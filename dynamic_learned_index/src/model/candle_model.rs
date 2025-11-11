use candle_core::{DType, Tensor, D};
use candle_core::{Device, Result as CandleResult};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
use candle_nn::{loss, ops};
use log::info;
use rand::rng;
use rand::seq::SliceRandom;

use crate::errors::BuildError;
use crate::model::{ModelDevice, ModelLayer, TrainParams};
use crate::structs::LabelMethod;
use crate::types::ArraySlice;
use crate::{clustering, sampling};

#[derive(Debug, Default)]
pub struct ModelBuilder {
    device: Option<ModelDevice>,
    input_nodes: Option<i64>,
    layers: Vec<ModelLayer>,
    labels: Option<usize>,
    train_params: Option<TrainParams>,
    label_method: Option<LabelMethod>,
    retrain_params: Option<TrainParams>,
}

impl ModelBuilder {
    pub fn device(&mut self, device: ModelDevice) -> &mut Self {
        self.device = Some(device);
        self
    }

    pub fn input_nodes(&mut self, input_nodes: i64) -> &mut Self {
        self.input_nodes = Some(input_nodes);
        self
    }

    pub fn add_layer(&mut self, layer: ModelLayer) -> &mut Self {
        self.layers.push(layer);
        self
    }

    pub fn labels(&mut self, labels: usize) -> &mut Self {
        self.labels = Some(labels);
        self
    }

    pub fn retrain_params(&mut self, retrain_params: TrainParams) -> &mut Self {
        self.retrain_params = Some(retrain_params);
        self
    }

    pub fn train_params(&mut self, train_params: TrainParams) -> &mut Self {
        self.train_params = Some(train_params);
        self
    }

    pub fn label_method(&mut self, label_method: LabelMethod) -> &mut Self {
        self.label_method = Some(label_method);
        self
    }

    pub fn build(&self) -> Result<Model, BuildError> {
        let device = self.device.as_ref().ok_or(BuildError::MissingAttribute)?;
        let device = match device {
            ModelDevice::Cpu => Device::Cpu,
            ModelDevice::Gpu(_) => todo!(),
        };
        let label_method = self.label_method.ok_or(BuildError::MissingAttribute)?;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let input_nodes = self.input_nodes.ok_or(BuildError::MissingAttribute)? as usize;
        let labels = self.labels.ok_or(BuildError::MissingAttribute)?;
        assert!(labels > 0, "labels must be greater than 0");
        let train_params = self.train_params.clone().unwrap_or_default();
        let retrain_params = self.retrain_params.clone().unwrap_or_default();
        let (mut layers, in_nodes) = self.layers.iter().enumerate().fold(
            (Vec::new(), input_nodes),
            |(mut layers, input_nodes), (i, layer)| {
                let (layers, output_nodes) = match layer {
                    ModelLayer::Linear(nodes) => {
                        let lin = linear(input_nodes, *nodes, vs.pp(format!("layer_{i}"))).unwrap();
                        layers.push(CandleModelLayer::Linear(lin));
                        (layers, *nodes)
                    }
                    ModelLayer::ReLU => {
                        layers.push(CandleModelLayer::ReLU);
                        (layers, input_nodes)
                    }
                };
                (layers, output_nodes)
            },
        );
        let lin =
            linear(in_nodes, labels, vs.pp("final")).map_err(|_| BuildError::ModelCreation)?;
        layers.push(CandleModelLayer::Linear(lin));
        let model = CandleModel { layers };
        let model = Model {
            model,
            varmap,
            labels,
            device,
            train_params,
            retrain_params,
            input_shape: input_nodes,
            label_method,
        };
        Ok(model)
    }
}

pub struct Model {
    model: CandleModel,
    varmap: VarMap,
    labels: usize,
    device: Device,
    pub input_shape: usize,
    train_params: TrainParams,
    retrain_params: TrainParams,
    label_method: LabelMethod,
}

impl Model {
    pub fn predict(&self, xs: &ArraySlice) -> Vec<(usize, f32)> {
        let tensor_test_votes = Tensor::from_vec(xs.to_vec(), (1, self.input_shape), &self.device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let logits = self.model.forward(&tensor_test_votes).unwrap();
        let final_result = ops::softmax(&logits, D::Minus1).unwrap();
        let predictions = final_result.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
        let mut predictions = predictions.into_iter().enumerate().collect::<Vec<_>>();
        predictions.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert!(predictions.len() <= self.labels);
        predictions
    }

    pub fn predict_many(&self, xs: &ArraySlice) -> Vec<usize> {
        let dim = xs.len() / self.input_shape;
        let tensor_test_votes =
            Tensor::from_vec(xs.to_vec(), (dim, self.input_shape), &self.device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();

        let final_result = self
            .model
            .forward(&tensor_test_votes)
            .unwrap()
            .argmax(1)
            .unwrap();
        final_result
            .squeeze(0)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
            .into_iter()
            .map(|v| v as usize)
            .collect()
    }

    pub fn train(&mut self, xs: &ArraySlice) {
        let sample_size = sampling::select_sample_size(
            self.labels,
            xs.len() / self.input_shape,
            self.train_params.threshold_samples,
        );
        info!(sample_size = sample_size, total = xs.len() / self.input_shape ; "model:train");
        let xs = sampling::sample(xs, sample_size, self.input_shape);
        let ys = clustering::compute_labels(
            &xs,
            &self.label_method,
            self.labels,
            self.input_shape,
            self.train_params.max_iters,
        );

        let optim_config = candle_nn::ParamsAdamW {
            lr: 1e-3,
            weight_decay: 0.0, // Make it behave like regular Adam
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(self.varmap.all_vars(), optim_config).unwrap();

        for _ in 0..self.train_params.epochs {
            // Create shuffled batches
            let batches = self.create_shuffled_batches(&xs, &ys);

            for (batch_xs, batch_ys) in batches {
                let logits = self.model.forward(&batch_xs).unwrap();
                let log_sm = ops::log_softmax(&logits, D::Minus1).unwrap();
                let loss = loss::nll(&log_sm, &batch_ys).unwrap();
                opt.backward_step(&loss).unwrap();
            }
        }
    }

    fn create_shuffled_batches(&self, xs: &[f32], ys: &[i32]) -> Vec<(Tensor, Tensor)> {
        let total_samples = ys.len();
        let batch_size = self.train_params.batch_size;

        // Create indices for shuffling
        let mut indices: Vec<usize> = (0..total_samples).collect();

        // Proper shuffle using rng
        indices.shuffle(&mut rng());

        let mut batches = Vec::new();

        for chunk_indices in indices.chunks(batch_size) {
            if chunk_indices.is_empty() {
                continue;
            }

            // Collect batch data
            let mut batch_xs_vec = Vec::with_capacity(chunk_indices.len() * self.input_shape);
            let mut batch_ys_vec = Vec::with_capacity(chunk_indices.len());

            for &idx in chunk_indices {
                let start_idx = idx * self.input_shape;
                let end_idx = start_idx + self.input_shape;
                batch_xs_vec.extend_from_slice(&xs[start_idx..end_idx]);
                batch_ys_vec.push(ys[idx] as i64);
            }

            // Create tensors for this batch
            let batch_xs = Tensor::from_vec(
                batch_xs_vec,
                (chunk_indices.len(), self.input_shape),
                &self.device,
            )
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

            let batch_ys =
                Tensor::from_vec(batch_ys_vec, (chunk_indices.len(),), &self.device).unwrap();

            batches.push((batch_xs, batch_ys));
        }

        batches
    }

    pub fn retrain(&mut self, _xs: &ArraySlice) {
        info!(epochs = self.retrain_params.epochs; "model:retrain");
    }
}

enum CandleModelLayer {
    Linear(Linear),
    ReLU,
}
struct CandleModel {
    layers: Vec<CandleModelLayer>,
}

// Implement the forward pass for our model, which is required by the
// `Module` trait. This is where we chain the layers and activations.
impl Module for CandleModel {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        self.layers
            .iter()
            .try_fold(xs.clone(), |acc, layer| match layer {
                CandleModelLayer::Linear(lin) => lin.forward(&acc),
                CandleModelLayer::ReLU => acc.relu(),
            })
    }
}
