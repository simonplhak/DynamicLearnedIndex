use std::path::PathBuf;

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
use crate::{clustering, sampling, ModelConfig};

#[derive(Debug, Default)]
pub struct ModelBuilder {
    device: Option<ModelDevice>,
    input_nodes: Option<i64>,
    layers: Vec<ModelLayer>,
    labels: Option<usize>,
    train_params: Option<TrainParams>,
    label_method: Option<LabelMethod>,
    retrain_params: Option<TrainParams>,
    weights_path: Option<PathBuf>,
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

    pub fn weights_path(&mut self, weights_path: PathBuf) -> &mut Self {
        self.weights_path = Some(weights_path);
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
        let vs = match &self.weights_path {
            Some(weights_path) => unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device).unwrap()
            },
            None => VarBuilder::from_varmap(&varmap, DType::F32, &device),
        };
        let input_nodes = self.input_nodes.ok_or(BuildError::MissingAttribute)? as usize;
        let labels = self.labels.ok_or(BuildError::MissingAttribute)?;
        assert!(labels > 0, "labels must be greater than 0");
        let train_params = self.train_params.clone().unwrap_or_default();
        let retrain_params = self.retrain_params.clone().unwrap_or_default();
        let (mut layers, in_nodes) = self
            .layers
            .iter()
            .enumerate()
            .take(self.layers.len() - 1)
            .fold(
                (Vec::new(), input_nodes),
                |(mut layers, input_nodes), (i, layer)| {
                    let (layers, output_nodes) = match layer {
                        ModelLayer::Linear(nodes) => {
                            let lin = linear(input_nodes, *nodes, vs.pp(format!("{i}", i = 2 * i)))
                                .unwrap();
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
        let lin = linear(
            in_nodes,
            labels,
            vs.pp(format!("{i}", i = 2 * layers.len())),
        )
        .unwrap();
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

// Iterator for streaming batches without allocating a Vec of batches.
struct BatchIter {
    dataset: Tensor,
    labels: Tensor,
    batch_size: usize,
    total_samples: usize,
    start_sample: usize,
}

impl Iterator for BatchIter {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.start_sample >= self.total_samples {
            return None;
        }
        let end_sample = std::cmp::min(self.start_sample + self.batch_size, self.total_samples);
        let len = end_sample - self.start_sample;

        // Create zero-copy views for the batch
        let batch_xs = self.dataset.narrow(0, self.start_sample, len).unwrap();
        let batch_ys = self.labels.narrow(0, self.start_sample, len).unwrap();

        self.start_sample = end_sample;
        Some((batch_xs, batch_ys))
    }
}

impl Model {
    pub fn predict(&self, xs: &ArraySlice) -> Vec<(usize, f32)> {
        let tensor_test_votes = Tensor::from_slice(xs, (1, self.input_shape), &self.device)
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
        let dataset = Tensor::from_slice(xs, (dim, self.input_shape), &self.device).unwrap();
        self.model
            .forward(&dataset)
            .unwrap()
            .argmax(1)
            .unwrap()
            .squeeze(0)
            .unwrap()
            .to_vec1::<u32>()
            .unwrap()
            .into_iter()
            .map(|v| v as usize)
            .collect::<Vec<_>>()
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

    fn create_shuffled_batches(&self, xs: &[f32], ys: &[i32]) -> BatchIter {
        let total_samples = ys.len();
        let batch_size = if self.train_params.batch_size > 0 {
            self.train_params.batch_size
        } else {
            total_samples
        };

        // Create indices for shuffling and permute the whole dataset once per epoch
        let mut indices: Vec<usize> = (0..total_samples).collect();
        indices.shuffle(&mut rng());

        // Permute xs and ys into contiguous buffers (one full dataset copy per epoch)
        let mut permuted_xs: Vec<f32> = Vec::with_capacity(xs.len());
        let mut permuted_ys: Vec<i64> = Vec::with_capacity(total_samples);

        for &idx in &indices {
            let start = idx * self.input_shape;
            let end = start + self.input_shape;
            permuted_xs.extend_from_slice(&xs[start..end]);
            permuted_ys.push(ys[idx] as i64);
        }

        // Create a single dataset tensor (consumes the Vec) and a labels tensor.
        // On CPU this should be zero-copy: the Vec's buffer becomes the Tensor storage.
        let dataset =
            Tensor::from_vec(permuted_xs, (total_samples, self.input_shape), &self.device)
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap();

        let labels_tensor = Tensor::from_vec(permuted_ys, (total_samples,), &self.device).unwrap();

        BatchIter {
            dataset,
            labels: labels_tensor,
            batch_size,
            total_samples,
            start_sample: 0,
        }
    }

    pub fn retrain(&mut self, _xs: &ArraySlice) {
        info!(epochs = self.retrain_params.epochs; "model:retrain");
    }

    pub fn dump(&self, weights_filename: PathBuf) -> ModelConfig {
        self.varmap.save(&weights_filename).unwrap();
        ModelConfig {
            layers: self
                .model
                .layers
                .iter()
                .map(|layer| match layer {
                    CandleModelLayer::Linear(lin) => ModelLayer::Linear(lin.weight().dims()[0]),
                    CandleModelLayer::ReLU => ModelLayer::ReLU,
                })
                .collect(),
            train_params: self.train_params.clone(),
            retrain_params: self.retrain_params.clone(),
            weights_path: Some(weights_filename),
        }
    }
}

enum CandleModelLayer {
    Linear(Linear),
    ReLU,
}
struct CandleModel {
    layers: Vec<CandleModelLayer>,
}

impl Module for CandleModel {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let mut current = match &self.layers[0] {
            CandleModelLayer::Linear(lin) => lin.forward(xs)?,
            CandleModelLayer::ReLU => xs.relu()?,
        };
        for layer in &self.layers[1..] {
            current = match layer {
                CandleModelLayer::Linear(lin) => lin.forward(&current)?,
                CandleModelLayer::ReLU => current.relu()?,
            };
        }
        Ok(current)
    }
}
