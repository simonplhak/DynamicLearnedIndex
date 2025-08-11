use std::env::var;

use candle_core::{DType, Tensor, D};
use candle_core::{Device, Result as CandleResult};
use candle_nn::{
    linear, loss::mse, Activation, AdamW, Linear, Module, Optimizer, VarBuilder, VarMap,
};
use candle_nn::{loss, ops};
use log::info;

use crate::distance_fn::LabelMethod;
use crate::errors::BuildError;
use crate::model::{ModelLayer, TrainParams};
use crate::types::ArraySlice;
use crate::{clustering, sampling, ModelDevice};

#[derive(Debug, Default)]
pub struct ModelNewBuilder {
    device: Option<ModelDevice>,
    input_nodes: Option<i64>,
    layers: Vec<ModelLayer>,
    labels: Option<usize>,
    train_params: Option<TrainParams>,
    label_method: Option<LabelMethod>,
    retrain_params: Option<TrainParams>,
}

impl ModelNewBuilder {
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

    pub fn build(&self) -> Result<ModelNew, BuildError> {
        let device = self.device.as_ref().ok_or(BuildError::MissingAttribute)?;
        let device = match device {
            ModelDevice::Cpu => Device::Cpu,
            ModelDevice::Gpu(_) => todo!(),
        };
        let label_method = self.label_method.ok_or(BuildError::MissingAttribute)?;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let input_nodes = self.input_nodes.ok_or(BuildError::MissingAttribute)?;
        let labels = self.labels.ok_or(BuildError::MissingAttribute)?;
        assert!(labels > 0, "labels must be greater than 0");
        let train_params = self.train_params.clone().unwrap_or_default();
        let retrain_params = self.retrain_params.clone().unwrap_or_default();
        let model = CandleModel {
            ln1: linear(input_nodes as usize, 256, vs.pp("ln1")).unwrap(),
            ln2: linear(256, labels, vs.pp("ln2")).unwrap(),
        };
        let model = ModelNew {
            model,
            varmap,
            labels,
            device,
            train_params,
            retrain_params,
            input_shape: input_nodes as usize,
            label_method,
        };
        Ok(model)
    }
}

pub struct ModelNew {
    model: CandleModel,
    varmap: VarMap,
    labels: usize,
    device: Device,
    pub input_shape: usize,
    train_params: TrainParams,
    retrain_params: TrainParams,
    label_method: LabelMethod,
}

impl ModelNew {
    pub fn predict(&self, xs: &ArraySlice) -> Vec<(usize, f32)> {
        let tensor_test_votes = Tensor::from_vec(xs.to_vec(), (1, self.input_shape), &self.device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();

        let final_result = self.model.forward(&tensor_test_votes).unwrap();
        let x = final_result.squeeze(0).unwrap().to_vec1::<f32>().unwrap();
        x.into_iter().enumerate().collect::<Vec<_>>()
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
        let xs = sampling::sample(xs, self.train_params.threshold_samples, self.input_shape);
        let ys = clustering::compute_labels(
            &xs,
            &self.label_method,
            self.labels,
            self.input_shape,
            self.train_params.max_iters,
        );
        let dim = xs.len() / self.input_shape;
        let xs = Tensor::from_vec(xs.to_vec(), (dim, self.input_shape), &self.device)
            .unwrap()
            .to_dtype(DType::F32)
            .unwrap();
        let ys = Tensor::from_vec(
            ys.into_iter().map(|y| y as i64).collect::<Vec<i64>>(),
            (dim,),
            &self.device,
        )
        .unwrap();
        let mut sgd = candle_nn::SGD::new(self.varmap.all_vars(), 0.001).unwrap();
        for _ in 0..self.train_params.epochs {
            let logits = self.model.forward(&xs).unwrap();
            let log_sm = ops::log_softmax(&logits, D::Minus1).unwrap();
            let loss = loss::nll(&log_sm, &ys).unwrap();
            sgd.backward_step(&loss).unwrap();
        }
    }

    pub fn retrain(&mut self, _xs: &ArraySlice) {
        info!(epochs = self.retrain_params.epochs; "model:retrain");
    }
}

struct CandleModel {
    ln1: Linear,
    ln2: Linear,
}

// Implement the constructor for our model.
impl CandleModel {
    fn new(vs: VarBuilder) -> CandleResult<Self> {
        let ln1 = linear(10, 256, vs.pp("ln1"))?;
        let ln2 = linear(256, 256, vs.pp("ln2"))?;

        Ok(Self { ln1, ln2 })
    }
}

// Implement the forward pass for our model, which is required by the
// `Module` trait. This is where we chain the layers and activations.
impl Module for CandleModel {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        // 1. Pass input through the first linear layer.
        let xs = self.ln1.forward(xs)?;

        // 2. Apply the ReLU activation function.
        let xs = xs.relu()?;

        // 3. Pass the result through the second linear layer.
        let xs = self.ln2.forward(&xs)?;

        // Return the final tensor.
        Ok(xs)
    }
}
