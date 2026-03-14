mod features;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::errors::DliResult;
use crate::structs::LabelMethod;
use crate::types::ArraySlice;
use std::marker::PhantomData;

pub trait ModelInterface {
    fn predict(&self, xs: &Self::TensorType) -> DliResult<Vec<(usize, f32)>>;
    fn predict_many(&self, xs: &ArraySlice) -> DliResult<Vec<usize>>;
    fn train(&mut self, xs: &ArraySlice) -> DliResult<()>;
    fn retrain(&mut self, xs: &ArraySlice) -> DliResult<()>;
    fn dump(&self, weights_filename: PathBuf) -> DliResult<ModelConfig>;
    fn memory_usage(&self) -> usize;
    fn vec2tensor(&self, xs: &[f32]) -> DliResult<Self::TensorType>;

    type TensorType;
}

#[derive(Default)]
pub struct CandleBackend;

#[derive(Default)]
pub struct TchBackend;

#[derive(Debug, Clone, Default)]
pub struct BaseModelBuilder<B> {
    pub device: Option<ModelDevice>,
    pub input_nodes: Option<i64>,
    pub layers: Vec<ModelLayer>,
    pub labels: Option<usize>,
    pub train_params: Option<TrainParams>,
    pub label_method: Option<LabelMethod>,
    pub weights_path: Option<PathBuf>,
    _backend: PhantomData<B>,
}

impl<B> BaseModelBuilder<B> {
    pub fn device(&mut self, device: ModelDevice) -> &mut Self {
        self.device = Some(device);
        self
    }

    pub fn input_nodes(&mut self, input_nodes: i64) -> &mut Self {
        self.input_nodes = Some(input_nodes);
        self
    }

    pub fn layers(&mut self, layers: Vec<ModelLayer>) -> &mut Self {
        self.layers = layers;
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
}

#[derive(Debug, Serialize, Deserialize, Default, Clone, Copy)]
pub enum ModelDevice {
    #[default]
    #[serde(rename = "cpu")]
    Cpu,
    #[serde(rename = "gpu")]
    Gpu(usize),
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub struct TrainParams {
    pub threshold_samples: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub max_iters: usize, // Added for clustering iterations
    pub retrain_strategy: RetrainStrategy,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            threshold_samples: 1000,
            batch_size: 256,
            epochs: 3,
            max_iters: 10, // Default max iterations for clustering
            retrain_strategy: RetrainStrategy::NoRetrain,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub enum RetrainStrategy {
    #[serde(rename = "no_retrain")]
    NoRetrain,
    #[serde(rename = "from_scratch")]
    FromScratch,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub layers: Vec<ModelLayer>,
    pub train_params: TrainParams,
    pub weights_path: Option<PathBuf>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            layers: vec![ModelLayer::Linear(128), ModelLayer::ReLU],
            train_params: Default::default(),
            weights_path: None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(tag = "type", content = "value")]
pub enum ModelLayer {
    #[serde(rename = "linear")]
    Linear(usize),
    #[serde(rename = "relu")]
    ReLU,
}

#[cfg(feature = "candle")]
pub mod candle_model;

#[cfg(feature = "tch")]
pub mod tch_model;

#[cfg(feature = "candle")]
pub type Model = candle_model::Model;

#[cfg(feature = "tch")]
pub type Model = tch_model::Model;

#[cfg(feature = "candle")]
pub type ModelBuilder = BaseModelBuilder<CandleBackend>;

#[cfg(feature = "tch")]
pub type ModelBuilder = BaseModelBuilder<TchBackend>;
