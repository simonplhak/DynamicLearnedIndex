pub mod candle_model;
#[cfg(feature = "tch")]
pub mod mix_model;
#[cfg(feature = "tch")]
pub mod tch_model;

use std::path::PathBuf;

#[cfg(not(feature = "tch"))]
pub use candle_model::{Model, ModelBuilder};
#[cfg(feature = "tch")]
pub use mix_model::{Model, ModelBuilder};
use serde::{Deserialize, Serialize};

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
