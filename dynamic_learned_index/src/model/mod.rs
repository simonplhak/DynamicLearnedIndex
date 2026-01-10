pub mod candle_model;
#[cfg(feature = "tch")]
pub mod tch_model;

use std::path::PathBuf;

pub use candle_model::{Model, ModelBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub enum ModelDevice {
    #[default]
    #[serde(rename = "cpu")]
    Cpu,
    #[serde(rename = "gpu")]
    Gpu(usize),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainParams {
    pub threshold_samples: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub max_iters: usize, // Added for clustering iterations
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            threshold_samples: 1000,
            batch_size: 8,
            epochs: 3,
            max_iters: 10, // Default max iterations for clustering
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub enum RetrainStrategy {
    NoRetrain,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RetrainParams {
    pub strategy: RetrainStrategy,
    pub threshold_samples: usize,
    pub batch_size: usize,
    pub epochs: usize,
    pub max_iters: usize, // Added for clustering iterations
}

impl Default for RetrainParams {
    fn default() -> Self {
        Self {
            strategy: RetrainStrategy::NoRetrain,
            threshold_samples: 1000,
            batch_size: 8,
            epochs: 3,
            max_iters: 10, // Default max iterations for clustering
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelConfig {
    pub layers: Vec<ModelLayer>,
    pub train_params: TrainParams,
    pub retrain_params: RetrainParams,
    pub weights_path: Option<PathBuf>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            layers: vec![ModelLayer::Linear(128), ModelLayer::ReLU],
            train_params: Default::default(),
            retrain_params: Default::default(),
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
