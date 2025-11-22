// pub use candle::*;
pub mod candle_model;
pub mod tch_model;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
// pub use tch_model::{Model, ModelBuilder};
pub use candle_model::{Model, ModelBuilder};

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
pub struct ModelConfig {
    pub layers: Vec<ModelLayer>,
    pub train_params: TrainParams,
    pub retrain_params: TrainParams,
    pub weights_path: Option<PathBuf>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            layers: vec![
                ModelLayer::Linear(256),
                ModelLayer::ReLU,
                ModelLayer::Linear(256),
                ModelLayer::ReLU,
            ],
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
