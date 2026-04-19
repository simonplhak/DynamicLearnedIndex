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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_builder_with_all_params() {
        // Arrange
        let input_nodes = 128;
        let labels = 10;
        let train_params = TrainParams {
            threshold_samples: 500,
            batch_size: 16,
            epochs: 5,
            max_iters: 20,
            retrain_strategy: RetrainStrategy::NoRetrain,
        };

        // Act
        let model = ModelBuilder::default()
            .device(ModelDevice::Cpu)
            .input_nodes(input_nodes)
            .add_layer(ModelLayer::Linear(256))
            .add_layer(ModelLayer::ReLU)
            .add_layer(ModelLayer::Linear(128))
            .add_layer(ModelLayer::ReLU)
            .labels(labels)
            .train_params(train_params)
            .label_method(LabelMethod::KMeans)
            .build();

        // Assert
        assert!(model.is_ok(), "Model should build successfully");
        let model = model.unwrap();
        assert_eq!(model.input_shape, input_nodes as usize);
        assert_eq!(
            model.train_params.threshold_samples,
            train_params.threshold_samples
        );
        assert_eq!(model.train_params.batch_size, train_params.batch_size);
        assert_eq!(model.train_params.epochs, train_params.epochs);
        assert_eq!(model.train_params.max_iters, train_params.max_iters);
    }

    #[test]
    fn test_model_save_and_load_weights() {
        use tempfile::NamedTempFile;

        // Create and train a model
        let mut builder = ModelBuilder::default();
        let mut model = builder
            .device(ModelDevice::Cpu)
            .input_nodes(10)
            .add_layer(ModelLayer::Linear(32))
            .add_layer(ModelLayer::ReLU)
            .add_layer(ModelLayer::Linear(16))
            .labels(3)
            .train_params(TrainParams {
                epochs: 5,
                batch_size: 10,
                threshold_samples: 50,
                max_iters: 10,
                retrain_strategy: RetrainStrategy::NoRetrain,
            })
            .label_method(LabelMethod::KMeans)
            .build()
            .unwrap();

        // Create training data (50 samples, 10 features each)
        let training_data: Vec<f32> = (0..500).map(|i| (i % 100) as f32 / 100.0).collect();

        // Train the model
        model.train(&training_data).unwrap();

        // Create test queries
        let test_queries: Vec<Vec<f32>> = vec![
            (0..10).map(|i| i as f32 / 10.0).collect(),
            (0..10).map(|i| (i + 5) as f32 / 10.0).collect(),
            (0..10).map(|i| (i * 2) as f32 / 10.0).collect(),
        ];

        // Get predictions from original model
        let original_predictions: Vec<Vec<(usize, f32)>> = test_queries
            .iter()
            .map(|query| model.predict(&model.vec2tensor(query).unwrap()).unwrap())
            .collect::<Vec<_>>();

        // Save model to temporary file
        let temp_file = NamedTempFile::new().expect("Failed to create temp file");
        let weights_path = temp_file.path().to_path_buf();
        model.dump(weights_path.clone()).unwrap();

        // Load model from weights
        let mut loaded_builder = ModelBuilder::default();
        let loaded_model = loaded_builder
            .device(ModelDevice::Cpu)
            .input_nodes(10)
            .add_layer(ModelLayer::Linear(32))
            .add_layer(ModelLayer::ReLU)
            .add_layer(ModelLayer::Linear(16))
            .labels(3)
            .label_method(LabelMethod::KMeans)
            .weights_path(weights_path)
            .build()
            .unwrap();

        // Get predictions from loaded model
        let loaded_predictions: Vec<Vec<(usize, f32)>> = test_queries
            .iter()
            .map(|query| {
                loaded_model
                    .predict(&loaded_model.vec2tensor(query).unwrap())
                    .unwrap()
            })
            .collect::<Vec<_>>();

        // Verify predictions match
        assert_eq!(original_predictions.len(), loaded_predictions.len());
        for (original, loaded) in original_predictions.iter().zip(loaded_predictions.iter()) {
            assert_eq!(original.len(), loaded.len());
            for ((orig_label, orig_prob), (load_label, load_prob)) in
                original.iter().zip(loaded.iter())
            {
                assert_eq!(orig_label, load_label, "Labels should match");
                assert!(
                    (orig_prob - load_prob).abs() < 1e-5,
                    "Probabilities should match (original: {orig_prob}, loaded: {load_prob})"
                );
            }
        }
    }

    #[test]
    fn test_predict_many_consistency() {
        // Arrange
        let input_nodes = 10;
        let labels = 5;
        let mut builder = ModelBuilder::default();
        let mut model = builder
            .device(ModelDevice::Cpu)
            .input_nodes(input_nodes)
            .add_layer(ModelLayer::Linear(32))
            .add_layer(ModelLayer::ReLU)
            .labels(labels)
            .train_params(TrainParams {
                epochs: 1,
                batch_size: 10,
                threshold_samples: 50,
                max_iters: 10,
                retrain_strategy: RetrainStrategy::NoRetrain,
            })
            .label_method(LabelMethod::KMeans)
            .build()
            .unwrap();

        // Create random training data to initialize weights properly
        let training_data: Vec<f32> = (0..100).map(|i| (i % 100) as f32 / 100.0).collect();
        model.train(&training_data).unwrap();

        // Create test queries (batch of 5 vectors)
        let batch_size = 5;
        let test_data: Vec<f32> = (0..batch_size * input_nodes as usize)
            .map(|i| (i % 100) as f32 / 100.0)
            .collect();

        // Act 1: Predict individually
        let mut individual_predictions = Vec::new();
        for i in 0..batch_size {
            let start = i * input_nodes as usize;
            let end = start + input_nodes as usize;
            let query = &test_data[start..end];
            let result = model.predict(&model.vec2tensor(query).unwrap()).unwrap();
            // predict returns sorted (label, prob), so first one is top-1
            individual_predictions.push(result[0].0);
        }

        // Act 2: Predict as a batch
        let batch_predictions = model.predict_many(&test_data).unwrap();

        // Assert
        assert_eq!(
            individual_predictions.len(),
            batch_predictions.len(),
            "Should have same number of predictions"
        );

        for (i, (individual, batch)) in individual_predictions
            .iter()
            .zip(batch_predictions.iter())
            .enumerate()
        {
            assert_eq!(
                individual, batch,
                "Prediction mismatch at index {}: individual={}, batch={}",
                i, individual, batch
            );
        }
    }

    #[test]
    fn test_basic_learning_capability() {
        // Arrange
        let input_nodes = 2;
        let labels = 2;
        let mut builder = ModelBuilder::default();
        let mut model = builder
            .device(ModelDevice::Cpu)
            .input_nodes(input_nodes)
            .add_layer(ModelLayer::Linear(16))
            .add_layer(ModelLayer::ReLU)
            .labels(labels)
            .train_params(TrainParams {
                epochs: 50,
                batch_size: 10,
                threshold_samples: 200,
                max_iters: 10,
                retrain_strategy: RetrainStrategy::NoRetrain,
            })
            .label_method(LabelMethod::KMeans)
            .build()
            .unwrap();

        // Create two distinct clusters
        let mut training_data = Vec::new();
        // Cluster 1: around 0.2
        for _ in 0..50 {
            training_data.push(0.2 + rand::random::<f32>() * 0.0001);
            training_data.push(0.2 + rand::random::<f32>() * 0.0001);
        }
        // Cluster 2: around 0.8
        for _ in 0..50 {
            training_data.push(0.8 + rand::random::<f32>() * 0.0001);
            training_data.push(0.8 + rand::random::<f32>() * 0.0001);
        }

        // Act
        model.train(&training_data).unwrap();

        // Assert
        // Check predictions for representative points
        let p1 = model.vec2tensor(&[0.2, 0.2]).unwrap();
        let p2 = model.vec2tensor(&[0.8, 0.8]).unwrap();

        let res1 = model.predict(&p1).unwrap();
        let res2 = model.predict(&p2).unwrap();

        let label1 = res1[0].0;
        let label2 = res2[0].0;

        assert_ne!(
            label1, label2,
            "Model should assign different labels to distinct clusters"
        );

        // Verify consistency within clusters
        let p1_b = model.vec2tensor(&[0.21, 0.19]).unwrap();
        let res1_b = model.predict(&p1_b).unwrap();
        assert_eq!(
            res1_b[0].0, label1,
            "Model should be consistent within cluster 1"
        );

        let p2_b = model.vec2tensor(&[0.79, 0.81]).unwrap();
        let res2_b = model.predict(&p2_b).unwrap();
        assert_eq!(
            res2_b[0].0, label2,
            "Model should be consistent within cluster 2"
        );
    }
}
