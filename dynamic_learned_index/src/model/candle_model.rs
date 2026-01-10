use std::path::PathBuf;

use candle_core::{DType, Tensor, D};
use candle_core::{Device, Result as CandleResult};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
use candle_nn::{loss, ops};
use log::info;
use rand::rng;
use rand::seq::SliceRandom;

use crate::errors::{DliError, DliResult};
use crate::model::{ModelDevice, ModelLayer, RetrainParams, RetrainStrategy, TrainParams};
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
    retrain_params: Option<RetrainParams>,
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

    pub fn retrain_params(&mut self, retrain_params: RetrainParams) -> &mut Self {
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

    pub fn build(&self) -> DliResult<Model> {
        let device = self
            .device
            .as_ref()
            .ok_or(DliError::MissingAttribute("device"))?;
        let device = match device {
            ModelDevice::Cpu => Device::Cpu,
            ModelDevice::Gpu(_) => todo!(),
        };
        let label_method = self
            .label_method
            .ok_or(DliError::MissingAttribute("label_method"))?;
        let varmap = VarMap::new();
        let vs = match &self.weights_path {
            Some(weights_path) => unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)?
            },
            None => VarBuilder::from_varmap(&varmap, DType::F32, &device),
        };
        let input_nodes = self
            .input_nodes
            .ok_or(DliError::MissingAttribute("input_nodes"))? as usize;
        let labels = self.labels.ok_or(DliError::MissingAttribute("labels"))?;
        assert!(labels > 0, "labels must be greater than 0");
        let train_params = self.train_params.unwrap_or_default();
        let retrain_params = self.retrain_params.unwrap_or_default();
        let mut i = 0;
        let (mut layers, in_nodes) = self.layers.iter().try_fold(
            (Vec::<CandleModelLayer>::new(), input_nodes),
            |(mut layers, input_nodes), layer| -> DliResult<(Vec<CandleModelLayer>, usize)> {
                let (layers, output_nodes) = match layer {
                    ModelLayer::Linear(nodes) => {
                        let lin = linear(input_nodes, *nodes, vs.pp(format!("{i}", i = 2 * i)))?;
                        i += 1;
                        layers.push(CandleModelLayer::Linear(lin));
                        (layers, *nodes)
                    }
                    ModelLayer::ReLU => {
                        layers.push(CandleModelLayer::ReLU);
                        (layers, input_nodes)
                    }
                };
                Ok((layers, output_nodes))
            },
        )?;
        let lin = linear(in_nodes, labels, vs.pp(format!("{i}", i = 2 * i)))?;
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
    retrain_params: RetrainParams,
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
    pub fn predict(&self, xs: &ArraySlice) -> DliResult<Vec<(usize, f32)>> {
        let tensor_test_votes =
            Tensor::from_slice(xs, (1, self.input_shape), &self.device)?.to_dtype(DType::F32)?;

        let logits = self.model.forward(&tensor_test_votes)?;
        let final_result = ops::softmax(&logits, D::Minus1)?;
        let predictions = final_result.squeeze(0)?.to_vec1::<f32>()?;
        let mut predictions = predictions.into_iter().enumerate().collect::<Vec<_>>();
        predictions.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert!(predictions.len() <= self.labels);
        Ok(predictions)
    }

    pub fn predict_many(&self, xs: &ArraySlice) -> DliResult<Vec<usize>> {
        let dim = xs.len() / self.input_shape;
        let dataset = Tensor::from_slice(xs, (dim, self.input_shape), &self.device)?;
        let rs = self.model.forward(&dataset)?.argmax(1)?.to_vec1::<u32>()?;
        Ok(rs.into_iter().map(|v| v as usize).collect::<Vec<_>>())
    }

    pub fn train(&mut self, xs: &ArraySlice) -> DliResult<()> {
        self._train(xs, self.train_params)
    }

    fn _train(&mut self, xs: &ArraySlice, train_params: TrainParams) -> DliResult<()> {
        let sample_size = sampling::select_sample_size(
            self.labels,
            xs.len() / self.input_shape,
            train_params.threshold_samples,
        );
        info!(sample_size = sample_size, total = xs.len() / self.input_shape ; "model:train");
        let xs = sampling::sample(xs, sample_size, self.input_shape);
        let ys = clustering::compute_labels(
            &xs,
            &self.label_method,
            self.labels,
            self.input_shape,
            train_params.max_iters,
        );

        let optim_config = candle_nn::ParamsAdamW {
            lr: 1e-3,
            weight_decay: 0.0, // Make it behave like regular Adam
            ..Default::default()
        };
        let mut opt = candle_nn::AdamW::new(self.varmap.all_vars(), optim_config)?;

        for _ in 0..train_params.epochs {
            // Create shuffled batches
            let batches = self.create_shuffled_batches(&xs, &ys, train_params.batch_size)?;

            for (batch_xs, batch_ys) in batches {
                let logits = self.model.forward(&batch_xs)?;
                let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &batch_ys)?;
                opt.backward_step(&loss)?;
            }
        }
        Ok(())
    }

    fn create_shuffled_batches(
        &self,
        xs: &[f32],
        ys: &[i32],
        batch_size: usize,
    ) -> DliResult<BatchIter> {
        let total_samples = ys.len();
        let batch_size = if batch_size > 0 {
            batch_size
        } else {
            total_samples
        };

        let mut indices: Vec<usize> = (0..total_samples).collect();
        indices.shuffle(&mut rng());

        let mut permuted_xs: Vec<f32> = Vec::with_capacity(xs.len());
        let mut permuted_ys: Vec<i64> = Vec::with_capacity(total_samples);

        for &idx in &indices {
            let start = idx * self.input_shape;
            let end = start + self.input_shape;
            permuted_xs.extend_from_slice(&xs[start..end]);
            permuted_ys.push(ys[idx] as i64);
        }

        let dataset =
            Tensor::from_vec(permuted_xs, (total_samples, self.input_shape), &self.device)?
                .to_dtype(DType::F32)?;

        let labels_tensor = Tensor::from_vec(permuted_ys, (total_samples,), &self.device)?;

        Ok(BatchIter {
            dataset,
            labels: labels_tensor,
            batch_size,
            total_samples,
            start_sample: 0,
        })
    }

    pub fn retrain(&mut self, _xs: &ArraySlice) -> DliResult<()> {
        info!(epochs = self.retrain_params.epochs; "model:retrain");
        match self.retrain_params.strategy {
            RetrainStrategy::NoRetrain => {
                info!("No retraining performed as per strategy.");
            }
            RetrainStrategy::FromScratch => {
                reset_model(&mut self.varmap, &self.device)?;
                self._train(_xs, self.retrain_params.into())?;
            }
        };
        Ok(())
    }

    pub fn dump(&self, weights_filename: PathBuf) -> DliResult<ModelConfig> {
        self.varmap.save(&weights_filename)?;
        Ok(ModelConfig {
            layers: self
                .model
                .layers
                .iter()
                .take(self.model.layers.len() - 1)
                .map(|layer| match layer {
                    CandleModelLayer::Linear(lin) => ModelLayer::Linear(lin.weight().dims()[0]),
                    CandleModelLayer::ReLU => ModelLayer::ReLU,
                })
                .collect(),
            train_params: self.train_params,
            retrain_params: self.retrain_params,
            weights_path: Some(weights_filename),
        })
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

fn reset_model(var_map: &mut VarMap, device: &Device) -> CandleResult<()> {
    for var in var_map.all_vars() {
        let dims = var.dims();
        if dims.len() == 2 {
            let fan_in = dims[1];
            let bound = (6.0 / fan_in as f64).sqrt();
            let new_values = Tensor::rand(-bound, bound, dims, device)?.to_dtype(var.dtype())?;
            var.set(&new_values)?;
        } else {
            // Initialize biases to zero
            let new_values = Tensor::zeros(dims, var.dtype(), device)?;
            var.set(&new_values)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::model::RetrainStrategy;

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
        };
        let retrain_params = RetrainParams {
            threshold_samples: 1000,
            batch_size: 32,
            epochs: 10,
            max_iters: 15,
            strategy: RetrainStrategy::NoRetrain,
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
            .retrain_params(retrain_params)
            .label_method(LabelMethod::KMeans)
            .build();

        // Assert
        assert!(model.is_ok(), "Model should build successfully");
        let model = model.unwrap();
        assert_eq!(model.input_shape, input_nodes as usize);
        assert_eq!(model.labels, labels);
        assert_eq!(
            model.train_params.threshold_samples,
            train_params.threshold_samples
        );
        assert_eq!(model.train_params.batch_size, train_params.batch_size);
        assert_eq!(model.train_params.epochs, train_params.epochs);
        assert_eq!(model.train_params.max_iters, train_params.max_iters);
        assert_eq!(
            model.retrain_params.threshold_samples,
            retrain_params.threshold_samples
        );
        assert_eq!(model.retrain_params.batch_size, retrain_params.batch_size);
        assert_eq!(model.retrain_params.epochs, retrain_params.epochs);
        assert_eq!(model.retrain_params.max_iters, retrain_params.max_iters);

        // Verify the model has the correct number of layers
        // We added 4 layers + 1 final output layer = 5 total
        assert_eq!(model.model.layers.len(), 5);
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
            .map(|query| model.predict(query).unwrap())
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
            .map(|query| loaded_model.predict(query).unwrap())
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
            let result = model.predict(query).unwrap();
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
            })
            .label_method(LabelMethod::KMeans)
            .build()
            .unwrap();

        // Create two distinct clusters
        let mut training_data = Vec::new();
        // Cluster 1: around 0.2
        for _ in 0..50 {
            training_data.push(0.2);
            training_data.push(0.2);
        }
        // Cluster 2: around 0.8
        for _ in 0..50 {
            training_data.push(0.8);
            training_data.push(0.8);
        }

        // Act
        model.train(&training_data).unwrap();

        // Assert
        // Check predictions for representative points
        let p1 = vec![0.2, 0.2];
        let p2 = vec![0.8, 0.8];

        let res1 = model.predict(&p1).unwrap();
        let res2 = model.predict(&p2).unwrap();

        let label1 = res1[0].0;
        let label2 = res2[0].0;

        assert_ne!(
            label1, label2,
            "Model should assign different labels to distinct clusters"
        );

        // Verify consistency within clusters
        let p1_b = vec![0.21, 0.19];
        let res1_b = model.predict(&p1_b).unwrap();
        assert_eq!(
            res1_b[0].0, label1,
            "Model should be consistent within cluster 1"
        );

        let p2_b = vec![0.79, 0.81];
        let res2_b = model.predict(&p2_b).unwrap();
        assert_eq!(
            res2_b[0].0, label2,
            "Model should be consistent within cluster 2"
        );
    }

    #[test]
    fn test_builder_validation() {
        // Test missing device
        let res = ModelBuilder::default().build();
        assert!(matches!(res, Err(DliError::MissingAttribute("device"))));

        // Test missing label_method (device provided)
        let res = ModelBuilder::default().device(ModelDevice::Cpu).build();
        assert!(matches!(
            res,
            Err(DliError::MissingAttribute("label_method"))
        ));

        // Test missing input_nodes (device and label_method provided)
        let res = ModelBuilder::default()
            .device(ModelDevice::Cpu)
            .label_method(LabelMethod::KMeans)
            .build();
        assert!(matches!(
            res,
            Err(DliError::MissingAttribute("input_nodes"))
        ));

        // Test missing labels (others provided)
        let res = ModelBuilder::default()
            .device(ModelDevice::Cpu)
            .label_method(LabelMethod::KMeans)
            .input_nodes(10)
            .build();
        assert!(matches!(res, Err(DliError::MissingAttribute("labels"))));

        // Test success
        let res = ModelBuilder::default()
            .device(ModelDevice::Cpu)
            .label_method(LabelMethod::KMeans)
            .input_nodes(10)
            .labels(2)
            .build();
        assert!(res.is_ok());
    }
}
