use std::path::PathBuf;

use candle_core::{DType, IndexOp as _, Tensor, D};
use candle_core::{Device, Result as CandleResult};
use candle_nn::{linear, Linear, Module, Optimizer, VarBuilder, VarMap};
use candle_nn::{loss, ops};
use log::debug;
use measure_time_macro::log_time;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rng;
use std::collections::HashMap;

use crate::errors::{DliError, DliResult};
use crate::model::{CandleBackend, ModelDevice, ModelLayer, RetrainStrategy, TrainParams};
use crate::structs::LabelMethod;
use crate::types::ArraySlice;
use crate::{clustering, sampling, ModelConfig};

impl crate::model::BaseModelBuilder<CandleBackend> {
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
    input_shape: usize,
    train_params: TrainParams,
    label_method: LabelMethod,
}

impl crate::model::ModelInterface for Model {
    type TensorType = Tensor;

    fn predict(&self, xs: &Tensor) -> DliResult<Vec<(usize, f32)>> {
        let logits = self.model.forward(xs)?;
        let final_result = ops::softmax(&logits, D::Minus1)?;
        let predictions = final_result.squeeze(0)?.to_vec1::<f32>()?;
        let mut predictions = predictions.into_iter().enumerate().collect::<Vec<_>>();
        predictions.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert!(predictions.len() <= self.labels);
        Ok(predictions)
    }

    fn predict_many(&self, xs: &ArraySlice) -> DliResult<Vec<usize>> {
        let dim = xs.len() / self.input_shape;
        let batch_size = 4096;

        let mut predictions = Vec::with_capacity(dim);

        for chunk in xs.chunks(batch_size * self.input_shape) {
            let chunk_dim = chunk.len() / self.input_shape;
            let dataset = Tensor::from_slice(chunk, (chunk_dim, self.input_shape), &self.device)?;
            let rs = self.model.forward(&dataset)?.argmax(1)?.to_vec1::<u32>()?;
            predictions.extend(rs.into_iter().map(|v| v as usize));
        }
        Ok(predictions)
    }

    fn train(&mut self, xs: &ArraySlice) -> DliResult<()> {
        let sample_size = sampling::select_sample_size(
            self.labels,
            xs.len() / self.input_shape,
            self.train_params.threshold_samples,
        );
        debug!(sample_size = sample_size, total = xs.len() / self.input_shape ; "model:train");
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
        let mut opt = candle_nn::AdamW::new(self.varmap.all_vars(), optim_config)?;
        self._train(&xs, &ys, &mut opt)?;
        Ok(())
    }

    fn retrain(&mut self, _xs: &ArraySlice) -> DliResult<()> {
        match self.train_params.retrain_strategy {
            RetrainStrategy::NoRetrain => {
                debug!("No retraining performed as per strategy.");
            }
            RetrainStrategy::FromScratch => {
                reset_model(&mut self.varmap, &self.device)?;
                self.train(_xs)?;
            }
        };
        Ok(())
    }

    fn dump(&self, weights_filename: PathBuf) -> DliResult<ModelConfig> {
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
            weights_path: Some(weights_filename),
        })
    }

    fn memory_usage(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();

        let varmap_size: usize = self
            .varmap
            .all_vars()
            .iter()
            .map(|var| var.elem_count() * var.dtype().size_in_bytes())
            .sum();

        if varmap_size > 0 {
            total += varmap_size;
        } else {
            for layer in &self.model.layers {
                match layer {
                    CandleModelLayer::Linear(lin) => {
                        total += lin.weight().elem_count() * lin.weight().dtype().size_in_bytes();
                        if let Some(bias) = lin.bias() {
                            total += bias.elem_count() * bias.dtype().size_in_bytes();
                        }
                    }
                    CandleModelLayer::ReLU => {}
                }
            }
        }
        total
    }

    fn vec2tensor(&self, xs: &[f32]) -> DliResult<Tensor> {
        Tensor::from_slice(
            xs,
            (xs.len() / self.input_shape, self.input_shape),
            &self.device,
        )?
        .to_dtype(DType::F32)
        .map_err(|e| e.into())
    }

    fn input_shape(&self) -> usize {
        self.input_shape
    }
}

impl Model {
    #[log_time]
    fn _train(&mut self, xs: &ArraySlice, ys: &[i64], opt: &mut candle_nn::AdamW) -> DliResult<()> {
        let weighted_index = Self::weighted_index(ys)?;

        let dataset_tensor = Tensor::from_slice(
            xs,
            (xs.len() / self.input_shape, self.input_shape),
            &self.device,
        )?;
        let labels_tensor = Tensor::from_slice(ys, (ys.len(),), &self.device)?;

        let mut rng = rng();
        let total_samples = ys.len();
        let batch_size = self.train_params.batch_size;
        let generate_indices = |rng: &mut _| {
            Tensor::from_iter(
                (0..total_samples).map(|_| weighted_index.sample(rng) as u32),
                &self.device,
            )
        };
        let mut indices = generate_indices(&mut rng)?;
        for epoch in 0..self.train_params.epochs {
            let mut start = 0;
            while start < total_samples {
                let end = std::cmp::min(start + batch_size, total_samples);
                let batch_idx_tensor = indices.i(start..end)?;

                let batch_xs = dataset_tensor.index_select(&batch_idx_tensor, 0)?;
                let batch_ys = labels_tensor.index_select(&batch_idx_tensor, 0)?;

                let logits = self.model.forward(&batch_xs)?;
                let log_sm = ops::log_softmax(&logits, D::Minus1)?;
                let loss = loss::nll(&log_sm, &batch_ys)?;
                opt.backward_step(&loss)?;

                start = end;
            }
            if epoch < self.train_params.epochs - 1 {
                indices = generate_indices(&mut rng)?;
            }
        }
        Ok(())
    }

    fn weighted_index(ys: &[i64]) -> DliResult<WeightedIndex<f64>> {
        let mut class_counts = HashMap::new();
        for &y in ys {
            *class_counts.entry(y).or_insert(0) += 1;
        }

        let weights: Vec<f64> = ys
            .iter()
            .map(|&y| {
                let count = class_counts.get(&y).unwrap_or(&1);
                1.0 / (*count as f64)
            })
            .collect();

        WeightedIndex::new(&weights)
            .map_err(|_| DliError::ModelCreation("Failed to create WeightedIndex"))
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
    use crate::model::{BaseModelBuilder, ModelInterface, RetrainStrategy};

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
        let model = BaseModelBuilder::<CandleBackend>::default()
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
        assert_eq!(model.labels, labels);
        assert_eq!(
            model.train_params.threshold_samples,
            train_params.threshold_samples
        );
        assert_eq!(model.train_params.batch_size, train_params.batch_size);
        assert_eq!(model.train_params.epochs, train_params.epochs);
        assert_eq!(model.train_params.max_iters, train_params.max_iters);

        // Verify the model has the correct number of layers
        // We added 4 layers + 1 final output layer = 5 total
        assert_eq!(model.model.layers.len(), 5);
    }

    #[test]
    fn test_model_save_and_load_weights() {
        use tempfile::NamedTempFile;

        // Create and train a model
        let mut builder = BaseModelBuilder::<CandleBackend>::default();
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
        let mut loaded_builder = BaseModelBuilder::<CandleBackend>::default();
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
        let mut builder = BaseModelBuilder::<CandleBackend>::default();
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
        let mut builder = BaseModelBuilder::<CandleBackend>::default();
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
