use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_core::{DType, IndexOp as _, Tensor, D};
use candle_core::{Device, Result as CandleResult};
use candle_nn::{linear, Module, Optimizer, Sequential, VarBuilder, VarMap};
use candle_nn::{loss, ops};
use log::debug;
use measure_time_macro::log_time;
use rand::distr::weighted::WeightedIndex;
use rand::distr::Distribution;
use rand::rng;
use std::collections::HashMap;
use std::path::PathBuf;

use crate::errors::{DliError, DliResult};
use crate::model::{CandleBackend, ModelDevice, ModelLayer, RetrainStrategy, TrainParams};
use crate::structs::{FloatElement, LabelMethod};
use crate::types::ArraySlice;
use crate::{clustering, sampling, ModelConfig};

impl<F: FloatElement> crate::model::BaseModelBuilder<CandleBackend, F> {
    pub fn build(&self) -> DliResult<Model<F>> {
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
        let mut varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let input_nodes = self
            .input_nodes
            .ok_or(DliError::MissingAttribute("input_nodes"))? as usize;
        let labels = self.labels.ok_or(DliError::MissingAttribute("labels"))?;
        assert!(labels > 0, "labels must be greater than 0");
        let train_params = self.train_params.unwrap_or_default();
        let mut i = 0;
        let (seq, in_nodes) = self.layers.iter().try_fold(
            (candle_nn::seq(), input_nodes),
            |(seq, input_nodes), layer| -> DliResult<(Sequential, usize)> {
                let (layers, output_nodes) = match layer {
                    ModelLayer::Linear(nodes) => {
                        let seq = seq.add(linear(
                            input_nodes,
                            *nodes,
                            vs.pp(format!("{i}", i = 2 * i)),
                        )?);
                        i += 1;

                        (seq, *nodes)
                    }
                    ModelLayer::ReLU => {
                        let seq = seq.add_fn(|xs| xs.relu());
                        (seq, input_nodes)
                    }
                };
                Ok((layers, output_nodes))
            },
        )?;
        let seq = seq.add(linear(in_nodes, labels, vs.pp(format!("{i}", i = 2 * i)))?);

        if let Some(weights_path) = &self.weights_path {
            varmap.load(weights_path)?;
        }

        let mut model = Model {
            model: seq,
            varmap,
            labels,
            device,
            train_params,
            input_shape: input_nodes,
            label_method,
            layers: self.layers.clone(),
            use_quantization: self.quantize,
            seed: self.seed,
            _marker: std::marker::PhantomData,
            quant_model: None,
        };
        if self.quantize && self.weights_path.is_some() {
            model.quantize()?;
        }
        Ok(model)
    }
}

pub struct Model<F: FloatElement> {
    model: Sequential,
    varmap: VarMap,
    pub labels: usize,
    device: Device,
    pub input_shape: usize,
    pub train_params: TrainParams,
    label_method: LabelMethod,
    layers: Vec<ModelLayer>,
    use_quantization: bool,
    seed: u64,
    quant_model: Option<Sequential>,
    _marker: std::marker::PhantomData<F>,
}

// UNSAFE: TODO
unsafe impl<F: FloatElement> Send for Model<F> {}

// UNSAFE: TODO
unsafe impl<F: FloatElement> Sync for Model<F> {}

impl<F: FloatElement> crate::model::ModelInterface<F> for Model<F> {
    type TensorType = Tensor;

    fn predict(&self, xs: &Tensor) -> DliResult<Vec<(usize, f32)>> {
        let logits = match self.use_quantization {
            true => self.quant_model.as_ref().unwrap().forward(xs)?,
            false => self.model.forward(xs)?,
        };
        let final_result = ops::softmax(&logits, D::Minus1)?;
        let predictions = final_result.squeeze(0)?.to_vec1::<f32>()?;
        let predictions = predictions.into_iter().enumerate().collect::<Vec<_>>();
        assert!(predictions.len() <= self.labels);
        Ok(predictions)
    }

    #[log_time]
    fn predict_many(&self, xs: &[F]) -> DliResult<Vec<Vec<(usize, f32)>>> {
        let dim = xs.len() / self.input_shape;
        let batch_size = 4096;

        let mut predictions: Vec<Vec<(usize, f32)>> = Vec::with_capacity(dim);

        for chunk in xs.chunks(batch_size * self.input_shape) {
            let chunk_dim = chunk.len() / self.input_shape;
            let dataset = Tensor::from_slice(chunk, (chunk_dim, self.input_shape), &self.device)?
                .to_dtype(DType::F32)?;
            let res = match self.use_quantization {
                true => self.quant_model.as_ref().unwrap().forward(&dataset)?,
                false => self.model.forward(&dataset)?,
            };
            let res = ops::softmax(&res, D::Minus1)?;
            let res = res.to_vec2::<f32>()?;
            let res: Vec<Vec<(usize, f32)>> = res
                .into_iter()
                .map(|preds| preds.into_iter().enumerate().collect())
                .collect();
            predictions.extend(res);
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
        let xs = sampling::sample(xs, sample_size, self.input_shape, self.seed);

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
        if self.use_quantization {
            self.quantize()?;
        }
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
            layers: self.layers.clone(),
            train_params: self.train_params,
            weights_path: Some(weights_filename),
            quantize: self.use_quantization,
            seed: self.seed,
        })
    }

    fn memory_usage(&self) -> usize {
        let mut total = std::mem::size_of::<Self>();

        // Calculate memory used by all variables in varmap (weights and biases)
        let varmap_size: usize = self
            .varmap
            .all_vars()
            .iter()
            .map(|var| var.elem_count() * var.dtype().size_in_bytes())
            .sum();

        total += varmap_size;
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
}

impl<F: FloatElement> Model<F> {
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

    pub fn quantize(&mut self) -> DliResult<()> {
        let mut quantized_seq = candle_nn::seq();
        let vars = self.varmap.data().lock().unwrap();
        let mut linear_idx = 0;
        let load_linear = |linear_idx| -> DliResult<QuantizedLinear> {
            let weight_name = format!("{}.weight", linear_idx * 2);
            let bias_name = format!("{}.bias", linear_idx * 2);
            let weight_var = vars
                .get(&weight_name)
                .unwrap_or_else(|| panic!("Weight tensor '{}' not found in VarMap", weight_name));
            let weight = weight_var.as_tensor();
            let q_weight = QTensor::quantize(weight, GgmlDType::F16)?;
            let q_matmul = QMatMul::from_qtensor(q_weight)?;
            let bias = vars.get(&bias_name).map(|v| v.as_tensor().clone()).unwrap();
            Ok(QuantizedLinear { q_matmul, bias })
        };

        for layer in &self.layers {
            match layer {
                ModelLayer::Linear(_out_features) => {
                    let q_linear = load_linear(linear_idx)?;
                    quantized_seq = quantized_seq.add(q_linear);
                    linear_idx += 1;
                }
                ModelLayer::ReLU => {
                    quantized_seq = quantized_seq.add(candle_nn::Activation::Relu);
                }
            }
        }
        let q_linear = load_linear(linear_idx)?;
        quantized_seq = quantized_seq.add(q_linear);
        self.quant_model = Some(quantized_seq);
        Ok(())
    }
}

#[derive(Debug)]
pub struct QuantizedLinear {
    pub q_matmul: QMatMul,
    pub bias: Tensor,
}

impl Module for QuantizedLinear {
    fn forward(&self, xs: &Tensor) -> CandleResult<Tensor> {
        let out = self.q_matmul.forward(xs)?;
        out.broadcast_add(&self.bias)
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
