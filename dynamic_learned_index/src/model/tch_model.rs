use std::path::PathBuf;
use std::sync::Mutex;

use log::debug;
use tch::{
    nn::{self, OptimizerConfig},
    vision::dataset::Dataset,
    Device, Tensor,
};

use crate::{
    clustering::{self},
    errors::{DliError, DliResult},
    model::{ModelDevice, ModelLayer, TchBackend, TrainParams},
    sampling,
    structs::LabelMethod,
    types::ArraySlice,
    ModelConfig,
};

fn to_tch_device(device: ModelDevice) -> Device {
    match device {
        ModelDevice::Cpu => Device::Cpu,
        ModelDevice::Gpu(gpu_no) => Device::Cuda(gpu_no),
    }
}

impl crate::model::BaseModelBuilder<TchBackend> {
    pub fn build(&self) -> DliResult<Model> {
        let device_mdl = self
            .device
            .as_ref()
            .ok_or(DliError::MissingAttribute("device"))?;
        let device = to_tch_device(*device_mdl);
        let label_method = self
            .label_method
            .ok_or(DliError::MissingAttribute("label_method"))?;
        let mut vs = nn::VarStore::new(device);
        let vs_root = vs.root();
        let input_nodes = self
            .input_nodes
            .ok_or(DliError::MissingAttribute("input_nodes"))?;
        let labels = self.labels.ok_or(DliError::MissingAttribute("labels"))?;
        assert!(labels > 0, "labels must be greater than 0");
        let mut i = 0;
        let (mut model, output_nodes) =
            self.layers
                .iter()
                .fold((nn::seq(), input_nodes), |(model, input_nodes), layer| {
                    let (model, output_nodes) = match layer {
                        ModelLayer::Linear(nodes) => {
                            let nodes = *nodes as i64;
                            let r = (
                                model.add(nn::linear(
                                    &vs_root / format!("{i}", i = 2 * i),
                                    input_nodes,
                                    nodes,
                                    Default::default(),
                                )),
                                nodes,
                            );
                            i += 1;
                            r
                        }
                        ModelLayer::ReLU => (model.add_fn(|xs| xs.relu()), input_nodes),
                    };
                    (model, output_nodes)
                });
        model = model.add(nn::linear(
            &vs_root / format!("{i}", i = 2 * i),
            output_nodes,
            labels as i64,
            Default::default(),
        ));
        if let Some(path) = &self.weights_path {
            vs.load(path)?;
        }
        let train_params = self.train_params.unwrap_or_default();
        let model = Model {
            model: Mutex::new(Box::new(model)),
            vs,
            labels,
            device,
            train_params,
            input_shape: input_nodes as usize,
            label_method,
            layers: self.layers.clone(),
        };
        Ok(model)
    }
}

// todo reset model after flush
#[derive(Debug)]
pub struct Model {
    model: Mutex<Box<dyn nn::Module>>,
    vs: nn::VarStore,
    labels: usize,
    device: Device,
    pub input_shape: usize,
    train_params: TrainParams,
    label_method: LabelMethod,
    layers: Vec<ModelLayer>,
}

impl crate::model::ModelInterface for Model {
    type TensorType = Tensor;

    fn predict(&self, xs: &Tensor) -> DliResult<Vec<(usize, f32)>> {
        let xs = match self.device {
            Device::Cpu => xs.shallow_clone(),
            _ => xs.to_device(self.device),
        };
        let predictions = tensor2vec(&self.model.lock().unwrap().forward(&xs).softmax(-1, tch::Kind::Float));
        let mut predictions = predictions.into_iter().enumerate().collect::<Vec<_>>();
        predictions.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert!(predictions.len() <= self.labels);
        Ok(predictions)
    }

    fn predict_many(&self, xs: &ArraySlice) -> DliResult<Vec<usize>> {
        let xs_tensor = Tensor::from_slice(xs);
        let xs_tensor = xs_tensor.view((
            (xs.len() / self.input_shape) as i64,
            self.input_shape as i64,
        ));
        let labels = self.model.lock().unwrap().forward(&xs_tensor).argmax(1, false);
        Ok(tensor2vec_usize(&labels))
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
        assert_eq!(ys.len(), sample_size);
        let dataset = self.dataset(&xs, &ys);
        let mut opt = nn::Adam::default().build(&self.vs, 1e-3).unwrap();
        for _ in 0..self.train_params.epochs {
            for (xs, ys) in dataset
                .train_iter(self.train_params.batch_size as i64)
                .shuffle()
            {
                let loss = self.model.lock().unwrap().forward(&xs).cross_entropy_for_logits(&ys);
                opt.backward_step(&loss);
            }
        }
        Ok(())
    }

    fn retrain(&mut self, _xs: &ArraySlice) -> DliResult<()> {
        Ok(())
    }

    fn dump(&self, weights_filename: PathBuf) -> DliResult<ModelConfig> {
        self.vs.save(&weights_filename)?;
        Ok(ModelConfig {
            train_params: self.train_params,
            weights_path: Some(weights_filename),
            layers: self.layers.clone(),
        })
    }

    fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + (self
                .vs
                .variables()
                .into_values()
                .map(|tensor| {
                    let numel = tensor.size().iter().product::<i64>() as u64;
                    let element_size = match tensor.kind() {
                        tch::Kind::Float => 4,
                        tch::Kind::Double => 8,
                        tch::Kind::Int64 => 8,
                        tch::Kind::Int => 4,
                        _ => 4, // default to 4 bytes
                    };
                    numel * element_size
                })
                .sum::<u64>() as usize)
    }

    fn vec2tensor(&self, xs: &[f32]) -> DliResult<Tensor> {
        Ok(vec2tensor(xs))
    }
}

impl Model {
    pub fn dataset(&self, xs: &[f32], ys: &[i64]) -> Dataset {
        let total_queries = ys.len();
        assert!(xs.len().is_multiple_of(self.input_shape));
        assert!(xs.len() / self.input_shape == ys.len());
        let xs = Tensor::from_slice(xs);
        let xs = xs.view((
            xs.size()[0] / self.input_shape as i64,
            self.input_shape as i64,
        ));
        let xs = match self.device {
            Device::Cpu => xs,
            _ => xs.to_device(self.device),
        };
        assert!(
            xs.size()[0] as usize == total_queries,
            "{} != {total_queries}, {:?}",
            xs.size()[0],
            xs.size()
        );
        let ys = Tensor::from_slice(ys).to_kind(tch::Kind::Int64);
        let ys = match self.device {
            Device::Cpu => ys,
            _ => ys.to_device(self.device),
        };
        assert!(xs.size()[0] == ys.size()[0]);
        assert!(xs.size()[0] == ys.size()[0]);
        assert!(ys.kind() == tch::Kind::Int64);
        let options = (xs.kind(), xs.device());
        Dataset {
            train_images: xs,
            train_labels: ys,
            test_images: Tensor::empty(0, options),
            test_labels: Tensor::empty(0, options),
            labels: self.labels as i64,
        }
    }

    pub fn dump(&self, weights_filename: PathBuf) -> DliResult<ModelConfig> {
        self.vs.save(&weights_filename)?;
        Ok(ModelConfig {
            train_params: self.train_params,
            weights_path: Some(weights_filename),
            layers: self.layers.clone(),
        })
    }

    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + (self
                .vs
                .variables()
                .into_values()
                .map(|tensor| {
                    let numel = tensor.size().iter().product::<i64>() as u64;
                    let element_size = match tensor.kind() {
                        tch::Kind::Float => 4,
                        tch::Kind::Double => 8,
                        tch::Kind::Int64 => 8,
                        tch::Kind::Int => 4,
                        _ => 4, // default to 4 bytes
                    };
                    numel * element_size
                })
                .sum::<u64>() as usize)
    }
}

fn tensor2vec(tensor: &tch::Tensor) -> Vec<f32> {
    tensor.try_into().unwrap()
}

fn tensor2vec_usize(tensor: &tch::Tensor) -> Vec<usize> {
    let x: Vec<i64> = tensor.try_into().unwrap();
    x.iter().map(|&v| v as usize).collect()
}

fn vec2tensor(vec: &ArraySlice) -> tch::Tensor {
    tch::Tensor::from_slice(vec)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_device_to_tch_device() {
        let cpu_device = ModelDevice::Cpu;
        assert!(matches!(to_tch_device(cpu_device), tch::Device::Cpu));

        let gpu_device = ModelDevice::Gpu(0);
        assert!(matches!(to_tch_device(gpu_device), tch::Device::Cuda(0)));
    }
}
