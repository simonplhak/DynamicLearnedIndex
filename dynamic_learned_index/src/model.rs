use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, OptimizerConfig},
    vision::dataset::Dataset,
    Device, Tensor,
};

use crate::{
    clustering::{self},
    distance_fn::LabelMethod,
    errors::BuildError,
    sampling,
    types::{Array, ArraySlice},
};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct TrainParams {
    pub threshold_samples: usize,
    pub batch_size: i64,
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
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(tag = "type", content = "value")]
pub enum ModelLayer {
    #[serde(rename = "linear")]
    Linear(i64),
    #[serde(rename = "relu")]
    ReLU,
}

#[derive(Debug, Default)]
pub(crate) struct ModelBuilder {
    device: Option<Device>,
    input_nodes: Option<i64>,
    layers: Vec<ModelLayer>,
    labels: Option<i64>,
    train_params: Option<TrainParams>,
    label_method: Option<LabelMethod>,
}

impl ModelBuilder {
    pub fn device(&mut self, device: Device) -> &mut Self {
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

    pub fn labels(&mut self, labels: i64) -> &mut Self {
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

    pub fn build(&self) -> Result<Model, BuildError> {
        let device = self.device.ok_or(BuildError::MissingAttribute)?;
        let label_method = self.label_method.ok_or(BuildError::MissingAttribute)?;
        let vs = nn::VarStore::new(device);
        let vs_root = vs.root();
        let input_nodes = self.input_nodes.ok_or(BuildError::MissingAttribute)?;
        let labels = self.labels.ok_or(BuildError::MissingAttribute)?;
        assert!(labels > 0, "labels must be greater than 0");
        let (mut model, output_nodes) = self.layers.iter().enumerate().fold(
            (nn::seq(), input_nodes),
            |(model, input_nodes), (i, layer)| {
                let (model, output_nodes) = match layer {
                    ModelLayer::Linear(nodes) => (
                        model.add(nn::linear(
                            &vs_root / format!("layer_{i}"),
                            input_nodes,
                            *nodes,
                            Default::default(),
                        )),
                        *nodes,
                    ),
                    ModelLayer::ReLU => (model.add_fn(|xs| xs.relu()), input_nodes),
                };
                (model, output_nodes)
            },
        );
        model = model.add(nn::linear(
            &vs_root / "output",
            output_nodes,
            labels,
            Default::default(),
        ));
        let train_params = self.train_params.clone().unwrap_or_default();
        let model = Model {
            model: Box::new(model),
            vs,
            labels,
            device,
            train_params,
            input_shape: input_nodes as usize,
            label_method,
        };
        Ok(model)
    }
}

// todo reset model after flush
#[derive(Debug)]
pub(crate) struct Model {
    model: Box<dyn nn::Module>,
    vs: nn::VarStore,
    labels: i64,
    device: Device,
    input_shape: usize,
    train_params: TrainParams,
    label_method: LabelMethod,
}

impl Model {
    /// returns vec of tuples (label, confidence) sorted by confidence
    pub fn predict(&self, xs: &ArraySlice) -> Vec<(usize, f32)> {
        let xs = vec2tensor(xs).to_device(self.device);
        let predictions = tensor2vec(&self.model.forward(&xs));
        let mut predictions = predictions.into_iter().enumerate().collect::<Vec<_>>();
        predictions.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        assert!(predictions.len() <= self.labels as usize);
        predictions
    }

    pub fn predict_many(&self, xs: &[Array]) -> Vec<usize> {
        let xs = Tensor::cat(
            &xs.iter()
                .map(|x| vec2tensor(x).unsqueeze(0))
                .collect::<Vec<_>>(),
            0,
        )
        .to_device(self.device);
        let labels = self.model.forward(&xs).argmax(1, false);
        tensor2vec_usize(&labels)
    }

    pub fn train(&mut self, xs: &[Array], k: usize) {
        let xs = sampling::sample(xs, self.train_params.threshold_samples);
        let ys = clustering::compute_labels(
            &xs,
            &self.label_method,
            k,
            self.input_shape,
            self.train_params.max_iters,
        );
        let dataset = self.dataset(&xs, &ys);
        let mut opt = nn::Adam::default().build(&self.vs, 1e-3).unwrap();
        for _ in 0..self.train_params.epochs {
            for (xs, ys) in dataset.train_iter(self.train_params.batch_size).shuffle() {
                let loss = self.model.forward(&xs).cross_entropy_for_logits(&ys);
                opt.backward_step(&loss);
            }
        }
    }

    fn dataset(&self, xs: &[f32], ys: &[i32]) -> Dataset {
        let total_queries = ys.len();
        assert!(xs.len() % self.input_shape == 0);
        assert!(xs.len() / self.input_shape == ys.len());
        let tensors = (0..total_queries)
            .map(|i| {
                Tensor::from_slice(&xs[i * self.input_shape..(i + 1) * self.input_shape])
                    .unsqueeze(0)
            })
            .collect::<Vec<_>>();
        let xs = Tensor::cat(&tensors, 0).to_device(self.device);
        assert!(
            xs.size()[0] as usize == total_queries,
            "{} != {total_queries}, {:?}",
            xs.size()[0],
            xs.size()
        );
        let ys = Tensor::from_slice(ys)
            .to_kind(tch::Kind::Int64)
            .to_device(self.device);
        assert!(xs.size()[0] == ys.size()[0]);
        assert!(xs.size()[0] == ys.size()[0]);
        assert!(ys.kind() == tch::Kind::Int64);
        let options = (xs.kind(), xs.device());
        Dataset {
            train_images: xs,
            train_labels: ys,
            test_images: Tensor::empty(0, options),
            test_labels: Tensor::empty(0, options),
            labels: self.labels,
        }
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
    tch::Tensor::from_slice(vec).to_kind(tch::kind::Kind::Float)
}
