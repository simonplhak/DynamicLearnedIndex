use log::info;
use serde::{Deserialize, Serialize};
use tch::{
    nn::{self, OptimizerConfig},
    vision::dataset::Dataset,
    Device, Tensor,
};

use crate::errors::BuildError;

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct ModelConfig {
    pub layers: Vec<ModelLayer>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(tag = "type", content = "value")]
pub enum ModelLayer {
    Linear(i64),
    ReLU,
}

#[derive(Debug)]
pub(crate) struct ModelBuilder {
    device: Device,
    input_nodes: Option<i64>,
    layers: Vec<ModelLayer>,
    labels: Option<i64>,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            layers: Vec::new(),
            input_nodes: None,
            labels: None,
        }
    }
}

impl ModelBuilder {
    pub fn device(&mut self, device: Device) -> &mut Self {
        self.device = device;
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

    pub fn build(&self) -> Result<Model, BuildError> {
        let vs = nn::VarStore::new(self.device);
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
                            &vs_root / format!("layer_{}", i),
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
        let model = Model {
            model: Box::new(model),
            vs,
            labels,
        };
        Ok(model)
    }
}

#[derive(Debug)]
pub(crate) struct Model {
    model: Box<dyn nn::Module>,
    vs: nn::VarStore,
    labels: i64,
}

impl Model {
    pub fn predict(&self, xs: &tch::Tensor) -> usize {
        let label = self.model.forward(xs).argmax(0, false).int64_value(&[]);
        assert!(
            label >= 0 && label < self.labels,
            "label out of range: {}",
            label
        );
        label as usize
    }

    pub fn train(&mut self, xs: Tensor, ys: Tensor) {
        info!(queries=xs.size()[0]; "model:train_started");
        let dataset = self.dataset(xs, ys);
        let batch_size = 32; // todo take from config
        let mut opt = nn::Adam::default().build(&self.vs, 1e-3).unwrap(); // todo handle unwrap
        for _ in 1..3 {
            for (xs, ys) in dataset.train_iter(batch_size) {
                let loss = self.model.forward(&xs).cross_entropy_for_logits(&ys);
                opt.backward_step(&loss);
            }
        }
        info!("model:train_finished");
    }

    fn dataset(&self, xs: Tensor, ys: Tensor) -> Dataset {
        assert!(
            xs.size()[0] == ys.size()[0],
            "xs and ys must have the same size: {} != {}",
            xs.size()[0],
            ys.size()[0]
        );
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
