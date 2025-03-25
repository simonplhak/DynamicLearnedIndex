use serde::{Deserialize, Serialize};
use tch::{nn, Device, Tensor};

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
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            device: Device::Cpu,
            layers: Vec::new(),
            input_nodes: None,
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

    pub fn build(&self) -> Result<Model, BuildError> {
        let vs = nn::VarStore::new(self.device);
        let vs_root = vs.root();
        let input_nodes = self.input_nodes.ok_or(BuildError::MissingAttribute)?;
        let (model, _) = self.layers.iter().enumerate().fold(
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
        let model = Model {
            model: Box::new(model),
            vs,
        };
        Ok(model)
    }
}

#[derive(Debug)]
pub(crate) struct Model {
    model: Box<dyn nn::Module>,
    vs: nn::VarStore,
}

impl Model {
    pub fn predict(&self, xs: &tch::Tensor) -> usize {
        self.model.forward(xs).argmax(0, false).int64_value(&[]) as usize
    }

    pub fn train(&mut self, queries: &[Tensor], labels: &Tensor) {
        // todo
        // let mut opt = nn::Adam::default().build(&vs, 1e-3)?;
        // for epoch in 1..200 {
        //     let loss = *self
        //         .model
        //         .forward(&m.train_images)
        //         .cross_entropy_for_logits(&m.train_labels);
        //     opt.backward_step(&loss);
        //     let test_accuracy = net
        //         .forward(&m.test_images)
        //         .accuracy_for_logits(&m.test_labels);
        //     println!(
        //         "epoch: {:4} train loss: {:8.5} test acc: {:5.2}%",
        //         epoch,
        //         f64::from(&loss),
        //         100. * f64::from(&test_accuracy),
        //     );
        // }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub(crate) enum LabelMethod {
    #[default]
    #[serde(rename = "knn")]
    Knn,
    #[serde(rename = "random")]
    Random,
}

pub(crate) fn compute_labels(data: &[Tensor], label_method: &LabelMethod, k: i64) -> Tensor {
    debug_assert!(!data.is_empty());
    match label_method {
        LabelMethod::Knn => {
            todo!()
            // let knn_index = index_factory(d, description, metric)
        }
        LabelMethod::Random => {
            let shape = data[0].size();
            Tensor::randint(k, &shape, tch::kind::INT64_CPU)
        } // todo handle device
    }
}
