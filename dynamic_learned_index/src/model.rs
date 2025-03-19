use serde::{Deserialize, Serialize};
use tch::{nn, Device};

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

    pub fn build(&self) -> Result<Box<dyn nn::Module>, BuildError> {
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
        Ok(Box::new(model))
    }
}
