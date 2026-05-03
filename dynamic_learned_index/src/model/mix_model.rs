use std::path::PathBuf;

use crate::model::candle_model;
use crate::model::tch_model;
use crate::model::BaseModelBuilder;
use crate::model::CandleBackend;
use crate::model::ModelInterface as _;
use crate::model::TchBackend;
use crate::structs::FloatElement;
use crate::DliError;
use candle_core::Tensor;
#[cfg(feature = "measure_time")]
use log::debug;
#[cfg(feature = "measure_time")]
use measure_time_macro::log_time;
use rand::Rng as _;

use crate::{model::MixBackend, ArraySlice, DliResult, ModelConfig};

impl<F: FloatElement> crate::model::BaseModelBuilder<MixBackend, F> {
    pub fn build(&self) -> DliResult<Model<F>> {
        let device = self.device.ok_or(DliError::MissingAttribute("device"))?;
        let input_nodes = self
            .input_nodes
            .ok_or(DliError::MissingAttribute("input_nodes"))?;
        let labels = self.labels.ok_or(DliError::MissingAttribute("labels"))?;
        let train_params = self.train_params.unwrap_or_default();
        let label_method = self
            .label_method
            .ok_or(DliError::MissingAttribute("label_method"))?;
        let mut tch_builder = BaseModelBuilder::<TchBackend, F>::default();
        tch_builder
            .device(device)
            .input_nodes(input_nodes)
            .labels(labels)
            .train_params(train_params)
            .label_method(label_method)
            .layers(self.layers.clone());
        let mut candle_builder = BaseModelBuilder::<CandleBackend, F>::default();
        candle_builder
            .device(device)
            .input_nodes(input_nodes)
            .labels(labels)
            .train_params(train_params)
            .label_method(label_method)
            .quantize(self.quantize)
            .layers(self.layers.clone());
        if let Some(weights_path) = self.weights_path.clone() {
            tch_builder.weights_path(weights_path.clone());
        }
        let mut model = Model {
            tch_model: tch_builder.build()?,
            candle_model: candle_builder.build()?,
            candle_builder,
            input_shape: input_nodes as usize,
        };
        model.sync_weights_from_tch_to_candle()?;
        Ok(model)
    }
}

pub struct Model<F: FloatElement> {
    tch_model: tch_model::Model<F>,
    candle_model: candle_model::Model<F>,
    candle_builder: BaseModelBuilder<CandleBackend, F>,
    pub input_shape: usize,
}

impl<F: FloatElement> crate::model::ModelInterface<F> for Model<F> {
    type TensorType = Tensor;

    fn predict(&self, xs: &Self::TensorType) -> DliResult<Vec<(usize, f32)>> {
        self.candle_model.predict(xs)
    }

    #[log_time]
    fn predict_many(&self, xs: &[F]) -> DliResult<Vec<usize>> {
        self.candle_model.predict_many(xs)
    }

    fn train(&mut self, xs: &ArraySlice) -> DliResult<()> {
        self.tch_model.train(xs)?;
        self.sync_weights_from_tch_to_candle()
    }

    fn retrain(&mut self, xs: &ArraySlice) -> DliResult<()> {
        self.tch_model.retrain(xs)?;
        self.sync_weights_from_tch_to_candle()
    }

    fn dump(&self, weights_filename: PathBuf) -> DliResult<ModelConfig> {
        self.tch_model.dump(weights_filename)
    }

    fn memory_usage(&self) -> usize {
        self.candle_model.memory_usage() + self.tch_model.memory_usage()
    }

    fn vec2tensor(&self, xs: &[f32]) -> DliResult<Self::TensorType> {
        self.candle_model.vec2tensor(xs)
    }
}

impl<F: FloatElement> Model<F> {
    pub fn sync_weights_from_tch_to_candle(&mut self) -> DliResult<()> {
        let random_id = rand::rng().random::<u64>();
        let weights_path = std::path::PathBuf::from("/tmp")
            .join(format!("mix_model_weights_{}.safetensors", random_id));
        self.tch_model.dump(weights_path.clone())?;
        self.candle_model = self.candle_builder.weights_path(weights_path).build()?;
        Ok(())
    }
}
