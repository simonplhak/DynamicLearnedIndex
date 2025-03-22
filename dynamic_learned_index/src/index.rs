use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use tch::{nn, Device, Tensor};

use crate::{
    bucket::{self, StaticBucket},
    errors::BuildError,
    model::{self, ModelConfig},
    Id,
};

#[derive(Debug, Serialize, Deserialize, Default)]
pub enum ModelDevice {
    #[default]
    #[serde(rename = "cpu")]
    Cpu,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Index {
    #[serde(rename = "levels")]
    levels_config: HashMap<usize, LevelIndexConfig>,
    #[serde(rename = "bucket")]
    bucket_config: bucket::BucketConfig,
    input_shape: i64,
    arity: i64,
    device: ModelDevice,
    #[serde(skip)]
    levels: Vec<LevelIndex>,
}

impl Index {
    pub fn validate(&self) -> Result<(), BuildError> {
        if self.levels_config.is_empty() {
            return Err(BuildError::MissingAttribute);
        }
        if !self.levels_config.contains_key(&0) {
            return Err(BuildError::MissingAttribute);
        }
        Ok(())
    }
    pub fn search(&self, key: &Tensor) -> Tensor {
        let res = self
            .levels
            .iter()
            .map(|level_index| level_index.search(key))
            .collect::<Vec<_>>();
        todo!()
    }

    pub fn insert(&mut self, value: Tensor) {
        // if !self.has_space() {
        //     self.add_level();
        // }
        // todo!()
    }

    fn get_level_index_config(&self) -> LevelIndexConfig {
        let curr_level = self.levels.len();
        // self.levels_config.iter().find_map(|level_config| {
        //     level_config.get(&curr_level).map(|config| config.clone())
        // })
        self.levels_config
            .iter()
            .take_while(|(level, _)| **level <= curr_level)
            .last()
            .map(|(_, config)| config.to_owned())
            .unwrap()
    }

    fn add_level(&mut self) {
        let level_index_config = self.get_level_index_config();
        let level_index = LevelIndexBuilder::default()
            .size(self.arity)
            .input_shape(self.input_shape) // todo this should be self.artity ** level, but waiting for buffer specification
            .model(level_index_config.model.clone())
            .bucket(self.bucket_config.clone())
            .build()
            .unwrap();
        self.levels.push(level_index);
        println!("Added level");
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct LevelIndexConfig {
    pub model: ModelConfig,
}

#[derive(Debug, Default)]
pub struct LevelIndexBuilder {
    size: Option<i64>,
    model_config: Option<ModelConfig>,
    bucket_config: Option<bucket::BucketConfig>,
    input_shape: Option<i64>,
}

impl LevelIndexBuilder {
    pub fn size(&mut self, size: i64) -> &mut Self {
        self.size = Some(size);
        self
    }

    pub fn model(&mut self, model: ModelConfig) -> &mut Self {
        self.model_config = Some(model);
        self
    }

    pub fn bucket(&mut self, bucket: bucket::BucketConfig) -> &mut Self {
        self.bucket_config = Some(bucket);
        self
    }

    pub fn input_shape(&mut self, input_shape: i64) -> &mut Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn build(&self) -> Result<LevelIndex, BuildError> {
        let size = self.size.ok_or(BuildError::MissingAttribute)?;
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;
        let model_config = self
            .model_config
            .as_ref()
            .ok_or(BuildError::MissingAttribute)?;
        let bucket_config = self
            .bucket_config
            .as_ref()
            .ok_or(BuildError::MissingAttribute)?;
        let mut model_builder = model::ModelBuilder::default();
        model_builder.device(Device::Cpu).input_nodes(input_shape);
        model_config.layers.iter().for_each(|layer| {
            model_builder.add_layer(*layer);
        });
        let model = model_builder.build()?;
        let mut bucket_builder = bucket::BucketBuilder::default();
        bucket_builder
            .input_shape(input_shape)
            .size(size)
            .bucket_type(bucket_config.bucket_type);
        let buckets = (0..size)
            .map(|_| bucket_builder.build())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(LevelIndex { model, buckets })
    }
}

#[derive(Debug)]
pub struct LevelIndex {
    model: Box<dyn nn::Module>,
    buckets: Vec<Box<dyn bucket::Bucket>>,
}

impl LevelIndex {
    pub fn search(&self, key: &Tensor) -> Tensor {
        let bucket_idx = self.model.forward(key).argmax(0, true).int64_value(&[]) as usize;
        self.buckets[bucket_idx].search(key);
        // self.model.forward(&key)
        todo!()
    }

    pub fn insert(&mut self, value: Tensor, id: Id) {
        todo!()
    }

    fn has_space(&self) -> bool {
        self.buckets.iter().any(|bucket| bucket.has_space())
    }
}
