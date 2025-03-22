use std::{collections::HashMap, fmt};

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

#[derive(Debug, Serialize, Deserialize, Default)]
pub enum Levelling {
    #[default]
    #[serde(rename = "bentley_saxe")]
    BentleySaxe,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct IndexConfig {
    levelling: Levelling,
    levels: HashMap<usize, LevelIndexConfig>,
    buffer_size: i64,
    bucket: bucket::BucketConfig,
    input_shape: i64,
    arity: i64,
    device: ModelDevice,
}

impl IndexConfig {
    pub fn build(self) -> Result<Index, BuildError> {
        if self.levels.is_empty() {
            return Err(BuildError::MissingAttribute);
        }
        if !self.levels.contains_key(&1) {
            return Err(BuildError::MissingAttribute);
        }
        let buffer = LevelIndexBuilder::default()
            .size(self.buffer_size)
            .input_shape(self.input_shape)
            .is_buffer(true)
            .build()?;
        let levels = vec![buffer];
        let index = Index {
            levelling: self.levelling,
            levels_config: self.levels,
            bucket_config: self.bucket,
            input_shape: self.input_shape,
            arity: self.arity,
            device: self.device,
            levels,
        };
        Ok(index)
    }
}

#[derive(Debug, Default)]
pub struct Index {
    levelling: Levelling,
    levels_config: HashMap<usize, LevelIndexConfig>,
    bucket_config: bucket::BucketConfig,
    input_shape: i64,
    arity: i64,
    device: ModelDevice,
    levels: Vec<Box<dyn LevelIndex>>,
}

impl Index {
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
    levelling: Option<Levelling>,
    is_buffer: Option<bool>,
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

    pub fn levelling(&mut self, levelling: Levelling) -> &mut Self {
        self.levelling = Some(levelling);
        self
    }

    pub fn is_buffer(&mut self, is_buffer: bool) -> &mut Self {
        self.is_buffer = Some(is_buffer);
        self
    }

    pub fn build(&self) -> Result<Box<dyn LevelIndex>, BuildError> {
        if self.is_buffer.unwrap_or(false) {
            return self.build_buffer();
        }
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
        let level_index = match self.levelling {
            Some(Levelling::BentleySaxe) => BentleySaxe { model, buckets },
            None => return Err(BuildError::MissingAttribute),
        };
        Ok(Box::new(level_index))
    }

    fn build_buffer(&self) -> Result<Box<dyn LevelIndex>, BuildError> {
        let size = self.size.ok_or(BuildError::MissingAttribute)?;
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;
        Ok(Box::new(Buffer {
            records: Tensor::zeros([size, input_shape], tch::kind::FLOAT_CPU),
            ids: vec![0; size as usize],
            pointer: 0,
        }))
    }
}

enum LevelIndexError {
    Overflow,
}

trait LevelIndex {
    fn search(&self, key: &Tensor) -> Tensor;
    fn insert(&mut self, value: Tensor, id: Id) -> Result<(), LevelIndexError>;
}

impl fmt::Debug for dyn LevelIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bucket") // todo
    }
}

#[derive(Debug)]
pub struct Buffer {
    records: Tensor, // (size, ..dim)
    ids: Vec<Id>,
    pointer: i64,
}

impl Buffer {
    fn has_space(&self) -> bool {
        self.pointer < self.records.size()[0]
    }
}

impl LevelIndex for Buffer {
    fn search(&self, key: &Tensor) -> Tensor {
        todo!()
    }

    fn insert(&mut self, value: Tensor, id: Id) -> Result<(), LevelIndexError> {
        todo!()
    }
}

#[derive(Debug)]
pub struct BentleySaxe {
    model: Box<dyn nn::Module>,
    buckets: Vec<Box<dyn bucket::Bucket>>,
}

impl BentleySaxe {
    fn has_space(&self) -> bool {
        self.buckets.iter().any(|bucket| bucket.has_space())
    }
}

impl LevelIndex for BentleySaxe {
    fn search(&self, key: &Tensor) -> Tensor {
        let bucket_idx = self.model.forward(key).argmax(0, true).int64_value(&[]) as usize;
        self.buckets[bucket_idx].search(key);
        // self.model.forward(&key)
        todo!()
    }

    fn insert(&mut self, value: Tensor, id: Id) -> Result<(), LevelIndexError> {
        todo!()
    }
}
