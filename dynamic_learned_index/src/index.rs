use crate::{
    bucket::{self, Bucket, BucketBuilder},
    errors::BuildError,
    model::{self, Model, ModelConfig},
    types::{Array, ArraySlice},
    Id, SearchStrategy,
};
use log::{debug, info};
use measure_time_macro::log_time;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tch::Device;

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub enum ModelDevice {
    #[default]
    #[serde(rename = "cpu")]
    Cpu,
    #[serde(rename = "gpu")]
    Gpu(usize),
}

impl ModelDevice {
    pub fn to_tch_device(&self) -> Device {
        match self {
            ModelDevice::Cpu => Device::Cpu,
            ModelDevice::Gpu(gpu_no) => Device::Cuda(*gpu_no),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub enum Levelling {
    #[default]
    #[serde(rename = "bentley_saxe")]
    BentleySaxe,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    levelling: Levelling,
    levels: HashMap<usize, LevelIndexConfig>,
    pub buffer_size: usize,
    pub input_shape: usize,
    arity: usize,
    pub device: ModelDevice,
}

impl Default for IndexConfig {
    fn default() -> Self {
        let mut levels = HashMap::new();
        levels.insert(0, Default::default());
        Self {
            levelling: Default::default(),
            levels,
            buffer_size: 5000,
            input_shape: 768,
            arity: 3,
            device: Default::default(),
        }
    }
}

impl IndexConfig {
    pub fn from_yaml(file: &str) -> Result<Self, BuildError> {
        let content = std::fs::read_to_string(file).map_err(|_| BuildError::NonExistentFile)?;
        serde_yaml::from_str(&content).map_err(|e| BuildError::InvalidYamlConfig(e.to_string()))
    }

    pub fn build(self) -> Result<Index, BuildError> {
        if self.levels.is_empty() {
            return Err(BuildError::MissingAttribute);
        }
        if !self.levels.contains_key(&0) {
            return Err(BuildError::MissingAttribute);
        }
        let buffer = BucketBuilder::default()
            .id("buffer".to_string())
            .input_shape(self.input_shape)
            .size(self.buffer_size)
            .build()?;
        let index = match self.levelling {
            Levelling::BentleySaxe => {
                let index = BentleySaxeIndex {
                    levels_config: self.levels,
                    input_shape: self.input_shape,
                    arity: self.arity,
                    device: self.device,
                    levels: Vec::new(),
                    buffer,
                    search_startegy: Default::default(),
                };
                Index::BentleySaxe(index)
            }
        };
        Ok(index)
    }
}

#[derive(Debug)]
pub enum Index {
    BentleySaxe(BentleySaxeIndex),
}

impl Index {
    #[log_time]
    pub fn search(&self, query: &ArraySlice, k: usize) -> Vec<Id> {
        match self {
            Index::BentleySaxe(index) => index.search(query, k),
        }
    }

    pub fn insert(&mut self, value: Array, id: Id) {
        match self {
            Index::BentleySaxe(index) => {
                index.insert(value, id);
            }
        }
    }

    pub fn search_strategy(&mut self, search_strategy: SearchStrategy) {
        match self {
            Index::BentleySaxe(index) => {
                index.search_startegy = search_strategy;
            }
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Index::BentleySaxe(index) => index.size(),
        }
    }
}

#[derive(Debug)]
pub struct BentleySaxeIndex {
    levels_config: HashMap<usize, LevelIndexConfig>,
    input_shape: usize,
    arity: usize,
    device: ModelDevice,
    levels: Vec<LevelIndex>,
    buffer: Bucket,
    search_startegy: SearchStrategy,
}

impl BentleySaxeIndex {
    fn available_level(&self) -> Option<usize> {
        let mut count = self.buffer.size();
        self.levels
            .iter()
            .enumerate()
            .find(|(_, level)| {
                let occupied = level.occupied();
                let fits = level.size() - occupied >= count;
                if !fits {
                    count += occupied;
                }
                fits
            })
            .map(|(i, _)| i)
    }

    fn get_level_index_config(&self) -> LevelIndexConfig {
        let curr_level = self.levels.len();
        self.levels_config
            .iter()
            .take_while(|(level, _)| **level <= curr_level)
            .last()
            .map(|(_, config)| config.to_owned())
            .unwrap()
    }

    fn add_level(&mut self) -> usize {
        let level_index_config = self.get_level_index_config();
        let n_buckets = self.arity.pow(self.levels.len() as u32 + 1);
        let level_index = LevelIndexBuilder::default()
            .id(format!("{}", self.levels.len()))
            .n_buckets(n_buckets)
            .input_shape(self.input_shape)
            .model(level_index_config.model.clone())
            .model_device(self.device.clone())
            .bucket_size(level_index_config.bucket_size)
            .build()
            .unwrap();
        self.levels.push(level_index);
        self.levels.len() - 1
    }

    fn lower_level_data(&mut self, level_idx: usize) -> (Vec<Array>, Vec<Id>) {
        let (data, ids): (Vec<Vec<Array>>, Vec<Vec<Id>>) = self
            .levels
            .iter_mut()
            .take(level_idx)
            .map(|level| level.get_data())
            .unzip();
        let (buffer_data, buffer_ids) = self.buffer.get_data();
        let data = data
            .into_iter()
            .flatten()
            .chain(buffer_data)
            .collect::<Vec<_>>();
        let ids = ids
            .into_iter()
            .flatten()
            .chain(buffer_ids)
            .collect::<Vec<_>>();
        (data, ids)
    }

    fn buckets2visit(&self, query: &ArraySlice) -> Vec<&Bucket> {
        let nprobe = self.search_startegy.nprobe();
        match &self.search_startegy {
            SearchStrategy::Base(_) => self
                .levels
                .iter()
                .flat_map(|level| level.buckets2visit(query, nprobe))
                .collect(),
        }
    }

    #[log_time]
    fn search(&self, query: &ArraySlice, k: usize) -> Vec<Id> {
        let buckets2visit = self.buckets2visit(query);
        let (ids, distances): (Vec<_>, Vec<_>) = buckets2visit
            .par_iter()
            .map(|bucket| bucket.search(query, k))
            .unzip();
        let ids = ids.into_iter().flatten().collect::<Vec<_>>();
        let distances = distances.into_iter().flatten().collect::<Vec<_>>();
        let mut results = ids.into_iter().zip(distances.iter()).collect::<Vec<_>>();
        results.sort_by(|(_, dist_a), (_, dist_b)| {
            dist_a
                .partial_cmp(dist_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.into_iter().take(k).map(|(id, _)| id).collect()
    }

    fn insert(&mut self, value: Array, id: Id) {
        if self.buffer.has_space(1) {
            self.buffer.insert(value, id);
            return; // value fits into buffer
        }
        debug!(buffer_size = self.buffer.size(); "index:buffer_flush");
        match self.available_level() {
            Some(level_idx) => {
                let (data, ids) = self.lower_level_data(level_idx);
                let level = &mut self.levels[level_idx];
                level.insert(data, ids);
            }
            None => {
                let level_idx = self.add_level();
                let (data, ids) = self.lower_level_data(level_idx);
                let level = &mut self.levels[level_idx];
                level.train(&data, level.n_buckets());
                level.insert_many(data, ids);
            }
        };
    }

    fn size(&self) -> usize {
        self.levels.iter().map(|level| level.size()).sum::<usize>() + self.buffer.size()
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LevelIndexConfig {
    pub model: ModelConfig,
    pub bucket_size: usize,
}

impl Default for LevelIndexConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig::default(),
            bucket_size: 5000,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct LevelIndexBuilder {
    id: Option<String>,
    n_buckets: Option<usize>,
    model_config: Option<ModelConfig>,
    bucket_size: Option<usize>,
    input_shape: Option<usize>,
    model_device: ModelDevice,
}

impl LevelIndexBuilder {
    pub fn n_buckets(&mut self, size: usize) -> &mut Self {
        self.n_buckets = Some(size);
        self
    }

    pub fn model(&mut self, model: ModelConfig) -> &mut Self {
        self.model_config = Some(model);
        self
    }

    pub fn bucket_size(&mut self, bucket_size: usize) -> &mut Self {
        self.bucket_size = Some(bucket_size);
        self
    }

    pub fn input_shape(&mut self, input_shape: usize) -> &mut Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn id(&mut self, id: String) -> &mut Self {
        self.id = Some(id);
        self
    }

    pub fn model_device(&mut self, model_device: ModelDevice) -> &mut Self {
        self.model_device = model_device;
        self
    }

    pub fn build(&self) -> Result<LevelIndex, BuildError> {
        let n_buckets = self.n_buckets.ok_or(BuildError::MissingAttribute)?;
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;
        let id = self.id.clone().ok_or(BuildError::MissingAttribute)?;
        let bucket_size = self.bucket_size.ok_or(BuildError::MissingAttribute)?;
        let model_config = self
            .model_config
            .as_ref()
            .ok_or(BuildError::MissingAttribute)?;
        let mut model_builder = model::ModelBuilder::default();
        model_builder
            .device(self.model_device.to_tch_device())
            .input_nodes(input_shape as i64)
            .train_params(model_config.train_params.clone())
            .labels(n_buckets as i64);
        model_config.layers.iter().for_each(|layer| {
            model_builder.add_layer(*layer);
        });
        let model = model_builder.build()?;
        let mut bucket_builder = bucket::BucketBuilder::default();
        bucket_builder
            .input_shape(input_shape)
            .size(bucket_size)
            .is_dynamic(true);
        let buckets = (0..n_buckets)
            .map(|bucket_id| bucket_builder.id(format!("{id}:{bucket_id}")).build())
            .collect::<Result<Vec<_>, _>>()?;
        let level_index = LevelIndex { model, buckets };
        Ok(level_index)
    }
}

#[derive(Debug)]
pub struct LevelIndex {
    model: Model,
    buckets: Vec<Bucket>,
}

impl LevelIndex {
    fn size(&self) -> usize {
        self.buckets.iter().map(|bucket| bucket.size()).sum()
    }

    fn occupied(&self) -> usize {
        self.buckets.iter().map(|bucket| bucket.occupied()).sum()
    }

    fn buckets2visit(&self, query: &ArraySlice, nprobe: usize) -> Vec<&Bucket> {
        self.model
            .predict(query)
            .into_iter()
            .take(nprobe)
            .map(|(bucket_idx, _)| &self.buckets[bucket_idx])
            .collect()
    }

    #[log_time]
    fn train(&mut self, xs: &[Array], k: usize) {
        self.model.train(xs, k);
    }

    fn insert(&mut self, data: Vec<Array>, ids: Vec<Id>) {
        data.into_iter().zip(ids).for_each(|(data, id)| {
            let predictions = self.model.predict(&data);
            assert!(predictions.len() == self.buckets.len());
            let bucket_idx = predictions[0].0;
            self.buckets[bucket_idx].insert(data, id);
        });
    }

    fn insert_many(&mut self, data: Vec<Array>, ids: Vec<Id>) {
        assert!(data.len() == ids.len());
        let asigments = self.model.predict_many(&data);
        assert!(asigments.len() == data.len());
        asigments
            .into_iter()
            .zip(data.into_iter().zip(ids))
            .for_each(|(bucket_idx, (query, id))| {
                self.buckets[bucket_idx].insert(query, id);
            });
    }

    fn get_data(&mut self) -> (Vec<Array>, Vec<Id>) {
        let (data, ids): (Vec<Vec<Array>>, Vec<Vec<Id>>) = self
            .buckets
            .iter_mut()
            .filter(|bucket| bucket.occupied() > 0)
            .map(|bucket| bucket.get_data())
            .unzip();
        (
            data.into_iter().flatten().collect(),
            ids.into_iter().flatten().collect(),
        )
    }

    fn n_buckets(&self) -> usize {
        self.buckets.len()
    }
}
