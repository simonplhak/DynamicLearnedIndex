use std::collections::HashMap;

use crate::{
    bucket::{self, Bucket, BucketBuilder, BucketType},
    clustering::{compute_labels, LabelMethod},
    errors::BuildError,
    model::{self, Model, ModelConfig},
    types::Array,
    util, Id,
};
use log::info;
use measure_time_macro::log_time;
use serde::{Deserialize, Serialize};
use tch::{Device, Tensor};

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub enum ModelDevice {
    #[default]
    #[serde(rename = "cpu")]
    Cpu,
}

impl ModelDevice {
    pub fn to_tch_device(&self) -> Device {
        match self {
            ModelDevice::Cpu => Device::Cpu,
        }
    }
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
    buffer_size: usize,
    input_shape: i64,
    arity: i64,
    label_method: LabelMethod,
    device: ModelDevice,
}

impl IndexConfig {
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
            .bucket_type(BucketType::New)
            .build()?;
        let index = match self.levelling {
            Levelling::BentleySaxe => {
                let index = BentleySaxeIndex {
                    levels_config: self.levels,
                    input_shape: self.input_shape,
                    arity: self.arity,
                    device: self.device,
                    levels: Vec::new(),
                    label_method: self.label_method,
                    buffer,
                };
                Index::BentleySaxe(index)
            }
        };
        Ok(index)
    }
}

pub enum Index {
    BentleySaxe(BentleySaxeIndex),
}

impl Index {
    #[log_time]
    pub fn search(&self, query: &Tensor, k: usize) -> Vec<Id> {
        match self {
            Index::BentleySaxe(index) => index.search(query, k),
        }
    }

    pub fn insert(&mut self, value: Tensor, id: Id) {
        match self {
            Index::BentleySaxe(index) => {
                let value = util::tensor2vec(&value);
                index.insert(value, id);
            }
        }
    }
}

#[derive(Debug)]
pub struct BentleySaxeIndex {
    levels_config: HashMap<usize, LevelIndexConfig>,
    input_shape: i64,
    arity: i64,
    label_method: LabelMethod,
    device: ModelDevice, // todo propagate to model
    levels: Vec<LevelIndex>,
    buffer: Bucket,
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

    fn buckets2visit(&self, query: &Tensor) -> Vec<&Bucket> {
        self.levels
            .iter()
            .map(|level| level.bucket2visit(query))
            .collect()
    }

    #[log_time]
    fn search(&self, query: &Tensor, k: usize) -> Vec<Id> {
        let buckets2visit = self.buckets2visit(query);
        let (ids, distances): (Vec<_>, Vec<_>) = buckets2visit
            .iter()
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
        info!(buffer_size = self.buffer.size(); "index:buffer_flush");
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
                let (data, ids) =
                    compute_labels(data, ids, &self.label_method, level.n_buckets() as i64);
                let cluster_shape = data
                    .iter()
                    .map(|x| x.len().to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                info!(cluster_shape = cluster_shape, level_idx = level_idx; "index:cluster_shape");
                let data_refs: Vec<&[Array]> = data.iter().map(|inner| inner.as_slice()).collect();
                level.train(&data_refs);
                level.insert_many(data, ids);
            }
        };
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct LevelIndexConfig {
    pub model: ModelConfig,
    pub bucket_size: usize,
}

#[derive(Debug, Default)]
pub(crate) struct LevelIndexBuilder {
    id: Option<String>,
    n_buckets: Option<i64>,
    model_config: Option<ModelConfig>,
    bucket_size: Option<usize>,
    input_shape: Option<i64>,
    model_device: ModelDevice,
}

impl LevelIndexBuilder {
    pub fn n_buckets(&mut self, size: i64) -> &mut Self {
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

    pub fn input_shape(&mut self, input_shape: i64) -> &mut Self {
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
            .input_nodes(input_shape)
            .labels(n_buckets);
        model_config.layers.iter().for_each(|layer| {
            model_builder.add_layer(*layer);
        });
        let model = model_builder.build()?;
        let mut bucket_builder = bucket::BucketBuilder::default();
        bucket_builder
            .input_shape(input_shape)
            .size(bucket_size)
            .is_dynamic(true)
            .bucket_type(BucketType::New);
        let buckets = (0..n_buckets)
            .map(|bucket_id| bucket_builder.id(format!("{}:{}", id, bucket_id)).build())
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

    fn bucket2visit(&self, query: &Tensor) -> &Bucket {
        let bucket_idx = self.model.predict(query);
        &self.buckets[bucket_idx]
    }

    #[log_time]
    fn train(&mut self, queries: &[&[Array]]) {
        assert!(self.buckets.len() == queries.len());
        self.model.train(queries);
    }

    fn insert(&mut self, data: Vec<Array>, ids: Vec<Id>) {
        data.into_iter().zip(ids).for_each(|(data, id)| {
            let tensor = util::vec2tensor(&data);
            let bucket_idx = self.model.predict(&tensor);
            self.buckets[bucket_idx].insert(data, id);
        });
    }

    fn insert_many(&mut self, data: Vec<Vec<Array>>, ids: Vec<Vec<Id>>) {
        assert!(data.len() == self.buckets.len());
        assert!(ids.len() == self.buckets.len());
        data.into_iter()
            .zip(ids)
            .enumerate()
            .for_each(|(bucket_idx, (data, ids))| {
                assert!(data.len() == ids.len());
                self.buckets[bucket_idx].insert_many(data, ids);
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
