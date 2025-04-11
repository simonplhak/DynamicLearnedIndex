use std::{collections::HashMap, fmt};

use crate::{
    bucket::{self, Bucket, StaticBucket},
    clustering::{compute_labels, LabelMethod},
    errors::BuildError,
    model::{self, Model, ModelConfig},
    Id,
};
use log::info;
use serde::{Deserialize, Serialize};
use tch::{Device, Tensor};

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
    buffer_size: usize,
    bucket: bucket::BucketType,
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
        let buffer = Bucket::Static(StaticBucket::new(self.buffer_size, self.input_shape));
        let index = match self.levelling {
            Levelling::BentleySaxe => {
                let index = BentleySaxeIndex {
                    levels_config: self.levels,
                    bucket_type: self.bucket,
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
        info!(index:? = index; "index:build");
        Ok(index)
    }
}

#[derive(Serialize)]
pub enum Index {
    BentleySaxe(BentleySaxeIndex),
}

impl fmt::Debug for Index {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            serde_json::to_string(self).map_err(|_| fmt::Error)?
        )
    }
}

impl Index {
    pub fn search(&self, key: &Tensor) -> (Tensor, Tensor) {
        match self {
            Index::BentleySaxe(index) => index.search(key),
        }
    }

    pub fn insert(&mut self, value: Tensor, id: Id) {
        match self {
            Index::BentleySaxe(index) => index.insert(value, id),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct BentleySaxeIndex {
    levels_config: HashMap<usize, LevelIndexConfig>,
    bucket_type: bucket::BucketType,
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
            .n_buckets(n_buckets)
            .input_shape(self.input_shape)
            .model(level_index_config.model.clone())
            .bucket(self.bucket_type)
            .build()
            .unwrap();
        self.levels.push(level_index);
        self.levels.len() - 1
    }

    fn lower_level_data(&mut self, level_idx: usize) -> (Vec<Tensor>, Vec<Id>) {
        let (data, ids): (Vec<Vec<Tensor>>, Vec<Vec<Id>>) = self
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

    fn search(&self, query: &Tensor) -> (Tensor, Tensor) {
        let buckets2visit = self.buckets2visit(query);
        // todo: how to merge results?
        todo!()
    }

    fn insert(&mut self, value: Tensor, id: Id) {
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
                info!(cluster_shape = cluster_shape; "index:cluster_shape");
                let data_refs: Vec<&[Tensor]> = data.iter().map(|inner| inner.as_slice()).collect();
                level.train(&data_refs);
                level.insert_many(data, ids);
            }
        };
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct LevelIndexConfig {
    pub model: ModelConfig,
}

#[derive(Debug, Default)]
pub(crate) struct LevelIndexBuilder {
    n_buckets: Option<i64>,
    model_config: Option<ModelConfig>,
    bucket_type: Option<bucket::BucketType>,
    input_shape: Option<i64>,
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

    pub fn bucket(&mut self, bucket: bucket::BucketType) -> &mut Self {
        self.bucket_type = Some(bucket);
        self
    }

    pub fn input_shape(&mut self, input_shape: i64) -> &mut Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn build(&self) -> Result<LevelIndex, BuildError> {
        let n_buckets = self.n_buckets.ok_or(BuildError::MissingAttribute)?;
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;
        let model_config = self
            .model_config
            .as_ref()
            .ok_or(BuildError::MissingAttribute)?;
        let mut model_builder = model::ModelBuilder::default();
        model_builder
            .device(Device::Cpu)
            .input_nodes(input_shape)
            .labels(n_buckets);
        model_config.layers.iter().for_each(|layer| {
            model_builder.add_layer(*layer);
        });
        let model = model_builder.build()?;
        let mut bucket_builder = bucket::BucketBuilder::default();
        let bucket_type = self
            .bucket_type
            .as_ref()
            .ok_or(BuildError::MissingAttribute)?;
        bucket_builder
            .input_shape(input_shape)
            .bucket_type(*bucket_type);
        let buckets = (0..n_buckets)
            .map(|_| bucket_builder.build())
            .collect::<Result<Vec<_>, _>>()?;
        let level_index = LevelIndex { model, buckets };
        Ok(level_index)
    }
}

#[derive(Debug, Serialize)]
pub struct LevelIndex {
    #[serde(skip)]
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

    fn search(&self, query: &Tensor) -> Tensor {
        let bucket_idx = self.model.predict(query);
        self.buckets[bucket_idx].search(query);
        // self.model.forward(&key)
        todo!()
    }

    fn train(&mut self, queries: &[&[Tensor]]) {
        assert!(self.buckets.len() == queries.len());
        let total_queries = queries.iter().map(|x| x.len()).sum::<usize>() as i64;
        let xs: Tensor = Tensor::cat(
            &queries
                .iter()
                .map(|xs| Tensor::cat(&xs.iter().map(|x| x.unsqueeze(0)).collect::<Vec<_>>(), 0))
                .collect::<Vec<_>>(),
            0,
        );
        assert!(
            xs.size()[0] == total_queries,
            "xs and buckets must have the same length: xs={:?}, queries={}",
            xs.size(),
            total_queries
        );

        let ys = Tensor::cat(
            &queries
                .iter()
                .enumerate()
                .map(|(y, x)| {
                    Tensor::full([x.len() as i64], y as i64, (tch::Kind::Float, Device::Cpu))
                }) // todo use specified device
                .collect::<Vec<_>>(),
            0,
        );
        assert!(
            xs.size()[0] == ys.size()[0],
            "xs and ys must have the same length: xs={:?}, ys={:?}",
            xs.size(),
            ys.size()
        );
        self.model.train(xs, ys);
    }

    fn insert(&mut self, data: Vec<Tensor>, ids: Vec<Id>) {
        data.into_iter().zip(ids).for_each(|(data, id)| {
            let bucket_idx = self.model.predict(&data);
            self.buckets[bucket_idx].insert(data, id);
        });
    }

    fn insert_many(&mut self, data: Vec<Vec<Tensor>>, ids: Vec<Vec<Id>>) {
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

    fn get_data(&mut self) -> (Vec<Tensor>, Vec<Id>) {
        let (data, ids): (Vec<Vec<Tensor>>, Vec<Vec<Id>>) = self
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
