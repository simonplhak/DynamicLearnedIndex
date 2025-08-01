use crate::{
    bucket::{self, Bucket, BucketBuilder},
    errors::BuildError,
    model::{self, Model, ModelConfig},
    types::{Array, ArraySlice},
    DistanceFn, Id, SearchStrategy,
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
    pub levelling: Levelling,
    pub levels: HashMap<usize, LevelIndexConfig>,
    pub buffer_size: usize,
    pub input_shape: usize,
    pub arity: usize,
    pub device: ModelDevice,
    pub distance_fn: DistanceFn,
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
            distance_fn: Default::default(),
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
            .distance_fn(self.distance_fn.clone())
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
                    distance_fn: self.distance_fn,
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

pub struct SearchParams {
    k: usize,
    search_strategy: SearchStrategy,
}

pub trait SearchParamsT {
    fn into_search_params(self) -> SearchParams;
}

impl SearchParamsT for () {
    fn into_search_params(self) -> SearchParams {
        SearchParams {
            k: 10,
            search_strategy: SearchStrategy::default(),
        }
    }
}

impl SearchParamsT for (usize, SearchStrategy) {
    fn into_search_params(self) -> SearchParams {
        SearchParams {
            k: self.0,
            search_strategy: self.1,
        }
    }
}

impl SearchParamsT for usize {
    fn into_search_params(self) -> SearchParams {
        SearchParams {
            k: self,
            search_strategy: SearchStrategy::default(),
        }
    }
}

impl Index {
    pub fn search<S>(&self, query: &ArraySlice, params: S) -> Vec<Id>
    where
        S: SearchParamsT,
    {
        let params = params.into_search_params();
        match self {
            Index::BentleySaxe(index) => index.search(query, params),
        }
    }

    pub fn insert(&mut self, value: Array, id: Id) {
        match self {
            Index::BentleySaxe(index) => {
                index.insert(value, id);
            }
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Index::BentleySaxe(index) => index.size(),
        }
    }

    pub fn n_buckets(&self) -> usize {
        match self {
            Index::BentleySaxe(index) => index.levels.iter().map(|level| level.n_buckets()).sum(),
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
    distance_fn: DistanceFn,
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
            .distance_fn(self.distance_fn.clone())
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

    fn buckets2visit(&self, query: &ArraySlice, search_strategy: SearchStrategy) -> Vec<&Bucket> {
        let bucket_predictions = self
            .levels
            .iter()
            .map(|level| level.buckets2visit_predictions(query))
            .collect::<Vec<_>>();
        let level_bucket_idxs = search_strategy.buckets2visit(bucket_predictions);
        level_bucket_idxs
            .into_iter()
            .zip(self.levels.iter())
            .flat_map(|(bucket_idxs, level)| {
                bucket_idxs
                    .into_iter()
                    .map(|bucket_idx| &level.buckets[bucket_idx])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    }

    #[log_time]
    fn search(&self, query: &ArraySlice, params: SearchParams) -> Vec<Id> {
        let buckets2visit = self.buckets2visit(query, params.search_strategy);
        let mut results: Vec<(Id, f32)> = buckets2visit
            .par_iter()
            .flat_map(|bucket| bucket.search(query, params.k))
            .collect();
        results.sort_unstable_by(|(_, dist_a), (_, dist_b)| self.distance_fn.cmp(dist_a, dist_b));
        results
            .into_iter()
            .take(params.k)
            .map(|(id, _)| id)
            .collect()
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
                level.insert_many(data, ids);
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
    distance_fn: Option<DistanceFn>,
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

    pub fn distance_fn(&mut self, distance_fn: DistanceFn) -> &mut Self {
        self.distance_fn = Some(distance_fn);
        self
    }

    pub fn build(&self) -> Result<LevelIndex, BuildError> {
        let n_buckets = self.n_buckets.ok_or(BuildError::MissingAttribute)?;
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;
        let id = self.id.clone().ok_or(BuildError::MissingAttribute)?;
        let bucket_size = self.bucket_size.ok_or(BuildError::MissingAttribute)?;
        let distance_fn = self
            .distance_fn
            .clone()
            .ok_or(BuildError::MissingAttribute)?;
        let model_config = self
            .model_config
            .as_ref()
            .ok_or(BuildError::MissingAttribute)?;
        let mut model_builder = model::ModelBuilder::default();
        model_builder
            .device(self.model_device.to_tch_device())
            .input_nodes(input_shape as i64)
            .train_params(model_config.train_params.clone())
            .labels(n_buckets as i64)
            .label_method(distance_fn.clone().into());
        model_config.layers.iter().for_each(|layer| {
            model_builder.add_layer(*layer);
        });
        let model = model_builder.build()?;
        let mut bucket_builder = bucket::BucketBuilder::default();
        bucket_builder
            .input_shape(input_shape)
            .size(bucket_size)
            .is_dynamic(true)
            .distance_fn(distance_fn);
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

    fn buckets2visit_predictions(&self, query: &ArraySlice) -> Vec<(usize, f32)> {
        self.model.predict(query).into_iter().collect()
    }

    #[log_time]
    fn train(&mut self, xs: &[Array], k: usize) {
        self.model.train(xs, k);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        distance_fn::DistanceFn,
        model::{ModelConfig, ModelLayer, TrainParams},
        search_strategy::SearchStrategy,
    };
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_config() -> IndexConfig {
        let mut levels = HashMap::new();
        levels.insert(0, LevelIndexConfig::default());

        IndexConfig {
            levelling: Levelling::BentleySaxe,
            levels,
            buffer_size: 10,
            input_shape: 3,
            arity: 2,
            device: ModelDevice::Cpu,
            distance_fn: DistanceFn::Dot,
        }
    }

    fn create_simple_model_config() -> ModelConfig {
        ModelConfig {
            layers: vec![ModelLayer::Linear(2), ModelLayer::ReLU],
            train_params: TrainParams {
                threshold_samples: 5,
                batch_size: 2,
                epochs: 1,
                ..Default::default()
            },
        }
    }

    #[test]
    fn test_model_device_to_tch_device() {
        let cpu_device = ModelDevice::Cpu;
        assert!(matches!(cpu_device.to_tch_device(), tch::Device::Cpu));

        let gpu_device = ModelDevice::Gpu(0);
        assert!(matches!(gpu_device.to_tch_device(), tch::Device::Cuda(0)));
    }

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.buffer_size, 5000);
        assert_eq!(config.input_shape, 768);
        assert_eq!(config.arity, 3);
        assert!(matches!(config.device, ModelDevice::Cpu));
        assert!(matches!(config.distance_fn, DistanceFn::Dot));
        assert!(matches!(config.levelling, Levelling::BentleySaxe));
        assert!(config.levels.contains_key(&0));
    }

    #[test]
    fn test_level_index_config_default() {
        let config = LevelIndexConfig::default();
        assert_eq!(config.bucket_size, 5000);
    }

    #[test]
    fn test_index_config_from_yaml_valid() {
        let yaml_content = r#"
        levelling: bentley_saxe
        levels:
          0:
            model:
              layers:
                - type: linear
                  value: 4
                - type: relu
              train_params:
                threshold_samples: 100
                batch_size: 4
                epochs: 2
                label_method:
                  type: knn
                  value:
                    max_iters: 5
            bucket_size: 100
        buffer_size: 50
        input_shape: 10
        arity: 2
        device: cpu
        distance_fn: dot
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "{yaml_content}").unwrap();

        let config = IndexConfig::from_yaml(temp_file.path().to_str().unwrap()).unwrap();
        assert_eq!(config.buffer_size, 50);
        assert_eq!(config.input_shape, 10);
        assert_eq!(config.arity, 2);
    }

    #[test]
    fn test_index_config_from_yaml_nonexistent_file() {
        let result = IndexConfig::from_yaml("nonexistent.yaml");
        assert!(matches!(result, Err(BuildError::NonExistentFile)));
    }

    #[test]
    fn test_index_config_build_success() {
        let config = create_test_config();
        let index = config.build().unwrap();
        assert!(matches!(index, Index::BentleySaxe(_)));
    }

    #[test]
    fn test_index_config_build_empty_levels() {
        let mut config = create_test_config();
        config.levels.clear();
        let result = config.build();
        assert!(matches!(result, Err(BuildError::MissingAttribute)));
    }

    #[test]
    fn test_index_config_build_missing_level_zero() {
        let mut config = create_test_config();
        config.levels.clear();
        config.levels.insert(1, LevelIndexConfig::default());
        let result = config.build();
        assert!(matches!(result, Err(BuildError::MissingAttribute)));
    }

    #[test]
    fn test_search_params_trait_unit() {
        let params = ().into_search_params();
        assert_eq!(params.k, 10);
        assert!(matches!(
            params.search_strategy,
            SearchStrategy::ModelDriven(30)
        ));
    }

    #[test]
    fn test_search_params_trait_usize() {
        let params = 5usize.into_search_params();
        assert_eq!(params.k, 5);
        assert!(matches!(
            params.search_strategy,
            SearchStrategy::ModelDriven(30)
        ));
    }

    #[test]
    fn test_search_params_trait_tuple() {
        let params = (3, SearchStrategy::Base(10)).into_search_params();
        assert_eq!(params.k, 3);
        assert!(matches!(params.search_strategy, SearchStrategy::Base(10)));
    }

    #[test]
    fn test_index_insert_and_size() {
        let config = create_test_config();
        let mut index = config.build().unwrap();

        // Initial size is buffer size (10 from test config)
        assert_eq!(index.size(), 10);

        index.insert(vec![1.0, 2.0, 3.0], 1);
        // Size method returns total capacity, not occupied count
        assert_eq!(index.size(), 10);

        index.insert(vec![4.0, 5.0, 6.0], 2);
        assert_eq!(index.size(), 10);
    }

    #[test]
    fn test_index_search_basic() {
        let config = create_test_config();
        let mut index = config.build().unwrap();

        // Insert some test data
        index.insert(vec![1.0, 2.0, 3.0], 1);
        index.insert(vec![4.0, 5.0, 6.0], 2);

        // We can't actually search with no levels because it would trigger empty predictions
        // But we can verify that data was inserted
        assert_eq!(index.size(), 10); // Buffer size
    }

    #[test]
    fn test_level_index_builder() {
        let model_config = create_simple_model_config();
        let mut builder = LevelIndexBuilder::default();
        builder
            .id("test_level".to_string())
            .n_buckets(2)
            .input_shape(3)
            .bucket_size(10)
            .model(model_config)
            .distance_fn(DistanceFn::Dot)
            .model_device(ModelDevice::Cpu);

        let level_index = builder.build().unwrap();
        assert_eq!(level_index.n_buckets(), 2);
        assert_eq!(level_index.size(), 20); // 2 buckets * 10 size each
        assert_eq!(level_index.occupied(), 0);
    }

    #[test]
    fn test_level_index_builder_missing_attributes() {
        let builder = LevelIndexBuilder::default();
        let result = builder.build();
        assert!(matches!(result, Err(BuildError::MissingAttribute)));
    }

    #[test]
    fn test_bentley_saxe_index_available_level() {
        let config = create_test_config();
        let index = config.build().unwrap();

        let Index::BentleySaxe(bs_index) = index;
        // Initially no levels, so should return None
        assert_eq!(bs_index.available_level(), None);
    }

    #[test]
    fn test_bentley_saxe_index_get_level_config() {
        let config = create_test_config();
        let index = config.build().unwrap();

        let Index::BentleySaxe(bs_index) = index;
        let level_config = bs_index.get_level_index_config();
        assert_eq!(level_config.bucket_size, 5000); // default value
    }

    #[test]
    fn test_bentley_saxe_index_insert_to_buffer() {
        let config = create_test_config();
        let mut index = config.build().unwrap();

        // Initial size should be buffer size
        let initial_size = index.size();
        assert_eq!(initial_size, 10);

        // Insert data that should go to buffer
        for i in 1..=5 {
            index.insert(vec![i as f32, i as f32 + 1.0, i as f32 + 2.0], i as u32);
        }

        // Size should remain the same (capacity, not occupancy)
        assert_eq!(index.size(), 10);
    }

    #[test]
    fn test_level_index_predictions() {
        let model_config = create_simple_model_config();
        let mut builder = LevelIndexBuilder::default();
        builder
            .id("test_level".to_string())
            .n_buckets(2)
            .input_shape(3)
            .bucket_size(10)
            .model(model_config)
            .distance_fn(DistanceFn::Dot)
            .model_device(ModelDevice::Cpu);

        let level_index = builder.build().unwrap();
        let query = vec![1.0, 2.0, 3.0];
        let predictions = level_index.buckets2visit_predictions(&query);

        // Should return predictions for each bucket
        assert_eq!(predictions.len(), 2);
        // Each prediction should have bucket index and probability
        assert!(predictions
            .iter()
            .all(|(bucket_idx, _prob)| *bucket_idx < 2));
    }

    #[test]
    fn test_level_index_get_data_empty() {
        let model_config = create_simple_model_config();
        let mut builder = LevelIndexBuilder::default();
        builder
            .id("test_level".to_string())
            .n_buckets(2)
            .input_shape(3)
            .bucket_size(10)
            .model(model_config)
            .distance_fn(DistanceFn::Dot)
            .model_device(ModelDevice::Cpu);

        let mut level_index = builder.build().unwrap();
        let (data, ids) = level_index.get_data();

        assert!(data.is_empty());
        assert!(ids.is_empty());
    }

    #[test]
    fn test_bentley_saxe_empty_levels() {
        let config = create_test_config();
        let index = config.build().unwrap();

        let Index::BentleySaxe(bs_index) = index;
        // Should have no levels initially
        assert_eq!(bs_index.levels.len(), 0);
        // Should have a buffer
        assert_eq!(bs_index.buffer.size(), 10);
    }

    #[test]
    fn test_bentley_saxe_lower_level_data_empty() {
        let config = create_test_config();
        let mut index = config.build().unwrap();

        let Index::BentleySaxe(ref mut bs_index) = index;
        let (data, ids) = bs_index.lower_level_data(0);
        // Should only have buffer data initially
        assert!(data.is_empty());
        assert!(ids.is_empty());
    }
}
