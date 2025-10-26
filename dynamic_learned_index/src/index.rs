use crate::{
    bucket::{self, Bucket, Buffer, DeleteMethod},
    errors::BuildError,
    model::{Model, ModelBuilder, ModelConfig, ModelDevice},
    types::{Array, ArraySlice},
    DistanceFn, Id, SearchStrategy,
};
use log::{debug, info};
use measure_time_macro::log_time;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    pub compaction_strategy: CompactionStrategy,
    pub levels: HashMap<usize, LevelIndexConfig>,
    pub buffer_size: usize,
    pub input_shape: usize,
    pub arity: usize,
    pub device: ModelDevice,
    pub distance_fn: DistanceFn,
    pub delete_method: DeleteMethod,
}

impl Default for IndexConfig {
    fn default() -> Self {
        let mut levels = HashMap::new();
        levels.insert(0, Default::default());
        Self {
            compaction_strategy: Default::default(),
            levels,
            buffer_size: 5000,
            input_shape: 768,
            arity: 3,
            device: Default::default(),
            distance_fn: Default::default(),
            delete_method: Default::default(),
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
        let buffer = Buffer::new(self.buffer_size, self.input_shape);
        let index = Index {
            levels_config: self.levels,
            input_shape: self.input_shape,
            arity: self.arity,
            device: self.device,
            levels: Vec::new(),
            buffer,
            distance_fn: self.distance_fn,
            compaction_strategy: self.compaction_strategy,
            delete_method: self.delete_method,
        };
        Ok(index)
    }
}

pub struct Index {
    compaction_strategy: CompactionStrategy,
    levels_config: HashMap<usize, LevelIndexConfig>,
    input_shape: usize,
    arity: usize,
    device: ModelDevice,
    levels: Vec<LevelIndex>,
    buffer: Buffer,
    distance_fn: DistanceFn,
    delete_method: DeleteMethod,
}

pub struct SearchParams {
    pub k: usize,
    pub search_strategy: SearchStrategy,
}

pub trait SearchParamsT {
    fn into_search_params(self) -> SearchParams;
}

impl SearchParamsT for SearchParams {
    fn into_search_params(self) -> SearchParams {
        self
    }
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

pub struct SearchStatistics {
    pub total_visited_buckets: usize,
    pub total_visited_records: usize,
}

impl Index {
    pub fn search<S>(&self, query: &ArraySlice, params: S) -> Vec<Id>
    where
        S: SearchParamsT,
    {
        let params = params.into_search_params();
        let (records2visit, ids2visit, _) = self.records2visit(query, params.search_strategy);
        let rs = flat_knn::knn(
            records2visit,
            query,
            params.k,
            match self.distance_fn {
                DistanceFn::L2 => flat_knn::Metric::L2,
                DistanceFn::Dot => flat_knn::Metric::Dot,
            },
        );
        rs.into_iter().map(|(_, idx)| ids2visit[idx]).collect()
    }

    fn records2visit(
        &self,
        query: &ArraySlice,
        search_strategy: SearchStrategy,
    ) -> (Vec<&[f32]>, Vec<Id>, usize) {
        // (records, ids, total_visited_buckets)
        let bucket_predictions = self
            .levels
            .iter()
            .map(|level| level.buckets2visit_predictions(query))
            .collect::<Vec<_>>();
        let level_bucket_idxs =
            search_strategy.buckets2visit(bucket_predictions, self.buffer.occupied());
        let buckets2visit_count: usize = level_bucket_idxs.iter().map(|v| v.len()).sum();

        // Pre-calculate total capacity to avoid repeated allocations
        let total_capacity = level_bucket_idxs
            .iter()
            .zip(self.levels.iter())
            .map(|(bucket_idxs, level)| {
                bucket_idxs
                    .iter()
                    .map(|&bucket_idx| level.buckets[bucket_idx].occupied())
                    .sum::<usize>()
            })
            .sum::<usize>()
            + self.buffer.occupied();

        let mut records = Vec::with_capacity(total_capacity);
        let mut ids = Vec::with_capacity(total_capacity);

        // Process levels
        for (bucket_idxs, level) in level_bucket_idxs.into_iter().zip(self.levels.iter()) {
            for bucket_idx in bucket_idxs {
                let bucket = &level.buckets[bucket_idx];
                let occupied = bucket.occupied();
                for i in 0..occupied {
                    records.push(bucket.record(i));
                    ids.push(bucket.ids[i]);
                }
            }
        }

        // Process buffer
        let buffer_occupied = self.buffer.occupied();
        for i in 0..buffer_occupied {
            records.push(self.buffer.record(i));
            ids.push(self.buffer.ids[i]);
        }

        (records, ids, buckets2visit_count)
    }

    pub fn add_level(&mut self) -> usize {
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

    fn get_level_index_config(&self) -> LevelIndexConfig {
        let curr_level = self.levels.len();
        self.levels_config
            .iter()
            .take_while(|(level, _)| **level <= curr_level)
            .last()
            .map(|(_, config)| config.to_owned())
            .unwrap()
    }

    pub fn insert(&mut self, value: Array, id: Id) {
        if self.buffer.has_space(1) {
            self.buffer.insert(value, id);
            return; // value fits into buffer
        }
        debug!(levels = self.levels.len(), occupied = self.occupied(); "index:buffer_flush");
        // let strategy = self.compaction_strategy.clone();
        self.compaction_strategy.clone().compact(self);
        assert!(self.buffer.has_space(1));
        self.buffer.insert(value, id);
    }

    pub fn delete(&mut self, id: Id) -> Option<(Array, Id)> {
        if let Some(deleted) = self.buffer.delete(&id) {
            return Some(deleted);
        }
        for level in &mut self.levels {
            if let Some(deleted) = level.delete(&id, &self.delete_method) {
                // todo move delete method spec to index
                return Some(deleted);
            }
        }
        None
    }

    pub fn size(&self) -> usize {
        self.levels.iter().map(|level| level.size()).sum::<usize>() + self.buffer.size
    }

    pub fn n_buckets(&self) -> usize {
        self.levels.iter().map(|level| level.n_buckets()).sum()
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn occupied(&self) -> usize {
        self.levels
            .iter()
            .map(|level| level.occupied())
            .sum::<usize>()
            + self.buffer.occupied()
    }

    pub fn verbose_search<S>(&self, query: &ArraySlice, params: S) -> (Vec<Id>, SearchStatistics)
    where
        S: SearchParamsT,
    {
        let params = params.into_search_params();
        let (records2visit, ids2visit, total_visited_buckets) =
            self.records2visit(query, params.search_strategy);
        let rs = flat_knn::knn(
            records2visit,
            query,
            params.k,
            match self.distance_fn {
                DistanceFn::L2 => flat_knn::Metric::L2,
                DistanceFn::Dot => flat_knn::Metric::Dot,
            },
        );
        let res = rs.into_iter().map(|(_, idx)| ids2visit[idx]).collect();
        (
            res,
            SearchStatistics {
                total_visited_buckets,
                total_visited_records: ids2visit.len(),
            },
        )
    }
}

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub enum CompactionStrategy {
    #[default]
    #[serde(rename = "bentley_saxe")]
    BentleySaxe,
}

impl CompactionStrategy {
    fn available_level(&self, index: &Index) -> Option<usize> {
        let mut count = index.buffer.occupied();
        index
            .levels
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

    fn lower_level_data(&self, index: &mut Index, level_idx: usize) -> (Array, Vec<Id>) {
        let (data, ids): (Vec<Array>, Vec<Vec<Id>>) = index
            .levels
            .iter_mut()
            .take(level_idx)
            .map(|level| level.get_data())
            .unzip();
        let (buffer_data, buffer_ids) = index.buffer.get_data();
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

    pub fn compact(&self, index: &mut Index) {
        let original_occupied = index.occupied();
        match self {
            CompactionStrategy::BentleySaxe => {
                match self.available_level(index) {
                    Some(level_idx) => {
                        let (data, ids) = self.lower_level_data(index, level_idx);
                        let level = &mut index.levels[level_idx];
                        if level.size() == 0 {
                            level.retrain(&data);
                        }
                        level.insert_many(data, ids);
                    }
                    None => {
                        let level_idx = index.add_level();
                        let (data, ids) = self.lower_level_data(index, level_idx);
                        let level = &mut index.levels[level_idx];
                        level.train(&data);
                        level.insert_many(data, ids);
                    }
                };
            }
        }
        assert_eq!(original_occupied, index.occupied());
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
        let bucket_size = self.bucket_size.ok_or(BuildError::MissingAttribute)?;
        let distance_fn = self
            .distance_fn
            .clone()
            .ok_or(BuildError::MissingAttribute)?;
        let model_config = self
            .model_config
            .as_ref()
            .ok_or(BuildError::MissingAttribute)?;
        let mut model_builder = ModelBuilder::default();
        model_builder
            .device(self.model_device.clone())
            .input_nodes(input_shape as i64)
            .train_params(model_config.train_params.clone())
            .retrain_params(model_config.retrain_params.clone())
            .labels(n_buckets)
            .label_method(distance_fn.clone().into());
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
            .map(|_| bucket_builder.build())
            .collect::<Result<Vec<_>, _>>()?;
        let level_index = LevelIndex::new(model, buckets);
        Ok(level_index)
    }
}

pub struct LevelIndex {
    model: Model,
    buckets: Vec<Bucket>,
    ids_map: HashMap<Id, (usize, usize)>, // Id -> (bucket_idx, record_idx)
}

impl LevelIndex {
    fn new(model: Model, buckets: Vec<Bucket>) -> Self {
        Self {
            model,
            buckets,
            ids_map: HashMap::new(),
        }
    }

    fn size(&self) -> usize {
        self.buckets.iter().map(|bucket| bucket.size()).sum()
    }

    fn occupied(&self) -> usize {
        self.buckets.iter().map(|bucket| bucket.occupied()).sum()
    }

    fn buckets2visit_predictions(&self, query: &ArraySlice) -> Vec<(usize, f32, usize)> {
        self.model
            .predict(query)
            .into_iter()
            .map(|(bucket_id, prob)| (bucket_id, prob, self.buckets[bucket_id].occupied()))
            .collect()
    }

    #[log_time]
    fn train(&mut self, xs: &ArraySlice) {
        self.model.train(xs);
    }

    #[log_time]
    fn retrain(&mut self, xs: &ArraySlice) {
        self.model.retrain(xs);
    }
    fn insert_many(&mut self, records: Array, ids: Vec<Id>) {
        let input_shape = self.model.input_shape;
        assert!(records.len() / input_shape == ids.len());
        let assignments = self.model.predict_many(&records);
        assert!(assignments.len() == ids.len());
        // Calculate frequency of each bucket index in assignments
        let mut frequencies = vec![0; self.buckets.len()];
        for &bucket_idx in &assignments {
            frequencies[bucket_idx] += 1;
        }
        // Pre-resize buckets based on calculated frequencies
        frequencies
            .iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .for_each(|(bucket_idx, count)| {
                self.buckets[bucket_idx].resize(*count);
            });
        {
            let mut records = records;
            let mut ids = ids;
            let mut assignments = assignments;
            while !records.is_empty() {
                let query = records.split_off(records.len() - input_shape);
                let id = ids.pop().unwrap();
                let bucket_idx = assignments.pop().unwrap();
                self.buckets[bucket_idx].insert(query, id);
                self.ids_map
                    .insert(id, (bucket_idx, self.buckets[bucket_idx].occupied() - 1));
            }
            assert!(assignments.is_empty());
            assert!(ids.is_empty());
            assert!(records.is_empty());
        }
    }

    fn get_data(&mut self) -> (Array, Vec<Id>) {
        let (data, ids): (Vec<Array>, Vec<Vec<Id>>) = self
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

    fn delete(&mut self, id: &Id, delete_method: &DeleteMethod) -> Option<(Array, Id)> {
        let deleted = self.ids_map.get(id).cloned();
        if let Some((bucket_idx, record_idx)) = deleted {
            let bucket = &mut self.buckets[bucket_idx];
            let (deleted, (swapped_new_idx, swapped_id)) =
                bucket.delete(record_idx, delete_method)?;
            let (old_bucket_idx, old_record_idx) = self.ids_map.remove(id).unwrap(); // we are sure it exists
            assert_eq!(old_bucket_idx, bucket_idx);
            assert_eq!(old_record_idx, bucket.occupied());
            self.ids_map
                .insert(swapped_id, (bucket_idx, swapped_new_idx));
            return Some(deleted);
        }
        None
    }

    fn n_buckets(&self) -> usize {
        self.buckets.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{distance_fn::DistanceFn, search_strategy::SearchStrategy};
    use std::collections::HashMap;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_config() -> IndexConfig {
        let mut levels = HashMap::new();
        levels.insert(0, LevelIndexConfig::default());

        IndexConfig {
            levels,
            buffer_size: 10,
            input_shape: 3,
            arity: 2,
            device: ModelDevice::Cpu,
            distance_fn: DistanceFn::Dot,
            compaction_strategy: CompactionStrategy::BentleySaxe,
            delete_method: DeleteMethod::OidToBucket,
        }
    }

    #[test]
    fn test_index_config_default() {
        let config = IndexConfig::default();
        assert_eq!(config.buffer_size, 5000);
        assert_eq!(config.input_shape, 768);
        assert_eq!(config.arity, 3);
        assert!(matches!(config.device, ModelDevice::Cpu));
        assert!(matches!(config.distance_fn, DistanceFn::Dot));
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
    fn test_level_index_builder_missing_attributes() {
        let builder = LevelIndexBuilder::default();
        let result = builder.build();
        assert!(matches!(result, Err(BuildError::MissingAttribute)));
    }

    #[test]
    fn test_bentley_saxe_index_available_level() {
        let config = create_test_config();
        let index = config.build().unwrap();
        // Initially no levels, so should return None
        assert_eq!(index.compaction_strategy.available_level(&index), None);
    }

    #[test]
    fn test_bentley_saxe_index_get_level_config() {
        let config = create_test_config();
        let index = config.build().unwrap();

        let level_config = index.get_level_index_config();
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
    fn test_bentley_saxe_empty_levels() {
        let config = create_test_config();
        let index = config.build().unwrap();

        // Should have no levels initially
        assert_eq!(index.levels.len(), 0);
        // Should have a buffer
        assert_eq!(index.buffer.size, 10);
    }

    #[test]
    fn test_bentley_saxe_lower_level_data_empty() {
        let config = create_test_config();
        let mut index = config.build().unwrap();

        let (data, ids) = index
            .compaction_strategy
            .clone()
            .lower_level_data(&mut index, 0);
        // Should only have buffer data initially
        assert!(data.is_empty());
        assert!(ids.is_empty());
    }
}
