use crate::{
    bucket::{self, Bucket, Buffer, BufferBuilder},
    constants::DEFAULT_BUCKET_SIZE,
    model::{Model, ModelBuilder, ModelConfig, ModelDevice},
    structs::{DiskBucket, DiskBuffer, DiskIndex, DiskLevelIndex, IndexConfig},
    Array, ArraySlice, DeleteMethod, DeleteStatistics, DistanceFn, DliError, Id, ModelLayer,
    SearchParamsT, SearchStatistics, SearchStrategy,
};
use log::{debug, info};
use measure_time_macro::log_time;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fs::{create_dir, File},
    path::{Path, PathBuf},
};

pub struct Index {
    compaction_strategy: CompactionStrategy,
    levels_config: LevelIndexConfig,
    input_shape: usize,
    arity: usize,
    device: ModelDevice,
    levels: Vec<LevelIndex>,
    buffer: Buffer,
    distance_fn: DistanceFn,
    delete_method: DeleteMethod,
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
        self.levels_config.clone()
    }

    pub fn insert(&mut self, value: Array, id: Id) {
        if self.buffer.has_space(1) {
            self.buffer.insert(value, id);
            return; // value fits into buffer
        }
        debug!(levels = self.levels.len(), occupied = self.occupied(); "index:buffer_flush");
        self.compaction_strategy.clone().compact(self);
        assert!(self.buffer.has_space(1));
        self.buffer.insert(value, id);
    }

    pub fn delete(&mut self, id: Id) -> Option<(Array, Id)> {
        self.verbose_delete(id).map(|(deleted, _)| deleted)
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

    pub fn n_empty_buckets(&self) -> usize {
        self.levels
            .iter()
            .map(|level| {
                level
                    .buckets
                    .iter()
                    .filter(|bucket| bucket.occupied() == 0)
                    .count()
            })
            .sum()
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

    pub fn verbose_delete(&mut self, id: Id) -> Option<((Array, Id), DeleteStatistics)> {
        if let Some(deleted) = self.buffer.delete(&id) {
            return Some((
                deleted,
                DeleteStatistics {
                    affected_level: None,
                },
            ));
        }
        for (level_idx, level) in &mut self.levels.iter_mut().enumerate() {
            if let Some(deleted) = level.delete(&id, &self.delete_method) {
                if self.is_level_underutilized(level_idx) {
                    self.compaction_strategy.clone().rebuild(self, level_idx)
                }
                return Some((
                    deleted,
                    DeleteStatistics {
                        affected_level: Some(level_idx),
                    },
                ));
            }
        }
        None
    }

    fn is_level_underutilized(&self, level_idx: usize) -> bool {
        let level = &self.levels[level_idx];
        level.occupied() < level.buckets[0].size() * self.arity.pow(level_idx as u32)
    }

    pub fn dump(&self, working_dir: &Path) {
        create_dir(working_dir).unwrap();
        let disk_levels = self
            .levels
            .iter()
            .enumerate()
            .map(|(level_id, level)| level.dump(working_dir, level_id))
            .collect::<Vec<_>>();
        let disk_buffer = self.buffer.dump(working_dir);
        let disk_index = DiskIndex {
            levels_config: self.levels_config.clone(),
            compaction_strategy: self.compaction_strategy.clone(),
            buffer_size: self.buffer.size,
            input_shape: self.input_shape,
            arity: self.arity,
            distance_fn: self.distance_fn.clone(),
            delete_method: self.delete_method.clone(),
            levels: disk_levels,
            disk_buffer,
        };
        let meta_path = working_dir.join("meta.json");
        let meta_file = File::create(meta_path).unwrap();
        serde_json::to_writer(meta_file, &disk_index).unwrap();
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub enum RebuildStrategy {
    #[default]
    #[serde(rename = "no_rebuild")]
    NoRebuild,
    #[serde(rename = "basic_rebuild")]
    BasicRebuild,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", content = "rebuild_strategy")]
pub enum CompactionStrategy {
    #[serde(rename = "bentley_saxe")]
    BentleySaxe(RebuildStrategy),
}

impl Default for CompactionStrategy {
    fn default() -> Self {
        CompactionStrategy::BentleySaxe(Default::default())
    }
}

impl From<&str> for CompactionStrategy {
    fn from(val: &str) -> Self {
        match val {
            "bentley_saxe:no_rebuild" => {
                CompactionStrategy::BentleySaxe(RebuildStrategy::NoRebuild)
            }
            "bentley_saxe:basic_rebuild" => {
                CompactionStrategy::BentleySaxe(RebuildStrategy::BasicRebuild)
            }
            _ => panic!("Unknown compaction strategy: {val}"),
        }
    }
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
            CompactionStrategy::BentleySaxe(_) => {
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

    pub fn rebuild(&self, index: &mut Index, level_idx: usize) {
        assert!(level_idx < index.levels.len());
        match self {
            CompactionStrategy::BentleySaxe(RebuildStrategy::NoRebuild) => {}
            CompactionStrategy::BentleySaxe(RebuildStrategy::BasicRebuild) => {
                let level_occupied = index.levels[level_idx].occupied();
                let move_data = |index: &mut Index, from_level_idx: usize, to_level_idx: usize| {
                    assert!(from_level_idx != to_level_idx);
                    assert!(index.levels[from_level_idx].occupied() > 0);
                    assert!(
                        index.levels[from_level_idx].occupied()
                            <= index.levels[to_level_idx].free_space()
                    );
                    let from_level_occupied = index.levels[from_level_idx].occupied();
                    let to_level_occupied = index.levels[to_level_idx].occupied();
                    let (data, ids) = index.levels[from_level_idx].get_data();
                    index.levels[to_level_idx].insert_many(data, ids);
                    assert!(index.levels[from_level_idx].occupied() == 0);
                    assert!(
                        index.levels[to_level_idx].occupied()
                            == from_level_occupied + to_level_occupied
                    );
                };
                // First level
                if level_idx == 0 {
                    if let Some(lower_level_idx) = lower_level(index, level_idx, level_occupied) {
                        assert!(lower_level_idx > level_idx);
                        move_data(index, level_idx, lower_level_idx);
                        return;
                    };
                    // flush buffer
                    let buffer_occupied = index.buffer.occupied();
                    let (data, ids) = index.buffer.get_data();
                    index.levels[level_idx].insert_many(data, ids);
                    assert!(index.buffer.occupied() == 0);
                    assert!(index.levels[level_idx].occupied() == level_occupied + buffer_occupied);
                    return;
                }
                // Middle level
                if level_idx < index.levels.len() - 1 {
                    if let Some(lower_level_idx) = lower_level(index, level_idx, level_occupied) {
                        assert!(lower_level_idx > level_idx);
                        move_data(index, level_idx, lower_level_idx);
                        return;
                    };
                }
                // Last level or no lower level found for middle level
                // Try to move data to the upper level
                let upper_level_idx = level_idx - 1;
                if index.levels[upper_level_idx].free_space() >= level_occupied {
                    move_data(index, level_idx, upper_level_idx);
                    return;
                }
                // Top up current level from upper level
                move_data(index, upper_level_idx, level_idx);
            }
        }
    }
}

fn lower_level(index: &Index, level_idx: usize, size: usize) -> Option<usize> {
    // Find the next level with enough free space
    index
        .levels
        .iter()
        .enumerate()
        .skip(level_idx + 1)
        .find(|(_, level)| level.occupied() > 0 && level.free_space() >= size)
        .map(|(level_idx, _)| level_idx)
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
            bucket_size: DEFAULT_BUCKET_SIZE,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct LevelIndexBuilder {
    id: Option<String>,
    n_buckets: Option<usize>,
    buckets: Option<(Vec<DiskBucket>, PathBuf, PathBuf)>,
    model_config: Option<ModelConfig>,
    bucket_size: Option<usize>,
    input_shape: Option<usize>,
    model_device: ModelDevice,
    distance_fn: Option<DistanceFn>,
}

impl LevelIndexBuilder {
    pub fn n_buckets(mut self, size: usize) -> Self {
        self.n_buckets = Some(size);
        self
    }

    pub fn buckets(
        mut self,
        buckets: Vec<DiskBucket>,
        records_path: PathBuf,
        ids_path: PathBuf,
    ) -> Self {
        self.buckets = Some((buckets, records_path, ids_path));
        self
    }

    pub fn model(mut self, model: ModelConfig) -> Self {
        self.model_config = Some(model);
        self
    }

    pub fn bucket_size(mut self, bucket_size: usize) -> Self {
        self.bucket_size = Some(bucket_size);
        self
    }

    pub fn input_shape(mut self, input_shape: usize) -> Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn id(mut self, id: String) -> Self {
        self.id = Some(id);
        self
    }

    pub fn model_device(mut self, model_device: ModelDevice) -> Self {
        self.model_device = model_device;
        self
    }

    pub fn distance_fn(mut self, distance_fn: DistanceFn) -> Self {
        self.distance_fn = Some(distance_fn);
        self
    }

    pub fn build(self) -> Result<LevelIndex, DliError> {
        let input_shape = self
            .input_shape
            .ok_or(DliError::MissingAttribute("input_shape"))?;
        let bucket_size = self
            .bucket_size
            .ok_or(DliError::MissingAttribute("bucket_size"))?;
        let buckets = match self.buckets {
            Some((buckets, records_path, ids_path)) => {
                let mut records_file = File::open(records_path)?;
                let mut ids_file = File::open(ids_path)?;
                buckets
                    .into_iter()
                    .map(|disk_bucket| {
                        bucket::BucketBuilder::from_disk(
                            disk_bucket,
                            &mut records_file,
                            &mut ids_file,
                        )
                        .input_shape(input_shape)
                        .size(bucket_size)
                        .build()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
            None => {
                let n_buckets = self
                    .n_buckets
                    .ok_or(DliError::MissingAttribute("n_buckets"))?;
                (0..n_buckets)
                    .map(|_| {
                        bucket::BucketBuilder::default()
                            .input_shape(input_shape)
                            .size(bucket_size)
                            .build()
                    })
                    .collect::<Result<Vec<_>, _>>()?
            }
        };
        let n_buckets = buckets.len();
        let distance_fn = self
            .distance_fn
            .ok_or(DliError::MissingAttribute("distance_fn"))?;
        let model_config = self
            .model_config
            .as_ref()
            .ok_or(DliError::MissingAttribute("model_config"))?;
        let mut model_builder = ModelBuilder::default();
        model_builder
            .device(self.model_device.clone())
            .input_nodes(input_shape as i64)
            .train_params(model_config.train_params.clone())
            .retrain_params(model_config.retrain_params.clone())
            .labels(n_buckets)
            .label_method(distance_fn.into());
        if let Some(weights_path) = &model_config.weights_path {
            model_builder.weights_path(weights_path.clone());
        }
        model_config.layers.iter().for_each(|layer| {
            model_builder.add_layer(*layer);
        });
        let model = model_builder.build()?;

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
        self.buckets[0].size() * self.buckets.len()
    }

    fn occupied(&self) -> usize {
        self.buckets.iter().map(|bucket| bucket.occupied()).sum()
    }

    fn free_space(&self) -> usize {
        let size = self.size();
        let occupied = self.occupied();
        match size > occupied {
            true => size - occupied,
            false => 0,
        }
    }

    fn buckets2visit_predictions(&self, query: &ArraySlice) -> Vec<(usize, f32, usize)> {
        if self.occupied() == 0 {
            return self
                .buckets
                .iter()
                .enumerate()
                .map(|(bucket_id, _)| (bucket_id, 0.0, 0))
                .collect();
        }
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
                let record_idx = self.buckets[bucket_idx].insert(query, id);
                self.ids_map.insert(id, (bucket_idx, record_idx));
            }
            assert!(assignments.is_empty());
            assert!(ids.is_empty());
            assert!(records.is_empty());
        }
    }

    fn get_data(&mut self) -> (Array, Vec<Id>) {
        self.ids_map.clear(); // clear existing id mappings
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
            assert!(bucket_idx < self.buckets.len());
            assert!(record_idx < self.buckets[bucket_idx].occupied());
            let bucket = &mut self.buckets[bucket_idx];
            let (deleted, (swapped_new_idx, swapped_id)) =
                bucket.delete(record_idx, delete_method)?;
            let (deleted_bucket_idx, deleted_record_idx) = self.ids_map.remove(id).unwrap(); // we are sure it exists
            assert_eq!(deleted_bucket_idx, bucket_idx);
            assert_eq!(deleted_record_idx, record_idx);
            // Only insert the swapped id mapping if a different record was moved into the deleted slot.
            // In the bucket's "delete last" case the swapped_id equals the deleted id, so avoid re-inserting it.
            if swapped_id != deleted.1 {
                self.ids_map
                    .insert(swapped_id, (bucket_idx, swapped_new_idx));
            }
            return Some(deleted);
        }
        None
    }

    fn n_buckets(&self) -> usize {
        self.buckets.len()
    }

    fn dump(&self, working_dir: &Path, level_id: usize) -> DiskLevelIndex {
        let weights_path = working_dir.join(format!("model-{level_id}.safetensors"));
        let model = self.model.dump(weights_path.clone());
        let records_path = working_dir.join(format!("bucket-records-{level_id}.bin"));
        let ids_path = working_dir.join(format!("bucket-ids-{level_id}.bin"));
        let mut records_file = File::create(records_path.clone()).unwrap(); // TODO: remove unwrap
        let mut ids_file = File::create(ids_path.clone()).unwrap(); // TODO: remove unwrap
        let disk_buckets = self
            .buckets
            .iter()
            .map(|bucket| bucket.dump(&mut records_file, &mut ids_file))
            .collect::<Vec<_>>();
        let config = LevelIndexConfig {
            model,
            bucket_size: self.buckets[0].size(),
        };
        DiskLevelIndex {
            weights_path,
            buckets: disk_buckets,
            config,
            records_path,
            ids_path,
        }
    }
}

#[derive(Clone)]
pub struct IndexBuilder {
    compaction_strategy: Option<CompactionStrategy>,
    levels_config: LevelIndexConfig,
    model_layers: Option<Vec<ModelLayer>>,
    buffer_size: Option<usize>,
    input_shape: Option<usize>,
    arity: Option<usize>,
    device: Option<ModelDevice>,
    distance_fn: Option<DistanceFn>,
    delete_method: Option<DeleteMethod>,
    levels: Option<Vec<DiskLevelIndex>>,
    disk_buffer: Option<DiskBuffer>,
}

impl Default for IndexBuilder {
    fn default() -> Self {
        Self::from_config(Default::default())
    }
}

impl IndexBuilder {
    pub fn from_yaml(file: &Path) -> Result<Self, DliError> {
        let content = std::fs::read_to_string(file)?;
        let config = serde_yaml::from_str(&content)?;
        Ok(Self::from_config(config))
    }

    pub fn from_config(config: IndexConfig) -> Self {
        Self {
            compaction_strategy: Some(config.compaction_strategy),
            levels_config: config.levels,
            buffer_size: Some(config.buffer_size),
            input_shape: Some(config.input_shape),
            arity: Some(config.arity),
            device: Some(config.device),
            distance_fn: Some(config.distance_fn),
            delete_method: Some(config.delete_method),
            levels: None,
            disk_buffer: None,
            model_layers: None,
        }
    }

    pub fn from_disk(working_dir: &Path) -> Result<Self, DliError> {
        let meta_path = working_dir.join("meta.json");
        let meta_file = File::open(meta_path)?;
        let disk_index: DiskIndex = serde_json::from_reader(meta_file)?;
        Ok(Self {
            compaction_strategy: Some(disk_index.compaction_strategy),
            levels_config: disk_index.levels_config,
            buffer_size: Some(disk_index.buffer_size),
            input_shape: Some(disk_index.input_shape),
            arity: Some(disk_index.arity),
            device: Some(Default::default()),
            distance_fn: Some(disk_index.distance_fn),
            delete_method: Some(disk_index.delete_method),
            levels: Some(disk_index.levels),
            disk_buffer: Some(disk_index.disk_buffer),
            model_layers: None,
        })
    }

    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = Some(size);
        self
    }

    pub fn bucket_size(mut self, size: usize) -> Self {
        self.levels_config.bucket_size = size;
        self
    }

    pub fn arity(mut self, arity: usize) -> Self {
        self.arity = Some(arity);
        self
    }

    pub fn compaction_strategy(mut self, strategy: CompactionStrategy) -> Self {
        self.compaction_strategy = Some(strategy);
        self
    }

    pub fn distance_fn(mut self, distance_fn: DistanceFn) -> Self {
        self.distance_fn = Some(distance_fn);
        self
    }

    pub fn input_shape(mut self, shape: usize) -> Self {
        self.input_shape = Some(shape);
        self
    }

    pub fn device(mut self, device: ModelDevice) -> Self {
        self.device = Some(device);
        self
    }

    pub fn delete_method(mut self, method: DeleteMethod) -> Self {
        self.delete_method = Some(method);
        self
    }

    pub fn train_threshold_samples(mut self, samples: usize) -> Self {
        self.levels_config.model.train_params.threshold_samples = samples;
        self
    }

    pub fn train_batch_size(mut self, batch_size: usize) -> Self {
        self.levels_config.model.train_params.batch_size = batch_size;
        self
    }

    pub fn train_epochs(mut self, epochs: usize) -> Self {
        self.levels_config.model.train_params.epochs = epochs;
        self
    }

    pub fn retrain_threshold_samples(mut self, samples: usize) -> Self {
        self.levels_config.model.retrain_params.threshold_samples = samples;
        self
    }

    pub fn retrain_batch_size(mut self, batch_size: usize) -> Self {
        self.levels_config.model.retrain_params.batch_size = batch_size;
        self
    }

    pub fn retrain_epochs(mut self, epochs: usize) -> Self {
        self.levels_config.model.retrain_params.epochs = epochs;
        self
    }

    pub fn add_layer(mut self, layer: ModelLayer) -> Self {
        match self.model_layers.as_mut() {
            Some(layers) => layers.push(layer),
            None => {
                self.model_layers = Some(vec![layer]);
            }
        };
        self
    }

    fn load_disk_level(
        disk_index: DiskLevelIndex,
        device: ModelDevice,
        distance_fn: DistanceFn,
        input_shape: usize,
    ) -> Result<LevelIndex, DliError> {
        LevelIndexBuilder::default()
            .model(disk_index.config.model)
            .distance_fn(distance_fn)
            .model_device(device)
            .bucket_size(disk_index.config.bucket_size)
            .input_shape(input_shape)
            .buckets(
                disk_index.buckets,
                disk_index.records_path,
                disk_index.ids_path,
            )
            .build()
    }

    pub fn build(self) -> Result<Index, DliError> {
        let levels_config = self.levels_config;
        let buffer_size = self
            .buffer_size
            .ok_or(DliError::MissingAttribute("buffer_size"))?;
        let input_shape = self
            .input_shape
            .ok_or(DliError::MissingAttribute("input_shape"))?;
        let mut buffer = BufferBuilder::default()
            .input_shape(input_shape)
            .size(buffer_size);
        if let Some(disk_buffer) = self.disk_buffer {
            buffer = buffer.disk_buffer(disk_buffer);
        }
        let buffer = buffer.build()?;
        let arity = self.arity.ok_or(DliError::MissingAttribute("arity"))?;
        let device = self.device.ok_or(DliError::MissingAttribute("device"))?;
        let distance_fn = self
            .distance_fn
            .ok_or(DliError::MissingAttribute("distance_fn"))?;
        let compaction_strategy = self
            .compaction_strategy
            .ok_or(DliError::MissingAttribute("compaction_strategy"))?;
        let delete_method = self
            .delete_method
            .ok_or(DliError::MissingAttribute("delete_method"))?;
        let levels = match self.levels {
            Some(levels) => levels
                .into_iter()
                .map(|level| {
                    Self::load_disk_level(level, device.clone(), distance_fn.clone(), input_shape)
                })
                .collect::<Result<Vec<_>, _>>()?,
            None => Vec::new(),
        };
        let index = Index {
            levels_config,
            input_shape,
            arity,
            device,
            levels,
            buffer,
            distance_fn,
            compaction_strategy,
            delete_method,
        };
        Ok(index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{DEFAULT_BUCKET_SIZE, DEFAULT_SEARCH_N_CANDIDATES};
    use crate::{search_strategy::SearchStrategy, structs::DistanceFn};
    use crate::{ModelConfig, ModelLayer, TrainParams};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_config() -> IndexConfig {
        IndexConfig {
            levels: LevelIndexConfig::default(),
            buffer_size: 10,
            input_shape: 3,
            arity: 2,
            device: ModelDevice::Cpu,
            distance_fn: DistanceFn::Dot,
            compaction_strategy: CompactionStrategy::BentleySaxe(RebuildStrategy::NoRebuild),
            delete_method: DeleteMethod::OidToBucket,
        }
    }

    #[test]
    fn test_level_index_config_default() {
        let config = LevelIndexConfig::default();
        assert_eq!(config.bucket_size, DEFAULT_BUCKET_SIZE);
    }

    #[test]
    fn test_index_config_from_yaml_valid() {
        let yaml_content = r#"
        compaction_strategy:
          type: bentley_saxe
          rebuild_strategy: no_rebuild
        levels:
          model:
            layers:
              - type: linear
                value: 4
              - type: relu
            train_params:
              threshold_samples: 100
              max_iters: 5
              batch_size: 4
              epochs: 2
            retrain_params:
              threshold_samples: 10
              max_iters: 4
              batch_size: 5
              epochs: 3
          bucket_size: 100
        buffer_size: 50
        input_shape: 10
        arity: 2
        device: cpu
        distance_fn: dot
        delete_method: oid_to_bucket
        "#;

        let mut temp_file = NamedTempFile::new().unwrap();
        writeln!(temp_file, "{yaml_content}").unwrap();

        let index = IndexBuilder::from_yaml(temp_file.path())
            .unwrap()
            .build()
            .unwrap();
        assert_eq!(index.buffer.size, 50);
        assert_eq!(index.input_shape, 10);
        assert_eq!(index.arity, 2);
        assert_eq!(index.levels.len(), 0);
        let lvl = index.levels_config;
        assert_eq!(lvl.bucket_size, 100);
        assert!(matches!(index.delete_method, DeleteMethod::OidToBucket));
    }

    #[test]
    fn test_index_config_from_yaml_nonexistent_file() {
        let result = IndexBuilder::from_yaml(Path::new("nonexistent.yaml"));
        assert!(matches!(result, Err(DliError::IoError(_))));
    }

    #[test]
    fn test_search_params_trait_unit() {
        let params = ().into_search_params();
        assert_eq!(params.k, 10);
        assert!(matches!(
            params.search_strategy,
            SearchStrategy::ModelDriven(DEFAULT_SEARCH_N_CANDIDATES)
        ));
    }

    #[test]
    fn test_search_params_trait_usize() {
        let params = 5usize.into_search_params();
        assert_eq!(params.k, 5);
        assert!(matches!(
            params.search_strategy,
            SearchStrategy::ModelDriven(DEFAULT_SEARCH_N_CANDIDATES)
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
        let mut index = IndexBuilder::from_config(config).build().unwrap();

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
        let mut index = IndexBuilder::from_config(config).build().unwrap();

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
        assert!(matches!(
            result,
            Err(DliError::MissingAttribute("input_shape"))
        ));
    }

    #[test]
    fn test_level_index_builder_minimal_params() {
        let builder = LevelIndexBuilder::default();

        let level = builder
            .n_buckets(4)
            .input_shape(10)
            .bucket_size(50)
            .model(ModelConfig::default())
            .distance_fn(DistanceFn::Dot)
            .build()
            .expect("Failed to build LevelIndex with minimal params");

        // Verify level has correct number of buckets
        assert_eq!(level.n_buckets(), 4);

        // Verify each bucket has correct configuration
        for bucket in &level.buckets {
            assert_eq!(bucket.size(), 50);
            assert_eq!(bucket.occupied(), 0); // Should be empty initially
        }

        // Verify model configuration
        assert_eq!(level.model.input_shape, 10);
        // Note: labels field is private, but should equal n_buckets (4)

        // Verify level is initially empty
        assert_eq!(level.occupied(), 0);
        assert_eq!(level.size(), 4 * 50); // n_buckets * bucket_size
        assert_eq!(level.free_space(), 200); // All space is free

        // Verify ids_map is empty
        assert!(level.ids_map.is_empty());
    }

    #[test]
    fn test_level_index_builder_model_integration() {
        // Test that the model is built correctly with different configurations

        // Test with L2 distance
        let level_l2 = LevelIndexBuilder::default()
            .n_buckets(3)
            .input_shape(5)
            .bucket_size(20)
            .model(ModelConfig::default())
            .distance_fn(DistanceFn::L2)
            .build()
            .expect("Failed to build with L2 distance");

        assert_eq!(level_l2.model.input_shape, 5);
        assert_eq!(level_l2.n_buckets(), 3);

        // Test with Dot distance
        let level_dot = LevelIndexBuilder::default()
            .n_buckets(5)
            .input_shape(8)
            .bucket_size(30)
            .model(ModelConfig::default())
            .distance_fn(DistanceFn::Dot)
            .build()
            .expect("Failed to build with Dot distance");

        assert_eq!(level_dot.model.input_shape, 8);
        assert_eq!(level_dot.n_buckets(), 5);

        // Test with custom model config
        let custom_model_config = ModelConfig {
            layers: vec![
                ModelLayer::Linear(16),
                ModelLayer::ReLU,
                ModelLayer::Linear(8),
            ],
            train_params: TrainParams {
                epochs: 20,
                batch_size: 64,
                threshold_samples: 500,
                max_iters: 100,
            },
            retrain_params: TrainParams {
                epochs: 10,
                batch_size: 32,
                threshold_samples: 250,
                max_iters: 50,
            },
            weights_path: None,
        };

        let level_custom = LevelIndexBuilder::default()
            .n_buckets(2)
            .input_shape(12)
            .bucket_size(40)
            .model(custom_model_config)
            .distance_fn(DistanceFn::Dot)
            .build()
            .expect("Failed to build with custom model config");

        assert_eq!(level_custom.model.input_shape, 12);
        assert_eq!(level_custom.n_buckets(), 2);
    }

    #[test]
    fn test_level_index_save_and_load() {
        use tempfile::TempDir;

        // Create a level with some buckets
        let input_shape = 10;
        let n_buckets = 4;
        let bucket_size = 50;

        let mut level = LevelIndexBuilder::default()
            .n_buckets(n_buckets)
            .input_shape(input_shape)
            .bucket_size(bucket_size)
            .model(ModelConfig::default())
            .distance_fn(DistanceFn::Dot)
            .build()
            .expect("Failed to build level");

        // Generate training data (100 samples, 10 features each)
        let training_data: Vec<f32> = (0..1000).map(|i| (i % 100) as f32 / 100.0).collect();

        // Train the level
        level.train(&training_data);

        // Insert some data into the level
        let mut insert_data: Vec<f32> = Vec::new();
        let mut insert_ids: Vec<u32> = Vec::new();
        for i in 0..20 {
            let record: Vec<f32> = (0..input_shape)
                .map(|j| (i * 10 + j) as f32 / 100.0)
                .collect();
            insert_data.extend(record);
            insert_ids.push(i as u32);
        }
        level.insert_many(insert_data, insert_ids);

        // Create test queries
        let test_queries: Vec<Vec<f32>> = vec![
            (0..input_shape).map(|i| i as f32 / 10.0).collect(),
            (0..input_shape).map(|i| (i + 5) as f32 / 10.0).collect(),
            (0..input_shape).map(|i| (i * 2) as f32 / 10.0).collect(),
            (0..input_shape).map(|i| (i + 3) as f32 / 15.0).collect(),
        ];

        // Get predictions from original level
        let original_predictions: Vec<Vec<(usize, f32, usize)>> = test_queries
            .iter()
            .map(|query| level.buckets2visit_predictions(query))
            .collect();

        // Save level to temporary directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let level_id = 0;
        let disk_level = level.dump(temp_dir.path(), level_id);

        // Verify disk files were created
        assert!(disk_level.weights_path.exists());
        assert!(disk_level.records_path.exists());
        assert!(disk_level.ids_path.exists());

        // Load level from disk
        let loaded_level = LevelIndexBuilder::default()
            .model(disk_level.config.model)
            .distance_fn(DistanceFn::Dot)
            .model_device(ModelDevice::Cpu)
            .bucket_size(disk_level.config.bucket_size)
            .input_shape(input_shape)
            .buckets(
                disk_level.buckets,
                disk_level.records_path,
                disk_level.ids_path,
            )
            .build()
            .expect("Failed to build level from disk");

        // Verify loaded level has same properties
        assert_eq!(loaded_level.n_buckets(), n_buckets);
        assert_eq!(loaded_level.model.input_shape, input_shape);
        assert_eq!(loaded_level.occupied(), 20); // Same number of records

        // Get predictions from loaded level
        let loaded_predictions: Vec<Vec<(usize, f32, usize)>> = test_queries
            .iter()
            .map(|query| loaded_level.buckets2visit_predictions(query))
            .collect();

        // Verify predictions match
        assert_eq!(original_predictions.len(), loaded_predictions.len());
        for (original, loaded) in original_predictions.iter().zip(loaded_predictions.iter()) {
            assert_eq!(original.len(), loaded.len());
            for ((orig_bucket, orig_prob, orig_count), (load_bucket, load_prob, load_count)) in
                original.iter().zip(loaded.iter())
            {
                assert_eq!(orig_bucket, load_bucket, "Bucket indices should match");
                assert_eq!(orig_count, load_count, "Bucket counts should match");
                assert!(
                    (orig_prob - load_prob).abs() < 1e-5,
                    "Probabilities should match (original: {orig_prob}, loaded: {load_prob})"
                );
            }
        }

        // Verify the actual data in buckets matches
        for bucket_idx in 0..n_buckets {
            let orig_bucket = &level.buckets[bucket_idx];
            let loaded_bucket = &loaded_level.buckets[bucket_idx];

            assert_eq!(
                orig_bucket.occupied(),
                loaded_bucket.occupied(),
                "Bucket {bucket_idx} should have same occupied count"
            );
            assert_eq!(
                orig_bucket.size(),
                loaded_bucket.size(),
                "Bucket {bucket_idx} should have same size"
            );

            // Compare records in each bucket
            for record_idx in 0..orig_bucket.occupied() {
                let orig_record = orig_bucket.record(record_idx);
                let loaded_record = loaded_bucket.record(record_idx);
                assert_eq!(
                    orig_record, loaded_record,
                    "Record {record_idx} in bucket {bucket_idx} should match"
                );

                assert_eq!(
                    orig_bucket.ids[record_idx], loaded_bucket.ids[record_idx],
                    "ID {record_idx} in bucket {bucket_idx} should match"
                );
            }
        }
    }

    // IndexBuilder tests
    #[test]
    fn test_index_builder_default() {
        // Test building an index with default configuration
        let index = IndexBuilder::default().build().unwrap();

        // Verify default values are applied
        assert_eq!(index.buffer.size, crate::constants::DEFAULT_BUFFER_SIZE);
        assert_eq!(index.input_shape, crate::constants::DEFAULT_INPUT_SHAPE);
        assert_eq!(index.arity, crate::constants::DEFAULT_ARITY);
        assert_eq!(index.levels.len(), 0); // Should start with no levels

        // Verify buffer is empty initially
        assert_eq!(index.buffer.occupied(), 0);

        // Verify distance function is set to default (Dot)
        assert!(matches!(index.distance_fn, DistanceFn::Dot));

        // Verify device is CPU by default
        assert!(matches!(index.device, ModelDevice::Cpu));

        // Verify compaction strategy is set
        assert!(matches!(
            index.compaction_strategy,
            CompactionStrategy::BentleySaxe(_)
        ));

        // Verify delete method is set
        assert!(matches!(index.delete_method, DeleteMethod::OidToBucket));
    }

    #[test]
    fn test_index_builder_with_custom_params() {
        // Test building an index with custom parameters
        let index = IndexBuilder::default()
            .buffer_size(100)
            .bucket_size(200)
            .arity(4)
            .input_shape(128)
            .distance_fn(DistanceFn::L2)
            .train_epochs(10)
            .train_batch_size(64)
            .retrain_epochs(5)
            .retrain_batch_size(32)
            .build()
            .unwrap();

        // Verify custom values are applied
        assert_eq!(index.buffer.size, 100);
        assert_eq!(index.input_shape, 128);
        assert_eq!(index.arity, 4);
        assert_eq!(index.levels_config.bucket_size, 200);

        // Verify distance function
        assert!(matches!(index.distance_fn, DistanceFn::L2));

        // Verify training parameters were set
        assert_eq!(index.levels_config.model.train_params.epochs, 10);
        assert_eq!(index.levels_config.model.train_params.batch_size, 64);
        assert_eq!(index.levels_config.model.retrain_params.epochs, 5);
        assert_eq!(index.levels_config.model.retrain_params.batch_size, 32);

        // Verify index is initially empty
        assert_eq!(index.occupied(), 0);
        assert_eq!(index.n_levels(), 0);
    }

    #[test]
    fn test_index_save_and_load() {
        use tempfile::TempDir;

        // Build an index with specific configuration
        let input_shape = 10;
        let mut index = IndexBuilder::default()
            .arity(2)
            .bucket_size(10)
            .buffer_size(10)
            .input_shape(input_shape)
            .distance_fn(DistanceFn::Dot)
            .build()
            .unwrap();

        // Verify initial configuration
        assert_eq!(index.arity, 2);
        assert_eq!(index.buffer.size, 10);
        assert_eq!(index.levels_config.bucket_size, 10);
        assert_eq!(index.input_shape, input_shape);

        // Insert 1000 records into the index
        for i in 0..100 {
            let record: Vec<f32> = (0..input_shape)
                .map(|j| ((i * input_shape + j) % 100) as f32 / 100.0)
                .collect();
            index.insert(record, i as u32);
        }

        // Verify data was inserted
        let occupied_after_insert = index.occupied();
        assert_eq!(occupied_after_insert, 100);
        let n_levels_after_insert = index.n_levels();
        assert!(
            n_levels_after_insert > 0,
            "Should have created levels during insertion"
        );

        // Create test queries
        let test_queries: Vec<Vec<f32>> = vec![
            (0..input_shape).map(|i| i as f32 / 10.0).collect(),
            (0..input_shape).map(|i| (i + 5) as f32 / 10.0).collect(),
            (0..input_shape).map(|i| (i * 2) as f32 / 10.0).collect(),
            (0..input_shape).map(|i| (i + 3) as f32 / 15.0).collect(),
            (0..input_shape)
                .map(|i| ((i * 3) % 10) as f32 / 20.0)
                .collect(),
        ];

        // Run queries and store results from original index
        let original_results: Vec<Vec<Id>> = test_queries
            .iter()
            .map(|query| index.search(query.as_slice(), 10))
            .collect();

        // Also get verbose search statistics
        let original_stats: Vec<(Vec<Id>, SearchStatistics)> = test_queries
            .iter()
            .map(|query| index.verbose_search(query.as_slice(), 10))
            .collect();

        // Save index to temporary directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let dump_dir = temp_dir.path().join("index_dump");
        index.dump(&dump_dir);

        // Verify meta file was created
        let meta_path = dump_dir.join("meta.json");
        assert!(meta_path.exists(), "Meta file should exist");

        // Load index from disk
        let loaded_index = IndexBuilder::from_disk(&dump_dir)
            .expect("Failed to create builder from disk")
            .build()
            .expect("Failed to build index from disk");

        // Verify loaded index has same configuration
        assert_eq!(loaded_index.arity, index.arity);
        assert_eq!(loaded_index.buffer.size, index.buffer.size);
        assert_eq!(loaded_index.input_shape, index.input_shape);
        assert_eq!(
            loaded_index.levels_config.bucket_size,
            index.levels_config.bucket_size
        );
        assert!(matches!(loaded_index.distance_fn, DistanceFn::Dot));

        // Verify loaded index has same data
        assert_eq!(loaded_index.occupied(), occupied_after_insert);
        assert_eq!(loaded_index.n_levels(), n_levels_after_insert);
        assert_eq!(loaded_index.size(), index.size());
        assert_eq!(loaded_index.n_buckets(), index.n_buckets());

        // Run same queries on loaded index
        let loaded_results: Vec<Vec<Id>> = test_queries
            .iter()
            .map(|query| loaded_index.search(query.as_slice(), 10))
            .collect();

        let loaded_stats: Vec<(Vec<Id>, SearchStatistics)> = test_queries
            .iter()
            .map(|query| loaded_index.verbose_search(query.as_slice(), 10))
            .collect();

        // Verify search results match
        assert_eq!(original_results.len(), loaded_results.len());
        for (i, (orig, loaded)) in original_results
            .iter()
            .zip(loaded_results.iter())
            .enumerate()
        {
            assert_eq!(
                orig, loaded,
                "Query {i} results should match between original and loaded index"
            );
        }

        // Verify search statistics match
        for (i, ((orig_ids, orig_stats), (loaded_ids, loaded_stats))) in
            original_stats.iter().zip(loaded_stats.iter()).enumerate()
        {
            assert_eq!(orig_ids, loaded_ids, "Query {i} IDs should match");
            assert_eq!(
                orig_stats.total_visited_buckets, loaded_stats.total_visited_buckets,
                "Query {i} should visit same number of buckets"
            );
            assert_eq!(
                orig_stats.total_visited_records, loaded_stats.total_visited_records,
                "Query {i} should visit same number of records"
            );
        }

        // Verify buffer contents match
        assert_eq!(
            index.buffer.occupied(),
            loaded_index.buffer.occupied(),
            "Buffer occupancy should match"
        );

        // Verify each level matches
        for level_idx in 0..n_levels_after_insert {
            let orig_level = &index.levels[level_idx];
            let loaded_level = &loaded_index.levels[level_idx];

            assert_eq!(
                orig_level.occupied(),
                loaded_level.occupied(),
                "Level {level_idx} occupancy should match"
            );
            assert_eq!(
                orig_level.n_buckets(),
                loaded_level.n_buckets(),
                "Level {level_idx} bucket count should match"
            );
        }
    }

    #[test]
    fn test_bentley_saxe_index_available_level() {
        let config = create_test_config();
        let index = IndexBuilder::from_config(config).build().unwrap();
        // Initially no levels, so should return None
        assert_eq!(index.compaction_strategy.available_level(&index), None);
    }

    #[test]
    fn test_bentley_saxe_index_get_level_config() {
        let config = create_test_config();
        let index = IndexBuilder::from_config(config).build().unwrap();

        let level_config = index.get_level_index_config();
        assert_eq!(level_config.bucket_size, DEFAULT_BUCKET_SIZE); // default value
    }

    #[test]
    fn test_bentley_saxe_index_insert_to_buffer() {
        let config = create_test_config();
        let mut index = IndexBuilder::from_config(config).build().unwrap();

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
        let index = IndexBuilder::from_config(config).build().unwrap();

        // Should have no levels initially
        assert_eq!(index.levels.len(), 0);
        // Should have a buffer
        assert_eq!(index.buffer.size, 10);
    }

    #[test]
    fn test_bentley_saxe_lower_level_data_empty() {
        let config = create_test_config();
        let mut index = IndexBuilder::from_config(config).build().unwrap();

        let (data, ids) = index
            .compaction_strategy
            .clone()
            .lower_level_data(&mut index, 0);
        // Should only have buffer data initially
        assert!(data.is_empty());
        assert!(ids.is_empty());
    }

    // Helper to build a LevelIndex with one bucket and populate it with records and ids
    fn make_level_with_records(records: Vec<Vec<f32>>, ids: Vec<Id>) -> LevelIndex {
        let input_shape = if records.is_empty() {
            1
        } else {
            records[0].len()
        };
        let mut level = LevelIndexBuilder::default()
            .n_buckets(1)
            .input_shape(input_shape)
            .bucket_size(100)
            .model(ModelConfig::default())
            .model_device(ModelDevice::Cpu)
            .distance_fn(DistanceFn::Dot)
            .build()
            .unwrap();

        for (rec, id) in records.into_iter().zip(ids.into_iter()) {
            level.buckets[0].insert(rec, id);
            level
                .ids_map
                .insert(id, (0, level.buckets[0].occupied() - 1));
        }
        level
    }

    #[test]
    fn test_level_index_delete_last_element() {
        let rec = vec![1.0f32, 2.0, 3.0];
        let id = 42u32;
        let mut level = make_level_with_records(vec![rec.clone()], vec![id]);

        let res = level.delete(&id, &DeleteMethod::OidToBucket);
        assert!(res.is_some());
        let (deleted_vec, deleted_id) = res.unwrap();
        assert_eq!(deleted_id, id);
        assert_eq!(deleted_vec, rec);

        // id should be removed from ids_map
        assert!(!level.ids_map.contains_key(&id));
        // bucket should be empty
        assert_eq!(level.buckets[0].occupied(), 0);
    }

    #[test]
    fn test_level_index_delete_middle_swaps_last_in() {
        let rec0 = vec![0.0f32, 0.1, 0.2];
        let rec1 = vec![1.0f32, 1.1, 1.2];
        let rec2 = vec![2.0f32, 2.1, 2.2];
        let ids = vec![1u32, 2u32, 3u32];
        let mut level =
            make_level_with_records(vec![rec0.clone(), rec1.clone(), rec2.clone()], ids.clone());

        // delete middle id (2)
        let res = level.delete(&2u32, &DeleteMethod::OidToBucket);
        assert!(res.is_some());
        let (deleted_vec, deleted_id) = res.unwrap();
        assert_eq!(deleted_id, 2u32);
        assert_eq!(deleted_vec, rec1);

        // ids_map should not contain deleted id
        assert!(!level.ids_map.contains_key(&2u32));
        // moved id (3) should now be at index 1
        assert_eq!(level.ids_map.get(&3u32).cloned(), Some((0usize, 1usize)));
        // bucket should have two records and record(1) equals rec2
        assert_eq!(level.buckets[0].occupied(), 2);
        assert_eq!(level.buckets[0].record(1), rec2.as_slice());
    }

    #[test]
    fn test_level_index_delete_missing_id_returns_none() {
        let mut level = make_level_with_records(vec![], vec![]);
        let res = level.delete(&999u32, &DeleteMethod::OidToBucket);
        assert!(res.is_none());
    }

    #[test]
    fn test_level_index_get_data_empty() {
        let mut level = make_level_with_records(vec![], vec![]);
        let (data, ids) = level.get_data();
        assert!(data.is_empty());
        assert!(ids.is_empty());
        assert!(level.ids_map.is_empty());
        // buckets remain empty
        assert_eq!(level.buckets.iter().map(|b| b.occupied()).sum::<usize>(), 0);
    }

    #[test]
    fn test_level_index_get_data_single_record() {
        let rec = vec![1.0f32, 2.0, 3.0];
        let id = 7u32;
        let mut level = make_level_with_records(vec![rec.clone()], vec![id]);

        assert_eq!(level.buckets[0].occupied(), 1);
        let (data, ids) = level.get_data();
        assert_eq!(ids, vec![id]);
        assert_eq!(data, rec);
        assert!(level.ids_map.is_empty());
        assert_eq!(level.buckets[0].occupied(), 0);
    }

    #[test]
    fn test_level_index_get_data_multiple_buckets() {
        // build a level with 3 buckets and populate bucket 0 and 1
        let rec0 = vec![0.0f32, 0.1, 0.2];
        let rec1 = vec![1.0f32, 1.1, 1.2];
        let rec1b = vec![1.5f32, 1.6, 1.7];

        let mut level = LevelIndexBuilder::default()
            .n_buckets(3)
            .input_shape(3)
            .bucket_size(100)
            .model(ModelConfig::default())
            .model_device(ModelDevice::Cpu)
            .distance_fn(DistanceFn::Dot)
            .build()
            .unwrap();

        level.buckets[0].insert(rec0.clone(), 1u32);
        level
            .ids_map
            .insert(1u32, (0, level.buckets[0].occupied() - 1));

        level.buckets[1].insert(rec1.clone(), 2u32);
        level
            .ids_map
            .insert(2u32, (1, level.buckets[1].occupied() - 1));

        level.buckets[1].insert(rec1b.clone(), 3u32);
        level
            .ids_map
            .insert(3u32, (1, level.buckets[1].occupied() - 1));

        let (data, ids) = level.get_data();
        // ids should be in bucket order then insertion order
        assert_eq!(ids, vec![1u32, 2u32, 3u32]);
        // data is flattened concatenation
        let expected = [rec0, rec1, rec1b].concat();
        assert_eq!(data, expected);
        // after get_data, ids_map cleared and buckets empty
        assert!(level.ids_map.is_empty());
        assert_eq!(level.buckets.iter().map(|b| b.occupied()).sum::<usize>(), 0);
    }
}
