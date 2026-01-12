use crate::{
    bucket::{Buffer, BufferBuilder},
    level_index::{LevelIndex, LevelIndexBuilder},
    model::{ModelDevice, RetrainStrategy},
    structs::{
        DiskBuffer, DiskIndex, DiskLevelIndex, IndexConfig, LevelIndexConfig, Records2Visit,
    },
    Array, ArraySlice, DeleteMethod, DeleteStatistics, DistanceFn, DliError, DliResult, Id,
    ModelLayer, SearchParams, SearchParamsT, SearchStatistics, SearchStrategy,
};
use log::{debug, info};
use measure_time_macro::log_time;
use serde::{Deserialize, Serialize};
use std::{
    fs::{create_dir, File},
    path::{absolute, Path},
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
    pub fn search<S>(&self, query: &ArraySlice, params: S) -> DliResult<Vec<Id>>
    where
        S: SearchParamsT,
    {
        self.verbose_search(query, params).map(|(res, _)| res)
    }

    fn records2visit(
        &'_ self,
        predictions: Vec<Vec<(usize, f32, usize)>>,
        search_strategy: SearchStrategy,
    ) -> Records2Visit<'_> {
        match search_strategy {
            SearchStrategy::Base(_nprobe) => todo!(),
            SearchStrategy::ModelDriven(ncandidates) => {
                let arity = self.arity;
                let normalize_probability =
                    |prob: f32, level_idx| (arity.pow(level_idx) as f32) * prob.max(0.0);
                let levels = predictions.len();
                let mut buckets2visit = Vec::with_capacity(self.n_buckets() + 1);
                for (level_idx, level_predictions) in predictions.iter().enumerate() {
                    for (bucket_id, prob, occupied) in level_predictions {
                        buckets2visit.push((
                            level_idx,
                            *bucket_id,
                            normalize_probability(*prob, level_idx as u32),
                            *occupied,
                        ));
                    }
                }
                // add buffer as a special "bucket"
                buckets2visit.push((levels, self.n_buckets(), 1.0, self.buffer.occupied()));
                buckets2visit.sort_by(|a, b| b.2.total_cmp(&a.2));
                let total_visited_buckets = buckets2visit.len();
                let mut records = Vec::new();
                let mut ids = Vec::new();
                let mut total_occupied = 0;
                for (level_idx, bucket_id, _prob, occupied) in buckets2visit {
                    if occupied > 0 && total_occupied < ncandidates {
                        if level_idx == levels {
                            // buffer
                            for i in 0..self.buffer.occupied() {
                                records.push(self.buffer.record(i));
                                ids.push(self.buffer.ids[i]);
                            }
                        } else {
                            let level = &self.levels[level_idx];
                            let bucket = &level.bucket(bucket_id);
                            let occupied = bucket.occupied();
                            for i in 0..occupied {
                                records.push(bucket.record(i));
                                ids.push(bucket.ids[i]);
                            }
                        }
                        total_occupied += occupied;
                    }
                    if total_occupied >= ncandidates {
                        break;
                    }
                }
                Records2Visit {
                    records,
                    ids,
                    total_visited_buckets,
                }
            }
        }
    }

    pub fn add_level(&mut self) -> DliResult<usize> {
        let level_index_config = self.get_level_index_config();
        let n_buckets = self.arity.pow(self.levels.len() as u32 + 1);
        let level_index = LevelIndexBuilder::default()
            .id(format!("{}", self.levels.len()))
            .n_buckets(n_buckets)
            .input_shape(self.input_shape)
            .model(level_index_config.model.clone())
            .model_device(self.device)
            .bucket_size(level_index_config.bucket_size)
            .distance_fn(self.distance_fn.clone())
            .build()?;
        self.levels.push(level_index);
        Ok(self.levels.len() - 1)
    }

    fn get_level_index_config(&self) -> LevelIndexConfig {
        self.levels_config.clone()
    }

    #[log_time]
    pub fn insert(&mut self, value: Array, id: Id) -> DliResult<()> {
        self.buffer.insert(value, id);
        if self.buffer.has_space(1) {
            return Ok(()); // buffer is not full yet
        }
        debug!(levels = self.levels.len(), occupied = self.occupied(); "index:buffer_flush");
        self.compaction_strategy.clone().compact(self)?;
        assert!(self.buffer.occupied() == 0);
        Ok(())
    }

    pub fn delete(&mut self, id: Id) -> DliResult<Option<(Array, Id)>> {
        Ok(self.verbose_delete(id)?.map(|(deleted, _)| deleted))
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
            + self.buffer_occupied()
    }

    pub fn buffer_occupied(&self) -> usize {
        self.buffer.occupied()
    }

    pub fn level_occupied(&self, level_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        self.levels[level_idx].occupied()
    }

    pub fn level_n_buckets(&self, level_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        self.levels[level_idx].n_buckets()
    }

    pub fn level_total_size(&self, level_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        self.levels[level_idx].size()
    }

    pub fn level_n_empty_buckets(&self, level_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        self.levels[level_idx].n_empty_buckets()
    }

    pub fn bucket_occupied(&self, level_idx: usize, bucket_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        assert!(bucket_idx < self.levels[level_idx].n_buckets());
        self.levels[level_idx].bucket(bucket_idx).occupied()
    }

    pub fn n_empty_buckets(&self) -> usize {
        self.levels
            .iter()
            .map(|level| level.n_empty_buckets())
            .sum()
    }

    #[log_time]
    fn bucket_selection(&self, query: &ArraySlice) -> DliResult<Vec<Vec<(usize, f32, usize)>>> {
        self.levels
            .iter()
            .map(|level| level.buckets2visit_predictions(query))
            .collect::<DliResult<Vec<_>>>()
    }

    #[log_time]
    fn merge_results(
        &self,
        records2visit: &Vec<&[f32]>,
        query: &ArraySlice,
        params: &SearchParams,
    ) -> Vec<(f32, usize)> {
        flat_knn::knn(
            records2visit,
            query,
            params.k,
            match self.distance_fn {
                DistanceFn::L2 => flat_knn::Metric::L2,
                DistanceFn::Dot => flat_knn::Metric::Dot,
            },
        )
    }

    pub fn verbose_search<S>(
        &self,
        query: &ArraySlice,
        params: S,
    ) -> DliResult<(Vec<Id>, SearchStatistics)>
    where
        S: SearchParamsT,
    {
        let params = params.into_search_params();
        let predictions = self.bucket_selection(query)?;
        let records2visit = self.records2visit(predictions, params.search_strategy);
        let rs = self.merge_results(&records2visit.records, query, &params);
        let res = rs
            .into_iter()
            .map(|(_, idx)| records2visit.ids[idx])
            .collect();
        Ok((
            res,
            SearchStatistics {
                total_visited_buckets: records2visit.total_visited_buckets,
                total_visited_records: records2visit.ids.len(),
            },
        ))
    }

    #[log_time]
    pub fn verbose_delete(&mut self, id: Id) -> DliResult<Option<((Array, Id), DeleteStatistics)>> {
        if let Some(deleted) = self.buffer.delete(&id) {
            return Ok(Some((
                deleted,
                DeleteStatistics {
                    affected_level: None,
                },
            )));
        }
        for (level_idx, level) in &mut self.levels.iter_mut().enumerate() {
            if let Some(deleted) = level.delete(&id, &self.delete_method)? {
                if self.is_level_underutilized(level_idx) {
                    self.compaction_strategy.clone().rebuild(self, level_idx)?;
                }
                return Ok(Some((
                    deleted,
                    DeleteStatistics {
                        affected_level: Some(level_idx),
                    },
                )));
            }
        }
        Ok(None)
    }

    fn is_level_underutilized(&self, level_idx: usize) -> bool {
        let level = &self.levels[level_idx];
        level.occupied() < level.bucket_size() * self.arity.pow(level_idx as u32)
    }

    pub fn dump(&self, working_dir: &Path) -> DliResult<()> {
        let working_dir = absolute(working_dir)?;
        create_dir(&working_dir)?;
        let disk_levels = self
            .levels
            .iter()
            .enumerate()
            .map(|(level_id, level)| level.dump(&working_dir, level_id))
            .collect::<DliResult<Vec<_>>>()?;
        let disk_buffer = self.buffer.dump(&working_dir);
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
        let meta_file = File::create(meta_path)?;
        serde_json::to_writer(meta_file, &disk_index)?;
        Ok(())
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

    #[log_time]
    pub fn compact(&self, index: &mut Index) -> DliResult<()> {
        let original_occupied = index.occupied();
        match self {
            CompactionStrategy::BentleySaxe(_) => {
                match self.available_level(index) {
                    Some(level_idx) => {
                        let (data, ids) = self.lower_level_data(index, level_idx);
                        let level = &mut index.levels[level_idx];
                        if level.size() == 0 {
                            info!("index:retrain");
                            level.retrain(&data)?;
                        }
                        info!(
                            level_idx = level_idx,
                            data_size = ids.len();
                            "index:compact",
                        );
                        level.insert_many(data, ids)?;
                    }
                    None => {
                        info!("index:new_level");
                        let level_idx = index.add_level()?;
                        let (data, ids) = self.lower_level_data(index, level_idx);
                        let level = &mut index.levels[level_idx];
                        level.train(&data)?;
                        info!(
                            level_idx = level_idx,
                            data_size = ids.len();
                            "index:compact",
                        );
                        level.insert_many(data, ids)?;
                    }
                };
            }
        }
        assert_eq!(original_occupied, index.occupied());
        Ok(())
    }

    #[log_time]
    pub fn rebuild(&self, index: &mut Index, level_idx: usize) -> DliResult<()> {
        assert!(level_idx < index.levels.len());

        match self {
            CompactionStrategy::BentleySaxe(RebuildStrategy::NoRebuild) => {}
            CompactionStrategy::BentleySaxe(RebuildStrategy::BasicRebuild) => {
                let level_occupied = index.levels[level_idx].occupied();
                info!(level_idx = level_idx, occupied = level_occupied; "index:rebuild");
                match Self::find_source_target_levels(index, level_idx, level_occupied) {
                    Some((from_level_idx, to_level_idx)) => {
                        move_data(index, &[from_level_idx], to_level_idx);
                    }
                    None => {
                        flush_buffer(index, level_idx, level_occupied)?;
                    }
                }
            }
            
        }
        Ok(())
    }

    fn find_source_target_levels(
        index: &mut Index,
        level_idx: usize,
        level_occupied: usize,
    ) -> Option<(usize, usize)> {
        // First level
        if level_idx == 0 {
            if let Some(lower_level_idx) = lower_level(index, level_idx, level_occupied) {
                assert!(lower_level_idx > level_idx);
                return Some((level_idx, lower_level_idx));
            };
            return None;
        }
        // Middle level
        if level_idx < index.levels.len() - 1 {
            if let Some(lower_level_idx) = lower_level(index, level_idx, level_occupied) {
                assert!(lower_level_idx > level_idx);
                return Some((level_idx, lower_level_idx));
            };
        }
        // Last level or no lower level found for middle level
        // Try to move data to the upper level
        let upper_level_idx = level_idx - 1;
        if index.levels[upper_level_idx].free_space() >= level_occupied {
            return Some((level_idx, upper_level_idx));
        }
        // Top up current level from upper level
        Some((level_idx - 1, level_idx))
    }
}

fn flush_buffer(index: &mut Index, level_idx: usize, level_occupied: usize) -> DliResult<()> {
    let buffer_occupied = index.buffer.occupied();
    let (data, ids) = index.buffer.get_data();
    index.levels[level_idx].insert_many(data, ids)?;
    assert!(index.buffer.occupied() == 0);
    assert!(index.levels[level_idx].occupied() == level_occupied + buffer_occupied);
    Ok(())
}

fn move_data(index: &mut Index, from_level_idxs: &[usize], to_level_idx: usize) {
    assert!(from_level_idxs
        .iter()
        .all(|&idx| idx < index.levels.len() && idx != to_level_idx));
    let from_levels_occupied = from_level_idxs
        .iter()
        .map(|&idx| index.levels[idx].occupied())
        .sum::<usize>();
    assert!(from_levels_occupied <= index.levels[to_level_idx].free_space());
    let to_level_occupied = index.levels[to_level_idx].occupied();
    for idx in from_level_idxs {
        let (data, ids) = index.levels[*idx].get_data();
        index.levels[to_level_idx]
            .insert_many(data, ids)
            .expect("insert_many failed inside rebuild move_data closure");
    }
    assert!(from_level_idxs
        .iter()
        .all(|&idx| index.levels[idx].occupied() == 0));
    assert!(index.levels[to_level_idx].occupied() == from_levels_occupied + to_level_occupied);
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
    pub fn from_yaml(file: &Path) -> DliResult<Self> {
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

    pub fn from_disk(working_dir: &Path) -> DliResult<Self> {
        let meta_path = working_dir.join("meta.json");
        println!("Loading index metadata from {:?}", meta_path);
        let meta_file = File::open(meta_path)?;
        println!("Deserializing index metadata...");
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

    pub fn retrain_strategy(mut self, strategy: RetrainStrategy) -> Self {
        self.levels_config.model.train_params.retrain_strategy = strategy;
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
    ) -> DliResult<LevelIndex> {
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

    pub fn build(self) -> DliResult<Index> {
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
                .map(|level| Self::load_disk_level(level, device, distance_fn.clone(), input_shape))
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
    use crate::bucket::{self, Bucket};
    use crate::constants::{DEFAULT_BUCKET_SIZE, DEFAULT_SEARCH_N_CANDIDATES};
    use crate::errors::DliResult;
    use crate::level_index::LevelIndex;
    use crate::structs::{DistanceFn, SearchStrategy};
    use crate::ModelConfig;
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

    /// Helper function to create a Level with predefined buckets containing records.
    /// Each bucket contains a list of (records, ids) tuples.
    #[allow(dead_code)]
    fn create_level_with_records_per_bucket(
        input_shape: usize,
        bucket_size: usize,
        buckets_data: Vec<(Array, Vec<Id>)>,
    ) -> DliResult<LevelIndex> {
        let mut buckets: Vec<Bucket> = buckets_data
            .iter()
            .map(|(records, ids)| {
                let mut bucket = bucket::BucketBuilder::default()
                    .input_shape(input_shape)
                    .size(bucket_size)
                    .build()
                    .expect("Failed to create bucket");
                // Populate the bucket with records
                for (rec, id) in records.chunks_exact(input_shape).zip(ids.iter()) {
                    bucket.insert(rec.to_vec(), *id);
                }
                bucket
            })
            .collect();

        // If no buckets provided, create a single empty one for model training purposes
        if buckets.is_empty() {
            buckets.push(
                bucket::BucketBuilder::default()
                    .input_shape(input_shape)
                    .size(bucket_size)
                    .build()
                    .expect("Failed to create empty bucket"),
            );
        }

        let _n_buckets = buckets.len();
        let level = LevelIndexBuilder::default()
            .input_shape(input_shape)
            .bucket_size(bucket_size)
            .buckets_in_memory(buckets)
            .model(ModelConfig::default())
            .model_device(ModelDevice::Cpu)
            .distance_fn(DistanceFn::Dot)
            .build()?;

        Ok(level)
    }

    #[test]
    fn test_level_index_config_default() {
        let config = LevelIndexConfig::default();
        assert_eq!(config.bucket_size, DEFAULT_BUCKET_SIZE);
    }

    #[test]
    fn test_index_config_from_yaml_valid() -> DliResult<()> {
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
              retrain_strategy: no_retrain
          bucket_size: 100
        buffer_size: 50
        input_shape: 10
        arity: 2
        device: cpu
        distance_fn: dot
        delete_method: oid_to_bucket
        "#;

        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "{yaml_content}")?;

        let index = IndexBuilder::from_yaml(temp_file.path())?.build()?;
        assert_eq!(index.buffer.size, 50);
        assert_eq!(index.input_shape, 10);
        assert_eq!(index.arity, 2);
        assert_eq!(index.levels.len(), 0);
        let lvl = index.levels_config;
        assert_eq!(lvl.bucket_size, 100);
        assert!(matches!(index.delete_method, DeleteMethod::OidToBucket));
        Ok(())
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
    fn test_index_insert_and_size() -> DliResult<()> {
        let config = create_test_config();
        let mut index = IndexBuilder::from_config(config).build()?;

        // Initial size is buffer size (10 from test config)
        assert_eq!(index.size(), 10);

        index.insert(vec![1.0, 2.0, 3.0], 1)?;
        // Size method returns total capacity, not occupied count
        assert_eq!(index.size(), 10);

        index.insert(vec![4.0, 5.0, 6.0], 2)?;
        assert_eq!(index.size(), 10);
        Ok(())
    }

    #[test]
    fn test_index_search_basic() -> DliResult<()> {
        let config = create_test_config();
        let mut index = IndexBuilder::from_config(config).build()?;

        // Insert some test data
        index.insert(vec![1.0, 2.0, 3.0], 1)?;
        index.insert(vec![4.0, 5.0, 6.0], 2)?;

        // We can't actually search with no levels because it would trigger empty predictions
        // But we can verify that data was inserted
        assert_eq!(index.size(), 10); // Buffer size
        Ok(())
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

        // Verify index is initially empty
        assert_eq!(index.occupied(), 0);
        assert_eq!(index.n_levels(), 0);
    }

    #[test]
    fn test_index_save_and_load() -> DliResult<()> {
        use tempfile::TempDir;

        // Build an index with specific configuration
        let input_shape = 10;
        let mut index = IndexBuilder::default()
            .arity(2)
            .bucket_size(10)
            .buffer_size(10)
            .input_shape(input_shape)
            .distance_fn(DistanceFn::Dot)
            .build()?;

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
            index.insert(record, i as u32)?;
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
        let original_results = test_queries
            .iter()
            .map(|query| index.search(query.as_slice(), 10))
            .collect::<DliResult<Vec<_>>>()?;

        // Also get verbose search statistics
        let original_stats = test_queries
            .iter()
            .map(|query| index.verbose_search(query.as_slice(), 10))
            .collect::<DliResult<Vec<_>>>()?;

        // Save index to temporary directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let dump_dir = temp_dir.path().join("index_dump");
        index.dump(&dump_dir)?;

        // Verify meta file was created
        let meta_path = dump_dir.join("meta.json");
        assert!(meta_path.exists(), "Meta file should exist");

        // Load index from disk
        let loaded_index = IndexBuilder::from_disk(&dump_dir)?.build()?;

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
        let loaded_results = test_queries
            .iter()
            .map(|query| loaded_index.search(query.as_slice(), 10))
            .collect::<DliResult<Vec<_>>>()?;

        let loaded_stats = test_queries
            .iter()
            .map(|query| loaded_index.verbose_search(query.as_slice(), 10))
            .collect::<DliResult<Vec<_>>>()?;

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
        Ok(())
    }

    #[test]
    fn test_bentley_saxe_index_available_level() -> DliResult<()> {
        let config = create_test_config();
        let index = IndexBuilder::from_config(config).build()?;
        // Initially no levels, so should return None
        assert_eq!(index.compaction_strategy.available_level(&index), None);
        Ok(())
    }

    #[test]
    fn test_bentley_saxe_index_get_level_config() -> DliResult<()> {
        let config = create_test_config();
        let index = IndexBuilder::from_config(config).build()?;

        let level_config = index.get_level_index_config();
        assert_eq!(level_config.bucket_size, DEFAULT_BUCKET_SIZE); // default value
        Ok(())
    }

    #[test]
    fn test_bentley_saxe_index_insert_to_buffer() -> DliResult<()> {
        let config = create_test_config();
        let mut index = IndexBuilder::from_config(config).build()?;

        // Initial size should be buffer size
        let initial_size = index.size();
        assert_eq!(initial_size, 10);

        // Insert data that should go to buffer
        for i in 1..=5 {
            index.insert(vec![i as f32, i as f32 + 1.0, i as f32 + 2.0], i as u32)?;
        }

        // Size should remain the same (capacity, not occupancy)
        assert_eq!(index.size(), 10);
        Ok(())
    }

    #[test]
    fn test_bentley_saxe_empty_levels() -> DliResult<()> {
        let config = create_test_config();
        let index = IndexBuilder::from_config(config).build()?;

        // Should have no levels initially
        assert_eq!(index.levels.len(), 0);
        // Should have a buffer
        assert_eq!(index.buffer.size, 10);
        Ok(())
    }

    #[test]
    fn test_bentley_saxe_lower_level_data_empty() -> DliResult<()> {
        let config = create_test_config();
        let mut index = IndexBuilder::from_config(config).build()?;

        let (data, ids) = index
            .compaction_strategy
            .clone()
            .lower_level_data(&mut index, 0);
        // Should only have buffer data initially
        assert!(data.is_empty());
        assert!(ids.is_empty());
        Ok(())
    }

    #[test]
    fn test_records2visit() -> DliResult<()> {
        // Create an index with one level with predefined bucket structure
        let input_shape = 3;
        let bucket_size = 50;

        // Bucket 0: 4 records (IDs 1, 2, 3, 4)
        let bucket_0_records = [
            1.0, 2.0, 3.0, // ID 1
            4.0, 5.0, 6.0, // ID 2
            7.0, 8.0, 9.0, // ID 3
            10.0, 11.0, 12.0, // ID 4
        ];
        let bucket_0_ids = [1u32, 2u32, 3u32, 4u32];

        // Bucket 1: 2 records (IDs 5, 6)
        let bucket_1_records = [
            13.0, 14.0, 15.0, // ID 5
            16.0, 17.0, 18.0, // ID 6
        ];
        let bucket_1_ids = [5u32, 6u32];

        // Create buckets in memory
        let mut bucket_0 = bucket::BucketBuilder::default()
            .input_shape(input_shape)
            .size(bucket_size)
            .build()?;
        for (rec, id) in bucket_0_records
            .chunks_exact(input_shape)
            .zip(bucket_0_ids.iter())
        {
            bucket_0.insert(rec.to_vec(), *id);
        }

        let mut bucket_1 = bucket::BucketBuilder::default()
            .input_shape(input_shape)
            .size(bucket_size)
            .build()?;
        for (rec, id) in bucket_1_records
            .chunks_exact(input_shape)
            .zip(bucket_1_ids.iter())
        {
            bucket_1.insert(rec.to_vec(), *id);
        }

        // Build the index with one level containing these predefined buckets
        let mut index = IndexBuilder::default()
            .input_shape(input_shape)
            .buffer_size(5)
            .distance_fn(DistanceFn::Dot)
            .build()?;

        // Manually create and add a level with in-memory buckets
        let level = LevelIndexBuilder::default()
            .input_shape(input_shape)
            .bucket_size(bucket_size)
            .buckets_in_memory(vec![bucket_0, bucket_1])
            .model(ModelConfig::default())
            .model_device(ModelDevice::Cpu)
            .distance_fn(DistanceFn::Dot)
            .build()?;

        index.levels.push(level);

        // Add some records to buffer for testing
        index.insert(vec![19.0, 20.0, 21.0], 7)?;
        index.insert(vec![22.0, 23.0, 24.0], 8)?;

        // Verify the structure we created
        assert_eq!(index.n_levels(), 1);
        assert_eq!(index.levels[0].n_buckets(), 2);
        assert_eq!(index.levels[0].bucket(0).occupied(), 4); // bucket 0 has 4 records
        assert_eq!(index.levels[0].bucket(1).occupied(), 2); // bucket 1 has 2 records
        assert_eq!(index.buffer.occupied(), 2); // buffer has 2 records

        // Create mock predictions with controlled probabilities
        // Bucket 0 has higher probability (0.8) than bucket 1 (0.2)
        let predictions = vec![vec![
            (0, 0.8, 4), // bucket 0, high prob, 4 records
            (1, 0.2, 2), // bucket 1, low prob, 2 records
        ]];

        // Test 1: with ncandidates = 5
        // Bucket normalization: arity=2, level_idx=0
        // bucket 0: 0.8 * 2^0 = 0.8
        // bucket 1: 0.2 * 2^0 = 0.2
        // buffer: 1.0
        // Sorted: buffer(1.0), bucket 0(0.8), bucket 1(0.2)
        // Collection: buffer(2) + bucket 0(4) = 6 total (exceeds ncandidates=5 but whole buckets collected)
        let ncandidates = 5;
        let search_strategy = SearchStrategy::ModelDriven(ncandidates);
        let result = index.records2visit(predictions.clone(), search_strategy);

        // 1. records and ids should have same length
        assert_eq!(result.records.len(), result.ids.len());

        // 2. Should collect whole buckets, may exceed ncandidates
        // Buffer (2 records) + bucket 0 (4 records) = 6 total
        assert_eq!(result.records.len(), 6);

        // 3. All collected IDs should be from buffer and bucket 0
        // Buffer IDs: 7, 8; Bucket 0 IDs: 1, 2, 3, 4
        for id in &result.ids {
            assert!(
                (*id >= 1 && *id <= 4) || (*id == 7 || *id == 8),
                "Invalid id: {id}"
            );
        }

        // 4. total_visited_buckets should count all buckets + buffer
        // In this case: 2 data buckets + 1 buffer = 3
        assert_eq!(result.total_visited_buckets, 3);

        // Test 2: with larger ncandidates to verify all records collected
        let ncandidates_large = 20;
        let search_strategy_large = SearchStrategy::ModelDriven(ncandidates_large);
        let result_large = index.records2visit(predictions, search_strategy_large);

        // Should collect all available records: 4 + 2 + 2 = 8
        assert_eq!(result_large.records.len(), 8);
        assert_eq!(result_large.ids.len(), 8);

        // All IDs from 1-8 should be present
        for id in &result_large.ids {
            assert!(*id >= 1 && *id <= 8, "Invalid id: {id}");
        }

        Ok(())
    }

    #[test]
    fn test_search_in_buffer() -> DliResult<()> {
        let mut config = create_test_config();
        config.distance_fn = DistanceFn::L2;
        let mut index = IndexBuilder::from_config(config).build()?;

        // Insert items into buffer (buffer size is 10 in create_test_config)
        let id1 = 1;
        let vec1 = vec![1.0, 2.0, 3.0];
        index.insert(vec1.clone(), id1)?;

        let id2 = 2;
        let vec2 = vec![4.0, 5.0, 6.0];
        index.insert(vec2.clone(), id2)?;

        // Search for the first item
        // We use exact match search (k=1)
        let results = index.search(&vec1, 1)?;

        assert!(!results.is_empty(), "Search should return results");
        assert_eq!(results[0], id1, "Should find the inserted ID in buffer");

        // Search for the second item
        let results = index.search(&vec2, 1)?;
        assert!(!results.is_empty());
        assert_eq!(results[0], id2);

        Ok(())
    }

    #[test]
    fn test_delete_from_buffer() -> DliResult<()> {
        let mut config = create_test_config();
        config.distance_fn = DistanceFn::L2;
        let mut index = IndexBuilder::from_config(config).build()?;

        // Insert items into buffer
        let id1 = 1;
        let vec1 = vec![1.0, 2.0, 3.0];
        index.insert(vec1.clone(), id1)?;

        let id2 = 2;
        let vec2 = vec![4.0, 5.0, 6.0];
        index.insert(vec2.clone(), id2)?;

        assert_eq!(index.occupied(), 2, "Should have 2 items initially");

        // Delete the first item
        let deleted = index.delete(id1)?;
        assert!(deleted.is_some(), "Should return deleted item");
        let (deleted_vec, deleted_id) = deleted.unwrap();
        assert_eq!(deleted_id, id1);
        assert_eq!(deleted_vec, vec1);

        // Verify occupancy decreased
        assert_eq!(index.occupied(), 1, "Should have 1 item after delete");

        // Verify item is gone from search
        // Searching for vec1 should now return id2 as it is the only remaining item
        let results = index.search(&vec1, 1)?;
        assert!(!results.is_empty());
        assert_ne!(results[0], id1, "Deleted ID should not be found");
        assert_eq!(results[0], id2, "Should find the other item");

        // Verify the other item is still there and findable
        let results2 = index.search(&vec2, 1)?;
        assert!(!results2.is_empty());
        assert_eq!(results2[0], id2);

        Ok(())
    }

    #[test]
    fn test_compaction_trigger() -> DliResult<()> {
        let mut config = create_test_config();
        config.buffer_size = 10;
        let mut index = IndexBuilder::from_config(config).build()?;

        // Insert 10 items (filling the buffer)
        for i in 0..10 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        // User expectation: Buffer should have flushed when it became full (at 10 items).
        // So we expect 1 level and empty buffer.
        assert_eq!(
            index.n_levels(),
            1,
            "Should have flushed to level after 10 inserts"
        );
        assert_eq!(
            index.buffer.occupied(),
            0,
            "Buffer should be empty after flush"
        );

        Ok(())
    }

    #[test]
    fn test_delete_from_level() -> DliResult<()> {
        let mut config = create_test_config();
        config.distance_fn = DistanceFn::L2;
        config.buffer_size = 10;
        let mut index = IndexBuilder::from_config(config).build()?;

        // Insert 15 items.
        // First 10 items (0..10) will fill the buffer.
        // The 11th item (10) will trigger compaction, moving 0..10 to level 0.
        // Items 10..15 will end up in the buffer.
        for i in 0..15 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        // Verify structure: Should have levels now
        assert!(
            index.n_levels() > 0,
            "Should have at least one level after compaction"
        );
        assert_eq!(index.occupied(), 15, "Total occupancy should be 15");

        // Pick an ID that should be in the level (ID 5 is in the first batch of 10)
        let id_to_delete = 5;
        let vec_to_delete = vec![5.0, 5.0, 5.0];

        // Verify it exists before deletion
        let results = index.search(&vec_to_delete, 1)?;
        assert!(!results.is_empty());
        assert_eq!(results[0], id_to_delete);

        // Delete it
        let deleted = index.delete(id_to_delete)?;
        assert!(deleted.is_some(), "Should return deleted item");
        let (deleted_vec, deleted_id) = deleted.unwrap();
        assert_eq!(deleted_id, id_to_delete);
        assert_eq!(deleted_vec, vec_to_delete);

        // Verify occupancy decreased
        assert_eq!(index.occupied(), 14, "Should have 14 items after delete");

        // Verify it is gone
        // Searching for the deleted vector should return the nearest neighbor (e.g. 4 or 6), but not 5
        let results = index.search(&vec_to_delete, 1)?;
        if !results.is_empty() {
            assert_ne!(results[0], id_to_delete, "Deleted ID should not be found");
        }

        Ok(())
    }

    #[test]
    fn test_search_empty_index() -> DliResult<()> {
        let config = create_test_config();
        let index = IndexBuilder::from_config(config).build()?;

        // Search for any vector
        let query = vec![1.0, 2.0, 3.0];
        let results = index.search(&query, 5)?;

        assert!(
            results.is_empty(),
            "Search on empty index should return empty results"
        );

        Ok(())
    }
}
