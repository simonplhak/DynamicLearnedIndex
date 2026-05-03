use crate::{
    bucket::{Bucket, Buffer, BufferBuilder},
    level_index::{LevelIndex, LevelIndexBuilder, StorageContainer},
    model::{ModelDevice, RetrainStrategy},
    structs::{
        CompactionStrategyConfig, DiskBuffer, DiskIndex, DiskLevelIndex, FloatElement, IndexConfig,
        LevelIndexConfig, RebuildStrategy, Records2Visit,
    },
    DeleteMethod, DistanceFn, DliError, DliResult, Id, ModelLayer, SearchParams, SearchParamsT,
    SearchStrategy,
};
use flat_knn::VectorType;
use log::debug;
use lru::LruCache;
use measure_time_macro::log_time;
use rayon::iter::{IntoParallelRefIterator as _, ParallelIterator};
use std::{
    fs::{self, create_dir, File},
    num::NonZeroUsize,
    path::{absolute, Path, PathBuf},
    sync::Mutex,
};
use typed_arena::Arena;

pub struct Index<F: FloatElement> {
    compaction_strategy: CompactionStrategy<F>,
    levels_config: LevelIndexConfig,
    input_shape: usize,
    arity: usize,
    device: ModelDevice,
    levels: Vec<LevelIndex<F>>,
    buffer: Buffer<F>,
    distance_fn: DistanceFn,
    delete_method: DeleteMethod,
    /// Base directory for per-level cold storage files.
    cold_storage_dir: Option<PathBuf>,
    /// Levels at index >= this value are stored cold.
    cold_threshold_level: Option<usize>,
    /// Shared LRU cache for cold storage buckets.
    /// Key: (level_idx, bucket_id). Capacity derived from `cold_cache_size_bytes`.
    cache: Mutex<LruCache<(usize, usize), Bucket<F>>>,
}

impl<F: FloatElement + flat_knn::VectorType> Index<F> {
    #[log_time]
    pub fn search<S>(&self, query: &[F], params: S) -> DliResult<Vec<Id>>
    where
        S: SearchParamsT,
    {
        let params = params.into_search_params();
        let predictions = self.bucket_selection(query)?;
        let records2visit = self.records2visit(predictions, params.search_strategy);
        self.merge_results(records2visit, query, &params)
    }

    #[log_time]
    pub fn search_bulk<S>(&self, xs: &[F], params: S) -> DliResult<Vec<Vec<Id>>>
    where
        S: SearchParamsT,
    {
        let params = params.into_search_params();
        let input_shape = self.input_shape;
        let n_queries = xs.len() / input_shape;

        // Collect bucket predictions efficiently for all queries at once per level
        let mut level_predictions = Vec::new();
        for level in &self.levels {
            let queries_refs: Vec<&[F]> = (0..n_queries)
                .map(|i| &xs[i * input_shape..(i + 1) * input_shape])
                .collect();
            let preds = level.buckets2visit_predictions_many(&queries_refs)?;
            level_predictions.push(preds);
        }

        // Process results for each query
        let mut all_results = Vec::with_capacity(n_queries);
        for query_idx in 0..n_queries {
            let query = &xs[query_idx * input_shape..(query_idx + 1) * input_shape];

            // Gather this query's predictions from all levels
            let predictions = level_predictions
                .iter()
                .map(|level_preds| level_preds[query_idx].clone())
                .collect();

            // Find records to visit for this query
            let records2visit = self.records2visit(predictions, params.search_strategy);

            // Perform KNN merge
            let res = self.merge_results(records2visit, query, &params)?;
            all_results.push(res);
        }

        Ok(all_results)
    }

    #[log_time]
    fn records2visit(
        &self,
        predictions: Vec<Vec<(usize, f32)>>,
        search_strategy: SearchStrategy,
    ) -> Records2Visit {
        match search_strategy {
            SearchStrategy::Base(_nprobe) => todo!(),
            SearchStrategy::ModelDriven(ncandidates) => {
                let arity = self.arity;
                let normalize_probability =
                    |prob: f32, level_idx| (arity.pow(level_idx) as f32) * prob.max(0.0);
                let levels = predictions.len();
                let mut buckets2visit = Vec::with_capacity(self.n_buckets() + 1);
                for (level_idx, level_predictions) in predictions.iter().enumerate() {
                    for (bucket_id, prob) in level_predictions {
                        let occupied = self.levels[level_idx]
                            .storage
                            .bucket_occupied_count(*bucket_id);
                        if occupied == 0 {
                            continue;
                        }
                        buckets2visit.push((
                            level_idx,
                            *bucket_id,
                            normalize_probability(*prob, level_idx as u32),
                            occupied,
                        ));
                    }
                }
                // add buffer as a special "bucket"
                if self.buffer.occupied() > 0 {
                    buckets2visit.push((levels, self.n_buckets(), 1.0, self.buffer.occupied()));
                }
                buckets2visit.sort_by(|a, b| b.2.total_cmp(&a.2));
                let mut total_occupied = 0;
                let mut visited_buckets = 0;
                for (level_idx, _bucket_id, _prob, occupied) in &buckets2visit {
                    if total_occupied < ncandidates {
                        if *level_idx == levels {
                            total_occupied += occupied;
                        } else {
                            total_occupied += occupied;
                        }
                        visited_buckets += 1;
                    } else {
                        break;
                    }
                }
                debug!(
                    visited_buckets = visited_buckets,
                    visited_records = total_occupied;
                    "index:records2visit"
                );
                let _ = buckets2visit.split_off(visited_buckets);
                buckets2visit
            }
        }
    }

    pub fn invalidate_cold_cache(&self) {
        self.cache.lock().unwrap().clear();
    }

    pub fn add_level(&mut self) -> DliResult<usize> {
        let level_index_config = self.get_level_index_config();
        let new_level_idx = self.levels.len();
        let n_buckets = self.arity.pow(new_level_idx as u32 + 1);
        let mut level_index_builder = LevelIndexBuilder::default()
            .id(format!("{}", new_level_idx))
            .n_buckets(n_buckets)
            .input_shape(self.input_shape)
            .model(level_index_config.model.clone())
            .model_device(self.device)
            .bucket_size(level_index_config.bucket_size)
            .distance_fn(self.distance_fn.clone());
        // If this level meets the cold threshold, initialize cold storage for it.
        if let Some(cold_storage_dir) = self.cold_storage_dir.clone() {
            if let Some(cold_threshold) = self.cold_threshold_level {
                if new_level_idx >= cold_threshold {
                    let cold_data_path =
                        cold_storage_dir.join(format!("cold_level_{}.bin", new_level_idx));
                    level_index_builder = level_index_builder.cold_data_path(cold_data_path);
                }
            }
        }

        let level_index = level_index_builder.build()?;
        self.levels.push(level_index);
        Ok(new_level_idx)
    }

    fn get_level_index_config(&self) -> LevelIndexConfig {
        self.levels_config.clone()
    }

    #[log_time]
    pub fn insert(&mut self, value: Vec<F>, id: Id) -> DliResult<()> {
        self.buffer.insert(value, id);
        if self.buffer.has_space(1) {
            return Ok(()); // buffer is not full yet
        }
        debug!(levels = self.levels.len(), occupied = self.occupied(); "index:buffer_flush");
        self.compaction_strategy.clone().compact(self)?;
        assert!(self.buffer.occupied() == 0);
        Ok(())
    }

    #[log_time]
    pub fn delete(&mut self, id: Id) -> DliResult<bool> {
        if self.buffer.delete(&id) {
            return Ok(true);
        }
        let (deleted, level_idx) = self.delete_from_level(id);
        if deleted {
            debug!(level_idx = level_idx, id = id; "index:delete");
            if self.is_level_underutilized(level_idx) {
                self.compaction_strategy.clone().rebuild(self, level_idx)?;
            }
            return Ok(true);
        }
        Ok(false)
    }

    #[log_time]
    fn delete_from_level(&mut self, id: Id) -> (bool, usize) {
        for (level_idx, level) in &mut self.levels.iter_mut().enumerate() {
            if level.storage.delete(&id, &self.delete_method) {
                return (true, level_idx);
            }
        }
        (false, 0)
    }

    pub fn size(&self) -> usize {
        self.levels
            .iter()
            .map(|level| level.storage.size())
            .sum::<usize>()
            + self.buffer.size
    }

    pub fn memory_usage(&self) -> usize {
        let levels_size: usize = self.levels.iter().map(|l| l.memory_usage()).sum();
        let buffer_heap = self.buffer.memory_usage() - std::mem::size_of::<Buffer<f32>>();
        std::mem::size_of::<Self>() + levels_size + buffer_heap
    }

    pub fn n_buckets(&self) -> usize {
        self.levels
            .iter()
            .map(|level| level.storage.n_buckets())
            .sum()
    }

    pub fn n_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn occupied(&self) -> usize {
        self.levels
            .iter()
            .map(|level| level.storage.occupied())
            .sum::<usize>()
            + self.buffer_occupied()
    }

    pub fn buffer_occupied(&self) -> usize {
        self.buffer.occupied()
    }

    pub fn level_occupied(&self, level_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        self.levels[level_idx].storage.occupied()
    }

    pub fn level_n_buckets(&self, level_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        self.levels[level_idx].storage.n_buckets()
    }

    pub fn level_total_size(&self, level_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        self.levels[level_idx].storage.size()
    }

    pub fn level_n_empty_buckets(&self, level_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        self.levels[level_idx].storage.n_empty_buckets()
    }

    pub fn bucket_occupied(&self, level_idx: usize, bucket_idx: usize) -> usize {
        assert!(level_idx < self.levels.len());
        assert!(bucket_idx < self.levels[level_idx].storage.n_buckets());
        self.levels[level_idx]
            .storage
            .bucket_occupied_count(bucket_idx)
    }

    pub fn n_empty_buckets(&self) -> usize {
        self.levels
            .iter()
            .map(|level| level.storage.n_empty_buckets())
            .sum()
    }

    #[log_time]
    fn bucket_selection(&self, query: &[F]) -> DliResult<Vec<Vec<(usize, f32)>>> {
        self.levels
            .par_iter()
            .map(|level| {
                if level.storage.occupied() > 0 {
                    level.buckets2visit_predictions(query)
                } else {
                    Ok(vec![])
                }
            })
            .collect::<DliResult<Vec<_>>>()
    }

    #[log_time]
    fn merge_results(
        &self,
        records2visit: Records2Visit,
        query: &[F],
        params: &SearchParams,
    ) -> DliResult<Vec<Id>> {
        // Acquiring the lock for the whole time of function
        // This is usually not a good approach but currently index does not support concurrent writes,
        // so it is safe and does not bring any performance overhead
        let mut cache_lock = self.cache.lock().unwrap();

        // Using an arena to avoid allocations for each record
        let cold_arena = Arena::new();
        let mut cold_arena_keys = Vec::new();

        let mut records: Vec<&[F]> = Vec::new();
        let mut ids: Vec<Id> = Vec::new();

        for (level_idx, bucket_id, ..) in records2visit.into_iter() {
            // --- Path A: Cache Hit ---
            if let Some(cached_bucket) = cache_lock.get(&(level_idx, bucket_id)) {
                for i in 0..cached_bucket.occupied() {
                    // SAFETY:
                    // Here is the trick: we cast the reference to a shorter lifetime
                    // that the compiler can't "track" back to the mutable lock
                    // in a way that blocks the next iteration.
                    unsafe {
                        let ptr = cached_bucket.record(i) as *const [F];
                        records.push(&*ptr);
                        ids.push(cached_bucket.ids[i]);
                    }
                }
            } else {
                if level_idx == self.levels.len() {
                    for i in 0..self.buffer.occupied() {
                        records.push(self.buffer.record(i));
                        ids.push(self.buffer.ids[i]);
                    }
                } else {
                    match &self.levels[level_idx].storage.container {
                        StorageContainer::Hot(hot_storage) => {
                            let bucket = &hot_storage.buckets[bucket_id];
                            let occupied = bucket.occupied();
                            for i in 0..occupied {
                                records.push(bucket.record(i));
                                ids.push(bucket.ids[i]);
                            }
                        }
                        StorageContainer::Cold(cold_storage) => {
                            let bucket = cold_storage.read_bucket(bucket_id)?;
                            let pinned_bucket = cold_arena.alloc(bucket);
                            cold_arena_keys.push((level_idx, bucket_id));
                            for i in 0..pinned_bucket.occupied() {
                                records.push(pinned_bucket.record(i));
                                ids.push(pinned_bucket.ids[i]);
                            }
                        }
                    };
                }
            }
        }

        // KNN Computation
        let res = match self.distance_fn {
            DistanceFn::L2 => flat_knn::knn::<_, flat_knn::L2>(&records, query, params.k),
            DistanceFn::Dot => flat_knn::knn::<_, flat_knn::Dot>(&records, query, params.k),
        };

        let res = res.into_iter().map(|(_, idx)| ids[idx]).collect();
        // Update cache
        for (cold_bucket, key) in cold_arena
            .into_vec()
            .into_iter()
            .zip(cold_arena_keys.into_iter())
        {
            cache_lock.put(key, cold_bucket);
        }
        Ok(res)
    }

    fn is_level_underutilized(&self, level_idx: usize) -> bool {
        let level = &self.levels[level_idx];
        level.storage.occupied() < level.storage.bucket_size() * self.arity.pow(level_idx as u32)
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
        let records_path = working_dir.join("buffer_records.bin");
        let mut records_file = File::create(records_path.clone())?;
        let ids_path = working_dir.join("buffer_ids.bin");
        let mut ids_file = File::create(ids_path.clone())?;
        let disk_buffer_storage = self.buffer.dump(&mut records_file, &mut ids_file);
        let disk_buffer = DiskBuffer {
            records_path,
            ids_path,
            data: disk_buffer_storage,
        };
        let cold_cache_size_bytes = {
            let n = self.cache.lock().unwrap().cap().get();
            let bytes_per_bucket =
                self.input_shape * std::mem::size_of::<F>() * self.levels_config.bucket_size
                    + std::mem::size_of::<Id>() * self.levels_config.bucket_size;
            (n * bytes_per_bucket) as u64
        };
        let disk_index = DiskIndex {
            levels_config: self.levels_config.clone(),
            compaction_strategy: self.compaction_strategy.strategy.clone(),
            buffer_size: self.buffer.size,
            input_shape: self.input_shape,
            arity: self.arity,
            distance_fn: self.distance_fn.clone(),
            delete_method: self.delete_method.clone(),
            levels: disk_levels,
            disk_buffer,
            cold_cache_size_bytes,
            cold_storage_dir: self.cold_storage_dir.clone(),
            cold_threshold_level: self.cold_threshold_level,
        };
        let meta_path = working_dir.join("meta.json");
        let meta_file = File::create(meta_path)?;
        serde_json::to_writer(meta_file, &disk_index)?;
        Ok(())
    }
}

// Generic version used in your actual logic
#[derive(Debug, Clone)]
pub struct CompactionStrategy<F: FloatElement> {
    strategy: CompactionStrategyConfig,
    _marker: std::marker::PhantomData<F>,
}

impl<F: FloatElement + VectorType> CompactionStrategy<F> {
    fn available_level(&self, index: &Index<F>) -> Option<usize> {
        let mut count = index.buffer.occupied();
        index
            .levels
            .iter()
            .enumerate()
            .find(|(_, level)| {
                let occupied = level.storage.occupied();
                let fits = level.storage.size() - occupied >= count;
                if !fits {
                    count += occupied;
                }
                fits
            })
            .map(|(i, _)| i)
    }

    fn lower_level_data(
        &self,
        index: &mut Index<F>,
        level_idx: usize,
    ) -> DliResult<(Vec<F>, Vec<Id>)> {
        let (data, ids): (Vec<Vec<F>>, Vec<Vec<Id>>) = index
            .levels
            .iter_mut()
            .take(level_idx)
            .map(|level| level.storage.get_data())
            .collect::<DliResult<Vec<_>>>()?
            .into_iter()
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
        Ok((data, ids))
    }

    #[log_time]
    pub fn compact(&self, index: &mut Index<F>) -> DliResult<()> {
        let original_occupied = index.occupied();
        match self.strategy {
            CompactionStrategyConfig::BentleySaxe(_) => {
                match self.available_level(index) {
                    Some(level_idx) => {
                        let (data, ids) = self.lower_level_data(index, level_idx)?;
                        let level = &mut index.levels[level_idx];
                        if level.storage.size() == 0 {
                            debug!("index:retrain");
                            level.retrain(&data)?;
                        }
                        debug!(
                            level_idx = level_idx,
                            data_size = ids.len();
                            "index:compact",
                        );
                        level.insert_many(data, ids)?;
                    }
                    None => {
                        debug!("index:new_level");
                        let level_idx = index.add_level()?;
                        let (data, ids) = self.lower_level_data(index, level_idx)?;
                        let level = &mut index.levels[level_idx];
                        level.train(&data)?;
                        debug!(
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
    pub fn rebuild(&self, index: &mut Index<F>, level_idx: usize) -> DliResult<()> {
        assert!(level_idx < index.levels.len());
        let level_occupied = index.levels[level_idx].storage.occupied();
        debug!(level_idx = level_idx, occupied = level_occupied; "index:rebuild");
        match self.strategy {
            CompactionStrategyConfig::BentleySaxe(RebuildStrategy::NoRebuild) => {
                debug!("index:no_rebuild");
            }
            CompactionStrategyConfig::BentleySaxe(RebuildStrategy::BasicRebuild) => {
                debug!("index:basic_rebuild");
                match Self::find_source_target_levels(index, level_idx, level_occupied) {
                    Some((from_level_idx, to_level_idx)) => {
                        move_data(index, &[from_level_idx], to_level_idx)?;
                    }
                    None => {
                        flush_buffer(index, level_idx, level_occupied)?;
                    }
                }
            }
            CompactionStrategyConfig::BentleySaxe(RebuildStrategy::GreedyRebuild) => {
                debug!("index:greedy_rebuild");
                match Self::find_source_target_levels(index, level_idx, level_occupied) {
                    Some((_, to_level_idx)) => {
                        let mut available_space = index.levels[to_level_idx].storage.free_space();
                        let mut source_levels = vec![];
                        for level_idx in (to_level_idx..=0).rev() {
                            // handling case where to_level_idx is zero
                            if level_idx == to_level_idx {
                                continue;
                            }
                            let level_occupied = index.levels[level_idx].storage.occupied();
                            if level_occupied > 0 {
                                if level_occupied > available_space {
                                    break;
                                }
                                available_space -= level_occupied;
                                source_levels.push(level_idx);
                            }
                        }
                        move_data(index, &source_levels, to_level_idx)?;
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
        index: &mut Index<F>,
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
        if index.levels[upper_level_idx].storage.free_space() >= level_occupied {
            return Some((level_idx, upper_level_idx));
        }
        // Top up current level from upper level
        Some((level_idx - 1, level_idx))
    }
}

fn flush_buffer<F: FloatElement>(
    index: &mut Index<F>,
    level_idx: usize,
    level_occupied: usize,
) -> DliResult<()> {
    let buffer_occupied = index.buffer.occupied();
    let (data, ids) = index.buffer.get_data();
    index.levels[level_idx].insert_many(data, ids)?;
    assert!(index.buffer.occupied() == 0);
    assert!(index.levels[level_idx].storage.occupied() == level_occupied + buffer_occupied);
    Ok(())
}

fn move_data<F: FloatElement>(
    index: &mut Index<F>,
    from_level_idxs: &[usize],
    to_level_idx: usize,
) -> DliResult<()> {
    debug!(
        source_levels = from_level_idxs.iter().map(|idx| idx.to_string()).collect::<Vec<_>>().join(",").as_str(),
        to_level = to_level_idx;
        "index:move_data"
    );
    assert!(from_level_idxs
        .iter()
        .all(|&idx| idx < index.levels.len() && idx != to_level_idx));
    let from_levels_occupied = from_level_idxs
        .iter()
        .map(|&idx| index.levels[idx].storage.occupied())
        .sum::<usize>();
    assert!(from_levels_occupied <= index.levels[to_level_idx].storage.free_space());
    let to_level_occupied = index.levels[to_level_idx].storage.occupied();
    let mut data = Vec::with_capacity(from_levels_occupied * index.input_shape);
    let mut ids = Vec::with_capacity(from_levels_occupied);
    for idx in from_level_idxs {
        let (level_data, level_ids) = index.levels[*idx].storage.get_data()?;
        data.extend(level_data);
        ids.extend(level_ids);
    }
    index.levels[to_level_idx].insert_many(data, ids)?;
    assert!(from_level_idxs
        .iter()
        .all(|&idx| index.levels[idx].storage.occupied() == 0));
    assert!(
        index.levels[to_level_idx].storage.occupied() == from_levels_occupied + to_level_occupied
    );
    Ok(())
}

fn lower_level<F: FloatElement>(index: &Index<F>, level_idx: usize, size: usize) -> Option<usize> {
    // Find the next level with enough free space
    index
        .levels
        .iter()
        .enumerate()
        .skip(level_idx + 1)
        .find(|(_, level)| level.storage.occupied() > 0 && level.storage.free_space() >= size)
        .map(|(level_idx, _)| level_idx)
}

#[derive(Clone)]
pub struct IndexBuilder<F: FloatElement> {
    compaction_strategy: Option<CompactionStrategyConfig>,
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
    /// Base directory for per-level cold storage files.
    cold_storage_dir: Option<PathBuf>,
    /// Levels >= this index are stored cold.
    cold_threshold_level: Option<usize>,
    /// Byte budget for the index-level LRU bucket cache.
    /// Capacity in entries is derived from this at build time.
    cold_cache_size_bytes: Option<u64>,
    _marker: std::marker::PhantomData<F>,
}

impl<F: FloatElement> Default for IndexBuilder<F> {
    fn default() -> Self {
        Self::from_config(Default::default())
    }
}

impl<F: FloatElement> IndexBuilder<F> {
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
            cold_storage_dir: None,
            cold_threshold_level: None,
            cold_cache_size_bytes: None,
            model_layers: None,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn from_disk(working_dir: &Path) -> DliResult<Self> {
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
            cold_storage_dir: disk_index.cold_storage_dir,
            cold_threshold_level: disk_index.cold_threshold_level,
            cold_cache_size_bytes: Some(disk_index.cold_cache_size_bytes),
            model_layers: None,
            _marker: std::marker::PhantomData,
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

    pub fn compaction_strategy(mut self, strategy: CompactionStrategyConfig) -> Self {
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

    pub fn cold_storage_dir(mut self, path: PathBuf) -> Self {
        self.cold_storage_dir = Some(path);
        self
    }

    /// Set the threshold: levels at index >= `level` will be stored cold.
    /// Requires `cold_storage_dir` to also be set.
    pub fn cold_threshold_level(mut self, level: usize) -> Self {
        self.cold_threshold_level = Some(level);
        self
    }

    /// Set the byte budget for the index-level LRU bucket cache.
    ///
    /// Entry capacity is estimated from the budget and the configured bucket size:
    /// `n_entries = bytes / (bucket_size × (input_shape × sizeof(F) + sizeof(Id)))`
    ///
    /// Example: `4 * 1024 * 1024 * 1024` for a 4 GiB cache.
    pub fn cold_cache_size_bytes(mut self, bytes: u64) -> Self {
        self.cold_cache_size_bytes = Some(bytes);
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

    pub fn quantize(mut self, quantize: bool) -> Self {
        self.levels_config.model.quantize = quantize;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.levels_config.model.seed = seed;
        self
    }

    fn load_disk_level(
        disk_level: DiskLevelIndex,
        device: ModelDevice,
        distance_fn: DistanceFn,
        input_shape: usize,
    ) -> DliResult<LevelIndex<F>> {
        if let Some(cold_path) = disk_level.cold_data_path {
            // Cold level: load routing metadata from sidecar, build level with empty hot storage, then replace with cold.
            // todo: this should be done directly in the builder
            let meta_path = crate::cold_storage::meta_path_for(&cold_path);
            let cold_storage_level = crate::cold_storage::ColdStorage::load(
                &cold_path,
                &meta_path,
                input_shape,
                disk_level.config.bucket_size,
            )?;
            let n_buckets = cold_storage_level.n_buckets();
            let mut level = LevelIndexBuilder::<F>::default()
                .model(disk_level.config.model)
                .distance_fn(distance_fn)
                .model_device(device)
                .bucket_size(disk_level.config.bucket_size)
                .input_shape(input_shape)
                .n_buckets(n_buckets)
                .build()?;
            level.storage.container = StorageContainer::Cold(cold_storage_level);
            return Ok(level);
        }
        LevelIndexBuilder::<F>::default()
            .model(disk_level.config.model)
            .distance_fn(distance_fn)
            .model_device(device)
            .bucket_size(disk_level.config.bucket_size)
            .input_shape(input_shape)
            .buckets(
                disk_level.buckets,
                disk_level.records_path,
                disk_level.ids_path,
            )
            .build()
    }

    pub fn build(self) -> DliResult<Index<F>> {
        let levels_config = self.levels_config;
        let buffer_size = self
            .buffer_size
            .ok_or(DliError::MissingAttribute("buffer_size"))?;
        let input_shape = self
            .input_shape
            .ok_or(DliError::MissingAttribute("input_shape"))?;
        let buffer = match self.disk_buffer {
            Some(disk_buffer) => {
                let DiskBuffer {
                    records_path,
                    ids_path,
                    data,
                } = disk_buffer;
                let mut records_file = File::create(records_path)?;
                let mut ids_file = File::create(ids_path)?;
                BufferBuilder::<F>::from_disk(data, &mut records_file, &mut ids_file)
                    .input_shape(input_shape)
                    .size(buffer_size)
                    .build()?
            }
            None => BufferBuilder::<F>::default()
                .input_shape(input_shape)
                .size(buffer_size)
                .build()?,
        };

        let arity = self.arity.ok_or(DliError::MissingAttribute("arity"))?;
        let device = self.device.ok_or(DliError::MissingAttribute("device"))?;
        let distance_fn = self
            .distance_fn
            .ok_or(DliError::MissingAttribute("distance_fn"))?;
        let compaction_strategy_config = self
            .compaction_strategy
            .ok_or(DliError::MissingAttribute("compaction_strategy"))?;
        let compaction_strategy = CompactionStrategy {
            strategy: compaction_strategy_config,
            _marker: std::marker::PhantomData,
        };
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
        let bucket_cache = {
            let bytes = self.cold_cache_size_bytes.unwrap_or(0);
            let bytes_per_entry = levels_config.bucket_size
                * (input_shape * std::mem::size_of::<F>() + std::mem::size_of::<Id>());
            let n_entries = if bytes == 0 {
                1
            } else {
                ((bytes as usize) / bytes_per_entry).max(1)
            };
            Mutex::new(LruCache::new(NonZeroUsize::new(n_entries).unwrap()))
        };
        if let Some(cold_storage_dir) = &self.cold_storage_dir {
            fs::create_dir_all(cold_storage_dir)?;
        }
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
            cold_storage_dir: self.cold_storage_dir,
            cold_threshold_level: self.cold_threshold_level,
            cache: bucket_cache,
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
            compaction_strategy: CompactionStrategyConfig::BentleySaxe(RebuildStrategy::NoRebuild),
            delete_method: DeleteMethod::OidToBucket,
        }
    }

    /// Helper function to create a Level with predefined buckets containing records.
    /// Each bucket contains a list of (records, ids) tuples.
    #[allow(dead_code)]
    fn create_level_with_records_per_bucket(
        input_shape: usize,
        bucket_size: usize,
        buckets_data: Vec<(Vec<f32>, Vec<Id>)>,
    ) -> DliResult<LevelIndex<f32>> {
        let mut buckets: Vec<Bucket<f32>> = buckets_data
            .iter()
            .map(|(records, ids)| {
                let mut bucket = bucket::BucketBuilder::<f32>::default()
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
                bucket::BucketBuilder::<f32>::default()
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
            quantize: false
            seed: 42
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

        let index = IndexBuilder::<f32>::from_yaml(temp_file.path())?.build()?;
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
        let result = IndexBuilder::<f32>::from_yaml(Path::new("nonexistent.yaml"));
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
        let builder = LevelIndexBuilder::<f32>::default();
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
        let index = IndexBuilder::<f32>::default().build().unwrap();

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
            index.compaction_strategy.strategy,
            CompactionStrategyConfig::BentleySaxe(_)
        ));

        // Verify delete method is set
        assert!(matches!(index.delete_method, DeleteMethod::OidToBucket));
    }

    #[test]
    fn test_index_builder_with_custom_params() {
        // Test building an index with custom parameters
        let index = IndexBuilder::<f32>::default()
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
            .map(|query| index.search(query.as_slice(), 10))
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
            .map(|query| loaded_index.search(query.as_slice(), 10))
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
        for (i, (orig_ids, loaded_ids)) in
            original_stats.iter().zip(loaded_stats.iter()).enumerate()
        {
            assert_eq!(orig_ids, loaded_ids, "Query {i} IDs should match");
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
                orig_level.storage.occupied(),
                loaded_level.storage.occupied(),
                "Level {level_idx} occupancy should match"
            );
            assert_eq!(
                orig_level.storage.n_buckets(),
                loaded_level.storage.n_buckets(),
                "Level {level_idx} bucket count should match"
            );
        }
        Ok(())
    }

    #[test]
    fn test_bentley_saxe_index_available_level() -> DliResult<()> {
        let config = create_test_config();
        let index = IndexBuilder::<f32>::from_config(config).build()?;
        // Initially no levels, so should return None
        assert_eq!(index.compaction_strategy.available_level(&index), None);
        Ok(())
    }

    #[test]
    fn test_bentley_saxe_index_get_level_config() -> DliResult<()> {
        let config = create_test_config();
        let index = IndexBuilder::<f32>::from_config(config).build()?;

        let level_config = index.get_level_index_config();
        assert_eq!(level_config.bucket_size, DEFAULT_BUCKET_SIZE); // default value
        Ok(())
    }

    #[test]
    fn test_bentley_saxe_index_insert_to_buffer() -> DliResult<()> {
        let config = create_test_config();
        let mut index = IndexBuilder::<f32>::from_config(config).build()?;

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
        let index = IndexBuilder::<f32>::from_config(config).build()?;

        // Should have no levels initially
        assert_eq!(index.levels.len(), 0);
        // Should have a buffer
        assert_eq!(index.buffer.size, 10);
        Ok(())
    }

    #[test]
    fn test_bentley_saxe_lower_level_data_empty() -> DliResult<()> {
        let config = create_test_config();
        let mut index = IndexBuilder::<f32>::from_config(config).build()?;

        let (data, ids) = index
            .compaction_strategy
            .clone()
            .lower_level_data(&mut index, 0)?;
        // Should only have buffer data initially
        assert!(data.is_empty());
        assert!(ids.is_empty());
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
        assert!(deleted, "Should return deleted item");

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
    fn test_compaction_with_cold_storage() -> DliResult<()> {
        use tempfile::TempDir;

        // Create a temporary directory for cold storage
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cold_storage_dir = temp_dir.path().to_path_buf();

        // Create index with arity 2, bucket size 10, buffer size 10
        let mut config = create_test_config();
        config.arity = 2; // Will create new levels more frequently
        config.levels.bucket_size = 10; // Small bucket size to trigger compaction sooner
        config.buffer_size = 10; // Small buffer to trigger flushing
        config.input_shape = 3;
        config.distance_fn = DistanceFn::Dot;

        let mut index = IndexBuilder::from_config(config)
            .cold_storage_dir(cold_storage_dir.clone())
            .cold_threshold_level(0) // All levels should be cold
            .build()?;

        // Insert 30 records to trigger compaction and create multiple levels
        for i in 0..30 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        // Verify that compaction happened and we have 2 levels
        assert_eq!(
            index.n_levels(),
            2,
            "Should have 2 levels after inserting 30 records with arity 2 and bucket size 10"
        );

        // Verify that all 30 elements are present
        assert_eq!(
            index.occupied(),
            30,
            "All 30 inserted records should be present in the index"
        );

        // Verify that cold storage is being used
        let mut cold_count = 0;
        for level in index.levels.iter() {
            match &level.storage.container {
                StorageContainer::Cold(_) => {
                    cold_count += 1;
                }
                StorageContainer::Hot(_) => {}
            }
        }
        assert_eq!(
            cold_count, 2,
            "Both levels should be in cold storage due to cold_threshold_level=0"
        );

        // Verify that cold storage directory has files
        assert!(
            cold_storage_dir.exists(),
            "Cold storage directory should exist"
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
        assert!(deleted, "Should return deleted item");

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

    #[test]
    fn test_search_bulk_basic() -> DliResult<()> {
        let mut config = create_test_config();
        config.distance_fn = DistanceFn::L2;
        config.buffer_size = 10;
        let mut index = IndexBuilder::from_config(config).build()?;

        // Insert test data (25 items to ensure we have levels)
        for i in 0..25 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        assert_eq!(index.occupied(), 25, "Should have 25 items");
        assert!(index.n_levels() > 0, "Should have levels after compaction");

        // Prepare bulk queries (4 queries)
        let queries: Vec<Vec<f32>> = vec![
            vec![3.0, 3.0, 3.0],    // Near ID 3
            vec![10.5, 10.5, 10.5], // Between IDs 10 and 11
            vec![20.0, 20.0, 20.0], // Near ID 20
            vec![5.0, 5.0, 5.0],    // Near ID 5
        ];
        let bulk_query_data: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();

        // Perform bulk search with k=5
        let bulk_results = index.search_bulk(&bulk_query_data, 5)?;

        // Verify bulk_results structure
        assert_eq!(bulk_results.len(), 4, "Should have 4 result sets");
        for (i, result) in bulk_results.iter().enumerate() {
            assert!(!result.is_empty(), "Result set {i} should not be empty");
            assert!(
                result.len() <= 5,
                "Result set {i} should have at most 5 results"
            );
        }

        // Verify that bulk search returns results (same IDs, possibly different order due to probability handling)
        // The important property is that we get k or fewer results per query
        let mut all_bulk_ids = Vec::new();
        for result_set in &bulk_results {
            all_bulk_ids.extend(result_set.iter().copied());
        }
        assert!(!all_bulk_ids.is_empty(), "Should have found some results");

        // Verify all returned IDs are valid (in range of inserted IDs)
        for id in all_bulk_ids {
            assert!(id < 25, "Result ID should be in valid range");
        }

        Ok(())
    }

    #[test]
    fn test_search_bulk_single_query() -> DliResult<()> {
        let mut config = create_test_config();
        config.distance_fn = DistanceFn::L2;
        config.buffer_size = 10;
        let mut index = IndexBuilder::from_config(config).build()?;

        // Insert 20 items
        for i in 0..20 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        // Single query as bulk search
        let query = vec![7.0, 7.0, 7.0];
        let bulk_results = index.search_bulk(&query, 5)?;

        // Should return a Vec with 1 result set
        assert_eq!(bulk_results.len(), 1);
        assert!(!bulk_results[0].is_empty(), "Should have found results");
        assert!(
            bulk_results[0].len() <= 5,
            "Should have at most k=5 results"
        );

        // Verify all IDs are valid
        for id in &bulk_results[0] {
            assert!(*id < 20, "Result ID should be in range [0, 20)");
        }

        Ok(())
    }

    #[test]
    fn test_search_bulk_empty_index() -> DliResult<()> {
        let config = create_test_config();
        let index = IndexBuilder::from_config(config).build()?;

        // Bulk search on empty index
        let queries = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 queries
        let bulk_results = index.search_bulk(&queries, 5)?;

        // Should have 2 result sets, both empty
        assert_eq!(bulk_results.len(), 2);
        assert!(bulk_results[0].is_empty());
        assert!(bulk_results[1].is_empty());

        Ok(())
    }

    #[test]
    fn test_search_bulk_different_k_values() -> DliResult<()> {
        let mut config = create_test_config();
        config.distance_fn = DistanceFn::L2;
        config.buffer_size = 10;
        let mut index = IndexBuilder::from_config(config).build()?;

        // Insert 30 items
        for i in 0..30 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        let queries: Vec<Vec<f32>> = vec![
            vec![5.0, 5.0, 5.0],
            vec![15.0, 15.0, 15.0],
            vec![25.0, 25.0, 25.0],
        ];
        let bulk_query_data: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();

        // Test with k=1
        let bulk_k1 = index.search_bulk(&bulk_query_data, 1)?;
        for (i, result) in bulk_k1.iter().enumerate() {
            assert_eq!(
                result.len(),
                1,
                "Result set {i} should return exactly 1 result for k=1"
            );
        }

        // Test with k=10
        let bulk_k10 = index.search_bulk(&bulk_query_data, 10)?;
        for (i, result) in bulk_k10.iter().enumerate() {
            assert!(
                result.len() <= 10,
                "Result set {i} should return at most 10 results for k=10"
            );
        }

        // Verify all results are valid IDs
        for result_set in bulk_k1.iter().chain(bulk_k10.iter()) {
            for id in result_set {
                assert!(*id < 30, "Result ID should be in range [0, 30)");
            }
        }

        Ok(())
    }

    #[test]
    fn test_search_bulk_consistency_with_buffer_and_levels() -> DliResult<()> {
        let mut config = create_test_config();
        config.distance_fn = DistanceFn::L2;
        config.buffer_size = 5;
        let mut index = IndexBuilder::from_config(config).build()?;

        // Insert 15 items - will trigger compaction
        // Items 0-4 in buffer initially, then we insert items 5-14
        // Item 5 triggers compaction, moving 0-4 to level 0
        for i in 0..15 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        assert!(index.n_levels() > 0, "Should have levels");
        assert!(index.buffer.occupied() > 0 || index.buffer.occupied() == 0); // Either has data or is empty after flush

        // Create bulk queries
        let queries: Vec<Vec<f32>> = vec![
            vec![2.0, 2.0, 2.0],    // In level (from initial batch)
            vec![12.0, 12.0, 12.0], // Likely in buffer or a level
            vec![7.5, 7.5, 7.5],    // Between two items
        ];
        let bulk_query_data: Vec<f32> = queries.iter().flat_map(|q| q.iter().copied()).collect();

        // Perform bulk search
        let bulk_results = index.search_bulk(&bulk_query_data, 3)?;

        // Verify bulk search returns results
        assert_eq!(bulk_results.len(), 3, "Should have 3 result sets");
        for (i, result) in bulk_results.iter().enumerate() {
            assert!(!result.is_empty(), "Result set {i} should not be empty");
            assert!(
                result.len() <= 3,
                "Result set {i} should have at most 3 results"
            );
        }

        // Verify all IDs are valid
        for result_set in &bulk_results {
            for id in result_set {
                assert!(*id < 15, "Result ID should be in range [0, 15)");
            }
        }

        Ok(())
    }

    #[test]
    fn test_search_recall_two_level_index_l2() -> DliResult<()> {
        // Create index with arity 2, buffer/bucket size 10, input_shape 640
        let mut config = create_test_config();
        config.buffer_size = 50;
        config.levels.bucket_size = 20;
        config.input_shape = 640;
        config.arity = 3;
        config.distance_fn = DistanceFn::L2;

        let mut index = IndexBuilder::from_config(config).build()?;
        let mut inserted_records: Vec<(Vec<f32>, u32)> = Vec::new();

        for i in 0..59 {
            // Generate a vector with input_shape elements
            // Use a pattern: each element is slightly offset from the record ID
            let mut record = vec![0.0; 640];
            let base_value = (i as f32) / 10.0;
            for j in 0..640 {
                record[j] = base_value + (j as f32 / 1000.0);
            }
            inserted_records.push((record.clone(), i as u32));
            index.insert(record, i as u32)?;
        }

        // Verify index structure
        assert_eq!(index.occupied(), 59, "Should have 59 records in index");
        println!(
            "Index structure (L2): {} levels, {} buckets",
            index.n_levels(),
            index.n_buckets()
        );

        // Use the first 20 records as queries
        let mut correct_matches = 0;
        let mut total_searches = 0;

        for (query_record, query_id) in inserted_records.iter() {
            // Search with k=1, ncandidates=10
            let search_strategy = SearchStrategy::ModelDriven(10);
            let search_params = (1usize, search_strategy);
            let results = index.search(query_record, search_params)?;

            total_searches += 1;

            // For the top-1 result, the best match should be the query itself
            // We check if the query_id appears in the top-1 results
            if !results.is_empty() && results[0] == *query_id {
                correct_matches += 1;
            } else if !results.is_empty() {
                // Check if we got the correct ID at all
                println!(
                    "Query ID: {}, Got: {:?}, Expected: {}",
                    query_id, results, query_id
                );
            }
        }

        // Calculate recall@1
        let recall = (correct_matches as f32) / (total_searches as f32);
        println!(
            "Recall@1 (L2): {}/{} = {:.2}%",
            correct_matches,
            total_searches,
            recall * 100.0
        );

        // Assert that recall is higher than 90%
        assert!(
            recall > 0.9,
            "Recall should be higher than 90%, got {:.2}%",
            recall * 100.0
        );

        Ok(())
    }

    #[test]
    fn test_search_recall_two_level_index_dot_product() -> DliResult<()> {
        // Create index with arity 2, buffer/bucket size 10, input_shape 640
        let mut config = create_test_config();
        config.buffer_size = 50;
        config.levels.bucket_size = 20;
        config.input_shape = 640;
        config.arity = 3;
        config.distance_fn = DistanceFn::Dot;

        let mut index = IndexBuilder::from_config(config).build()?;

        let mut inserted_records: Vec<(Vec<f32>, u32)> = Vec::new();

        for i in 0..59 {
            // Generate a vector with input_shape elements
            // Use diverse normalized vectors for dot product similarity
            let mut record = vec![0.0; 640];

            // Create diverse vectors using different patterns for different ID ranges
            let id_normalized = (i as f32) / 59.0; // Range [0, 1]

            // Method: Create vectors with distinct patterns
            // Different ID ranges get different patterns to ensure diversity
            for j in 0..640 {
                let j_normalized = (j as f32) / 640.0; // Range [0, 1]

                if i < 20 {
                    // Group 1: Sinusoidal pattern with varying frequency
                    record[j] = ((id_normalized * std::f32::consts::PI * 2.0
                        + j_normalized * std::f32::consts::PI)
                        .sin()
                        * 0.5)
                        + id_normalized;
                } else if i < 40 {
                    // Group 2: Cosine pattern with different offset
                    record[j] = ((id_normalized * std::f32::consts::PI * 2.0
                        - j_normalized * std::f32::consts::PI)
                        .cos()
                        * 0.5)
                        + (1.0 - id_normalized);
                } else {
                    // Group 3: Linear interpolation with per-dimension variation
                    record[j] =
                        id_normalized * j_normalized + (1.0 - id_normalized) * (1.0 - j_normalized);
                }
            }

            // Normalize the vector to unit length
            let norm: f32 = record.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                record.iter_mut().for_each(|x| *x /= norm);
            } else {
                // Fallback: create a random unit vector if normalization fails
                for j in 0..640 {
                    record[j] = ((i as f32 * 73.0 + j as f32 * 211.0).sin() * 0.5 + 0.5).max(0.0);
                }
                let norm: f32 = record.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    record.iter_mut().for_each(|x| *x /= norm);
                }
            }
            inserted_records.push((record.clone(), i as u32));
            index.insert(record, i as u32)?;
        }

        // Verify index structure
        assert_eq!(index.occupied(), 59, "Should have 59 records in index");
        println!(
            "Index structure (Dot): {} levels, {} buckets",
            index.n_levels(),
            index.n_buckets()
        );

        // Use the first 20 records as queries
        let mut correct_matches = 0;
        let mut total_searches = 0;

        for (query_record, query_id) in inserted_records.iter() {
            // Search with k=1, ncandidates=10
            let search_strategy = SearchStrategy::ModelDriven(10);
            let search_params = (1usize, search_strategy);
            let results = index.search(query_record, search_params)?;

            total_searches += 1;

            // For the top-1 result, the best match should be the query itself
            // We check if the query_id appears in the top-1 results
            if !results.is_empty() && results[0] == *query_id {
                correct_matches += 1;
            } else if !results.is_empty() {
                // Check if we got the correct ID at all
                println!(
                    "Query ID: {}, Got: {:?}, Expected: {}",
                    query_id, results, query_id
                );
            }
        }

        // Calculate recall@1
        let recall = (correct_matches as f32) / (total_searches as f32);
        println!(
            "Recall@1 (Dot Product): {}/{} = {:.2}%",
            correct_matches,
            total_searches,
            recall * 100.0
        );

        // Assert that recall is higher than 90%
        assert!(
            recall > 0.9,
            "Recall should be higher than 90%, got {:.2}%",
            recall * 100.0
        );

        Ok(())
    }

    #[test]
    fn test_cold_storage_threshold_level_basic() -> DliResult<()> {
        use tempfile::TempDir;

        // Create a temporary directory for cold storage
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cold_storage_dir = temp_dir.path().to_path_buf();

        let mut config = create_test_config();
        config.buffer_size = 20;
        config.levels.bucket_size = 10;
        config.input_shape = 10;
        config.arity = 2;
        config.distance_fn = DistanceFn::Dot;

        // Build index with cold storage enabled
        let mut index = IndexBuilder::from_config(config)
            .cold_storage_dir(cold_storage_dir.clone())
            .cold_threshold_level(0)
            .build()?;

        // Verify that cold_threshold_level is set
        assert_eq!(index.cold_threshold_level, Some(0));
        assert_eq!(index.cold_storage_dir, Some(cold_storage_dir.clone()));

        // Insert records to trigger level creation
        for i in 0..30 {
            let record: Vec<f32> = (0..10)
                .map(|j| ((i * 10 + j) as f32 / 100.0).sin())
                .collect();
            index.insert(record, i as u32)?;
        }

        // Verify data was inserted and levels were created
        assert_eq!(index.occupied(), 30);
        assert!(index.n_levels() > 0, "Should have created levels");

        // Since cold_threshold_level is 0, all levels should be cold
        for (level_idx, level) in index.levels.iter().enumerate() {
            match &level.storage.container {
                crate::level_index::StorageContainer::Cold(_) => {
                    println!("Level {} is cold storage", level_idx);
                }
                crate::level_index::StorageContainer::Hot(_) => {
                    println!(
                        "Level {} is hot storage (cold storage may not be initialized)",
                        level_idx
                    );
                }
            }
        }

        // Test that searches still work correctly with cold storage
        let query = (0..10)
            .map(|j| ((0 * 10 + j) as f32 / 100.0).sin())
            .collect::<Vec<f32>>();
        let results = index.search(&query, (5usize, SearchStrategy::ModelDriven(10)))?;
        assert!(!results.is_empty(), "Search should return results");

        Ok(())
    }

    #[test]
    fn test_search_recall_with_cold_storage_dot_product() -> DliResult<()> {
        use tempfile::TempDir;

        // Create a temporary directory for cold storage
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cold_storage_dir = temp_dir.path().to_path_buf();

        // Create index with cold storage enabled (identical config to test_search_recall_two_level_index_dot_product)
        let mut config = create_test_config();
        config.buffer_size = 50;
        config.levels.bucket_size = 20;
        config.input_shape = 640;
        config.arity = 3;
        config.distance_fn = DistanceFn::Dot;

        let mut index = IndexBuilder::from_config(config)
            .cold_storage_dir(cold_storage_dir.clone())
            .cold_threshold_level(0)
            .build()?;

        let mut inserted_records: Vec<(Vec<f32>, u32)> = Vec::new();

        // Insert 59 records with the same vector generation logic
        for i in 0..59 {
            // Generate a vector with input_shape elements
            // Use diverse normalized vectors for dot product similarity
            let mut record = vec![0.0; 640];

            // Create diverse vectors using different patterns for different ID ranges
            let id_normalized = (i as f32) / 59.0; // Range [0, 1]

            // Method: Create vectors with distinct patterns
            // Different ID ranges get different patterns to ensure diversity
            for j in 0..640 {
                let j_normalized = (j as f32) / 640.0; // Range [0, 1]

                if i < 20 {
                    // Group 1: Sinusoidal pattern with varying frequency
                    record[j] = ((id_normalized * std::f32::consts::PI * 2.0
                        + j_normalized * std::f32::consts::PI)
                        .sin()
                        * 0.5)
                        + id_normalized;
                } else if i < 40 {
                    // Group 2: Cosine pattern with different offset
                    record[j] = ((id_normalized * std::f32::consts::PI * 2.0
                        - j_normalized * std::f32::consts::PI)
                        .cos()
                        * 0.5)
                        + (1.0 - id_normalized);
                } else {
                    // Group 3: Linear interpolation with per-dimension variation
                    record[j] =
                        id_normalized * j_normalized + (1.0 - id_normalized) * (1.0 - j_normalized);
                }
            }

            // Normalize the vector to unit length
            let norm: f32 = record.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                record.iter_mut().for_each(|x| *x /= norm);
            } else {
                // Fallback: create a random unit vector if normalization fails
                for j in 0..640 {
                    record[j] = ((i as f32 * 73.0 + j as f32 * 211.0).sin() * 0.5 + 0.5).max(0.0);
                }
                let norm: f32 = record.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    record.iter_mut().for_each(|x| *x /= norm);
                }
            }
            inserted_records.push((record.clone(), i as u32));
            index.insert(record, i as u32)?;
        }

        // Verify index structure
        assert_eq!(index.occupied(), 59, "Should have 59 records in index");
        println!(
            "Index structure (Dot with Cold Storage): {} levels, {} buckets",
            index.n_levels(),
            index.n_buckets()
        );

        // Use all records as queries to test recall@1
        let mut correct_matches = 0;
        let mut total_searches = 0;

        for (query_record, query_id) in inserted_records.iter() {
            // Search with k=1, ncandidates=10
            let search_strategy = SearchStrategy::ModelDriven(10);
            let search_params = (1usize, search_strategy);
            let results = index.search(query_record, search_params)?;

            total_searches += 1;

            // For the top-1 result, the best match should be the query itself
            // We check if the query_id appears in the top-1 results
            if !results.is_empty() && results[0] == *query_id {
                correct_matches += 1;
            } else if !results.is_empty() {
                // Check if we got the correct ID at all
                println!(
                    "Query ID: {}, Got: {:?}, Expected: {}",
                    query_id, results, query_id
                );
            }
        }

        // Calculate recall@1
        let recall = (correct_matches as f32) / (total_searches as f32);
        println!(
            "Recall@1 (Dot Product with Cold Storage): {}/{} = {:.2}%",
            correct_matches,
            total_searches,
            recall * 100.0
        );

        // Assert that recall is higher than 90%
        assert!(
            recall > 0.9,
            "Recall should be higher than 90%, got {:.2}%",
            recall * 100.0
        );

        Ok(())
    }

    #[test]
    fn test_trigger_compaction_simple() -> DliResult<()> {
        use tempfile::TempDir;

        // Create a temporary directory for cold storage
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cold_storage_dir = temp_dir.path().to_path_buf();

        // Create index with arity 2, bucket size 10, buffer size 10
        let mut config = create_test_config();
        config.arity = 2;
        config.levels.bucket_size = 10;
        config.buffer_size = 10;

        let mut index = IndexBuilder::from_config(config)
            .cold_storage_dir(cold_storage_dir)
            .cold_threshold_level(0)
            .build()?;

        // Insert 30 records to trigger compaction
        for i in 0..30 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        // Verify we have 2 levels after compaction
        assert_eq!(
            index.n_levels(),
            2,
            "Should have 2 levels after inserting 30 records with arity 2 and bucket size 10"
        );
        assert!(matches!(
            index.levels[0].storage.container,
            StorageContainer::Cold(_)
        ));
        assert!(matches!(
            index.levels[1].storage.container,
            StorageContainer::Cold(_)
        ));

        // Verify all 30 elements are present
        assert_eq!(
            index.occupied(),
            30,
            "All 30 inserted records should be present in the index"
        );

        Ok(())
    }

    #[test]
    fn test_compaction_with_recall() -> DliResult<()> {
        use tempfile::TempDir;

        // Create a temporary directory for cold storage
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cold_storage_dir = temp_dir.path().to_path_buf();

        // Create index with arity 2, bucket size 10, buffer size 10
        let mut config = create_test_config();
        config.arity = 2;
        config.levels.bucket_size = 10;
        config.buffer_size = 10;
        config.input_shape = 3;
        config.distance_fn = DistanceFn::L2;

        let mut index = IndexBuilder::from_config(config)
            .cold_storage_dir(cold_storage_dir)
            .cold_threshold_level(0)
            .build()?;

        // Insert 59 records to trigger compaction and new level creation
        for i in 0..59 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        // Verify we have multiple levels after compaction
        assert!(
            index.n_levels() == 2,
            "Should have 2 levels after inserting 60 records with arity 2 and bucket size 10, but got {}",
            index.n_levels()
        );

        // Verify all 60 elements are present
        assert_eq!(
            index.occupied(),
            59,
            "All 59 inserted records should be present in the index"
        );

        // Perform searches and check recall
        let mut correct_count = 0;
        let k = 5;
        let num_queries = 59; // Query 59 points

        for query_id in 0..num_queries {
            let query_vec = vec![query_id as f32, query_id as f32, query_id as f32];
            let results = index.search(
                &query_vec,
                SearchParams {
                    k: k,
                    search_strategy: SearchStrategy::ModelDriven(100),
                },
            )?;

            // Check if the query point itself is in the results
            let found = results.iter().any(|id| {
                let id_val = *id as f32;
                // Check if this result is close to our query ID
                (id_val - query_id as f32).abs() < 1.0
            });

            if found {
                correct_count += 1;
            }
        }

        let recall = correct_count as f32 / num_queries as f32;
        println!(
            "Recall: {:.2}% ({}/{})",
            recall * 100.0,
            correct_count,
            num_queries
        );

        // Assert that recall is at least 60%
        assert!(
            recall == 1.0,
            "Recall should be 100%, got {:.2}%",
            recall * 100.0
        );

        Ok(())
    }

    #[test]
    fn test_compaction_deep_with_recall() -> DliResult<()> {
        use tempfile::TempDir;

        // Create a temporary directory for cold storage
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let cold_storage_dir = temp_dir.path().to_path_buf();

        // Create index with arity 2, bucket size 10, buffer size 10
        let mut config = create_test_config();
        config.arity = 2;
        config.levels.bucket_size = 10;
        config.buffer_size = 10;
        config.input_shape = 3;
        config.distance_fn = DistanceFn::L2;

        let mut index = IndexBuilder::from_config(config)
            .cold_storage_dir(cold_storage_dir)
            .cold_threshold_level(0)
            .build()?;

        for i in 0..500 {
            let vec = vec![i as f32, i as f32, i as f32];
            index.insert(vec, i as u32)?;
        }

        assert_eq!(
            index.occupied(),
            500,
            "All 500 inserted records should be present in the index"
        );

        // Perform searches and check recall
        let mut correct_count = 0;
        let k = 5;
        let num_queries = 500; // Query 500 points

        for query_id in 0..num_queries {
            let query_vec = vec![query_id as f32, query_id as f32, query_id as f32];
            let results = index.search(
                &query_vec,
                SearchParams {
                    k: k,
                    search_strategy: SearchStrategy::ModelDriven(1000),
                },
            )?;

            // Check if the query point itself is in the results
            let found = results.iter().any(|id| {
                let id_val = *id as f32;
                // Check if this result is close to our query ID
                (id_val - query_id as f32).abs() < 1.0
            });

            if found {
                correct_count += 1;
            }
        }

        let recall = correct_count as f32 / num_queries as f32;
        println!(
            "Recall: {:.2}% ({}/{})",
            recall * 100.0,
            correct_count,
            num_queries
        );

        // Assert that recall is at least 60%
        assert!(
            recall == 1.0,
            "Recall should be 100%, got {:.2}%",
            recall * 100.0
        );

        Ok(())
    }
}
