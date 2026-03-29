use crate::{
    bucket::{self, Bucket, FloatElement},
    model::{Model, ModelBuilder, ModelConfig, ModelDevice, ModelInterface as _},
    structs::{DiskBucket, DiskLevelIndex, LevelIndexConfig},
    ArraySlice, DeleteMethod, DistanceFn, DliError, DliResult, Id,
};
#[cfg(feature = "measure_time")]
use log::debug;
use measure_time_macro::log_time;
use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

#[derive(Debug, Default)]
pub(crate) struct LevelIndexBuilder<F: FloatElement> {
    id: Option<String>,
    n_buckets: Option<usize>,
    buckets: Option<(Vec<DiskBucket>, PathBuf, PathBuf)>,
    buckets_in_memory: Option<Vec<Bucket<F>>>,
    model_config: Option<ModelConfig>,
    bucket_size: Option<usize>,
    input_shape: Option<usize>,
    model_device: ModelDevice,
    distance_fn: Option<DistanceFn>,
}

impl<F: FloatElement> LevelIndexBuilder<F> {
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

    #[allow(dead_code)]
    pub fn buckets_in_memory(mut self, buckets: Vec<Bucket<F>>) -> Self {
        self.buckets_in_memory = Some(buckets);
        self
    }

    pub fn build(self) -> DliResult<LevelIndex<F>> {
        let input_shape = self
            .input_shape
            .ok_or(DliError::MissingAttribute("input_shape"))?;
        let bucket_size = self
            .bucket_size
            .ok_or(DliError::MissingAttribute("bucket_size"))?;
        let buckets = if let Some(buckets) = self.buckets_in_memory {
            // Use pre-built in-memory buckets directly
            buckets
        } else if let Some((buckets, records_path, ids_path)) = self.buckets {
            // Load buckets from disk
            let mut records_file = File::open(records_path)?;
            let mut ids_file = File::open(ids_path)?;
            buckets
                .into_iter()
                .map(|disk_bucket| {
                    bucket::BucketBuilder::<F>::from_disk(
                        disk_bucket,
                        &mut records_file,
                        &mut ids_file,
                    )
                    .input_shape(input_shape)
                    .size(bucket_size)
                    .build()
                })
                .collect::<Result<Vec<_>, _>>()?
        } else {
            // Create empty buckets
            let n_buckets = self
                .n_buckets
                .ok_or(DliError::MissingAttribute("n_buckets"))?;
            (0..n_buckets)
                .map(|_| {
                    bucket::BucketBuilder::<F>::default()
                        .input_shape(input_shape)
                        .size(bucket_size)
                        .build()
                })
                .collect::<Result<Vec<_>, _>>()?
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
            .device(self.model_device)
            .input_nodes(input_shape as i64)
            .train_params(model_config.train_params)
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

pub struct LevelIndex<F: FloatElement> {
    model: Model,
    buckets: Vec<Bucket<F>>,
    ids_map: HashMap<Id, (usize, usize)>, // Id -> (bucket_idx, record_idx)
}

impl<F: FloatElement> LevelIndex<F> {
    fn new(model: Model, buckets: Vec<Bucket<F>>) -> Self {
        Self {
            model,
            buckets,
            ids_map: HashMap::new(),
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.buckets[0].size() * self.buckets.len()
    }

    pub(crate) fn bucket_size(&self) -> usize {
        self.buckets[0].size()
    }

    pub(crate) fn occupied(&self) -> usize {
        self.buckets.iter().map(|bucket| bucket.occupied()).sum()
    }

    pub(crate) fn free_space(&self) -> usize {
        let size = self.size();
        let occupied = self.occupied();
        match size > occupied {
            true => size - occupied,
            false => 0,
        }
    }

    pub(crate) fn n_buckets(&self) -> usize {
        self.buckets.len()
    }

    pub(crate) fn n_empty_buckets(&self) -> usize {
        self.buckets
            .iter()
            .filter(|bucket| bucket.occupied() == 0)
            .count()
    }

    pub(crate) fn bucket(&self, bucket_idx: usize) -> &Bucket<F> {
        &self.buckets[bucket_idx]
    }

    pub fn memory_usage(&self) -> usize {
        let buckets_size: usize = self.buckets.iter().map(|b| b.memory_usage()).sum();
        let map_capacity = self.ids_map.capacity();
        // size of Key + Value + 1 byte control (SwissTable)
        let entry_size = std::mem::size_of::<Id>() + std::mem::size_of::<(usize, usize)>() + 1;
        let map_size = map_capacity * entry_size;

        std::mem::size_of::<Self>() + self.model.memory_usage() + buckets_size + map_size
    }

    pub(crate) fn buckets2visit_predictions(
        &self,
        query: &ArraySlice,
    ) -> DliResult<Vec<(usize, f32, usize)>> {
        if self.occupied() == 0 {
            return Ok(self
                .buckets
                .iter()
                .enumerate()
                .map(|(bucket_id, _)| (bucket_id, 0.0, 0))
                .collect());
        }
        let query = self.model.vec2tensor(query)?;
        let preds = self
            .model
            .predict(&query)?
            .into_iter()
            .map(|(bucket_id, prob)| (bucket_id, prob, self.buckets[bucket_id].occupied()))
            .collect();
        Ok(preds)
    }

    pub(crate) fn buckets2visit_predictions_many(
        &self,
        queries: &[&[f32]],
    ) -> DliResult<Vec<Vec<(usize, f32, usize)>>> {
        if self.occupied() == 0 {
            let empty_predictions = self
                .buckets
                .iter()
                .enumerate()
                .map(|(bucket_id, _)| (bucket_id, 0.0, 0))
                .collect::<Vec<_>>();
            return Ok(vec![empty_predictions; queries.len()]);
        }

        let mut flat_queries = Vec::new();
        for query in queries {
            flat_queries.extend_from_slice(query);
        }

        // Get batch predictions (bucket assignments) using predict_many
        let assignments = self.model.predict_many(&flat_queries)?;

        // Group assignments by query and convert to the required format
        let mut results = vec![Vec::new(); queries.len()];
        for (assignment_idx, &bucket_idx) in assignments.iter().enumerate() {
            let query_idx = assignment_idx;
            let bucket = &self.buckets[bucket_idx];
            // (bucket_idx, probability=0.0, occupied_count)
            results[query_idx].push((bucket_idx, 0.0, bucket.occupied()));
        }

        Ok(results)
    }

    #[log_time]
    pub(crate) fn train(&mut self, xs: &[F]) -> DliResult<()> {
        self.model.train(F::to_f32_slice(xs))?;
        Ok(())
    }

    #[log_time]
    pub(crate) fn retrain(&mut self, xs: &[F]) -> DliResult<()> {
        self.model.retrain(F::to_f32_slice(xs))?;
        Ok(())
    }

    #[log_time]
    pub(crate) fn insert_many(&mut self, records: Vec<F>, ids: Vec<Id>) -> DliResult<()> {
        let input_shape = self.model.input_shape;
        assert!(records.len() / input_shape == ids.len());
        if records.is_empty() {
            return Ok(());
        }
        let assignments = self.model.predict_many(F::to_f32_slice(&records))?;
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
                let id = ids.pop().expect("ids mismatch");
                let bucket_idx = assignments.pop().expect("assignments mismatch");
                let record_idx = self.buckets[bucket_idx].insert(query, id);
                self.ids_map.insert(id, (bucket_idx, record_idx));
            }
            assert!(assignments.is_empty());
            assert!(ids.is_empty());
            assert!(records.is_empty());
        }
        Ok(())
    }

    pub(crate) fn get_data(&mut self) -> (Vec<F>, Vec<Id>) {
        self.ids_map.clear(); // clear existing id mappings
        let (data, ids): (Vec<_>, Vec<Vec<Id>>) = self
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

    pub(crate) fn delete(&mut self, id: &Id, delete_method: &DeleteMethod) -> Option<(Vec<F>, Id)> {
        let deleted = self.ids_map.get(id).cloned();
        if let Some((bucket_idx, record_idx)) = deleted {
            assert!(bucket_idx < self.buckets.len());
            assert!(record_idx < self.buckets[bucket_idx].occupied());
            let bucket = &mut self.buckets[bucket_idx];
            let (deleted, (swapped_new_idx, swapped_id)) =
                bucket.delete(record_idx, delete_method)?;
            let (deleted_bucket_idx, deleted_record_idx) = self.ids_map.remove(id)?; // we are sure it exists
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

    pub(crate) fn dump(&self, working_dir: &Path, level_id: usize) -> DliResult<DiskLevelIndex> {
        let model = self
            .model
            .dump(working_dir.join(format!("model-{level_id}.safetensors")))?;
        let records_path = working_dir.join(format!("bucket-records-{level_id}.bin"));
        let ids_path = working_dir.join(format!("bucket-ids-{level_id}.bin"));
        let mut records_file = File::create(records_path.clone())?;
        let mut ids_file = File::create(ids_path.clone())?;
        let disk_buckets = self
            .buckets
            .iter()
            .map(|bucket| bucket.dump(&mut records_file, &mut ids_file))
            .collect::<Vec<_>>();
        let config = LevelIndexConfig {
            model,
            bucket_size: self.buckets[0].size(),
        };
        Ok(DiskLevelIndex {
            buckets: disk_buckets,
            config,
            records_path,
            ids_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::DliResult;
    use crate::model::RetrainStrategy;
    use crate::structs::DistanceFn;
    use crate::{ModelConfig, ModelLayer, TrainParams};

    fn make_level_with_records(records: Vec<Vec<f32>>, ids: Vec<Id>) -> LevelIndex<f32> {
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
            .expect("level build failed");

        for (rec, id) in records.into_iter().zip(ids.into_iter()) {
            level.buckets[0].insert(rec, id);
            level
                .ids_map
                .insert(id, (0, level.buckets[0].occupied() - 1));
        }
        level
    }

    #[test]
    fn test_level_index_builder_minimal_params() {
        let builder = LevelIndexBuilder::<f32>::default();

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
        let level_l2 = LevelIndexBuilder::<f32>::default()
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
        let level_dot = LevelIndexBuilder::<f32>::default()
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
                retrain_strategy: RetrainStrategy::NoRetrain,
            },
            weights_path: None,
        };

        let level_custom = LevelIndexBuilder::<f32>::default()
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
    fn test_level_index_save_and_load() -> DliResult<()> {
        use tempfile::TempDir;

        // Create a level with some buckets
        let input_shape = 10;
        let n_buckets = 4;
        let bucket_size = 50;

        let mut level = LevelIndexBuilder::<f32>::default()
            .n_buckets(n_buckets)
            .input_shape(input_shape)
            .bucket_size(bucket_size)
            .model(ModelConfig::default())
            .distance_fn(DistanceFn::Dot)
            .build()?;

        // Generate training data (100 samples, 10 features each)
        let training_data: Vec<f32> = (0..1000).map(|i| (i % 100) as f32 / 100.0).collect();

        // Train the level
        level.train(&training_data)?;

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
        level.insert_many(insert_data, insert_ids)?;

        // Create test queries
        let test_queries: Vec<Vec<f32>> = vec![
            (0..input_shape).map(|i| i as f32 / 10.0).collect(),
            (0..input_shape).map(|i| (i + 5) as f32 / 10.0).collect(),
            (0..input_shape).map(|i| (i * 2) as f32 / 10.0).collect(),
            (0..input_shape).map(|i| (i + 3) as f32 / 15.0).collect(),
        ];

        // Get predictions from original level
        let original_predictions = test_queries
            .iter()
            .map(|query| level.buckets2visit_predictions(query))
            .collect::<DliResult<Vec<_>>>()?;

        // Save level to temporary directory
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let level_id = 0;
        let disk_level = level.dump(temp_dir.path(), level_id)?;

        // Verify disk files were created
        assert!(disk_level.records_path.exists());
        assert!(disk_level.ids_path.exists());

        // Load level from disk
        let loaded_level = LevelIndexBuilder::<f32>::default()
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
            .build()?;

        // Verify loaded level has same properties
        assert_eq!(loaded_level.n_buckets(), n_buckets);
        assert_eq!(loaded_level.model.input_shape, input_shape);
        assert_eq!(loaded_level.occupied(), 20); // Same number of records

        // Get predictions from loaded level
        let loaded_predictions = test_queries
            .iter()
            .map(|query| loaded_level.buckets2visit_predictions(query))
            .collect::<DliResult<Vec<_>>>()?;

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
        Ok(())
    }

    #[test]
    fn test_level_index_delete_middle_swaps_last_in() -> DliResult<()> {
        let rec0 = vec![0.0f32, 0.1, 0.2];
        let rec1 = vec![1.0f32, 1.1, 1.2];
        let rec2 = vec![2.0f32, 2.1, 2.2];
        let ids = vec![1u32, 2u32, 3u32];
        let mut level =
            make_level_with_records(vec![rec0.clone(), rec1.clone(), rec2.clone()], ids.clone());

        // delete middle id (2)
        let res = level.delete(&2u32, &DeleteMethod::OidToBucket);
        let (deleted_vec, deleted_id) = res.expect("expected middle deletion");
        assert_eq!(deleted_id, 2u32);
        assert_eq!(deleted_vec, rec1);

        // ids_map should not contain deleted id
        assert!(!level.ids_map.contains_key(&2u32));
        // moved id (3) should now be at index 1
        assert_eq!(level.ids_map.get(&3u32).cloned(), Some((0usize, 1usize)));
        // bucket should have two records and record(1) equals rec2
        assert_eq!(level.buckets[0].occupied(), 2);
        assert_eq!(level.buckets[0].record(1), rec2.as_slice());
        Ok(())
    }

    #[test]
    fn test_level_index_delete_last_element() -> DliResult<()> {
        let rec = vec![1.0f32, 2.0, 3.0];
        let id = 42u32;
        let mut level = make_level_with_records(vec![rec.clone()], vec![id]);
        let res = level.delete(&id, &DeleteMethod::OidToBucket);
        let (deleted_vec, deleted_id) = res.expect("expected deletion");
        assert_eq!(deleted_id, id);
        assert_eq!(deleted_vec, rec);

        // id should be removed from ids_map
        assert!(!level.ids_map.contains_key(&id));
        // bucket should be empty
        assert_eq!(level.buckets[0].occupied(), 0);
        Ok(())
    }

    #[test]
    fn test_level_index_delete_missing_id_returns_none() -> DliResult<()> {
        let mut level = make_level_with_records(vec![], vec![]);
        let res = level.delete(&999u32, &DeleteMethod::OidToBucket);
        assert!(res.is_none());
        Ok(())
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

    #[test]
    fn test_level_builder_with_in_memory_buckets() -> DliResult<()> {
        // Test that we can build a level directly with pre-populated buckets in memory
        let input_shape = 3;
        let bucket_size = 50;

        // Create records for two buckets
        let bucket_0_records = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 records
        let bucket_0_ids = [1u32, 2u32];

        let bucket_1_records = [7.0, 8.0, 9.0]; // 1 record
        let bucket_1_ids = [3u32];

        // Create buckets in memory
        let mut bucket_0 = bucket::BucketBuilder::<f32>::default()
            .input_shape(input_shape)
            .size(bucket_size)
            .build()?;
        for (rec, id) in bucket_0_records
            .chunks_exact(input_shape)
            .zip(bucket_0_ids.iter())
        {
            bucket_0.insert(rec.to_vec(), *id);
        }

        let mut bucket_1 = bucket::BucketBuilder::<f32>::default()
            .input_shape(input_shape)
            .size(bucket_size)
            .build()?;
        for (rec, id) in bucket_1_records
            .chunks_exact(input_shape)
            .zip(bucket_1_ids.iter())
        {
            bucket_1.insert(rec.to_vec(), *id);
        }

        // Build level with in-memory buckets
        let level = LevelIndexBuilder::default()
            .input_shape(input_shape)
            .bucket_size(bucket_size)
            .buckets_in_memory(vec![bucket_0, bucket_1])
            .model(ModelConfig::default())
            .model_device(ModelDevice::Cpu)
            .distance_fn(DistanceFn::Dot)
            .build()?;

        // Verify the level was created correctly
        assert_eq!(level.n_buckets(), 2);
        assert_eq!(level.occupied(), 3); // 2 + 1 records
        assert_eq!(level.buckets[0].occupied(), 2);
        assert_eq!(level.buckets[1].occupied(), 1);

        // Verify the records are correct
        assert_eq!(level.buckets[0].ids, vec![1u32, 2u32]);
        assert_eq!(level.buckets[1].ids, vec![3u32]);

        Ok(())
    }

    #[test]
    fn test_mixed_operations() -> DliResult<()> {
        // 1. Initialize LevelIndex with 1 bucket
        let input_shape = 3;
        let mut level = LevelIndexBuilder::default()
            .n_buckets(1)
            .input_shape(input_shape)
            .bucket_size(100)
            .model(ModelConfig::default())
            .model_device(ModelDevice::Cpu)
            .distance_fn(DistanceFn::Dot)
            .build()?;

        let rec_a = vec![1.0, 1.0, 1.0];
        let id_a = 1u32;
        let rec_b = vec![2.0, 2.0, 2.0];
        let id_b = 2u32;
        let rec_c = vec![3.0, 3.0, 3.0];
        let id_c = 3u32;

        // 2. Insert A
        level.insert_many(rec_a.clone(), vec![id_a])?;
        assert_eq!(level.ids_map.get(&id_a), Some(&(0, 0)));

        // 3. Insert B
        level.insert_many(rec_b.clone(), vec![id_b])?;
        assert_eq!(level.ids_map.get(&id_a), Some(&(0, 0)));
        assert_eq!(level.ids_map.get(&id_b), Some(&(0, 1)));

        // 4. Delete A
        // Since we have [A, B], deleting A (idx 0) should swap B (last) to idx 0.
        level.delete(&id_a, &DeleteMethod::OidToBucket);

        assert!(!level.ids_map.contains_key(&id_a));
        assert_eq!(
            level.ids_map.get(&id_b),
            Some(&(0, 0)),
            "B should have moved to index 0"
        );

        // Verify bucket content
        assert_eq!(level.buckets[0].occupied(), 1);
        assert_eq!(level.buckets[0].record(0), rec_b.as_slice());

        // 5. Insert C
        level.insert_many(rec_c.clone(), vec![id_c])?;

        // 6. Verify ids_map consistency
        assert_eq!(level.ids_map.get(&id_b), Some(&(0, 0)));
        assert_eq!(level.ids_map.get(&id_c), Some(&(0, 1)));

        // 7. Verify bucket data
        assert_eq!(level.buckets[0].occupied(), 2);
        assert_eq!(level.buckets[0].record(0), rec_b.as_slice());
        assert_eq!(level.buckets[0].record(1), rec_c.as_slice());

        Ok(())
    }

    #[test]
    fn test_insert_many_empty() -> DliResult<()> {
        let mut level = LevelIndexBuilder::<f32>::default()
            .n_buckets(1)
            .input_shape(3)
            .bucket_size(100)
            .model(ModelConfig::default())
            .model_device(ModelDevice::Cpu)
            .distance_fn(DistanceFn::Dot)
            .build()?;

        // Initial state check
        assert_eq!(level.occupied(), 0);
        assert!(level.ids_map.is_empty());

        // Call insert_many with empty vectors
        level.insert_many(vec![], vec![])?;

        // Verify state hasn't changed
        assert_eq!(level.occupied(), 0);
        assert!(level.ids_map.is_empty());

        Ok(())
    }

    #[test]
    fn test_level_stats() -> DliResult<()> {
        let bucket_size = 10;
        let n_buckets = 2;
        let input_shape = 2;

        // Create level
        let mut level = LevelIndexBuilder::default()
            .n_buckets(n_buckets)
            .input_shape(input_shape)
            .bucket_size(bucket_size)
            .model(ModelConfig::default())
            .model_device(ModelDevice::Cpu)
            .distance_fn(DistanceFn::Dot)
            .build()?;

        // 1. Initialization
        assert_eq!(level.size(), 20);
        assert_eq!(level.occupied(), 0);
        assert_eq!(level.free_space(), 20);
        assert_eq!(level.n_empty_buckets(), 2);

        // 2. Manual Insertion to control distribution
        // Insert into bucket 0
        let rec1 = vec![1.0, 1.0];
        let id1 = 1u32;
        level.buckets[0].insert(rec1.clone(), id1);
        level.ids_map.insert(id1, (0, 0));

        assert_eq!(level.size(), 20);
        assert_eq!(level.occupied(), 1);
        assert_eq!(level.free_space(), 19);
        assert_eq!(level.n_empty_buckets(), 1);

        // 3. Insert into bucket 1
        let rec2 = vec![2.0, 2.0];
        let id2 = 2u32;
        level.buckets[1].insert(rec2.clone(), id2);
        level.ids_map.insert(id2, (1, 0));

        assert_eq!(level.occupied(), 2);
        assert_eq!(level.free_space(), 18);
        assert_eq!(level.n_empty_buckets(), 0);

        // 4. Insert another into bucket 0
        let rec3 = vec![3.0, 3.0];
        let id3 = 3u32;
        level.buckets[0].insert(rec3.clone(), id3);
        level.ids_map.insert(id3, (0, 1));

        assert_eq!(level.occupied(), 3);
        assert_eq!(level.free_space(), 17);
        assert_eq!(level.n_empty_buckets(), 0);

        // 5. Deletion from bucket 1 (making it empty)
        level.delete(&id2, &DeleteMethod::OidToBucket);

        assert_eq!(level.occupied(), 2);
        assert_eq!(level.free_space(), 18);
        assert_eq!(level.n_empty_buckets(), 1);

        Ok(())
    }
}
