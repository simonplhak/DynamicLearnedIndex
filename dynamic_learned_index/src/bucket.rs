use std::fmt::Debug;

use crate::{
    errors::BuildError,
    types::{Array, ArrayNumType, ArraySlice},
    DistanceFn, Id,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct Buffer {
    records: Vec<ArrayNumType>,
    pub ids: Vec<Id>,
    size: usize,
    input_shape: usize,
}

impl Buffer {
    pub fn new(size: usize, input_shape: usize) -> Self {
        Self {
            records: Vec::with_capacity(size * input_shape),
            ids: Vec::with_capacity(size),
            size,
            input_shape,
        }
    }

    pub fn record(&self, i: usize) -> &ArraySlice {
        let start = i * self.input_shape;
        let end = start + self.input_shape;
        &self.records[start..end]
    }

    pub fn search(
        &self,
        query: &ArraySlice,
        k: usize,
        distance_fn: &DistanceFn,
    ) -> Vec<(Id, ArrayNumType)> {
        assert!(k > 0);
        let res = flat_knn::knn(
            (&self.records, self.input_shape),
            query,
            k,
            match distance_fn {
                DistanceFn::L2 => flat_knn::Metric::L2,
                DistanceFn::Dot => flat_knn::Metric::Dot,
            },
        );
        res.into_iter()
            .map(|(dist, idx)| (self.ids[idx], dist))
            .collect()
    }

    pub fn insert(&mut self, record: Array, id: Id) {
        if !self.has_space(1) {
            panic!("Buffer is full, cannot insert new record");
        }
        self.records.extend(record);
        self.ids.push(id);
    }

    pub fn get_data(&mut self) -> (Array, Vec<Id>) {
        let records = std::mem::replace(
            &mut self.records,
            Vec::with_capacity(self.size * self.input_shape),
        );
        let ids = std::mem::replace(&mut self.ids, Vec::with_capacity(self.size));

        (records, ids)
    }

    pub fn has_space(&self, count: usize) -> bool {
        self.occupied() + count <= self.size
    }

    pub fn occupied(&self) -> usize {
        self.ids.len()
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct Bucket {
    records: Vec<ArrayNumType>,
    pub ids: Vec<Id>,
    size: usize,
    input_shape: usize,
}

impl Bucket {
    fn new(size: usize, input_shape: usize) -> Self {
        Self {
            records: Vec::with_capacity(size * input_shape),
            ids: Vec::with_capacity(size),
            size,
            input_shape,
        }
    }

    pub fn record(&self, i: usize) -> &ArraySlice {
        let start = i * self.input_shape;
        let end = start + self.input_shape;
        &self.records[start..end]
    }

    pub fn search(
        &self,
        query: &ArraySlice,
        k: usize,
        distance_fn: &DistanceFn,
    ) -> Vec<(Id, ArrayNumType)> {
        assert!(k > 0);
        let res = flat_knn::knn(
            (&self.records, self.input_shape),
            query,
            k,
            match distance_fn {
                DistanceFn::L2 => flat_knn::Metric::L2,
                DistanceFn::Dot => flat_knn::Metric::Dot,
            },
        );
        res.into_iter()
            .map(|(dist, idx)| (self.ids[idx], dist))
            .collect()
    }

    pub fn insert(&mut self, record: Array, id: Id) {
        if !self.has_space(1) {
            self.resize(1)
        }
        self.records.extend(record);
        self.ids.push(id);
    }

    pub fn resize(&mut self, new_n_objects: usize) {
        assert!(new_n_objects > 0);
        self.records.reserve(new_n_objects * self.input_shape);
        self.ids.reserve(new_n_objects);
    }

    pub fn get_data(&mut self) -> (Array, Vec<Id>) {
        let records = std::mem::take(&mut self.records);
        let ids = std::mem::take(&mut self.ids);
        (records, ids)
    }

    pub fn has_space(&self, count: usize) -> bool {
        self.occupied() + count <= self.size
    }

    pub fn occupied(&self) -> usize {
        self.ids.len()
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

#[derive(Debug, Default)]
pub(crate) struct BucketBuilder {
    input_shape: Option<usize>,
    size: Option<usize>,
    is_dynamic: bool,
}

impl BucketBuilder {
    pub fn input_shape(&mut self, input_shape: usize) -> &mut Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn size(&mut self, size: usize) -> &mut Self {
        self.size = Some(size);
        self
    }

    pub fn is_dynamic(&mut self, is_dynamic: bool) -> &mut Self {
        self.is_dynamic = is_dynamic;
        self
    }

    pub fn build(&self) -> Result<Bucket, BuildError> {
        let size = self.size.ok_or(BuildError::MissingAttribute)?;
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;
        Ok(Bucket::new(size, input_shape))
    }
}

#[cfg(test)]
mod tests {
    use crate::DistanceFn;

    use super::*;

    fn create_bucket() -> Bucket {
        Bucket::new(10, 5)
    }

    fn create_static_bucket() -> Bucket {
        Bucket::new(3, 2)
    }

    #[test]
    fn test_new_bucket() {
        let bucket = create_bucket();
        assert_eq!(bucket.size, 10);
        assert_eq!(bucket.input_shape, 5);
        assert_eq!(bucket.occupied(), 0);
    }

    #[test]
    fn test_bucket_builder() {
        let bucket = BucketBuilder::default()
            .size(20)
            .input_shape(3)
            .is_dynamic(false)
            .build()
            .unwrap();

        assert_eq!(bucket.size(), 20);
        assert_eq!(bucket.input_shape, 3);
    }

    #[test]
    fn test_bucket_builder_missing_attributes() {
        let result = BucketBuilder::default().build();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), BuildError::MissingAttribute));
    }

    #[test]
    fn test_insert_single_record() {
        let mut bucket = create_bucket();
        let record = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        bucket.insert(record.clone(), 100);

        assert_eq!(bucket.occupied(), 1);
        assert_eq!(bucket._record(0), record.as_slice());
        assert_eq!(bucket.ids[0], 100);
    }

    #[test]
    fn test_insert_multiple_records() {
        let mut bucket = create_bucket();
        let record1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let record2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let record3 = vec![2.5, 3.5, 4.5, 5.5, 6.5];

        bucket.insert(record1.clone(), 1);
        bucket.insert(record2.clone(), 2);
        bucket.insert(record3.clone(), 3);

        assert_eq!(bucket.occupied(), 3);
        assert_eq!(bucket._record(0), record1.as_slice());
        assert_eq!(bucket._record(1), record2.as_slice());
        assert_eq!(bucket._record(2), record3.as_slice());
        assert_eq!(bucket.ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_has_space() {
        let mut bucket = create_static_bucket(); // size 3
        assert!(bucket.has_space(1));
        assert!(bucket.has_space(3));
        assert!(!bucket.has_space(4));

        bucket.insert(vec![1.0, 2.0], 1);
        assert!(bucket.has_space(2));
        assert!(!bucket.has_space(3));

        bucket.insert(vec![3.0, 4.0], 2);
        bucket.insert(vec![5.0, 6.0], 3);
        assert!(!bucket.has_space(1));
    }

    // #[test]
    // fn test_search_single_record() {
    //     let mut bucket = create_bucket(DistanceFn::Dot);
    //     let record = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    //     bucket.insert(record.clone(), 42);

    //     let query = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    //     let (ids, distances) = bucket.search(&query, 1);

    //     assert_eq!(ids.len(), 1);
    //     assert_eq!(ids[0], 42);
    //     assert_eq!(distances.len(), 1);
    //     // For dot product with identical vectors, we expect a high similarity
    //     assert!(distances[0] > 0.0);
    // }

    // #[test]
    // fn test_search_multiple_records_dot() {
    //     let mut bucket = create_bucket(DistanceFn::Dot);
    //     let record1 = vec![1.0, 0.0, 0.0, 0.0, 0.0]; // Should have lower dot product with query
    //     let record2 = vec![1.0, 1.0, 1.0, 1.0, 1.0]; // Should have higher dot product with query
    //     let record3 = vec![2.0, 2.0, 2.0, 2.0, 2.0]; // Should have highest dot product with query

    //     bucket.insert(record1, 1);
    //     bucket.insert(record2, 2);
    //     bucket.insert(record3, 3);

    //     let query = vec![1.0, 1.0, 1.0, 1.0, 1.0];
    //     let (ids, _distances) = bucket.search(&query, 2);

    //     // For dot product, higher values are better, so id 3 should come first
    //     assert_eq!(ids.len(), 2);
    //     assert_eq!(ids[0], 3); // record3 has highest dot product
    //     assert_eq!(ids[1], 2); // record2 has second highest
    // }

    // #[test]
    // fn test_search_multiple_records_l2() {
    //     let mut bucket = Bucket::new("test".to_string(), 10, 3, true, DistanceFn::L2);
    //     let record1 = vec![1.0, 1.0, 1.0]; // Distance sqrt(3) from origin
    //     let record2 = vec![2.0, 2.0, 2.0]; // Distance sqrt(12) from origin
    //     let record3 = vec![0.5, 0.5, 0.5]; // Distance sqrt(0.75) from origin

    //     bucket.insert(record1, 1);
    //     bucket.insert(record2, 2);
    //     bucket.insert(record3, 3);

    //     let query = vec![0.0, 0.0, 0.0]; // Origin
    //     let (ids, _distances) = bucket.search(&query, 2);

    //     // For L2 distance, smaller values are better, so id 3 should come first
    //     assert_eq!(ids.len(), 2);
    //     assert_eq!(ids[0], 3); // record3 has smallest L2 distance
    //     assert_eq!(ids[1], 1); // record1 has second smallest
    // }

    #[test]
    fn test_get_data() {
        let mut bucket = create_bucket();
        let record1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let record2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];

        bucket.insert(record1.clone(), 10);
        bucket.insert(record2.clone(), 20);

        let (records, ids) = bucket.get_data();

        // Records should be a flattened array containing all values
        let expected_records = vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(records, expected_records);
        assert_eq!(ids, vec![10, 20]);

        // After get_data, bucket should be empty
        assert_eq!(bucket.occupied(), 0);
    }

    #[test]
    fn test_resize_dynamic_bucket() {
        let mut bucket = Bucket::new(2, 3);

        // Fill the bucket
        bucket.insert(vec![1.0, 2.0, 3.0], 1);
        bucket.insert(vec![4.0, 5.0, 6.0], 2);
        assert_eq!(bucket.occupied(), 2);
        assert!(!bucket.has_space(1));

        // Insert one more - should trigger resize
        bucket.insert(vec![7.0, 8.0, 9.0], 3);
        assert_eq!(bucket.occupied(), 3);
    }

    #[test]
    fn test_distance_fn_l2() {
        let distance_fn = DistanceFn::L2;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let distance = distance_fn.distance(&a, &b);
        // L2 distance between [1,2,3] and [4,5,6] should be sqrt(27) ≈ 5.196
        assert!((distance - 5.196).abs() < 0.01);
    }

    #[test]
    fn test_distance_fn_dot() {
        let distance_fn = DistanceFn::Dot;
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let distance = distance_fn.distance(&a, &b);
        // Dot product of [1,2,3] and [4,5,6] should be 1*4 + 2*5 + 3*6 = 32
        assert_eq!(distance, 32.0);
    }

    #[test]
    fn test_distance_fn_cmp() {
        let l2_fn = DistanceFn::L2;
        let dot_fn = DistanceFn::Dot;

        // For L2, smaller is better
        assert_eq!(l2_fn.cmp(&1.0, &2.0), std::cmp::Ordering::Less);
        assert_eq!(l2_fn.cmp(&2.0, &1.0), std::cmp::Ordering::Greater);

        // For Dot, larger is better (reversed comparison)
        assert_eq!(dot_fn.cmp(&1.0, &2.0), std::cmp::Ordering::Greater);
        assert_eq!(dot_fn.cmp(&2.0, &1.0), std::cmp::Ordering::Less);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_distance_fn_mismatched_lengths() {
        let distance_fn = DistanceFn::L2;
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        distance_fn.distance(&a, &b);
    }

    #[test]
    #[should_panic]
    fn test_resize_static_bucket_panics() {
        let mut bucket = create_static_bucket();
        // Try to force resize on a static bucket
        bucket.resize(1);
    }

    #[test]
    fn test_record_access() {
        let mut bucket = create_bucket();
        let record1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let record2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        bucket.insert(record1.clone(), 1);
        bucket.insert(record2.clone(), 2);

        assert_eq!(bucket._record(0), record1.as_slice());
        assert_eq!(bucket._record(1), record2.as_slice());
    }
}
