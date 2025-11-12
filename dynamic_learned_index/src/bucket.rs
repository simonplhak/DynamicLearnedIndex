use std::fmt::Debug;

use crate::{Array, ArrayNumType, ArraySlice, BuildError, DeleteMethod, Id};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct Buffer {
    records: Vec<ArrayNumType>,
    pub ids: Vec<Id>,
    pub size: usize,
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

    pub fn delete(&mut self, id: &Id) -> Option<(Array, Id)> {
        let idx = self.ids.iter().position(|inner_id| inner_id == id);
        match idx {
            Some(idx) => swap_and_pop(&mut self.records, &mut self.ids, idx, self.input_shape)
                .map(|(deleted, _)| deleted),
            None => None,
        }
    }

    pub fn has_space(&self, count: usize) -> bool {
        self.occupied() + count <= self.size
    }

    pub fn occupied(&self) -> usize {
        self.ids.len()
    }
}

fn swap_and_pop(
    records: &mut Array,
    ids: &mut Vec<Id>,
    idx: usize,
    input_shape: usize,
) -> Option<((Array, Id), (usize, Id))> // (Deleted Array, Deleted Id), (New Index of Swapped Record, Swapped Id)
{
    let occupied = ids.len();
    match occupied - 1 == idx {
        true => {
            // idx is the last one, just pop
            let inner_id = ids.pop().unwrap(); // we are sure that there is something
            let record_start = idx * input_shape;
            let removed_vector: Vec<f32> = records.drain(record_start..).collect();
            Some(((removed_vector, inner_id), (idx, inner_id)))
        }
        false => {
            // idx is not the last one, swap with the last and pop
            ids.swap(idx, occupied - 1);
            let inner_id = ids.pop().unwrap(); // we are sure that there is something
                                               // swap last record with the one to be removed
            let record_start = idx * input_shape;
            let last_record_start = (occupied - 1) * input_shape;
            for i in 0..input_shape {
                records.swap(record_start + i, last_record_start + i);
            }
            // Remove the record from the end
            let removed_vector: Vec<f32> = records.drain(last_record_start..).collect();
            Some(((removed_vector, inner_id), (idx, ids[idx])))
        }
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

    pub fn insert(&mut self, record: Array, id: Id) -> usize {
        if !self.has_space(1) {
            self.resize(1)
        }
        self.records.extend(record);
        self.ids.push(id);
        self.occupied() - 1
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

    pub fn delete(
        &mut self,
        record_idx: usize,
        delete_method: &DeleteMethod,
    ) -> Option<((Array, Id), (usize, Id))> // (Deleted Array, Deleted Id), (New Index of Swapped Record, Swapped Id)
    {
        match delete_method {
            DeleteMethod::OidToBucket => swap_and_pop(
                &mut self.records,
                &mut self.ids,
                record_idx,
                self.input_shape,
            ),
        }
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

    use super::*;

    fn create_bucket() -> Bucket {
        Bucket::new(10, 5)
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
        let idx = bucket.insert(record.clone(), 100);
        assert_eq!(idx, 0);

        assert_eq!(bucket.occupied(), 1);
        assert_eq!(bucket.record(0), record.as_slice());
        assert_eq!(bucket.ids[0], 100);
    }

    #[test]
    fn test_insert_multiple_records() {
        let mut bucket = create_bucket();
        let record1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let record2 = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let record3 = vec![2.5, 3.5, 4.5, 5.5, 6.5];

        let idx = bucket.insert(record1.clone(), 1);
        assert_eq!(idx, 0);
        let idx = bucket.insert(record2.clone(), 2);
        assert_eq!(idx, 1);
        let idx = bucket.insert(record3.clone(), 3);
        assert_eq!(idx, 2);

        assert_eq!(bucket.occupied(), 3);
        assert_eq!(bucket.record(0), record1.as_slice());
        assert_eq!(bucket.record(1), record2.as_slice());
        assert_eq!(bucket.record(2), record3.as_slice());
        assert_eq!(bucket.ids, vec![1, 2, 3]);
    }

    #[test]
    fn test_has_space() {
        let mut bucket = Bucket::new(3, 2);
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
    fn test_record_access() {
        let mut bucket = create_bucket();
        let record1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let record2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];

        bucket.insert(record1.clone(), 1);
        bucket.insert(record2.clone(), 2);

        assert_eq!(bucket.record(0), record1.as_slice());
        assert_eq!(bucket.record(1), record2.as_slice());
    }

    // Helper to flatten per-record vectors into the `records` layout used by the bucket
    fn flatten(records: &[Vec<f32>]) -> Vec<f32> {
        records.iter().flat_map(|r| r.iter().copied()).collect()
    }

    #[test]
    fn test_swap_and_pop_delete_last() {
        let mut ids: Vec<Id> = vec![100u32];
        let mut records = vec![1.0f32, 2.0f32, 3.0f32];
        let input_shape = 3;
        let idx = 0;

        let ((deleted_vec, deleted_id), (new_idx, swapped_id)) =
            swap_and_pop(&mut records, &mut ids, idx, input_shape).unwrap();

        assert_eq!(deleted_vec, vec![1.0, 2.0, 3.0]);
        assert_eq!(deleted_id, 100u32);
        assert_eq!(new_idx, 0);
        assert_eq!(swapped_id, 100u32);
        assert!(ids.is_empty());
        assert!(records.is_empty());
    }

    #[test]
    fn test_swap_and_pop_delete_middle_swaps_last_in() {
        let mut ids: Vec<Id> = vec![1u32, 2u32, 3u32];
        let rec0 = vec![0.0f32, 0.1, 0.2];
        let rec1 = vec![1.0f32, 1.1, 1.2];
        let rec2 = vec![2.0f32, 2.1, 2.2];
        let mut records = flatten(&[rec0.clone(), rec1.clone(), rec2.clone()]);
        let input_shape = 3;
        let idx = 1; // delete rec1

        let ((deleted_vec, deleted_id), (new_idx, swapped_id)) =
            swap_and_pop(&mut records, &mut ids, idx, input_shape).unwrap();

        // The function removes the element that was at `idx` (rec1)
        assert_eq!(deleted_vec, rec1);
        assert_eq!(deleted_id, 2u32);

        // The last record (rec2) should have been moved into position `idx`
        assert_eq!(new_idx, idx);
        assert_eq!(swapped_id, 3u32);

        // ids and records must reflect the removal and the swap
        assert_eq!(ids, vec![1u32, 3u32]);
        assert_eq!(records, flatten(&[rec0, rec2]));
    }

    #[test]
    fn test_swap_and_pop_input_shape_one() {
        let mut ids: Vec<Id> = vec![10u32, 11u32, 12u32];
        let mut records = vec![10.0f32, 11.0f32, 12.0f32];
        let input_shape = 1;
        let idx = 1; // remove middle element (11)

        let ((deleted_vec, deleted_id), (new_idx, swapped_id)) =
            swap_and_pop(&mut records, &mut ids, idx, input_shape).unwrap();

        assert_eq!(deleted_vec, vec![11.0f32]);
        assert_eq!(deleted_id, 11u32);
        assert_eq!(new_idx, 1);
        assert_eq!(swapped_id, 12u32);
        assert_eq!(ids, vec![10u32, 12u32]);
        assert_eq!(records, vec![10.0f32, 12.0f32]);
    }
}
