use std::{
    fmt::Debug,
    fs::File,
    io::{Read as _, Seek as _, Write as _},
    path::Path,
};

use crate::{
    structs::{DiskBucket, DiskBuffer},
    Array, ArrayNumType, ArraySlice, DeleteMethod, DliError, DliResult, Id,
};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct Buffer {
    records: Vec<ArrayNumType>,
    pub ids: Vec<Id>,
    pub size: usize,
    input_shape: usize,
}

impl Buffer {
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

    pub fn memory_usage(&self) -> usize {
        let records_size = self.records.capacity() * std::mem::size_of::<ArrayNumType>();
        let ids_size = self.ids.capacity() * std::mem::size_of::<Id>();
        std::mem::size_of::<Self>() + records_size + ids_size
    }

    pub fn dump(&self, working_dir: &Path) -> DiskBuffer {
        let records_path = working_dir.join("buffer_records.bin");
        let mut records_file = File::create(records_path.clone()).unwrap();
        let records_bytes: &[u8] = bytemuck::cast_slice(&self.records);
        records_file.write_all(records_bytes).unwrap();

        let ids_path = working_dir.join("buffer_ids.bin");
        let mut ids_file = File::create(ids_path.clone()).unwrap();
        let ids_bytes: &[u8] = bytemuck::cast_slice(&self.ids);
        ids_file.write_all(ids_bytes).unwrap();

        DiskBuffer {
            records_path,
            ids_path,
            count: self.occupied(),
        }
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

    pub fn memory_usage(&self) -> usize {
        let records_size = self.records.capacity() * std::mem::size_of::<ArrayNumType>();
        let ids_size = self.ids.capacity() * std::mem::size_of::<Id>();
        std::mem::size_of::<Self>() + records_size + ids_size
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn dump(&self, records_file: &mut File, ids_file: &mut File) -> DiskBucket {
        let records_offset = records_file.stream_position().unwrap();
        let records_bytes: &[u8] = bytemuck::cast_slice(&self.records);
        records_file.write_all(records_bytes).unwrap();
        let ids_offset = ids_file.stream_position().unwrap();
        let ids_bytes: &[u8] = bytemuck::cast_slice(&self.ids);
        ids_file.write_all(ids_bytes).unwrap();
        DiskBucket {
            records_offset,
            ids_offset,
            count: self.occupied(),
        }
    }
}

#[derive(Default)]
pub struct BufferBuilder {
    size: Option<usize>,
    input_shape: Option<usize>,
    disk_buffer: Option<DiskBuffer>,
}

impl BufferBuilder {
    pub fn size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    pub fn input_shape(mut self, input_shape: usize) -> Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn disk_buffer(mut self, disk_buffer: DiskBuffer) -> Self {
        self.disk_buffer = Some(disk_buffer);
        self
    }

    pub fn build(self) -> DliResult<Buffer> {
        let size = self.size.ok_or(DliError::MissingAttribute("size"))?;
        let input_shape = self
            .input_shape
            .ok_or(DliError::MissingAttribute("input_shape"))?;
        let (records, ids) = match self.disk_buffer {
            Some(disk_buffer) => {
                let mut records_file = File::open(&disk_buffer.records_path)?;
                let mut ids_file = File::open(&disk_buffer.ids_path)?;

                // Read records
                records_file.seek(std::io::SeekFrom::Start(0)).unwrap();
                let mut records = vec![0.0f32; disk_buffer.count * input_shape];
                let records_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut records);
                records_file.read_exact(records_bytes).unwrap();

                // Read ids
                ids_file.seek(std::io::SeekFrom::Start(0)).unwrap();
                let mut ids = vec![0u32; disk_buffer.count];
                let ids_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut ids);
                ids_file.read_exact(ids_bytes).unwrap();

                (records, ids)
            }
            None => (
                Vec::with_capacity(size * input_shape),
                Vec::with_capacity(size),
            ),
        };
        Ok(Buffer {
            records,
            ids,
            size,
            input_shape,
        })
    }
}

#[derive(Debug, Default)]
pub(crate) struct BucketBuilder<'a> {
    input_shape: Option<usize>,
    size: Option<usize>,
    disk_bucket: Option<DiskBucket>,
    records_file: Option<&'a mut File>,
    ids_file: Option<&'a mut File>,
}

impl<'a> BucketBuilder<'a> {
    pub fn from_disk(
        disk_bucket: DiskBucket,
        records_file: &'a mut File,
        ids_file: &'a mut File,
    ) -> Self {
        Self {
            input_shape: None,
            size: None,
            disk_bucket: Some(disk_bucket),
            records_file: Some(records_file),
            ids_file: Some(ids_file),
        }
    }

    pub fn input_shape(mut self, input_shape: usize) -> Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn size(mut self, size: usize) -> Self {
        self.size = Some(size);
        self
    }

    pub fn build(self) -> DliResult<Bucket> {
        let size = self.size.ok_or(DliError::MissingAttribute("size"))?;
        let input_shape = self
            .input_shape
            .ok_or(DliError::MissingAttribute("input_shape"))?;
        let (records, ids) = match self.disk_bucket {
            Some(disk_bucket) => {
                let records_file = self
                    .records_file
                    .ok_or(DliError::MissingAttribute("records_file"))?;
                let ids_file = self
                    .ids_file
                    .ok_or(DliError::MissingAttribute("ids_file"))?;

                // Read records
                records_file
                    .seek(std::io::SeekFrom::Start(disk_bucket.records_offset))
                    .unwrap();
                let mut records = vec![0.0f32; disk_bucket.count * input_shape];
                let records_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut records);
                records_file.read_exact(records_bytes).unwrap();

                // Read ids
                ids_file
                    .seek(std::io::SeekFrom::Start(disk_bucket.ids_offset))
                    .unwrap();
                let mut ids = vec![0u32; disk_bucket.count];
                let ids_bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut ids);
                ids_file.read_exact(ids_bytes).unwrap();

                (records, ids)
            }
            None => (Vec::new(), Vec::new()),
        };
        Ok(Bucket {
            records,
            ids,
            size,
            input_shape,
        })
    }
}

#[cfg(test)]
mod tests {

    use tempfile::NamedTempFile;

    use super::*;

    fn create_bucket() -> Bucket {
        BucketBuilder::default()
            .size(10)
            .input_shape(5)
            .build()
            .unwrap()
    }

    #[test]
    fn test_bucket_serialize() {
        let mut original_bucket = create_bucket();
        original_bucket.insert(vec![1.0; original_bucket.input_shape], 1);
        original_bucket.insert(vec![2.0; original_bucket.input_shape], 2);
        let mut records_file = NamedTempFile::new().unwrap();
        let mut ids_file = NamedTempFile::new().unwrap();
        let dump = original_bucket.dump(records_file.as_file_mut(), ids_file.as_file_mut());
        let deserialized =
            BucketBuilder::from_disk(dump, records_file.as_file_mut(), ids_file.as_file_mut())
                .input_shape(original_bucket.input_shape)
                .size(original_bucket.size)
                .build()
                .unwrap();

        assert_eq!(original_bucket.size, deserialized.size);
        assert_eq!(original_bucket.input_shape, deserialized.input_shape);
        assert_eq!(original_bucket.records, deserialized.records);
        assert_eq!(original_bucket.ids, deserialized.ids);
        assert_eq!(original_bucket.occupied(), deserialized.occupied());
        assert_eq!(original_bucket.record(0), deserialized.record(0));
        assert_eq!(original_bucket.record(1), deserialized.record(1));
    }

    #[test]
    fn test_bucket_empty_serialize() {
        let original_bucket = create_bucket();
        let mut records_file = NamedTempFile::new().unwrap();
        let mut ids_file = NamedTempFile::new().unwrap();
        let dump = original_bucket.dump(records_file.as_file_mut(), ids_file.as_file_mut());
        let deserialized =
            BucketBuilder::from_disk(dump, records_file.as_file_mut(), ids_file.as_file_mut())
                .input_shape(original_bucket.input_shape)
                .size(original_bucket.size)
                .build()
                .unwrap();

        assert_eq!(original_bucket.size, deserialized.size);
        assert_eq!(original_bucket.input_shape, deserialized.input_shape);
        assert_eq!(original_bucket.records, deserialized.records);
        assert_eq!(original_bucket.ids, deserialized.ids);
        assert_eq!(original_bucket.occupied(), deserialized.occupied());
    }

    #[test]
    fn test_buffer_serialize() {
        let mut original_buffer = BufferBuilder::default()
            .size(10)
            .input_shape(5)
            .build()
            .unwrap();
        original_buffer.insert(vec![1.0; original_buffer.input_shape], 1);
        original_buffer.insert(vec![2.0; original_buffer.input_shape], 2);
        let working_dir = tempfile::tempdir().unwrap();
        let dump = original_buffer.dump(working_dir.path());
        let deserialized = BufferBuilder::default()
            .size(original_buffer.size)
            .input_shape(original_buffer.input_shape)
            .disk_buffer(dump)
            .build()
            .unwrap();

        assert_eq!(original_buffer.size, deserialized.size);
        assert_eq!(original_buffer.input_shape, deserialized.input_shape);
        assert_eq!(original_buffer.records, deserialized.records);
        assert_eq!(original_buffer.ids, deserialized.ids);
        assert_eq!(original_buffer.occupied(), deserialized.occupied());
        assert_eq!(original_buffer.record(0), deserialized.record(0));
        assert_eq!(original_buffer.record(1), deserialized.record(1));
    }

    #[test]
    fn test_buffer_empty_serialize() {
        let original_buffer = BufferBuilder::default()
            .size(10)
            .input_shape(5)
            .build()
            .unwrap();
        let working_dir = tempfile::tempdir().unwrap();
        let dump = original_buffer.dump(working_dir.path());
        let deserialized = BufferBuilder::default()
            .size(original_buffer.size)
            .input_shape(original_buffer.input_shape)
            .disk_buffer(dump)
            .build()
            .unwrap();

        assert_eq!(original_buffer.size, deserialized.size);
        assert_eq!(original_buffer.input_shape, deserialized.input_shape);
        assert_eq!(original_buffer.records, deserialized.records);
        assert_eq!(original_buffer.ids, deserialized.ids);
        assert_eq!(original_buffer.occupied(), deserialized.occupied());
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
            .build()
            .unwrap();

        assert_eq!(bucket.size(), 20);
        assert_eq!(bucket.input_shape, 3);
    }

    #[test]
    fn test_bucket_builder_missing_attributes() {
        let result = BucketBuilder::default().build();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            DliError::MissingAttribute("size")
        ));
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
        let mut bucket = BucketBuilder::default()
            .size(3)
            .input_shape(2)
            .build()
            .unwrap();
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
        let mut bucket = BucketBuilder::default()
            .size(2)
            .input_shape(3)
            .build()
            .unwrap();

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

    #[test]
    fn test_buffer_delete() {
        let mut buffer = BufferBuilder::default()
            .size(5)
            .input_shape(2)
            .build()
            .unwrap();

        // Insert records: ID 1 -> [1.0, 1.0], ID 2 -> [2.0, 2.0], ID 3 -> [3.0, 3.0]
        buffer.insert(vec![1.0, 1.0], 1);
        buffer.insert(vec![2.0, 2.0], 2);
        buffer.insert(vec![3.0, 3.0], 3);

        assert_eq!(buffer.occupied(), 3);

        // Case A: Delete existing ID (ID 2)
        // This is the middle element, so ID 3 should be swapped into its place.
        let result = buffer.delete(&2);
        assert!(result.is_some());
        let (deleted_record, deleted_id) = result.unwrap();

        assert_eq!(deleted_id, 2);
        assert_eq!(deleted_record, vec![2.0, 2.0]);
        assert_eq!(buffer.occupied(), 2);

        // Verify swap: Index 0 is still ID 1, Index 1 should now be ID 3
        assert_eq!(buffer.ids, vec![1, 3]);
        assert_eq!(buffer.record(0), &[1.0, 1.0]);
        assert_eq!(buffer.record(1), &[3.0, 3.0]);

        // Case B: Delete non-existent ID (ID 99)
        let result = buffer.delete(&99);
        assert!(result.is_none());
        assert_eq!(buffer.occupied(), 2);
        assert_eq!(buffer.ids, vec![1, 3]);
    }

    #[test]
    fn test_buffer_get_data_state() {
        let mut buffer = BufferBuilder::default()
            .size(5)
            .input_shape(2)
            .build()
            .unwrap();

        // Fill partially
        buffer.insert(vec![1.0, 1.0], 1);
        buffer.insert(vec![2.0, 2.0], 2);
        assert_eq!(buffer.occupied(), 2);

        // Call get_data
        let (records, ids) = buffer.get_data();

        // Verify returned data
        assert_eq!(ids, vec![1, 2]);
        assert_eq!(records, vec![1.0, 1.0, 2.0, 2.0]);

        // Verify buffer state after get_data
        assert_eq!(buffer.occupied(), 0);
        assert_eq!(buffer.size, 5); // Capacity/Size setting should remain
        assert_eq!(buffer.input_shape, 2); // Input shape should remain

        // Verify reuse
        buffer.insert(vec![3.0, 3.0], 3);
        assert_eq!(buffer.occupied(), 1);
        assert_eq!(buffer.ids, vec![3]);
        assert_eq!(buffer.record(0), &[3.0, 3.0]);
    }
}
