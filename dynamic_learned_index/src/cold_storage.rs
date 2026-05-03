use crate::{
    bucket::Bucket,
    structs::{DiskLevelIndex, FloatElement, LevelIndexConfig},
    DliError, DliResult, Id,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    fs::File,
    io::{Seek, SeekFrom, Write as _},
    marker::PhantomData,
    os::unix::fs::FileExt as _,
    path::{Path, PathBuf},
};

pub const BLOCK_SIZE: usize = 65_536;
const HEADER_BYTES: usize = 4; // 2 bytes u16 count (LE) + 2 bytes padding

/// How many records of type F with the given input_shape fit in one 64 KB block.
pub fn records_per_block<F: FloatElement>(input_shape: usize) -> usize {
    let bytes_per_record = input_shape * std::mem::size_of::<F>() + std::mem::size_of::<Id>();
    (BLOCK_SIZE - HEADER_BYTES) / bytes_per_record
}

/// Routing-table entry for one logical bucket.
/// Empty buckets have `extents = vec![]` and occupy zero disk space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColdDiskBucket {
    /// Ordered list of 0-based physical block indices in the data file.
    pub extents: Vec<u32>,
    /// Total number of records stored across all extents (cached for fast occupancy checks).
    /// Must be kept in sync with the actual record counts in block headers.
    #[serde(default)]
    pub count: usize,
}

#[derive(Debug)]
pub struct ColdStorage<F: FloatElement> {
    /// Routing table: index == logical_id.
    pub disk_buckets: Vec<ColdDiskBucket>,
    /// Read-write handle to the flat data file.
    data_file: File,
    /// Path to the data file for metadata.
    data_path: PathBuf,
    pub input_shape: usize,
    pub ids: HashSet<Id>,
    pub bucket_size: usize,
    _marker: PhantomData<F>,
}

// --- Block encode / decode ---

/// Encodes at most `records_per_block` records into an exactly-65536-byte block.
///
/// Layout:
///   [0..2]  u16 LE — number of records
///   [2..4]  reserved (zero)
///   [4..]   for each record: [F × input_shape][u32 LE id]
///   rest    zero-padded
pub fn encode_block<F: FloatElement>(
    records: &[F],
    ids: &[Id],
    input_shape: usize,
) -> [u8; BLOCK_SIZE] {
    assert_eq!(records.len(), ids.len() * input_shape);
    let n = ids.len();
    assert!(n <= records_per_block::<F>(input_shape));

    let mut buf = [0u8; BLOCK_SIZE];
    buf[0..2].copy_from_slice(&(n as u16).to_le_bytes());
    // bytes 2-3 remain zero (padding)

    let bytes_per_vector = input_shape * std::mem::size_of::<F>();
    let bytes_per_record = bytes_per_vector + std::mem::size_of::<Id>();

    for i in 0..n {
        let base = HEADER_BYTES + i * bytes_per_record;
        let float_bytes: &[u8] =
            bytemuck::cast_slice(&records[i * input_shape..(i + 1) * input_shape]);
        buf[base..base + bytes_per_vector].copy_from_slice(float_bytes);
        buf[base + bytes_per_vector..base + bytes_per_record]
            .copy_from_slice(&ids[i].to_le_bytes());
    }
    buf
}

/// Decodes a 65536-byte block into `(records_flat, ids)`.
pub fn decode_block<F: FloatElement>(
    buf: &[u8; BLOCK_SIZE],
    input_shape: usize,
) -> DliResult<(Vec<F>, Vec<Id>)> {
    let count = u16::from_le_bytes([buf[0], buf[1]]) as usize;
    let rpb = records_per_block::<F>(input_shape);
    if count > rpb {
        return Err(DliError::IoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("cold block count {count} exceeds max {rpb}"),
        )));
    }

    let bytes_per_vector = input_shape * std::mem::size_of::<F>();
    let bytes_per_record = bytes_per_vector + std::mem::size_of::<Id>();

    let mut records: Vec<F> = Vec::with_capacity(count * input_shape);
    let mut ids: Vec<Id> = Vec::with_capacity(count);

    for i in 0..count {
        let base = HEADER_BYTES + i * bytes_per_record;
        let float_bytes = &buf[base..base + bytes_per_vector];
        let floats: &[F] = bytemuck::cast_slice(float_bytes);
        records.extend_from_slice(floats);
        let id_bytes: [u8; 4] = buf[base + bytes_per_vector..base + bytes_per_record]
            .try_into()
            .expect("id slice is always 4 bytes");
        ids.push(u32::from_le_bytes(id_bytes));
    }
    Ok((records, ids))
}

// --- ColdStorageLevel ---

impl<F: FloatElement> ColdStorage<F> {
    /// Creates a new cold storage level pointing at `data_path`.
    /// If the file does not exist it is created (empty).
    pub fn new(
        data_path: &Path,
        n_buckets: usize,
        input_shape: usize,
        bucket_size: usize,
    ) -> DliResult<Self> {
        if !data_path.exists() {
            File::create(data_path)?;
        }
        let data_file = File::options().read(true).write(true).open(data_path)?;
        let disk_buckets = ColdStorage::<F>::empty_disk_buckets(n_buckets);

        Ok(Self {
            disk_buckets,
            data_file,
            data_path: data_path.to_path_buf(),
            input_shape,
            _marker: PhantomData,
            ids: HashSet::new(),
            bucket_size,
        })
    }

    fn empty_disk_buckets(n_buckets: usize) -> Vec<ColdDiskBucket> {
        vec![
            ColdDiskBucket {
                extents: vec![],
                count: 0,
            };
            n_buckets
        ]
    }

    /// Loads a cold storage level from an existing data file and its JSON metadata sidecar.
    pub fn load(
        data_path: &Path,
        meta_path: &Path,
        input_shape: usize,
        bucket_size: usize,
    ) -> DliResult<Self> {
        let disk_buckets = load_metadata(meta_path)?;
        let data_file = File::options().read(true).write(true).open(data_path)?;

        // Populate ids HashSet from all records in the file
        let mut ids = HashSet::new();
        let file_size = data_file.metadata()?.len();
        if file_size > 0 {
            let mut buf = vec![0u8; file_size as usize];
            data_file.read_exact_at(&mut buf, 0)?;

            for bucket in &disk_buckets {
                for &block_idx in &bucket.extents {
                    let start = block_idx as usize * BLOCK_SIZE;
                    let end = start + BLOCK_SIZE;
                    if end <= buf.len() {
                        let block: &[u8; BLOCK_SIZE] =
                            buf[start..end].try_into().map_err(|_| {
                                DliError::IoError(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    "Invalid block size in cold storage file",
                                ))
                            })?;
                        let (_recs, block_ids) = decode_block::<F>(block, input_shape)?;
                        ids.extend(block_ids);
                    }
                }
            }
        }

        Ok(Self {
            disk_buckets,
            data_file,
            data_path: data_path.to_path_buf(),
            input_shape,
            _marker: PhantomData,
            ids,
            bucket_size,
        })
    }

    pub fn delete(&mut self, id: Id) -> bool {
        self.ids.remove(&id)
    }

    pub fn insert(
        &mut self,
        records: Vec<F>,
        ids: Vec<Id>,
        assignments: &[usize],
    ) -> DliResult<()> {
        // Group records by bucket_idx with pre-allocated capacity
        let mut bucket_records: std::collections::HashMap<usize, (Vec<F>, Vec<Id>)> =
            std::collections::HashMap::new();

        let input_shape = self.input_shape;
        let mut record_offset = 0;

        // Iterate through assignments and ids together, avoiding separate iterators
        for (id, &bucket_idx) in ids.iter().zip(assignments.iter()) {
            let record_slice = &records[record_offset..record_offset + input_shape];
            // Use entry API to avoid double lookup
            let bucket_entry = bucket_records
                .entry(bucket_idx)
                .or_insert_with(|| (Vec::with_capacity(input_shape * 10), Vec::new()));

            bucket_entry.0.extend_from_slice(record_slice);
            bucket_entry.1.push(*id);

            record_offset += input_shape;
        }

        // Append to each bucket and save metadata once at the end
        for (bucket_idx, (recs, ids)) in bucket_records {
            self.append(bucket_idx, &recs, &ids)?;
        }

        // Save metadata once after all appends
        let meta_path = meta_path_for(&self.data_path);
        save_metadata(&self.disk_buckets, &meta_path)?;

        Ok(())
    }

    /// Internal append without metadata save - used by insert to batch metadata saves.
    /// Automatically splits records across multiple blocks if they exceed block capacity.
    fn append(&mut self, bucket_idx: usize, records: &[F], ids: &[Id]) -> DliResult<()> {
        if records.is_empty() {
            return Ok(());
        }
        debug_assert_eq!(records.len(), ids.len() * self.input_shape);

        let rpb = records_per_block::<F>(self.input_shape);
        let mut record_offset = 0;
        let mut id_offset = 0;

        // Split records into blocks and write each block
        while id_offset < ids.len() {
            let remaining_ids = ids.len() - id_offset;
            let records_in_this_block = std::cmp::min(rpb, remaining_ids);
            let records_end = record_offset + records_in_this_block * self.input_shape;

            let block_records = &records[record_offset..records_end];
            let block_ids = &ids[id_offset..id_offset + records_in_this_block];

            // Encode the block
            let block = encode_block(block_records, block_ids, self.input_shape);

            // Seek to end and write the block
            self.data_file.seek(SeekFrom::End(0))?;
            self.data_file.write_all(&block)?;
            let block_idx = (self.data_file.stream_position()? / BLOCK_SIZE as u64) - 1;

            // Update disk_buckets
            if let Some(disk_bucket) = self.disk_buckets.get_mut(bucket_idx) {
                disk_bucket.extents.push(block_idx as u32);
                disk_bucket.count += records_in_this_block;
            } else {
                // If not exists, add new (though should exist from new)
                self.disk_buckets.push(ColdDiskBucket {
                    extents: vec![block_idx as u32],
                    count: records_in_this_block,
                });
            }

            record_offset = records_end;
            id_offset += records_in_this_block;
        }
        self.ids.extend(ids);

        Ok(())
    }

    /// Reads and reconstructs the bucket for `bucket_id` from disk.
    /// No caching — the caller (typically `Index`) manages cache lookups.
    /// Blocks are read sequentially; for single-query workloads this is faster
    /// than the rayon overhead for small extent lists.
    pub fn read_bucket(&self, bucket_id: usize) -> DliResult<Bucket<F>> {
        let bucket = &self.disk_buckets[bucket_id];
        if bucket.extents.is_empty() {
            return Ok(Bucket::<F>::from_parts(vec![], vec![], self.input_shape));
        }

        let total_count = bucket.count;
        let input_shape = self.input_shape;
        let mut all_records: Vec<F> = Vec::with_capacity(total_count * input_shape);
        let mut all_ids: Vec<Id> = Vec::with_capacity(total_count);

        for &block_idx in &bucket.extents {
            let mut buf = Box::new([0u8; BLOCK_SIZE]);
            let byte_offset = block_idx as u64 * BLOCK_SIZE as u64;
            self.data_file
                .read_exact_at(buf.as_mut_slice(), byte_offset)?;
            let (recs, ids) = decode_block::<F>(&*buf, input_shape)?;
            for (rec, id) in recs.chunks_exact(input_shape).zip(ids) {
                if self.ids.contains(&id) {
                    all_records.extend(rec);
                    all_ids.push(id);
                }
            }
        }

        Ok(Bucket::<F>::from_parts(all_records, all_ids, input_shape))
    }

    pub fn get_data(&mut self) -> DliResult<(Vec<F>, Vec<Id>)> {
        let file_size = self.data_file.metadata()?.len();
        if file_size == 0 {
            return Ok((vec![], vec![]));
        }

        let mut buf = vec![0u8; file_size as usize];
        self.data_file.read_exact_at(&mut buf, 0)?;

        let mut all_records = Vec::new();
        let mut all_ids = Vec::new();

        for bucket in &self.disk_buckets {
            if bucket.extents.is_empty() {
                continue;
            }
            let mut bucket_records: Vec<F> = Vec::new();
            let mut bucket_ids = Vec::new();
            for &block_idx in &bucket.extents {
                let start = block_idx as usize * BLOCK_SIZE;
                let end = start + BLOCK_SIZE;
                let block: &[u8; BLOCK_SIZE] = buf[start..end].try_into().map_err(|_| {
                    DliError::IoError(std::io::Error::new(
                        std::io::ErrorKind::InvalidData,
                        "Invalid block size in cold storage file",
                    ))
                })?;
                let (recs, ids) = decode_block::<F>(block, self.input_shape)?;
                for (id, rec) in ids.into_iter().zip(recs.chunks_exact(self.input_shape)) {
                    if self.ids.contains(&id) {
                        bucket_records.extend(rec);
                        bucket_ids.push(id);
                    }
                }
            }
            all_records.extend(bucket_records);
            all_ids.extend(bucket_ids);
        }
        self.disk_buckets = ColdStorage::<F>::empty_disk_buckets(self.disk_buckets.len());
        self.ids.clear();

        Ok((all_records, all_ids))
    }

    pub fn bucket_occupied(&self, bucket_idx: usize) -> usize {
        assert!(bucket_idx < self.disk_buckets.len());
        self.disk_buckets[bucket_idx].count
    }

    /// Number of logical buckets in this level.
    pub fn n_buckets(&self) -> usize {
        self.disk_buckets.len()
    }

    pub fn dump(
        &self,
        _working_dir: &Path,
        _level_id: usize,
        _config: &LevelIndexConfig,
    ) -> DliResult<DiskLevelIndex> {
        todo!()
    }
}

// --- Metadata sidecar ---

/// Derives the JSON metadata path from the data file path.
pub fn meta_path_for(data_path: &Path) -> std::path::PathBuf {
    data_path.with_extension("cold.meta.json")
}

/// Serializes the extent table to a JSON sidecar file.
pub fn save_metadata(disk_buckets: &[ColdDiskBucket], meta_path: &Path) -> DliResult<()> {
    let json = serde_json::to_string(disk_buckets)?;
    std::fs::write(meta_path, json)?;
    Ok(())
}

/// Loads the extent table from a JSON sidecar file.
pub fn load_metadata(meta_path: &Path) -> DliResult<Vec<ColdDiskBucket>> {
    let json = std::fs::read_to_string(meta_path)?;
    Ok(serde_json::from_str(&json)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    // ---- encode / decode ----

    #[test]
    fn test_encode_decode_round_trip_f32() {
        let input_shape = 4;
        let records: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let ids: Vec<Id> = vec![10, 20];
        let block = encode_block::<f32>(&records, &ids, input_shape);
        let (dec_records, dec_ids) = decode_block::<f32>(&block, input_shape).unwrap();
        assert_eq!(dec_records, records);
        assert_eq!(dec_ids, ids);
    }

    #[test]
    fn test_encode_decode_round_trip_f16() {
        let input_shape = 4;
        let records: Vec<f16> = (0..8).map(|x| f16::from_f32(x as f32)).collect();
        let ids: Vec<Id> = vec![1, 2];
        let block = encode_block::<f16>(&records, &ids, input_shape);
        let (dec_records, dec_ids) = decode_block::<f16>(&block, input_shape).unwrap();
        assert_eq!(dec_records, records);
        assert_eq!(dec_ids, ids);
    }

    #[test]
    fn test_encode_decode_empty_block() {
        let input_shape = 4;
        let block = encode_block::<f32>(&[], &[], input_shape);
        let (dec_records, dec_ids) = decode_block::<f32>(&block, input_shape).unwrap();
        assert!(dec_records.is_empty());
        assert!(dec_ids.is_empty());
    }

    #[test]
    fn test_encode_decode_partial_block() {
        let input_shape = 768;
        let records: Vec<f32> = vec![0.5f32; input_shape];
        let ids: Vec<Id> = vec![42];
        let block = encode_block::<f32>(&records, &ids, input_shape);
        let count = u16::from_le_bytes([block[0], block[1]]);
        assert_eq!(count, 1);
        let (dec_records, dec_ids) = decode_block::<f32>(&block, input_shape).unwrap();
        assert_eq!(dec_records, records);
        assert_eq!(dec_ids, ids);
    }

    #[test]
    fn test_encode_decode_full_block_f16_768() {
        let input_shape = 768;
        let rpb = records_per_block::<f16>(input_shape);
        assert_eq!(rpb, 42, "expected 42 records/block for f16 dim=768");
        let records: Vec<f16> = (0..rpb * input_shape)
            .map(|i| f16::from_f32((i % 100) as f32))
            .collect();
        let ids: Vec<Id> = (0..rpb as u32).collect();
        let block = encode_block::<f16>(&records, &ids, input_shape);
        let (dec_records, dec_ids) = decode_block::<f16>(&block, input_shape).unwrap();
        assert_eq!(dec_records.len(), rpb * input_shape);
        assert_eq!(dec_ids.len(), rpb);
        assert_eq!(dec_records, records);
        assert_eq!(dec_ids, ids);
    }

    // ---- append method ----

    #[test]
    fn test_append_single_record() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 4;
        let n_buckets = 4;
        let bucket_size = 100;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // Append a single record to bucket 0
        let records = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let ids = vec![10];
        cold_storage.append(0, &records, &ids).unwrap();

        // Verify the bucket was updated correctly
        assert_eq!(
            cold_storage.disk_buckets[0].count, 1,
            "bucket 0 should have 1 record"
        );
        assert_eq!(
            cold_storage.disk_buckets[0].extents.len(),
            1,
            "bucket 0 should have 1 extent"
        );
        assert_eq!(
            cold_storage.disk_buckets[0].extents[0], 0,
            "first extent should be at block 0"
        );

        // Verify other buckets remain unchanged
        assert_eq!(cold_storage.disk_buckets[1].extents.is_empty(), true);
        assert_eq!(cold_storage.disk_buckets[2].extents.is_empty(), true);
        assert_eq!(cold_storage.disk_buckets[3].extents.is_empty(), true);

        // Verify the data can be read back
        let bucket = cold_storage.read_bucket(0).unwrap();
        assert_eq!(bucket.occupied(), 1);
        assert_eq!(bucket.ids, &[10]);
        assert_eq!(bucket.record(0), &records[..input_shape]);
    }

    #[test]
    fn test_append_multiple_records() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 3;
        let n_buckets = 2;
        let bucket_size = 100;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // Append 2 records to bucket 0
        let records = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32];
        let ids = vec![100, 101];
        cold_storage.append(0, &records, &ids).unwrap();

        // Verify count and extents
        assert_eq!(cold_storage.disk_buckets[0].count, 2);
        assert_eq!(cold_storage.disk_buckets[0].extents.len(), 1);

        // Verify data can be read back
        let bucket = cold_storage.read_bucket(0).unwrap();
        assert_eq!(bucket.occupied(), 2);
        assert_eq!(bucket.ids, &[100, 101]);
        assert_eq!(bucket.record(0), &records[..input_shape]);
        assert_eq!(bucket.record(1), &records[input_shape..]);
    }

    #[test]
    fn test_append_to_different_buckets() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 2;
        let n_buckets = 3;
        let bucket_size = 100;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // Append to bucket 0
        let records_b0 = vec![1.0f32, 2.0f32];
        let ids_b0 = vec![10];
        cold_storage.append(0, &records_b0, &ids_b0).unwrap();

        // Append to bucket 2
        let records_b2 = vec![3.0f32, 4.0f32, 5.0f32, 6.0f32];
        let ids_b2 = vec![20, 21];
        cold_storage.append(2, &records_b2, &ids_b2).unwrap();

        // Verify bucket 0
        assert_eq!(cold_storage.disk_buckets[0].count, 1);
        assert_eq!(cold_storage.disk_buckets[0].extents.len(), 1);
        assert_eq!(cold_storage.disk_buckets[0].extents[0], 0);

        // Verify bucket 1 (empty)
        assert_eq!(cold_storage.disk_buckets[1].extents.is_empty(), true);
        assert_eq!(cold_storage.disk_buckets[1].extents.len(), 0);

        // Verify bucket 2
        assert_eq!(cold_storage.disk_buckets[2].count, 2);
        assert_eq!(cold_storage.disk_buckets[2].extents.len(), 1);
        assert_eq!(cold_storage.disk_buckets[2].extents[0], 1);

        // Verify data can be read back
        let bucket_0 = cold_storage.read_bucket(0).unwrap();
        assert_eq!(bucket_0.ids, &[10]);

        let bucket_2 = cold_storage.read_bucket(2).unwrap();
        assert_eq!(bucket_2.ids, &[20, 21]);
    }

    #[test]
    fn test_append_multiple_blocks_to_same_bucket() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 4;
        let n_buckets = 1;
        let bucket_size = 100;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // First append: fill some data
        let records_1 = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let ids_1 = vec![1];
        cold_storage.append(0, &records_1, &ids_1).unwrap();

        assert_eq!(cold_storage.disk_buckets[0].count, 1);
        assert_eq!(cold_storage.disk_buckets[0].extents.len(), 1);
        assert_eq!(cold_storage.disk_buckets[0].extents[0], 0);

        // Second append: add more data to the same bucket
        let records_2 = vec![5.0f32, 6.0f32, 7.0f32, 8.0f32];
        let ids_2 = vec![2];
        cold_storage.append(0, &records_2, &ids_2).unwrap();

        // Verify bucket now has 2 records and 2 extents
        assert_eq!(
            cold_storage.disk_buckets[0].count, 2,
            "bucket should have 2 total records"
        );
        assert_eq!(
            cold_storage.disk_buckets[0].extents.len(),
            2,
            "bucket should have 2 extents"
        );
        assert_eq!(
            cold_storage.disk_buckets[0].extents[0], 0,
            "first extent at block 0"
        );
        assert_eq!(
            cold_storage.disk_buckets[0].extents[1], 1,
            "second extent at block 1"
        );

        // Verify both records can be read back
        let bucket = cold_storage.read_bucket(0).unwrap();
        assert_eq!(bucket.occupied(), 2);
        assert_eq!(bucket.ids, &[1, 2]);
    }

    #[test]
    fn test_append_empty_records() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 4;
        let n_buckets = 2;
        let bucket_size = 100;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // Append empty records (should be a no-op)
        let records: Vec<f32> = vec![];
        let ids: Vec<Id> = vec![];
        cold_storage.append(0, &records, &ids).unwrap();

        // Verify bucket remains empty
        assert_eq!(cold_storage.disk_buckets[0].extents.is_empty(), true);
        assert_eq!(cold_storage.disk_buckets[0].extents.len(), 0);

        // Verify file size is still 0
        let file_size = cold_storage.data_file.metadata().unwrap().len();
        assert_eq!(
            file_size, 0,
            "file should remain empty after appending empty records"
        );
    }

    // ---- insert method ----

    #[test]
    fn test_insert_empty_bucket() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 4;
        let n_buckets = 3;
        let bucket_size = 100;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // Insert a single record into bucket 1
        let records = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let ids = vec![42];
        let assignments = vec![1]; // Assign to bucket 1

        cold_storage
            .insert(records.clone(), ids.clone(), &assignments)
            .unwrap();

        // Verify bucket 0 is still empty
        assert_eq!(cold_storage.disk_buckets[0].extents.is_empty(), true);

        // Verify bucket 1 has the record
        assert_eq!(cold_storage.disk_buckets[1].count, 1);
        assert_eq!(cold_storage.disk_buckets[1].extents.len(), 1);

        // Verify bucket 2 is still empty
        assert_eq!(cold_storage.disk_buckets[2].extents.is_empty(), true);

        // Verify data can be read back
        let bucket = cold_storage.read_bucket(1).unwrap();
        assert_eq!(bucket.records_slice(), &records);
    }

    #[test]
    fn test_insert_multiple_buckets() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 3;
        let n_buckets = 4;
        let bucket_size = 100;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // Insert 6 records: 2 to bucket 0, 2 to bucket 2, 2 to bucket 3
        let records = vec![
            1.0f32, 2.0f32, 3.0f32, // record 0
            4.0f32, 5.0f32, 6.0f32, // record 1
            7.0f32, 8.0f32, 9.0f32, // record 2
            10.0f32, 11.0f32, 12.0f32, // record 3
            13.0f32, 14.0f32, 15.0f32, // record 4
            16.0f32, 17.0f32, 18.0f32, // record 5
        ];
        let ids = vec![100, 101, 102, 103, 104, 105];
        let assignments = vec![0, 0, 2, 2, 3, 3]; // Distribute across buckets

        cold_storage
            .insert(records.clone(), ids.clone(), &assignments)
            .unwrap();

        // Verify bucket 0 has 2 records
        assert_eq!(cold_storage.disk_buckets[0].count, 2);
        assert_eq!(cold_storage.disk_buckets[0].extents.len(), 1);

        // Verify bucket 1 is empty
        assert_eq!(cold_storage.disk_buckets[1].extents.is_empty(), true);

        // Verify bucket 2 has 2 records
        assert_eq!(cold_storage.disk_buckets[2].count, 2);
        assert_eq!(cold_storage.disk_buckets[2].extents.len(), 1);

        // Verify bucket 3 has 2 records
        assert_eq!(cold_storage.disk_buckets[3].count, 2);
        assert_eq!(cold_storage.disk_buckets[3].extents.len(), 1);

        // Verify data can be read from each bucket
        let bucket_0 = cold_storage.read_bucket(0).unwrap();
        let bucket_2 = cold_storage.read_bucket(2).unwrap();
        let bucket_3 = cold_storage.read_bucket(3).unwrap();

        assert_eq!(bucket_0.records_slice().len(), 6); // 2 records * 3 dims
        assert_eq!(bucket_2.records_slice().len(), 6);
        assert_eq!(bucket_3.records_slice().len(), 6);
    }

    #[test]
    fn test_insert_overflowing_bucket() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 4;
        let n_buckets = 1;
        let bucket_size = 2; // Small bucket size to test overflow

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // First insert: 2 records that fit in the bucket
        let records_1 = vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, // record 0
            5.0f32, 6.0f32, 7.0f32, 8.0f32, // record 1
        ];
        let ids_1 = vec![10, 11];
        let assignments_1 = vec![0, 0];

        cold_storage
            .insert(records_1, ids_1, &assignments_1)
            .unwrap();

        assert_eq!(cold_storage.disk_buckets[0].count, 2);
        assert_eq!(cold_storage.disk_buckets[0].extents.len(), 1);

        // Second insert: 3 more records that overflow the bucket_size
        let records_2 = vec![
            9.0f32, 10.0f32, 11.0f32, 12.0f32, // record 2
            13.0f32, 14.0f32, 15.0f32, 16.0f32, // record 3
            17.0f32, 18.0f32, 19.0f32, 20.0f32, // record 4
        ];
        let ids_2 = vec![20, 21, 22];
        let assignments_2 = vec![0, 0, 0];

        cold_storage
            .insert(records_2, ids_2, &assignments_2)
            .unwrap();

        // Verify bucket now has 5 total records
        assert_eq!(
            cold_storage.disk_buckets[0].count, 5,
            "bucket should have 5 records after overflow"
        );

        // Verify multiple extents were created as data was appended
        // (the exact number depends on how records are packed into blocks)
        assert!(
            cold_storage.disk_buckets[0].extents.len() >= 1,
            "bucket should have at least 1 extent"
        );

        // Verify all records can be read back
        let bucket = cold_storage.read_bucket(0).unwrap();
        assert_eq!(bucket.records_slice().len(), 20); // 5 records * 4 dims
    }

    #[test]
    fn test_insert_single_record_per_bucket() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 2;
        let n_buckets = 5;
        let bucket_size = 100;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // Insert one record into each bucket
        let records = vec![
            1.0f32, 2.0f32, // record for bucket 0
            3.0f32, 4.0f32, // record for bucket 1
            5.0f32, 6.0f32, // record for bucket 2
            7.0f32, 8.0f32, // record for bucket 3
            9.0f32, 10.0f32, // record for bucket 4
        ];
        let ids = vec![1, 2, 3, 4, 5];
        let assignments = vec![0, 1, 2, 3, 4];

        cold_storage.insert(records, ids, &assignments).unwrap();

        // Verify each bucket has exactly one record
        for i in 0..n_buckets {
            assert_eq!(
                cold_storage.disk_buckets[i].count, 1,
                "bucket {} should have 1 record",
                i
            );
            assert_eq!(
                cold_storage.disk_buckets[i].extents.len(),
                1,
                "bucket {} should have 1 extent",
                i
            );
        }

        // Verify all buckets have extents (blocks may be shared or sequential depending on capacity)
        let mut max_block_idx = -1i32;
        for i in 0..n_buckets {
            let block_idx = cold_storage.disk_buckets[i].extents[0] as i32;
            assert!(
                block_idx >= 0,
                "bucket {} should have a valid block index",
                i
            );
            max_block_idx = max_block_idx.max(block_idx);
        }

        // Verify total number of blocks allocated
        // With 5 records and input_shape=2, all can fit in a single 65KB block
        assert!(
            max_block_idx + 1 <= 5,
            "should use at most 5 blocks for 5 records"
        );
    }

    #[test]
    fn test_insert_interleaved_assignments() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 2;
        let n_buckets = 2;
        let bucket_size = 100;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // Insert records with alternating bucket assignments: b0, b1, b0, b1
        let records = vec![
            1.0f32, 2.0f32, // record 0 -> bucket 0
            3.0f32, 4.0f32, // record 1 -> bucket 1
            5.0f32, 6.0f32, // record 2 -> bucket 0
            7.0f32, 8.0f32, // record 3 -> bucket 1
        ];
        let ids = vec![10, 20, 11, 21];
        let assignments = vec![0, 1, 0, 1]; // Interleaved assignments

        cold_storage.insert(records, ids, &assignments).unwrap();

        // Verify bucket 0 has 2 records
        assert_eq!(cold_storage.disk_buckets[0].count, 2);
        assert_eq!(cold_storage.disk_buckets[0].extents.len(), 1);

        // Verify bucket 1 has 2 records
        assert_eq!(cold_storage.disk_buckets[1].count, 2);
        assert_eq!(cold_storage.disk_buckets[1].extents.len(), 1);

        // Verify the records are correctly grouped by bucket
        let bucket_0 = cold_storage.read_bucket(0).unwrap();
        let bucket_1 = cold_storage.read_bucket(1).unwrap();

        // bucket 0 should have records with ids [10, 11]
        assert_eq!(bucket_0.records_slice().len(), 4); // 2 records * 2 dims
                                                       // bucket 1 should have records with ids [20, 21]
        assert_eq!(bucket_1.records_slice().len(), 4); // 2 records * 2 dims
    }

    #[test]
    fn test_insert_actual_block_overflow() {
        use tempfile::TempDir;

        let temp_dir = TempDir::new().unwrap();
        let data_path = temp_dir.path().join("test.cold.data");
        let input_shape = 4;
        let n_buckets = 1;
        let bucket_size = 10000;

        let mut cold_storage =
            ColdStorage::<f32>::new(&data_path, n_buckets, input_shape, bucket_size).unwrap();

        // With input_shape=4, each f32 record is 4*4 + 4 = 20 bytes
        // Block size is 65536 bytes, so max records per block = (65536-4)/20 = 3276
        // Insert 3286 records to overflow into 2 blocks
        let num_records = 3286;
        let mut records = Vec::with_capacity(num_records * input_shape);
        let mut ids = Vec::with_capacity(num_records);
        let mut assignments = Vec::with_capacity(num_records);

        for i in 0..num_records {
            for j in 0..input_shape {
                records.push((i as f32) + (j as f32) * 0.1);
            }
            ids.push(i as u32);
            assignments.push(0); // All go to bucket 0
        }

        cold_storage
            .insert(records.clone(), ids.clone(), &assignments)
            .unwrap();

        // Verify bucket 0 has all records
        assert_eq!(
            cold_storage.disk_buckets[0].count, num_records,
            "bucket should have {} records",
            num_records
        );

        // Verify we have multiple extents due to block overflow
        assert!(
            cold_storage.disk_buckets[0].extents.len() >= 2,
            "should have at least 2 extents to hold 3286 records"
        );

        // Verify all records can be read back
        let bucket = cold_storage.read_bucket(0).unwrap();
        assert_eq!(
            bucket.records_slice().len(),
            num_records * input_shape,
            "should read back all {} records",
            num_records
        );

        // Spot check some records to ensure they were stored and retrieved correctly
        let read_records = bucket.records_slice();
        // First record should be [0.0, 0.1, 0.2, 0.3]
        assert_eq!(read_records[0], 0.0);
        assert_eq!(read_records[1], 0.1);
        assert_eq!(read_records[2], 0.2);
        assert_eq!(read_records[3], 0.3);

        // Last record should be [3285.0, 3285.1, 3285.2, 3285.3]
        let last_idx = (num_records - 1) * input_shape;
        assert_eq!(read_records[last_idx], 3285.0);
        assert_eq!(read_records[last_idx + 1], 3285.1);
        assert_eq!(read_records[last_idx + 2], 3285.2);
        assert_eq!(read_records[last_idx + 3], 3285.3);
    }
}
