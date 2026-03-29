use std::{
    collections::HashMap,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use crate::{ArrayNumType, DliError, DliResult, Id};
use serde::{Deserialize, Serialize};

/// Metadata for a cold bucket stored on disk
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ColdBucketMetadata {
    pub bucket_id: usize,
    pub level_id: usize,
    pub count: usize,
    pub records_size: usize,
    pub compressed_size: usize,
    pub compressed_path: PathBuf,
}

/// Index mapping bucket locations (hot or cold)
#[derive(Debug, Clone)]
pub enum BucketLocation {
    Hot,
    Cold(ColdBucketMetadata),
}

/// Manager for cold storage operations
pub struct ColdStorageManager {
    cold_dir: PathBuf,
    /// Map from (level_id, bucket_id) -> ColdBucketMetadata
    cold_index: HashMap<(usize, usize), ColdBucketMetadata>,
    /// Hot memory limit in bytes
    hot_memory_limit: usize,
    /// Current hot memory usage in bytes
    current_hot_memory: usize,
}

impl ColdStorageManager {
    /// Create a new cold storage manager
    pub fn new(working_dir: &Path, hot_memory_limit: usize) -> DliResult<Self> {
        let cold_dir = working_dir.join("cold");
        fs::create_dir_all(&cold_dir).map_err(|e| {
            DliError::IoError(format!("Failed to create cold storage directory: {}", e))
        })?;

        Ok(Self {
            cold_dir,
            cold_index: HashMap::new(),
            hot_memory_limit,
            current_hot_memory: 0,
        })
    }

    /// Store a bucket to cold storage with Zstd compression
    pub fn demote_bucket(
        &mut self,
        level_id: usize,
        bucket_id: usize,
        records: &[ArrayNumType],
        ids: &[Id],
        input_shape: usize,
    ) -> DliResult<ColdBucketMetadata> {
        // Serialize bucket data
        let mut buffer = Vec::new();

        // Write record count and input shape
        let count = ids.len() as u32;
        buffer.extend_from_slice(&count.to_le_bytes());
        buffer.extend_from_slice(&(input_shape as u32).to_le_bytes());

        // Write records as bytes
        let records_bytes: &[u8] = bytemuck::cast_slice(records);
        buffer.extend_from_slice(records_bytes);

        // Write IDs as bytes
        let ids_bytes: &[u8] = bytemuck::cast_slice(ids);
        buffer.extend_from_slice(ids_bytes);

        let uncompressed_size = buffer.len();

        // Compress with Zstd
        let mut compressed = Vec::new();
        let mut encoder = zstd::stream::Encoder::new(&mut compressed, 3)
            .map_err(|e| DliError::SerializationError(format!("Zstd encoding failed: {}", e)))?;
        encoder.write_all(&buffer).map_err(|e| {
            DliError::SerializationError(format!("Failed to compress bucket: {}", e))
        })?;
        encoder
            .finish()
            .map_err(|e| DliError::SerializationError(format!("Zstd finish failed: {}", e)))?;

        let compressed_size = compressed.len();

        // Write compressed data to disk
        let compressed_path = self
            .cold_dir
            .join(format!("level_{}_bucket_{}.zst", level_id, bucket_id));
        let mut file = File::create(&compressed_path)
            .map_err(|e| DliError::IoError(format!("Failed to create cold bucket file: {}", e)))?;
        file.write_all(&compressed)
            .map_err(|e| DliError::IoError(format!("Failed to write cold bucket data: {}", e)))?;

        let metadata = ColdBucketMetadata {
            bucket_id,
            level_id,
            count: ids.len(),
            records_size: uncompressed_size,
            compressed_size,
            compressed_path,
        };

        self.cold_index
            .insert((level_id, bucket_id), metadata.clone());

        Ok(metadata)
    }

    /// Load a bucket from cold storage
    pub fn promote_bucket(
        &mut self,
        level_id: usize,
        bucket_id: usize,
        input_shape: usize,
    ) -> DliResult<(Vec<ArrayNumType>, Vec<Id>)> {
        let metadata = self
            .cold_index
            .get(&(level_id, bucket_id))
            .ok_or_else(|| {
                DliError::NotFound(format!(
                    "Bucket ({}, {}) not found in cold storage",
                    level_id, bucket_id
                ))
            })?
            .clone();

        // Read compressed data
        let mut file = File::open(&metadata.compressed_path)
            .map_err(|e| DliError::IoError(format!("Failed to open cold bucket file: {}", e)))?;
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed)
            .map_err(|e| DliError::IoError(format!("Failed to read cold bucket data: {}", e)))?;

        // Decompress
        let decompressed = zstd::decode_all(compressed.as_slice())
            .map_err(|e| DliError::SerializationError(format!("Zstd decoding failed: {}", e)))?;

        // Parse decompressed data
        if decompressed.len() < 8 {
            return Err(DliError::SerializationError(
                "Compressed bucket data too small".to_string(),
            ));
        }

        let count = u32::from_le_bytes([
            decompressed[0],
            decompressed[1],
            decompressed[2],
            decompressed[3],
        ]) as usize;
        let parsed_input_shape = u32::from_le_bytes([
            decompressed[4],
            decompressed[5],
            decompressed[6],
            decompressed[7],
        ]) as usize;

        if parsed_input_shape != input_shape {
            return Err(DliError::SerializationError(
                "Input shape mismatch in cold bucket".to_string(),
            ));
        }

        let mut offset = 8;
        let records_size = count * input_shape * std::mem::size_of::<ArrayNumType>();
        if offset + records_size > decompressed.len() {
            return Err(DliError::SerializationError(
                "Invalid bucket data: insufficient records".to_string(),
            ));
        }

        let records_bytes = &decompressed[offset..offset + records_size];
        let records: Vec<ArrayNumType> = bytemuck::cast_slice(records_bytes).to_vec();

        offset += records_size;
        let ids_size = count * std::mem::size_of::<Id>();
        if offset + ids_size > decompressed.len() {
            return Err(DliError::SerializationError(
                "Invalid bucket data: insufficient IDs".to_string(),
            ));
        }

        let ids_bytes = &decompressed[offset..offset + ids_size];
        let ids: Vec<Id> = bytemuck::cast_slice(ids_bytes).to_vec();

        // Remove from cold index
        self.cold_index.remove(&(level_id, bucket_id));

        Ok((records, ids))
    }

    /// Check if a bucket is in cold storage
    pub fn is_cold(&self, level_id: usize, bucket_id: usize) -> bool {
        self.cold_index.contains_key(&(level_id, bucket_id))
    }

    /// Get all cold buckets
    pub fn cold_buckets(&self) -> Vec<&ColdBucketMetadata> {
        self.cold_index.values().collect()
    }

    /// Update hot memory usage
    pub fn set_hot_memory_usage(&mut self, usage: usize) {
        self.current_hot_memory = usage;
    }

    /// Check if we exceed memory limit
    pub fn exceeds_memory_limit(&self) -> bool {
        self.current_hot_memory > self.hot_memory_limit
    }

    /// Get memory usage ratio (0.0 to 1.0+)
    pub fn memory_ratio(&self) -> f64 {
        self.current_hot_memory as f64 / self.hot_memory_limit as f64
    }

    /// Delete cold bucket file from disk
    pub fn delete_cold_bucket(&mut self, level_id: usize, bucket_id: usize) -> DliResult<()> {
        if let Some(metadata) = self.cold_index.remove(&(level_id, bucket_id)) {
            fs::remove_file(&metadata.compressed_path).map_err(|e| {
                DliError::IoError(format!("Failed to delete cold bucket file: {}", e))
            })?;
        }
        Ok(())
    }

    /// Persist cold storage index to disk
    pub fn save_index(&self, working_dir: &Path) -> DliResult<()> {
        let index_path = working_dir.join("cold_index.json");
        let index_data: Vec<_> = self.cold_index.values().collect();
        let json = serde_json::to_string(&index_data)
            .map_err(|e| DliError::SerializationError(e.to_string()))?;
        fs::write(&index_path, json)
            .map_err(|e| DliError::IoError(format!("Failed to save cold storage index: {}", e)))?;
        Ok(())
    }

    /// Load cold storage index from disk
    pub fn load_index(&mut self, working_dir: &Path) -> DliResult<()> {
        let index_path = working_dir.join("cold_index.json");
        if !index_path.exists() {
            return Ok(()); // No cold storage index yet
        }

        let json = fs::read_to_string(&index_path)
            .map_err(|e| DliError::IoError(format!("Failed to read cold storage index: {}", e)))?;
        let metadatas: Vec<ColdBucketMetadata> =
            serde_json::from_str(&json).map_err(|e| DliError::SerializationError(e.to_string()))?;

        for metadata in metadatas {
            self.cold_index
                .insert((metadata.level_id, metadata.bucket_id), metadata);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_storage_manager() -> DliResult<()> {
        let temp_dir = tempfile::tempdir().unwrap();
        let mut manager = ColdStorageManager::new(temp_dir.path(), 1024 * 1024)?;

        // Test demotion
        let records = vec![1.0, 2.0, 3.0, 4.0];
        let ids = vec![1, 2];
        let metadata = manager.demote_bucket(0, 0, &records, &ids, 2)?;
        assert_eq!(metadata.count, 2);
        assert!(manager.is_cold(0, 0));

        // Test promotion
        let (recovered_records, recovered_ids) = manager.promote_bucket(0, 0, 2)?;
        assert_eq!(recovered_records, records);
        assert_eq!(recovered_ids, ids);
        assert!(!manager.is_cold(0, 0));

        Ok(())
    }
}
