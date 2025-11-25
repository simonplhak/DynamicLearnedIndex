use std::{collections::HashMap, path::PathBuf};

use serde::{Deserialize, Serialize};

use crate::{
    constants::{DEFAULT_ARITY, DEFAULT_BUFFER_SIZE, DEFAULT_INPUT_SHAPE, DEFAULT_SEARCH_K},
    CompactionStrategy, LevelIndexConfig, ModelDevice, SearchStrategy,
};

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
            buffer_size: DEFAULT_BUFFER_SIZE,
            input_shape: DEFAULT_INPUT_SHAPE,
            arity: DEFAULT_ARITY,
            device: Default::default(),
            distance_fn: Default::default(),
            delete_method: Default::default(),
        }
    }
}

#[derive(Default, Deserialize, Serialize, Debug, Clone)]
pub enum DistanceFn {
    #[serde(rename = "l2")]
    L2,
    #[default]
    #[serde(rename = "dot")]
    Dot,
}

impl From<DistanceFn> for LabelMethod {
    fn from(val: DistanceFn) -> Self {
        match val {
            DistanceFn::L2 => LabelMethod::KMeans,
            DistanceFn::Dot => LabelMethod::SphericalKMeans,
        }
    }
}

impl From<&str> for DistanceFn {
    fn from(val: &str) -> Self {
        match val {
            "l2" => DistanceFn::L2,
            "dot" => DistanceFn::Dot,
            _ => panic!("Unknown distance function: {val}"),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LabelMethod {
    KMeans,
    SphericalKMeans,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum DeleteMethod {
    #[default]
    #[serde(rename = "oid_to_bucket")]
    OidToBucket,
}

impl From<&str> for DeleteMethod {
    fn from(val: &str) -> Self {
        match val {
            "oid_to_bucket" => DeleteMethod::OidToBucket,
            _ => panic!("Unknown delete method: {val}"),
        }
    }
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
            k: DEFAULT_SEARCH_K,
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

pub struct DeleteStatistics {
    pub affected_level: Option<usize>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskBucket {
    pub records_offset: u64,
    pub ids_offset: u64,
    pub count: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskBuffer {
    pub records_path: PathBuf,
    pub ids_path: PathBuf,
    pub count: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskLevelIndex {
    pub weights_path: PathBuf,
    pub records_path: PathBuf,
    pub ids_path: PathBuf,
    pub buckets: Vec<DiskBucket>,
    pub config: LevelIndexConfig,
}
