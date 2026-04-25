use serde::{Deserialize, Serialize};
use std::{borrow::Cow, path::PathBuf};

use crate::{
    constants::{
        DEFAULT_ARITY, DEFAULT_BUCKET_SIZE, DEFAULT_BUFFER_SIZE, DEFAULT_INPUT_SHAPE,
        DEFAULT_SEARCH_K, DEFAULT_SEARCH_N_CANDIDATES,
    },
    Id, ModelConfig, ModelDevice,
};
use half::f16;

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

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub enum RebuildStrategy {
    #[default]
    #[serde(rename = "no_rebuild")]
    NoRebuild,
    #[serde(rename = "basic_rebuild")]
    BasicRebuild,
    #[serde(rename = "greedy_rebuild")]
    GreedyRebuild,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", content = "rebuild_strategy")]
pub enum CompactionStrategyConfig {
    #[serde(rename = "bentley_saxe")]
    BentleySaxe(RebuildStrategy),
}

impl Default for CompactionStrategyConfig {
    fn default() -> Self {
        CompactionStrategyConfig::BentleySaxe(Default::default())
    }
}

impl From<&str> for CompactionStrategyConfig {
    fn from(val: &str) -> Self {
        match val {
            "bentley_saxe:no_rebuild" => {
                CompactionStrategyConfig::BentleySaxe(RebuildStrategy::NoRebuild)
            }
            "bentley_saxe:basic_rebuild" => {
                CompactionStrategyConfig::BentleySaxe(RebuildStrategy::BasicRebuild)
            }
            "bentley_saxe:greedy_rebuild" => {
                CompactionStrategyConfig::BentleySaxe(RebuildStrategy::GreedyRebuild)
            }
            _ => panic!("Unknown compaction strategy: {val}"),
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct IndexConfig {
    pub compaction_strategy: CompactionStrategyConfig,
    pub levels: LevelIndexConfig,
    pub buffer_size: usize,
    pub input_shape: usize,
    pub arity: usize,
    pub device: ModelDevice,
    pub distance_fn: DistanceFn,
    pub delete_method: DeleteMethod,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            compaction_strategy: Default::default(),
            levels: Default::default(),
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

#[derive(Debug, Copy, Clone)]
pub enum SearchStrategy {
    Base(usize), // todo rename to KnnDriven
    ModelDriven(usize),
}

impl Default for SearchStrategy {
    fn default() -> Self {
        SearchStrategy::ModelDriven(DEFAULT_SEARCH_N_CANDIDATES)
    }
}

#[cfg(any(feature = "candle", feature = "mix"))]
pub trait CandleFloat: candle_core::WithDType {}
#[cfg(not(any(feature = "candle", feature = "mix")))]
pub trait CandleFloat {}

// Blanket impls
#[cfg(any(feature = "candle", feature = "mix"))]
impl<T: candle_core::WithDType> CandleFloat for T {}
#[cfg(not(any(feature = "candle", feature = "mix")))]
impl<T> CandleFloat for T {}

#[cfg(any(feature = "tch", feature = "mix"))]
pub trait TchFloat: tch::kind::Element {}
#[cfg(not(any(feature = "tch", feature = "mix")))]
pub trait TchFloat {}

#[cfg(any(feature = "tch", feature = "mix"))]
impl<T: tch::kind::Element> TchFloat for T {}
#[cfg(not(any(feature = "tch", feature = "mix")))]
impl<T> TchFloat for T {}

pub trait FloatElement: bytemuck::Pod + Default + CandleFloat + TchFloat {
    fn to_f32_slice(slice: &[Self]) -> Cow<'_, [f32]>;

    #[cfg(any(feature = "candle", feature = "mix"))]
    fn to_candle_dtype() -> candle_core::DType;

    fn zero() -> Self;
}

impl FloatElement for f32 {
    fn to_f32_slice(slice: &[Self]) -> Cow<'_, [f32]> {
        Cow::Borrowed(slice)
    }

    #[cfg(any(feature = "candle", feature = "mix"))]
    fn to_candle_dtype() -> candle_core::DType {
        candle_core::DType::F32
    }

    fn zero() -> Self {
        0.0_f32
    }
}

impl FloatElement for f16 {
    fn to_f32_slice(slice: &[Self]) -> Cow<'_, [f32]> {
        // HalfFloatVecExt::to_f32_slice is not zero-copy, it allocates a new Vec<f32> and converts each f16 to f32
        let mut v = Vec::with_capacity(slice.len());
        v.extend(slice.iter().map(|x| x.to_f32()));
        Cow::Owned(v)
    }

    #[cfg(any(feature = "candle", feature = "mix"))]
    fn to_candle_dtype() -> candle_core::DType {
        candle_core::DType::F16
    }

    fn zero() -> Self {
        f16::from_f32(0.0)
    }
}

pub struct Records2Visit<'a, F: FloatElement> {
    pub records: Vec<&'a [F]>,
    pub ids: Vec<Id>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskBucket {
    pub records_offset: u64,
    pub ids_offset: u64,
    pub count: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskLevelIndex {
    pub records_path: PathBuf,
    pub ids_path: PathBuf,
    pub buckets: Vec<DiskBucket>,
    pub config: LevelIndexConfig,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskBuffer {
    pub records_path: PathBuf,
    pub ids_path: PathBuf,
    pub data: DiskBucket,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DiskIndex {
    pub compaction_strategy: CompactionStrategyConfig,
    pub levels_config: LevelIndexConfig,
    pub buffer_size: usize,
    pub input_shape: usize,
    pub arity: usize,
    pub distance_fn: DistanceFn,
    pub delete_method: DeleteMethod,
    pub levels: Vec<DiskLevelIndex>,
    pub disk_buffer: DiskBuffer,
}
