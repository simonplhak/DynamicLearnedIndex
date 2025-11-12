use serde::{Deserialize, Serialize};

use crate::{constants::DEFAULT_SEARCH_K, SearchStrategy};

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
