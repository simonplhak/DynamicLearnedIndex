use serde::{Deserialize, Serialize};

use crate::types::{ArrayNumType, ArraySlice};
use simsimd::SpatialSimilarity;

#[derive(Default, Deserialize, Serialize, Debug, Clone)]
pub enum DistanceFn {
    #[default]
    #[serde(rename = "l2")]
    L2,
    #[serde(rename = "dot")]
    Dot,
}

impl DistanceFn {
    #[inline]
    pub(crate) fn distance(&self, a: &ArraySlice, b: &ArraySlice) -> ArrayNumType {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");
        match self {
            DistanceFn::L2 => f32::l2(a, b).unwrap() as f32,
            DistanceFn::Dot => f32::dot(a, b).unwrap() as f32,
        }
    }

    #[inline]
    pub(crate) fn cmp(&self, a: &f32, b: &f32) -> std::cmp::Ordering {
        match self {
            DistanceFn::L2 => a.total_cmp(b),
            DistanceFn::Dot => b.total_cmp(a), // Higher is better for inner product
        }
    }
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
