use log::debug;
use serde::{Deserialize, Serialize};

use crate::constants;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", content = "value")]
pub enum LabelMethod {
    #[serde(rename = "knn")]
    Knn(KMeansConfig),
}

impl Default for LabelMethod {
    fn default() -> Self {
        LabelMethod::Knn(KMeansConfig { max_iters: 10 })
    }
}

pub(crate) fn compute_labels(
    data: &Vec<f32>,
    label_method: &LabelMethod,
    k: usize,
    input_shape: usize,
) -> Vec<i32> {
    debug_assert!(!data.is_empty());
    let data_len = data.len() / input_shape;
    assert!(data_len * input_shape == data.len());
    debug!(
        data_len=data_len,
        k=k,
        label_method = format!("{:?}", label_method); "clustering:compute_labels"
    );
    match label_method {
        LabelMethod::Knn(kmeans) => k_means_clustering_new(data, input_shape, k, kmeans.max_iters),
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct KMeansConfig {
    max_iters: usize,
}

impl Default for KMeansConfig {
    fn default() -> Self {
        KMeansConfig { max_iters: 10 }
    }
}

fn k_means_clustering_new(
    data: &Vec<f32>,
    input_shape: usize,
    k: usize,
    max_iters: usize,
) -> Vec<i32> {
    let count = data.len() / input_shape;
    assert!(count * input_shape == data.len());
    assert!(k > 0);
    assert!(k <= count);
    assert!(max_iters > 0);
    let kmean: kmeans::KMeans<_, { constants::LANES }, _> =
        kmeans::KMeans::new(data, count, input_shape, kmeans::EuclideanDistance);
    let result = kmean.kmeans_lloyd(
        k,
        max_iters,
        kmeans::KMeans::init_kmeanplusplus,
        &kmeans::KMeansConfig::default(),
    );
    debug!(error = result.distsum ;"kmeans:metrics");
    assert!(result.assignments.len() == count);
    result.assignments.into_iter().map(|x| x as i32).collect()
}
