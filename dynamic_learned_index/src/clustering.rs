use log::debug;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::constants;

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type", content = "value")]
pub enum LabelMethod {
    #[serde(rename = "knn")]
    Knn(KMeansConfig),
    #[serde(rename = "spherical_knn")]
    SphericalKnn(KMeansConfig),
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
    debug!(data_len = data_len, k = k; "clustering:compute_labels");
    match label_method {
        LabelMethod::Knn(kmeans) => k_means_clustering_new(data, input_shape, k, kmeans.max_iters),
        LabelMethod::SphericalKnn(kmeans) => {
            k_means_clustering_spherical(data, input_shape, k, kmeans.max_iters)
        }
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

fn k_means_clustering_spherical(
    data: &[f32],
    input_shape: usize,
    k: usize,
    max_iters: usize,
) -> Vec<i32> {
    let count = data.len() / input_shape;
    assert!(count * input_shape == data.len());
    assert!(k > 0);
    assert!(k <= count);
    assert!(max_iters > 0);
    let mut kmeans = kentro::KMeans::new(k).with_iterations(max_iters);

    // Convert the Vec<f32> to Array2 with shape (count, input_shape)
    let data = Array2::from_shape_vec((count, input_shape), data.to_vec())
        .expect("Failed to reshape data into Array2");
    let clusters = kmeans.train(data.view(), None).unwrap();
    let mut result = vec![0; count];
    clusters
        .into_iter()
        .enumerate()
        .for_each(|(cluster, query_ids)| {
            query_ids.into_iter().for_each(|query_id| {
                assert!(query_id < count);
                result[query_id] = cluster as i32;
            });
        });
    // debug!(error = result.distsum ;"kmeans:metrics");
    assert!(result.len() == count);
    result
}
