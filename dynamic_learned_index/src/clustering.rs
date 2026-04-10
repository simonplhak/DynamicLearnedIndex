use log::debug;
use measure_time_macro::log_time;
use ndarray::{Array2, ArrayView};

use crate::structs::LabelMethod;

#[log_time]
pub(crate) fn compute_labels(
    data: &[f32],
    label_method: &LabelMethod,
    k: usize,
    input_shape: usize,
    max_iters: usize,
) -> Vec<i64> {
    debug_assert!(!data.is_empty());
    let data_len = data.len() / input_shape;
    assert!(data_len * input_shape == data.len());
    debug!(data_len = data_len, k = k; "clustering:compute_labels");
    match label_method {
        LabelMethod::KMeans => k_means_clustering(data, input_shape, k, max_iters),
        LabelMethod::SphericalKMeans => {
            k_means_clustering_spherical(data, input_shape, k, max_iters)
        }
    }
}

fn k_means_clustering(data: &[f32], input_shape: usize, k: usize, max_iters: usize) -> Vec<i64> {
    let count = data.len() / input_shape;
    assert!(count * input_shape == data.len());
    assert!(k > 0);
    assert!(k <= count);
    assert!(max_iters > 0);
    let mut kmeans = kentro::KMeans::new(k)
        .with_iterations(max_iters)
        .with_euclidean(true);

    // Convert the Vec<f32> to Array2 with shape (count, input_shape)
    let data = ArrayView::from_shape((count, input_shape), data)
        .expect("Failed to reshape data into Array2");
    let clusters = kmeans.train(data.view(), None).unwrap();
    let mut result = vec![0; count];
    clusters
        .into_iter()
        .enumerate()
        .for_each(|(cluster, query_ids)| {
            query_ids.into_iter().for_each(|query_id| {
                assert!(query_id < count);
                result[query_id] = cluster as i64;
            });
        });
    // debug!(error = result.distsum ;"kmeans:metrics");
    assert!(result.len() == count);
    result
}

fn k_means_clustering_spherical(
    data: &[f32],
    input_shape: usize,
    k: usize,
    max_iters: usize,
) -> Vec<i64> {
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
                result[query_id] = cluster as i64;
            });
        });
    // debug!(error = result.distsum ;"kmeans:metrics");
    assert!(result.len() == count);
    result
}
