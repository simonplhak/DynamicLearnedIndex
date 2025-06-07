use log::info;
use serde::{Deserialize, Serialize};

use crate::{constants, types::Array, Id};

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub(crate) enum LabelMethod {
    #[serde(rename = "knn")]
    Knn(KMeansConfig),
}

impl Default for LabelMethod {
    fn default() -> Self {
        LabelMethod::Knn(KMeansConfig { max_iters: 10 })
    }
}

pub(crate) fn compute_labels(
    data: Vec<Array>,
    ids: Vec<Id>,
    label_method: &LabelMethod,
    k: usize,
    input_shape: usize,
) -> (Vec<Vec<Array>>, Vec<Vec<Id>>) {
    debug_assert!(!data.is_empty());
    info!(
        data_len = data.len(),
        k = k,
        label_method = format!("{:?}", label_method); "clustering:compute_labels"
    );
    match label_method {
        LabelMethod::Knn(kmeans) => {
            // todo get_data method from index should return data in the right shape
            let data = data.into_iter().flatten().collect::<Vec<_>>();
            k_means_clustering_new(data, input_shape, ids, k, kmeans.max_iters)
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub(crate) struct KMeansConfig {
    max_iters: usize,
}

fn k_means_clustering_new(
    data: Vec<f32>,
    input_shape: usize,
    ids: Vec<Id>,
    k: usize,
    max_iters: usize,
) -> (Vec<Vec<Array>>, Vec<Vec<Id>>) {
    let count = data.len() / input_shape;
    assert!(count * input_shape == data.len());
    assert!(count == ids.len());
    let kmean: kmeans::KMeans<_, { constants::LANES }, _> =
        kmeans::KMeans::new(&data, count, input_shape, kmeans::EuclideanDistance);
    let result = kmean.kmeans_lloyd(
        k,
        max_iters,
        kmeans::KMeans::init_kmeanplusplus,
        &kmeans::KMeansConfig::default(),
    );
    info!(error = result.distsum ;"kmeans:metrics");
    assert!(result.assignments.len() == count);
    let mut cluster_data = vec![Vec::new(); k];
    let mut cluster_ids = vec![Vec::new(); k];
    let mut data = data;
    result
        .assignments
        .iter()
        .zip(ids.iter())
        .enumerate()
        .rev()
        .for_each(|(i, (assigment, id))| {
            let v = data.drain(i * input_shape..).collect::<Vec<_>>();
            assert!(v.len() == input_shape);
            cluster_data[*assigment].push(v);
            cluster_ids[*assigment].push(*id);
        });
    (cluster_data, cluster_ids)
}
