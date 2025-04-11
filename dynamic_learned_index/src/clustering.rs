use log::info;
use rand::seq::IteratorRandom;
use serde::{Deserialize, Serialize};
use tch::Tensor;

use crate::Id;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub(crate) enum LabelMethod {
    #[serde(rename = "knn")]
    Knn(KMeans),
}

impl Default for LabelMethod {
    fn default() -> Self {
        LabelMethod::Knn(KMeans { max_iters: 10 })
    }
}

pub(crate) fn compute_labels(
    data: Vec<Tensor>,
    ids: Vec<Id>,
    label_method: &LabelMethod,
    k: i64,
) -> (Vec<Vec<Tensor>>, Vec<Vec<Id>>) {
    debug_assert!(!data.is_empty());
    info!(
        data_len = data.len(),
        k = k,
        label_method = format!("{:?}", label_method); "clustering:compute_labels"
    );
    match label_method {
        LabelMethod::Knn(kmeans) => k_means_clustering(data, ids, k as usize, kmeans.max_iters),
    }
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub(crate) struct KMeans {
    max_iters: usize,
}

fn cmp_vecs(point: &Tensor, a: &Tensor, b: &Tensor) -> std::cmp::Ordering {
    let dist_a = (point - a).square().sum(tch::Kind::Float).double_value(&[]);
    let dist_b = (point - b).square().sum(tch::Kind::Float).double_value(&[]);
    dist_a
        .partial_cmp(&dist_b)
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn k_means_clustering(
    data: Vec<Tensor>,
    ids: Vec<Id>,
    k: usize,
    max_iters: usize,
) -> (Vec<Vec<Tensor>>, Vec<Vec<Id>>) {
    assert!(k > 0, "k must be greater than 0");
    assert!(
        data.len() == ids.len(),
        "data and ids must have the same length"
    );
    assert!(data.len() > k, "data must have more elements than k");
    assert!(max_iters > 0, "max_iters must be greater than 0");

    let mut rng = rand::rng();
    let mut centroids: Vec<Tensor> = data
        .iter()
        .choose_multiple(&mut rng, k)
        .into_iter()
        .map(|t| t.copy())
        .collect();
    // let mut data = split_into_k_chunks(data, k);

    for _ in 0..max_iters {
        let assignments: Vec<usize> = data
            .iter()
            // .flatten()
            .map(|point| {
                assert!(centroids.len() == k);
                assert!(!centroids.is_empty());
                centroids
                    .iter()
                    .enumerate()
                    .min_by(|(_, a), (_, b)| cmp_vecs(point, a, b))
                    .map(|(i, _)| i)
                    .unwrap()
            })
            .collect();

        let mut new_centroids = Vec::with_capacity(k);
        for cluster in 0..k {
            let members = data
                .iter()
                .zip(&assignments)
                .filter(|(_, &c)| c == cluster)
                .map(|(p, _)| p)
                .collect::<Vec<_>>();

            let centroid = if members.is_empty() {
                centroids[cluster].copy()
            } else {
                Tensor::stack(&members, 0).mean_dim(0, false, members[0].kind())
            };
            new_centroids.push(centroid);
        }

        if centroids
            .iter()
            .zip(&new_centroids)
            .all(|(a, b)| (a - b).abs().max().double_value(&[]) < 1e-10)
        {
            break;
        }
        centroids = new_centroids;
    }
    let mut capacities = vec![0; k];
    let final_assignments: Vec<usize> = data
        .iter()
        .map(|point| {
            let cluster_idx = centroids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| cmp_vecs(point, a, b))
                .map(|(i, _)| i)
                .unwrap();
            capacities[cluster_idx] += 1;
            cluster_idx
        })
        .collect();

    let mut tensor_clusters = capacities
        .iter()
        .map(|capacity| Vec::with_capacity(*capacity))
        .collect::<Vec<_>>();
    let mut id_clusters = capacities
        .iter()
        .map(|capacity| Vec::with_capacity(*capacity))
        .collect::<Vec<_>>();

    for ((tensor, id), cluster_idx) in data.into_iter().zip(ids).zip(final_assignments) {
        tensor_clusters[cluster_idx].push(tensor);
        id_clusters[cluster_idx].push(id);
    }

    (tensor_clusters, id_clusters)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_k_means_clustering_basic() {
        let data = vec![
            Tensor::from_slice(&[1.0, 2.0]),
            Tensor::from_slice(&[1.1, 2.1]),
            Tensor::from_slice(&[10.0, 10.0]),
            Tensor::from_slice(&[10.1, 10.1]),
        ];
        let ids = vec![1, 2, 3, 4];
        let k = 2;
        let max_iters = 10;

        let (tensor_clusters, id_clusters) = k_means_clustering(data, ids, k, max_iters);

        assert_eq!(tensor_clusters.len(), k);
        assert_eq!(id_clusters.len(), k);

        // Ensure all data points are assigned to a cluster
        let total_points: usize = tensor_clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total_points, 4);

        let total_ids: usize = id_clusters.iter().map(|c| c.len()).sum();
        assert_eq!(total_ids, 4);
        assert_eq!(tensor_clusters[0].len(), 2);
        assert_eq!(tensor_clusters[1].len(), 2);
        assert_eq!(id_clusters[0].len(), 2);
        assert_eq!(id_clusters[1].len(), 2);
        assert_ne!(tensor_clusters[0], tensor_clusters[1]);
        assert_ne!(id_clusters[0], id_clusters[1]);
        let converges = matches!(id_clusters[0].as_slice(), [1, 2] | [3, 4]);
        assert!(converges);
        let converges = matches!(id_clusters[1].as_slice(), [1, 2] | [3, 4]);
        assert!(converges);
    }
}
