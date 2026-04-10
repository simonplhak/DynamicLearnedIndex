#![feature(portable_simd)]
#![feature(test)]

extern crate test;
#[cfg(feature = "kmeans")]
use kmeans::*;
use rand::Rng;

const _VECTOR_DIM: usize = 768;
const _SAMPLE_SIZE: usize = 10_000;
const _NUM_CLUSTERS: usize = 81;
const _NUM_ITERS: usize = 1;
#[cfg(feature = "kmeans")]
const LANES: usize = 8; // Assuming 256-bit SIMD with f32 (32 bytes / 4 bytes per f32)

fn _generate_random_data(size: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..size * dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}

#[cfg(feature = "kmeans")]
fn kmeans(data: &Vec<f32>, input_shape: usize, k: usize, max_iters: usize) -> Vec<i64> {
    let count = data.len() / input_shape;
    let kmean: KMeans<_, { LANES }, _> = KMeans::new(data, count, input_shape, EuclideanDistance);
    let result = kmean.kmeans_lloyd(
        k,
        max_iters,
        KMeans::init_kmeanplusplus,
        &KMeansConfig::default(),
    );
    result.assignments.into_iter().map(|x| x as i64).collect()
}

#[cfg(feature = "kmeans")]
#[bench]
fn bench_kmeans(b: &mut test::Bencher) {
    let data = _generate_random_data(_SAMPLE_SIZE, _VECTOR_DIM);

    b.iter(|| {
        kmeans(&data, _VECTOR_DIM, _NUM_CLUSTERS, _NUM_ITERS);
    });
}

#[cfg(feature = "kentro")]
fn kentro(data: &[f32], input_shape: usize, k: usize, max_iters: usize) -> Vec<i64> {
    use ndarray::Array2;
    let count = data.len() / input_shape;
    let mut kmeans = kentro::KMeans::new(k)
        .with_iterations(max_iters)
        .with_euclidean(true);

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
    result
}

#[cfg(feature = "kentro")]
#[bench]
fn bench_kentro(b: &mut test::Bencher) {
    let data = _generate_random_data(_SAMPLE_SIZE, _VECTOR_DIM);

    b.iter(|| {
        kentro(&data, _VECTOR_DIM, _NUM_CLUSTERS, _NUM_ITERS);
    });
}
