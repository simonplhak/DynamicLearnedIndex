#![feature(test)]

extern crate test;
use ndarray::Array2;
use rand::Rng;
use test::Bencher;

const VECTOR_DIM: usize = 768;
const SAMPLE_SIZE: usize = 10_000;
const NUM_CLUSTERS: usize = 81;
const NUM_ITERS: usize = 1;
const LANES: usize = 8; // Assuming 256-bit SIMD with f32 (32 bytes / 4 bytes per f32)

fn generate_random_data(size: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..size * dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}

fn kmeans(data: &Vec<f32>, input_shape: usize, k: usize, max_iters: usize) -> Vec<i64> {
    let count = data.len() / input_shape;
    let kmean: kmeans::KMeans<_, { LANES }, _> =
        kmeans::KMeans::new(data, count, input_shape, kmeans::EuclideanDistance);
    let result = kmean.kmeans_lloyd(
        k,
        max_iters,
        kmeans::KMeans::init_kmeanplusplus,
        &kmeans::KMeansConfig::default(),
    );
    result.assignments.into_iter().map(|x| x as i64).collect()
}

#[bench]
fn bench_kmeans(b: &mut Bencher) {
    let data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);

    b.iter(|| {
        kmeans(&data, VECTOR_DIM, NUM_CLUSTERS, NUM_ITERS);
    });
}

fn kentro(data: &[f32], input_shape: usize, k: usize, max_iters: usize) -> Vec<i64> {
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

#[bench]
fn bench_kentro(b: &mut Bencher) {
    let data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);

    b.iter(|| {
        kentro(&data, VECTOR_DIM, NUM_CLUSTERS, NUM_ITERS);
    });
}
