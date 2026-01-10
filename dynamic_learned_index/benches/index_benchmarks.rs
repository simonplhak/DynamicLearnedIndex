use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{seq::SliceRandom, SeedableRng};
use std::{hint::black_box, path::PathBuf};

use dynamic_learned_index::{Array, IndexBuilder, SearchStrategy};

const QUERIES_DATASET_PATH: &str = "data/k300/queries.h5";
const QUERIES_DATASET_NAME: &str = "emb";
const INDEX_DUMP_DIR: &str = "index_dump";
const MAX_K: usize = 10;
const SEARCH_STRATEGY: SearchStrategy = SearchStrategy::ModelDriven(10_000);
const SEED: u64 = 42;

fn load_h5(path: &PathBuf, dataset_name: &str) -> Vec<Array> {
    let emb = hdf5::File::open(path)
        .expect("Failed to open HDF5 file")
        .dataset(dataset_name)
        .expect("Failed to open dataset");
    let data = emb.read_2d::<f32>().expect("Failed to read dataset");
    data.outer_iter().map(|row| row.to_vec()).collect()
}

fn index_search_parametrized_benchmark(c: &mut Criterion) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let queries_path = root.join(QUERIES_DATASET_PATH);
    let queries = load_h5(&queries_path, QUERIES_DATASET_NAME);
    let scenarios = [
        ("First Query", &queries[0]),
        ("Middle Query", &queries[queries.len() / 2]),
        ("Last Query", &queries[queries.len() - 1]),
    ];
    let mut group = c.benchmark_group("Search Latency");
    group.throughput(Throughput::Elements(1));
    let index_path = root.join(INDEX_DUMP_DIR);
    println!("Loading index from {:?}", index_path);
    let index = IndexBuilder::from_disk(&index_path)
        .expect("Failed to load index from disk")
        .build()
        .expect("Failed to build index");
    for (scenario_name, query_vec) in scenarios {
        group.bench_with_input(
            BenchmarkId::from_parameter(scenario_name),
            query_vec,
            |b, query| {
                b.iter(|| {
                    let _ = index.search(black_box(query), (MAX_K, SEARCH_STRATEGY));
                });
            },
        );
    }
    group.finish();
}

fn index_search_throughput_benchmark(c: &mut Criterion) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let queries_path = root.join(QUERIES_DATASET_PATH);
    let mut queries = load_h5(&queries_path, QUERIES_DATASET_NAME);

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    queries.shuffle(&mut rng);

    let mut group = c.benchmark_group("Index Throughput");
    group.throughput(Throughput::Elements(1));
    let index_path = root.join(INDEX_DUMP_DIR);
    let index = IndexBuilder::from_disk(&index_path)
        .expect("Failed to load index from disk")
        .build()
        .expect("Failed to build index");
    group.bench_function("random_access_mix", |b| {
        let mut query_iter = queries.iter().cycle();
        b.iter(|| {
            let query = query_iter.next().unwrap();
            let _ = index.search(black_box(query), black_box((MAX_K, SEARCH_STRATEGY)));
        })
    });
    group.finish();
}

criterion_group!(benches, index_search_parametrized_benchmark);
criterion_group!(throughput_benches, index_search_throughput_benchmark);
criterion_main!(benches, throughput_benches);
