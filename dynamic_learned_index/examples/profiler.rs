use std::fs::File;
use std::hint::black_box;
use std::path::PathBuf;

use dynamic_learned_index::{Array, IndexBuilder, SearchStrategy};
use pprof::ProfilerGuard;

const QUERIES_DATASET_PATH: &str = "data/k300/queries.h5";
const QUERIES_DATASET_NAME: &str = "emb";
const INDEX_DUMP_DIR: &str = "index_dump";
const MAX_K: usize = 10;
const SEARCH_STRATEGY: SearchStrategy = SearchStrategy::ModelDriven(100);

#[cfg(feature = "hdf5")]
fn load_h5(path: &PathBuf, dataset_name: &str) -> Vec<Array> {
    let emb = hdf5::File::open(path)
        .expect("Failed to open HDF5 file")
        .dataset(dataset_name)
        .expect("Failed to open dataset");
    let data = emb.read_2d::<f32>().expect("Failed to read dataset");
    data.outer_iter().map(|row| row.to_vec()).collect()
}

#[cfg(not(feature = "hdf5"))]
fn load_h5(_path: &PathBuf, _dataset_name: &str) -> Vec<Array> {
    println!("Run with --feature hdf5");
    vec![]
}

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let queries_path = root.join(QUERIES_DATASET_PATH);
    println!("Loading queries from {:?}", queries_path);
    let queries = load_h5(&queries_path, QUERIES_DATASET_NAME);

    let index_path = root.join(INDEX_DUMP_DIR);
    println!("Loading index from {:?}", index_path);
    let index = IndexBuilder::from_disk(&index_path)
        .expect("Failed to load index from disk")
        .build()
        .expect("Failed to build index");

    println!("Starting profiler...");
    let guard = ProfilerGuard::new(100).unwrap();

    println!("Running search on first 100 queries...");

    for query in &queries[0..100] {
        let _ = index.search(black_box(query), black_box((MAX_K, SEARCH_STRATEGY)));
    }

    println!("Creating flamegraph...");
    if let Ok(report) = guard.report().build() {
        let file = File::create("flamegraph.svg").unwrap();
        report.flamegraph(file).unwrap();
        println!("Flamegraph generated at flamegraph.svg");
    };
}
