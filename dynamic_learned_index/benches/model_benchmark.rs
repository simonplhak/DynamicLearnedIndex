#![feature(test)]

extern crate test;

use dynamic_learned_index::candle_model::{Model as ModelNew, ModelBuilder as ModelNewBuilder};
use dynamic_learned_index::distance_fn::LabelMethod;
use dynamic_learned_index::model::{Model, ModelBuilder, ModelLayer, TrainParams};
use dynamic_learned_index::ModelDevice;
use rand::Rng;
use test::Bencher;

const VECTOR_DIM: usize = 768;
const SAMPLE_SIZE: usize = 10000;
const NUM_CLUSTERS: usize = 9;
const BATCH_SIZE: i64 = 256;
const HIDDEN_NEURONS: i64 = 256;
const EPOCHS: usize = 1;

fn generate_random_data(size: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..size * dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}

fn pytorch_model() -> Model {
    ModelBuilder::default()
        .device(ModelDevice::Cpu)
        .input_nodes(VECTOR_DIM as i64)
        .add_layer(ModelLayer::Linear(HIDDEN_NEURONS))
        .add_layer(ModelLayer::ReLU)
        .add_layer(ModelLayer::Linear(HIDDEN_NEURONS))
        .labels(NUM_CLUSTERS)
        .train_params(TrainParams {
            threshold_samples: SAMPLE_SIZE,
            batch_size: BATCH_SIZE,
            epochs: EPOCHS,
            max_iters: 10,
        })
        .label_method(LabelMethod::KMeans)
        .build()
        .unwrap()
}

#[bench]
fn bench_pytorch_model_train(b: &mut Bencher) {
    let data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);

    b.iter(|| {
        let mut model = pytorch_model();
        model.train(&data);
    });
}

#[bench]
fn bench_pytorch_model_predict_single(b: &mut Bencher) {
    let train_data = generate_random_data(1000, VECTOR_DIM); // Smaller training set
    let test_vector = generate_random_data(1, VECTOR_DIM);

    let mut model = pytorch_model();

    model.train(&train_data);

    b.iter(|| {
        let _result = model.predict(&test_vector);
    });
}

#[bench]
fn bench_pytorch_model_predict_many(b: &mut Bencher) {
    let train_data = generate_random_data(1000, VECTOR_DIM);
    let test_data = generate_random_data(1000, VECTOR_DIM); // 100 test vectors

    let mut model = pytorch_model();

    model.train(&train_data);

    b.iter(|| {
        let _result = model.predict_many(&test_data);
    });
}

fn candle_model() -> ModelNew {
    ModelNewBuilder::default()
        .device(ModelDevice::Cpu)
        .input_nodes(VECTOR_DIM as i64)
        .add_layer(ModelLayer::Linear(HIDDEN_NEURONS))
        .add_layer(ModelLayer::ReLU)
        .add_layer(ModelLayer::Linear(HIDDEN_NEURONS))
        .labels(NUM_CLUSTERS)
        .train_params(TrainParams {
            threshold_samples: 1000,
            batch_size: BATCH_SIZE,
            epochs: EPOCHS,
            max_iters: 10,
        })
        .label_method(LabelMethod::KMeans)
        .build()
        .unwrap()
}

#[bench]
fn bench_candle_model_train(b: &mut Bencher) {
    let data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);

    b.iter(|| {
        let mut model = candle_model();
        model.train(&data);
    });
}

#[bench]
fn bench_candle_model_predict_single(b: &mut Bencher) {
    let train_data = generate_random_data(1000, VECTOR_DIM);
    let test_vector = generate_random_data(1, VECTOR_DIM);

    let mut model = candle_model();

    model.train(&train_data);

    b.iter(|| {
        let _result = model.predict(&test_vector);
    });
}

#[bench]
fn bench_candle_model_predict_many(b: &mut Bencher) {
    let train_data = generate_random_data(1000, VECTOR_DIM);
    let test_data = generate_random_data(1000, VECTOR_DIM);

    let mut model = candle_model();

    model.train(&train_data);

    b.iter(|| {
        let _result = model.predict_many(&test_data);
    });
}
