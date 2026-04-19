#![feature(test)]

extern crate test;

use dynamic_learned_index::model::{
    Model, ModelBuilder, ModelInterface, ModelLayer, RetrainStrategy, TrainParams,
};
use dynamic_learned_index::structs::LabelMethod;
use dynamic_learned_index::ModelDevice;
use rand::Rng;
use test::Bencher;

const VECTOR_DIM: usize = 768;
const SAMPLE_SIZE: usize = 1000;
const NUM_CLUSTERS: usize = 9;
const BATCH_SIZE: usize = 256;
const HIDDEN_NEURONS: usize = 256;
const EPOCHS: usize = 1;
const PREDICT_MANY_SIZE: usize = 1_000;

fn generate_random_data(size: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..size * dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}

fn model() -> Model<f32> {
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
            retrain_strategy: RetrainStrategy::NoRetrain,
        })
        .label_method(LabelMethod::KMeans)
        .build()
        .unwrap()
}

#[bench]
fn bench_model_train(b: &mut Bencher) {
    let data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);

    b.iter(|| {
        let mut model = model();
        model.train(&data).unwrap();
    });
}

#[bench]
fn bench_model_predict_single(b: &mut Bencher) {
    let train_data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);
    let mut model = model();
    let test_vector = model
        .vec2tensor(&generate_random_data(1, VECTOR_DIM))
        .unwrap();

    model.train(&train_data).unwrap();

    b.iter(|| {
        let _result = model.predict(&test_vector);
    });
}

#[bench]
fn bench_model_predict_many(b: &mut Bencher) {
    let train_data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);
    let test_data = generate_random_data(PREDICT_MANY_SIZE, VECTOR_DIM);

    let mut model = model();

    model.train(&train_data).unwrap();

    b.iter(|| {
        let _result = model.predict_many(&test_data);
    });
}
