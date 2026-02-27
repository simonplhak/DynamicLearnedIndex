#![feature(test)]

extern crate test;

use dynamic_learned_index::model::candle_model::{
    Model as ModelNew, ModelBuilder as ModelNewBuilder,
};
#[cfg(feature = "tch")]
use dynamic_learned_index::model::mix_model::{Model as MixModel, ModelBuilder as MixModelBuilder};
#[cfg(feature = "tch")]
use dynamic_learned_index::model::tch_model::{Model, ModelBuilder};
use dynamic_learned_index::model::{ModelLayer, RetrainStrategy, TrainParams};
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
const PREDICT_MANY_SIZE: usize = 10_000;

fn generate_random_data(size: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    (0..size * dim)
        .map(|_| rng.random_range(-1.0..1.0))
        .collect()
}

#[cfg(feature = "tch")]
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
            retrain_strategy: RetrainStrategy::NoRetrain,
        })
        .label_method(LabelMethod::KMeans)
        .build()
        .unwrap()
}

#[cfg(feature = "tch")]
#[bench]
fn bench_pytorch_model_train(b: &mut Bencher) {
    let data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);

    b.iter(|| {
        let mut model = pytorch_model();
        model.train(&data);
    });
}

#[cfg(feature = "tch")]
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

#[cfg(feature = "tch")]
#[bench]
fn bench_pytorch_model_predict_many(b: &mut Bencher) {
    let train_data = generate_random_data(1000, VECTOR_DIM);
    let test_data = generate_random_data(PREDICT_MANY_SIZE, VECTOR_DIM); // 100 test vectors

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
fn bench_candle_model_train(b: &mut Bencher) {
    let data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);

    b.iter(|| {
        let mut model = candle_model();
        model.train(&data).unwrap();
    });
}

#[bench]
fn bench_candle_model_predict_single(b: &mut Bencher) {
    let train_data = generate_random_data(1000, VECTOR_DIM);
    let mut model = candle_model();
    let test_vector = model
        .vec2tensor(&generate_random_data(1, VECTOR_DIM))
        .unwrap();

    model.train(&train_data).unwrap();

    b.iter(|| {
        let _result = model.predict(&test_vector);
    });
}

#[bench]
fn bench_candle_model_predict_many(b: &mut Bencher) {
    let train_data = generate_random_data(1000, VECTOR_DIM);
    let test_data = generate_random_data(PREDICT_MANY_SIZE, VECTOR_DIM);

    let mut model = candle_model();

    model.train(&train_data).unwrap();

    b.iter(|| {
        let _result = model.predict_many(&test_data);
    });
}

fn mix_model() -> MixModel {
    MixModelBuilder::default()
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
fn bench_mix_model_train(b: &mut Bencher) {
    let data = generate_random_data(SAMPLE_SIZE, VECTOR_DIM);

    b.iter(|| {
        let mut model = mix_model();
        model.train(&data).unwrap();
    });
}

#[bench]
fn bench_mix_model_predict_single(b: &mut Bencher) {
    let train_data = generate_random_data(1000, VECTOR_DIM);
    let mut model = mix_model();
    let test_vector = model
        .vec2tensor(&generate_random_data(1, VECTOR_DIM))
        .unwrap();

    model.train(&train_data).unwrap();

    b.iter(|| {
        let _result = model.predict(&test_vector);
    });
}

#[bench]
fn bench_mix_model_predict_many(b: &mut Bencher) {
    let train_data = generate_random_data(1000, VECTOR_DIM);
    let test_data = generate_random_data(PREDICT_MANY_SIZE, VECTOR_DIM);

    let mut model = mix_model();

    model.train(&train_data).unwrap();

    b.iter(|| {
        let _result = model.predict_many(&test_data);
    });
}
