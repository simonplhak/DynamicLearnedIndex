use dynamic_learned_index::{
    model::{ModelBuilder, ModelDevice, ModelLayer, RetrainStrategy},
    structs::LabelMethod,
    DliResult, TrainParams,
};
use log::info;
use rand::Rng;
use std::env;

fn generate_dummy_data(num_samples: usize, input_dim: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let mut data = Vec::with_capacity(num_samples * input_dim);

    for _ in 0..num_samples * input_dim {
        data.push(rng.random_range(-1.0..1.0));
    }

    data
}

fn run_performance_test(
    num_samples: usize,
    input_dim: usize,
    n_buckets: usize,
    epochs: usize,
    batch_size: usize,
) -> DliResult<()> {
    info!("=== Performance Test Configuration ===");
    info!("Samples: {}", num_samples);
    info!("Input dimension: {}", input_dim);
    info!("Number of buckets: {}", n_buckets);
    info!("Epochs: {}", epochs);
    info!("Batch size: {}", batch_size);
    info!("=====================================");

    let data_gen_start = std::time::Instant::now();
    let data = generate_dummy_data(num_samples, input_dim);
    let data_gen_time = data_gen_start.elapsed();
    info!("Data generation took: {:?}", data_gen_time);

    let model_build_start = std::time::Instant::now();
    let mut model = ModelBuilder::default()
        .device(ModelDevice::Cpu)
        .input_nodes(input_dim as i64)
        .add_layer(ModelLayer::Linear(128))
        .add_layer(ModelLayer::ReLU)
        .add_layer(ModelLayer::Linear(64))
        .add_layer(ModelLayer::ReLU)
        .labels(n_buckets)
        .train_params(TrainParams {
            epochs,
            batch_size,
            threshold_samples: num_samples,
            max_iters: 20, // K-means iterations - reduced from 100 for performance
            retrain_strategy: RetrainStrategy::NoRetrain,
        })
        .label_method(LabelMethod::KMeans)
        .build()?;
    let model_build_time = model_build_start.elapsed();
    info!("Model building took: {:?}", model_build_time);

    info!("Starting training...");
    let training_start = std::time::Instant::now();
    model.train(&data)?;
    let training_time = training_start.elapsed();

    info!("=== Performance Results ===");
    info!("Total training time: {:?}", training_time);
    info!(
        "Samples per second: {:.2}",
        num_samples as f64 / training_time.as_secs_f64()
    );
    info!("===========================");

    Ok(())
}

fn main() -> DliResult<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("debug")).init();

    let args: Vec<String> = env::args().collect();

    let (num_samples, input_dim, n_buckets, epochs, batch_size) = if args.len() >= 6 {
        (
            args[1].parse().expect("Invalid num_samples"),
            args[2].parse().expect("Invalid input_dim"),
            args[3].parse().expect("Invalid n_buckets"),
            args[4].parse().expect("Invalid epochs"),
            args[5].parse().expect("Invalid batch_size"),
        )
    } else {
        // Default parameters - moderate size for initial testing
        (10000, 50, 32, 5, 256)
    };

    run_performance_test(num_samples, input_dim, n_buckets, epochs, batch_size)?;

    // Run additional test configurations
    info!("\n=== Running additional test configurations ===");

    // // Small dataset test
    // info!("\n--- Small dataset test ---");
    // run_performance_test(1000, 20, 16, 3, 128)?;

    // // Large dataset test
    // info!("\n--- Large dataset test ---");
    // run_performance_test(50000, 100, 64, 3, 512)?;

    // // High dimensional test
    // info!("\n--- High dimensional test ---");
    // run_performance_test(5000, 200, 32, 3, 256)?;

    Ok(())
}
