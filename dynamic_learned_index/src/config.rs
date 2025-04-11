use dotenvy::dotenv;
use log::info;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::env;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub bucket_scaling_factor: f64,
    pub skip_insert_log: usize,
}

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    dotenv().ok(); // load from .env
    let bucket_scaling_factor = env::var("BUCKET_SCALING_FACTOR")
        .unwrap_or_else(|_| "1.0".to_string())
        .parse::<f64>()
        .expect("BUCKET_SCALING_FACTOR must be a valid number");
    let skip_insert_log = env::var("SKIP_INSERT_LOGS")
        .unwrap_or_else(|_| "10".to_string())
        .parse::<usize>()
        .expect("SKIP_INSERT_LOGS must be a valid number");
    info!(
        bucket_scaling_factor = bucket_scaling_factor,
        skip_insert_log = skip_insert_log;
        "config:load"
    );
    Config {
        bucket_scaling_factor,
        skip_insert_log,
    }
});
