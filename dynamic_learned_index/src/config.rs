use dotenvy::dotenv;
use log::info;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::env;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub bucket_scaling_factor: f64,
}

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    dotenv().ok(); // load from .env
    let bucket_scaling_factor = env::var("BUCKET_SCALING_FACTOR")
        .expect("BUCKET_SCALING_FACTOR environment variable must be set")
        .parse::<f64>()
        .expect("BUCKET_SCALING_FACTOR must be a valid number");
    info!(
        bucket_scaling_factor = bucket_scaling_factor;
        "config:load"
    );
    Config {
        bucket_scaling_factor,
    }
});
