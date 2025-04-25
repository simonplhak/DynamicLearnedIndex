use dotenvy::dotenv;
use log::info;
use once_cell::sync::Lazy;
use serde::Deserialize;
use std::env;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub skip_insert_log: usize,
}

pub static CONFIG: Lazy<Config> = Lazy::new(|| {
    dotenv().ok(); // load from .env
    let skip_insert_log = env::var("SKIP_INSERT_LOGS")
        .unwrap_or_else(|_| "10".to_string())
        .parse::<usize>()
        .expect("SKIP_INSERT_LOGS must be a valid number");
    info!(
        skip_insert_log = skip_insert_log;
        "config:load"
    );
    Config { skip_insert_log }
});
