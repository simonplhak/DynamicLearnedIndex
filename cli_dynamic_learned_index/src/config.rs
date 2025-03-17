use dotenvy::dotenv;
use serde::Deserialize;
use std::{env, path::PathBuf};

#[derive(Debug, Deserialize)]
pub struct Config {
    pub data_dir: PathBuf,
    pub experiments_dir: PathBuf,
}

impl Config {
    pub fn new() -> Self {
        dotenv().ok(); // Load .env file if present

        Self {
            data_dir: PathBuf::from(env::var("DATA_DIR").unwrap_or_else(|_| "./data".into())),
            experiments_dir: PathBuf::from(
                env::var("EXPERIMENTS_DIR").unwrap_or_else(|_| "experiments_data".into()),
            ),
        }
    }
}
