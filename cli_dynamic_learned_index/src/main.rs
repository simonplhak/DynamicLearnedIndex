use anyhow::Result;
use clap::{Parser, Subcommand};
use dataset::{config_from_yaml, load_dataset};
use dynamic_learned_index::{self};
use log::info;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use tch::{IndexOp, Tensor};
mod config;
mod dataset;

#[derive(Parser, Debug)]
#[command(name = "cli_app", version = "1.0", about = "CLI tool as an entrypoint to dynamic_learned_index crate", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Experiment(ExperimentConfig),
    Test,
}

#[derive(Parser, Debug, Serialize, Deserialize)]
struct ExperimentConfig {
    experiment_name: PathBuf,
    #[arg(short, long)]
    dataset_config: PathBuf,
    #[arg(short, long)]
    force: bool,
}

fn experiment(experiment_config: &ExperimentConfig) -> Result<()> {
    let config = config::Config::new();
    let dataset_config = config_from_yaml(&experiment_config.dataset_config)?;
    let experiment_dir = config
        .experiments_dir
        .join(&experiment_config.experiment_name);
    if experiment_dir.exists() {
        if !experiment_config.force {
            return Err(anyhow::anyhow!(
                "Experiment dir already exists: {}",
                experiment_config.experiment_name.display()
            ));
        } else {
            fs::remove_dir_all(experiment_dir.clone())?;
        }
    }
    fs::create_dir(experiment_dir.clone())?;
    let ds = dataset::load_dataset(&dataset_config.dataset)?;
    let config_yaml = serde_yaml::to_string(&experiment_config)?;
    fs::write(experiment_dir.join("config.yaml"), config_yaml)?;
    info!(dataset_size:? = ds.size(); "experiment");
    Ok(())
}

fn test() -> Result<()> {
    let path = PathBuf::from("configs/example.yaml");
    let config_content = fs::read_to_string(path)?;
    let index_config = serde_yaml::from_str::<dynamic_learned_index::IndexConfig>(&config_content)?;
    let mut index = index_config.build()?;
    let dataset_config = config_from_yaml(&PathBuf::from("data/example/config.yaml"))?;
    let ds = load_dataset(&dataset_config.dataset)?;
    let limit = 10;
    (0..limit).for_each(|i| {
        let tensor = ds.i((i, ..));
        println!("Inserting tensor: {} shape={:?}", i, tensor.size());
        index.insert(tensor, i as u32);
    });
    println!("{:?}", ds.size());
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    structured_logger::Builder::default().init();

    match &cli.command {
        Commands::Experiment(config) => experiment(config),
        Commands::Test => test(),
    }
}
