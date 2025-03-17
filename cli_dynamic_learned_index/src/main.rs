use anyhow::Result;
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
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
}

#[derive(Parser, Debug, Serialize, Deserialize)]
struct ExperimentConfig {
    experiment_name: PathBuf,
    #[arg(short, long, default_value_t = dataset::Dataset::K300)]
    dataset: dataset::Dataset,
    #[arg(short, long)]
    force: bool,
}

fn experiment(experiment_config: &ExperimentConfig) -> Result<()> {
    let config = config::Config::new();

    let dataset_config = experiment_config.dataset.config();
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
    let ds = dataset::load_dataset(&dataset_config)?;
    let config_yaml = serde_yaml::to_string(&experiment_config)?;
    fs::write(experiment_dir.join("config.yaml"), config_yaml)?;

    println!("{:?}, {:?}, {:?}", experiment_config, dataset_config, ds);

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Experiment(config) => experiment(config),
    }
}
