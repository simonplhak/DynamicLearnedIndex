use anyhow::Result;
use clap::{Parser, Subcommand};
use dataset::{config_from_yaml, load_dataset};
use dynamic_learned_index::{self};
use eval::{eval_queries, insert_all_data};
use log::info;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use structured_logger::json::new_writer;
use tch::{IndexOp, Tensor};
mod config;
mod dataset;
mod eval;

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
#[derive(Parser, Debug, Serialize, Deserialize, Clone, clap::ValueEnum)]
enum LogOuptut {
    Stdout,
    File,
}

#[derive(Parser, Debug, Serialize, Deserialize)]
struct ExperimentConfig {
    experiment_name: PathBuf,
    #[arg(short, long)]
    dataset_config: PathBuf,
    #[arg(short, long)]
    index_config: PathBuf,
    #[arg(short, long)]
    force: bool,
    #[arg(short, long, default_value = "stdout")]
    log_output: LogOuptut,
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
    let log_writer = match experiment_config.log_output {
        LogOuptut::Stdout => new_writer(std::io::stdout()),
        LogOuptut::File => new_writer(fs::File::create(experiment_dir.join("logs.jsonl"))?),
    };
    structured_logger::Builder::with_level("info")
        .with_target_writer("*", log_writer)
        .init();
    let config_yaml = serde_yaml::to_string(&experiment_config)?;
    fs::write(experiment_dir.join("config.yaml"), config_yaml)?;
    let config_content = fs::read_to_string(&experiment_config.index_config)?;
    let index_config = serde_yaml::from_str::<dynamic_learned_index::IndexConfig>(&config_content)?;
    fs::write(experiment_dir.join("index_config.yaml"), config_content)?;
    let mut index = index_config.build()?;
    let data = load_dataset(&dataset_config.dataset)?;
    let gt = load_dataset(&dataset_config.ground_truth)?;
    let queries = load_dataset(&dataset_config.queries)?;
    insert_all_data(&mut index, data);
    let metrics = eval_queries(&index, gt, queries);
    info!(total = metrics.total, recall_top1=metrics.recall_top1, recall_top5=metrics.recall_top5, recall_top10=metrics.recall_top10; "metrics");
    Ok(())
}

fn test() -> Result<()> {
    let experiment_dir = PathBuf::from("experiments_data/test");
    if !experiment_dir.exists() {
        fs::create_dir_all(&experiment_dir)?;
    }
    structured_logger::Builder::with_level("info")
        .with_target_writer("*", new_writer(std::io::stdout()))
        // .with_target_writer(
        //     "*",
        //     new_writer(fs::File::create(experiment_dir.join("logs.jsonl"))?),
        // )
        .init();
    let path = PathBuf::from("configs/example.yaml");
    let config_content = fs::read_to_string(path)?;
    let index_config = serde_yaml::from_str::<dynamic_learned_index::IndexConfig>(&config_content)?;
    let mut index = index_config.build()?;
    let dataset_config = config_from_yaml(&PathBuf::from("data/k300/config.yaml"))?;
    let data = load_dataset(&dataset_config.dataset)?;
    let gt = load_dataset(&dataset_config.ground_truth)?;
    let queries = load_dataset(&dataset_config.queries)?;
    insert_all_data(&mut index, data);
    let metrics = eval_queries(&index, gt, queries);
    info!(total = metrics.total, recall_top1=metrics.recall_top1, recall_top5=metrics.recall_top5, recall_top10=metrics.recall_top10; "metrics");
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    // let matrix = Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).reshape([2, 3]);
    // Example vector: shape (3,)
    // let result = {
    //     let vector = Tensor::from_slice(&[10, 20, 30]);
    //     let result = tch::Tensor::cat(matrix, vector);
    //     &matrix + &vector
    // };
    // Broadcasting: add vector to each row
    // result.print();
    // let a = Tensor::from_slice(&[1, 2]).unsqueeze(0); // shape (1, 2)
    // let b = Tensor::from_slice(&[3, 4]).unsqueeze(0);

    // let result = Tensor::cat(&[a, b], 0); // concatenate along dimension 1

    // result.print();
    // Ok(())
    match &cli.command {
        Commands::Experiment(config) => experiment(config),
        Commands::Test => test(),
    }
}
