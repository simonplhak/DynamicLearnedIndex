use anyhow::Result;
use clap::{Parser, Subcommand};
use dataset::{config_from_yaml, load_dataset};
use dynamic_learned_index::{self};
use log::info;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf};
use structured_logger::json::new_writer;
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
    let ds = dataset::load_dataset(&dataset_config.dataset)?;
    let config_yaml = serde_yaml::to_string(&experiment_config)?;
    fs::write(experiment_dir.join("config.yaml"), config_yaml)?;
    info!(dataset_size:? = ds.size(); "experiment");
    Ok(())
}

fn test() -> Result<()> {
    let experiment_dir = PathBuf::from("experiments_data/example");
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
    let dataset_config = config_from_yaml(&PathBuf::from("data/example/config.yaml"))?;
    let ds = load_dataset(&dataset_config.dataset)?;
    let limit = 100;
    (0..limit).for_each(|i| {
        let tensor = ds.i((i, ..));
        println!("Inserting tensor: {} shape={:?}", i, tensor.size());
        index.insert(tensor, i as u32);
    });
    println!("Insert finished");
    (0..limit).for_each(|i| {
        let tensor = ds.i((i, ..));
        println!("Searching tensor: {} shape={:?}", i, tensor.size());
        let result = index.search(&tensor, 1);
        println!("Result: {:?}", result);
    });
    println!("{:?}", ds.size());
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
