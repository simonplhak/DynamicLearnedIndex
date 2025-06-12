use anyhow::Result;
use clap::{Parser, Subcommand};
use dataset::load_dataset_config;
use dynamic_learned_index::{self};
use eval::{eval_queries, insert_all_data};
use log::info;
use serde::{Deserialize, Serialize};
use std::{fs, path::PathBuf, str::FromStr};
use structured_logger::json::new_writer;

use crate::dataset::DatasetConfig;

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
    Defaults(DefaultsConfig),
}

#[derive(Parser, Debug, Serialize, Deserialize, Clone, clap::ValueEnum)]
enum LogOuptut {
    Stdout,
    File,
}

#[derive(Parser, Debug, Serialize, Deserialize)]
struct ExperimentConfig {
    experiment_name: PathBuf,
    /// directory with data, optionally can contain config.yaml that changes default paths
    ///
    /// It expects that directory contains the following data: dataset, queries, gt
    ///
    /// You can display default values with command `defaults dataset`
    ///
    /// Accepted dataset types: h5
    dataset_dir: PathBuf,
    /// Path to index config file, it can be a relative path to the current directory
    /// or an absolute path.
    ///
    /// It expects that the file is in YAML format and contains the index configuration.
    ///
    /// You can display default values with command `defaults index`
    #[arg(short, long)]
    index_config: Option<PathBuf>,
    /// Forces the experiment to overwrite existing data
    /// If the experiment directory already exists, it will be removed and recreated.
    #[arg(short, long)]
    force: bool,
    /// Where are logs outputed
    #[arg(long, default_value = "file")]
    log_output: LogOuptut,
    /// Validates index on inserted queries after n insterted queries
    #[arg(long, default_value = "10000")]
    validate_after_n: usize,
    /// Into validation dataset is included each n-th value
    #[arg(long, default_value = "1000")]
    include_each_n_val: usize,
    /// Skips validation
    #[arg(long)]
    skip_validation: bool,
    /// Limits original dataset size
    #[arg(short, long)]
    limit: Option<usize>,
    /// Where are artifacts of run saved,
    /// more specifically data are stored under experiments_dir / {name} directory
    #[arg(short, long, default_value = "experiments_data")]
    experiments_dir: PathBuf,
    /// Some datasets start indexing from 1, not 0 (from some **** reason)
    /// To mitigate this, we can set this flag to true
    #[arg(long, default_value = "false")]
    start_from_one: bool,
    /// Set verbosity level
    /// -v for info, -vv for debug, -vvv for trace
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,
}

fn experiment(config: &ExperimentConfig) -> Result<()> {
    let dataset_config = load_dataset_config(&config.dataset_dir)?;
    let experiment_dir = config.experiments_dir.join(&config.experiment_name);
    if experiment_dir.exists() {
        if !config.force {
            return Err(anyhow::anyhow!(
                "Experiment dir already exists: {}",
                config.experiment_name.display()
            ));
        } else {
            fs::remove_dir_all(experiment_dir.clone())?;
        }
    }
    fs::create_dir(experiment_dir.clone())?;
    let log_writer = match config.log_output {
        LogOuptut::Stdout => new_writer(std::io::stdout()),
        LogOuptut::File => new_writer(fs::File::create(experiment_dir.join("logs.jsonl"))?),
    };
    let log_level = match config.verbose {
        0 => "info",
        _ => "debug",
    };
    structured_logger::Builder::with_level(log_level)
        .with_target_writer("*", log_writer)
        .init();
    let config_yaml = serde_yaml::to_string(&config)?;
    fs::write(experiment_dir.join("config.yaml"), config_yaml)?;
    let index_config = match &config.index_config {
        Some(index_config_path) => {
            if !index_config_path.exists() {
                return Err(anyhow::anyhow!(
                    "Index config file does not exist: {}",
                    index_config_path.display()
                ));
            }
            let config_content = fs::read_to_string(index_config_path)?;
            serde_yaml::from_str::<dynamic_learned_index::IndexConfig>(&config_content)?
        }
        None => dynamic_learned_index::IndexConfig::default(),
    };

    fs::write(
        experiment_dir.join("index_config.yaml"),
        serde_yaml::to_string(&index_config)?,
    )?;
    let mut index = index_config.build()?;
    let (queries, test_queries, gt) = dataset_config.load()?;
    let validation_options = match config.skip_validation {
        true => None,
        false => Some(eval::ValidationOptions {
            validate_after_n: config.validate_after_n,
            include_each_n: config.include_each_n_val,
        }),
    };
    insert_all_data(
        &mut index,
        queries,
        config.limit,
        validation_options,
        config.start_from_one,
    );
    let metrics = eval_queries(&index, &gt, &test_queries);
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
    let dataset_config = load_dataset_config(&PathBuf::from("data/k300"))?;
    let (queries, test_queries, gt) = dataset_config.load()?;
    let validation_options = eval::ValidationOptions {
        validate_after_n: 100,
        include_each_n: 10,
    };
    insert_all_data(
        &mut index,
        queries,
        Some(200),
        Some(validation_options),
        true,
    );
    let metrics = eval_queries(&index, &gt, &test_queries);
    info!(total=metrics.total, recall_top1=metrics.recall_top1, recall_top5=metrics.recall_top5, recall_top10=metrics.recall_top10; "metrics");
    Ok(())
}

#[derive(Parser, Debug)]
struct DefaultsConfig {
    default_for: DefaultFor,
}

#[derive(Parser, Debug, Clone, clap::ValueEnum)]
enum DefaultFor {
    Dataset,
    Index,
}

fn defaults(config: &DefaultsConfig) -> Result<()> {
    match config.default_for {
        DefaultFor::Dataset => {
            let empty_dir = PathBuf::from_str("")?;
            let default_dataset = DatasetConfig::new(&empty_dir);
            let default_dataset = serde_yaml::to_string(&default_dataset)?;
            println!("{default_dataset}");
        }
        DefaultFor::Index => {
            let default_index = dynamic_learned_index::IndexConfig::default();
            let default_index = serde_yaml::to_string(&default_index)?;
            println!("{default_index}");
        }
    };
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Experiment(config) => experiment(config),
        Commands::Test => test(),
        Commands::Defaults(config) => defaults(config),
    }
}
