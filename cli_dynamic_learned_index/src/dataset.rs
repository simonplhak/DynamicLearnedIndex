use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tch::{Kind, Tensor};

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    identifier: String,
    size: u64,
    dataset_path: PathBuf,
    queries_path: PathBuf,
    ground_truth_path: PathBuf,
    load_method: LoadMethod,
}

#[derive(Debug, Serialize, Deserialize)]
enum LoadMethod {
    HDF5(String),
}

pub(crate) fn load_dataset(config: &DatasetConfig) -> Result<Tensor> {
    match &config.load_method {
        LoadMethod::HDF5(dataset_name) => load_hdf5(&config.dataset_path, dataset_name),
    }
}

fn load_hdf5(path: &PathBuf, dataset_name: &str) -> Result<Tensor> {
    let emb = hdf5::File::open(path)?.dataset(dataset_name)?;
    let data = emb.read_2d::<f32>()?;
    let x = Tensor::from_slice(data.as_slice().unwrap()).to_kind(Kind::Float);
    let x = x.reshape([
        i64::try_from(emb.shape()[0]).unwrap(),
        i64::try_from(emb.shape()[1]).unwrap(),
    ]);

    Ok(x)
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, Serialize, Deserialize)]
pub(crate) enum Dataset {
    K300,
    M10,
    M100,
}

impl std::fmt::Display for Dataset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Dataset {
    pub(crate) fn config(&self) -> DatasetConfig {
        match &self {
            Dataset::K300 => DatasetConfig {
                identifier: "K300".to_string(),
                size: 300_000,
                dataset_path: PathBuf::from("data/laion2B-en-clip768v2-n=300K.h5"),
                queries_path: PathBuf::from("data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5"),
                ground_truth_path: PathBuf::from(
                    "gold-standard-dbsize=300K--public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
                ),
                load_method: LoadMethod::HDF5("emb".to_string()),
            },
            Dataset::M10 => DatasetConfig {
                identifier: "M10".to_string(),
                size: 10_120_191,
                dataset_path: PathBuf::from("data/laion2B-en-clip768v2-n=10M.h5"),
                queries_path: PathBuf::from("data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5"),
                ground_truth_path: PathBuf::from(
                    "data/gold-standard-dbsize=10M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
                ),
                load_method: LoadMethod::HDF5("emb".to_string()),
            },
            Dataset::M100 => DatasetConfig {
                identifier: "M100".to_string(),
                size: 102_144_212,
                dataset_path: PathBuf::from("data/laion2B-en-clip768v2-n=100M.h5"),
                queries_path: PathBuf::from("data/public-queries-2024-laion2B-en-clip768v2-n=10k.h5"),
                ground_truth_path: PathBuf::from(
                    "data/gold-standard-dbsize=100M--public-queries-2024-laion2B-en-clip768v2-n=10k.h5",
                ),
                load_method: LoadMethod::HDF5("emb".to_string()),
            },
        }
    }
}
