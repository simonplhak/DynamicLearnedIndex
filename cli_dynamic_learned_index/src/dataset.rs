use anyhow::Result;
use dynamic_learned_index::types::Id;
use half::f16;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

type Dataset = Vec<Vec<f16>>;
type Queries = Vec<Vec<f16>>;
type GroundTruth = Vec<Vec<Id>>;

pub(crate) fn load_dataset_config(path: &Path) -> Result<DatasetConfig> {
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "Dataset directory does not exist: {}",
            path.display()
        ));
    }
    let config_path = path.join("config.yaml");
    match !config_path.exists() {
        true => {
            let config = std::fs::read_to_string(config_path)?;
            let config = serde_yaml::from_str(&config)?;
            Ok(config)
        }
        false => Ok(DatasetConfig::new(path)),
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub dataset: LoadMethod,
    pub queries: LoadMethod,
    pub ground_truth: LoadMethod,
}

impl DatasetConfig {
    pub(crate) fn new(path: &Path) -> Self {
        Self {
            dataset: LoadMethod::H5(H5LoadMethod {
                path: path.join("data.h5"),
                dataset_name: "emb".into(),
            }),
            queries: LoadMethod::H5(H5LoadMethod {
                path: path.join("queries.h5"),
                dataset_name: "emb".into(),
            }),
            ground_truth: LoadMethod::H5(H5LoadMethod {
                path: path.join("gt.h5"),
                dataset_name: "knns".into(),
            }),
        }
    }

    pub(crate) fn load(&self) -> Result<(Dataset, Queries, GroundTruth)> {
        let queries = load_dataset(&self.dataset)?;
        let test_queries = load_dataset(&self.queries)?;
        let ground_truth = load_dataset_ids(&self.ground_truth)?;
        Ok((queries, test_queries, ground_truth))
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum LoadMethod {
    #[serde(rename = "h5")]
    H5(H5LoadMethod),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct H5LoadMethod {
    path: PathBuf,
    dataset_name: String,
}

fn load_dataset(load_method: &LoadMethod) -> Result<Vec<Vec<f16>>> {
    match &load_method {
        LoadMethod::H5(dataset) => load_h5(&dataset.path, &dataset.dataset_name),
    }
}

fn load_h5(path: &PathBuf, dataset_name: &str) -> Result<Vec<Vec<f16>>> {
    let emb = hdf5::File::open(path)?.dataset(dataset_name)?;
    let data = emb.read_2d::<f16>()?;
    Ok(data.outer_iter().map(|row| row.to_vec()).collect())
}

pub(crate) fn load_dataset_ids(load_method: &LoadMethod) -> Result<Vec<Vec<Id>>> {
    match &load_method {
        LoadMethod::H5(dataset) => load_h5_ids(&dataset.path, &dataset.dataset_name),
    }
}

fn load_h5_ids(path: &PathBuf, dataset_name: &str) -> Result<Vec<Vec<Id>>> {
    let emb = hdf5::File::open(path)?.dataset(dataset_name)?;
    let data = emb.read_2d::<Id>()?;
    Ok(data.outer_iter().map(|row| row.to_vec()).collect())
}
