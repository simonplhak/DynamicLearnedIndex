use anyhow::Result;
use dynamic_learned_index::types::{Array, Id};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

pub(crate) fn config_from_yaml(path: &PathBuf) -> Result<DatasetConfig> {
    let config = std::fs::read_to_string(path)?;
    let config = serde_yaml::from_str(&config)?;
    Ok(config)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetConfig {
    // todo add docs
    pub dataset: LoadMethod,
    pub queries: LoadMethod,
    pub ground_truth: LoadMethod,
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

pub(crate) fn load_dataset(load_method: &LoadMethod) -> Result<Vec<Array>> {
    match &load_method {
        LoadMethod::H5(dataset) => load_h5(&dataset.path, &dataset.dataset_name),
    }
}

fn load_h5(path: &PathBuf, dataset_name: &str) -> Result<Vec<Array>> {
    let emb = hdf5::File::open(path)?.dataset(dataset_name)?;
    let data = emb.read_2d::<f32>()?;
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
