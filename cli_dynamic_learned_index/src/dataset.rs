use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tch::{Kind, Tensor};

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

pub(crate) fn load_dataset(load_method: &LoadMethod) -> Result<Tensor> {
    match &load_method {
        LoadMethod::H5(dataset) => load_h5(&dataset.path, &dataset.dataset_name),
    }
}

fn load_h5(path: &PathBuf, dataset_name: &str) -> Result<Tensor> {
    let emb = hdf5::File::open(path)?.dataset(dataset_name)?;
    let data = emb.read_2d::<f32>()?;
    let x = Tensor::from_slice(
        data.as_slice()
            .ok_or_else(|| anyhow::anyhow!("Failed to convert data to slice"))?,
    )
    .to_kind(Kind::Float);
    let x = x.reshape([
        i64::try_from(emb.shape()[0])?,
        i64::try_from(emb.shape()[1])?,
    ]);

    Ok(x)
}
