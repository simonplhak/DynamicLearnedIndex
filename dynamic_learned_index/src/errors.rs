use thiserror::Error;

#[derive(Debug, Error)]
pub enum DliError {
    #[error("Missing attribute: {0}")]
    MissingAttribute(&'static str),

    #[error("File operation failed: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid YAML config: {0}")]
    YamlError(#[from] serde_yaml::Error),

    #[error("Invalid JSON config: {0}")]
    JsonError(#[from] serde_json::Error),

    #[cfg(any(feature = "candle", feature = "mix"))]
    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Model creation failed: {0}")]
    ModelCreation(&'static str),

    #[error("Generic error: {0}")]
    Generic(String),
}

pub type DliResult<T> = Result<T, DliError>;

// tch-rs integration: automatic error conversion
#[cfg(any(feature = "tch", feature = "mix"))]
impl From<tch::TchError> for DliError {
    fn from(err: tch::TchError) -> Self {
        DliError::ModelCreation(Box::leak(err.to_string().into_boxed_str()))
    }
}

// PyO3 integration: automatic error conversion
#[cfg(feature = "pyo3")]
impl From<DliError> for pyo3::PyErr {
    fn from(err: DliError) -> pyo3::PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}
