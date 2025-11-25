use std::fmt;

#[derive(Debug)]
pub enum BuildError {
    MissingAttribute,
    NonExistentFile,
    InvalidYamlConfig(String),
    ModelCreation(&'static str),
    MissingAttributeStr(&'static str),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::MissingAttribute => write!(f, "MissingAttribute"),
            BuildError::NonExistentFile => write!(f, "NonExistentFile"),
            BuildError::InvalidYamlConfig(err) => write!(f, "Invalid YAML config: {err}"),
            BuildError::ModelCreation(msg) => write!(f, "Model creation failed: {msg}"),
            BuildError::MissingAttributeStr(err) => write!(f, "Missing attribute: {err}"),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
