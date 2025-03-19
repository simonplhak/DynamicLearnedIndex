use std::fmt;

#[derive(Debug)]
pub enum BuildError {
    MissingAttribute,
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BuildError::MissingAttribute => write!(f, "MissingAttribute"),
        }
    }
}

impl std::error::Error for BuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
