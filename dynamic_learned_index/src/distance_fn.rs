use serde::{Deserialize, Serialize};

#[derive(Default, Deserialize, Serialize, Debug, Clone)]
pub enum DistanceFn {
    #[serde(rename = "l2")]
    L2,
    #[default]
    #[serde(rename = "dot")]
    Dot,
}

impl From<DistanceFn> for LabelMethod {
    fn from(val: DistanceFn) -> Self {
        match val {
            DistanceFn::L2 => LabelMethod::KMeans,
            DistanceFn::Dot => LabelMethod::SphericalKMeans,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum LabelMethod {
    KMeans,
    SphericalKMeans,
}
