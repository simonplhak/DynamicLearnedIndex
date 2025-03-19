use std::fmt;

use serde::{Deserialize, Serialize};
use tch::{kind, Tensor};

use crate::errors::BuildError;

#[derive(Debug, Default, Serialize, Deserialize, Clone)]
pub struct BucketConfig {
    pub bucket_type: BucketType,
}

enum BucketError {
    Overflow,
}

#[derive(Debug, Default, Serialize, Deserialize, Clone, Copy)]
pub(crate) enum BucketType {
    #[default]
    #[serde(rename = "static")]
    Static,
}

pub(crate) trait Bucket {
    fn search(&self, key: &Tensor) -> (Tensor, Tensor);
    fn insert(&mut self, value: Tensor, id: i64) -> Result<(), BucketError>;
}

impl fmt::Debug for dyn Bucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bucket") // todo
    }
}

#[derive(Debug, Default)]
pub(crate) struct BucketBuilder {
    size: Option<i64>,
    input_shape: Option<i64>,
    bucket_type: BucketType,
}

impl BucketBuilder {
    pub fn size(&mut self, size: i64) -> &mut Self {
        self.size = Some(size);
        self
    }

    pub fn input_shape(&mut self, input_shape: i64) -> &mut Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn bucket_type(&mut self, bucket_type: BucketType) -> &mut Self {
        self.bucket_type = bucket_type;
        self
    }

    pub fn build(&self) -> Result<Box<dyn Bucket>, BuildError> {
        let bucket = match self.bucket_type {
            BucketType::Static => self.build_static_bucket(),
        };
        bucket.map(|bucket| Box::new(bucket) as Box<dyn Bucket>)
    }

    fn build_static_bucket(&self) -> Result<StaticBucket, BuildError> {
        let size = self.size.ok_or(BuildError::MissingAttribute)?;
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;

        Ok(StaticBucket {
            records: Tensor::zeros([size, input_shape], kind::FLOAT_CPU),
            ids: Tensor::zeros(&[size], kind::INT64_CPU),
        })
    }
}

#[derive(Debug)]
pub(crate) struct StaticBucket {
    records: Tensor, // (size, ..dim)
    ids: Tensor,     // (size,)
}

impl Bucket for StaticBucket {
    fn search(&self, key: &Tensor) -> (Tensor, Tensor) {
        todo!()
    }

    fn insert(&mut self, value: Tensor, id: i64) -> Result<(), BucketError> {
        todo!()
    }
}
