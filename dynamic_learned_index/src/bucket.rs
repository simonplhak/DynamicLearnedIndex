use std::fmt::Debug;

use serde::{Deserialize, Serialize};
use tch::Tensor;

use crate::{errors::BuildError, Id};

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
#[serde(tag = "type", content = "value")]
pub(crate) enum BucketType {
    #[serde(rename = "static")]
    Static(usize),
}

impl Default for BucketType {
    fn default() -> Self {
        BucketType::Static(10)
    }
}

#[derive(Debug, Serialize)]
pub(crate) enum Bucket {
    Static(StaticBucket),
}

impl Bucket {
    pub fn search(&self, query: &Tensor) -> (Tensor, Tensor) {
        match self {
            Bucket::Static(bucket) => bucket.search(query),
        }
    }

    pub fn insert(&mut self, value: Tensor, id: Id) {
        match self {
            Bucket::Static(bucket) => bucket.insert(value, id),
        }
    }

    pub fn size(&self) -> usize {
        match self {
            Bucket::Static(bucket) => bucket.size(),
        }
    }

    pub fn has_space(&self, count: usize) -> bool {
        match self {
            Bucket::Static(bucket) => bucket.has_space(count),
        }
    }

    pub fn occupied(&self) -> usize {
        match self {
            Bucket::Static(bucket) => bucket.occupied(),
        }
    }

    pub fn get_data(&mut self) -> (Vec<Tensor>, Vec<Id>) {
        match self {
            Bucket::Static(bucket) => bucket.get_data(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct BucketBuilder {
    input_shape: Option<i64>,
    bucket_type: BucketType,
}

impl BucketBuilder {
    pub fn input_shape(&mut self, input_shape: i64) -> &mut Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn bucket_type(&mut self, bucket_type: BucketType) -> &mut Self {
        self.bucket_type = bucket_type;
        self
    }

    pub fn build(&self) -> Result<Bucket, BuildError> {
        let bucket = match self.bucket_type {
            BucketType::Static(size) => Bucket::Static(self.build_static_bucket(size)?),
        };
        Ok(bucket)
    }

    fn build_static_bucket(&self, size: usize) -> Result<StaticBucket, BuildError> {
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;
        Ok(StaticBucket::new(size, input_shape))
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct StaticBucket {
    #[serde(skip)]
    records: Vec<Tensor>,
    #[serde(skip)]
    ids: Vec<Id>,
    size: usize,
    input_shape: i64,
}

impl StaticBucket {
    pub fn new(size: usize, input_shape: i64) -> Self {
        let records = Vec::with_capacity(size);
        let ids = Vec::with_capacity(size);
        Self {
            records,
            ids,
            size,
            input_shape,
        }
    }

    fn search(&self, query: &Tensor) -> (Tensor, Tensor) {
        // todo implement search, this is just for ilustration
        (
            Tensor::randint(
                self.records.len() as i64,
                [self.records.len() as i64],
                tch::kind::INT64_CPU,
            ),
            Tensor::rand([self.records.len() as i64], (tch::kind::FLOAT_CPU)),
        )
    }

    fn insert(&mut self, value: Tensor, id: Id) {
        debug_assert!(self.has_space(1), "Bucket is full size={}", self.size);
        debug_assert_eq!(value.size()[0], self.input_shape);
        self.records.push(value);
        self.ids.push(id);
    }

    fn has_space(&self, count: usize) -> bool {
        self.records.len() + count <= self.size
    }

    fn occupied(&self) -> usize {
        self.records.len()
    }
    fn size(&self) -> usize {
        self.size
    }

    fn get_data(&mut self) -> (Vec<Tensor>, Vec<Id>) {
        let size = self.size();
        let records = std::mem::replace(&mut self.records, Vec::with_capacity(size));
        let ids = std::mem::replace(&mut self.ids, Vec::with_capacity(size));
        (records, ids)
    }
}
