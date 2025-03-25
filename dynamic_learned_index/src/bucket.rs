use std::fmt;

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

pub(crate) trait Bucket {
    fn search(&self, query: &Tensor) -> (Tensor, Tensor);
    fn insert(&mut self, value: Tensor, id: Id);
    fn size(&self) -> usize;
    fn has_space(&self, count: usize) -> bool;
    fn occupied(&self) -> usize;
    fn get_data(&mut self) -> (Vec<Tensor>, Vec<Id>);
}

impl fmt::Debug for dyn Bucket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bucket") // todo
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

    pub fn build(&self) -> Result<Box<dyn Bucket>, BuildError> {
        let bucket = match self.bucket_type {
            BucketType::Static(size) => self.build_static_bucket(size),
        };
        bucket.map(|bucket| Box::new(bucket) as Box<dyn Bucket>)
    }

    fn build_static_bucket(&self, size: usize) -> Result<StaticBucket, BuildError> {
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;
        Ok(StaticBucket::new(size as usize, input_shape))
    }
}

#[derive(Debug)]
pub(crate) struct StaticBucket {
    records: Vec<Tensor>,
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
}

impl Bucket for StaticBucket {
    fn search(&self, query: &Tensor) -> (Tensor, Tensor) {
        todo!()
    }

    fn insert(&mut self, value: Tensor, id: Id) {
        debug_assert!(self.has_space(1), "Bucket is full size={}", self.size);
        debug_assert_eq!(value.size()[0], self.input_shape);
        println!("Inserting value: {:?}", value);
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
