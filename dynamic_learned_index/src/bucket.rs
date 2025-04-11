use std::fmt::Debug;

use log::info;
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
    pub fn search(&self, query: &Tensor, k: usize) -> (Vec<Id>, Vec<f64>) {
        match self {
            Bucket::Static(bucket) => bucket.search(query, k),
        }
    }

    pub fn insert(&mut self, value: Tensor, id: Id) {
        // todo take from env varaible
        // log only every 10th insert to avoid flooding
        if self.occupied() % 5 == 0 {
            info!(size=self.size(), occupied=self.occupied(), id=self.id(); "bucket:insert");
        }
        assert!(self.has_space(1), "Bucket is full size={}", self.size());
        match self {
            Bucket::Static(bucket) => bucket.insert(value, id),
        }
    }

    pub fn insert_many(&mut self, values: Vec<Tensor>, ids: Vec<Id>) {
        info!(size=self.size(), occupied=self.occupied(), id=self.id(), values_len=values.len(); "bucket:insert_many");
        assert!(values.len() == ids.len());
        assert!(
            self.has_space(values.len()),
            "Bucket is full size={}",
            self.size()
        );
        match self {
            Bucket::Static(bucket) => bucket.insert_many(values, ids),
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

    fn id(&self) -> &str {
        match self {
            Bucket::Static(bucket) => bucket.id(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct BucketBuilder {
    input_shape: Option<i64>,
    bucket_type: BucketType,
    id: Option<String>,
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

    pub fn id(&mut self, id: String) -> &mut Self {
        self.id = Some(id);
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
        let id = self.id.clone().ok_or(BuildError::MissingAttribute)?;
        Ok(StaticBucket::new(id, size, input_shape))
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct StaticBucket {
    id: String,
    #[serde(skip)]
    records: Vec<Tensor>,
    #[serde(skip)]
    ids: Vec<Id>,
    size: usize,
    input_shape: i64,
}

impl StaticBucket {
    pub fn new(id: String, size: usize, input_shape: i64) -> Self {
        let records = Vec::with_capacity(size);
        let ids = Vec::with_capacity(size);
        Self {
            id,
            records,
            ids,
            size,
            input_shape,
        }
    }

    fn search(&self, query: &Tensor, k: usize) -> (Vec<Id>, Vec<f64>) {
        assert!(self.occupied() > 0);
        assert!(self.occupied() >= k);
        assert!(self.size() > 0);
        let mut distances = self
            .records
            .iter()
            .zip(self.ids.iter())
            .map(|(x, id)| (id, euclidean_distance(query, x)))
            .collect::<Vec<_>>();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances.into_iter().unzip()
    }

    fn insert(&mut self, value: Tensor, id: Id) {
        assert_eq!(value.size()[0], self.input_shape);
        self.records.push(value);
        self.ids.push(id);
    }

    fn insert_many(&mut self, values: Vec<Tensor>, ids: Vec<Id>) {
        self.records.extend(values);
        self.ids.extend(ids);
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

    fn id(&self) -> &str {
        &self.id
    }
}

fn euclidean_distance(a: &Tensor, b: &Tensor) -> f64 {
    let diff = a - b;
    let sum = diff.square().sum(tch::Kind::Float).double_value(&[]);
    sum.sqrt()
}
