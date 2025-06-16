use std::fmt::Debug;

use log::debug;
use serde::Serialize;

use crate::{
    config::CONFIG,
    errors::BuildError,
    types::{Array, ArrayNumType, ArraySlice},
    Id,
};

#[derive(Debug, Serialize)]
pub(crate) struct Bucket {
    id: String,
    records: Vec<ArrayNumType>,
    ids: Vec<Id>,
    size: usize,
    input_shape: usize,
    current_size: usize,
    is_dynamic: bool,
}

impl Bucket {
    fn new(id: String, size: usize, input_shape: usize, is_dynamic: bool) -> Self {
        Self {
            id,
            records: Vec::with_capacity(size * input_shape),
            ids: Vec::with_capacity(size),
            size,
            input_shape,
            current_size: size,
            is_dynamic,
        }
    }

    fn record(&self, i: usize) -> &ArraySlice {
        let start = i * self.input_shape;
        let end = start + self.input_shape;
        &self.records[start..end]
    }

    pub fn search(&self, query: &ArraySlice, k: usize) -> (Vec<Id>, Vec<ArrayNumType>) {
        assert!(k > 0);
        let mut distances = self
            .ids
            .iter()
            .enumerate()
            .map(|(i, id)| (id, euclidean_distance_new(query, self.record(i))))
            .collect::<Vec<_>>();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        distances.truncate(k);
        distances.into_iter().unzip()
    }

    pub fn insert(&mut self, record: Array, id: Id) {
        if !self.has_space(1) {
            self.resize(1)
        }
        if self.occupied() % CONFIG.skip_insert_log == 0 {
            debug!(size=self.size(), occupied=self.occupied(), id=self.id(); "bucket:insert");
        }
        self.records.extend(record);
        self.ids.push(id);
    }

    fn resize(&mut self, new_n_objects: usize) {
        assert!(self.is_dynamic);
        assert!(new_n_objects > 0);
        let resize_factor =
            ((new_n_objects + self.current_size) as f64 / self.current_size as f64).ceil() as usize;
        assert!(resize_factor > 1);
        let to_add_size = self.size * (resize_factor - 1);
        assert!(to_add_size > 0);
        // debug!(to_add_size=to_add_size, id=self.id; "bucket:rescale");
        self.records.reserve(to_add_size * self.input_shape);
        self.ids.reserve(to_add_size);
        self.current_size += to_add_size;
    }

    pub fn get_data(&mut self) -> (Vec<Array>, Vec<Id>) {
        let size = self.size();
        let mut records = std::mem::replace(
            &mut self.records,
            Vec::with_capacity(size * self.input_shape),
        );
        assert!(records.len() % self.input_shape == 0);
        assert!(records.len() == self.occupied() * self.input_shape);
        let records = (0..self.occupied())
            .map(|_| records.drain(..self.input_shape).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let ids = std::mem::replace(&mut self.ids, Vec::with_capacity(size));
        (records, ids)
    }

    pub fn has_space(&self, count: usize) -> bool {
        self.occupied() + count <= self.current_size
    }

    pub fn occupied(&self) -> usize {
        self.ids.len()
    }

    pub fn size(&self) -> usize {
        self.size
    }

    fn id(&self) -> &str {
        &self.id
    }
}

#[derive(Debug, Default)]
pub(crate) struct BucketBuilder {
    input_shape: Option<usize>,
    id: Option<String>,
    size: Option<usize>,
    is_dynamic: bool,
}

impl BucketBuilder {
    pub fn input_shape(&mut self, input_shape: usize) -> &mut Self {
        self.input_shape = Some(input_shape);
        self
    }

    pub fn id(&mut self, id: String) -> &mut Self {
        self.id = Some(id);
        self
    }

    pub fn size(&mut self, size: usize) -> &mut Self {
        self.size = Some(size);
        self
    }

    pub fn is_dynamic(&mut self, is_dynamic: bool) -> &mut Self {
        self.is_dynamic = is_dynamic;
        self
    }

    pub fn build(&self) -> Result<Bucket, BuildError> {
        let size = self.size.ok_or(BuildError::MissingAttribute)?;
        let input_shape = self.input_shape.ok_or(BuildError::MissingAttribute)?;
        let id = self.id.clone().ok_or(BuildError::MissingAttribute)?;
        Ok(Bucket::new(id, size, input_shape, self.is_dynamic))
    }
}

fn euclidean_distance_new(a: &ArraySlice, b: &ArraySlice) -> ArrayNumType {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
