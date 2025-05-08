// use faiss::index_factory;
use tch::Tensor;

pub(crate) mod bucket;
mod clustering;
mod config;
mod errors;
pub mod index;
mod model;
pub mod types;
mod util;
pub use index::{Index, IndexConfig};
pub use types::Id;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

pub fn print_tensor() {
    let t = Tensor::from_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
}

pub fn test() {
    // let knn_index = index_factory(d, description, metric)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
