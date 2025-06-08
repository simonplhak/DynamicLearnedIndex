use rand::seq::IteratorRandom;

use crate::types::Array;

pub(crate) fn sample(queries: &[Array], n: usize) -> Array {
    assert!(!queries.is_empty(), "Queries cannot be empty");
    assert!(n > 0, "Sample size must be greater than zero");

    let mut rng = rand::rng();
    let indices: Vec<usize> = (0..queries.len()).choose_multiple(&mut rng, n);
    let input_shape = queries[0].len();
    let n = indices.len();
    let shape = n * input_shape;
    let mut sampled_queries = Vec::with_capacity(shape);
    indices.iter().for_each(|i| {
        sampled_queries.extend_from_slice(&queries[*i]);
    });
    assert!(
        sampled_queries.len() == shape,
        "Sampled queries length mismatch: {} != {}",
        sampled_queries.len(),
        shape
    );
    sampled_queries
}
