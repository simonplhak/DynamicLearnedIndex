use crate::types::{Array, ArraySlice};
use log::info;
use measure_time_macro::log_time;
use rand::rngs::SmallRng;
use rand::SeedableRng;

#[log_time]
pub(crate) fn sample(queries: &ArraySlice, n: usize, shape: usize) -> Array {
    // todo possibility to return ArraySlice to avoid cloning??
    info!(n=n ;"sampling");
    assert!(!queries.is_empty(), "Queries cannot be empty");
    assert!(queries.len().is_multiple_of(shape));
    assert!(n > 0, "Sample size must be greater than zero");
    let num_queries = queries.len() / shape;
    if n >= num_queries {
        return queries.to_vec();
    }

    let mut rng = SmallRng::seed_from_u64(42); // Fixed seed for consistency, or use: SmallRng::from_entropy()
    let idxs = rand::seq::index::sample(&mut rng, num_queries, n).into_vec();

    // Pre-allocate exactly what we need:
    let mut out = Array::with_capacity(n * shape);

    // Bulk-copy each selected slice:
    for &i in &idxs {
        let start = i * shape;
        let end = start + shape;
        let slice = &queries[start..end];
        out.extend_from_slice(slice);
    }

    out
}

/// Selects the sample size based on the total number of objects and number of clusters.
/// See: https://github.com/facebookresearch/faiss/wiki/FAQ/5e5b0a1d95b4b12fc3fc92700e8e717c01ce7943#how-many-training-points-do-i-need-for-k-means
pub(crate) fn select_sample_size(k: usize, total_objects: usize, sample_threshold: usize) -> usize {
    let base_size = k * 40;
    if base_size < sample_threshold {
        total_objects.min(sample_threshold)
    } else {
        total_objects.min(base_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ArrayNumType;

    #[test]
    fn test_sample_basic_functionality() {
        let queries = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let sampled = sample(&queries, 3, 3);

        // Should return 3 samples with 3 elements each
        assert_eq!(sampled.len(), 9);

        // All values should be from the original queries
        for value in &sampled {
            assert!((1.0..=15.0).contains(value));
        }
    }

    #[test]
    fn test_sample_single_query() {
        let queries = vec![1.0, 2.0, 3.0];
        let sampled = sample(&queries, 1, 3);

        assert_eq!(sampled.len(), 3);
        assert_eq!(sampled, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sample_all_queries() {
        let queries = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let original_count = queries.len() / 3;
        let sampled = sample(&queries, original_count, 3);

        // Should return all queries
        assert_eq!(sampled.len(), queries.len());
    }

    #[test]
    fn test_sample_more_than_available() {
        let queries = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let sampled = sample(&queries, 10, 3); // More than 5 available

        // Should return all available queries
        assert_eq!(sampled.len(), queries.len());
    }

    #[test]
    fn test_sample_preserves_array_structure() {
        let queries = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let sampled = sample(&queries, 2, 2);

        // Should have 2 * 2 = 4 elements
        assert_eq!(sampled.len(), 4);

        // Check that we have pairs of values from original arrays
        for chunk in sampled.chunks(2) {
            assert_eq!(chunk.len(), 2);
            // Each pair should be consecutive values from original data
            // (since we're sampling from a flattened array with shape=2)
            let diff = chunk[1] - chunk[0];
            assert!((diff - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_sample_different_sized_shape() {
        let queries = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sampled = sample(&queries, 1, 4);

        assert_eq!(sampled.len(), 4);
    }

    #[test]
    fn test_sample_randomness() {
        let queries = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];

        // Reset RNG for predictable sequence
        let sample1 = sample(&queries, 3, 3);

        // Reset RNG again to get same sequence
        let sample2 = sample(&queries, 3, 3);

        // With fixed seed, samples should be identical
        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_sample_maintains_value_integrity() {
        let queries = vec![1.1, 2.2, 3.3, 4.4, 5.5, 6.6];
        let sampled = sample(&queries, 2, 3);

        assert_eq!(sampled.len(), 6);

        // Check that all sampled values exist in original data
        for value in &sampled {
            assert!(queries.contains(value));
        }
    }

    #[test]
    #[should_panic(expected = "Queries cannot be empty")]
    fn test_sample_empty_queries_panics() {
        let queries: Array = vec![];
        sample(&queries, 1, 1);
    }

    #[test]
    #[should_panic(expected = "Sample size must be greater than zero")]
    fn test_sample_zero_size_panics() {
        let queries = vec![1.0, 2.0, 3.0];
        sample(&queries, 0, 3);
    }

    #[test]
    fn test_sample_single_element_shape() {
        let queries = vec![1.0, 2.0, 3.0, 4.0];
        let sampled = sample(&queries, 2, 1);

        assert_eq!(sampled.len(), 2);
        for value in &sampled {
            assert!((1.0..=4.0).contains(value));
        }
    }

    #[test]
    fn test_sample_large_arrays() {
        let queries: Array = (0..300).map(|i| i as ArrayNumType).collect();
        let sampled = sample(&queries, 2, 100);

        assert_eq!(sampled.len(), 200); // 2 * 100
    }

    #[test]
    fn test_sample_deterministic_with_fixed_seed() {
        // Reset RNG for predictable results

        let queries = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
        ];
        let first_run = sample(&queries, 3, 3);

        // Reset RNG again
        let second_run = sample(&queries, 3, 3);

        // With fixed seed, results should be identical
        assert_eq!(first_run, second_run);
    }
}
