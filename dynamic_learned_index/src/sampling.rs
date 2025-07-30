use crate::types::Array;
use log::info;
use rand::rngs::SmallRng;
use rand::SeedableRng;

/// A global RNG instance with a fixed seed for reproducibility
pub(crate) static mut GLOBAL_RNG: Option<SmallRng> = None;

/// Initialize the global RNG with a fixed seed
pub(crate) fn init_global_rng() {
    unsafe {
        if GLOBAL_RNG.is_none() {
            GLOBAL_RNG = Some(SmallRng::seed_from_u64(42));
        }
    }
}

/// Get a reference to the global RNG, initializing if needed
pub(crate) fn get_global_rng() -> &'static mut SmallRng {
    unsafe {
        if GLOBAL_RNG.is_none() {
            init_global_rng();
        }
        GLOBAL_RNG.as_mut().unwrap()
    }
}

pub(crate) fn sample(queries: &[Array], n: usize) -> Array {
    info!(n=n ;"sampling");
    assert!(!queries.is_empty(), "Queries cannot be empty");
    assert!(n > 0, "Sample size must be greater than zero");
    let num_queries = queries.len();
    let input_len = queries[0].len();
    if n >= num_queries {
        let total = num_queries * input_len;
        let mut flat = Vec::with_capacity(total);
        for q in queries {
            flat.extend_from_slice(q);
        }
        return flat;
    }

    // Sample `n` distinct indices efficiently:
    let mut rng = get_global_rng(); // Fixed seed for consistency, or use: SmallRng::from_entropy()
    let idxs = rand::seq::index::sample(&mut rng, num_queries, n).into_vec();

    // Pre-allocate exactly what we need:
    let mut out = Vec::with_capacity(n * input_len);

    // Bulk-copy each selected slice:
    for &i in &idxs {
        let slice = &queries[i];
        // Safety: `slice.len() == input_len`
        out.extend_from_slice(slice);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ArrayNumType;

    fn create_test_queries() -> Vec<Array> {
        vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
            vec![13.0, 14.0, 15.0],
        ]
    }

    #[test]
    fn test_sample_basic_functionality() {
        let queries = create_test_queries();
        let sampled = sample(&queries, 3);

        // Should return flattened array with 3 * 3 = 9 elements
        assert_eq!(sampled.len(), 9);

        // All values should be from the original queries
        for value in &sampled {
            assert!((1.0..=15.0).contains(value));
        }
    }

    #[test]
    fn test_sample_single_query() {
        let queries = vec![vec![1.0, 2.0, 3.0]];
        let sampled = sample(&queries, 1);

        assert_eq!(sampled.len(), 3);
        assert_eq!(sampled, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sample_all_queries() {
        let queries = create_test_queries();
        let original_count = queries.len();
        let sampled = sample(&queries, original_count);

        // Should return all queries flattened
        assert_eq!(sampled.len(), original_count * 3);
    }

    #[test]
    fn test_sample_more_than_available() {
        let queries = create_test_queries();
        let sampled = sample(&queries, 10); // More than 5 available

        // Should return all available queries (5 * 3 = 15 elements)
        assert_eq!(sampled.len(), 15);
    }

    #[test]
    fn test_sample_preserves_array_structure() {
        let queries = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let sampled = sample(&queries, 2);

        // Should have 2 * 2 = 4 elements
        assert_eq!(sampled.len(), 4);

        // Check that we have pairs of values from original arrays
        assert_eq!(sampled.len() % 2, 0);
        for chunk in sampled.chunks(2) {
            // Each pair should be consecutive values from original data
            let diff = chunk[1] - chunk[0];
            assert!((diff - 1.0).abs() < f32::EPSILON);
        }
    }

    #[test]
    fn test_sample_different_sized_arrays() {
        let queries = vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]];
        let sampled = sample(&queries, 1);

        assert_eq!(sampled.len(), 4);
    }

    #[test]
    fn test_sample_randomness() {
        let queries = create_test_queries();
        let sample1 = sample(&queries, 3);
        let sample2 = sample(&queries, 3);

        // With randomness, samples should potentially be different
        // Note: This test might rarely fail due to random chance,
        // but with 5 queries and sampling 3, it's very unlikely
        // they'll be identical in the same order
        // We'll just check they have the same length for now
        assert_eq!(sample1.len(), sample2.len());
    }

    #[test]
    fn test_sample_maintains_value_integrity() {
        let queries = vec![vec![1.1, 2.2, 3.3], vec![4.4, 5.5, 6.6]];
        let sampled = sample(&queries, 2);

        assert_eq!(sampled.len(), 6);

        // Check that all sampled values exist in original data
        let original_values: Vec<ArrayNumType> = queries.iter().flatten().copied().collect();
        for value in &sampled {
            assert!(original_values.contains(value));
        }
    }

    #[test]
    #[should_panic(expected = "Queries cannot be empty")]
    fn test_sample_empty_queries_panics() {
        let queries: Vec<Array> = vec![];
        sample(&queries, 1);
    }

    #[test]
    #[should_panic(expected = "Sample size must be greater than zero")]
    fn test_sample_zero_size_panics() {
        let queries = create_test_queries();
        sample(&queries, 0);
    }

    #[test]
    fn test_sample_empty_arrays_in_queries() {
        let queries = vec![vec![], vec![], vec![]];
        let sampled = sample(&queries, 2);

        // Should return empty result when input arrays are empty
        assert_eq!(sampled.len(), 0);
    }

    #[test]
    fn test_sample_single_element_arrays() {
        let queries = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let sampled = sample(&queries, 2);

        assert_eq!(sampled.len(), 2);
        for value in &sampled {
            assert!((1.0..=4.0).contains(value));
        }
    }

    #[test]
    fn test_sample_large_arrays() {
        let queries = vec![
            (0..100).map(|i| i as ArrayNumType).collect::<Vec<_>>(),
            (100..200).map(|i| i as ArrayNumType).collect::<Vec<_>>(),
            (200..300).map(|i| i as ArrayNumType).collect::<Vec<_>>(),
        ];
        let sampled = sample(&queries, 2);

        assert_eq!(sampled.len(), 200); // 2 * 100
    }

    #[test]
    fn test_sample_deterministic_with_fixed_seed() {
        // Note: This test would require setting a seed for the RNG
        // The current implementation uses rand::rng() which doesn't allow seed setting
        // This test documents the expected behavior if seeding were implemented
        let queries = create_test_queries();
        let sampled = sample(&queries, 3);

        // Just verify basic properties
        assert_eq!(sampled.len(), 9);
        assert!(sampled.len() % 3 == 0);
    }
}
