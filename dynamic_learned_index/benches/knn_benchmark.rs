#![feature(test)]

extern crate test;
use simsimd::SpatialSimilarity;
use std::arch::x86_64::*;

const INPUT_SHAPE: usize = 768;
const LINES: usize = 8;

/// original
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// from https://blog.lancedb.com/my-simd-is-faster-than-yours-fb2989bf25e7/
#[inline]
fn l2_f32(from: &[f32], to: &[f32]) -> f32 {
    unsafe {
        // Get the potion of the vector that is aligned to 32 bytes.
        let len = from.len() / 8 * LINES;
        let mut sums = _mm256_setzero_ps();
        for i in (0..len).step_by(8) {
            let left = _mm256_loadu_ps(from.as_ptr().add(i));
            let right = _mm256_loadu_ps(to.as_ptr().add(i));
            let sub = _mm256_sub_ps(left, right);
            // sum = sub * sub + sum
            sums = _mm256_fmadd_ps(sub, sub, sums);
        }
        // Shift and add vector, until only 1 value left.
        // sums = [x0-x7], shift = [x4-x7]
        let mut shift = _mm256_permute2f128_ps(sums, sums, 1);
        // [x0+x4, x1+x5, ..]
        sums = _mm256_add_ps(sums, shift);
        shift = _mm256_permute_ps(sums, 14);
        sums = _mm256_add_ps(sums, shift);
        sums = _mm256_hadd_ps(sums, sums);
        let mut results: [f32; 8] = [0f32; 8];
        _mm256_storeu_ps(results.as_mut_ptr(), sums);

        results[0] += euclidean_distance(&from[len..], &to[len..]);
        results[0]
    }
}

/// via simsimd crate
fn l2_simsimd(from: &[f32], to: &[f32]) -> f32 {
    f32::l2(from, to).unwrap() as f32
}

#[bench]
fn bench_euclidean_distance(b: &mut test::Bencher) {
    let v1 = vec![0.1f32; INPUT_SHAPE];
    let v2 = vec![0.2f32; INPUT_SHAPE];

    b.iter(|| {
        test::black_box(euclidean_distance(&v1, &v2));
    });
}

#[bench]
fn bench_euclidean_distance_simd(b: &mut test::Bencher) {
    let v1 = vec![0.1f32; INPUT_SHAPE];
    let v2 = vec![0.2f32; INPUT_SHAPE];

    b.iter(|| {
        test::black_box(l2_f32(&v1, &v2));
    });
}

#[bench]
fn bench_l2_simsimd(b: &mut test::Bencher) {
    let v1 = vec![0.1f32; INPUT_SHAPE];
    let v2 = vec![0.2f32; INPUT_SHAPE];

    b.iter(|| {
        test::black_box(l2_simsimd(&v1, &v2));
    });
}
