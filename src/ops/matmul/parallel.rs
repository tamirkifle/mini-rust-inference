//! Row-parallel GEMM using rayon — commit 12.2.
//!
//! # Strategy
//!
//! Each output row `C[i, :]` is independent of all other output rows: it only
//! depends on the corresponding input row `A[i, :]` and the full weight matrix
//! `B`.  Rayon's `par_chunks_mut` assigns one row of `C` to each logical task,
//! which rayon then schedules across the available thread-pool workers.
//!
//! ```text
//! C[0, :]  ←── thread 0
//! C[1, :]  ←── thread 1
//! C[2, :]  ←── thread 2   (work-stealing if unequal load)
//!   …
//! C[M-1,:] ←── thread T-1
//! ```
//!
//! No synchronisation is required because each task writes to a non-overlapping
//! slice of `c_data`.  Inner loops follow the `i-p-j` register-scalar hoist
//! pattern from the blocked kernel.
//!
//! # When to use
//!
//! Prefer this over `matmul_blocked` when:
//! - `M` (number of rows / prompt tokens) > ~8 AND
//! - multiple CPU cores are available.
//!
//! For single-token decode (`M = 1`) there is nothing to parallelise; fall
//! back to `matmul_blocked` to avoid rayon thread-pool overhead.

use std::borrow::Cow;
use rayon::prelude::*;
use crate::tensor::{Result, Tensor, TensorError};

/// Row-parallel GEMM: `C = A @ B` where each output row is computed by an
/// independent rayon task.
///
/// # Errors
///
/// Same error conditions as `matmul_blocked`: both inputs must be 2-D and
/// their inner dimensions must match.
#[must_use = "returns a new tensor; result is not stored in-place"]
pub fn matmul_parallel(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    // ── validation ─────────────────────────────────────────────────────────
    if a.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_parallel: `a` must be 2-D, got {}D (shape {:?})",
                a.ndim(), a.dims()
            ),
        });
    }
    if b.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_parallel: `b` must be 2-D, got {}D (shape {:?})",
                b.ndim(), b.dims()
            ),
        });
    }

    let [m, k]  = [a.dims()[0], a.dims()[1]];
    let [k2, n] = [b.dims()[0], b.dims()[1]];

    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, k],
            got:      vec![k2, n],
        });
    }

    // ── contiguity gate ────────────────────────────────────────────────────
    let a_c: Cow<Tensor<f32>> = if a.is_contiguous() { Cow::Borrowed(a) } else { Cow::Owned(a.contiguous()) };
    let b_c: Cow<Tensor<f32>> = if b.is_contiguous() { Cow::Borrowed(b) } else { Cow::Owned(b.contiguous()) };

    let a_slice: &[f32] = a_c.as_slice();
    let b_slice: &[f32] = b_c.as_slice();

    // ── row-parallel kernel ────────────────────────────────────────────────
    // par_chunks_mut(n) splits c_data into M non-overlapping rows of width n.
    // Each rayon task receives exactly one row and has exclusive write access.
    let mut c_data = vec![0.0_f32; m * n];

    c_data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, c_row)| {
            // Compute C[i, :] = A[i, :] · B
            for p in 0..k {
                let a_ip = a_slice[i * k + p];
                for j in 0..n {
                    c_row[j] += a_ip * b_slice[p * n + j];
                }
            }
        });

    Tensor::from_vec(c_data, vec![m, n])
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul::matmul_blocked;

    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    fn assert_matches_blocked(a: &Tensor<f32>, b: &Tensor<f32>) {
        let expected = matmul_blocked(a, b).unwrap();
        let got      = matmul_parallel(a, b).unwrap();
        assert_eq!(got.dims(), expected.dims());
        for (i, (g, e)) in got.as_slice().iter().zip(expected.as_slice()).enumerate() {
            let denom   = g.abs().max(e.abs()).max(1.0);
            let rel_err = (g - e).abs() / denom;
            assert!(rel_err < 1e-4, "element {i}: par={g} blocked={e} rel_err={rel_err:.2e}");
        }
    }

    #[test]
    fn test_parallel_2x2_known_result() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = matmul_parallel(&a, &b).unwrap();
        assert!(close_slice(c.as_slice(), &[19.0, 22.0, 43.0, 50.0], 1e-5));
    }

    #[test]
    fn test_parallel_1x1() {
        let a = Tensor::from_vec(vec![3.0_f32], vec![1, 1]).unwrap();
        let b = Tensor::from_vec(vec![7.0_f32], vec![1, 1]).unwrap();
        let c = matmul_parallel(&a, &b).unwrap();
        assert!((c.as_slice()[0] - 21.0).abs() < 1e-5);
    }

    #[test]
    fn test_parallel_identity() {
        let a  = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c  = matmul_parallel(&a, &id).unwrap();
        assert!(close_slice(c.as_slice(), &[1.0, 2.0, 3.0, 4.0], 1e-5));
    }

    #[test]
    fn test_parallel_matches_blocked_square_32() {
        let n = 32_usize;
        let a = Tensor::from_vec((0..n*n).map(|i| i as f32 * 0.01).collect(), vec![n, n]).unwrap();
        let b = Tensor::from_vec((0..n*n).map(|i| (n*n - i) as f32 * 0.005).collect(), vec![n, n]).unwrap();
        assert_matches_blocked(&a, &b);
    }

    #[test]
    fn test_parallel_matches_blocked_rectangular() {
        let (m, k, n) = (50, 70, 40);
        let a = Tensor::from_vec((0..m*k).map(|i| i as f32 * 0.003).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..k*n).map(|i| (k*n - i) as f32 * 0.002).collect(), vec![k, n]).unwrap();
        assert_matches_blocked(&a, &b);
    }

    #[test]
    fn test_parallel_matches_blocked_100x80x60() {
        let (m, k, n) = (100, 80, 60);
        let a = Tensor::from_vec((0..m*k).map(|i| i as f32 * 0.001).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..k*n).map(|i| (k*n - i) as f32 * 0.001).collect(), vec![k, n]).unwrap();
        assert_matches_blocked(&a, &b);
    }

    #[test]
    fn test_parallel_non_contiguous_a() {
        let a   = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let a_t = a.transpose(0, 1).unwrap();
        let id  = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c   = matmul_parallel(&a_t, &id).unwrap();
        assert!(close_slice(c.as_slice(), &[1.0, 2.0, 3.0, 4.0], 1e-5));
    }

    #[test]
    fn test_parallel_shape_mismatch_rejected() {
        let a = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32; 3], vec![3, 1]).unwrap();
        assert!(matches!(matmul_parallel(&a, &b), Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_parallel_non_2d_a_rejected() {
        let a = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        assert!(matches!(matmul_parallel(&a, &b), Err(TensorError::InvalidShape { .. })));
    }

    /// Output rows must be independent: parallel produces same result as sequential.
    #[test]
    fn test_parallel_no_cross_row_contamination() {
        // Each output row of a 4×4 @ 4×4 must be independent.
        let a = Tensor::from_vec((0..16).map(|i| i as f32).collect(), vec![4, 4]).unwrap();
        let b = Tensor::from_vec((0..16).map(|i| (16 - i) as f32).collect(), vec![4, 4]).unwrap();
        let par  = matmul_parallel(&a, &b).unwrap();
        let seq  = matmul_blocked(&a, &b).unwrap();
        assert!(close_slice(par.as_slice(), seq.as_slice(), 1e-4));
    }
}
