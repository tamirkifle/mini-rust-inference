//! Cache-blocked (tiled) matrix multiplication — commit 5.2.
//!
//! # Strategy
//!
//! The naive `i-p-j` loop stalls on L1/L2 cache misses for large matrices
//! because entire rows of `B` stop fitting in cache as `N` grows.  Tiling
//! partitions the three loop dimensions into blocks that each fit comfortably
//! in the L1 data cache, so every element loaded is reused `BLOCK` times
//! before being evicted.
//!
//! ## Tile sizes
//!
//! The default tile is **32 × 32 × 32** (i-tile × p-tile × j-tile):
//! - One 32×32 f32 tile = 4 096 bytes = 4 KiB.
//! - Three tiles (A-panel, B-panel, C-panel) = 12 KiB, well inside a typical
//!   32 KiB L1D cache.
//! - Adjust `BLOCK` via [`matmul_blocked_with_block_size`] to tune for your
//!   microarchitecture.
//!
//! ## Loop order inside a tile
//!
//! The inner loops still follow `i-p-j` (register-friendly scalar hoist of
//! `a[i,p]`), identical to the naive kernel, so the correctness argument is
//! the same.
//!
//! ## Non-contiguous inputs
//!
//! Inputs are forced contiguous via `Cow` before tiling, identical to the
//! naive kernel.  This is a one-time copy and does not affect asymptotic
//! complexity.

use std::borrow::Cow;

use crate::tensor::{Result, Tensor, TensorError};

// ── public constants ───────────────────────────────────────────────────────

/// Default tile (block) size along each dimension.
///
/// 32 × 32 × 32 → three tiles of 4 KiB each ≈ 12 KiB total, fits in L1D.
pub const DEFAULT_BLOCK_SIZE: usize = 32;

// ── public entry-points ────────────────────────────────────────────────────

/// Cache-blocked GEMM with the default 32×32×32 tile size.
///
/// Drop-in replacement for [`super::matmul_naive`].  Results must match
/// the naive kernel within `1e-4` relative error (floating-point reordering
/// may shift the last ULP).
///
/// # Errors
///
/// Same error conditions as `matmul_naive`.
#[must_use = "returns a new tensor; result is not stored in-place"] // CHANGED
pub fn matmul_blocked(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    matmul_blocked_with_block_size(a, b, DEFAULT_BLOCK_SIZE)
}

/// Cache-blocked GEMM with a caller-specified tile size.
///
/// Useful for benchmarking different tile sizes or for small matrices where
/// a smaller tile avoids unnecessary overhead.
///
/// # Arguments
///
/// * `block` – tile size along every dimension (i, p, j).  Must be ≥ 1.
///
/// # Errors
///
/// Returns [`TensorError::InvalidShape`] if `block == 0`, in addition to
/// the usual 2-D / inner-dim checks.
#[must_use = "returns a new tensor; result is not stored in-place"] // CHANGED
pub fn matmul_blocked_with_block_size(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    block: usize,
) -> Result<Tensor<f32>> {
    // ── validation ─────────────────────────────────────────────────────────
    if block == 0 {
        return Err(TensorError::InvalidShape {
            reason: "matmul_blocked: block size must be ≥ 1".into(),
        });
    }
    if a.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_blocked: `a` must be 2-D, got {}D (shape {:?})",
                a.ndim(),
                a.dims()
            ),
        });
    }
    if b.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_blocked: `b` must be 2-D, got {}D (shape {:?})",
                b.ndim(),
                b.dims()
            ),
        });
    }

    let [m, k] = [a.dims()[0], a.dims()[1]]; // CHANGED: destructure
    let [k2, n] = [b.dims()[0], b.dims()[1]];

    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, k],
            got: vec![k2, n],
        });
    }

    // ── contiguity gate ────────────────────────────────────────────────────
    // CHANGED: same Cow pattern as naive — one-time copy for non-contiguous.
    let a_c: Cow<Tensor<f32>> = if a.is_contiguous() {
        Cow::Borrowed(a)
    } else {
        Cow::Owned(a.contiguous())
    };
    let b_c: Cow<Tensor<f32>> = if b.is_contiguous() {
        Cow::Borrowed(b)
    } else {
        Cow::Owned(b.contiguous())
    };

    let a_data = a_c.as_slice();
    let b_data = b_c.as_slice();

    // ── tiled i-p-j loop ───────────────────────────────────────────────────
    // CHANGED: outer tile loops advance in steps of `block`; inner loops
    //          are clamped to the actual matrix bounds (handles non-multiple sizes).
    let mut c_data = vec![0.0_f32; m * n];

    let mut ii = 0;
    while ii < m {
        let i_end = (ii + block).min(m); // CHANGED: clamp to matrix bound

        let mut pp = 0;
        while pp < k {
            let p_end = (pp + block).min(k);

            let mut jj = 0;
            while jj < n {
                let j_end = (jj + block).min(n);

                // ── micro-kernel: operate on the (i_end-ii) × (p_end-pp) × (j_end-jj) tile
                for i in ii..i_end {
                    for p in pp..p_end {
                        let a_ip = a_data[i * k + p]; // CHANGED: scalar hoist
                        for j in jj..j_end {
                            c_data[i * n + j] += a_ip * b_data[p * n + j];
                        }
                    }
                }

                jj += block;
            }
            pp += block;
        }
        ii += block;
    }

    Tensor::from_vec(c_data, vec![m, n])
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul::matmul_naive;

    const EPS_REL: f32 = 1e-4; // relative tolerance vs naive

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    fn close_slice(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
    }

    /// Assert blocked result matches naive within relative tolerance.
    fn assert_matches_naive(a: &Tensor<f32>, b: &Tensor<f32>) {
        let expected = matmul_naive(a, b).unwrap();
        let got = matmul_blocked(a, b).unwrap();

        assert_eq!(got.dims(), expected.dims(), "shape mismatch");

        for (i, (g, e)) in got.as_slice().iter().zip(expected.as_slice()).enumerate() {
            let denom = g.abs().max(e.abs()).max(1.0);
            let rel_err = (g - e).abs() / denom;
            assert!(
                rel_err < EPS_REL,
                "element {i}: blocked={g} naive={e} rel_err={rel_err:.2e}"
            );
        }
    }

    // ── correctness: small exact cases ────────────────────────────────────

    #[test]
    fn test_2x2_identity() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = matmul_blocked(&a, &id).unwrap();
        assert!(close_slice(c.as_slice(), &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_2x2_known_result() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = matmul_blocked(&a, &b).unwrap();
        assert!(close_slice(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn test_1x1() {
        let a = Tensor::from_vec(vec![3.0_f32], vec![1, 1]).unwrap();
        let b = Tensor::from_vec(vec![7.0_f32], vec![1, 1]).unwrap();
        let c = matmul_blocked(&a, &b).unwrap();
        assert!(close(c.as_slice()[0], 21.0));
    }

    // ── correctness: matches naive ─────────────────────────────────────────

    #[test]
    fn test_matches_naive_square_32() {
        // CHANGED: exact multiple of default block — simplest tiling case
        let n = 32_usize;
        let a = Tensor::from_vec((0..(n * n)).map(|i| i as f32 * 0.01).collect(), vec![n, n]).unwrap();
        let b = Tensor::from_vec((0..(n * n)).map(|i| (n * n - i) as f32 * 0.005).collect(), vec![n, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_non_multiple_of_block() {
        // CHANGED: 50×70 — not a multiple of 32, exercises boundary clamping
        let (m, k, n) = (50, 70, 40);
        let a = Tensor::from_vec((0..(m * k)).map(|i| i as f32 * 0.003).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..(k * n)).map(|i| (k * n - i) as f32 * 0.002).collect(), vec![k, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_rectangular() {
        #[rustfmt::skip]
        let a = Tensor::from_vec(vec![
            1.0_f32, 2.0, 3.0,
            4.0,     5.0, 6.0,
        ], vec![2, 3]).unwrap();
        #[rustfmt::skip]
        let b = Tensor::from_vec(vec![
             7.0_f32,  8.0,  9.0, 10.0,
            11.0,     12.0, 13.0, 14.0,
            15.0,     16.0, 17.0, 18.0,
        ], vec![3, 4]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_larger_non_square() {
        // CHANGED: 100×80 @ 80×60 — spans multiple tiles in all three dims
        let (m, k, n) = (100, 80, 60);
        let a = Tensor::from_vec((0..(m * k)).map(|i| i as f32 * 0.001).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..(k * n)).map(|i| (k * n - i) as f32 * 0.001).collect(), vec![k, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    // ── block-size variants ────────────────────────────────────────────────

    #[test]
    fn test_block_size_1_equals_naive() {
        // CHANGED: block=1 degenerates to scalar updates — still correct
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]).unwrap();
        let expected = matmul_naive(&a, &b).unwrap();
        let got = matmul_blocked_with_block_size(&a, &b, 1).unwrap();
        assert!(close_slice(got.as_slice(), expected.as_slice()));
    }

    #[test]
    fn test_block_size_larger_than_matrix() {
        // CHANGED: block > matrix dims — clamping makes it equivalent to naive
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let expected = matmul_naive(&a, &b).unwrap();
        let got = matmul_blocked_with_block_size(&a, &b, 1024).unwrap();
        assert!(close_slice(got.as_slice(), expected.as_slice()));
    }

    // ── non-contiguous inputs ──────────────────────────────────────────────

    #[test]
    fn test_non_contiguous_transpose_a() {
        let a = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let a_t = a.transpose(0, 1).unwrap();
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = matmul_blocked(&a_t, &id).unwrap();
        assert!(close_slice(c.as_slice(), &[1.0, 2.0, 3.0, 4.0]));
    }

    // ── error handling ─────────────────────────────────────────────────────

    #[test]
    fn test_shape_mismatch_error() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3, 1]).unwrap();
        assert!(matches!(
            matmul_blocked(&a, &b),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_zero_block_size_error() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert!(matches!(
            matmul_blocked_with_block_size(&a, &b, 0),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_non_2d_error() {
        let a = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        assert!(matches!(
            matmul_blocked(&a, &b),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
