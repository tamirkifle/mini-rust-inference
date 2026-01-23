//! Matrix-vector multiplication — commit 5.3.
//!
//! # Why a dedicated matvec?
//!
//! During autoregressive inference the batch size is almost always 1: we
//! multiply a weight matrix `W [M, K]` by a single activation vector `x [K]`
//! to produce `y [M]`.  Falling back to a general GEMM for this case wastes
//! work: the N=1 outer loop never tiles, the output has no j-dimension to
//! vectorise across, and the GEMM bookkeeping (three tile loops, extra bounds
//! checks) adds overhead for what is essentially `M` independent dot products.
//!
//! This module provides:
//!
//! | Function | Signature | Notes |
//! |----------|-----------|-------|
//! | [`matvec`] | `(W: [M,K], x: [K]) → y: [M]` | 1-D input/output |
//! | [`matvec_2d`] | `(W: [M,K], x: [K,1]) → y: [M,1]` | 2-D column vector |
//!
//! Both functions share the same inner kernel: a simple row-dot-product loop
//! that the compiler can auto-vectorise with `-O2` / `--release`.
//!
//! ## Fallback relationship
//!
//! `matvec(W, x)` is mathematically equivalent to
//! `matmul_naive(W, x.reshape([K,1]))?.reshape([M])`, and the tests verify
//! this.  The dedicated path avoids the reshape + GEMM overhead.

use std::borrow::Cow;

use crate::tensor::{Result, Tensor, TensorError};

// ── public API ─────────────────────────────────────────────────────────────

/// Multiply matrix `w` by vector `x`, returning a 1-D result vector.
///
/// # Arguments
///
/// * `w` – 2-D weight tensor of shape `[M, K]`
/// * `x` – 1-D input vector of shape `[K]`
///
/// # Returns
///
/// A contiguous `Tensor<f32>` of shape `[M]`.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `w` is not 2-D or `x` is not 1-D.
/// * [`TensorError::ShapeMismatch`] if `w.dims()[1] != x.dims()[0]`.
#[must_use = "returns a new tensor"] // CHANGED
pub fn matvec(w: &Tensor<f32>, x: &Tensor<f32>) -> Result<Tensor<f32>> {
    // ── validation ─────────────────────────────────────────────────────────
    if w.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matvec: `w` must be 2-D, got {}D (shape {:?})",
                w.ndim(), w.dims()
            ),
        });
    }
    if x.ndim() != 1 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matvec: `x` must be 1-D, got {}D (shape {:?})",
                x.ndim(), x.dims()
            ),
        });
    }

    let m = w.dims()[0];
    let k = w.dims()[1];
    let kx = x.dims()[0];

    if k != kx {
        return Err(TensorError::ShapeMismatch {
            expected: vec![k],
            got: vec![kx],
        });
    }

    // ── contiguity gate ────────────────────────────────────────────────────
    // CHANGED: force contiguous so we can use direct slice indexing.
    let w_c: Cow<Tensor<f32>> = if w.is_contiguous() { Cow::Borrowed(w) } else { Cow::Owned(w.contiguous()) };
    let x_c: Cow<Tensor<f32>> = if x.is_contiguous() { Cow::Borrowed(x) } else { Cow::Owned(x.contiguous()) };

    let w_data = w_c.as_slice();
    let x_data = x_c.as_slice();

    // ── kernel: M row-dot-products ─────────────────────────────────────────
    // CHANGED: each row of W is contiguous [K] floats; dot with x_data.
    let mut y = vec![0.0_f32; m];
    for i in 0..m {
        let row = &w_data[i * k..(i + 1) * k]; // CHANGED: zero-copy row slice
        let mut acc = 0.0_f32;
        for j in 0..k {
            acc += row[j] * x_data[j];
        }
        y[i] = acc;
    }

    Tensor::from_vec(y, vec![m])
}

/// Multiply matrix `w` by a 2-D column vector `x`, returning a 2-D result.
///
/// Convenience wrapper for callers that prefer to keep everything 2-D.
///
/// # Arguments
///
/// * `w` – 2-D weight tensor of shape `[M, K]`
/// * `x` – 2-D column vector of shape `[K, 1]`
///
/// # Returns
///
/// A contiguous `Tensor<f32>` of shape `[M, 1]`.
///
/// # Errors
///
/// Same as [`matvec`], plus [`TensorError::InvalidShape`] if `x` is not shape
/// `[*, 1]`.
#[must_use = "returns a new tensor"] // CHANGED
pub fn matvec_2d(w: &Tensor<f32>, x: &Tensor<f32>) -> Result<Tensor<f32>> {
    if x.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matvec_2d: `x` must be 2-D [K,1], got {}D (shape {:?})",
                x.ndim(), x.dims()
            ),
        });
    }
    if x.dims()[1] != 1 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matvec_2d: `x` must have shape [K,1], got {:?}",
                x.dims()
            ),
        });
    }

    // CHANGED: squeeze the trailing dim, delegate to matvec, then unsqueeze.
    let k = x.dims()[0];
    let x_1d = x.reshape(vec![k])?;
    let y_1d = matvec(w, &x_1d)?;
    let m = y_1d.numel();
    y_1d.reshape(vec![m, 1])
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul::matmul_naive;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
    fn close_slice(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
    }

    // ── basic correctness ──────────────────────────────────────────────────

    #[test]
    fn test_identity_matrix() {
        // I @ x = x
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let x  = Tensor::from_vec(vec![3.0_f32, 7.0], vec![2]).unwrap();
        let y  = matvec(&id, &x).unwrap();
        assert_eq!(y.dims(), &[2]);
        assert!(close_slice(y.as_slice(), &[3.0, 7.0]));
    }

    #[test]
    fn test_known_2x3() {
        // [[1,2,3],[4,5,6]] @ [1,2,3]
        // row0 = 1+4+9 = 14
        // row1 = 4+10+18 = 32
        let w = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3]).unwrap();
        let y = matvec(&w, &x).unwrap();
        assert_eq!(y.dims(), &[2]); // CHANGED: [M] = [2]
        assert!(close_slice(y.as_slice(), &[14.0, 32.0]));
    }

    #[test]
    fn test_single_row() {
        // [1x4] @ [4] -> [1]
        let w = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![1, 4]).unwrap();
        let x = Tensor::from_vec(vec![1.0_f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        let y = matvec(&w, &x).unwrap();
        assert_eq!(y.dims(), &[1]);
        assert!(close(y.as_slice()[0], 10.0));
    }

    #[test]
    fn test_zeros() {
        let w = Tensor::<f32>::zeros(vec![4, 8]);
        let x = Tensor::from_vec(vec![1.0_f32; 8], vec![8]).unwrap();
        let y = matvec(&w, &x).unwrap();
        assert!(y.as_slice().iter().all(|&v| close(v, 0.0)));
    }

    // ── matches matmul_naive ───────────────────────────────────────────────

    /// matvec(W, x) must equal matmul_naive(W, x_col).squeeze()
    fn assert_matches_gemm(w_data: &[f32], w_shape: (usize, usize), x_data: &[f32]) {
        let (m, k) = w_shape;
        let w  = Tensor::from_vec(w_data.to_vec(), vec![m, k]).unwrap();
        let x  = Tensor::from_vec(x_data.to_vec(), vec![k]).unwrap();
        let mv = matvec(&w, &x).unwrap();

        // GEMM reference: x as column vector
        let x_col = Tensor::from_vec(x_data.to_vec(), vec![k, 1]).unwrap();
        let mm    = matmul_naive(&w, &x_col).unwrap(); // [M, 1]

        assert_eq!(mv.dims(), &[m]);
        assert_eq!(mm.dims(), &[m, 1]);
        for (i, (mv_v, mm_v)) in mv.as_slice().iter().zip(mm.as_slice()).enumerate() {
            assert!(close(*mv_v, *mm_v), "row {i}: matvec={mv_v} gemm={mm_v}");
        }
    }

    #[test]
    fn test_matches_gemm_square_4() {
        let n = 4_usize;
        let w: Vec<f32> = (0..(n*n)).map(|i| i as f32 * 0.1).collect();
        let x: Vec<f32> = (0..n).map(|i| i as f32 + 1.0).collect();
        assert_matches_gemm(&w, (n, n), &x);
    }

    #[test]
    fn test_matches_gemm_rectangular() {
        // CHANGED: non-square — common in LLM weight projections (M != K)
        let (m, k) = (64, 32);
        let w: Vec<f32> = (0..(m*k)).map(|i| (i as f32).sin()).collect();
        let x: Vec<f32> = (0..k).map(|i| (i as f32).cos()).collect();
        assert_matches_gemm(&w, (m, k), &x);
    }

    #[test]
    fn test_matches_gemm_tall() {
        // CHANGED: tall matrix (M >> K) — the feed-forward up-projection case
        let (m, k) = (256, 64);
        let w: Vec<f32> = (0..(m*k)).map(|i| i as f32 * 1e-3).collect();
        let x: Vec<f32> = (0..k).map(|i| i as f32 * 0.5).collect();
        assert_matches_gemm(&w, (m, k), &x);
    }

    // ── non-contiguous inputs ──────────────────────────────────────────────

    #[test]
    fn test_non_contiguous_weight() {
        // CHANGED: transpose makes w non-contiguous; matvec must still work
        // w_orig [2,3], w_T [3,2]; matvec(w_T, x[2]) -> y[3]
        let w_orig = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let w_t = w_orig.transpose(0, 1).unwrap(); // [3,2], non-contiguous
        let x   = Tensor::from_vec(vec![1.0_f32, 0.0], vec![2]).unwrap();
        let y   = matvec(&w_t, &x).unwrap();
        // w_T rows: [1,4], [2,5], [3,6]  dot  [1,0] -> [1, 2, 3]
        assert_eq!(y.dims(), &[3]);
        assert!(close_slice(y.as_slice(), &[1.0, 2.0, 3.0]));
    }

    // ── matvec_2d ──────────────────────────────────────────────────────────

    #[test]
    fn test_matvec_2d_known() {
        let w = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3, 1]).unwrap();
        let y = matvec_2d(&w, &x).unwrap();
        assert_eq!(y.dims(), &[2, 1]);
        assert!(close_slice(y.as_slice(), &[14.0, 32.0]));
    }

    #[test]
    fn test_matvec_2d_matches_1d() {
        let (m, k) = (16, 8);
        let w: Vec<f32> = (0..(m*k)).map(|i| i as f32 * 0.05).collect();
        let x: Vec<f32> = (0..k).map(|i| i as f32).collect();

        let wt  = Tensor::from_vec(w.clone(), vec![m, k]).unwrap();
        let x1d = Tensor::from_vec(x.clone(), vec![k]).unwrap();
        let x2d = Tensor::from_vec(x, vec![k, 1]).unwrap();

        let y1d = matvec(&wt, &x1d).unwrap();
        let y2d = matvec_2d(&wt, &x2d).unwrap();

        assert_eq!(y2d.dims(), &[m, 1]);
        assert!(close_slice(y1d.as_slice(), y2d.as_slice()));
    }

    // ── error handling ─────────────────────────────────────────────────────

    #[test]
    fn test_w_not_2d() {
        let w = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 2, 2]).unwrap();
        let x = Tensor::from_vec(vec![1.0_f32; 4], vec![4]).unwrap();
        assert!(matches!(matvec(&w, &x), Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn test_x_not_1d() {
        let w = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let x = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        assert!(matches!(matvec(&w, &x), Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn test_shape_mismatch() {
        let w = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap();
        let x = Tensor::from_vec(vec![1.0_f32; 4], vec![4]).unwrap();
        assert!(matches!(matvec(&w, &x), Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_matvec_2d_x_not_column() {
        let w = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap();
        let x = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap(); // not [K,1]
        assert!(matches!(matvec_2d(&w, &x), Err(TensorError::InvalidShape { .. })));
    }
}
