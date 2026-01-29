//! Numerically stable softmax — commit 6.4.
//!
//! # Algorithm
//!
//! Naive softmax `exp(xᵢ) / Σ exp(xⱼ)` overflows for large logits.
//! The standard fix: subtract the row maximum before exponentiation.
//!
//! ```text
//! m    = max(x)
//! sᵢ   = exp(xᵢ − m)
//! out  = s / sum(s)
//! ```
//!
//! This is mathematically identical to naive softmax but never overflows
//! (the largest exponent is always `exp(0) = 1`), and underflow at the
//! tails only zeroes out negligible probabilities.
//!
//! # Supported input shapes
//!
//! | Shape                  | dim=-1 default behaviour             |
//! |------------------------|--------------------------------------|
//! | `[n]`                  | single distribution over n classes   |
//! | `[batch, n]`           | independent softmax per row          |
//! | `[batch, heads, seq]`  | independent softmax per last dim     |
//!
//! The normalisation always runs over the **last** dimension.
//! Use [`softmax_dim`] to target a different axis.

use crate::tensor::{Result, Tensor, TensorError};

// ── private row kernel ──────────────────────────────────────────────────────

/// Stable softmax over a single contiguous slice, written in-place.
/// Returns early with all-zero output if the slice is empty.
#[inline]
fn softmax_row(row: &mut [f32]) { // CHANGED
    if row.is_empty() { return; }

    // 1. find max for numerical stability
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max); // CHANGED

    // 2. exp(x - max) in-place
    let mut sum = 0.0_f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v; // CHANGED: accumulate while warm in L1
    }

    // 3. normalise
    let inv_sum = 1.0 / sum; // CHANGED: one division, many multiplications
    for v in row.iter_mut() {
        *v *= inv_sum;
    }
}

// ── public API ──────────────────────────────────────────────────────────────

/// Apply softmax over the **last** dimension of `x`, returning a new tensor.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `x` is 0-D.
#[must_use = "returns a new tensor; the input is not modified"] // CHANGED
pub fn softmax(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    softmax_dim(x, x.ndim().wrapping_sub(1)) // CHANGED: last dim; wrapping_sub safe because 0-D is rejected
}

/// Apply softmax over a specific dimension `dim`, returning a new tensor.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `x` is 0-D or `dim >= x.ndim()`.
#[must_use = "returns a new tensor"] // CHANGED
pub fn softmax_dim(x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>> {
    if x.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "softmax: input must be at least 1-D".to_string(),
        });
    }
    if dim >= x.ndim() {
        return Err(TensorError::InvalidShape {
            reason: format!("softmax: dim {dim} out of range for {}D tensor", x.ndim()),
        });
    }

    // CHANGED: make contiguous so we can index linearly
    let x_c = if x.is_contiguous() {
        std::borrow::Cow::Borrowed(x)
    } else {
        std::borrow::Cow::Owned(x.contiguous())
    };

    let mut out_data = x_c.as_slice().to_vec();
    let dims = x_c.dims();

    if dim == x_c.ndim() - 1 {
        // Fast path: last dim → rows are contiguous slices // CHANGED
        let n = dims[dim];
        let n_rows = x_c.numel() / n;
        for r in 0..n_rows {
            softmax_row(&mut out_data[r * n..(r + 1) * n]);
        }
    } else {
        // General path: dim is not the last — iterate over dim slices // CHANGED
        // outer = product of dims before `dim`
        // inner = product of dims after `dim`
        let outer: usize = dims[..dim].iter().product();
        let n    = dims[dim];
        let inner: usize = dims[dim + 1..].iter().product();

        // For each (outer, inner) pair, collect the `n` values along `dim`,
        // run softmax, and write back.
        let mut buf = vec![0.0_f32; n];
        for o in 0..outer {
            for i in 0..inner {
                // gather
                for d in 0..n {
                    buf[d] = out_data[(o * n + d) * inner + i];
                }
                softmax_row(&mut buf);
                // scatter back
                for d in 0..n {
                    out_data[(o * n + d) * inner + i] = buf[d];
                }
            }
        }
    }

    Tensor::from_vec(out_data, x_c.shape().clone())
}

/// Apply softmax in-place over the last dimension of `x`.
///
/// Requires `x` to be contiguous.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `x` is 0-D or non-contiguous.
pub fn softmax_inplace(x: &mut Tensor<f32>) -> Result<()> { // CHANGED
    if x.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "softmax_inplace: input must be at least 1-D".to_string(),
        });
    }
    if !x.is_contiguous() {
        return Err(TensorError::InvalidShape {
            reason: "softmax_inplace: x must be contiguous; call x.contiguous() first".to_string(),
        });
    }
    let n = x.dims()[x.ndim() - 1];
    let n_rows = x.numel() / n;
    let data = x.as_slice_mut();
    for r in 0..n_rows {
        softmax_row(&mut data[r * n..(r + 1) * n]);
    }
    Ok(())
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }
    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y, tol))
    }
    fn sum(s: &[f32]) -> f32 { s.iter().sum() }

    // ── output sums to 1 ─────────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one_1d() {
        let x   = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let out = softmax(&x).unwrap();
        assert!(close(sum(out.as_slice()), 1.0, 1e-6));
    }

    #[test]
    fn test_softmax_sums_to_one_each_row_2d() {
        // CHANGED: every row of a [3,4] tensor must sum to 1
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let x   = Tensor::from_vec(data, vec![3, 4]).unwrap();
        let out = softmax(&x).unwrap();
        for r in 0..3 {
            let row_sum = sum(&out.as_slice()[r * 4..(r + 1) * 4]);
            assert!(close(row_sum, 1.0, 1e-6), "row {r} sums to {row_sum}");
        }
    }

    // ── all outputs are in (0, 1) ─────────────────────────────────────────

    #[test]
    fn test_softmax_outputs_in_range() {
        let data: Vec<f32> = (-6..6).map(|i| i as f32).collect();
        let x   = Tensor::from_vec(data, vec![12]).unwrap();
        let out = softmax(&x).unwrap();
        for &v in out.as_slice() {
            assert!(v > 0.0 && v < 1.0, "output {v} not in (0,1)");
        }
    }

    // ── uniform logits → uniform distribution ─────────────────────────────

    #[test]
    fn test_softmax_uniform_logits() {
        // CHANGED: all-same input → all outputs = 1/n
        let n = 5_usize;
        let x   = Tensor::full(vec![n], 2.0_f32);
        let out = softmax(&x).unwrap();
        let expected = 1.0 / n as f32;
        for &v in out.as_slice() {
            assert!(close(v, expected, 1e-6));
        }
    }

    // ── numerical stability: large logits produce no NaN/Inf ──────────────

    #[test]
    fn test_softmax_large_positive_no_nan() {
        // CHANGED: without the max-subtract trick these would all overflow
        let data = vec![1000.0_f32, 1001.0, 1002.0];
        let x   = Tensor::from_vec(data, vec![3]).unwrap();
        let out = softmax(&x).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(),      "NaN in softmax output");
            assert!(!v.is_infinite(), "Inf in softmax output");
        }
        assert!(close(sum(out.as_slice()), 1.0, 1e-6));
    }

    #[test]
    fn test_softmax_large_negative_no_nan() {
        let data = vec![-1000.0_f32, -999.0, -998.0];
        let x   = Tensor::from_vec(data, vec![3]).unwrap();
        let out = softmax(&x).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(),      "NaN in softmax output");
            assert!(!v.is_infinite(), "Inf in softmax output");
        }
        assert!(close(sum(out.as_slice()), 1.0, 1e-5));
    }

    // ── PyTorch reference values ──────────────────────────────────────────

    #[test]
    fn test_softmax_pytorch_reference() {
        // CHANGED: torch.nn.functional.softmax(torch.tensor([1.,2.,3.,4.]), dim=0)
        // → [0.0321, 0.0871, 0.2369, 0.6439]
        let x   = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let out = softmax(&x).unwrap();
        let expected = [0.032_058_604, 0.087_144_31, 0.236_882_8, 0.643_914_24];
        assert!(close_slice(out.as_slice(), &expected, 1e-5),
            "got {:?}", out.as_slice());
    }

    // ── one-hot: one logit >> others → prob ≈ 1 ──────────────────────────

    #[test]
    fn test_softmax_onehot_spike() {
        // CHANGED: a large spike should produce near-1 probability at that index
        let data = vec![0.0_f32, 0.0, 100.0, 0.0];
        let x   = Tensor::from_vec(data, vec![4]).unwrap();
        let out = softmax(&x).unwrap();
        assert!(out.as_slice()[2] > 0.999, "spike prob = {}", out.as_slice()[2]);
        assert!(close(sum(out.as_slice()), 1.0, 1e-6));
    }

    // ── shape preservation ────────────────────────────────────────────────

    #[test]
    fn test_softmax_shape_preserved_3d() {
        let x   = Tensor::from_vec((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]).unwrap();
        let out = softmax(&x).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    // ── softmax_dim: non-last axis ────────────────────────────────────────

    #[test]
    fn test_softmax_dim0_2d() {
        // CHANGED: softmax along dim=0 → each *column* sums to 1
        let data = vec![1.0_f32, 2.0,
                        3.0,     4.0];
        let x   = Tensor::from_vec(data, vec![2, 2]).unwrap();
        let out = softmax_dim(&x, 0).unwrap();
        assert_eq!(out.dims(), &[2, 2]);
        // col 0: softmax([1,3]), col 1: softmax([2,4])
        let col0_sum = out.as_slice()[0] + out.as_slice()[2];
        let col1_sum = out.as_slice()[1] + out.as_slice()[3];
        assert!(close(col0_sum, 1.0, 1e-6), "col0 sum = {col0_sum}");
        assert!(close(col1_sum, 1.0, 1e-6), "col1 sum = {col1_sum}");
    }

    #[test]
    fn test_softmax_dim_last_equals_softmax() {
        // CHANGED: softmax_dim(x, last) must equal softmax(x)
        let x = Tensor::from_vec((0..12).map(|i| i as f32).collect(), vec![3, 4]).unwrap();
        let a = softmax(&x).unwrap();
        let b = softmax_dim(&x, 1).unwrap();
        assert!(close_slice(a.as_slice(), b.as_slice(), 1e-7));
    }

    // ── in-place variant ──────────────────────────────────────────────────

    #[test]
    fn test_softmax_inplace_matches_allocating() {
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let x_alloc    = Tensor::from_vec(data.clone(), vec![2, 4]).unwrap();
        let mut x_ip   = Tensor::from_vec(data, vec![2, 4]).unwrap();
        let out_alloc  = softmax(&x_alloc).unwrap();
        softmax_inplace(&mut x_ip).unwrap();
        assert!(close_slice(out_alloc.as_slice(), x_ip.as_slice(), 1e-7));
    }

    #[test]
    fn test_softmax_inplace_rejects_strided() {
        let orig       = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let mut strided = orig.transpose(0, 1).unwrap();
        assert!(softmax_inplace(&mut strided).is_err());
    }

    // ── non-contiguous input handled ──────────────────────────────────────

    #[test]
    fn test_softmax_non_contiguous_input() {
        let orig = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let t    = orig.transpose(0, 1).unwrap(); // non-contiguous [2,2]
        let out  = softmax(&t);
        assert!(out.is_ok());
        // rows of transposed are [1,2] and [3,4]; each must sum to 1
        let s = out.unwrap();
        let r0 = s.as_slice()[0] + s.as_slice()[1];
        let r1 = s.as_slice()[2] + s.as_slice()[3];
        assert!(close(r0, 1.0, 1e-6));
        assert!(close(r1, 1.0, 1e-6));
    }

    // ── error handling ────────────────────────────────────────────────────

    #[test]
    fn test_softmax_dim_out_of_range() {
        let x = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap();
        assert!(matches!(softmax_dim(&x, 2), Err(TensorError::InvalidShape { .. })));
    }
}
