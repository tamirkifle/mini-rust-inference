//! SiLU (Sigmoid Linear Unit) activation — commit 6.2.
//!
//! # Formula
//!
//! ```text
//! silu(x) = x · σ(x) = x / (1 + exp(-x))
//! ```
//!
//! Also known as the Swish activation (Swish-1).  Used in Llama's FFN gate
//! projection and as the non-linearity inside SwiGLU.
//!
//! # Supported input shapes
//!
//! Any rank ≥ 1.  The operation is elementwise and shape-preserving.

use std::borrow::Cow;
use crate::tensor::{Result, Tensor, TensorError};

// ── scalar kernel (pub so SwiGLU can reuse it) ─────────────────────────────

/// Compute `silu(x) = x · sigmoid(x)` for a single `f32`.
#[inline]
pub fn silu_scalar(x: f32) -> f32 { // CHANGED
    x / (1.0 + (-x).exp())
}

// ── allocating entry point ──────────────────────────────────────────────────

/// Apply SiLU elementwise to every element of `x`, returning a new tensor.
///
/// # Errors
///
/// Returns [`TensorError::InvalidShape`] if `x` is 0-D (scalar).
#[must_use = "returns a new tensor; the input is not modified"] // CHANGED
pub fn silu(x: &Tensor<f32>) -> Result<Tensor<f32>> {
    if x.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "silu: input must be at least 1-D".to_string(),
        });
    }

    // CHANGED: contiguity gate — zero-cost on the fast path
    let x_c: Cow<Tensor<f32>> = if x.is_contiguous() {
        Cow::Borrowed(x)
    } else {
        Cow::Owned(x.contiguous())
    };

    let out: Vec<f32> = x_c.as_slice().iter().map(|&v| silu_scalar(v)).collect(); // CHANGED
    Tensor::from_vec(out, x.shape().clone())
}

/// Apply SiLU in-place to every element of `x`.
///
/// Requires `x` to be contiguous.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `x` is 0-D or non-contiguous.
pub fn silu_inplace(x: &mut Tensor<f32>) -> Result<()> { // CHANGED
    if x.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "silu_inplace: input must be at least 1-D".to_string(),
        });
    }
    if !x.is_contiguous() {
        return Err(TensorError::InvalidShape {
            reason: "silu_inplace: x must be contiguous; call x.contiguous() first".to_string(),
        });
    }
    for v in x.as_slice_mut() {
        *v = silu_scalar(*v); // CHANGED
    }
    Ok(())
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    // ── scalar kernel ─────────────────────────────────────────────────────

    #[test]
    fn test_silu_scalar_zero() {
        // CHANGED: silu(0) = 0 · sigmoid(0) = 0 · 0.5 = 0
        assert!(close(silu_scalar(0.0), 0.0, 1e-7));
    }

    #[test]
    fn test_silu_scalar_positive_large() {
        // silu(x) → x for large x (sigmoid → 1)
        let x = 20.0_f32;
        assert!(close(silu_scalar(x), x, 1e-3));
    }

    #[test]
    fn test_silu_scalar_negative_large() {
        // silu(x) → 0 for large negative x (sigmoid → 0)
        assert!(close(silu_scalar(-20.0), 0.0, 1e-3));
    }

    #[test]
    fn test_silu_scalar_pytorch_reference() {
        // CHANGED: cross-check with PyTorch:
        //   import torch; torch.nn.functional.silu(torch.tensor([1., -1., 2., -2.]))
        //   → [0.7311, -0.2689,  1.7616, -0.2384]
        let cases = [(1.0_f32, 0.731_059), (-1.0, -0.268_941), (2.0, 1.761_594), (-2.0, -0.238_406)];
        for (x, expected) in cases {
            assert!(
                close(silu_scalar(x), expected, 1e-5),
                "silu({x}) = {}, expected {expected}", silu_scalar(x)
            );
        }
    }

    // ── tensor API ────────────────────────────────────────────────────────

    #[test]
    fn test_silu_shape_preserved_1d() {
        let x = Tensor::from_vec(vec![1.0_f32, -1.0, 0.0], vec![3]).unwrap();
        let out = silu(&x).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_silu_shape_preserved_2d() {
        let x = Tensor::from_vec((0..6).map(|i| i as f32 - 3.0).collect(), vec![2, 3]).unwrap();
        let out = silu(&x).unwrap();
        assert_eq!(out.dims(), &[2, 3]);
    }

    #[test]
    fn test_silu_values_match_scalar() {
        // CHANGED: tensor version must apply the scalar kernel to each element
        let data = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let x   = Tensor::from_vec(data.clone(), vec![5]).unwrap();
        let out = silu(&x).unwrap();
        for (&got, &raw) in out.as_slice().iter().zip(data.iter()) {
            assert!(close(got, silu_scalar(raw), 1e-7), "element mismatch");
        }
    }

    #[test]
    fn test_silu_non_contiguous_input() {
        // CHANGED: transposed (strided) input must still produce correct output
        let x_orig = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let x_t    = x_orig.transpose(0, 1).unwrap();
        let out    = silu(&x_t).unwrap();
        assert_eq!(out.dims(), &[2, 2]);
        // Contiguous layout of transposed [[1,3],[2,4]] → elements [1,3,2,4]
        let expected: Vec<f32> = [1.0_f32, 3.0, 2.0, 4.0].iter().map(|&v| silu_scalar(v)).collect();
        for (&got, exp) in out.as_slice().iter().zip(expected.iter()) {
            assert!(close(got, *exp, 1e-6));
        }
    }

    #[test]
    fn test_silu_inplace_matches_allocating() {
        let data = vec![1.0_f32, -1.0, 2.0, -2.0];
        let x_alloc = Tensor::from_vec(data.clone(), vec![4]).unwrap();
        let mut x_ip = Tensor::from_vec(data, vec![4]).unwrap();

        let out_alloc = silu(&x_alloc).unwrap();
        silu_inplace(&mut x_ip).unwrap();

        for (&a, &b) in out_alloc.as_slice().iter().zip(x_ip.as_slice().iter()) {
            assert!(close(a, b, 1e-7));
        }
    }

    #[test]
    fn test_silu_inplace_rejects_strided() {
        let x_orig = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let mut x_t = x_orig.transpose(0, 1).unwrap();
        assert!(silu_inplace(&mut x_t).is_err());
    }

    #[test]
    fn test_silu_no_nan_inf() {
        // CHANGED: large positive and negative values must not produce NaN/Inf
        let data: Vec<f32> = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        let x   = Tensor::from_vec(data, vec![5]).unwrap();
        let out = silu(&x).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(), "NaN in silu output");
            assert!(!v.is_infinite(), "Inf in silu output");
        }
    }
}
