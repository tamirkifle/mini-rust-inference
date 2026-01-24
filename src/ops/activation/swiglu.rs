//! SwiGLU gating activation — commit 6.2.
//!
//! # Formula
//!
//! ```text
//! swiglu(gate, up) = silu(gate) ⊙ up
//! ```
//!
//! where `⊙` is elementwise multiplication.  In Llama's FFN:
//!
//! ```text
//! gate = x @ W_gate    // shape [seq, d_ff]
//! up   = x @ W_up      // shape [seq, d_ff]
//! h    = swiglu(gate, up)
//! out  = h @ W_down    // shape [seq, d_model]
//! ```
//!
//! This module provides:
//! - [`swiglu`] — allocating version (returns a new tensor)
//! - [`swiglu_inplace`] — in-place on the `gate` tensor (saves one allocation)
//!
//! # Shape contract
//!
//! `gate` and `up` must have identical shapes.  Any rank ≥ 1 is accepted.

use std::borrow::Cow;
use crate::tensor::{Result, Tensor, TensorError};
use super::silu::silu_scalar; // CHANGED: reuse the scalar kernel

// ── allocating ─────────────────────────────────────────────────────────────

/// Compute `silu(gate) ⊙ up`, returning a new tensor.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if either input is 0-D.
/// * [`TensorError::ShapeMismatch`] if `gate` and `up` have different shapes.
#[must_use = "returns a new tensor"] // CHANGED
pub fn swiglu(gate: &Tensor<f32>, up: &Tensor<f32>) -> Result<Tensor<f32>> {
    validate(gate, up)?;

    // CHANGED: contiguity gate — borrow when already contiguous
    let g: Cow<Tensor<f32>> = if gate.is_contiguous() { Cow::Borrowed(gate) } else { Cow::Owned(gate.contiguous()) };
    let u: Cow<Tensor<f32>> = if up.is_contiguous()   { Cow::Borrowed(up)   } else { Cow::Owned(up.contiguous())   };

    let out: Vec<f32> = g.as_slice()
        .iter()
        .zip(u.as_slice().iter())
        .map(|(&g_i, &u_i)| silu_scalar(g_i) * u_i) // CHANGED: silu(gate) * up
        .collect();

    Tensor::from_vec(out, gate.shape().clone())
}

/// Compute `silu(gate) ⊙ up` in-place, writing results back into `gate`.
///
/// Requires both tensors to be contiguous.  Use this in the forward pass to
/// avoid allocating a third `[seq, d_ff]` buffer.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if either tensor is 0-D or non-contiguous.
/// * [`TensorError::ShapeMismatch`] if shapes differ.
pub fn swiglu_inplace(gate: &mut Tensor<f32>, up: &Tensor<f32>) -> Result<()> { // CHANGED
    validate(gate, up)?;

    if !gate.is_contiguous() {
        return Err(TensorError::InvalidShape {
            reason: "swiglu_inplace: gate must be contiguous".to_string(),
        });
    }
    if !up.is_contiguous() {
        return Err(TensorError::InvalidShape {
            reason: "swiglu_inplace: up must be contiguous".to_string(),
        });
    }

    let up_data = up.as_slice().to_vec(); // CHANGED: snapshot before mutable borrow
    for (g_i, &u_i) in gate.as_slice_mut().iter_mut().zip(up_data.iter()) {
        *g_i = silu_scalar(*g_i) * u_i;
    }
    Ok(())
}

// ── shared validation ───────────────────────────────────────────────────────

fn validate(gate: &Tensor<f32>, up: &Tensor<f32>) -> Result<()> { // CHANGED
    if gate.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "swiglu: gate must be at least 1-D".to_string(),
        });
    }
    if up.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "swiglu: up must be at least 1-D".to_string(),
        });
    }
    if gate.shape() != up.shape() {
        return Err(TensorError::ShapeMismatch {
            expected: gate.dims().to_vec(),
            got:      up.dims().to_vec(),
        });
    }
    Ok(())
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::silu::silu_scalar;

    fn close(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    // ── correctness ───────────────────────────────────────────────────────

    #[test]
    fn test_swiglu_known_values() {
        // CHANGED: swiglu([1,-1], [2,3]) = [silu(1)*2, silu(-1)*3]
        let gate = Tensor::from_vec(vec![1.0_f32, -1.0], vec![2]).unwrap();
        let up   = Tensor::from_vec(vec![2.0_f32,  3.0], vec![2]).unwrap();
        let out  = swiglu(&gate, &up).unwrap();
        let expected = [silu_scalar(1.0) * 2.0, silu_scalar(-1.0) * 3.0];
        for (&got, &exp) in out.as_slice().iter().zip(expected.iter()) {
            assert!(close(got, exp, 1e-6));
        }
    }

    #[test]
    fn test_swiglu_up_ones_equals_silu_gate() {
        // CHANGED: up = 1 → swiglu == silu(gate)
        let data = vec![-2.0_f32, -1.0, 0.0, 1.0, 2.0];
        let gate = Tensor::from_vec(data.clone(), vec![5]).unwrap();
        let up   = Tensor::ones(vec![5]);
        let out  = swiglu(&gate, &up).unwrap();
        for (&got, &raw) in out.as_slice().iter().zip(data.iter()) {
            assert!(close(got, silu_scalar(raw), 1e-7));
        }
    }

    #[test]
    fn test_swiglu_gate_zero_zeroes_output() {
        // silu(0) = 0 → output all zeros regardless of up
        let gate = Tensor::zeros(vec![4]);
        let up   = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let out  = swiglu(&gate, &up).unwrap();
        for &v in out.as_slice() {
            assert!(close(v, 0.0, 1e-7), "expected 0.0, got {v}");
        }
    }

    // ── shape handling ────────────────────────────────────────────────────

    #[test]
    fn test_swiglu_shape_preserved_2d() {
        let (seq, d) = (4, 8);
        let gate = Tensor::from_vec((0..(seq*d)).map(|i| i as f32 * 0.1 - 2.0).collect(), vec![seq, d]).unwrap();
        let up   = Tensor::from_vec((0..(seq*d)).map(|i| i as f32 * 0.05 + 1.0).collect(), vec![seq, d]).unwrap();
        let out  = swiglu(&gate, &up).unwrap();
        assert_eq!(out.dims(), &[seq, d]);
    }

    #[test]
    fn test_swiglu_non_contiguous_inputs() {
        // CHANGED: transposed inputs must still give correct result
        let g_orig = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let u_orig = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let g_t    = g_orig.transpose(0, 1).unwrap();
        let u_t    = u_orig.transpose(0, 1).unwrap();
        // Both non-contiguous — allocating version must handle it
        let out = swiglu(&g_t, &u_t).unwrap();
        assert_eq!(out.dims(), &[2, 2]);
    }

    // ── in-place ──────────────────────────────────────────────────────────

    #[test]
    fn test_swiglu_inplace_matches_allocating() {
        let g_data = vec![-1.0_f32, 0.5, 1.5, -0.3];
        let u_data = vec![ 2.0_f32, 1.0, 0.5,  3.0];
        let gate_alloc = Tensor::from_vec(g_data.clone(), vec![4]).unwrap();
        let up_alloc   = Tensor::from_vec(u_data.clone(), vec![4]).unwrap();
        let mut gate_ip = Tensor::from_vec(g_data, vec![4]).unwrap();
        let up_ip       = Tensor::from_vec(u_data, vec![4]).unwrap();

        let out_alloc = swiglu(&gate_alloc, &up_alloc).unwrap();
        swiglu_inplace(&mut gate_ip, &up_ip).unwrap();

        for (&a, &b) in out_alloc.as_slice().iter().zip(gate_ip.as_slice().iter()) {
            assert!(close(a, b, 1e-7));
        }
    }

    #[test]
    fn test_swiglu_inplace_rejects_strided_gate() {
        let g = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let mut g_t = g.transpose(0, 1).unwrap();
        let up = Tensor::ones(vec![2, 2]);
        assert!(swiglu_inplace(&mut g_t, &up).is_err());
    }

    #[test]
    fn test_swiglu_inplace_rejects_strided_up() {
        let mut gate = Tensor::ones(vec![2, 2]);
        let u_orig   = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let u_t      = u_orig.transpose(0, 1).unwrap();
        assert!(swiglu_inplace(&mut gate, &u_t).is_err());
    }

    // ── error handling ────────────────────────────────────────────────────

    #[test]
    fn test_swiglu_shape_mismatch() {
        let gate = Tensor::from_vec(vec![1.0_f32; 4], vec![4]).unwrap();
        let up   = Tensor::from_vec(vec![1.0_f32; 6], vec![6]).unwrap();
        assert!(matches!(swiglu(&gate, &up), Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_swiglu_no_nan_inf() {
        // CHANGED: extreme values must not produce NaN/Inf
        let data: Vec<f32> = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        let gate = Tensor::from_vec(data.clone(), vec![5]).unwrap();
        let up   = Tensor::from_vec(data, vec![5]).unwrap();
        let out  = swiglu(&gate, &up).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(), "NaN in swiglu output");
            assert!(!v.is_infinite(), "Inf in swiglu output");
        }
    }

    // ── pytorch reference ─────────────────────────────────────────────────

    #[test]
    fn test_swiglu_pytorch_reference() {
        // CHANGED: Python reference:
        //   import torch, torch.nn.functional as F
        //   gate = torch.tensor([1., 2., -1., -2.])
        //   up   = torch.tensor([1., 0.5, 2., 3.])
        //   F.silu(gate) * up
        //   → [0.7311, 0.8808, -0.5379, -0.7151]
        let gate = Tensor::from_vec(vec![1.0_f32, 2.0, -1.0, -2.0], vec![4]).unwrap();
        let up   = Tensor::from_vec(vec![1.0_f32, 0.5,  2.0,  3.0], vec![4]).unwrap();
        let out  = swiglu(&gate, &up).unwrap();
        let expected = [0.731_059, 0.880_797, -0.537_882, -0.715_219];
        for (&got, &exp) in out.as_slice().iter().zip(expected.iter()) {
            assert!(close(got, exp, 1e-4), "got {got}, expected {exp}");
        }
    }
}
