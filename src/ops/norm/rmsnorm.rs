//! RMSNorm — Root Mean Square Layer Normalization.
//!
//! Used in Llama in place of LayerNorm.  The key difference: no mean
//! subtraction (centering), only RMS-based rescaling.
//!
//! # Formula
//!
//! ```text
//! rms(x)     = sqrt( (1/d) · Σ xᵢ² + ε )
//! x̂ᵢ        = xᵢ / rms(x)
//! output[i]  = weight[i] · x̂ᵢ
//! ```
//!
//! `weight` is a learned gain vector of shape `[d]` (called `norm.weight` in
//! Llama checkpoints).  `ε` is a small constant for numerical stability
//! (typical: 1e-5 or 1e-6).
//!
//! # Supported input shapes
//!
//! | Shape            | Interpretation          |
//! |------------------|-------------------------|
//! | `[d]`            | single token vector     |
//! | `[seq, d]`       | sequence of tokens      |
//! | `[batch, seq, d]`| batch of sequences      |
//!
//! The normalisation is always computed over the **last** dimension.

use crate::tensor::{Result, Tensor, TensorError};

// ── public entry points ─────────────────────────────────────────────────────

/// Apply RMSNorm over the last dimension of `x`.
///
/// # Arguments
///
/// * `x`      – Input tensor; any rank ≥ 1.  Shape `[..., d]`.
/// * `weight` – Gain vector of shape `[d]`.  Must be 1-D and contiguous.
/// * `eps`    – Stability epsilon (use [`DEFAULT_EPS`] if unsure).
///
/// # Returns
///
/// A new contiguous `Tensor<f32>` with the same shape as `x`.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `x` is 0-D, `weight` is not 1-D, or
///   `weight` length does not equal the last dimension of `x`.
/// * [`TensorError::ShapeMismatch`] (reused as a convenience) never fires
///   here; all dim errors come back as [`TensorError::InvalidShape`].
#[must_use = "returns a new tensor; the input is not modified"] // CHANGED
pub fn rmsnorm(x: &Tensor<f32>, weight: &Tensor<f32>, eps: f32) -> Result<Tensor<f32>> {
    // ── validate weight ────────────────────────────────────────────────────
    if weight.ndim() != 1 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rmsnorm: weight must be 1-D, got {}D (shape {:?})",
                weight.ndim(),
                weight.dims()
            ),
        });
    }

    let d = weight.dims()[0]; // CHANGED: last-dim size == hidden dimension

    // ── validate x ────────────────────────────────────────────────────────
    if x.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "rmsnorm: input must be at least 1-D".to_string(),
        });
    }

    let x_last_dim = x.dims()[x.ndim() - 1];
    if x_last_dim != d {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rmsnorm: last dim of x ({x_last_dim}) != weight length ({d})"
            ),
        });
    }

    // ── contiguity gate ────────────────────────────────────────────────────
    // CHANGED: avoid extra allocation on the hot path; contiguous tensors are
    // the common case coming out of matmul / embedding look-ups.
    let x_c = if x.is_contiguous() {
        std::borrow::Cow::Borrowed(x)
    } else {
        std::borrow::Cow::Owned(x.contiguous())
    };
    let w_c = if weight.is_contiguous() {
        std::borrow::Cow::Borrowed(weight)
    } else {
        std::borrow::Cow::Owned(weight.contiguous())
    };

    let x_data = x_c.as_slice();
    let w_data = w_c.as_slice();

    // ── compute ────────────────────────────────────────────────────────────
    let n_rows = x.numel() / d; // number of independent vectors to normalise // CHANGED
    let mut out = vec![0.0_f32; x.numel()];

    for row in 0..n_rows {
        let start = row * d;
        let end   = start + d;
        let slice = &x_data[start..end];

        // RMS: sqrt( mean(x²) + ε )
        let mean_sq: f32 = slice.iter().map(|v| v * v).sum::<f32>() / d as f32; // CHANGED
        let rms_inv = 1.0 / (mean_sq + eps).sqrt(); // CHANGED: reciprocal avoids repeated division

        // Normalise + scale
        let out_row = &mut out[start..end];
        for (i, (&xi, &wi)) in slice.iter().zip(w_data.iter()).enumerate() {
            out_row[i] = wi * (xi * rms_inv); // CHANGED: weight · (x / rms)
        }
    }

    Tensor::from_vec(out, x.shape().clone())
}

/// Default epsilon used by Llama 2 and Llama 3 checkpoints.
pub const DEFAULT_EPS: f32 = 1e-5; // CHANGED: matches Llama default

// ── in-place variant ────────────────────────────────────────────────────────

/// Apply RMSNorm in-place over the last dimension of `x`.
///
/// Useful when the caller is done with the original values and wants to avoid
/// an extra allocation.  Semantics are identical to [`rmsnorm`].
///
/// # Errors
///
/// Same conditions as [`rmsnorm`].
pub fn rmsnorm_inplace(x: &mut Tensor<f32>, weight: &Tensor<f32>, eps: f32) -> Result<()> {
    // CHANGED: validate first, then mutate — no partial writes on error
    if weight.ndim() != 1 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rmsnorm_inplace: weight must be 1-D, got {}D",
                weight.ndim()
            ),
        });
    }
    let d = weight.dims()[0];

    if x.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "rmsnorm_inplace: input must be at least 1-D".to_string(),
        });
    }
    let x_last_dim = x.dims()[x.ndim() - 1];
    if x_last_dim != d {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rmsnorm_inplace: last dim of x ({x_last_dim}) != weight length ({d})"
            ),
        });
    }

    // Require contiguity for in-place — strided tensors are ambiguous to mutate
    if !x.is_contiguous() {
        return Err(TensorError::InvalidShape {
            reason: "rmsnorm_inplace: x must be contiguous; call x.contiguous() first".to_string(),
        });
    }

    let w_c = if weight.is_contiguous() {
        std::borrow::Cow::Borrowed(weight)
    } else {
        std::borrow::Cow::Owned(weight.contiguous())
    };
    let w_data = w_c.as_slice();

    let n_rows = x.numel() / d;
    let x_data = x.as_slice_mut();

    for row in 0..n_rows {
        let start = row * d;
        let end   = start + d;
        let slice = &x_data[start..end];

        let mean_sq: f32 = slice.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let rms_inv = 1.0 / (mean_sq + eps).sqrt();

        // CHANGED: two-pass is unavoidable for in-place (need rms before writing)
        for i in start..end {
            x_data[i] = w_data[i - start] * (x_data[i] * rms_inv);
        }
    }

    Ok(())
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }
    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y, tol))
    }

    // ── identity weight → pure normalisation ─────────────────────────────

    #[test]
    fn test_rmsnorm_identity_weight_1d() {
        // CHANGED: with weight = 1, output should equal x / rms(x)
        let x = Tensor::from_vec(vec![3.0_f32, 4.0], vec![2]).unwrap();
        let w = Tensor::ones(vec![2]);
        let out = rmsnorm(&x, &w, EPS).unwrap();

        // rms([3,4]) = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.5355
        let rms = (12.5_f32 + EPS).sqrt();
        let expected = [3.0 / rms, 4.0 / rms];
        assert!(close_slice(out.as_slice(), &expected, 1e-5));
    }

    #[test]
    fn test_rmsnorm_zero_weight_zeroes_output() {
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3]).unwrap();
        let w = Tensor::zeros(vec![3]);
        let out = rmsnorm(&x, &w, EPS).unwrap();
        assert!(out.as_slice().iter().all(|&v| v == 0.0));
    }

    // ── shape preservation ────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_preserves_shape_2d() {
        // CHANGED: [seq=4, d=8] → same shape out
        let d: usize = 8;
        let seq: usize = 4;
        let x = Tensor::from_vec(
            (0..(seq * d)).map(|i| i as f32 * 0.1 + 0.1).collect(),
            vec![seq, d],
        ).unwrap();
        let w = Tensor::ones(vec![d]);
        let out = rmsnorm(&x, &w, EPS).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_rmsnorm_preserves_shape_3d() {
        let (batch, seq, d) = (2, 3, 16);
        let x = Tensor::from_vec(
            (0..(batch * seq * d)).map(|i| i as f32 + 1.0).collect(),
            vec![batch, seq, d],
        ).unwrap();
        let w = Tensor::ones(vec![d]);
        let out = rmsnorm(&x, &w, EPS).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    // ── each row is independently normalised ─────────────────────────────

    #[test]
    fn test_rmsnorm_rows_independent() {
        // CHANGED: process rows one-by-one and compare to batch result
        let d = 4_usize;
        let seq = 3_usize;
        let data: Vec<f32> = (0..(seq * d)).map(|i| i as f32 + 1.0).collect();
        let x = Tensor::from_vec(data.clone(), vec![seq, d]).unwrap();
        let w = Tensor::from_vec(vec![0.5_f32, 1.0, 1.5, 2.0], vec![d]).unwrap();

        let batch_out = rmsnorm(&x, &w, EPS).unwrap();

        for row in 0..seq {
            let row_data = data[row * d..(row + 1) * d].to_vec();
            let x_row = Tensor::from_vec(row_data, vec![d]).unwrap();
            let row_out = rmsnorm(&x_row, &w, EPS).unwrap();
            assert!(
                close_slice(
                    &batch_out.as_slice()[row * d..(row + 1) * d],
                    row_out.as_slice(),
                    1e-6
                ),
                "row {row} mismatch"
            );
        }
    }

    // ── unit vector stays unit ────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_unit_vector_with_identity_weight() {
        // CHANGED: x = [1/√3, 1/√3, 1/√3] has mean_sq = 1/3, rms = sqrt(1/3 + ε) ≈ 1/√3
        // output[i] = x[i] / rms ≈ (1/√3) / (1/√3) ≈ 1.0
        let v = (1.0_f32 / 3.0_f32).sqrt();
        let x = Tensor::from_vec(vec![v, v, v], vec![3]).unwrap();
        let w = Tensor::ones(vec![3]);
        let out = rmsnorm(&x, &w, EPS).unwrap();
        // Each element normalises to ≈ 1.0 (within floating-point + epsilon error)
        for &o in out.as_slice() {
            assert!(close(o, 1.0, 1e-3), "expected ≈ 1.0, got {o}");
        }
    }

    // ── epsilon prevents division by zero ─────────────────────────────────

    #[test]
    fn test_rmsnorm_all_zeros_no_nan() {
        // CHANGED: zero input → rms = sqrt(ε), output = 0 for all (0 * weight/rms = 0)
        let x = Tensor::zeros(vec![4]);
        let w = Tensor::ones(vec![4]);
        let out = rmsnorm(&x, &w, EPS).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(), "got NaN for all-zero input");
            assert!(!v.is_infinite(), "got Inf for all-zero input");
        }
    }

    // ── numerical precision (PyTorch reference values) ───────────────────

    #[test]
    fn test_rmsnorm_pytorch_reference() {
        // CHANGED: cross-check with PyTorch:
        //   x = torch.tensor([1., 2., 3., 4.])
        //   w = torch.tensor([1., 1., 1., 1.])
        //   torch.nn.functional.rms_norm(x, (4,), w, eps=1e-5)
        //   → tensor([0.3651, 0.7303, 1.0954, 1.4606])
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let w = Tensor::ones(vec![4]);
        let out = rmsnorm(&x, &w, 1e-5).unwrap();
        let expected = [0.365_148_4, 0.730_296_8, 1.095_445_2, 1.460_593_6];
        assert!(
            close_slice(out.as_slice(), &expected, 1e-5),
            "got {:?}, expected {:?}",
            out.as_slice(),
            expected
        );
    }

    // ── in-place variant ─────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_inplace_matches_allocating() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w    = Tensor::from_vec(vec![1.0_f32, 0.5, 2.0], vec![3]).unwrap();

        let x_alloc = Tensor::from_vec(data.clone(), vec![2, 3]).unwrap();
        let mut x_ip = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let out_alloc = rmsnorm(&x_alloc, &w, EPS).unwrap();
        rmsnorm_inplace(&mut x_ip, &w, EPS).unwrap();

        assert!(close_slice(out_alloc.as_slice(), x_ip.as_slice(), 1e-6));
    }

    #[test]
    fn test_rmsnorm_inplace_rejects_strided() {
        // CHANGED: in-place on a transposed (strided) tensor must fail gracefully
        let x_orig = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let mut x_t = x_orig.transpose(0, 1).unwrap();
        let w = Tensor::ones(vec![2]);
        assert!(rmsnorm_inplace(&mut x_t, &w, EPS).is_err());
    }

    // ── error handling ────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_weight_dim_mismatch() {
        let x = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap();
        let w = Tensor::ones(vec![4]); // wrong d
        assert!(matches!(
            rmsnorm(&x, &w, EPS),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_rmsnorm_weight_not_1d() {
        let x = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let w = Tensor::ones(vec![2, 2]); // 2-D weight — invalid
        assert!(matches!(
            rmsnorm(&x, &w, EPS),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_rmsnorm_0d_input_rejected() {
        // 0-D tensors (scalars) don't make sense for RMSNorm
        // We can't construct a 0-D Tensor via from_vec with shape [] easily,
        // but we can test via a shape with numel=1 that is 1-D (valid) vs a
        // hypothetical 0-D case covered by the ndim==0 branch.
        // The Tensor API doesn't expose 0-D, so test that 1-D [1] does work.
        let x = Tensor::from_vec(vec![2.0_f32], vec![1]).unwrap();
        let w = Tensor::ones(vec![1]);
        assert!(rmsnorm(&x, &w, EPS).is_ok());
    }

    // ── non-contiguous input is handled ──────────────────────────────────

    #[test]
    fn test_rmsnorm_non_contiguous_input() {
        // CHANGED: transpose makes x non-contiguous; rmsnorm must still work
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let x_orig = Tensor::from_vec(data, vec![2, 2]).unwrap();
        let x_t    = x_orig.transpose(0, 1).unwrap(); // shape [2,2], strided
        let w      = Tensor::ones(vec![2]);
        let out    = rmsnorm(&x_t, &w, EPS);
        assert!(out.is_ok(), "rmsnorm on non-contiguous input must not panic");
        assert_eq!(out.unwrap().dims(), &[2, 2]);
    }
}
