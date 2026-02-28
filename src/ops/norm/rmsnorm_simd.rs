//! SIMD-accelerated RMSNorm — commit 15.3.
//!
//! Vectorises the two hot loops of [`super::rmsnorm`]:
//!
//! 1. **Sum-of-squares**: `Σ xᵢ²` → `simd::f32::dot(row, row)`
//!    (self-dot computes the same value; NEON vfmaq_f32 / AVX2 vfmadd_ps)
//!
//! 2. **Normalise + weight**: `wᵢ · (xᵢ · rms_inv)` is split into
//!    - `simd::f32::scale_into(tmp, row, rms_inv)` → `tmp[i] = xᵢ · rms_inv`
//!    - `simd::f32::mul_into(out_row, tmp, weight)`  → `out[i] = wᵢ · tmp[i]`
//!
//! # Correctness guarantee
//!
//! Both `rmsnorm_simd` and `rmsnorm_simd_inplace` must produce results within
//! `1e-5` absolute of the scalar [`super::rmsnorm`] / [`super::rmsnorm_inplace`].

use crate::simd::f32 as simd;
use crate::tensor::{Result, Tensor, TensorError};

/// Apply RMSNorm over the last dimension of `x` using SIMD-accelerated loops.
///
/// Drop-in replacement for [`super::rmsnorm`].  Same signature, same error
/// conditions, same output shape.
#[must_use = "returns a new tensor; the input is not modified"]
pub fn rmsnorm_simd(x: &Tensor<f32>, weight: &Tensor<f32>, eps: f32) -> Result<Tensor<f32>> {
    if weight.ndim() != 1 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rmsnorm_simd: weight must be 1-D, got {}D",
                weight.ndim()
            ),
        });
    }
    let d = weight.dims()[0];
    if x.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "rmsnorm_simd: input must be at least 1-D".to_string(),
        });
    }
    if x.dims()[x.ndim() - 1] != d {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rmsnorm_simd: last dim of x ({}) != weight length ({d})",
                x.dims()[x.ndim() - 1]
            ),
        });
    }

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

    let n_rows = x.numel() / d;
    let mut out = vec![0.0_f32; x.numel()];

    // One reusable scratch buffer for the scaled (pre-weight) row.
    let mut tmp = vec![0.0_f32; d];

    for row in 0..n_rows {
        let start = row * d;
        let slice = &x_data[start..start + d];
        let out_row = &mut out[start..start + d];

        // ① sum-of-squares via self-dot (SIMD)
        let sum_sq = simd::dot(slice, slice);
        let rms_inv = 1.0 / (sum_sq / d as f32 + eps).sqrt();

        // ② tmp[i] = xᵢ · rms_inv  (SIMD scale)
        simd::scale_into(&mut tmp, slice, rms_inv);

        // ③ out[i] = wᵢ · tmp[i]   (SIMD element-wise mul)
        simd::mul_into(out_row, &tmp, w_data);
    }

    Tensor::from_vec(out, x.shape().clone())
}

/// Apply RMSNorm in-place using SIMD-accelerated loops.
///
/// Drop-in replacement for [`super::rmsnorm_inplace`].
pub fn rmsnorm_simd_inplace(x: &mut Tensor<f32>, weight: &Tensor<f32>, eps: f32) -> Result<()> {
    if weight.ndim() != 1 {
        return Err(TensorError::InvalidShape {
            reason: format!("rmsnorm_simd_inplace: weight must be 1-D, got {}D", weight.ndim()),
        });
    }
    let d = weight.dims()[0];
    if x.ndim() == 0 {
        return Err(TensorError::InvalidShape {
            reason: "rmsnorm_simd_inplace: input must be at least 1-D".to_string(),
        });
    }
    if x.dims()[x.ndim() - 1] != d {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rmsnorm_simd_inplace: last dim ({}) != weight length ({d})",
                x.dims()[x.ndim() - 1]
            ),
        });
    }
    if !x.is_contiguous() {
        return Err(TensorError::InvalidShape {
            reason: "rmsnorm_simd_inplace: x must be contiguous".to_string(),
        });
    }

    let w_c = if weight.is_contiguous() {
        std::borrow::Cow::Borrowed(weight)
    } else {
        std::borrow::Cow::Owned(weight.contiguous())
    };
    let w_data = w_c.as_slice();

    let n_rows = x.numel() / d;
    let mut tmp = vec![0.0_f32; d];

    for row in 0..n_rows {
        let start = row * d;

        // Read sum-of-squares from the existing data.
        let sum_sq = {
            let slice = &x.as_slice()[start..start + d];
            simd::dot(slice, slice)
        };
        let rms_inv = 1.0 / (sum_sq / d as f32 + eps).sqrt();

        // tmp = x_row · rms_inv, then x_row = w · tmp
        {
            let slice = &x.as_slice()[start..start + d];
            simd::scale_into(&mut tmp, slice, rms_inv);
        }
        simd::mul_into(&mut x.as_slice_mut()[start..start + d], &tmp, w_data);
    }

    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::norm::rmsnorm;

    const EPS: f32 = 1e-5;

    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    // ── allocating variant matches scalar ────────────────────────────────

    #[test]
    fn test_simd_matches_scalar_1d() {
        let x = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let w = Tensor::ones(vec![4]);
        let scalar = rmsnorm(&x, &w, EPS).unwrap();
        let simd   = rmsnorm_simd(&x, &w, EPS).unwrap();
        assert!(close_slice(scalar.as_slice(), simd.as_slice(), 1e-5),
            "scalar={:?}\nsimd={:?}", scalar.as_slice(), simd.as_slice());
    }

    #[test]
    fn test_simd_matches_scalar_2d() {
        let data: Vec<f32> = (0..32).map(|i| i as f32 * 0.3 + 0.1).collect();
        let x = Tensor::from_vec(data, vec![4, 8]).unwrap();
        let w = Tensor::from_vec((1..=8).map(|i| i as f32 * 0.5).collect(), vec![8]).unwrap();
        let scalar = rmsnorm(&x, &w, EPS).unwrap();
        let simd   = rmsnorm_simd(&x, &w, EPS).unwrap();
        assert!(close_slice(scalar.as_slice(), simd.as_slice(), 1e-5));
    }

    #[test]
    fn test_simd_matches_scalar_3d() {
        let (batch, seq, d) = (2, 3, 16);
        let x = Tensor::from_vec(
            (0..(batch * seq * d)).map(|i| i as f32 * 0.05 + 0.01).collect(),
            vec![batch, seq, d],
        ).unwrap();
        let w = Tensor::ones(vec![d]);
        let scalar = rmsnorm(&x, &w, EPS).unwrap();
        let simd   = rmsnorm_simd(&x, &w, EPS).unwrap();
        assert!(close_slice(scalar.as_slice(), simd.as_slice(), 1e-5));
    }

    #[test]
    fn test_simd_non_contiguous_input() {
        let orig = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let x_t  = orig.transpose(0, 1).unwrap();
        let w    = Tensor::ones(vec![2]);
        let scalar = rmsnorm(&x_t, &w, EPS).unwrap();
        let simd   = rmsnorm_simd(&x_t, &w, EPS).unwrap();
        assert!(close_slice(scalar.as_slice(), simd.as_slice(), 1e-5));
    }

    #[test]
    fn test_simd_all_zeros_no_nan() {
        let x = Tensor::zeros(vec![8]);
        let w = Tensor::ones(vec![8]);
        let out = rmsnorm_simd(&x, &w, EPS).unwrap();
        assert!(out.as_slice().iter().all(|v| !v.is_nan() && !v.is_infinite()));
    }

    // ── shape / error handling ────────────────────────────────────────────

    #[test]
    fn test_simd_shape_preserved() {
        let x = Tensor::from_vec((0..24).map(|i| i as f32).collect(), vec![2, 3, 4]).unwrap();
        let w = Tensor::ones(vec![4]);
        let out = rmsnorm_simd(&x, &w, EPS).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn test_simd_weight_dim_mismatch() {
        let x = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap();
        let w = Tensor::ones(vec![5]);
        assert!(rmsnorm_simd(&x, &w, EPS).is_err());
    }

    // ── in-place variant ─────────────────────────────────────────────────

    #[test]
    fn test_simd_inplace_matches_allocating() {
        let data: Vec<f32> = (0..20).map(|i| i as f32 * 0.1 + 0.5).collect();
        let w = Tensor::from_vec((1..=4).map(|i| i as f32 * 0.25).collect(), vec![4]).unwrap();
        let x_alloc = Tensor::from_vec(data.clone(), vec![5, 4]).unwrap();
        let mut x_ip = Tensor::from_vec(data, vec![5, 4]).unwrap();
        let alloc_out = rmsnorm_simd(&x_alloc, &w, EPS).unwrap();
        rmsnorm_simd_inplace(&mut x_ip, &w, EPS).unwrap();
        assert!(close_slice(alloc_out.as_slice(), x_ip.as_slice(), 1e-6));
    }

    #[test]
    fn test_simd_inplace_rejects_strided() {
        let orig = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let mut x_t = orig.transpose(0, 1).unwrap();
        let w = Tensor::ones(vec![2]);
        assert!(rmsnorm_simd_inplace(&mut x_t, &w, EPS).is_err());
    }
}
