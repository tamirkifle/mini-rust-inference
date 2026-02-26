//! INT8 × INT8 → INT32 → f32 matrix multiplication kernel.
//!
//! # Numerics
//!
//! Given:
//!   - activation row `a[k]`  quantized with per-tensor scale `s_a`
//!   - weight row `w[n, k]`   quantized with per-channel scale `s_w[n]`
//!
//! The integer dot product is:
//!
//! ```text
//! acc_int[n] = Σ_k  a_q[k] * w_q[n, k]          (i32 accumulation)
//! out[n]     = acc_int[n] * s_a * s_w[n]          (rescale to f32)
//! ```
//!
//! Accumulating in `i32` prevents overflow: worst case per element is
//! 127 × 127 = 16 129; with K = 4096 that gives 66 125 824, well within
//! `i32::MAX = 2 147 483 647`.
//!
//! # API
//!
//! `matmul_int8` is the primary entry point.  It accepts:
//! - `act_q`   — INT8 activations already quantized (flat `[M, K]` row-major)
//! - `act_scale` — the single per-tensor scale from `quantize_symmetric`
//! - `weights`  — `&QuantizedMatrix` (per-channel, from `quantize_per_channel`)
//!
//! and returns a `Tensor<f32>` of shape `[M, N]`.
//!
//! `matmul_int8_from_f32` is a convenience wrapper that quantizes activations
//! on the fly, useful for testing and mixed-precision forward passes.

use std::borrow::Cow;

use crate::quant::int8::per_channel::QuantizedMatrix;
use crate::quant::int8::symmetric::quantize_symmetric;
use crate::tensor::{Result, Tensor, TensorError};

// ── primary kernel ────────────────────────────────────────────────────────────

/// INT8 × INT8 → INT32 → f32 GEMM.
///
/// Computes `output[m, n] = Σ_k act_q[m, k] * weights.data[n, k]`,
/// accumulating in `i32`, then rescales:
/// `output[m, n] = acc * act_scale * weights.scales[n]`.
///
/// # Arguments
///
/// * `act_q`     – flat `[M × K]` row-major INT8 activations
/// * `act_scale` – per-tensor f32 scale used to quantize `act_q`
/// * `weights`   – per-channel quantized weight matrix `[N, K]`
/// * `m`         – number of activation rows (batch / sequence length)
///
/// # Returns
///
/// Contiguous `Tensor<f32>` of shape `[M, N]`.
///
/// # Errors
///
/// [`TensorError::InvalidShape`] if dimensions are inconsistent.
#[must_use = "returns a new tensor"]
pub fn matmul_int8(
    act_q: &[i8],
    act_scale: f32,
    weights: &QuantizedMatrix,
    m: usize,
) -> Result<Tensor<f32>> {
    let n = weights.n_out;
    let k = weights.k_in;

    if act_q.len() != m * k {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_int8: act_q.len()={} != m*k={}*{}={}",
                act_q.len(), m, k, m * k
            ),
        });
    }

    let mut out = vec![0.0_f32; m * n];

    for m_i in 0..m {
        let a_row = &act_q[m_i * k..(m_i + 1) * k];
        for n_i in 0..n {
            let w_row = weights.row(n_i);
            // i32 accumulation — no overflow for K ≤ ~100 000
            let mut acc: i32 = 0;
            for k_i in 0..k {
                acc += i32::from(a_row[k_i]) * i32::from(w_row[k_i]);
            }
            out[m_i * n + n_i] = acc as f32 * act_scale * weights.scales[n_i];
        }
    }

    Tensor::from_vec(out, vec![m, n])
}

// ── convenience wrapper ───────────────────────────────────────────────────────

/// Quantizes f32 activations on the fly, then calls [`matmul_int8`].
///
/// Useful for testing and mixed-precision forward passes where activations
/// arrive as f32 but weights are already quantized.
///
/// `input` must be a 2-D contiguous-or-not `Tensor<f32>` of shape `[M, K]`.
///
/// # Errors
///
/// [`TensorError::InvalidShape`] if `input` is not 2-D or K mismatches weights.
#[must_use = "returns a new tensor"]
pub fn matmul_int8_from_f32(
    input: &Tensor<f32>,
    weights: &QuantizedMatrix,
) -> Result<Tensor<f32>> {
    if input.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_int8_from_f32: input must be 2-D, got {}D (shape {:?})",
                input.ndim(), input.dims()
            ),
        });
    }

    let m = input.dims()[0];
    let k_in = input.dims()[1];

    if k_in != weights.k_in {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, weights.k_in],
            got: vec![m, k_in],
        });
    }

    // Ensure contiguous before slicing
    let input_c: Cow<Tensor<f32>> = if input.is_contiguous() {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(input.contiguous())
    };

    let (act_q, act_scale) = quantize_symmetric(input_c.as_slice());
    matmul_int8(&act_q, act_scale, weights, m)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul::naive::matmul_naive;
    use crate::quant::int8::per_channel::quantize_per_channel;
    use crate::quant::int8::symmetric::quantize_symmetric;

    // Tolerance: sum of activation quantization error + weight quantization error
    // In the worst case both contribute ~0.5 LSB each per accumulation step.
    // For K=32 a loose relative tolerance of 2% is empirically generous.
    const REL_TOL: f32 = 0.02;

    fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| {
            let denom = x.abs().max(y.abs()).max(1e-3);
            (x - y).abs() / denom
        }).fold(0.0_f32, f32::max)
    }

    // ── shape and basic correctness ───────────────────────────────────────

    #[test]
    fn output_shape_is_m_by_n() {
        let w: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let qw = quantize_per_channel(&w, 2, 32);
        let input = Tensor::from_vec(vec![0.5_f32; 96], vec![3, 32]).unwrap();
        let out = matmul_int8_from_f32(&input, &qw).unwrap();
        assert_eq!(out.dims(), &[3, 2]);
    }

    #[test]
    fn zero_input_gives_zero_output() {
        let w: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 - 6.0).collect();
        let qw = quantize_per_channel(&w, 4, 32);
        let input = Tensor::zeros(vec![2, 32]);
        let out = matmul_int8_from_f32(&input, &qw).unwrap();
        for &v in out.as_slice() {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn zero_weights_give_zero_output() {
        let w = vec![0.0_f32; 64];
        let qw = quantize_per_channel(&w, 2, 32);
        let inp: Vec<f32> = (0..96).map(|i| i as f32 * 0.1 - 5.0).collect();
        let input = Tensor::from_vec(inp, vec![3, 32]).unwrap();
        let out = matmul_int8_from_f32(&input, &qw).unwrap();
        for &v in out.as_slice() {
            assert!(v.abs() < 1e-4, "expected ~0, got {v}");
        }
    }

    // ── within quantization error of f32 reference ────────────────────────

    #[test]
    fn matches_naive_within_quant_error_small() {
        // [2, 32] × [4, 32]^T → [2, 4]
        let w: Vec<f32> = (0..128).map(|i| (i as f32) * 0.03 - 2.0).collect();
        let inp: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.6).collect();

        let qw = quantize_per_channel(&w, 4, 32);
        let input = Tensor::from_vec(inp.clone(), vec![2, 32]).unwrap();

        // f32 reference: input @ weight^T
        let w_tensor = Tensor::from_vec(w, vec![4, 32]).unwrap();
        let ref_out = matmul_naive(&input, &w_tensor.transpose(0, 1).unwrap()).unwrap();

        let int8_out = matmul_int8_from_f32(&input, &qw).unwrap();
        let mre = max_rel_err(int8_out.as_slice(), ref_out.as_slice());
        assert!(mre < REL_TOL, "max relative error {mre:.4} >= {REL_TOL}");
    }

    #[test]
    fn matches_naive_within_quant_error_larger() {
        // [4, 128] × [8, 128]^T → [4, 8]
        let w: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let inp: Vec<f32> = (0..512).map(|i| (i as f32) * 0.02 - 5.0).collect();

        let qw = quantize_per_channel(&w, 8, 128);
        let input = Tensor::from_vec(inp.clone(), vec![4, 128]).unwrap();

        let w_tensor = Tensor::from_vec(w, vec![8, 128]).unwrap();
        let ref_out = matmul_naive(&input, &w_tensor.transpose(0, 1).unwrap()).unwrap();
        let int8_out = matmul_int8_from_f32(&input, &qw).unwrap();

        let mre = max_rel_err(int8_out.as_slice(), ref_out.as_slice());
        assert!(mre < REL_TOL, "max relative error {mre:.4} >= {REL_TOL}");
    }

    #[test]
    fn single_row_single_output_channel() {
        // 1×32 input, 1×32 weight — scalar dot product check
        let w: Vec<f32> = vec![1.0_f32; 32];
        let inp: Vec<f32> = vec![0.5_f32; 32];
        // expected dot product = 32 * 0.5 * 1.0 = 16.0
        let qw = quantize_per_channel(&w, 1, 32);
        let input = Tensor::from_vec(inp, vec![1, 32]).unwrap();
        let out = matmul_int8_from_f32(&input, &qw).unwrap();
        assert_eq!(out.dims(), &[1, 1]);
        let got = out.as_slice()[0];
        assert!((got - 16.0).abs() < 0.5, "expected ~16.0, got {got}");
    }

    // ── low-level `matmul_int8` with pre-quantized activations ────────────

    #[test]
    fn matmul_int8_direct_matches_wrapper() {
        let w: Vec<f32> = (0..96).map(|i| (i as f32) * 0.04 - 2.0).collect();
        let inp: Vec<f32> = (0..64).map(|i| (i as f32) * 0.06 - 2.0).collect();

        let qw = quantize_per_channel(&w, 3, 32);
        let input = Tensor::from_vec(inp.clone(), vec![2, 32]).unwrap();

        let (act_q, act_scale) = quantize_symmetric(&inp);

        let via_wrapper = matmul_int8_from_f32(&input, &qw).unwrap();
        let via_direct  = matmul_int8(&act_q, act_scale, &qw, 2).unwrap();

        assert_eq!(via_wrapper.as_slice(), via_direct.as_slice());
    }

    // ── non-contiguous input ──────────────────────────────────────────────

    #[test]
    fn non_contiguous_input_handled() {
        let data: Vec<f32> = (0..128).map(|i| i as f32 * 0.05 - 3.0).collect();
        let t = Tensor::from_vec(data, vec![64, 2]).unwrap();
        let input_nc = t.transpose(0, 1).unwrap(); // [2, 64], non-contiguous
        assert!(!input_nc.is_contiguous());

        let w: Vec<f32> = (0..192).map(|i| (i as f32) * 0.02 - 2.0).collect();
        let qw = quantize_per_channel(&w, 3, 64);

        // Should not panic, and should match the contiguous equivalent
        let out_nc  = matmul_int8_from_f32(&input_nc, &qw).unwrap();
        let out_c   = matmul_int8_from_f32(&input_nc.contiguous(), &qw).unwrap();
        assert_eq!(out_nc.as_slice(), out_c.as_slice());
    }

    // ── error handling ────────────────────────────────────────────────────

    #[test]
    fn error_on_non_2d_input() {
        let w = vec![0.0_f32; 32];
        let qw = quantize_per_channel(&w, 1, 32);
        let input = Tensor::from_vec(vec![1.0_f32; 64], vec![2, 4, 8]).unwrap();
        assert!(matches!(
            matmul_int8_from_f32(&input, &qw),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn error_on_k_mismatch() {
        let w = vec![0.0_f32; 64];
        let qw = quantize_per_channel(&w, 2, 32); // expects k=32
        let input = Tensor::from_vec(vec![1.0_f32; 64], vec![2, 32]).unwrap();
        // k_in matches (32==32) — this should succeed
        let _ = matmul_int8_from_f32(&input, &qw).unwrap();

        // Now mismatch: input k=64 vs weight k=32
        let qw2 = quantize_per_channel(&w, 1, 64);
        let input2 = Tensor::from_vec(vec![1.0_f32; 32], vec![1, 32]).unwrap();
        assert!(matches!(
            matmul_int8_from_f32(&input2, &qw2),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn error_on_bad_act_q_length() {
        let w = vec![0.0_f32; 64];
        let qw = quantize_per_channel(&w, 2, 32);
        let act_q = vec![0_i8; 10]; // wrong length for m=1, k=32
        assert!(matches!(
            matmul_int8(&act_q, 1.0, &qw, 1),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    // ── i32 accumulation doesn't overflow for large K ────────────────────

    #[test]
    fn no_overflow_large_k() {
        // K=4096, all activations=127, all weights=127 → acc = 127*127*4096 = 66_060_288
        // well within i32::MAX=2_147_483_647
        let k = 4096;
        let n_out = 1;
        let w: Vec<f32> = vec![127.0_f32; k];
        let qw = quantize_per_channel(&w, n_out, k);
        // Force act_q to all-127 by making input all equal to max representable
        let inp = vec![127.0_f32; k]; // scale=1.0, all quant to 127
        let input = Tensor::from_vec(inp, vec![1, k]).unwrap();
        let out = matmul_int8_from_f32(&input, &qw).unwrap();
        // Should not panic; value should be close to 127*127*4096 ≈ 66M
        let expected = 127.0_f32 * 127.0 * k as f32;
        let got = out.as_slice()[0];
        let rel = (got - expected).abs() / expected;
        assert!(rel < 0.02, "rel err {rel:.4}, got={got}, expected={expected}");
    }
}
