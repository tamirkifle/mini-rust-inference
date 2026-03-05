//! ARM NEON INT8 × INT8 → INT32 → f32 matrix multiplication kernel — commit 16.2.
//!
//! # Algorithm
//!
//! The inner dot product uses NEON 128-bit registers (16 `i8` per load):
//!
//! ```text
//! va_lo, va_hi = split vld1q_s8(a_row[0..16]) into two 8-element halves
//! vb_lo, vb_hi = split vld1q_s8(b_row[0..16]) into two 8-element halves
//! prod_lo = vmull_s8(va_lo, vb_lo)   // i8×i8→i16, 8 lanes
//! prod_hi = vmull_s8(va_hi, vb_hi)   // i8×i8→i16, 8 lanes
//! acc     = vpadalq_s16(acc, prod_lo) // pairwise add i16→i32, accumulate
//! acc     = vpadalq_s16(acc, prod_hi)
//! ```
//!
//! Each NEON iteration processes **16 `i8` elements**.  For K = 4096 that is
//! 256 iterations versus 4096 scalar multiplications.
//!
//! # Overflow guarantee
//!
//! `vmull_s8` produces `i16` results: max `|127 × 127| = 16_129`, fits in
//! `i16` (max 32_767).  `vpadalq_s16` accumulates pairs into `i32`: max
//! `32_258`, fits in `i32`.  After 256 iterations (K=4096):
//! `256 × 32_258 = 8_258_048`, well within `i32::MAX`. ✓
//!
//! # Conditional compilation
//!
//! The NEON intrinsic block is guarded by `#[cfg(target_arch = "aarch64")]`.
//! On x86_64 (including CI runners without NEON) the scalar fallback is used
//! transparently.
//!
//! # Public API
//!
//! | Function            | Description                                  |
//! |---------------------|----------------------------------------------|
//! | [`dot_i8_neon`]     | Single dot product `Σ a[i]·b[i]` in `i32`  |
//! | [`matmul_int8_neon`]| Full GEMM: `act_q × weights^T → f32`        |

use crate::quant::int8::per_channel::QuantizedMatrix;
use crate::tensor::{Result, Tensor, TensorError};

// ── NEON inner kernel (aarch64 only) ─────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use std::arch::aarch64::*;

    /// NEON `i8` dot product: `Σ a[i] · b[i]` with `i32` accumulation.
    ///
    /// Processes 16 elements per SIMD iteration using:
    /// - `vmull_s8`   : `i8×i8 → i16` (8 lanes × 2 halves = 16 elements)
    /// - `vpadalq_s16`: pairwise add `i16 → i32` + accumulate
    ///
    /// # Safety
    /// NEON is mandatory on all aarch64 targets.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
        let n       = a.len(); // caller guarantees a.len() == b.len()
        let chunks  = n / 16;  // 16 i8 per 128-bit NEON register
        let ap      = a.as_ptr();
        let bp      = b.as_ptr();
        let mut acc = vdupq_n_s32(0);

        for i in 0..chunks {
            // Load 16 signed bytes into one 128-bit register each.
            let va = vld1q_s8(ap.add(i * 16));
            let vb = vld1q_s8(bp.add(i * 16));

            // Split into low and high 8-lane halves.
            let va_lo = vget_low_s8(va);
            let va_hi = vget_high_s8(va);
            let vb_lo = vget_low_s8(vb);
            let vb_hi = vget_high_s8(vb);

            // vmull_s8: i8×i8 → i16, 8 lanes (widening multiply).
            // Overflow impossible: max |127×127| = 16_129 < i16::MAX = 32_767.
            let prod_lo = vmull_s8(va_lo, vb_lo); // int16x8_t
            let prod_hi = vmull_s8(va_hi, vb_hi); // int16x8_t

            // vpadalq_s16: pairwise add i16 lanes → i32, accumulate into acc.
            // i32 lane[j] += prod[2j] + prod[2j+1]  (both i16; sum ≤ 32_258)
            acc = vpadalq_s16(acc, prod_lo);
            acc = vpadalq_s16(acc, prod_hi);
        }

        // Horizontal sum of the 4 i32 accumulator lanes.
        let mut result = vaddvq_s32(acc);

        // Scalar tail (< 16 remaining elements).
        let tail_start = chunks * 16;
        for i in tail_start..n {
            result += i32::from(*ap.add(i)) * i32::from(*bp.add(i));
        }

        result
    }
}

// ── scalar fallback ───────────────────────────────────────────────────────

#[allow(dead_code)] // only reached on non-aarch64 targets
#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum()
}

// ── public API ────────────────────────────────────────────────────────────

/// Signed `i8` dot product: `Σ a[i] · b[i]` accumulated in `i32`.
///
/// On aarch64 this uses NEON `vmull_s8` / `vpadalq_s16`, processing
/// 16 elements per SIMD iteration.  Falls back to scalar on x86_64 and
/// other architectures (where the AVX2 path in [`super::int8_avx2`] is
/// preferred instead).
///
/// # Panics (debug)
///
/// Debug-panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_i8_neon(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len(), "dot_i8_neon: length mismatch");

    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on all aarch64 targets per the AArch64 ABI.
    return unsafe { neon_impl::dot_i8(a, b) };

    #[cfg(not(target_arch = "aarch64"))]
    dot_i8_scalar(a, b)
}

/// NEON-accelerated INT8 × INT8 → INT32 → f32 GEMM.
///
/// Computes `output[m_i, n_i] = Σ_k act_q[m_i,k] * weights[n_i,k]`,
/// accumulating in `i32`, then rescales:
/// `output = acc * act_scale * weights.scales[n_i]`.
///
/// On aarch64 the inner dot product uses NEON 16-element chunks;
/// a scalar fallback is used on x86_64 and other architectures.
///
/// # Arguments
///
/// * `act_q`      – flat `[M × K]` row-major INT8 activations
/// * `act_scale`  – per-tensor f32 scale used to quantize `act_q`
/// * `weights`    – per-channel [`QuantizedMatrix`] of shape `[N, K]`
/// * `m`          – number of activation rows
///
/// # Returns
///
/// Contiguous `Tensor<f32>` of shape `[M, N]`.
///
/// # Errors
///
/// [`TensorError::InvalidShape`] if `act_q.len() != m * k`.
#[must_use = "returns a new tensor"]
pub fn matmul_int8_neon(
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
                "matmul_int8_neon: act_q.len()={} != m*k={}*{}={}",
                act_q.len(), m, k, m * k
            ),
        });
    }

    let mut out = vec![0.0_f32; m * n];

    for m_i in 0..m {
        let a_row = &act_q[m_i * k..(m_i + 1) * k];
        for n_i in 0..n {
            let w_row = weights.row(n_i);
            let acc   = dot_i8_neon(a_row, w_row);
            out[m_i * n + n_i] = acc as f32 * act_scale * weights.scales[n_i];
        }
    }

    Tensor::from_vec(out, vec![m, n])
}

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::int8::per_channel::quantize_per_channel;
    use crate::quant::int8::symmetric::quantize_symmetric;

    // --- dot_i8_neon ---

    #[test]
    fn dot_empty() {
        assert_eq!(dot_i8_neon(&[], &[]), 0);
    }

    #[test]
    fn dot_scalar_tail_only() {
        // 5 elements: no 16-element NEON chunks, pure scalar tail.
        let a = vec![1_i8, 2, 3, 4, 5];
        let b = vec![5_i8, 4, 3, 2, 1];
        // 5+8+9+8+5 = 35
        assert_eq!(dot_i8_neon(&a, &b), 35);
    }

    #[test]
    fn dot_exactly_16_elements() {
        // Exactly one NEON chunk.
        let a: Vec<i8> = (0..16).map(|i| (i as i8) - 8).collect();
        let b: Vec<i8> = (0..16).map(|i| (i as i8) - 7).collect();
        assert_eq!(dot_i8_neon(&a, &b), dot_i8_scalar(&a, &b));
    }

    #[test]
    fn dot_17_elements() {
        // 16-element chunk + 1-element tail.
        let a: Vec<i8> = (0..17).map(|i| ((i * 3) as i8).wrapping_sub(25)).collect();
        let b: Vec<i8> = (0..17).map(|i| ((i * 5) as i8).wrapping_sub(40)).collect();
        assert_eq!(dot_i8_neon(&a, &b), dot_i8_scalar(&a, &b));
    }

    #[test]
    fn dot_all_positive() {
        let a = vec![127_i8; 32];
        let b = vec![1_i8; 32];
        assert_eq!(dot_i8_neon(&a, &b), 127 * 32);
    }

    #[test]
    fn dot_mixed_signs_zero_sum() {
        // Alternating +1 / -1 dotted with all +2 → sum = 0.
        let n = 32_usize;
        let a: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let b = vec![2_i8; n];
        assert_eq!(dot_i8_neon(&a, &b), 0);
    }

    #[test]
    fn dot_all_negative() {
        let a = vec![-4_i8; 16];
        let b = vec![-7_i8; 16];
        // dot = 28 * 16 = 448
        assert_eq!(dot_i8_neon(&a, &b), 448);
    }

    #[test]
    fn dot_large_k_no_overflow() {
        // K=4096 all 127 × 1 → 521_152, comfortably within i32::MAX.
        let a = vec![127_i8; 4096];
        let b = vec![1_i8; 4096];
        assert_eq!(dot_i8_neon(&a, &b), 127 * 4096);
    }

    #[test]
    fn dot_matches_scalar_k_128() {
        let k = 128_usize;
        let a: Vec<i8> = (0..k).map(|i| ((i * 7 + 3) as i8).wrapping_sub(64)).collect();
        let b: Vec<i8> = (0..k).map(|i| ((i * 11 + 1) as i8).wrapping_sub(64)).collect();
        assert_eq!(dot_i8_neon(&a, &b), dot_i8_scalar(&a, &b));
    }

    // --- matmul_int8_neon ---

    const REL_TOL: f32 = 0.02;

    fn max_rel_err(got: &[f32], expected: &[f32]) -> f32 {
        got.iter().zip(expected).map(|(g, e)| {
            let denom = g.abs().max(e.abs()).max(1e-3);
            (g - e).abs() / denom
        }).fold(0.0_f32, f32::max)
    }

    #[test]
    fn output_shape() {
        let w: Vec<f32> = (0..64).map(|i| i as f32 * 0.01).collect();
        let qw = quantize_per_channel(&w, 2, 32);
        let act: Vec<f32> = vec![0.5; 96];
        let (act_q, act_scale) = quantize_symmetric(&act);
        let out = matmul_int8_neon(&act_q, act_scale, &qw, 3).unwrap();
        assert_eq!(out.dims(), &[3, 2]);
    }

    #[test]
    fn matches_scalar_int8_kernel() {
        use crate::ops::matmul::matmul_int8;
        let k = 64_usize;
        let n_out = 4_usize;
        let m_rows = 3_usize;
        let w: Vec<f32> = (0..n_out * k).map(|i| (i as f32) * 0.03 - 2.0).collect();
        let inp: Vec<f32> = (0..m_rows * k).map(|i| (i as f32) * 0.05 - 1.6).collect();
        let qw = quantize_per_channel(&w, n_out, k);
        let (act_q, act_scale) = quantize_symmetric(&inp);
        let neon_out   = matmul_int8_neon(&act_q, act_scale, &qw, m_rows).unwrap();
        let scalar_out = matmul_int8(&act_q, act_scale, &qw, m_rows).unwrap();
        let mre = max_rel_err(neon_out.as_slice(), scalar_out.as_slice());
        assert!(mre < 1e-6, "NEON differs from scalar by {mre:.2e}");
    }

    #[test]
    fn neon_and_avx2_agree() {
        use crate::ops::matmul::int8_avx2::matmul_int8_avx2;
        let k = 128_usize;
        let n_out = 6_usize;
        let m_rows = 4_usize;
        let w: Vec<f32> = (0..n_out * k).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let inp: Vec<f32> = (0..m_rows * k).map(|i| (i as f32) * 0.02 - 5.0).collect();
        let qw = quantize_per_channel(&w, n_out, k);
        let (act_q, act_scale) = quantize_symmetric(&inp);
        let neon_out = matmul_int8_neon(&act_q, act_scale, &qw, m_rows).unwrap();
        let avx2_out = matmul_int8_avx2(&act_q, act_scale, &qw, m_rows).unwrap();
        let mre = max_rel_err(neon_out.as_slice(), avx2_out.as_slice());
        assert!(mre < 1e-6, "NEON and AVX2 kernels disagree by {mre:.2e}");
    }

    #[test]
    fn matches_f32_reference_within_quant_error() {
        use crate::ops::matmul::matmul_naive;
        let k = 64_usize;
        let n_out = 4_usize;
        let w: Vec<f32> = (0..n_out * k).map(|i| (i as f32) * 0.04 - 2.0).collect();
        let inp: Vec<f32> = (0..2 * k).map(|i| (i as f32) * 0.06 - 2.0).collect();
        let qw = quantize_per_channel(&w, n_out, k);
        let input_t = Tensor::from_vec(inp.clone(), vec![2, k]).unwrap();
        let w_t = Tensor::from_vec(w, vec![n_out, k]).unwrap();
        let ref_out = matmul_naive(&input_t, &w_t.transpose(0, 1).unwrap()).unwrap();
        let (act_q, act_scale) = quantize_symmetric(&inp);
        let neon_out = matmul_int8_neon(&act_q, act_scale, &qw, 2).unwrap();
        let mre = max_rel_err(neon_out.as_slice(), ref_out.as_slice());
        assert!(mre < REL_TOL, "mre={mre:.4}");
    }

    #[test]
    fn error_on_bad_act_q_length() {
        let w = vec![0.0_f32; 64];
        let qw = quantize_per_channel(&w, 2, 32);
        let act_q = vec![0_i8; 10];
        assert!(matches!(
            matmul_int8_neon(&act_q, 1.0, &qw, 1),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
