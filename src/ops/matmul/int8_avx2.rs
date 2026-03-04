//! AVX2 INT8 × INT8 → INT32 → f32 matrix multiplication kernel — commit 16.1.
//!
//! # Algorithm
//!
//! The inner dot product between two `i8` vectors uses the classic
//! `vpmaddubsw` + `vpmaddwd` two-instruction pattern:
//!
//! ```text
//! va_abs  = abs(a_row[0..32])           // treat as u8: [0, 127]
//! vb_sign = sign(b_row[0..32], a_row)   // negate b where a < 0
//! prod16  = maddubs(va_abs, vb_sign)    // u8×i8→i16, add adjacent pairs
//! prod32  = madd(prod16, 1)             // i16→i32, add adjacent pairs
//! acc    += prod32                       // i32 accumulation (no overflow)
//! ```
//!
//! Each AVX2 iteration processes **32 i8 elements** per dot product, versus
//! 1 element for the scalar fallback in `int8.rs`.
//!
//! # Overflow guarantee
//!
//! Worst-case per 32-element chunk: `127 × 127 = 16_129` per element,
//! `16_129 × 2 = 32_258` after `maddubs` pairs, `32_258 × 2 = 64_516`
//! after `madd` pairs — all safely within `i16::MAX = 32_767` and
//! `i32::MAX = 2_147_483_647`.  For K = 4096 (32-element chunks = 128):
//! `128 × 64_516 = 8_258_048`, well within `i32::MAX`. ✓
//!
//! # Public API
//!
//! | Function            | Description                                  |
//! |---------------------|----------------------------------------------|
//! | [`dot_i8_avx2`]     | Single dot product `Σ a[i]·b[i]` in `i32`  |
//! | [`matmul_int8_avx2`]| Full GEMM: `act_q × weights^T → f32`        |

use crate::quant::int8::per_channel::QuantizedMatrix;
use crate::tensor::{Result, Tensor, TensorError};

// ── AVX2 inner kernel (x86_64 only) ──────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use std::arch::x86_64::*;

    /// Horizontal sum of 8 `i32` lanes in a 256-bit AVX2 register.
    ///
    /// # Safety
    /// Caller must guarantee AVX2 is available.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hsum_epi32(v: __m256i) -> i32 {
        // Fold upper 128 bits onto lower 128 bits.
        let lo128 = _mm256_castsi256_si128(v);
        let hi128 = _mm256_extracti128_si256(v, 1);
        let sum4  = _mm_add_epi32(lo128, hi128);
        // Pair-sum the 4 lanes.
        let shuf  = _mm_shuffle_epi32(sum4, 0b10_11_00_01); // [b,a,d,c]
        let sum2  = _mm_add_epi32(sum4, shuf);              // [a+b, _, c+d, _]
        let shuf2 = _mm_shuffle_epi32(sum2, 0b01_00_11_10); // move c+d to lo
        let sum1  = _mm_add_epi32(sum2, shuf2);
        _mm_cvtsi128_si32(sum1)
    }

    /// AVX2 `i8` dot product: `Σ a[i] · b[i]` with `i32` accumulation.
    ///
    /// Processes 32 elements per SIMD iteration.
    ///
    /// # Safety
    /// Caller must guarantee AVX2 is available.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
        let n      = a.len(); // caller guarantees a.len() == b.len()
        let chunks = n / 32;
        let ap     = a.as_ptr();
        let bp     = b.as_ptr();
        let ones   = _mm256_set1_epi16(1_i16);
        let mut acc = _mm256_setzero_si256();

        for i in 0..chunks {
            // Load 32 signed i8 values.
            let va = _mm256_loadu_si256(ap.add(i * 32) as *const __m256i);
            let vb = _mm256_loadu_si256(bp.add(i * 32) as *const __m256i);

            // abs(a): treat as u8 so maddubs accepts it as the unsigned arg.
            let va_abs  = _mm256_abs_epi8(va);
            // sign(b, a): negates b[k] where a[k] < 0, zeroes where a[k]==0.
            // Combined with va_abs, this restores the correct signed product:
            //   abs(a[k]) * sign(b[k], a[k]) == a[k] * b[k]
            let vb_sign = _mm256_sign_epi8(vb, va);

            // _mm256_maddubs_epi16: (u8, i8) → i16, multiply + add adjacent pairs.
            //   prod16[j] = va_abs[2j] * vb_sign[2j] + va_abs[2j+1] * vb_sign[2j+1]
            // Max value: 127*127 + 127*127 = 32_258 — fits in i16 (max 32_767). ✓
            let prod16 = _mm256_maddubs_epi16(va_abs, vb_sign);

            // _mm256_madd_epi16(v, ones): i16 → i32, add adjacent pairs.
            //   prod32[j] = prod16[2j] * 1 + prod16[2j+1] * 1
            let prod32 = _mm256_madd_epi16(prod16, ones);

            acc = _mm256_add_epi32(acc, prod32);
        }

        // Horizontal sum of the 8 i32 accumulator lanes.
        let mut result = hsum_epi32(acc);

        // Scalar tail (< 32 remaining elements).
        let tail_start = chunks * 32;
        for i in tail_start..n {
            result += i32::from(*ap.add(i)) * i32::from(*bp.add(i));
        }

        result
    }
}

// ── scalar fallback ───────────────────────────────────────────────────────

#[inline]
fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| i32::from(x) * i32::from(y)).sum()
}

// ── public API ────────────────────────────────────────────────────────────

/// Signed `i8` dot product: `Σ a[i] · b[i]` accumulated in `i32`.
///
/// On x86_64 with AVX2 this uses the `vpmaddubsw`/`vpmaddwd` trick,
/// processing 32 elements per SIMD cycle.  Falls back to scalar on
/// non-AVX2 x86_64 and all other architectures.
///
/// # Panics (debug)
///
/// Debug-panics if `a.len() != b.len()`.
#[must_use]
pub fn dot_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len(), "dot_i8_avx2: length mismatch");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 confirmed by runtime detection.
            return unsafe { avx2_impl::dot_i8(a, b) };
        }
    }

    dot_i8_scalar(a, b)
}

/// AVX2-accelerated INT8 × INT8 → INT32 → f32 GEMM.
///
/// Computes `output[m_i, n_i] = Σ_k act_q[m_i,k] * weights[n_i,k]`,
/// accumulating in `i32`, then rescales:
/// `output = acc * act_scale * weights.scales[n_i]`.
///
/// On x86_64 the inner dot product uses 32-element AVX2 SIMD chunks;
/// a scalar fallback is used on other architectures.
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
pub fn matmul_int8_avx2(
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
                "matmul_int8_avx2: act_q.len()={} != m*k={}*{}={}",
                act_q.len(), m, k, m * k
            ),
        });
    }

    let mut out = vec![0.0_f32; m * n];

    for m_i in 0..m {
        let a_row = &act_q[m_i * k..(m_i + 1) * k];
        for n_i in 0..n {
            let w_row  = weights.row(n_i);
            let acc    = dot_i8_avx2(a_row, w_row);
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

    // --- dot_i8_avx2 ---

    #[test]
    fn dot_empty() {
        assert_eq!(dot_i8_avx2(&[], &[]), 0);
    }

    #[test]
    fn dot_scalar_only() {
        // 3 elements: no SIMD chunks, pure tail
        let a = vec![1_i8, 2, 3];
        let b = vec![4_i8, 5, 6];
        assert_eq!(dot_i8_avx2(&a, &b), 4 + 10 + 18); // 32
    }

    #[test]
    fn dot_matches_scalar_32_elements() {
        // Exactly one AVX2 chunk (32 elements).
        let a: Vec<i8> = (0..32).map(|i| (i as i8).wrapping_sub(16)).collect();
        let b: Vec<i8> = (0..32).map(|i| (i as i8).wrapping_mul(2).wrapping_sub(30)).collect();
        let simd   = dot_i8_avx2(&a, &b);
        let scalar = dot_i8_scalar(&a, &b);
        assert_eq!(simd, scalar, "SIMD={simd} scalar={scalar}");
    }

    #[test]
    fn dot_matches_scalar_33_elements() {
        // 32-element chunk + 1-element tail.
        let a: Vec<i8> = (0..33).map(|i| ((i * 3) as i8).wrapping_sub(50)).collect();
        let b: Vec<i8> = (0..33).map(|i| ((i * 5) as i8).wrapping_sub(80)).collect();
        assert_eq!(dot_i8_avx2(&a, &b), dot_i8_scalar(&a, &b));
    }

    #[test]
    fn dot_all_positive() {
        let a = vec![127_i8; 64];
        let b = vec![1_i8; 64];
        // dot = 127 * 64 = 8128
        assert_eq!(dot_i8_avx2(&a, &b), 8128);
    }

    #[test]
    fn dot_mixed_signs() {
        // a = [1, -1, 1, -1, ...] b = [2, 2, 2, 2, ...]
        // dot = n/2 * (2) + n/2 * (-2) = 0  for even n
        let n = 64_usize;
        let a: Vec<i8> = (0..n).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect();
        let b = vec![2_i8; n];
        assert_eq!(dot_i8_avx2(&a, &b), 0);
    }

    #[test]
    fn dot_all_negative() {
        let a = vec![-3_i8; 32];
        let b = vec![-5_i8; 32];
        // dot = 32 * 15 = 480
        assert_eq!(dot_i8_avx2(&a, &b), 480);
    }

    #[test]
    fn dot_large_k_no_overflow() {
        // K=4096, a=127, b=1 → dot=521152 — well within i32
        let a = vec![127_i8; 4096];
        let b = vec![1_i8; 4096];
        let got = dot_i8_avx2(&a, &b);
        assert_eq!(got, 127 * 4096);
    }

    #[test]
    fn dot_matches_scalar_k_128() {
        let k = 128_usize;
        let a: Vec<i8> = (0..k).map(|i| ((i * 7 + 3) as i8).wrapping_sub(64)).collect();
        let b: Vec<i8> = (0..k).map(|i| ((i * 11 + 1) as i8).wrapping_sub(64)).collect();
        assert_eq!(dot_i8_avx2(&a, &b), dot_i8_scalar(&a, &b));
    }

    // --- matmul_int8_avx2 ---

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
        let out = matmul_int8_avx2(&act_q, act_scale, &qw, 3).unwrap();
        assert_eq!(out.dims(), &[3, 2]);
    }

    #[test]
    fn matches_scalar_int8_small() {
        use crate::ops::matmul::matmul_int8;
        let w: Vec<f32> = (0..128).map(|i| (i as f32) * 0.03 - 2.0).collect();
        let inp: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.6).collect();
        let qw = quantize_per_channel(&w, 4, 32);
        let (act_q, act_scale) = quantize_symmetric(&inp);

        let avx2_out   = matmul_int8_avx2(&act_q, act_scale, &qw, 2).unwrap();
        let scalar_out = matmul_int8(&act_q, act_scale, &qw, 2).unwrap();

        let mre = max_rel_err(avx2_out.as_slice(), scalar_out.as_slice());
        assert!(mre < 1e-6, "AVX2 differs from scalar by {mre:.2e}");
    }

    #[test]
    fn matches_scalar_int8_larger_k() {
        use crate::ops::matmul::matmul_int8;
        let k = 128_usize;
        let n_out = 8_usize;
        let m_rows = 4_usize;
        let w: Vec<f32> = (0..n_out * k).map(|i| (i as f32) * 0.01 - 5.0).collect();
        let inp: Vec<f32> = (0..m_rows * k).map(|i| (i as f32) * 0.02 - 5.0).collect();
        let qw = quantize_per_channel(&w, n_out, k);
        let (act_q, act_scale) = quantize_symmetric(&inp);

        let avx2_out   = matmul_int8_avx2(&act_q, act_scale, &qw, m_rows).unwrap();
        let scalar_out = matmul_int8(&act_q, act_scale, &qw, m_rows).unwrap();
        let mre = max_rel_err(avx2_out.as_slice(), scalar_out.as_slice());
        assert!(mre < 1e-6, "mre={mre:.2e}");
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
        let avx2_out = matmul_int8_avx2(&act_q, act_scale, &qw, 2).unwrap();
        let mre = max_rel_err(avx2_out.as_slice(), ref_out.as_slice());
        assert!(mre < REL_TOL, "mre={mre:.4}");
    }

    #[test]
    fn error_on_bad_act_q_length() {
        let w = vec![0.0_f32; 64];
        let qw = quantize_per_channel(&w, 2, 32);
        let act_q = vec![0_i8; 10];
        assert!(matches!(
            matmul_int8_avx2(&act_q, 1.0, &qw, 1),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
