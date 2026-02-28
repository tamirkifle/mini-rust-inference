//! Portable SIMD f32 primitives — commit 15.1.
//!
//! All public functions are **safe**. Platform-specific kernels use `unsafe`
//! intrinsics dispatched at runtime (x86_64 AVX2+FMA detection) or at
//! compile-time (aarch64, where NEON is always present).
//!
//! # Operations
//!
//! | Function       | Semantics                  | Key consumer          |
//! |----------------|----------------------------|-----------------------|
//! | [`hsum`]       | Σ aᵢ                       | RMSNorm, softmax      |
//! | [`dot`]        | Σ aᵢ·bᵢ                    | matmul inner loop     |
//! | [`add_into`]   | dst[i] = a[i] + b[i]       | residual connections  |
//! | [`mul_into`]   | dst[i] = a[i] · b[i]       | SwiGLU gate           |
//! | [`scale_into`] | dst[i] = src[i] · s        | RMSNorm weight·norm   |
//! | [`fma_into`]   | dst[i] = a[i]·b[i] + c[i] | fused bias-add        |

// ── scalar fallbacks (reference + non-SIMD platform) ──────────────────────

#[inline]
pub(crate) fn hsum_scalar(a: &[f32]) -> f32 {
    a.iter().sum()
}

#[inline]
pub(crate) fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
#[allow(dead_code)] // fallback for non-SIMD platforms; also used in tests
fn add_into_scalar(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..dst.len() {
        dst[i] = a[i] + b[i];
    }
}

#[inline]
#[allow(dead_code)]
fn mul_into_scalar(dst: &mut [f32], a: &[f32], b: &[f32]) {
    for i in 0..dst.len() {
        dst[i] = a[i] * b[i];
    }
}

#[inline]
#[allow(dead_code)]
fn scale_into_scalar(dst: &mut [f32], src: &[f32], s: f32) {
    for i in 0..dst.len() {
        dst[i] = src[i] * s;
    }
}

#[inline]
#[allow(dead_code)]
fn fma_into_scalar(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    for i in 0..dst.len() {
        dst[i] = a[i].mul_add(b[i], c[i]);
    }
}

// ── x86_64 AVX2 kernels ───────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// Reduce an 8-lane AVX register to a single f32 via tree reduction.
    ///
    /// # Safety
    /// Caller must ensure AVX2 is available (implied by the `avx2` target-feature).
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hsum_m256(v: __m256) -> f32 {
        // fold upper 128 bits onto lower 128 bits
        let hi128 = _mm256_extractf128_ps(v, 1);
        let lo128 = _mm256_castps256_ps128(v);
        let sum4  = _mm_add_ps(lo128, hi128);
        // [a,b,c,d] → shuffle to [b,a,d,c], then add: [a+b, a+b, c+d, c+d]
        let shuf  = _mm_shuffle_ps(sum4, sum4, 0b10_11_00_01);
        let sum2  = _mm_add_ps(sum4, shuf);
        // movehl gives [c+d, c+d, _, _]; add low scalar
        let shuf2 = _mm_movehl_ps(sum2, sum2);
        let sum1  = _mm_add_ss(sum2, shuf2);
        _mm_cvtss_f32(sum1)
    }

    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn hsum(a: &[f32]) -> f32 {
        let n      = a.len();
        let chunks = n / 8;
        let p      = a.as_ptr();
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            acc = _mm256_add_ps(acc, _mm256_loadu_ps(p.add(i * 8)));
        }
        hsum_m256(acc) + super::hsum_scalar(&a[chunks * 8..])
    }

    /// # Safety
    /// Requires AVX2 + FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n      = a.len();    // caller guarantees a.len() == b.len()
        let chunks = n / 8;
        let ap     = a.as_ptr();
        let bp     = b.as_ptr();
        let mut acc = _mm256_setzero_ps();
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ap.add(i * 8));
            let vb = _mm256_loadu_ps(bp.add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }
        hsum_m256(acc) + super::dot_scalar(&a[chunks * 8..], &b[chunks * 8..])
    }

    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
        let n      = dst.len();
        let chunks = n / 8;
        let dp     = dst.as_mut_ptr();
        let ap     = a.as_ptr();
        let bp     = b.as_ptr();
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ap.add(i * 8));
            let vb = _mm256_loadu_ps(bp.add(i * 8));
            _mm256_storeu_ps(dp.add(i * 8), _mm256_add_ps(va, vb));
        }
        for i in (chunks * 8)..n {
            dst[i] = a[i] + b[i];
        }
    }

    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn mul_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
        let n      = dst.len();
        let chunks = n / 8;
        let dp     = dst.as_mut_ptr();
        let ap     = a.as_ptr();
        let bp     = b.as_ptr();
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ap.add(i * 8));
            let vb = _mm256_loadu_ps(bp.add(i * 8));
            _mm256_storeu_ps(dp.add(i * 8), _mm256_mul_ps(va, vb));
        }
        for i in (chunks * 8)..n {
            dst[i] = a[i] * b[i];
        }
    }

    /// # Safety
    /// Requires AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn scale_into(dst: &mut [f32], src: &[f32], s: f32) {
        let n      = dst.len();
        let chunks = n / 8;
        let dp     = dst.as_mut_ptr();
        let sp     = src.as_ptr();
        let vs     = _mm256_set1_ps(s);
        for i in 0..chunks {
            let va = _mm256_loadu_ps(sp.add(i * 8));
            _mm256_storeu_ps(dp.add(i * 8), _mm256_mul_ps(va, vs));
        }
        for i in (chunks * 8)..n {
            dst[i] = src[i] * s;
        }
    }

    /// # Safety
    /// Requires AVX2 + FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn fma_into(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
        let n      = dst.len();
        let chunks = n / 8;
        let dp     = dst.as_mut_ptr();
        let ap     = a.as_ptr();
        let bp     = b.as_ptr();
        let cp     = c.as_ptr();
        for i in 0..chunks {
            let va = _mm256_loadu_ps(ap.add(i * 8));
            let vb = _mm256_loadu_ps(bp.add(i * 8));
            let vc = _mm256_loadu_ps(cp.add(i * 8));
            // _mm256_fmadd_ps(a, b, c) = a*b + c
            _mm256_storeu_ps(dp.add(i * 8), _mm256_fmadd_ps(va, vb, vc));
        }
        for i in (chunks * 8)..n {
            dst[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

// ── aarch64 NEON kernels ──────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    /// # Safety
    /// Requires NEON (always available on aarch64).
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn hsum(a: &[f32]) -> f32 {
        let n      = a.len();
        let chunks = n / 4;
        let p      = a.as_ptr();
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            acc = vaddq_f32(acc, vld1q_f32(p.add(i * 4)));
        }
        vaddvq_f32(acc) + super::hsum_scalar(&a[chunks * 4..])
    }

    /// # Safety
    /// Requires NEON.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(a: &[f32], b: &[f32]) -> f32 {
        let n      = a.len();
        let chunks = n / 4;
        let ap     = a.as_ptr();
        let bp     = b.as_ptr();
        let mut acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let va = vld1q_f32(ap.add(i * 4));
            let vb = vld1q_f32(bp.add(i * 4));
            // vfmaq_f32(a, b, c) = a + b*c
            acc = vfmaq_f32(acc, va, vb);
        }
        vaddvq_f32(acc) + super::dot_scalar(&a[chunks * 4..], &b[chunks * 4..])
    }

    /// # Safety
    /// Requires NEON.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
        let n      = dst.len();
        let chunks = n / 4;
        let dp     = dst.as_mut_ptr();
        let ap     = a.as_ptr();
        let bp     = b.as_ptr();
        for i in 0..chunks {
            vst1q_f32(dp.add(i * 4), vaddq_f32(vld1q_f32(ap.add(i * 4)), vld1q_f32(bp.add(i * 4))));
        }
        for i in (chunks * 4)..n {
            dst[i] = a[i] + b[i];
        }
    }

    /// # Safety
    /// Requires NEON.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn mul_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
        let n      = dst.len();
        let chunks = n / 4;
        let dp     = dst.as_mut_ptr();
        let ap     = a.as_ptr();
        let bp     = b.as_ptr();
        for i in 0..chunks {
            vst1q_f32(dp.add(i * 4), vmulq_f32(vld1q_f32(ap.add(i * 4)), vld1q_f32(bp.add(i * 4))));
        }
        for i in (chunks * 4)..n {
            dst[i] = a[i] * b[i];
        }
    }

    /// # Safety
    /// Requires NEON.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn scale_into(dst: &mut [f32], src: &[f32], s: f32) {
        let n      = dst.len();
        let chunks = n / 4;
        let dp     = dst.as_mut_ptr();
        let sp     = src.as_ptr();
        let vs     = vdupq_n_f32(s);
        for i in 0..chunks {
            vst1q_f32(dp.add(i * 4), vmulq_f32(vld1q_f32(sp.add(i * 4)), vs));
        }
        for i in (chunks * 4)..n {
            dst[i] = src[i] * s;
        }
    }

    /// # Safety
    /// Requires NEON.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn fma_into(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
        let n      = dst.len();
        let chunks = n / 4;
        let dp     = dst.as_mut_ptr();
        let ap     = a.as_ptr();
        let bp     = b.as_ptr();
        let cp     = c.as_ptr();
        for i in 0..chunks {
            let va = vld1q_f32(ap.add(i * 4));
            let vb = vld1q_f32(bp.add(i * 4));
            let vc = vld1q_f32(cp.add(i * 4));
            // vfmaq_f32(a, b, c) = a + b*c; so dst = vc + va*vb
            vst1q_f32(dp.add(i * 4), vfmaq_f32(vc, va, vb));
        }
        for i in (chunks * 4)..n {
            dst[i] = a[i].mul_add(b[i], c[i]);
        }
    }
}

// ── arch-specific dispatch functions ─────────────────────────────────────

// --- hsum ---
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_hsum(a: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: feature check guarantees AVX2 is available.
        unsafe { x86::hsum(a) }
    } else {
        hsum_scalar(a)
    }
}
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_hsum(a: &[f32]) -> f32 {
    // SAFETY: NEON is mandatory on aarch64.
    unsafe { neon::hsum(a) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_hsum(a: &[f32]) -> f32 { hsum_scalar(a) }

// --- dot ---
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_dot(a: &[f32], b: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 and FMA features confirmed at runtime.
        unsafe { x86::dot(a, b) }
    } else {
        dot_scalar(a, b)
    }
}
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_dot(a: &[f32], b: &[f32]) -> f32 {
    // SAFETY: NEON is mandatory on aarch64.
    unsafe { neon::dot(a, b) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_dot(a: &[f32], b: &[f32]) -> f32 { dot_scalar(a, b) }

// --- add_into ---
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    if is_x86_feature_detected!("avx2") {
        // SAFETY: AVX2 confirmed.
        unsafe { x86::add_into(dst, a, b) }
    } else {
        add_into_scalar(dst, a, b)
    }
}
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    unsafe { neon::add_into(dst, a, b) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_add_into(dst: &mut [f32], a: &[f32], b: &[f32]) { add_into_scalar(dst, a, b) }

// --- mul_into ---
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_mul_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    if is_x86_feature_detected!("avx2") {
        unsafe { x86::mul_into(dst, a, b) }
    } else {
        mul_into_scalar(dst, a, b)
    }
}
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_mul_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    unsafe { neon::mul_into(dst, a, b) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_mul_into(dst: &mut [f32], a: &[f32], b: &[f32]) { mul_into_scalar(dst, a, b) }

// --- scale_into ---
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_scale_into(dst: &mut [f32], src: &[f32], s: f32) {
    if is_x86_feature_detected!("avx2") {
        unsafe { x86::scale_into(dst, src, s) }
    } else {
        scale_into_scalar(dst, src, s)
    }
}
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_scale_into(dst: &mut [f32], src: &[f32], s: f32) {
    unsafe { neon::scale_into(dst, src, s) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_scale_into(dst: &mut [f32], src: &[f32], s: f32) { scale_into_scalar(dst, src, s) }

// --- fma_into ---
#[cfg(target_arch = "x86_64")]
#[inline]
fn dispatch_fma_into(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { x86::fma_into(dst, a, b, c) }
    } else {
        fma_into_scalar(dst, a, b, c)
    }
}
#[cfg(target_arch = "aarch64")]
#[inline]
fn dispatch_fma_into(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    unsafe { neon::fma_into(dst, a, b, c) }
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline]
fn dispatch_fma_into(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) { fma_into_scalar(dst, a, b, c) }

// ── public safe API ───────────────────────────────────────────────────────

/// Horizontal sum: `Σ aᵢ`.
#[must_use]
pub fn hsum(a: &[f32]) -> f32 {
    dispatch_hsum(a)
}

/// Dot product: `Σ aᵢ · bᵢ`.
///
/// # Panics
///
/// Debug-panics if `a.len() != b.len()`.
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "simd::f32::dot: length mismatch");
    dispatch_dot(a, b)
}

/// Element-wise add: `dst[i] = a[i] + b[i]`.
///
/// # Panics
///
/// Debug-panics if slice lengths differ.
pub fn add_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    dispatch_add_into(dst, a, b);
}

/// Element-wise multiply: `dst[i] = a[i] · b[i]`.
pub fn mul_into(dst: &mut [f32], a: &[f32], b: &[f32]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    dispatch_mul_into(dst, a, b);
}

/// Scale: `dst[i] = src[i] · s`.
pub fn scale_into(dst: &mut [f32], src: &[f32], s: f32) {
    debug_assert_eq!(dst.len(), src.len());
    dispatch_scale_into(dst, src, s);
}

/// Fused multiply-add: `dst[i] = a[i] · b[i] + c[i]`.
pub fn fma_into(dst: &mut [f32], a: &[f32], b: &[f32], c: &[f32]) {
    debug_assert_eq!(dst.len(), a.len());
    debug_assert_eq!(dst.len(), b.len());
    debug_assert_eq!(dst.len(), c.len());
    dispatch_fma_into(dst, a, b, c);
}

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-5;
    fn close(a: f32, b: f32) -> bool { (a - b).abs() < TOL }
    fn close_slice(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
    }

    // ── hsum ─────────────────────────────────────────────────────────────

    #[test]
    fn test_hsum_empty() {
        assert_eq!(hsum(&[]), 0.0_f32);
    }

    #[test]
    fn test_hsum_basic() {
        assert!(close(hsum(&[1.0, 2.0, 3.0, 4.0]), 10.0));
    }

    #[test]
    fn test_hsum_matches_scalar_large() {
        // 97 elements: 12 chunks of 8 (96) + tail of 1 for AVX2
        //               24 chunks of 4 (96) + tail of 1 for NEON
        let a: Vec<f32> = (0..97).map(|i| i as f32 * 0.5).collect();
        assert!(close(hsum(&a), hsum_scalar(&a)));
    }

    // ── dot ──────────────────────────────────────────────────────────────

    #[test]
    fn test_dot_basic() {
        // [1,2,3]·[4,5,6] = 4+10+18 = 32
        assert!(close(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0));
    }

    #[test]
    fn test_dot_matches_scalar_large() {
        // NEON uses fused vfmaq_f32 (single rounding per iteration) while the
        // scalar path does separate mul+add.  For 100 elements the two can
        // diverge by a few ULPs in the accumulated result, so we use a
        // relative tolerance of 1e-4 here rather than the strict 1e-5 used
        // for simpler cases.
        let a: Vec<f32> = (0..100).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..100).map(|i| (100 - i) as f32 * 0.1).collect();
        let simd_result   = dot(&a, &b);
        let scalar_result = dot_scalar(&a, &b);
        let rel_err = (simd_result - scalar_result).abs() / scalar_result.abs().max(1e-10);
        assert!(rel_err < 1e-4,
            "rel_err {rel_err:.2e} too large (simd={simd_result}, scalar={scalar_result})");
    }

    #[test]
    fn test_dot_tail_handling() {
        // 9 elements: tests chunk-of-8 + single tail element (AVX2 path)
        let a: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..9).map(|i| i as f32 + 1.0).collect();
        assert!(close(dot(&a, &b), dot_scalar(&a, &b)));
    }

    // ── add_into ─────────────────────────────────────────────────────────

    #[test]
    fn test_add_into_matches_scalar() {
        let a: Vec<f32> = (0..25).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..25).map(|i| (25 - i) as f32).collect();
        let mut dst_simd   = vec![0.0_f32; 25];
        let mut dst_scalar = vec![0.0_f32; 25];
        add_into(&mut dst_simd, &a, &b);
        add_into_scalar(&mut dst_scalar, &a, &b);
        assert!(close_slice(&dst_simd, &dst_scalar));
    }

    // ── mul_into ─────────────────────────────────────────────────────────

    #[test]
    fn test_mul_into_matches_scalar() {
        let a: Vec<f32> = (1..=17).map(|i| i as f32).collect();
        let b: Vec<f32> = (1..=17).map(|i| 1.0 / i as f32).collect();
        let mut dst_simd   = vec![0.0_f32; 16];
        let mut dst_scalar = vec![0.0_f32; 16];
        mul_into(&mut dst_simd, &a[..16], &b[..16]);
        mul_into_scalar(&mut dst_scalar, &a[..16], &b[..16]);
        assert!(close_slice(&dst_simd, &dst_scalar));
    }

    // ── scale_into ───────────────────────────────────────────────────────

    #[test]
    fn test_scale_into_matches_scalar() {
        let src: Vec<f32> = (0..13).map(|i| i as f32).collect();
        let s = 2.5_f32;
        let mut dst_simd   = vec![0.0_f32; 13];
        let mut dst_scalar = vec![0.0_f32; 13];
        scale_into(&mut dst_simd, &src, s);
        scale_into_scalar(&mut dst_scalar, &src, s);
        assert!(close_slice(&dst_simd, &dst_scalar));
    }

    // ── fma_into ─────────────────────────────────────────────────────────

    #[test]
    fn test_fma_into_matches_scalar() {
        let a: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..20).map(|i| i as f32 * 0.5).collect();
        let c: Vec<f32> = (0..20).map(|i| -(i as f32)).collect();
        let mut dst_simd   = vec![0.0_f32; 20];
        let mut dst_scalar = vec![0.0_f32; 20];
        fma_into(&mut dst_simd, &a, &b, &c);
        fma_into_scalar(&mut dst_scalar, &a, &b, &c);
        assert!(close_slice(&dst_simd, &dst_scalar));
    }

    #[test]
    fn test_fma_into_zero_addend() {
        // fma(a, b, 0) should equal a*b
        let a = vec![1.0_f32, 2.0, 3.0, 4.0];
        let b = vec![5.0_f32, 6.0, 7.0, 8.0];
        let c = vec![0.0_f32; 4];
        let mut dst = vec![0.0_f32; 4];
        fma_into(&mut dst, &a, &b, &c);
        let expected = [5.0, 12.0, 21.0, 32.0];
        assert!(close_slice(&dst, &expected));
    }
}
