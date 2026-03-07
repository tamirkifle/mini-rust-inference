//! ARM NEON f32 matrix multiplication kernel — commit 18.5.
//!
//! # Why this exists
//!
//! The blocked matmul (`blocked.rs`) uses three nested `while` loops with
//! runtime `.min()` boundary clamping.  LLVM cannot prove the inner `j` loop
//! has a fixed stride-1 width (it changes at tile boundaries), so it falls
//! back to scalar or narrow 2-wide SIMD instead of emitting wide NEON `fmla`
//! instructions.  The result is flat ~9 Gelem/s at all matrix sizes while the
//! naive `for j in 0..n` loop reaches 23 Gelem/s.
//!
//! Explicit NEON intrinsics bypass LLVM's ambiguity entirely.
//!
//! # Algorithm — "outer-product row" (axpy decomposition)
//!
//! ```text
//! for i in 0..M:
//!     c_row = [0.0; N]
//!     for kk in 0..K:
//!         va = vdupq_n_f32(A[i, kk])           // broadcast scalar → 4 lanes
//!         for j4 in 0..N/4:
//!             vb = vld1q_f32(&B[kk, j4*4])     // load 4 contiguous f32
//!             vc = vld1q_f32(&c_row[j4*4])
//!             vc = vfmaq_f32(vc, va, vb)        // vc += va * vb  (4 f32/cycle)
//!             vst1q_f32(&mut c_row[j4*4], vc)
//!     C[i, :] = c_row
//! ```
//!
//! B rows are contiguous in row-major layout so every `vld1q_f32(bp.add(j4*4))`
//! is a sequential load — no gather/scatter overhead.
//!
//! # Integration
//!
//! - `matmul_blocked` on aarch64 calls [`neon_gemm_slice`] instead of its
//!   tiled while loops.
//! - `matmul_parallel` on aarch64 calls [`neon_gemm_row_slice`] inside each
//!   rayon task, combining NEON throughput with multi-core parallelism.
//! - On x86_64 both functions use a scalar axpy fallback; use `matmul_avx2`
//!   for the fast f32 path on that architecture.
//!
//! # Public API
//!
//! | Function              | Description                                |
//! |-----------------------|--------------------------------------------|
//! | [`matmul_neon_f32`]   | Full GEMM on `Tensor<f32>` inputs          |
//! | [`neon_gemm_slice`]   | Raw-slice GEMM used by `blocked.rs`        |
//! | [`neon_gemm_row_slice`]| Single-row GEMM used by `parallel.rs`    |

use std::borrow::Cow;
use crate::tensor::{Result, Tensor, TensorError};

// ── NEON kernel (aarch64 only) ────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
mod neon_impl {
    use std::arch::aarch64::*;

    /// Compute one output row with NEON `vfmaq_f32` (4 f32/cycle).
    ///
    /// Zeros `c_row`, then computes `c_row[j] = Σ_k a_row[k] * b[k*n + j]`.
    ///
    /// # Arguments
    /// * `a_row` – `[K]` activation row
    /// * `b`     – `[K × N]` weight matrix, row-major
    /// * `c_row` – `[N]` output row (overwritten)
    ///
    /// # Safety
    /// NEON is mandatory on all aarch64 targets per the AArch64 ABI.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn gemm_row(
        a_row: &[f32],
        b: &[f32],
        c_row: &mut [f32],
        k: usize,
        n: usize,
    ) {
        let chunks = n / 4;
        let tail   = n % 4;
        let cp     = c_row.as_mut_ptr();

        // Zero c_row in 4-wide NEON stores.
        let zero = vdupq_n_f32(0.0_f32);
        for j4 in 0..chunks {
            vst1q_f32(cp.add(j4 * 4), zero);
        }
        for j in (chunks * 4)..n {
            *cp.add(j) = 0.0;
        }

        // axpy: for each k step, broadcast a_row[kk] and fma into c_row.
        for kk in 0..k {
            let va = vdupq_n_f32(*a_row.get_unchecked(kk));
            let bp = b.as_ptr().add(kk * n);

            for j4 in 0..chunks {
                let vb = vld1q_f32(bp.add(j4 * 4));
                let vc = vld1q_f32(cp.add(j4 * 4));
                vst1q_f32(cp.add(j4 * 4), vfmaq_f32(vc, va, vb));
            }

            // Scalar tail for n % 4 remaining elements.
            let base = chunks * 4;
            for j in 0..tail {
                *cp.add(base + j) += *a_row.get_unchecked(kk) * *bp.add(base + j);
            }
        }
    }

    /// Full GEMM: iterate over M rows, calling `gemm_row` for each.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn gemm(
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for i in 0..m {
            let a_row = a.get_unchecked(i * k..(i + 1) * k);
            let c_row = std::slice::from_raw_parts_mut(c.as_mut_ptr().add(i * n), n);
            gemm_row(a_row, b, c_row, k, n);
        }
    }
}

// ── scalar fallback ───────────────────────────────────────────────────────────

#[cfg(not(target_arch = "aarch64"))]
fn gemm_scalar_row(a_row: &[f32], b: &[f32], c_row: &mut [f32], k: usize, n: usize) {
    for j in 0..n { c_row[j] = 0.0; }
    for kk in 0..k {
        let a_ik  = a_row[kk];
        let b_row = &b[kk * n..(kk + 1) * n];
        for j in 0..n {
            c_row[j] += a_ik * b_row[j];
        }
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn gemm_scalar(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let a_row = &a[i * k..(i + 1) * k];
        let c_row = &mut c[i * n..(i + 1) * n];
        gemm_scalar_row(a_row, b, c_row, k, n);
    }
}

// ── pub(crate) raw-slice interfaces ──────────────────────────────────────────
//
// These are lower-level than the Tensor API and used by sibling modules
// (blocked.rs, parallel.rs) to avoid Tensor overhead in tight loops.

/// NEON-accelerated GEMM on flat f32 slices.
///
/// `a`: `[m × k]` row-major, `b`: `[k × n]` row-major, `c`: `[m × n]` output
/// (overwritten — all elements will be zeroed then filled).
///
/// Uses NEON `vfmaq_f32` on aarch64; scalar axpy fallback elsewhere.
pub(crate) fn neon_gemm_slice(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on all aarch64 targets per the AArch64 ABI.
    return unsafe { neon_impl::gemm(a, b, c, m, k, n) };

    #[cfg(not(target_arch = "aarch64"))]
    gemm_scalar(a, b, c, m, k, n);
}

/// NEON-accelerated single-row GEMM.
///
/// Computes `c_row[j] = Σ_k a_row[k] * b[k*n + j]`, zeroing `c_row` first.
///
/// * `a_row` – `[k]` activation row
/// * `b`     – `[k × n]` weight matrix, full row-major slice
/// * `c_row` – `[n]` output row (overwritten)
///
/// Uses NEON `vfmaq_f32` on aarch64; scalar axpy fallback elsewhere.
pub(crate) fn neon_gemm_row_slice(
    a_row: &[f32],
    b: &[f32],
    c_row: &mut [f32],
    k: usize,
    n: usize,
) {
    #[cfg(target_arch = "aarch64")]
    // SAFETY: NEON is mandatory on all aarch64 targets per the AArch64 ABI.
    return unsafe { neon_impl::gemm_row(a_row, b, c_row, k, n) };

    #[cfg(not(target_arch = "aarch64"))]
    gemm_scalar_row(a_row, b, c_row, k, n);
}

// ── public Tensor API ─────────────────────────────────────────────────────────

/// NEON-accelerated f32 GEMM: `C = A @ B`.
///
/// On aarch64 uses explicit `vfmaq_f32` intrinsics (4 f32 per cycle) to
/// achieve throughput the blocked while-loop kernel cannot, due to LLVM's
/// inability to auto-vectorize variable-width tile boundaries.
///
/// On non-aarch64 targets falls back to a scalar axpy loop — use
/// [`super::matmul_avx2`] for the fast path on x86_64.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if either input is not 2-D.
/// * [`TensorError::ShapeMismatch`] if inner dimensions don't match.
#[must_use = "returns a new tensor; result is not stored in-place"]
pub fn matmul_neon_f32(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_neon_f32: `a` must be 2-D, got {}D (shape {:?})",
                a.ndim(), a.dims()
            ),
        });
    }
    if b.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_neon_f32: `b` must be 2-D, got {}D (shape {:?})",
                b.ndim(), b.dims()
            ),
        });
    }
    let [m, k]  = [a.dims()[0], a.dims()[1]];
    let [k2, n] = [b.dims()[0], b.dims()[1]];
    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, k],
            got:      vec![k2, n],
        });
    }

    let a_c: Cow<Tensor<f32>> = if a.is_contiguous() { Cow::Borrowed(a) } else { Cow::Owned(a.contiguous()) };
    let b_c: Cow<Tensor<f32>> = if b.is_contiguous() { Cow::Borrowed(b) } else { Cow::Owned(b.contiguous()) };

    let mut c_data = vec![0.0_f32; m * n];
    neon_gemm_slice(a_c.as_slice(), b_c.as_slice(), &mut c_data, m, k, n);
    Tensor::from_vec(c_data, vec![m, n])
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul::matmul_naive;

    const REL_TOL: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool { (a - b).abs() < 1e-5 }

    fn assert_matches_naive(a: &Tensor<f32>, b: &Tensor<f32>) {
        let expected = matmul_naive(a, b).unwrap();
        let got      = matmul_neon_f32(a, b).unwrap();
        assert_eq!(got.dims(), expected.dims(), "shape mismatch");
        for (idx, (g, e)) in got.as_slice().iter().zip(expected.as_slice()).enumerate() {
            let denom = g.abs().max(e.abs()).max(1.0);
            let rel   = (g - e).abs() / denom;
            assert!(rel < REL_TOL, "elem {idx}: neon={g} naive={e} rel={rel:.2e}");
        }
    }

    #[test]
    fn test_2x2_identity() {
        let a  = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c  = matmul_neon_f32(&a, &id).unwrap();
        assert!(c.as_slice().iter().zip(&[1.0_f32, 2.0, 3.0, 4.0]).all(|(g, e)| close(*g, *e)));
    }

    #[test]
    fn test_2x2_known_result() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = matmul_neon_f32(&a, &b).unwrap();
        assert!(c.as_slice().iter().zip(&[19.0_f32, 22.0, 43.0, 50.0]).all(|(g, e)| close(*g, *e)));
    }

    #[test]
    fn test_1x1() {
        let a = Tensor::from_vec(vec![3.0_f32], vec![1, 1]).unwrap();
        let b = Tensor::from_vec(vec![7.0_f32], vec![1, 1]).unwrap();
        assert!(close(matmul_neon_f32(&a, &b).unwrap().as_slice()[0], 21.0));
    }

    #[test]
    fn test_matches_naive_n_not_multiple_of_4() {
        // N=13: 3 full 4-wide chunks + 1-element scalar tail.
        let (m, k, n) = (4, 16, 13);
        let a = Tensor::from_vec((0..m*k).map(|i| i as f32 * 0.1).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..k*n).map(|i| i as f32 * 0.05).collect(), vec![k, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_square_32() {
        let n = 32_usize;
        let a = Tensor::from_vec((0..n*n).map(|i| i as f32 * 0.01).collect(), vec![n, n]).unwrap();
        let b = Tensor::from_vec((0..n*n).map(|i| (n*n - i) as f32 * 0.005).collect(), vec![n, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_square_128() {
        let n = 128_usize;
        let a = Tensor::from_vec((0..n*n).map(|i| i as f32 * 0.001).collect(), vec![n, n]).unwrap();
        let b = Tensor::from_vec((0..n*n).map(|i| (n*n - i) as f32 * 0.001).collect(), vec![n, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_rectangular() {
        let (m, k, n) = (50, 70, 40);
        let a = Tensor::from_vec((0..m*k).map(|i| i as f32 * 0.003).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..k*n).map(|i| (k*n-i) as f32 * 0.002).collect(), vec![k, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_projection_shape() {
        // Scaled-down projection: seq=4, hidden=64, ffn=128
        let (m, k, n) = (4, 64, 128);
        let a = Tensor::from_vec((0..m*k).map(|i| (i as f32)*0.01 - 0.3).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..k*n).map(|i| (i as f32)*0.005 - 0.15).collect(), vec![k, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_non_contiguous_transpose_a() {
        let a   = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let a_t = a.transpose(0, 1).unwrap();
        let id  = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c   = matmul_neon_f32(&a_t, &id).unwrap();
        assert!(c.as_slice().iter().zip(&[1.0_f32, 2.0, 3.0, 4.0]).all(|(g, e)| close(*g, *e)));
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32; 3], vec![3, 1]).unwrap();
        assert!(matches!(matmul_neon_f32(&a, &b), Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_non_2d_error() {
        let a = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        assert!(matches!(matmul_neon_f32(&a, &b), Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn test_matches_avx2_on_shared_shape() {
        // On aarch64 avx2 falls back to blocked; this verifies neon and blocked agree.
        use crate::ops::matmul::matmul_avx2;
        let (m, k, n) = (8, 32, 16);
        let a = Tensor::from_vec((0..m*k).map(|i| i as f32 * 0.01).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..k*n).map(|i| (k*n-i) as f32 * 0.01).collect(), vec![k, n]).unwrap();
        let neon_out = matmul_neon_f32(&a, &b).unwrap();
        let avx2_out = matmul_avx2(&a, &b).unwrap();
        for (i, (g, e)) in neon_out.as_slice().iter().zip(avx2_out.as_slice()).enumerate() {
            let denom = g.abs().max(e.abs()).max(1.0);
            let rel   = (g - e).abs() / denom;
            assert!(rel < REL_TOL, "elem {i}: neon={g} avx2/blocked={e} rel={rel:.2e}");
        }
    }
}
