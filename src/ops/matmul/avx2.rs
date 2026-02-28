//! AVX2 f32 matrix multiplication kernel — commit 15.2.
//!
//! # Algorithm
//!
//! Row-accumulation ("outer product row") decomposition:
//!
//! ```text
//! for i in 0..M:
//!     c_row = [0; N]
//!     for k in 0..K:
//!         c_row[j..] += A[i,k] * B[k,j..]   ← SIMD axpy (8 f32/cycle on AVX2)
//!     C[i,:] = c_row
//! ```
//!
//! B rows are contiguous in row-major layout, so each axpy is a single
//! `_mm256_fmadd_ps` loop over N elements with no scatter/gather overhead.
//!
//! # Throughput
//!
//! AVX2 + FMA processes 8 f32s per instruction.  For K=4096 (Llama hidden
//! dim) and N=4096, the inner axpy loop is ~512 FMA instructions per output
//! row — compared to ~4096 scalar multiplications for the naive kernel.
//!
//! # Runtime dispatch
//!
//! [`matmul_avx2`] detects AVX2+FMA at call time and falls back to
//! [`super::matmul_blocked`] on older CPUs (including Apple Silicon, which
//! uses NEON and will take the scalar path here).

use std::borrow::Cow;

use crate::tensor::{Result, Tensor, TensorError};

// ── validation helper (shared by both paths) ───────────────────────────────

fn validate_2d_gemm(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    label: &str,
) -> Result<(usize, usize, usize)> {
    if a.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "{label}: `a` must be 2-D, got {}D (shape {:?})",
                a.ndim(),
                a.dims()
            ),
        });
    }
    if b.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "{label}: `b` must be 2-D, got {}D (shape {:?})",
                b.ndim(),
                b.dims()
            ),
        });
    }
    let (m, k) = (a.dims()[0], a.dims()[1]);
    let (k2, n) = (b.dims()[0], b.dims()[1]);
    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, k],
            got: vec![k2, n],
        });
    }
    Ok((m, k, n))
}

// ── AVX2 kernel (x86_64 only) ─────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
mod avx2_impl {
    use std::arch::x86_64::*;

    /// Inner AVX2+FMA GEMM — called only after feature detection confirms AVX2 + FMA.
    ///
    /// # Safety
    /// Caller must guarantee AVX2 and FMA are available.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn gemm_kernel(
        a: &[f32], b: &[f32], c: &mut [f32],
        m: usize, k: usize, n: usize,
    ) {
        // Allocate a temporary accumulator row on the stack for small N,
        // or re-use c directly.  We write directly into `c` row-by-row.
        for i in 0..m {
            let a_row = &a[i * k..(i + 1) * k];
            let c_row = &mut c[i * n..(i + 1) * n];

            // Zero the output row (c_row may contain garbage from allocation).
            // We zero it in AVX2 chunks.
            let chunks_n = n / 8;
            let cp = c_row.as_mut_ptr();
            let zero = _mm256_setzero_ps();
            for j8 in 0..chunks_n {
                _mm256_storeu_ps(cp.add(j8 * 8), zero);
            }
            for j in (chunks_n * 8)..n {
                c_row[j] = 0.0;
            }

            // For each k: accumulate A[i,kk] * B[kk,:] into c_row.
            for kk in 0..k {
                let a_ik = *a_row.get_unchecked(kk);
                let va = _mm256_set1_ps(a_ik);
                let b_row = b.get_unchecked(kk * n..(kk + 1) * n);
                let bp = b_row.as_ptr();

                for j8 in 0..chunks_n {
                    let vb = _mm256_loadu_ps(bp.add(j8 * 8));
                    let vc = _mm256_loadu_ps(cp.add(j8 * 8));
                    // c[j8*8..] += a_ik * b[kk, j8*8..]
                    _mm256_storeu_ps(cp.add(j8 * 8), _mm256_fmadd_ps(va, vb, vc));
                }
                // scalar tail
                for j in (chunks_n * 8)..n {
                    *cp.add(j) += a_ik * *bp.add(j);
                }
            }
        }
    }
}

// ── public entry point ─────────────────────────────────────────────────────

/// AVX2-accelerated 2-D GEMM: `C = A @ B`.
///
/// Falls back to [`super::matmul_blocked`] when AVX2 + FMA are not available
/// (e.g. Apple Silicon, older x86 CPUs).
///
/// # Arguments
///
/// * `a` – shape `[M, K]`
/// * `b` – shape `[K, N]`
///
/// # Returns
///
/// A new contiguous `Tensor<f32>` of shape `[M, N]`.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if either tensor is not 2-D.
/// * [`TensorError::ShapeMismatch`] if inner dimensions don't match.
#[must_use = "returns a new tensor; result is not stored in-place"]
pub fn matmul_avx2(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    let (m, k, n) = validate_2d_gemm(a, b, "matmul_avx2")?;

    // Contiguity gate
    let a_c: Cow<Tensor<f32>> = if a.is_contiguous() {
        Cow::Borrowed(a)
    } else {
        Cow::Owned(a.contiguous())
    };
    let b_c: Cow<Tensor<f32>> = if b.is_contiguous() {
        Cow::Borrowed(b)
    } else {
        Cow::Owned(b.contiguous())
    };

    // Suppress "unused" warnings on non-x86 targets where (m,k,n,a_c,b_c)
    // are consumed only inside the #[cfg(target_arch = "x86_64")] block below.
    let _ = (m, k, n, &a_c, &b_c);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let mut c_data = vec![0.0_f32; m * n];
            // SAFETY: we just confirmed AVX2 + FMA are available.
            unsafe {
                avx2_impl::gemm_kernel(
                    a_c.as_slice(),
                    b_c.as_slice(),
                    &mut c_data,
                    m, k, n,
                );
            }
            return Tensor::from_vec(c_data, vec![m, n]);
        }
    }

    // Fallback — non-x86 or no AVX2
    super::matmul_blocked(a, b)
}

// ── tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul::matmul_naive;

    const REL_TOL: f32 = 1e-4;

    fn assert_matches_naive(a: &Tensor<f32>, b: &Tensor<f32>) {
        let expected = matmul_naive(a, b).unwrap();
        let got = matmul_avx2(a, b).unwrap();
        assert_eq!(got.dims(), expected.dims(), "shape mismatch");
        for (idx, (g, e)) in got.as_slice().iter().zip(expected.as_slice()).enumerate() {
            let denom = g.abs().max(e.abs()).max(1.0);
            let rel = (g - e).abs() / denom;
            assert!(rel < REL_TOL, "element {idx}: avx2={g} naive={e} rel={rel:.2e}");
        }
    }

    #[test]
    fn test_2x2_identity() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = matmul_avx2(&a, &id).unwrap();
        for (g, e) in c.as_slice().iter().zip(&[1.0_f32, 2.0, 3.0, 4.0]) {
            assert!((g - e).abs() < 1e-6, "got {g}, expected {e}");
        }
    }

    #[test]
    fn test_2x2_known_result() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = matmul_avx2(&a, &b).unwrap();
        let expected = [19.0_f32, 22.0, 43.0, 50.0];
        for (g, e) in c.as_slice().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "got {g}, expected {e}");
        }
    }

    #[test]
    fn test_matches_naive_square_32() {
        let n = 32_usize;
        let a = Tensor::from_vec((0..(n * n)).map(|i| i as f32 * 0.01).collect(), vec![n, n]).unwrap();
        let b = Tensor::from_vec((0..(n * n)).map(|i| (n * n - i) as f32 * 0.005).collect(), vec![n, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_square_64() {
        // 64×64 — two full 8-wide AVX2 rows, multiple accumulation loops
        let n = 64_usize;
        let a = Tensor::from_vec((0..(n * n)).map(|i| i as f32 * 0.001).collect(), vec![n, n]).unwrap();
        let b = Tensor::from_vec((0..(n * n)).map(|i| (i % 17) as f32 * 0.1).collect(), vec![n, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_non_multiple_of_8() {
        // N=13: 1 full 8-wide chunk + 5-element scalar tail
        let (m, k, n) = (4, 16, 13);
        let a = Tensor::from_vec((0..(m * k)).map(|i| i as f32 * 0.1).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..(k * n)).map(|i| i as f32 * 0.05).collect(), vec![k, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_rectangular_tall() {
        // Tall A: typical projection matrix shape in Llama (seq=1, emb=dim)
        let (m, k, n) = (128, 64, 32);
        let a = Tensor::from_vec((0..(m * k)).map(|i| i as f32 * 0.001).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..(k * n)).map(|i| i as f32 * 0.002).collect(), vec![k, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_matches_naive_larger() {
        // 200×150 @ 150×100 — exercises multi-chunk accumulation
        let (m, k, n) = (200, 150, 100);
        let a = Tensor::from_vec((0..(m * k)).map(|i| (i % 31) as f32 * 0.01).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..(k * n)).map(|i| (i % 37) as f32 * 0.01).collect(), vec![k, n]).unwrap();
        assert_matches_naive(&a, &b);
    }

    #[test]
    fn test_non_contiguous_input() {
        // Transposed A is non-contiguous; should still produce correct result.
        let a_raw = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let a = a_raw.transpose(0, 1).unwrap(); // [[1,2],[3,4]]
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let c = matmul_avx2(&a, &id).unwrap();
        // After transpose: A = [[1,2],[3,4]], A@I = A
        let expected = [1.0_f32, 2.0, 3.0, 4.0];
        for (g, e) in c.as_slice().iter().zip(&expected) {
            assert!((g - e).abs() < 1e-5, "got {g}, expected {e}");
        }
    }

    #[test]
    fn test_shape_mismatch_error() {
        let a = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32; 8], vec![4, 2]).unwrap();
        assert!(matches!(matmul_avx2(&a, &b), Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_non_2d_error() {
        let a = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        assert!(matches!(matmul_avx2(&a, &b), Err(TensorError::InvalidShape { .. })));
    }
}
