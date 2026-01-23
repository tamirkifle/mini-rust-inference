//! Naive (triple-nested loop) matrix multiplication.
//!
//! This is the correctness **reference implementation**. It is intentionally
//! unoptimized — every subsequent matmul commit (blocked, SIMD, INT8, …) must
//! produce results within floating-point tolerance of this baseline.
//!
//! Loop order is `i-p-j` (a.k.a. `i-k-j`), which keeps the `b` row hot in
//! cache and avoids the scattered writes of the naïve `i-j-k` order, while
//! still being trivially readable.

use std::borrow::Cow;

use crate::tensor::{Result, Tensor, TensorError};

/// Naive GEMM: computes `C = A × B`.
///
/// # Arguments
///
/// * `a` – 2-D tensor of shape `[M, K]`
/// * `b` – 2-D tensor of shape `[K, N]`
///
/// # Returns
///
/// A new contiguous `Tensor<f32>` of shape `[M, N]`.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] – if either tensor is not exactly 2-D.
/// * [`TensorError::ShapeMismatch`] – if `a.dims()[1] != b.dims()[0]`.
#[must_use = "returns a new tensor; result is not stored in-place"] // CHANGED
pub fn matmul_naive(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    // ── dimensionality checks ──────────────────────────────────────────────
    if a.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_naive: `a` must be 2-D, got {}D (shape {:?})",
                a.ndim(),
                a.dims()
            ),
        });
    }
    if b.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_naive: `b` must be 2-D, got {}D (shape {:?})",
                b.ndim(),
                b.dims()
            ),
        });
    }

    let [m, k] = [a.dims()[0], a.dims()[1]]; // CHANGED: destructure for clarity
    let [k2, n] = [b.dims()[0], b.dims()[1]];

    if k != k2 {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, k],
            got: vec![k2, n],
        });
    }

    // ── contiguity gate ────────────────────────────────────────────────────
    // CHANGED: force contiguous layout so stride-1 indexing is safe.
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

    let a_data = a_c.as_slice();
    let b_data = b_c.as_slice();

    // ── i-p-j triple loop ──────────────────────────────────────────────────
    // CHANGED: accumulate into c_data[i*n + j] using the ikj traversal order.
    let mut c_data = vec![0.0_f32; m * n];

    for i in 0..m {
        for p in 0..k {
            let a_ip = a_data[i * k + p]; // a[i, p]  – scalar hoist
            for j in 0..n {
                // CHANGED: c[i,j] += a[i,p] * b[p,j]
                c_data[i * n + j] += a_ip * b_data[p * n + j];
            }
        }
    }

    Tensor::from_vec(c_data, vec![m, n])
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn close_slice(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
    }

    // ── ndarray reference ──────────────────────────────────────────────────
    /// Convenience: compare our result against ndarray directly.
    fn assert_matches_ndarray(
        a_data: &[f32],
        a_shape: (usize, usize),
        b_data: &[f32],
        b_shape: (usize, usize),
    ) {
        use ndarray::Array2;
        let our_a = Tensor::from_vec(a_data.to_vec(), vec![a_shape.0, a_shape.1]).unwrap();
        let our_b = Tensor::from_vec(b_data.to_vec(), vec![b_shape.0, b_shape.1]).unwrap();
        let our_c = matmul_naive(&our_a, &our_b).unwrap();

        let nd_a = Array2::from_shape_vec(a_shape, a_data.to_vec()).unwrap();
        let nd_b = Array2::from_shape_vec(b_shape, b_data.to_vec()).unwrap();
        let nd_c: Vec<f32> = nd_a.dot(&nd_b).into_raw_vec();

        assert!(
            close_slice(our_c.as_slice(), &nd_c),
            "mismatch vs ndarray:\n  ours={:?}\n  ndarray={:?}",
            our_c.as_slice(),
            nd_c,
        );
    }

    // ── basic correctness ──────────────────────────────────────────────────

    #[test]
    fn test_2x2_identity() {
        // A @ I = A
        let a_data = [1.0_f32, 2.0, 3.0, 4.0];
        let i_data = [1.0_f32, 0.0, 0.0, 1.0];

        let a = Tensor::from_vec(a_data.to_vec(), vec![2, 2]).unwrap();
        let id = Tensor::from_vec(i_data.to_vec(), vec![2, 2]).unwrap();

        let c = matmul_naive(&a, &id).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        assert!(close_slice(c.as_slice(), &a_data));
    }

    #[test]
    fn test_2x2_known_result() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = matmul_naive(&a, &b).unwrap();

        assert_eq!(c.dims(), &[2, 2]);
        assert!(close_slice(c.as_slice(), &[19.0, 22.0, 43.0, 50.0]));
    }

    #[test]
    fn test_rectangular_matmul() {
        // [2x3] @ [3x4] -> [2x4]
        // CHANGED: rectangular case exercises the M != K != N path
        #[rustfmt::skip]
        let a_data = [
            1.0_f32, 2.0, 3.0,
            4.0,     5.0, 6.0,
        ];
        #[rustfmt::skip]
        let b_data = [
             7.0_f32,  8.0,  9.0, 10.0,
            11.0,     12.0, 13.0, 14.0,
            15.0,     16.0, 17.0, 18.0,
        ];
        let a = Tensor::from_vec(a_data.to_vec(), vec![2, 3]).unwrap();
        let b = Tensor::from_vec(b_data.to_vec(), vec![3, 4]).unwrap();
        let c = matmul_naive(&a, &b).unwrap();

        assert_eq!(c.dims(), &[2, 4]);
        // Row 0: [1*7+2*11+3*15, 1*8+2*12+3*16, 1*9+2*13+3*17, 1*10+2*14+3*18]
        //      = [7+22+45, 8+24+48, 9+26+51, 10+28+54]
        //      = [74, 80, 86, 92]
        // Row 1: [4*7+5*11+6*15, 4*8+5*12+6*16, 4*9+5*13+6*17, 4*10+5*14+6*18]
        //      = [28+55+90, 32+60+96, 36+65+102, 40+70+108]
        //      = [173, 188, 203, 218]
        let expected = [74.0_f32, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0];
        assert!(close_slice(c.as_slice(), &expected));
    }

    #[test]
    fn test_matmul_1x1() {
        let a = Tensor::from_vec(vec![3.0_f32], vec![1, 1]).unwrap();
        let b = Tensor::from_vec(vec![7.0_f32], vec![1, 1]).unwrap();
        let c = matmul_naive(&a, &b).unwrap();
        assert!(close(c.as_slice()[0], 21.0));
    }

    #[test]
    fn test_matmul_vector_outer_product() {
        // column [3x1] @ row [1x3] = [3x3] outer product
        let col = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3, 1]).unwrap();
        let row = Tensor::from_vec(vec![4.0_f32, 5.0, 6.0], vec![1, 3]).unwrap();
        let c = matmul_naive(&col, &row).unwrap();

        assert_eq!(c.dims(), &[3, 3]);
        let expected = [
            4.0_f32, 5.0, 6.0,  // 1 * [4,5,6]
            8.0, 10.0, 12.0,     // 2 * [4,5,6]
            12.0, 15.0, 18.0,    // 3 * [4,5,6]
        ];
        assert!(close_slice(c.as_slice(), &expected));
    }

    // ── non-contiguous inputs ──────────────────────────────────────────────

    #[test]
    fn test_non_contiguous_transpose_a() {
        // CHANGED: verify contiguity gate — transposed A is non-contiguous
        // [[1,3],[2,4]]^T = [[1,2],[3,4]]
        // [[1,2],[3,4]] @ [[1,0],[0,1]] = [[1,2],[3,4]]
        let a = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let a_t = a.transpose(0, 1).unwrap(); // non-contiguous [[1,2],[3,4]]
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();

        let c = matmul_naive(&a_t, &id).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        assert!(close_slice(c.as_slice(), &[1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn test_non_contiguous_transpose_b() {
        // A @ B^T where B^T is non-contiguous
        // A = [[1,2],[3,4]] (2x2), B^T of [[1,3],[2,4]] = [[1,2],[3,4]]
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let b_t = b.transpose(0, 1).unwrap(); // non-contiguous

        // contiguous b_t = [[1,2],[3,4]], so A @ b_t = [[1,2],[3,4]]@[[1,2],[3,4]]
        // = [[7,10],[15,22]]
        let c = matmul_naive(&a, &b_t).unwrap();
        assert_eq!(c.dims(), &[2, 2]);
        assert!(close_slice(c.as_slice(), &[7.0, 10.0, 15.0, 22.0]));
    }

    // ── ndarray comparison ─────────────────────────────────────────────────

    #[test]
    fn test_matches_ndarray_3x3() {
        #[rustfmt::skip]
        assert_matches_ndarray(
            &[1.0, 2.0, 3.0,
              4.0, 5.0, 6.0,
              7.0, 8.0, 9.0_f32],
            (3, 3),
            &[9.0, 8.0, 7.0,
              6.0, 5.0, 4.0,
              3.0, 2.0, 1.0_f32],
            (3, 3),
        );
    }

    #[test]
    fn test_matches_ndarray_non_square() {
        assert_matches_ndarray(
            &[1.0, 0.5, -1.0, 2.0, 0.25, 3.0_f32],
            (2, 3),
            &[1.0, 2.0, -1.0, 0.0, 0.5, 1.0_f32],
            (3, 2),
        );
    }

    #[test]
    fn test_matches_ndarray_larger() {
        // CHANGED: 16x16 to catch any loop-bound off-by-one.
        // Uses a looser relative tolerance because f32 accumulation order differs
        // between our ikj loop and ndarray's internal kernel; absolute differences
        // remain < 0.001 (< 0.001% relative for values ~2600).
        use ndarray::Array2;
        let n = 16_usize;
        let a_data: Vec<f32> = (0..(n * n)).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..(n * n)).map(|i| (n * n - i) as f32 * 0.05).collect();

        let our_a = Tensor::from_vec(a_data.clone(), vec![n, n]).unwrap();
        let our_b = Tensor::from_vec(b_data.clone(), vec![n, n]).unwrap();
        let our_c = matmul_naive(&our_a, &our_b).unwrap();

        let nd_a = Array2::from_shape_vec((n, n), a_data).unwrap();
        let nd_b = Array2::from_shape_vec((n, n), b_data).unwrap();
        let nd_c: Vec<f32> = nd_a.dot(&nd_b).into_raw_vec();

        // Relative tolerance: accept if |a-b| / max(|a|,|b|,1) < 1e-4
        let rel_tol = 1e-4_f32;
        for (ours, theirs) in our_c.as_slice().iter().zip(nd_c.iter()) {
            let denom = ours.abs().max(theirs.abs()).max(1.0);
            let rel_err = (ours - theirs).abs() / denom;
            assert!(
                rel_err < rel_tol,
                "relative error {rel_err:.2e} > {rel_tol:.2e}: ours={ours}, ndarray={theirs}"
            );
        }
    }

    // ── error handling ─────────────────────────────────────────────────────

    #[test]
    fn test_shape_mismatch_error() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3, 1]).unwrap();
        // k=2 for a, k2=3 for b — mismatch
        assert!(matches!(
            matmul_naive(&a, &b),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_non_2d_a_error() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3, 1]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], vec![3, 1]).unwrap();
        assert!(matches!(
            matmul_naive(&a, &b),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_non_2d_b_error() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2, 1]).unwrap();
        assert!(matches!(
            matmul_naive(&a, &b),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
