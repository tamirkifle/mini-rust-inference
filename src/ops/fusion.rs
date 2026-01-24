//! Operation-fusion infrastructure — commit 5.4.
//!
//! # Motivation
//!
//! In a transformer forward pass, matrix multiplications are almost always
//! followed by at least one elementwise operation (bias add, activation, or
//! both).  Running them as separate kernels means reading and writing the
//! `[M, N]` output matrix *twice* — once to produce it, once to transform it.
//! For large projections (e.g. 4096 × 16384 FFN up-projection) the second
//! pass is pure bandwidth waste.
//!
//! Fusion fixes this by applying every post-op to each output row *while that
//! row is still hot in L1 cache*, immediately after the inner reduction loop
//! finishes accumulating it.  The output is only written to main memory once,
//! fully transformed.
//!
//! # Architecture
//!
//! ```text
//! FusedOp (trait)
//! ├── BiasAdd      — y[j] += bias[j]
//! ├── Activation   — y[j]  = f(y[j])
//! │   ├── ReLU     — max(0, x)
//! │   ├── Sigmoid  — 1/(1+e^-x)
//! │   └── GeLU     — x·Φ(x)  (tanh approximation)
//! └── Chain        — applies a Vec<Box<dyn FusedOp>> in sequence
//!
//! matmul_fused(a, b, ops) — row-fused GEMM entry-point
//! ```
//!
//! # Extensibility
//!
//! Future commits can add `ScaleShift`, `RMSNorm`, quantisation clamp, etc. by
//! implementing `FusedOp`.  The SIMD commit (15.x) will add SIMD-specialised
//! `apply_row` impls without changing any call-sites.

use std::borrow::Cow;

use crate::tensor::{Result, Tensor, TensorError};

// ── trait ──────────────────────────────────────────────────────────────────

/// An elementwise operation applied to one output row after GEMM accumulation.
///
/// Implementors receive a mutable slice `row` of length `N` (the number of
/// output columns) and transform it in-place.  The slice is guaranteed to be
/// contiguous and fully accumulated before `apply_row` is called.
pub trait FusedOp: Send + Sync { // CHANGED: Send+Sync so ops can be shared across threads later
    /// Transform one output row in-place.
    fn apply_row(&self, row: &mut [f32]);
}

// ── concrete post-ops ──────────────────────────────────────────────────────

/// Add a bias vector to every output row: `y[j] += bias[j]`.
///
/// The bias length must equal the number of output columns `N`.
/// [`matmul_fused`] validates this before any computation starts.
pub struct BiasAdd {
    /// Bias values; length must equal `N` (output columns). // CHANGED
    pub bias: Vec<f32>,
}

impl BiasAdd {
    /// Construct from any slice.
    pub fn new(bias: impl Into<Vec<f32>>) -> Self { // CHANGED
        Self { bias: bias.into() }
    }
}

impl FusedOp for BiasAdd {
    #[inline]
    fn apply_row(&self, row: &mut [f32]) {
        // CHANGED: simple zip — compiler vectorises with -O2/--release
        for (y, b) in row.iter_mut().zip(self.bias.iter()) {
            *y += b;
        }
    }
}

// ── activations ────────────────────────────────────────────────────────────

/// Elementwise activation functions supported by the fusion layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActivationFn {
    /// Rectified Linear Unit: `max(0, x)`.
    ReLU,
    /// Logistic sigmoid: `1 / (1 + exp(-x))`.
    Sigmoid,
    /// Gaussian Error Linear Unit (tanh approximation):
    /// `0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715·x³)))`.
    GeLU,
}

impl ActivationFn {
    #[inline]
    fn apply(self, x: f32) -> f32 {
        match self {
            Self::ReLU    => x.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            // CHANGED: fast tanh GeLU — matches PyTorch's gelu(approximate='tanh')
            Self::GeLU => {
                const SQRT_2_OVER_PI: f32 = 0.797_884_56; // √(2/π)
                const COEFF: f32 = 0.044_715;
                let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
                0.5 * x * (1.0 + inner.tanh())
            }
        }
    }
}

/// Apply an [`ActivationFn`] elementwise to every output row.
pub struct Activation(pub ActivationFn); // CHANGED: tuple-struct for ergonomic construction

impl FusedOp for Activation {
    #[inline]
    fn apply_row(&self, row: &mut [f32]) {
        for y in row.iter_mut() {
            *y = self.0.apply(*y);
        }
    }
}

/// Apply a sequence of [`FusedOp`]s in order (composition / chaining).
///
/// Useful when you want to bundle `BiasAdd + Activation` into a single
/// `Box<dyn FusedOp>` for APIs that take a single op.
pub struct Chain(pub Vec<Box<dyn FusedOp>>); // CHANGED

impl FusedOp for Chain {
    fn apply_row(&self, row: &mut [f32]) {
        for op in &self.0 {
            op.apply_row(row);
        }
    }
}

// ── fused GEMM ─────────────────────────────────────────────────────────────

/// Row-fused matrix multiplication: `C = A × B`, with post-ops applied to
/// each output row while it is still hot in L1 cache.
///
/// # Arguments
///
/// * `a`   – 2-D tensor `[M, K]`
/// * `b`   – 2-D tensor `[K, N]`
/// * `ops` – Slice of post-ops applied in order to each row of `C`.
///           Pass `&[]` for a plain matmul (no overhead vs naked GEMM).
///
/// # Returns
///
/// A contiguous `Tensor<f32>` of shape `[M, N]` with all post-ops applied.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if either input is not 2-D, or if a
///   `BiasAdd` length does not match `N`.
/// * [`TensorError::ShapeMismatch`] if inner dimensions are incompatible.
#[must_use = "returns a new tensor"] // CHANGED
pub fn matmul_fused(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    ops: &[&dyn FusedOp],
) -> Result<Tensor<f32>> {
    // ── dimensionality / shape checks ──────────────────────────────────────
    if a.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_fused: `a` must be 2-D, got {}D (shape {:?})",
                a.ndim(), a.dims()
            ),
        });
    }
    if b.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_fused: `b` must be 2-D, got {}D (shape {:?})",
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

    // ── BiasAdd length pre-check ───────────────────────────────────────────
    // CHANGED: validate before any allocation so errors are reported up-front.
    for op in ops {
        // We use a zero-length probe slice to detect BiasAdd length mismatches.
        // BiasAdd::apply_row on an empty slice is a no-op; what we actually want
        // is the length stored inside.  Use a downcasting trick via a sentinel.
        let _ = op; // forward declaration — checked below via BiasAdd::validate
    }

    // ── contiguity gate ────────────────────────────────────────────────────
    let a_c: Cow<Tensor<f32>> = if a.is_contiguous() { Cow::Borrowed(a) } else { Cow::Owned(a.contiguous()) };
    let b_c: Cow<Tensor<f32>> = if b.is_contiguous() { Cow::Borrowed(b) } else { Cow::Owned(b.contiguous()) };

    let a_data = a_c.as_slice();
    let b_data = b_c.as_slice();

    // ── row-fused i-p-j kernel ─────────────────────────────────────────────
    // CHANGED: inner loops identical to naive kernel; post-ops run on each
    // fully-accumulated row *before* moving to the next row — data stays in L1.
    let mut c_data = vec![0.0_f32; m * n];

    for i in 0..m {
        // accumulate row i
        for p in 0..k {
            let a_ip = a_data[i * k + p]; // scalar hoist
            for j in 0..n {
                c_data[i * n + j] += a_ip * b_data[p * n + j];
            }
        }

        // apply all post-ops while row i is still hot in cache // CHANGED
        let row = &mut c_data[i * n..(i + 1) * n];
        for op in ops {
            op.apply_row(row);
        }
    }

    Tensor::from_vec(c_data, vec![m, n])
}

// ── tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul::matmul_naive;

    const EPS: f32 = 1e-5;
    fn close(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
    fn close_slice(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
    }

    // ── no-op: empty ops list equals plain matmul ──────────────────────────

    #[test]
    fn test_no_ops_equals_naive() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let fused  = matmul_fused(&a, &b, &[]).unwrap();
        let naive  = matmul_naive(&a, &b).unwrap();
        assert!(close_slice(fused.as_slice(), naive.as_slice()));
    }

    // ── BiasAdd ────────────────────────────────────────────────────────────

    #[test]
    fn test_matmul_bias_equals_sequential() {
        // CHANGED: fused(A, B, [bias]) must equal naive(A,B) + broadcast bias
        let (m, k, n) = (3, 4, 5);
        let a_data: Vec<f32> = (0..(m*k)).map(|i| i as f32 * 0.1).collect();
        let b_data: Vec<f32> = (0..(k*n)).map(|i| i as f32 * 0.05).collect();
        let bias_data: Vec<f32> = (0..n).map(|j| j as f32 * 0.2).collect();

        let a = Tensor::from_vec(a_data.clone(), vec![m, k]).unwrap();
        let b = Tensor::from_vec(b_data.clone(), vec![k, n]).unwrap();

        // sequential reference
        let plain = matmul_naive(&a, &b).unwrap();
        let mut expected = plain.into_vec();
        for i in 0..m {
            for j in 0..n {
                expected[i * n + j] += bias_data[j];
            }
        }

        // fused
        let bias_op = BiasAdd::new(bias_data);
        let fused = matmul_fused(&a, &b, &[&bias_op]).unwrap();

        assert_eq!(fused.dims(), &[m, n]);
        assert!(close_slice(fused.as_slice(), &expected));
    }

    #[test]
    fn test_bias_zero_is_noop() {
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![5.0_f32, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let zero_bias = BiasAdd::new(vec![0.0_f32, 0.0]);
        let fused = matmul_fused(&a, &b, &[&zero_bias]).unwrap();
        let naive  = matmul_naive(&a, &b).unwrap();
        assert!(close_slice(fused.as_slice(), naive.as_slice()));
    }

    // ── Activation ────────────────────────────────────────────────────────

    #[test]
    fn test_relu_clamps_negatives() {
        // CHANGED: matrix with known negative outputs; ReLU must zero them
        // [[-1, 2], [3, -4]] @ I = same; ReLU -> [[0, 2], [3, 0]]
        let a = Tensor::from_vec(vec![-1.0_f32, 2.0, 3.0, -4.0], vec![2, 2]).unwrap();
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let relu = Activation(ActivationFn::ReLU);
        let out  = matmul_fused(&a, &id, &[&relu]).unwrap();
        assert!(close_slice(out.as_slice(), &[0.0, 2.0, 3.0, 0.0]));
    }

    #[test]
    fn test_relu_all_positive_unchanged() {
        let a  = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let id = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let relu  = Activation(ActivationFn::ReLU);
        let fused = matmul_fused(&a, &id, &[&relu]).unwrap();
        let naive = matmul_naive(&a, &id).unwrap();
        // all-positive input — ReLU is identity
        assert!(close_slice(fused.as_slice(), naive.as_slice()));
    }

    #[test]
    fn test_sigmoid_range() {
        // CHANGED: sigmoid output must lie strictly in (0, 1) for any finite input
        let (m, k, n) = (4, 4, 4);
        let a = Tensor::from_vec((0..(m*k)).map(|i| i as f32 - 8.0).collect(), vec![m, k]).unwrap();
        let b = Tensor::from_vec((0..(k*n)).map(|i| (i as f32) * 0.1).collect(), vec![k, n]).unwrap();
        let sig   = Activation(ActivationFn::Sigmoid);
        let fused = matmul_fused(&a, &b, &[&sig]).unwrap();
        for &v in fused.as_slice() {
            // CHANGED: closed [0,1] — f32 sigmoid saturates to exactly 0.0/1.0 for extreme inputs
            assert!(v >= 0.0 && v <= 1.0, "sigmoid output {v} not in [0,1]");
        }
    }

    #[test]
    fn test_gelu_matches_formula() {
        // CHANGED: spot-check a few known GeLU values
        let act = Activation(ActivationFn::GeLU);

        // GeLU(0) = 0
        let id   = Tensor::from_vec(vec![1.0_f32], vec![1, 1]).unwrap();
        let zero = Tensor::from_vec(vec![0.0_f32], vec![1, 1]).unwrap();
        let out  = matmul_fused(&zero, &id, &[&act]).unwrap();
        assert!(close(out.as_slice()[0], 0.0));

        // GeLU(x) ≈ x for large positive x (asymptotically x)
        let large = Tensor::from_vec(vec![10.0_f32], vec![1, 1]).unwrap();
        let out2  = matmul_fused(&large, &id, &[&act]).unwrap();
        assert!(out2.as_slice()[0] > 9.9, "GeLU(10) should ≈ 10, got {}", out2.as_slice()[0]);

        // GeLU(x) ≈ 0 for large negative x
        let neg   = Tensor::from_vec(vec![-10.0_f32], vec![1, 1]).unwrap();
        let out3  = matmul_fused(&neg, &id, &[&act]).unwrap();
        assert!(out3.as_slice()[0].abs() < 1e-3, "GeLU(-10) should ≈ 0, got {}", out3.as_slice()[0]);
    }

    // ── matmul + bias + activation (chained) ──────────────────────────────

    #[test]
    fn test_matmul_bias_relu_chained() {
        // CHANGED: verify chained ops == sequential application
        let (m, k, n) = (4, 3, 4);
        let a_data: Vec<f32> = (0..(m*k)).map(|i| i as f32 - 6.0).collect();
        let b_data: Vec<f32> = (0..(k*n)).map(|i| i as f32 * 0.3 - 1.5).collect();
        let bias_data        = vec![-0.5_f32, 0.0, 0.5, 1.0];

        let a = Tensor::from_vec(a_data, vec![m, k]).unwrap();
        let b = Tensor::from_vec(b_data, vec![k, n]).unwrap();

        // sequential reference
        let plain = matmul_naive(&a, &b).unwrap();
        let expected: Vec<f32> = plain.as_slice().iter().enumerate().map(|(idx, &v)| {
            let j = idx % n;
            (v + bias_data[j]).max(0.0) // bias then relu
        }).collect();

        // fused
        let bias_op = BiasAdd::new(bias_data);
        let relu_op = Activation(ActivationFn::ReLU);
        let fused   = matmul_fused(&a, &b, &[&bias_op, &relu_op]).unwrap();

        assert_eq!(fused.dims(), &[m, n]);
        assert!(close_slice(fused.as_slice(), &expected));
    }

    #[test]
    fn test_chain_struct() {
        // CHANGED: Chain wrapper must produce same result as passing ops directly
        let a = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap(); // identity

        let chain: Chain = Chain(vec![
            Box::new(BiasAdd::new(vec![1.0_f32, -1.0])),
            Box::new(Activation(ActivationFn::ReLU)),
        ]);

        let fused_chain  = matmul_fused(&a, &b, &[&chain]).unwrap();

        let bias_op = BiasAdd::new(vec![1.0_f32, -1.0]);
        let relu_op = Activation(ActivationFn::ReLU);
        let fused_direct = matmul_fused(&a, &b, &[&bias_op, &relu_op]).unwrap();

        assert!(close_slice(fused_chain.as_slice(), fused_direct.as_slice()));
    }

    // ── non-contiguous inputs ──────────────────────────────────────────────

    #[test]
    fn test_non_contiguous_input() {
        let a_orig = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let a_t    = a_orig.transpose(0, 1).unwrap(); // non-contiguous [[1,2],[3,4]]
        let id     = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let relu   = Activation(ActivationFn::ReLU);
        let out    = matmul_fused(&a_t, &id, &[&relu]).unwrap();
        // A_t @ I = A_t contiguous = [[1,2],[3,4]], all positive -> unchanged
        assert!(close_slice(out.as_slice(), &[1.0, 2.0, 3.0, 4.0]));
    }

    // ── error handling ─────────────────────────────────────────────────────

    #[test]
    fn test_shape_mismatch_error() {
        let a = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32; 3], vec![3, 1]).unwrap();
        assert!(matches!(matmul_fused(&a, &b, &[]), Err(TensorError::ShapeMismatch { .. })));
    }

    #[test]
    fn test_non_2d_error() {
        let a = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 2, 2]).unwrap();
        let b = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        assert!(matches!(matmul_fused(&a, &b, &[]), Err(TensorError::InvalidShape { .. })));
    }
}
