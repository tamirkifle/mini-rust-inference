//! Scaled dot-product attention — commit 7.1.
//!
//! # Algorithm
//!
//! ```text
//! scores  = Q @ Kᵀ / sqrt(d_k)      shape: [seq_q, seq_k]
//! weights = softmax(scores, dim=-1)   shape: [seq_q, seq_k]
//! out     = weights @ V               shape: [seq_q, d_v]
//! ```
//!
//! This is the "Attention is All You Need" formulation.  The `1/sqrt(d_k)`
//! scale factor prevents the dot products from growing large in magnitude
//! as the key dimension increases, keeping gradients well-behaved.
//!
//! # Input shapes (2-D, single head)
//!
//! | Tensor | Shape          | Description                         |
//! |--------|----------------|-------------------------------------|
//! | `q`    | `[seq_q, d_k]` | Query matrix                        |
//! | `k`    | `[seq_k, d_k]` | Key matrix                          |
//! | `v`    | `[seq_k, d_v]` | Value matrix                        |
//!
//! Output shape: `[seq_q, d_v]`
//!
//! Multi-head usage: call this function per head after splitting the
//! `[seq, n_heads * d_k]` tensor into `n_heads` × `[seq, d_k]` slices.

use crate::tensor::{Result, Tensor, TensorError};
use crate::ops::{matmul_naive, softmax};

// ── public API ──────────────────────────────────────────────────────────────

/// Compute scaled dot-product attention.
///
/// # Arguments
///
/// * `q` – Query tensor of shape `[seq_q, d_k]`
/// * `k` – Key tensor of shape `[seq_k, d_k]`
/// * `v` – Value tensor of shape `[seq_k, d_v]`
///
/// # Returns
///
/// Attention output of shape `[seq_q, d_v]`.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if any input is not exactly 2-D.
/// * [`TensorError::ShapeMismatch`] if `q.dims()[1] != k.dims()[1]`
///   (key dim mismatch) or `k.dims()[0] != v.dims()[0]` (seq_k mismatch).
#[must_use = "returns the attention output; result is not stored in-place"] // CHANGED
pub fn scaled_dot_product_attention(
    q: &Tensor<f32>, // CHANGED
    k: &Tensor<f32>, // CHANGED
    v: &Tensor<f32>, // CHANGED
) -> Result<Tensor<f32>> {
    // ── shape checks ──────────────────────────────────────────────────────
    if q.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "sdpa: q must be 2-D, got {}D {:?}", q.ndim(), q.dims()
            ),
        });
    }
    if k.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "sdpa: k must be 2-D, got {}D {:?}", k.ndim(), k.dims()
            ),
        });
    }
    if v.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "sdpa: v must be 2-D, got {}D {:?}", v.ndim(), v.dims()
            ),
        });
    }

    let [_seq_q, d_k]  = [q.dims()[0], q.dims()[1]]; // CHANGED
    let [seq_k, d_k_k] = [k.dims()[0], k.dims()[1]]; // CHANGED
    let [seq_k_v, _d_v] = [v.dims()[0], v.dims()[1]]; // CHANGED

    if d_k != d_k_k { // CHANGED
        return Err(TensorError::ShapeMismatch {
            expected: vec![seq_k, d_k],
            got:      vec![k.dims()[0], k.dims()[1]],
        });
    }
    if seq_k != seq_k_v { // CHANGED
        return Err(TensorError::ShapeMismatch {
            expected: vec![seq_k, v.dims()[1]],
            got:      vec![v.dims()[0], v.dims()[1]],
        });
    }

    // ── step 1: scores = Q @ Kᵀ ─────────────────────────────────────────
    // k_t shape: [d_k, seq_k]  (zero-copy view, non-contiguous)
    let k_t = k.transpose(0, 1)?; // CHANGED

    // matmul_naive handles non-contiguous inputs via its internal Cow gate
    let scores_raw = matmul_naive(q, &k_t)?; // CHANGED: [seq_q, seq_k]

    // ── step 2: scale by 1/sqrt(d_k) ────────────────────────────────────
    let scale = 1.0_f32 / (d_k as f32).sqrt(); // CHANGED
    let scaled_data: Vec<f32> = scores_raw // CHANGED
        .as_slice()
        .iter()
        .map(|&x| x * scale)
        .collect();
    let scores = Tensor::from_vec(scaled_data, scores_raw.shape().clone())?; // CHANGED

    // ── step 3: softmax over last dim (seq_k) ────────────────────────────
    // softmax() already normalises over the last dimension
    let weights = softmax(&scores)?; // CHANGED: [seq_q, seq_k]

    // ── step 4: out = weights @ V ────────────────────────────────────────
    // weights: [seq_q, seq_k], v: [seq_k, d_v]  →  out: [seq_q, d_v]
    // Ensure v is contiguous for matmul (Cow gate inside matmul_naive handles it)
    let out = matmul_naive(&weights, v)?; // CHANGED: [seq_q, d_v]

    Ok(out) // CHANGED
}

/// Apply a pre-computed additive bias to scores before softmax.
///
/// This helper is used by causal masking (commit 7.2): pass `−∞` for
/// positions that should receive zero attention weight.
///
/// # Errors
///
/// * [`TensorError::ShapeMismatch`] if `scores` and `bias` shapes differ.
#[must_use = "returns a new tensor with bias added"] // CHANGED
pub fn add_attention_bias(
    scores: &Tensor<f32>, // CHANGED
    bias:   &Tensor<f32>, // CHANGED
) -> Result<Tensor<f32>> {
    if scores.dims() != bias.dims() { // CHANGED
        return Err(TensorError::ShapeMismatch {
            expected: scores.dims().to_vec(),
            got:      bias.dims().to_vec(),
        });
    }
    let data: Vec<f32> = scores // CHANGED
        .as_slice()
        .iter()
        .zip(bias.as_slice())
        .map(|(&s, &b)| s + b)
        .collect();
    Tensor::from_vec(data, scores.shape().clone()) // CHANGED
}

/// Scaled dot-product attention with an additive score bias.
///
/// Equivalent to:
/// ```text
/// scores  = Q @ Kᵀ / sqrt(d_k) + bias
/// weights = softmax(scores)
/// out     = weights @ V
/// ```
///
/// The `bias` tensor must match the score shape `[seq_q, seq_k]`.
/// Pass a causal mask (filled with `0.0` or `f32::NEG_INFINITY`) as `bias`.
///
/// # Errors
///
/// Same as [`scaled_dot_product_attention`] plus shape mismatch for `bias`.
#[must_use = "returns the attention output"] // CHANGED
pub fn scaled_dot_product_attention_with_bias(
    q:    &Tensor<f32>, // CHANGED
    k:    &Tensor<f32>, // CHANGED
    v:    &Tensor<f32>, // CHANGED
    bias: &Tensor<f32>, // CHANGED
) -> Result<Tensor<f32>> {
    // ── shape checks (same as plain SDPA) ────────────────────────────────
    if q.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("sdpa_biased: q must be 2-D, got {}D {:?}", q.ndim(), q.dims()),
        });
    }
    if k.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("sdpa_biased: k must be 2-D, got {}D {:?}", k.ndim(), k.dims()),
        });
    }
    if v.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("sdpa_biased: v must be 2-D, got {}D {:?}", v.ndim(), v.dims()),
        });
    }

    let d_k = q.dims()[1]; // CHANGED
    if k.dims()[1] != d_k { // CHANGED
        return Err(TensorError::ShapeMismatch {
            expected: vec![k.dims()[0], d_k],
            got:      k.dims().to_vec(),
        });
    }
    let seq_k = k.dims()[0]; // CHANGED
    if v.dims()[0] != seq_k { // CHANGED
        return Err(TensorError::ShapeMismatch {
            expected: vec![seq_k, v.dims()[1]],
            got:      v.dims().to_vec(),
        });
    }

    let k_t = k.transpose(0, 1)?; // CHANGED
    let scores_raw = matmul_naive(q, &k_t)?; // CHANGED

    let scale = 1.0_f32 / (d_k as f32).sqrt(); // CHANGED
    let scaled_data: Vec<f32> = scores_raw.as_slice().iter().map(|&x| x * scale).collect(); // CHANGED
    let scores = Tensor::from_vec(scaled_data, scores_raw.shape().clone())?; // CHANGED

    // CHANGED: add bias before softmax (e.g. causal mask)
    let scores_biased = add_attention_bias(&scores, bias)?; // CHANGED

    let weights = softmax(&scores_biased)?; // CHANGED
    let out = matmul_naive(&weights, v)?; // CHANGED

    Ok(out) // CHANGED
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    // ── output shape ─────────────────────────────────────────────────────

    #[test]
    fn test_sdpa_output_shape_square() {
        // Q: [4, 8], K: [4, 8], V: [4, 8] → out: [4, 8]
        let q = Tensor::from_vec(vec![0.0_f32; 32], vec![4, 8]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 32], vec![4, 8]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; 32], vec![4, 8]).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(out.dims(), &[4, 8]);
    }

    #[test]
    fn test_sdpa_output_shape_rectangular() {
        // Q: [2, 4], K: [6, 4], V: [6, 8] → out: [2, 8]
        let q = Tensor::from_vec(vec![1.0_f32; 8],  vec![2, 4]).unwrap();
        let k = Tensor::from_vec(vec![1.0_f32; 24], vec![6, 4]).unwrap();
        let v = Tensor::from_vec(vec![1.0_f32; 48], vec![6, 8]).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(out.dims(), &[2, 8]);
    }

    // ── trivial case: uniform attention should average V ─────────────────

    #[test]
    fn test_sdpa_uniform_attention_averages_v() {
        // All-zero Q and K → all-equal scores → uniform softmax → output = mean(V)
        // Q: [2, 4] zeros, K: [3, 4] zeros → scores = [[0,0,0],[0,0,0]] → uniform
        // V: rows [1,2,3], [4,5,6], [7,8,9] → each output row = mean = [4, 5, 6]
        let q = Tensor::from_vec(vec![0.0_f32; 8], vec![2, 4]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 12], vec![3, 4]).unwrap();
        let v_data = vec![
            1.0_f32, 2.0, 3.0,
            4.0,     5.0, 6.0,
            7.0,     8.0, 9.0,
        ];
        let v = Tensor::from_vec(v_data, vec![3, 3]).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v).unwrap();

        // Each output row should be [4, 5, 6] (mean of V rows)
        let expected = [4.0_f32, 5.0, 6.0, 4.0, 5.0, 6.0];
        assert!(close_slice(out.as_slice(), &expected, 1e-5),
            "got {:?}, expected {:?}", out.as_slice(), expected);
    }

    // ── hand-computed reference ───────────────────────────────────────────

    #[test]
    fn test_sdpa_hand_computed_2x2() {
        // Q = [[1,0],[0,1]], K = [[1,0],[0,1]], V = [[1,2],[3,4]]
        // d_k = 2, scale = 1/sqrt(2) ≈ 0.7071
        //
        // scores = Q @ Kᵀ = I @ I = I → [[1,0],[0,1]]
        // scaled = [[0.7071, 0], [0, 0.7071]]
        // softmax row 0: exp(0.7071)/(exp(0.7071)+1) ≈ 0.6700, 0.3300
        // softmax row 1: same flipped   ≈ 0.3300, 0.6700
        // out[0] = 0.6700*[1,2] + 0.3300*[3,4] = [1.660, 2.660]
        // out[1] = 0.3300*[1,2] + 0.6700*[3,4] = [2.340, 3.340]

        let q = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let k = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let v = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let out = scaled_dot_product_attention(&q, &k, &v).unwrap();
        assert_eq!(out.dims(), &[2, 2]);

        let expected = [1.6602_f32, 2.6602, 2.3398, 3.3398];
        assert!(close_slice(out.as_slice(), &expected, 1e-3),
            "got {:?}", out.as_slice());
    }

    // ── scale factor applied correctly ────────────────────────────────────

    #[test]
    fn test_sdpa_scale_reduces_sharpness() {
        // The 1/sqrt(d_k) scale dampens scores when d_k grows, *holding the
        // per-dimension contribution constant*.  We test this by constructing
        // Q and K so that the raw dot product is the same scalar value (4.0)
        // for both d_k=1 and d_k=4, but only the first key matches.
        //
        // d_k=1: Q=[4], K=[[4],[0]]  → raw score = [16, 0], scale=1/1=1.0
        //   → scaled = [16, 0] → softmax → w0 ≈ sigmoid(16) ≈ 1.0 (very sharp)
        //
        // d_k=4: Q=[2,0,0,0], K=[[2,0,0,0],[0,0,0,0]] → raw=[4, 0], scale=1/2=0.5
        //   → scaled = [2, 0] → softmax → w0 = e^2/(e^2+1) ≈ 0.88  (flatter)
        //
        // So d_k=4 should give a *lower* w0 than d_k=1 for same raw dot product.

        // d_k = 1: raw score = 4*4 = 16 → scale = 1.0 → softmax([16, 0])
        let q1 = Tensor::from_vec(vec![4.0_f32], vec![1, 1]).unwrap(); // CHANGED
        let k1 = Tensor::from_vec(vec![4.0_f32, 0.0], vec![2, 1]).unwrap(); // CHANGED
        let v1 = Tensor::from_vec(vec![1.0_f32, 0.0], vec![2, 1]).unwrap();
        let out1 = scaled_dot_product_attention(&q1, &k1, &v1).unwrap();
        let w0_sharp = out1.as_slice()[0]; // should be ~1.0 (very sharp)

        // d_k = 4: raw score = 2*2 = 4 → scale = 1/sqrt(4) = 0.5 → softmax([2, 0])
        let q4 = Tensor::from_vec(vec![2.0_f32, 0.0, 0.0, 0.0], vec![1, 4]).unwrap(); // CHANGED
        let k4 = Tensor::from_vec(vec![2.0_f32, 0.0, 0.0, 0.0,  // row 0
                                       0.0,     0.0, 0.0, 0.0], vec![2, 4]).unwrap(); // CHANGED
        let v4 = Tensor::from_vec(vec![1.0_f32, 0.0], vec![2, 1]).unwrap();
        let out4 = scaled_dot_product_attention(&q4, &k4, &v4).unwrap();
        let w0_flat = out4.as_slice()[0]; // should be ~0.88

        // Larger d_k with same raw score but stronger scaling → flatter distribution
        assert!(w0_flat < w0_sharp, // CHANGED
            "expected flatter attention for d_k=4 (w0={w0_flat:.4}), \
             got sharper than d_k=1 (w0={w0_sharp:.4})");
    }

    // ── 1x1 degenerate case ───────────────────────────────────────────────

    #[test]
    fn test_sdpa_1x1_degenerate() {
        // Q=[3], K=[2], V=[5] — single token, single key
        let q = Tensor::from_vec(vec![3.0_f32], vec![1, 1]).unwrap();
        let k = Tensor::from_vec(vec![2.0_f32], vec![1, 1]).unwrap();
        let v = Tensor::from_vec(vec![5.0_f32], vec![1, 1]).unwrap();
        let out = scaled_dot_product_attention(&q, &k, &v).unwrap();
        // Only one key: softmax([anything]) = [1.0] → output = 1.0 * V = [5.0]
        assert_eq!(out.dims(), &[1, 1]);
        assert!(close(out.as_slice()[0], 5.0));
    }

    // ── non-contiguous V input ────────────────────────────────────────────

    #[test]
    fn test_sdpa_non_contiguous_v() {
        // Transposing V makes it non-contiguous; SDPA should still produce correct shape
        let q = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let k = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let v_base = Tensor::from_vec(vec![1.0_f32, 3.0, 2.0, 4.0], vec![2, 2]).unwrap();
        let v_t = v_base.transpose(0, 1).unwrap(); // non-contiguous [2,2]
        let out = scaled_dot_product_attention(&q, &k, &v_t);
        assert!(out.is_ok());
        assert_eq!(out.unwrap().dims(), &[2, 2]);
    }

    // ── add_attention_bias ────────────────────────────────────────────────

    #[test]
    fn test_add_attention_bias_zero_mask() {
        let scores = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bias   = Tensor::from_vec(vec![0.0_f32; 4], vec![2, 2]).unwrap();
        let out = add_attention_bias(&scores, &bias).unwrap();
        assert!(close_slice(out.as_slice(), scores.as_slice(), 1e-7));
    }

    #[test]
    fn test_add_attention_bias_neg_inf_masks() {
        // Adding -inf to a position should drive that attention weight to ~0 after softmax
        let scores = Tensor::from_vec(vec![1.0_f32, 1.0], vec![1, 2]).unwrap();
        let bias   = Tensor::from_vec(vec![0.0_f32, f32::NEG_INFINITY], vec![1, 2]).unwrap();
        let biased = add_attention_bias(&scores, &bias).unwrap();
        // softmax([1, -inf]) → [1.0, 0.0]
        let weights = crate::ops::softmax(&biased).unwrap();
        assert!(close(weights.as_slice()[0], 1.0));
        assert!(close(weights.as_slice()[1], 0.0));
    }

    // ── biased SDPA ───────────────────────────────────────────────────────

    #[test]
    fn test_sdpa_with_zero_bias_matches_plain() {
        let q = Tensor::from_vec(vec![1.0_f32, 0.5, 0.25, 0.1], vec![2, 2]).unwrap();
        let k = Tensor::from_vec(vec![0.3_f32, 0.7, 0.9, 0.2], vec![2, 2]).unwrap();
        let v = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let bias = Tensor::from_vec(vec![0.0_f32; 4], vec![2, 2]).unwrap();

        let plain  = scaled_dot_product_attention(&q, &k, &v).unwrap();
        let biased = scaled_dot_product_attention_with_bias(&q, &k, &v, &bias).unwrap();

        assert!(close_slice(plain.as_slice(), biased.as_slice(), 1e-6),
            "biased(zero) != plain:\n  plain={:?}\n  biased={:?}",
            plain.as_slice(), biased.as_slice());
    }

    // ── error handling ────────────────────────────────────────────────────

    #[test]
    fn test_sdpa_error_non_2d_q() {
        let q = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 2, 2]).unwrap();
        let k = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let v = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        assert!(matches!(
            scaled_dot_product_attention(&q, &k, &v),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_sdpa_error_dk_mismatch() {
        let q = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap();
        let k = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 4]).unwrap(); // d_k=4 ≠ 3
        let v = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        assert!(matches!(
            scaled_dot_product_attention(&q, &k, &v),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_sdpa_error_seqk_mismatch() {
        let q = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap();
        let k = Tensor::from_vec(vec![1.0_f32; 9], vec![3, 3]).unwrap(); // seq_k=3
        let v = Tensor::from_vec(vec![1.0_f32; 8], vec![4, 2]).unwrap(); // seq_k=4 ≠ 3
        assert!(matches!(
            scaled_dot_product_attention(&q, &k, &v),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }
}
