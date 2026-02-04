//! Grouped-Query Attention (GQA) — commit 7.4.
//!
//! # Background
//!
//! Standard MHA has `n_kv_heads == n_heads`: every query head gets its own K/V
//! head.  GQA generalises this so that `n_kv_heads` divides `n_heads`, and each
//! group of `n_heads / n_kv_heads` query heads **shares** one K/V head.  This
//! reduces the KV-cache memory footprint by the same ratio.
//!
//! Special cases:
//! - `n_kv_heads == n_heads`  →  standard MHA  (Llama 1, GPT-2)
//! - `n_kv_heads == 1`        →  MQA (Multi-Query Attention)
//! - `1 < n_kv_heads < n_heads` →  GQA  (Llama 2 70B, Llama 3 all sizes)
//!
//! # Algorithm
//!
//! ```text
//! groups = n_heads / n_kv_heads          (must divide evenly)
//!
//! For q_head h in 0..n_heads:
//!   kv_head = h / groups                 (integer division)
//!   q_h  = Q[:, h*d_k .. (h+1)*d_k]     [seq_q, d_k]
//!   k_kv = K[:, kv_head*d_k ..]         [seq_k, d_k]
//!   v_kv = V[:, kv_head*d_v ..]         [seq_k, d_v]
//!   out_h = sdpa(q_h, k_kv, v_kv)       [seq_q, d_v]
//!
//! out = concat(out_0 .. out_{n_heads-1}) [seq_q, n_heads * d_v]
//! ```

use crate::tensor::{Result, Tensor, TensorError};
use crate::attention::sdpa::scaled_dot_product_attention_with_bias;
use crate::attention::mask::{causal_mask, causal_mask_with_offset};
use crate::attention::multihead::{split_head, concat_heads};

// ── validation helper ───────────────────────────────────────────────────────

/// Validate GQA preconditions.  Returns `(seq_q, seq_k, d_k, d_v, groups)`.
fn gqa_check( // CHANGED
    q:          &Tensor<f32>,
    k:          &Tensor<f32>,
    v:          &Tensor<f32>,
    n_heads:    usize,
    n_kv_heads: usize,
) -> Result<(usize, usize, usize, usize, usize)> {
    if q.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("gqa: q must be 2-D, got {}D", q.ndim()),
        });
    }
    if k.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("gqa: k must be 2-D, got {}D", k.ndim()),
        });
    }
    if v.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("gqa: v must be 2-D, got {}D", v.ndim()),
        });
    }
    if n_heads == 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: "gqa: n_heads must be > 0".to_string(),
        });
    }
    if n_kv_heads == 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: "gqa: n_kv_heads must be > 0".to_string(),
        });
    }
    if n_heads % n_kv_heads != 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "gqa: n_heads({n_heads}) must be divisible by n_kv_heads({n_kv_heads})"
            ),
        });
    }
    let q_total = q.dims()[1]; // CHANGED
    let k_total = k.dims()[1]; // CHANGED
    let v_total = v.dims()[1]; // CHANGED
    if q_total % n_heads != 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("gqa: q dim({q_total}) not divisible by n_heads({n_heads})"),
        });
    }
    let d_k = q_total / n_heads; // CHANGED
    if k_total != n_kv_heads * d_k { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "gqa: k dim({k_total}) != n_kv_heads({n_kv_heads}) * d_k({d_k})"
            ),
        });
    }
    if v_total % n_kv_heads != 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("gqa: v dim({v_total}) not divisible by n_kv_heads({n_kv_heads})"),
        });
    }
    let d_v = v_total / n_kv_heads; // CHANGED
    if k.dims()[0] != v.dims()[0] { // CHANGED
        return Err(TensorError::ShapeMismatch {
            expected: vec![k.dims()[0], k_total],
            got:      v.dims().to_vec(),
        });
    }
    let seq_q  = q.dims()[0]; // CHANGED
    let seq_k  = k.dims()[0]; // CHANGED
    let groups = n_heads / n_kv_heads; // CHANGED
    Ok((seq_q, seq_k, d_k, d_v, groups)) // CHANGED
}

// ── core GQA functions ──────────────────────────────────────────────────────

/// Grouped-query attention (no mask).
///
/// # Arguments
///
/// * `q`          – `[seq_q, n_heads * d_k]`
/// * `k`          – `[seq_k, n_kv_heads * d_k]`
/// * `v`          – `[seq_k, n_kv_heads * d_v]`
/// * `n_heads`    – total number of query heads
/// * `n_kv_heads` – number of key/value heads (`n_heads % n_kv_heads == 0`)
///
/// # Returns
///
/// `[seq_q, n_heads * d_v]`
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] / [`TensorError::ShapeMismatch`] on bad inputs.
#[must_use = "returns the GQA output"] // CHANGED
pub fn grouped_query_attention( // CHANGED
    q:          &Tensor<f32>,
    k:          &Tensor<f32>,
    v:          &Tensor<f32>,
    n_heads:    usize,
    n_kv_heads: usize,
) -> Result<Tensor<f32>> {
    let (_seq_q, _seq_k, _d_k, _d_v, groups) = gqa_check(q, k, v, n_heads, n_kv_heads)?; // CHANGED

    let mut head_outputs: Vec<Tensor<f32>> = Vec::with_capacity(n_heads); // CHANGED
    for h in 0..n_heads { // CHANGED
        let kv_h  = h / groups; // CHANGED: query head h maps to KV head kv_h
        let q_h   = split_head(q, n_heads,    h)?;   // CHANGED: [seq_q, d_k]
        let k_kv  = split_head(k, n_kv_heads, kv_h)?; // CHANGED: [seq_k, d_k]
        let v_kv  = split_head(v, n_kv_heads, kv_h)?; // CHANGED: [seq_k, d_v]
        let bias  = Tensor::from_vec( // CHANGED: zero bias = no mask
            vec![0.0_f32; q_h.dims()[0] * k_kv.dims()[0]],
            vec![q_h.dims()[0], k_kv.dims()[0]],
        )?;
        let out_h = scaled_dot_product_attention_with_bias(&q_h, &k_kv, &v_kv, &bias)?; // CHANGED
        head_outputs.push(out_h); // CHANGED
    }
    concat_heads(&head_outputs) // CHANGED
}

/// GQA with causal mask (full-sequence prefill).  Requires `seq_q == seq_k`.
///
/// # Errors
///
/// Same as [`grouped_query_attention`] plus [`TensorError::InvalidShape`] if
/// `seq_q != seq_k`.
#[must_use = "returns the causally-masked GQA output"] // CHANGED
pub fn grouped_query_attention_causal( // CHANGED
    q:          &Tensor<f32>,
    k:          &Tensor<f32>,
    v:          &Tensor<f32>,
    n_heads:    usize,
    n_kv_heads: usize,
) -> Result<Tensor<f32>> {
    let (seq_q, seq_k, _d_k, _d_v, groups) = gqa_check(q, k, v, n_heads, n_kv_heads)?; // CHANGED
    if seq_q != seq_k { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "gqa_causal: seq_q({seq_q}) != seq_k({seq_k}); \
                 use grouped_query_attention_causal_with_offset for decode"
            ),
        });
    }
    let mask = causal_mask(seq_q)?; // CHANGED: one mask shared across all heads

    let mut head_outputs: Vec<Tensor<f32>> = Vec::with_capacity(n_heads); // CHANGED
    for h in 0..n_heads { // CHANGED
        let kv_h = h / groups; // CHANGED
        let q_h  = split_head(q, n_heads,    h)?;    // CHANGED
        let k_kv = split_head(k, n_kv_heads, kv_h)?; // CHANGED
        let v_kv = split_head(v, n_kv_heads, kv_h)?; // CHANGED
        let out_h = scaled_dot_product_attention_with_bias(&q_h, &k_kv, &v_kv, &mask)?; // CHANGED
        head_outputs.push(out_h); // CHANGED
    }
    concat_heads(&head_outputs) // CHANGED
}

/// GQA with causal mask for the **KV-cache decode step**.
///
/// `start_pos` is the absolute position of the first query token.
///
/// # Errors
///
/// Same as [`grouped_query_attention`] plus range errors from
/// [`causal_mask_with_offset`].
#[must_use = "returns the masked GQA output"] // CHANGED
pub fn grouped_query_attention_causal_with_offset( // CHANGED
    q:          &Tensor<f32>,
    k:          &Tensor<f32>,
    v:          &Tensor<f32>,
    n_heads:    usize,
    n_kv_heads: usize,
    start_pos:  usize,
) -> Result<Tensor<f32>> {
    let (seq_q, seq_k, _d_k, _d_v, groups) = gqa_check(q, k, v, n_heads, n_kv_heads)?; // CHANGED
    let mask = causal_mask_with_offset(seq_q, seq_k, start_pos)?; // CHANGED

    let mut head_outputs: Vec<Tensor<f32>> = Vec::with_capacity(n_heads); // CHANGED
    for h in 0..n_heads { // CHANGED
        let kv_h = h / groups; // CHANGED
        let q_h  = split_head(q, n_heads,    h)?;    // CHANGED
        let k_kv = split_head(k, n_kv_heads, kv_h)?; // CHANGED
        let v_kv = split_head(v, n_kv_heads, kv_h)?; // CHANGED
        let out_h = scaled_dot_product_attention_with_bias(&q_h, &k_kv, &v_kv, &mask)?; // CHANGED
        head_outputs.push(out_h); // CHANGED
    }
    concat_heads(&head_outputs) // CHANGED
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::multihead::multi_head_attention;

    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    // ── n_kv_heads == n_heads → identical to standard MHA ────────────────

    #[test]
    fn test_gqa_mha_equivalence() { // CHANGED
        // When n_kv_heads == n_heads, GQA must produce the same result as MHA.
        let q = Tensor::from_vec(
            vec![1.0_f32, 0.5, 0.25, 0.1,  0.9, 0.8, 0.3, 0.2,
                 0.4,     0.6, 0.7,  0.15, 0.05,0.95,0.35,0.55],
            vec![2, 8]).unwrap();
        let k = Tensor::from_vec(
            vec![0.6_f32, 0.4, 0.7, 0.3,  0.1, 0.9, 0.5, 0.2,
                 0.8,     0.2, 0.3, 0.85, 0.45,0.55,0.65,0.75],
            vec![2, 8]).unwrap();
        let v = Tensor::from_vec(
            vec![1.0_f32, 2.0, 3.0, 4.0,  5.0, 6.0, 7.0, 8.0,
                 0.1,     0.2, 0.3, 0.4,  0.5, 0.6, 0.7, 0.8],
            vec![2, 8]).unwrap();

        let gqa_out = grouped_query_attention(&q, &k, &v, 4, 4).unwrap(); // CHANGED
        let mha_out = multi_head_attention(&q, &k, &v, 4).unwrap();

        assert!(close_slice(gqa_out.as_slice(), mha_out.as_slice(), 1e-6),
            "GQA(n_kv==n_q) != MHA:\n  gqa={:?}\n  mha={:?}",
            gqa_out.as_slice(), mha_out.as_slice());
    }

    // ── MQA: n_kv_heads == 1 ─────────────────────────────────────────────

    #[test]
    fn test_gqa_mqa_output_shape() { // CHANGED
        // n_heads=4, n_kv_heads=1: Q [3,8], K [3,2], V [3,2] → out [3,8]
        let q = Tensor::from_vec(vec![0.0_f32; 24], vec![3, 8]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32;  6], vec![3, 2]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32;  6], vec![3, 2]).unwrap();
        let out = grouped_query_attention(&q, &k, &v, 4, 1).unwrap();
        assert_eq!(out.dims(), &[3, 8]);
    }

    #[test]
    fn test_gqa_mqa_all_heads_same_kv() { // CHANGED
        // With a single KV head all query heads see the same K/V.
        // Use all-zero Q/K → uniform attention over 2 V rows.
        // V = [[1,10],[2,20]], mean = [1.5, 15] for every query head.
        let q = Tensor::from_vec(vec![0.0_f32; 4], vec![1, 4]).unwrap(); // 1 token, 2 q-heads, d_k=2
        let k = Tensor::from_vec(vec![0.0_f32; 4], vec![2, 2]).unwrap(); // 2 keys, 1 kv-head, d_k=2
        let v = Tensor::from_vec(vec![1.0_f32, 10.0, 2.0, 20.0], vec![2, 2]).unwrap();

        let out = grouped_query_attention(&q, &k, &v, 2, 1).unwrap(); // CHANGED
        assert_eq!(out.dims(), &[1, 4]); // 1 tok × (2 heads × 2 d_v)

        // Both heads should produce [1.5, 15.0]
        assert!((out.as_slice()[0] - 1.5).abs()  < 1e-5, "h0[0]={}", out.as_slice()[0]);
        assert!((out.as_slice()[1] - 15.0).abs() < 1e-5, "h0[1]={}", out.as_slice()[1]);
        assert!((out.as_slice()[2] - 1.5).abs()  < 1e-5, "h1[0]={}", out.as_slice()[2]);
        assert!((out.as_slice()[3] - 15.0).abs() < 1e-5, "h1[1]={}", out.as_slice()[3]);
    }

    // ── groups = 2: two Q heads share one KV head ─────────────────────────

    #[test]
    fn test_gqa_groups2_kv_sharing() { // CHANGED
        // n_heads=4, n_kv_heads=2, groups=2
        // q_heads 0,1 share kv_head 0; q_heads 2,3 share kv_head 1.
        // K[kv0] and K[kv1] are orthogonal so each group attends differently.
        // All-zero Q → uniform distribution within each shared KV group.
        // V kv0 rows: [1,0],[2,0] → mean=[1.5, 0]
        // V kv1 rows: [0,3],[0,4] → mean=[0, 3.5]
        // Expected output heads 0,1: [1.5, 0]; heads 2,3: [0, 3.5]

        let seq = 2_usize;
        // Q: [2, 8], all zeros (4 heads × d_k=2)
        let q = Tensor::from_vec(vec![0.0_f32; 16], vec![seq, 8]).unwrap();
        // K: [2, 4], two KV heads × d_k=2, all zeros → uniform attention
        let k = Tensor::from_vec(vec![0.0_f32;  8], vec![seq, 4]).unwrap();
        // V: [2, 4], kv_head 0 cols [0,1], kv_head 1 cols [2,3]
        let v = Tensor::from_vec(vec![
            1.0_f32, 0.0,  0.0, 3.0, // row 0
            2.0,     0.0,  0.0, 4.0, // row 1
        ], vec![seq, 4]).unwrap();

        let out = grouped_query_attention(&q, &k, &v, 4, 2).unwrap(); // CHANGED
        assert_eq!(out.dims(), &[2, 8]); // seq=2, 4 heads × d_v=2

        // For each output token: heads 0,1 → [1.5,0.0]; heads 2,3 → [0.0,3.5]
        for tok in 0..seq {
            let base = tok * 8;
            let s = out.as_slice();
            assert!((s[base]   - 1.5).abs() < 1e-5, "tok{tok} h0[0]={}", s[base]);
            assert!((s[base+1] - 0.0).abs() < 1e-5, "tok{tok} h0[1]={}", s[base+1]);
            assert!((s[base+2] - 1.5).abs() < 1e-5, "tok{tok} h1[0]={}", s[base+2]);
            assert!((s[base+3] - 0.0).abs() < 1e-5, "tok{tok} h1[1]={}", s[base+3]);
            assert!((s[base+4] - 0.0).abs() < 1e-5, "tok{tok} h2[0]={}", s[base+4]);
            assert!((s[base+5] - 3.5).abs() < 1e-5, "tok{tok} h2[1]={}", s[base+5]);
            assert!((s[base+6] - 0.0).abs() < 1e-5, "tok{tok} h3[0]={}", s[base+6]);
            assert!((s[base+7] - 3.5).abs() < 1e-5, "tok{tok} h3[1]={}", s[base+7]);
        }
    }

    // ── causal GQA ────────────────────────────────────────────────────────

    #[test]
    fn test_gqa_causal_first_token_self_only() { // CHANGED
        // Token 0 may only see itself; its output should equal V[kv_head_0, row0].
        // n_heads=2, n_kv_heads=1, groups=2; both Q heads share the single KV head.
        // V row0 = [5, 7], V row1 = [99, 99] (should be masked for token 0)
        let q = Tensor::from_vec(vec![
            1.0_f32, 0.0, 1.0, 0.0, // token 0: both heads identical
            0.5,     0.5, 0.5, 0.5, // token 1
        ], vec![2, 4]).unwrap();
        let k = Tensor::from_vec(vec![
            1.0_f32, 0.0, // kv row 0
            0.0,     1.0, // kv row 1
        ], vec![2, 2]).unwrap();
        let v = Tensor::from_vec(vec![
            5.0_f32, 7.0,   // kv row 0
            99.0,    99.0,  // kv row 1 — must be masked for token 0
        ], vec![2, 2]).unwrap();

        let out = grouped_query_attention_causal(&q, &k, &v, 2, 1).unwrap(); // CHANGED
        assert_eq!(out.dims(), &[2, 4]);

        // Token 0, both heads → V[kv0][row0] = [5, 7]
        let tok0 = &out.as_slice()[0..4];
        assert!((tok0[0] - 5.0).abs() < 1e-4, "tok0 h0[0]={}", tok0[0]);
        assert!((tok0[1] - 7.0).abs() < 1e-4, "tok0 h0[1]={}", tok0[1]);
        assert!((tok0[2] - 5.0).abs() < 1e-4, "tok0 h1[0]={}", tok0[2]);
        assert!((tok0[3] - 7.0).abs() < 1e-4, "tok0 h1[1]={}", tok0[3]);
    }

    #[test]
    fn test_gqa_causal_with_offset_shape() { // CHANGED
        // decode: 1 query token, 4 KV tokens, start_pos=3
        // n_heads=4, n_kv_heads=2
        let q = Tensor::from_vec(vec![0.0_f32; 8],  vec![1, 8]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 16], vec![4, 4]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; 16], vec![4, 4]).unwrap();
        let out = grouped_query_attention_causal_with_offset(&q, &k, &v, 4, 2, 3).unwrap();
        assert_eq!(out.dims(), &[1, 8]);
    }

    // ── error handling ────────────────────────────────────────────────────

    #[test]
    fn test_gqa_n_heads_not_divisible_by_kv_heads() { // CHANGED
        let q = Tensor::from_vec(vec![0.0_f32; 12], vec![2, 6]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32;  8], vec![2, 4]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32;  8], vec![2, 4]).unwrap();
        assert!(matches!(
            grouped_query_attention(&q, &k, &v, 3, 2),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_gqa_k_dim_mismatch() { // CHANGED
        // n_heads=4, n_kv_heads=2, d_k=2 → k must be [*, 4]; give [*, 6]
        let q = Tensor::from_vec(vec![0.0_f32; 8],  vec![1, 8]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 6],  vec![1, 6]).unwrap(); // wrong
        let v = Tensor::from_vec(vec![0.0_f32; 4],  vec![1, 4]).unwrap();
        assert!(matches!(
            grouped_query_attention(&q, &k, &v, 4, 2),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_gqa_zero_kv_heads_rejected() { // CHANGED
        let q = Tensor::from_vec(vec![0.0_f32; 4], vec![1, 4]).unwrap();
        let k = q.clone();
        let v = q.clone();
        assert!(matches!(
            grouped_query_attention(&q, &k, &v, 2, 0),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_gqa_causal_seq_mismatch_rejected() { // CHANGED
        let q = Tensor::from_vec(vec![0.0_f32; 4], vec![2, 2]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 6], vec![3, 2]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; 6], vec![3, 2]).unwrap();
        assert!(matches!(
            grouped_query_attention_causal(&q, &k, &v, 1, 1),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
