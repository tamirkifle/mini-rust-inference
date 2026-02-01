//! Multi-head attention — commit 7.3.
//!
//! # Algorithm
//!
//! ```text
//! Input Q: [seq_q, n_heads * d_k]
//! Input K: [seq_k, n_heads * d_k]
//! Input V: [seq_k, n_heads * d_v]
//!
//! For h in 0..n_heads:
//!   q_h = Q[:, h*d_k .. (h+1)*d_k]   shape: [seq_q, d_k]
//!   k_h = K[:, h*d_k .. (h+1)*d_k]   shape: [seq_k, d_k]
//!   v_h = V[:, h*d_v .. (h+1)*d_v]   shape: [seq_k, d_v]
//!   out_h = sdpa(q_h, k_h, v_h)       shape: [seq_q, d_v]
//!
//! out = concat(out_0, out_1, ...) along dim=1   shape: [seq_q, n_heads * d_v]
//! ```
//!
//! # Notes
//!
//! - Projection matrices (W_q, W_k, W_v, W_o) are **not** included here.
//!   The transformer block (commit 8.1) owns those projections.
//! - Both causal and non-causal variants are provided.
//! - GQA (n_kv_heads != n_heads) lives in commit 7.4 (`gqa.rs`).

use crate::tensor::{Result, Tensor, TensorError};
use crate::attention::sdpa::scaled_dot_product_attention_with_bias;
use crate::attention::mask::causal_mask;
use crate::ops::softmax;
use crate::ops::matmul_naive;

// ── head split / concat helpers ─────────────────────────────────────────────

/// Extract head `h` from a packed `[seq, n_heads * d_head]` tensor.
///
/// Returns a **contiguous** `[seq, d_head]` tensor (copy required because
/// head columns are strided in the packed layout).
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `x` is not 2-D.
/// * [`TensorError::InvalidShape`] if `x.dims()[1] % n_heads != 0` or `h >= n_heads`.
#[must_use = "returns a new contiguous head tensor"] // CHANGED
pub fn split_head( // CHANGED
    x:       &Tensor<f32>,
    n_heads: usize,
    h:       usize,
) -> Result<Tensor<f32>> {
    if x.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("split_head: x must be 2-D, got {}D {:?}", x.ndim(), x.dims()),
        });
    }
    let seq      = x.dims()[0]; // CHANGED
    let total_d  = x.dims()[1]; // CHANGED
    if total_d % n_heads != 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "split_head: total_d({total_d}) is not divisible by n_heads({n_heads})"
            ),
        });
    }
    if h >= n_heads { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("split_head: head index {h} >= n_heads {n_heads}"),
        });
    }
    let d_head = total_d / n_heads; // CHANGED
    let src    = if x.is_contiguous() { // CHANGED
        std::borrow::Cow::Borrowed(x)
    } else {
        std::borrow::Cow::Owned(x.contiguous())
    };
    let src_data = src.as_slice(); // CHANGED

    // CHANGED: gather head h's columns into a contiguous buffer
    let mut data = vec![0.0_f32; seq * d_head]; // CHANGED
    for i in 0..seq { // CHANGED
        let src_off = i * total_d + h * d_head; // CHANGED
        let dst_off = i * d_head; // CHANGED
        data[dst_off..dst_off + d_head] // CHANGED
            .copy_from_slice(&src_data[src_off..src_off + d_head]); // CHANGED
    }
    Tensor::from_vec(data, vec![seq, d_head]) // CHANGED
}

/// Concatenate a slice of same-shape `[seq, d_head]` tensors along dim=1.
///
/// Returns `[seq, n_heads * d_head]`.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `heads` is empty or shapes are inconsistent.
#[must_use = "returns the concatenated tensor"] // CHANGED
pub fn concat_heads(heads: &[Tensor<f32>]) -> Result<Tensor<f32>> { // CHANGED
    if heads.is_empty() { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: "concat_heads: heads slice must not be empty".to_string(),
        });
    }
    let seq    = heads[0].dims()[0]; // CHANGED
    let d_head = heads[0].dims()[1]; // CHANGED
    for (i, h) in heads.iter().enumerate() { // CHANGED
        if h.ndim() != 2 || h.dims()[0] != seq || h.dims()[1] != d_head { // CHANGED
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "concat_heads: head {i} has shape {:?}, expected [{seq}, {d_head}]",
                    h.dims()
                ),
            });
        }
    }
    let n_heads  = heads.len(); // CHANGED
    let total_d  = n_heads * d_head; // CHANGED
    let mut data = vec![0.0_f32; seq * total_d]; // CHANGED
    for (h_idx, head) in heads.iter().enumerate() { // CHANGED
        let src = if head.is_contiguous() { // CHANGED
            std::borrow::Cow::Borrowed(head)
        } else {
            std::borrow::Cow::Owned(head.contiguous())
        };
        let src_data = src.as_slice(); // CHANGED
        for i in 0..seq { // CHANGED
            let dst_off = i * total_d + h_idx * d_head; // CHANGED
            let src_off = i * d_head; // CHANGED
            data[dst_off..dst_off + d_head] // CHANGED
                .copy_from_slice(&src_data[src_off..src_off + d_head]); // CHANGED
        }
    }
    Tensor::from_vec(data, vec![seq, total_d]) // CHANGED
}

// ── core MHA implementation ─────────────────────────────────────────────────

/// Validate common MHA preconditions; return `(seq_q, seq_k, d_k, d_v)`.
fn mha_check( // CHANGED
    q:       &Tensor<f32>,
    k:       &Tensor<f32>,
    v:       &Tensor<f32>,
    n_heads: usize,
) -> Result<(usize, usize, usize, usize)> {
    if q.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("mha: q must be 2-D, got {}D", q.ndim()),
        });
    }
    if k.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("mha: k must be 2-D, got {}D", k.ndim()),
        });
    }
    if v.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("mha: v must be 2-D, got {}D", v.ndim()),
        });
    }
    if n_heads == 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: "mha: n_heads must be > 0".to_string(),
        });
    }
    let q_total = q.dims()[1]; // CHANGED
    let k_total = k.dims()[1]; // CHANGED
    let v_total = v.dims()[1]; // CHANGED
    if q_total != k_total { // CHANGED
        return Err(TensorError::ShapeMismatch {
            expected: q.dims().to_vec(),
            got:      k.dims().to_vec(),
        });
    }
    if q_total % n_heads != 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "mha: q dim ({q_total}) not divisible by n_heads ({n_heads})"
            ),
        });
    }
    if v_total % n_heads != 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "mha: v dim ({v_total}) not divisible by n_heads ({n_heads})"
            ),
        });
    }
    if k.dims()[0] != v.dims()[0] { // CHANGED
        return Err(TensorError::ShapeMismatch {
            expected: vec![k.dims()[0], v.dims()[1]],
            got:      v.dims().to_vec(),
        });
    }
    let seq_q = q.dims()[0]; // CHANGED
    let seq_k = k.dims()[0]; // CHANGED
    let d_k   = q_total / n_heads; // CHANGED
    let d_v   = v_total / n_heads; // CHANGED
    Ok((seq_q, seq_k, d_k, d_v)) // CHANGED
}

/// Multi-head attention (no mask).
///
/// # Arguments
///
/// * `q`       – `[seq_q, n_heads * d_k]`
/// * `k`       – `[seq_k, n_heads * d_k]`
/// * `v`       – `[seq_k, n_heads * d_v]`
/// * `n_heads` – number of attention heads
///
/// # Returns
///
/// `[seq_q, n_heads * d_v]`
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] / [`TensorError::ShapeMismatch`] on bad inputs.
#[must_use = "returns the multi-head attention output"] // CHANGED
pub fn multi_head_attention( // CHANGED
    q:       &Tensor<f32>,
    k:       &Tensor<f32>,
    v:       &Tensor<f32>,
    n_heads: usize,
) -> Result<Tensor<f32>> {
    let (_seq_q, _seq_k, _d_k, _d_v) = mha_check(q, k, v, n_heads)?; // CHANGED

    // CHANGED: run SDPA per head, collect outputs
    let mut head_outputs: Vec<Tensor<f32>> = Vec::with_capacity(n_heads); // CHANGED
    for h in 0..n_heads { // CHANGED
        let q_h = split_head(q, n_heads, h)?; // CHANGED: [seq_q, d_k]
        let k_h = split_head(k, n_heads, h)?; // CHANGED: [seq_k, d_k]
        let v_h = split_head(v, n_heads, h)?; // CHANGED: [seq_k, d_v]
        // zero bias = no mask
        let bias = Tensor::from_vec( // CHANGED
            vec![0.0_f32; q_h.dims()[0] * k_h.dims()[0]],
            vec![q_h.dims()[0], k_h.dims()[0]],
        )?;
        let out_h = scaled_dot_product_attention_with_bias(&q_h, &k_h, &v_h, &bias)?; // CHANGED
        head_outputs.push(out_h); // CHANGED
    }
    concat_heads(&head_outputs) // CHANGED
}

/// Multi-head attention with causal mask (autoregressive / full-sequence).
///
/// Requires `seq_q == seq_k`. For the KV-cache decode step use
/// [`multi_head_attention_causal_with_offset`].
///
/// # Errors
///
/// Same as [`multi_head_attention`] plus [`TensorError::InvalidShape`] if
/// `seq_q != seq_k`.
#[must_use = "returns the causally-masked multi-head attention output"] // CHANGED
pub fn multi_head_attention_causal( // CHANGED
    q:       &Tensor<f32>,
    k:       &Tensor<f32>,
    v:       &Tensor<f32>,
    n_heads: usize,
) -> Result<Tensor<f32>> {
    let (seq_q, seq_k, _d_k, _d_v) = mha_check(q, k, v, n_heads)?; // CHANGED
    if seq_q != seq_k { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "multi_head_attention_causal: seq_q({seq_q}) != seq_k({seq_k}); \
                 use multi_head_attention_causal_with_offset for decode"
            ),
        });
    }
    let mask = causal_mask(seq_q)?; // CHANGED: shared across all heads

    let mut head_outputs: Vec<Tensor<f32>> = Vec::with_capacity(n_heads); // CHANGED
    for h in 0..n_heads { // CHANGED
        let q_h = split_head(q, n_heads, h)?; // CHANGED
        let k_h = split_head(k, n_heads, h)?; // CHANGED
        let v_h = split_head(v, n_heads, h)?; // CHANGED
        let out_h = scaled_dot_product_attention_with_bias(&q_h, &k_h, &v_h, &mask)?; // CHANGED
        head_outputs.push(out_h); // CHANGED
    }
    concat_heads(&head_outputs) // CHANGED
}

/// Multi-head attention with causal mask for the **KV-cache decode step**.
///
/// `start_pos` is the absolute position of the first query token.
///
/// # Errors
///
/// Same as [`multi_head_attention`]; see also [`causal_mask_with_offset`].
#[must_use = "returns the masked multi-head attention output"] // CHANGED
pub fn multi_head_attention_causal_with_offset( // CHANGED
    q:         &Tensor<f32>,
    k:         &Tensor<f32>,
    v:         &Tensor<f32>,
    n_heads:   usize,
    start_pos: usize,
) -> Result<Tensor<f32>> {
    let (seq_q, seq_k, _d_k, _d_v) = mha_check(q, k, v, n_heads)?; // CHANGED
    let mask = crate::attention::mask::causal_mask_with_offset(seq_q, seq_k, start_pos)?; // CHANGED

    let mut head_outputs: Vec<Tensor<f32>> = Vec::with_capacity(n_heads); // CHANGED
    for h in 0..n_heads { // CHANGED
        let q_h = split_head(q, n_heads, h)?; // CHANGED
        let k_h = split_head(k, n_heads, h)?; // CHANGED
        let v_h = split_head(v, n_heads, h)?; // CHANGED
        let out_h = scaled_dot_product_attention_with_bias(&q_h, &k_h, &v_h, &mask)?; // CHANGED
        head_outputs.push(out_h); // CHANGED
    }
    concat_heads(&head_outputs) // CHANGED
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::sdpa::scaled_dot_product_attention;

    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    // ── split_head ────────────────────────────────────────────────────────

    #[test]
    fn test_split_head_shape() { // CHANGED
        // [4, 6] with n_heads=3 → each head is [4, 2]
        let x = Tensor::from_vec(vec![0.0_f32; 24], vec![4, 6]).unwrap();
        let h = split_head(&x, 3, 0).unwrap();
        assert_eq!(h.dims(), &[4, 2]);
    }

    #[test]
    fn test_split_head_values() { // CHANGED
        // [2, 4], n_heads=2: head 0 = cols 0,1; head 1 = cols 2,3
        // row 0: [10,20,30,40], row 1: [50,60,70,80]
        let x = Tensor::from_vec(
            vec![10.0_f32, 20.0, 30.0, 40.0,
                 50.0,     60.0, 70.0, 80.0],
            vec![2, 4]).unwrap();
        let h0 = split_head(&x, 2, 0).unwrap();
        let h1 = split_head(&x, 2, 1).unwrap();
        assert!(close_slice(h0.as_slice(), &[10.0, 20.0, 50.0, 60.0], 1e-6));
        assert!(close_slice(h1.as_slice(), &[30.0, 40.0, 70.0, 80.0], 1e-6));
    }

    #[test]
    fn test_split_head_out_of_range() { // CHANGED
        let x = Tensor::from_vec(vec![0.0_f32; 8], vec![2, 4]).unwrap();
        assert!(matches!(split_head(&x, 2, 2), Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn test_split_head_not_divisible() { // CHANGED
        let x = Tensor::from_vec(vec![0.0_f32; 6], vec![2, 3]).unwrap();
        assert!(matches!(split_head(&x, 2, 0), Err(TensorError::InvalidShape { .. })));
    }

    // ── concat_heads ──────────────────────────────────────────────────────

    #[test]
    fn test_concat_heads_shape() { // CHANGED
        let h0 = Tensor::from_vec(vec![0.0_f32; 8], vec![4, 2]).unwrap();
        let h1 = Tensor::from_vec(vec![0.0_f32; 8], vec![4, 2]).unwrap();
        let h2 = Tensor::from_vec(vec![0.0_f32; 8], vec![4, 2]).unwrap();
        let out = concat_heads(&[h0, h1, h2]).unwrap();
        assert_eq!(out.dims(), &[4, 6]);
    }

    #[test]
    fn test_concat_heads_roundtrip() { // CHANGED
        // split then concat should recover the original tensor
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x  = Tensor::from_vec(data.clone(), vec![2, 8]).unwrap();
        let h0 = split_head(&x, 4, 0).unwrap();
        let h1 = split_head(&x, 4, 1).unwrap();
        let h2 = split_head(&x, 4, 2).unwrap();
        let h3 = split_head(&x, 4, 3).unwrap();
        let out = concat_heads(&[h0, h1, h2, h3]).unwrap();
        assert!(close_slice(out.as_slice(), &data, 1e-7),
            "roundtrip failed: {:?}", out.as_slice());
    }

    #[test]
    fn test_concat_heads_empty_rejected() { // CHANGED
        assert!(matches!(concat_heads(&[]), Err(TensorError::InvalidShape { .. })));
    }

    // ── multi_head_attention output shape ────────────────────────────────

    #[test]
    fn test_mha_output_shape_1_head() { // CHANGED
        // With 1 head MHA == SDPA
        let q = Tensor::from_vec(vec![0.0_f32; 12], vec![3, 4]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 12], vec![3, 4]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; 12], vec![3, 4]).unwrap();
        let out = multi_head_attention(&q, &k, &v, 1).unwrap();
        assert_eq!(out.dims(), &[3, 4]);
    }

    #[test]
    fn test_mha_output_shape_4_heads() { // CHANGED
        // Q/K: [5, 8] with 4 heads → d_k=2; V: [5, 8] → d_v=2; out: [5, 8]
        let q = Tensor::from_vec(vec![0.0_f32; 40], vec![5, 8]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 40], vec![5, 8]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; 40], vec![5, 8]).unwrap();
        let out = multi_head_attention(&q, &k, &v, 4).unwrap();
        assert_eq!(out.dims(), &[5, 8]);
    }

    // ── 1-head MHA must match plain SDPA exactly ─────────────────────────

    #[test]
    fn test_mha_1head_matches_sdpa() { // CHANGED
        let data_q = vec![1.0_f32, 0.0, 0.5, 0.5];
        let data_k = vec![0.3_f32, 0.7, 0.9, 0.1];
        let data_v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let q = Tensor::from_vec(data_q.clone(), vec![2, 2]).unwrap();
        let k = Tensor::from_vec(data_k.clone(), vec![2, 2]).unwrap();
        let v = Tensor::from_vec(data_v.clone(), vec![2, 2]).unwrap();

        let mha_out  = multi_head_attention(&q, &k, &v, 1).unwrap();
        let sdpa_out = scaled_dot_product_attention(&q, &k, &v).unwrap();

        assert!(close_slice(mha_out.as_slice(), sdpa_out.as_slice(), 1e-6),
            "1-head MHA differs from SDPA:\n  mha={:?}\n  sdpa={:?}",
            mha_out.as_slice(), sdpa_out.as_slice());
    }

    // ── multi-head: each head operates independently ──────────────────────

    #[test]
    fn test_mha_heads_independent() { // CHANGED
        // Build Q, K, V such that head 0 gets all-zero inputs and head 1 gets
        // distinct inputs.  The output for head 0 should equal uniform-attention
        // on V head 0, regardless of what head 1 does.
        //
        // seq=2, n_heads=2, d_k=2, d_v=2
        // Q: [2, 4]  head0 cols=[0,1], head1 cols=[2,3]
        // Set head0 of Q and K to 0 (uniform attn), head1 to identity-like values
        let q = Tensor::from_vec(vec![
            0.0_f32, 0.0, 1.0, 0.0, // token 0: head0=[0,0], head1=[1,0]
            0.0,     0.0, 0.0, 1.0, // token 1: head0=[0,0], head1=[0,1]
        ], vec![2, 4]).unwrap();
        let k = q.clone();
        let v = Tensor::from_vec(vec![
            10.0_f32, 20.0, 1.0, 2.0, // head0 values=[10,20], head1 values=[1,2]
            30.0,     40.0, 3.0, 4.0,
        ], vec![2, 4]).unwrap();

        let out = multi_head_attention(&q, &k, &v, 2).unwrap();
        assert_eq!(out.dims(), &[2, 4]);

        // Head 0: uniform attn over 2 tokens → mean([10,20],[30,40]) = [20,30]
        // Both output rows for head 0 should be [20, 30]
        let row0_h0 = [out.as_slice()[0], out.as_slice()[1]];
        let row1_h0 = [out.as_slice()[4], out.as_slice()[5]];
        assert!(close_slice(&row0_h0, &[20.0, 30.0], 1e-4),
            "head0 row0 = {:?}", row0_h0);
        assert!(close_slice(&row1_h0, &[20.0, 30.0], 1e-4),
            "head0 row1 = {:?}", row1_h0);
    }

    // ── causal MHA: token 0 attends only to itself ────────────────────────

    #[test]
    fn test_mha_causal_first_token_self_only() { // CHANGED
        // Same premise as mask tests: with 2 heads × 2 tokens,
        // token 0 may only attend to itself in the causal variant.
        let q = Tensor::from_vec(vec![
            1.0_f32, 0.0,  0.0, 1.0, // token 0
            0.0,     1.0,  1.0, 0.0, // token 1
        ], vec![2, 4]).unwrap();
        let k = q.clone();
        // V head0: row0=[1,2], row1=[3,4];  V head1: row0=[5,6], row1=[7,8]
        let v = Tensor::from_vec(vec![
            1.0_f32, 2.0, 5.0, 6.0,
            3.0,     4.0, 7.0, 8.0,
        ], vec![2, 4]).unwrap();

        let out = multi_head_attention_causal(&q, &k, &v, 2).unwrap();
        assert_eq!(out.dims(), &[2, 4]);

        // Token 0 must equal V[0] for both heads: [1,2] and [5,6]
        let tok0 = &out.as_slice()[0..4];
        assert!((tok0[0] - 1.0).abs() < 1e-4, "tok0 head0[0]={}", tok0[0]);
        assert!((tok0[1] - 2.0).abs() < 1e-4, "tok0 head0[1]={}", tok0[1]);
        assert!((tok0[2] - 5.0).abs() < 1e-4, "tok0 head1[0]={}", tok0[2]);
        assert!((tok0[3] - 6.0).abs() < 1e-4, "tok0 head1[1]={}", tok0[3]);
    }

    // ── causal with offset (decode step) ─────────────────────────────────

    #[test]
    fn test_mha_causal_offset_output_shape() { // CHANGED
        // decode: q=[1, 4], k=[3, 4], v=[3, 4], n_heads=2, start_pos=2
        let q = Tensor::from_vec(vec![0.0_f32; 4],  vec![1, 4]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 12], vec![3, 4]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; 12], vec![3, 4]).unwrap();
        let out = multi_head_attention_causal_with_offset(&q, &k, &v, 2, 2).unwrap();
        assert_eq!(out.dims(), &[1, 4]);
    }

    // ── error handling ────────────────────────────────────────────────────

    #[test]
    fn test_mha_zero_heads_rejected() { // CHANGED
        let q = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let k = q.clone();
        let v = q.clone();
        assert!(matches!(multi_head_attention(&q, &k, &v, 0), Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn test_mha_heads_not_divisible_rejected() { // CHANGED
        let q = Tensor::from_vec(vec![1.0_f32; 6], vec![2, 3]).unwrap();
        let k = q.clone();
        let v = q.clone();
        assert!(matches!(multi_head_attention(&q, &k, &v, 2), Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn test_mha_causal_seq_mismatch_rejected() { // CHANGED
        let q = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let k = Tensor::from_vec(vec![1.0_f32; 6], vec![3, 2]).unwrap();
        let v = Tensor::from_vec(vec![1.0_f32; 6], vec![3, 2]).unwrap();
        assert!(matches!(
            multi_head_attention_causal(&q, &k, &v, 1),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
