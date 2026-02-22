//! Sliding window attention mask — commit 12.3.
//!
//! # Motivation
//!
//! Standard causal attention has O(N) memory and compute cost per decode step:
//! each new token attends over the full accumulated context of length N.
//! For very long sequences this means the attention score matrix grows without
//! bound, eventually dominating both latency and memory.
//!
//! Sliding window attention (Beltagy et al. 2020; used in Mistral 7B) bounds
//! this cost by restricting each token to attend only to the W most recent
//! tokens (plus itself).  The KV-cache working set is capped at W rows per
//! layer regardless of total sequence length, giving O(1) per-step memory.
//!
//! # Mask shape
//!
//! For a token at absolute position `abs_pos = start_pos + i`, the allowed
//! attention window is:
//!
//! ```text
//! [max(0, abs_pos − window_size + 1) ..= abs_pos]   (inclusive)
//! ```
//!
//! All keys outside this window receive `−∞` so they contribute zero weight
//! after softmax.  Future keys (`j > abs_pos`) are also masked — the sliding
//! window mask is a *strict superset* of the causal mask.
//!
//! # API
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`sliding_window_mask`] | Build an additive `[seq_q, seq_k]` bias tensor |
//! | [`sliding_window_sdpa`] | SDPA + sliding window mask (single-head) |
//! | [`sliding_window_gqa`]  | GQA + sliding window mask (Llama/Mistral style) |
//!
//! # Relationship to causal mask
//!
//! `sliding_window_mask(seq_q, seq_k, start_pos, window_size = usize::MAX)`
//! is numerically identical to `causal_mask_with_offset(seq_q, seq_k,
//! start_pos)` — an infinite window degenerates to standard causal attention.
//! This is verified in [`tests::test_infinite_window_matches_causal_mask`].

use crate::attention::sdpa::scaled_dot_product_attention_with_bias;
use crate::tensor::{Result, Tensor, TensorError};

// ── mask constructor ──────────────────────────────────────────────────────────

/// Build a sliding-window additive mask of shape `[seq_q, seq_k]`.
///
/// Entry `[i, j]` is:
/// - `0.0`                if `j` is within the window and not in the future
/// - `f32::NEG_INFINITY`  otherwise
///
/// The window for query token `i` (absolute position `start_pos + i`) covers
/// key positions in `[max(0, abs_pos − window_size + 1) ..= abs_pos]`.
///
/// # Arguments
///
/// * `seq_q`       – number of query tokens
/// * `seq_k`       – total number of key/value positions in the cache
/// * `start_pos`   – absolute sequence position of the first query token
/// * `window_size` – maximum number of past tokens to attend to (including
///                   self).  Pass `usize::MAX` for full causal attention.
///
/// # Errors
///
/// [`TensorError::InvalidShape`] if `seq_q == 0`, `seq_k == 0`, or
/// `start_pos + seq_q > seq_k`.
#[must_use = "returns a new mask tensor"]
pub fn sliding_window_mask(
    seq_q:       usize,
    seq_k:       usize,
    start_pos:   usize,
    window_size: usize,
) -> Result<Tensor<f32>> {
    if seq_q == 0 {
        return Err(TensorError::InvalidShape {
            reason: "sliding_window_mask: seq_q must be > 0".to_string(),
        });
    }
    if seq_k == 0 {
        return Err(TensorError::InvalidShape {
            reason: "sliding_window_mask: seq_k must be > 0".to_string(),
        });
    }
    if window_size == 0 {
        return Err(TensorError::InvalidShape {
            reason: "sliding_window_mask: window_size must be > 0".to_string(),
        });
    }
    if start_pos + seq_q > seq_k {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "sliding_window_mask: start_pos({start_pos}) + seq_q({seq_q}) \
                 > seq_k({seq_k})"
            ),
        });
    }

    let mut data = vec![f32::NEG_INFINITY; seq_q * seq_k];

    for i in 0..seq_q {
        let abs_pos    = start_pos + i;
        // window_left is the earliest key position this query may attend to
        let window_left = abs_pos.saturating_sub(window_size - 1);
        for j in window_left..=abs_pos {
            if j < seq_k {
                data[i * seq_k + j] = 0.0;
            }
        }
    }

    Tensor::from_vec(data, vec![seq_q, seq_k])
}

// ── convenience wrappers ──────────────────────────────────────────────────────

/// Single-head SDPA with sliding window causal mask.
///
/// Equivalent to calling [`sliding_window_mask`] then
/// [`scaled_dot_product_attention_with_bias`].
///
/// # Arguments
///
/// * `q`, `k`, `v`  – 2-D tensors `[seq_q, d]`, `[seq_k, d]`, `[seq_k, d_v]`
/// * `start_pos`    – absolute position of the first query token
/// * `window_size`  – sliding window width (tokens)
///
/// # Errors
///
/// [`TensorError::InvalidShape`] on dimension mismatches or an empty window.
#[must_use = "returns the masked attention output"]
pub fn sliding_window_sdpa(
    q:           &Tensor<f32>,
    k:           &Tensor<f32>,
    v:           &Tensor<f32>,
    start_pos:   usize,
    window_size: usize,
) -> Result<Tensor<f32>> {
    if q.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!("sliding_window_sdpa: q must be 2-D, got {}D", q.ndim()),
        });
    }
    if k.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!("sliding_window_sdpa: k must be 2-D, got {}D", k.ndim()),
        });
    }
    let seq_q = q.dims()[0];
    let seq_k = k.dims()[0];
    let mask  = sliding_window_mask(seq_q, seq_k, start_pos, window_size)?;
    scaled_dot_product_attention_with_bias(q, k, v, &mask)
}

/// Grouped-query attention with sliding window causal mask.
///
/// This is the primary entry point for Mistral-style inference.
/// The mask restricts each query to the `window_size` most recent key
/// positions; `grouped_query_attention_causal_with_offset` is used underneath
/// for multi-head grouping, with the sliding window mask injected in place of
/// the standard causal mask via `scaled_dot_product_attention_with_bias`.
///
/// # Arguments
///
/// * `q`           – `[seq_q, n_heads * head_dim]`
/// * `k`           – `[seq_k, n_kv_heads * head_dim]`
/// * `v`           – `[seq_k, n_kv_heads * head_dim]`
/// * `n_heads`     – total query head count
/// * `n_kv_heads`  – KV head count (`n_heads % n_kv_heads == 0`)
/// * `start_pos`   – absolute position of the first query token
/// * `window_size` – sliding window width
///
/// # Errors
///
/// [`TensorError::InvalidShape`] on dimension mismatches or empty inputs.
///
/// # Note on `start_pos` for full-window queries
///
/// During prefill with `start_pos = 0` and `window_size ≥ seq_len`, the
/// sliding window degenerates to standard causal attention — every token
/// can attend to all prior tokens.
#[must_use = "returns the masked attention output"]
pub fn sliding_window_gqa(
    q:           &Tensor<f32>,
    k:           &Tensor<f32>,
    v:           &Tensor<f32>,
    n_heads:     usize,
    n_kv_heads:  usize,
    start_pos:   usize,
    window_size: usize,
) -> Result<Tensor<f32>> {
    if q.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!("sliding_window_gqa: q must be 2-D, got {}D", q.ndim()),
        });
    }
    if k.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!("sliding_window_gqa: k must be 2-D, got {}D", k.ndim()),
        });
    }
    if n_heads == 0 || n_kv_heads == 0 {
        return Err(TensorError::InvalidShape {
            reason: "sliding_window_gqa: n_heads and n_kv_heads must be > 0".to_string(),
        });
    }
    if n_heads % n_kv_heads != 0 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "sliding_window_gqa: n_heads({n_heads}) not divisible by n_kv_heads({n_kv_heads})"
            ),
        });
    }

    let seq_q  = q.dims()[0];
    let seq_k  = k.dims()[0];
    let groups = n_heads / n_kv_heads;

    // Build sliding window mask [seq_q, seq_k] — shared across all heads
    let mask = sliding_window_mask(seq_q, seq_k, start_pos, window_size)?;

    // Use split_head(x, n_heads, h) to extract one head at a time
    use crate::attention::multihead::{split_head, concat_heads};

    let mut out_heads: Vec<Tensor<f32>> = Vec::with_capacity(n_heads);
    for h in 0..n_heads {
        let kv_h = h / groups;
        let q_h  = split_head(q, n_heads,    h)?;    // [seq_q, head_dim]
        let k_h  = split_head(k, n_kv_heads, kv_h)?; // [seq_k, kv_d]
        let v_h  = split_head(v, n_kv_heads, kv_h)?; // [seq_k, kv_d]
        let out_h = scaled_dot_product_attention_with_bias(&q_h, &k_h, &v_h, &mask)?;
        out_heads.push(out_h);
    }

    // concat_heads([seq_q, head_dim] × n_heads) → [seq_q, n_heads * head_dim]
    concat_heads(&out_heads)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::mask::causal_mask_with_offset;

    const EPS: f32 = 1e-5;
    fn close(a: f32, b: f32) -> bool { (a - b).abs() < EPS }

    // ── mask shape and values ─────────────────────────────────────────────

    #[test]
    fn test_mask_shape() {
        let m = sliding_window_mask(2, 5, 2, 3).unwrap();
        assert_eq!(m.dims(), &[2, 5]);
    }

    #[test]
    fn test_mask_single_token_window_one() {
        // Window=1: a token at pos 3 may only attend to itself (key 3).
        let m = sliding_window_mask(1, 4, 3, 1).unwrap();
        let s = m.as_slice();
        assert!(s[0].is_infinite() && s[0] < 0.0, "key 0 should be masked");
        assert!(s[1].is_infinite() && s[1] < 0.0, "key 1 should be masked");
        assert!(s[2].is_infinite() && s[2] < 0.0, "key 2 should be masked");
        assert!(close(s[3], 0.0),                 "key 3 (self) should be 0.0");
    }

    #[test]
    fn test_mask_window_larger_than_context() {
        // Window > pos: token at pos=2 with window=10 can see all keys [0,1,2].
        let m = sliding_window_mask(1, 3, 2, 10).unwrap();
        let s = m.as_slice();
        assert!(close(s[0], 0.0), "key 0 allowed");
        assert!(close(s[1], 0.0), "key 1 allowed");
        assert!(close(s[2], 0.0), "key 2 (self) allowed");
    }

    #[test]
    fn test_mask_window_exactly_at_boundary() {
        // Token at pos=3, window=2: can see keys [2, 3], not [0, 1].
        let m = sliding_window_mask(1, 4, 3, 2).unwrap();
        let s = m.as_slice();
        assert!(s[0].is_infinite() && s[0] < 0.0, "key 0 outside window");
        assert!(s[1].is_infinite() && s[1] < 0.0, "key 1 outside window");
        assert!(close(s[2], 0.0), "key 2 in window");
        assert!(close(s[3], 0.0), "key 3 in window (self)");
    }

    #[test]
    fn test_mask_future_keys_always_masked() {
        // Even with a large window, future keys must always be masked.
        // seq_q=2, seq_k=4, start_pos=0, window=100
        // row 0 (abs pos 0): can see key 0 only (future=keys 1,2,3 masked)
        // row 1 (abs pos 1): can see keys 0,1 (future=keys 2,3 masked)
        let m = sliding_window_mask(2, 4, 0, 100).unwrap();
        let s = m.as_slice();
        // row 0
        assert!(close(s[0], 0.0));
        assert!(s[1].is_infinite() && s[1] < 0.0, "future key 1 must be masked");
        assert!(s[2].is_infinite() && s[2] < 0.0, "future key 2 must be masked");
        assert!(s[3].is_infinite() && s[3] < 0.0, "future key 3 must be masked");
        // row 1
        assert!(close(s[4], 0.0));
        assert!(close(s[5], 0.0));
        assert!(s[6].is_infinite() && s[6] < 0.0, "future key 2 must be masked");
        assert!(s[7].is_infinite() && s[7] < 0.0, "future key 3 must be masked");
    }

    #[test]
    fn test_mask_multi_query_sliding() {
        // seq_q=3, seq_k=6, start_pos=3, window=2
        // row 0 (abs 3): window [2,3] → keys 2,3 open; 0,1,4,5 masked
        // row 1 (abs 4): window [3,4] → keys 3,4 open; 0,1,2,5 masked
        // row 2 (abs 5): window [4,5] → keys 4,5 open; 0,1,2,3 masked
        let m = sliding_window_mask(3, 6, 3, 2).unwrap();
        let s = m.as_slice();
        // row 0
        assert!(s[0].is_infinite()); assert!(s[1].is_infinite());
        assert!(close(s[2], 0.0));   assert!(close(s[3], 0.0));
        assert!(s[4].is_infinite()); assert!(s[5].is_infinite());
        // row 1
        assert!(s[6].is_infinite()); assert!(s[7].is_infinite()); assert!(s[8].is_infinite());
        assert!(close(s[9], 0.0));   assert!(close(s[10], 0.0));
        assert!(s[11].is_infinite());
        // row 2
        for j in 0..4 { assert!(s[12+j].is_infinite(), "key {j} should be masked"); }
        assert!(close(s[16], 0.0));  assert!(close(s[17], 0.0));
    }

    /// An infinite window must produce a mask identical to causal_mask_with_offset.
    #[test]
    fn test_infinite_window_matches_causal_mask() {
        let (seq_q, seq_k, start_pos) = (3, 6, 3);
        let causal  = causal_mask_with_offset(seq_q, seq_k, start_pos).unwrap();
        let sliding = sliding_window_mask(seq_q, seq_k, start_pos, usize::MAX).unwrap();
        for (i, (c, w)) in causal.as_slice().iter().zip(sliding.as_slice()).enumerate() {
            assert_eq!(c.is_infinite(), w.is_infinite(),
                "element {i}: causal={c} sliding={w}");
            if !c.is_infinite() { assert!(close(*c, *w)); }
        }
    }

    // ── error cases ───────────────────────────────────────────────────────

    #[test]
    fn test_zero_seq_q_rejected() {
        assert!(matches!(
            sliding_window_mask(0, 4, 0, 2),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_zero_seq_k_rejected() {
        assert!(matches!(
            sliding_window_mask(1, 0, 0, 2),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_zero_window_size_rejected() {
        assert!(matches!(
            sliding_window_mask(1, 4, 0, 0),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_query_exceeds_keys_rejected() {
        // start_pos=3 + seq_q=2 = 5 > seq_k=4
        assert!(matches!(
            sliding_window_mask(2, 4, 3, 2),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    // ── sliding_window_sdpa ───────────────────────────────────────────────

    #[test]
    fn test_sdpa_output_shape() {
        let q = Tensor::from_vec(vec![0.0_f32; 3 * 4], vec![3, 4]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 5 * 4], vec![5, 4]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; 5 * 4], vec![5, 4]).unwrap();
        let out = sliding_window_sdpa(&q, &k, &v, 2, 2).unwrap();
        assert_eq!(out.dims(), &[3, 4]);
    }

    #[test]
    fn test_sdpa_window_blocks_old_tokens() {
        // Token at pos=2, window=1: must see only V[2], not V[0] or V[1].
        // V[2]=[10,10], V[0]=V[1]=[99,99]. If window works, output ≈ [10,10].
        let q = Tensor::from_vec(vec![0.0_f32; 2], vec![1, 2]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 6], vec![3, 2]).unwrap();
        let v = Tensor::from_vec(vec![
            99.0_f32, 99.0,  // pos 0 — should be masked
            99.0,     99.0,  // pos 1 — should be masked
            10.0,     10.0,  // pos 2 — in window
        ], vec![3, 2]).unwrap();
        let out = sliding_window_sdpa(&q, &k, &v, 2, 1).unwrap();
        assert!((out.as_slice()[0] - 10.0).abs() < 1e-4,
            "got {} instead of 10.0 — old token leaked", out.as_slice()[0]);
    }

    #[test]
    fn test_sdpa_infinite_window_matches_causal() {
        // With window=usize::MAX, sliding SDPA must agree with standard causal.
        use crate::attention::mask::masked_sdpa_with_offset;
        let q = Tensor::from_vec((0..8).map(|i| i as f32 * 0.1).collect(), vec![2, 4]).unwrap();
        let k = Tensor::from_vec((0..16).map(|i| i as f32 * 0.05).collect(), vec![4, 4]).unwrap();
        let v = Tensor::from_vec((0..16).map(|i| (16-i) as f32 * 0.1).collect(), vec![4, 4]).unwrap();
        let sliding = sliding_window_sdpa(&q, &k, &v, 2, usize::MAX).unwrap();
        let causal  = masked_sdpa_with_offset(&q, &k, &v, 2).unwrap();
        for (i, (s, c)) in sliding.as_slice().iter().zip(causal.as_slice()).enumerate() {
            assert!((s - c).abs() < 1e-5, "element {i}: sliding={s} causal={c}");
        }
    }

    // ── sliding_window_gqa ────────────────────────────────────────────────

    #[test]
    fn test_gqa_output_shape() {
        let (seq_q, seq_k) = (2, 5);
        let (n_heads, n_kv, head_dim) = (4, 2, 8);
        let q = Tensor::from_vec(vec![0.1_f32; seq_q * n_heads * head_dim],
            vec![seq_q, n_heads * head_dim]).unwrap();
        let k = Tensor::from_vec(vec![0.1_f32; seq_k * n_kv * head_dim],
            vec![seq_k, n_kv * head_dim]).unwrap();
        let v = Tensor::from_vec(vec![0.1_f32; seq_k * n_kv * head_dim],
            vec![seq_k, n_kv * head_dim]).unwrap();
        let out = sliding_window_gqa(&q, &k, &v, n_heads, n_kv, 3, 3).unwrap();
        assert_eq!(out.dims(), &[seq_q, n_heads * head_dim]);
    }

    #[test]
    fn test_gqa_no_nan() {
        let (seq_q, seq_k) = (3, 6);
        let (n_heads, n_kv, head_dim) = (2, 2, 4);
        let q = Tensor::from_vec(
            (0..seq_q * n_heads * head_dim).map(|i| i as f32 * 0.01).collect(),
            vec![seq_q, n_heads * head_dim]).unwrap();
        let k = Tensor::from_vec(
            (0..seq_k * n_kv * head_dim).map(|i| i as f32 * 0.02).collect(),
            vec![seq_k, n_kv * head_dim]).unwrap();
        let v = k.clone();
        let out = sliding_window_gqa(&q, &k, &v, n_heads, n_kv, 3, 2).unwrap();
        for &x in out.as_slice() {
            assert!(!x.is_nan() && !x.is_infinite(), "non-finite in sliding GQA output");
        }
    }

    /// With infinite window size, sliding_window_gqa must agree with
    /// grouped_query_attention_causal_with_offset.
    #[test]
    fn test_gqa_infinite_window_matches_standard_gqa() {
        use crate::attention::gqa::grouped_query_attention_causal_with_offset;
        let (seq_q, seq_k) = (2, 4);
        let (n_heads, n_kv, head_dim) = (4, 2, 4);
        let q = Tensor::from_vec(
            (0..seq_q * n_heads * head_dim).map(|i| i as f32 * 0.05).collect(),
            vec![seq_q, n_heads * head_dim]).unwrap();
        let k = Tensor::from_vec(
            (0..seq_k * n_kv * head_dim).map(|i| i as f32 * 0.03).collect(),
            vec![seq_k, n_kv * head_dim]).unwrap();
        let v = Tensor::from_vec(
            (0..seq_k * n_kv * head_dim).map(|i| (seq_k*n_kv*head_dim - i) as f32 * 0.02).collect(),
            vec![seq_k, n_kv * head_dim]).unwrap();
        let start_pos = 2_usize;
        let standard = grouped_query_attention_causal_with_offset(
            &q, &k, &v, n_heads, n_kv, start_pos).unwrap();
        let sliding  = sliding_window_gqa(
            &q, &k, &v, n_heads, n_kv, start_pos, usize::MAX).unwrap();
        for (i, (s, g)) in sliding.as_slice().iter().zip(standard.as_slice()).enumerate() {
            assert!((s - g).abs() < 1e-4, "element {i}: sliding={s} gqa={g}");
        }
    }

    #[test]
    fn test_gqa_window_limits_context() {
        // With window=1, each query token sees only itself.
        // All-zero Q/K → uniform attention, but only over the single allowed key.
        // V row at self-position = [pos as f32; head_dim].
        // seq_q=2, seq_k=4, start_pos=2, window=1
        // query 0 (abs 2) sees only key 2:  V[2] = [2.0; 4]
        // query 1 (abs 3) sees only key 3:  V[3] = [3.0; 4]
        let (seq_q, seq_k) = (2, 4);
        let (n_heads, n_kv, head_dim) = (2, 2, 4);
        let q = Tensor::from_vec(vec![0.0_f32; seq_q * n_heads * head_dim],
            vec![seq_q, n_heads * head_dim]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; seq_k * n_kv * head_dim],
            vec![seq_k, n_kv * head_dim]).unwrap();
        // V[i, :] = [i as f32; n_kv * head_dim]
        let v_data: Vec<f32> = (0..seq_k).flat_map(|i|
            vec![i as f32; n_kv * head_dim]
        ).collect();
        let v = Tensor::from_vec(v_data, vec![seq_k, n_kv * head_dim]).unwrap();
        let out = sliding_window_gqa(&q, &k, &v, n_heads, n_kv, 2, 1).unwrap();
        // Each head in query row 0 must be ≈ 2.0; query row 1 must be ≈ 3.0
        let row0 = &out.as_slice()[..n_heads * head_dim];
        let row1 = &out.as_slice()[n_heads * head_dim..];
        for &v in row0 { assert!((v - 2.0).abs() < 1e-4, "row0: got {v}, expected 2.0"); }
        for &v in row1 { assert!((v - 3.0).abs() < 1e-4, "row1: got {v}, expected 3.0"); }
    }

    #[test]
    fn test_gqa_zero_kv_heads_rejected() {
        let q = Tensor::from_vec(vec![0.0_f32; 8], vec![1, 8]).unwrap();
        let k = q.clone(); let v = q.clone();
        assert!(matches!(
            sliding_window_gqa(&q, &k, &v, 2, 0, 0, 4),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_gqa_indivisible_heads_rejected() {
        let q = Tensor::from_vec(vec![0.0_f32; 8], vec![1, 8]).unwrap();
        let k = q.clone(); let v = q.clone();
        assert!(matches!(
            sliding_window_gqa(&q, &k, &v, 3, 2, 0, 4),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
