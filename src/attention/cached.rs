//! Cached attention — commit 9.2.
//!
//! # Prefill vs Decode
//!
//! LLM generation has two distinct computation phases:
//!
//! | Phase     | Input shape      | K/V written to cache        | Attention over |
//! |-----------|------------------|-----------------------------|----------------|
//! | Prefill   | `[seq_len, d]`   | all `seq_len` rows at once  | full prompt     |
//! | Decode    | `[1, d]`         | single row at `pos`         | all 0..=pos     |
//!
//! Both paths use [`crate::attention::gqa::grouped_query_attention_causal_with_offset`]
//! for the actual attention computation; they differ only in **how the K/V cache is
//! populated** and **what K/V slice is passed to the attention kernel**.
//!
//! # Correctness guarantee
//!
//! `cached_attention_prefill` with `start_pos = 0` produces the same output as
//! calling `grouped_query_attention_causal` directly on the same Q/K/V.

use crate::cache::KvCache;
use crate::tensor::{Result, Tensor, TensorError};
use crate::attention::gqa::grouped_query_attention_causal_with_offset;

// ── prefill ───────────────────────────────────────────────────────────────────

/// Prefill path: write K/V for the full input sequence to cache, then run
/// causally-masked GQA over the full context `[0 .. start_pos + seq_len)`.
///
/// # Arguments
///
/// * `cache`      – mutable KV-cache; will be written at positions
///                  `[start_pos .. start_pos + seq_len)`.
/// * `layer`      – which transformer layer's cache slot to write.
/// * `start_pos`  – cache offset for the first token in `q` (0 for the first
///                  prefill of a new sequence).
/// * `q`          – `[seq_len, n_heads * d_k]`
/// * `k`          – `[seq_len, n_kv_heads * d_k]` — must already have RoPE applied.
/// * `v`          – `[seq_len, n_kv_heads * d_v]`
/// * `n_heads`    – total query head count.
/// * `n_kv_heads` – KV head count (`n_heads % n_kv_heads == 0`).
///
/// # Returns
///
/// Attention output `[seq_len, n_heads * d_v]`.
///
/// # Errors
///
/// [`TensorError::InvalidShape`] if any dimension is inconsistent, `layer` is
/// out of range, or positions overflow `max_seq_len`.
#[must_use = "returns the prefill attention output"]
pub fn cached_attention_prefill(
    cache:      &mut KvCache,
    layer:      usize,
    start_pos:  usize,
    q:          &Tensor<f32>,
    k:          &Tensor<f32>,
    v:          &Tensor<f32>,
    n_heads:    usize,
    n_kv_heads: usize,
) -> Result<Tensor<f32>> {
    if q.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!("cached_attention_prefill: q must be 2-D, got {}D", q.ndim()),
        });
    }
    let seq_len = q.dims()[0];
    let kv_dim  = cache.kv_dim();

    // Ensure K and V are contiguous before slicing.
    let k_cont;
    let v_cont;
    let k_slice = if k.is_contiguous() {
        k.as_slice()
    } else {
        k_cont = k.contiguous();
        k_cont.as_slice()
    };
    let v_slice = if v.is_contiguous() {
        v.as_slice()
    } else {
        v_cont = v.contiguous();
        v_cont.as_slice()
    };

    // Validate K/V width matches cache kv_dim.
    if k.dims().get(1).copied().unwrap_or(0) != kv_dim {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "cached_attention_prefill: k dim[1]={} != cache kv_dim={kv_dim}",
                k.dims().get(1).copied().unwrap_or(0),
            ),
        });
    }

    // Write each K/V row to the cache.
    for i in 0..seq_len {
        cache.write_k(layer, start_pos + i, &k_slice[i * kv_dim..(i + 1) * kv_dim])?;
        cache.write_v(layer, start_pos + i, &v_slice[i * kv_dim..(i + 1) * kv_dim])?;
    }

    // Read back full K/V context (includes any prefix from start_pos > 0).
    let total_len = start_pos + seq_len;
    let k_full = cache.read_k(layer, total_len)?;
    let v_full = cache.read_v(layer, total_len)?;

    // Causal GQA with start_pos offset so prefix tokens are not masked.
    grouped_query_attention_causal_with_offset(q, &k_full, &v_full, n_heads, n_kv_heads, start_pos)
}

// ── decode ────────────────────────────────────────────────────────────────────

/// Decode path: append one new K/V row to the cache, then run GQA over the
/// full accumulated context `[0 .. pos + 1]`.
///
/// This is the hot path called once per generated token after prefill.
/// The K/V tensors here represent **only the new token** (`[1, kv_dim]`); the
/// full history is assembled by reading back from the cache.
///
/// # Arguments
///
/// * `cache`      – mutable KV-cache; will be written at position `pos`.
/// * `layer`      – transformer layer index.
/// * `pos`        – absolute sequence position of the new token (0-based;
///                  equals the number of tokens already in the cache).
/// * `q`          – `[1, n_heads * d_k]`
/// * `k`          – `[1, n_kv_heads * d_k]` — RoPE already applied.
/// * `v`          – `[1, n_kv_heads * d_v]`
/// * `n_heads`    – total query head count.
/// * `n_kv_heads` – KV head count.
///
/// # Returns
///
/// Attention output `[1, n_heads * d_v]`.
///
/// # Errors
///
/// [`TensorError::InvalidShape`] if `pos >= max_seq_len`, dimensions mismatch,
/// or `layer` is out of range.
#[must_use = "returns the decode attention output"]
pub fn cached_attention_decode(
    cache:      &mut KvCache,
    layer:      usize,
    pos:        usize,
    q:          &Tensor<f32>,
    k:          &Tensor<f32>,
    v:          &Tensor<f32>,
    n_heads:    usize,
    n_kv_heads: usize,
) -> Result<Tensor<f32>> {
    if q.ndim() != 2 || q.dims()[0] != 1 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "cached_attention_decode: q must be [1, d], got {:?}",
                q.dims()
            ),
        });
    }
    let kv_dim = cache.kv_dim();

    let k_cont;
    let v_cont;
    let k_slice = if k.is_contiguous() {
        k.as_slice()
    } else {
        k_cont = k.contiguous();
        k_cont.as_slice()
    };
    let v_slice = if v.is_contiguous() {
        v.as_slice()
    } else {
        v_cont = v.contiguous();
        v_cont.as_slice()
    };

    // Write the new token's K/V at position `pos`.
    cache.write_k(layer, pos, &k_slice[..kv_dim])?;
    cache.write_v(layer, pos, &v_slice[..kv_dim])?;

    // Read back full context: positions 0..=pos → [pos+1, kv_dim].
    let seq_len = pos + 1;
    let k_full  = cache.read_k(layer, seq_len)?;
    let v_full  = cache.read_v(layer, seq_len)?;

    // GQA with offset: query is at absolute position `pos`, so it may attend
    // to all keys at positions 0..=pos (causal_mask_with_offset allows this).
    grouped_query_attention_causal_with_offset(q, &k_full, &v_full, n_heads, n_kv_heads, pos)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;
    use crate::attention::gqa::grouped_query_attention_causal;

    fn close(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }
    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y, tol))
    }

    // ── prefill matches uncached GQA ──────────────────────────────────────

    #[test]
    fn test_prefill_matches_uncached_gqa_mha() {
        // With start_pos=0, cached_attention_prefill must match
        // grouped_query_attention_causal exactly (n_heads == n_kv_heads).
        let n_heads = 2_usize;
        let n_kv_heads = 2_usize;
        let head_dim = 4_usize;
        let seq = 3_usize;
        let n_heads_total = n_heads * head_dim;
        let kv_total = n_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..seq * n_heads_total)
            .map(|i| i as f32 * 0.1).collect();
        let k_data: Vec<f32> = (0..seq * kv_total)
            .map(|i| i as f32 * 0.05).collect();
        let v_data: Vec<f32> = (0..seq * kv_total)
            .map(|i| (i as f32).sin()).collect();

        let q = Tensor::from_vec(q_data.clone(), vec![seq, n_heads_total]).unwrap();
        let k = Tensor::from_vec(k_data.clone(), vec![seq, kv_total]).unwrap();
        let v = Tensor::from_vec(v_data.clone(), vec![seq, kv_total]).unwrap();

        let uncached = grouped_query_attention_causal(&q, &k, &v, n_heads, n_kv_heads).unwrap();

        let mut cache = KvCache::new(1, 64, n_kv_heads, head_dim);
        let cached = cached_attention_prefill(&mut cache, 0, 0, &q, &k, &v, n_heads, n_kv_heads)
            .unwrap();

        assert!(
            close_slice(cached.as_slice(), uncached.as_slice(), 1e-5),
            "prefill output differs from uncached GQA:\n cached={:?}\nuncached={:?}",
            &cached.as_slice()[..8], &uncached.as_slice()[..8]
        );
    }

    #[test]
    fn test_prefill_output_shape() {
        let n_heads = 4_usize;
        let n_kv_heads = 2_usize;
        let head_dim = 8_usize;
        let seq = 5_usize;
        let q = Tensor::from_vec(vec![0.1_f32; seq * n_heads * head_dim],
            vec![seq, n_heads * head_dim]).unwrap();
        let k = Tensor::from_vec(vec![0.2_f32; seq * n_kv_heads * head_dim],
            vec![seq, n_kv_heads * head_dim]).unwrap();
        let v = Tensor::from_vec(vec![0.3_f32; seq * n_kv_heads * head_dim],
            vec![seq, n_kv_heads * head_dim]).unwrap();

        let mut cache = KvCache::new(1, 128, n_kv_heads, head_dim);
        let out = cached_attention_prefill(&mut cache, 0, 0, &q, &k, &v, n_heads, n_kv_heads)
            .unwrap();
        // Output should be [seq, n_heads * head_dim]
        assert_eq!(out.dims(), &[seq, n_heads * head_dim]);
    }

    // ── decode output matches equivalent single-token prefill ─────────────

    #[test]
    fn test_decode_single_token_output_shape() {
        let n_heads = 2_usize;
        let n_kv_heads = 2_usize;
        let head_dim = 4_usize;
        let kv_dim = n_kv_heads * head_dim;

        let q = Tensor::from_vec(vec![0.5_f32; n_heads * head_dim],
            vec![1, n_heads * head_dim]).unwrap();
        let k = Tensor::from_vec(vec![0.3_f32; kv_dim], vec![1, kv_dim]).unwrap();
        let v = Tensor::from_vec(vec![0.7_f32; kv_dim], vec![1, kv_dim]).unwrap();

        let mut cache = KvCache::new(1, 64, n_kv_heads, head_dim);
        let out = cached_attention_decode(&mut cache, 0, 0, &q, &k, &v, n_heads, n_kv_heads)
            .unwrap();
        assert_eq!(out.dims(), &[1, n_heads * head_dim]);
    }

    #[test]
    fn test_decode_at_pos0_matches_prefill_single_token() {
        // Decode at pos=0 with a single token should give the same result as
        // prefill with start_pos=0 and a single token (both have seq_q=1, seq_k=1).
        let n_heads = 2_usize;
        let n_kv_heads = 2_usize;
        let head_dim = 4_usize;
        let kv_dim = n_kv_heads * head_dim;

        let q_data: Vec<f32> = (0..n_heads * head_dim).map(|i| i as f32 * 0.1).collect();
        let k_data: Vec<f32> = (0..kv_dim).map(|i| i as f32 * 0.05).collect();
        let v_data: Vec<f32> = (0..kv_dim).map(|i| (i as f32).sin()).collect();

        let q = Tensor::from_vec(q_data, vec![1, n_heads * head_dim]).unwrap();
        let k = Tensor::from_vec(k_data, vec![1, kv_dim]).unwrap();
        let v = Tensor::from_vec(v_data, vec![1, kv_dim]).unwrap();

        let mut cache_p = KvCache::new(1, 64, n_kv_heads, head_dim);
        let mut cache_d = KvCache::new(1, 64, n_kv_heads, head_dim);

        let out_p = cached_attention_prefill(&mut cache_p, 0, 0, &q, &k, &v, n_heads, n_kv_heads)
            .unwrap();
        let out_d = cached_attention_decode(&mut cache_d, 0, 0, &q, &k, &v, n_heads, n_kv_heads)
            .unwrap();

        assert!(
            close_slice(out_p.as_slice(), out_d.as_slice(), 1e-5),
            "prefill single-token != decode at pos=0"
        );
    }

    // ── cache accumulation: decode sees all prior K/V ─────────────────────

    #[test]
    fn test_decode_step2_attends_to_all_prior_kv() {
        // After prefill of 2 tokens, decode at pos=2 should attend to all 3 positions.
        // V rows: [1,0], [2,0], [3,0] for KV head 0; all queries are zero → uniform.
        // Expected output per head: mean([1,2,3], [0,0,0]) = [2.0, 0.0]
        let n_heads = 1_usize;
        let n_kv_heads = 1_usize;
        let head_dim = 2_usize;
        let kv_dim = n_kv_heads * head_dim;

        let mut cache = KvCache::new(1, 16, n_kv_heads, head_dim);

        // Prefill 2 tokens
        let q_p = Tensor::from_vec(vec![0.0_f32; 2 * head_dim], vec![2, head_dim]).unwrap();
        let k_p = Tensor::from_vec(vec![0.0_f32; 2 * kv_dim],   vec![2, kv_dim]).unwrap();
        let v_p = Tensor::from_vec(vec![
            1.0_f32, 0.0,  // pos 0
            2.0,     0.0,  // pos 1
        ], vec![2, kv_dim]).unwrap();
        cached_attention_prefill(&mut cache, 0, 0, &q_p, &k_p, &v_p, n_heads, n_kv_heads)
            .unwrap();

        // Decode step: new token at pos=2
        let q_d = Tensor::from_vec(vec![0.0_f32; head_dim], vec![1, head_dim]).unwrap();
        let k_d = Tensor::from_vec(vec![0.0_f32; kv_dim],   vec![1, kv_dim]).unwrap();
        let v_d = Tensor::from_vec(vec![3.0_f32, 0.0],       vec![1, kv_dim]).unwrap();

        let out = cached_attention_decode(&mut cache, 0, 2, &q_d, &k_d, &v_d, n_heads, n_kv_heads)
            .unwrap();

        assert_eq!(out.dims(), &[1, head_dim]);
        // Uniform attention over 3 positions → mean of [1,2,3] = 2.0
        assert!((out.as_slice()[0] - 2.0).abs() < 1e-4,
            "expected 2.0, got {}", out.as_slice()[0]);
        assert!((out.as_slice()[1] - 0.0).abs() < 1e-4,
            "expected 0.0, got {}", out.as_slice()[1]);
    }

    // ── error handling ────────────────────────────────────────────────────

    #[test]
    fn test_prefill_kv_dim_mismatch_rejected() {
        let n_heads = 2_usize;
        let n_kv_heads = 2_usize;
        let head_dim = 4_usize;
        let seq = 2_usize;
        let q = Tensor::from_vec(vec![0.0_f32; seq * n_heads * head_dim],
            vec![seq, n_heads * head_dim]).unwrap();
        // K has wrong second dim (not n_kv_heads * head_dim)
        let k = Tensor::from_vec(vec![0.0_f32; seq * 3], vec![seq, 3]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; seq * n_kv_heads * head_dim],
            vec![seq, n_kv_heads * head_dim]).unwrap();

        let mut cache = KvCache::new(1, 64, n_kv_heads, head_dim);
        assert!(matches!(
            cached_attention_prefill(&mut cache, 0, 0, &q, &k, &v, n_heads, n_kv_heads),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_decode_non_1d_query_rejected() {
        let n_heads = 2_usize;
        let n_kv_heads = 2_usize;
        let head_dim = 4_usize;
        let kv_dim = n_kv_heads * head_dim;
        // q has 2 tokens — not allowed for decode
        let q = Tensor::from_vec(vec![0.0_f32; 2 * n_heads * head_dim],
            vec![2, n_heads * head_dim]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; kv_dim], vec![1, kv_dim]).unwrap();
        let v = Tensor::from_vec(vec![0.0_f32; kv_dim], vec![1, kv_dim]).unwrap();

        let mut cache = KvCache::new(1, 64, n_kv_heads, head_dim);
        assert!(matches!(
            cached_attention_decode(&mut cache, 0, 0, &q, &k, &v, n_heads, n_kv_heads),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
