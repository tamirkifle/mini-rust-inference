//! Causal (autoregressive) attention masks — commit 7.2.
//!
//! # Purpose
//!
//! In autoregressive generation each token may only attend to itself and
//! **previous** tokens, never future ones.  This is enforced by adding
//! `−∞` to score positions `[i, j]` where `j > i` before the softmax step.
//! After softmax those positions become exactly `0.0`.
//!
//! # Functions
//!
//! | Function                    | Use-case                                       |
//! |-----------------------------|------------------------------------------------|
//! | [`causal_mask`]             | Full-sequence prefill  (`seq_q == seq_k`)      |
//! | [`causal_mask_with_offset`] | KV-cache decode step   (`seq_q < seq_k`)       |
//! | [`masked_sdpa`]             | SDPA + causal mask in one call (full-sequence) |
//! | [`masked_sdpa_with_offset`] | SDPA + causal mask for cached decode step      |
//!
//! # Mask convention
//!
//! Additive bias tensors use the "add before softmax" convention:
//! - `0.0`              → position is **allowed**
//! - `f32::NEG_INFINITY` → position is **masked** (will become `0.0` after softmax)

use crate::tensor::{Result, Tensor, TensorError};
use crate::attention::sdpa::scaled_dot_product_attention_with_bias;

// ── mask constructors ───────────────────────────────────────────────────────

/// Build a square causal mask of shape `[seq_len, seq_len]`.
///
/// Entry `[i, j]` is:
/// - `0.0`               if `j <= i`  (token `i` can attend to token `j`)
/// - `f32::NEG_INFINITY` if `j >  i`  (token `i` cannot attend to future token `j`)
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `seq_len == 0`.
#[must_use = "returns a new mask tensor"] // CHANGED
pub fn causal_mask(seq_len: usize) -> Result<Tensor<f32>> { // CHANGED
    if seq_len == 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: "causal_mask: seq_len must be > 0".to_string(),
        });
    }
    let mut data = vec![0.0_f32; seq_len * seq_len]; // CHANGED
    for i in 0..seq_len { // CHANGED
        for j in (i + 1)..seq_len { // CHANGED
            data[i * seq_len + j] = f32::NEG_INFINITY; // CHANGED: mask future positions
        }
    }
    Tensor::from_vec(data, vec![seq_len, seq_len]) // CHANGED
}

/// Build a causal mask for the **KV-cache decode step**.
///
/// During cached decoding, the query tensor has shape `[seq_q, d_k]` (often
/// `seq_q == 1` for single-token generation) while the key/value tensors span
/// the full `[seq_k, d_k]` context accumulated so far.
///
/// `start_pos` is the index of the first query token in the full sequence.
/// Token `i` (0-based in the query) corresponds to absolute position
/// `start_pos + i`.  It may attend to any key at absolute position
/// `<= start_pos + i`.
///
/// Returns a mask of shape `[seq_q, seq_k]`.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `seq_q == 0`, `seq_k == 0`, or
///   `start_pos + seq_q > seq_k` (query extends beyond available keys).
#[must_use = "returns a new mask tensor"] // CHANGED
pub fn causal_mask_with_offset( // CHANGED
    seq_q:     usize,
    seq_k:     usize,
    start_pos: usize,
) -> Result<Tensor<f32>> {
    if seq_q == 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: "causal_mask_with_offset: seq_q must be > 0".to_string(),
        });
    }
    if seq_k == 0 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: "causal_mask_with_offset: seq_k must be > 0".to_string(),
        });
    }
    if start_pos + seq_q > seq_k { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "causal_mask_with_offset: start_pos({start_pos}) + seq_q({seq_q}) \
                 > seq_k({seq_k}); query extends beyond available keys"
            ),
        });
    }

    let mut data = vec![0.0_f32; seq_q * seq_k]; // CHANGED
    for i in 0..seq_q { // CHANGED
        let abs_pos = start_pos + i; // CHANGED: absolute position of query token i
        for j in (abs_pos + 1)..seq_k { // CHANGED: mask all keys after this position
            data[i * seq_k + j] = f32::NEG_INFINITY; // CHANGED
        }
    }
    Tensor::from_vec(data, vec![seq_q, seq_k]) // CHANGED
}

// ── convenience wrappers ─────────────────────────────────────────────────────

/// Causally-masked scaled dot-product attention (full sequence, prefill).
///
/// Equivalent to calling [`causal_mask`] then
/// [`scaled_dot_product_attention_with_bias`].
///
/// `Q`, `K`, `V` must all be 2-D with `seq_q == seq_k`.
///
/// # Errors
///
/// Same errors as [`scaled_dot_product_attention_with_bias`] plus
/// [`TensorError::InvalidShape`] if `seq_q != seq_k`.
#[must_use = "returns the masked attention output"] // CHANGED
pub fn masked_sdpa( // CHANGED
    q: &Tensor<f32>,
    k: &Tensor<f32>,
    v: &Tensor<f32>,
) -> Result<Tensor<f32>> {
    if q.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("masked_sdpa: q must be 2-D, got {}D", q.ndim()),
        });
    }
    let seq_q = q.dims()[0]; // CHANGED
    let seq_k = if k.ndim() >= 1 { k.dims()[0] } else { 0 }; // CHANGED
    if seq_q != seq_k { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!(
                "masked_sdpa: seq_q({seq_q}) != seq_k({seq_k}); \
                 use masked_sdpa_with_offset for the decode step"
            ),
        });
    }
    let mask = causal_mask(seq_q)?; // CHANGED
    scaled_dot_product_attention_with_bias(q, k, v, &mask) // CHANGED
}

/// Causally-masked SDPA for the **KV-cache decode step**.
///
/// `start_pos` is the absolute sequence position of the first query token.
/// Typically `start_pos == current_kv_length - seq_q`.
///
/// # Errors
///
/// Same as [`masked_sdpa`] plus out-of-range errors from
/// [`causal_mask_with_offset`].
#[must_use = "returns the masked attention output"] // CHANGED
pub fn masked_sdpa_with_offset( // CHANGED
    q:         &Tensor<f32>,
    k:         &Tensor<f32>,
    v:         &Tensor<f32>,
    start_pos: usize,
) -> Result<Tensor<f32>> {
    if q.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("masked_sdpa_with_offset: q must be 2-D, got {}D", q.ndim()),
        });
    }
    if k.ndim() != 2 { // CHANGED
        return Err(TensorError::InvalidShape {
            reason: format!("masked_sdpa_with_offset: k must be 2-D, got {}D", k.ndim()),
        });
    }
    let seq_q = q.dims()[0]; // CHANGED
    let seq_k = k.dims()[0]; // CHANGED
    let mask = causal_mask_with_offset(seq_q, seq_k, start_pos)?; // CHANGED
    scaled_dot_product_attention_with_bias(q, k, v, &mask) // CHANGED
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;
    fn close(a: f32, b: f32) -> bool { (a - b).abs() < EPS }

    // ── causal_mask shape and values ──────────────────────────────────────

    #[test]
    fn test_causal_mask_shape() { // CHANGED
        let m = causal_mask(4).unwrap();
        assert_eq!(m.dims(), &[4, 4]);
    }

    #[test]
    fn test_causal_mask_lower_triangle_zero() { // CHANGED
        // All entries [i, j] with j <= i must be 0.0
        let n = 5_usize;
        let m = causal_mask(n).unwrap();
        for i in 0..n {
            for j in 0..=i {
                assert!(
                    close(m.as_slice()[i * n + j], 0.0),
                    "[{i},{j}] should be 0.0, got {}", m.as_slice()[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_upper_triangle_neg_inf() { // CHANGED
        // All entries [i, j] with j > i must be -inf
        let n = 5_usize;
        let m = causal_mask(n).unwrap();
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    m.as_slice()[i * n + j].is_infinite() && m.as_slice()[i * n + j] < 0.0,
                    "[{i},{j}] should be -inf, got {}", m.as_slice()[i * n + j]
                );
            }
        }
    }

    #[test]
    fn test_causal_mask_1x1() { // CHANGED: degenerate single token
        let m = causal_mask(1).unwrap();
        assert_eq!(m.dims(), &[1, 1]);
        assert!(close(m.as_slice()[0], 0.0));
    }

    #[test]
    fn test_causal_mask_zero_rejected() { // CHANGED
        assert!(matches!(causal_mask(0), Err(TensorError::InvalidShape { .. })));
    }

    // ── causal_mask_with_offset ───────────────────────────────────────────

    #[test]
    fn test_causal_mask_offset_shape() { // CHANGED
        // decode step: seq_q=1, seq_k=4, start_pos=3
        let m = causal_mask_with_offset(1, 4, 3).unwrap();
        assert_eq!(m.dims(), &[1, 4]);
    }

    #[test]
    fn test_causal_mask_offset_single_token_decode() { // CHANGED
        // Token at position 2 (0-based) may attend to positions 0, 1, 2
        // seq_k = 4, so positions 3 should be masked.
        let m = causal_mask_with_offset(1, 4, 2).unwrap();
        // row 0: [0.0, 0.0, 0.0, -inf]
        let s = m.as_slice();
        assert!(close(s[0], 0.0), "pos 0 should be 0.0");
        assert!(close(s[1], 0.0), "pos 1 should be 0.0");
        assert!(close(s[2], 0.0), "pos 2 should be 0.0 (self)");
        assert!(s[3].is_infinite() && s[3] < 0.0, "pos 3 should be -inf");
    }

    #[test]
    fn test_causal_mask_offset_matches_causal_mask_when_full() { // CHANGED
        // causal_mask_with_offset(n, n, 0) should equal causal_mask(n)
        let n = 4_usize;
        let full  = causal_mask(n).unwrap();
        let off   = causal_mask_with_offset(n, n, 0).unwrap();
        for (a, b) in full.as_slice().iter().zip(off.as_slice()) {
            // Both 0.0 or both -inf
            assert_eq!(a.is_infinite(), b.is_infinite());
            if !a.is_infinite() { assert!(close(*a, *b)); }
        }
    }

    #[test]
    fn test_causal_mask_offset_multi_query() { // CHANGED
        // seq_q=2, seq_k=4, start_pos=1: queries at abs positions 1 and 2
        // row 0 (abs 1): attend to [0,1], mask [2,3]
        // row 1 (abs 2): attend to [0,1,2], mask [3]
        let m = causal_mask_with_offset(2, 4, 1).unwrap();
        let s = m.as_slice();
        // row 0
        assert!(close(s[0], 0.0));
        assert!(close(s[1], 0.0));
        assert!(s[2].is_infinite() && s[2] < 0.0);
        assert!(s[3].is_infinite() && s[3] < 0.0);
        // row 1
        assert!(close(s[4], 0.0));
        assert!(close(s[5], 0.0));
        assert!(close(s[6], 0.0));
        assert!(s[7].is_infinite() && s[7] < 0.0);
    }

    #[test]
    fn test_causal_mask_offset_out_of_range_rejected() { // CHANGED
        // start_pos=3, seq_q=2 → needs seq_k >= 5, but we only have 4
        assert!(matches!(
            causal_mask_with_offset(2, 4, 3),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    // ── masked attention: future positions have zero weight ───────────────

    #[test]
    fn test_masked_sdpa_future_positions_zero() { // CHANGED
        // With 4 tokens, token 0 should attend only to itself.
        // We set all V rows to unique values and verify token 0's output
        // equals V row 0 exactly (weight on positions 1,2,3 = 0).
        let q = Tensor::from_vec(
            vec![1.0_f32, 0.0,  // token 0: [1, 0]
                 0.0,     1.0,  // token 1
                 1.0,     1.0,  // token 2
                 0.5,     0.5], // token 3
            vec![4, 2]).unwrap();
        let k = q.clone(); // same as Q for simplicity
        let v = Tensor::from_vec(
            vec![10.0_f32, 20.0, // row 0
                 30.0,     40.0, // row 1
                 50.0,     60.0, // row 2
                 70.0,     80.0],// row 3
            vec![4, 2]).unwrap();

        let out = masked_sdpa(&q, &k, &v).unwrap();

        // Token 0 can only see itself → output must equal V[0] = [10, 20]
        assert!(close(out.as_slice()[0], 10.0),
            "out[0,0] = {}, expected 10.0", out.as_slice()[0]);
        assert!(close(out.as_slice()[1], 20.0),
            "out[0,1] = {}, expected 20.0", out.as_slice()[1]);
    }

    #[test]
    fn test_masked_sdpa_last_token_sees_all() { // CHANGED
        // Token N-1 should be able to attend to all previous tokens.
        // All-zero Q and K → uniform distribution over all visible tokens.
        // With causal mask, token 3 (last of 4) attends to all 4 equally.
        // V rows are [0,0], [1,1], [2,2], [3,3] → mean of all 4 = [1.5, 1.5]
        let q = Tensor::from_vec(vec![0.0_f32; 8], vec![4, 2]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 8], vec![4, 2]).unwrap();
        let v = Tensor::from_vec(
            vec![0.0_f32, 0.0,
                 1.0,     1.0,
                 2.0,     2.0,
                 3.0,     3.0],
            vec![4, 2]).unwrap();

        let out = masked_sdpa(&q, &k, &v).unwrap();

        // Last row (token 3) should be mean([0,1,2,3], [0,1,2,3]) = [1.5, 1.5]
        let last = &out.as_slice()[6..8];
        assert!((last[0] - 1.5).abs() < 1e-5,
            "out[3,0] = {}, expected 1.5", last[0]);
        assert!((last[1] - 1.5).abs() < 1e-5,
            "out[3,1] = {}, expected 1.5", last[1]);
    }

    #[test]
    fn test_masked_sdpa_seq_mismatch_rejected() { // CHANGED
        // masked_sdpa requires seq_q == seq_k
        let q = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let k = Tensor::from_vec(vec![1.0_f32; 6], vec![3, 2]).unwrap();
        let v = Tensor::from_vec(vec![1.0_f32; 6], vec![3, 2]).unwrap();
        assert!(matches!(masked_sdpa(&q, &k, &v), Err(TensorError::InvalidShape { .. })));
    }

    // ── masked_sdpa_with_offset (decode step) ─────────────────────────────

    #[test]
    fn test_masked_sdpa_with_offset_single_token() { // CHANGED
        // Decode: single new query, full KV context of 4 tokens.
        // start_pos=3 → query at abs position 3 may attend to all 4 keys.
        let q = Tensor::from_vec(vec![0.0_f32; 2], vec![1, 2]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 8], vec![4, 2]).unwrap();
        let v = Tensor::from_vec(
            vec![0.0_f32, 0.0,
                 1.0,     1.0,
                 2.0,     2.0,
                 3.0,     3.0],
            vec![4, 2]).unwrap();

        let out = masked_sdpa_with_offset(&q, &k, &v, 3).unwrap();

        // All 4 keys attend → uniform over 4 → mean = [1.5, 1.5]
        assert_eq!(out.dims(), &[1, 2]);
        assert!((out.as_slice()[0] - 1.5).abs() < 1e-5,
            "out[0] = {}, expected 1.5", out.as_slice()[0]);
    }

    #[test]
    fn test_masked_sdpa_with_offset_blocks_future() { // CHANGED
        // Decode at start_pos=1: query at abs pos 1 may attend to keys 0,1 only,
        // NOT key 2.  V[2] = [99, 99] — if it leaks through the output will be wrong.
        let q = Tensor::from_vec(vec![0.0_f32; 2], vec![1, 2]).unwrap();
        let k = Tensor::from_vec(vec![0.0_f32; 6], vec![3, 2]).unwrap();
        let v = Tensor::from_vec(
            vec![1.0_f32, 1.0,
                 1.0,     1.0,
                 99.0,    99.0], // should be masked
            vec![3, 2]).unwrap();

        let out = masked_sdpa_with_offset(&q, &k, &v, 1).unwrap();

        // Output should be mean of V[0] and V[1] = [1, 1], not contaminated by V[2]
        assert!((out.as_slice()[0] - 1.0).abs() < 1e-4,
            "out[0] = {} — future token leaked!", out.as_slice()[0]);
        assert!((out.as_slice()[1] - 1.0).abs() < 1e-4,
            "out[1] = {} — future token leaked!", out.as_slice()[1]);
    }
}
