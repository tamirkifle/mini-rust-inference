//! Single Llama transformer block — commit 8.1.
//!
//! Implements one transformer layer:
//!
//! ```text
//! x ─── RMSNorm ──▶ Q,K,V projections ──▶ RoPE ──▶ GQA ──▶ out-proj ──▶ (+x) ──▶
//!        └─────────────────────────────────────────────────────────────────────────┐
//!                                                                                  │
//! ──── RMSNorm ──▶ gate,up projections ──▶ SwiGLU ──▶ down-proj ──▶ (+) ──▶ output
//!        └───────────────────────────────────────────────────────────┘
//! ```
//!
//! All weight matrices are stored in GGUF convention: `[out_features, in_features]`.
//! Every projection computes `x @ W.T` via an explicit transpose + contiguous copy.
//!
//! # Shapes (at construction)
//!
//! | Weight | Shape |
//! |--------|-------|
//! | `wq` | `[n_heads * head_dim, embed_dim]` |
//! | `wk` | `[n_kv_heads * head_dim, embed_dim]` |
//! | `wv` | `[n_kv_heads * head_dim, embed_dim]` |
//! | `wo` | `[embed_dim, n_heads * head_dim]` |
//! | `attn_norm` | `[embed_dim]` |
//! | `wgate` | `[ffn_dim, embed_dim]` |
//! | `wup` | `[ffn_dim, embed_dim]` |
//! | `wdown` | `[embed_dim, ffn_dim]` |
//! | `ffn_norm` | `[embed_dim]` |

use crate::model::{ModelError, Result};
use crate::ops::{
    matmul::matmul_blocked,
    norm::rmsnorm,
    activation::swiglu,
    rope::{RopeTable, rope_apply},
};
use crate::attention::gqa::grouped_query_attention_causal_with_offset;
use crate::tensor::Tensor;

// ── TransformerBlock ─────────────────────────────────────────────────────────

/// One Llama transformer block (attention + FFN with residual connections).
///
/// Constructed by injecting pre-loaded weight tensors.  In a full pipeline
/// these come from `TensorExtractor`; in tests they can be synthetic.
pub struct TransformerBlock { // CHANGED
    wq: Tensor<f32>,
    wk: Tensor<f32>,
    wv: Tensor<f32>,
    wo: Tensor<f32>,
    attn_norm: Tensor<f32>,
    wgate: Tensor<f32>,
    wup: Tensor<f32>,
    wdown: Tensor<f32>,
    ffn_norm: Tensor<f32>,
    rope_table: RopeTable,
    n_heads: usize,
    n_kv_heads: usize,
    rms_norm_eps: f32,
}

impl TransformerBlock {
    /// Create a new transformer block from pre-loaded weight tensors.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new( // CHANGED
        wq: Tensor<f32>, wk: Tensor<f32>, wv: Tensor<f32>, wo: Tensor<f32>,
        attn_norm: Tensor<f32>, wgate: Tensor<f32>, wup: Tensor<f32>,
        wdown: Tensor<f32>, ffn_norm: Tensor<f32>, rope_table: RopeTable,
        n_heads: usize, n_kv_heads: usize, rms_norm_eps: f32,
    ) -> Self {
        Self { wq, wk, wv, wo, attn_norm, wgate, wup, wdown, ffn_norm,
               rope_table, n_heads, n_kv_heads, rms_norm_eps }
    }

    /// Run one forward pass through the transformer block.
    ///
    /// `x` — `[seq_len, embed_dim]`.  `start_pos` — 0 for prefill.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if any underlying tensor operation fails.
    pub fn forward(&self, x: &Tensor<f32>, start_pos: usize) -> Result<Tensor<f32>> { // CHANGED
        let x = self.attention_sublayer(x, start_pos)?;
        let x = self.ffn_sublayer(&x)?;
        Ok(x)
    }

    fn attention_sublayer(&self, x: &Tensor<f32>, start_pos: usize) -> Result<Tensor<f32>> {
        let seq = x.dims()[0]; // CHANGED
        let normed = rmsnorm(x, &self.attn_norm, self.rms_norm_eps)?; // CHANGED
        let mut q = proj(&normed, &self.wq)?; // CHANGED
        let mut k = proj(&normed, &self.wk)?; // CHANGED
        let v = proj(&normed, &self.wv)?;     // CHANGED
        let head_dim_q = q.dims()[1] / self.n_heads;
        let head_dim_k = k.dims()[1] / self.n_kv_heads;
        q = q.reshape(vec![seq, self.n_heads, head_dim_q])?;    // CHANGED
        k = k.reshape(vec![seq, self.n_kv_heads, head_dim_k])?; // CHANGED
        rope_apply(&mut q, &self.rope_table, start_pos)?; // CHANGED
        rope_apply(&mut k, &self.rope_table, start_pos)?; // CHANGED
        q = q.reshape(vec![seq, self.n_heads * head_dim_q])?;    // CHANGED
        k = k.reshape(vec![seq, self.n_kv_heads * head_dim_k])?; // CHANGED
        let attn_out = grouped_query_attention_causal_with_offset( // CHANGED
            &q, &k, &v, self.n_heads, self.n_kv_heads, start_pos,
        )?;
        let projected = proj(&attn_out, &self.wo)?; // CHANGED
        add_elementwise(x, &projected)              // CHANGED
    }

    fn ffn_sublayer(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let normed = rmsnorm(x, &self.ffn_norm, self.rms_norm_eps)?; // CHANGED
        let gate = proj(&normed, &self.wgate)?; // CHANGED
        let up   = proj(&normed, &self.wup)?;   // CHANGED
        let hidden = swiglu(&gate, &up)?;        // CHANGED
        let ffn_out = proj(&hidden, &self.wdown)?; // CHANGED
        add_elementwise(x, &ffn_out)               // CHANGED
    }
}

// ── private helpers ──────────────────────────────────────────────────────────

/// Linear projection: `input @ weight.T` (GGUF stores weights as [out, in]).
fn proj(input: &Tensor<f32>, weight: &Tensor<f32>) -> Result<Tensor<f32>> { // CHANGED
    let wt = weight.transpose(0, 1)?.contiguous();
    Ok(matmul_blocked(input, &wt)?)
}

/// Element-wise addition — both tensors must have identical shapes.
fn add_elementwise(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> { // CHANGED
    if a.dims() != b.dims() {
        return Err(ModelError::TensorError(
            crate::tensor::TensorError::ShapeMismatch {
                expected: a.dims().to_vec(),
                got: b.dims().to_vec(),
            },
        ));
    }
    let a_data = if a.is_contiguous() {
        std::borrow::Cow::Borrowed(a.as_slice())
    } else {
        std::borrow::Cow::Owned(a.contiguous().as_slice().to_vec())
    };
    let b_data = if b.is_contiguous() {
        std::borrow::Cow::Borrowed(b.as_slice())
    } else {
        std::borrow::Cow::Owned(b.contiguous().as_slice().to_vec())
    };
    let result: Vec<f32> = a_data.iter().zip(b_data.iter()).map(|(x, y)| x + y).collect(); // CHANGED
    Ok(Tensor::from_vec(result, a.shape().clone())?)
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_block(embed: usize, n_heads: usize, n_kv_heads: usize, ffn_dim: usize) -> TransformerBlock {
        let head_dim = embed / n_heads;
        let q_dim  = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        TransformerBlock::new(
            Tensor::zeros(vec![q_dim, embed]),   // wq
            Tensor::zeros(vec![kv_dim, embed]),  // wk
            Tensor::zeros(vec![kv_dim, embed]),  // wv
            Tensor::zeros(vec![embed, q_dim]),   // wo
            Tensor::ones(vec![embed]),            // attn_norm
            Tensor::zeros(vec![ffn_dim, embed]), // wgate
            Tensor::zeros(vec![ffn_dim, embed]), // wup
            Tensor::zeros(vec![embed, ffn_dim]), // wdown
            Tensor::ones(vec![embed]),            // ffn_norm
            RopeTable::new(512, head_dim, 10_000.0),
            n_heads, n_kv_heads, 1e-5,
        )
    }

    #[test]
    fn test_block_output_shape_matches_input() { // CHANGED
        let block = make_block(8, 2, 2, 16);
        let x = Tensor::from_vec((0..16).map(|i| i as f32).collect(), vec![2, 8]).unwrap();
        let out = block.forward(&x, 0).unwrap();
        assert_eq!(out.dims(), &[2, 8]);
    }

    #[test]
    fn test_block_single_token_shape() { // CHANGED
        let block = make_block(8, 2, 2, 16);
        let x = Tensor::from_vec(vec![0.1_f32; 8], vec![1, 8]).unwrap();
        assert_eq!(block.forward(&x, 0).unwrap().dims(), &[1, 8]);
    }

    #[test]
    fn test_block_zero_weights_preserves_input() { // CHANGED
        let embed = 8_usize;
        let block = make_block(embed, 2, 2, 16);
        let data: Vec<f32> = (0..embed).map(|i| (i + 1) as f32 * 0.1).collect();
        let x = Tensor::from_vec(data.clone(), vec![1, embed]).unwrap();
        let out = block.forward(&x, 0).unwrap();
        for (i, (&e, &a)) in data.iter().zip(out.as_slice()).enumerate() {
            assert!((e - a).abs() < 1e-5, "element {i}: expected {e}, got {a}");
        }
    }

    #[test]
    fn test_block_no_nan_or_inf() { // CHANGED
        let block = make_block(8, 2, 2, 16);
        let x = Tensor::from_vec(vec![1.0,-1.0,0.5,-0.5,2.0,-2.0,0.1,-0.1], vec![1,8]).unwrap();
        let out = block.forward(&x, 0).unwrap();
        for (i, &v) in out.as_slice().iter().enumerate() {
            assert!(!v.is_nan(), "NaN at {i}");
            assert!(!v.is_infinite(), "Inf at {i}");
        }
    }

    #[test]
    fn test_block_gqa_2_kv_heads() { // CHANGED
        let block = make_block(8, 4, 2, 16);
        let x = Tensor::from_vec(vec![0.5_f32; 8], vec![1, 8]).unwrap();
        assert_eq!(block.forward(&x, 0).unwrap().dims(), &[1, 8]);
    }

    #[test]
    fn test_block_mqa_1_kv_head() { // CHANGED
        let block = make_block(8, 2, 1, 16);
        let x = Tensor::from_vec(vec![1.0_f32; 8], vec![1, 8]).unwrap();
        assert_eq!(block.forward(&x, 0).unwrap().dims(), &[1, 8]);
    }

    #[test]
    fn test_block_prefill_start_pos_zero_only() { // CHANGED
        let block = make_block(8, 2, 2, 16);
        let x = Tensor::from_vec((0..48).map(|i| i as f32 * 0.01).collect(), vec![6, 8]).unwrap();
        let out = block.forward(&x, 0).unwrap();
        assert_eq!(out.dims(), &[6, 8]);
        for &v in out.as_slice() { assert!(!v.is_nan()); }
    }
}
