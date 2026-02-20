//! Full Llama model forward pass — commit 8.2.
//!
//! Wires together: token embedding → N transformer blocks → final RMSNorm → logits.
//!
//! ```text
//! tokens[seq]
//!   │
//!   ▼  token_embd [vocab, embed]
//! x [seq, embed]
//!   │
//!   ├─▶ block_0 ──▶ block_1 ──▶ … ──▶ block_{N-1}
//!   │
//!   ▼  output_norm (RMSNorm)
//! x_norm [seq, embed]
//!   │
//!   ▼  output [vocab, embed]  →  x_norm @ output.T
//! logits [seq, vocab]
//! ```
//!
//! # Loading from GGUF
//!
//! `LlamaModel::from_loader` extracts all weights via the generic
//! `TensorExtractor::extract` which auto-handles F32, F16, Q8_0, Q4_0.
//!
//! # Current limitations
//!
//! - `forward` uses `start_pos = 0` (prefill only — no KV-cache decode yet).
//!   Incremental decode with `start_pos > 0` will be enabled in commit 9.x
//!   once the KV-cache is implemented.

use crate::gguf::{GgufLoader, TensorExtractor};
use crate::model::{Result, ModelError};
use crate::cache::KvCache;
use crate::model::llama::{
    LlamaConfig, TransformerBlock,
    weights::{WeightRole, GlobalWeightRole, weight_name, global_weight_name},
};
use crate::ops::{norm::rmsnorm, matmul::matmul_blocked};
use crate::ops::rope::RopeTable;
use crate::tensor::Tensor;

// ── LlamaModel ───────────────────────────────────────────────────────────────

/// Full Llama model: embedding + N transformer blocks + final norm + unembedding.
pub struct LlamaModel { // CHANGED
    config: LlamaConfig,
    /// Token embedding table: `[vocab_size, embed_dim]`
    token_embd: Tensor<f32>,
    /// Transformer blocks (one per layer).
    blocks: Vec<TransformerBlock>,
    /// Final RMSNorm scale: `[embed_dim]`
    output_norm: Tensor<f32>,
    /// Unembedding / LM-head matrix: `[vocab_size, embed_dim]`
    output: Tensor<f32>,
}

impl LlamaModel {
    /// Construct from pre-built components.
    ///
    /// Used in unit tests and custom loading pipelines.
    #[must_use]
    pub fn new( // CHANGED
        config: LlamaConfig,
        token_embd: Tensor<f32>,
        blocks: Vec<TransformerBlock>,
        output_norm: Tensor<f32>,
        output: Tensor<f32>,
    ) -> Self {
        Self { config, token_embd, blocks, output_norm, output }
    }

    /// Return the model configuration.
    #[must_use]
    pub fn config(&self) -> &LlamaConfig { &self.config }

    // ── accessors used by ChunkedPrefill ──────────────────────────────────

    /// Slice of transformer blocks (one per layer).
    #[must_use]
    pub fn blocks(&self) -> &[TransformerBlock] { &self.blocks }

    /// Final RMSNorm scale vector `[embed_dim]`.
    #[must_use]
    pub fn output_norm(&self) -> &Tensor<f32> { &self.output_norm }

    /// Unembedding matrix `[vocab_size, embed_dim]`.
    #[must_use]
    pub fn output_weight(&self) -> &Tensor<f32> { &self.output }

    /// Embed a slice of token IDs into `[seq, embed_dim]`.
    ///
    /// Public so that `ChunkedPrefill` can call it without duplicating the logic.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::InvalidConfig`] if any token ID ≥ vocab_size.
    pub fn embed_tokens(&self, tokens: &[u32]) -> Result<Tensor<f32>> {
        embed(&self.token_embd, tokens)
    }

    /// Load a model from a GGUF file.
    ///
    /// Extracts all weights using `TensorExtractor::extract` which auto-handles
    /// F32, F16, Q8_0, and Q4_0 tensor types.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::LoadError`] if any tensor is missing or unreadable,
    /// or [`ModelError::MissingMetadataKey`] / [`ModelError::InvalidConfig`]
    /// if the metadata is incomplete.
    pub fn from_loader(loader: &GgufLoader) -> Result<Self> { // CHANGED
        let config = LlamaConfig::from_metadata(loader.metadata())?;
        let ex = TensorExtractor::new(loader);

        // ── global weights ─────────────────────────────────────────────────
        let token_embd  = ex.extract(global_weight_name(GlobalWeightRole::TokenEmbd))?;
        let output_norm = ex.extract(global_weight_name(GlobalWeightRole::OutputNorm))?;
        let output      = ex.extract(global_weight_name(GlobalWeightRole::Output))?;

        // ── shared RoPE table ──────────────────────────────────────────────
        let head_dim   = config.head_dim() as usize;
        let rope_table = RopeTable::new(
            config.context_length as usize,
            head_dim,
            config.rope_freq_base,
        );

        // ── per-layer blocks ───────────────────────────────────────────────
        let mut blocks = Vec::with_capacity(config.block_count as usize);
        for layer in 0..config.block_count as usize {
            let wq        = ex.extract(&weight_name(layer, WeightRole::AttnQ))?;
            let wk        = ex.extract(&weight_name(layer, WeightRole::AttnK))?;
            let wv        = ex.extract(&weight_name(layer, WeightRole::AttnV))?;
            let wo        = ex.extract(&weight_name(layer, WeightRole::AttnOutput))?;
            let attn_norm = ex.extract(&weight_name(layer, WeightRole::AttnNorm))?;
            let wgate     = ex.extract(&weight_name(layer, WeightRole::FfnGate))?;
            let wup       = ex.extract(&weight_name(layer, WeightRole::FfnUp))?;
            let wdown     = ex.extract(&weight_name(layer, WeightRole::FfnDown))?;
            let ffn_norm  = ex.extract(&weight_name(layer, WeightRole::FfnNorm))?;
            blocks.push(TransformerBlock::new(
                wq, wk, wv, wo, attn_norm,
                wgate, wup, wdown, ffn_norm,
                rope_table.clone(), // CHANGED: clone per block
                config.n_heads as usize, config.n_kv_heads as usize,
                config.rms_norm_eps,
            ));
        }

        Ok(Self::new(config, token_embd, blocks, output_norm, output))
    }

    /// Run the full forward pass.
    ///
    /// # Arguments
    ///
    /// * `tokens` — Slice of token IDs.  Each must be `< vocab_size`.
    ///
    /// # Returns
    ///
    /// Logits tensor of shape `[seq_len, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] on out-of-range token IDs or tensor shape failures.
    pub fn forward(&self, tokens: &[u32]) -> Result<Tensor<f32>> { // CHANGED
        if tokens.is_empty() {
            return Err(ModelError::InvalidConfig {
                reason: "forward: token list must not be empty".to_string(),
            });
        }

        // 1. Token embedding lookup: gather rows → [seq, embed_dim]
        let mut x = embed(&self.token_embd, tokens)?; // CHANGED

        // 2. Pass through all transformer blocks (prefill, start_pos = 0)
        for block in &self.blocks {
            x = block.forward(&x, 0)?; // CHANGED
        }

        // 3. Final RMSNorm
        x = rmsnorm(&x, &self.output_norm, self.config.rms_norm_eps)?; // CHANGED

        // 4. Logits: x @ output.T  →  [seq, vocab_size]
        let output_t = self.output.transpose(0, 1)?.contiguous(); // CHANGED
        let logits = matmul_blocked(&x, &output_t)?;              // CHANGED
        Ok(logits)
    }

    /// Decode a single new token at absolute position `pos` using the KV-cache.
    ///
    /// Faster than `forward` for incremental generation: processes exactly one
    /// token and reuses all prior K/V projections from `cache`.
    ///
    /// # Arguments
    ///
    /// * `token` – ID of the token to decode.  Must be `< vocab_size`.
    /// * `pos`   – absolute sequence position of this token in the cache.
    ///             Must equal the number of tokens already written (0-based).
    /// * `cache` – shared KV-cache (will be written at `pos` for every layer).
    ///
    /// # Returns
    ///
    /// Logits vector `[vocab_size]` for the new token position.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] on out-of-range token ID, cache overflow, or
    /// any tensor shape failure.
    pub fn forward_decode(
        &self,
        token: u32,
        pos:   usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        // 1. Embed the single token → [1, embed_dim]
        let mut x = self.embed_tokens(&[token])?;

        // 2. Pass through every transformer block (decode path)
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward_decode(&x, pos, cache, layer_idx)?;
        }

        // 3. Final RMSNorm
        x = rmsnorm(&x, &self.output_norm, self.config.rms_norm_eps)?;

        // 4. Unembedding: x @ output.T → [1, vocab_size]
        let output_t = self.output.transpose(0, 1)?.contiguous();
        let logits   = matmul_blocked(&x, &output_t)?;

        Ok(logits.as_slice().to_vec())
    }
}

/// Gather token embedding rows: `[vocab, embed] → [seq, embed]`.
fn embed(token_embd: &Tensor<f32>, tokens: &[u32]) -> Result<Tensor<f32>> { // CHANGED
    let vocab_size = token_embd.dims()[0];
    let embed_dim  = token_embd.dims()[1];
    let data = token_embd.as_slice(); // contiguous (extracted from GGUF)
    let mut out = Vec::with_capacity(tokens.len() * embed_dim);
    for &tok in tokens {
        let id = tok as usize;
        if id >= vocab_size {
            return Err(ModelError::InvalidConfig {
                reason: format!("token id {id} >= vocab_size {vocab_size}"),
            });
        }
        let start = id * embed_dim;
        out.extend_from_slice(&data[start..start + embed_dim]);
    }
    Ok(Tensor::from_vec(out, vec![tokens.len(), embed_dim])?)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;
    use crate::model::llama::{config::LlamaConfig, block::TransformerBlock};
    use crate::ops::rope::RopeTable;
    use crate::gguf::MetadataValue;
    use crate::gguf::Metadata;

    /// Minimal config for a tiny synthetic model.
    fn tiny_config() -> LlamaConfig {
        let mut m = Metadata::new();
        for (k, v) in [
            ("llama.block_count",          MetadataValue::Uint32(2)),
            ("llama.embedding_length",     MetadataValue::Uint32(8)),
            ("llama.attention.head_count", MetadataValue::Uint32(2)),
            ("llama.feed_forward_length",  MetadataValue::Uint32(16)),
            ("llama.context_length",       MetadataValue::Uint32(64)),
            ("llama.vocab_size",           MetadataValue::Uint32(32)),
        ] { m.insert(k.to_string(), v); }
        LlamaConfig::from_metadata(&m).unwrap()
    }

    /// Build a `LlamaModel` with all-zero weights for shape/smoke tests.
    fn make_model(cfg: &LlamaConfig) -> LlamaModel {
        let embed  = cfg.embedding_length as usize;
        let vocab  = cfg.vocab_size       as usize;
        let ffn    = cfg.feed_forward_length as usize;
        let heads  = cfg.n_heads          as usize;
        let kv     = cfg.n_kv_heads       as usize;
        let hd     = cfg.head_dim()       as usize;
        let qd     = heads * hd;
        let kvd    = kv   * hd;

        let rope = RopeTable::new(cfg.context_length as usize, hd, cfg.rope_freq_base);
        let blocks: Vec<_> = (0..cfg.block_count as usize).map(|_| {
            TransformerBlock::new(
                Tensor::zeros(vec![qd,    embed]),
                Tensor::zeros(vec![kvd,   embed]),
                Tensor::zeros(vec![kvd,   embed]),
                Tensor::zeros(vec![embed, qd   ]),
                Tensor::ones(vec![embed]),
                Tensor::zeros(vec![ffn,   embed]),
                Tensor::zeros(vec![ffn,   embed]),
                Tensor::zeros(vec![embed, ffn  ]),
                Tensor::ones(vec![embed]),
                rope.clone(),
                heads, kv, cfg.rms_norm_eps,
            )
        }).collect();

        LlamaModel::new(
            cfg.clone(),
            Tensor::zeros(vec![vocab, embed]),
            blocks,
            Tensor::ones(vec![embed]),
            Tensor::zeros(vec![vocab, embed]),
        )
    }

    #[test]
    fn test_forward_single_token_output_shape() { // CHANGED
        let cfg   = tiny_config();
        let vocab = cfg.vocab_size as usize;
        let model = make_model(&cfg);
        let out   = model.forward(&[0]).unwrap();
        assert_eq!(out.dims(), &[1, vocab]);
    }

    #[test]
    fn test_forward_multi_token_output_shape() { // CHANGED
        let cfg   = tiny_config();
        let vocab = cfg.vocab_size as usize;
        let model = make_model(&cfg);
        let out   = model.forward(&[0, 1, 2, 3]).unwrap();
        assert_eq!(out.dims(), &[4, vocab]);
    }

    #[test]
    fn test_forward_no_nan_or_inf() { // CHANGED
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let out   = model.forward(&[0, 5, 31]).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(),      "NaN in logits");
            assert!(!v.is_infinite(), "Inf in logits");
        }
    }

    #[test]
    fn test_forward_empty_tokens_rejected() { // CHANGED
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        assert!(model.forward(&[]).is_err());
    }

    #[test]
    fn test_forward_out_of_range_token_rejected() { // CHANGED
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        // vocab_size = 32, so token 32 is out of range
        assert!(model.forward(&[32]).is_err());
    }

    #[test]
    fn test_embed_gathers_correct_rows() { // CHANGED
        // Build a token_embd where row i is filled with i as f32
        let vocab = 4_usize;
        let dim   = 3_usize;
        let data: Vec<f32> = (0..vocab).flat_map(|i| vec![i as f32; dim]).collect();
        let token_embd = Tensor::from_vec(data, vec![vocab, dim]).unwrap();
        let out = embed(&token_embd, &[2, 0, 3]).unwrap();
        assert_eq!(out.dims(), &[3, dim]);
        // row 0 of out should be all 2.0
        assert!(out.as_slice()[..dim].iter().all(|&v: &f32| (v - 2.0).abs() < 1e-7));
        // row 1 should be all 0.0
        assert!(out.as_slice()[dim..2*dim].iter().all(|&v: &f32| v == 0.0));
        // row 2 should be all 3.0
        assert!(out.as_slice()[2*dim..].iter().all(|&v: &f32| (v - 3.0).abs() < 1e-7));
    }

    #[test]
    fn test_config_accessor() { // CHANGED
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        assert_eq!(model.config().block_count, 2);
        assert_eq!(model.config().vocab_size, 32);
    }

    // ── forward_decode ────────────────────────────────────────────────────

    #[test]
    fn test_forward_decode_output_length() {
        let cfg   = tiny_config();
        let vocab = cfg.vocab_size as usize;
        let model = make_model(&cfg);
        let n_layers   = cfg.block_count as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim   = cfg.head_dim() as usize;
        let mut cache  = KvCache::new(n_layers, cfg.context_length as usize, n_kv_heads, head_dim);
        let logits = model.forward_decode(0, 0, &mut cache).unwrap();
        assert_eq!(logits.len(), vocab);
    }

    #[test]
    fn test_forward_decode_no_nan() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let n_layers   = cfg.block_count as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim   = cfg.head_dim() as usize;
        let mut cache  = KvCache::new(n_layers, cfg.context_length as usize, n_kv_heads, head_dim);
        let logits = model.forward_decode(5, 0, &mut cache).unwrap();
        for (i, &v) in logits.iter().enumerate() {
            assert!(!v.is_nan(),      "NaN in decode logits[{i}]");
            assert!(!v.is_infinite(), "Inf in decode logits[{i}]");
        }
    }

    /// At pos=0 with a fresh cache, `forward_decode` must produce the same
    /// logits as the first (and only) row of `forward(&[token])`.
    #[test]
    fn test_forward_decode_matches_forward_single_token() {
        let cfg   = tiny_config();
        let vocab = cfg.vocab_size as usize;
        let model = make_model(&cfg);
        let n_layers   = cfg.block_count as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim   = cfg.head_dim() as usize;

        let logits_full = model.forward(&[5]).unwrap();
        let mut cache   = KvCache::new(n_layers, cfg.context_length as usize, n_kv_heads, head_dim);
        let logits_dec  = model.forward_decode(5, 0, &mut cache).unwrap();

        assert_eq!(logits_dec.len(), vocab);
        for (i, (&d, &f)) in logits_dec.iter().zip(logits_full.as_slice()).enumerate() {
            assert!((d - f).abs() < 1e-4,
                "logit[{i}]: decode={d} forward={f}");
        }
    }
}
