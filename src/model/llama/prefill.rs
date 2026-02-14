//! Chunked prefill for long prompts — commit 11.2.
//!
//! # Motivation
//!
//! The standard `LlamaModel::forward` allocates activations proportional to
//! the full sequence length `seq`:
//!
//! * QKV projections: `3 × seq × embed_dim`
//! * Attention score matrix: `n_heads × seq × seq`  ← O(seq²) peak
//!
//! For a 4 K prompt on a 7B model this can spike to several GB *per forward
//! pass* — well above available RAM on consumer hardware.
//!
//! Chunked prefill processes the prompt in windows of `chunk_size` tokens.
//! The KV-cache absorbs each chunk's keys and values, so later chunks can
//! attend over the full prior context without recomputing it.  Peak
//! intermediate memory is now proportional to `chunk_size × embed_dim` rather
//! than `seq × embed_dim`, and the attention score matrix never exceeds
//! `n_heads × chunk_size × total_seen`.
//!
//! # Correctness
//!
//! `ChunkedPrefill::run` produces logits `[seq, vocab]` numerically identical
//! to `LlamaModel::forward` for the same token sequence (within f32 error),
//! as long as both are run without quantisation effects.
//!
//! # Usage
//!
//! ```rust,ignore
//! use llm_engine::model::llama::prefill::ChunkedPrefill;
//!
//! let prefill = ChunkedPrefill::new(/*chunk_size=*/ 512);
//! let logits  = prefill.run(&model, &tokens)?;
//! // logits: [seq, vocab_size]
//! ```

use crate::cache::KvCache;
use crate::model::{ModelError, Result};
use crate::model::llama::LlamaModel;
use crate::ops::{norm::rmsnorm, matmul::matmul_blocked};
use crate::tensor::Tensor;

// ── ChunkedPrefill ────────────────────────────────────────────────────────────

/// Configuration for chunked prompt processing.
///
/// Holds a single tunable parameter: `chunk_size`.  Everything else (number of
/// layers, heads, etc.) is read from the model at call time.
#[derive(Debug, Clone)]
pub struct ChunkedPrefill {
    /// Maximum number of tokens processed per chunk.  Must be ≥ 1.
    pub chunk_size: usize,
}

impl ChunkedPrefill {
    /// Create a new `ChunkedPrefill` with the given chunk size.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size == 0`.
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be ≥ 1");
        Self { chunk_size }
    }

    /// Run chunked prefill over `tokens`, returning logits `[seq_len, vocab_size]`.
    ///
    /// The output is numerically equivalent to `model.forward(tokens)` but uses
    /// a KV-cache internally to bound peak activation memory to
    /// `O(chunk_size × embed_dim)` per layer.
    ///
    /// # Arguments
    ///
    /// * `model`  – fully-loaded Llama model.
    /// * `tokens` – prompt token IDs; must be non-empty.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if any token is out of range, any tensor
    /// operation fails, or the sequence length exceeds the model's context
    /// window.
    pub fn run(&self, model: &LlamaModel, tokens: &[u32]) -> Result<Tensor<f32>> {
        if tokens.is_empty() {
            return Err(ModelError::InvalidConfig {
                reason: "chunked_prefill: token list must not be empty".to_string(),
            });
        }

        let cfg       = model.config();
        let seq_len   = tokens.len();
        let max_ctx   = cfg.context_length as usize;

        if seq_len > max_ctx {
            return Err(ModelError::InvalidConfig {
                reason: format!(
                    "chunked_prefill: seq_len {seq_len} exceeds context_length {max_ctx}"
                ),
            });
        }

        // Pre-allocate a KV-cache large enough for the entire prompt.
        let n_layers   = cfg.block_count as usize;
        let n_kv_heads = cfg.n_kv_heads  as usize;
        let head_dim   = cfg.head_dim()  as usize;

        let mut cache = KvCache::new(n_layers, max_ctx, n_kv_heads, head_dim);

        // Collect logits for every chunk and concatenate at the end.
        let mut all_logits: Vec<f32> = Vec::with_capacity(seq_len * cfg.vocab_size as usize);

        let mut start = 0_usize;
        while start < seq_len {
            let end        = (start + self.chunk_size).min(seq_len);
            let chunk_toks = &tokens[start..end];

            // 1. Token embedding lookup for this chunk.
            let mut x = model.embed_tokens(chunk_toks)?;

            // 2. Pass through all transformer blocks with KV-cache.
            for (layer_idx, block) in model.blocks().iter().enumerate() {
                x = block.forward_cached(&x, start, &mut cache, layer_idx)?;
            }

            // 3. Final RMSNorm.
            x = rmsnorm(&x, model.output_norm(), cfg.rms_norm_eps)?;

            // 4. Unembedding: x @ output.T → [chunk_size, vocab]
            let output_t = model.output_weight().transpose(0, 1)?.contiguous();
            let chunk_logits = matmul_blocked(&x, &output_t)?;

            all_logits.extend_from_slice(chunk_logits.as_slice());
            start = end;
        }

        Ok(Tensor::from_vec(all_logits, vec![seq_len, cfg.vocab_size as usize])?)
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::llama::{LlamaModel, LlamaConfig, TransformerBlock};
    use crate::ops::rope::RopeTable;
    use crate::gguf::{MetadataValue, Metadata};

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

    fn make_model(cfg: &LlamaConfig) -> LlamaModel {
        let embed  = cfg.embedding_length   as usize;
        let vocab  = cfg.vocab_size         as usize;
        let ffn    = cfg.feed_forward_length as usize;
        let heads  = cfg.n_heads            as usize;
        let kv     = cfg.n_kv_heads         as usize;
        let hd     = cfg.head_dim()         as usize;
        let qd     = heads * hd;
        let kvd    = kv   * hd;
        let rope   = RopeTable::new(cfg.context_length as usize, hd, cfg.rope_freq_base);

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

    // ── output shape ─────────────────────────────────────────────────────

    #[test]
    fn test_chunked_output_shape_single_chunk() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let p     = ChunkedPrefill::new(8);
        let out   = p.run(&model, &[0, 1, 2, 3]).unwrap();
        assert_eq!(out.dims(), &[4, cfg.vocab_size as usize]);
    }

    #[test]
    fn test_chunked_output_shape_multi_chunk() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        // chunk_size=3 over 8 tokens → chunks [0..3), [3..6), [6..8)
        let p   = ChunkedPrefill::new(3);
        let out = p.run(&model, &[0,1,2,3,4,5,6,7]).unwrap();
        assert_eq!(out.dims(), &[8, cfg.vocab_size as usize]);
    }

    #[test]
    fn test_chunked_single_token() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let p     = ChunkedPrefill::new(4);
        let out   = p.run(&model, &[5]).unwrap();
        assert_eq!(out.dims(), &[1, cfg.vocab_size as usize]);
    }

    // ── no NaN / Inf ─────────────────────────────────────────────────────

    #[test]
    fn test_chunked_no_nan() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let p     = ChunkedPrefill::new(2);
        let out   = p.run(&model, &[0,1,2,3,4,5]).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(),      "NaN in chunked logits");
            assert!(!v.is_infinite(), "Inf in chunked logits");
        }
    }

    // ── chunk_size=seq_len matches forward() ────────────────────────────

    #[test]
    fn test_chunked_matches_forward_single_chunk() {
        let cfg    = tiny_config();
        let model  = make_model(&cfg);
        let tokens = &[0u32, 3, 7, 15];

        // Full-sequence forward
        let logits_full = model.forward(tokens).unwrap();

        // Chunked with chunk_size=seq_len (single chunk, no cross-chunk cache)
        let p = ChunkedPrefill::new(tokens.len());
        let logits_chunked = p.run(&model, tokens).unwrap();

        assert_eq!(logits_full.dims(), logits_chunked.dims());
        for (i, (&a, &b)) in logits_full.as_slice().iter()
            .zip(logits_chunked.as_slice()).enumerate()
        {
            assert!((a - b).abs() < 1e-4,
                "logit[{i}] mismatch: forward={a} chunked={b}");
        }
    }

    // ── errors ────────────────────────────────────────────────────────────

    #[test]
    fn test_empty_tokens_rejected() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let p     = ChunkedPrefill::new(4);
        assert!(p.run(&model, &[]).is_err());
    }

    #[test]
    fn test_seq_exceeds_context_length_rejected() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let p     = ChunkedPrefill::new(4);
        // context_length = 64; send 65 tokens
        let tokens: Vec<u32> = (0..65).map(|i| i % 32).collect();
        assert!(p.run(&model, &tokens).is_err());
    }
}
