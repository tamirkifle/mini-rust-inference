//! Parallel chunked prefill — commit 12.2.
//!
//! [`ParallelPrefill`] mirrors [`ChunkedPrefill`] but uses
//! [`matmul_parallel`] (rayon row-parallel GEMM) for every projection inside
//! each transformer block.  On multi-core hardware this yields measurable
//! speedup for chunks ≥ 8 tokens, where rayon's task-scheduling overhead is
//! amortised across enough rows.
//!
//! # When to choose `ParallelPrefill` over `ChunkedPrefill`
//!
//! | Scenario                   | Recommendation                         |
//! |----------------------------|----------------------------------------|
//! | Single-core or tiny prompt | `ChunkedPrefill` (no thread overhead)  |
//! | Multi-core, prompt ≥ 64    | `ParallelPrefill` (higher throughput)  |
//! | Benchmarking peak TFLOP/s  | `ParallelPrefill`                      |
//!
//! [`ChunkedPrefill`]: crate::model::llama::prefill::ChunkedPrefill

use crate::cache::KvCache;
use crate::model::{ModelError, Result};
use crate::model::llama::LlamaModel;
use crate::ops::{norm::rmsnorm, matmul::matmul_parallel};
use crate::tensor::Tensor;

// ── ParallelPrefill ───────────────────────────────────────────────────────────

/// Configuration for parallel prompt processing.
#[derive(Debug, Clone)]
pub struct ParallelPrefill {
    /// Maximum number of tokens processed per chunk.  Must be ≥ 1.
    pub chunk_size: usize,
}

impl ParallelPrefill {
    /// Create a new `ParallelPrefill` with the given chunk size.
    ///
    /// # Panics
    ///
    /// Panics if `chunk_size == 0`.
    #[must_use]
    pub fn new(chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be ≥ 1");
        Self { chunk_size }
    }

    /// Run parallel chunked prefill over `tokens`, returning logits `[seq_len, vocab_size]`.
    ///
    /// Equivalent to [`ChunkedPrefill::run`] in output, but uses
    /// `forward_cached_parallel` inside each transformer block so that all
    /// six projection matmuls (Q, K, V, O, gate, up, down) run in parallel
    /// across rayon worker threads.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] if any token is out of range, any tensor
    /// operation fails, or the sequence length exceeds the model's context window.
    ///
    /// [`ChunkedPrefill::run`]: crate::model::llama::prefill::ChunkedPrefill::run
    pub fn run(&self, model: &LlamaModel, tokens: &[u32]) -> Result<Tensor<f32>> {
        if tokens.is_empty() {
            return Err(ModelError::InvalidConfig {
                reason: "parallel_prefill: token list must not be empty".to_string(),
            });
        }

        let cfg     = model.config();
        let seq_len = tokens.len();
        let max_ctx = cfg.context_length as usize;

        if seq_len > max_ctx {
            return Err(ModelError::InvalidConfig {
                reason: format!(
                    "parallel_prefill: seq_len {seq_len} exceeds context_length {max_ctx}"
                ),
            });
        }

        let n_layers   = cfg.block_count as usize;
        let n_kv_heads = cfg.n_kv_heads  as usize;
        let head_dim   = cfg.head_dim()  as usize;
        let mut cache  = KvCache::new(n_layers, max_ctx, n_kv_heads, head_dim);

        let mut all_logits: Vec<f32> = Vec::with_capacity(seq_len * cfg.vocab_size as usize);

        let mut start = 0_usize;
        while start < seq_len {
            let end        = (start + self.chunk_size).min(seq_len);
            let chunk_toks = &tokens[start..end];

            // 1. Token embedding for this chunk.
            let mut x = model.embed_tokens(chunk_toks)?;

            // 2. All transformer blocks — parallel projection path.
            for (layer_idx, block) in model.blocks().iter().enumerate() {
                x = block.forward_cached_parallel(&x, start, &mut cache, layer_idx)?;
            }

            // 3. Final RMSNorm.
            x = rmsnorm(&x, model.output_norm(), cfg.rms_norm_eps)?;

            // 4. Unembedding with parallel matmul: x @ output.T → [chunk, vocab]
            let output_t     = model.output_weight().transpose(0, 1)?.contiguous();
            let chunk_logits = matmul_parallel(&x, &output_t)?;

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
    use crate::model::llama::prefill::ChunkedPrefill;
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
        let rope   = RopeTable::new(cfg.context_length as usize, hd, cfg.rope_freq_base);
        let blocks: Vec<_> = (0..cfg.block_count as usize).map(|_| {
            TransformerBlock::new(
                Tensor::zeros(vec![heads*hd,  embed]),
                Tensor::zeros(vec![kv*hd,     embed]),
                Tensor::zeros(vec![kv*hd,     embed]),
                Tensor::zeros(vec![embed, heads*hd]),
                Tensor::ones(vec![embed]),
                Tensor::zeros(vec![ffn, embed]),
                Tensor::zeros(vec![ffn, embed]),
                Tensor::zeros(vec![embed, ffn]),
                Tensor::ones(vec![embed]),
                rope.clone(), heads, kv, cfg.rms_norm_eps,
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
    fn test_parallel_output_shape_single_chunk() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let p     = ParallelPrefill::new(8);
        let out   = p.run(&model, &[0, 1, 2, 3]).unwrap();
        assert_eq!(out.dims(), &[4, cfg.vocab_size as usize]);
    }

    #[test]
    fn test_parallel_output_shape_multi_chunk() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let p     = ParallelPrefill::new(3);
        let out   = p.run(&model, &[0,1,2,3,4,5,6,7]).unwrap();
        assert_eq!(out.dims(), &[8, cfg.vocab_size as usize]);
    }

    #[test]
    fn test_parallel_no_nan() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let p     = ParallelPrefill::new(2);
        let out   = p.run(&model, &[0,1,2,3,4,5]).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan() && !v.is_infinite());
        }
    }

    /// With chunk_size == seq_len, ParallelPrefill must agree with ChunkedPrefill.
    #[test]
    fn test_parallel_matches_chunked_single_chunk() {
        let cfg    = tiny_config();
        let model  = make_model(&cfg);
        let tokens = &[0u32, 3, 7, 15];

        let chunked_logits  = ChunkedPrefill::new(tokens.len()).run(&model, tokens).unwrap();
        let parallel_logits = ParallelPrefill::new(tokens.len()).run(&model, tokens).unwrap();

        assert_eq!(chunked_logits.dims(), parallel_logits.dims());
        for (i, (&c, &p)) in chunked_logits.as_slice().iter()
            .zip(parallel_logits.as_slice()).enumerate()
        {
            assert!((c - p).abs() < 1e-4,
                "logit[{i}]: chunked={c} parallel={p}");
        }
    }

    #[test]
    fn test_parallel_single_token() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let p     = ParallelPrefill::new(4);
        let out   = p.run(&model, &[5]).unwrap();
        assert_eq!(out.dims(), &[1, cfg.vocab_size as usize]);
    }

    #[test]
    fn test_empty_rejected() {
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        assert!(ParallelPrefill::new(4).run(&model, &[]).is_err());
    }

    #[test]
    fn test_exceeds_context_rejected() {
        let cfg    = tiny_config();
        let model  = make_model(&cfg);
        let tokens: Vec<u32> = (0..65).map(|i| i % 32).collect();
        assert!(ParallelPrefill::new(4).run(&model, &tokens).is_err());
    }
}
