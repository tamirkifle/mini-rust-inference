//! Generation session — commit 12.1.
//!
//! [`Session`] encapsulates a model, tokenizer, KV-cache, and sampling state
//! into a single self-contained unit.  Each session owns its own KV-cache and
//! PRNG, so multiple sessions against the same model are fully isolated.
//!
//! # Phases
//!
//! | Phase   | Method          | Description                                     |
//! |---------|-----------------|---------------------------------------------------|
//! | Prefill | (internal)      | Encode prompt tokens, write K/V into cache       |
//! | Decode  | (internal loop) | Sample one token at a time using the cache       |
//! | Reset   | `reset`         | Clear cache and restart from an empty context    |
//!
//! # Multi-turn usage
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use llm_engine::session::Session;
//! use llm_engine::config::SessionConfig;
//!
//! let mut session = Session::new(Arc::clone(&model), Arc::clone(&tok), SessionConfig::default());
//!
//! let reply1 = session.generate("What is the capital of France?")?;
//! // KV-cache now holds: [BOS, prompt tokens, reply1 tokens]
//!
//! let reply2 = session.extend("And its population?")?;
//! // KV-cache extended with follow-up and new reply
//! ```

use std::sync::Arc;

use crate::cache::KvCache;
use crate::config::SessionConfig;
use crate::model::llama::forward::LlamaModel;
use crate::model::{ModelError, Result};
use crate::ops::{matmul::matmul_blocked, norm::rmsnorm};
use crate::sampling::{sample, SimpleRng};
use crate::tokenizer::Tokenizer;

// ── Session ───────────────────────────────────────────────────────────────────

/// Stateful autoregressive generation session.
///
/// Wraps a shared model and tokenizer with a session-local KV-cache and PRNG.
/// Multiple sessions sharing the same `Arc<LlamaModel>` do not share state.
pub struct Session {
    model:     Arc<LlamaModel>,
    tokenizer: Arc<Tokenizer>,
    cache:     KvCache,
    /// Tokens currently written into `cache` (= next available write position).
    position:  usize,
    config:    SessionConfig,
    rng:       SimpleRng,
}

impl Session {
    /// Create a new session.  The KV-cache is pre-allocated for
    /// `config.max_seq_len` positions.
    #[must_use]
    pub fn new(
        model:     Arc<LlamaModel>,
        tokenizer: Arc<Tokenizer>,
        config:    SessionConfig,
    ) -> Self {
        let cfg        = model.config();
        let n_layers   = cfg.block_count as usize;
        let n_kv_heads = cfg.n_kv_heads  as usize;
        let head_dim   = cfg.head_dim()  as usize;
        let cache      = KvCache::new(n_layers, config.max_seq_len, n_kv_heads, head_dim);
        let rng        = SimpleRng::new(config.generate.seed);
        Self { model, tokenizer, cache, position: 0, config, rng }
    }

    /// Clear the KV-cache and rewind position to 0.
    ///
    /// The pre-allocated memory is retained — call this before every new
    /// conversation turn that starts from scratch.
    pub fn reset(&mut self) {
        self.cache.clear();
        self.position = 0;
        self.rng = SimpleRng::new(self.config.generate.seed);
    }

    /// Number of tokens currently held in the KV-cache.
    #[must_use]
    pub fn tokens_used(&self) -> usize { self.position }

    /// Borrow the session configuration.
    #[must_use]
    pub fn config(&self) -> &SessionConfig { &self.config }

    // ── public generation API ─────────────────────────────────────────────

    /// Start a new conversation.
    ///
    /// Resets the KV-cache, encodes `prompt` (prepending BOS), runs chunked
    /// prefill, then decodes up to `config.generate.max_new_tokens` tokens.
    ///
    /// Returns the generated text, not including the prompt.
    ///
    /// # Errors
    ///
    /// [`ModelError::InvalidConfig`] if the prompt overflows `max_seq_len`,
    /// or [`ModelError::TensorError`] on any tensor shape failure.
    pub fn generate(&mut self, prompt: &str) -> Result<String> {
        self.reset();
        let mut tokens = vec![self.tokenizer.bos_id];
        tokens.extend(self.tokenizer.encode(prompt));
        let last_logits = self.prefill(&tokens)?;
        self.decode_loop(last_logits)
    }

    /// Continue an existing conversation without resetting the KV-cache.
    ///
    /// Encodes `follow_up`, prefills it at the current cache position, then
    /// decodes a new response.  Typical use: multi-turn chat.
    ///
    /// # Errors
    ///
    /// [`ModelError::InvalidConfig`] if `follow_up` encodes to zero tokens or
    /// the sequence would overflow `max_seq_len`.
    pub fn extend(&mut self, follow_up: &str) -> Result<String> {
        let tokens: Vec<u32> = self.tokenizer.encode(follow_up);
        if tokens.is_empty() {
            return Err(ModelError::InvalidConfig {
                reason: "Session::extend: follow_up encodes to zero tokens".to_string(),
            });
        }
        let last_logits = self.prefill(&tokens)?;
        self.decode_loop(last_logits)
    }

    // ── private helpers ───────────────────────────────────────────────────

    /// Chunked prefill: write `tokens` into the KV-cache starting at
    /// `self.position` and return logits `[vocab_size]` for the **last** token.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            return Err(ModelError::InvalidConfig {
                reason: "Session::prefill: empty token list".to_string(),
            });
        }
        let max_seq = self.config.max_seq_len;
        if self.position + tokens.len() > max_seq {
            return Err(ModelError::InvalidConfig {
                reason: format!(
                    "Session: {} + {} tokens would exceed max_seq_len {}",
                    self.position, tokens.len(), max_seq,
                ),
            });
        }

        let vocab      = self.model.config().vocab_size as usize;
        let chunk_size = self.config.chunk_size;
        let mut last_logits = vec![0.0_f32; vocab];
        let mut chunk_start = 0_usize;

        while chunk_start < tokens.len() {
            let chunk_end = (chunk_start + chunk_size).min(tokens.len());
            let chunk     = &tokens[chunk_start..chunk_end];
            let cache_pos = self.position + chunk_start;

            // 1. Embed chunk tokens → [chunk_len, embed_dim]
            let mut x = self.model.embed_tokens(chunk)?;

            // 2. All transformer blocks (cached prefill path)
            for (layer_idx, block) in self.model.blocks().iter().enumerate() {
                x = block.forward_cached(&x, cache_pos, &mut self.cache, layer_idx)?;
            }

            // 3. Final norm + unembedding → logits [chunk_len, vocab]
            let cfg    = self.model.config();
            let normed = rmsnorm(&x, self.model.output_norm(), cfg.rms_norm_eps)?;
            let out_t  = self.model.output_weight().transpose(0, 1)?.contiguous();
            let logits = matmul_blocked(&normed, &out_t)?;

            // Keep the last token's logit row for sampling
            let last_start = (chunk.len() - 1) * vocab;
            last_logits = logits.as_slice()[last_start..last_start + vocab].to_vec();

            chunk_start = chunk_end;
        }

        self.position += tokens.len();
        Ok(last_logits)
    }

    /// Autoregressive decode loop.
    ///
    /// Consumes `last_logits` (from prefill or the previous step) and emits
    /// at most `config.generate.max_new_tokens` tokens, stopping at EOS or
    /// when the cache is full.
    fn decode_loop(&mut self, mut last_logits: Vec<f32>) -> Result<String> {
        let max_new = self.config.generate.max_new_tokens;
        let max_seq = self.config.max_seq_len;
        let mut generated: Vec<u32> = Vec::with_capacity(max_new);

        for _ in 0..max_new {
            if self.position >= max_seq { break; }

            let next_id = sample(
                &last_logits,
                &self.config.generate.sampling,
                &mut self.rng,
            );
            if next_id == self.tokenizer.eos_id { break; }

            generated.push(next_id);

            last_logits = self.model
                .forward_decode(next_id, self.position, &mut self.cache)?;
            self.position += 1;
        }

        Ok(self.tokenizer.decode(&generated))
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SessionConfig;
    use crate::generate::GenerateConfig;
    use crate::gguf::{Metadata, MetadataValue};
    use crate::model::llama::{LlamaConfig, LlamaModel, TransformerBlock};
    use crate::ops::rope::RopeTable;
    use crate::tensor::Tensor;
    use crate::tokenizer::bpe::Tokenizer;

    fn tiny_config() -> LlamaConfig {
        let mut m = Metadata::new();
        for (k, v) in [
            ("llama.block_count",          MetadataValue::Uint32(1)),
            ("llama.embedding_length",     MetadataValue::Uint32(8)),
            ("llama.attention.head_count", MetadataValue::Uint32(2)),
            ("llama.feed_forward_length",  MetadataValue::Uint32(16)),
            ("llama.context_length",       MetadataValue::Uint32(64)),
            ("llama.vocab_size",           MetadataValue::Uint32(10)),
        ] { m.insert(k.to_string(), v); }
        LlamaConfig::from_metadata(&m).unwrap()
    }

    fn make_model(cfg: &LlamaConfig) -> LlamaModel {
        let embed = cfg.embedding_length as usize;
        let vocab = cfg.vocab_size as usize;
        let ffn   = cfg.feed_forward_length as usize;
        let heads = cfg.n_heads as usize;
        let kv    = cfg.n_kv_heads as usize;
        let hd    = cfg.head_dim() as usize;
        let rope  = RopeTable::new(cfg.context_length as usize, hd, cfg.rope_freq_base);
        let block = TransformerBlock::new(
            Tensor::zeros(vec![heads*hd, embed]), Tensor::zeros(vec![kv*hd, embed]),
            Tensor::zeros(vec![kv*hd, embed]),    Tensor::zeros(vec![embed, heads*hd]),
            Tensor::ones(vec![embed]),
            Tensor::zeros(vec![ffn, embed]),       Tensor::zeros(vec![ffn, embed]),
            Tensor::zeros(vec![embed, ffn]),       Tensor::ones(vec![embed]),
            rope, heads, kv, cfg.rms_norm_eps,
        );
        LlamaModel::new(cfg.clone(),
            Tensor::zeros(vec![vocab, embed]), vec![block],
            Tensor::ones(vec![embed]), Tensor::zeros(vec![vocab, embed]))
    }

    fn make_tokenizer() -> Tokenizer {
        use crate::gguf::keys;
        let vocab: Vec<String> =
            ["<unk>","<s>","</s>","a","b","c","d","e","f","g"]
            .iter().map(|s| s.to_string()).collect();
        let scores = vec![
            f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY,
            -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
        ];
        let types = vec![2i32, 3, 3, 1, 1, 1, 1, 1, 1, 1];
        let mut m = Metadata::new();
        m.insert(keys::TOKENIZER_GGML_TOKENS.to_string(),
            MetadataValue::StringArray(vocab));
        m.insert(keys::TOKENIZER_GGML_SCORES.to_string(),
            MetadataValue::Float32Array(scores));
        m.insert(keys::TOKENIZER_GGML_TOKEN_TYPE.to_string(),
            MetadataValue::Int32Array(types));
        m.insert(keys::TOKENIZER_GGML_BOS_TOKEN_ID.to_string(),
            MetadataValue::Uint32(1));
        m.insert(keys::TOKENIZER_GGML_EOS_TOKEN_ID.to_string(),
            MetadataValue::Uint32(2));
        Tokenizer::from_metadata(&m).unwrap()
    }

    fn make_session(max_new_tokens: usize) -> Session {
        let cfg   = tiny_config();
        let model = Arc::new(make_model(&cfg));
        let tok   = Arc::new(make_tokenizer());
        Session::new(model, tok, SessionConfig::new(GenerateConfig::greedy(max_new_tokens), 4, 64))
    }

    #[test]
    fn test_session_generate_returns_string() {
        let mut s = make_session(5);
        let _ = s.generate("abc").unwrap();
    }

    #[test]
    fn test_session_greedy_deterministic() {
        let cfg   = tiny_config();
        let model = Arc::new(make_model(&cfg));
        let tok   = Arc::new(make_tokenizer());
        let scfg  = SessionConfig::new(GenerateConfig::greedy(5), 4, 64);
        let mut s1 = Session::new(Arc::clone(&model), Arc::clone(&tok), scfg.clone());
        let mut s2 = Session::new(Arc::clone(&model), Arc::clone(&tok), scfg);
        assert_eq!(s1.generate("a").unwrap(), s2.generate("a").unwrap(),
            "greedy sessions must be deterministic");
    }

    #[test]
    fn test_session_generate_resets_between_calls() {
        let mut s = make_session(5);
        let r1 = s.generate("a").unwrap();
        let r2 = s.generate("a").unwrap(); // generate() calls reset internally
        assert_eq!(r1, r2, "repeated generate must produce identical output");
    }

    #[test]
    fn test_session_tokens_used_advances_after_generate() {
        let mut s = make_session(3);
        assert_eq!(s.tokens_used(), 0);
        s.generate("a").unwrap();
        assert!(s.tokens_used() > 0, "tokens_used should be > 0 after generate");
    }

    #[test]
    fn test_session_extend_advances_position_further() {
        let mut s  = make_session(2);
        s.generate("a").unwrap();
        let pos_after_generate = s.tokens_used();
        s.extend("b").unwrap();
        assert!(s.tokens_used() > pos_after_generate,
            "extend must advance position beyond the post-generate position");
    }

    #[test]
    fn test_multiple_sessions_isolated() {
        let cfg   = tiny_config();
        let model = Arc::new(make_model(&cfg));
        let tok   = Arc::new(make_tokenizer());
        let scfg  = SessionConfig::new(GenerateConfig::greedy(3), 4, 64);
        let mut s1 = Session::new(Arc::clone(&model), Arc::clone(&tok), scfg.clone());
        let mut s2 = Session::new(Arc::clone(&model), Arc::clone(&tok), scfg);
        // Separate runs — session caches must not bleed into each other
        let r1 = s1.generate("abc").unwrap();
        let r2 = s2.generate("abc").unwrap();
        assert_eq!(r1, r2, "isolated sessions with same input must agree");
        assert_eq!(s1.tokens_used(), s2.tokens_used());
    }

    #[test]
    fn test_session_respects_max_new_tokens() {
        let mut s = make_session(2);
        let out   = s.generate("a").unwrap();
        assert!(out.chars().count() <= 2,
            "output must not exceed max_new_tokens=2 chars, got {:?}", out);
    }

    #[test]
    fn test_session_config_accessor() {
        let s = make_session(10);
        assert_eq!(s.config().max_seq_len, 64);
        assert_eq!(s.config().chunk_size,   4);
    }
}
