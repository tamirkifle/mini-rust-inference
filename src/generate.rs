//! Autoregressive generation loop — commit 8.5.
//!
//! Ties together `LlamaModel`, `Tokenizer`, and the sampling strategies into
//! a single `generate` call.
//!
//! # Algorithm
//!
//! ```text
//! prompt text
//!   │  tokenizer.encode()
//!   ▼
//! [bos, tok_0, tok_1, …, tok_n]  ← context
//!   │
//!   ▼  loop for max_new_tokens steps:
//!   │    logits = model.forward(&context)           [seq, vocab]
//!   │    next_id = sample(logits[last_row], config) scalar
////   │    if next_id == eos → break
//!   │    context.push(next_id)
//!   ▼
//! generated token ids (excluding prompt)
//!   │  tokenizer.decode()
//!   ▼
//! output text
//! ```
//!
//! # Complexity note
//!
//! Without a KV-cache (commit 9.x) every step re-runs the full forward pass on
//! the entire growing context: O(n²) total compute.  Correct, but slow for long
//! sequences.  The interface is identical after the KV-cache lands — only the
//! internals of `LlamaModel::forward` change.

use crate::model::llama::forward::LlamaModel;
use crate::model::{ModelError, Result};
use crate::tokenizer::Tokenizer;
use crate::sampling::{SamplingConfig, SimpleRng, sample};

/// Configuration for a generation run.
#[derive(Debug, Clone)]
pub struct GenerateConfig { // CHANGED
    /// Maximum number of new tokens to generate (not counting prompt).
    pub max_new_tokens: usize,
    /// Sampling hyper-parameters.
    pub sampling: SamplingConfig,
    /// PRNG seed (use different values for varied outputs).
    pub seed: u64,
}

impl Default for GenerateConfig {
    fn default() -> Self { // CHANGED
        Self {
            max_new_tokens: 128,
            sampling: SamplingConfig::default(),
            seed: 42,
        }
    }
}

impl GenerateConfig {
    /// Greedy (deterministic) preset with a token budget.
    #[must_use]
    pub fn greedy(max_new_tokens: usize) -> Self { // CHANGED
        Self {
            max_new_tokens,
            sampling: SamplingConfig::greedy(),
            seed: 0,
        }
    }
}

/// Generate text from `prompt` using `model` and `tokenizer`.
///
/// # Returns
///
/// The newly generated text (does **not** include the prompt).
///
/// # Errors
///
/// Returns [`ModelError`] if the forward pass or token encoding fails,
/// or if the prompt encodes to zero tokens.
pub fn generate( // CHANGED
    model: &LlamaModel,
    tokenizer: &Tokenizer,
    prompt: &str,
    config: &GenerateConfig,
) -> Result<String> {
    // ── encode prompt ──────────────────────────────────────────────────────
    let mut context: Vec<u32> = vec![tokenizer.bos_id];
    context.extend(tokenizer.encode(prompt));

    if context.len() == 1 && prompt.is_empty() {
        return Err(ModelError::InvalidConfig {
            reason: "generate: prompt encodes to zero tokens".to_string(),
        });
    }

    // ── generation loop ────────────────────────────────────────────────────
    let mut rng = SimpleRng::new(config.seed);
    let mut generated: Vec<u32> = Vec::with_capacity(config.max_new_tokens);

    for _ in 0..config.max_new_tokens {
        // Full forward pass on the current context (O(n) per step without KV-cache)
        let logits = model.forward(&context)?; // CHANGED

        // Extract last-position logits: shape [seq, vocab] → slice [vocab]
        let vocab = model.config().vocab_size as usize;
        let seq   = logits.dims()[0];
        let last_row_start = (seq - 1) * vocab;
        let last_logits = &logits.as_slice()[last_row_start..last_row_start + vocab]; // CHANGED

        // Sample next token
        let next_id = sample(last_logits, &config.sampling, &mut rng); // CHANGED

        // Stop at EOS
        if next_id == tokenizer.eos_id { break; } // CHANGED

        context.push(next_id);
        generated.push(next_id);
    }

    // ── decode generated tokens ────────────────────────────────────────────
    Ok(tokenizer.decode(&generated)) // CHANGED
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::llama::{LlamaModel, LlamaConfig, TransformerBlock};
    use crate::ops::rope::RopeTable;
    use crate::tensor::Tensor;
    use crate::gguf::{Metadata, MetadataValue};
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

    /// Minimal tokenizer: vocab [<unk>,<s>,</s>,a,b,c,d,e,f,g]
    fn make_tokenizer() -> Tokenizer {
        use crate::gguf::keys;
        let vocab: Vec<String> =
            ["<unk>","<s>","</s>","a","b","c","d","e","f","g"]
            .iter().map(|s| s.to_string()).collect();
        let scores  = vec![f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY,
                           -1.0,-1.0,-1.0,-1.0,-1.0,-1.0,-1.0];
        let types   = vec![2i32, 3, 3, 1,1,1,1,1,1,1];
        let mut m = Metadata::new();
        m.insert(keys::TOKENIZER_GGML_TOKENS.to_string(),     MetadataValue::StringArray(vocab));
        m.insert(keys::TOKENIZER_GGML_SCORES.to_string(),     MetadataValue::Float32Array(scores));
        m.insert(keys::TOKENIZER_GGML_TOKEN_TYPE.to_string(), MetadataValue::Int32Array(types));
        m.insert(keys::TOKENIZER_GGML_BOS_TOKEN_ID.to_string(), MetadataValue::Uint32(1));
        m.insert(keys::TOKENIZER_GGML_EOS_TOKEN_ID.to_string(), MetadataValue::Uint32(2));
        Tokenizer::from_metadata(&m).unwrap()
    }

    #[test]
    fn test_generate_returns_string() { // CHANGED
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let tok   = make_tokenizer();
        let cfg_g = GenerateConfig::greedy(5);
        let out   = generate(&model, &tok, "abc", &cfg_g).unwrap();
        // Just check it's a String and doesn't panic
        let _ = out;
    }

    #[test]
    fn test_generate_greedy_deterministic() { // CHANGED
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let tok   = make_tokenizer();
        let cfg_g = GenerateConfig::greedy(5);
        let a = generate(&model, &tok, "a", &cfg_g).unwrap();
        let b = generate(&model, &tok, "a", &cfg_g).unwrap();
        assert_eq!(a, b, "greedy must be deterministic");
    }

    #[test]
    fn test_generate_respects_max_new_tokens() { // CHANGED
        let cfg   = tiny_config();
        let model = make_model(&cfg);
        let tok   = make_tokenizer();
        // max_new_tokens=3; output can have at most 3 decoded tokens
        let cfg_g = GenerateConfig::greedy(3);
        let out   = generate(&model, &tok, "a", &cfg_g).unwrap();
        // Decoded output is ≤ 3 characters (each token ≤ 1 char in this tiny vocab)
        assert!(out.chars().count() <= 3);
    }

    #[test]
    fn test_generate_config_default() { // CHANGED
        let cfg = GenerateConfig::default();
        assert_eq!(cfg.max_new_tokens, 128);
    }
}
