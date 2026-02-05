//! Llama model configuration, populated from GGUF metadata.
//!
//! GGUF stores all hyper-parameters as metadata key-value pairs.
//! `LlamaConfig::from_metadata` extracts and validates the fields that the
//! forward-pass code needs.
//!
//! # Required GGUF keys
//!
//! | Key | Type | Notes |
//! |-----|------|-------|
//! | `llama.block_count` | u32 | Number of transformer layers |
//! | `llama.embedding_length` | u32 | Model hidden dim (d_model) |
//! | `llama.attention.head_count` | u32 | Number of Q heads |
//! | `llama.feed_forward_length` | u32 | FFN intermediate dim |
//! | `llama.context_length` | u32 | Maximum sequence length |
//!
//! # Optional GGUF keys (with defaults)
//!
//! | Key | Default | Notes |
//! |-----|---------|-------|
//! | `llama.attention.head_count_kv` | = n_heads | GQA KV-head count |
//! | `llama.vocab_size` | from token list | Vocabulary size |
//! | `llama.rope.freq_base` | 10 000.0 | RoPE base frequency |
//! | `llama.attention.layer_norm_rms_epsilon` | 1e-5 | RMSNorm epsilon |

use crate::gguf::Metadata;
use crate::model::{ModelError, Result};

/// Hyper-parameters for a Llama-style model.
#[derive(Debug, Clone)]
pub struct LlamaConfig {
    /// Number of transformer blocks (layers).
    pub block_count: u32,
    /// Hidden dimension / embedding size.
    pub embedding_length: u32,
    /// Number of query attention heads.
    pub n_heads: u32,
    /// Number of key-value heads (for GQA/MQA; equals `n_heads` for standard MHA).
    pub n_kv_heads: u32,
    /// Feed-forward intermediate dimension.
    pub feed_forward_length: u32,
    /// Maximum context length (sequence positions).
    pub context_length: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// RoPE base frequency (default: 10 000.0).
    pub rope_freq_base: f32,
    /// RMSNorm epsilon (default: 1e-5).
    pub rms_norm_eps: f32,
}

impl LlamaConfig {
    /// Dimension of each attention head.
    #[must_use]
    pub fn head_dim(&self) -> u32 {
        self.embedding_length / self.n_heads // CHANGED
    }

    /// Construct a `LlamaConfig` from parsed GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::MissingMetadataKey`] for any required key that is
    /// absent, or [`ModelError::InvalidConfig`] if values are internally
    /// inconsistent (e.g., `n_heads` doesn't divide `embedding_length`).
    pub fn from_metadata(meta: &Metadata) -> Result<Self> {
        // ── required fields ────────────────────────────────────────────────
        let block_count = get_u32_wide(meta, "llama.block_count") // CHANGED
            .ok_or(ModelError::MissingMetadataKey { key: "llama.block_count" })?;

        let embedding_length = get_u32_wide(meta, "llama.embedding_length") // CHANGED
            .ok_or(ModelError::MissingMetadataKey { key: "llama.embedding_length" })?;

        let n_heads = get_u32_wide(meta, "llama.attention.head_count") // CHANGED
            .ok_or(ModelError::MissingMetadataKey { key: "llama.attention.head_count" })?;

        let feed_forward_length = get_u32_wide(meta, "llama.feed_forward_length") // CHANGED
            .ok_or(ModelError::MissingMetadataKey { key: "llama.feed_forward_length" })?;

        let context_length = get_u32_wide(meta, "llama.context_length") // CHANGED
            .ok_or(ModelError::MissingMetadataKey { key: "llama.context_length" })?;

        // ── optional / derived fields ──────────────────────────────────────
        // n_kv_heads: falls back to n_heads (standard MHA) if not present
        let n_kv_heads = get_u32_wide(meta, "llama.attention.head_count_kv") // CHANGED
            .unwrap_or(n_heads);

        // vocab_size: try direct key, then count the token list
        let vocab_size = get_u32_wide(meta, "llama.vocab_size") // CHANGED
            .or_else(|| {
                meta.get("tokenizer.ggml.tokens")
                    .and_then(|v| v.as_string_array())
                    .map(|a| a.len() as u32)
            })
            .ok_or(ModelError::MissingMetadataKey { key: "llama.vocab_size" })?;

        let rope_freq_base = meta.get_f32("llama.rope.freq_base") // CHANGED
            .unwrap_or(10_000.0);

        let rms_norm_eps = meta.get_f32("llama.attention.layer_norm_rms_epsilon") // CHANGED
            .unwrap_or(1e-5);

        // ── validate invariants ────────────────────────────────────────────
        if n_heads == 0 {
            return Err(ModelError::InvalidConfig {
                reason: "n_heads must be > 0".to_string(),
            });
        }
        if embedding_length % n_heads != 0 {
            return Err(ModelError::InvalidConfig {
                reason: format!(
                    "embedding_length {embedding_length} not divisible by n_heads {n_heads}"
                ),
            });
        }
        if n_kv_heads == 0 || n_heads % n_kv_heads != 0 {
            return Err(ModelError::InvalidConfig {
                reason: format!(
                    "n_heads {n_heads} not divisible by n_kv_heads {n_kv_heads}"
                ),
            });
        }

        Ok(Self { // CHANGED
            block_count,
            embedding_length,
            n_heads,
            n_kv_heads,
            feed_forward_length,
            context_length,
            vocab_size,
            rope_freq_base,
            rms_norm_eps,
        })
    }
}

// ── private helpers ──────────────────────────────────────────────────────────

/// Read a metadata value as u32, accepting u8/u16/u32/u64 (with range check).
fn get_u32_wide(meta: &Metadata, key: &str) -> Option<u32> { // CHANGED
    meta.get_u32(key)
        .or_else(|| meta.get_u64(key).and_then(|v| u32::try_from(v).ok()))
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::MetadataValue;

    fn make_meta(entries: &[(&str, MetadataValue)]) -> Metadata {
        let mut m = Metadata::new();
        for (k, v) in entries {
            m.insert((*k).to_string(), (*v).clone());
        }
        m
    }

    fn minimal_entries() -> Vec<(&'static str, MetadataValue)> {
        vec![
            ("llama.block_count",         MetadataValue::Uint32(32)),
            ("llama.embedding_length",    MetadataValue::Uint32(4096)),
            ("llama.attention.head_count",MetadataValue::Uint32(32)),
            ("llama.feed_forward_length", MetadataValue::Uint32(11008)),
            ("llama.context_length",      MetadataValue::Uint32(4096)),
            ("llama.vocab_size",          MetadataValue::Uint32(32000)),
        ]
    }

    #[test]
    fn test_config_from_minimal_metadata() { // CHANGED
        let meta = make_meta(&minimal_entries());
        let cfg = LlamaConfig::from_metadata(&meta).unwrap();

        assert_eq!(cfg.block_count,         32);
        assert_eq!(cfg.embedding_length,    4096);
        assert_eq!(cfg.n_heads,             32);
        assert_eq!(cfg.n_kv_heads,          32); // defaults to n_heads
        assert_eq!(cfg.feed_forward_length, 11008);
        assert_eq!(cfg.context_length,      4096);
        assert_eq!(cfg.vocab_size,          32000);
        assert!((cfg.rope_freq_base - 10_000.0).abs() < 1e-6);
        assert!((cfg.rms_norm_eps   - 1e-5).abs()     < 1e-10);
    }

    #[test]
    fn test_config_head_dim() { // CHANGED
        let meta = make_meta(&minimal_entries());
        let cfg = LlamaConfig::from_metadata(&meta).unwrap();
        assert_eq!(cfg.head_dim(), 128); // 4096 / 32
    }

    #[test]
    fn test_config_gqa_kv_heads() { // CHANGED
        let mut entries = minimal_entries();
        entries.push(("llama.attention.head_count_kv", MetadataValue::Uint32(8)));
        let meta = make_meta(&entries);
        let cfg = LlamaConfig::from_metadata(&meta).unwrap();
        assert_eq!(cfg.n_kv_heads, 8);
    }

    #[test]
    fn test_config_custom_rope_base() { // CHANGED
        let mut entries = minimal_entries();
        entries.push(("llama.rope.freq_base", MetadataValue::Float32(500_000.0)));
        let meta = make_meta(&entries);
        let cfg = LlamaConfig::from_metadata(&meta).unwrap();
        assert!((cfg.rope_freq_base - 500_000.0).abs() < 1.0);
    }

    #[test]
    fn test_config_custom_rms_eps() { // CHANGED
        let mut entries = minimal_entries();
        entries.push(("llama.attention.layer_norm_rms_epsilon", MetadataValue::Float32(1e-6)));
        let meta = make_meta(&entries);
        let cfg = LlamaConfig::from_metadata(&meta).unwrap();
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-12);
    }

    #[test]
    fn test_config_vocab_from_token_list() { // CHANGED
        // No llama.vocab_size — should be derived from tokenizer.ggml.tokens length
        let tokens: Vec<String> = (0..500).map(|i| format!("tok{i}")).collect();
        let mut entries = minimal_entries();
        // Remove vocab_size entry
        entries.retain(|(k, _)| *k != "llama.vocab_size");
        entries.push(("tokenizer.ggml.tokens", MetadataValue::StringArray(tokens)));
        let meta = make_meta(&entries);
        let cfg = LlamaConfig::from_metadata(&meta).unwrap();
        assert_eq!(cfg.vocab_size, 500);
    }

    #[test]
    fn test_config_missing_required_key() { // CHANGED
        let meta = Metadata::new(); // empty
        let result = LlamaConfig::from_metadata(&meta);
        assert!(matches!(result, Err(ModelError::MissingMetadataKey { .. })));
    }

    #[test]
    fn test_config_invalid_heads_divisibility() { // CHANGED
        let mut entries = minimal_entries();
        // embedding_length = 4096, n_heads = 7 → 4096 % 7 != 0
        entries.retain(|(k, _)| *k != "llama.attention.head_count");
        entries.push(("llama.attention.head_count", MetadataValue::Uint32(7)));
        let meta = make_meta(&entries);
        let result = LlamaConfig::from_metadata(&meta);
        assert!(matches!(result, Err(ModelError::InvalidConfig { .. })));
    }

    #[test]
    fn test_config_invalid_kv_head_divisibility() { // CHANGED
        let mut entries = minimal_entries();
        // n_heads=32, n_kv_heads=5 → 32 % 5 != 0
        entries.push(("llama.attention.head_count_kv", MetadataValue::Uint32(5)));
        let meta = make_meta(&entries);
        let result = LlamaConfig::from_metadata(&meta);
        assert!(matches!(result, Err(ModelError::InvalidConfig { .. })));
    }

    #[test]
    fn test_config_u64_vocab_size() { // CHANGED: some GGUF files store as uint64
        let mut entries = minimal_entries();
        entries.retain(|(k, _)| *k != "llama.vocab_size");
        entries.push(("llama.vocab_size", MetadataValue::Uint64(32000)));
        let meta = make_meta(&entries);
        let cfg = LlamaConfig::from_metadata(&meta).unwrap();
        assert_eq!(cfg.vocab_size, 32000);
    }
}
