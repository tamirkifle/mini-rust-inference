//! Session-level configuration — commit 12.1.
//!
//! [`SessionConfig`] bundles sampling parameters, chunked-prefill window size,
//! and the sequence-length budget into one struct so each [`Session`] is
//! fully self-contained and isolated from other sessions.
//!
//! [`Session`]: crate::session::Session

use crate::generate::GenerateConfig;

/// Configuration for a single generation session.
///
/// Passed to [`Session::new`] at construction time.  Each session owns a copy,
/// so changing a config after construction does not affect running sessions.
///
/// # Defaults
///
/// ```
/// use llm_engine::config::SessionConfig;
/// let cfg = SessionConfig::default();
/// assert_eq!(cfg.chunk_size,  512);
/// assert_eq!(cfg.max_seq_len, 2048);
/// ```
///
/// [`Session::new`]: crate::session::Session::new
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Sampling hyper-parameters and token budget.
    pub generate: GenerateConfig,
    /// Number of tokens processed per prefill chunk.
    ///
    /// Smaller values reduce peak activation memory; larger values may be
    /// faster thanks to better matrix utilisation.  Must be ≥ 1.
    pub chunk_size: usize,
    /// Maximum total sequence length (prompt + generated tokens combined).
    ///
    /// Determines KV-cache pre-allocation size.  Must not exceed the model's
    /// `context_length`.
    pub max_seq_len: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            generate:    GenerateConfig::default(),
            chunk_size:  512,
            max_seq_len: 2048,
        }
    }
}

impl SessionConfig {
    /// Construct with explicit parameters.
    #[must_use]
    pub fn new(generate: GenerateConfig, chunk_size: usize, max_seq_len: usize) -> Self {
        assert!(chunk_size > 0,  "SessionConfig: chunk_size must be ≥ 1");
        assert!(max_seq_len > 0, "SessionConfig: max_seq_len must be ≥ 1");
        Self { generate, chunk_size, max_seq_len }
    }

    /// Greedy-decoding preset.
    ///
    /// Uses `temperature = 0`, no top-k or top-p, and the supplied token budget.
    #[must_use]
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            generate:    GenerateConfig::greedy(max_new_tokens),
            chunk_size:  512,
            max_seq_len: 2048,
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_values() {
        let cfg = SessionConfig::default();
        assert_eq!(cfg.chunk_size,  512);
        assert_eq!(cfg.max_seq_len, 2048);
        assert_eq!(cfg.generate.max_new_tokens, 128);
    }

    #[test]
    fn test_greedy_preset() {
        let cfg = SessionConfig::greedy(64);
        assert_eq!(cfg.generate.max_new_tokens, 64);
        assert_eq!(cfg.generate.sampling.temperature, 0.0);
        assert_eq!(cfg.chunk_size,  512);
        assert_eq!(cfg.max_seq_len, 2048);
    }

    #[test]
    fn test_new_explicit() {
        use crate::generate::GenerateConfig;
        let cfg = SessionConfig::new(GenerateConfig::greedy(10), 32, 256);
        assert_eq!(cfg.chunk_size,  32);
        assert_eq!(cfg.max_seq_len, 256);
        assert_eq!(cfg.generate.max_new_tokens, 10);
    }
}
