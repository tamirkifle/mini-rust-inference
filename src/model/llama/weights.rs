//! GGUF tensor name mapping for Llama model weights.
//!
//! Maps logical weight roles to the actual tensor names stored in GGUF files.
//!
//! # Naming convention
//!
//! Per-layer weights follow the pattern `blk.{layer}.{role}.weight`:
//!
//! ```text
//! blk.0.attn_q.weight      ← query projection, layer 0
//! blk.0.attn_k.weight      ← key projection, layer 0
//! blk.31.ffn_down.weight   ← FFN down-projection, layer 31
//! ```
//!
//! Global weights (no layer index):
//!
//! ```text
//! token_embd.weight    ← token embedding table
//! output_norm.weight   ← final RMSNorm scale
//! output.weight        ← LM head (unembedding)
//! ```

// ── per-layer weight roles ───────────────────────────────────────────────────

/// Logical role of a per-layer weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightRole {
    /// Query projection matrix `W_Q`.
    AttnQ,
    /// Key projection matrix `W_K`.
    AttnK,
    /// Value projection matrix `W_V`.
    AttnV,
    /// Attention output projection `W_O`.
    AttnOutput,
    /// Pre-attention RMSNorm scale vector.
    AttnNorm,
    /// Pre-FFN RMSNorm scale vector.
    FfnNorm,
    /// FFN gate projection (SwiGLU gate branch).
    FfnGate,
    /// FFN up projection (SwiGLU up branch).
    FfnUp,
    /// FFN down projection.
    FfnDown,
}

/// Returns the GGUF tensor name for a per-layer weight.
///
/// # Examples
///
/// ```
/// use llm_engine::model::llama::weights::{weight_name, WeightRole};
/// assert_eq!(weight_name(3, WeightRole::AttnQ),     "blk.3.attn_q.weight");
/// assert_eq!(weight_name(0, WeightRole::FfnDown),   "blk.0.ffn_down.weight");
/// assert_eq!(weight_name(31, WeightRole::AttnNorm), "blk.31.attn_norm.weight");
/// ```
#[must_use]
pub fn weight_name(layer: usize, role: WeightRole) -> String { // CHANGED
    let suffix = match role {
        WeightRole::AttnQ      => "attn_q.weight",
        WeightRole::AttnK      => "attn_k.weight",
        WeightRole::AttnV      => "attn_v.weight",
        WeightRole::AttnOutput => "attn_output.weight",
        WeightRole::AttnNorm   => "attn_norm.weight",
        WeightRole::FfnNorm    => "ffn_norm.weight",
        WeightRole::FfnGate    => "ffn_gate.weight",
        WeightRole::FfnUp      => "ffn_up.weight",
        WeightRole::FfnDown    => "ffn_down.weight",
    };
    format!("blk.{layer}.{suffix}")
}

// ── global (non-layer) weight roles ─────────────────────────────────────────

/// Logical role of a global (non-per-layer) weight tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GlobalWeightRole {
    /// Token embedding table `[vocab_size, embed_dim]`.
    TokenEmbd,
    /// Final RMSNorm scale vector `[embed_dim]`.
    OutputNorm,
    /// LM-head unembedding matrix `[vocab_size, embed_dim]`.
    Output,
}

/// Returns the GGUF tensor name for a global weight.
///
/// # Examples
///
/// ```
/// use llm_engine::model::llama::weights::{global_weight_name, GlobalWeightRole};
/// assert_eq!(global_weight_name(GlobalWeightRole::TokenEmbd),  "token_embd.weight");
/// assert_eq!(global_weight_name(GlobalWeightRole::OutputNorm), "output_norm.weight");
/// assert_eq!(global_weight_name(GlobalWeightRole::Output),     "output.weight");
/// ```
#[must_use]
pub fn global_weight_name(role: GlobalWeightRole) -> &'static str { // CHANGED
    match role {
        GlobalWeightRole::TokenEmbd  => "token_embd.weight",
        GlobalWeightRole::OutputNorm => "output_norm.weight",
        GlobalWeightRole::Output     => "output.weight",
    }
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_name_layer_0() { // CHANGED
        assert_eq!(weight_name(0, WeightRole::AttnQ),      "blk.0.attn_q.weight");
        assert_eq!(weight_name(0, WeightRole::AttnK),      "blk.0.attn_k.weight");
        assert_eq!(weight_name(0, WeightRole::AttnV),      "blk.0.attn_v.weight");
        assert_eq!(weight_name(0, WeightRole::AttnOutput), "blk.0.attn_output.weight");
        assert_eq!(weight_name(0, WeightRole::AttnNorm),   "blk.0.attn_norm.weight");
        assert_eq!(weight_name(0, WeightRole::FfnNorm),    "blk.0.ffn_norm.weight");
        assert_eq!(weight_name(0, WeightRole::FfnGate),    "blk.0.ffn_gate.weight");
        assert_eq!(weight_name(0, WeightRole::FfnUp),      "blk.0.ffn_up.weight");
        assert_eq!(weight_name(0, WeightRole::FfnDown),    "blk.0.ffn_down.weight");
    }

    #[test]
    fn test_weight_name_layer_31() { // CHANGED
        assert_eq!(weight_name(31, WeightRole::AttnQ),    "blk.31.attn_q.weight");
        assert_eq!(weight_name(31, WeightRole::FfnDown),  "blk.31.ffn_down.weight");
        assert_eq!(weight_name(31, WeightRole::AttnNorm), "blk.31.attn_norm.weight");
    }

    #[test]
    fn test_weight_name_arbitrary_layer() { // CHANGED
        for layer in [0, 1, 7, 15, 31, 63, 100] {
            let name = weight_name(layer, WeightRole::AttnQ);
            assert!(name.starts_with(&format!("blk.{layer}.")));
            assert!(name.ends_with(".weight"));
        }
    }

    #[test]
    fn test_global_weight_names() { // CHANGED
        assert_eq!(global_weight_name(GlobalWeightRole::TokenEmbd),  "token_embd.weight");
        assert_eq!(global_weight_name(GlobalWeightRole::OutputNorm), "output_norm.weight");
        assert_eq!(global_weight_name(GlobalWeightRole::Output),     "output.weight");
    }

    #[test]
    fn test_weight_name_round_trip() { // CHANGED: each role produces a unique name at the same layer
        use std::collections::HashSet;
        let roles = [
            WeightRole::AttnQ, WeightRole::AttnK, WeightRole::AttnV,
            WeightRole::AttnOutput, WeightRole::AttnNorm, WeightRole::FfnNorm,
            WeightRole::FfnGate, WeightRole::FfnUp, WeightRole::FfnDown,
        ];
        let names: HashSet<String> = roles.iter().map(|&r| weight_name(0, r)).collect();
        assert_eq!(names.len(), roles.len(), "all roles must produce distinct names");
    }
}
