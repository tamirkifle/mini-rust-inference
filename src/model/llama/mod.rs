//! Llama model architecture (Llama 2 / Llama 3 style transformer).

// CHANGED: commit 8.0 — config and weight name mapping
pub mod config;
pub mod weights;

// CHANGED: commit 8.1 — single transformer block
pub mod block;

// CHANGED: commit 8.2 — full model forward pass
pub mod forward;

// Re-exports for convenience
pub use config::LlamaConfig;
pub use weights::{GlobalWeightRole, WeightRole, global_weight_name, weight_name};
pub use block::TransformerBlock;
pub use forward::LlamaModel; // CHANGED: commit 8.2
