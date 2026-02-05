//! Llama model architecture (Llama 2 / Llama 3 style transformer).

// CHANGED: commit 8.0 — config and weight name mapping
pub mod config;
pub mod weights;

// CHANGED: commit 8.1 — single transformer block
pub mod block;

// Re-exports for convenience
pub use config::LlamaConfig;
pub use weights::{GlobalWeightRole, WeightRole, global_weight_name, weight_name};
pub use block::TransformerBlock; // CHANGED: commit 8.1
