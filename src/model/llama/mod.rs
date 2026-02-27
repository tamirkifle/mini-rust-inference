//! Llama model architecture (Llama 2 / Llama 3 style transformer).

// CHANGED: commit 8.0 — config and weight name mapping
pub mod config;
pub mod weights;

// CHANGED: commit 8.1 — single transformer block
pub mod block;

// CHANGED: commit 8.2 — full model forward pass
pub mod forward;

// CHANGED: commit 11.2 — chunked prefill
pub mod prefill;

// CHANGED: commit 12.2 — parallel chunked prefill (rayon matmul)
pub mod parallel_prefill;

// CHANGED: commit 14.1 — mixed-precision forward pass (INT8 weights, f32 activations)
pub mod forward_int8;

// Re-exports for convenience
pub use config::LlamaConfig;
pub use weights::{GlobalWeightRole, WeightRole, global_weight_name, weight_name};
pub use block::TransformerBlock;
pub use forward::LlamaModel;
pub use forward_int8::{LlamaModelInt8, TransformerBlockInt8};
pub use prefill::ChunkedPrefill;
pub use parallel_prefill::ParallelPrefill;
