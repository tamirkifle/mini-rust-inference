//! Core neural-network operations.
//!
//! This module is the home for all compute primitives:
//!
//! | Sub-module  | Contents                              | Status      |
//! |-------------|---------------------------------------|-------------|
//! | `matmul`    | GEMM variants (naive → SIMD)          | 🔜 M2 week 5 |
//! | `norm`      | RMSNorm, LayerNorm                    | 🔜 M2 week 6 |
//! | `activation`| SiLU, SwiGLU                          | 🔜 M2 week 6 |
//! | `rope`      | Rotary Positional Embeddings          | 🔜 M2 week 6 |
//! | `softmax`   | Numerically-stable softmax            | 🔜 M2 week 6 |
//! | `fusion`    | Op-fusion infrastructure              | 🔜 M2 week 5 |

// CHANGED: declare sub-modules as they land commit-by-commit
pub mod matmul;
pub mod matvec;       // commit 5.3
pub mod fusion;       // commit 5.4
pub mod norm;         // commit 6.1
pub mod activation;   // commit 6.2
pub mod rope;         // CHANGED: commit 6.3

// CHANGED: flat re-exports for the most common entry-points
pub use matmul::{matmul_naive, matmul_blocked, matmul_blocked_with_block_size};
pub use matvec::{matvec, matvec_2d};
pub use fusion::{matmul_fused, FusedOp, BiasAdd, Activation, ActivationFn, Chain}; // commit 5.4
pub use norm::{rmsnorm, rmsnorm_inplace, DEFAULT_EPS};                    // commit 6.1
pub use activation::{silu, silu_inplace, silu_scalar, swiglu, swiglu_inplace}; // commit 6.2
pub use rope::{RopeTable, rope_apply, rope_apply_copy};                        // CHANGED: commit 6.3
