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
pub mod matvec; // CHANGED: commit 5.3

// CHANGED: flat re-exports for the most common entry-points
pub use matmul::{matmul_naive, matmul_blocked, matmul_blocked_with_block_size};
pub use matvec::{matvec, matvec_2d}; // CHANGED: commit 5.3
