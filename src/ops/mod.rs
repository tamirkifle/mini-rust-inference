//! Core neural-network operations.
//!
//! This module is the home for all compute primitives:
//!
//! | Sub-module    | Contents                              | Status      |
//! |---------------|---------------------------------------|-------------|
//! | `matmul`      | GEMM variants (naive → SIMD)          | ✅ M2 week 5 |
//! | `norm`        | RMSNorm, LayerNorm                    | ✅ M2 week 6 |
//! | `activation`  | SiLU, SwiGLU                          | ✅ M2 week 6 |
//! | `rope`        | Rotary Positional Embeddings          | ✅ M2 week 6 |
//! | `rope_scaled` | Extended-context RoPE scaling         | ✅ M3 week 11 |
//! | `softmax`     | Numerically-stable softmax            | ✅ M2 week 6 |
//! | `fusion`      | Op-fusion infrastructure              | ✅ M2 week 5 |

pub mod matmul;
pub mod matvec;       // commit 5.3
pub mod fusion;       // commit 5.4
pub mod norm;         // commit 6.1
pub mod activation;   // commit 6.2
pub mod rope;         // commit 6.3
pub mod rope_scaled;  // commit 11.1
pub mod softmax;      // commit 6.4
pub mod softmax_simd; // commit 15.3 — SIMD-accelerated softmax

pub use matmul::{matmul_naive, matmul_blocked, matmul_blocked_with_block_size};
pub use matvec::{matvec, matvec_2d};
pub use fusion::{matmul_fused, FusedOp, BiasAdd, Activation, ActivationFn, Chain};
pub use norm::{rmsnorm, rmsnorm_inplace, DEFAULT_EPS, rmsnorm_simd, rmsnorm_simd_inplace};
pub use activation::{silu, silu_inplace, silu_scalar, swiglu, swiglu_inplace};
pub use rope::{RopeTable, rope_apply, rope_apply_copy};
pub use rope_scaled::{RopeScaling, ScaledRopeTable, rope_apply_scaled, rope_apply_scaled_copy};
pub use softmax::{softmax, softmax_dim, softmax_inplace};
pub use softmax_simd::{softmax_simd, softmax_simd_dim, softmax_simd_inplace};
