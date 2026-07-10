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

pub mod activation; // commit 6.2
pub mod fusion; // commit 5.4
pub mod matmul;
pub mod matvec; // commit 5.3
pub mod norm; // commit 6.1
pub mod rope; // commit 6.3
pub mod rope_scaled; // commit 11.1
pub mod softmax; // commit 6.4
pub mod softmax_simd; // commit 15.3 — SIMD-accelerated softmax

pub use activation::{silu, silu_inplace, silu_scalar, swiglu, swiglu_inplace};
pub use fusion::{matmul_fused, Activation, ActivationFn, BiasAdd, Chain, FusedOp};
pub use matmul::{matmul_blocked, matmul_blocked_with_block_size, matmul_naive};
pub use matvec::{matvec, matvec_2d};
pub use norm::{rmsnorm, rmsnorm_inplace, rmsnorm_simd, rmsnorm_simd_inplace, DEFAULT_EPS};
pub use rope::{rope_apply, rope_apply_copy, RopeTable};
pub use rope_scaled::{rope_apply_scaled, rope_apply_scaled_copy, RopeScaling, ScaledRopeTable};
pub use softmax::{softmax, softmax_dim, softmax_inplace};
pub use softmax_simd::{softmax_simd, softmax_simd_dim, softmax_simd_inplace};
