//! Normalisation operations.
//!
//! | Module      | Op                          | Status     |
//! |-------------|---------------------------  |------------|
//! | `rmsnorm`   | Root Mean Square LayerNorm  | ✅ commit 6.1 |

pub mod rmsnorm;       // commit 6.1
pub mod rmsnorm_simd;  // commit 15.3 — SIMD-accelerated RMSNorm

pub use rmsnorm::{rmsnorm, rmsnorm_inplace, DEFAULT_EPS};
pub use rmsnorm_simd::{rmsnorm_simd, rmsnorm_simd_inplace};
