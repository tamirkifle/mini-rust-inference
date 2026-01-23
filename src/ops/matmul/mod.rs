//! Matrix multiplication operations.
//!
//! # Implementations
//!
//! | Module    | Strategy               | Status      |
//! |-----------|------------------------|-------------|
//! | `naive`   | Triple-nested loop     | ✅ commit 5.1 |
//! | `blocked` | Cache-tiled GEMM       | 🔜 commit 5.2 |
//!
//! All implementations must match `matmul_naive` within `1e-5` absolute error.

pub mod naive;

// CHANGED: re-export the canonical naive entry-point at this level
pub use naive::matmul_naive;
