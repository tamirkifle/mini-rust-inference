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
pub mod blocked; // CHANGED: commit 5.2

// CHANGED: re-export both kernels at this level
pub use naive::matmul_naive;
pub use blocked::{matmul_blocked, matmul_blocked_with_block_size, DEFAULT_BLOCK_SIZE};
