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
pub mod blocked;
pub mod parallel; // commit 12.2 — row-parallel GEMM via rayon

pub use naive::matmul_naive;
pub use blocked::{matmul_blocked, matmul_blocked_with_block_size, DEFAULT_BLOCK_SIZE};
pub use parallel::matmul_parallel;
