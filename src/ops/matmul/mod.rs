//! Matrix multiplication operations.
//!
//! # Implementations
//!
//! | Module    | Strategy                             | Commit |
//! |-----------|--------------------------------------|--------|
//! | `naive`   | Triple-nested ikj loop (reference)   | 5.1    |
//! | `blocked` | Cache-tiled GEMM (32×32 tiles)       | 5.2    |
//! | `parallel`| Row-parallel GEMM via rayon          | 12.2   |
//! | `q4_0`    | Q4_0 lazy-dequant × f32 activations  | 13.1   |
//!
//! All f32 implementations must match `matmul_naive` within `1e-5` absolute error.

pub mod naive;
pub mod blocked;
pub mod parallel; // commit 12.2 — row-parallel GEMM via rayon
pub mod q4_0;    // commit 13.1 — Q4_0 dequantised matmul

pub use naive::matmul_naive;
pub use blocked::{matmul_blocked, matmul_blocked_with_block_size, DEFAULT_BLOCK_SIZE};
pub use parallel::matmul_parallel;
pub use q4_0::matmul_q4_0_dequant;
