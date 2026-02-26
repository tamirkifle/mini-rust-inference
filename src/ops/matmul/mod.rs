//! Matrix multiplication operations.
//!
//! # Implementations
//!
//! | Module     | Strategy                              | Commit |
//! |------------|---------------------------------------|--------|
//! | `naive`    | Triple-nested ikj loop (reference)    | 5.1    |
//! | `blocked`  | Cache-tiled GEMM (32×32 tiles)        | 5.2    |
//! | `parallel` | Row-parallel GEMM via rayon           | 12.2   |
//! | `q4_0`     | Q4_0 lazy-dequant × f32 activations   | 13.1   |
//! | `int8`     | INT8×INT8→INT32→f32 quantized GEMM    | 13.4   |
//!
//! All f32 implementations must match `matmul_naive` within `1e-5` absolute error.
//! INT8 implementations are expected to match within quantization error (~2% relative).

pub mod naive;
pub mod blocked;
pub mod parallel; // commit 12.2 — row-parallel GEMM via rayon
pub mod q4_0;    // commit 13.1 — Q4_0 dequantised matmul
pub mod int8;    // commit 13.4 — INT8×INT8→INT32→f32

pub use naive::matmul_naive;
pub use blocked::{matmul_blocked, matmul_blocked_with_block_size, DEFAULT_BLOCK_SIZE};
pub use parallel::matmul_parallel;
pub use q4_0::matmul_q4_0_dequant;
pub use int8::{matmul_int8, matmul_int8_from_f32};
