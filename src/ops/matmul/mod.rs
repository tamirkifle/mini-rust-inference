//! Matrix multiplication operations.
//!
//! # Implementations
//!
//! | Module        | Strategy                              | Commit |
//! |---------------|---------------------------------------|--------|
//! | `naive`       | Triple-nested ikj loop (reference)    | 5.1    |
//! | `blocked`     | Cache-tiled GEMM (32×32 tiles)        | 5.2    |
//! | `parallel`    | Row-parallel GEMM via rayon; INT8 parallel GEMM | 12.2, 16.3 |
//! | `q4_0`        | Q4_0 lazy-dequant × f32 activations   | 13.1   |
//! | `int8`        | INT8×INT8→INT32→f32 quantized GEMM    | 13.4   |
//! | `q8_direct`   | Q8_0 direct (zero weight dequant)     | 14.2   |
//! | `avx2`        | AVX2 f32 GEMM kernel                  | 15.2   |
//! | `int8_avx2`   | AVX2 INT8 dot product kernel          | 16.1   |
//! | `int8_neon`   | NEON INT8 dot product kernel          | 16.2   |
//!
//! All f32 implementations must match `matmul_naive` within `1e-5` absolute error.
//! INT8 implementations are expected to match within quantization error (~2-3% relative).

pub mod naive;
pub mod blocked;
pub mod parallel;    // commit 12.2 — row-parallel GEMM via rayon
pub mod q4_0;        // commit 13.1 — Q4_0 dequantised matmul
pub mod int8;        // commit 13.4 — INT8×INT8→INT32→f32
pub mod q8_direct;   // commit 14.2 — direct Q8_0 inference (zero weight dequant)
pub mod avx2;        // commit 15.2 — AVX2 f32 GEMM kernel
pub mod int8_avx2;   // commit 16.1 — AVX2 INT8 dot product kernel
pub mod int8_neon;   // commit 16.2 — NEON INT8 dot product kernel

pub use naive::matmul_naive;
pub use blocked::{matmul_blocked, matmul_blocked_with_block_size, DEFAULT_BLOCK_SIZE};
pub use parallel::{matmul_parallel, matmul_int8_parallel};
pub use q4_0::matmul_q4_0_dequant;
pub use int8::{matmul_int8, matmul_int8_from_f32};
pub use q8_direct::{matmul_q8_0_direct, Q8_0WeightMatrix};
pub use avx2::matmul_avx2;
pub use int8_avx2::{dot_i8_avx2, matmul_int8_avx2};
pub use int8_neon::{dot_i8_neon, matmul_int8_neon};
