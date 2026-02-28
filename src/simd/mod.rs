//! SIMD-accelerated compute primitives — Milestone 4, Week 15.
//!
//! # Design
//!
//! All public APIs are **safe**. Arch-specific kernels are `unsafe` internally
//! and dispatched at call-site based on CPU feature detection.
//!
//! | Sub-module  | Contents                                         |
//! |-------------|--------------------------------------------------|
//! | `f32`       | Vectorized f32 add/mul/fma/dot/hsum primitives   |
//! | `dispatch`  | Runtime CPU feature detection + kernel selection |
//!
//! # Platform support
//!
//! | Architecture | SIMD level | Notes                                         |
//! |--------------|------------|-----------------------------------------------|
//! | x86_64       | AVX2 + FMA | Runtime detection via `is_x86_feature_detected!` |
//! | aarch64      | NEON       | Always enabled (mandatory AArch64 ABI)        |
//! | other        | Scalar     | Pure Rust fallback                            |

#[allow(clippy::module_inception)]
pub mod f32; // commit-15.1 — vectorized f32 primitives
             // commit-15.4: pub mod dispatch;

pub use self::f32::{hsum, dot, add_into, mul_into, scale_into, fma_into};
