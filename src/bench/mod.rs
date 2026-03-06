//! Benchmarking utilities — commit 17.2.
//!
//! Provides timing and memory-tracking primitives used by both the Criterion
//! benchmark harness (`benches/`) and production-mode profiling.
//!
//! # Modules
//!
//! | Module    | Contents                                    |
//! |-----------|---------------------------------------------|
//! | `metrics` | Wall-clock timer, `InferenceMetrics`, RSS   |

pub mod metrics;
pub use metrics::{InferenceMetrics, Timer, measure_generate};
