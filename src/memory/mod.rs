//! Memory management subsystem — Weeks 10–12.
//!
//! # Modules
//!
//! - [`pool`]: Pre-allocated buffer pool for intermediate tensors (commit 10.1)
//! - `arena`: Bump allocator for per-forward-pass scratch memory (commit 10.2)
//! - `stats`: Allocation instrumentation and reporting (commit 10.4)

pub mod pool;
pub mod arena;
pub mod stats;

pub use pool::{PoolStats, TensorPool};
pub use arena::Arena;
pub use stats::{format_bytes, query_rss, MemorySnapshot, MemoryTracker};
