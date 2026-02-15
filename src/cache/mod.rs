//! KV-cache for efficient autoregressive generation — Milestone 3, Weeks 9–12.
//!
//! | Module       | Contents                                          | Status          |
//! |--------------|---------------------------------------------------|-----------------|
//! | `kv_cache`   | Per-layer K/V storage pre-allocated for max_seq   | ✅ commit 9.1   |
//! | `position`   | Sequence position cursor with overflow guard      | ✅ commit 9.3   |
//! | `management` | Cache reset and suffix truncation helpers         | ✅ commit 9.4   |
//! | `paged`      | Block-based KV storage with lazy page allocation  | ✅ commit 11.3  |

pub mod kv_cache;    // commit 9.1
pub mod position;    // commit 9.3
pub mod management;  // commit 9.4
pub mod paged;       // commit 11.3

pub use kv_cache::KvCache;
pub use position::CachePosition;
pub use management::{cache_reset, cache_truncate};
pub use paged::PagedKvCache;
