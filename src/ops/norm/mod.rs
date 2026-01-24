//! Normalisation operations.
//!
//! | Module      | Op                          | Status     |
//! |-------------|---------------------------  |------------|
//! | `rmsnorm`   | Root Mean Square LayerNorm  | ✅ commit 6.1 |

pub mod rmsnorm; // CHANGED: commit 6.1

// CHANGED: flat re-exports for the most common entry-points
pub use rmsnorm::{rmsnorm, rmsnorm_inplace, DEFAULT_EPS};
