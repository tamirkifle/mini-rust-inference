//! Activation functions.
//!
//! | Module    | Op                              | Status          |
//! |-----------|---------------------------------|-----------------|
//! | `silu`    | SiLU / Swish-1                  | ✅ commit 6.2   |
//! | `swiglu`  | SwiGLU gating (silu ⊙ up)       | ✅ commit 6.2   |

pub mod silu;   // CHANGED: commit 6.2
pub mod swiglu; // CHANGED: commit 6.2

// CHANGED: flat re-exports
pub use silu::{silu, silu_inplace, silu_scalar};
pub use swiglu::{swiglu, swiglu_inplace};
