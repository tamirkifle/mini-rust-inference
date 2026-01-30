//! Attention mechanisms.
//!
//! | Module       | Contents                                   | Status        |
//! |--------------|--------------------------------------------|---------------|
//! | `sdpa`       | Scaled dot-product attention (single-head) | ✅ commit 7.1  |
//! | `mask`       | Causal / padding masks                     | 🔜 commit 7.2  |
//! | `multihead`  | Multi-head attention with head splitting   | 🔜 commit 7.3  |
//! | `gqa`        | Grouped-query attention (Llama 2/3 style)  | 🔜 commit 7.4  |

// CHANGED: declare sub-modules as they land commit-by-commit
pub mod sdpa; // commit 7.1

// CHANGED: flat re-exports for common entry-points
pub use sdpa::scaled_dot_product_attention;
