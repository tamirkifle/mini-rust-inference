//! Attention mechanisms.
//!
//! | Module       | Contents                                   | Status        |
//! |--------------|--------------------------------------------|---------------|
//! | `sdpa`       | Scaled dot-product attention (single-head) | ✅ commit 7.1  |
//! | `mask`       | Causal / padding masks                     | ✅ commit 7.2  |
//! | `multihead`  | Multi-head attention with head splitting   | ✅ commit 7.3  |
//! | `gqa`        | Grouped-query attention (Llama 2/3 style)  | ✅ commit 7.4  |
//! | `cached`     | KV-cache prefill and decode paths          | ✅ commit 9.2  |

// CHANGED: declare sub-modules as they land commit-by-commit
pub mod sdpa;      // commit 7.1
pub mod mask;      // commit 7.2
pub mod multihead; // commit 7.3
pub mod gqa;       // CHANGED: commit 7.4
pub mod cached;    // CHANGED: commit 9.2
pub mod sliding;   // commit 12.3 — sliding window attention

// CHANGED: flat re-exports for common entry-points
pub use sdpa::scaled_dot_product_attention;
pub use mask::{causal_mask, causal_mask_with_offset, masked_sdpa, masked_sdpa_with_offset};
pub use multihead::{
    split_head, concat_heads,
    multi_head_attention,
    multi_head_attention_causal,
    multi_head_attention_causal_with_offset,
};
pub use gqa::{                                                                 // CHANGED: commit 7.4
    grouped_query_attention,
    grouped_query_attention_causal,
    grouped_query_attention_causal_with_offset,
};
pub use cached::{cached_attention_prefill, cached_attention_decode};           // CHANGED: commit 9.2
pub use sliding::{sliding_window_mask, sliding_window_sdpa, sliding_window_gqa}; // commit 12.3
