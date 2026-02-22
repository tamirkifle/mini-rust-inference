//! Quantization utilities for inference.
//!
//! # Modules
//!
//! | Module              | Format | Commit |
//! |---------------------|--------|--------|
//! | `int8::symmetric`   | INT8 per-tensor symmetric | 13.2 |
//! | `int8::per_channel` | INT8 per-channel (weights) | 13.3 |

pub mod int8;
