//! INT8 quantization sub-modules.
//!
//! | Module         | Scope                            | Commit |
//! |----------------|----------------------------------|--------|
//! | `symmetric`    | Per-tensor activation quant      | 13.2   |
//! | `per_channel`  | Per-output-channel weight quant  | 13.3   |
pub mod per_channel;
pub mod symmetric;
