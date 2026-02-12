//! BPE tokenizer — commit 8.4.
//!
//! Loads a SentencePiece-style BPE tokenizer from GGUF metadata and provides
//! `encode` (text → token IDs) and `decode` (token IDs → text) operations.
//!
//! # GGUF metadata keys consumed
//!
//! | Key | Type | Required |
//! |-----|------|----------|
//! | `tokenizer.ggml.tokens` | StringArray | yes |
//! | `tokenizer.ggml.scores` | Float32Array | yes |
//! | `tokenizer.ggml.token_type` | Int32Array | yes |
//! | `tokenizer.ggml.bos_token_id` | u32 | yes |
//! | `tokenizer.ggml.eos_token_id` | u32 | yes |
//! | `tokenizer.ggml.merges` | StringArray | optional |
//!
//! # Encoding strategy
//!
//! Uses the SentencePiece max-score BPE algorithm:
//! 1. Prepend `▁` (U+2581) to the text and replace every space with `▁`.
//! 2. Split into individual Unicode characters; map each to a token via byte
//!    fallback (`<0xHH>`) if the character isn't directly in the vocabulary.
//! 3. Repeatedly merge the adjacent pair whose joined string has the highest
//!    score in the vocabulary, until no more valid merges exist.
//!
//! # Decoding
//!
//! Concatenate token strings, replace `▁` → ` `, trim one leading space.
//! Control tokens (type 3) are skipped.

pub mod bpe;

pub use bpe::Tokenizer;
