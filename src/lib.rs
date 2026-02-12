//! # LLM Inference Engine
//!
//! A production-quality LLM inference engine built from scratch in Rust.
//!
//! ## Architecture
//!
//! - `tensor`: N-dimensional array abstraction with stride support
//! - `gguf`: GGUF model format parser with memory-mapped loading and tensor extraction
//! - `ops`: Core operations (matmul, attention, normalization) - *future*
//! - `model`: Model architectures (Llama) - *future*
//! - `cache`: KV-cache management for efficient generation - *future*
//! - `quant`: Quantization (INT8/INT4) for memory efficiency - *future*
//! - `simd`: SIMD-accelerated kernels (AVX2/NEON) - *future*
//!
//! ## Quick Start
//!
//! ```no_run
//! use llm_engine::gguf::{GgufLoader, TensorExtractor};
//! use llm_engine::Tensor;
//!
//! // Load a GGUF model file
//! let loader = GgufLoader::open("model.gguf")?;
//! println!("Model has {} tensors", loader.header().tensor_count());
//!
//! // Extract F32 tensor data
//! let extractor = TensorExtractor::new(&loader);
//! let weights = extractor.extract_f32("output.weight")?;
//! println!("Weight shape: {:?}", weights.dims());
//! # Ok::<(), llm_engine::gguf::GgufError>(())
//! ```

#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

// Module declarations
pub mod gguf;
pub mod tensor;

// Re-exports for convenience
pub use tensor::{Shape, Stride, Tensor, TensorView};

// CHANGED: ops module activated in commit 5.1
pub mod ops;

// CHANGED: attention module activated in commit 7.1
pub mod attention;

// CHANGED: model module activated in commit 8.0
pub mod model;

// CHANGED: tokenizer module activated in commit 8.4
pub mod tokenizer;

// CHANGED: sampling + generation loop activated in commit 8.5
pub mod sampling;
pub mod generate;

// Future modules
// pub mod cache;
// pub mod quant;
// pub mod simd;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_exists() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn tensor_re_export_works() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        assert_eq!(tensor.dims(), &[2, 2]);
    }
}