//! GGUF file format parser and tensor extraction.
//!
//! GGUF (GGML Universal Format) is a binary file format for storing
//! large language model weights with metadata. It supports various
//! quantization schemes and is designed for efficient loading.
//!
//! # File Structure
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │              GGUF Header                │
//! ├─────────────────────────────────────────┤
//! │  Magic     │ "GGUF" (4 bytes)           │
//! │  Version   │ u32 (2 or 3)               │
//! │  Tensors   │ u64 (tensor count)         │
//! │  Metadata  │ u64 (KV pair count)        │
//! ├─────────────────────────────────────────┤
//! │           Metadata KV Pairs             │
//! │  (model config, tokenizer, etc.)        │
//! ├─────────────────────────────────────────┤
//! │            Tensor Infos                 │
//! │  (names, shapes, types, offsets)        │
//! ├─────────────────────────────────────────┤
//! │            Tensor Data                  │
//! │  (raw weight bytes, possibly quantized) │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Example: Loading and Inspecting
//!
//! ```no_run
//! use llm_engine::gguf::{GgufLoader, TensorExtractor};
//!
//! // Open a GGUF model file
//! let loader = GgufLoader::open("model.gguf")?;
//!
//! // Inspect metadata
//! println!("GGUF version: {}", loader.header().version());
//! if let Some(arch) = loader.metadata().get_str("general.architecture") {
//!     println!("Architecture: {arch}");
//! }
//!
//! // Extract F32 tensor data
//! let extractor = TensorExtractor::new(&loader);
//! if let Ok(tensor) = extractor.extract_f32("output.weight") {
//!     println!("Extracted tensor shape: {:?}", tensor.dims());
//! }
//! # Ok::<(), llm_engine::gguf::GgufError>(())
//! ```
//!
//! # References
//!
//! - [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
//! - [llama.cpp](https://github.com/ggerganov/llama.cpp)

pub mod cache;
mod dtype;
pub mod dequant;
mod error;
mod extract;
pub mod f16;
mod header;
mod loader;
mod metadata;
mod mmap;
mod quantization;
mod tensor_info;

// Core types
pub use dtype::GgmlType;
pub use error::{GgufError, Result};
pub use header::{GgufHeader, GGUF_MAGIC, SUPPORTED_VERSIONS};
pub use loader::{inspect, GgufInspection, GgufLoader, GgufLoaderBuilder};
pub use metadata::{keys, GgufValueType, Metadata, MetadataValue};
pub use mmap::{MappedFile, MappedSlice};

// Tensor extraction
pub use extract::{
    extract_f32_from_bytes, extract_f32_into, validate_tensor_data, ExtractionInfo,
    TensorExtractor,
};

// F16 (half-precision) support
pub use f16::{
    extract_f16_as_f32, extract_f16_as_f32_into, f16_to_f32_batch, f16_to_f32_batch_into,
    f32_to_f16_batch,
};

// Quantization support
pub use quantization::{
    block_bytes_for_type, block_size_for_type, f16_to_f32, f32_to_f16, BlockQ4_0, BlockQ4_1,
    BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1, QK_K, QK_LEGACY,
};

// Tensor info utilities
pub use tensor_info::{
    align_offset, padding_for_alignment, TensorInfo, TensorInfos, TensorSummary,
    DEFAULT_ALIGNMENT,
};

// Reader utilities are used internally by submodules via `super::header::`

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper to write a GGUF string (length-prefixed).
    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Creates a minimal valid GGUF file in memory.
    fn create_minimal_gguf() -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC); // Magic
        data.extend_from_slice(&3u32.to_le_bytes()); // Version
        data.extend_from_slice(&2u64.to_le_bytes()); // Tensor count
        data.extend_from_slice(&3u64.to_le_bytes()); // Metadata count

        // Metadata entry 1: general.architecture = "llama"
        write_string(&mut data, "general.architecture");
        data.extend_from_slice(&8u32.to_le_bytes()); // Type: String
        write_string(&mut data, "llama");

        // Metadata entry 2: llama.block_count = 32
        write_string(&mut data, "llama.block_count");
        data.extend_from_slice(&4u32.to_le_bytes()); // Type: Uint32
        data.extend_from_slice(&32u32.to_le_bytes());

        // Metadata entry 3: llama.embedding_length = 4096
        write_string(&mut data, "llama.embedding_length");
        data.extend_from_slice(&4u32.to_le_bytes()); // Type: Uint32
        data.extend_from_slice(&4096u32.to_le_bytes());

        data
    }

    #[test]
    fn test_parse_minimal_gguf() {
        let data = create_minimal_gguf();
        let mut cursor = Cursor::new(data);

        // Parse header
        let header = GgufHeader::read(&mut cursor).unwrap();
        assert_eq!(header.version(), 3);
        assert_eq!(header.tensor_count(), 2);
        assert_eq!(header.metadata_kv_count(), 3);

        // Parse metadata
        let metadata = Metadata::read(&mut cursor, header.metadata_kv_count()).unwrap();
        assert_eq!(metadata.len(), 3);
        assert_eq!(metadata.get_str("general.architecture"), Some("llama"));
        assert_eq!(metadata.get_u32("llama.block_count"), Some(32));
        assert_eq!(metadata.get_u32("llama.embedding_length"), Some(4096));
    }

    #[test]
    fn test_type_conversions() {
        // Test f16 conversions
        let f32_val = 1.5f32;
        let f16_bits = f32_to_f16(f32_val);
        let back = f16_to_f32(f16_bits);
        assert!((back - f32_val).abs() < 0.001);

        // Test type ID conversions
        assert_eq!(GgmlType::from_u32(0).unwrap(), GgmlType::F32);
        assert_eq!(GgmlType::from_u32(1).unwrap(), GgmlType::F16);
        assert_eq!(GgmlType::from_u32(2).unwrap(), GgmlType::Q4_0);
    }

    #[test]
    fn test_extract_f32_bytes() {
        let values: Vec<f32> = vec![1.0, 2.5, -3.14, 0.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let extracted = extract_f32_from_bytes(&bytes).unwrap();

        assert_eq!(extracted.len(), 4);
        for (i, &val) in extracted.iter().enumerate() {
            assert!(
                (val - values[i]).abs() < 1e-6,
                "Mismatch at {i}: expected {}, got {val}",
                values[i]
            );
        }
    }

    #[test]
    fn test_alignment_utilities() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);

        assert_eq!(padding_for_alignment(0, 32), 0);
        assert_eq!(padding_for_alignment(1, 32), 31);
        assert_eq!(padding_for_alignment(32, 32), 0);
    }
}