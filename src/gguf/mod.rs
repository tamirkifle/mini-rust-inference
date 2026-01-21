//! GGUF file format parser.
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
//! # Example
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::BufReader;
//! use llm_engine::gguf::{GgufHeader, Metadata};
//!
//! let file = File::open("model.gguf").expect("open file");
//! let mut reader = BufReader::new(file);
//!
//! let header = GgufHeader::read(&mut reader).expect("parse header");
//! println!("GGUF version: {}", header.version());
//! println!("Tensor count: {}", header.tensor_count());
//!
//! let metadata = Metadata::read(&mut reader, header.metadata_kv_count())
//!     .expect("parse metadata");
//! if let Some(arch) = metadata.get_str("general.architecture") {
//!     println!("Architecture: {arch}");
//! }
//! ```
//!
//! # References
//!
//! - [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)
//! - [llama.cpp](https://github.com/ggerganov/llama.cpp)

mod dtype;
mod error;
mod header;
mod metadata;
mod quantization;
mod tensor_info;

pub use dtype::GgmlType;
pub use error::{GgufError, Result};
pub use header::{GgufHeader, GGUF_MAGIC, SUPPORTED_VERSIONS};
pub use metadata::{keys, GgufValueType, Metadata, MetadataValue};
pub use quantization::{
    f16_to_f32, f32_to_f16, BlockQ4_0, BlockQ4_1, BlockQ5_0, BlockQ5_1, BlockQ8_0, BlockQ8_1,
    QK_K, QK_LEGACY,
};
pub use tensor_info::{
    align_offset, padding_for_alignment, TensorInfo, TensorInfos, TensorSummary,
    DEFAULT_ALIGNMENT,
};

// Re-export reader utilities for use by other modules (tensor_info, etc.)
pub(crate) use header::{
    read_f32, read_f64, read_i16, read_i32, read_i64, read_i8, read_string, read_u16, read_u32,
    read_u64, read_u8,
};

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
        data.extend_from_slice(&GGUF_MAGIC);            // Magic
        data.extend_from_slice(&3u32.to_le_bytes());    // Version
        data.extend_from_slice(&2u64.to_le_bytes());    // Tensor count
        data.extend_from_slice(&3u64.to_le_bytes());    // Metadata count

        // Metadata entry 1: general.architecture = "llama"
        write_string(&mut data, "general.architecture");
        data.extend_from_slice(&8u32.to_le_bytes());    // Type: String
        write_string(&mut data, "llama");

        // Metadata entry 2: llama.block_count = 32
        write_string(&mut data, "llama.block_count");
        data.extend_from_slice(&4u32.to_le_bytes());    // Type: Uint32
        data.extend_from_slice(&32u32.to_le_bytes());

        // Metadata entry 3: llama.embedding_length = 4096
        write_string(&mut data, "llama.embedding_length");
        data.extend_from_slice(&4u32.to_le_bytes());    // Type: Uint32
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
    fn test_gguf_with_arrays() {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());    // No tensors
        data.extend_from_slice(&1u64.to_le_bytes());    // 1 metadata entry

        // Metadata: tokenizer.ggml.token_type = [1, 2, 1, 1]
        write_string(&mut data, "tokenizer.ggml.token_type");
        data.extend_from_slice(&9u32.to_le_bytes());    // Type: Array
        data.extend_from_slice(&4u32.to_le_bytes());    // Element type: Uint32
        data.extend_from_slice(&4u64.to_le_bytes());    // Length: 4
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());

        let mut cursor = Cursor::new(data);

        let header = GgufHeader::read(&mut cursor).unwrap();
        let metadata = Metadata::read(&mut cursor, header.metadata_kv_count()).unwrap();

        let arr = metadata.get("tokenizer.ggml.token_type").unwrap();
        assert_eq!(arr.as_u32_array(), Some(&[1u32, 2, 1, 1][..]));
    }

    #[test]
    fn test_gguf_version_2() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&2u32.to_le_bytes());    // Version 2
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let header = GgufHeader::read(&mut cursor).unwrap();

        assert_eq!(header.version(), 2);
    }

    #[test]
    fn test_invalid_gguf_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGML");  // Wrong magic
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let result = GgufHeader::read(&mut cursor);

        assert!(result.is_err());
        if let Err(GgufError::InvalidMagic { got }) = result {
            assert_eq!(&got, b"GGML");
        } else {
            panic!("Expected InvalidMagic error");
        }
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&99u32.to_le_bytes());  // Future version
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let result = GgufHeader::read(&mut cursor);

        assert!(matches!(
            result,
            Err(GgufError::UnsupportedVersion { version: 99, .. })
        ));
    }

    #[test]
    fn test_metadata_keys_constants() {
        // Verify key constants are defined correctly
        assert_eq!(keys::GENERAL_ARCHITECTURE, "general.architecture");
        assert_eq!(keys::LLAMA_BLOCK_COUNT, "llama.block_count");
        assert_eq!(keys::LLAMA_EMBEDDING_LENGTH, "llama.embedding_length");
    }

    #[test]
    fn test_ggml_type_sizes() {
        // Verify type sizes match expected values
        assert_eq!(GgmlType::F32.tensor_size(100), 400);
        assert_eq!(GgmlType::F16.tensor_size(100), 200);
        assert_eq!(GgmlType::Q4_0.tensor_size(32), 18);
        assert_eq!(GgmlType::Q8_0.tensor_size(32), 34);
    }

    #[test]
    fn test_quantization_block_sizes() {
        // Verify block constants match type definitions
        assert_eq!(QK_LEGACY, GgmlType::Q4_0.block_size());
        assert_eq!(QK_LEGACY, GgmlType::Q8_0.block_size());
        assert_eq!(QK_K, GgmlType::Q4K.block_size());
    }

    #[test]
    fn test_f16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 100.0];
        for v in values {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((v - back).abs() < 0.01, "Failed for {v}");
        }
    }

    #[test]
    fn test_tensor_info_integration() {
        // Test that tensor info works with dtype size calculations
        let dtype = GgmlType::Q4_0;
        let numel = 4096 * 4096; // 16M elements

        // Verify size calculation
        let expected_size = dtype.tensor_size(numel);
        assert_eq!(expected_size, (numel / 32) * 18); // 32 elements per block, 18 bytes per block
    }

    #[test]
    fn test_alignment_utilities() {
        assert_eq!(align_offset(100, DEFAULT_ALIGNMENT as u64), 128);
        assert_eq!(padding_for_alignment(100, DEFAULT_ALIGNMENT as u64), 28);
    }
}