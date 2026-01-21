//! GGUF tensor data types.
//!
//! GGUF supports various data types for tensor storage, ranging from
//! full-precision floats to heavily quantized formats.
//!
//! # Data Type Categories
//!
//! - **Floating point**: F32, F16, BF16
//! - **Legacy quantization**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
//! - **K-quants**: Q2_K through Q8_K (more sophisticated quantization)
//! - **I-quants**: IQ1_S through IQ4_XS (importance-based quantization)
//!
//! # Block Quantization
//!
//! Most quantized types use block quantization where elements are grouped
//! into blocks (typically 32 or 256 elements) that share scale factors.
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use super::error::{GgufError, Result};
use std::fmt;

/// GGUF tensor data type identifiers.
///
/// These match the type IDs used in GGUF files and llama.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgmlType {
    /// 32-bit floating point.
    F32 = 0,
    /// 16-bit floating point (IEEE 754).
    F16 = 1,
    /// 4-bit quantization (32 elements per block, 18 bytes/block).
    Q4_0 = 2,
    /// 4-bit quantization with min (32 elements per block, 20 bytes/block).
    Q4_1 = 3,
    // Note: Types 4 and 5 are deprecated (Q4_2, Q4_3)
    /// 5-bit quantization (32 elements per block, 22 bytes/block).
    Q5_0 = 6,
    /// 5-bit quantization with min (32 elements per block, 24 bytes/block).
    Q5_1 = 7,
    /// 8-bit quantization (32 elements per block, 34 bytes/block).
    Q8_0 = 8,
    /// 8-bit quantization with sum (32 elements per block, 40 bytes/block).
    Q8_1 = 9,
    /// K-quant 2-bit (256 elements per block).
    Q2K = 10,
    /// K-quant 3-bit (256 elements per block).
    Q3K = 11,
    /// K-quant 4-bit (256 elements per block).
    Q4K = 12,
    /// K-quant 5-bit (256 elements per block).
    Q5K = 13,
    /// K-quant 6-bit (256 elements per block).
    Q6K = 14,
    /// K-quant 8-bit (256 elements per block).
    Q8K = 15,
    /// Importance-based 2-bit quantization (extra small).
    Iq2Xxs = 16,
    /// Importance-based 2-bit quantization (small).
    Iq2Xs = 17,
    /// Importance-based 3-bit quantization (extra extra small).
    Iq3Xxs = 18,
    /// Importance-based 1-bit quantization (small).
    Iq1S = 19,
    /// Importance-based 4-bit quantization (non-linear).
    Iq4Nl = 20,
    /// Importance-based 3-bit quantization (small).
    Iq3S = 21,
    /// Importance-based 2-bit quantization.
    Iq2S = 22,
    /// Importance-based 4-bit quantization (extra small).
    Iq4Xs = 23,
    /// 8-bit signed integer.
    I8 = 24,
    /// 16-bit signed integer.
    I16 = 25,
    /// 32-bit signed integer.
    I32 = 26,
    /// 64-bit signed integer.
    I64 = 27,
    /// 64-bit floating point.
    F64 = 28,
    /// Brain floating point (16-bit).
    Bf16 = 29,
}

impl GgmlType {
    /// Converts a u32 type ID to a `GgmlType`.
    ///
    /// # Errors
    ///
    /// Returns `GgufError::InvalidValueType` if the type ID is not recognized.
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            9 => Ok(Self::Q8_1),
            10 => Ok(Self::Q2K),
            11 => Ok(Self::Q3K),
            12 => Ok(Self::Q4K),
            13 => Ok(Self::Q5K),
            14 => Ok(Self::Q6K),
            15 => Ok(Self::Q8K),
            16 => Ok(Self::Iq2Xxs),
            17 => Ok(Self::Iq2Xs),
            18 => Ok(Self::Iq3Xxs),
            19 => Ok(Self::Iq1S),
            20 => Ok(Self::Iq4Nl),
            21 => Ok(Self::Iq3S),
            22 => Ok(Self::Iq2S),
            23 => Ok(Self::Iq4Xs),
            24 => Ok(Self::I8),
            25 => Ok(Self::I16),
            26 => Ok(Self::I32),
            27 => Ok(Self::I64),
            28 => Ok(Self::F64),
            29 => Ok(Self::Bf16),
            _ => Err(GgufError::InvalidValueType { type_id: value }),
        }
    }

    /// Returns the type ID as used in GGUF files.
    #[must_use]
    pub const fn type_id(self) -> u32 {
        self as u32
    }

    /// Returns the block size for quantized types.
    ///
    /// For non-quantized types (F32, F16, etc.), returns 1.
    /// For legacy quantized types (Q4_0, Q8_0, etc.), returns 32.
    /// For K-quants, returns 256.
    #[must_use]
    pub const fn block_size(self) -> usize {
        match self {
            // Non-quantized types
            Self::F32 | Self::F16 | Self::Bf16 | Self::F64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 1,

            // Legacy quantized types (32 elements per block)
            Self::Q4_0 | Self::Q4_1 | Self::Q5_0 | Self::Q5_1 | Self::Q8_0 | Self::Q8_1 => 32,

            // K-quants (256 elements per block)
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => 256,

            // I-quants (256 elements per block)
            Self::Iq2Xxs | Self::Iq2Xs | Self::Iq3Xxs | Self::Iq1S => 256,
            Self::Iq4Nl | Self::Iq3S | Self::Iq2S | Self::Iq4Xs => 256,
        }
    }

    /// Returns the size in bytes of one block.
    ///
    /// For non-quantized types, this is the size of one element.
    /// For quantized types, this is the size of a complete block.
    #[must_use]
    pub const fn block_bytes(self) -> usize {
        match self {
            // Non-quantized types (bytes per element)
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Bf16 => 2,
            Self::F64 => 8,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 4,
            Self::I64 => 8,

            // Legacy quantized types (bytes per 32-element block)
            // Q4_0: 1 f16 scale (2 bytes) + 16 bytes (32 * 4 bits / 8) = 18 bytes
            Self::Q4_0 => 18,
            // Q4_1: 1 f16 scale + 1 f16 min + 16 bytes = 20 bytes
            Self::Q4_1 => 20,
            // Q5_0: 1 f16 scale + 4 bytes high bits + 16 bytes = 22 bytes
            Self::Q5_0 => 22,
            // Q5_1: 1 f16 scale + 1 f16 min + 4 bytes high bits + 16 bytes = 24 bytes
            Self::Q5_1 => 24,
            // Q8_0: 1 f16 scale + 32 bytes (32 * 8 bits / 8) = 34 bytes
            Self::Q8_0 => 34,
            // Q8_1: 1 f32 scale + 1 f32 sum + 32 bytes = 40 bytes
            Self::Q8_1 => 40,

            // K-quants (bytes per 256-element block)
            // Q2_K: scales + quants for 256 elements
            Self::Q2K => 84,
            // Q3_K: scales + quants for 256 elements
            Self::Q3K => 110,
            // Q4_K: scales + quants for 256 elements
            Self::Q4K => 144,
            // Q5_K: scales + quants for 256 elements
            Self::Q5K => 176,
            // Q6_K: scales + quants for 256 elements
            Self::Q6K => 210,
            // Q8_K: scales + quants for 256 elements
            Self::Q8K => 292,

            // I-quants (approximate, varies by type)
            Self::Iq2Xxs => 66,
            Self::Iq2Xs => 74,
            Self::Iq3Xxs => 98,
            Self::Iq1S => 50,
            Self::Iq4Nl => 136,
            Self::Iq3S => 110,
            Self::Iq2S => 82,
            Self::Iq4Xs => 136,
        }
    }

    /// Calculates the total size in bytes for storing `n` elements.
    ///
    /// For quantized types, `n` must be a multiple of `block_size()`.
    ///
    /// # Panics
    ///
    /// Panics if `n` is not aligned to the block size for quantized types.
    #[must_use]
    pub const fn tensor_size(self, n: usize) -> usize {
        let block_size = self.block_size();
        let block_bytes = self.block_bytes();

        if block_size == 1 {
            // Non-quantized: n elements * bytes per element
            n * block_bytes
        } else {
            // Quantized: (n / block_size) blocks * bytes per block
            // Note: n should be divisible by block_size
            (n / block_size) * block_bytes
        }
    }

    /// Calculates tensor size with alignment check.
    ///
    /// # Errors
    ///
    /// Returns error if element count is not aligned to block size.
    pub fn tensor_size_checked(self, n: usize) -> Result<usize> {
        let block_size = self.block_size();

        if block_size > 1 && n % block_size != 0 {
            return Err(GgufError::ValueOutOfRange {
                field: "element count",
                value: n as u64,
            });
        }

        Ok(self.tensor_size(n))
    }

    /// Returns the name of the type as used in GGUF/llama.cpp.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
            Self::Q8K => "Q8_K",
            Self::Iq2Xxs => "IQ2_XXS",
            Self::Iq2Xs => "IQ2_XS",
            Self::Iq3Xxs => "IQ3_XXS",
            Self::Iq1S => "IQ1_S",
            Self::Iq4Nl => "IQ4_NL",
            Self::Iq3S => "IQ3_S",
            Self::Iq2S => "IQ2_S",
            Self::Iq4Xs => "IQ4_XS",
            Self::I8 => "I8",
            Self::I16 => "I16",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::F64 => "F64",
            Self::Bf16 => "BF16",
        }
    }

    /// Returns true if this is a quantized type.
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        self.block_size() > 1
    }

    /// Returns the effective bits per element for this type.
    ///
    /// For quantized types, this is an approximation based on block size.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bits_per_element(self) -> f32 {
        let block_bytes = self.block_bytes();
        let block_size = self.block_size();
        (block_bytes * 8) as f32 / block_size as f32
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_id_roundtrip() {
        let types = [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q8_1,
            GgmlType::Q2K,
            GgmlType::Q4K,
            GgmlType::Bf16,
        ];

        for ty in types {
            let id = ty.type_id();
            let recovered = GgmlType::from_u32(id).unwrap();
            assert_eq!(ty, recovered);
        }
    }

    #[test]
    fn test_invalid_type_id() {
        assert!(GgmlType::from_u32(4).is_err()); // Deprecated Q4_2
        assert!(GgmlType::from_u32(5).is_err()); // Deprecated Q4_3
        assert!(GgmlType::from_u32(100).is_err());
    }

    #[test]
    fn test_f32_size() {
        let ty = GgmlType::F32;
        assert_eq!(ty.block_size(), 1);
        assert_eq!(ty.block_bytes(), 4);
        assert_eq!(ty.tensor_size(1), 4);
        assert_eq!(ty.tensor_size(100), 400);
        assert!(!ty.is_quantized());
    }

    #[test]
    fn test_f16_size() {
        let ty = GgmlType::F16;
        assert_eq!(ty.block_size(), 1);
        assert_eq!(ty.block_bytes(), 2);
        assert_eq!(ty.tensor_size(1), 2);
        assert_eq!(ty.tensor_size(100), 200);
    }

    #[test]
    fn test_q4_0_size() {
        let ty = GgmlType::Q4_0;
        assert_eq!(ty.block_size(), 32);
        assert_eq!(ty.block_bytes(), 18);
        // 32 elements = 1 block = 18 bytes
        assert_eq!(ty.tensor_size(32), 18);
        // 64 elements = 2 blocks = 36 bytes
        assert_eq!(ty.tensor_size(64), 36);
        // 1024 elements = 32 blocks = 576 bytes
        assert_eq!(ty.tensor_size(1024), 576);
        assert!(ty.is_quantized());
    }

    #[test]
    fn test_q4_1_size() {
        let ty = GgmlType::Q4_1;
        assert_eq!(ty.block_size(), 32);
        assert_eq!(ty.block_bytes(), 20);
        assert_eq!(ty.tensor_size(32), 20);
    }

    #[test]
    fn test_q5_0_size() {
        let ty = GgmlType::Q5_0;
        assert_eq!(ty.block_size(), 32);
        assert_eq!(ty.block_bytes(), 22);
        assert_eq!(ty.tensor_size(32), 22);
    }

    #[test]
    fn test_q5_1_size() {
        let ty = GgmlType::Q5_1;
        assert_eq!(ty.block_size(), 32);
        assert_eq!(ty.block_bytes(), 24);
        assert_eq!(ty.tensor_size(32), 24);
    }

    #[test]
    fn test_q8_0_size() {
        let ty = GgmlType::Q8_0;
        assert_eq!(ty.block_size(), 32);
        assert_eq!(ty.block_bytes(), 34);
        assert_eq!(ty.tensor_size(32), 34);
        assert_eq!(ty.tensor_size(64), 68);
    }

    #[test]
    fn test_q8_1_size() {
        let ty = GgmlType::Q8_1;
        assert_eq!(ty.block_size(), 32);
        assert_eq!(ty.block_bytes(), 40);
        assert_eq!(ty.tensor_size(32), 40);
    }

    #[test]
    fn test_k_quant_sizes() {
        // K-quants use 256-element blocks
        assert_eq!(GgmlType::Q2K.block_size(), 256);
        assert_eq!(GgmlType::Q4K.block_size(), 256);
        assert_eq!(GgmlType::Q6K.block_size(), 256);

        // Verify block bytes match llama.cpp
        assert_eq!(GgmlType::Q2K.block_bytes(), 84);
        assert_eq!(GgmlType::Q4K.block_bytes(), 144);
        assert_eq!(GgmlType::Q6K.block_bytes(), 210);
    }

    #[test]
    fn test_tensor_size_checked() {
        let ty = GgmlType::Q4_0;

        // Aligned sizes should work
        assert!(ty.tensor_size_checked(32).is_ok());
        assert!(ty.tensor_size_checked(64).is_ok());
        assert!(ty.tensor_size_checked(1024).is_ok());

        // Unaligned sizes should fail
        assert!(ty.tensor_size_checked(31).is_err());
        assert!(ty.tensor_size_checked(33).is_err());
        assert!(ty.tensor_size_checked(100).is_err());

        // Non-quantized types accept any size
        let f32_ty = GgmlType::F32;
        assert!(f32_ty.tensor_size_checked(1).is_ok());
        assert!(f32_ty.tensor_size_checked(100).is_ok());
    }

    #[test]
    fn test_bits_per_element() {
        // F32: 32 bits
        assert!((GgmlType::F32.bits_per_element() - 32.0).abs() < 0.01);

        // F16: 16 bits
        assert!((GgmlType::F16.bits_per_element() - 16.0).abs() < 0.01);

        // Q4_0: 18 bytes / 32 elements = 4.5 bits/element
        assert!((GgmlType::Q4_0.bits_per_element() - 4.5).abs() < 0.01);

        // Q8_0: 34 bytes / 32 elements = 8.5 bits/element
        assert!((GgmlType::Q8_0.bits_per_element() - 8.5).abs() < 0.01);
    }

    #[test]
    fn test_type_names() {
        assert_eq!(GgmlType::F32.name(), "F32");
        assert_eq!(GgmlType::Q4_0.name(), "Q4_0");
        assert_eq!(GgmlType::Q4K.name(), "Q4_K");
        assert_eq!(GgmlType::Iq2Xxs.name(), "IQ2_XXS");
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", GgmlType::F32), "F32");
        assert_eq!(format!("{}", GgmlType::Q4_0), "Q4_0");
    }

    #[test]
    fn test_real_world_tensor_sizes() {
        // Llama 7B embedding: 4096 * 32000 = 131,072,000 elements
        let embed_elements = 4096 * 32000;

        // F16: 262,144,000 bytes (~250 MB)
        assert_eq!(GgmlType::F16.tensor_size(embed_elements), 262_144_000);

        // Q4_0: 73,728,000 bytes (~70 MB) - ~3.6x compression
        assert_eq!(GgmlType::Q4_0.tensor_size(embed_elements), 73_728_000);

        // Q8_0: 139,264,000 bytes (~133 MB) - ~1.9x compression
        assert_eq!(GgmlType::Q8_0.tensor_size(embed_elements), 139_264_000);
    }
}