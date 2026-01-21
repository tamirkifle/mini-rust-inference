//! Quantization block structures and dequantization.
//!
//! This module defines the memory layout of quantized blocks used in GGUF files
//! and provides utilities for working with quantized data.
//!
//! # Block Quantization Overview
//!
//! Quantized tensors are divided into blocks, where each block stores:
//! - Scale factor(s) for the block
//! - Optionally, minimum/offset values
//! - Packed quantized values
//!
//! # Legacy Quantization (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
//!
//! These use 32-element blocks with simple scale factors.
//!
//! # K-Quantization (Q2_K through Q8_K)
//!
//! These use 256-element blocks with more sophisticated scale factors
//! and generally achieve better quality for the same bit rate.
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/src/ggml-quants.c>

use super::dtype::GgmlType;

/// Block size for legacy quantization formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1).
pub const QK_LEGACY: usize = 32;

/// Block size for K-quantization formats (Q2_K through Q8_K).
pub const QK_K: usize = 256;

/// Q4_0 quantization block.
///
/// 32 elements quantized to 4 bits each with a shared f16 scale.
///
/// Memory layout (18 bytes total):
/// - `d`: f16 scale factor (2 bytes)
/// - `qs`: 16 bytes of packed 4-bit values (32 * 4 bits / 8 = 16 bytes)
///
/// Dequantization: `x[i] = qs[i] * d`
/// where `qs[i]` is a signed 4-bit value in range [-8, 7].
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct BlockQ4_0 {
    /// Scale factor (f16 stored as u16).
    pub d: u16,
    /// Packed 4-bit quantized values (2 values per byte).
    pub qs: [u8; 16],
}

impl BlockQ4_0 {
    /// Block size in elements.
    pub const BLOCK_SIZE: usize = QK_LEGACY;
    /// Block size in bytes.
    pub const BLOCK_BYTES: usize = 18;

    /// Returns the scale factor as f32.
    #[must_use]
    pub fn scale(&self) -> f32 {
        f16_to_f32(self.d)
    }

    /// Extracts quantized value at index (0..31).
    ///
    /// Returns signed 4-bit value in range [-8, 7].
    #[must_use]
    pub fn quant(&self, idx: usize) -> i8 {
        debug_assert!(idx < 32);
        let byte_idx = idx / 2;
        let nibble = if idx % 2 == 0 {
            self.qs[byte_idx] & 0x0F
        } else {
            self.qs[byte_idx] >> 4
        };
        // Convert unsigned 4-bit to signed: subtract 8
        (nibble as i8) - 8
    }

    /// Dequantizes the block to f32 values.
    #[must_use]
    pub fn dequantize(&self) -> [f32; 32] {
        let d = self.scale();
        let mut result = [0.0f32; 32];

        for i in 0..32 {
            result[i] = f32::from(self.quant(i)) * d;
        }

        result
    }
}

/// Q4_1 quantization block.
///
/// 32 elements quantized to 4 bits each with f16 scale and f16 minimum.
///
/// Memory layout (20 bytes total):
/// - `d`: f16 scale factor (2 bytes)
/// - `m`: f16 minimum value (2 bytes)
/// - `qs`: 16 bytes of packed 4-bit values
///
/// Dequantization: `x[i] = qs[i] * d + m`
/// where `qs[i]` is an unsigned 4-bit value in range [0, 15].
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct BlockQ4_1 {
    /// Scale factor (f16 stored as u16).
    pub d: u16,
    /// Minimum value (f16 stored as u16).
    pub m: u16,
    /// Packed 4-bit quantized values.
    pub qs: [u8; 16],
}

impl BlockQ4_1 {
    /// Block size in elements.
    pub const BLOCK_SIZE: usize = QK_LEGACY;
    /// Block size in bytes.
    pub const BLOCK_BYTES: usize = 20;

    /// Returns the scale factor as f32.
    #[must_use]
    pub fn scale(&self) -> f32 {
        f16_to_f32(self.d)
    }

    /// Returns the minimum value as f32.
    #[must_use]
    pub fn min(&self) -> f32 {
        f16_to_f32(self.m)
    }

    /// Extracts quantized value at index (0..31).
    ///
    /// Returns unsigned 4-bit value in range [0, 15].
    #[must_use]
    pub fn quant(&self, idx: usize) -> u8 {
        debug_assert!(idx < 32);
        let byte_idx = idx / 2;
        if idx % 2 == 0 {
            self.qs[byte_idx] & 0x0F
        } else {
            self.qs[byte_idx] >> 4
        }
    }

    /// Dequantizes the block to f32 values.
    #[must_use]
    pub fn dequantize(&self) -> [f32; 32] {
        let d = self.scale();
        let m = self.min();
        let mut result = [0.0f32; 32];

        for i in 0..32 {
            result[i] = f32::from(self.quant(i)) * d + m;
        }

        result
    }
}

/// Q5_0 quantization block.
///
/// 32 elements quantized to 5 bits each with f16 scale.
///
/// Memory layout (22 bytes total):
/// - `d`: f16 scale factor (2 bytes)
/// - `qh`: 4 bytes for high bits (1 bit per element, 32 bits = 4 bytes)
/// - `qs`: 16 bytes of packed 4-bit low values
///
/// Each value = (4 low bits from qs) | (1 high bit from qh << 4)
/// Dequantization: `x[i] = (q5[i] - 16) * d`
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct BlockQ5_0 {
    /// Scale factor (f16 stored as u16).
    pub d: u16,
    /// High bits for each element (1 bit per element).
    pub qh: [u8; 4],
    /// Packed 4-bit low values.
    pub qs: [u8; 16],
}

impl BlockQ5_0 {
    /// Block size in elements.
    pub const BLOCK_SIZE: usize = QK_LEGACY;
    /// Block size in bytes.
    pub const BLOCK_BYTES: usize = 22;

    /// Returns the scale factor as f32.
    #[must_use]
    pub fn scale(&self) -> f32 {
        f16_to_f32(self.d)
    }

    /// Extracts 5-bit quantized value at index (0..31).
    #[must_use]
    pub fn quant(&self, idx: usize) -> i8 {
        debug_assert!(idx < 32);

        // Get low 4 bits
        let byte_idx = idx / 2;
        let low = if idx % 2 == 0 {
            self.qs[byte_idx] & 0x0F
        } else {
            self.qs[byte_idx] >> 4
        };

        // Get high bit
        let qh_byte = idx / 8;
        let qh_bit = idx % 8;
        let high = (self.qh[qh_byte] >> qh_bit) & 1;

        // Combine: 5-bit value = low | (high << 4)
        let q5 = low | (high << 4);

        // Convert to signed: subtract 16
        (q5 as i8) - 16
    }

    /// Dequantizes the block to f32 values.
    #[must_use]
    pub fn dequantize(&self) -> [f32; 32] {
        let d = self.scale();
        let mut result = [0.0f32; 32];

        for i in 0..32 {
            result[i] = f32::from(self.quant(i)) * d;
        }

        result
    }
}

/// Q5_1 quantization block.
///
/// 32 elements quantized to 5 bits each with f16 scale and f16 minimum.
///
/// Memory layout (24 bytes total):
/// - `d`: f16 scale factor (2 bytes)
/// - `m`: f16 minimum value (2 bytes)
/// - `qh`: 4 bytes for high bits
/// - `qs`: 16 bytes of packed 4-bit low values
///
/// Dequantization: `x[i] = q5[i] * d + m`
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct BlockQ5_1 {
    /// Scale factor (f16 stored as u16).
    pub d: u16,
    /// Minimum value (f16 stored as u16).
    pub m: u16,
    /// High bits for each element.
    pub qh: [u8; 4],
    /// Packed 4-bit low values.
    pub qs: [u8; 16],
}

impl BlockQ5_1 {
    /// Block size in elements.
    pub const BLOCK_SIZE: usize = QK_LEGACY;
    /// Block size in bytes.
    pub const BLOCK_BYTES: usize = 24;

    /// Returns the scale factor as f32.
    #[must_use]
    pub fn scale(&self) -> f32 {
        f16_to_f32(self.d)
    }

    /// Returns the minimum value as f32.
    #[must_use]
    pub fn min(&self) -> f32 {
        f16_to_f32(self.m)
    }

    /// Extracts 5-bit quantized value at index (0..31).
    #[must_use]
    pub fn quant(&self, idx: usize) -> u8 {
        debug_assert!(idx < 32);

        let byte_idx = idx / 2;
        let low = if idx % 2 == 0 {
            self.qs[byte_idx] & 0x0F
        } else {
            self.qs[byte_idx] >> 4
        };

        let qh_byte = idx / 8;
        let qh_bit = idx % 8;
        let high = (self.qh[qh_byte] >> qh_bit) & 1;

        low | (high << 4)
    }

    /// Dequantizes the block to f32 values.
    #[must_use]
    pub fn dequantize(&self) -> [f32; 32] {
        let d = self.scale();
        let m = self.min();
        let mut result = [0.0f32; 32];

        for i in 0..32 {
            result[i] = f32::from(self.quant(i)) * d + m;
        }

        result
    }
}

/// Q8_0 quantization block.
///
/// 32 elements quantized to 8 bits each with f16 scale.
///
/// Memory layout (34 bytes total):
/// - `d`: f16 scale factor (2 bytes)
/// - `qs`: 32 bytes of signed 8-bit values
///
/// Dequantization: `x[i] = qs[i] * d`
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct BlockQ8_0 {
    /// Scale factor (f16 stored as u16).
    pub d: u16,
    /// Signed 8-bit quantized values.
    pub qs: [i8; 32],
}

impl BlockQ8_0 {
    /// Block size in elements.
    pub const BLOCK_SIZE: usize = QK_LEGACY;
    /// Block size in bytes.
    pub const BLOCK_BYTES: usize = 34;

    /// Returns the scale factor as f32.
    #[must_use]
    pub fn scale(&self) -> f32 {
        f16_to_f32(self.d)
    }

    /// Dequantizes the block to f32 values.
    #[must_use]
    pub fn dequantize(&self) -> [f32; 32] {
        let d = self.scale();
        let mut result = [0.0f32; 32];

        for i in 0..32 {
            result[i] = f32::from(self.qs[i]) * d;
        }

        result
    }
}

/// Q8_1 quantization block.
///
/// 32 elements quantized to 8 bits each with f32 scale and f32 sum.
///
/// Memory layout (40 bytes total):
/// - `d`: f32 scale factor (4 bytes)
/// - `s`: f32 sum of original values (4 bytes)
/// - `qs`: 32 bytes of signed 8-bit values
///
/// The sum is stored for efficient dot product computation.
/// Dequantization: `x[i] = qs[i] * d`
#[derive(Debug, Clone, Copy)]
#[repr(C, packed)]
pub struct BlockQ8_1 {
    /// Scale factor (f32).
    pub d: f32,
    /// Sum of original values (f32).
    pub s: f32,
    /// Signed 8-bit quantized values.
    pub qs: [i8; 32],
}

impl BlockQ8_1 {
    /// Block size in elements.
    pub const BLOCK_SIZE: usize = QK_LEGACY;
    /// Block size in bytes.
    pub const BLOCK_BYTES: usize = 40;

    /// Returns the scale factor.
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.d
    }

    /// Returns the sum of original values.
    #[must_use]
    pub const fn sum(&self) -> f32 {
        self.s
    }

    /// Dequantizes the block to f32 values.
    #[must_use]
    pub fn dequantize(&self) -> [f32; 32] {
        let d = self.d;
        let mut result = [0.0f32; 32];

        for i in 0..32 {
            result[i] = f32::from(self.qs[i]) * d;
        }

        result
    }
}

/// Converts a half-precision float (stored as u16) to f32.
///
/// IEEE 754 half-precision format:
/// - 1 bit sign
/// - 5 bits exponent (bias 15)
/// - 10 bits mantissa
#[must_use]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let mant = h & 0x3FF;

    if exp == 0 {
        // Subnormal or zero
        if mant == 0 {
            // Zero (preserve sign)
            if sign == 1 {
                -0.0f32
            } else {
                0.0f32
            }
        } else {
            // Subnormal: (-1)^sign * 2^(-14) * (mant/1024)
            let val = (mant as f32) * (1.0 / 1024.0) * (1.0 / 16384.0);
            if sign == 1 { -val } else { val }
        }
    } else if exp == 31 {
        // Infinity or NaN
        if mant == 0 {
            if sign == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        // Normalized: (-1)^sign * 2^(exp-15) * (1 + mant/1024)
        let val = (1.0 + (mant as f32) / 1024.0) * 2.0f32.powi(i32::from(exp) - 15);
        if sign == 1 { -val } else { val }
    }
}

/// Converts f32 to half-precision float (stored as u16).
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 255 {
        // Infinity or NaN
        if mant == 0 {
            // Infinity
            (sign << 15) | 0x7C00
        } else {
            // NaN
            (sign << 15) | 0x7C00 | ((mant >> 13) as u16).max(1)
        }
    } else if exp > 142 {
        // Overflow to infinity
        (sign << 15) | 0x7C00
    } else if exp < 103 {
        // Underflow to zero
        sign << 15
    } else if exp < 113 {
        // Subnormal
        let mant_with_implicit = mant | 0x80_0000;
        let shift = 126 - exp;
        let mant16 = (mant_with_implicit >> (shift + 13)) as u16;
        (sign << 15) | mant16
    } else {
        // Normalized
        let exp16 = ((exp - 112) as u16) & 0x1F;
        let mant16 = (mant >> 13) as u16;
        (sign << 15) | (exp16 << 10) | mant16
    }
}

/// Returns the appropriate block size for a given type.
#[must_use]
pub const fn block_size_for_type(dtype: GgmlType) -> usize {
    dtype.block_size()
}

/// Returns the byte size of one block for a given type.
#[must_use]
pub const fn block_bytes_for_type(dtype: GgmlType) -> usize {
    dtype.block_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_conversion_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 2.0, 100.0, -0.125, 65504.0];

        for &v in &values {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            // f16 has limited precision, so allow some error
            assert!((v - back).abs() < v.abs() * 0.001 + 0.001, "Failed for {v}");
        }
    }

    #[test]
    fn test_f16_special_values() {
        // Zero
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);

        // One
        assert!((f16_to_f32(0x3C00) - 1.0).abs() < 1e-6);

        // Infinity
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7C00).is_sign_positive());
        assert!(f16_to_f32(0xFC00).is_infinite());
        assert!(f16_to_f32(0xFC00).is_sign_negative());

        // NaN
        assert!(f16_to_f32(0x7C01).is_nan());
    }

    #[test]
    fn test_block_q4_0_size() {
        assert_eq!(std::mem::size_of::<BlockQ4_0>(), 18);
        assert_eq!(BlockQ4_0::BLOCK_SIZE, 32);
        assert_eq!(BlockQ4_0::BLOCK_BYTES, 18);
    }

    #[test]
    fn test_block_q4_1_size() {
        assert_eq!(std::mem::size_of::<BlockQ4_1>(), 20);
        assert_eq!(BlockQ4_1::BLOCK_SIZE, 32);
        assert_eq!(BlockQ4_1::BLOCK_BYTES, 20);
    }

    #[test]
    fn test_block_q5_0_size() {
        assert_eq!(std::mem::size_of::<BlockQ5_0>(), 22);
        assert_eq!(BlockQ5_0::BLOCK_SIZE, 32);
        assert_eq!(BlockQ5_0::BLOCK_BYTES, 22);
    }

    #[test]
    fn test_block_q5_1_size() {
        assert_eq!(std::mem::size_of::<BlockQ5_1>(), 24);
        assert_eq!(BlockQ5_1::BLOCK_SIZE, 32);
        assert_eq!(BlockQ5_1::BLOCK_BYTES, 24);
    }

    #[test]
    fn test_block_q8_0_size() {
        assert_eq!(std::mem::size_of::<BlockQ8_0>(), 34);
        assert_eq!(BlockQ8_0::BLOCK_SIZE, 32);
        assert_eq!(BlockQ8_0::BLOCK_BYTES, 34);
    }

    #[test]
    fn test_block_q8_1_size() {
        assert_eq!(std::mem::size_of::<BlockQ8_1>(), 40);
        assert_eq!(BlockQ8_1::BLOCK_SIZE, 32);
        assert_eq!(BlockQ8_1::BLOCK_BYTES, 40);
    }

    #[test]
    fn test_q4_0_quant_extraction() {
        // Create a block with known values
        let block = BlockQ4_0 {
            d: f32_to_f16(1.0),
            qs: [
                0x10, // elements 0=0, 1=1
                0x32, // elements 2=2, 3=3
                0x54, // elements 4=4, 5=5
                0x76, // elements 6=6, 7=7
                0x98, // elements 8=8, 9=9
                0xBA, // elements 10=10, 11=11
                0xDC, // elements 12=12, 13=13
                0xFE, // elements 14=14, 15=15
                0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            ],
        };

        // Check extraction (subtract 8 for signed conversion)
        assert_eq!(block.quant(0), 0 - 8);
        assert_eq!(block.quant(1), 1 - 8);
        assert_eq!(block.quant(2), 2 - 8);
        assert_eq!(block.quant(3), 3 - 8);
    }

    #[test]
    fn test_q4_1_quant_extraction() {
        let block = BlockQ4_1 {
            d: f32_to_f16(0.1),
            m: f32_to_f16(0.0),
            qs: [
                0x10, 0x32, 0x54, 0x76,
                0x98, 0xBA, 0xDC, 0xFE,
                0x00, 0x00, 0x00, 0x00,
                0x00, 0x00, 0x00, 0x00,
            ],
        };

        // Q4_1 uses unsigned values
        assert_eq!(block.quant(0), 0);
        assert_eq!(block.quant(1), 1);
        assert_eq!(block.quant(2), 2);
        assert_eq!(block.quant(3), 3);
    }

    #[test]
    fn test_q8_0_dequantize() {
        let block = BlockQ8_0 {
            d: f32_to_f16(0.5),
            qs: [
                0, 1, 2, 3, 4, 5, 6, 7,
                -8, -7, -6, -5, -4, -3, -2, -1,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            ],
        };

        let dequant = block.dequantize();
        let scale = block.scale();

        assert!((dequant[0] - 0.0).abs() < 0.01);
        assert!((dequant[1] - scale).abs() < 0.01);
        assert!((dequant[8] - (-4.0)).abs() < 0.01);
    }

    #[test]
    fn test_block_size_constants() {
        assert_eq!(QK_LEGACY, 32);
        assert_eq!(QK_K, 256);
    }
}