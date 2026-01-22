//! Dequantization routines for GGUF quantized tensor formats.
//!
//! This module provides functions to convert quantized tensor data back to
//! full-precision (F32) values. Quantization reduces model size by storing
//! weights with fewer bits, and dequantization is required before computation.
//!
//! # Supported Formats
//!
//! Currently implemented:
//! - **Q8_0**: 8-bit quantization with block-wise scaling
//!
//! Planned:
//! - Q4_0, Q4_1: 4-bit quantization variants
//! - Q5_0, Q5_1: 5-bit quantization variants
//! - Q8_1: 8-bit with min value offset
//!
//! # Q8_0 Format
//!
//! Q8_0 uses 8-bit signed integers with a shared scale per block of 32 values:
//!
//! ```text
//! ┌────────────────┬────────────────────────────────┐
//! │  scale (f16)   │  quants[32] (i8 × 32)          │
//! │   2 bytes      │   32 bytes                     │
//! └────────────────┴────────────────────────────────┘
//! ```
//!
//! Dequantization formula: `value[i] = scale × quants[i]`
//!
//! # Example
//!
//! ```
//! use llm_engine::gguf::dequant::{dequantize_q8_0, DequantStats};
//!
//! // Raw Q8_0 block data (34 bytes per block)
//! let block_data: Vec<u8> = create_q8_0_block(1.0, &[0i8; 32]);
//! let values = dequantize_q8_0(&block_data).unwrap();
//! assert_eq!(values.len(), 32);
//!
//! # fn create_q8_0_block(scale: f32, quants: &[i8; 32]) -> Vec<u8> {
//! #     let mut data = Vec::new();
//! #     data.extend_from_slice(&llm_engine::gguf::f32_to_f16(scale).to_le_bytes());
//! #     for &q in quants { data.push(q as u8); }
//! #     data
//! # }
//! ```
//!
//! # Performance
//!
//! The dequantization functions are optimized for:
//! - Sequential memory access patterns
//! - Minimal branching in the inner loop
//! - Potential SIMD vectorization by the compiler
//!
//! For large tensors, consider using the `_into` variants to reuse buffers.

use super::error::{GgufError, Result};
use super::quantization::{f16_to_f32, BlockQ8_0, QK_LEGACY};

/// Size of a Q8_0 block in bytes (2 bytes scale + 32 bytes quants).
pub const Q8_0_BLOCK_SIZE: usize = std::mem::size_of::<BlockQ8_0>();

/// Number of elements per Q8_0 block.
pub const Q8_0_BLOCK_ELEMENTS: usize = QK_LEGACY;

/// Dequantizes Q8_0 data to F32 values.
///
/// Q8_0 stores 32 values per block with a shared F16 scale factor.
/// Each quantized value is an i8 that gets multiplied by the scale.
///
/// # Arguments
///
/// * `data` - Raw byte slice containing Q8_0 blocks
///
/// # Returns
///
/// A vector of f32 values.
///
/// # Errors
///
/// Returns an error if the data length is not a multiple of the Q8_0 block size (34 bytes).
///
/// # Example
///
/// ```
/// use llm_engine::gguf::dequant::dequantize_q8_0;
/// use llm_engine::gguf::f32_to_f16;
///
/// // Create a single Q8_0 block with scale=0.5 and quants=[2, 4, 6, ...]
/// let mut block_data = Vec::new();
/// block_data.extend_from_slice(&f32_to_f16(0.5).to_le_bytes()); // scale
/// for i in 0..32i8 {
///     block_data.push((i * 2) as u8); // quants as bytes
/// }
///
/// let values = dequantize_q8_0(&block_data).unwrap();
/// assert_eq!(values.len(), 32);
/// // First value: 0.5 * 0 = 0.0
/// assert!((values[0] - 0.0).abs() < 1e-3);
/// // Second value: 0.5 * 2 = 1.0
/// assert!((values[1] - 1.0).abs() < 1e-3);
/// ```
pub fn dequantize_q8_0(data: &[u8]) -> Result<Vec<f32>> {
    // Verify data length is a multiple of block size
    if data.len() % Q8_0_BLOCK_SIZE != 0 {
        return Err(GgufError::AlignmentError {
            expected: Q8_0_BLOCK_SIZE,
            actual: data.len() % Q8_0_BLOCK_SIZE,
        });
    }

    let num_blocks = data.len() / Q8_0_BLOCK_SIZE;
    let num_elements = num_blocks * Q8_0_BLOCK_ELEMENTS;
    let mut result = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * Q8_0_BLOCK_SIZE;
        let block_data = &data[block_start..block_start + Q8_0_BLOCK_SIZE];

        // Extract scale (first 2 bytes, F16 little-endian)
        let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
        let scale = f16_to_f32(scale_bits);

        // Extract and dequantize the 32 quantized values
        let quants = &block_data[2..];
        for &q in quants {
            // q is stored as u8 but represents i8
            let q_signed = q as i8;
            result.push(scale * f32::from(q_signed));
        }
    }

    Ok(result)
}

/// Dequantizes Q8_0 data into a provided buffer.
///
/// This variant allows reusing a buffer to avoid allocations during
/// repeated dequantization operations.
///
/// # Arguments
///
/// * `data` - Raw byte slice containing Q8_0 blocks
/// * `buffer` - Destination buffer (will be cleared and filled)
///
/// # Errors
///
/// Returns an error if the data length is not a multiple of the Q8_0 block size.
pub fn dequantize_q8_0_into(data: &[u8], buffer: &mut Vec<f32>) -> Result<()> {
    if data.len() % Q8_0_BLOCK_SIZE != 0 {
        return Err(GgufError::AlignmentError {
            expected: Q8_0_BLOCK_SIZE,
            actual: data.len() % Q8_0_BLOCK_SIZE,
        });
    }

    let num_blocks = data.len() / Q8_0_BLOCK_SIZE;
    let num_elements = num_blocks * Q8_0_BLOCK_ELEMENTS;

    buffer.clear();
    buffer.reserve(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * Q8_0_BLOCK_SIZE;
        let block_data = &data[block_start..block_start + Q8_0_BLOCK_SIZE];

        let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
        let scale = f16_to_f32(scale_bits);

        let quants = &block_data[2..];
        for &q in quants {
            let q_signed = q as i8;
            buffer.push(scale * f32::from(q_signed));
        }
    }

    Ok(())
}

/// Dequantizes a single Q8_0 block.
///
/// Useful when processing blocks individually or for testing.
///
/// # Arguments
///
/// * `block` - Exactly 34 bytes representing one Q8_0 block
///
/// # Returns
///
/// An array of 32 f32 values.
///
/// # Errors
///
/// Returns an error if the block is not exactly 34 bytes.
pub fn dequantize_q8_0_block(block: &[u8]) -> Result<[f32; Q8_0_BLOCK_ELEMENTS]> {
    if block.len() != Q8_0_BLOCK_SIZE {
        return Err(GgufError::ShapeMismatch {
            expected: Q8_0_BLOCK_SIZE,
            got: block.len(),
        });
    }

    let scale_bits = u16::from_le_bytes([block[0], block[1]]);
    let scale = f16_to_f32(scale_bits);

    let mut result = [0.0f32; Q8_0_BLOCK_ELEMENTS];
    let quants = &block[2..];

    for (i, &q) in quants.iter().enumerate() {
        let q_signed = q as i8;
        result[i] = scale * f32::from(q_signed);
    }

    Ok(result)
}

/// Statistics collected during dequantization.
#[derive(Debug, Clone, Default)]
pub struct DequantStats {
    /// Number of blocks processed.
    pub num_blocks: usize,
    /// Number of elements dequantized.
    pub num_elements: usize,
    /// Minimum scale factor encountered.
    pub min_scale: f32,
    /// Maximum scale factor encountered.
    pub max_scale: f32,
    /// Minimum dequantized value.
    pub min_value: f32,
    /// Maximum dequantized value.
    pub max_value: f32,
    /// Number of zero values.
    pub zero_count: usize,
}

impl DequantStats {
    /// Creates new stats with initial values.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_scale: f32::MAX,
            max_scale: f32::MIN,
            min_value: f32::MAX,
            max_value: f32::MIN,
            ..Default::default()
        }
    }

    /// Updates stats with a scale value.
    fn update_scale(&mut self, scale: f32) {
        if scale < self.min_scale {
            self.min_scale = scale;
        }
        if scale > self.max_scale {
            self.max_scale = scale;
        }
    }

    /// Updates stats with a dequantized value.
    fn update_value(&mut self, value: f32) {
        if value < self.min_value {
            self.min_value = value;
        }
        if value > self.max_value {
            self.max_value = value;
        }
        if value == 0.0 {
            self.zero_count += 1;
        }
    }

    /// Returns the sparsity ratio (fraction of zero values).
    #[must_use]
    pub fn sparsity(&self) -> f32 {
        if self.num_elements == 0 {
            0.0
        } else {
            self.zero_count as f32 / self.num_elements as f32
        }
    }
}

/// Dequantizes Q8_0 data with statistics collection.
///
/// Useful for analyzing the distribution of quantized weights.
///
/// # Arguments
///
/// * `data` - Raw byte slice containing Q8_0 blocks
///
/// # Returns
///
/// A tuple of (dequantized values, statistics).
///
/// # Errors
///
/// Returns an error if the data length is not a multiple of the Q8_0 block size.
pub fn dequantize_q8_0_with_stats(data: &[u8]) -> Result<(Vec<f32>, DequantStats)> {
    if data.len() % Q8_0_BLOCK_SIZE != 0 {
        return Err(GgufError::AlignmentError {
            expected: Q8_0_BLOCK_SIZE,
            actual: data.len() % Q8_0_BLOCK_SIZE,
        });
    }

    let num_blocks = data.len() / Q8_0_BLOCK_SIZE;
    let num_elements = num_blocks * Q8_0_BLOCK_ELEMENTS;

    let mut result = Vec::with_capacity(num_elements);
    let mut stats = DequantStats::new();
    stats.num_blocks = num_blocks;
    stats.num_elements = num_elements;

    for block_idx in 0..num_blocks {
        let block_start = block_idx * Q8_0_BLOCK_SIZE;
        let block_data = &data[block_start..block_start + Q8_0_BLOCK_SIZE];

        let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
        let scale = f16_to_f32(scale_bits);
        stats.update_scale(scale);

        let quants = &block_data[2..];
        for &q in quants {
            let q_signed = q as i8;
            let value = scale * f32::from(q_signed);
            stats.update_value(value);
            result.push(value);
        }
    }

    Ok((result, stats))
}

/// Calculates the expected number of elements from Q8_0 data size.
///
/// # Arguments
///
/// * `data_size` - Size of Q8_0 data in bytes
///
/// # Returns
///
/// Number of f32 elements that would result from dequantization,
/// or `None` if the size is not a valid Q8_0 data size.
#[must_use]
pub const fn q8_0_element_count(data_size: usize) -> Option<usize> {
    if data_size % Q8_0_BLOCK_SIZE != 0 {
        None
    } else {
        Some((data_size / Q8_0_BLOCK_SIZE) * Q8_0_BLOCK_ELEMENTS)
    }
}

/// Calculates the Q8_0 data size needed for a given number of elements.
///
/// # Arguments
///
/// * `num_elements` - Number of f32 elements
///
/// # Returns
///
/// Size in bytes of Q8_0 data needed, or `None` if the element count
/// is not a multiple of 32 (the block size).
#[must_use]
pub const fn q8_0_data_size(num_elements: usize) -> Option<usize> {
    if num_elements % Q8_0_BLOCK_ELEMENTS != 0 {
        None
    } else {
        Some((num_elements / Q8_0_BLOCK_ELEMENTS) * Q8_0_BLOCK_SIZE)
    }
}

/// Creates a Q8_0 block from f32 values (for testing).
///
/// This performs simple quantization by finding the maximum absolute value
/// and scaling all values to fit in i8 range.
///
/// # Arguments
///
/// * `values` - Exactly 32 f32 values to quantize
///
/// # Returns
///
/// A 34-byte Q8_0 block.
///
/// # Panics
///
/// Panics if `values` does not contain exactly 32 elements.
#[must_use]
pub fn create_q8_0_block(values: &[f32]) -> [u8; Q8_0_BLOCK_SIZE] {
    assert_eq!(
        values.len(),
        Q8_0_BLOCK_ELEMENTS,
        "Q8_0 block requires exactly 32 values"
    );

    // Find maximum absolute value for scale calculation
    let max_abs = values
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, |a, b| a.max(b));

    // Calculate scale (map max_abs to 127)
    let scale = if max_abs == 0.0 {
        0.0
    } else {
        max_abs / 127.0
    };

    // Convert scale to F16
    let scale_f16 = super::quantization::f32_to_f16(scale);

    let mut block = [0u8; Q8_0_BLOCK_SIZE];

    // Write scale (little-endian)
    block[0..2].copy_from_slice(&scale_f16.to_le_bytes());

    // Quantize values
    if scale > 0.0 {
        let inv_scale = 1.0 / scale;
        for (i, &v) in values.iter().enumerate() {
            let q = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
            block[2 + i] = q as u8;
        }
    }
    // If scale is 0, quants are already initialized to 0

    block
}

// =============================================================================
// Q4_0 Dequantization
// =============================================================================

/// Size of a Q4_0 block in bytes (2 bytes scale + 16 bytes quants).
pub const Q4_0_BLOCK_SIZE: usize = 18;

/// Number of elements per Q4_0 block.
pub const Q4_0_BLOCK_ELEMENTS: usize = QK_LEGACY; // 32

/// Dequantizes Q4_0 data to F32 values.
///
/// Q4_0 stores 32 values per block with a shared F16 scale factor.
/// Each byte contains two 4-bit quantized values (low nibble first).
/// Values are unsigned (0-15) and shifted by -8 to get signed range (-8 to +7).
///
/// # Arguments
///
/// * `data` - Raw byte slice containing Q4_0 blocks
///
/// # Returns
///
/// A vector of f32 values.
///
/// # Errors
///
/// Returns an error if the data length is not a multiple of the Q4_0 block size (18 bytes).
///
/// # Example
///
/// ```
/// use llm_engine::gguf::dequant::dequantize_q4_0;
///
/// // Create a minimal Q4_0 block (18 bytes)
/// let mut block_data = vec![0u8; 18];
/// // Scale = 1.0 in F16 = 0x3C00
/// block_data[0] = 0x00;
/// block_data[1] = 0x3C;
/// // First byte: low nibble = 8 (becomes 0), high nibble = 9 (becomes 1)
/// block_data[2] = 0x98; // (9 << 4) | 8
///
/// let values = dequantize_q4_0(&block_data).unwrap();
/// assert_eq!(values.len(), 32);
/// // First value: 1.0 * (8 - 8) = 0.0
/// assert!((values[0] - 0.0).abs() < 0.1);
/// // Second value: 1.0 * (9 - 8) = 1.0
/// assert!((values[1] - 1.0).abs() < 0.1);
/// ```
pub fn dequantize_q4_0(data: &[u8]) -> Result<Vec<f32>> {
    // Verify data length is a multiple of block size
    if data.len() % Q4_0_BLOCK_SIZE != 0 {
        return Err(GgufError::AlignmentError {
            expected: Q4_0_BLOCK_SIZE,
            actual: data.len() % Q4_0_BLOCK_SIZE,
        });
    }

    let num_blocks = data.len() / Q4_0_BLOCK_SIZE;
    let num_elements = num_blocks * Q4_0_BLOCK_ELEMENTS;
    let mut result = Vec::with_capacity(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * Q4_0_BLOCK_SIZE;
        let block_data = &data[block_start..block_start + Q4_0_BLOCK_SIZE];

        // Extract scale (first 2 bytes, F16 little-endian)
        let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
        let scale = f16_to_f32(scale_bits);

        // Extract and dequantize the 32 quantized values (16 bytes, 2 values per byte)
        let quants = &block_data[2..];
        for &byte in quants {
            // Low nibble first (bits 0-3)
            let q_low = (byte & 0x0F) as i32 - 8;
            result.push(scale * q_low as f32);

            // High nibble second (bits 4-7)
            let q_high = ((byte >> 4) & 0x0F) as i32 - 8;
            result.push(scale * q_high as f32);
        }
    }

    Ok(result)
}

/// Dequantizes Q4_0 data into a provided buffer.
///
/// This variant allows reusing a buffer to avoid allocations during
/// repeated dequantization operations.
///
/// # Arguments
///
/// * `data` - Raw byte slice containing Q4_0 blocks
/// * `buffer` - Destination buffer (will be cleared and filled)
///
/// # Errors
///
/// Returns an error if the data length is not a multiple of the Q4_0 block size.
pub fn dequantize_q4_0_into(data: &[u8], buffer: &mut Vec<f32>) -> Result<()> {
    if data.len() % Q4_0_BLOCK_SIZE != 0 {
        return Err(GgufError::AlignmentError {
            expected: Q4_0_BLOCK_SIZE,
            actual: data.len() % Q4_0_BLOCK_SIZE,
        });
    }

    let num_blocks = data.len() / Q4_0_BLOCK_SIZE;
    let num_elements = num_blocks * Q4_0_BLOCK_ELEMENTS;

    buffer.clear();
    buffer.reserve(num_elements);

    for block_idx in 0..num_blocks {
        let block_start = block_idx * Q4_0_BLOCK_SIZE;
        let block_data = &data[block_start..block_start + Q4_0_BLOCK_SIZE];

        let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
        let scale = f16_to_f32(scale_bits);

        let quants = &block_data[2..];
        for &byte in quants {
            let q_low = (byte & 0x0F) as i32 - 8;
            buffer.push(scale * q_low as f32);

            let q_high = ((byte >> 4) & 0x0F) as i32 - 8;
            buffer.push(scale * q_high as f32);
        }
    }

    Ok(())
}

/// Dequantizes a single Q4_0 block.
///
/// Useful when processing blocks individually or for testing.
///
/// # Arguments
///
/// * `block` - Exactly 18 bytes representing one Q4_0 block
///
/// # Returns
///
/// An array of 32 f32 values.
///
/// # Errors
///
/// Returns an error if the block is not exactly 18 bytes.
pub fn dequantize_q4_0_block(block: &[u8]) -> Result<[f32; Q4_0_BLOCK_ELEMENTS]> {
    if block.len() != Q4_0_BLOCK_SIZE {
        return Err(GgufError::ShapeMismatch {
            expected: Q4_0_BLOCK_SIZE,
            got: block.len(),
        });
    }

    let scale_bits = u16::from_le_bytes([block[0], block[1]]);
    let scale = f16_to_f32(scale_bits);

    let mut result = [0.0f32; Q4_0_BLOCK_ELEMENTS];
    let quants = &block[2..];

    for (i, &byte) in quants.iter().enumerate() {
        let q_low = (byte & 0x0F) as i32 - 8;
        result[i * 2] = scale * q_low as f32;

        let q_high = ((byte >> 4) & 0x0F) as i32 - 8;
        result[i * 2 + 1] = scale * q_high as f32;
    }

    Ok(result)
}

/// Dequantizes Q4_0 data with statistics collection.
///
/// Useful for analyzing the distribution of quantized weights.
///
/// # Arguments
///
/// * `data` - Raw byte slice containing Q4_0 blocks
///
/// # Returns
///
/// A tuple of (dequantized values, statistics).
///
/// # Errors
///
/// Returns an error if the data length is not a multiple of the Q4_0 block size.
pub fn dequantize_q4_0_with_stats(data: &[u8]) -> Result<(Vec<f32>, DequantStats)> {
    if data.len() % Q4_0_BLOCK_SIZE != 0 {
        return Err(GgufError::AlignmentError {
            expected: Q4_0_BLOCK_SIZE,
            actual: data.len() % Q4_0_BLOCK_SIZE,
        });
    }

    let num_blocks = data.len() / Q4_0_BLOCK_SIZE;
    let num_elements = num_blocks * Q4_0_BLOCK_ELEMENTS;

    let mut result = Vec::with_capacity(num_elements);
    let mut stats = DequantStats::new();
    stats.num_blocks = num_blocks;
    stats.num_elements = num_elements;

    for block_idx in 0..num_blocks {
        let block_start = block_idx * Q4_0_BLOCK_SIZE;
        let block_data = &data[block_start..block_start + Q4_0_BLOCK_SIZE];

        let scale_bits = u16::from_le_bytes([block_data[0], block_data[1]]);
        let scale = f16_to_f32(scale_bits);
        stats.update_scale(scale);

        let quants = &block_data[2..];
        for &byte in quants {
            let q_low = (byte & 0x0F) as i32 - 8;
            let value_low = scale * q_low as f32;
            stats.update_value(value_low);
            result.push(value_low);

            let q_high = ((byte >> 4) & 0x0F) as i32 - 8;
            let value_high = scale * q_high as f32;
            stats.update_value(value_high);
            result.push(value_high);
        }
    }

    Ok((result, stats))
}

/// Calculates the expected number of elements from Q4_0 data size.
///
/// # Arguments
///
/// * `data_size` - Size of Q4_0 data in bytes
///
/// # Returns
///
/// Number of f32 elements that would result from dequantization,
/// or `None` if the size is not a valid Q4_0 data size.
#[must_use]
pub const fn q4_0_element_count(data_size: usize) -> Option<usize> {
    if data_size % Q4_0_BLOCK_SIZE != 0 {
        None
    } else {
        Some((data_size / Q4_0_BLOCK_SIZE) * Q4_0_BLOCK_ELEMENTS)
    }
}

/// Calculates the Q4_0 data size needed for a given number of elements.
///
/// # Arguments
///
/// * `num_elements` - Number of f32 elements
///
/// # Returns
///
/// Size in bytes of Q4_0 data needed, or `None` if the element count
/// is not a multiple of 32 (the block size).
#[must_use]
pub const fn q4_0_data_size(num_elements: usize) -> Option<usize> {
    if num_elements % Q4_0_BLOCK_ELEMENTS != 0 {
        None
    } else {
        Some((num_elements / Q4_0_BLOCK_ELEMENTS) * Q4_0_BLOCK_SIZE)
    }
}

/// Creates a Q4_0 block from f32 values (for testing).
///
/// This performs simple quantization by finding the maximum absolute value
/// and scaling all values to fit in 4-bit signed range (-8 to +7).
///
/// # Arguments
///
/// * `values` - Exactly 32 f32 values to quantize
///
/// # Returns
///
/// An 18-byte Q4_0 block.
///
/// # Panics
///
/// Panics if `values` does not contain exactly 32 elements.
#[must_use]
pub fn create_q4_0_block(values: &[f32]) -> [u8; Q4_0_BLOCK_SIZE] {
    assert_eq!(
        values.len(),
        Q4_0_BLOCK_ELEMENTS,
        "Q4_0 block requires exactly 32 values"
    );

    // Find maximum absolute value for scale calculation
    let max_abs = values
        .iter()
        .map(|v| v.abs())
        .fold(0.0f32, |a, b| a.max(b));

    // Calculate scale (map max_abs to 7, the max positive value in 4-bit signed)
    let scale = if max_abs == 0.0 {
        0.0
    } else {
        max_abs / 7.0
    };

    // Convert scale to F16
    let scale_f16 = super::quantization::f32_to_f16(scale);

    let mut block = [0u8; Q4_0_BLOCK_SIZE];

    // Write scale (little-endian)
    block[0..2].copy_from_slice(&scale_f16.to_le_bytes());

    // Quantize values (pack 2 per byte)
    if scale > 0.0 {
        let inv_scale = 1.0 / scale;
        for i in 0..16 {
            // Low nibble (even index)
            let v_low = values[i * 2];
            let q_low = ((v_low * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;

            // High nibble (odd index)
            let v_high = values[i * 2 + 1];
            let q_high = ((v_high * inv_scale).round() as i32 + 8).clamp(0, 15) as u8;

            block[2 + i] = q_low | (q_high << 4);
        }
    } else {
        // If scale is 0, all quants should be 8 (which maps to 0 after -8 offset)
        for i in 0..16 {
            block[2 + i] = 0x88; // Both nibbles = 8
        }
    }

    block
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-2; // Q8_0 has limited precision

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_dequantize_q8_0_single_block() {
        // Create a block with scale=1.0 and quants=[0, 1, 2, ..., 31]
        let mut block_data = Vec::new();

        // Scale = 1.0 in F16 = 0x3C00
        block_data.extend_from_slice(&0x3C00u16.to_le_bytes());

        // Quants: 0, 1, 2, ..., 31
        for i in 0..32i8 {
            block_data.push(i as u8);
        }

        let values = dequantize_q8_0(&block_data).unwrap();

        assert_eq!(values.len(), 32);
        for (i, &v) in values.iter().enumerate() {
            let expected = i as f32;
            assert!(
                approx_eq(v, expected),
                "Mismatch at {i}: expected {expected}, got {v}"
            );
        }
    }

    #[test]
    fn test_dequantize_q8_0_negative_values() {
        let mut block_data = Vec::new();

        // Scale = 0.5 in F16 = 0x3800
        block_data.extend_from_slice(&0x3800u16.to_le_bytes());

        // Quants: -128 to 127 (first 32 values starting from -16)
        for i in -16..16i8 {
            block_data.push(i as u8);
        }

        let values = dequantize_q8_0(&block_data).unwrap();

        assert_eq!(values.len(), 32);
        // First value: 0.5 * (-16) = -8.0
        assert!(approx_eq(values[0], -8.0));
        // Middle value (index 16): 0.5 * 0 = 0.0
        assert!(approx_eq(values[16], 0.0));
        // Last value: 0.5 * 15 = 7.5
        assert!(approx_eq(values[31], 7.5));
    }

    #[test]
    fn test_dequantize_q8_0_multiple_blocks() {
        let mut data = Vec::new();

        // Block 1: scale=1.0, all zeros
        data.extend_from_slice(&0x3C00u16.to_le_bytes());
        data.extend_from_slice(&[0u8; 32]);

        // Block 2: scale=2.0, all ones
        data.extend_from_slice(&0x4000u16.to_le_bytes());
        data.extend_from_slice(&[1u8; 32]);

        let values = dequantize_q8_0(&data).unwrap();

        assert_eq!(values.len(), 64);

        // First block: all zeros
        for &v in &values[0..32] {
            assert!(approx_eq(v, 0.0));
        }

        // Second block: all 2.0 (scale=2.0 * quant=1)
        for &v in &values[32..64] {
            assert!(approx_eq(v, 2.0));
        }
    }

    #[test]
    fn test_dequantize_q8_0_invalid_length() {
        // Not a multiple of 34 bytes
        let data = vec![0u8; 35];
        let result = dequantize_q8_0(&data);

        assert!(result.is_err());
        assert!(matches!(result, Err(GgufError::AlignmentError { .. })));
    }

    #[test]
    fn test_dequantize_q8_0_empty() {
        let data: Vec<u8> = vec![];
        let values = dequantize_q8_0(&data).unwrap();

        assert!(values.is_empty());
    }

    #[test]
    fn test_dequantize_q8_0_into() {
        let mut block_data = Vec::new();
        block_data.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale=1.0
        block_data.extend_from_slice(&[5u8; 32]); // all 5s

        let mut buffer = Vec::new();
        dequantize_q8_0_into(&block_data, &mut buffer).unwrap();

        assert_eq!(buffer.len(), 32);
        for &v in &buffer {
            assert!(approx_eq(v, 5.0));
        }

        // Reuse buffer
        block_data.clear();
        block_data.extend_from_slice(&0x4000u16.to_le_bytes()); // scale=2.0
        block_data.extend_from_slice(&[3u8; 32]); // all 3s

        dequantize_q8_0_into(&block_data, &mut buffer).unwrap();
        assert_eq!(buffer.len(), 32);
        for &v in &buffer {
            assert!(approx_eq(v, 6.0)); // 2.0 * 3 = 6.0
        }
    }

    #[test]
    fn test_dequantize_q8_0_block() {
        let mut block = [0u8; 34];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale=1.0
        for i in 0..32 {
            block[2 + i] = (i * 2) as u8;
        }

        let values = dequantize_q8_0_block(&block).unwrap();

        assert_eq!(values.len(), 32);
        for (i, &v) in values.iter().enumerate() {
            let expected = (i * 2) as f32;
            assert!(approx_eq(v, expected));
        }
    }

    #[test]
    fn test_dequantize_q8_0_block_wrong_size() {
        let block = [0u8; 33]; // Wrong size
        let result = dequantize_q8_0_block(&block);

        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q8_0_with_stats() {
        let mut data = Vec::new();

        // Block with scale=0.5, values from -10 to 21
        data.extend_from_slice(&0x3800u16.to_le_bytes());
        for i in -10..22i8 {
            data.push(i as u8);
        }

        let (values, stats) = dequantize_q8_0_with_stats(&data).unwrap();

        assert_eq!(values.len(), 32);
        assert_eq!(stats.num_blocks, 1);
        assert_eq!(stats.num_elements, 32);
        assert!(approx_eq(stats.min_scale, 0.5));
        assert!(approx_eq(stats.max_scale, 0.5));
        assert!(approx_eq(stats.min_value, -5.0)); // 0.5 * (-10)
        assert!(approx_eq(stats.max_value, 10.5)); // 0.5 * 21
        assert_eq!(stats.zero_count, 1); // quant=0 gives value=0
    }

    #[test]
    fn test_q8_0_element_count() {
        assert_eq!(q8_0_element_count(0), Some(0));
        assert_eq!(q8_0_element_count(34), Some(32));
        assert_eq!(q8_0_element_count(68), Some(64));
        assert_eq!(q8_0_element_count(35), None); // Invalid
    }

    #[test]
    fn test_q8_0_data_size() {
        assert_eq!(q8_0_data_size(0), Some(0));
        assert_eq!(q8_0_data_size(32), Some(34));
        assert_eq!(q8_0_data_size(64), Some(68));
        assert_eq!(q8_0_data_size(33), None); // Not multiple of 32
    }

    #[test]
    fn test_create_q8_0_block() {
        // Create block from known values
        let values: Vec<f32> = (0..32).map(|i| i as f32).collect();
        let block = create_q8_0_block(&values);

        // Dequantize and verify roundtrip
        let recovered = dequantize_q8_0_block(&block).unwrap();

        // Q8_0 has limited precision, so we allow some error
        for (i, (&orig, &recov)) in values.iter().zip(recovered.iter()).enumerate() {
            let error = (orig - recov).abs();
            assert!(
                error < 0.5,
                "Roundtrip error at {i}: orig={orig}, recovered={recov}, error={error}"
            );
        }
    }

    #[test]
    fn test_create_q8_0_block_zeros() {
        let values = vec![0.0f32; 32];
        let block = create_q8_0_block(&values);
        let recovered = dequantize_q8_0_block(&block).unwrap();

        for &v in &recovered {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn test_create_q8_0_block_negative() {
        let values: Vec<f32> = (-16..16).map(|i| i as f32).collect();
        let block = create_q8_0_block(&values);
        let recovered = dequantize_q8_0_block(&block).unwrap();

        for (orig, recov) in values.iter().zip(recovered.iter()) {
            let error = (orig - recov).abs();
            assert!(error < 0.5);
        }
    }

    #[test]
    fn test_dequant_stats_sparsity() {
        let mut stats = DequantStats::new();
        stats.num_elements = 100;
        stats.zero_count = 25;

        assert!((stats.sparsity() - 0.25).abs() < 1e-6);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_dequantize_q8_0_large() {
        // Test with multiple blocks
        let num_blocks = 100;
        let mut data = Vec::with_capacity(num_blocks * Q8_0_BLOCK_SIZE);

        for block_idx in 0..num_blocks {
            // Varying scales
            let scale = (block_idx + 1) as f32 * 0.01;
            let scale_f16 = super::super::quantization::f32_to_f16(scale);
            data.extend_from_slice(&scale_f16.to_le_bytes());

            // Varying quants
            for i in 0..32u8 {
                data.push(i);
            }
        }

        let values = dequantize_q8_0(&data).unwrap();

        assert_eq!(values.len(), num_blocks * 32);

        // Verify first block (scale ≈ 0.01)
        assert!(values[0].abs() < 0.01); // 0.01 * 0 = 0
        assert!((values[1] - 0.01).abs() < 0.01); // 0.01 * 1 ≈ 0.01
    }

    // =========================================================================
    // Q4_0 Tests
    // =========================================================================

    const Q4_0_EPSILON: f32 = 0.5; // Q4_0 has lower precision than Q8_0

    fn q4_0_approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < Q4_0_EPSILON
    }

    #[test]
    fn test_dequantize_q4_0_single_block() {
        // Create a Q4_0 block with scale=1.0
        // Each byte packs 2 values: low nibble (bits 0-3) and high nibble (bits 4-7)
        // Value = scale * (nibble - 8), so nibble=8 -> 0, nibble=9 -> 1, etc.
        let mut block_data = Vec::new();

        // Scale = 1.0 in F16 = 0x3C00
        block_data.extend_from_slice(&0x3C00u16.to_le_bytes());

        // 16 bytes for 32 values
        // We'll create values: 0, 1, 0, 1, 0, 1, ... (alternating)
        // nibble=8 gives 0, nibble=9 gives 1
        for _ in 0..16 {
            // Low nibble = 8 (value 0), high nibble = 9 (value 1)
            block_data.push(0x98); // (9 << 4) | 8
        }

        let values = dequantize_q4_0(&block_data).unwrap();

        assert_eq!(values.len(), 32);
        for i in 0..32 {
            let expected = if i % 2 == 0 { 0.0 } else { 1.0 };
            assert!(
                q4_0_approx_eq(values[i], expected),
                "Mismatch at {i}: expected {expected}, got {}",
                values[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q4_0_negative_values() {
        let mut block_data = Vec::new();

        // Scale = 2.0 in F16 = 0x4000
        block_data.extend_from_slice(&0x4000u16.to_le_bytes());

        // Create values that span the range
        // nibble=0 -> -8, nibble=15 -> +7
        // Let's do: (-8, +7) pairs
        for _ in 0..16 {
            // Low nibble = 0 (value = 2.0 * (0-8) = -16)
            // High nibble = 15 (value = 2.0 * (15-8) = 14)
            block_data.push(0xF0); // (15 << 4) | 0
        }

        let values = dequantize_q4_0(&block_data).unwrap();

        assert_eq!(values.len(), 32);
        // Even indices: -16.0
        assert!(q4_0_approx_eq(values[0], -16.0));
        // Odd indices: +14.0
        assert!(q4_0_approx_eq(values[1], 14.0));
    }

    #[test]
    fn test_dequantize_q4_0_multiple_blocks() {
        let mut data = Vec::new();

        // Block 1: scale=1.0, all zeros (nibbles = 8)
        data.extend_from_slice(&0x3C00u16.to_le_bytes());
        data.extend_from_slice(&[0x88u8; 16]); // (8 << 4) | 8 = 0x88

        // Block 2: scale=0.5, alternating -4 and +3
        // -4 = 0.5 * (4-8), +3 = 0.5 * (14-8) = 0.5 * 6 = 3
        // Wait, let me recalculate: +3 needs (q-8)*0.5 = 3, so q-8 = 6, q = 14
        data.extend_from_slice(&0x3800u16.to_le_bytes()); // scale=0.5
        data.extend_from_slice(&[0xE4u8; 16]); // (14 << 4) | 4 = 0xE4

        let values = dequantize_q4_0(&data).unwrap();

        assert_eq!(values.len(), 64);

        // First block: all zeros
        for &v in &values[0..32] {
            assert!(q4_0_approx_eq(v, 0.0), "Expected 0.0, got {v}");
        }

        // Second block: alternating -2.0 and +3.0
        // Actually: 0.5 * (4-8) = 0.5 * (-4) = -2.0
        //           0.5 * (14-8) = 0.5 * 6 = 3.0
        for i in 32..64 {
            let expected = if (i - 32) % 2 == 0 { -2.0 } else { 3.0 };
            assert!(
                q4_0_approx_eq(values[i], expected),
                "Mismatch at {i}: expected {expected}, got {}",
                values[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q4_0_invalid_length() {
        // Not a multiple of 18 bytes
        let data = vec![0u8; 19];
        let result = dequantize_q4_0(&data);

        assert!(result.is_err());
        assert!(matches!(result, Err(GgufError::AlignmentError { .. })));
    }

    #[test]
    fn test_dequantize_q4_0_empty() {
        let data: Vec<u8> = vec![];
        let values = dequantize_q4_0(&data).unwrap();

        assert!(values.is_empty());
    }

    #[test]
    fn test_dequantize_q4_0_into() {
        let mut block_data = Vec::new();
        block_data.extend_from_slice(&0x3C00u16.to_le_bytes()); // scale=1.0
        block_data.extend_from_slice(&[0x88u8; 16]); // all zeros

        let mut buffer = Vec::new();
        dequantize_q4_0_into(&block_data, &mut buffer).unwrap();

        assert_eq!(buffer.len(), 32);
        for &v in &buffer {
            assert!(q4_0_approx_eq(v, 0.0));
        }

        // Reuse buffer
        block_data.clear();
        block_data.extend_from_slice(&0x4000u16.to_le_bytes()); // scale=2.0
        block_data.extend_from_slice(&[0x99u8; 16]); // nibbles = 9, value = 2*(9-8) = 2

        dequantize_q4_0_into(&block_data, &mut buffer).unwrap();
        assert_eq!(buffer.len(), 32);
        for &v in &buffer {
            assert!(q4_0_approx_eq(v, 2.0));
        }
    }

    #[test]
    fn test_dequantize_q4_0_block() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&0x3C00u16.to_le_bytes()); // scale=1.0
        // Create sequential values: 0, 1, 2, 3, 4, 5, 6, 7, -8, -7, ...
        // This requires nibbles: 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, ...
        for i in 0..16 {
            let low = ((i * 2) % 16) as u8; // Will wrap: 0,2,4,6,8,10,12,14,0,2,...
            let high = ((i * 2 + 1) % 16) as u8;
            block[2 + i] = low | (high << 4);
        }

        let values = dequantize_q4_0_block(&block).unwrap();

        assert_eq!(values.len(), 32);
        // Verify the pattern
        for i in 0..32 {
            let nibble = (i % 16) as i32;
            let expected = (nibble - 8) as f32;
            assert!(
                q4_0_approx_eq(values[i], expected),
                "Mismatch at {i}: expected {expected}, got {}",
                values[i]
            );
        }
    }

    #[test]
    fn test_dequantize_q4_0_block_wrong_size() {
        let block = [0u8; 17]; // Wrong size
        let result = dequantize_q4_0_block(&block);

        assert!(result.is_err());
    }

    #[test]
    fn test_dequantize_q4_0_with_stats() {
        let mut data = Vec::new();

        // Block with scale=1.0, values from -8 to +7 (nibbles 0 to 15)
        data.extend_from_slice(&0x3C00u16.to_le_bytes());
        for i in 0..16u8 {
            // Pack sequential nibbles
            let low = i;
            let high = (i + 1) % 16;
            data.push(low | (high << 4));
        }

        let (values, stats) = dequantize_q4_0_with_stats(&data).unwrap();

        assert_eq!(values.len(), 32);
        assert_eq!(stats.num_blocks, 1);
        assert_eq!(stats.num_elements, 32);
        assert!(approx_eq(stats.min_scale, 1.0));
        assert!(approx_eq(stats.max_scale, 1.0));
        // Min value should be -8 (nibble 0)
        assert!(q4_0_approx_eq(stats.min_value, -8.0));
        // Max value should be +7 (nibble 15)
        assert!(q4_0_approx_eq(stats.max_value, 7.0));
    }

    #[test]
    fn test_q4_0_element_count() {
        assert_eq!(q4_0_element_count(0), Some(0));
        assert_eq!(q4_0_element_count(18), Some(32));
        assert_eq!(q4_0_element_count(36), Some(64));
        assert_eq!(q4_0_element_count(19), None); // Invalid
    }

    #[test]
    fn test_q4_0_data_size() {
        assert_eq!(q4_0_data_size(0), Some(0));
        assert_eq!(q4_0_data_size(32), Some(18));
        assert_eq!(q4_0_data_size(64), Some(36));
        assert_eq!(q4_0_data_size(33), None); // Not multiple of 32
    }

    #[test]
    fn test_create_q4_0_block() {
        // Create block from known values
        let values: Vec<f32> = (-8..24).map(|i| i as f32).take(32).collect();
        let block = create_q4_0_block(&values);

        // Dequantize and verify roundtrip
        let recovered = dequantize_q4_0_block(&block).unwrap();

        // Q4_0 has lower precision, so we allow larger error
        for (i, (&orig, &recov)) in values.iter().zip(recovered.iter()).enumerate() {
            let error = (orig - recov).abs();
            // With only 4 bits, error can be significant
            assert!(
                error < 3.0,
                "Roundtrip error at {i}: orig={orig}, recovered={recov}, error={error}"
            );
        }
    }

    #[test]
    fn test_create_q4_0_block_zeros() {
        let values = vec![0.0f32; 32];
        let block = create_q4_0_block(&values);
        let recovered = dequantize_q4_0_block(&block).unwrap();

        for &v in &recovered {
            assert!(q4_0_approx_eq(v, 0.0));
        }
    }

    #[test]
    fn test_create_q4_0_block_small_values() {
        // Test with values that fit well in 4-bit range
        let values: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let block = create_q4_0_block(&values);
        let recovered = dequantize_q4_0_block(&block).unwrap();

        for (orig, recov) in values.iter().zip(recovered.iter()) {
            let error = (orig - recov).abs();
            assert!(error < 2.0);
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_dequantize_q4_0_large() {
        // Test with multiple blocks
        let num_blocks = 100;
        let mut data = Vec::with_capacity(num_blocks * Q4_0_BLOCK_SIZE);

        for block_idx in 0..num_blocks {
            // Varying scales
            let scale = (block_idx + 1) as f32 * 0.1;
            let scale_f16 = super::super::quantization::f32_to_f16(scale);
            data.extend_from_slice(&scale_f16.to_le_bytes());

            // All nibbles = 8 (value = 0)
            data.extend_from_slice(&[0x88u8; 16]);
        }

        let values = dequantize_q4_0(&data).unwrap();

        assert_eq!(values.len(), num_blocks * 32);

        // All values should be 0 (scale * (8-8) = 0)
        for &v in &values {
            assert!(q4_0_approx_eq(v, 0.0));
        }
    }
}