//! Half-precision (F16) floating point utilities.
//!
//! This module provides functions for working with IEEE 754 half-precision
//! floating point values, which are commonly used in GGUF model files to
//! reduce storage size while maintaining reasonable precision.
//!
//! # IEEE 754 Half-Precision Format
//!
//! ```text
//! ┌───┬───────────┬────────────────────┐
//! │ S │  Exponent │     Mantissa       │
//! │ 1 │   5 bits  │     10 bits        │
//! └───┴───────────┴────────────────────┘
//!  15    14-10         9-0
//! ```
//!
//! - **Sign (S)**: 1 bit (0 = positive, 1 = negative)
//! - **Exponent**: 5 bits with bias of 15
//! - **Mantissa**: 10 bits (implicit leading 1 for normalized values)
//!
//! # Precision and Range
//!
//! | Property | F16 | F32 |
//! |----------|-----|-----|
//! | Exponent bits | 5 | 8 |
//! | Mantissa bits | 10 | 23 |
//! | Max value | ~65504 | ~3.4e38 |
//! | Min positive | ~6.1e-5 | ~1.2e-38 |
//! | Precision | ~3 decimal digits | ~7 decimal digits |
//!
//! # Example
//!
//! ```
//! use llm_engine::gguf::f16::{f16_to_f32_batch, extract_f16_as_f32};
//!
//! // Convert a batch of F16 values (stored as u16)
//! let f16_bits: Vec<u16> = vec![0x3C00, 0x4000, 0x4200]; // 1.0, 2.0, 3.0
//! let f32_values = f16_to_f32_batch(&f16_bits);
//! assert!((f32_values[0] - 1.0).abs() < 1e-3);
//! ```

use super::error::{GgufError, Result};
use super::quantization::{f16_to_f32, f32_to_f16};

/// Converts a batch of F16 values (stored as u16) to F32.
///
/// This function efficiently converts multiple half-precision values
/// to single-precision, which is required for computation.
///
/// # Arguments
///
/// * `f16_bits` - Slice of u16 values containing F16 bit patterns
///
/// # Returns
///
/// A vector of f32 values.
///
/// # Example
///
/// ```
/// use llm_engine::gguf::f16::f16_to_f32_batch;
///
/// let f16_bits = vec![0x3C00u16, 0x4000, 0x4200]; // 1.0, 2.0, 3.0
/// let f32_values = f16_to_f32_batch(&f16_bits);
///
/// assert!((f32_values[0] - 1.0).abs() < 1e-3);
/// assert!((f32_values[1] - 2.0).abs() < 1e-3);
/// assert!((f32_values[2] - 3.0).abs() < 1e-3);
/// ```
#[must_use]
pub fn f16_to_f32_batch(f16_bits: &[u16]) -> Vec<f32> {
    f16_bits.iter().map(|&bits| f16_to_f32(bits)).collect()
}

/// Converts a batch of F16 values into a provided buffer.
///
/// This variant allows reusing a buffer to avoid allocations during
/// repeated conversions.
///
/// # Arguments
///
/// * `f16_bits` - Slice of u16 values containing F16 bit patterns
/// * `buffer` - Destination buffer (will be cleared and filled)
///
/// # Example
///
/// ```
/// use llm_engine::gguf::f16::f16_to_f32_batch_into;
///
/// let f16_bits = vec![0x3C00u16]; // 1.0
/// let mut buffer = Vec::new();
/// f16_to_f32_batch_into(&f16_bits, &mut buffer);
///
/// assert_eq!(buffer.len(), 1);
/// assert!((buffer[0] - 1.0).abs() < 1e-3);
/// ```
pub fn f16_to_f32_batch_into(f16_bits: &[u16], buffer: &mut Vec<f32>) {
    buffer.clear();
    buffer.reserve(f16_bits.len());
    buffer.extend(f16_bits.iter().map(|&bits| f16_to_f32(bits)));
}

/// Converts a batch of F32 values to F16 (stored as u16).
///
/// This is useful for testing and potentially for saving tensors back
/// to F16 format.
///
/// # Arguments
///
/// * `f32_values` - Slice of f32 values to convert
///
/// # Returns
///
/// A vector of u16 values containing F16 bit patterns.
///
/// # Note
///
/// Values outside the F16 range will be clamped to infinity or zero.
/// Precision loss is expected for values requiring more than ~3 decimal digits.
#[must_use]
pub fn f32_to_f16_batch(f32_values: &[f32]) -> Vec<u16> {
    f32_values.iter().map(|&v| f32_to_f16(v)).collect()
}

/// Extracts F16 values from raw bytes and converts to F32.
///
/// This function handles the complete pipeline of:
/// 1. Interpreting bytes as u16 (little-endian)
/// 2. Converting F16 bit patterns to F32 values
///
/// # Arguments
///
/// * `bytes` - Raw byte slice containing F16 data (little-endian)
///
/// # Returns
///
/// A vector of f32 values.
///
/// # Errors
///
/// Returns an error if the byte slice length is not a multiple of 2.
///
/// # Example
///
/// ```
/// use llm_engine::gguf::f16::extract_f16_as_f32;
///
/// // F16 representation of 1.0 is 0x3C00 (little-endian: [0x00, 0x3C])
/// let bytes = vec![0x00, 0x3C, 0x00, 0x40]; // 1.0, 2.0
/// let values = extract_f16_as_f32(&bytes).unwrap();
///
/// assert_eq!(values.len(), 2);
/// assert!((values[0] - 1.0).abs() < 1e-3);
/// assert!((values[1] - 2.0).abs() < 1e-3);
/// ```
pub fn extract_f16_as_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    // Verify length is multiple of 2 (size of u16)
    if bytes.len() % 2 != 0 {
        return Err(GgufError::AlignmentError {
            expected: 2,
            actual: bytes.len() % 2,
        });
    }

    let count = bytes.len() / 2;
    let mut result = Vec::with_capacity(count);

    // Check alignment for potential fast path
    let ptr = bytes.as_ptr();
    let is_aligned = ptr.align_offset(std::mem::align_of::<u16>()) == 0;

    if is_aligned && cfg!(target_endian = "little") {
        // Fast path: aligned data on little-endian system
        // SAFETY: We verified alignment and length
        let u16_slice = unsafe { std::slice::from_raw_parts(ptr.cast::<u16>(), count) };
        result.extend(u16_slice.iter().map(|&bits| f16_to_f32(bits)));
    } else {
        // Slow path: handle unaligned or big-endian systems
        for chunk in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            result.push(f16_to_f32(bits));
        }
    }

    Ok(result)
}

/// Extracts F16 values from raw bytes into a provided buffer.
///
/// This variant allows reusing a buffer to avoid allocations.
///
/// # Arguments
///
/// * `bytes` - Raw byte slice containing F16 data (little-endian)
/// * `buffer` - Destination buffer (will be cleared and filled)
///
/// # Errors
///
/// Returns an error if the byte slice length is not a multiple of 2.
pub fn extract_f16_as_f32_into(bytes: &[u8], buffer: &mut Vec<f32>) -> Result<()> {
    // Verify length is multiple of 2
    if bytes.len() % 2 != 0 {
        return Err(GgufError::AlignmentError {
            expected: 2,
            actual: bytes.len() % 2,
        });
    }

    let count = bytes.len() / 2;
    buffer.clear();
    buffer.reserve(count);

    // Check alignment for potential fast path
    let ptr = bytes.as_ptr();
    let is_aligned = ptr.align_offset(std::mem::align_of::<u16>()) == 0;

    if is_aligned && cfg!(target_endian = "little") {
        // Fast path
        let u16_slice = unsafe { std::slice::from_raw_parts(ptr.cast::<u16>(), count) };
        buffer.extend(u16_slice.iter().map(|&bits| f16_to_f32(bits)));
    } else {
        // Slow path
        for chunk in bytes.chunks_exact(2) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            buffer.push(f16_to_f32(bits));
        }
    }

    Ok(())
}

/// Common F16 bit patterns for reference and testing.
pub mod constants {
    /// F16 representation of 0.0
    pub const F16_ZERO: u16 = 0x0000;

    /// F16 representation of -0.0
    pub const F16_NEG_ZERO: u16 = 0x8000;

    /// F16 representation of 1.0
    pub const F16_ONE: u16 = 0x3C00;

    /// F16 representation of -1.0
    pub const F16_NEG_ONE: u16 = 0xBC00;

    /// F16 representation of 2.0
    pub const F16_TWO: u16 = 0x4000;

    /// F16 representation of 0.5
    pub const F16_HALF: u16 = 0x3800;

    /// F16 representation of positive infinity
    pub const F16_INFINITY: u16 = 0x7C00;

    /// F16 representation of negative infinity
    pub const F16_NEG_INFINITY: u16 = 0xFC00;

    /// F16 representation of NaN (one of many)
    pub const F16_NAN: u16 = 0x7C01;

    /// Maximum finite F16 value (~65504)
    pub const F16_MAX: u16 = 0x7BFF;

    /// Minimum positive normalized F16 value (~6.1e-5)
    pub const F16_MIN_POSITIVE: u16 = 0x0400;

    /// Smallest positive subnormal F16 value (~5.96e-8)
    pub const F16_MIN_SUBNORMAL: u16 = 0x0001;

    /// F16 machine epsilon (~0.00097656)
    pub const F16_EPSILON: u16 = 0x1400;
}

/// Checks if an F16 value (as u16 bits) is NaN.
#[must_use]
pub const fn is_f16_nan(bits: u16) -> bool {
    // NaN: exponent is all 1s and mantissa is non-zero
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    exp == 0x1F && mant != 0
}

/// Checks if an F16 value (as u16 bits) is infinite.
#[must_use]
pub const fn is_f16_infinite(bits: u16) -> bool {
    // Infinity: exponent is all 1s and mantissa is zero
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    exp == 0x1F && mant == 0
}

/// Checks if an F16 value (as u16 bits) is zero (positive or negative).
#[must_use]
pub const fn is_f16_zero(bits: u16) -> bool {
    // Zero: all bits except sign are zero
    (bits & 0x7FFF) == 0
}

/// Checks if an F16 value (as u16 bits) is subnormal.
#[must_use]
pub const fn is_f16_subnormal(bits: u16) -> bool {
    // Subnormal: exponent is zero and mantissa is non-zero
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    exp == 0 && mant != 0
}

/// Returns the sign of an F16 value (0 for positive, 1 for negative).
#[must_use]
pub const fn f16_sign(bits: u16) -> u16 {
    (bits >> 15) & 1
}

/// Returns the biased exponent of an F16 value (0-31).
#[must_use]
pub const fn f16_exponent(bits: u16) -> u16 {
    (bits >> 10) & 0x1F
}

/// Returns the mantissa of an F16 value (0-1023).
#[must_use]
pub const fn f16_mantissa(bits: u16) -> u16 {
    bits & 0x3FF
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::constants::*;

    const EPSILON: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_infinite() && b.is_infinite() {
            return a.is_sign_positive() == b.is_sign_positive();
        }
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_f16_to_f32_batch_basic() {
        let f16_bits = vec![F16_ONE, F16_TWO, F16_HALF];
        let result = f16_to_f32_batch(&f16_bits);

        assert_eq!(result.len(), 3);
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 2.0));
        assert!(approx_eq(result[2], 0.5));
    }

    #[test]
    fn test_f16_to_f32_batch_special_values() {
        let f16_bits = vec![F16_ZERO, F16_NEG_ZERO, F16_INFINITY, F16_NEG_INFINITY, F16_NAN];
        let result = f16_to_f32_batch(&f16_bits);

        assert!(approx_eq(result[0], 0.0));
        assert!(approx_eq(result[1], -0.0));
        assert!(result[2].is_infinite() && result[2].is_sign_positive());
        assert!(result[3].is_infinite() && result[3].is_sign_negative());
        assert!(result[4].is_nan());
    }

    #[test]
    fn test_f16_to_f32_batch_into() {
        let f16_bits = vec![F16_ONE, F16_NEG_ONE];
        let mut buffer = Vec::new();

        f16_to_f32_batch_into(&f16_bits, &mut buffer);

        assert_eq!(buffer.len(), 2);
        assert!(approx_eq(buffer[0], 1.0));
        assert!(approx_eq(buffer[1], -1.0));

        // Test buffer reuse
        let f16_bits2 = vec![F16_TWO];
        f16_to_f32_batch_into(&f16_bits2, &mut buffer);
        assert_eq!(buffer.len(), 1);
        assert!(approx_eq(buffer[0], 2.0));
    }

    #[test]
    fn test_f32_to_f16_batch() {
        let f32_values = vec![1.0f32, 2.0, 0.5, -1.0];
        let result = f32_to_f16_batch(&f32_values);

        assert_eq!(result.len(), 4);
        assert_eq!(result[0], F16_ONE);
        assert_eq!(result[1], F16_TWO);
        assert_eq!(result[2], F16_HALF);
        assert_eq!(result[3], F16_NEG_ONE);
    }

    #[test]
    fn test_extract_f16_as_f32() {
        // F16 1.0 = 0x3C00, little-endian: [0x00, 0x3C]
        // F16 2.0 = 0x4000, little-endian: [0x00, 0x40]
        let bytes = vec![0x00, 0x3C, 0x00, 0x40];
        let result = extract_f16_as_f32(&bytes).unwrap();

        assert_eq!(result.len(), 2);
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 2.0));
    }

    #[test]
    fn test_extract_f16_as_f32_invalid_length() {
        let bytes = vec![0x00, 0x3C, 0x00]; // 3 bytes, not multiple of 2
        let result = extract_f16_as_f32(&bytes);

        assert!(result.is_err());
        assert!(matches!(result, Err(GgufError::AlignmentError { .. })));
    }

    #[test]
    fn test_extract_f16_as_f32_empty() {
        let bytes: Vec<u8> = vec![];
        let result = extract_f16_as_f32(&bytes).unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn test_extract_f16_as_f32_into() {
        let bytes = vec![0x00, 0x3C]; // F16 1.0
        let mut buffer = Vec::new();

        extract_f16_as_f32_into(&bytes, &mut buffer).unwrap();

        assert_eq!(buffer.len(), 1);
        assert!(approx_eq(buffer[0], 1.0));
    }

    #[test]
    fn test_f16_inspection_functions() {
        // Test is_f16_nan
        assert!(is_f16_nan(F16_NAN));
        assert!(!is_f16_nan(F16_ONE));
        assert!(!is_f16_nan(F16_INFINITY));

        // Test is_f16_infinite
        assert!(is_f16_infinite(F16_INFINITY));
        assert!(is_f16_infinite(F16_NEG_INFINITY));
        assert!(!is_f16_infinite(F16_NAN));
        assert!(!is_f16_infinite(F16_ONE));

        // Test is_f16_zero
        assert!(is_f16_zero(F16_ZERO));
        assert!(is_f16_zero(F16_NEG_ZERO));
        assert!(!is_f16_zero(F16_ONE));

        // Test is_f16_subnormal
        assert!(is_f16_subnormal(F16_MIN_SUBNORMAL));
        assert!(!is_f16_subnormal(F16_MIN_POSITIVE)); // Normalized
        assert!(!is_f16_subnormal(F16_ZERO));
    }

    #[test]
    fn test_f16_component_extraction() {
        // F16 1.0 = 0x3C00 = 0 01111 0000000000
        assert_eq!(f16_sign(F16_ONE), 0);
        assert_eq!(f16_exponent(F16_ONE), 15); // Bias is 15, so exp=15 means 2^0
        assert_eq!(f16_mantissa(F16_ONE), 0);

        // F16 -1.0 = 0xBC00 = 1 01111 0000000000
        assert_eq!(f16_sign(F16_NEG_ONE), 1);
        assert_eq!(f16_exponent(F16_NEG_ONE), 15);
        assert_eq!(f16_mantissa(F16_NEG_ONE), 0);

        // F16 2.0 = 0x4000 = 0 10000 0000000000
        assert_eq!(f16_sign(F16_TWO), 0);
        assert_eq!(f16_exponent(F16_TWO), 16); // 2^1
        assert_eq!(f16_mantissa(F16_TWO), 0);
    }

    #[test]
    fn test_roundtrip_precision() {
        // Test that common values roundtrip within F16 precision
        let test_values = [0.0f32, 1.0, -1.0, 2.0, 0.5, 100.0, 0.001];

        for &val in &test_values {
            let f16_bits = f32_to_f16(val);
            let back = f16_to_f32(f16_bits);
            
            if val == 0.0 {
                assert!(back == 0.0 || back == -0.0);
            } else {
                // F16 has ~3 decimal digits of precision
                let relative_error = ((back - val) / val).abs();
                assert!(
                    relative_error < 0.01,
                    "Roundtrip failed for {val}: got {back}, error {relative_error}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_large_batch_conversion() {
        // Test with a larger batch to verify performance path
        let count = 10000;
        let f16_bits: Vec<u16> = (0..count)
            .map(|i| f32_to_f16((i as f32) / 100.0))
            .collect();

        let result = f16_to_f32_batch(&f16_bits);

        assert_eq!(result.len(), count);
        // Spot check a few values
        assert!(approx_eq(result[0], 0.0));
        assert!(approx_eq(result[100], 1.0));
        assert!(approx_eq(result[1000], 10.0));
    }
}