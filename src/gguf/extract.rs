//! Tensor data extraction from GGUF files.
//!
//! This module provides functionality to extract tensor data from GGUF files
//! and convert it into usable `Tensor<f32>` instances. It handles:
//!
//! - Byte alignment verification
//! - Endianness conversion (GGUF uses little-endian)
//! - Shape validation and tensor construction
//! - Type-specific extraction (F32, F16, quantized types)
//!
//! # Example
//!
//! ```no_run
//! use llm_engine::gguf::{GgufLoader, TensorExtractor};
//!
//! let loader = GgufLoader::open("model.gguf")?;
//! let extractor = TensorExtractor::new(&loader);
//!
//! // Extract an F32 tensor
//! let tensor = extractor.extract_f32("model.embed_tokens.weight")?;
//! println!("Extracted tensor with shape: {:?}", tensor.dims());
//! # Ok::<(), llm_engine::gguf::GgufError>(())
//! ```
//!
//! # Memory Layout
//!
//! GGUF stores tensor data in little-endian format. For F32 tensors:
//! - Each element is 4 bytes (IEEE 754 single-precision)
//! - Data is stored contiguously with optional alignment padding
//! - Dimensions in GGUF are stored in column-major order but we convert
//!   to row-major for compatibility with standard tensor libraries
//!
//! # Alignment
//!
//! GGUF files typically align tensor data to 32 bytes for efficient
//! SIMD operations. The extractor verifies alignment for direct memory
//! access when possible, falling back to byte-by-byte copying when needed.

use crate::tensor::Tensor;

use super::dtype::GgmlType;
use super::error::{GgufError, Result};
use super::f16::extract_f16_as_f32;
use super::loader::GgufLoader;
use super::tensor_info::TensorInfo;

/// Tensor data extractor for GGUF files.
///
/// Provides methods to extract tensor data from a loaded GGUF file
/// and convert it into `Tensor<f32>` instances suitable for computation.
///
/// The extractor maintains a reference to the loader, allowing efficient
/// zero-copy access to memory-mapped tensor data when alignment permits.
#[derive(Debug)]
pub struct TensorExtractor<'a> {
    /// Reference to the GGUF loader containing memory-mapped data.
    loader: &'a GgufLoader,
}

impl<'a> TensorExtractor<'a> {
    /// Creates a new tensor extractor for the given loader.
    ///
    /// # Arguments
    ///
    /// * `loader` - Reference to a loaded GGUF file
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llm_engine::gguf::{GgufLoader, TensorExtractor};
    ///
    /// let loader = GgufLoader::open("model.gguf")?;
    /// let extractor = TensorExtractor::new(&loader);
    /// # Ok::<(), llm_engine::gguf::GgufError>(())
    /// ```
    #[must_use]
    pub const fn new(loader: &'a GgufLoader) -> Self {
        Self { loader }
    }

    /// Extracts an F32 tensor by name.
    ///
    /// This method retrieves the raw tensor data from the GGUF file and
    /// converts it into a `Tensor<f32>`. The tensor must be stored as
    /// F32 type in the file.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name (e.g., "model.layers.0.attention.wq.weight")
    ///
    /// # Returns
    ///
    /// A `Tensor<f32>` with the extracted data and proper shape.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not found in the file
    /// - The tensor is not F32 type
    /// - The data is corrupted or misaligned
    /// - Shape validation fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llm_engine::gguf::{GgufLoader, TensorExtractor};
    ///
    /// let loader = GgufLoader::open("model.gguf")?;
    /// let extractor = TensorExtractor::new(&loader);
    ///
    /// let weights = extractor.extract_f32("output.weight")?;
    /// println!("Shape: {:?}, Elements: {}", weights.dims(), weights.numel());
    /// # Ok::<(), llm_engine::gguf::GgufError>(())
    /// ```
    pub fn extract_f32(&self, name: &str) -> Result<Tensor<f32>> {
        // Get tensor info
        let info = self
            .loader
            .tensors()
            .get(name)
            .ok_or_else(|| GgufError::KeyNotFound {
                key: name.to_string(),
            })?;

        // Verify type is F32
        if info.dtype() != GgmlType::F32 {
            return Err(GgufError::TypeMismatch {
                expected: "F32".to_string(),
                got: info.dtype().name().to_string(),
            });
        }

        self.extract_f32_from_info(info)
    }

    /// Extracts an F32 tensor from tensor info.
    ///
    /// This is useful when you already have a `TensorInfo` reference
    /// and want to avoid a name lookup.
    ///
    /// # Arguments
    ///
    /// * `info` - Reference to the tensor's metadata
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not F32 type
    /// - Data extraction fails
    /// - Shape validation fails
    pub fn extract_f32_from_info(&self, info: &TensorInfo) -> Result<Tensor<f32>> {
        // Verify type is F32
        if info.dtype() != GgmlType::F32 {
            return Err(GgufError::TypeMismatch {
                expected: "F32".to_string(),
                got: info.dtype().name().to_string(),
            });
        }

        // Get raw tensor data
        let raw_data = self
            .loader
            .tensor_data_for(info)
            .ok_or_else(|| GgufError::TensorDataUnavailable {
                name: info.name().to_string(),
            })?;

        // Calculate expected size
        let numel = info.numel() as usize;
        let expected_bytes = numel * std::mem::size_of::<f32>();

        if raw_data.len() != expected_bytes {
            return Err(GgufError::ShapeMismatch {
                expected: expected_bytes,
                got: raw_data.len(),
            });
        }

        // Extract f32 values
        let data = extract_f32_from_bytes(raw_data.as_bytes())?;

        // Create tensor with row-major shape (reversed from GGUF's column-major)
        let shape = info.shape_row_major();

        Tensor::from_vec(data, shape).map_err(|e| GgufError::ShapeMismatch {
            expected: numel,
            got: e.to_string().parse().unwrap_or(0),
        })
    }

    /// Checks if a tensor can be extracted as F32.
    ///
    /// Returns `true` if the tensor exists and is stored as F32 type.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name to check
    #[must_use]
    pub fn can_extract_f32(&self, name: &str) -> bool {
        self.loader
            .tensors()
            .get(name)
            .map(|info| info.dtype() == GgmlType::F32)
            .unwrap_or(false)
    }

    /// Lists all F32 tensors in the file.
    ///
    /// Returns an iterator over tensor names that are stored as F32 type.
    pub fn f32_tensor_names(&self) -> impl Iterator<Item = &str> {
        self.loader
            .tensors()
            .iter()
            .filter(|info| info.dtype() == GgmlType::F32)
            .map(TensorInfo::name)
    }

    /// Returns the number of F32 tensors in the file.
    #[must_use]
    pub fn f32_tensor_count(&self) -> usize {
        self.loader
            .tensors()
            .iter()
            .filter(|info| info.dtype() == GgmlType::F32)
            .count()
    }

    // =========================================================================
    // F16 Tensor Extraction
    // =========================================================================

    /// Extracts an F16 tensor by name, converting to F32.
    ///
    /// F16 (half-precision) tensors are automatically dequantized to F32
    /// for computation. This is a common operation since many models store
    /// weights in F16 to reduce file size.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name in the GGUF file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor doesn't exist
    /// - The tensor is not F16 type
    /// - Data extraction fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llm_engine::gguf::{GgufLoader, TensorExtractor};
    ///
    /// let loader = GgufLoader::open("model.gguf")?;
    /// let extractor = TensorExtractor::new(&loader);
    ///
    /// // Extract F16 tensor and convert to F32
    /// let tensor = extractor.extract_f16("model.embed_tokens.weight")?;
    /// println!("Shape: {:?}", tensor.dims());
    /// # Ok::<(), llm_engine::gguf::GgufError>(())
    /// ```
    pub fn extract_f16(&self, name: &str) -> Result<Tensor<f32>> {
        let info = self
            .loader
            .tensors()
            .get(name)
            .ok_or_else(|| GgufError::KeyNotFound {
                key: name.to_string(),
            })?;

        // Verify type is F16
        if info.dtype() != GgmlType::F16 {
            return Err(GgufError::TypeMismatch {
                expected: "F16".to_string(),
                got: info.dtype().name().to_string(),
            });
        }

        self.extract_f16_from_info(info)
    }

    /// Extracts an F16 tensor from tensor info, converting to F32.
    ///
    /// This is useful when you already have a `TensorInfo` reference
    /// and want to avoid a name lookup.
    ///
    /// # Arguments
    ///
    /// * `info` - Reference to the tensor's metadata
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not F16 type
    /// - Data extraction fails
    pub fn extract_f16_from_info(&self, info: &TensorInfo) -> Result<Tensor<f32>> {
        // Verify type is F16
        if info.dtype() != GgmlType::F16 {
            return Err(GgufError::TypeMismatch {
                expected: "F16".to_string(),
                got: info.dtype().name().to_string(),
            });
        }

        // Get raw tensor data
        let raw_data = self
            .loader
            .tensor_data_for(info)
            .ok_or_else(|| GgufError::TensorDataUnavailable {
                name: info.name().to_string(),
            })?;

        // Calculate expected size (F16 = 2 bytes per element)
        let numel = info.numel() as usize;
        let expected_bytes = numel * 2;

        if raw_data.len() != expected_bytes {
            return Err(GgufError::ShapeMismatch {
                expected: expected_bytes,
                got: raw_data.len(),
            });
        }

        // Extract and convert F16 to F32
        let data = extract_f16_as_f32(raw_data.as_bytes())?;

        // Create tensor with row-major shape
        let shape = info.shape_row_major();

        Tensor::from_vec(data, shape).map_err(|e| GgufError::ShapeMismatch {
            expected: numel,
            got: e.to_string().parse().unwrap_or(0),
        })
    }

    /// Checks if a tensor can be extracted as F16.
    ///
    /// Returns `true` if the tensor exists and is stored as F16 type.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name to check
    #[must_use]
    pub fn can_extract_f16(&self, name: &str) -> bool {
        self.loader
            .tensors()
            .get(name)
            .map(|info| info.dtype() == GgmlType::F16)
            .unwrap_or(false)
    }

    /// Lists all F16 tensors in the file.
    ///
    /// Returns an iterator over tensor names that are stored as F16 type.
    pub fn f16_tensor_names(&self) -> impl Iterator<Item = &str> {
        self.loader
            .tensors()
            .iter()
            .filter(|info| info.dtype() == GgmlType::F16)
            .map(TensorInfo::name)
    }

    /// Returns the number of F16 tensors in the file.
    #[must_use]
    pub fn f16_tensor_count(&self) -> usize {
        self.loader
            .tensors()
            .iter()
            .filter(|info| info.dtype() == GgmlType::F16)
            .count()
    }

    // =========================================================================
    // Q8_0 Tensor Extraction (Dequantization)
    // =========================================================================

    /// Extracts a Q8_0 tensor by name, dequantizing to F32.
    ///
    /// Q8_0 is an 8-bit quantization format with a shared scale per block
    /// of 32 elements. This method dequantizes the data to full precision.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name in the GGUF file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor doesn't exist
    /// - The tensor is not Q8_0 type
    /// - Data extraction or dequantization fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llm_engine::gguf::{GgufLoader, TensorExtractor};
    ///
    /// let loader = GgufLoader::open("model.gguf")?;
    /// let extractor = TensorExtractor::new(&loader);
    ///
    /// // Extract Q8_0 tensor and dequantize to F32
    /// let tensor = extractor.extract_q8_0("model.layers.0.attn.weight")?;
    /// println!("Shape: {:?}", tensor.dims());
    /// # Ok::<(), llm_engine::gguf::GgufError>(())
    /// ```
    pub fn extract_q8_0(&self, name: &str) -> Result<Tensor<f32>> {
        let info = self
            .loader
            .tensors()
            .get(name)
            .ok_or_else(|| GgufError::KeyNotFound {
                key: name.to_string(),
            })?;

        // Verify type is Q8_0
        if info.dtype() != GgmlType::Q8_0 {
            return Err(GgufError::TypeMismatch {
                expected: "Q8_0".to_string(),
                got: info.dtype().name().to_string(),
            });
        }

        self.extract_q8_0_from_info(info)
    }

    /// Extracts a Q8_0 tensor from tensor info, dequantizing to F32.
    ///
    /// This is useful when you already have a `TensorInfo` reference
    /// and want to avoid a name lookup.
    ///
    /// # Arguments
    ///
    /// * `info` - Reference to the tensor's metadata
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor is not Q8_0 type
    /// - Data extraction or dequantization fails
    pub fn extract_q8_0_from_info(&self, info: &TensorInfo) -> Result<Tensor<f32>> {
        use super::dequant::{dequantize_q8_0, Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_SIZE};

        // Verify type is Q8_0
        if info.dtype() != GgmlType::Q8_0 {
            return Err(GgufError::TypeMismatch {
                expected: "Q8_0".to_string(),
                got: info.dtype().name().to_string(),
            });
        }

        // Get raw tensor data
        let raw_data = self
            .loader
            .tensor_data_for(info)
            .ok_or_else(|| GgufError::TensorDataUnavailable {
                name: info.name().to_string(),
            })?;

        // Calculate expected size
        let numel = info.numel() as usize;

        // Q8_0: must be a multiple of 32 elements
        if numel % Q8_0_BLOCK_ELEMENTS != 0 {
            return Err(GgufError::ShapeMismatch {
                expected: numel - (numel % Q8_0_BLOCK_ELEMENTS) + Q8_0_BLOCK_ELEMENTS,
                got: numel,
            });
        }

        let num_blocks = numel / Q8_0_BLOCK_ELEMENTS;
        let expected_bytes = num_blocks * Q8_0_BLOCK_SIZE;

        if raw_data.len() != expected_bytes {
            return Err(GgufError::ShapeMismatch {
                expected: expected_bytes,
                got: raw_data.len(),
            });
        }

        // Dequantize Q8_0 to F32
        let data = dequantize_q8_0(raw_data.as_bytes())?;

        // Create tensor with row-major shape
        let shape = info.shape_row_major();

        Tensor::from_vec(data, shape).map_err(|e| GgufError::ShapeMismatch {
            expected: numel,
            got: e.to_string().parse().unwrap_or(0),
        })
    }

    /// Checks if a tensor can be extracted as Q8_0.
    ///
    /// Returns `true` if the tensor exists and is stored as Q8_0 type.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name to check
    #[must_use]
    pub fn can_extract_q8_0(&self, name: &str) -> bool {
        self.loader
            .tensors()
            .get(name)
            .map(|info| info.dtype() == GgmlType::Q8_0)
            .unwrap_or(false)
    }

    /// Lists all Q8_0 tensors in the file.
    ///
    /// Returns an iterator over tensor names that are stored as Q8_0 type.
    pub fn q8_0_tensor_names(&self) -> impl Iterator<Item = &str> {
        self.loader
            .tensors()
            .iter()
            .filter(|info| info.dtype() == GgmlType::Q8_0)
            .map(TensorInfo::name)
    }

    /// Returns the number of Q8_0 tensors in the file.
    #[must_use]
    pub fn q8_0_tensor_count(&self) -> usize {
        self.loader
            .tensors()
            .iter()
            .filter(|info| info.dtype() == GgmlType::Q8_0)
            .count()
    }

    // =========================================================================
    // Generic Extraction (auto-detect type)
    // =========================================================================

    /// Extracts a tensor by name, automatically handling F32, F16, and Q8_0 types.
    ///
    /// This method detects the tensor's storage type and performs the
    /// appropriate extraction (direct copy for F32, conversion for F16,
    /// dequantization for Q8_0).
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name in the GGUF file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The tensor doesn't exist
    /// - The tensor type is not supported
    /// - Data extraction fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llm_engine::gguf::{GgufLoader, TensorExtractor};
    ///
    /// let loader = GgufLoader::open("model.gguf")?;
    /// let extractor = TensorExtractor::new(&loader);
    ///
    /// // Automatically handles F32, F16, or Q8_0 tensors
    /// let tensor = extractor.extract("model.embed_tokens.weight")?;
    /// # Ok::<(), llm_engine::gguf::GgufError>(())
    /// ```
    pub fn extract(&self, name: &str) -> Result<Tensor<f32>> {
        let info = self
            .loader
            .tensors()
            .get(name)
            .ok_or_else(|| GgufError::KeyNotFound {
                key: name.to_string(),
            })?;

        match info.dtype() {
            GgmlType::F32 => self.extract_f32_from_info(info),
            GgmlType::F16 => self.extract_f16_from_info(info),
            GgmlType::Q8_0 => self.extract_q8_0_from_info(info),
            other => Err(GgufError::TypeMismatch {
                expected: "F32, F16, or Q8_0".to_string(),
                got: other.name().to_string(),
            }),
        }
    }

    /// Returns a reference to the underlying loader.
    #[must_use]
    pub const fn loader(&self) -> &'a GgufLoader {
        self.loader
    }
}

/// Extracts f32 values from a byte slice.
///
/// This function handles the conversion from raw bytes to f32 values,
/// accounting for endianness (GGUF uses little-endian).
///
/// # Arguments
///
/// * `bytes` - Raw byte slice containing f32 data
///
/// # Returns
///
/// A vector of f32 values extracted from the bytes.
///
/// # Errors
///
/// Returns an error if:
/// - The byte slice length is not a multiple of 4
/// - Memory alignment cannot be satisfied
///
/// # Performance
///
/// When the input slice is properly aligned (common with memory-mapped files),
/// this function can perform a fast reinterpretation. Otherwise, it falls back
/// to byte-by-byte conversion.
#[allow(clippy::cast_ptr_alignment)]
pub fn extract_f32_from_bytes(bytes: &[u8]) -> Result<Vec<f32>> {
    // Verify length is multiple of f32 size
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(GgufError::AlignmentError {
            expected: std::mem::size_of::<f32>(),
            actual: bytes.len() % std::mem::size_of::<f32>(),
        });
    }

    let count = bytes.len() / std::mem::size_of::<f32>();
    let mut result = Vec::with_capacity(count);

    // Check if we can do a fast aligned copy
    let ptr = bytes.as_ptr();
    let is_aligned = ptr.align_offset(std::mem::align_of::<f32>()) == 0;

    if is_aligned && cfg!(target_endian = "little") {
        // Fast path: aligned data on little-endian system
        // SAFETY: We verified alignment and length, and f32 is Copy
        let f32_slice = unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), count) };
        result.extend_from_slice(f32_slice);
    } else {
        // Slow path: handle unaligned or big-endian systems
        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            result.push(value);
        }
    }

    Ok(result)
}

/// Extracts f32 values from a byte slice with a provided buffer.
///
/// This variant allows reusing a buffer to avoid allocations during
/// repeated extractions.
///
/// # Arguments
///
/// * `bytes` - Raw byte slice containing f32 data
/// * `buffer` - Destination buffer (will be cleared and filled)
///
/// # Errors
///
/// Returns an error if the byte slice length is not a multiple of 4.
pub fn extract_f32_into(bytes: &[u8], buffer: &mut Vec<f32>) -> Result<()> {
    // Verify length is multiple of f32 size
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(GgufError::AlignmentError {
            expected: std::mem::size_of::<f32>(),
            actual: bytes.len() % std::mem::size_of::<f32>(),
        });
    }

    let count = bytes.len() / std::mem::size_of::<f32>();
    buffer.clear();
    buffer.reserve(count);

    // Check alignment for fast path
    let ptr = bytes.as_ptr();
    let is_aligned = ptr.align_offset(std::mem::align_of::<f32>()) == 0;

    if is_aligned && cfg!(target_endian = "little") {
        // Fast path: aligned data on little-endian system
        // SAFETY: We verified alignment and length
        let f32_slice =
            unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), count) };
        buffer.extend_from_slice(f32_slice);
    } else {
        // Slow path: handle unaligned or big-endian systems
        for chunk in bytes.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            buffer.push(value);
        }
    }

    Ok(())
}

/// Verifies that tensor data is accessible and valid.
///
/// This function performs validation checks on tensor data without
/// actually extracting it, useful for pre-flight validation.
///
/// # Arguments
///
/// * `loader` - Reference to the GGUF loader
/// * `name` - Tensor name to validate
///
/// # Errors
///
/// Returns an error if:
/// - The tensor is not found
/// - The data is not accessible
/// - Size validation fails
pub fn validate_tensor_data(loader: &GgufLoader, name: &str) -> Result<()> {
    let info = loader
        .tensors()
        .get(name)
        .ok_or_else(|| GgufError::KeyNotFound {
            key: name.to_string(),
        })?;

    let data = loader
        .tensor_data_for(info)
        .ok_or_else(|| GgufError::TensorDataUnavailable {
            name: name.to_string(),
        })?;

    let expected_size = info.size_bytes();
    if data.len() != expected_size {
        return Err(GgufError::ShapeMismatch {
            expected: expected_size,
            got: data.len(),
        });
    }

    Ok(())
}

/// Information about extraction capabilities for a tensor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExtractionInfo {
    /// Tensor name.
    pub name: String,
    /// Data type in the file.
    pub dtype: GgmlType,
    /// Shape (row-major order).
    pub shape: Vec<usize>,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Number of elements.
    pub numel: usize,
    /// Whether direct F32 extraction is supported (no conversion needed).
    pub supports_f32_direct: bool,
    /// Whether F16 extraction (with conversion to F32) is supported.
    pub supports_f16_extraction: bool,
    /// Whether Q8_0 dequantization is supported.
    pub supports_q8_0_dequant: bool,
    /// Whether the tensor is quantized.
    pub is_quantized: bool,
}

impl ExtractionInfo {
    /// Creates extraction info from tensor info.
    #[must_use]
    pub fn from_tensor_info(info: &TensorInfo) -> Self {
        Self {
            name: info.name().to_string(),
            dtype: info.dtype(),
            shape: info.shape_row_major(),
            size_bytes: info.size_bytes(),
            numel: info.numel() as usize,
            supports_f32_direct: info.dtype() == GgmlType::F32,
            supports_f16_extraction: info.dtype() == GgmlType::F16,
            supports_q8_0_dequant: info.dtype() == GgmlType::Q8_0,
            is_quantized: info.is_quantized(),
        }
    }

    /// Returns true if the tensor can be extracted to F32 (directly, via conversion, or dequantization).
    #[must_use]
    pub fn can_extract_to_f32(&self) -> bool {
        self.supports_f32_direct || self.supports_f16_extraction || self.supports_q8_0_dequant
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_f32_from_bytes_aligned() {
        // Create aligned f32 data
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let extracted = extract_f32_from_bytes(&bytes).unwrap();

        assert_eq!(extracted.len(), 5);
        for (i, &val) in extracted.iter().enumerate() {
            assert!(
                (val - values[i]).abs() < 1e-6,
                "Mismatch at index {i}: expected {}, got {val}",
                values[i]
            );
        }
    }

    #[test]
    fn test_extract_f32_from_bytes_special_values() {
        // Test special floating point values
        let values: Vec<f32> = vec![
            0.0,
            -0.0,
            1.0,
            -1.0,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            f32::EPSILON,
        ];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let extracted = extract_f32_from_bytes(&bytes).unwrap();

        assert_eq!(extracted.len(), values.len());
        for (i, &val) in extracted.iter().enumerate() {
            if values[i].is_nan() {
                assert!(val.is_nan(), "Expected NaN at index {i}");
            } else {
                assert_eq!(val.to_bits(), values[i].to_bits(), "Mismatch at index {i}");
            }
        }
    }

    #[test]
    fn test_extract_f32_from_bytes_invalid_length() {
        // Not a multiple of 4 bytes
        let bytes = vec![0u8; 7];
        let result = extract_f32_from_bytes(&bytes);

        assert!(result.is_err());
        assert!(matches!(result, Err(GgufError::AlignmentError { .. })));
    }

    #[test]
    fn test_extract_f32_from_bytes_empty() {
        let bytes: Vec<u8> = vec![];
        let extracted = extract_f32_from_bytes(&bytes).unwrap();

        assert!(extracted.is_empty());
    }

    #[test]
    fn test_extract_f32_into_reuses_buffer() {
        let values1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let bytes1: Vec<u8> = values1.iter().flat_map(|v| v.to_le_bytes()).collect();

        let values2: Vec<f32> = vec![4.0, 5.0];
        let bytes2: Vec<u8> = values2.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut buffer = Vec::new();

        // First extraction
        extract_f32_into(&bytes1, &mut buffer).unwrap();
        assert_eq!(buffer.len(), 3);

        // Second extraction reuses buffer
        extract_f32_into(&bytes2, &mut buffer).unwrap();
        assert_eq!(buffer.len(), 2);
        assert!((buffer[0] - 4.0).abs() < 1e-6);
        assert!((buffer[1] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_extraction_info_from_tensor_info() {
        // Create a mock TensorInfo
        // Note: This test would need actual TensorInfo in a real scenario
        // For now, we test the struct construction
        let info = ExtractionInfo {
            name: "test.weight".to_string(),
            dtype: GgmlType::F32,
            shape: vec![4, 8],
            size_bytes: 128,
            numel: 32,
            supports_f32_direct: true,
            supports_f16_extraction: false,
            supports_q8_0_dequant: false,
            is_quantized: false,
        };

        assert_eq!(info.name, "test.weight");
        assert!(info.supports_f32_direct);
        assert!(!info.supports_f16_extraction);
        assert!(!info.supports_q8_0_dequant);
        assert!(!info.is_quantized);
        assert!(info.can_extract_to_f32());
    }

    #[test]
    fn test_extraction_info_quantized() {
        let info = ExtractionInfo {
            name: "test.quantized".to_string(),
            dtype: GgmlType::Q4_0,
            shape: vec![256, 256],
            size_bytes: 36864, // Q4_0 compressed size
            numel: 65536,
            supports_f32_direct: false,
            supports_f16_extraction: false,
            supports_q8_0_dequant: false,
            is_quantized: true,
        };

        assert!(!info.supports_f32_direct);
        assert!(!info.supports_f16_extraction);
        assert!(!info.supports_q8_0_dequant);
        assert!(info.is_quantized);
        assert!(!info.can_extract_to_f32()); // Q4_0 not yet supported
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_extract_f32_large_tensor() {
        // Test with a larger tensor to verify performance path
        let count = 10000;
        let values: Vec<f32> = (0..count).map(|i| i as f32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let extracted = extract_f32_from_bytes(&bytes).unwrap();

        assert_eq!(extracted.len(), count);
        // Verify first and last values
        assert!((extracted[0] - 0.0).abs() < 1e-6);
        assert!((extracted[count - 1] - (count - 1) as f32).abs() < 1e-6);
    }

    #[test]
    fn test_extract_f32_infinity_and_nan() {
        let values: Vec<f32> = vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();

        let extracted = extract_f32_from_bytes(&bytes).unwrap();

        assert!(extracted[0].is_infinite() && extracted[0].is_sign_positive());
        assert!(extracted[1].is_infinite() && extracted[1].is_sign_negative());
        assert!(extracted[2].is_nan());
    }

    // =========================================================================
    // F16 Extraction Tests
    // =========================================================================

    #[test]
    fn test_extract_f16_as_f32_basic() {
        use super::super::quantization::f32_to_f16;

        // Create F16 bytes for [1.0, 2.0, 0.5]
        let f16_values = vec![
            f32_to_f16(1.0),
            f32_to_f16(2.0),
            f32_to_f16(0.5),
        ];
        let bytes: Vec<u8> = f16_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let extracted = extract_f16_as_f32(&bytes).unwrap();

        assert_eq!(extracted.len(), 3);
        assert!((extracted[0] - 1.0).abs() < 1e-3);
        assert!((extracted[1] - 2.0).abs() < 1e-3);
        assert!((extracted[2] - 0.5).abs() < 1e-3);
    }

    #[test]
    fn test_extract_f16_as_f32_special_values() {
        use super::super::f16::constants;

        // Create bytes for special values
        let f16_values: Vec<u16> = vec![
            constants::F16_ZERO,
            constants::F16_NEG_ZERO,
            constants::F16_INFINITY,
            constants::F16_NEG_INFINITY,
            constants::F16_NAN,
        ];
        let bytes: Vec<u8> = f16_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let extracted = extract_f16_as_f32(&bytes).unwrap();

        assert_eq!(extracted.len(), 5);
        assert_eq!(extracted[0], 0.0);
        assert!(extracted[1].is_sign_negative() && extracted[1] == 0.0);
        assert!(extracted[2].is_infinite() && extracted[2].is_sign_positive());
        assert!(extracted[3].is_infinite() && extracted[3].is_sign_negative());
        assert!(extracted[4].is_nan());
    }

    #[test]
    fn test_extract_f16_as_f32_invalid_length() {
        // 3 bytes is not a multiple of 2
        let bytes = vec![0u8; 3];
        let result = extract_f16_as_f32(&bytes);

        assert!(result.is_err());
        assert!(matches!(result, Err(GgufError::AlignmentError { .. })));
    }

    #[test]
    fn test_extract_f16_as_f32_empty() {
        let bytes: Vec<u8> = vec![];
        let extracted = extract_f16_as_f32(&bytes).unwrap();

        assert!(extracted.is_empty());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_extract_f16_as_f32_large_batch() {
        use super::super::quantization::f32_to_f16;

        // Test with a larger batch
        let count = 10000;
        let f16_values: Vec<u16> = (0..count)
            .map(|i| f32_to_f16((i as f32) * 0.01))
            .collect();
        let bytes: Vec<u8> = f16_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        let extracted = extract_f16_as_f32(&bytes).unwrap();

        assert_eq!(extracted.len(), count);
        // Verify samples (allowing for F16 precision loss)
        assert!((extracted[0] - 0.0).abs() < 1e-3);
        assert!((extracted[100] - 1.0).abs() < 1e-2);
        assert!((extracted[1000] - 10.0).abs() < 0.1);
    }

    #[test]
    fn test_extraction_info_f16() {
        let info = ExtractionInfo {
            name: "test.f16_weight".to_string(),
            dtype: GgmlType::F16,
            shape: vec![4, 8],
            size_bytes: 64, // 32 elements * 2 bytes
            numel: 32,
            supports_f32_direct: false,
            supports_f16_extraction: true,
            supports_q8_0_dequant: false,
            is_quantized: false,
        };

        assert_eq!(info.name, "test.f16_weight");
        assert!(!info.supports_f32_direct); // F16 needs conversion
        assert!(info.supports_f16_extraction);
        assert!(!info.supports_q8_0_dequant);
        assert!(!info.is_quantized); // F16 is not quantized, just half-precision
        assert!(info.can_extract_to_f32()); // Can extract via conversion
    }

    #[test]
    fn test_extraction_info_q8_0() {
        let info = ExtractionInfo {
            name: "test.q8_0_weight".to_string(),
            dtype: GgmlType::Q8_0,
            shape: vec![256, 256],
            size_bytes: 65536 / 32 * 34, // 34 bytes per 32 elements
            numel: 65536,
            supports_f32_direct: false,
            supports_f16_extraction: false,
            supports_q8_0_dequant: true,
            is_quantized: true,
        };

        assert_eq!(info.name, "test.q8_0_weight");
        assert!(!info.supports_f32_direct);
        assert!(!info.supports_f16_extraction);
        assert!(info.supports_q8_0_dequant);
        assert!(info.is_quantized);
        assert!(info.can_extract_to_f32()); // Can extract via dequantization
    }
}