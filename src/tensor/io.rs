//! Binary I/O for tensor serialization.
//!
//! Simple binary format for saving/loading tensors:
//!
//! ```text
//! Format:
//!   Magic: 4 bytes ("TENS")
//!   Version: 1 byte
//!   DType: 1 byte (0=f32, 1=f64, 2=i32, 3=i64)
//!   NDim: 4 bytes (u32 LE)
//!   Dims: NDim * 8 bytes (u64 LE each)
//!   Data: numel * sizeof(dtype) bytes
//! ```
//!
//! This format is designed for debugging and testing, not production use.
//! For production, use GGUF or safetensors.

use super::error::{Result, TensorError};
use super::shape::Shape;
use super::Tensor;
use std::io::{Read, Write};

/// Magic bytes for tensor file format.
const MAGIC: &[u8; 4] = b"TENS";

/// Current format version.
const VERSION: u8 = 1;

/// Data type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DType {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    I64 = 3,
}

impl DType {
    fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::F32),
            1 => Some(Self::F64),
            2 => Some(Self::I32),
            3 => Some(Self::I64),
            _ => None,
        }
    }
}

/// Writes a tensor to a binary writer.
///
/// # Errors
///
/// Returns error if writing fails.
#[allow(clippy::cast_possible_truncation)]
pub fn write_tensor<W: Write>(tensor: &Tensor<f32>, writer: &mut W) -> Result<()> {
    // Magic
    writer.write_all(MAGIC).map_err(|e| io_error(&e))?;

    // Version
    writer.write_all(&[VERSION]).map_err(|e| io_error(&e))?;

    // DType
    writer.write_all(&[DType::F32 as u8]).map_err(|e| io_error(&e))?;

    // NDim (safe truncation: tensors won't have > 4B dimensions)
    let ndim = tensor.ndim() as u32;
    writer.write_all(&ndim.to_le_bytes()).map_err(|e| io_error(&e))?;

    // Dims
    for &dim in tensor.dims() {
        let dim_u64 = dim as u64;
        writer.write_all(&dim_u64.to_le_bytes()).map_err(|e| io_error(&e))?;
    }

    // Data
    let data = tensor.as_slice();
    for &val in data {
        writer.write_all(&val.to_le_bytes()).map_err(|e| io_error(&e))?;
    }

    Ok(())
}

/// Reads a tensor from a binary reader.
///
/// # Errors
///
/// Returns error if reading fails or format is invalid.
#[allow(clippy::cast_possible_truncation)]
pub fn read_tensor<R: Read>(reader: &mut R) -> Result<Tensor<f32>> {
    // Magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic).map_err(|e| io_error(&e))?;
    if &magic != MAGIC {
        return Err(TensorError::InvalidShape {
            reason: format!("invalid magic bytes: expected TENS, got {magic:?}"),
        });
    }

    // Version
    let mut version = [0u8; 1];
    reader.read_exact(&mut version).map_err(|e| io_error(&e))?;
    if version[0] != VERSION {
        return Err(TensorError::InvalidShape {
            reason: format!("unsupported version: {}", version[0]),
        });
    }

    // DType
    let mut dtype = [0u8; 1];
    reader.read_exact(&mut dtype).map_err(|e| io_error(&e))?;
    let dtype = DType::from_u8(dtype[0]).ok_or_else(|| TensorError::InvalidShape {
        reason: format!("unknown dtype: {}", dtype[0]),
    })?;

    if dtype != DType::F32 {
        return Err(TensorError::InvalidShape {
            reason: format!("expected F32, got {dtype:?}"),
        });
    }

    // NDim
    let mut ndim_bytes = [0u8; 4];
    reader.read_exact(&mut ndim_bytes).map_err(|e| io_error(&e))?;
    let ndim = u32::from_le_bytes(ndim_bytes) as usize;

    // Dims (safe truncation on 32-bit: dims won't exceed usize)
    let mut dims = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        let mut dim_bytes = [0u8; 8];
        reader.read_exact(&mut dim_bytes).map_err(|e| io_error(&e))?;
        dims.push(u64::from_le_bytes(dim_bytes) as usize);
    }

    let shape = Shape::new(dims);
    let numel = shape.numel();

    // Data
    let mut data = Vec::with_capacity(numel);
    for _ in 0..numel {
        let mut val_bytes = [0u8; 4];
        reader.read_exact(&mut val_bytes).map_err(|e| io_error(&e))?;
        data.push(f32::from_le_bytes(val_bytes));
    }

    Tensor::from_vec(data, shape)
}

fn io_error(e: &std::io::Error) -> TensorError {
    TensorError::InvalidShape {
        reason: format!("I/O error: {e}"),
    }
}

/// Saves a tensor to a file.
///
/// # Errors
///
/// Returns error if file cannot be created or writing fails.
pub fn save_tensor(tensor: &Tensor<f32>, path: &std::path::Path) -> Result<()> {
    let mut file = std::fs::File::create(path).map_err(|e| io_error(&e))?;
    write_tensor(tensor, &mut file)
}

/// Loads a tensor from a file.
///
/// # Errors
///
/// Returns error if file cannot be read or format is invalid.
pub fn load_tensor(path: &std::path::Path) -> Result<Tensor<f32>> {
    let mut file = std::fs::File::open(path).map_err(|e| io_error(&e))?;
    read_tensor(&mut file)
}

// Extension methods for Tensor
impl Tensor<f32> {
    /// Saves the tensor to a binary file.
    ///
    /// # Errors
    ///
    /// Returns error if saving fails.
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        save_tensor(self, path)
    }

    /// Loads a tensor from a binary file.
    ///
    /// # Errors
    ///
    /// Returns error if loading fails.
    pub fn load(path: &std::path::Path) -> Result<Self> {
        load_tensor(path)
    }

    /// Serializes the tensor to bytes.
    ///
    /// # Errors
    ///
    /// Returns error if serialization fails.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut buffer = Vec::new();
        write_tensor(self, &mut buffer)?;
        Ok(buffer)
    }

    /// Deserializes a tensor from bytes.
    ///
    /// # Errors
    ///
    /// Returns error if deserialization fails.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        read_tensor(&mut cursor)
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_roundtrip_memory() {
        let original = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        let mut buffer = Vec::new();
        write_tensor(&original, &mut buffer).unwrap();

        let mut cursor = Cursor::new(buffer);
        let loaded = read_tensor(&mut cursor).unwrap();

        assert_eq!(original.dims(), loaded.dims());
        for (a, b) in original.as_slice().iter().zip(loaded.as_slice().iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn test_roundtrip_bytes() {
        let original = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        let bytes = original.to_bytes().unwrap();
        let loaded = Tensor::from_bytes(&bytes).unwrap();

        assert_eq!(original.dims(), loaded.dims());
        for (a, b) in original.as_slice().iter().zip(loaded.as_slice().iter()) {
            assert!(approx_eq(*a, *b));
        }
    }

    #[test]
    fn test_format_structure() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]).unwrap();
        let bytes = tensor.to_bytes().unwrap();

        // Check magic
        assert_eq!(&bytes[0..4], b"TENS");
        // Check version
        assert_eq!(bytes[4], 1);
        // Check dtype (F32 = 0)
        assert_eq!(bytes[5], 0);
        // Check ndim (1)
        assert_eq!(u32::from_le_bytes([bytes[6], bytes[7], bytes[8], bytes[9]]), 1);
    }

    #[test]
    fn test_invalid_magic() {
        let bad_data = b"BADM\x01\x00\x01\x00\x00\x00\x02\x00\x00\x00\x00\x00\x00\x00";
        let result = Tensor::from_bytes(bad_data);
        assert!(result.is_err());
    }

    #[test]
    fn test_3d_tensor() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let original = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        let bytes = original.to_bytes().unwrap();
        let loaded = Tensor::from_bytes(&bytes).unwrap();

        assert_eq!(original.dims(), loaded.dims());
        assert_eq!(original.numel(), loaded.numel());
    }

    #[test]
    fn test_scalar() {
        let original = Tensor::from_vec(vec![42.0f32], vec![]).unwrap();

        let bytes = original.to_bytes().unwrap();
        let loaded = Tensor::from_bytes(&bytes).unwrap();

        assert_eq!(loaded.ndim(), 0);
        assert!(approx_eq(*loaded.as_slice().first().unwrap(), 42.0));
    }
}