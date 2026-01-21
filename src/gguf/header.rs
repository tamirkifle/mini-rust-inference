//! GGUF file header parsing.
//!
//! The GGUF header contains:
//! - Magic bytes: "GGUF" (4 bytes)
//! - Version: u32 (little-endian)
//! - Tensor count: u64 (little-endian)
//! - Metadata KV count: u64 (little-endian)
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use super::error::{GgufError, Result};
use std::io::Read;

/// GGUF magic bytes: "GGUF" in ASCII.
pub const GGUF_MAGIC: [u8; 4] = [0x47, 0x47, 0x55, 0x46];

/// Supported GGUF versions.
pub const SUPPORTED_VERSIONS: &[u32] = &[2, 3];

/// GGUF file header.
///
/// Contains metadata about the file structure including version,
/// tensor count, and metadata entry count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    /// GGUF format version (2 or 3).
    version: u32,

    /// Number of tensors in the file.
    tensor_count: u64,

    /// Number of metadata key-value pairs.
    metadata_kv_count: u64,
}

impl GgufHeader {
    /// Header size in bytes (magic + version + tensor_count + metadata_kv_count).
    pub const SIZE: usize = 4 + 4 + 8 + 8; // 24 bytes

    /// Creates a new GGUF header.
    #[must_use]
    pub const fn new(version: u32, tensor_count: u64, metadata_kv_count: u64) -> Self {
        Self {
            version,
            tensor_count,
            metadata_kv_count,
        }
    }

    /// Reads and parses a GGUF header from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Magic bytes don't match "GGUF"
    /// - Version is not supported (currently 2 or 3)
    /// - I/O error occurs during reading
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        // Read and validate magic
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic).map_err(|e| {
            if e.kind() == std::io::ErrorKind::UnexpectedEof {
                GgufError::UnexpectedEof { context: "magic bytes" }
            } else {
                GgufError::from(e)
            }
        })?;

        if magic != GGUF_MAGIC {
            return Err(GgufError::InvalidMagic { got: magic });
        }

        // Read version
        let version = read_u32(reader, "version")?;
        if !SUPPORTED_VERSIONS.contains(&version) {
            return Err(GgufError::UnsupportedVersion {
                version,
                supported: SUPPORTED_VERSIONS,
            });
        }

        // Read counts
        let tensor_count = read_u64(reader, "tensor count")?;
        let metadata_kv_count = read_u64(reader, "metadata KV count")?;

        Ok(Self {
            version,
            tensor_count,
            metadata_kv_count,
        })
    }

    /// Returns the GGUF format version.
    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    /// Returns the number of tensors in the file.
    #[must_use]
    pub const fn tensor_count(&self) -> u64 {
        self.tensor_count
    }

    /// Returns the number of metadata key-value pairs.
    #[must_use]
    pub const fn metadata_kv_count(&self) -> u64 {
        self.metadata_kv_count
    }
}

/// Reads a little-endian u32 from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if not enough bytes available.
pub fn read_u32<R: Read>(reader: &mut R, context: &'static str) -> Result<u32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(u32::from_le_bytes(buf))
}

/// Reads a little-endian u64 from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if not enough bytes available.
pub fn read_u64<R: Read>(reader: &mut R, context: &'static str) -> Result<u64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(u64::from_le_bytes(buf))
}

/// Reads a little-endian i32 from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if not enough bytes available.
pub fn read_i32<R: Read>(reader: &mut R, context: &'static str) -> Result<i32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(i32::from_le_bytes(buf))
}

/// Reads a little-endian i64 from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if not enough bytes available.
pub fn read_i64<R: Read>(reader: &mut R, context: &'static str) -> Result<i64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(i64::from_le_bytes(buf))
}

/// Reads a little-endian f32 from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if not enough bytes available.
pub fn read_f32<R: Read>(reader: &mut R, context: &'static str) -> Result<f32> {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(f32::from_le_bytes(buf))
}

/// Reads a little-endian f64 from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if not enough bytes available.
pub fn read_f64<R: Read>(reader: &mut R, context: &'static str) -> Result<f64> {
    let mut buf = [0u8; 8];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(f64::from_le_bytes(buf))
}

/// Reads a single byte from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if no byte available.
pub fn read_u8<R: Read>(reader: &mut R, context: &'static str) -> Result<u8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(buf[0])
}

/// Reads a signed byte from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if no byte available.
pub fn read_i8<R: Read>(reader: &mut R, context: &'static str) -> Result<i8> {
    let mut buf = [0u8; 1];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(buf[0] as i8)
}

/// Reads a little-endian u16 from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if not enough bytes available.
pub fn read_u16<R: Read>(reader: &mut R, context: &'static str) -> Result<u16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(u16::from_le_bytes(buf))
}

/// Reads a little-endian i16 from a reader.
///
/// # Errors
///
/// Returns `GgufError::UnexpectedEof` if not enough bytes available.
pub fn read_i16<R: Read>(reader: &mut R, context: &'static str) -> Result<i16> {
    let mut buf = [0u8; 2];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;
    Ok(i16::from_le_bytes(buf))
}

/// Reads a length-prefixed UTF-8 string from a reader.
///
/// GGUF strings are stored as:
/// - Length: u64 (little-endian)
/// - Bytes: [u8; length]
///
/// # Errors
///
/// Returns an error if:
/// - Not enough bytes available
/// - Bytes are not valid UTF-8
pub fn read_string<R: Read>(reader: &mut R, context: &'static str) -> Result<String> {
    let len = read_u64(reader, context)?;

    // Sanity check to prevent OOM on malformed files
    if len > 1024 * 1024 {
        return Err(GgufError::Overflow { context });
    }

    #[allow(clippy::cast_possible_truncation)]
    let len = len as usize;
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).map_err(|e| {
        if e.kind() == std::io::ErrorKind::UnexpectedEof {
            GgufError::UnexpectedEof { context }
        } else {
            GgufError::from(e)
        }
    })?;

    String::from_utf8(buf).map_err(|_| GgufError::InvalidUtf8 { context })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_valid_header_v3() {
        // Construct a valid GGUF v3 header
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);           // Magic
        data.extend_from_slice(&3u32.to_le_bytes());   // Version 3
        data.extend_from_slice(&10u64.to_le_bytes());  // 10 tensors
        data.extend_from_slice(&5u64.to_le_bytes());   // 5 metadata entries

        let mut cursor = Cursor::new(data);
        let header = GgufHeader::read(&mut cursor).unwrap();

        assert_eq!(header.version(), 3);
        assert_eq!(header.tensor_count(), 10);
        assert_eq!(header.metadata_kv_count(), 5);
    }

    #[test]
    fn test_valid_header_v2() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&2u32.to_le_bytes());   // Version 2
        data.extend_from_slice(&100u64.to_le_bytes());
        data.extend_from_slice(&20u64.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let header = GgufHeader::read(&mut cursor).unwrap();

        assert_eq!(header.version(), 2);
        assert_eq!(header.tensor_count(), 100);
        assert_eq!(header.metadata_kv_count(), 20);
    }

    #[test]
    fn test_invalid_magic() {
        let mut data = Vec::new();
        data.extend_from_slice(b"GGML");  // Wrong magic
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let result = GgufHeader::read(&mut cursor);

        assert!(matches!(result, Err(GgufError::InvalidMagic { .. })));
    }

    #[test]
    fn test_unsupported_version() {
        let mut data = Vec::new();
        data.extend_from_slice(&GGUF_MAGIC);
        data.extend_from_slice(&1u32.to_le_bytes());  // Version 1 (unsupported)
        data.extend_from_slice(&0u64.to_le_bytes());
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let result = GgufHeader::read(&mut cursor);

        assert!(matches!(
            result,
            Err(GgufError::UnsupportedVersion { version: 1, .. })
        ));
    }

    #[test]
    fn test_truncated_header() {
        // Only magic bytes, missing rest
        let data = GGUF_MAGIC.to_vec();
        let mut cursor = Cursor::new(data);
        let result = GgufHeader::read(&mut cursor);

        assert!(matches!(result, Err(GgufError::UnexpectedEof { .. })));
    }

    #[test]
    fn test_empty_input() {
        let data: Vec<u8> = vec![];
        let mut cursor = Cursor::new(data);
        let result = GgufHeader::read(&mut cursor);

        assert!(matches!(result, Err(GgufError::UnexpectedEof { .. })));
    }

    #[test]
    fn test_read_primitives() {
        // Test u32
        let data = 0x1234_5678u32.to_le_bytes();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_u32(&mut cursor, "test").unwrap(), 0x1234_5678);

        // Test u64
        let data = 0x1234_5678_9ABC_DEF0u64.to_le_bytes();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_u64(&mut cursor, "test").unwrap(), 0x1234_5678_9ABC_DEF0);

        // Test i32
        let data = (-12345i32).to_le_bytes();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_i32(&mut cursor, "test").unwrap(), -12345);

        // Test f32
        let data = 3.14159f32.to_le_bytes();
        let mut cursor = Cursor::new(data);
        let val = read_f32(&mut cursor, "test").unwrap();
        assert!((val - 3.14159).abs() < 1e-5);
    }

    #[test]
    fn test_read_string() {
        let mut data = Vec::new();
        let s = "Hello, GGUF!";
        data.extend_from_slice(&(s.len() as u64).to_le_bytes());
        data.extend_from_slice(s.as_bytes());

        let mut cursor = Cursor::new(data);
        let result = read_string(&mut cursor, "test string").unwrap();

        assert_eq!(result, "Hello, GGUF!");
    }

    #[test]
    fn test_read_empty_string() {
        let mut data = Vec::new();
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let result = read_string(&mut cursor, "empty string").unwrap();

        assert_eq!(result, "");
    }

    #[test]
    fn test_read_string_invalid_utf8() {
        let mut data = Vec::new();
        data.extend_from_slice(&3u64.to_le_bytes());
        data.extend_from_slice(&[0xFF, 0xFE, 0xFD]);  // Invalid UTF-8

        let mut cursor = Cursor::new(data);
        let result = read_string(&mut cursor, "test");

        assert!(matches!(result, Err(GgufError::InvalidUtf8 { .. })));
    }

    #[test]
    fn test_header_size_constant() {
        assert_eq!(GgufHeader::SIZE, 24);
    }
}