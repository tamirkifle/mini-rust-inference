//! GGUF metadata key-value parsing.
//!
//! Metadata in GGUF files consists of key-value pairs where:
//! - Key: length-prefixed UTF-8 string
//! - Value type: u32 enum
//! - Value: type-dependent encoding
//!
//! Supported value types:
//! - Integers: u8, i8, u16, i16, u32, i32, u64, i64
//! - Floats: f32, f64
//! - Bool: single byte (0 or 1)
//! - String: length-prefixed UTF-8
//! - Array: homogeneous typed arrays (no nesting)
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use super::error::{GgufError, Result};
use super::header::{
    read_f32, read_f64, read_i16, read_i32, read_i64, read_i8, read_string, read_u16, read_u32,
    read_u64, read_u8,
};
use std::collections::HashMap;
use std::io::Read;

/// GGUF metadata value type identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufValueType {
    /// Unsigned 8-bit integer.
    Uint8 = 0,
    /// Signed 8-bit integer.
    Int8 = 1,
    /// Unsigned 16-bit integer.
    Uint16 = 2,
    /// Signed 16-bit integer.
    Int16 = 3,
    /// Unsigned 32-bit integer.
    Uint32 = 4,
    /// Signed 32-bit integer.
    Int32 = 5,
    /// 32-bit floating point.
    Float32 = 6,
    /// Boolean (0 or 1).
    Bool = 7,
    /// Length-prefixed UTF-8 string.
    String = 8,
    /// Homogeneous typed array.
    Array = 9,
    /// Unsigned 64-bit integer.
    Uint64 = 10,
    /// Signed 64-bit integer.
    Int64 = 11,
    /// 64-bit floating point.
    Float64 = 12,
}

impl GgufValueType {
    /// Converts a u32 to a value type.
    ///
    /// # Errors
    ///
    /// Returns `GgufError::UnknownValueType` if the type ID is not recognized.
    pub fn from_u32(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Uint8),
            1 => Ok(Self::Int8),
            2 => Ok(Self::Uint16),
            3 => Ok(Self::Int16),
            4 => Ok(Self::Uint32),
            5 => Ok(Self::Int32),
            6 => Ok(Self::Float32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::Uint64),
            11 => Ok(Self::Int64),
            12 => Ok(Self::Float64),
            _ => Err(GgufError::UnknownValueType { type_id: value }),
        }
    }
}

/// A metadata value in a GGUF file.
///
/// Supports all GGUF value types including scalar values, strings, and arrays.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// Unsigned 8-bit integer.
    Uint8(u8),
    /// Signed 8-bit integer.
    Int8(i8),
    /// Unsigned 16-bit integer.
    Uint16(u16),
    /// Signed 16-bit integer.
    Int16(i16),
    /// Unsigned 32-bit integer.
    Uint32(u32),
    /// Signed 32-bit integer.
    Int32(i32),
    /// 32-bit floating point.
    Float32(f32),
    /// Boolean value.
    Bool(bool),
    /// UTF-8 string.
    String(String),
    /// Unsigned 64-bit integer.
    Uint64(u64),
    /// Signed 64-bit integer.
    Int64(i64),
    /// 64-bit floating point.
    Float64(f64),
    /// Array of unsigned 8-bit integers.
    Uint8Array(Vec<u8>),
    /// Array of signed 8-bit integers.
    Int8Array(Vec<i8>),
    /// Array of unsigned 16-bit integers.
    Uint16Array(Vec<u16>),
    /// Array of signed 16-bit integers.
    Int16Array(Vec<i16>),
    /// Array of unsigned 32-bit integers.
    Uint32Array(Vec<u32>),
    /// Array of signed 32-bit integers.
    Int32Array(Vec<i32>),
    /// Array of 32-bit floats.
    Float32Array(Vec<f32>),
    /// Array of booleans.
    BoolArray(Vec<bool>),
    /// Array of strings.
    StringArray(Vec<String>),
    /// Array of unsigned 64-bit integers.
    Uint64Array(Vec<u64>),
    /// Array of signed 64-bit integers.
    Int64Array(Vec<i64>),
    /// Array of 64-bit floats.
    Float64Array(Vec<f64>),
}

impl MetadataValue {
    /// Reads a metadata value from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Unknown value type encountered
    /// - I/O error during reading
    /// - Invalid boolean value
    /// - Nested arrays encountered
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        let type_id = read_u32(reader, "value type")?;
        let value_type = GgufValueType::from_u32(type_id)?;
        Self::read_typed(reader, value_type, false)
    }

    /// Reads a metadata value of a known type.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or nested arrays are encountered.
    fn read_typed<R: Read>(reader: &mut R, value_type: GgufValueType, in_array: bool) -> Result<Self> {
        match value_type {
            GgufValueType::Uint8 => Ok(Self::Uint8(read_u8(reader, "uint8 value")?)),
            GgufValueType::Int8 => Ok(Self::Int8(read_i8(reader, "int8 value")?)),
            GgufValueType::Uint16 => Ok(Self::Uint16(read_u16(reader, "uint16 value")?)),
            GgufValueType::Int16 => Ok(Self::Int16(read_i16(reader, "int16 value")?)),
            GgufValueType::Uint32 => Ok(Self::Uint32(read_u32(reader, "uint32 value")?)),
            GgufValueType::Int32 => Ok(Self::Int32(read_i32(reader, "int32 value")?)),
            GgufValueType::Float32 => Ok(Self::Float32(read_f32(reader, "float32 value")?)),
            GgufValueType::Bool => {
                let byte = read_u8(reader, "bool value")?;
                match byte {
                    0 => Ok(Self::Bool(false)),
                    1 => Ok(Self::Bool(true)),
                    _ => Err(GgufError::InvalidBool { value: byte }),
                }
            }
            GgufValueType::String => Ok(Self::String(read_string(reader, "string value")?)),
            GgufValueType::Uint64 => Ok(Self::Uint64(read_u64(reader, "uint64 value")?)),
            GgufValueType::Int64 => Ok(Self::Int64(read_i64(reader, "int64 value")?)),
            GgufValueType::Float64 => Ok(Self::Float64(read_f64(reader, "float64 value")?)),
            GgufValueType::Array => {
                if in_array {
                    return Err(GgufError::NestedArray);
                }
                Self::read_array(reader)
            }
        }
    }

    /// Reads an array value from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or nested arrays are encountered.
    #[allow(clippy::cast_possible_truncation)]
    fn read_array<R: Read>(reader: &mut R) -> Result<Self> {
        let element_type_id = read_u32(reader, "array element type")?;
        let element_type = GgufValueType::from_u32(element_type_id)?;
        let len = read_u64(reader, "array length")?;

        // Sanity check to prevent OOM
        if len > 100_000_000 {
            return Err(GgufError::Overflow {
                context: "array length",
            });
        }
        let len = len as usize;

        match element_type {
            GgufValueType::Uint8 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_u8(reader, "array element")?);
                }
                Ok(Self::Uint8Array(arr))
            }
            GgufValueType::Int8 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_i8(reader, "array element")?);
                }
                Ok(Self::Int8Array(arr))
            }
            GgufValueType::Uint16 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_u16(reader, "array element")?);
                }
                Ok(Self::Uint16Array(arr))
            }
            GgufValueType::Int16 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_i16(reader, "array element")?);
                }
                Ok(Self::Int16Array(arr))
            }
            GgufValueType::Uint32 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_u32(reader, "array element")?);
                }
                Ok(Self::Uint32Array(arr))
            }
            GgufValueType::Int32 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_i32(reader, "array element")?);
                }
                Ok(Self::Int32Array(arr))
            }
            GgufValueType::Float32 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_f32(reader, "array element")?);
                }
                Ok(Self::Float32Array(arr))
            }
            GgufValueType::Bool => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    let byte = read_u8(reader, "array element")?;
                    match byte {
                        0 => arr.push(false),
                        1 => arr.push(true),
                        _ => return Err(GgufError::InvalidBool { value: byte }),
                    }
                }
                Ok(Self::BoolArray(arr))
            }
            GgufValueType::String => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_string(reader, "array element")?);
                }
                Ok(Self::StringArray(arr))
            }
            GgufValueType::Uint64 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_u64(reader, "array element")?);
                }
                Ok(Self::Uint64Array(arr))
            }
            GgufValueType::Int64 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_i64(reader, "array element")?);
                }
                Ok(Self::Int64Array(arr))
            }
            GgufValueType::Float64 => {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(read_f64(reader, "array element")?);
                }
                Ok(Self::Float64Array(arr))
            }
            GgufValueType::Array => Err(GgufError::NestedArray),
        }
    }

    /// Returns the value as a u32, if applicable.
    #[must_use]
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            Self::Uint8(v) => Some(u32::from(*v)),
            Self::Uint16(v) => Some(u32::from(*v)),
            Self::Uint32(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as a u64, if applicable.
    #[must_use]
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(u64::from(*v)),
            Self::Uint16(v) => Some(u64::from(*v)),
            Self::Uint32(v) => Some(u64::from(*v)),
            Self::Uint64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as an i64, if applicable.
    #[must_use]
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int8(v) => Some(i64::from(*v)),
            Self::Int16(v) => Some(i64::from(*v)),
            Self::Int32(v) => Some(i64::from(*v)),
            Self::Int64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as an f32, if applicable.
    #[must_use]
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as an f64, if applicable.
    #[must_use]
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float32(v) => Some(f64::from(*v)),
            Self::Float64(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as a bool, if applicable.
    #[must_use]
    pub const fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the value as a string slice, if applicable.
    #[must_use]
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(v) => Some(v.as_str()),
            _ => None,
        }
    }

    /// Returns the value as a u32 array slice, if applicable.
    #[must_use]
    pub fn as_u32_array(&self) -> Option<&[u32]> {
        match self {
            Self::Uint32Array(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Returns the value as a string array slice, if applicable.
    #[must_use]
    pub fn as_string_array(&self) -> Option<&[String]> {
        match self {
            Self::StringArray(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// A collection of metadata key-value pairs.
///
/// Provides convenient access to model configuration values.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    /// Key-value pairs indexed by key name.
    entries: HashMap<String, MetadataValue>,
}

impl Metadata {
    /// Creates a new empty metadata collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Reads metadata entries from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - I/O error during reading
    /// - Invalid value types encountered
    /// - Duplicate keys found
    #[allow(clippy::cast_possible_truncation)]
    pub fn read<R: Read>(reader: &mut R, count: u64) -> Result<Self> {
        // Sanity check
        if count > 100_000 {
            return Err(GgufError::Overflow {
                context: "metadata count",
            });
        }

        let mut entries = HashMap::with_capacity(count as usize);

        for _ in 0..count {
            let key = read_string(reader, "metadata key")?;
            let value = MetadataValue::read(reader)?;

            if entries.contains_key(&key) {
                return Err(GgufError::DuplicateKey { key });
            }

            entries.insert(key, value);
        }

        Ok(Self { entries })
    }

    /// Returns the number of metadata entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if there are no metadata entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Gets a metadata value by key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Gets a u32 value by key.
    #[must_use]
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.get(key).and_then(MetadataValue::as_u32)
    }

    /// Gets a u64 value by key.
    #[must_use]
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(MetadataValue::as_u64)
    }

    /// Gets an i64 value by key.
    #[must_use]
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(MetadataValue::as_i64)
    }

    /// Gets an f32 value by key.
    #[must_use]
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.get(key).and_then(MetadataValue::as_f32)
    }

    /// Gets an f64 value by key.
    #[must_use]
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(MetadataValue::as_f64)
    }

    /// Gets a bool value by key.
    #[must_use]
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(MetadataValue::as_bool)
    }

    /// Gets a string value by key.
    #[must_use]
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(MetadataValue::as_str)
    }

    /// Returns an iterator over all keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Returns an iterator over all key-value pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetadataValue)> {
        self.entries.iter()
    }

    /// Inserts a metadata entry.
    pub fn insert(&mut self, key: String, value: MetadataValue) {
        self.entries.insert(key, value);
    }
}

/// Common GGUF metadata keys for Llama models.
pub mod keys {
    /// General architecture name (e.g., "llama").
    pub const GENERAL_ARCHITECTURE: &str = "general.architecture";
    /// Model name.
    pub const GENERAL_NAME: &str = "general.name";
    /// Number of parameters.
    pub const GENERAL_FILE_TYPE: &str = "general.file_type";

    /// Embedding dimension.
    pub const LLAMA_EMBEDDING_LENGTH: &str = "llama.embedding_length";
    /// Number of attention heads.
    pub const LLAMA_ATTENTION_HEAD_COUNT: &str = "llama.attention.head_count";
    /// Number of key-value heads (for GQA).
    pub const LLAMA_ATTENTION_HEAD_COUNT_KV: &str = "llama.attention.head_count_kv";
    /// Number of layers.
    pub const LLAMA_BLOCK_COUNT: &str = "llama.block_count";
    /// Feed-forward hidden dimension.
    pub const LLAMA_FEED_FORWARD_LENGTH: &str = "llama.feed_forward_length";
    /// RoPE dimension count.
    pub const LLAMA_ROPE_DIMENSION_COUNT: &str = "llama.rope.dimension_count";
    /// Context length.
    pub const LLAMA_CONTEXT_LENGTH: &str = "llama.context_length";
    /// Vocabulary size.
    pub const TOKENIZER_GGML_TOKENS: &str = "tokenizer.ggml.tokens";
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn test_value_type_from_u32() {
        assert_eq!(GgufValueType::from_u32(0).unwrap(), GgufValueType::Uint8);
        assert_eq!(GgufValueType::from_u32(6).unwrap(), GgufValueType::Float32);
        assert_eq!(GgufValueType::from_u32(8).unwrap(), GgufValueType::String);
        assert!(GgufValueType::from_u32(99).is_err());
    }

    #[test]
    fn test_read_uint32_value() {
        let mut data = Vec::new();
        data.extend_from_slice(&4u32.to_le_bytes());  // Type: Uint32
        data.extend_from_slice(&42u32.to_le_bytes()); // Value: 42

        let mut cursor = Cursor::new(data);
        let value = MetadataValue::read(&mut cursor).unwrap();

        assert_eq!(value, MetadataValue::Uint32(42));
        assert_eq!(value.as_u32(), Some(42));
    }

    #[test]
    fn test_read_string_value() {
        let mut data = Vec::new();
        data.extend_from_slice(&8u32.to_le_bytes());  // Type: String
        write_string(&mut data, "llama");

        let mut cursor = Cursor::new(data);
        let value = MetadataValue::read(&mut cursor).unwrap();

        assert_eq!(value.as_str(), Some("llama"));
    }

    #[test]
    fn test_read_bool_values() {
        // Test true
        let mut data = Vec::new();
        data.extend_from_slice(&7u32.to_le_bytes());  // Type: Bool
        data.push(1);  // true

        let mut cursor = Cursor::new(data);
        let value = MetadataValue::read(&mut cursor).unwrap();
        assert_eq!(value.as_bool(), Some(true));

        // Test false
        let mut data = Vec::new();
        data.extend_from_slice(&7u32.to_le_bytes());
        data.push(0);  // false

        let mut cursor = Cursor::new(data);
        let value = MetadataValue::read(&mut cursor).unwrap();
        assert_eq!(value.as_bool(), Some(false));

        // Test invalid
        let mut data = Vec::new();
        data.extend_from_slice(&7u32.to_le_bytes());
        data.push(2);  // invalid

        let mut cursor = Cursor::new(data);
        let result = MetadataValue::read(&mut cursor);
        assert!(matches!(result, Err(GgufError::InvalidBool { value: 2 })));
    }

    #[test]
    fn test_read_float32_value() {
        let mut data = Vec::new();
        data.extend_from_slice(&6u32.to_le_bytes());       // Type: Float32
        data.extend_from_slice(&3.14f32.to_le_bytes());    // Value

        let mut cursor = Cursor::new(data);
        let value = MetadataValue::read(&mut cursor).unwrap();

        let f = value.as_f32().unwrap();
        assert!((f - 3.14).abs() < 1e-5);
    }

    #[test]
    fn test_read_u32_array() {
        let mut data = Vec::new();
        data.extend_from_slice(&9u32.to_le_bytes());   // Type: Array
        data.extend_from_slice(&4u32.to_le_bytes());   // Element type: Uint32
        data.extend_from_slice(&3u64.to_le_bytes());   // Length: 3
        data.extend_from_slice(&10u32.to_le_bytes());  // Elements
        data.extend_from_slice(&20u32.to_le_bytes());
        data.extend_from_slice(&30u32.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let value = MetadataValue::read(&mut cursor).unwrap();

        assert_eq!(value.as_u32_array(), Some(&[10u32, 20, 30][..]));
    }

    #[test]
    fn test_read_string_array() {
        let mut data = Vec::new();
        data.extend_from_slice(&9u32.to_le_bytes());   // Type: Array
        data.extend_from_slice(&8u32.to_le_bytes());   // Element type: String
        data.extend_from_slice(&2u64.to_le_bytes());   // Length: 2
        write_string(&mut data, "hello");
        write_string(&mut data, "world");

        let mut cursor = Cursor::new(data);
        let value = MetadataValue::read(&mut cursor).unwrap();

        let arr = value.as_string_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0], "hello");
        assert_eq!(arr[1], "world");
    }

    #[test]
    fn test_nested_array_rejected() {
        let mut data = Vec::new();
        data.extend_from_slice(&9u32.to_le_bytes());   // Type: Array
        data.extend_from_slice(&9u32.to_le_bytes());   // Element type: Array (nested!)
        data.extend_from_slice(&1u64.to_le_bytes());   // Length: 1

        let mut cursor = Cursor::new(data);
        let result = MetadataValue::read(&mut cursor);

        assert!(matches!(result, Err(GgufError::NestedArray)));
    }

    #[test]
    fn test_metadata_read() {
        let mut data = Vec::new();

        // Entry 1: "general.architecture" = "llama" (string)
        write_string(&mut data, "general.architecture");
        data.extend_from_slice(&8u32.to_le_bytes());  // Type: String
        write_string(&mut data, "llama");

        // Entry 2: "llama.block_count" = 32 (u32)
        write_string(&mut data, "llama.block_count");
        data.extend_from_slice(&4u32.to_le_bytes());  // Type: Uint32
        data.extend_from_slice(&32u32.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let metadata = Metadata::read(&mut cursor, 2).unwrap();

        assert_eq!(metadata.len(), 2);
        assert_eq!(metadata.get_str("general.architecture"), Some("llama"));
        assert_eq!(metadata.get_u32("llama.block_count"), Some(32));
    }

    #[test]
    fn test_metadata_duplicate_key() {
        let mut data = Vec::new();

        // Entry 1
        write_string(&mut data, "key");
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());

        // Entry 2 with same key
        write_string(&mut data, "key");
        data.extend_from_slice(&4u32.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let result = Metadata::read(&mut cursor, 2);

        assert!(matches!(result, Err(GgufError::DuplicateKey { .. })));
    }

    #[test]
    fn test_metadata_empty() {
        let data: Vec<u8> = vec![];
        let mut cursor = Cursor::new(data);
        let metadata = Metadata::read(&mut cursor, 0).unwrap();

        assert!(metadata.is_empty());
        assert_eq!(metadata.len(), 0);
    }

    #[test]
    fn test_metadata_iteration() {
        let mut metadata = Metadata::new();
        metadata.insert("key1".to_string(), MetadataValue::Uint32(1));
        metadata.insert("key2".to_string(), MetadataValue::Uint32(2));

        let keys: Vec<_> = metadata.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&&"key1".to_string()));
        assert!(keys.contains(&&"key2".to_string()));
    }
}