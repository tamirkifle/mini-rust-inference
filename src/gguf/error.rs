//! Error types for GGUF parsing and extraction operations.

use std::fmt;

/// Errors that can occur during GGUF parsing and tensor extraction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    /// Invalid magic bytes (expected "GGUF").
    InvalidMagic { got: [u8; 4] },

    /// Unsupported GGUF version.
    UnsupportedVersion {
        version: u32,
        supported: &'static [u32],
    },

    /// Unexpected end of file during parsing.
    UnexpectedEof { context: &'static str },

    /// Invalid UTF-8 in string data.
    InvalidUtf8 { context: &'static str },

    /// Unknown metadata value type.
    UnknownValueType { type_id: u32 },

    /// Invalid boolean value (expected 0 or 1).
    InvalidBool { value: u8 },

    /// Array nesting not allowed (arrays of arrays).
    NestedArray,

    /// I/O error during reading.
    Io { message: String },

    /// Integer overflow during size calculation.
    Overflow { context: &'static str },

    /// Duplicate metadata key.
    DuplicateKey { key: String },

    /// Value is out of valid range.
    ValueOutOfRange { field: &'static str, value: u64 },

    /// Key not found in metadata or tensor collection.
    KeyNotFound { key: String },

    /// Data type mismatch during extraction.
    TypeMismatch {
        expected: String,
        got: String,
    },

    /// Alignment error during tensor extraction.
    AlignmentError {
        expected: usize,
        actual: usize,
    },

    /// Tensor data not available or out of bounds.
    TensorDataUnavailable { name: String },

    /// Shape mismatch during tensor creation.
    ShapeMismatch { expected: usize, got: usize },
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic { got } => {
                write!(
                    f,
                    "invalid GGUF magic: expected [0x47, 0x47, 0x55, 0x46] ('GGUF'), got {got:?}"
                )
            }
            Self::UnsupportedVersion { version, supported } => {
                write!(
                    f,
                    "unsupported GGUF version {version}, supported versions: {supported:?}"
                )
            }
            Self::UnexpectedEof { context } => {
                write!(f, "unexpected end of file while reading {context}")
            }
            Self::InvalidUtf8 { context } => {
                write!(f, "invalid UTF-8 in {context}")
            }
            Self::UnknownValueType { type_id } => {
                write!(f, "unknown metadata value type: {type_id}")
            }
            Self::InvalidBool { value } => {
                write!(f, "invalid boolean value: {value} (expected 0 or 1)")
            }
            Self::NestedArray => {
                write!(f, "nested arrays are not supported in GGUF")
            }
            Self::Io { message } => {
                write!(f, "I/O error: {message}")
            }
            Self::Overflow { context } => {
                write!(f, "integer overflow while calculating {context}")
            }
            Self::DuplicateKey { key } => {
                write!(f, "duplicate metadata key: {key}")
            }
            Self::ValueOutOfRange { field, value } => {
                write!(f, "value out of range for {field}: {value}")
            }
            Self::KeyNotFound { key } => {
                write!(f, "key not found: {key}")
            }
            Self::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            Self::AlignmentError { expected, actual } => {
                write!(
                    f,
                    "alignment error: expected {expected}-byte alignment, got {actual}"
                )
            }
            Self::TensorDataUnavailable { name } => {
                write!(f, "tensor data unavailable: {name}")
            }
            Self::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "shape mismatch: expected {expected} elements, got {got}"
                )
            }
        }
    }
}

impl std::error::Error for GgufError {}

impl From<std::io::Error> for GgufError {
    fn from(err: std::io::Error) -> Self {
        Self::Io {
            message: err.to_string(),
        }
    }
}

/// Result type for GGUF operations.
pub type Result<T> = std::result::Result<T, GgufError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = GgufError::InvalidMagic {
            got: [0x47, 0x47, 0x4D, 0x4C],
        };
        assert!(err.to_string().contains("invalid GGUF magic"));

        let err = GgufError::TypeMismatch {
            expected: "F32".to_string(),
            got: "Q4_0".to_string(),
        };
        assert!(err.to_string().contains("type mismatch"));

        let err = GgufError::TensorDataUnavailable {
            name: "test.weight".to_string(),
        };
        assert!(err.to_string().contains("tensor data unavailable"));
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let gguf_err: GgufError = io_err.into();
        assert!(matches!(gguf_err, GgufError::Io { .. }));
    }
}