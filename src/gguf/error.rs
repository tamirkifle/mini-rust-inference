//! Error types for GGUF parsing operations.

use std::fmt;

/// Errors that can occur during GGUF parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufError {
    /// Invalid magic bytes (expected "GGUF").
    InvalidMagic { got: [u8; 4] },

    /// Unsupported GGUF version.
    UnsupportedVersion { version: u32, supported: &'static [u32] },

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