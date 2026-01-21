//! Error types for tensor operations.

use std::fmt;

/// Errors that can occur during tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Shape mismatch in operation.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Invalid shape specification.
    InvalidShape { reason: String },
    /// Invalid stride specification.
    InvalidStride { reason: String },
    /// Index out of bounds.
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    /// Element count mismatch between shape and data.
    ElementCountMismatch { shape_elements: usize, data_len: usize },
    /// Cannot reshape to target shape.
    ReshapeError { from: Vec<usize>, to: Vec<usize> },
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            }
            Self::InvalidShape { reason } => {
                write!(f, "invalid shape: {reason}")
            }
            Self::InvalidStride { reason } => {
                write!(f, "invalid stride: {reason}")
            }
            Self::IndexOutOfBounds { index, shape } => {
                write!(f, "index {index:?} out of bounds for shape {shape:?}")
            }
            Self::ElementCountMismatch {
                shape_elements,
                data_len,
            } => {
                write!(
                    f,
                    "element count mismatch: shape requires {shape_elements} elements, got {data_len}"
                )
            }
            Self::ReshapeError { from, to } => {
                write!(f, "cannot reshape from {from:?} to {to:?}")
            }
        }
    }
}

impl std::error::Error for TensorError {}

/// Result type for tensor operations.
pub type Result<T> = std::result::Result<T, TensorError>;
