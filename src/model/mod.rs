//! Model architectures.
//!
//! Currently supported:
//! - `llama`: Llama 2 / Llama 3 transformer architecture

// CHANGED: model module activated in commit 8.0
pub mod llama;

use crate::tensor::TensorError;

/// Errors that can occur during model construction or forward passes.
#[derive(Debug, Clone)]
pub enum ModelError {
    /// A required GGUF metadata key was not present.
    MissingMetadataKey { key: &'static str },
    /// An underlying tensor operation failed.
    TensorError(TensorError),
    /// A configuration value is invalid (e.g., n_heads doesn't divide embed_dim).
    InvalidConfig { reason: String },
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingMetadataKey { key } => {
                write!(f, "missing required metadata key: {key}")
            }
            Self::TensorError(e) => write!(f, "tensor error: {e}"),
            Self::InvalidConfig { reason } => write!(f, "invalid model config: {reason}"),
        }
    }
}

impl std::error::Error for ModelError {}

// CHANGED: auto-convert TensorError → ModelError so `?` works in model code
impl From<TensorError> for ModelError {
    fn from(e: TensorError) -> Self {
        Self::TensorError(e)
    }
}

/// Convenience alias — all model functions return this.
pub type Result<T> = std::result::Result<T, ModelError>;
