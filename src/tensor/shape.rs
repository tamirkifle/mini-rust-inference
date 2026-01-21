//! N-dimensional shape representation.
//!
//! Shapes define the dimensions of a tensor. For example:
//! - Scalar: `[]` (0 dimensions)
//! - Vector: `[n]` (1 dimension)
//! - Matrix: `[rows, cols]` (2 dimensions)
//! - 3D tensor: `[batch, rows, cols]` (3 dimensions)

use super::error::{Result, TensorError};

/// N-dimensional shape.
///
/// Stores dimensions in row-major order (outermost dimension first).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    /// Creates a new shape from dimensions.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_engine::tensor::Shape;
    ///
    /// let matrix = Shape::new(vec![3, 4]); // 3x4 matrix
    /// assert_eq!(matrix.ndim(), 2);
    /// assert_eq!(matrix.numel(), 12);
    /// ```
    #[must_use]
    pub const fn new(dims: Vec<usize>) -> Self {
        Self { dims }
    }

    /// Creates a scalar shape (0 dimensions).
    #[must_use]
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }

    /// Creates a 1D shape (vector).
    #[must_use]
    pub fn vector(len: usize) -> Self {
        Self { dims: vec![len] }
    }

    /// Creates a 2D shape (matrix).
    #[must_use]
    pub fn matrix(rows: usize, cols: usize) -> Self {
        Self {
            dims: vec![rows, cols],
        }
    }

    /// Returns the number of dimensions (rank).
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.dims.iter().product::<usize>().max(1)
    }

    /// Returns the dimensions as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Returns the size of a specific dimension.
    ///
    /// Supports negative indexing: -1 is the last dimension.
    #[must_use]
    pub fn dim(&self, index: isize) -> Option<usize> {
        let idx = if index < 0 {
            let neg_index = index.unsigned_abs();
            self.dims.len().checked_sub(neg_index)?
        } else {
            index.unsigned_abs()
        };
        self.dims.get(idx).copied()
    }

    /// Checks if this shape is compatible with another for element-wise operations.
    ///
    /// Shapes are compatible if they are equal or can be broadcast together.
    #[must_use]
    pub fn is_compatible(&self, other: &Self) -> bool {
        if self.dims == other.dims {
            return true;
        }
        // Basic broadcasting: shapes must match from the right
        let self_iter = self.dims.iter().rev();
        let other_iter = other.dims.iter().rev();

        for (a, b) in self_iter.zip(other_iter) {
            if *a != *b && *a != 1 && *b != 1 {
                return false;
            }
        }
        true
    }

    /// Validates that indices are within bounds.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::IndexOutOfBounds` if any index exceeds its dimension.
    pub fn validate_indices(&self, indices: &[usize]) -> Result<()> {
        if indices.len() != self.dims.len() {
            return Err(TensorError::IndexOutOfBounds {
                index: indices.to_vec(),
                shape: self.dims.clone(),
            });
        }
        for (idx, &dim) in indices.iter().zip(self.dims.iter()) {
            if *idx >= dim {
                return Err(TensorError::IndexOutOfBounds {
                    index: indices.to_vec(),
                    shape: self.dims.clone(),
                });
            }
        }
        Ok(())
    }

    /// Validates that this shape can be reshaped to the target shape.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ReshapeError` if element counts don't match.
    pub fn validate_reshape(&self, target: &Self) -> Result<()> {
        if self.numel() != target.numel() {
            return Err(TensorError::ReshapeError {
                from: self.dims.clone(),
                to: target.dims.clone(),
            });
        }
        Ok(())
    }
}

impl From<Vec<usize>> for Shape {
    fn from(dims: Vec<usize>) -> Self {
        Self::new(dims)
    }
}

impl From<&[usize]> for Shape {
    fn from(dims: &[usize]) -> Self {
        Self::new(dims.to_vec())
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{dim}")?;
        }
        write!(f, "]")
    }
}

// Convenience macros for creating shapes
#[macro_export]
macro_rules! shape {
    () => { $crate::tensor::Shape::scalar() };
    ($($dim:expr),+ $(,)?) => {
        $crate::tensor::Shape::new(vec![$($dim),+])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_scalar_shape() {
        let shape = Shape::scalar();
        assert_eq!(shape.ndim(), 0);
        assert_eq!(shape.numel(), 1);
    }

    #[test]
    fn test_dim_access() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(shape.dim(0), Some(2));
        assert_eq!(shape.dim(1), Some(3));
        assert_eq!(shape.dim(2), Some(4));
        assert_eq!(shape.dim(-1), Some(4));
        assert_eq!(shape.dim(-2), Some(3));
        assert_eq!(shape.dim(3), None);
    }

    #[test]
    fn test_shape_compatibility() {
        let shape_3x4 = Shape::new(vec![3, 4]);
        let shape_3x4_dup = Shape::new(vec![3, 4]);
        let shape_1x4 = Shape::new(vec![1, 4]);
        let shape_3x1 = Shape::new(vec![3, 1]);
        let shape_2x4 = Shape::new(vec![2, 4]);

        assert!(shape_3x4.is_compatible(&shape_3x4_dup));
        assert!(shape_3x4.is_compatible(&shape_1x4)); // Broadcasting
        assert!(shape_3x4.is_compatible(&shape_3x1)); // Broadcasting
        assert!(!shape_3x4.is_compatible(&shape_2x4)); // Incompatible
    }

    #[test]
    fn test_validate_indices() {
        let shape = Shape::new(vec![2, 3]);
        assert!(shape.validate_indices(&[0, 0]).is_ok());
        assert!(shape.validate_indices(&[1, 2]).is_ok());
        assert!(shape.validate_indices(&[2, 0]).is_err());
        assert!(shape.validate_indices(&[0, 3]).is_err());
        assert!(shape.validate_indices(&[0]).is_err());
    }

    #[test]
    fn test_validate_reshape() {
        let shape = Shape::new(vec![2, 6]);
        assert!(shape.validate_reshape(&Shape::new(vec![3, 4])).is_ok());
        assert!(shape.validate_reshape(&Shape::new(vec![12])).is_ok());
        assert!(shape.validate_reshape(&Shape::new(vec![2, 3, 2])).is_ok());
        assert!(shape.validate_reshape(&Shape::new(vec![5, 3])).is_err());
    }

    #[test]
    fn test_shape_display() {
        assert_eq!(Shape::scalar().to_string(), "[]");
        assert_eq!(Shape::vector(5).to_string(), "[5]");
        assert_eq!(Shape::matrix(3, 4).to_string(), "[3, 4]");
    }
}