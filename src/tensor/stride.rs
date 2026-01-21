//! Stride calculations for tensor memory layout.
//!
//! Strides define how many elements to skip in memory to advance one position
//! in each dimension. For a row-major (C-style) contiguous tensor:
//!
//! ```text
//! Shape:  [2, 3, 4]
//! Stride: [12, 4, 1]
//!
//! To access element [i, j, k]:
//!   offset = i * 12 + j * 4 + k * 1
//! ```

use super::error::{Result, TensorError};
use super::shape::Shape;

/// Stride values for each dimension.
///
/// Stride[i] indicates how many elements to skip in linear memory
/// to advance one position in dimension i.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Stride {
    values: Vec<usize>,
}

impl Stride {
    /// Creates a new stride from values.
    #[must_use]
    pub const fn new(values: Vec<usize>) -> Self {
        Self { values }
    }

    /// Computes contiguous row-major (C-style) strides for a shape.
    ///
    /// For shape `[d0, d1, d2, ...]`, strides are:
    /// `[d1*d2*..., d2*d3*..., ..., 1]`
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_engine::tensor::{Shape, Stride};
    ///
    /// let shape = Shape::new(vec![2, 3, 4]);
    /// let stride = Stride::contiguous(&shape);
    /// assert_eq!(stride.values(), &[12, 4, 1]);
    /// ```
    #[must_use]
    pub fn contiguous(shape: &Shape) -> Self {
        let dims = shape.dims();
        if dims.is_empty() {
            return Self { values: vec![] };
        }

        let mut strides = vec![1usize; dims.len()];

        // Compute strides from right to left
        for i in (0..dims.len() - 1).rev() {
            strides[i] = strides[i + 1] * dims[i + 1];
        }

        Self { values: strides }
    }

    /// Computes column-major (Fortran-style) strides for a shape.
    ///
    /// For shape `[d0, d1, d2, ...]`, strides are:
    /// `[1, d0, d0*d1, ...]`
    #[must_use]
    pub fn column_major(shape: &Shape) -> Self {
        let dims = shape.dims();
        if dims.is_empty() {
            return Self { values: vec![] };
        }

        let mut strides = vec![1usize; dims.len()];

        // Compute strides from left to right
        for i in 1..dims.len() {
            strides[i] = strides[i - 1] * dims[i - 1];
        }

        Self { values: strides }
    }

    /// Returns the stride values as a slice.
    #[must_use]
    pub fn values(&self) -> &[usize] {
        &self.values
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.values.len()
    }

    /// Computes the linear offset for a given set of indices.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_engine::tensor::{Shape, Stride};
    ///
    /// let shape = Shape::new(vec![2, 3]);
    /// let stride = Stride::contiguous(&shape);
    /// assert_eq!(stride.offset(&[0, 0]), 0);
    /// assert_eq!(stride.offset(&[0, 1]), 1);
    /// assert_eq!(stride.offset(&[1, 0]), 3);
    /// assert_eq!(stride.offset(&[1, 2]), 5);
    /// ```
    #[must_use]
    pub fn offset(&self, indices: &[usize]) -> usize {
        indices
            .iter()
            .zip(self.values.iter())
            .map(|(idx, stride)| idx * stride)
            .sum()
    }

    /// Checks if this stride represents a contiguous row-major layout.
    #[must_use]
    pub fn is_contiguous(&self, shape: &Shape) -> bool {
        *self == Self::contiguous(shape)
    }

    /// Computes new strides after transposing two dimensions.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidStride` if dimensions are out of bounds.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.values.len() || dim1 >= self.values.len() {
            return Err(TensorError::InvalidStride {
                reason: format!(
                    "transpose dimensions ({dim0}, {dim1}) out of bounds for {} dims",
                    self.values.len()
                ),
            });
        }

        let mut new_values = self.values.clone();
        new_values.swap(dim0, dim1);
        Ok(Self { values: new_values })
    }

    /// Computes new strides after permuting dimensions.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidStride` if permutation is invalid.
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        if dims.len() != self.values.len() {
            return Err(TensorError::InvalidStride {
                reason: format!(
                    "permute requires {} dims, got {}",
                    self.values.len(),
                    dims.len()
                ),
            });
        }

        // Validate permutation
        let mut seen = vec![false; dims.len()];
        for &d in dims {
            if d >= dims.len() {
                return Err(TensorError::InvalidStride {
                    reason: format!("invalid dimension {d} in permutation"),
                });
            }
            if seen[d] {
                return Err(TensorError::InvalidStride {
                    reason: format!("duplicate dimension {d} in permutation"),
                });
            }
            seen[d] = true;
        }

        let new_values: Vec<usize> = dims.iter().map(|&d| self.values[d]).collect();
        Ok(Self { values: new_values })
    }
}

impl From<Vec<usize>> for Stride {
    fn from(values: Vec<usize>) -> Self {
        Self::new(values)
    }
}

impl std::fmt::Display for Stride {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Stride{:?}", self.values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_stride() {
        // 1D
        let shape_1d = Shape::vector(5);
        assert_eq!(Stride::contiguous(&shape_1d).values(), &[1]);

        // 2D
        let shape_2d = Shape::matrix(3, 4);
        assert_eq!(Stride::contiguous(&shape_2d).values(), &[4, 1]);

        // 3D
        let shape_3d = Shape::new(vec![2, 3, 4]);
        assert_eq!(Stride::contiguous(&shape_3d).values(), &[12, 4, 1]);

        // Scalar
        let shape_scalar = Shape::scalar();
        assert_eq!(Stride::contiguous(&shape_scalar).values(), &[] as &[usize]);
    }

    #[test]
    fn test_column_major_stride() {
        let shape = Shape::new(vec![2, 3, 4]);
        assert_eq!(Stride::column_major(&shape).values(), &[1, 2, 6]);
    }

    #[test]
    fn test_offset_calculation() {
        let shape = Shape::new(vec![2, 3, 4]);
        let stride = Stride::contiguous(&shape);

        assert_eq!(stride.offset(&[0, 0, 0]), 0);
        assert_eq!(stride.offset(&[0, 0, 1]), 1);
        assert_eq!(stride.offset(&[0, 1, 0]), 4);
        assert_eq!(stride.offset(&[1, 0, 0]), 12);
        assert_eq!(stride.offset(&[1, 2, 3]), 12 + 8 + 3);
    }

    #[test]
    fn test_is_contiguous() {
        let shape = Shape::matrix(3, 4);
        let contig = Stride::contiguous(&shape);
        let col_major = Stride::column_major(&shape);

        assert!(contig.is_contiguous(&shape));
        assert!(!col_major.is_contiguous(&shape));
    }

    #[test]
    fn test_transpose() {
        let stride = Stride::new(vec![12, 4, 1]);
        let transposed = stride.transpose(0, 2).unwrap();
        assert_eq!(transposed.values(), &[1, 4, 12]);
    }

    #[test]
    fn test_permute() {
        let stride = Stride::new(vec![12, 4, 1]);

        // Reverse dimensions
        let permuted = stride.permute(&[2, 1, 0]).unwrap();
        assert_eq!(permuted.values(), &[1, 4, 12]);

        // Invalid permutation
        assert!(stride.permute(&[0, 0, 1]).is_err()); // Duplicate
        assert!(stride.permute(&[0, 1]).is_err()); // Wrong length
        assert!(stride.permute(&[0, 1, 3]).is_err()); // Out of bounds
    }
}