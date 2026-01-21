//! Tensor layout combining shape and stride.
//!
//! Layout defines the complete memory organization of a tensor:
//! - Shape: the logical dimensions
//! - Stride: how to navigate memory

use super::error::{Result, TensorError};
use super::shape::Shape;
use super::stride::Stride;

/// Memory layout of a tensor.
///
/// Combines shape (logical dimensions) with stride (memory navigation).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    shape: Shape,
    stride: Stride,
}

impl Layout {
    /// Creates a new layout from shape and stride.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidStride` if shape and stride dimensions don't match.
    pub fn new(shape: Shape, stride: Stride) -> Result<Self> {
        if shape.ndim() != stride.ndim() {
            return Err(TensorError::InvalidStride {
                reason: format!(
                    "shape has {} dims but stride has {} dims",
                    shape.ndim(),
                    stride.ndim()
                ),
            });
        }
        Ok(Self { shape, stride })
    }

    /// Creates a contiguous row-major layout for a shape.
    #[must_use]
    pub fn contiguous(shape: Shape) -> Self {
        let stride = Stride::contiguous(&shape);
        Self { shape, stride }
    }

    /// Creates a contiguous layout from dimensions.
    #[must_use]
    pub fn from_dims(dims: Vec<usize>) -> Self {
        Self::contiguous(Shape::new(dims))
    }

    /// Returns the shape.
    #[must_use]
    pub const fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the stride.
    #[must_use]
    pub const fn stride(&self) -> &Stride {
        &self.stride
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Returns the dimensions as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        self.shape.dims()
    }

    /// Checks if the layout is contiguous in memory.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.stride.is_contiguous(&self.shape)
    }

    /// Computes the linear memory offset for indices.
    #[must_use]
    pub fn offset(&self, indices: &[usize]) -> usize {
        self.stride.offset(indices)
    }

    /// Validates indices and returns the offset.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::IndexOutOfBounds` if indices are invalid.
    pub fn checked_offset(&self, indices: &[usize]) -> Result<usize> {
        self.shape.validate_indices(indices)?;
        Ok(self.stride.offset(indices))
    }

    /// Computes the minimum buffer size needed for this layout.
    ///
    /// For contiguous tensors, this equals `numel()`.
    /// For strided tensors, this may be larger.
    #[must_use]
    pub fn min_buffer_size(&self) -> usize {
        if self.shape.numel() == 0 {
            return 0;
        }

        // Maximum offset + 1
        let dims = self.shape.dims();
        let strides = self.stride.values();

        dims.iter()
            .zip(strides.iter())
            .map(|(&d, &s)| if d > 0 { (d - 1) * s } else { 0 })
            .sum::<usize>()
            + 1
    }

    /// Creates a new layout with transposed dimensions.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidShape` if dimensions are out of bounds.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let mut new_dims = self.shape.dims().to_vec();
        if dim0 >= new_dims.len() || dim1 >= new_dims.len() {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "transpose dimensions ({dim0}, {dim1}) out of bounds for {} dims",
                    new_dims.len()
                ),
            });
        }
        new_dims.swap(dim0, dim1);

        let new_stride = self.stride.transpose(dim0, dim1)?;

        Ok(Self {
            shape: Shape::new(new_dims),
            stride: new_stride,
        })
    }

    /// Creates a new layout with permuted dimensions.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidShape` if permutation is invalid.
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        let old_dims = self.shape.dims();
        if dims.len() != old_dims.len() {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "permute requires {} dims, got {}",
                    old_dims.len(),
                    dims.len()
                ),
            });
        }

        let new_dims: Vec<usize> = dims
            .iter()
            .map(|&d| {
                old_dims.get(d).copied().ok_or_else(|| TensorError::InvalidShape {
                    reason: format!("invalid dimension {d} in permutation"),
                })
            })
            .collect::<Result<_>>()?;

        let new_stride = self.stride.permute(dims)?;

        Ok(Self {
            shape: Shape::new(new_dims),
            stride: new_stride,
        })
    }

    /// Creates a reshaped layout if the tensor is contiguous.
    ///
    /// Reshaping a non-contiguous tensor requires copying data.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ReshapeError` if element counts don't match
    /// or if the tensor is not contiguous.
    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
        self.shape.validate_reshape(&new_shape)?;

        if !self.is_contiguous() {
            return Err(TensorError::ReshapeError {
                from: self.shape.dims().to_vec(),
                to: new_shape.dims().to_vec(),
            });
        }

        Ok(Self::contiguous(new_shape))
    }
}

impl std::fmt::Display for Layout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Layout {{ shape: {}, stride: {:?}, contiguous: {} }}",
            self.shape,
            self.stride.values(),
            self.is_contiguous()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contiguous_layout() {
        let layout = Layout::contiguous(Shape::new(vec![2, 3, 4]));
        assert!(layout.is_contiguous());
        assert_eq!(layout.numel(), 24);
        assert_eq!(layout.min_buffer_size(), 24);
    }

    #[test]
    fn test_offset_calculation() {
        let layout = Layout::contiguous(Shape::matrix(3, 4));

        assert_eq!(layout.offset(&[0, 0]), 0);
        assert_eq!(layout.offset(&[0, 3]), 3);
        assert_eq!(layout.offset(&[1, 0]), 4);
        assert_eq!(layout.offset(&[2, 3]), 11);

        assert!(layout.checked_offset(&[3, 0]).is_err());
        assert!(layout.checked_offset(&[0, 4]).is_err());
    }

    #[test]
    fn test_transpose() {
        let layout = Layout::contiguous(Shape::matrix(3, 4));
        let transposed = layout.transpose(0, 1).unwrap();

        assert_eq!(transposed.dims(), &[4, 3]);
        assert!(!transposed.is_contiguous());

        // Check that element access still works correctly
        // Original [1, 2] should equal transposed [2, 1]
        assert_eq!(layout.offset(&[1, 2]), transposed.offset(&[2, 1]));
    }

    #[test]
    fn test_permute() {
        let layout = Layout::contiguous(Shape::new(vec![2, 3, 4]));
        let permuted = layout.permute(&[2, 0, 1]).unwrap();

        assert_eq!(permuted.dims(), &[4, 2, 3]);
        assert!(!permuted.is_contiguous());
    }

    #[test]
    fn test_reshape() {
        let layout = Layout::contiguous(Shape::matrix(3, 4));

        // Valid reshapes
        let reshaped = layout.reshape(Shape::new(vec![2, 6])).unwrap();
        assert_eq!(reshaped.dims(), &[2, 6]);
        assert!(reshaped.is_contiguous());

        let flat = layout.reshape(Shape::vector(12)).unwrap();
        assert_eq!(flat.dims(), &[12]);

        // Invalid reshape
        assert!(layout.reshape(Shape::new(vec![5, 3])).is_err());

        // Cannot reshape non-contiguous
        let transposed = layout.transpose(0, 1).unwrap();
        assert!(transposed.reshape(Shape::vector(12)).is_err());
    }

    #[test]
    fn test_min_buffer_size() {
        // Contiguous: buffer size equals element count
        let contig = Layout::contiguous(Shape::matrix(3, 4));
        assert_eq!(contig.min_buffer_size(), 12);

        // Strided (transposed): buffer size may differ
        let transposed = contig.transpose(0, 1).unwrap();
        assert_eq!(transposed.min_buffer_size(), 12); // Same underlying data
    }
}