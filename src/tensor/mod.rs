//! Tensor module - N-dimensional array abstraction.
//!
//! This module provides the core tensor types for the inference engine:
//!
//! - [`Shape`]: N-dimensional shape representation
//! - [`Stride`]: Memory stride calculations
//! - [`Layout`]: Combined shape and stride
//! - [`Tensor`]: Owned N-dimensional array
//!
//! # Memory Layout
//!
//! Tensors use row-major (C-style) contiguous layout by default:
//!
//! ```text
//! Shape:  [2, 3]
//! Stride: [3, 1]
//! Memory: [a, b, c, d, e, f]
//!
//! Logical view:
//!   [[a, b, c],
//!    [d, e, f]]
//! ```
//!
//! # Examples
//!
//! ```
//! use llm_engine::tensor::Tensor;
//!
//! // Create a 2x3 matrix
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
//! let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();
//!
//! assert_eq!(tensor.shape().dims(), &[2, 3]);
//! assert_eq!(tensor.get(&[0, 0]), Some(&1.0));
//! assert_eq!(tensor.get(&[1, 2]), Some(&6.0));
//! ```

mod error;
mod layout;
mod shape;
mod stride;

pub use error::{Result, TensorError};
pub use layout::Layout;
pub use shape::Shape;
pub use stride::Stride;

use std::fmt;

/// N-dimensional tensor with owned data.
///
/// Generic over element type `T`. For inference, typically `f32` or quantized types.
#[derive(Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    layout: Layout,
}

impl<T> Tensor<T> {
    /// Creates a tensor from data and shape.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ElementCountMismatch` if data length doesn't match shape.
    pub fn from_vec(data: Vec<T>, shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let expected_len = shape.numel();

        if data.len() != expected_len {
            return Err(TensorError::ElementCountMismatch {
                shape_elements: expected_len,
                data_len: data.len(),
            });
        }

        Ok(Self {
            data,
            layout: Layout::contiguous(shape),
        })
    }

    /// Creates a tensor from data with explicit layout.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ElementCountMismatch` if data length is insufficient for layout.
    pub fn from_vec_with_layout(data: Vec<T>, layout: Layout) -> Result<Self> {
        let min_size = layout.min_buffer_size();
        if data.len() < min_size {
            return Err(TensorError::ElementCountMismatch {
                shape_elements: min_size,
                data_len: data.len(),
            });
        }

        Ok(Self { data, layout })
    }

    /// Returns the tensor's shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// Returns the tensor's stride.
    #[must_use]
    pub fn stride(&self) -> &Stride {
        self.layout.stride()
    }

    /// Returns the tensor's layout.
    #[must_use]
    pub const fn layout(&self) -> &Layout {
        &self.layout
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> usize {
        self.layout.numel()
    }

    /// Returns the dimensions as a slice.
    #[must_use]
    pub fn dims(&self) -> &[usize] {
        self.layout.dims()
    }

    /// Checks if the tensor is contiguous in memory.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Gets a reference to an element by indices.
    #[must_use]
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.layout.checked_offset(indices).ok()?;
        self.data.get(offset)
    }

    /// Gets a mutable reference to an element by indices.
    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let offset = self.layout.checked_offset(indices).ok()?;
        self.data.get_mut(offset)
    }

    /// Returns the underlying data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Returns the underlying data as a mutable slice.
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the tensor and returns the underlying data.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Returns a raw pointer to the data.
    #[must_use]
    pub const fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    /// Returns a mutable raw pointer to the data.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

impl<T: Clone> Tensor<T> {
    /// Reshapes the tensor to a new shape.
    ///
    /// For non-contiguous tensors, this will copy the data.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ReshapeError` if element counts don't match.
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Self> {
        let new_shape = new_shape.into();
        self.shape().validate_reshape(&new_shape)?;

        if self.is_contiguous() {
            // Can reuse data
            Ok(Self {
                data: self.data.clone(),
                layout: Layout::contiguous(new_shape),
            })
        } else {
            // Must copy to contiguous
            let data = self.to_contiguous_vec();
            Ok(Self {
                data,
                layout: Layout::contiguous(new_shape),
            })
        }
    }

    /// Creates a contiguous copy of the tensor data.
    fn to_contiguous_vec(&self) -> Vec<T> {
        if self.is_contiguous() {
            return self.data.clone();
        }

        // Iterate in logical order and copy
        let mut result = Vec::with_capacity(self.numel());
        self.for_each_index(|indices| {
            if let Some(val) = self.get(indices) {
                result.push(val.clone());
            }
        });
        result
    }

    /// Iterates over all valid indices in row-major order.
    fn for_each_index<F>(&self, mut f: F)
    where
        F: FnMut(&[usize]),
    {
        let dims = self.dims();
        if dims.is_empty() {
            f(&[]);
            return;
        }

        let mut indices = vec![0usize; dims.len()];
        loop {
            f(&indices);

            // Increment indices (row-major order)
            let mut i = dims.len() - 1;
            loop {
                indices[i] += 1;
                if indices[i] < dims[i] {
                    break;
                }
                indices[i] = 0;
                if i == 0 {
                    return;
                }
                i -= 1;
            }
        }
    }

    /// Transposes two dimensions, returning a new tensor.
    ///
    /// This creates a view with different strides. Call `contiguous()` to
    /// copy the data to a contiguous layout.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::InvalidShape` if dimensions are out of bounds.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let new_layout = self.layout.transpose(dim0, dim1)?;
        Ok(Self {
            data: self.data.clone(),
            layout: new_layout,
        })
    }

    /// Returns a contiguous copy of the tensor.
    #[must_use]
    pub fn contiguous(&self) -> Self {
        if self.is_contiguous() {
            return self.clone();
        }

        let data = self.to_contiguous_vec();
        Self {
            data,
            layout: Layout::contiguous(self.shape().clone()),
        }
    }
}

impl<T: Default + Clone> Tensor<T> {
    /// Creates a tensor filled with the default value.
    #[must_use]
    pub fn zeros(shape: impl Into<Shape>) -> Self {
        let shape = shape.into();
        let data = vec![T::default(); shape.numel()];
        Self {
            data,
            layout: Layout::contiguous(shape),
        }
    }
}

impl<T: Clone> Tensor<T> {
    /// Creates a tensor filled with a specific value.
    #[must_use]
    pub fn full(shape: impl Into<Shape>, value: T) -> Self {
        let shape = shape.into();
        let data = vec![value; shape.numel()];
        Self {
            data,
            layout: Layout::contiguous(shape),
        }
    }
}

impl Tensor<f32> {
    /// Creates a tensor filled with ones.
    #[must_use]
    pub fn ones(shape: impl Into<Shape>) -> Self {
        Self::full(shape, 1.0)
    }
}

impl<T: fmt::Debug> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("stride", &self.stride().values())
            .field("contiguous", &self.is_contiguous())
            .field("data_len", &self.data.len())
            .finish_non_exhaustive()
    }
}

impl<T: fmt::Display + Clone> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor(shape={}, ", self.shape())?;

        // Show first few elements
        let preview: Vec<String> = self
            .data
            .iter()
            .take(6)
            .map(|x| format!("{x}"))
            .collect();

        if self.numel() > 6 {
            write!(f, "[{}, ...])", preview.join(", "))
        } else {
            write!(f, "[{}])", preview.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_tensor_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.dims(), &[2, 3]);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_tensor_element_access() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        assert!(approx_eq(*tensor.get(&[0, 0]).unwrap(), 1.0));
        assert!(approx_eq(*tensor.get(&[0, 1]).unwrap(), 2.0));
        assert!(approx_eq(*tensor.get(&[0, 2]).unwrap(), 3.0));
        assert!(approx_eq(*tensor.get(&[1, 0]).unwrap(), 4.0));
        assert!(approx_eq(*tensor.get(&[1, 1]).unwrap(), 5.0));
        assert!(approx_eq(*tensor.get(&[1, 2]).unwrap(), 6.0));

        // Out of bounds
        assert_eq!(tensor.get(&[2, 0]), None);
        assert_eq!(tensor.get(&[0, 3]), None);
    }

    #[test]
    fn test_tensor_mutate() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut tensor = Tensor::from_vec(data, vec![2, 2]).unwrap();

        *tensor.get_mut(&[0, 1]).unwrap() = 10.0;
        assert!(approx_eq(*tensor.get(&[0, 1]).unwrap(), 10.0));
    }

    #[test]
    fn test_zeros_and_ones() {
        let zeros = Tensor::<f32>::zeros(vec![2, 3]);
        assert_eq!(zeros.numel(), 6);
        assert!(zeros.as_slice().iter().all(|&x| approx_eq(x, 0.0)));

        let ones = Tensor::<f32>::ones(vec![2, 3]);
        assert!(ones.as_slice().iter().all(|&x| approx_eq(x, 1.0)));
    }

    #[test]
    fn test_reshape() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let reshaped = tensor.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.dims(), &[3, 2]);
        assert!(approx_eq(*reshaped.get(&[0, 0]).unwrap(), 1.0));
        assert!(approx_eq(*reshaped.get(&[0, 1]).unwrap(), 2.0));
        assert!(approx_eq(*reshaped.get(&[1, 0]).unwrap(), 3.0));

        // Invalid reshape
        assert!(tensor.reshape(vec![4, 2]).is_err());
    }

    #[test]
    fn test_transpose() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();

        let transposed = tensor.transpose(0, 1).unwrap();
        assert_eq!(transposed.dims(), &[3, 2]);
        assert!(!transposed.is_contiguous());

        // Check element mapping
        // Original [0, 1] = 2.0 should be transposed [1, 0]
        assert!(approx_eq(*transposed.get(&[0, 0]).unwrap(), 1.0));
        assert!(approx_eq(*transposed.get(&[1, 0]).unwrap(), 2.0));
        assert!(approx_eq(*transposed.get(&[0, 1]).unwrap(), 4.0));
        assert!(approx_eq(*transposed.get(&[2, 1]).unwrap(), 6.0));
    }

    #[test]
    fn test_contiguous_copy() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor = Tensor::from_vec(data, vec![2, 3]).unwrap();
        let transposed = tensor.transpose(0, 1).unwrap();

        assert!(!transposed.is_contiguous());

        let contiguous = transposed.contiguous();
        assert!(contiguous.is_contiguous());
        assert_eq!(contiguous.dims(), &[3, 2]);

        // Data should be in new order
        let expected = &[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (got, exp) in contiguous.as_slice().iter().zip(expected.iter()) {
            assert!(approx_eq(*got, *exp));
        }
    }

    #[test]
    fn test_data_size_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let result = Tensor::from_vec(data, vec![2, 3]);
        assert!(result.is_err());
    }
}