//! Borrowed tensor views with zero-copy semantics.
//!
//! `TensorView` provides a read-only view into tensor data without copying.
//! This is useful for:
//! - Slicing existing tensors
//! - Memory-mapped data
//! - Avoiding copies in function parameters
//!
//! # Examples
//!
//! ```
//! use llm_engine::tensor::{Tensor, TensorView};
//!
//! let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let view = tensor.view();
//!
//! assert_eq!(view.get(&[0, 0]), Some(&1.0));
//! assert_eq!(view.get(&[1, 1]), Some(&4.0));
//! ```

use super::error::{Result, TensorError};
use super::layout::Layout;
use super::shape::Shape;
use super::storage::BorrowedStorage;
use super::stride::Stride;
use super::Tensor;

use std::fmt;

/// A borrowed view into tensor data.
///
/// `TensorView` does not own its data—it borrows from an existing tensor
/// or slice. This enables zero-copy operations like slicing and reshaping.
#[derive(Clone)]
pub struct TensorView<'a, T> {
    storage: BorrowedStorage<'a, T>,
    layout: Layout,
}

impl<'a, T> TensorView<'a, T> {
    /// Creates a view from a slice and shape.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ElementCountMismatch` if slice length doesn't match shape.
    pub fn new(data: &'a [T], shape: impl Into<Shape>) -> Result<Self> {
        let shape = shape.into();
        let expected_len = shape.numel();

        if data.len() != expected_len {
            return Err(TensorError::ElementCountMismatch {
                shape_elements: expected_len,
                data_len: data.len(),
            });
        }

        Ok(Self {
            storage: BorrowedStorage::new(data),
            layout: Layout::contiguous(shape),
        })
    }

    /// Creates a view from a slice with explicit layout.
    ///
    /// # Errors
    ///
    /// Returns `TensorError::ElementCountMismatch` if slice is too small for layout.
    pub fn with_layout(data: &'a [T], layout: Layout) -> Result<Self> {
        let min_size = layout.min_buffer_size();
        if data.len() < min_size {
            return Err(TensorError::ElementCountMismatch {
                shape_elements: min_size,
                data_len: data.len(),
            });
        }

        Ok(Self {
            storage: BorrowedStorage::new(data),
            layout,
        })
    }

    /// Creates a view from a slice, inferring a 1D shape.
    #[must_use]
    pub fn from_slice(data: &'a [T]) -> Self {
        let len = data.len();
        Self {
            storage: BorrowedStorage::new(data),
            layout: Layout::contiguous(Shape::vector(len)),
        }
    }

    /// Returns the view's shape.
    #[must_use]
    pub fn shape(&self) -> &Shape {
        self.layout.shape()
    }

    /// Returns the view's stride.
    #[must_use]
    pub fn stride(&self) -> &Stride {
        self.layout.stride()
    }

    /// Returns the view's layout.
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

    /// Checks if the view is contiguous in memory.
    #[must_use]
    pub fn is_contiguous(&self) -> bool {
        self.layout.is_contiguous()
    }

    /// Gets a reference to an element by indices.
    #[must_use]
    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let offset = self.layout.checked_offset(indices).ok()?;
        self.storage.as_slice().get(offset)
    }

    /// Returns the underlying data as a slice.
    #[must_use]
    pub fn as_slice(&self) -> &'a [T] {
        self.storage.as_slice()
    }

    /// Returns a raw pointer to the data.
    #[must_use]
    pub fn as_ptr(&self) -> *const T {
        self.storage.as_slice().as_ptr()
    }
}

impl<T: Clone> TensorView<'_, T> {
    /// Creates an owned tensor by copying the view's data.
    ///
    /// # Panics
    ///
    /// This method uses `.expect()` internally but will never panic in practice
    /// as the view's shape is guaranteed to match its data length.
    #[must_use]
    pub fn to_owned(&self) -> Tensor<T> {
        if self.is_contiguous() {
            Tensor::from_vec(self.storage.as_slice().to_vec(), self.shape().clone())
                .expect("contiguous view should have matching size")
        } else {
            // Copy in logical order for non-contiguous views
            let data = self.to_contiguous_vec();
            Tensor::from_vec(data, self.shape().clone())
                .expect("collected data should match shape")
        }
    }

    /// Creates a contiguous copy of the data.
    fn to_contiguous_vec(&self) -> Vec<T> {
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

    /// Returns a new view with transposed dimensions.
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are out of bounds.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        let new_layout = self.layout.transpose(dim0, dim1)?;
        Ok(Self {
            storage: self.storage.clone(),
            layout: new_layout,
        })
    }

    /// Returns a reshaped view if the data is contiguous.
    ///
    /// # Errors
    ///
    /// Returns error if element counts don't match or view is not contiguous.
    pub fn reshape(&self, new_shape: impl Into<Shape>) -> Result<Self> {
        let new_shape = new_shape.into();
        self.shape().validate_reshape(&new_shape)?;

        if !self.is_contiguous() {
            return Err(TensorError::ReshapeError {
                from: self.shape().dims().to_vec(),
                to: new_shape.dims().to_vec(),
            });
        }

        Ok(Self {
            storage: self.storage.clone(),
            layout: Layout::contiguous(new_shape),
        })
    }
}

// Conversion from owned Tensor to view
impl<T> Tensor<T> {
    /// Creates a borrowed view of this tensor.
    #[must_use]
    pub fn view(&self) -> TensorView<'_, T> {
        TensorView {
            storage: BorrowedStorage::new(self.as_slice()),
            layout: self.layout().clone(),
        }
    }

    /// Creates a view of a slice of this tensor's data.
    ///
    /// # Errors
    ///
    /// Returns error if the range is out of bounds or shape doesn't match.
    pub fn slice_view(&self, start: usize, shape: impl Into<Shape>) -> Result<TensorView<'_, T>> {
        let shape = shape.into();
        let end = start + shape.numel();

        if end > self.as_slice().len() {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![start, end],
                shape: vec![self.as_slice().len()],
            });
        }

        TensorView::new(&self.as_slice()[start..end], shape)
    }
}

impl<T: fmt::Debug> fmt::Debug for TensorView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("shape", &self.shape())
            .field("stride", &self.stride().values())
            .field("contiguous", &self.is_contiguous())
            .field("data_len", &self.storage.len())
            .finish_non_exhaustive()
    }
}

impl<T: fmt::Display> fmt::Display for TensorView<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TensorView(shape={}, ", self.shape())?;

        // Show first few elements
        let slice = self.storage.as_slice();
        let preview: Vec<String> = slice.iter().take(6).map(|x| format!("{x}")).collect();

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
    fn test_view_creation() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();

        assert_eq!(view.ndim(), 2);
        assert_eq!(view.numel(), 6);
        assert_eq!(view.dims(), &[2, 3]);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_view_from_slice() {
        let data = [1.0f32, 2.0, 3.0];
        let view = TensorView::from_slice(&data);

        assert_eq!(view.ndim(), 1);
        assert_eq!(view.dims(), &[3]);
    }

    #[test]
    fn test_view_element_access() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();

        assert!(approx_eq(*view.get(&[0, 0]).unwrap(), 1.0));
        assert!(approx_eq(*view.get(&[0, 2]).unwrap(), 3.0));
        assert!(approx_eq(*view.get(&[1, 0]).unwrap(), 4.0));
        assert!(approx_eq(*view.get(&[1, 2]).unwrap(), 6.0));

        assert_eq!(view.get(&[2, 0]), None);
    }

    #[test]
    fn test_tensor_to_view() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let view = tensor.view();

        assert_eq!(view.dims(), tensor.dims());
        assert!(approx_eq(*view.get(&[0, 0]).unwrap(), 1.0));
        assert!(approx_eq(*view.get(&[1, 1]).unwrap(), 4.0));
    }

    #[test]
    fn test_view_to_owned() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let view = TensorView::new(&data, vec![2, 2]).unwrap();
        let owned = view.to_owned();

        assert_eq!(owned.dims(), &[2, 2]);
        assert!(approx_eq(*owned.get(&[0, 0]).unwrap(), 1.0));
        assert!(approx_eq(*owned.get(&[1, 1]).unwrap(), 4.0));
    }

    #[test]
    fn test_view_reshape() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();

        let reshaped = view.reshape(vec![3, 2]).unwrap();
        assert_eq!(reshaped.dims(), &[3, 2]);

        // Same underlying data
        assert!(std::ptr::eq(view.as_ptr(), reshaped.as_ptr()));
    }

    #[test]
    fn test_view_transpose() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let view = TensorView::new(&data, vec![2, 3]).unwrap();

        let transposed = view.transpose(0, 1).unwrap();
        assert_eq!(transposed.dims(), &[3, 2]);
        assert!(!transposed.is_contiguous());

        // Check element mapping
        assert!(approx_eq(*transposed.get(&[0, 0]).unwrap(), 1.0));
        assert!(approx_eq(*transposed.get(&[1, 0]).unwrap(), 2.0));
        assert!(approx_eq(*transposed.get(&[0, 1]).unwrap(), 4.0));
    }

    #[test]
    fn test_slice_view() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();

        let view = tensor.slice_view(2, vec![3]).unwrap();
        assert_eq!(view.dims(), &[3]);
        assert!(approx_eq(*view.get(&[0]).unwrap(), 3.0));
        assert!(approx_eq(*view.get(&[2]).unwrap(), 5.0));
    }

    #[test]
    fn test_zero_copy() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let view = TensorView::new(&data, vec![2, 2]).unwrap();

        // Verify view points to same memory
        assert!(std::ptr::eq(data.as_ptr(), view.as_ptr()));

        // Reshape should also be zero-copy
        let reshaped = view.reshape(vec![4]).unwrap();
        assert!(std::ptr::eq(data.as_ptr(), reshaped.as_ptr()));
    }

    #[test]
    fn test_lifetime_correctness() {
        let owned = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // View borrows from tensor
        let view = owned.view();
        assert!(approx_eq(*view.get(&[0, 0]).unwrap(), 1.0));

        // Can create owned copy while view exists
        let copy = view.to_owned();
        assert!(approx_eq(*copy.get(&[0, 0]).unwrap(), 1.0));
    }
}