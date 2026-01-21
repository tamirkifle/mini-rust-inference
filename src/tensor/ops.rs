//! Tensor shape operations.
//!
//! This module provides non-computational operations that transform tensor
//! shape and layout without modifying the underlying data:
//!
//! - `reshape`: Change dimensions while preserving element count
//! - `transpose`: Swap two dimensions
//! - `permute`: Reorder all dimensions
//! - `squeeze`: Remove dimensions of size 1
//! - `unsqueeze`: Add a dimension of size 1
//! - `flatten`: Collapse to 1D
//!
//! Most operations return views when possible (zero-copy).

use super::error::{Result, TensorError};
use super::shape::Shape;
use super::view::TensorView;
use super::Tensor;

/// Extension trait for tensor shape operations.
///
/// Implemented for both `Tensor<T>` and `TensorView<'a, T>`.
pub trait ShapeOps {
    /// Returns the shape of the tensor.
    fn shape(&self) -> &Shape;

    /// Returns true if the tensor is contiguous in memory.
    fn is_contiguous(&self) -> bool;
}

impl<T> ShapeOps for Tensor<T> {
    fn shape(&self) -> &Shape {
        self.layout().shape()
    }

    fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }
}

impl<T> ShapeOps for TensorView<'_, T> {
    fn shape(&self) -> &Shape {
        self.layout().shape()
    }

    fn is_contiguous(&self) -> bool {
        self.layout().is_contiguous()
    }
}

// Additional operations for Tensor<T>
impl<T: Clone> Tensor<T> {
    /// Permutes the dimensions of the tensor.
    ///
    /// # Arguments
    ///
    /// * `dims` - New order of dimensions (e.g., `[2, 0, 1]` moves dim 2 to front)
    ///
    /// # Errors
    ///
    /// Returns error if permutation is invalid.
    ///
    /// # Panics
    ///
    /// This method uses `.expect()` internally but will never panic in practice
    /// as permutation preserves buffer size.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_engine::tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0f32; 24], vec![2, 3, 4]).unwrap();
    /// let permuted = t.permute(&[2, 0, 1]).unwrap();
    /// assert_eq!(permuted.dims(), &[4, 2, 3]);
    /// ```
    pub fn permute(&self, dims: &[usize]) -> Result<Self> {
        let new_layout = self.layout().permute(dims)?;
        Ok(Self::from_vec_with_layout(self.as_slice().to_vec(), new_layout)
            .expect("permute should preserve buffer size"))
    }

    /// Removes dimensions of size 1.
    ///
    /// If `dim` is `Some`, only that dimension is squeezed (error if not size 1).
    /// If `dim` is `None`, all size-1 dimensions are removed.
    ///
    /// # Errors
    ///
    /// Returns error if specified dimension is not size 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_engine::tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![1, 3, 1]).unwrap();
    /// let squeezed = t.squeeze(None).unwrap();
    /// assert_eq!(squeezed.dims(), &[3]);
    /// ```
    pub fn squeeze(&self, dim: Option<usize>) -> Result<Self> {
        let old_dims = self.dims();

        let new_dims: Vec<usize> = match dim {
            Some(d) => {
                if d >= old_dims.len() {
                    return Err(TensorError::InvalidShape {
                        reason: format!("dimension {d} out of bounds for {} dims", old_dims.len()),
                    });
                }
                if old_dims[d] != 1 {
                    return Err(TensorError::InvalidShape {
                        reason: format!(
                            "cannot squeeze dimension {d} with size {}",
                            old_dims[d]
                        ),
                    });
                }
                old_dims
                    .iter()
                    .enumerate()
                    .filter(|&(i, _)| i != d)
                    .map(|(_, &v)| v)
                    .collect()
            }
            None => old_dims.iter().copied().filter(|&d| d != 1).collect(),
        };

        // Handle case where all dims were 1 (scalar)
        let new_shape = if new_dims.is_empty() {
            Shape::scalar()
        } else {
            Shape::new(new_dims)
        };

        self.reshape(new_shape)
    }

    /// Adds a dimension of size 1 at the specified position.
    ///
    /// # Errors
    ///
    /// Returns error if dimension is out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_engine::tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap();
    /// let expanded = t.unsqueeze(0).unwrap();
    /// assert_eq!(expanded.dims(), &[1, 3]);
    /// ```
    pub fn unsqueeze(&self, dim: usize) -> Result<Self> {
        let old_dims = self.dims();

        if dim > old_dims.len() {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "dimension {dim} out of bounds for {} dims (max {})",
                    old_dims.len(),
                    old_dims.len()
                ),
            });
        }

        let mut new_dims = old_dims.to_vec();
        new_dims.insert(dim, 1);

        self.reshape(Shape::new(new_dims))
    }

    /// Flattens the tensor to 1D.
    ///
    /// # Panics
    ///
    /// This method uses `.expect()` internally but will never panic in practice
    /// as flattening always produces a valid shape.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_engine::tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0f32; 24], vec![2, 3, 4]).unwrap();
    /// let flat = t.flatten();
    /// assert_eq!(flat.dims(), &[24]);
    /// ```
    #[must_use]
    pub fn flatten(&self) -> Self {
        self.reshape(Shape::vector(self.numel()))
            .expect("flatten should always succeed")
    }

    /// Flattens dimensions from `start_dim` to `end_dim` (inclusive).
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are out of bounds or invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_engine::tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0f32; 24], vec![2, 3, 4]).unwrap();
    /// let partial = t.flatten_range(1, 2).unwrap();
    /// assert_eq!(partial.dims(), &[2, 12]);
    /// ```
    pub fn flatten_range(&self, start_dim: usize, end_dim: usize) -> Result<Self> {
        let old_dims = self.dims();

        if start_dim >= old_dims.len() || end_dim >= old_dims.len() {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "dimensions [{start_dim}, {end_dim}] out of bounds for {} dims",
                    old_dims.len()
                ),
            });
        }

        if start_dim > end_dim {
            return Err(TensorError::InvalidShape {
                reason: format!("start_dim {start_dim} > end_dim {end_dim}"),
            });
        }

        let mut new_dims = Vec::new();

        // Dims before start_dim
        new_dims.extend_from_slice(&old_dims[..start_dim]);

        // Flattened range
        let flat_size: usize = old_dims[start_dim..=end_dim].iter().product();
        new_dims.push(flat_size);

        // Dims after end_dim
        if end_dim + 1 < old_dims.len() {
            new_dims.extend_from_slice(&old_dims[end_dim + 1..]);
        }

        self.reshape(Shape::new(new_dims))
    }

    /// Returns a view of a contiguous slice along the first dimension.
    ///
    /// # Errors
    ///
    /// Returns error if indices are out of bounds.
    ///
    /// # Examples
    ///
    /// ```
    /// use llm_engine::tensor::Tensor;
    ///
    /// let t = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).unwrap();
    /// let sliced = t.narrow(0, 1, 2).unwrap(); // rows 1-2
    /// assert_eq!(sliced.dims(), &[2, 2]);
    /// ```
    pub fn narrow(&self, dim: usize, start: usize, length: usize) -> Result<Self> {
        let old_dims = self.dims();

        if dim >= old_dims.len() {
            return Err(TensorError::InvalidShape {
                reason: format!("dimension {dim} out of bounds for {} dims", old_dims.len()),
            });
        }

        if start + length > old_dims[dim] {
            return Err(TensorError::IndexOutOfBounds {
                index: vec![start, start + length],
                shape: vec![old_dims[dim]],
            });
        }

        // For contiguous tensors with dim=0, we can slice directly
        if dim == 0 && self.is_contiguous() {
            let stride: usize = old_dims[1..].iter().product();
            let start_offset = start * stride;
            let end_offset = start_offset + length * stride;

            let mut new_dims = old_dims.to_vec();
            new_dims[0] = length;

            return Tensor::from_vec(
                self.as_slice()[start_offset..end_offset].to_vec(),
                Shape::new(new_dims),
            );
        }

        // For other cases, we need to copy element by element
        let mut new_dims = old_dims.to_vec();
        new_dims[dim] = length;
        let new_shape = Shape::new(new_dims);

        let mut data = Vec::with_capacity(new_shape.numel());

        // Iterate through all indices of the new shape
        let mut indices = vec![0usize; old_dims.len()];
        for _ in 0..new_shape.numel() {
            // Map new indices to old indices
            let mut old_indices = indices.clone();
            old_indices[dim] += start;

            if let Some(val) = self.get(&old_indices) {
                data.push(val.clone());
            }

            // Increment indices
            for i in (0..indices.len()).rev() {
                indices[i] += 1;
                if i == dim {
                    if indices[i] < length {
                        break;
                    }
                } else if indices[i] < old_dims[i] {
                    break;
                }
                indices[i] = 0;
            }
        }

        Tensor::from_vec(data, new_shape)
    }
}

/// Broadcasts two shapes to a common shape.
///
/// Broadcasting rules (NumPy-style):
/// 1. Shapes are compared from right to left
/// 2. Dimensions are compatible if equal or one is 1
/// 3. Missing dimensions are treated as 1
///
/// # Errors
///
/// Returns error if shapes are not broadcastable.
pub fn broadcast_shapes(a: &Shape, b: &Shape) -> Result<Shape> {
    let a_dims = a.dims();
    let b_dims = b.dims();

    let max_ndim = a_dims.len().max(b_dims.len());
    let mut result = Vec::with_capacity(max_ndim);

    for i in 0..max_ndim {
        let a_dim = if i < a_dims.len() {
            a_dims[a_dims.len() - 1 - i]
        } else {
            1
        };
        let b_dim = if i < b_dims.len() {
            b_dims[b_dims.len() - 1 - i]
        } else {
            1
        };

        if a_dim == b_dim {
            result.push(a_dim);
        } else if a_dim == 1 {
            result.push(b_dim);
        } else if b_dim == 1 {
            result.push(a_dim);
        } else {
            return Err(TensorError::ShapeMismatch {
                expected: a_dims.to_vec(),
                got: b_dims.to_vec(),
            });
        }
    }

    result.reverse();
    Ok(Shape::new(result))
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-6;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_permute() {
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![2, 3, 4]).unwrap();

        let permuted = tensor.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.dims(), &[4, 2, 3]);

        // Verify element mapping: original [i,j,k] -> permuted [k,i,j]
        assert!(approx_eq(*tensor.get(&[0, 0, 0]).unwrap(), *permuted.get(&[0, 0, 0]).unwrap()));
        assert!(approx_eq(*tensor.get(&[1, 2, 3]).unwrap(), *permuted.get(&[3, 1, 2]).unwrap()));
    }

    #[test]
    fn test_squeeze_all() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![1, 3, 1]).unwrap();
        let squeezed = tensor.squeeze(None).unwrap();
        assert_eq!(squeezed.dims(), &[3]);
    }

    #[test]
    fn test_squeeze_specific() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![1, 3, 1]).unwrap();

        let squeezed_0 = tensor.squeeze(Some(0)).unwrap();
        assert_eq!(squeezed_0.dims(), &[3, 1]);

        let squeezed_2 = tensor.squeeze(Some(2)).unwrap();
        assert_eq!(squeezed_2.dims(), &[1, 3]);

        // Cannot squeeze non-1 dimension
        assert!(tensor.squeeze(Some(1)).is_err());
    }

    #[test]
    fn test_unsqueeze() {
        let tensor = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], vec![3]).unwrap();

        let expanded_0 = tensor.unsqueeze(0).unwrap();
        assert_eq!(expanded_0.dims(), &[1, 3]);

        let expanded_1 = tensor.unsqueeze(1).unwrap();
        assert_eq!(expanded_1.dims(), &[3, 1]);
    }

    #[test]
    fn test_flatten() {
        let tensor = Tensor::from_vec(vec![1.0f32; 24], vec![2, 3, 4]).unwrap();
        let flat = tensor.flatten();
        assert_eq!(flat.dims(), &[24]);
    }

    #[test]
    fn test_flatten_range() {
        let tensor = Tensor::from_vec(vec![1.0f32; 24], vec![2, 3, 4]).unwrap();

        let partial = tensor.flatten_range(1, 2).unwrap();
        assert_eq!(partial.dims(), &[2, 12]);

        let full = tensor.flatten_range(0, 2).unwrap();
        assert_eq!(full.dims(), &[24]);
    }

    #[test]
    fn test_narrow() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data, vec![4, 3]).unwrap();

        // Narrow along dim 0 (rows)
        let sliced = tensor.narrow(0, 1, 2).unwrap();
        assert_eq!(sliced.dims(), &[2, 3]);
        assert!(approx_eq(*sliced.get(&[0, 0]).unwrap(), 3.0)); // Original [1, 0]
        assert!(approx_eq(*sliced.get(&[1, 2]).unwrap(), 8.0)); // Original [2, 2]
    }

    #[test]
    fn test_broadcast_shapes() {
        // Same shapes
        let result = broadcast_shapes(&Shape::new(vec![3, 4]), &Shape::new(vec![3, 4])).unwrap();
        assert_eq!(result.dims(), &[3, 4]);

        // Broadcast scalar
        let result = broadcast_shapes(&Shape::new(vec![3, 4]), &Shape::scalar()).unwrap();
        assert_eq!(result.dims(), &[3, 4]);

        // Broadcast with 1s
        let result = broadcast_shapes(&Shape::new(vec![3, 4]), &Shape::new(vec![1, 4])).unwrap();
        assert_eq!(result.dims(), &[3, 4]);

        // Different ranks
        let result = broadcast_shapes(&Shape::new(vec![2, 3, 4]), &Shape::new(vec![4])).unwrap();
        assert_eq!(result.dims(), &[2, 3, 4]);

        // Incompatible
        let result = broadcast_shapes(&Shape::new(vec![3, 4]), &Shape::new(vec![2, 4]));
        assert!(result.is_err());
    }

    #[test]
    fn test_reshape_transpose_roundtrip() {
        // Property: reshape(transpose(x)) should preserve data (when made contiguous)
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let tensor = Tensor::from_vec(data.clone(), vec![3, 4]).unwrap();

        let transposed = tensor.transpose(0, 1).unwrap();
        let contiguous = transposed.contiguous();
        let reshaped = contiguous.reshape(vec![12]).unwrap();

        // All original elements should still be present
        let result_data = reshaped.as_slice();
        for val in &data {
            assert!(result_data.iter().any(|x| approx_eq(*x, *val)));
        }
    }
}