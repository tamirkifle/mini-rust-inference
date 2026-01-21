//! Storage backends for tensor data.
//!
//! Tensors can store data in two ways:
//! - **Owned**: `Vec<T>` — tensor owns its data
//! - **Borrowed**: `&[T]` — tensor borrows data from elsewhere
//!
//! This module provides traits and types to abstract over storage.

use std::ops::Deref;

/// Trait for types that can provide a slice of tensor data.
///
/// Implemented by both owned (`Vec<T>`) and borrowed (`&[T]`) storage.
pub trait Storage<T>: Deref<Target = [T]> {
    /// Returns the number of elements in storage.
    fn len(&self) -> usize {
        self.deref().len()
    }

    /// Returns true if storage is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Storage<T> for Vec<T> {}
impl<T> Storage<T> for &[T] {}
impl<T> Storage<T> for Box<[T]> {}

/// Owned storage backed by a `Vec<T>`.
#[derive(Debug, Clone)]
pub struct OwnedStorage<T> {
    data: Vec<T>,
}

impl<T> OwnedStorage<T> {
    /// Creates new owned storage from a vector.
    #[must_use]
    pub const fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    /// Returns a reference to the underlying vector.
    #[must_use]
    pub const fn as_vec(&self) -> &Vec<T> {
        &self.data
    }

    /// Consumes the storage and returns the underlying vector.
    #[must_use]
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// Returns a mutable reference to the underlying data.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Deref for OwnedStorage<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> Storage<T> for OwnedStorage<T> {}

impl<T> From<Vec<T>> for OwnedStorage<T> {
    fn from(data: Vec<T>) -> Self {
        Self::new(data)
    }
}

/// Borrowed storage backed by a slice reference.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedStorage<'a, T> {
    data: &'a [T],
}

impl<'a, T> BorrowedStorage<'a, T> {
    /// Creates new borrowed storage from a slice.
    #[must_use]
    pub const fn new(data: &'a [T]) -> Self {
        Self { data }
    }

    /// Returns the underlying slice.
    #[must_use]
    pub const fn as_slice(&self) -> &'a [T] {
        self.data
    }
}

impl<T> Deref for BorrowedStorage<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<T> Storage<T> for BorrowedStorage<'_, T> {}

impl<'a, T> From<&'a [T]> for BorrowedStorage<'a, T> {
    fn from(data: &'a [T]) -> Self {
        Self::new(data)
    }
}

impl<'a, T> From<&'a Vec<T>> for BorrowedStorage<'a, T> {
    fn from(data: &'a Vec<T>) -> Self {
        Self::new(data.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_owned_storage() {
        let storage = OwnedStorage::new(vec![1.0f32, 2.0, 3.0]);
        assert_eq!(storage.len(), 3);
        assert_eq!(&storage[0..2], &[1.0, 2.0]);
    }

    #[test]
    fn test_borrowed_storage() {
        let data = vec![1.0f32, 2.0, 3.0];
        let storage = BorrowedStorage::new(&data);
        assert_eq!(storage.len(), 3);
        assert_eq!(&storage[1..], &[2.0, 3.0]);
    }

    #[test]
    fn test_storage_trait() {
        fn use_storage<S: Storage<f32>>(s: &S) -> usize {
            s.len()
        }

        let owned = OwnedStorage::new(vec![1.0f32, 2.0]);
        let data = vec![1.0f32, 2.0, 3.0];
        let borrowed = BorrowedStorage::new(&data);

        assert_eq!(use_storage(&owned), 2);
        assert_eq!(use_storage(&borrowed), 3);
    }
}