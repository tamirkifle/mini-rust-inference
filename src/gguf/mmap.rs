//! Memory-mapped file access for GGUF tensor data.
//!
//! This module provides zero-copy access to tensor data using OS memory mapping.
//! When a GGUF file is memory-mapped:
//!
//! - The file is not loaded into RAM immediately
//! - Pages are loaded on-demand when accessed (demand paging)
//! - Multiple processes can share the same physical pages
//! - The OS manages caching and eviction automatically
//!
//! # Benefits
//!
//! - **Constant memory overhead**: Opening a 13GB model uses ~constant RAM
//! - **Fast startup**: No need to read entire file before use
//! - **Zero-copy**: Tensor data accessed directly from mapped pages
//! - **OS-managed caching**: Frequently accessed data stays in RAM
//!
//! # Platform Notes
//!
//! - Linux: Uses `mmap(2)` with `MAP_PRIVATE`
//! - macOS: Uses `mmap(2)` with similar semantics
//! - Windows: Uses `CreateFileMapping`/`MapViewOfFile`
//!
//! # Safety
//!
//! The memory-mapped region must outlive any references to tensor data.
//! This is enforced through Rust's lifetime system.

use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::Path;

use memmap2::Mmap;

/// A memory-mapped file providing read-only access.
///
/// This wraps `memmap2::Mmap` and provides a safe interface for
/// accessing file contents as a byte slice.
///
/// # Example
///
/// ```no_run
/// use llm_engine::gguf::MappedFile;
///
/// let mapped = MappedFile::open("model.gguf").expect("failed to map file");
/// println!("File size: {} bytes", mapped.len());
///
/// // Access data as a byte slice
/// let header_bytes = &mapped[0..24];
/// ```
#[derive(Debug)]
pub struct MappedFile {
    /// The underlying memory map.
    mmap: Mmap,

    /// Original file path (for error messages).
    path: String,
}

impl MappedFile {
    /// Opens and memory-maps a file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to map
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or mapped.
    ///
    /// # Safety
    ///
    /// This function is safe to call, but the resulting `MappedFile` must
    /// outlive any slices or references derived from it.
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref)?;

        // SAFETY: The file is opened read-only, and we ensure the MappedFile
        // outlives any references to the mapped data through Rust's lifetime system.
        let mmap = unsafe { Mmap::map(&file)? };

        Ok(Self {
            mmap,
            path: path_ref.display().to_string(),
        })
    }

    /// Opens and memory-maps a file from an already-opened File handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be mapped.
    pub fn from_file(file: &File, path: &str) -> io::Result<Self> {
        // SAFETY: Same as open()
        let mmap = unsafe { Mmap::map(file)? };

        Ok(Self {
            mmap,
            path: path.to_string(),
        })
    }

    /// Returns the length of the mapped region in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Returns true if the mapped region is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.mmap.is_empty()
    }

    /// Returns the file path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns a pointer to the start of the mapped region.
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.mmap.as_ptr()
    }

    /// Returns the mapped data as a byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.mmap
    }

    /// Returns a slice of the mapped data at the given range.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    #[must_use]
    pub fn slice(&self, start: usize, len: usize) -> &[u8] {
        &self.mmap[start..start + len]
    }

    /// Returns a slice of the mapped data at the given range, or None if out of bounds.
    #[must_use]
    pub fn try_slice(&self, start: usize, len: usize) -> Option<&[u8]> {
        let end = start.checked_add(len)?;
        if end <= self.mmap.len() {
            Some(&self.mmap[start..end])
        } else {
            None
        }
    }

    /// Advises the OS about expected access patterns.
    ///
    /// This is a hint and may be ignored by the OS.
    ///
    /// # Errors
    ///
    /// Returns an error if the advise call fails (rare).
    #[cfg(unix)]
    pub fn advise_sequential(&self) -> io::Result<()> {
        self.mmap.advise(memmap2::Advice::Sequential)
    }

    /// Advises the OS that data will be accessed randomly.
    #[cfg(unix)]
    pub fn advise_random(&self) -> io::Result<()> {
        self.mmap.advise(memmap2::Advice::Random)
    }

    /// Advises the OS that data will be needed soon.
    #[cfg(unix)]
    pub fn advise_willneed(&self) -> io::Result<()> {
        self.mmap.advise(memmap2::Advice::WillNeed)
    }

    /// Advises the OS about a specific byte range within the mapping.
    ///
    /// `offset` and `len` are in bytes relative to the start of the mapping.
    /// Pass `willneed = true` to prefetch (WillNeed).
    /// `false` is a no-op on this API; `DontNeed` requires `unsafe` in memmap2.
    ///
    /// Silently no-ops if `offset + len` exceeds the mapping length.
    #[cfg(unix)]
    pub fn advise_range(&self, willneed: bool, offset: usize, len: usize) -> io::Result<()> {
        if !willneed {
            return Ok(()); // DontNeed requires unsafe; skip — OS evicts naturally
        }
        if offset.saturating_add(len) > self.mmap.len() {
            return Ok(());
        }
        self.mmap.advise_range(memmap2::Advice::WillNeed, offset, len)
    }

    /// Non-Unix platforms: no-op
    #[cfg(not(unix))]
    pub fn advise_sequential(&self) -> io::Result<()> {
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn advise_random(&self) -> io::Result<()> {
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn advise_willneed(&self) -> io::Result<()> {
        Ok(())
    }

    #[cfg(not(unix))]
    pub fn advise_range(&self, _willneed: bool, _offset: usize, _len: usize) -> io::Result<()> {
        Ok(())
    }
}

impl Deref for MappedFile {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        &self.mmap
    }
}

impl AsRef<[u8]> for MappedFile {
    fn as_ref(&self) -> &[u8] {
        &self.mmap
    }
}

/// A view into a portion of a memory-mapped file.
///
/// This provides a safe way to reference a slice of mapped data
/// while ensuring the underlying `MappedFile` remains valid.
#[derive(Debug, Clone, Copy)]
pub struct MappedSlice<'a> {
    data: &'a [u8],
}

impl<'a> MappedSlice<'a> {
    /// Creates a new mapped slice.
    #[must_use]
    pub const fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Returns the length of the slice.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if the slice is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a pointer to the start of the slice.
    #[must_use]
    pub const fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    /// Returns the underlying byte slice.
    #[must_use]
    pub const fn as_bytes(&self) -> &'a [u8] {
        self.data
    }

    /// Interprets the slice as a slice of a different type.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - The slice is properly aligned for type T
    /// - The slice length is a multiple of `size_of::<T>()`
    /// - The byte pattern is valid for type T
    ///
    /// # Panics
    ///
    /// Panics if alignment or size requirements are not met.
    #[must_use]
    pub unsafe fn cast<T: Copy>(&self) -> &'a [T] {
        let ptr = self.data.as_ptr();
        let len = self.data.len();

        assert!(
            ptr.align_offset(std::mem::align_of::<T>()) == 0,
            "slice not aligned for type"
        );
        assert!(
            len % std::mem::size_of::<T>() == 0,
            "slice length not multiple of type size"
        );

        let count = len / std::mem::size_of::<T>();
        std::slice::from_raw_parts(ptr.cast::<T>(), count)
    }

    /// Safely interprets the slice as f32 values.
    ///
    /// Returns None if alignment or size requirements are not met.
    #[must_use]
    pub fn as_f32(&self) -> Option<&'a [f32]> {
        let ptr = self.data.as_ptr();
        let len = self.data.len();

        if ptr.align_offset(std::mem::align_of::<f32>()) != 0 {
            return None;
        }
        if len % std::mem::size_of::<f32>() != 0 {
            return None;
        }

        let count = len / std::mem::size_of::<f32>();
        // SAFETY: We checked alignment and size
        Some(unsafe { std::slice::from_raw_parts(ptr.cast::<f32>(), count) })
    }

    /// Safely interprets the slice as f16 values (stored as u16).
    ///
    /// Returns None if alignment or size requirements are not met.
    #[must_use]
    pub fn as_f16_bits(&self) -> Option<&'a [u16]> {
        let ptr = self.data.as_ptr();
        let len = self.data.len();

        if ptr.align_offset(std::mem::align_of::<u16>()) != 0 {
            return None;
        }
        if len % std::mem::size_of::<u16>() != 0 {
            return None;
        }

        let count = len / std::mem::size_of::<u16>();
        // SAFETY: We checked alignment and size
        Some(unsafe { std::slice::from_raw_parts(ptr.cast::<u16>(), count) })
    }
}

impl<'a> Deref for MappedSlice<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a> AsRef<[u8]> for MappedSlice<'a> {
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

impl<'a> From<&'a [u8]> for MappedSlice<'a> {
    fn from(data: &'a [u8]) -> Self {
        Self::new(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn create_test_file(data: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(data).expect("write data");
        file.flush().expect("flush");
        file
    }

    #[test]
    fn test_mapped_file_basic() {
        let data = b"Hello, memory-mapped world!";
        let temp = create_test_file(data);

        let mapped = MappedFile::open(temp.path()).expect("open mapped file");

        assert_eq!(mapped.len(), data.len());
        assert!(!mapped.is_empty());
        assert_eq!(mapped.as_slice(), data);
    }

    #[test]
    fn test_mapped_file_slice() {
        let data = b"0123456789ABCDEF";
        let temp = create_test_file(data);

        let mapped = MappedFile::open(temp.path()).expect("open mapped file");

        assert_eq!(mapped.slice(0, 4), b"0123");
        assert_eq!(mapped.slice(10, 6), b"ABCDEF");
        assert_eq!(mapped.try_slice(0, 4), Some(&b"0123"[..]));
        assert_eq!(mapped.try_slice(100, 4), None);
    }

    #[test]
    fn test_mapped_file_deref() {
        let data = b"test data";
        let temp = create_test_file(data);

        let mapped = MappedFile::open(temp.path()).expect("open mapped file");

        // Can use slice indexing through Deref
        assert_eq!(&mapped[0..4], b"test");
        assert_eq!(mapped.len(), 9);
    }

    #[test]
    fn test_mapped_slice_as_f32() {
        // Create aligned f32 data
        let floats: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(floats.as_ptr().cast::<u8>(), std::mem::size_of_val(&floats))
        };

        let temp = create_test_file(bytes);
        let mapped = MappedFile::open(temp.path()).expect("open mapped file");

        let slice = MappedSlice::new(mapped.as_slice());
        let f32_slice = slice.as_f32().expect("should be aligned");

        assert_eq!(f32_slice.len(), 4);
        assert!((f32_slice[0] - 1.0).abs() < 1e-6);
        assert!((f32_slice[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_mapped_slice_unaligned() {
        let data = b"12345"; // 5 bytes, not aligned for f32
        let temp = create_test_file(data);

        let mapped = MappedFile::open(temp.path()).expect("open mapped file");
        let slice = MappedSlice::new(&mapped[1..5]); // Offset by 1, definitely unaligned

        // Should return None for unaligned/wrong-sized data
        // Note: as_f32() checks both alignment and size
        assert!(slice.as_f32().is_none() || slice.len() % 4 != 0);
    }

    #[test]
    fn test_mapped_file_path() {
        let data = b"test";
        let temp = create_test_file(data);
        let path_str = temp.path().display().to_string();

        let mapped = MappedFile::open(temp.path()).expect("open mapped file");

        assert_eq!(mapped.path(), path_str);
    }

    #[test]
    fn test_advise_calls() {
        let data = b"test data for advise";
        let temp = create_test_file(data);

        let mapped = MappedFile::open(temp.path()).expect("open mapped file");

        // These should not fail (they're hints)
        assert!(mapped.advise_sequential().is_ok());
        assert!(mapped.advise_random().is_ok());
        assert!(mapped.advise_willneed().is_ok());
    }

    #[test]
    fn test_empty_file() {
        let temp = create_test_file(b"");

        let mapped = MappedFile::open(temp.path()).expect("open mapped file");

        assert!(mapped.is_empty());
        assert_eq!(mapped.len(), 0);
    }

    #[test]
    fn test_large_slice_boundaries() {
        let data: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let temp = create_test_file(&data);

        let mapped = MappedFile::open(temp.path()).expect("open mapped file");

        // Test boundaries
        assert_eq!(mapped.slice(0, 1024).len(), 1024);
        assert_eq!(mapped.try_slice(0, 1025), None);
        assert_eq!(mapped.try_slice(1000, 24), Some(&data[1000..1024]));
        assert_eq!(mapped.try_slice(1000, 25), None);
    }
}