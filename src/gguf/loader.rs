//! High-level GGUF file loader.
//!
//! This module provides a unified interface for loading GGUF model files,
//! combining header parsing, metadata extraction, tensor info, and
//! memory-mapped data access.
//!
//! # Example
//!
//! ```no_run
//! use llm_engine::gguf::GgufLoader;
//!
//! // Open a GGUF model file
//! let loader = GgufLoader::open("model.gguf").expect("failed to open model");
//!
//! // Inspect model metadata
//! println!("Version: {}", loader.header().version());
//! println!("Tensors: {}", loader.header().tensor_count());
//!
//! if let Some(arch) = loader.metadata().get_str("general.architecture") {
//!     println!("Architecture: {arch}");
//! }
//!
//! // List all tensors
//! for tensor in loader.tensors().iter() {
//!     println!("{}: {:?} ({})", tensor.name(), tensor.dims(), tensor.dtype());
//! }
//!
//! // Get tensor data (zero-copy from memory map)
//! if let Some(data) = loader.tensor_data("model.embed_tokens.weight") {
//!     println!("Embedding data: {} bytes", data.len());
//! }
//! ```
//!
//! # Memory Efficiency
//!
//! The loader uses memory mapping, so:
//! - Opening a file is nearly instant regardless of size
//! - Memory usage is proportional to accessed data, not file size
//! - The OS handles caching and paging automatically

use super::dtype::GgmlType;
use super::error::{GgufError, Result};
use super::header::GgufHeader;
use super::metadata::Metadata;
use super::mmap::{MappedFile, MappedSlice};
use super::tensor_info::{align_offset, TensorInfo, TensorInfos, TensorSummary, DEFAULT_ALIGNMENT};

use std::io::Cursor;
use std::path::Path;

/// A loaded GGUF file with memory-mapped tensor data.
///
/// This is the main entry point for working with GGUF model files.
/// It provides access to:
/// - File header (version, counts)
/// - Metadata key-value pairs
/// - Tensor information (names, shapes, types)
/// - Tensor data (zero-copy via memory mapping)
#[derive(Debug)]
pub struct GgufLoader {
    /// Memory-mapped file contents.
    mmap: MappedFile,

    /// Parsed file header.
    header: GgufHeader,

    /// Parsed metadata key-value pairs.
    metadata: Metadata,

    /// Parsed tensor information.
    tensors: TensorInfos,
}

impl GgufLoader {
    /// Opens a GGUF file and parses its structure.
    ///
    /// This memory-maps the file and parses the header, metadata, and
    /// tensor information. Tensor data is not loaded until accessed.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened or mapped
    /// - File format is invalid (bad magic, version, etc.)
    /// - Metadata or tensor info parsing fails
    ///
    /// # Example
    ///
    /// ```no_run
    /// use llm_engine::gguf::GgufLoader;
    ///
    /// let loader = GgufLoader::open("model.gguf")?;
    /// println!("Loaded model with {} tensors", loader.header().tensor_count());
    /// # Ok::<(), llm_engine::gguf::GgufError>(())
    /// ```
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let mmap = MappedFile::open(&path).map_err(|e| GgufError::Io {
            message: e.to_string(),
        })?;

        // Parse from memory-mapped data
        Self::from_mapped(mmap)
    }

    /// Creates a loader from an already memory-mapped file.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn from_mapped(mmap: MappedFile) -> Result<Self> {
        let data = mmap.as_slice();

        // Create a cursor for reading
        let mut cursor = Cursor::new(data);

        // Parse header
        let header = GgufHeader::read(&mut cursor)?;

        // Parse metadata
        let metadata = Metadata::read(&mut cursor, header.metadata_kv_count())?;

        // Parse tensor infos
        let mut tensors = TensorInfos::read(&mut cursor, header.tensor_count())?;

        // Calculate data section offset (current position + alignment)
        let current_pos = cursor.position();
        let alignment = Self::get_alignment(&metadata);
        let data_offset = align_offset(current_pos, alignment as u64);
        tensors.set_data_offset(data_offset);

        Ok(Self {
            mmap,
            header,
            metadata,
            tensors,
        })
    }

    /// Opens a GGUF file using buffered I/O instead of memory mapping.
    ///
    /// This is useful for systems where memory mapping is not available
    /// or for very small files where mapping overhead isn't worth it.
    ///
    /// Note: This only parses metadata; tensor data access requires
    /// separate file reads.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or parsed.
    pub fn open_buffered<P: AsRef<Path>>(path: P) -> Result<Self> {
        // For buffered mode, we still memory-map but could add a non-mmap path
        Self::open(path)
    }

    /// Returns the file header.
    #[must_use]
    pub const fn header(&self) -> &GgufHeader {
        &self.header
    }

    /// Returns the metadata collection.
    #[must_use]
    pub const fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Returns the tensor information collection.
    #[must_use]
    pub const fn tensors(&self) -> &TensorInfos {
        &self.tensors
    }

    /// Returns the total file size in bytes.
    #[must_use]
    pub fn file_size(&self) -> usize {
        self.mmap.len()
    }

    /// Returns the file path.
    #[must_use]
    pub fn path(&self) -> &str {
        self.mmap.path()
    }

    /// Gets raw tensor data by name.
    ///
    /// Returns a slice of the memory-mapped data for the tensor.
    /// The slice lifetime is tied to the loader.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name (e.g., "model.layers.0.attention.wq.weight")
    ///
    /// # Returns
    ///
    /// `Some(MappedSlice)` if the tensor exists, `None` otherwise.
    #[must_use]
    pub fn tensor_data(&self, name: &str) -> Option<MappedSlice<'_>> {
        let info = self.tensors.get(name)?;
        self.tensor_data_for(info)
    }

    /// Gets raw tensor data for a TensorInfo.
    ///
    /// # Returns
    ///
    /// `Some(MappedSlice)` if the data is within bounds, `None` otherwise.
    #[must_use]
    pub fn tensor_data_for(&self, info: &TensorInfo) -> Option<MappedSlice<'_>> {
        let offset = self.tensors.absolute_offset(info) as usize;
        let size = info.size_bytes();

        self.mmap.try_slice(offset, size).map(MappedSlice::new)
    }

    /// Gets tensor data as f32 slice (only valid for F32 tensors).
    ///
    /// # Returns
    ///
    /// `Some(&[f32])` if the tensor exists and is F32 type, `None` otherwise.
    #[must_use]
    pub fn tensor_f32(&self, name: &str) -> Option<&[f32]> {
        let info = self.tensors.get(name)?;

        if info.dtype() != GgmlType::F32 {
            return None;
        }

        let slice = self.tensor_data_for(info)?;
        slice.as_f32()
    }

    /// Gets tensor data as f16 bits (u16 slice, only valid for F16 tensors).
    ///
    /// The returned u16 values are raw f16 bit patterns.
    /// Use `f16_to_f32` to convert individual values.
    ///
    /// # Returns
    ///
    /// `Some(&[u16])` if the tensor exists and is F16 type, `None` otherwise.
    #[must_use]
    pub fn tensor_f16_bits(&self, name: &str) -> Option<&[u16]> {
        let info = self.tensors.get(name)?;

        if info.dtype() != GgmlType::F16 {
            return None;
        }

        let slice = self.tensor_data_for(info)?;
        slice.as_f16_bits()
    }

    /// Computes a summary of all tensors.
    #[must_use]
    pub fn summary(&self) -> TensorSummary {
        TensorSummary::from_tensors(&self.tensors)
    }

    /// Returns the alignment used for tensor data.
    #[must_use]
    pub fn alignment(&self) -> usize {
        Self::get_alignment(&self.metadata)
    }

    /// Exposes the raw memory-map bytes for offset calculations.
    ///
    /// Primarily used by `model::mmap_weights::WeightAccessor` to issue
    /// per-region OS page hints.
    #[must_use]
    pub fn mmap_data(&self) -> &[u8] {
        self.mmap.as_slice()
    }

    /// Advises the OS to use sequential read-ahead for the full mmap.
    ///
    /// Useful during prefill when weights are accessed layer by layer.
    ///
    /// # Errors
    ///
    /// Returns an error if the OS advise call fails (rare; treated as a hint).
    pub fn mmap_advise_sequential(&self) -> std::io::Result<()> {
        self.mmap.advise_sequential()
    }

    /// Advises the OS to disable read-ahead for the full mmap.
    ///
    /// Useful during decode when weights are accessed in small random jumps.
    ///
    /// # Errors
    ///
    /// Same as [`mmap_advise_sequential`](Self::mmap_advise_sequential).
    pub fn mmap_advise_random(&self) -> std::io::Result<()> {
        self.mmap.advise_random()
    }

    /// Issues a per-region `WillNeed` or `DontNeed` hint for a byte range.
    ///
    /// Pass `willneed = true` to prefetch `[offset, offset+len)`, or
    /// `false` to release those pages under memory pressure.
    ///
    /// # Errors
    ///
    /// Same as [`mmap_advise_sequential`](Self::mmap_advise_sequential).
    pub fn mmap_advise_region(
        &self,
        willneed: bool,
        offset: usize,
        len: usize,
    ) -> std::io::Result<()> {
        self.mmap.advise_range(willneed, offset, len)
    }

    /// Extracts alignment from metadata or uses default.
    fn get_alignment(metadata: &Metadata) -> usize {
        metadata
            .get_u32("general.alignment")
            .map(|v| v as usize)
            .unwrap_or(DEFAULT_ALIGNMENT)
    }

    /// Validates that tensor data offsets are within file bounds.
    ///
    /// # Errors
    ///
    /// Returns an error if any tensor's data extends beyond the file.
    pub fn validate(&self) -> Result<()> {
        let file_size = self.mmap.len() as u64;

        for tensor in self.tensors.iter() {
            let start = self.tensors.absolute_offset(tensor);
            let end = start + tensor.size_bytes() as u64;

            if end > file_size {
                return Err(GgufError::ValueOutOfRange {
                    field: "tensor data",
                    value: end,
                });
            }
        }

        Ok(())
    }

    /// Advises the OS that tensor data will be accessed sequentially.
    ///
    /// This can improve performance when iterating through all tensors.
    ///
    /// # Errors
    ///
    /// Returns an error if the advise call fails.
    pub fn advise_sequential(&self) -> std::io::Result<()> {
        self.mmap.advise_sequential()
    }

    /// Advises the OS that tensor data will be accessed randomly.
    ///
    /// # Errors
    ///
    /// Returns an error if the advise call fails.
    pub fn advise_random(&self) -> std::io::Result<()> {
        self.mmap.advise_random()
    }

    /// Prefetches tensor data into memory.
    ///
    /// This advises the OS that the data will be needed soon.
    ///
    /// # Errors
    ///
    /// Returns an error if the advise call fails.
    pub fn prefetch(&self) -> std::io::Result<()> {
        self.mmap.advise_willneed()
    }
}

/// Builder for loading GGUF files with custom options.
#[derive(Debug, Default)]
pub struct GgufLoaderBuilder {
    /// Whether to validate tensor offsets after loading.
    validate: bool,

    /// Whether to prefetch tensor data.
    prefetch: bool,

    /// Expected architecture (for validation).
    expected_arch: Option<String>,
}

impl GgufLoaderBuilder {
    /// Creates a new builder with default options.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enables validation of tensor data offsets.
    #[must_use]
    pub const fn validate(mut self, validate: bool) -> Self {
        self.validate = validate;
        self
    }

    /// Enables prefetching of tensor data.
    #[must_use]
    pub const fn prefetch(mut self, prefetch: bool) -> Self {
        self.prefetch = prefetch;
        self
    }

    /// Sets expected architecture for validation.
    #[must_use]
    pub fn expected_arch(mut self, arch: impl Into<String>) -> Self {
        self.expected_arch = Some(arch.into());
        self
    }

    /// Opens a GGUF file with the configured options.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - File cannot be opened
    /// - Validation fails (if enabled)
    /// - Architecture doesn't match (if expected_arch set)
    pub fn open<P: AsRef<Path>>(self, path: P) -> Result<GgufLoader> {
        let loader = GgufLoader::open(path)?;

        // Validate if requested
        if self.validate {
            loader.validate()?;
        }

        // Check architecture if expected
        if let Some(expected) = &self.expected_arch {
            let actual = loader.metadata().get_str("general.architecture");
            if actual != Some(expected.as_str()) {
                return Err(GgufError::TypeMismatch {
                    expected: expected.clone(),
                    got: actual.unwrap_or("(none)").to_string(),
                });
            }
        }

        // Prefetch if requested
        if self.prefetch {
            let _ = loader.prefetch(); // Ignore errors (it's just a hint)
        }

        Ok(loader)
    }
}

/// Quickly inspect a GGUF file without fully loading it.
///
/// This is useful for listing models or checking compatibility.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or parsed.
pub fn inspect<P: AsRef<Path>>(path: P) -> Result<GgufInspection> {
    let loader = GgufLoader::open(path)?;

    let architecture = loader
        .metadata()
        .get_str("general.architecture")
        .map(String::from);

    let name = loader.metadata().get_str("general.name").map(String::from);

    let summary = loader.summary();

    Ok(GgufInspection {
        path: loader.path().to_string(),
        version: loader.header().version(),
        architecture,
        name,
        tensor_count: loader.tensors().len(),
        total_params: summary.total_params,
        total_bytes: summary.total_bytes,
        num_layers: summary.num_layers,
    })
}

/// Quick inspection results for a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufInspection {
    /// File path.
    pub path: String,
    /// GGUF format version.
    pub version: u32,
    /// Model architecture (e.g., "llama").
    pub architecture: Option<String>,
    /// Model name.
    pub name: Option<String>,
    /// Number of tensors.
    pub tensor_count: usize,
    /// Total parameter count.
    pub total_params: u64,
    /// Total size in bytes.
    pub total_bytes: usize,
    /// Number of layers.
    pub num_layers: usize,
}

impl std::fmt::Display for GgufInspection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "GGUF File: {}", self.path)?;
        writeln!(f, "  Version: {}", self.version)?;

        if let Some(arch) = &self.architecture {
            writeln!(f, "  Architecture: {arch}")?;
        }
        if let Some(name) = &self.name {
            writeln!(f, "  Name: {name}")?;
        }

        writeln!(f, "  Tensors: {}", self.tensor_count)?;
        writeln!(f, "  Parameters: {}", format_params(self.total_params))?;
        writeln!(f, "  Size: {}", format_bytes(self.total_bytes))?;
        writeln!(f, "  Layers: {}", self.num_layers)?;

        Ok(())
    }
}

/// Formats a byte count in human-readable form.
fn format_bytes(bytes: usize) -> String {
    let bytes = bytes as f64;
    if bytes >= 1e9 {
        format!("{:.2} GB", bytes / 1e9)
    } else if bytes >= 1e6 {
        format!("{:.2} MB", bytes / 1e6)
    } else if bytes >= 1e3 {
        format!("{:.2} KB", bytes / 1e3)
    } else {
        format!("{bytes} B")
    }
}

/// Formats a parameter count in human-readable form.
fn format_params(params: u64) -> String {
    let params = params as f64;
    if params >= 1e9 {
        format!("{:.2}B", params / 1e9)
    } else if params >= 1e6 {
        format!("{:.2}M", params / 1e6)
    } else if params >= 1e3 {
        format!("{:.2}K", params / 1e3)
    } else {
        format!("{params}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Helper to write a length-prefixed string.
    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Creates a minimal valid GGUF file for testing.
    fn create_test_gguf() -> Vec<u8> {
        let mut data = Vec::new();

        // Header
        data.extend_from_slice(b"GGUF"); // Magic
        data.extend_from_slice(&3u32.to_le_bytes()); // Version
        data.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
        data.extend_from_slice(&2u64.to_le_bytes()); // 2 metadata entries

        // Metadata 1: general.architecture = "llama"
        write_string(&mut data, "general.architecture");
        data.extend_from_slice(&8u32.to_le_bytes()); // Type: String
        write_string(&mut data, "llama");

        // Metadata 2: llama.block_count = 32
        write_string(&mut data, "llama.block_count");
        data.extend_from_slice(&4u32.to_le_bytes()); // Type: Uint32
        data.extend_from_slice(&32u32.to_le_bytes());

        // Tensor info
        write_string(&mut data, "test.weight");
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims = 1
        data.extend_from_slice(&32u64.to_le_bytes()); // dim = 32
        data.extend_from_slice(&0u32.to_le_bytes()); // type = F32
        data.extend_from_slice(&0u64.to_le_bytes()); // offset = 0

        // Alignment padding to 32 bytes
        let current_len = data.len();
        let aligned = ((current_len + 31) / 32) * 32;
        data.resize(aligned, 0);

        // Tensor data: 32 f32 values
        for i in 0..32u32 {
            data.extend_from_slice(&(i as f32).to_le_bytes());
        }

        data
    }

    fn create_test_file(data: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("create temp file");
        file.write_all(data).expect("write data");
        file.flush().expect("flush");
        file
    }

    #[test]
    fn test_loader_open() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let loader = GgufLoader::open(temp.path()).expect("open loader");

        assert_eq!(loader.header().version(), 3);
        assert_eq!(loader.header().tensor_count(), 1);
        assert_eq!(loader.header().metadata_kv_count(), 2);
    }

    #[test]
    fn test_loader_metadata() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let loader = GgufLoader::open(temp.path()).expect("open loader");

        assert_eq!(
            loader.metadata().get_str("general.architecture"),
            Some("llama")
        );
        assert_eq!(loader.metadata().get_u32("llama.block_count"), Some(32));
    }

    #[test]
    fn test_loader_tensors() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let loader = GgufLoader::open(temp.path()).expect("open loader");

        assert_eq!(loader.tensors().len(), 1);

        let tensor = loader.tensors().get("test.weight").expect("find tensor");
        assert_eq!(tensor.name(), "test.weight");
        assert_eq!(tensor.dims(), &[32]);
        assert_eq!(tensor.dtype(), GgmlType::F32);
    }

    #[test]
    fn test_loader_tensor_data() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let loader = GgufLoader::open(temp.path()).expect("open loader");

        let tensor_data = loader.tensor_data("test.weight").expect("get data");
        assert_eq!(tensor_data.len(), 32 * 4); // 32 f32 values
    }

    #[test]
    fn test_loader_tensor_f32() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let loader = GgufLoader::open(temp.path()).expect("open loader");

        let f32_data = loader.tensor_f32("test.weight").expect("get f32 data");
        assert_eq!(f32_data.len(), 32);
        assert!((f32_data[0] - 0.0).abs() < 1e-6);
        assert!((f32_data[1] - 1.0).abs() < 1e-6);
        assert!((f32_data[31] - 31.0).abs() < 1e-6);
    }

    #[test]
    fn test_loader_validate() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let loader = GgufLoader::open(temp.path()).expect("open loader");

        // Should pass validation
        assert!(loader.validate().is_ok());
    }

    #[test]
    fn test_loader_summary() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let loader = GgufLoader::open(temp.path()).expect("open loader");
        let summary = loader.summary();

        assert_eq!(summary.count, 1);
        assert_eq!(summary.total_params, 32);
        assert_eq!(summary.total_bytes, 128); // 32 * 4 bytes
    }

    #[test]
    fn test_builder_validate() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let loader = GgufLoaderBuilder::new()
            .validate(true)
            .open(temp.path())
            .expect("open with validation");

        assert_eq!(loader.header().version(), 3);
    }

    #[test]
    fn test_builder_expected_arch() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        // Correct architecture should succeed
        let result = GgufLoaderBuilder::new()
            .expected_arch("llama")
            .open(temp.path());
        assert!(result.is_ok());

        // Wrong architecture should fail
        let result = GgufLoaderBuilder::new()
            .expected_arch("gpt2")
            .open(temp.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_inspect() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let inspection = inspect(temp.path()).expect("inspect");

        assert_eq!(inspection.version, 3);
        assert_eq!(inspection.architecture, Some("llama".to_string()));
        assert_eq!(inspection.tensor_count, 1);
        assert_eq!(inspection.total_params, 32);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1500), "1.50 KB");
        assert_eq!(format_bytes(1_500_000), "1.50 MB");
        assert_eq!(format_bytes(1_500_000_000), "1.50 GB");
    }

    #[test]
    fn test_format_params() {
        assert_eq!(format_params(500), "500");
        assert_eq!(format_params(7_000_000_000), "7.00B");
        assert_eq!(format_params(125_000_000), "125.00M");
    }

    #[test]
    fn test_alignment_default() {
        let data = create_test_gguf();
        let temp = create_test_file(&data);

        let loader = GgufLoader::open(temp.path()).expect("open loader");

        // Without explicit alignment in metadata, should use default
        assert_eq!(loader.alignment(), DEFAULT_ALIGNMENT);
    }
}