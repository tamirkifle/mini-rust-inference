//! GGUF tensor information parsing.
//!
//! Each tensor in a GGUF file has associated metadata:
//! - Name: UTF-8 string identifying the tensor (e.g., "model.layers.0.attention.wq.weight")
//! - Shape: N-dimensional array of sizes
//! - Type: Data type (F32, F16, quantized types)
//! - Offset: Byte offset from the start of the tensor data section
//!
//! # File Layout
//!
//! ```text
//! ┌──────────────────────────────────┐
//! │            Header                │
//! ├──────────────────────────────────┤
//! │         Metadata KV              │
//! ├──────────────────────────────────┤
//! │     Tensor Info Array            │  ← This module parses this
//! │  ┌────────────────────────────┐  │
//! │  │ name, ndims, dims[], type, │  │
//! │  │ offset                     │  │
//! │  └────────────────────────────┘  │
//! │           ... × n_tensors        │
//! ├──────────────────────────────────┤
//! │  Alignment Padding (to 32 bytes) │
//! ├──────────────────────────────────┤
//! │         Tensor Data              │  ← Offsets are relative to here
//! │  (raw bytes, possibly quantized) │
//! └──────────────────────────────────┘
//! ```
//!
//! Reference: <https://github.com/ggerganov/ggml/blob/master/docs/gguf.md>

use super::dtype::GgmlType;
use super::error::{GgufError, Result};
use super::header::{read_string, read_u32, read_u64};
use std::collections::HashMap;
use std::io::Read;

/// Default alignment for tensor data in GGUF files.
pub const DEFAULT_ALIGNMENT: usize = 32;

/// Information about a single tensor in a GGUF file.
///
/// This struct contains all metadata needed to locate and interpret
/// tensor data within the file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    /// Tensor name (e.g., "model.layers.0.attention.wq.weight").
    name: String,

    /// Number of dimensions.
    n_dims: u32,

    /// Size of each dimension.
    /// Stored in GGUF order (which may differ from row-major).
    dims: Vec<u64>,

    /// Data type (F32, F16, Q4_0, etc.).
    dtype: GgmlType,

    /// Byte offset from the start of the tensor data section.
    offset: u64,
}

impl TensorInfo {
    /// Reads tensor info from a reader.
    ///
    /// # Errors
    ///
    /// Returns error if reading fails or data is invalid.
    pub fn read<R: Read>(reader: &mut R) -> Result<Self> {
        // Name (length-prefixed string)
        let name = read_string(reader, "tensor name")?;

        // Number of dimensions
        let n_dims = read_u32(reader, "tensor n_dims")?;

        // Validate n_dims (reasonable limit)
        if n_dims > 8 {
            return Err(GgufError::ValueOutOfRange {
                field: "tensor n_dims",
                value: u64::from(n_dims),
            });
        }

        // Dimensions
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let dim = read_u64(reader, "tensor dimension")?;
            dims.push(dim);
        }

        // Data type
        let type_id = read_u32(reader, "tensor type")?;
        let dtype = GgmlType::from_u32(type_id)?;

        // Offset (relative to tensor data section)
        let offset = read_u64(reader, "tensor offset")?;

        Ok(Self {
            name,
            n_dims,
            dims,
            dtype,
            offset,
        })
    }

    /// Returns the tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub const fn n_dims(&self) -> u32 {
        self.n_dims
    }

    /// Returns the dimensions as a slice.
    #[must_use]
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Returns the dimensions as usize slice (for tensor creation).
    ///
    /// # Panics
    ///
    /// Panics if any dimension exceeds `usize::MAX` (unlikely on 64-bit).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn dims_usize(&self) -> Vec<usize> {
        self.dims.iter().map(|&d| d as usize).collect()
    }

    /// Returns the shape in row-major order (reversed from GGUF storage).
    ///
    /// GGUF stores dimensions in column-major order, but most frameworks
    /// expect row-major. This method returns dimensions suitable for
    /// creating row-major tensors.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn shape_row_major(&self) -> Vec<usize> {
        self.dims.iter().rev().map(|&d| d as usize).collect()
    }

    /// Returns the data type.
    #[must_use]
    pub const fn dtype(&self) -> GgmlType {
        self.dtype
    }

    /// Returns the byte offset from the tensor data section start.
    #[must_use]
    pub const fn offset(&self) -> u64 {
        self.offset
    }

    /// Returns the total number of elements.
    #[must_use]
    pub fn numel(&self) -> u64 {
        if self.dims.is_empty() {
            1 // Scalar
        } else {
            self.dims.iter().product()
        }
    }

    /// Returns the size in bytes for this tensor's data.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn size_bytes(&self) -> usize {
        self.dtype.tensor_size(self.numel() as usize)
    }

    /// Checks if this tensor uses a quantized data type.
    #[must_use]
    pub fn is_quantized(&self) -> bool {
        self.dtype.is_quantized()
    }
}

/// Collection of tensor information for all tensors in a GGUF file.
///
/// Provides lookup by name and iteration over tensors.
#[derive(Debug, Clone, Default)]
pub struct TensorInfos {
    /// Tensors in file order.
    tensors: Vec<TensorInfo>,

    /// Index by name for fast lookup.
    name_index: HashMap<String, usize>,

    /// Start of tensor data section (byte offset from file start).
    data_offset: u64,
}

impl TensorInfos {
    /// Creates a new empty tensor info collection.
    #[must_use]
    pub fn new() -> Self {
        Self {
            tensors: Vec::new(),
            name_index: HashMap::new(),
            data_offset: 0,
        }
    }

    /// Reads all tensor infos from a reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - Reader positioned after metadata section
    /// * `count` - Number of tensors to read (from header)
    ///
    /// # Errors
    ///
    /// Returns error if reading fails or duplicate tensor names found.
    #[allow(clippy::cast_possible_truncation)]
    pub fn read<R: Read>(reader: &mut R, count: u64) -> Result<Self> {
        // Sanity check
        if count > 100_000 {
            return Err(GgufError::ValueOutOfRange {
                field: "tensor count",
                value: count,
            });
        }

        let mut tensors = Vec::with_capacity(count as usize);
        let mut name_index = HashMap::with_capacity(count as usize);

        for i in 0..count {
            let info = TensorInfo::read(reader)?;

            // Check for duplicate names
            if name_index.contains_key(&info.name) {
                return Err(GgufError::KeyNotFound {
                    key: format!("duplicate tensor name: {}", info.name),
                });
            }

            name_index.insert(info.name.clone(), i as usize);
            tensors.push(info);
        }

        Ok(Self {
            tensors,
            name_index,
            data_offset: 0,
        })
    }

    /// Sets the tensor data section offset.
    ///
    /// This should be called after computing where tensor data starts
    /// (after header + metadata + tensor infos + alignment padding).
    pub fn set_data_offset(&mut self, offset: u64) {
        self.data_offset = offset;
    }

    /// Returns the tensor data section offset.
    #[must_use]
    pub const fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Returns the number of tensors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Returns true if there are no tensors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Gets tensor info by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.name_index.get(name).map(|&i| &self.tensors[i])
    }

    /// Gets tensor info by index.
    #[must_use]
    pub fn get_by_index(&self, index: usize) -> Option<&TensorInfo> {
        self.tensors.get(index)
    }

    /// Returns an iterator over all tensor infos.
    pub fn iter(&self) -> impl Iterator<Item = &TensorInfo> {
        self.tensors.iter()
    }

    /// Returns an iterator over tensor names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }

    /// Computes the absolute file offset for a tensor's data.
    ///
    /// This is `data_offset + tensor.offset`.
    #[must_use]
    pub fn absolute_offset(&self, tensor: &TensorInfo) -> u64 {
        self.data_offset + tensor.offset()
    }

    /// Returns total size of all tensor data in bytes.
    #[must_use]
    pub fn total_data_size(&self) -> usize {
        self.tensors.iter().map(TensorInfo::size_bytes).sum()
    }

    /// Finds tensors matching a name pattern.
    ///
    /// Simple prefix/suffix matching (not regex).
    pub fn find(&self, pattern: &str) -> Vec<&TensorInfo> {
        self.tensors
            .iter()
            .filter(|t| t.name.contains(pattern))
            .collect()
    }

    /// Groups tensors by layer number.
    ///
    /// Assumes naming convention like "model.layers.N.xxx".
    /// Returns a map from layer number to tensors in that layer.
    #[must_use]
    pub fn group_by_layer(&self) -> HashMap<usize, Vec<&TensorInfo>> {
        let mut groups: HashMap<usize, Vec<&TensorInfo>> = HashMap::new();

        for tensor in &self.tensors {
            if let Some(layer) = extract_layer_number(&tensor.name) {
                groups.entry(layer).or_default().push(tensor);
            }
        }

        groups
    }
}

/// Extracts layer number from tensor name.
///
/// Looks for patterns like "layers.N." or "blk.N.".
fn extract_layer_number(name: &str) -> Option<usize> {
    // Try "layers.N."
    if let Some(start) = name.find("layers.") {
        let rest = &name[start + 7..];
        if let Some(end) = rest.find('.') {
            return rest[..end].parse().ok();
        }
    }

    // Try "blk.N."
    if let Some(start) = name.find("blk.") {
        let rest = &name[start + 4..];
        if let Some(end) = rest.find('.') {
            return rest[..end].parse().ok();
        }
    }

    None
}

/// Computes the aligned offset for tensor data.
///
/// GGUF files align tensor data to `alignment` bytes (default 32).
#[must_use]
pub const fn align_offset(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        offset
    } else {
        offset + (alignment - remainder)
    }
}

/// Computes padding needed to reach alignment.
#[must_use]
pub const fn padding_for_alignment(offset: u64, alignment: u64) -> u64 {
    let remainder = offset % alignment;
    if remainder == 0 {
        0
    } else {
        alignment - remainder
    }
}

/// Summary statistics for a set of tensor infos.
#[derive(Debug, Clone, Default)]
pub struct TensorSummary {
    /// Total number of tensors.
    pub count: usize,
    /// Total number of parameters (elements).
    pub total_params: u64,
    /// Total size in bytes.
    pub total_bytes: usize,
    /// Count by data type.
    pub by_dtype: HashMap<GgmlType, usize>,
    /// Number of layers detected.
    pub num_layers: usize,
}

impl TensorSummary {
    /// Creates a summary from tensor infos.
    #[must_use]
    pub fn from_tensors(tensors: &TensorInfos) -> Self {
        let mut by_dtype: HashMap<GgmlType, usize> = HashMap::new();
        let mut total_params = 0u64;
        let mut total_bytes = 0usize;
        let mut max_layer = 0usize;

        for tensor in tensors.iter() {
            total_params += tensor.numel();
            total_bytes += tensor.size_bytes();
            *by_dtype.entry(tensor.dtype()).or_insert(0) += 1;

            if let Some(layer) = extract_layer_number(tensor.name()) {
                max_layer = max_layer.max(layer + 1);
            }
        }

        Self {
            count: tensors.len(),
            total_params,
            total_bytes,
            by_dtype,
            num_layers: max_layer,
        }
    }

    /// Returns total size in human-readable format.
    #[must_use]
    pub fn size_string(&self) -> String {
        let bytes = self.total_bytes as f64;
        if bytes >= 1e9 {
            format!("{:.2} GB", bytes / 1e9)
        } else if bytes >= 1e6 {
            format!("{:.2} MB", bytes / 1e6)
        } else if bytes >= 1e3 {
            format!("{:.2} KB", bytes / 1e3)
        } else {
            format!("{} B", self.total_bytes)
        }
    }

    /// Returns parameter count in human-readable format.
    #[must_use]
    pub fn params_string(&self) -> String {
        let params = self.total_params as f64;
        if params >= 1e9 {
            format!("{:.2}B", params / 1e9)
        } else if params >= 1e6 {
            format!("{:.2}M", params / 1e6)
        } else if params >= 1e3 {
            format!("{:.2}K", params / 1e3)
        } else {
            format!("{}", self.total_params)
        }
    }
}

impl std::fmt::Display for TensorSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Tensor Summary:")?;
        writeln!(f, "  Tensors: {}", self.count)?;
        writeln!(f, "  Parameters: {}", self.params_string())?;
        writeln!(f, "  Size: {}", self.size_string())?;
        writeln!(f, "  Layers: {}", self.num_layers)?;

        if !self.by_dtype.is_empty() {
            writeln!(f, "  Data types:")?;
            let mut types: Vec<_> = self.by_dtype.iter().collect();
            types.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
            for (dtype, count) in types {
                writeln!(f, "    {}: {}", dtype, count)?;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Helper to write a length-prefixed string.
    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    #[test]
    fn test_read_tensor_info() {
        let mut data = Vec::new();

        // Name
        write_string(&mut data, "model.layers.0.attention.wq.weight");

        // n_dims = 2
        data.extend_from_slice(&2u32.to_le_bytes());

        // dims = [4096, 4096]
        data.extend_from_slice(&4096u64.to_le_bytes());
        data.extend_from_slice(&4096u64.to_le_bytes());

        // type = Q4_0 (2)
        data.extend_from_slice(&2u32.to_le_bytes());

        // offset = 0
        data.extend_from_slice(&0u64.to_le_bytes());

        let mut cursor = Cursor::new(data);
        let info = TensorInfo::read(&mut cursor).unwrap();

        assert_eq!(info.name(), "model.layers.0.attention.wq.weight");
        assert_eq!(info.n_dims(), 2);
        assert_eq!(info.dims(), &[4096, 4096]);
        assert_eq!(info.dtype(), GgmlType::Q4_0);
        assert_eq!(info.offset(), 0);
        assert_eq!(info.numel(), 4096 * 4096);
        assert!(info.is_quantized());
    }

    #[test]
    fn test_tensor_info_size_bytes() {
        let info = TensorInfo {
            name: "test".to_string(),
            n_dims: 2,
            dims: vec![32, 32], // 1024 elements
            dtype: GgmlType::Q4_0,
            offset: 0,
        };

        // Q4_0: 1024 elements / 32 per block * 18 bytes/block = 576 bytes
        assert_eq!(info.size_bytes(), 576);

        let info_f32 = TensorInfo {
            name: "test".to_string(),
            n_dims: 1,
            dims: vec![100],
            dtype: GgmlType::F32,
            offset: 0,
        };

        // F32: 100 elements * 4 bytes = 400 bytes
        assert_eq!(info_f32.size_bytes(), 400);
    }

    #[test]
    fn test_shape_row_major() {
        let info = TensorInfo {
            name: "test".to_string(),
            n_dims: 3,
            dims: vec![4, 32, 128], // GGUF order
            dtype: GgmlType::F32,
            offset: 0,
        };

        // Row-major is reversed
        assert_eq!(info.shape_row_major(), vec![128, 32, 4]);
    }

    #[test]
    fn test_read_tensor_infos() {
        let mut data = Vec::new();

        // Tensor 1
        write_string(&mut data, "weight1");
        data.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&256u64.to_le_bytes()); // dim
        data.extend_from_slice(&0u32.to_le_bytes()); // F32
        data.extend_from_slice(&0u64.to_le_bytes()); // offset

        // Tensor 2
        write_string(&mut data, "weight2");
        data.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        data.extend_from_slice(&64u64.to_le_bytes()); // dim 0
        data.extend_from_slice(&64u64.to_le_bytes()); // dim 1
        data.extend_from_slice(&1u32.to_le_bytes()); // F16
        data.extend_from_slice(&1024u64.to_le_bytes()); // offset

        let mut cursor = Cursor::new(data);
        let infos = TensorInfos::read(&mut cursor, 2).unwrap();

        assert_eq!(infos.len(), 2);
        assert!(infos.get("weight1").is_some());
        assert!(infos.get("weight2").is_some());
        assert!(infos.get("nonexistent").is_none());

        let w1 = infos.get("weight1").unwrap();
        assert_eq!(w1.dtype(), GgmlType::F32);

        let w2 = infos.get("weight2").unwrap();
        assert_eq!(w2.dtype(), GgmlType::F16);
        assert_eq!(w2.offset(), 1024);
    }

    #[test]
    fn test_tensor_infos_iteration() {
        let mut data = Vec::new();

        for i in 0..3 {
            write_string(&mut data, &format!("tensor{i}"));
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&100u64.to_le_bytes());
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&(i as u64 * 400).to_le_bytes());
        }

        let mut cursor = Cursor::new(data);
        let infos = TensorInfos::read(&mut cursor, 3).unwrap();

        let names: Vec<_> = infos.names().collect();
        assert_eq!(names, vec!["tensor0", "tensor1", "tensor2"]);
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32), 0);
        assert_eq!(align_offset(1, 32), 32);
        assert_eq!(align_offset(31, 32), 32);
        assert_eq!(align_offset(32, 32), 32);
        assert_eq!(align_offset(33, 32), 64);
        assert_eq!(align_offset(100, 32), 128);
    }

    #[test]
    fn test_padding_for_alignment() {
        assert_eq!(padding_for_alignment(0, 32), 0);
        assert_eq!(padding_for_alignment(1, 32), 31);
        assert_eq!(padding_for_alignment(31, 32), 1);
        assert_eq!(padding_for_alignment(32, 32), 0);
        assert_eq!(padding_for_alignment(33, 32), 31);
    }

    #[test]
    fn test_extract_layer_number() {
        assert_eq!(extract_layer_number("model.layers.0.attention.wq.weight"), Some(0));
        assert_eq!(extract_layer_number("model.layers.15.mlp.gate.weight"), Some(15));
        assert_eq!(extract_layer_number("blk.7.attn_k.weight"), Some(7));
        assert_eq!(extract_layer_number("token_embd.weight"), None);
        assert_eq!(extract_layer_number("output.weight"), None);
    }

    #[test]
    fn test_find_tensors() {
        let mut data = Vec::new();

        let names = [
            "model.layers.0.attention.wq.weight",
            "model.layers.0.attention.wk.weight",
            "model.layers.1.attention.wq.weight",
            "output.weight",
        ];

        for (i, name) in names.iter().enumerate() {
            write_string(&mut data, name);
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&100u64.to_le_bytes());
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&(i as u64 * 400).to_le_bytes());
        }

        let mut cursor = Cursor::new(data);
        let infos = TensorInfos::read(&mut cursor, 4).unwrap();

        // Find by pattern
        let wq_tensors = infos.find("wq");
        assert_eq!(wq_tensors.len(), 2);

        let layer0_tensors = infos.find("layers.0");
        assert_eq!(layer0_tensors.len(), 2);

        let output_tensors = infos.find("output");
        assert_eq!(output_tensors.len(), 1);
    }

    #[test]
    fn test_group_by_layer() {
        let mut data = Vec::new();

        let names = [
            "model.layers.0.attention.wq.weight",
            "model.layers.0.attention.wk.weight",
            "model.layers.1.attention.wq.weight",
            "model.layers.2.mlp.weight",
            "output.weight",
        ];

        for (i, name) in names.iter().enumerate() {
            write_string(&mut data, name);
            data.extend_from_slice(&1u32.to_le_bytes());
            data.extend_from_slice(&100u64.to_le_bytes());
            data.extend_from_slice(&0u32.to_le_bytes());
            data.extend_from_slice(&(i as u64 * 400).to_le_bytes());
        }

        let mut cursor = Cursor::new(data);
        let infos = TensorInfos::read(&mut cursor, 5).unwrap();

        let groups = infos.group_by_layer();

        assert_eq!(groups.len(), 3); // Layers 0, 1, 2
        assert_eq!(groups.get(&0).map(|v| v.len()), Some(2));
        assert_eq!(groups.get(&1).map(|v| v.len()), Some(1));
        assert_eq!(groups.get(&2).map(|v| v.len()), Some(1));
    }

    #[test]
    fn test_tensor_summary() {
        let mut data = Vec::new();

        // 2 F32 tensors, 1 Q4_0 tensor
        let tensors_data = [
            ("w1", 1u32, 1024u64, 0u32), // F32, 1024 elements
            ("w2", 1u32, 2048u64, 0u32), // F32, 2048 elements
            ("w3", 1u32, 1024u64, 2u32), // Q4_0, 1024 elements
        ];

        for (name, ndims, dim, dtype) in tensors_data {
            write_string(&mut data, name);
            data.extend_from_slice(&ndims.to_le_bytes());
            data.extend_from_slice(&dim.to_le_bytes());
            data.extend_from_slice(&dtype.to_le_bytes());
            data.extend_from_slice(&0u64.to_le_bytes());
        }

        let mut cursor = Cursor::new(data);
        let infos = TensorInfos::read(&mut cursor, 3).unwrap();

        let summary = TensorSummary::from_tensors(&infos);

        assert_eq!(summary.count, 3);
        assert_eq!(summary.total_params, 1024 + 2048 + 1024);
        assert_eq!(summary.by_dtype.get(&GgmlType::F32), Some(&2));
        assert_eq!(summary.by_dtype.get(&GgmlType::Q4_0), Some(&1));
    }

    #[test]
    fn test_absolute_offset() {
        let mut infos = TensorInfos::new();
        infos.set_data_offset(1000);

        let tensor = TensorInfo {
            name: "test".to_string(),
            n_dims: 1,
            dims: vec![100],
            dtype: GgmlType::F32,
            offset: 500,
        };

        assert_eq!(infos.absolute_offset(&tensor), 1500);
    }
}