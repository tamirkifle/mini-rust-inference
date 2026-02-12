//! KV-cache data structure — commit 9.1.
//!
//! # Purpose
//!
//! During autoregressive generation each new token needs to attend over every
//! previously computed key and value.  Recomputing those projections from
//! scratch on every step wastes O(n²) total compute.  A KV-cache stores the
//! K and V projections for every past position so they can be looked up in O(1).
//!
//! # Layout
//!
//! For each layer, K and V are stored as flat `Vec<f32>` buffers of length
//! `max_seq_len * kv_dim`, laid out row-major:
//!
//! ```text
//! k_data[layer][pos * kv_dim .. (pos+1) * kv_dim]  =  K row for position `pos`
//! ```
//!
//! Pre-allocating for `max_seq_len` positions means generation never heap-allocates
//! after construction.
//!
//! # Usage
//!
//! ```text
//! // Prefill: write all prompt K/V rows, then run full causal attention
//! for pos in 0..seq_len {
//!     cache.write_k(layer, pos, &k_slice[pos * kv_dim..(pos+1)*kv_dim])?;
//!     cache.write_v(layer, pos, &v_slice[pos * kv_dim..(pos+1)*kv_dim])?;
//! }
//!
//! // Decode: write new token, read back full context
//! cache.write_k(layer, pos, new_k_row)?;
//! cache.write_v(layer, pos, new_v_row)?;
//! let k_ctx = cache.read_k(layer, pos + 1)?;  // [pos+1, kv_dim]
//! let v_ctx = cache.read_v(layer, pos + 1)?;
//! ```

use crate::tensor::{Result, Tensor, TensorError};

/// Per-layer key-value cache for transformer generation.
///
/// Pre-allocates `n_layers × max_seq_len × kv_dim` f32 storage at construction.
/// Writes and reads are O(kv_dim) copies — no heap allocation during generation.
pub struct KvCache {
    /// K buffer: `k_data[layer]` has length `max_seq_len * kv_dim`.
    k_data: Vec<Vec<f32>>,
    /// V buffer: same layout as `k_data`.
    v_data: Vec<Vec<f32>>,
    n_layers: usize,
    max_seq_len: usize,
    /// `n_kv_heads * head_dim` — the width of each K/V row.
    kv_dim: usize,
}

impl KvCache {
    /// Allocate a new cache for `n_layers` transformer blocks.
    ///
    /// Each layer pre-allocates `max_seq_len × kv_dim` f32 values,
    /// where `kv_dim = n_kv_heads * head_dim`.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero (checked via debug assertion).
    #[must_use]
    pub fn new(
        n_layers:    usize,
        max_seq_len: usize,
        n_kv_heads:  usize,
        head_dim:    usize,
    ) -> Self {
        debug_assert!(n_layers > 0,    "KvCache: n_layers must be > 0");
        debug_assert!(max_seq_len > 0, "KvCache: max_seq_len must be > 0");
        debug_assert!(n_kv_heads > 0,  "KvCache: n_kv_heads must be > 0");
        debug_assert!(head_dim > 0,    "KvCache: head_dim must be > 0");

        let kv_dim     = n_kv_heads * head_dim;
        let layer_size = max_seq_len * kv_dim;
        Self {
            k_data:      vec![vec![0.0_f32; layer_size]; n_layers],
            v_data:      vec![vec![0.0_f32; layer_size]; n_layers],
            n_layers,
            max_seq_len,
            kv_dim,
        }
    }

    // ── write ──────────────────────────────────────────────────────────────

    /// Write one position's K row for `layer` at cache position `pos`.
    ///
    /// `row` must have exactly `kv_dim` elements.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if `layer`, `pos`, or `row.len()` are out
    /// of bounds.
    pub fn write_k(&mut self, layer: usize, pos: usize, row: &[f32]) -> Result<()> {
        self.bounds_check(layer, pos, row.len())?;
        let start = pos * self.kv_dim;
        self.k_data[layer][start..start + self.kv_dim].copy_from_slice(row);
        Ok(())
    }

    /// Write one position's V row for `layer` at cache position `pos`.
    ///
    /// `row` must have exactly `kv_dim` elements.
    ///
    /// # Errors
    ///
    /// Same as [`write_k`].
    pub fn write_v(&mut self, layer: usize, pos: usize, row: &[f32]) -> Result<()> {
        self.bounds_check(layer, pos, row.len())?;
        let start = pos * self.kv_dim;
        self.v_data[layer][start..start + self.kv_dim].copy_from_slice(row);
        Ok(())
    }

    // ── read ───────────────────────────────────────────────────────────────

    /// Return cached K for `layer` covering positions `0..seq_len` as a
    /// `[seq_len, kv_dim]` tensor.
    ///
    /// This copies the slice into a new allocation.  For single-token decode
    /// that's `kv_dim` floats per layer — cheap relative to the matmul cost.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if `layer >= n_layers` or
    /// `seq_len > max_seq_len`.
    pub fn read_k(&self, layer: usize, seq_len: usize) -> Result<Tensor<f32>> {
        self.read_check(layer, seq_len)?;
        let data = self.k_data[layer][..seq_len * self.kv_dim].to_vec();
        Ok(Tensor::from_vec(data, vec![seq_len, self.kv_dim])?)
    }

    /// Return cached V for `layer` covering positions `0..seq_len` as a
    /// `[seq_len, kv_dim]` tensor.
    ///
    /// # Errors
    ///
    /// Same as [`read_k`].
    pub fn read_v(&self, layer: usize, seq_len: usize) -> Result<Tensor<f32>> {
        self.read_check(layer, seq_len)?;
        let data = self.v_data[layer][..seq_len * self.kv_dim].to_vec();
        Ok(Tensor::from_vec(data, vec![seq_len, self.kv_dim])?)
    }

    // ── accessors ─────────────────────────────────────────────────────────

    /// Number of transformer layers this cache was built for.
    #[must_use]
    pub fn n_layers(&self) -> usize { self.n_layers }

    /// Maximum number of sequence positions this cache can hold.
    #[must_use]
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }

    /// Width of each K/V row: `n_kv_heads * head_dim`.
    #[must_use]
    pub fn kv_dim(&self) -> usize { self.kv_dim }

    // ── bulk operations ────────────────────────────────────────────────────

    /// Zero out all cached K and V data, retaining the allocated memory.
    ///
    /// Call this before processing a new, unrelated sequence.
    pub fn clear(&mut self) {
        for layer in 0..self.n_layers {
            self.k_data[layer].fill(0.0_f32);
            self.v_data[layer].fill(0.0_f32);
        }
    }

    /// Zero out cached data for positions `from_pos..until_pos` across all layers.
    ///
    /// Used by `cache_truncate` to invalidate evicted suffix positions so stale
    /// values cannot leak into future reads.
    pub(crate) fn zero_range(&mut self, from_pos: usize, until_pos: usize) {
        let kv_dim = self.kv_dim;
        for layer in 0..self.n_layers {
            let start = from_pos * kv_dim;
            let end   = until_pos * kv_dim;
            if end <= self.k_data[layer].len() {
                self.k_data[layer][start..end].fill(0.0_f32);
                self.v_data[layer][start..end].fill(0.0_f32);
            }
        }
    }
}

// ── private helpers ───────────────────────────────────────────────────────────

impl KvCache {
    fn bounds_check(&self, layer: usize, pos: usize, row_len: usize) -> Result<()> {
        if layer >= self.n_layers {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "KvCache::write: layer {layer} >= n_layers {}",
                    self.n_layers
                ),
            });
        }
        if pos >= self.max_seq_len {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "KvCache::write: pos {pos} >= max_seq_len {}",
                    self.max_seq_len
                ),
            });
        }
        if row_len != self.kv_dim {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "KvCache::write: row_len {row_len} != kv_dim {}",
                    self.kv_dim
                ),
            });
        }
        Ok(())
    }

    fn read_check(&self, layer: usize, seq_len: usize) -> Result<()> {
        if layer >= self.n_layers {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "KvCache::read: layer {layer} >= n_layers {}",
                    self.n_layers
                ),
            });
        }
        if seq_len == 0 {
            return Err(TensorError::InvalidShape {
                reason: "KvCache::read: seq_len must be > 0".to_string(),
            });
        }
        if seq_len > self.max_seq_len {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "KvCache::read: seq_len {seq_len} > max_seq_len {}",
                    self.max_seq_len
                ),
            });
        }
        Ok(())
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cache(n_layers: usize, max_seq: usize, kv_heads: usize, hd: usize) -> KvCache {
        KvCache::new(n_layers, max_seq, kv_heads, hd)
    }

    // ── construction ─────────────────────────────────────────────────────

    #[test]
    fn test_kvcache_construction_kv_dim() {
        let c = make_cache(4, 128, 8, 64);
        assert_eq!(c.kv_dim(),      8 * 64);
        assert_eq!(c.n_layers(),    4);
        assert_eq!(c.max_seq_len(), 128);
    }

    #[test]
    fn test_kvcache_initial_data_is_zero() {
        let c = make_cache(2, 4, 2, 4);
        let k = c.read_k(0, 4).unwrap();
        assert!(k.as_slice().iter().all(|&v| v == 0.0));
    }

    // ── write / read roundtrip ────────────────────────────────────────────

    #[test]
    fn test_kvcache_write_read_k_roundtrip() {
        let mut c = make_cache(1, 8, 2, 4);   // kv_dim = 8
        let row = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        c.write_k(0, 3, &row).unwrap();
        let k = c.read_k(0, 4).unwrap();      // read 4 rows → [4, 8]
        assert_eq!(k.dims(), &[4, 8]);
        assert_eq!(&k.as_slice()[24..32], row.as_slice()); // row 3 starts at offset 3*8=24
    }

    #[test]
    fn test_kvcache_write_read_v_roundtrip() {
        let mut c = make_cache(1, 8, 2, 4);
        let row = vec![-1.0_f32; 8];
        c.write_v(0, 0, &row).unwrap();
        let v = c.read_v(0, 1).unwrap();
        assert_eq!(v.dims(), &[1, 8]);
        assert!(v.as_slice().iter().all(|&x| (x + 1.0).abs() < 1e-7));
    }

    #[test]
    fn test_kvcache_multiple_layers_isolated() {
        let mut c = make_cache(3, 4, 1, 4);    // kv_dim = 4
        let row0 = vec![1.0_f32, 1.0, 1.0, 1.0];
        let row2 = vec![2.0_f32, 2.0, 2.0, 2.0];
        c.write_k(0, 0, &row0).unwrap();
        c.write_k(2, 0, &row2).unwrap();
        // Layer 1 should still be zero
        let k1 = c.read_k(1, 1).unwrap();
        assert!(k1.as_slice().iter().all(|&v| v == 0.0));
        // Layer 0 has row0
        let k0 = c.read_k(0, 1).unwrap();
        assert_eq!(k0.as_slice(), row0.as_slice());
        // Layer 2 has row2
        let k2 = c.read_k(2, 1).unwrap();
        assert_eq!(k2.as_slice(), row2.as_slice());
    }

    #[test]
    fn test_kvcache_sequential_writes_accumulate() {
        let mut c = make_cache(1, 4, 1, 2);   // kv_dim = 2
        for pos in 0..4_usize {
            c.write_k(0, pos, &[pos as f32, pos as f32 * 10.0]).unwrap();
        }
        let k = c.read_k(0, 4).unwrap();
        assert_eq!(k.dims(), &[4, 2]);
        for pos in 0..4_usize {
            assert!((k.as_slice()[pos * 2]      - pos as f32).abs()        < 1e-7);
            assert!((k.as_slice()[pos * 2 + 1]  - pos as f32 * 10.0).abs() < 1e-7);
        }
    }

    // ── clear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_kvcache_clear_zeros_all_data() {
        let mut c = make_cache(2, 4, 1, 4);
        let row = vec![5.0_f32; 4];
        c.write_k(0, 0, &row).unwrap();
        c.write_v(1, 2, &row).unwrap();
        c.clear();
        let k = c.read_k(0, 4).unwrap();
        let v = c.read_v(1, 4).unwrap();
        assert!(k.as_slice().iter().all(|&x| x == 0.0));
        assert!(v.as_slice().iter().all(|&x| x == 0.0));
    }

    // ── error cases ───────────────────────────────────────────────────────

    #[test]
    fn test_kvcache_write_layer_out_of_bounds() {
        let mut c = make_cache(2, 4, 1, 4);
        let row = vec![0.0_f32; 4];
        assert!(matches!(
            c.write_k(2, 0, &row),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_kvcache_write_pos_out_of_bounds() {
        let mut c = make_cache(1, 4, 1, 4);
        let row = vec![0.0_f32; 4];
        assert!(matches!(
            c.write_k(0, 4, &row),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_kvcache_write_row_wrong_length() {
        let mut c = make_cache(1, 4, 1, 4);
        let row = vec![0.0_f32; 3]; // kv_dim is 4, not 3
        assert!(matches!(
            c.write_k(0, 0, &row),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_kvcache_read_seq_exceeds_max() {
        let c = make_cache(1, 4, 1, 4);
        assert!(matches!(
            c.read_k(0, 5),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_kvcache_read_zero_seq_len_rejected() {
        let c = make_cache(1, 4, 1, 4);
        assert!(matches!(
            c.read_k(0, 0),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
