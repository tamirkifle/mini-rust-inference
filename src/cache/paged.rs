//! Paged KV-cache — commit 11.3.
//!
//! # Motivation
//!
//! The contiguous [`KvCache`] pre-allocates `n_layers × max_seq_len × kv_dim`
//! floats at construction time.  This is simple but has two drawbacks:
//!
//! 1. **Memory waste**: full capacity is reserved even when most sequences are
//!    much shorter than `max_seq_len`.
//! 2. **No sharing**: two requests with a common prefix must duplicate their
//!    K/V rows.  Future batching and prefix-caching want to share pages.
//!
//! `PagedKvCache` replaces the monolithic buffer with a pool of fixed-size
//! **pages**.  Each page holds `page_size` token rows.  Pages are allocated
//! on demand (the first write to a new logical page index triggers an
//! allocation).  Physical memory grows incrementally as the sequence does.
//!
//! # Layout
//!
//! ```text
//! logical position `pos`
//!   page_idx = pos / page_size       ← which page
//!   slot     = pos % page_size       ← which row within that page
//!
//! k_pages[layer][page_idx][slot * kv_dim .. (slot+1) * kv_dim]
//! ```
//!
//! # Compatibility
//!
//! `read_k` / `read_v` assemble pages into a contiguous `[seq_len, kv_dim]`
//! tensor — the same shape and semantics as [`KvCache::read_k`].  All
//! existing attention kernels consume this tensor without modification.
//!
//! # Future work
//!
//! - **Page sharing**: identical prefix pages can be ref-counted and shared
//!   across sequences (enables prefix caching).
//! - **Free lists**: released pages are returned to a pool for reuse.
//! - **Cross-sequence batching**: a scheduler assigns physical pages to
//!   concurrent sequences dynamically.

use crate::tensor::{Result, Tensor, TensorError};

// ── PagedKvCache ─────────────────────────────────────────────────────────────

/// Block-based KV-cache that allocates storage one page at a time.
///
/// Each page holds exactly `page_size` token rows.  Pages are created lazily
/// the first time a position in that page is written, so memory usage tracks
/// actual sequence length rather than maximum capacity.
pub struct PagedKvCache {
    /// `k_pages[layer]` — a growing list of K pages for that layer.
    /// Each page is a flat `Vec<f32>` of length `page_size * kv_dim`.
    k_pages: Vec<Vec<Vec<f32>>>,
    /// `v_pages[layer]` — same layout as `k_pages`.
    v_pages: Vec<Vec<Vec<f32>>>,
    n_layers:  usize,
    /// Number of token rows per page.
    page_size: usize,
    /// `n_kv_heads * head_dim` — width of each K/V row.
    kv_dim:    usize,
    /// Hard limit on pages per layer (= ceil(max_seq_len / page_size)).
    max_pages: usize,
}

impl PagedKvCache {
    /// Create a new paged cache.
    ///
    /// # Arguments
    ///
    /// * `n_layers`    – number of transformer layers.
    /// * `max_seq_len` – maximum sequence length; used to compute `max_pages`.
    /// * `n_kv_heads`  – number of KV attention heads.
    /// * `head_dim`    – dimension per head.
    /// * `page_size`   – tokens per page; must be ≥ 1.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero.
    #[must_use]
    pub fn new(
        n_layers:    usize,
        max_seq_len: usize,
        n_kv_heads:  usize,
        head_dim:    usize,
        page_size:   usize,
    ) -> Self {
        assert!(n_layers > 0,    "PagedKvCache: n_layers must be > 0");
        assert!(max_seq_len > 0, "PagedKvCache: max_seq_len must be > 0");
        assert!(n_kv_heads > 0,  "PagedKvCache: n_kv_heads must be > 0");
        assert!(head_dim > 0,    "PagedKvCache: head_dim must be > 0");
        assert!(page_size > 0,   "PagedKvCache: page_size must be > 0");

        let kv_dim    = n_kv_heads * head_dim;
        // ceil division
        let max_pages = (max_seq_len + page_size - 1) / page_size;

        Self {
            k_pages:   vec![Vec::new(); n_layers],
            v_pages:   vec![Vec::new(); n_layers],
            n_layers,
            page_size,
            kv_dim,
            max_pages,
        }
    }

    // ── write ─────────────────────────────────────────────────────────────

    /// Write one K row at logical position `pos` for `layer`.
    ///
    /// Allocates a new page if `pos` falls in a page not yet created.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if `layer`, `pos`, or `row.len()` are
    /// out of bounds.
    pub fn write_k(&mut self, layer: usize, pos: usize, row: &[f32]) -> Result<()> {
        self.validate_write(layer, pos, row.len())?;
        let (page_idx, slot) = self.location(pos);
        self.ensure_page_k(layer, page_idx);
        let start = slot * self.kv_dim;
        self.k_pages[layer][page_idx][start..start + self.kv_dim].copy_from_slice(row);
        Ok(())
    }

    /// Write one V row at logical position `pos` for `layer`.
    ///
    /// # Errors
    ///
    /// Same as [`write_k`].
    pub fn write_v(&mut self, layer: usize, pos: usize, row: &[f32]) -> Result<()> {
        self.validate_write(layer, pos, row.len())?;
        let (page_idx, slot) = self.location(pos);
        self.ensure_page_v(layer, page_idx);
        let start = slot * self.kv_dim;
        self.v_pages[layer][page_idx][start..start + self.kv_dim].copy_from_slice(row);
        Ok(())
    }

    // ── read ──────────────────────────────────────────────────────────────

    /// Assemble cached K for `layer` positions `0..seq_len` into a contiguous
    /// `[seq_len, kv_dim]` tensor.
    ///
    /// Rows that belong to pages never written are returned as zeros.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if `layer >= n_layers` or
    /// `seq_len > max_seq_len`.
    pub fn read_k(&self, layer: usize, seq_len: usize) -> Result<Tensor<f32>> {
        self.validate_read(layer, seq_len)?;
        Ok(Tensor::from_vec(self.gather_rows(&self.k_pages[layer], seq_len), vec![seq_len, self.kv_dim])?)
    }

    /// Assemble cached V for `layer` positions `0..seq_len`.
    ///
    /// # Errors
    ///
    /// Same as [`read_k`].
    pub fn read_v(&self, layer: usize, seq_len: usize) -> Result<Tensor<f32>> {
        self.validate_read(layer, seq_len)?;
        Ok(Tensor::from_vec(self.gather_rows(&self.v_pages[layer], seq_len), vec![seq_len, self.kv_dim])?)
    }

    // ── accessors ─────────────────────────────────────────────────────────

    /// Number of transformer layers this cache was built for.
    #[must_use]
    pub fn n_layers(&self) -> usize { self.n_layers }

    /// Tokens per page.
    #[must_use]
    pub fn page_size(&self) -> usize { self.page_size }

    /// `n_kv_heads * head_dim`.
    #[must_use]
    pub fn kv_dim(&self) -> usize { self.kv_dim }

    /// Maximum pages per layer (= `ceil(max_seq_len / page_size)`).
    #[must_use]
    pub fn max_pages(&self) -> usize { self.max_pages }

    /// Number of pages currently allocated for K in `layer`.
    #[must_use]
    pub fn allocated_pages_k(&self, layer: usize) -> usize {
        if layer < self.n_layers { self.k_pages[layer].len() } else { 0 }
    }

    /// Number of pages currently allocated for V in `layer`.
    #[must_use]
    pub fn allocated_pages_v(&self, layer: usize) -> usize {
        if layer < self.n_layers { self.v_pages[layer].len() } else { 0 }
    }

    // ── bulk operations ───────────────────────────────────────────────────

    /// Drop all allocated pages across every layer, resetting to an empty cache.
    ///
    /// Cheaper than zeroing — simply truncates the page vecs.
    pub fn clear(&mut self) {
        for layer in 0..self.n_layers {
            self.k_pages[layer].clear();
            self.v_pages[layer].clear();
        }
    }
}

// ── private helpers ───────────────────────────────────────────────────────────

impl PagedKvCache {
    /// Decompose a logical position into `(page_idx, slot)`.
    #[inline]
    fn location(&self, pos: usize) -> (usize, usize) {
        (pos / self.page_size, pos % self.page_size)
    }

    /// Grow K page list for `layer` to include `page_idx`, filling new pages with zeros.
    fn ensure_page_k(&mut self, layer: usize, page_idx: usize) {
        while self.k_pages[layer].len() <= page_idx {
            self.k_pages[layer].push(vec![0.0_f32; self.page_size * self.kv_dim]);
        }
    }

    /// Grow V page list for `layer` to include `page_idx`, filling new pages with zeros.
    fn ensure_page_v(&mut self, layer: usize, page_idx: usize) {
        while self.v_pages[layer].len() <= page_idx {
            self.v_pages[layer].push(vec![0.0_f32; self.page_size * self.kv_dim]);
        }
    }

    /// Gather `seq_len` rows from a page list into a flat Vec.
    ///
    /// Pages that haven't been allocated yet contribute zero rows.
    fn gather_rows(&self, pages: &[Vec<f32>], seq_len: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; seq_len * self.kv_dim];
        for pos in 0..seq_len {
            let (page_idx, slot) = (pos / self.page_size, pos % self.page_size);
            if page_idx < pages.len() {
                let src_start = slot * self.kv_dim;
                let dst_start = pos * self.kv_dim;
                out[dst_start..dst_start + self.kv_dim]
                    .copy_from_slice(&pages[page_idx][src_start..src_start + self.kv_dim]);
            }
            // else: row stays zero (page not yet allocated)
        }
        out
    }

    fn validate_write(&self, layer: usize, pos: usize, row_len: usize) -> Result<()> {
        if layer >= self.n_layers {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "PagedKvCache::write: layer {layer} >= n_layers {}",
                    self.n_layers
                ),
            });
        }
        let page_idx = pos / self.page_size;
        if page_idx >= self.max_pages {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "PagedKvCache::write: pos {pos} maps to page {page_idx} >= max_pages {}",
                    self.max_pages
                ),
            });
        }
        if row_len != self.kv_dim {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "PagedKvCache::write: row_len {row_len} != kv_dim {}",
                    self.kv_dim
                ),
            });
        }
        Ok(())
    }

    fn validate_read(&self, layer: usize, seq_len: usize) -> Result<()> {
        if layer >= self.n_layers {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "PagedKvCache::read: layer {layer} >= n_layers {}",
                    self.n_layers
                ),
            });
        }
        if seq_len == 0 {
            return Err(TensorError::InvalidShape {
                reason: "PagedKvCache::read: seq_len must be > 0".to_string(),
            });
        }
        let required_pages = (seq_len + self.page_size - 1) / self.page_size;
        if required_pages > self.max_pages {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "PagedKvCache::read: seq_len {seq_len} requires {required_pages} pages > max_pages {}",
                    self.max_pages
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
    use crate::cache::KvCache;

    fn make_paged(n_layers: usize, max_seq: usize, kv_heads: usize, hd: usize, ps: usize) -> PagedKvCache {
        PagedKvCache::new(n_layers, max_seq, kv_heads, hd, ps)
    }

    // ── construction ─────────────────────────────────────────────────────

    #[test]
    fn test_construction_dimensions() {
        let c = make_paged(4, 128, 8, 16, 16);
        assert_eq!(c.kv_dim(),    8 * 16);
        assert_eq!(c.n_layers(),  4);
        assert_eq!(c.page_size(), 16);
        // max_pages = ceil(128 / 16) = 8
        assert_eq!(c.max_pages(), 8);
    }

    #[test]
    fn test_no_pages_allocated_on_construction() {
        let c = make_paged(2, 64, 2, 8, 8);
        assert_eq!(c.allocated_pages_k(0), 0);
        assert_eq!(c.allocated_pages_v(0), 0);
    }

    // ── lazy page allocation ─────────────────────────────────────────────

    #[test]
    fn test_write_allocates_pages_lazily() {
        let mut c = make_paged(1, 64, 1, 4, 8); // kv_dim=4, page_size=8
        assert_eq!(c.allocated_pages_k(0), 0);

        // Writing pos=0 (page 0) → 1 page allocated
        c.write_k(0, 0, &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(c.allocated_pages_k(0), 1);

        // Writing pos=7 (still page 0) → still 1 page
        c.write_k(0, 7, &[5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(c.allocated_pages_k(0), 1);

        // Writing pos=8 (page 1) → 2 pages now
        c.write_k(0, 8, &[9.0, 10.0, 11.0, 12.0]).unwrap();
        assert_eq!(c.allocated_pages_k(0), 2);
    }

    // ── write / read roundtrip ────────────────────────────────────────────

    #[test]
    fn test_write_read_k_roundtrip() {
        let kv_dim = 4_usize;
        let mut c = make_paged(1, 32, 1, kv_dim, 8);
        let row = vec![1.0_f32, 2.0, 3.0, 4.0];
        c.write_k(0, 3, &row).unwrap();
        let k = c.read_k(0, 4).unwrap(); // [4, 4]
        assert_eq!(k.dims(), &[4, kv_dim]);
        // pos=3 is slot 3 in page 0 → offset 3*kv_dim = 12
        assert_eq!(&k.as_slice()[12..16], row.as_slice());
        // Other positions should be zero
        assert!(k.as_slice()[..12].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_write_read_v_roundtrip() {
        let kv_dim = 4_usize;
        let mut c = make_paged(1, 32, 1, kv_dim, 8);
        let row = vec![-1.0_f32; 4];
        c.write_v(0, 0, &row).unwrap();
        let v = c.read_v(0, 1).unwrap();
        assert_eq!(v.dims(), &[1, kv_dim]);
        assert!(v.as_slice().iter().all(|&x| (x + 1.0).abs() < 1e-7));
    }

    #[test]
    fn test_unwritten_positions_are_zero() {
        let kv_dim = 2_usize;
        let c = make_paged(1, 16, 1, kv_dim, 4);
        // Nothing written — read should be all zeros
        let k = c.read_k(0, 4).unwrap();
        assert!(k.as_slice().iter().all(|&v| v == 0.0));
    }

    // ── matches contiguous KvCache ────────────────────────────────────────

    #[test]
    fn test_paged_matches_contiguous_sequential_writes() {
        let n_layers   = 1_usize;
        let n_kv_heads = 2_usize;
        let head_dim   = 4_usize;
        let max_seq    = 32_usize;
        let page_size  = 4_usize;
        let seq_len    = 12_usize;

        let mut contiguous = KvCache::new(n_layers, max_seq, n_kv_heads, head_dim);
        let mut paged      = PagedKvCache::new(n_layers, max_seq, n_kv_heads, head_dim, page_size);

        for pos in 0..seq_len {
            let row_k: Vec<f32> = (0..n_kv_heads * head_dim)
                .map(|i| (pos * 100 + i) as f32)
                .collect();
            let row_v: Vec<f32> = (0..n_kv_heads * head_dim)
                .map(|i| -(pos as f32 * 10.0 + i as f32))
                .collect();
            contiguous.write_k(0, pos, &row_k).unwrap();
            contiguous.write_v(0, pos, &row_v).unwrap();
            paged.write_k(0, pos, &row_k).unwrap();
            paged.write_v(0, pos, &row_v).unwrap();
        }

        let ck = contiguous.read_k(0, seq_len).unwrap();
        let pk = paged.read_k(0, seq_len).unwrap();
        assert_eq!(ck.dims(), pk.dims());
        for (i, (&a, &b)) in ck.as_slice().iter().zip(pk.as_slice()).enumerate() {
            assert!((a - b).abs() < 1e-7, "K mismatch at {i}: contiguous={a} paged={b}");
        }

        let cv = contiguous.read_v(0, seq_len).unwrap();
        let pv = paged.read_v(0, seq_len).unwrap();
        for (i, (&a, &b)) in cv.as_slice().iter().zip(pv.as_slice()).enumerate() {
            assert!((a - b).abs() < 1e-7, "V mismatch at {i}: contiguous={a} paged={b}");
        }
    }

    // ── cross-page boundary ───────────────────────────────────────────────

    #[test]
    fn test_cross_page_boundary_reads_correctly() {
        let kv_dim = 2_usize;
        let mut c = make_paged(1, 32, 1, kv_dim, 3); // page_size=3
        // pos=2 is last slot of page 0; pos=3 is first slot of page 1
        c.write_k(0, 2, &[10.0, 20.0]).unwrap();
        c.write_k(0, 3, &[30.0, 40.0]).unwrap();

        let k = c.read_k(0, 4).unwrap(); // [4, 2]
        assert_eq!(&k.as_slice()[4..6],  &[10.0, 20.0]); // pos=2 → offset 2*2=4
        assert_eq!(&k.as_slice()[6..8],  &[30.0, 40.0]); // pos=3 → offset 3*2=6
    }

    // ── multiple layers isolated ─────────────────────────────────────────

    #[test]
    fn test_multiple_layers_isolated() {
        let kv_dim = 2_usize;
        let mut c = make_paged(3, 16, 1, kv_dim, 4);
        c.write_k(0, 0, &[1.0, 1.0]).unwrap();
        c.write_k(2, 0, &[2.0, 2.0]).unwrap();

        // Layer 1 untouched → zero
        let k1 = c.read_k(1, 1).unwrap();
        assert!(k1.as_slice().iter().all(|&v| v == 0.0));

        // Layer 0 has [1, 1]
        let k0 = c.read_k(0, 1).unwrap();
        assert_eq!(k0.as_slice(), &[1.0, 1.0]);

        // Layer 2 has [2, 2]
        let k2 = c.read_k(2, 1).unwrap();
        assert_eq!(k2.as_slice(), &[2.0, 2.0]);
    }

    // ── clear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_drops_all_pages() {
        let kv_dim = 2_usize;
        let mut c = make_paged(1, 16, 1, kv_dim, 4);
        c.write_k(0, 0, &[5.0, 5.0]).unwrap();
        c.write_v(0, 0, &[5.0, 5.0]).unwrap();
        assert_eq!(c.allocated_pages_k(0), 1);

        c.clear();
        assert_eq!(c.allocated_pages_k(0), 0);
        assert_eq!(c.allocated_pages_v(0), 0);

        // Reading after clear gives zeros
        let k = c.read_k(0, 2).unwrap();
        assert!(k.as_slice().iter().all(|&v| v == 0.0));
    }

    // ── page_size = 1 edge case ────────────────────────────────────────────

    #[test]
    fn test_page_size_1() {
        let kv_dim = 4_usize;
        let seq    = 5_usize;
        let mut c  = make_paged(1, 16, 1, kv_dim, 1); // every pos is its own page
        for pos in 0..seq {
            let row: Vec<f32> = (0..kv_dim).map(|i| (pos * kv_dim + i) as f32).collect();
            c.write_k(0, pos, &row).unwrap();
        }
        assert_eq!(c.allocated_pages_k(0), seq);
        let k = c.read_k(0, seq).unwrap();
        for pos in 0..seq {
            let expected: Vec<f32> = (0..kv_dim).map(|i| (pos * kv_dim + i) as f32).collect();
            assert_eq!(&k.as_slice()[pos*kv_dim..(pos+1)*kv_dim], expected.as_slice());
        }
    }

    // ── error paths ───────────────────────────────────────────────────────

    #[test]
    fn test_write_layer_out_of_bounds() {
        let mut c = make_paged(2, 16, 1, 4, 4);
        assert!(matches!(
            c.write_k(2, 0, &[0.0; 4]),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_write_pos_exceeds_max_pages() {
        let mut c = make_paged(1, 8, 1, 4, 4); // max_pages = 2
        // pos=8 → page_idx=2 >= max_pages=2
        assert!(matches!(
            c.write_k(0, 8, &[0.0; 4]),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_write_wrong_row_length() {
        let mut c = make_paged(1, 16, 1, 4, 4);
        assert!(matches!(
            c.write_k(0, 0, &[0.0; 3]), // kv_dim=4
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_read_zero_seq_len_rejected() {
        let c = make_paged(1, 16, 1, 4, 4);
        assert!(matches!(
            c.read_k(0, 0),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_read_layer_out_of_bounds() {
        let c = make_paged(1, 16, 1, 4, 4);
        assert!(matches!(
            c.read_k(1, 1),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_read_seq_exceeds_max_seq_len() {
        let c = make_paged(1, 8, 1, 4, 4); // max_pages=2, max_seq effectively 8
        // seq_len=9 → needs 3 pages > max_pages=2
        assert!(matches!(
            c.read_k(0, 9),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
