//! Memory-mapped weight access with demand-paging hints — commit 10.3.
//!
//! # Purpose
//!
//! Large language model weights (e.g. Llama 7B ≈ 13 GB for F16) vastly exceed
//! the available RAM of consumer hardware.  The GGUF loader already uses
//! `memmap2` so the OS *can* demand-page weights, but by default it has no
//! information about which pages will be needed next.  This leads to:
//!
//! - Random I/O when the OS guesses wrong and evicts pages that are soon reused.
//! - Latency spikes during the first forward pass ("cold start").
//! - Unnecessary RSS pressure from pages that will never be reused.
//!
//! `WeightAccessor` adds a thin layer of OS memory-advice (`madvise`) on top of
//! the existing mmap, giving the kernel the hints it needs to make good decisions:
//!
//! | Advice       | When                              | Effect                        |
//! |--------------|-----------------------------------|-------------------------------|
//! | `WillNeed`   | Before computing a layer          | Async prefetch from disk      |
//! | `Sequential` | Prefill (full prompt processing)  | Enable OS read-ahead          |
//! | `Random`     | Decode (single-token steps)       | Disable unhelpful read-ahead  |
//! | `DontNeed`   | After a layer (low-memory mode)   | Allow OS to reclaim pages     |
//!
//! # Design
//!
//! `WeightAccessor<'a>` borrows a `GgufLoader` and scans all tensor names to
//! compute per-layer byte ranges once at construction.  Every subsequent hint
//! is a single `madvise` syscall with zero allocation.
//!
//! # Usage
//!
//! ```no_run
//! use llm_engine::model::mmap_weights::{WeightAccessor, AccessPattern};
//! use llm_engine::gguf::GgufLoader;
//!
//! let loader = GgufLoader::open("tinyllama.gguf").unwrap();
//! let mut accessor = WeightAccessor::new(&loader, 22);
//!
//! // Hint: warm up global weights (embeddings, norms).
//! accessor.prefetch_global();
//!
//! // Hint: prefetch next layer while computing the current one.
//! accessor.prefetch_layer(1);
//!
//! // Notify OS of access pattern for best I/O scheduling.
//! accessor.set_pattern(AccessPattern::Sequential);
//!
//! // After processing: release pages for layer 0 under memory pressure.
//! accessor.evict_layer(0);
//!
//! println!("~{} bytes resident", accessor.stats().estimated_resident_bytes);
//! ```

use std::collections::HashMap;

use crate::gguf::GgufLoader;

// ── access pattern ────────────────────────────────────────────────────────────

/// Expected memory-access pattern for the current generation phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AccessPattern {
    /// Prefill — full prompt, layer-by-layer sequential sweep.
    /// OS read-ahead is beneficial.
    #[default]
    Sequential,
    /// Decode — one token per step, small random working set.
    /// OS read-ahead would waste I/O.
    Random,
}

// ── per-layer byte region ─────────────────────────────────────────────────────

/// Tightest bounding interval `[start, end)` over all tensors in a layer.
#[derive(Debug, Clone, Copy)]
struct LayerRegion {
    start: usize, // inclusive byte offset into mmap
    end:   usize, // exclusive
}

impl LayerRegion {
    #[inline]
    fn len(self) -> usize {
        self.end.saturating_sub(self.start)
    }
}

// ── stats ─────────────────────────────────────────────────────────────────────

/// Counters for memory-advice calls and RSS estimates.
#[derive(Debug, Clone, Default)]
pub struct AccessStats {
    /// `WillNeed` hints issued.
    pub prefetch_hints: u64,
    /// `DontNeed` hints issued.
    pub evict_hints: u64,
    /// Full-mmap `Sequential` hints issued.
    pub sequential_hints: u64,
    /// Full-mmap `Random` hints issued.
    pub random_hints: u64,
    /// Software estimate of bytes where `WillNeed` has been issued but
    /// `DontNeed` has not yet been issued.  Not a true RSS measurement.
    pub estimated_resident_bytes: usize,
    /// Layers that have been prefetched at least once.
    pub layers_prefetched: usize,
    /// Layers that have been evicted at least once.
    pub layers_evicted: usize,
}

// ── weight accessor ───────────────────────────────────────────────────────────

/// Memory-mapped weight accessor with OS demand-paging hints.
///
/// See the [module documentation](self) for design rationale and usage.
pub struct WeightAccessor<'a> {
    loader:         &'a GgufLoader,
    layer_regions:  HashMap<usize, LayerRegion>,
    global_region:  Option<LayerRegion>,
    pattern:        AccessPattern,
    stats:          AccessStats,
    /// Layers that have received `WillNeed` but not yet `DontNeed`.
    resident:       HashMap<usize, usize>, // layer_idx → bytes
}

impl<'a> WeightAccessor<'a> {
    /// Build a new accessor for a model with `n_layers` transformer blocks.
    ///
    /// Scans all tensor names once to compute per-layer byte regions.
    /// O(n_tensors) at construction; every subsequent hint is O(1).
    #[must_use]
    pub fn new(loader: &'a GgufLoader, n_layers: usize) -> Self {
        let (layer_regions, global_region) = build_regions(loader, n_layers);
        Self {
            loader,
            layer_regions,
            global_region,
            pattern: AccessPattern::default(),
            stats:   AccessStats::default(),
            resident: HashMap::new(),
        }
    }

    // ── hint API ──────────────────────────────────────────────────────────

    /// Issue a `WillNeed` prefetch hint for all tensors in `layer_idx`.
    ///
    /// Call this one layer ahead of the one currently being computed.
    /// No-op if `layer_idx` has no known tensors.
    pub fn prefetch_layer(&mut self, layer_idx: usize) {
        if let Some(&region) = self.layer_regions.get(&layer_idx) {
            let _ = self.loader.mmap_advise_region(true, region.start, region.len());
            self.stats.prefetch_hints += 1;
            let bytes = region.len();
            self.resident.entry(layer_idx).or_insert_with(|| {
                self.stats.layers_prefetched += 1;
                bytes
            });
            self.stats.estimated_resident_bytes = self.resident.values().sum();
        }
    }

    /// Issue a `DontNeed` eviction hint for all tensors in `layer_idx`.
    ///
    /// Allows the OS to reclaim pages under memory pressure.
    pub fn evict_layer(&mut self, layer_idx: usize) {
        if let Some(&region) = self.layer_regions.get(&layer_idx) {
            let _ = self.loader.mmap_advise_region(false, region.start, region.len());
            self.stats.evict_hints += 1;
            if self.resident.remove(&layer_idx).is_some() {
                self.stats.layers_evicted += 1;
            }
            self.stats.estimated_resident_bytes = self.resident.values().sum();
        }
    }

    /// Prefetch global tensors (embeddings, output norm, etc.).
    ///
    /// These are used on every forward step and should always be resident.
    pub fn prefetch_global(&mut self) {
        if let Some(region) = self.global_region {
            let _ = self.loader.mmap_advise_region(true, region.start, region.len());
            self.stats.prefetch_hints += 1;
        }
    }

    /// Notify the OS of the expected access pattern.
    ///
    /// Applied as a hint to the full mmap:
    /// - [`Sequential`](AccessPattern::Sequential): enables read-ahead.
    /// - [`Random`](AccessPattern::Random): disables read-ahead.
    pub fn set_pattern(&mut self, pattern: AccessPattern) {
        self.pattern = pattern;
        match pattern {
            AccessPattern::Sequential => {
                let _ = self.loader.mmap_advise_sequential();
                self.stats.sequential_hints += 1;
            }
            AccessPattern::Random => {
                let _ = self.loader.mmap_advise_random();
                self.stats.random_hints += 1;
            }
        }
    }

    // ── introspection ─────────────────────────────────────────────────────

    /// Number of transformer layers with at least one tensor found in the model.
    #[must_use]
    pub fn n_known_layers(&self) -> usize {
        self.layer_regions.len()
    }

    /// Byte range `(start_inclusive, end_exclusive)` for `layer_idx`, if known.
    #[must_use]
    pub fn layer_region(&self, layer_idx: usize) -> Option<(usize, usize)> {
        self.layer_regions.get(&layer_idx).map(|r| (r.start, r.end))
    }

    /// Total bytes spanned by all per-layer tensor regions.
    #[must_use]
    pub fn layer_bytes_total(&self) -> usize {
        self.layer_regions.values().map(|r| r.len()).sum()
    }

    /// Current access pattern.
    #[must_use]
    pub fn pattern(&self) -> AccessPattern {
        self.pattern
    }

    /// Read-only view of cumulative statistics.
    #[must_use]
    pub fn stats(&self) -> &AccessStats {
        &self.stats
    }

    /// Reset statistics without changing the access state.
    pub fn reset_stats(&mut self) {
        self.stats = AccessStats {
            estimated_resident_bytes: self.resident.values().sum(),
            layers_prefetched: self.resident.len(),
            ..AccessStats::default()
        };
    }
}

// ── region building ───────────────────────────────────────────────────────────

fn build_regions(
    loader: &GgufLoader,
    n_layers: usize,
) -> (HashMap<usize, LayerRegion>, Option<LayerRegion>) {
    let tensors = loader.tensors();

    let mut layer_bounds: HashMap<usize, (usize, usize)> = HashMap::new();
    let mut global_min = usize::MAX;
    let mut global_max: usize = 0;
    let mut has_global = false;

    for info in tensors.iter() {
        let abs_start = tensors.absolute_offset(info) as usize;
        let abs_end   = abs_start + info.size_bytes();

        if let Some(layer_idx) = extract_layer_index(info.name()) {
            if layer_idx < n_layers {
                let entry = layer_bounds
                    .entry(layer_idx)
                    .or_insert((abs_start, abs_end));
                entry.0 = entry.0.min(abs_start);
                entry.1 = entry.1.max(abs_end);
            }
        } else {
            global_min = global_min.min(abs_start);
            global_max = global_max.max(abs_end);
            has_global = true;
        }
    }

    let layer_regions = layer_bounds
        .into_iter()
        .map(|(idx, (s, e))| (idx, LayerRegion { start: s, end: e }))
        .collect();

    let global_region = if has_global && global_max > global_min {
        Some(LayerRegion { start: global_min, end: global_max })
    } else {
        None
    };

    (layer_regions, global_region)
}

/// Extract the layer index from a tensor name.
///
/// Handles common patterns:
/// - `"blk.7.attn_q.weight"` → `7`
/// - `"model.layers.7.self_attn.q_proj.weight"` → `7`
fn extract_layer_index(name: &str) -> Option<usize> {
    for segment in name.split('.') {
        if let Ok(n) = segment.parse::<usize>() {
            return Some(n);
        }
    }
    None
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufLoader;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── helpers ───────────────────────────────────────────────────────────

    /// Build a minimal valid GGUF file in memory (no real tensor data —
    /// just enough for the loader to parse metadata and tensor infos).
    fn make_minimal_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        // Magic + version (v3)
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        // tensor_count = 0, metadata_count = 0
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // metadata_count
        buf
    }

    fn temp_gguf(data: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(data).unwrap();
        f.flush().unwrap();
        f
    }

    // ── construction ─────────────────────────────────────────────────────

    #[test]
    fn test_accessor_no_tensors() {
        let data = make_minimal_gguf();
        let f = temp_gguf(&data);
        let loader = GgufLoader::open(f.path()).unwrap();
        let accessor = WeightAccessor::new(&loader, 32);
        assert_eq!(accessor.n_known_layers(), 0);
        assert_eq!(accessor.layer_bytes_total(), 0);
    }

    #[test]
    fn test_accessor_default_pattern_is_sequential() {
        let data = make_minimal_gguf();
        let f = temp_gguf(&data);
        let loader = GgufLoader::open(f.path()).unwrap();
        let accessor = WeightAccessor::new(&loader, 32);
        assert_eq!(accessor.pattern(), AccessPattern::Sequential);
    }

    // ── pattern hints ─────────────────────────────────────────────────────

    #[test]
    fn test_set_pattern_sequential_increments_stat() {
        let data = make_minimal_gguf();
        let f = temp_gguf(&data);
        let loader = GgufLoader::open(f.path()).unwrap();
        let mut accessor = WeightAccessor::new(&loader, 0);
        accessor.set_pattern(AccessPattern::Sequential);
        assert_eq!(accessor.stats().sequential_hints, 1);
    }

    #[test]
    fn test_set_pattern_random_increments_stat() {
        let data = make_minimal_gguf();
        let f = temp_gguf(&data);
        let loader = GgufLoader::open(f.path()).unwrap();
        let mut accessor = WeightAccessor::new(&loader, 0);
        accessor.set_pattern(AccessPattern::Random);
        assert_eq!(accessor.stats().random_hints, 1);
        assert_eq!(accessor.pattern(), AccessPattern::Random);
    }

    // ── prefetch / evict on empty model ──────────────────────────────────

    #[test]
    fn test_prefetch_unknown_layer_is_noop() {
        let data = make_minimal_gguf();
        let f = temp_gguf(&data);
        let loader = GgufLoader::open(f.path()).unwrap();
        let mut accessor = WeightAccessor::new(&loader, 32);
        accessor.prefetch_layer(5); // no tensors → no-op
        assert_eq!(accessor.stats().prefetch_hints, 0);
    }

    #[test]
    fn test_evict_unknown_layer_is_noop() {
        let data = make_minimal_gguf();
        let f = temp_gguf(&data);
        let loader = GgufLoader::open(f.path()).unwrap();
        let mut accessor = WeightAccessor::new(&loader, 32);
        accessor.evict_layer(5);
        assert_eq!(accessor.stats().evict_hints, 0);
    }

    // ── extract_layer_index ───────────────────────────────────────────────

    #[test]
    fn test_extract_layer_index_blk_format() {
        assert_eq!(extract_layer_index("blk.7.attn_q.weight"), Some(7));
    }

    #[test]
    fn test_extract_layer_index_layers_format() {
        assert_eq!(extract_layer_index("model.layers.3.self_attn.q_proj.weight"), Some(3));
    }

    #[test]
    fn test_extract_layer_index_global_tensor() {
        // "token_embd.weight" — no numeric segment
        assert_eq!(extract_layer_index("token_embd.weight"), None);
    }

    #[test]
    fn test_extract_layer_index_output_norm() {
        assert_eq!(extract_layer_index("output_norm.weight"), None);
    }

    // ── stats reset ───────────────────────────────────────────────────────

    #[test]
    fn test_reset_stats_clears_counters() {
        let data = make_minimal_gguf();
        let f = temp_gguf(&data);
        let loader = GgufLoader::open(f.path()).unwrap();
        let mut accessor = WeightAccessor::new(&loader, 0);
        accessor.set_pattern(AccessPattern::Random);
        accessor.set_pattern(AccessPattern::Sequential);
        assert_eq!(accessor.stats().sequential_hints, 1);
        accessor.reset_stats();
        assert_eq!(accessor.stats().sequential_hints, 0);
        assert_eq!(accessor.stats().random_hints, 0);
    }
}
