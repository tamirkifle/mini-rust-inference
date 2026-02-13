//! Tensor memory pool — commit 10.1.
//!
//! # Purpose
//!
//! Every forward pass allocates dozens of intermediate `Vec<f32>` buffers
//! (QKV projections, attention scores, FFN intermediates, etc.).  The default
//! allocator handles this fine for occasional inference, but the repeated
//! malloc/free pressure adds up over thousands of decode steps.
//!
//! `TensorPool` maintains per-size free lists.  When a caller requests a
//! buffer of `n` f32 elements:
//! - If a matching free buffer exists, it is returned immediately (zeroed).
//! - Otherwise a fresh `Vec<f32>` is allocated from the system heap.
//!
//! When the caller is done, it hands the buffer back with [`TensorPool::free`].
//! The pool clears it and saves it for the next request of the same size.
//!
//! # Design choices
//!
//! - **Keyed by element count** (`usize`), not byte size or shape — shapes are
//!   handled by the caller wrapping the raw buffer in a `Tensor`.
//! - **No per-buffer ownership tracking** — the pool trusts callers to return
//!   buffers at the right size.  Mismatched frees are silently dropped (pool
//!   only stores a buffer if its `capacity()` matches its `len()`).
//! - **`&mut self`** API — no interior mutability, no locks.  Use one pool per
//!   thread/session.
//! - **Free-list depth cap** ([`MAX_FREE_PER_SIZE`]) prevents unbounded growth
//!   if callers free more buffers than they later allocate.
//!
//! # Usage
//!
//! ```
//! use llm_engine::memory::TensorPool;
//!
//! let mut pool = TensorPool::new();
//!
//! // Allocate a 512-element buffer (zero-filled).
//! let mut buf = pool.alloc(512);
//! buf[0] = 1.0;
//!
//! // Return it for reuse.
//! pool.free(buf);
//!
//! // Next alloc of the same size hits the free list.
//! let buf2 = pool.alloc(512);
//! assert_eq!(buf2.len(), 512);
//! assert!(buf2.iter().all(|&x| x == 0.0)); // always zeroed on hand-out
//! ```

use std::collections::HashMap;

/// Maximum number of free buffers stored per size bucket.
///
/// Capped to prevent memory growth when a caller repeatedly frees without
/// reallocating (e.g., if a layer is skipped).
const MAX_FREE_PER_SIZE: usize = 8;

/// Pool statistics for diagnostics and memory usage reporting.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of `alloc()` calls.
    pub total_allocs: u64,
    /// Allocs satisfied from the free list (no heap allocation).
    pub reuses: u64,
    /// Allocs that required a fresh heap allocation.
    pub fresh_allocs: u64,
    /// Total number of `free()` calls.
    pub total_frees: u64,
    /// Frees that were stored in the pool.
    pub stored_frees: u64,
    /// Frees that were discarded (free list full or wrong size).
    pub dropped_frees: u64,
}

impl PoolStats {
    /// Fraction of allocations satisfied from the free list (`0.0`–`1.0`).
    ///
    /// Returns `0.0` if no allocations have been made yet.
    #[must_use]
    pub fn reuse_rate(&self) -> f64 {
        if self.total_allocs == 0 {
            0.0
        } else {
            self.reuses as f64 / self.total_allocs as f64
        }
    }
}

/// Pre-allocated buffer pool for intermediate `f32` tensors.
///
/// See the [module documentation](self) for design rationale and usage.
pub struct TensorPool {
    /// Free lists keyed by element count.
    free_lists: HashMap<usize, Vec<Vec<f32>>>,
    /// Cumulative allocation statistics.
    stats: PoolStats,
}

impl TensorPool {
    /// Create an empty pool with no pre-warmed buffers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            free_lists: HashMap::new(),
            stats: PoolStats::default(),
        }
    }

    /// Allocate a zeroed `Vec<f32>` of exactly `n` elements.
    ///
    /// Returns a buffer from the free list if one is available; otherwise
    /// allocates a fresh `Vec<f32>` from the system heap.
    ///
    /// The returned buffer is **always zeroed**, regardless of whether it came
    /// from the free list or was freshly allocated.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `n == 0`.
    pub fn alloc(&mut self, n: usize) -> Vec<f32> {
        debug_assert!(n > 0, "TensorPool::alloc: n must be > 0");
        self.stats.total_allocs += 1;

        if let Some(list) = self.free_lists.get_mut(&n) {
            if let Some(mut buf) = list.pop() {
                // Zero the buffer before handing it out.
                buf.fill(0.0_f32);
                self.stats.reuses += 1;
                return buf;
            }
        }

        // Fresh allocation.
        self.stats.fresh_allocs += 1;
        vec![0.0_f32; n]
    }

    /// Return a buffer to the pool for future reuse.
    ///
    /// The buffer is cleared (all elements set to zero) and stored under
    /// its current `len()` key.  If the free list for that size is already
    /// at capacity ([`MAX_FREE_PER_SIZE`]), the buffer is dropped.
    ///
    /// **Precondition**: `buf.len()` must equal the `n` passed to the
    /// corresponding `alloc()` call.  If the caller accidentally changes
    /// `buf`'s length (e.g. via `truncate`), it will be stored under the
    /// wrong key or dropped — the pool will still function correctly but
    /// may not reuse the memory as expected.
    pub fn free(&mut self, mut buf: Vec<f32>) {
        let n = buf.len();
        if n == 0 {
            return; // nothing useful to keep
        }
        self.stats.total_frees += 1;

        let list = self.free_lists.entry(n).or_default();
        if list.len() < MAX_FREE_PER_SIZE {
            buf.fill(0.0_f32);
            list.push(buf);
            self.stats.stored_frees += 1;
        } else {
            self.stats.dropped_frees += 1;
            // buf is dropped here — memory returned to the system allocator.
        }
    }

    /// Pre-warm the pool by allocating and immediately freeing `count` buffers
    /// of `n` elements each.
    ///
    /// Useful at session startup to avoid the first `count` allocs hitting the
    /// heap.
    pub fn prewarm(&mut self, n: usize, count: usize) {
        let capped = count.min(MAX_FREE_PER_SIZE);
        for _ in 0..capped {
            let buf = vec![0.0_f32; n];
            let list = self.free_lists.entry(n).or_default();
            if list.len() < MAX_FREE_PER_SIZE {
                list.push(buf);
            }
        }
    }

    /// Number of free buffers of size `n` currently in the pool.
    #[must_use]
    pub fn free_count(&self, n: usize) -> usize {
        self.free_lists.get(&n).map_or(0, Vec::len)
    }

    /// Total number of free buffers across all sizes.
    #[must_use]
    pub fn total_free_buffers(&self) -> usize {
        self.free_lists.values().map(Vec::len).sum()
    }

    /// Approximate bytes held in free-list buffers.
    ///
    /// Each element is 4 bytes (f32).
    #[must_use]
    pub fn pooled_bytes(&self) -> usize {
        self.free_lists
            .iter()
            .map(|(n, list)| n * list.len() * std::mem::size_of::<f32>())
            .sum()
    }

    /// Drop all free-list buffers, releasing pooled memory back to the OS.
    ///
    /// Statistics are preserved; allocations after a `shrink` will hit the
    /// heap again until the pool is re-warmed.
    pub fn shrink(&mut self) {
        self.free_lists.clear();
    }

    /// Read-only view of cumulative pool statistics.
    #[must_use]
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Reset statistics counters to zero without releasing buffers.
    pub fn reset_stats(&mut self) {
        self.stats = PoolStats::default();
    }
}

impl Default for TensorPool {
    fn default() -> Self {
        Self::new()
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── basic alloc/free ──────────────────────────────────────────────────

    #[test]
    fn test_alloc_returns_correct_size() {
        let mut pool = TensorPool::new();
        let buf = pool.alloc(256);
        assert_eq!(buf.len(), 256);
    }

    #[test]
    fn test_alloc_buffer_is_zeroed() {
        let mut pool = TensorPool::new();
        let buf = pool.alloc(64);
        assert!(buf.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_free_then_alloc_reuses_buffer() {
        let mut pool = TensorPool::new();
        let buf = pool.alloc(128);
        pool.free(buf);

        let buf2 = pool.alloc(128);
        assert_eq!(buf2.len(), 128);
        assert_eq!(pool.stats().reuses, 1);
        assert_eq!(pool.stats().fresh_allocs, 1); // only the first alloc was fresh
    }

    #[test]
    fn test_reused_buffer_is_zeroed() {
        let mut pool = TensorPool::new();
        let mut buf = pool.alloc(32);
        buf.iter_mut().for_each(|x| *x = 9.9);
        pool.free(buf);
        let buf2 = pool.alloc(32);
        assert!(buf2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_different_sizes_are_independent() {
        let mut pool = TensorPool::new();
        let b64  = pool.alloc(64);
        let b128 = pool.alloc(128);
        pool.free(b64);
        pool.free(b128);
        assert_eq!(pool.free_count(64),  1);
        assert_eq!(pool.free_count(128), 1);
    }

    // ── free-list cap ─────────────────────────────────────────────────────

    #[test]
    fn test_free_list_cap_drops_excess() {
        let mut pool = TensorPool::new();
        // Allocate MAX_FREE_PER_SIZE + 2 buffers first, then free them all at
        // once so the free list fills up and the last two are dropped.
        let bufs: Vec<_> = (0..MAX_FREE_PER_SIZE + 2)
            .map(|_| pool.alloc(16))
            .collect();
        for buf in bufs {
            pool.free(buf);
        }
        assert_eq!(pool.free_count(16), MAX_FREE_PER_SIZE);
        assert_eq!(pool.stats().dropped_frees, 2);
    }

    // ── prewarm ───────────────────────────────────────────────────────────

    #[test]
    fn test_prewarm_fills_free_list() {
        let mut pool = TensorPool::new();
        pool.prewarm(512, 4);
        assert_eq!(pool.free_count(512), 4);
    }

    #[test]
    fn test_prewarm_capped_at_max() {
        let mut pool = TensorPool::new();
        pool.prewarm(256, MAX_FREE_PER_SIZE + 10);
        assert_eq!(pool.free_count(256), MAX_FREE_PER_SIZE);
    }

    #[test]
    fn test_prewarm_then_alloc_reuses() {
        let mut pool = TensorPool::new();
        pool.prewarm(128, 3);
        let _b1 = pool.alloc(128);
        let _b2 = pool.alloc(128);
        let _b3 = pool.alloc(128);
        assert_eq!(pool.stats().reuses, 3);
        assert_eq!(pool.stats().fresh_allocs, 0);
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_stats_counts_are_accurate() {
        let mut pool = TensorPool::new();
        let b1 = pool.alloc(100); // fresh
        let b2 = pool.alloc(100); // fresh
        pool.free(b1);
        let _b3 = pool.alloc(100); // reuse
        pool.free(b2);
        // total_allocs = 3, fresh = 2, reuses = 1, total_frees = 2, stored = 2
        assert_eq!(pool.stats().total_allocs,  3);
        assert_eq!(pool.stats().fresh_allocs,  2);
        assert_eq!(pool.stats().reuses,        1);
        assert_eq!(pool.stats().total_frees,   2);
        assert_eq!(pool.stats().stored_frees,  2);
    }

    #[test]
    fn test_reuse_rate() {
        let mut pool = TensorPool::new();
        pool.prewarm(64, 2);
        let _b1 = pool.alloc(64); // reuse
        let _b2 = pool.alloc(64); // reuse
        let _b3 = pool.alloc(64); // fresh
        // rate = 2/3
        assert!((pool.stats().reuse_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    // ── pooled_bytes / total_free_buffers ────────────────────────────────

    #[test]
    fn test_pooled_bytes_accounting() {
        let mut pool = TensorPool::new();
        pool.prewarm(256, 2); // 2 × 256 × 4 = 2048 bytes
        pool.prewarm(128, 1); //     128 × 4 =  512 bytes
        assert_eq!(pool.pooled_bytes(), 2048 + 512);
        assert_eq!(pool.total_free_buffers(), 3);
    }

    // ── shrink ────────────────────────────────────────────────────────────

    #[test]
    fn test_shrink_releases_pooled_memory() {
        let mut pool = TensorPool::new();
        pool.prewarm(512, 4);
        assert!(pool.pooled_bytes() > 0);
        pool.shrink();
        assert_eq!(pool.pooled_bytes(), 0);
        assert_eq!(pool.total_free_buffers(), 0);
    }

    #[test]
    fn test_shrink_preserves_stats() {
        let mut pool = TensorPool::new();
        let buf = pool.alloc(64);
        pool.free(buf);
        pool.shrink();
        assert_eq!(pool.stats().total_allocs, 1);
    }

    // ── edge cases ────────────────────────────────────────────────────────

    #[test]
    fn test_free_zero_len_buffer_is_noop() {
        let mut pool = TensorPool::new();
        pool.free(vec![]); // should not panic or add to any list
        assert_eq!(pool.total_free_buffers(), 0);
        assert_eq!(pool.stats().total_frees, 0);
    }

    #[test]
    fn test_no_fragmentation_over_many_cycles() {
        let mut pool = TensorPool::new();
        pool.prewarm(1024, MAX_FREE_PER_SIZE);
        for _ in 0..1000 {
            let buf = pool.alloc(1024);
            pool.free(buf);
        }
        // All 1000 allocs after warmup should be reuses.
        assert_eq!(pool.stats().fresh_allocs, 0);
        assert_eq!(pool.stats().reuses, 1000);
    }
}
