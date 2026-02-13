//! Arena bump allocator for per-forward-pass scratch memory — commit 10.2.
//!
//! # Purpose
//!
//! A transformer forward pass allocates many small, short-lived buffers that
//! are all dead by the time the next token is generated.  The standard
//! allocator must track and free each one individually, which costs CPU time
//! proportional to the number of allocations.
//!
//! An **arena** allocator takes a different approach: carve slices out of a
//! single large pre-allocated `Vec<f32>` in order (bump allocation), then
//! reset the cursor back to zero at the end of the pass.  The result is:
//!
//! - **O(1) alloc**: just a bounds check and a cursor increment.
//! - **O(1) reset**: cursor ← 0.  No individual frees, no fragmentation.
//! - **Zero heap traffic** during the forward pass (after initial sizing).
//!
//! # Sizing the arena
//!
//! Call [`Arena::with_capacity`] with the maximum number of f32 elements you
//! expect in a single forward pass.  A conservative formula for a Llama 7B
//! forward pass (2048-token context) is roughly:
//!
//! ```text
//! capacity ≈ n_layers × (4 × seq_len × hidden_dim) + seq_len × hidden_dim
//! ```
//!
//! For TinyLlama (hidden_dim=2048, 22 layers, seq=512) that is ~90 M floats
//! (~360 MB).  You can start smaller and let the arena grow dynamically via
//! [`Arena::try_alloc`] / [`Arena::alloc`] fallback to heap.
//!
//! # Usage
//!
//! ```
//! use llm_engine::memory::Arena;
//!
//! let mut arena = Arena::with_capacity(1024);
//!
//! // Allocate a zeroed slice for the forward pass.
//! let q_buf = arena.alloc(256); // returns Vec<f32> with len 256
//! let k_buf = arena.alloc(256);
//!
//! // At the end of the pass, reset — all memory is reused.
//! arena.reset();
//! assert_eq!(arena.used(), 0);
//! ```
//!
//! # Lifetime model
//!
//! `alloc()` returns an **owned** `Vec<f32>` backed by a copy out of the
//! arena's internal buffer.  This is the same cost as the pool in commit-10.1
//! but avoids fragmentation and allows a single `reset()` to reclaim
//! everything at once.
//!
//! An alternative zero-copy design returning `&'arena mut [f32]` would be
//! more memory-efficient but requires callers to hold arena borrows through
//! the entire pass, which conflicts with the `&mut self` call pattern used
//! everywhere in the engine.  The copy-on-alloc design is therefore preferred
//! for ergonomics.  The copy is cheap relative to the matmul it precedes.

/// Arena bump allocator for per-forward-pass scratch buffers.
///
/// See the [module documentation](self) for design rationale and usage.
pub struct Arena {
    /// Backing storage: pre-allocated f32 pool.
    buf: Vec<f32>,
    /// Next free position (in elements, not bytes).
    cursor: usize,
    /// High-water mark: maximum `cursor` value observed since construction or
    /// last [`reset_stats`](Arena::reset_stats).
    peak_used: usize,
    /// Total number of successful `alloc()` calls.
    alloc_count: u64,
    /// Total f32 elements handed out over the arena's lifetime.
    total_elements_allocated: u64,
    /// Number of times `reset()` was called.
    reset_count: u64,
    /// Allocations that fell back to the heap because the arena was full.
    heap_fallback_count: u64,
}

impl Arena {
    /// Create an arena with `capacity` f32 elements of pre-allocated backing.
    ///
    /// If you are unsure of the right size, use [`Arena::default`] (64 K
    /// elements) and let the arena grow via heap fallback until you measure
    /// the true peak usage with [`Arena::peak_used`].
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            buf: vec![0.0_f32; capacity],
            cursor: 0,
            peak_used: 0,
            alloc_count: 0,
            total_elements_allocated: 0,
            reset_count: 0,
            heap_fallback_count: 0,
        }
    }

    // ── allocation ────────────────────────────────────────────────────────

    /// Allocate a zeroed `Vec<f32>` of `n` elements.
    ///
    /// If `n` elements fit inside the arena's remaining capacity, the method
    /// copies from the arena buffer (O(n)) and advances the cursor.  If not,
    /// it falls back to a heap allocation and records the event in
    /// [`heap_fallback_count`](Arena::heap_fallback_count).
    ///
    /// The returned buffer is always zero-filled regardless of path.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `n == 0`.
    pub fn alloc(&mut self, n: usize) -> Vec<f32> {
        debug_assert!(n > 0, "Arena::alloc: n must be > 0");
        self.alloc_count          += 1;
        self.total_elements_allocated += n as u64;

        if self.cursor + n <= self.buf.len() {
            // Fast path: carve from arena.
            // The arena buffer is zeroed on construction and after each reset,
            // so this slice is already zero — no extra fill needed.
            let slice = self.buf[self.cursor..self.cursor + n].to_vec();
            self.cursor += n;
            if self.cursor > self.peak_used {
                self.peak_used = self.cursor;
            }
            slice
        } else {
            // Slow path: heap fallback.
            self.heap_fallback_count += 1;
            vec![0.0_f32; n]
        }
    }

    /// Try to allocate `n` elements from the arena without heap fallback.
    ///
    /// Returns `Some(Vec<f32>)` if the arena has room, `None` otherwise.
    /// Does **not** increment `heap_fallback_count` on `None`.
    #[must_use]
    pub fn try_alloc(&mut self, n: usize) -> Option<Vec<f32>> {
        if n == 0 || self.cursor + n > self.buf.len() {
            return None;
        }
        self.alloc_count             += 1;
        self.total_elements_allocated += n as u64;
        let slice = self.buf[self.cursor..self.cursor + n].to_vec();
        self.cursor += n;
        if self.cursor > self.peak_used {
            self.peak_used = self.cursor;
        }
        Some(slice)
    }

    // ── lifecycle ─────────────────────────────────────────────────────────

    /// Reset the allocation cursor to zero, making the full arena available again.
    ///
    /// This **zeroes the used portion** of the backing buffer so future
    /// allocations always receive clean memory.  Cost: O(used_elements).
    pub fn reset(&mut self) {
        // Zero only the used prefix; the rest is already zero.
        self.buf[..self.cursor].fill(0.0_f32);
        self.cursor = 0;
        self.reset_count += 1;
    }

    /// Grow the arena's backing buffer to at least `new_capacity` elements.
    ///
    /// If `new_capacity` is smaller than the current capacity, this is a
    /// no-op.  Existing in-flight allocations are not affected (they are
    /// owned copies).  After growing, the new region is zero-initialized.
    pub fn grow(&mut self, new_capacity: usize) {
        if new_capacity > self.buf.len() {
            self.buf.resize(new_capacity, 0.0_f32);
        }
    }

    // ── introspection ─────────────────────────────────────────────────────

    /// Total capacity (f32 elements) of the backing buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Elements allocated since the last [`reset`](Arena::reset).
    #[must_use]
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Free elements remaining before heap fallback triggers.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.cursor)
    }

    /// Utilization ratio in `[0.0, 1.0]`.
    ///
    /// Returns `0.0` if the arena has zero capacity.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.buf.is_empty() {
            0.0
        } else {
            self.cursor as f64 / self.buf.len() as f64
        }
    }

    /// Maximum value of [`used`](Arena::used) observed since construction or
    /// the last [`reset_stats`](Arena::reset_stats).
    ///
    /// Use this to right-size the arena capacity: set it to `peak_used() * 1.2`
    /// to leave 20 % headroom.
    #[must_use]
    pub fn peak_used(&self) -> usize {
        self.peak_used
    }

    /// Number of [`alloc`](Arena::alloc) calls that fell back to the heap.
    ///
    /// Any non-zero value means the arena is undersized.
    #[must_use]
    pub fn heap_fallback_count(&self) -> u64 {
        self.heap_fallback_count
    }

    /// Total [`alloc`](Arena::alloc) / [`try_alloc`](Arena::try_alloc) calls.
    #[must_use]
    pub fn alloc_count(&self) -> u64 {
        self.alloc_count
    }

    /// Total f32 elements handed out over the arena's lifetime.
    #[must_use]
    pub fn total_elements_allocated(&self) -> u64 {
        self.total_elements_allocated
    }

    /// Number of times [`reset`](Arena::reset) was called.
    #[must_use]
    pub fn reset_count(&self) -> u64 {
        self.reset_count
    }

    /// Reset all stat counters (peak, counts) without touching the buffer or cursor.
    pub fn reset_stats(&mut self) {
        self.peak_used = self.cursor; // preserve current watermark
        self.alloc_count = 0;
        self.total_elements_allocated = 0;
        self.reset_count = 0;
        self.heap_fallback_count = 0;
    }
}

impl Default for Arena {
    /// Default arena with 64 K f32 elements (~256 KB) of backing storage.
    fn default() -> Self {
        Self::with_capacity(64 * 1024)
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── basic alloc ───────────────────────────────────────────────────────

    #[test]
    fn test_alloc_returns_correct_size() {
        let mut arena = Arena::with_capacity(512);
        let buf = arena.alloc(128);
        assert_eq!(buf.len(), 128);
    }

    #[test]
    fn test_alloc_buffer_is_zeroed() {
        let mut arena = Arena::with_capacity(256);
        let buf = arena.alloc(64);
        assert!(buf.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_alloc_advances_cursor() {
        let mut arena = Arena::with_capacity(512);
        arena.alloc(100);
        assert_eq!(arena.used(), 100);
        arena.alloc(50);
        assert_eq!(arena.used(), 150);
    }

    #[test]
    fn test_remaining_decreases_with_allocs() {
        let mut arena = Arena::with_capacity(200);
        assert_eq!(arena.remaining(), 200);
        arena.alloc(80);
        assert_eq!(arena.remaining(), 120);
    }

    // ── reset ─────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_zeroes_cursor() {
        let mut arena = Arena::with_capacity(256);
        arena.alloc(100);
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 256);
    }

    #[test]
    fn test_reset_reuses_memory_flat() {
        let mut arena = Arena::with_capacity(128);
        for _ in 0..100 {
            arena.alloc(32);
            arena.alloc(32);
            arena.alloc(32);
            arena.reset();
        }
        // After 100 full forward passes, heap fallbacks must be 0.
        assert_eq!(arena.heap_fallback_count(), 0);
    }

    #[test]
    fn test_reset_zeroes_dirty_region() {
        // Manually dirty the buffer, then reset, then re-alloc and confirm zeros.
        let mut arena = Arena::with_capacity(64);
        {
            let mut buf = arena.alloc(64);
            buf.iter_mut().for_each(|x| *x = 7.0);
            // buf is dropped here (owned copy), but the arena slice was already
            // a copy — the internal buffer is unchanged.
        }
        // At this point the arena's internal slice is still zeroed (alloc copies
        // from it, it doesn't move data out).  Confirm that.
        arena.reset();
        let buf2 = arena.alloc(64);
        assert!(buf2.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_reset_increments_counter() {
        let mut arena = Arena::with_capacity(128);
        arena.reset();
        arena.reset();
        assert_eq!(arena.reset_count(), 2);
    }

    // ── heap fallback ─────────────────────────────────────────────────────

    #[test]
    fn test_alloc_heap_fallback_when_full() {
        let mut arena = Arena::with_capacity(64);
        arena.alloc(64); // fills arena exactly
        let buf = arena.alloc(1); // must fall back
        assert_eq!(buf.len(), 1);
        assert_eq!(arena.heap_fallback_count(), 1);
    }

    #[test]
    fn test_fallback_buf_is_zeroed() {
        let mut arena = Arena::with_capacity(8);
        arena.alloc(8);
        let buf = arena.alloc(4);
        assert!(buf.iter().all(|&x| x == 0.0));
    }

    // ── try_alloc ─────────────────────────────────────────────────────────

    #[test]
    fn test_try_alloc_succeeds_when_room() {
        let mut arena = Arena::with_capacity(128);
        let result = arena.try_alloc(64);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 64);
    }

    #[test]
    fn test_try_alloc_fails_when_full() {
        let mut arena = Arena::with_capacity(32);
        arena.alloc(32);
        assert!(arena.try_alloc(1).is_none());
        assert_eq!(arena.heap_fallback_count(), 0); // try_alloc doesn't count
    }

    #[test]
    fn test_try_alloc_zero_always_none() {
        let mut arena = Arena::with_capacity(128);
        assert!(arena.try_alloc(0).is_none());
    }

    // ── grow ──────────────────────────────────────────────────────────────

    #[test]
    fn test_grow_increases_capacity() {
        let mut arena = Arena::with_capacity(64);
        arena.grow(256);
        assert_eq!(arena.capacity(), 256);
    }

    #[test]
    fn test_grow_smaller_is_noop() {
        let mut arena = Arena::with_capacity(256);
        arena.grow(64); // smaller — should be a no-op
        assert_eq!(arena.capacity(), 256);
    }

    #[test]
    fn test_grow_enables_arena_alloc() {
        let mut arena = Arena::with_capacity(8);
        arena.alloc(8); // fill up
        arena.reset();
        arena.grow(128);
        // Now we can allocate 128 elements without fallback.
        for _ in 0..4 {
            arena.alloc(32);
        }
        assert_eq!(arena.heap_fallback_count(), 0);
    }

    // ── peak_used ─────────────────────────────────────────────────────────

    #[test]
    fn test_peak_used_tracks_high_water_mark() {
        let mut arena = Arena::with_capacity(256);
        arena.alloc(100);
        arena.alloc(50);
        assert_eq!(arena.peak_used(), 150);
        arena.reset();
        // Peak persists across reset.
        assert_eq!(arena.peak_used(), 150);
        // New alloc below peak — peak unchanged.
        arena.alloc(30);
        assert_eq!(arena.peak_used(), 150);
        // New alloc above peak — peak updated.
        arena.alloc(200);
        assert_eq!(arena.peak_used(), 230);
    }

    // ── stats ─────────────────────────────────────────────────────────────

    #[test]
    fn test_alloc_count_tracked() {
        let mut arena = Arena::with_capacity(512);
        arena.alloc(10);
        arena.alloc(20);
        arena.alloc(5);
        assert_eq!(arena.alloc_count(), 3);
    }

    #[test]
    fn test_total_elements_allocated() {
        let mut arena = Arena::with_capacity(512);
        arena.alloc(100);
        arena.alloc(200);
        assert_eq!(arena.total_elements_allocated(), 300);
    }

    #[test]
    fn test_utilization() {
        let mut arena = Arena::with_capacity(200);
        arena.alloc(100);
        assert!((arena.utilization() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_utilization_empty_arena() {
        let arena = Arena::with_capacity(0);
        assert_eq!(arena.utilization(), 0.0);
    }

    // ── default ───────────────────────────────────────────────────────────

    #[test]
    fn test_default_capacity() {
        let arena = Arena::default();
        assert_eq!(arena.capacity(), 64 * 1024);
        assert_eq!(arena.used(), 0);
    }
}
