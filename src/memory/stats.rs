//! Memory usage tracking and reporting — commit 10.4.
//!
//! # Purpose
//!
//! Running Llama 7B on ≤16 GB RAM requires careful accounting of every
//! significant allocation.  This module provides a unified view of memory
//! usage across the engine's major subsystems:
//!
//! - **Model weights** — mmap region size (not necessarily RSS, but an upper bound)
//! - **KV-cache** — pre-allocated K/V buffers
//! - **Tensor pool** — free-list buffers from commit 10.1
//! - **Arena** — backing buffer from commit 10.2
//! - **System** — resident set size (RSS) from `/proc/self/status` on Linux
//!   or `task_info` on macOS (best-effort; returns 0 on unsupported platforms)
//!
//! # Usage
//!
//! ```no_run
//! use llm_engine::memory::{MemoryTracker, MemorySnapshot};
//! use llm_engine::memory::{TensorPool, Arena};
//!
//! let pool  = TensorPool::new();
//! let arena = Arena::with_capacity(1 << 20); // 1 M floats = 4 MB
//!
//! let tracker = MemoryTracker::new(
//!     Some(13_000_000_000), // model mmap size
//!     Some(512 * 1024 * 8 * 64 * 4 * 2), // KV-cache bytes
//! );
//!
//! let snap = tracker.snapshot(&pool, &arena);
//! snap.print_report();
//! ```
//!
//! # Accuracy
//!
//! - **Pool and arena bytes** are exact (we control the allocations).
//! - **KV-cache bytes** come from a size passed at construction — accurate
//!   for pre-allocated caches.
//! - **Model weight bytes** come from the mmap size — equals file size, not RSS.
//! - **Process RSS** is a best-effort system query.  On Linux it reads
//!   `/proc/self/status`; on macOS it calls `task_vm_info`; elsewhere it
//!   returns 0.

use crate::memory::{Arena, TensorPool};

// ── byte formatting ───────────────────────────────────────────────────────────

/// Format a byte count as a human-readable string (B / KB / MB / GB).
#[must_use]
pub fn format_bytes(bytes: usize) -> String {
    const KB: usize = 1024;
    const MB: usize = 1024 * KB;
    const GB: usize = 1024 * MB;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

// ── snapshot ──────────────────────────────────────────────────────────────────

/// A point-in-time view of memory usage across engine subsystems.
#[derive(Debug, Clone, Default)]
pub struct MemorySnapshot {
    /// Bytes occupied by model weight mmap (file size — not RSS).
    pub weight_mmap_bytes: usize,
    /// Bytes pre-allocated for the KV-cache (exact).
    pub kv_cache_bytes: usize,
    /// Bytes held in the tensor pool's free lists.
    pub pool_free_bytes: usize,
    /// Bytes in the arena's backing buffer (capacity, not just used portion).
    pub arena_capacity_bytes: usize,
    /// Bytes currently "live" in the arena (used * sizeof f32).
    pub arena_used_bytes: usize,
    /// Process RSS from the OS, in bytes.  0 if unavailable.
    pub process_rss_bytes: usize,
}

impl MemorySnapshot {
    /// Total bytes under engine control (sum of the four tracked regions).
    ///
    /// Does not include process RSS (which may overlap with the above).
    #[must_use]
    pub fn engine_total_bytes(&self) -> usize {
        self.weight_mmap_bytes
            + self.kv_cache_bytes
            + self.pool_free_bytes
            + self.arena_capacity_bytes
    }

    /// Print a formatted report to stdout.
    pub fn print_report(&self) {
        println!("── Memory Usage ─────────────────────────────");
        println!(
            "  Weight mmap  : {} (file size, not RSS)",
            format_bytes(self.weight_mmap_bytes)
        );
        println!("  KV-cache     : {}", format_bytes(self.kv_cache_bytes));
        println!("  Pool (free)  : {}", format_bytes(self.pool_free_bytes));
        println!(
            "  Arena        : {} capacity / {} used",
            format_bytes(self.arena_capacity_bytes),
            format_bytes(self.arena_used_bytes)
        );
        println!("  Engine total : {}", format_bytes(self.engine_total_bytes()));
        if self.process_rss_bytes > 0 {
            println!("  Process RSS  : {}", format_bytes(self.process_rss_bytes));
        }
        println!("─────────────────────────────────────────────");
    }
}

// ── tracker ───────────────────────────────────────────────────────────────────

/// Collects and reports memory usage across pool, arena, and fixed-size regions.
pub struct MemoryTracker {
    /// Size of the weight memory map (bytes), if a model is loaded.
    weight_mmap_bytes: usize,
    /// Size of the KV-cache allocation (bytes), if present.
    kv_cache_bytes: usize,
    /// Peak snapshot observed across all `snapshot()` calls.
    peak: MemorySnapshot,
}

impl MemoryTracker {
    /// Create a tracker.
    ///
    /// Pass `None` for any region that hasn't been allocated yet.
    #[must_use]
    pub fn new(weight_mmap_bytes: Option<usize>, kv_cache_bytes: Option<usize>) -> Self {
        Self {
            weight_mmap_bytes: weight_mmap_bytes.unwrap_or(0),
            kv_cache_bytes: kv_cache_bytes.unwrap_or(0),
            peak: MemorySnapshot::default(),
        }
    }

    /// Update the tracked weight mmap size (e.g. after loading a new model).
    pub fn set_weight_mmap_bytes(&mut self, bytes: usize) {
        self.weight_mmap_bytes = bytes;
    }

    /// Update the tracked KV-cache size (e.g. after resizing the cache).
    pub fn set_kv_cache_bytes(&mut self, bytes: usize) {
        self.kv_cache_bytes = bytes;
    }

    /// Take a snapshot of current memory usage.
    ///
    /// Queries pool + arena state directly; reads process RSS from the OS.
    /// Updates the internal peak if the new snapshot's engine total is higher.
    pub fn snapshot(&mut self, pool: &TensorPool, arena: &Arena) -> MemorySnapshot {
        let snap = MemorySnapshot {
            weight_mmap_bytes:    self.weight_mmap_bytes,
            kv_cache_bytes:       self.kv_cache_bytes,
            pool_free_bytes:      pool.pooled_bytes(),
            arena_capacity_bytes: arena.capacity() * std::mem::size_of::<f32>(),
            arena_used_bytes:     arena.used()      * std::mem::size_of::<f32>(),
            process_rss_bytes:    query_rss(),
        };

        if snap.engine_total_bytes() > self.peak.engine_total_bytes() {
            self.peak = snap.clone();
        }

        snap
    }

    /// The highest-watermark snapshot seen since construction.
    #[must_use]
    pub fn peak(&self) -> &MemorySnapshot {
        &self.peak
    }

    /// Human-readable summary of the peak snapshot.
    pub fn print_peak_report(&self) {
        println!("── Peak Memory Usage ────────────────────────");
        self.peak.print_report();
    }

    /// Estimated overhead ratio: `engine_total / process_rss`.
    ///
    /// Values > 1 suggest the engine's accounting is inflated (e.g. mmap
    /// pages are not actually resident).  Values < 1 suggest unmeasured
    /// allocations (Rust heap, thread stacks, etc.).
    ///
    /// Returns `None` if RSS is unavailable or zero.
    #[must_use]
    pub fn overhead_ratio(snap: &MemorySnapshot) -> Option<f64> {
        if snap.process_rss_bytes == 0 {
            None
        } else {
            Some(snap.engine_total_bytes() as f64 / snap.process_rss_bytes as f64)
        }
    }
}

// ── process RSS query ─────────────────────────────────────────────────────────

/// Query the current process Resident Set Size.
///
/// Returns 0 on unsupported platforms or when the query fails.
#[must_use]
pub fn query_rss() -> usize {
    #[cfg(target_os = "linux")]
    {
        read_linux_rss().unwrap_or(0)
    }
    #[cfg(target_os = "macos")]
    {
        read_macos_rss().unwrap_or(0)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0
    }
}

#[cfg(target_os = "linux")]
fn read_linux_rss() -> Option<usize> {
    let contents = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in contents.lines() {
        if line.starts_with("VmRSS:") {
            // Format: "VmRSS:     12345 kB"
            let kb: usize = line
                .split_whitespace()
                .nth(1)?
                .parse()
                .ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "macos")]
fn read_macos_rss() -> Option<usize> {
    // Use mach task_vm_info to read phys_footprint.
    // This is the same value Instruments and Activity Monitor show.
    use std::mem;

    // Safety: these are stable mach kernel types present on all macOS versions.
    extern "C" {
        fn task_self_trap() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut u32,
            task_info_count: *mut u32,
        ) -> i32;
    }

    const TASK_VM_INFO: u32 = 22;
    const TASK_VM_INFO_COUNT: u32 = 87; // sizeof(task_vm_info_data_t) / sizeof(natural_t)

    // task_vm_info_data_t is 87 u32-sized fields; we only need phys_footprint
    // which is at offset 6 (field index 12 as u64, since fields 0-5 are u64).
    // Layout (as of macOS 10.15+): virtual_size(u64), region_count(u32),
    // page_size(u32), resident_size(u64), resident_size_peak(u64), ...
    // phys_footprint(u64) is at byte offset 48.
    let mut info = vec![0u32; TASK_VM_INFO_COUNT as usize];
    let mut count = TASK_VM_INFO_COUNT;

    let ret = unsafe {
        task_info(
            task_self_trap(),
            TASK_VM_INFO,
            info.as_mut_ptr(),
            &mut count,
        )
    };

    if ret != 0 {
        return None;
    }

    // resident_size is at byte offset 16 (after virtual_size:u64 + region_count:u32 + page_size:u32)
    // = u64 at u32-index 4
    let resident: u64 = u64::from_le_bytes(unsafe {
        mem::transmute::<[u32; 2], [u8; 8]>([info[4], info[5]])
    });
    Some(resident as usize)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh() -> (TensorPool, Arena) {
        (TensorPool::new(), Arena::with_capacity(1024))
    }

    // ── format_bytes ──────────────────────────────────────────────────────

    #[test]
    fn test_format_bytes_bytes() {
        assert_eq!(format_bytes(512), "512 B");
    }

    #[test]
    fn test_format_bytes_kb() {
        assert_eq!(format_bytes(2048), "2.00 KB");
    }

    #[test]
    fn test_format_bytes_mb() {
        assert_eq!(format_bytes(5 * 1024 * 1024), "5.00 MB");
    }

    #[test]
    fn test_format_bytes_gb() {
        assert_eq!(format_bytes(2 * 1024 * 1024 * 1024), "2.00 GB");
    }

    // ── snapshot ──────────────────────────────────────────────────────────

    #[test]
    fn test_snapshot_arena_capacity_in_bytes() {
        let (pool, arena) = fresh(); // arena capacity = 1024 f32 = 4096 bytes
        let mut tracker = MemoryTracker::new(None, None);
        let snap = tracker.snapshot(&pool, &arena);
        assert_eq!(snap.arena_capacity_bytes, 1024 * 4);
    }

    #[test]
    fn test_snapshot_pool_free_bytes() {
        let mut pool = TensorPool::new();
        pool.prewarm(256, 2); // 2 × 256 × 4 = 2048 bytes
        let arena = Arena::default();
        let mut tracker = MemoryTracker::new(None, None);
        let snap = tracker.snapshot(&pool, &arena);
        assert_eq!(snap.pool_free_bytes, 2048);
    }

    #[test]
    fn test_snapshot_weight_mmap_bytes() {
        let (pool, arena) = fresh();
        let mut tracker = MemoryTracker::new(Some(1_000_000), None);
        let snap = tracker.snapshot(&pool, &arena);
        assert_eq!(snap.weight_mmap_bytes, 1_000_000);
    }

    #[test]
    fn test_snapshot_kv_cache_bytes() {
        let (pool, arena) = fresh();
        let mut tracker = MemoryTracker::new(None, Some(512_000));
        let snap = tracker.snapshot(&pool, &arena);
        assert_eq!(snap.kv_cache_bytes, 512_000);
    }

    #[test]
    fn test_engine_total_bytes() {
        let (pool, arena) = fresh();
        let mut tracker = MemoryTracker::new(Some(1000), Some(500));
        let snap = tracker.snapshot(&pool, &arena);
        let expected = 1000 + 500 + snap.pool_free_bytes + snap.arena_capacity_bytes;
        assert_eq!(snap.engine_total_bytes(), expected);
    }

    // ── peak tracking ─────────────────────────────────────────────────────

    #[test]
    fn test_peak_tracks_maximum_engine_total() {
        let (pool, mut arena) = fresh();
        let mut tracker = MemoryTracker::new(None, None);

        // First snapshot: small
        let _ = tracker.snapshot(&pool, &arena);

        // Grow arena → larger engine total
        arena.grow(64 * 1024);
        let snap2 = tracker.snapshot(&pool, &arena);

        // Peak should equal the second snapshot (larger)
        assert_eq!(
            tracker.peak().engine_total_bytes(),
            snap2.engine_total_bytes()
        );
    }

    #[test]
    fn test_peak_does_not_shrink() {
        let (pool, mut arena) = fresh();
        let mut tracker = MemoryTracker::new(None, None);

        arena.grow(64 * 1024);
        let peak_snap = tracker.snapshot(&pool, &arena);
        let peak_total = peak_snap.engine_total_bytes();

        // Shrink: create a new small arena
        let small_arena = Arena::with_capacity(1);
        let _ = tracker.snapshot(&pool, &small_arena);

        // Peak must not decrease
        assert_eq!(tracker.peak().engine_total_bytes(), peak_total);
    }

    // ── set methods ───────────────────────────────────────────────────────

    #[test]
    fn test_set_weight_mmap_bytes() {
        let (pool, arena) = fresh();
        let mut tracker = MemoryTracker::new(None, None);
        tracker.set_weight_mmap_bytes(9_000_000);
        let snap = tracker.snapshot(&pool, &arena);
        assert_eq!(snap.weight_mmap_bytes, 9_000_000);
    }

    #[test]
    fn test_set_kv_cache_bytes() {
        let (pool, arena) = fresh();
        let mut tracker = MemoryTracker::new(None, None);
        tracker.set_kv_cache_bytes(200_000);
        let snap = tracker.snapshot(&pool, &arena);
        assert_eq!(snap.kv_cache_bytes, 200_000);
    }

    // ── query_rss ─────────────────────────────────────────────────────────

    #[test]
    fn test_query_rss_returns_plausible_value_or_zero() {
        let rss = query_rss();
        // On Linux/macOS, RSS should be at least 1 MB for a Rust test binary.
        // On other platforms it returns 0. Either is acceptable.
        if rss > 0 {
            assert!(rss >= 1024 * 1024, "RSS suspiciously small: {rss}");
        }
    }

    // ── overhead_ratio ────────────────────────────────────────────────────

    #[test]
    fn test_overhead_ratio_none_when_rss_zero() {
        let snap = MemorySnapshot {
            process_rss_bytes: 0,
            ..Default::default()
        };
        assert!(MemoryTracker::overhead_ratio(&snap).is_none());
    }

    #[test]
    fn test_overhead_ratio_computed() {
        let snap = MemorySnapshot {
            weight_mmap_bytes: 1000,
            process_rss_bytes: 2000,
            ..Default::default()
        };
        let ratio = MemoryTracker::overhead_ratio(&snap).unwrap();
        assert!((ratio - 0.5).abs() < 1e-9);
    }
}
