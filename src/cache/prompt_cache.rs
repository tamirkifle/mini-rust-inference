//! Prompt caching for repeated prefixes — commit 11.4.
//!
//! # Problem
//!
//! Many LLM use-cases send the same long system prompt before each user turn.
//! Without caching, every new request re-runs the full prefill over those
//! tokens — paying their compute cost every time.
//!
//! # Solution
//!
//! `PromptCache` maintains a small set of **KV snapshots** keyed by token
//! sequence.  On each new request:
//!
//! 1. **Lookup** — find the longest stored prefix that matches the start of the
//!    new token list.  Returns a [`PrefixMatch`] with `matched_len` and a
//!    [`KvSnapshot`].
//! 2. **Restore** — call `snapshot.restore_into(cache)` to pre-populate a
//!    `KvCache` with the cached K/V rows.
//! 3. **Prefill only the tail** — run `ChunkedPrefill` (or the standard forward
//!    pass) starting at `start_pos = matched_len`, skipping the shared prefix.
//! 4. **Store** — after prefill completes, call `cache.store(tokens, cache, prefix_len)`
//!    to save the new snapshot for future requests.
//!
//! # Eviction
//!
//! When `capacity` is exceeded the **least-recently-used** entry is evicted.
//! Access order is tracked via a monotonic generation counter; no `HashMap` or
//! `BTreeMap` is needed at this scale.
//!
//! # Correctness guarantee
//!
//! A snapshot round-trips exactly: `store` then `lookup` then `restore_into`
//! produces a `KvCache` whose `read_k`/`read_v` outputs are bit-for-bit
//! identical to the original cache at the time of the store.

use crate::cache::KvCache;
use crate::tensor::{Result, TensorError};

// ── KvSnapshot ───────────────────────────────────────────────────────────────

/// An immutable snapshot of the first `seq_len` K/V rows from a `KvCache`.
///
/// Created by [`PromptCache::store`] and consumed by
/// [`KvSnapshot::restore_into`].
#[derive(Debug, Clone)]
pub struct KvSnapshot {
    /// Per-layer K data: `k_data[layer]` is a flat vec of `seq_len * kv_dim` f32.
    k_data:   Vec<Vec<f32>>,
    /// Per-layer V data: same layout as `k_data`.
    v_data:   Vec<Vec<f32>>,
    /// Number of token rows captured.
    pub seq_len:  usize,
    /// `n_kv_heads * head_dim`.
    pub kv_dim:   usize,
    /// Number of transformer layers.
    pub n_layers: usize,
}

impl KvSnapshot {
    /// Copy the snapshot's K/V rows into `cache`, overwriting positions
    /// `0 .. self.seq_len` in every layer.
    ///
    /// The caller is responsible for ensuring `cache` has the same `kv_dim`
    /// and `n_layers`, and that `cache.max_seq_len() >= self.seq_len`.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if dimensions are incompatible or any
    /// write is out of bounds.
    pub fn restore_into(&self, cache: &mut KvCache) -> Result<()> {
        if cache.kv_dim() != self.kv_dim {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "KvSnapshot::restore_into: kv_dim {} != cache.kv_dim {}",
                    self.kv_dim, cache.kv_dim()
                ),
            });
        }
        if cache.n_layers() < self.n_layers {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "KvSnapshot::restore_into: cache has {} layers < snapshot {}",
                    cache.n_layers(), self.n_layers
                ),
            });
        }
        for layer in 0..self.n_layers {
            for pos in 0..self.seq_len {
                let start = pos * self.kv_dim;
                let end   = start + self.kv_dim;
                cache.write_k(layer, pos, &self.k_data[layer][start..end])?;
                cache.write_v(layer, pos, &self.v_data[layer][start..end])?;
            }
        }
        Ok(())
    }
}

// ── PrefixMatch ───────────────────────────────────────────────────────────────

/// Result of a successful [`PromptCache::lookup`].
pub struct PrefixMatch {
    /// Number of leading tokens whose K/V data is cached.
    pub matched_len: usize,
    /// The cached K/V state for those tokens.
    pub snapshot:    KvSnapshot,
}

// ── PromptCache ───────────────────────────────────────────────────────────────

/// LRU cache of K/V snapshots keyed by token prefix.
///
/// Stores up to `capacity` entries.  When full, the least-recently-used entry
/// is evicted on the next `store` call.
pub struct PromptCache {
    entries:   Vec<CacheEntry>,
    capacity:  usize,
    /// Monotonically increasing clock for LRU tracking.
    clock:     u64,
    /// Dimensions every snapshot must match.
    n_layers:  usize,
    kv_dim:    usize,
}

struct CacheEntry {
    tokens:       Vec<u32>,
    snapshot:     KvSnapshot,
    last_used:    u64,
}

impl PromptCache {
    /// Create a new prompt cache.
    ///
    /// # Arguments
    ///
    /// * `capacity`   – maximum number of prefix snapshots to keep (≥ 1).
    /// * `n_layers`   – transformer layer count (used to validate stored caches).
    /// * `n_kv_heads` – KV head count.
    /// * `head_dim`   – dimension per head.
    ///
    /// # Panics
    ///
    /// Panics if `capacity == 0`.
    #[must_use]
    pub fn new(capacity: usize, n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        assert!(capacity > 0, "PromptCache: capacity must be ≥ 1");
        Self {
            entries:  Vec::with_capacity(capacity),
            capacity,
            clock:    0,
            n_layers,
            kv_dim:   n_kv_heads * head_dim,
        }
    }

    // ── lookup ────────────────────────────────────────────────────────────

    /// Find the longest cached prefix of `tokens`.
    ///
    /// Iterates all entries, picks the one with the longest token sequence that
    /// is a prefix of `tokens` (i.e. `tokens.starts_with(entry.tokens)`).
    ///
    /// Updates the entry's LRU timestamp.
    ///
    /// Returns `None` if no prefix matches at all (not even length 1).
    pub fn lookup(&mut self, tokens: &[u32]) -> Option<PrefixMatch> {
        let mut best_idx    = None::<usize>;
        let mut best_len    = 0_usize;

        for (idx, entry) in self.entries.iter().enumerate() {
            if entry.tokens.is_empty() { continue; }
            // Compute common prefix length between entry.tokens and tokens.
            let match_len = common_prefix_len(&entry.tokens, tokens);
            // Only count a hit if the *entire* stored prefix matched.
            // (A partial match would mean the cached K/V is for different tokens.)
            if match_len == entry.tokens.len() && match_len > best_len {
                best_len = match_len;
                best_idx = Some(idx);
            }
        }

        let idx = best_idx?;
        self.clock += 1;
        self.entries[idx].last_used = self.clock;
        Some(PrefixMatch {
            matched_len: best_len,
            snapshot:    self.entries[idx].snapshot.clone(),
        })
    }

    // ── store ─────────────────────────────────────────────────────────────

    /// Snapshot the first `prefix_len` K/V rows from `cache` and store them
    /// keyed by `tokens[..prefix_len]`.
    ///
    /// If an entry with the same token prefix already exists it is updated
    /// in-place.  Otherwise, if the cache is at capacity, the LRU entry is
    /// evicted before inserting the new one.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if `prefix_len > cache.max_seq_len()`,
    /// `cache.kv_dim() != self.kv_dim`, or `prefix_len > tokens.len()`.
    pub fn store(
        &mut self,
        tokens:     &[u32],
        cache:      &KvCache,
        prefix_len: usize,
    ) -> Result<()> {
        if prefix_len > tokens.len() {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "PromptCache::store: prefix_len {prefix_len} > tokens.len {}",
                    tokens.len()
                ),
            });
        }
        if cache.kv_dim() != self.kv_dim {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "PromptCache::store: cache.kv_dim {} != expected {}",
                    cache.kv_dim(), self.kv_dim
                ),
            });
        }
        if prefix_len > cache.max_seq_len() {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "PromptCache::store: prefix_len {prefix_len} > cache.max_seq_len {}",
                    cache.max_seq_len()
                ),
            });
        }

        let prefix_tokens = tokens[..prefix_len].to_vec();
        let snapshot      = self.snapshot_from_cache(cache, prefix_len)?;

        // Check for an existing entry with the same prefix — update in-place.
        if let Some(entry) = self.entries.iter_mut()
            .find(|e| e.tokens == prefix_tokens)
        {
            self.clock += 1;
            entry.snapshot  = snapshot;
            entry.last_used = self.clock;
            return Ok(());
        }

        // Evict LRU entry if at capacity.
        if self.entries.len() >= self.capacity {
            let lru_idx = self.entries.iter().enumerate()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(i, _)| i)
                .unwrap(); // safe: capacity ≥ 1, entries non-empty
            self.entries.swap_remove(lru_idx);
        }

        self.clock += 1;
        self.entries.push(CacheEntry {
            tokens:    prefix_tokens,
            snapshot,
            last_used: self.clock,
        });
        Ok(())
    }

    // ── accessors ─────────────────────────────────────────────────────────

    /// Number of snapshots currently stored.
    #[must_use]
    pub fn len(&self) -> usize { self.entries.len() }

    /// `true` if no snapshots are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }

    /// Maximum number of entries before eviction occurs.
    #[must_use]
    pub fn capacity(&self) -> usize { self.capacity }

    /// Remove all stored entries.
    pub fn clear(&mut self) { self.entries.clear(); }

    // ── private helpers ───────────────────────────────────────────────────

    /// Copy `prefix_len` rows per layer out of `cache` into a new `KvSnapshot`.
    fn snapshot_from_cache(&self, cache: &KvCache, prefix_len: usize) -> Result<KvSnapshot> {
        let mut k_data = Vec::with_capacity(self.n_layers);
        let mut v_data = Vec::with_capacity(self.n_layers);

        for layer in 0..self.n_layers {
            if prefix_len == 0 {
                k_data.push(Vec::new());
                v_data.push(Vec::new());
            } else {
                let k_tensor = cache.read_k(layer, prefix_len)?;
                let v_tensor = cache.read_v(layer, prefix_len)?;
                k_data.push(k_tensor.as_slice().to_vec());
                v_data.push(v_tensor.as_slice().to_vec());
            }
        }

        Ok(KvSnapshot {
            k_data,
            v_data,
            seq_len:  prefix_len,
            kv_dim:   self.kv_dim,
            n_layers: self.n_layers,
        })
    }
}

// ── free helpers ──────────────────────────────────────────────────────────────

/// Return the length of the longest common prefix between two slices.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;

    /// Build a KvCache with deterministic synthetic data written for `seq_len` positions.
    fn filled_cache(n_layers: usize, n_kv_heads: usize, head_dim: usize, seq_len: usize) -> KvCache {
        let mut cache = KvCache::new(n_layers, 128, n_kv_heads, head_dim);
        let kv_dim = n_kv_heads * head_dim;
        for layer in 0..n_layers {
            for pos in 0..seq_len {
                let row: Vec<f32> = (0..kv_dim)
                    .map(|i| (layer * 1000 + pos * 10 + i) as f32)
                    .collect();
                cache.write_k(layer, pos, &row).unwrap();
                cache.write_v(layer, pos, &row).unwrap();
            }
        }
        cache
    }

    fn make_prompt_cache(cap: usize, n_layers: usize, kv_heads: usize, hd: usize) -> PromptCache {
        PromptCache::new(cap, n_layers, kv_heads, hd)
    }

    // ── basic store / lookup ──────────────────────────────────────────────

    #[test]
    fn test_store_increments_len() {
        let cache = filled_cache(2, 2, 4, 8);
        let mut pc = make_prompt_cache(4, 2, 2, 4);
        assert_eq!(pc.len(), 0);
        pc.store(&[1, 2, 3, 4], &cache, 4).unwrap();
        assert_eq!(pc.len(), 1);
    }

    #[test]
    fn test_lookup_exact_match() {
        let n_layers = 2_usize;
        let n_kv_heads = 1_usize;
        let head_dim = 4_usize;
        let seq_len  = 4_usize;

        let cache = filled_cache(n_layers, n_kv_heads, head_dim, seq_len);
        let mut pc = make_prompt_cache(4, n_layers, n_kv_heads, head_dim);

        let tokens = &[10u32, 20, 30, 40];
        pc.store(tokens, &cache, seq_len).unwrap();

        // Lookup with exact same tokens
        let hit = pc.lookup(tokens).expect("should hit");
        assert_eq!(hit.matched_len, seq_len);
    }

    #[test]
    fn test_lookup_prefix_of_longer_sequence() {
        let n_layers = 1_usize;
        let n_kv_heads = 1_usize;
        let head_dim = 4_usize;
        let seq_len  = 4_usize;

        let cache = filled_cache(n_layers, n_kv_heads, head_dim, seq_len);
        let mut pc = make_prompt_cache(4, n_layers, n_kv_heads, head_dim);

        let prefix = &[1u32, 2, 3, 4];
        pc.store(prefix, &cache, seq_len).unwrap();

        // Query with prefix + more tokens — should still hit with matched_len=4
        let query = &[1u32, 2, 3, 4, 5, 6];
        let hit = pc.lookup(query).expect("should hit");
        assert_eq!(hit.matched_len, seq_len);
    }

    #[test]
    fn test_lookup_no_match_returns_none() {
        let cache = filled_cache(1, 1, 4, 4);
        let mut pc = make_prompt_cache(4, 1, 1, 4);
        pc.store(&[1, 2, 3, 4], &cache, 4).unwrap();

        // Completely different tokens — no match
        assert!(pc.lookup(&[9, 8, 7]).is_none());
    }

    #[test]
    fn test_lookup_partial_stored_prefix_does_not_match() {
        // Stored prefix=[1,2,3,4]; query starts with [1,2] but stored entry
        // is 4 tokens → we require the *entire* stored prefix to match.
        // A query shorter than the stored prefix should not match.
        let cache = filled_cache(1, 1, 4, 4);
        let mut pc = make_prompt_cache(4, 1, 1, 4);
        pc.store(&[1u32, 2, 3, 4], &cache, 4).unwrap();

        // Query is [1, 2] — shorter than stored prefix [1,2,3,4] → no match
        assert!(pc.lookup(&[1u32, 2]).is_none());
    }

    // ── snapshot round-trip ───────────────────────────────────────────────

    #[test]
    fn test_snapshot_restore_matches_original() {
        let n_layers   = 2_usize;
        let n_kv_heads = 1_usize;
        let head_dim   = 4_usize;
        let seq_len    = 5_usize;

        let src_cache = filled_cache(n_layers, n_kv_heads, head_dim, seq_len);
        let mut pc    = make_prompt_cache(4, n_layers, n_kv_heads, head_dim);

        let key: Vec<u32> = vec![0u32; seq_len];
        pc.store(&key, &src_cache, seq_len).unwrap();
        let hit = pc.lookup(&key).unwrap();

        // Restore into a fresh cache
        let mut dst_cache = KvCache::new(n_layers, 128, n_kv_heads, head_dim);
        hit.snapshot.restore_into(&mut dst_cache).unwrap();

        // Every layer's K and V should match the original
        for layer in 0..n_layers {
            let src_k = src_cache.read_k(layer, seq_len).unwrap();
            let dst_k = dst_cache.read_k(layer, seq_len).unwrap();
            for (i, (&a, &b)) in src_k.as_slice().iter().zip(dst_k.as_slice()).enumerate() {
                assert!((a - b).abs() < 1e-7,
                    "layer {layer} K[{i}] mismatch: src={a} dst={b}");
            }
            let src_v = src_cache.read_v(layer, seq_len).unwrap();
            let dst_v = dst_cache.read_v(layer, seq_len).unwrap();
            for (i, (&a, &b)) in src_v.as_slice().iter().zip(dst_v.as_slice()).enumerate() {
                assert!((a - b).abs() < 1e-7,
                    "layer {layer} V[{i}] mismatch: src={a} dst={b}");
            }
        }
    }

    // ── longest prefix wins ───────────────────────────────────────────────

    #[test]
    fn test_lookup_returns_longest_matching_prefix() {
        let n_layers = 1_usize; let kv_heads = 1_usize; let hd = 4_usize;
        let cache_short = filled_cache(n_layers, kv_heads, hd, 3);
        let cache_long  = filled_cache(n_layers, kv_heads, hd, 5);
        let mut pc = make_prompt_cache(4, n_layers, kv_heads, hd);

        pc.store(&[1u32, 2, 3],       &cache_short, 3).unwrap();
        pc.store(&[1u32, 2, 3, 4, 5], &cache_long,  5).unwrap();

        // Query starts with [1,2,3,4,5,...] — both entries match but longer wins
        let hit = pc.lookup(&[1u32, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(hit.matched_len, 5);
    }

    // ── LRU eviction ─────────────────────────────────────────────────────

    #[test]
    fn test_lru_eviction_removes_oldest() {
        let n_layers = 1_usize; let kv_heads = 1_usize; let hd = 4_usize;
        let cache = filled_cache(n_layers, kv_heads, hd, 2);
        let mut pc = make_prompt_cache(2, n_layers, kv_heads, hd); // capacity=2

        // Store two entries
        pc.store(&[1u32, 2], &cache, 2).unwrap(); // clock=1, entry A
        pc.store(&[3u32, 4], &cache, 2).unwrap(); // clock=2, entry B
        assert_eq!(pc.len(), 2);

        // Access entry A to make it most-recently used
        let _ = pc.lookup(&[1u32, 2]);  // clock=3, A.last_used=3

        // Store a third entry — capacity=2, should evict B (last_used=2)
        pc.store(&[5u32, 6], &cache, 2).unwrap(); // clock=4
        assert_eq!(pc.len(), 2);

        // A should still be present
        assert!(pc.lookup(&[1u32, 2]).is_some(), "entry A should survive LRU eviction");
        // B should be gone
        assert!(pc.lookup(&[3u32, 4]).is_none(), "entry B should have been evicted");
    }

    #[test]
    fn test_capacity_1_always_evicts_on_new_store() {
        let n_layers = 1_usize; let kv_heads = 1_usize; let hd = 4_usize;
        let cache = filled_cache(n_layers, kv_heads, hd, 2);
        let mut pc = make_prompt_cache(1, n_layers, kv_heads, hd);

        pc.store(&[1u32, 2], &cache, 2).unwrap();
        assert!(pc.lookup(&[1u32, 2]).is_some());

        // Store a different entry — should evict the first
        pc.store(&[3u32, 4], &cache, 2).unwrap();
        assert_eq!(pc.len(), 1);
        assert!(pc.lookup(&[1u32, 2]).is_none());
        assert!(pc.lookup(&[3u32, 4]).is_some());
    }

    // ── update existing entry ─────────────────────────────────────────────

    #[test]
    fn test_store_same_prefix_updates_in_place() {
        let n_layers = 1_usize; let kv_heads = 1_usize; let hd = 4_usize;
        let cache = filled_cache(n_layers, kv_heads, hd, 3);
        let mut pc = make_prompt_cache(4, n_layers, kv_heads, hd);
        let tokens = &[1u32, 2, 3];

        pc.store(tokens, &cache, 3).unwrap();
        pc.store(tokens, &cache, 3).unwrap(); // same prefix again
        // Should not grow beyond 1 entry
        assert_eq!(pc.len(), 1);
    }

    // ── clear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_clear_removes_all_entries() {
        let n_layers = 1_usize; let kv_heads = 1_usize; let hd = 4_usize;
        let cache = filled_cache(n_layers, kv_heads, hd, 3);
        let mut pc = make_prompt_cache(4, n_layers, kv_heads, hd);
        pc.store(&[1u32, 2, 3], &cache, 3).unwrap();
        pc.store(&[4u32, 5, 6], &cache, 3).unwrap();
        pc.clear();
        assert_eq!(pc.len(), 0);
        assert!(pc.lookup(&[1u32, 2, 3]).is_none());
    }

    // ── errors ────────────────────────────────────────────────────────────

    #[test]
    fn test_store_prefix_len_exceeds_tokens_rejected() {
        let cache = filled_cache(1, 1, 4, 4);
        let mut pc = make_prompt_cache(4, 1, 1, 4);
        // prefix_len=5 > tokens.len()=3
        assert!(matches!(
            pc.store(&[1u32, 2, 3], &cache, 5),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_store_kv_dim_mismatch_rejected() {
        // Cache with kv_dim=8 (2 heads × 4), PromptCache expects kv_dim=4 (1 head × 4)
        let cache = filled_cache(1, 2, 4, 4); // kv_dim = 2*4 = 8
        let mut pc = make_prompt_cache(4, 1, 1, 4); // kv_dim = 1*4 = 4
        assert!(matches!(
            pc.store(&[1u32, 2, 3, 4], &cache, 4),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_restore_kv_dim_mismatch_rejected() {
        let n_layers = 1_usize; let kv_heads = 1_usize; let hd = 4_usize;
        let cache = filled_cache(n_layers, kv_heads, hd, 3);
        let mut pc = make_prompt_cache(4, n_layers, kv_heads, hd);
        pc.store(&[1u32, 2, 3], &cache, 3).unwrap();
        let hit = pc.lookup(&[1u32, 2, 3]).unwrap();

        // Try restoring into a cache with wrong kv_dim
        let mut bad_cache = KvCache::new(n_layers, 64, 2, hd); // kv_dim = 2*4 = 8
        assert!(matches!(
            hit.snapshot.restore_into(&mut bad_cache),
            Err(TensorError::InvalidShape { .. })
        ));
    }
}
