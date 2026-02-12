//! Cache management operations — commit 9.4.
//!
//! High-level helpers that combine [`KvCache`] and [`CachePosition`] mutations
//! into safe, atomic operations.
//!
//! # Operations
//!
//! | Function          | Behaviour                                              |
//! |-------------------|--------------------------------------------------------|
//! | [`cache_reset`]   | Clear all K/V data and reset position to 0.            |
//! | [`cache_truncate`]| Evict suffix tokens, zero their data, rewind position. |
//!
//! # When to use `cache_truncate`
//!
//! Truncation is useful when re-generating from a shared prefix — e.g. the user
//! asks to regenerate a response starting from the same context.  The cached K/V
//! up to the chosen cut-point is preserved so the next prefill can start from
//! there, while the evicted suffix positions are zeroed to prevent stale data
//! from leaking into future reads.

use crate::cache::{KvCache, CachePosition};
use crate::tensor::{Result, TensorError};

// ── reset ─────────────────────────────────────────────────────────────────────

/// Clear all cached K/V data and reset the position cursor to 0.
///
/// Call this before processing an entirely new sequence to avoid stale tokens
/// from a prior generation run contaminating the current one.
///
/// Memory is **not** freed — the pre-allocated buffers are zeroed in place and
/// remain available for the next session.
pub fn cache_reset(cache: &mut KvCache, pos: &mut CachePosition) {
    cache.clear();
    pos.reset();
}

// ── truncate ──────────────────────────────────────────────────────────────────

/// Truncate the cache to the first `new_len` tokens.
///
/// Positions `[new_len .. pos.current())` are zeroed across all layers so that
/// future reads cannot observe stale values.  The position cursor is rewound to
/// `new_len`.
///
/// # Use case
///
/// Regeneration from a shared prompt prefix: call `cache_truncate(cache, pos, prompt_len)`
/// to discard all previously-generated tokens while preserving the prompt K/V
/// projections.
///
/// # Errors
///
/// [`TensorError::InvalidShape`] if `new_len > pos.current()`.
pub fn cache_truncate(
    cache:   &mut KvCache,
    pos:     &mut CachePosition,
    new_len: usize,
) -> Result<()> {
    let current = pos.current();
    if new_len > current {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "cache_truncate: new_len {new_len} > current pos {current}"
            ),
        });
    }

    // Zero out the evicted suffix so stale values don't leak into future reads.
    cache.zero_range(new_len, current);

    // Rewind the position cursor.
    pos.set(new_len)?;

    Ok(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::{KvCache, CachePosition};

    fn filled_cache(n_layers: usize, n_tokens: usize, kv_dim_val: usize) -> (KvCache, CachePosition) {
        // kv_dim_val == n_kv_heads * head_dim; split arbitrarily as 1 head × kv_dim_val
        let mut cache = KvCache::new(n_layers, 64, 1, kv_dim_val);
        let mut pos   = CachePosition::new(64);
        for layer in 0..n_layers {
            for p in 0..n_tokens {
                let row: Vec<f32> = (0..kv_dim_val).map(|i| (layer * 100 + p * 10 + i) as f32).collect();
                cache.write_k(layer, p, &row).unwrap();
                cache.write_v(layer, p, &row).unwrap();
            }
        }
        pos.advance(n_tokens).unwrap();
        (cache, pos)
    }

    // ── cache_reset ───────────────────────────────────────────────────────

    #[test]
    fn test_cache_reset_zeros_all_data() {
        let (mut cache, mut pos) = filled_cache(2, 5, 4);
        cache_reset(&mut cache, &mut pos);
        assert_eq!(pos.current(), 0);
        for layer in 0..2 {
            let k = cache.read_k(layer, 4).unwrap();
            assert!(k.as_slice().iter().all(|&v| v == 0.0),
                "layer {layer} K should be zeroed after reset");
        }
    }

    #[test]
    fn test_cache_reset_allows_reuse() {
        let (mut cache, mut pos) = filled_cache(1, 8, 4);
        cache_reset(&mut cache, &mut pos);

        // After reset, can write fresh data and read it back correctly.
        let row = vec![9.0_f32; 4];
        cache.write_k(0, 0, &row).unwrap();
        pos.advance(1).unwrap();

        let k = cache.read_k(0, 1).unwrap();
        assert_eq!(k.as_slice(), row.as_slice());
    }

    // ── cache_truncate ────────────────────────────────────────────────────

    #[test]
    fn test_cache_truncate_rewinds_position() {
        let (mut cache, mut pos) = filled_cache(1, 8, 4);
        cache_truncate(&mut cache, &mut pos, 4).unwrap();
        assert_eq!(pos.current(), 4);
    }

    #[test]
    fn test_cache_truncate_preserves_prefix() {
        let (mut cache, mut pos) = filled_cache(1, 6, 4);
        // Snapshot the first 3 rows before truncation.
        let k_before = cache.read_k(0, 3).unwrap();
        cache_truncate(&mut cache, &mut pos, 3).unwrap();
        let k_after = cache.read_k(0, 3).unwrap();
        assert_eq!(k_before.as_slice(), k_after.as_slice(),
            "prefix rows should be unchanged after truncation");
    }

    #[test]
    fn test_cache_truncate_zeros_evicted_suffix() {
        let (mut cache, mut pos) = filled_cache(1, 6, 4);
        cache_truncate(&mut cache, &mut pos, 3).unwrap();

        // After truncation to 3, the cache is allowed to read up to max_seq_len;
        // positions 3..6 should be zeroed.
        // We temporarily read 6 rows to inspect the suffix.
        let k = cache.read_k(0, 6).unwrap();
        let suffix = &k.as_slice()[3 * 4..6 * 4]; // rows 3, 4, 5
        assert!(suffix.iter().all(|&v| v == 0.0),
            "evicted rows should be zero after truncation, got {:?}", &suffix[..4]);
    }

    #[test]
    fn test_cache_truncate_to_zero() {
        let (mut cache, mut pos) = filled_cache(2, 4, 4);
        cache_truncate(&mut cache, &mut pos, 0).unwrap();
        assert_eq!(pos.current(), 0);
        for layer in 0..2 {
            let k = cache.read_k(layer, 4).unwrap();
            assert!(k.as_slice().iter().all(|&v| v == 0.0));
        }
    }

    #[test]
    fn test_cache_truncate_to_current_is_noop() {
        let (mut cache, mut pos) = filled_cache(1, 5, 4);
        let k_before = cache.read_k(0, 5).unwrap();
        cache_truncate(&mut cache, &mut pos, 5).unwrap();
        let k_after = cache.read_k(0, 5).unwrap();
        assert_eq!(pos.current(), 5);
        assert_eq!(k_before.as_slice(), k_after.as_slice());
    }

    #[test]
    fn test_cache_truncate_beyond_current_rejected() {
        let (mut cache, mut pos) = filled_cache(1, 4, 4);
        assert!(matches!(
            cache_truncate(&mut cache, &mut pos, 5),
            Err(TensorError::InvalidShape { .. })
        ));
        // Position should be unchanged.
        assert_eq!(pos.current(), 4);
    }

    #[test]
    fn test_cache_truncate_then_refill() {
        // Truncate to prefix, then write new tokens — they should be readable.
        let (mut cache, mut pos) = filled_cache(1, 6, 4);
        cache_truncate(&mut cache, &mut pos, 3).unwrap();

        let new_row = vec![42.0_f32; 4];
        cache.write_k(0, 3, &new_row).unwrap();
        pos.advance(1).unwrap();

        let k = cache.read_k(0, 4).unwrap();
        assert_eq!(&k.as_slice()[12..16], new_row.as_slice()); // row 3 at offset 3*4=12
    }
}
