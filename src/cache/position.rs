//! Cache position tracking — commit 9.3.
//!
//! `CachePosition` is a lightweight cursor that tracks how many tokens have
//! been written to a [`super::KvCache`] during a single generation session.
//!
//! It enforces the invariant `pos <= max_seq_len` so that the cache can never
//! be advanced past its pre-allocated capacity.
//!
//! # Typical lifecycle
//!
//! ```text
//! let mut pos = CachePosition::new(max_seq_len);
//!
//! // Prefill: process all prompt tokens at once
//! pos.advance(prompt_len)?;
//!
//! // Decode loop: one token per step
//! while !pos.is_full() {
//!     let token_pos = pos.current();
//!     // ... run cached attention at token_pos ...
//!     pos.advance(1)?;
//! }
//!
//! // New sequence
//! pos.reset();
//! ```

use crate::tensor::{Result, TensorError};

/// Cursor that tracks the current write position in a KV-cache.
///
/// `pos` represents the number of tokens whose K/V projections have been
/// stored: the **next** write should go at index `pos`.
#[derive(Debug, Clone)]
pub struct CachePosition {
    pos:         usize,
    max_seq_len: usize,
}

impl CachePosition {
    /// Create a new `CachePosition` starting at 0.
    #[must_use]
    pub fn new(max_seq_len: usize) -> Self {
        Self { pos: 0, max_seq_len }
    }

    // ── readers ───────────────────────────────────────────────────────────

    /// Current sequence position (tokens already in cache).
    #[must_use]
    pub fn current(&self) -> usize { self.pos }

    /// Maximum number of tokens this cache session can hold.
    #[must_use]
    pub fn max_seq_len(&self) -> usize { self.max_seq_len }

    /// `true` if the cache is at capacity — no more tokens can be added.
    #[must_use]
    pub fn is_full(&self) -> bool { self.pos >= self.max_seq_len }

    /// Number of token positions remaining before the cache is full.
    #[must_use]
    pub fn remaining(&self) -> usize { self.max_seq_len.saturating_sub(self.pos) }

    // ── mutation ──────────────────────────────────────────────────────────

    /// Advance the position by `n` tokens.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if `self.pos + n > max_seq_len`.
    pub fn advance(&mut self, n: usize) -> Result<()> {
        let next = self.pos + n;
        if next > self.max_seq_len {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "CachePosition: advance by {n} would overflow (pos={}, max={})",
                    self.pos, self.max_seq_len
                ),
            });
        }
        self.pos = next;
        Ok(())
    }

    /// Reset position to 0 (used before processing a new sequence).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Set position to an arbitrary value `≤ max_seq_len`.
    ///
    /// Used by cache management operations (e.g. `cache_truncate`).
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if `new_pos > max_seq_len`.
    pub fn set(&mut self, new_pos: usize) -> Result<()> {
        if new_pos > self.max_seq_len {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "CachePosition::set: new_pos {new_pos} > max_seq_len {}",
                    self.max_seq_len
                ),
            });
        }
        self.pos = new_pos;
        Ok(())
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_position_initial_state() {
        let p = CachePosition::new(128);
        assert_eq!(p.current(),     0);
        assert_eq!(p.max_seq_len(), 128);
        assert_eq!(p.remaining(),   128);
        assert!(!p.is_full());
    }

    #[test]
    fn test_cache_position_advance() {
        let mut p = CachePosition::new(64);
        p.advance(10).unwrap();
        assert_eq!(p.current(),   10);
        assert_eq!(p.remaining(), 54);
        assert!(!p.is_full());
    }

    #[test]
    fn test_cache_position_advance_to_full() {
        let mut p = CachePosition::new(8);
        p.advance(8).unwrap();
        assert_eq!(p.current(), 8);
        assert_eq!(p.remaining(), 0);
        assert!(p.is_full());
    }

    #[test]
    fn test_cache_position_advance_overflow_rejected() {
        let mut p = CachePosition::new(8);
        p.advance(7).unwrap();
        // 7 + 2 = 9 > 8
        assert!(matches!(
            p.advance(2),
            Err(TensorError::InvalidShape { .. })
        ));
        // position must not have changed
        assert_eq!(p.current(), 7);
    }

    #[test]
    fn test_cache_position_reset() {
        let mut p = CachePosition::new(32);
        p.advance(20).unwrap();
        p.reset();
        assert_eq!(p.current(),   0);
        assert_eq!(p.remaining(), 32);
        assert!(!p.is_full());
    }

    #[test]
    fn test_cache_position_advance_one_by_one() {
        let mut p = CachePosition::new(4);
        for expected in 1..=4_usize {
            p.advance(1).unwrap();
            assert_eq!(p.current(), expected);
        }
        assert!(p.is_full());
    }

    #[test]
    fn test_cache_position_set_valid() {
        let mut p = CachePosition::new(64);
        p.advance(30).unwrap();
        p.set(15).unwrap();
        assert_eq!(p.current(), 15);
    }

    #[test]
    fn test_cache_position_set_to_max() {
        let mut p = CachePosition::new(64);
        p.set(64).unwrap();
        assert!(p.is_full());
    }

    #[test]
    fn test_cache_position_set_overflow_rejected() {
        let mut p = CachePosition::new(64);
        assert!(matches!(
            p.set(65),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_cache_position_remaining_after_advance() {
        let mut p = CachePosition::new(100);
        p.advance(37).unwrap();
        assert_eq!(p.remaining(), 63);
    }
}
