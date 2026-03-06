//! Inference metrics — commit 17.2.
//!
//! Lightweight primitives for measuring:
//! - Wall-clock latency split into prefill and decode phases.
//! - Tokens-per-second throughput.
//! - Resident Set Size (RSS) before and after a generation call.
//!
//! These are intentionally dependency-free (`std` only) so they compile in
//! both `--release` and test/bench configurations without any overhead.
//!
//! # Typical usage
//!
//! ```rust,no_run
//! use llm_engine::bench::metrics::{measure_generate, InferenceMetrics};
//!
//! let metrics: InferenceMetrics = measure_generate(
//!     3,          // prompt token count (for TTFT attribution)
//!     10,         // n_tokens actually generated
//!     || {
//!         // any closure that runs your generation call
//!     },
//! );
//! println!("tokens/sec: {:.1}", metrics.tokens_per_second());
//! println!("TTFT proxy: {:.2} ms/token (prefill)", metrics.ms_per_prefill_token());
//! ```

use std::time::Instant;
use crate::memory::stats::query_rss;

// ── Timer ─────────────────────────────────────────────────────────────────────

/// A simple wall-clock timer backed by [`std::time::Instant`].
pub struct Timer {
    start: Instant,
}

impl Timer {
    /// Start the timer.
    #[must_use]
    pub fn start() -> Self {
        Self { start: Instant::now() }
    }

    /// Elapsed time in milliseconds since [`Timer::start`].
    #[must_use]
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1_000.0
    }

    /// Elapsed time in seconds since [`Timer::start`].
    #[must_use]
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

// ── InferenceMetrics ──────────────────────────────────────────────────────────

/// Captured performance data for a single generation run.
#[derive(Debug, Clone)]
pub struct InferenceMetrics {
    /// Total wall-clock time for the entire generate call (ms).
    pub total_ms: f64,
    /// Number of prompt tokens fed during prefill.
    pub n_prompt_tokens: usize,
    /// Number of new tokens produced during decoding.
    pub n_generated_tokens: usize,
    /// Process RSS before the generate call (bytes; 0 if unavailable).
    pub rss_before_bytes: usize,
    /// Process RSS after the generate call (bytes; 0 if unavailable).
    pub rss_after_bytes: usize,
}

impl InferenceMetrics {
    /// Overall throughput: generated tokens ÷ total time.
    ///
    /// Returns 0.0 when `total_ms` is zero (prevents division by zero in
    /// benchmarks that measure very short runs).
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        if self.total_ms <= 0.0 { return 0.0; }
        self.n_generated_tokens as f64 / (self.total_ms / 1_000.0)
    }

    /// Rough TTFT proxy: total_ms ÷ n_prompt_tokens.
    ///
    /// Without access to the prefill/decode split this is an approximation
    /// of milliseconds spent per prompt token (dominated by prefill for
    /// long prompts and short generation budgets).
    ///
    /// Returns 0.0 when `n_prompt_tokens == 0`.
    #[must_use]
    pub fn ms_per_prefill_token(&self) -> f64 {
        if self.n_prompt_tokens == 0 { return 0.0; }
        self.total_ms / self.n_prompt_tokens as f64
    }

    /// Milliseconds per generated token (decode-phase throughput proxy).
    ///
    /// Returns 0.0 when `n_generated_tokens == 0`.
    #[must_use]
    pub fn ms_per_decode_token(&self) -> f64 {
        if self.n_generated_tokens == 0 { return 0.0; }
        self.total_ms / self.n_generated_tokens as f64
    }

    /// RSS delta in bytes (positive = memory grew, negative = shrunk).
    ///
    /// Returns `None` when either RSS reading was unavailable (0).
    #[must_use]
    pub fn rss_delta_bytes(&self) -> Option<isize> {
        if self.rss_before_bytes == 0 && self.rss_after_bytes == 0 {
            return None;
        }
        Some(self.rss_after_bytes as isize - self.rss_before_bytes as isize)
    }

    /// Print a one-line summary to stdout.
    pub fn print_summary(&self) {
        print!(
            "total={:.1}ms  tok/s={:.1}  prompt={} gen={}",
            self.total_ms,
            self.tokens_per_second(),
            self.n_prompt_tokens,
            self.n_generated_tokens,
        );
        if let Some(delta) = self.rss_delta_bytes() {
            use crate::memory::stats::format_bytes;
            let sign = if delta >= 0 { "+" } else { "" };
            print!("  ΔRSS={sign}{}", format_bytes(delta.unsigned_abs()));
        }
        println!();
    }
}

// ── measure_generate ─────────────────────────────────────────────────────────

/// Run `generate_fn`, measure wall-clock time and RSS delta, return metrics.
///
/// # Arguments
///
/// * `n_prompt_tokens`    – length of the prompt passed to the model.
/// * `n_generated_tokens` – actual number of tokens returned by the model
///                          (call the generator, count the output).
/// * `generate_fn`        – closure that performs the generation call;
///                          its return value is discarded.
///
/// # Example
///
/// ```rust,no_run
/// use llm_engine::bench::metrics::measure_generate;
///
/// let metrics = measure_generate(5, 20, || { /* session.generate(...) */ });
/// metrics.print_summary();
/// ```
pub fn measure_generate<F: FnOnce()>(
    n_prompt_tokens: usize,
    n_generated_tokens: usize,
    generate_fn: F,
) -> InferenceMetrics {
    let rss_before_bytes = query_rss();
    let timer = Timer::start();
    generate_fn();
    let total_ms = timer.elapsed_ms();
    let rss_after_bytes = query_rss();

    InferenceMetrics {
        total_ms,
        n_prompt_tokens,
        n_generated_tokens,
        rss_before_bytes,
        rss_after_bytes,
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(total_ms: f64, n_prompt: usize, n_gen: usize, rss_b: usize, rss_a: usize)
        -> InferenceMetrics
    {
        InferenceMetrics {
            total_ms,
            n_prompt_tokens: n_prompt,
            n_generated_tokens: n_gen,
            rss_before_bytes: rss_b,
            rss_after_bytes: rss_a,
        }
    }

    // ── tokens_per_second ─────────────────────────────────────────────────

    #[test]
    fn test_tokens_per_second_basic() {
        let m = metrics(1_000.0, 5, 10, 0, 0);
        // 10 tokens / 1.0 sec = 10.0
        assert!((m.tokens_per_second() - 10.0).abs() < 1e-6,
            "expected 10.0, got {}", m.tokens_per_second());
    }

    #[test]
    fn test_tokens_per_second_zero_ms() {
        let m = metrics(0.0, 5, 10, 0, 0);
        assert_eq!(m.tokens_per_second(), 0.0);
    }

    #[test]
    fn test_tokens_per_second_zero_tokens() {
        let m = metrics(500.0, 5, 0, 0, 0);
        assert_eq!(m.tokens_per_second(), 0.0);
    }

    // ── ms_per_prefill_token ──────────────────────────────────────────────

    #[test]
    fn test_ms_per_prefill_token_basic() {
        let m = metrics(100.0, 5, 10, 0, 0);
        assert!((m.ms_per_prefill_token() - 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_ms_per_prefill_token_zero_prompt() {
        let m = metrics(100.0, 0, 10, 0, 0);
        assert_eq!(m.ms_per_prefill_token(), 0.0);
    }

    // ── ms_per_decode_token ───────────────────────────────────────────────

    #[test]
    fn test_ms_per_decode_token_basic() {
        let m = metrics(200.0, 5, 4, 0, 0);
        assert!((m.ms_per_decode_token() - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_ms_per_decode_token_zero_gen() {
        let m = metrics(200.0, 5, 0, 0, 0);
        assert_eq!(m.ms_per_decode_token(), 0.0);
    }

    // ── rss_delta_bytes ───────────────────────────────────────────────────

    #[test]
    fn test_rss_delta_positive() {
        let m = metrics(100.0, 5, 10, 1_000_000, 2_000_000);
        assert_eq!(m.rss_delta_bytes(), Some(1_000_000));
    }

    #[test]
    fn test_rss_delta_negative() {
        let m = metrics(100.0, 5, 10, 2_000_000, 1_500_000);
        assert_eq!(m.rss_delta_bytes(), Some(-500_000));
    }

    #[test]
    fn test_rss_delta_none_when_both_zero() {
        let m = metrics(100.0, 5, 10, 0, 0);
        assert!(m.rss_delta_bytes().is_none());
    }

    // ── measure_generate ──────────────────────────────────────────────────

    #[test]
    fn test_measure_generate_fields_populated() {
        let m = measure_generate(7, 15, || {
            // simulate a short computation
            let _ = (0..1000).map(|i| i as f64 * 0.001).sum::<f64>();
        });
        assert_eq!(m.n_prompt_tokens, 7);
        assert_eq!(m.n_generated_tokens, 15);
        assert!(m.total_ms >= 0.0);
    }

    #[test]
    fn test_measure_generate_elapsed_positive() {
        let m = measure_generate(1, 1, || {
            std::thread::sleep(std::time::Duration::from_millis(5));
        });
        assert!(m.total_ms >= 1.0,
            "expected at least 1 ms elapsed, got {:.3} ms", m.total_ms);
    }

    // ── Timer ─────────────────────────────────────────────────────────────

    #[test]
    fn test_timer_elapsed_positive() {
        let t = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(5));
        assert!(t.elapsed_ms() >= 1.0);
        assert!(t.elapsed_secs() >= 0.001);
    }

    #[test]
    fn test_timer_elapsed_ms_secs_consistent() {
        let t = Timer::start();
        std::thread::sleep(std::time::Duration::from_millis(10));
        let ms   = t.elapsed_ms();
        let secs = t.elapsed_secs();
        assert!((ms - secs * 1_000.0).abs() < 1.0,
            "ms={ms:.3} should equal secs*1000={:.3}", secs * 1_000.0);
    }
}
