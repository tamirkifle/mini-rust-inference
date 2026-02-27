//! Activation calibration for post-training quantization (PTQ) — commit 14.3.
//!
//! Running a model with `quantize_symmetric` computes the scale from each
//! individual activation tensor.  For production INT8 inference, per-tensor
//! scales should be *stable* across many representative inputs so the same
//! scale can be used at runtime without touching the data distribution.
//!
//! This module provides:
//!
//! - [`ActivationStats`] — running statistics (min, max, abs-max, running
//!   exponential moving average of abs-max) collected over calibration batches.
//! - [`Calibrator`] — accumulates stats for named activation sites across
//!   multiple forward passes, then computes final scales.
//! - [`CalibrationResult`] — per-site scale output, ready to pass to
//!   `quantize_symmetric_with_scale` (or any downstream INT8 path).
//!
//! # Typical calibration flow
//!
//! ```
//! use llm_engine::quant::calibrate::{Calibrator, CalibMethod};
//!
//! let mut cal = Calibrator::new(CalibMethod::MaxAbs);
//!
//! // Simulate two forward passes of sample activations
//! let act1 = vec![0.1_f32, -0.5, 0.3, 1.2, -0.8];
//! let act2 = vec![0.4_f32, -1.0, 0.6, 0.9, -0.2];
//! cal.observe("layer0.ffn", &act1);
//! cal.observe("layer0.ffn", &act2);
//!
//! let result = cal.finalize();
//! let scale = result.scale("layer0.ffn").unwrap();
//! assert!(scale > 0.0);
//! ```

use std::collections::HashMap;

// ── CalibMethod ───────────────────────────────────────────────────────────────

/// Strategy for computing the final per-site quantization scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CalibMethod {
    /// Use the global `max_abs` seen across all calibration batches.
    /// Most conservative — no clipping, highest dynamic range.
    MaxAbs,
    /// Use an exponential moving average of `max_abs` across batches.
    /// `alpha` controls how quickly old observations are forgotten.
    /// Formula: `ema = alpha * batch_max_abs + (1 - alpha) * ema`.
    Ema {
        /// Smoothing factor in `(0, 1]`.  Larger = faster adaptation.
        alpha: f32,
    },
    /// Use a percentile of the absolute-value distribution.
    /// Clips extreme outliers; often improves accuracy on long-tail
    /// distributions.  `percentile` in `(0.0, 100.0]`.
    Percentile {
        /// Percentile of abs values to use as the max (e.g. 99.9).
        percentile: f32,
    },
}

// ── ActivationStats ───────────────────────────────────────────────────────────

/// Running statistics for a single named activation site.
///
/// Updated by [`Calibrator::observe`] on each calibration batch.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Global minimum value seen.
    pub global_min:     f32,
    /// Global maximum value seen.
    pub global_max:     f32,
    /// Global maximum absolute value seen.
    pub global_abs_max: f32,
    /// Exponential moving average of per-batch `max_abs`.
    /// Initialised to the first batch's `max_abs`; updated on subsequent batches.
    pub ema_abs_max:    f32,
    /// EMA smoothing factor (from `CalibMethod::Ema`; stored for incremental update).
    ema_alpha:          f32,
    /// Flat list of all observed absolute values (used for percentile computation).
    /// Kept only when `CalibMethod::Percentile` is selected.
    abs_samples:        Option<Vec<f32>>,
    /// Number of batches observed.
    pub n_batches:      usize,
    /// Total number of scalar elements seen.
    pub n_elements:     usize,
}

impl ActivationStats {
    fn new(method: CalibMethod) -> Self {
        let ema_alpha    = if let CalibMethod::Ema { alpha } = method { alpha } else { 0.1 };
        let abs_samples  = if matches!(method, CalibMethod::Percentile { .. }) { Some(Vec::new()) } else { None };
        Self {
            global_min:     f32::INFINITY,
            global_max:     f32::NEG_INFINITY,
            global_abs_max: 0.0,
            ema_abs_max:    0.0,
            ema_alpha,
            abs_samples,
            n_batches:      0,
            n_elements:     0,
        }
    }

    /// Ingest one batch of activation values.
    pub fn update(&mut self, values: &[f32]) {
        if values.is_empty() { return; }

        let batch_min     = values.iter().cloned().fold(f32::INFINITY,     f32::min);
        let batch_max     = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let batch_abs_max = values.iter().map(|v| v.abs()).fold(0.0_f32,   f32::max);

        self.global_min     = self.global_min.min(batch_min);
        self.global_max     = self.global_max.max(batch_max);
        self.global_abs_max = self.global_abs_max.max(batch_abs_max);

        // EMA update
        if self.n_batches == 0 {
            self.ema_abs_max = batch_abs_max;
        } else {
            let a = self.ema_alpha;
            self.ema_abs_max = a * batch_abs_max + (1.0 - a) * self.ema_abs_max;
        }

        // Accumulate absolute values for percentile
        if let Some(ref mut samples) = self.abs_samples {
            samples.extend(values.iter().map(|v| v.abs()));
        }

        self.n_batches  += 1;
        self.n_elements += values.len();
    }

    /// Compute the final quantization scale using `method`.
    ///
    /// `scale = effective_max_abs / 127.0`
    ///
    /// Returns `1.0` if no data was ever observed (safe: zero activations stay zero).
    #[must_use]
    pub fn compute_scale(&self, method: CalibMethod) -> f32 {
        if self.n_batches == 0 { return 1.0; }

        let effective_max = match method {
            CalibMethod::MaxAbs           => self.global_abs_max,
            CalibMethod::Ema { .. }       => self.ema_abs_max,
            CalibMethod::Percentile { percentile } => {
                if let Some(ref samples) = self.abs_samples {
                    percentile_of_sorted(samples, percentile)
                } else {
                    self.global_abs_max // fallback if not collecting samples
                }
            }
        };

        if effective_max == 0.0 { 1.0 } else { effective_max / 127.0 }
    }
}

/// Compute the given percentile from an unsorted slice of non-negative f32 values.
fn percentile_of_sorted(values: &[f32], p: f32) -> f32 {
    if values.is_empty() { return 0.0; }
    let mut sorted = values.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((p / 100.0) * (sorted.len() - 1) as f32).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

// ── CalibrationResult ─────────────────────────────────────────────────────────

/// Final per-site quantization scales computed by [`Calibrator::finalize`].
#[derive(Debug, Clone)]
pub struct CalibrationResult {
    /// Map from site name to computed scale.
    pub scales: HashMap<String, f32>,
    /// The method used to compute scales.
    pub method: CalibMethod,
}

impl CalibrationResult {
    /// Look up the scale for a named activation site.
    #[must_use]
    pub fn scale(&self, site: &str) -> Option<f32> {
        self.scales.get(site).copied()
    }

    /// Return the number of calibrated sites.
    #[must_use]
    pub fn n_sites(&self) -> usize {
        self.scales.len()
    }
}

// ── Calibrator ────────────────────────────────────────────────────────────────

/// Accumulates activation statistics across forward passes and computes
/// stable per-site quantization scales.
///
/// # Usage
///
/// 1. Run several representative inputs through the model.
/// 2. After each forward pass, call [`observe`] for each activation tensor.
/// 3. Call [`finalize`] to obtain [`CalibrationResult`] with stable scales.
///
/// [`observe`]: Calibrator::observe
/// [`finalize`]: Calibrator::finalize
pub struct Calibrator {
    method: CalibMethod,
    stats:  HashMap<String, ActivationStats>,
}

impl Calibrator {
    /// Create a new calibrator with the given scale-computation method.
    #[must_use]
    pub fn new(method: CalibMethod) -> Self {
        Self { method, stats: HashMap::new() }
    }

    /// Record one activation tensor for `site`.
    ///
    /// Can be called multiple times per site (once per calibration batch).
    pub fn observe(&mut self, site: &str, values: &[f32]) {
        self.stats
            .entry(site.to_string())
            .or_insert_with(|| ActivationStats::new(self.method))
            .update(values);
    }

    /// Return the running statistics for a named site (read-only).
    #[must_use]
    pub fn stats(&self, site: &str) -> Option<&ActivationStats> {
        self.stats.get(site)
    }

    /// Return all sites observed so far.
    #[must_use]
    pub fn site_names(&self) -> Vec<&str> {
        self.stats.keys().map(String::as_str).collect()
    }

    /// Compute final scales for all observed sites and return a [`CalibrationResult`].
    ///
    /// The calibrator can still be used after calling `finalize`; subsequent
    /// `observe` calls will continue accumulating into the same running stats.
    #[must_use]
    pub fn finalize(&self) -> CalibrationResult {
        let scales = self.stats.iter()
            .map(|(name, s)| (name.clone(), s.compute_scale(self.method)))
            .collect();
        CalibrationResult { scales, method: self.method }
    }

    /// Reset all accumulated statistics (start fresh for a new calibration run).
    pub fn reset(&mut self) {
        self.stats.clear();
    }

    /// Number of sites currently tracked.
    #[must_use]
    pub fn n_sites(&self) -> usize { self.stats.len() }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ActivationStats ────────────────────────────────────────────────────

    #[test]
    fn stats_empty_has_safe_defaults() {
        let s = ActivationStats::new(CalibMethod::MaxAbs);
        // No batches → compute_scale should return 1.0 (safe default)
        assert_eq!(s.compute_scale(CalibMethod::MaxAbs), 1.0);
        assert_eq!(s.n_batches, 0);
    }

    #[test]
    fn stats_single_batch_max_abs() {
        let mut s = ActivationStats::new(CalibMethod::MaxAbs);
        s.update(&[0.1, -0.5, 0.3, 1.27, -0.8]);
        assert!((s.global_abs_max - 1.27).abs() < 1e-6);
        let scale = s.compute_scale(CalibMethod::MaxAbs);
        assert!((scale - 1.27 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn stats_multi_batch_max_abs_takes_global() {
        let mut s = ActivationStats::new(CalibMethod::MaxAbs);
        s.update(&[0.1, 0.2, 0.5]);
        s.update(&[1.0, 2.0, 0.3]); // batch 2 has higher max
        s.update(&[0.5, 0.1, 0.9]);
        assert!((s.global_abs_max - 2.0).abs() < 1e-6);
        let scale = s.compute_scale(CalibMethod::MaxAbs);
        assert!((scale - 2.0 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn stats_ema_smooths_over_batches() {
        let alpha = 0.5_f32;
        let mut s = ActivationStats::new(CalibMethod::Ema { alpha });
        s.update(&[1.0_f32; 4]); // batch 1: abs_max = 1.0 → ema = 1.0
        s.update(&[2.0_f32; 4]); // batch 2: abs_max = 2.0 → ema = 0.5*2 + 0.5*1 = 1.5
        let ema = s.ema_abs_max;
        assert!((ema - 1.5).abs() < 1e-5, "expected 1.5, got {ema}");
        let scale = s.compute_scale(CalibMethod::Ema { alpha });
        assert!((scale - 1.5 / 127.0).abs() < 1e-6);
    }

    #[test]
    fn stats_ema_first_batch_initialised_not_zeroed() {
        // alpha=0.5, first batch max_abs=3.0 → ema should be 3.0 (not smoothed vs 0)
        let alpha = 0.5_f32;
        let mut s = ActivationStats::new(CalibMethod::Ema { alpha });
        s.update(&[3.0_f32]);
        assert!((s.ema_abs_max - 3.0).abs() < 1e-6);
    }

    #[test]
    fn stats_percentile_99_clips_outlier() {
        let method = CalibMethod::Percentile { percentile: 50.0 };
        let mut s  = ActivationStats::new(method);
        // 9 values in [0, 1] and one huge outlier
        let vals: Vec<f32> = (0..9).map(|i| i as f32 * 0.1).collect();
        s.update(&vals);
        s.update(&[100.0]); // outlier
        // 50th percentile of {0, 0.1, …, 0.8, 100.0} should be much less than 100.
        let scale = s.compute_scale(method);
        assert!(scale < 1.0, "scale={scale} should be < 1.0 (outlier clipped)");
    }

    #[test]
    fn stats_min_max_tracked() {
        let mut s = ActivationStats::new(CalibMethod::MaxAbs);
        s.update(&[1.0, -3.0, 0.5]);
        s.update(&[4.0, -0.5, 2.0]);
        assert!((s.global_min - (-3.0)).abs() < 1e-6);
        assert!((s.global_max - 4.0).abs()  < 1e-6);
    }

    #[test]
    fn stats_n_elements_counted() {
        let mut s = ActivationStats::new(CalibMethod::MaxAbs);
        s.update(&[1.0; 10]);
        s.update(&[2.0; 5]);
        assert_eq!(s.n_elements, 15);
        assert_eq!(s.n_batches,   2);
    }

    #[test]
    fn stats_all_zeros_gives_scale_one() {
        let mut s = ActivationStats::new(CalibMethod::MaxAbs);
        s.update(&[0.0; 32]);
        assert_eq!(s.compute_scale(CalibMethod::MaxAbs), 1.0);
    }

    // ── Calibrator ─────────────────────────────────────────────────────────

    #[test]
    fn calibrator_registers_sites() {
        let mut cal = Calibrator::new(CalibMethod::MaxAbs);
        cal.observe("attn.q", &[0.5, -0.3, 1.0]);
        cal.observe("ffn.gate", &[0.1, 0.2]);
        assert_eq!(cal.n_sites(), 2);
    }

    #[test]
    fn calibrator_accumulates_across_batches() {
        let mut cal = Calibrator::new(CalibMethod::MaxAbs);
        cal.observe("layer0.q", &[0.5, 0.2]);  // abs_max = 0.5
        cal.observe("layer0.q", &[1.5, 0.1]);  // abs_max = 1.5
        let stats = cal.stats("layer0.q").unwrap();
        assert!((stats.global_abs_max - 1.5).abs() < 1e-6);
    }

    #[test]
    fn calibrator_finalize_returns_all_sites() {
        let mut cal = Calibrator::new(CalibMethod::MaxAbs);
        cal.observe("a", &[1.0]);
        cal.observe("b", &[2.0]);
        cal.observe("c", &[3.0]);
        let result = cal.finalize();
        assert_eq!(result.n_sites(), 3);
        assert!(result.scale("a").is_some());
        assert!(result.scale("b").is_some());
        assert!(result.scale("c").is_some());
    }

    #[test]
    fn calibrator_scale_is_max_abs_over_127() {
        let mut cal = Calibrator::new(CalibMethod::MaxAbs);
        cal.observe("layer0", &[0.0, 2.54, -1.27]);
        let result = cal.finalize();
        let scale  = result.scale("layer0").unwrap();
        assert!((scale - 2.54 / 127.0).abs() < 1e-6,
            "expected {}, got {scale}", 2.54 / 127.0);
    }

    #[test]
    fn calibrator_missing_site_returns_none() {
        let cal = Calibrator::new(CalibMethod::MaxAbs);
        let result = cal.finalize();
        assert!(result.scale("nonexistent").is_none());
    }

    #[test]
    fn calibrator_reset_clears_sites() {
        let mut cal = Calibrator::new(CalibMethod::MaxAbs);
        cal.observe("x", &[1.0]);
        assert_eq!(cal.n_sites(), 1);
        cal.reset();
        assert_eq!(cal.n_sites(), 0);
    }

    #[test]
    fn calibrator_finalize_idempotent() {
        let mut cal = Calibrator::new(CalibMethod::MaxAbs);
        cal.observe("site", &[3.0, -1.5]);
        let r1 = cal.finalize();
        let r2 = cal.finalize();
        assert_eq!(r1.scale("site"), r2.scale("site"));
    }

    #[test]
    fn calibrator_can_observe_after_finalize() {
        let mut cal = Calibrator::new(CalibMethod::MaxAbs);
        cal.observe("x", &[1.0]);
        let _r1 = cal.finalize();
        cal.observe("x", &[5.0]); // extend the same site
        let r2  = cal.finalize();
        // scale should now reflect max(1.0, 5.0) = 5.0
        let scale = r2.scale("x").unwrap();
        assert!((scale - 5.0 / 127.0).abs() < 1e-6);
    }

    // ── percentile helper ──────────────────────────────────────────────────

    #[test]
    fn percentile_100_returns_max() {
        let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let p100 = percentile_of_sorted(&vals, 100.0);
        assert!((p100 - 9.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_0_returns_min() {
        let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let p0 = percentile_of_sorted(&vals, 0.0);
        assert!((p0 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_50_returns_median() {
        let vals = vec![1.0_f32, 3.0, 5.0, 7.0, 9.0]; // already sorted
        let p50 = percentile_of_sorted(&vals, 50.0);
        assert!((p50 - 5.0).abs() < 1e-6);
    }

    #[test]
    fn percentile_of_empty_returns_zero() {
        assert_eq!(percentile_of_sorted(&[], 99.9), 0.0);
    }

    // ── site_names ─────────────────────────────────────────────────────────

    #[test]
    fn site_names_lists_all_observed() {
        let mut cal = Calibrator::new(CalibMethod::MaxAbs);
        cal.observe("alpha", &[1.0]);
        cal.observe("beta",  &[2.0]);
        let mut names = cal.site_names();
        names.sort_unstable();
        assert_eq!(names, vec!["alpha", "beta"]);
    }
}
