//! INT8 symmetric (per-tensor) quantization for activations.
//!
//! # Algorithm
//!
//! Given a float tensor `x`, symmetric quantization maps the range
//! `[-max_abs, +max_abs]` linearly onto `[-127, +127]` (we deliberately
//! avoid –128 to keep the range symmetric and avoid overflow on negation):
//!
//! ```text
//! scale  = max_abs / 127.0          (f32, stored alongside quantized data)
//! q[i]   = round(x[i] / scale)      (i8, clamped to [–127, 127])
//! x̂[i]  = q[i] * scale             (dequantised approximation)
//! ```
//!
//! # Error bounds
//!
//! Worst-case per-element error is `scale / 2 ≈ max_abs / 254`.
//! For a typical activation whose values are normally distributed the
//! average relative error is well under 1 %.

/// Maximum representable INT8 value (avoids –128 for symmetric range).
pub const INT8_MAX: i8 = 127;
/// Minimum representable INT8 value (symmetric counterpart to 127).
pub const INT8_MIN: i8 = -127;

// ── core API ─────────────────────────────────────────────────────────────────

/// Quantises a slice of f32 activations to INT8 using a single per-tensor scale.
///
/// Returns `(quantized_values, scale)`.
/// If all inputs are zero, scale is `1.0` and every quantized value is `0`.
///
/// # Example
/// ```
/// use llm_engine::quant::int8::symmetric::quantize_symmetric;
/// let (q, scale) = quantize_symmetric(&[0.0, 0.5, -1.0, 1.0]);
/// assert_eq!(scale, 1.0 / 127.0);
/// assert_eq!(q[2], -127);
/// assert_eq!(q[3],  127);
/// ```
#[must_use]
pub fn quantize_symmetric(values: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = values
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);

    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let inv_scale = 1.0 / scale;

    let quants = values
        .iter()
        .map(|&v| {
            let q = (v * inv_scale).round();
            q.clamp(-127.0, 127.0) as i8
        })
        .collect();

    (quants, scale)
}

/// Quantises in-place, writing into a caller-supplied `&mut Vec<i8>` buffer.
///
/// Clears and refills `out`; avoids the allocation of [`quantize_symmetric`].
/// Returns the scale factor.
pub fn quantize_symmetric_into(values: &[f32], out: &mut Vec<i8>) -> f32 {
    let max_abs = values
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f32, f32::max);

    let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
    let inv_scale = 1.0 / scale;

    out.clear();
    out.extend(values.iter().map(|&v| {
        let q = (v * inv_scale).round();
        q.clamp(-127.0, 127.0) as i8
    }));

    scale
}

/// Dequantises INT8 values back to f32 using the stored scale.
///
/// # Example
/// ```
/// use llm_engine::quant::int8::symmetric::{quantize_symmetric, dequantize_symmetric};
/// let vals = vec![1.0_f32, -0.5, 0.0, 0.75];
/// let (q, scale) = quantize_symmetric(&vals);
/// let rec = dequantize_symmetric(&q, scale);
/// for (a, b) in vals.iter().zip(&rec) {
///     assert!((a - b).abs() < 0.02);
/// }
/// ```
#[must_use]
pub fn dequantize_symmetric(quants: &[i8], scale: f32) -> Vec<f32> {
    quants.iter().map(|&q| f32::from(q) * scale).collect()
}

/// Dequantises into a caller-supplied buffer.
pub fn dequantize_symmetric_into(quants: &[i8], scale: f32, out: &mut Vec<f32>) {
    out.clear();
    out.extend(quants.iter().map(|&q| f32::from(q) * scale));
}

// ── statistics ────────────────────────────────────────────────────────────────

/// Summary statistics for a quantization operation.
#[derive(Debug, Clone)]
pub struct QuantStats {
    /// Per-tensor scale factor applied.
    pub scale: f32,
    /// Root-mean-square error between original and dequantised values.
    pub rmse: f32,
    /// Maximum absolute error across all elements.
    pub max_abs_err: f32,
    /// Mean relative error (|orig - rec| / |orig|), skipping near-zero originals.
    pub mean_rel_err: f32,
    /// Fraction of elements that clipped (hit ±127).
    pub clip_fraction: f32,
}

/// Quantises `values`, immediately dequantises, and returns both the quantized
/// data and error statistics — useful for calibration and unit-testing.
#[must_use]
pub fn quantize_with_stats(values: &[f32]) -> (Vec<i8>, f32, QuantStats) {
    let (quants, scale) = quantize_symmetric(values);
    let recovered = dequantize_symmetric(&quants, scale);

    let n = values.len() as f32;
    let mut sse = 0.0_f32;
    let mut max_abs_err = 0.0_f32;
    let mut rel_err_sum = 0.0_f32;
    let mut rel_count = 0_usize;
    let mut clip_count = 0_usize;

    for ((&orig, &rec), &q) in values.iter().zip(&recovered).zip(&quants) {
        let err = (orig - rec).abs();
        sse += err * err;
        if err > max_abs_err { max_abs_err = err; }
        let abs_orig = orig.abs();
        if abs_orig > 1e-6 {
            rel_err_sum += err / abs_orig;
            rel_count += 1;
        }
        if q == INT8_MAX || q == INT8_MIN { clip_count += 1; }
    }

    let stats = QuantStats {
        scale,
        rmse: if n > 0.0 { (sse / n).sqrt() } else { 0.0 },
        max_abs_err,
        mean_rel_err: if rel_count > 0 { rel_err_sum / rel_count as f32 } else { 0.0 },
        clip_fraction: if n > 0.0 { clip_count as f32 / n } else { 0.0 },
    };

    (quants, scale, stats)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── scale computation ─────────────────────────────────────────────────

    #[test]
    fn scale_is_max_abs_over_127() {
        let vals = vec![0.0_f32, 2.54, -1.27, 1.0];
        let (_, scale) = quantize_symmetric(&vals);
        let expected = 2.54_f32 / 127.0;
        assert!((scale - expected).abs() < 1e-6, "scale={scale} expected={expected}");
    }

    #[test]
    fn scale_is_one_when_all_zeros() {
        let vals = vec![0.0_f32; 16];
        let (q, scale) = quantize_symmetric(&vals);
        assert_eq!(scale, 1.0);
        assert!(q.iter().all(|&v| v == 0));
    }

    #[test]
    fn scale_handles_negative_max() {
        // max_abs comes from a negative value
        let vals = vec![-3.81_f32, 1.0, -0.5];
        let (_, scale) = quantize_symmetric(&vals);
        let expected = 3.81_f32 / 127.0;
        assert!((scale - expected).abs() < 1e-5);
    }

    // ── quantization values ───────────────────────────────────────────────

    #[test]
    fn max_value_maps_to_127() {
        let vals = vec![1.0_f32, -1.0, 0.5];
        let (q, _) = quantize_symmetric(&vals);
        assert_eq!(q[0],  127);
        assert_eq!(q[1], -127);
    }

    #[test]
    fn zero_maps_to_zero() {
        let vals = vec![0.0_f32, 1.0, -1.0];
        let (q, _) = quantize_symmetric(&vals);
        assert_eq!(q[0], 0);
    }

    #[test]
    fn rounding_is_nearest() {
        // Use a single-element tensor so scale = max_abs/127 = val/127,
        // and inv_scale = 127/max_abs, making q = round(val * 127/max_abs).
        //
        // val = 63.5 / 127  →  scale = 63.5/127 / 127 = 63.5/127²
        //   q = round(63.5/127 * 127²/63.5) = round(127) = 127  ← just the max
        //
        // Better: val=0.5, max_abs=1.0 → scale=1/127, q[0]=round(0.5*127)=round(63.5)=64
        let vals = vec![0.5_f32, 1.0];
        let (q, scale) = quantize_symmetric(&vals);
        // max_abs=1.0 → scale=1/127; q[1] must be 127
        assert_eq!(q[1], 127);
        // q[0] = round(0.5 * 127) = round(63.5); Rust rounds half-away-from-zero → 64
        assert_eq!(q[0], 64, "scale={scale} expected round(63.5)=64");
    }

    #[test]
    fn clamp_prevents_i8_overflow() {
        // Artificially feed a value that would exceed 127 if not clamped
        // (possible if scale is computed externally and is smaller than max_abs/127)
        let vals = vec![100.0_f32];
        let (q, _) = quantize_symmetric(&vals);
        // max_abs = 100 → scale = 100/127, inv = 127/100 = 1.27
        // round(100 * 1.27) = round(127) = 127 — right at the boundary
        assert_eq!(q[0], 127);
    }

    // ── dequantization roundtrip ───────────────────────────────────────────

    #[test]
    fn roundtrip_error_below_half_lsb() {
        let vals: Vec<f32> = (-64..=64).map(|i| i as f32 * 0.05).collect();
        let (q, scale) = quantize_symmetric(&vals);
        let rec = dequantize_symmetric(&q, scale);
        let max_err = vals
            .iter()
            .zip(&rec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // worst-case error = scale/2
        let tolerance = scale / 2.0 + 1e-6;
        assert!(
            max_err <= tolerance,
            "max_err={max_err} > scale/2={}", scale / 2.0
        );
    }

    #[test]
    fn mean_relative_error_below_1_percent_normal_like() {
        // Simulate a rough normal distribution using a deterministic sequence
        let vals: Vec<f32> = (0..256)
            .map(|i| {
                let x = (i as f32) * 0.1 - 12.8;
                // crude bell shape: weight toward center
                x * (-x * x / 50.0).exp()
            })
            .collect();

        let (_, _, stats) = quantize_with_stats(&vals);
        assert!(
            stats.mean_rel_err < 0.01,
            "mean relative error {:.4} >= 1%", stats.mean_rel_err
        );
    }

    // ── buffer reuse variants ──────────────────────────────────────────────

    #[test]
    fn into_variant_matches_allocating() {
        let vals: Vec<f32> = (0..64).map(|i| i as f32 * 0.03 - 1.0).collect();
        let (q_alloc, scale_a) = quantize_symmetric(&vals);
        let mut buf = Vec::new();
        let scale_b = quantize_symmetric_into(&vals, &mut buf);
        assert_eq!(scale_a, scale_b);
        assert_eq!(q_alloc, buf);
    }

    #[test]
    fn deq_into_variant_matches_allocating() {
        let quants: Vec<i8> = (-64..=63).collect();
        let scale = 0.01_f32;
        let alloc = dequantize_symmetric(&quants, scale);
        let mut buf = Vec::new();
        dequantize_symmetric_into(&quants, scale, &mut buf);
        assert_eq!(alloc, buf);
    }

    #[test]
    fn buffer_is_reused_without_leak() {
        let vals1: Vec<f32> = vec![1.0, -1.0, 0.5];
        let vals2: Vec<f32> = vec![2.0, -2.0];
        let mut buf: Vec<i8> = Vec::new();
        quantize_symmetric_into(&vals1, &mut buf);
        assert_eq!(buf.len(), 3);
        quantize_symmetric_into(&vals2, &mut buf);
        assert_eq!(buf.len(), 2); // old data gone
    }

    // ── stats ──────────────────────────────────────────────────────────────

    #[test]
    fn stats_clip_fraction_correct() {
        // Only max/min values clip
        let vals = vec![-1.0_f32, 0.0, 1.0];
        let (_, _, stats) = quantize_with_stats(&vals);
        // Two values (±1.0) map to ±127 → clip_fraction = 2/3
        assert!((stats.clip_fraction - 2.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn stats_rmse_zero_for_zero_input() {
        let vals = vec![0.0_f32; 32];
        let (_, _, stats) = quantize_with_stats(&vals);
        assert!(stats.rmse.abs() < 1e-9);
    }
}
