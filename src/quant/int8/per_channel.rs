//! INT8 per-channel (per-row) weight quantization.
//!
//! # Why per-channel?
//!
//! Transformer weight rows often have very different dynamic ranges:
//! a single per-tensor scale forces every row to share the worst-case
//! range, wasting bits on small-magnitude rows.  Per-channel assigns
//! each output row its own scale, reducing average quantization error
//! by roughly 2–4× compared to per-tensor on typical LLM weights.
//!
//! # Algorithm
//!
//! For a weight matrix `W` of shape `[N, K]`:
//!
//! ```text
//! scale[n]  = max_abs(W[n, :]) / 127.0     for n in 0..N
//! q[n, k]   = round(W[n, k] / scale[n])    clamped to [–127, 127]
//! Ŵ[n, k]  = q[n, k] * scale[n]
//! ```
//!
//! # Storage layout
//!
//! `QuantizedMatrix` stores:
//! - `data: Vec<i8>`    — row-major INT8 values, shape `[N, K]`
//! - `scales: Vec<f32>` — one scale per output row, length `N`
//! - `n_out: usize`     — number of output channels (rows)
//! - `k_in: usize`      — inner dimension (columns)

/// A weight matrix quantized to INT8 with one scale per output channel.
#[derive(Debug, Clone)]
pub struct QuantizedMatrix {
    /// Row-major INT8 weights: index `[n, k]` → `data[n * k_in + k]`.
    pub data: Vec<i8>,
    /// Per-row scale factors (length == `n_out`).
    pub scales: Vec<f32>,
    /// Number of output channels (rows).
    pub n_out: usize,
    /// Inner / input dimension (columns).
    pub k_in: usize,
}

impl QuantizedMatrix {
    /// Returns the INT8 row slice for output channel `n`.
    #[must_use]
    pub fn row(&self, n: usize) -> &[i8] {
        let start = n * self.k_in;
        &self.data[start..start + self.k_in]
    }

    /// Dequantises row `n` back to f32.
    #[must_use]
    pub fn dequantize_row(&self, n: usize) -> Vec<f32> {
        let scale = self.scales[n];
        self.row(n).iter().map(|&q| f32::from(q) * scale).collect()
    }

    /// Dequantises all rows and returns a `[n_out, k_in]` row-major f32 matrix.
    #[must_use]
    pub fn dequantize_all(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_out * self.k_in);
        for n in 0..self.n_out {
            let scale = self.scales[n];
            for &q in self.row(n) {
                out.push(f32::from(q) * scale);
            }
        }
        out
    }
}

// ── quantization API ──────────────────────────────────────────────────────────

/// Quantises a `[N, K]` row-major f32 weight matrix to INT8 per-channel.
///
/// Each row gets its own scale `max_abs(row) / 127`.
/// Rows that are all-zero get `scale = 1.0` and quantize to all zeros.
///
/// # Panics
///
/// Panics if `weights.len() != n_out * k_in`.
///
/// # Example
///
/// ```
/// use llm_engine::quant::int8::per_channel::quantize_per_channel;
///
/// let weights = vec![
///     1.0_f32, -2.0, 3.0,  // row 0: max_abs = 3.0 → scale = 3/127
///     0.1_f32,  0.2, 0.3,  // row 1: max_abs = 0.3 → scale = 0.3/127
/// ];
/// let qm = quantize_per_channel(&weights, 2, 3);
/// assert_eq!(qm.n_out, 2);
/// assert_eq!(qm.k_in, 3);
/// assert!((qm.scales[0] - 3.0 / 127.0).abs() < 1e-6);
/// assert!((qm.scales[1] - 0.3 / 127.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn quantize_per_channel(weights: &[f32], n_out: usize, k_in: usize) -> QuantizedMatrix {
    assert_eq!(
        weights.len(),
        n_out * k_in,
        "weights.len()={} != n_out*k_in={}",
        weights.len(),
        n_out * k_in
    );

    let mut data = Vec::with_capacity(n_out * k_in);
    let mut scales = Vec::with_capacity(n_out);

    for n in 0..n_out {
        let row = &weights[n * k_in..(n + 1) * k_in];
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let inv = 1.0 / scale;
        scales.push(scale);
        for &w in row {
            let q = (w * inv).round().clamp(-127.0, 127.0) as i8;
            data.push(q);
        }
    }

    QuantizedMatrix { data, scales, n_out, k_in }
}

/// Quantises into caller-supplied storage to avoid allocation.
///
/// `data_out` and `scales_out` are cleared and filled with the result.
pub fn quantize_per_channel_into(
    weights: &[f32],
    n_out: usize,
    k_in: usize,
    data_out: &mut Vec<i8>,
    scales_out: &mut Vec<f32>,
) {
    assert_eq!(weights.len(), n_out * k_in);
    data_out.clear();
    data_out.reserve(n_out * k_in);
    scales_out.clear();
    scales_out.reserve(n_out);

    for n in 0..n_out {
        let row = &weights[n * k_in..(n + 1) * k_in];
        let max_abs = row.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
        let inv = 1.0 / scale;
        scales_out.push(scale);
        for &w in row {
            data_out.push((w * inv).round().clamp(-127.0, 127.0) as i8);
        }
    }
}

// ── statistics ────────────────────────────────────────────────────────────────

/// Per-channel quantization error summary.
#[derive(Debug, Clone)]
pub struct PerChannelStats {
    /// Per-row RMSE between original and dequantized weights.
    pub row_rmse: Vec<f32>,
    /// Per-row max-absolute-error.
    pub row_max_err: Vec<f32>,
    /// Overall mean relative error across all elements.
    pub mean_rel_err: f32,
    /// Overall RMSE across the full matrix.
    pub global_rmse: f32,
}

/// Quantises weights and computes error statistics.
#[must_use]
pub fn quantize_with_stats(
    weights: &[f32],
    n_out: usize,
    k_in: usize,
) -> (QuantizedMatrix, PerChannelStats) {
    let qm = quantize_per_channel(weights, n_out, k_in);

    let mut row_rmse = Vec::with_capacity(n_out);
    let mut row_max_err = Vec::with_capacity(n_out);
    let mut global_sse = 0.0_f32;
    let mut rel_sum = 0.0_f32;
    let mut rel_count = 0_usize;

    for n in 0..n_out {
        let orig = &weights[n * k_in..(n + 1) * k_in];
        let scale = qm.scales[n];
        let row_q = qm.row(n);
        let mut sse = 0.0_f32;
        let mut max_err = 0.0_f32;
        for (&o, &q) in orig.iter().zip(row_q) {
            let rec = f32::from(q) * scale;
            let err = (o - rec).abs();
            sse += err * err;
            if err > max_err { max_err = err; }
            if o.abs() > 1e-6 { rel_sum += err / o.abs(); rel_count += 1; }
        }
        global_sse += sse;
        row_rmse.push((sse / k_in as f32).sqrt());
        row_max_err.push(max_err);
    }

    let total = (n_out * k_in) as f32;
    let stats = PerChannelStats {
        row_rmse,
        row_max_err,
        mean_rel_err: if rel_count > 0 { rel_sum / rel_count as f32 } else { 0.0 },
        global_rmse: if total > 0.0 { (global_sse / total).sqrt() } else { 0.0 },
    };
    (qm, stats)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }

    // ── scale correctness ─────────────────────────────────────────────────

    #[test]
    fn each_row_scale_is_max_abs_over_127() {
        let w = vec![
            1.0_f32, -2.0, 3.0,   // row 0: max_abs=3 → 3/127
            0.5_f32,  0.0, 0.25,  // row 1: max_abs=0.5 → 0.5/127
        ];
        let qm = quantize_per_channel(&w, 2, 3);
        assert!(approx(qm.scales[0], 3.0 / 127.0, EPS));
        assert!(approx(qm.scales[1], 0.5 / 127.0, EPS));
    }

    #[test]
    fn zero_row_gets_scale_one() {
        let w = vec![0.0_f32; 6]; // 2×3 zeros
        let qm = quantize_per_channel(&w, 2, 3);
        assert_eq!(qm.scales[0], 1.0);
        assert_eq!(qm.scales[1], 1.0);
        assert!(qm.data.iter().all(|&q| q == 0));
    }

    #[test]
    fn max_abs_element_maps_to_127() {
        let w = vec![-5.0_f32, 1.0, 2.5, 5.0];
        let qm = quantize_per_channel(&w, 1, 4);
        // max_abs = 5 → scale = 5/127; -5.0 * 127/5 = -127; 5.0 * 127/5 = 127
        assert_eq!(qm.data[0], -127);
        assert_eq!(qm.data[3],  127);
    }

    // ── independent rows ──────────────────────────────────────────────────

    #[test]
    fn rows_are_quantised_independently() {
        // Row 0 has large range, row 1 has small range
        let w = vec![
            0.0_f32, 0.0, 127.0,   // row 0: scale = 127/127 = 1.0
            0.0_f32, 0.0,   1.0,   // row 1: scale = 1/127
        ];
        let qm = quantize_per_channel(&w, 2, 3);
        // Row 0 last element → round(127 / 1.0) = 127
        assert_eq!(qm.row(0)[2], 127);
        // Row 1 last element → round(1.0 * 127) = 127
        assert_eq!(qm.row(1)[2], 127);
        // Scales differ
        assert!(approx(qm.scales[0], 1.0, EPS));
        assert!(approx(qm.scales[1], 1.0 / 127.0, EPS));
    }

    // ── roundtrip error ───────────────────────────────────────────────────

    #[test]
    fn roundtrip_error_within_half_lsb_per_row() {
        let w: Vec<f32> = (0..128).map(|i| (i as f32) * 0.03 - 2.0).collect();
        let qm = quantize_per_channel(&w, 4, 32); // 4 rows × 32 cols
        for n in 0..4 {
            let orig = &w[n * 32..(n + 1) * 32];
            let rec = qm.dequantize_row(n);
            let scale = qm.scales[n];
            let max_err = orig.iter().zip(&rec)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                max_err <= scale / 2.0 + 1e-6,
                "row {n}: max_err={max_err} > scale/2={}", scale / 2.0
            );
        }
    }

    #[test]
    fn dequantize_all_matches_row_by_row() {
        let w: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.2).collect();
        let qm = quantize_per_channel(&w, 8, 8);
        let all = qm.dequantize_all();
        for n in 0..8 {
            let row = qm.dequantize_row(n);
            assert_eq!(&all[n * 8..(n + 1) * 8], row.as_slice());
        }
    }

    #[test]
    fn per_channel_beats_per_tensor_on_varied_rows() {
        use crate::quant::int8::symmetric::quantize_symmetric;

        // Mix of very different row magnitudes
        let mut w = Vec::new();
        for n in 0..8_usize {
            let mag = 2.0_f32.powi(n as i32); // 1, 2, 4, … 128
            for k in 0..32_usize {
                w.push(mag * (k as f32 / 32.0 - 0.5));
            }
        }
        let n_out = 8;
        let k_in = 32;

        // Per-channel RMSE
        let (_, pc_stats) = quantize_with_stats(&w, n_out, k_in);

        // Per-tensor RMSE (baseline)
        let (_, pt_scale) = quantize_symmetric(&w);
        let pt_rmse = {
            let inv = 1.0 / pt_scale;
            let sse: f32 = w.iter().map(|&v| {
                let rec = (v * inv).round().clamp(-127.0, 127.0) as i8;
                let diff = v - f32::from(rec) * pt_scale;
                diff * diff
            }).sum();
            (sse / w.len() as f32).sqrt()
        };

        assert!(
            pc_stats.global_rmse < pt_rmse,
            "per-channel RMSE ({}) should be < per-tensor RMSE ({})",
            pc_stats.global_rmse, pt_rmse
        );
    }

    // ── buffer-reuse variant ──────────────────────────────────────────────

    #[test]
    fn into_variant_matches_allocating() {
        let w: Vec<f32> = (0..96).map(|i| i as f32 * 0.02 - 1.0).collect();
        let expected = quantize_per_channel(&w, 3, 32);
        let mut data_buf: Vec<i8> = Vec::new();
        let mut scale_buf: Vec<f32> = Vec::new();
        quantize_per_channel_into(&w, 3, 32, &mut data_buf, &mut scale_buf);
        assert_eq!(expected.data, data_buf);
        assert_eq!(expected.scales, scale_buf);
    }

    // ── stats ──────────────────────────────────────────────────────────────

    #[test]
    fn stats_zero_rmse_for_exactly_representable() {
        // max_abs = 127.0 → scale = 1.0.  Integer values in [0, 127] round-trip
        // with exactly zero error because round(v / 1.0) == v for all integers.
        let w: Vec<f32> = (0..=127).map(|i| i as f32).collect();
        let (_, stats) = quantize_with_stats(&w, 1, 128);
        assert!(
            stats.global_rmse < 1e-5,
            "expected ~0 rmse, got {}", stats.global_rmse
        );
    }

    #[test]
    fn mean_rel_err_below_1_percent() {
        let w: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.8).collect();
        let (_, stats) = quantize_with_stats(&w, 8, 32);
        assert!(
            stats.mean_rel_err < 0.01,
            "mean rel err {:.4} >= 1%", stats.mean_rel_err
        );
    }
}
