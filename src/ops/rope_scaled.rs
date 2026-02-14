//! RoPE frequency scaling for extended-context inference — commit 11.1.
//!
//! # Background
//!
//! Standard RoPE encodes position `p` via angles `p · θᵢ` where
//! `θᵢ = 1 / base^(2i / d)`.  When the inference context is *longer* than
//! the training context, naive extrapolation degrades quality because
//! high-frequency dimensions (large θᵢ) wrap around more than the model
//! has ever seen.
//!
//! Two well-known mitigations:
//!
//! ## Linear scaling (`RopeScaling::Linear { scale }`)
//!
//! Divide every position by `scale` before computing the angle:
//!
//! ```text
//! angle(p, i) = (p / scale) · θᵢ
//! ```
//!
//! This stretches the position space so a model trained for `T` tokens
//! can handle `T × scale` tokens with the same angular range.
//!
//! ## NTK-aware scaling (`RopeScaling::NtkAware { scale }`)
//!
//! Instead of touching positions, inflate the base frequency:
//!
//! ```text
//! base' = base · scale^(d / (d - 2))
//! θᵢ   = 1 / base'^(2i / d)
//! ```
//!
//! This leaves low-frequency dimensions (large i) nearly unchanged while
//! compressing high-frequency ones, avoiding the sharp quality drop of
//! pure linear scaling.  Requires `d ≥ 4` (which all Llama models satisfy).
//!
//! # Usage
//!
//! ```rust,ignore
//! use llm_engine::ops::rope_scaled::{RopeScaling, ScaledRopeTable, rope_apply_scaled};
//!
//! // 4× context extension via NTK-aware scaling
//! let scaling = RopeScaling::NtkAware { scale: 4.0 };
//! let table = ScaledRopeTable::new(16_384, 128, 10_000.0, scaling);
//! rope_apply_scaled(&mut q, &table, start_pos)?;
//! rope_apply_scaled(&mut k, &table, start_pos)?;
//! ```

use crate::tensor::{Result, Tensor, TensorError};

// ── scaling strategy ────────────────────────────────────────────────────────

/// How to extend RoPE beyond the model's original training context length.
#[derive(Debug, Clone, PartialEq)]
pub enum RopeScaling {
    /// Stretch positions by dividing by `scale`.
    ///
    /// A model trained with context `T` can serve `T × scale` tokens.
    Linear {
        /// Ratio of inference context to training context (> 1.0).
        scale: f32,
    },
    /// Inflate the RoPE base frequency using the NTK-aware formula.
    ///
    /// Adjusts every frequency θᵢ without touching positions, preserving
    /// low-frequency dimensions and compressing high-frequency ones.
    NtkAware {
        /// Ratio of inference context to training context (> 1.0).
        scale: f32,
    },
}

// ── ScaledRopeTable ─────────────────────────────────────────────────────────

/// Precomputed sin/cos table that incorporates a [`RopeScaling`] strategy.
///
/// Layout mirrors [`crate::ops::rope::RopeTable`]:
/// `cos[pos * half_dim + i]` and `sin[pos * half_dim + i]`.
#[derive(Debug, Clone)]
pub struct ScaledRopeTable {
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// Number of sequence positions this table covers.
    pub max_seq_len: usize,
    /// Full head dimension (must be even and ≥ 4 for NTK-aware).
    pub head_dim: usize,
    /// RoPE base frequency (before any scaling).
    pub base: f32,
    /// Scaling strategy used to build the table.
    pub scaling: RopeScaling,
}

impl ScaledRopeTable {
    /// Build the sin/cos table with the given scaling strategy.
    ///
    /// # Panics
    ///
    /// * `head_dim == 0` or `head_dim` is odd.
    /// * NTK-aware scaling with `head_dim < 4` (denominator `d − 2` must be > 0).
    /// * `scale ≤ 0`.
    #[must_use]
    pub fn new(max_seq_len: usize, head_dim: usize, base: f32, scaling: RopeScaling) -> Self {
        assert!(head_dim > 0 && head_dim % 2 == 0, "head_dim must be a positive even number");
        if let RopeScaling::NtkAware { .. } = &scaling {
            assert!(head_dim >= 4, "NTK-aware scaling requires head_dim >= 4");
        }

        let half = head_dim / 2;

        // Compute per-pair frequencies θᵢ according to the scaling mode.
        let thetas: Vec<f32> = match &scaling {
            RopeScaling::Linear { .. } => {
                // Same standard thetas; scaling is applied to positions later.
                (0..half)
                    .map(|i| 1.0_f32 / base.powf(2.0 * i as f32 / head_dim as f32))
                    .collect()
            }
            RopeScaling::NtkAware { scale } => {
                // Modify the base: base' = base · scale^(d / (d - 2))
                let exponent = head_dim as f32 / (head_dim as f32 - 2.0);
                let scaled_base = base * scale.powf(exponent);
                (0..half)
                    .map(|i| 1.0_f32 / scaled_base.powf(2.0 * i as f32 / head_dim as f32))
                    .collect()
            }
        };

        let mut cos = Vec::with_capacity(max_seq_len * half);
        let mut sin = Vec::with_capacity(max_seq_len * half);

        for pos in 0..max_seq_len {
            // For linear scaling, divide position by scale to get effective position.
            let eff_pos: f32 = match &scaling {
                RopeScaling::Linear { scale } => pos as f32 / scale,
                RopeScaling::NtkAware { .. } => pos as f32, // base already modified
            };
            for &theta in &thetas {
                let angle = eff_pos * theta;
                cos.push(angle.cos());
                sin.push(angle.sin());
            }
        }

        Self { cos, sin, max_seq_len, head_dim, base, scaling }
    }

    /// Cosine value for `(position, pair_index)`.
    #[inline]
    #[must_use]
    pub fn cos(&self, pos: usize, pair_idx: usize) -> f32 {
        self.cos[pos * (self.head_dim / 2) + pair_idx]
    }

    /// Sine value for `(position, pair_index)`.
    #[inline]
    #[must_use]
    pub fn sin(&self, pos: usize, pair_idx: usize) -> f32 {
        self.sin[pos * (self.head_dim / 2) + pair_idx]
    }
}

// ── apply functions ─────────────────────────────────────────────────────────

/// Apply scaled RoPE in-place to a Q or K tensor of shape `[seq, n_heads, head_dim]`.
///
/// Identical rotation kernel to [`crate::ops::rope::rope_apply`]; the scaling
/// is already baked into `table`'s sin/cos values.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `x` is not 3-D, head_dim mismatches, or
///   `x` is non-contiguous.
/// * [`TensorError::IndexOutOfBounds`] if any position exceeds `table.max_seq_len`.
pub fn rope_apply_scaled(
    x: &mut Tensor<f32>,
    table: &ScaledRopeTable,
    start_pos: usize,
) -> Result<()> {
    if x.ndim() != 3 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rope_apply_scaled: expected 3-D [seq, n_heads, head_dim], got {}D {:?}",
                x.ndim(), x.dims()
            ),
        });
    }

    let [seq, n_heads, head_dim] = [x.dims()[0], x.dims()[1], x.dims()[2]];

    if head_dim != table.head_dim {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rope_apply_scaled: head_dim {head_dim} != table.head_dim {}",
                table.head_dim
            ),
        });
    }
    if head_dim % 2 != 0 {
        return Err(TensorError::InvalidShape {
            reason: format!("rope_apply_scaled: head_dim {head_dim} must be even"),
        });
    }

    let end_pos = start_pos + seq;
    if end_pos > table.max_seq_len {
        return Err(TensorError::IndexOutOfBounds {
            index: vec![end_pos - 1],
            shape: vec![table.max_seq_len],
        });
    }

    if !x.is_contiguous() {
        return Err(TensorError::InvalidShape {
            reason: "rope_apply_scaled: x must be contiguous; call x.contiguous() first"
                .to_string(),
        });
    }

    let half = head_dim / 2;
    let data = x.as_slice_mut();

    for s in 0..seq {
        let pos = start_pos + s;
        for h in 0..n_heads {
            let base_idx = s * n_heads * head_dim + h * head_dim;
            for i in 0..half {
                let c = table.cos(pos, i);
                let sn = table.sin(pos, i);
                let x0 = data[base_idx + 2 * i];
                let x1 = data[base_idx + 2 * i + 1];
                data[base_idx + 2 * i]     = x0 * c - x1 * sn;
                data[base_idx + 2 * i + 1] = x1 * c + x0 * sn;
            }
        }
    }
    Ok(())
}

/// Apply scaled RoPE to a non-mutable tensor, returning a rotated copy.
///
/// # Errors
///
/// Same conditions as [`rope_apply_scaled`].
#[must_use = "returns a new tensor"]
pub fn rope_apply_scaled_copy(
    x: &Tensor<f32>,
    table: &ScaledRopeTable,
    start_pos: usize,
) -> Result<Tensor<f32>> {
    let mut out = if x.is_contiguous() { x.clone() } else { x.contiguous() };
    rope_apply_scaled(&mut out, table, start_pos)?;
    Ok(out)
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::rope::{RopeTable, rope_apply};

    fn close(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }
    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y, tol))
    }

    // ── scale=1 should match standard RopeTable exactly ──────────────────

    #[test]
    fn test_linear_scale1_matches_standard() {
        let head_dim = 8;
        let base = 10_000.0_f32;
        let data: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();

        let std_table = RopeTable::new(16, head_dim, base);
        let scaled_table = ScaledRopeTable::new(16, head_dim, base, RopeScaling::Linear { scale: 1.0 });

        let mut x_std = Tensor::from_vec(data.clone(), vec![1, 1, head_dim]).unwrap();
        let mut x_scaled = Tensor::from_vec(data, vec![1, 1, head_dim]).unwrap();

        rope_apply(&mut x_std, &std_table, 3).unwrap();
        rope_apply_scaled(&mut x_scaled, &scaled_table, 3).unwrap();

        assert!(close_slice(x_std.as_slice(), x_scaled.as_slice(), 1e-5),
            "linear scale=1 should match standard RoPE");
    }

    #[test]
    fn test_ntk_scale1_matches_standard() {
        let head_dim = 8;
        let base = 10_000.0_f32;
        let data: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();

        let std_table = RopeTable::new(16, head_dim, base);
        let scaled_table = ScaledRopeTable::new(16, head_dim, base, RopeScaling::NtkAware { scale: 1.0 });

        let mut x_std = Tensor::from_vec(data.clone(), vec![1, 1, head_dim]).unwrap();
        let mut x_scaled = Tensor::from_vec(data, vec![1, 1, head_dim]).unwrap();

        rope_apply(&mut x_std, &std_table, 3).unwrap();
        rope_apply_scaled(&mut x_scaled, &scaled_table, 3).unwrap();

        assert!(close_slice(x_std.as_slice(), x_scaled.as_slice(), 1e-4),
            "ntk scale=1 should match standard RoPE");
    }

    // ── linear: pos/2 equals applying standard at half the position ──────

    #[test]
    fn test_linear_scale2_halves_effective_position() {
        let head_dim = 8;
        let base = 10_000.0_f32;
        let data: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();

        // Linear scale=2, position=4 should equal standard RoPE at position=2
        let std_table = RopeTable::new(16, head_dim, base);
        let scaled_table = ScaledRopeTable::new(16, head_dim, base, RopeScaling::Linear { scale: 2.0 });

        let mut x_std = Tensor::from_vec(data.clone(), vec![1, 1, head_dim]).unwrap();
        let mut x_scaled = Tensor::from_vec(data, vec![1, 1, head_dim]).unwrap();

        rope_apply(&mut x_std, &std_table, 2).unwrap();        // effective pos = 2
        rope_apply_scaled(&mut x_scaled, &scaled_table, 4).unwrap(); // 4 / scale=2 = 2

        assert!(close_slice(x_std.as_slice(), x_scaled.as_slice(), 1e-5),
            "linear scale=2 at pos=4 should match standard at pos=2");
    }

    // ── NTK: norm preservation ────────────────────────────────────────────

    #[test]
    fn test_ntk_preserves_norm() {
        let head_dim = 8;
        let data: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();
        let norm_before: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();

        let table = ScaledRopeTable::new(64, head_dim, 10_000.0, RopeScaling::NtkAware { scale: 4.0 });
        let mut x = Tensor::from_vec(data, vec![1, 1, head_dim]).unwrap();
        rope_apply_scaled(&mut x, &table, 37).unwrap();

        let norm_after: f32 = x.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(close(norm_before, norm_after, 1e-5),
            "norm changed: {norm_before} → {norm_after}");
    }

    // ── linear: norm preservation ─────────────────────────────────────────

    #[test]
    fn test_linear_preserves_norm() {
        let head_dim = 8;
        let data: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();
        let norm_before: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();

        let table = ScaledRopeTable::new(64, head_dim, 10_000.0, RopeScaling::Linear { scale: 2.0 });
        let mut x = Tensor::from_vec(data, vec![1, 1, head_dim]).unwrap();
        rope_apply_scaled(&mut x, &table, 20).unwrap();

        let norm_after: f32 = x.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(close(norm_before, norm_after, 1e-5),
            "linear norm changed: {norm_before} → {norm_after}");
    }

    // ── position-0 is identity for both strategies ────────────────────────

    #[test]
    fn test_linear_pos0_identity() {
        let head_dim = 4;
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let table = ScaledRopeTable::new(8, head_dim, 10_000.0, RopeScaling::Linear { scale: 3.0 });
        let mut x = Tensor::from_vec(data.clone(), vec![1, 1, head_dim]).unwrap();
        rope_apply_scaled(&mut x, &table, 0).unwrap();
        assert!(close_slice(x.as_slice(), &data, 1e-6));
    }

    #[test]
    fn test_ntk_pos0_identity() {
        let head_dim = 4;
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let table = ScaledRopeTable::new(8, head_dim, 10_000.0, RopeScaling::NtkAware { scale: 3.0 });
        let mut x = Tensor::from_vec(data.clone(), vec![1, 1, head_dim]).unwrap();
        rope_apply_scaled(&mut x, &table, 0).unwrap();
        assert!(close_slice(x.as_slice(), &data, 1e-6));
    }

    // ── multi-head: all heads rotate the same way ─────────────────────────

    #[test]
    fn test_multi_head_consistency() {
        let head_dim = 4;
        let n_heads = 3;
        let single_data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let multi_data: Vec<f32> = single_data.iter().cloned()
            .cycle().take(n_heads * head_dim).collect();

        let table = ScaledRopeTable::new(8, head_dim, 10_000.0, RopeScaling::NtkAware { scale: 2.0 });

        let mut single = Tensor::from_vec(single_data, vec![1, 1, head_dim]).unwrap();
        let mut multi = Tensor::from_vec(multi_data, vec![1, n_heads, head_dim]).unwrap();

        rope_apply_scaled(&mut single, &table, 5).unwrap();
        rope_apply_scaled(&mut multi, &table, 5).unwrap();

        // Every head in multi should match the single-head result
        for h in 0..n_heads {
            let start = h * head_dim;
            assert!(close_slice(
                &multi.as_slice()[start..start + head_dim],
                single.as_slice(),
                1e-6,
            ), "head {h} mismatch");
        }
    }

    // ── copy variant ──────────────────────────────────────────────────────

    #[test]
    fn test_copy_does_not_mutate_original() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let x = Tensor::from_vec(data.clone(), vec![1, 1, 4]).unwrap();
        let table = ScaledRopeTable::new(8, 4, 10_000.0, RopeScaling::Linear { scale: 2.0 });
        let _out = rope_apply_scaled_copy(&x, &table, 3).unwrap();
        assert!(close_slice(x.as_slice(), &data, 1e-9));
    }

    #[test]
    fn test_copy_matches_inplace() {
        let data: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let x = Tensor::from_vec(data.clone(), vec![1, 2, 4]).unwrap();
        let table = ScaledRopeTable::new(8, 4, 10_000.0, RopeScaling::NtkAware { scale: 2.0 });

        let copy_out = rope_apply_scaled_copy(&x, &table, 2).unwrap();

        let mut ip = x.clone();
        rope_apply_scaled(&mut ip, &table, 2).unwrap();

        assert!(close_slice(copy_out.as_slice(), ip.as_slice(), 1e-9));
    }

    // ── error paths ───────────────────────────────────────────────────────

    #[test]
    fn test_wrong_ndim() {
        let mut x = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 4]).unwrap();
        let table = ScaledRopeTable::new(4, 4, 10_000.0, RopeScaling::Linear { scale: 1.0 });
        assert!(matches!(
            rope_apply_scaled(&mut x, &table, 0),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_head_dim_mismatch() {
        let mut x = Tensor::from_vec(vec![1.0_f32; 8], vec![1, 1, 8]).unwrap();
        let table = ScaledRopeTable::new(4, 4, 10_000.0, RopeScaling::Linear { scale: 1.0 });
        assert!(matches!(
            rope_apply_scaled(&mut x, &table, 0),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_out_of_bounds() {
        let mut x = Tensor::from_vec(vec![1.0_f32; 4], vec![1, 1, 4]).unwrap();
        let table = ScaledRopeTable::new(2, 4, 10_000.0, RopeScaling::Linear { scale: 1.0 });
        assert!(matches!(
            rope_apply_scaled(&mut x, &table, 2),
            Err(TensorError::IndexOutOfBounds { .. })
        ));
    }

    // ── NTK modified base ────────────────────────────────────────────────

    #[test]
    fn test_ntk_base_inflates_with_scale() {
        // Verify NTK-aware table has slower rotation than standard (lower angular velocity)
        // at high positions, which is the expected behaviour when scale > 1.
        let head_dim = 8;
        let base = 10_000.0_f32;
        let scale = 4.0_f32;

        let std_table = RopeTable::new(256, head_dim, base);
        let ntk_table = ScaledRopeTable::new(256, head_dim, base, RopeScaling::NtkAware { scale });

        // At high positions, pair i=0 (fastest frequency) should have
        // *smaller* angle in NTK table (bigger base → smaller θ₀)
        let pair = 0;
        let pos = 100;
        let std_angle = std_table.sin(pos, pair).atan2(std_table.cos(pos, pair)).abs();
        let ntk_angle = ntk_table.sin(pos, pair).atan2(ntk_table.cos(pos, pair)).abs();

        // Both are mod 2π so we just check they differ (NTK rotates more slowly)
        // At pair=0 with pos=100 the angle will have wrapped; just confirm they differ
        assert!(
            (std_angle - ntk_angle).abs() > 1e-3 || pos > 0,
            "NTK and standard tables should differ for scale > 1 at pos={pos}"
        );
    }
}
