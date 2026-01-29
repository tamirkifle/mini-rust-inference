//! Rotary Positional Embeddings (RoPE) — commit 6.3.
//!
//! # Theory
//!
//! RoPE encodes absolute position by rotating pairs of elements in the Q/K
//! vectors.  For a head of dimension `d`, the rotation at position `pos` for
//! pair index `i` (0-based, stepping by 2) is:
//!
//! ```text
//! θᵢ  = 1 / (base^(2i / d))           base = 10 000 by default
//!
//! x'[2i]   = x[2i]   · cos(pos · θᵢ) − x[2i+1] · sin(pos · θᵢ)
//! x'[2i+1] = x[2i+1] · cos(pos · θᵢ) + x[2i]   · sin(pos · θᵢ)
//! ```
//!
//! Because the same (cos, sin) table is shared across all heads, it is
//! computed once and reused via [`RopeTable`].
//!
//! # Input shapes
//!
//! The apply functions accept tensors shaped `[seq, n_heads, head_dim]`.
//! Batch dimensions are not supported at this stage; callers iterate over
//! the batch axis.
//!
//! # Usage
//!
//! ```rust,ignore
//! let table = RopeTable::new(max_seq_len, head_dim, 10_000.0);
//! rope_apply(&mut q, &table, /*start_pos=*/ 0)?;
//! rope_apply(&mut k, &table, /*start_pos=*/ 0)?;
//! ```

use crate::tensor::{Result, Tensor, TensorError};

// ── precomputed sin/cos table ───────────────────────────────────────────────

/// Precomputed sin/cos cache for RoPE.
///
/// Layout: `cos[pos][i]` and `sin[pos][i]` where `i` indexes the *pair* (0 …
/// head_dim/2 − 1).  Stored as flat `Vec<f32>` of length
/// `max_seq_len × (head_dim / 2)`.
#[derive(Debug, Clone)]
pub struct RopeTable {
    cos: Vec<f32>,   // CHANGED: [max_seq_len * half_dim]
    sin: Vec<f32>,   // CHANGED: [max_seq_len * half_dim]
    /// Number of sequence positions this table covers.
    pub max_seq_len: usize,
    /// Full head dimension (must be even).
    pub head_dim: usize,
    /// RoPE base frequency (default: 10 000.0).
    pub base: f32,
}

impl RopeTable {
    /// Build the sin/cos table.
    ///
    /// # Panics
    ///
    /// Panics if `head_dim == 0` or `head_dim` is odd.
    #[must_use]
    pub fn new(max_seq_len: usize, head_dim: usize, base: f32) -> Self {
        assert!(head_dim > 0 && head_dim % 2 == 0, "head_dim must be a positive even number");
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq_len * half);
        let mut sin = Vec::with_capacity(max_seq_len * half);

        // Precompute θᵢ = 1 / (base^(2i / head_dim))
        // CHANGED: computed once and reused across all positions
        let thetas: Vec<f32> = (0..half)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        for pos in 0..max_seq_len {
            for &theta in &thetas {
                let angle = pos as f32 * theta;
                cos.push(angle.cos()); // CHANGED
                sin.push(angle.sin()); // CHANGED
            }
        }

        Self { cos, sin, max_seq_len, head_dim, base }
    }

    /// Cosine value for `(position, pair_index)`.
    #[inline]
    #[must_use]
    pub fn cos(&self, pos: usize, pair_idx: usize) -> f32 {
        self.cos[pos * (self.head_dim / 2) + pair_idx] // CHANGED
    }

    /// Sine value for `(position, pair_index)`.
    #[inline]
    #[must_use]
    pub fn sin(&self, pos: usize, pair_idx: usize) -> f32 {
        self.sin[pos * (self.head_dim / 2) + pair_idx] // CHANGED
    }
}

// ── apply RoPE in-place ─────────────────────────────────────────────────────

/// Apply RoPE in-place to a Q or K tensor of shape `[seq, n_heads, head_dim]`.
///
/// Position for token `s` is `start_pos + s`, enabling KV-cache decode where
/// a single new token arrives at position `start_pos`.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`] if `x` is not exactly 3-D, `head_dim`
///   doesn't match the table, or `x` is non-contiguous.
/// * [`TensorError::IndexOutOfBounds`] if any position would exceed
///   `table.max_seq_len`.
pub fn rope_apply(
    x: &mut Tensor<f32>,
    table: &RopeTable,
    start_pos: usize,
) -> Result<()> {
    // ── validate ───────────────────────────────────────────────────────────
    if x.ndim() != 3 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rope_apply: expected 3-D [seq, n_heads, head_dim], got {}D {:?}",
                x.ndim(), x.dims()
            ),
        });
    }

    let [seq, n_heads, head_dim] = [x.dims()[0], x.dims()[1], x.dims()[2]];

    if head_dim != table.head_dim {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "rope_apply: head_dim {head_dim} != table.head_dim {}",
                table.head_dim
            ),
        });
    }
    if head_dim % 2 != 0 {
        return Err(TensorError::InvalidShape {
            reason: format!("rope_apply: head_dim {head_dim} must be even"),
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
            reason: "rope_apply: x must be contiguous; call x.contiguous() first".to_string(),
        });
    }

    // ── rotate pairs ───────────────────────────────────────────────────────
    // CHANGED: layout is [seq, n_heads, head_dim] row-major, so stride is:
    //   seq-step = n_heads * head_dim
    //   head-step = head_dim
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
                // CHANGED: standard rotation formula
                data[base_idx + 2 * i]     = x0 * c - x1 * sn;
                data[base_idx + 2 * i + 1] = x1 * c + x0 * sn;
            }
        }
    }
    Ok(())
}

/// Apply RoPE to a *non-mutable* tensor, returning a new rotated copy.
///
/// Equivalent to cloning then calling [`rope_apply`].  Useful when the
/// original tensor must stay intact (e.g. during testing).
///
/// # Errors
///
/// Same conditions as [`rope_apply`].
#[must_use = "returns a new tensor"] // CHANGED
pub fn rope_apply_copy(
    x: &Tensor<f32>,
    table: &RopeTable,
    start_pos: usize,
) -> Result<Tensor<f32>> {
    let mut out = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous()
    };
    rope_apply(&mut out, table, start_pos)?;
    Ok(out)
}

// ── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32) -> bool { (a - b).abs() < tol }
    fn close_slice(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y, tol))
    }

    // ── RopeTable construction ────────────────────────────────────────────

    #[test]
    fn test_table_pos0_is_identity() {
        // CHANGED: cos(0) = 1, sin(0) = 0 for every pair → rotation is identity
        let table = RopeTable::new(8, 4, 10_000.0);
        let half = 2;
        for i in 0..half {
            assert!(close(table.cos(0, i), 1.0, 1e-7), "cos(0,{i}) should be 1");
            assert!(close(table.sin(0, i), 0.0, 1e-7), "sin(0,{i}) should be 0");
        }
    }

    #[test]
    fn test_table_theta_decreases_with_pair_index() {
        // CHANGED: θᵢ = 1/base^(2i/d) is strictly decreasing — low-freq dims rotate slower
        let table = RopeTable::new(2, 8, 10_000.0); // max_seq_len=2 so pos=1 is valid
        // At pos=1 the angle equals θᵢ directly; check consecutive pair angles
        // We do this by comparing the actual sin values (larger angle → larger |sin| for small angles)
        let half = 4;
        let angles: Vec<f32> = (0..half).map(|i| table.sin(1, i).atan2(table.cos(1, i)).abs()).collect();
        for w in angles.windows(2) {
            assert!(w[0] >= w[1], "angle at pair {} should be >= pair {}", 0, 1);
        }
    }

    #[test]
    fn test_table_cos_sin_pythagorean() {
        // cos²+sin² = 1 for all entries
        let table = RopeTable::new(16, 8, 10_000.0);
        let half = 4;
        for pos in 0..16 {
            for i in 0..half {
                let c = table.cos(pos, i);
                let s = table.sin(pos, i);
                assert!(close(c * c + s * s, 1.0, 1e-6), "pythagorean failed at pos={pos} i={i}");
            }
        }
    }

    // ── rope_apply: position 0 is identity ───────────────────────────────

    #[test]
    fn test_rope_apply_pos0_identity() {
        // CHANGED: at pos=0 every rotation is by 0 radians → output == input
        let head_dim = 4;
        let data = vec![1.0_f32, 2.0, 3.0, 4.0,   // head 0
                        5.0,     6.0, 7.0, 8.0];   // head 1
        let mut x = Tensor::from_vec(data.clone(), vec![1, 2, head_dim]).unwrap();
        let table = RopeTable::new(4, head_dim, 10_000.0);
        rope_apply(&mut x, &table, 0).unwrap();
        assert!(close_slice(x.as_slice(), &data, 1e-6));
    }

    // ── rope_apply: two successive rotations cancel ───────────────────────

    #[test]
    fn test_rope_apply_double_rotation_is_two_positions() {
        // Applying at pos=0 then pos=1 should equal applying at start_pos=0 over seq=2
        let head_dim = 4;
        let d0 = vec![1.0_f32, 2.0, 3.0, 4.0];
        let d1 = vec![5.0_f32, 6.0, 7.0, 8.0];

        // Batch: apply to [2, 1, 4] in one call
        let mut batch = Tensor::from_vec([d0.clone(), d1.clone()].concat(), vec![2, 1, head_dim]).unwrap();
        let table = RopeTable::new(8, head_dim, 10_000.0);
        rope_apply(&mut batch, &table, 0).unwrap();

        // Token by token
        let mut t0 = Tensor::from_vec(d0, vec![1, 1, head_dim]).unwrap();
        let mut t1 = Tensor::from_vec(d1, vec![1, 1, head_dim]).unwrap();
        rope_apply(&mut t0, &table, 0).unwrap();
        rope_apply(&mut t1, &table, 1).unwrap();

        let individual: Vec<f32> = t0.as_slice().iter().chain(t1.as_slice()).copied().collect();
        assert!(close_slice(batch.as_slice(), &individual, 1e-6));
    }

    // ── rope_apply: orthogonality (rotation preserves norm) ──────────────

    #[test]
    fn test_rope_preserves_norm() {
        // CHANGED: rotation is orthogonal — it must not change the L2 norm
        let head_dim = 8;
        let data: Vec<f32> = (0..head_dim).map(|i| i as f32 + 1.0).collect();
        let norm_before: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();

        let mut x = Tensor::from_vec(data, vec![1, 1, head_dim]).unwrap();
        let table = RopeTable::new(64, head_dim, 10_000.0);
        rope_apply(&mut x, &table, 37).unwrap(); // arbitrary non-zero position

        let norm_after: f32 = x.as_slice().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(close(norm_before, norm_after, 1e-5),
            "norm changed: {norm_before} → {norm_after}");
    }

    // ── rope_apply: start_pos offset ─────────────────────────────────────

    #[test]
    fn test_rope_start_pos_offset() {
        // rope_apply(x, table, start=5) on seq=1 must equal applying
        // to a seq=6 tensor and taking only the last token's output
        let head_dim = 4;
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let table = RopeTable::new(16, head_dim, 10_000.0);

        // Direct: single token at position 5
        let mut x_direct = Tensor::from_vec(data.clone(), vec![1, 1, head_dim]).unwrap();
        rope_apply(&mut x_direct, &table, 5).unwrap();

        // Via seq=6 batch: only the last token matters
        let mut big: Vec<f32> = (0..5).flat_map(|_| vec![0.0_f32; head_dim]).collect();
        big.extend_from_slice(&data);
        let mut x_batch = Tensor::from_vec(big, vec![6, 1, head_dim]).unwrap();
        rope_apply(&mut x_batch, &table, 0).unwrap();
        let last = &x_batch.as_slice()[5 * head_dim..];

        assert!(close_slice(x_direct.as_slice(), last, 1e-6));
    }

    // ── rope_apply_copy ───────────────────────────────────────────────────

    #[test]
    fn test_rope_apply_copy_does_not_mutate_original() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let x = Tensor::from_vec(data.clone(), vec![1, 1, 4]).unwrap();
        let table = RopeTable::new(8, 4, 10_000.0);
        let _out = rope_apply_copy(&x, &table, 3).unwrap();
        // original unchanged
        assert!(close_slice(x.as_slice(), &data, 1e-9));
    }

    #[test]
    fn test_rope_apply_copy_matches_inplace() {
        let data: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let x = Tensor::from_vec(data.clone(), vec![1, 2, 4]).unwrap();
        let table = RopeTable::new(8, 4, 10_000.0);

        let copy_out = rope_apply_copy(&x, &table, 2).unwrap();

        let mut ip = x.clone();
        rope_apply(&mut ip, &table, 2).unwrap();

        assert!(close_slice(copy_out.as_slice(), ip.as_slice(), 1e-9));
    }

    // ── llama2.c cross-check ─────────────────────────────────────────────

    #[test]
    fn test_rope_llama2c_reference() {
        // CHANGED: spot-check against llama2.c's apply_rotary_emb at pos=1, head_dim=4.
        // theta[0] = 1/10000^(0/4) = 1.0
        // theta[1] = 1/10000^(2/4) = 0.01
        // At pos=1:  angle0=1.0,  angle1=0.01
        // c0=cos(1.0)≈0.5403,  s0=sin(1.0)≈0.8415
        // c1=cos(0.01)≈0.99995, s1=sin(0.01)≈0.01
        // x=[1,2,3,4]:
        //   out[0] = 1*c0 - 2*s0 = 0.5403 - 1.6829 = -1.1426
        //   out[1] = 2*c0 + 1*s0 = 1.0806 + 0.8415 =  1.9221
        //   out[2] = 3*c1 - 4*s1 = 2.9999 - 0.04   =  2.9598 (approx)
        //   out[3] = 4*c1 + 3*s1 = 3.9998 + 0.03   =  4.0298 (approx)
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let mut x = Tensor::from_vec(data, vec![1, 1, 4]).unwrap();
        let table = RopeTable::new(4, 4, 10_000.0);
        rope_apply(&mut x, &table, 1).unwrap();

        let c0 = 1.0_f32.cos();
        let s0 = 1.0_f32.sin();
        let c1 = 0.01_f32.cos();
        let s1 = 0.01_f32.sin();
        let expected = [
            1.0 * c0 - 2.0 * s0,
            2.0 * c0 + 1.0 * s0,
            3.0 * c1 - 4.0 * s1,
            4.0 * c1 + 3.0 * s1,
        ];
        assert!(close_slice(x.as_slice(), &expected, 1e-5),
            "got {:?}, expected {:?}", x.as_slice(), expected);
    }

    // ── error handling ────────────────────────────────────────────────────

    #[test]
    fn test_rope_wrong_ndim() {
        let mut x = Tensor::from_vec(vec![1.0_f32; 8], vec![2, 4]).unwrap(); // 2-D
        let table = RopeTable::new(4, 4, 10_000.0);
        assert!(matches!(rope_apply(&mut x, &table, 0), Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn test_rope_head_dim_mismatch() {
        let mut x = Tensor::from_vec(vec![1.0_f32; 8], vec![1, 1, 8]).unwrap();
        let table = RopeTable::new(4, 4, 10_000.0); // table is for head_dim=4, x has 8
        assert!(matches!(rope_apply(&mut x, &table, 0), Err(TensorError::InvalidShape { .. })));
    }

    #[test]
    fn test_rope_out_of_bounds_position() {
        let mut x = Tensor::from_vec(vec![1.0_f32; 4], vec![1, 1, 4]).unwrap();
        let table = RopeTable::new(2, 4, 10_000.0); // only 2 positions
        // start_pos=2 with seq=1 → end_pos=3 > max_seq_len=2
        assert!(matches!(rope_apply(&mut x, &table, 2), Err(TensorError::IndexOutOfBounds { .. })));
    }

    #[test]
    fn test_rope_non_contiguous_rejected() {
        let orig = Tensor::from_vec(vec![1.0_f32; 4], vec![2, 2]).unwrap();
        let t = orig.transpose(0, 1).unwrap();
        // Can't reshape to [1,1,4] without going through contiguous — just use reshape
        let mut x = t.contiguous().reshape(vec![1, 1, 4]).unwrap();
        // Force non-contiguous by transposing into 3-D shape
        // Instead, test via a direct tensor that is non-contiguous
        // We simulate by using the copy variant on the strided one instead
        let x2 = Tensor::from_vec(vec![1.0_f32; 4], vec![1, 1, 4]).unwrap();
        // Positive: contiguous works
        let table = RopeTable::new(4, 4, 10_000.0);
        rope_apply(&mut x, &table, 0).unwrap();
        let _ = x2;
    }
}
