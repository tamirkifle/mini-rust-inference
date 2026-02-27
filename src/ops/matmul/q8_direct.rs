//! Direct Q8_0 inference kernel — commit 14.2.
//!
//! # Motivation
//!
//! The standard path dequantizes Q8_0 weights to f32 when loading from GGUF,
//! producing a full-precision copy that costs 4× as much RAM as the quantized
//! form.  This module provides a **zero-dequant** alternative: weights stay in
//! Q8_0 byte format throughout the model's lifetime; only the 2-byte per-block
//! scale is converted to f32 at compute time.
//!
//! # Format Recap (Q8_0)
//!
//! Each block of 32 elements is stored as:
//!
//! ```text
//! ┌────────────┬─────────────────────────────┐
//! │ d (f16)    │  qs[32]  (i8 × 32)          │
//! │  2 bytes   │  32 bytes                   │
//! └────────────┴─────────────────────────────┘
//! Total: 34 bytes/block
//! ```
//!
//! Dequant formula per element: `f32 = qs[i] * f16_to_f32(d)`
//!
//! # Compute formula (W8A8 path)
//!
//! Given activation row `a[k]` quantized per-tensor with scale `s_a`:
//!
//! ```text
//! out[n] = Σ_b  (Σ_{k in block b}  a_q[k] * w_q[b,k])  *  s_a  *  d[n,b]
//! ```
//!
//! Accumulation in `i32`; rescale to f32 at the end of each block.
//! Max `|i32|` = 127 × 127 × 32 = 516 128 — well within `i32::MAX`.
//!
//! # Key difference vs `matmul_int8`
//!
//! `matmul_int8` uses **per-row** weight scales (`QuantizedMatrix`).
//! This kernel uses **per-block-of-32** scales, matching the Q8_0 format exactly.

use std::borrow::Cow;

use crate::gguf::dequant::{create_q8_0_block, Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_SIZE};
use crate::gguf::f16_to_f32;
use crate::quant::int8::symmetric::quantize_symmetric;
use crate::tensor::{Result, Tensor, TensorError};

// ── Q8_0WeightMatrix ──────────────────────────────────────────────────────────

/// A weight matrix stored in raw Q8_0 GGUF format, never fully dequantized to f32.
///
/// Memory layout: `n_out` rows, each row has `n_blocks` Q8_0 blocks of 34 bytes.
///
/// ```text
/// data[row n, block b]  →  byte offset: (n * n_blocks + b) * 34
///   [0..2]  f16 scale d
///   [2..34] i8 quants
/// ```
#[derive(Clone)]
pub struct Q8_0WeightMatrix {
    /// Raw Q8_0 bytes (GGUF layout).
    data: Vec<u8>,
    /// Number of output channels (rows).
    pub n_out: usize,
    /// Inner / input dimension (must be a multiple of 32).
    pub k_in: usize,
    /// `k_in / 32` — blocks per row.
    n_blocks: usize,
}

impl Q8_0WeightMatrix {
    // ── constructors ─────────────────────────────────────────────────────────

    /// Construct from raw Q8_0 bytes (e.g. directly from a GGUF mmap slice).
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidShape`] if `k_in` is not a multiple of 32
    /// or if `data.len()` does not equal `n_out * (k_in/32) * 34`.
    pub fn from_raw_bytes(data: Vec<u8>, n_out: usize, k_in: usize) -> Result<Self> {
        if k_in % Q8_0_BLOCK_ELEMENTS != 0 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Q8_0WeightMatrix: k_in={k_in} is not a multiple of {}",
                    Q8_0_BLOCK_ELEMENTS
                ),
            });
        }
        let n_blocks = k_in / Q8_0_BLOCK_ELEMENTS;
        let expected = n_out * n_blocks * Q8_0_BLOCK_SIZE;
        if data.len() != expected {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Q8_0WeightMatrix: data.len()={} != expected {} \
                     (n_out={n_out}, n_blocks={n_blocks}, block_size={Q8_0_BLOCK_SIZE})",
                    data.len(), expected
                ),
            });
        }
        Ok(Self { data, n_out, k_in, n_blocks })
    }

    /// Build a `Q8_0WeightMatrix` by quantizing a 2-D `Tensor<f32>`.
    ///
    /// Useful for testing without a real GGUF file.
    /// Shape must be `[n_out, k_in]`; `k_in` must be a multiple of 32.
    ///
    /// # Errors
    ///
    /// Returns [`TensorError::InvalidShape`] if the tensor is not 2-D or
    /// `k_in` is not a multiple of 32.
    pub fn from_f32_tensor(t: &Tensor<f32>) -> Result<Self> {
        if t.ndim() != 2 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Q8_0WeightMatrix::from_f32_tensor: expected 2-D, got {}D",
                    t.ndim()
                ),
            });
        }
        let n_out = t.dims()[0];
        let k_in  = t.dims()[1];
        if k_in % Q8_0_BLOCK_ELEMENTS != 0 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Q8_0WeightMatrix::from_f32_tensor: k_in={k_in} not a multiple of 32"
                ),
            });
        }
        let n_blocks = k_in / Q8_0_BLOCK_ELEMENTS;
        let slice: Cow<[f32]> = if t.is_contiguous() {
            Cow::Borrowed(t.as_slice())
        } else {
            Cow::Owned(t.contiguous().as_slice().to_vec())
        };

        let mut data = Vec::with_capacity(n_out * n_blocks * Q8_0_BLOCK_SIZE);
        for row in 0..n_out {
            for blk in 0..n_blocks {
                let start = row * k_in + blk * Q8_0_BLOCK_ELEMENTS;
                let block_f32 = &slice[start..start + Q8_0_BLOCK_ELEMENTS];
                let encoded = create_q8_0_block(block_f32);
                data.extend_from_slice(&encoded);
            }
        }
        Ok(Self { data, n_out, k_in, n_blocks })
    }

    // ── accessors ─────────────────────────────────────────────────────────────

    /// Return the f32 scale for block `b` of row `n`.
    #[inline]
    #[must_use]
    pub fn block_scale(&self, n: usize, b: usize) -> f32 {
        let off = (n * self.n_blocks + b) * Q8_0_BLOCK_SIZE;
        let bits = u16::from_le_bytes([self.data[off], self.data[off + 1]]);
        f16_to_f32(bits)
    }

    /// Return the 32 i8 quants for block `b` of row `n`.
    #[inline]
    #[must_use]
    pub fn block_quants(&self, n: usize, b: usize) -> &[u8] {
        let off = (n * self.n_blocks + b) * Q8_0_BLOCK_SIZE + 2;
        &self.data[off..off + Q8_0_BLOCK_ELEMENTS]
    }

    /// Dequantize all weights to a row-major `[n_out, k_in]` `Vec<f32>`.
    ///
    /// Used to produce the f32 reference for correctness tests.
    #[must_use]
    pub fn dequantize_all(&self) -> Vec<f32> {
        let mut out = vec![0.0_f32; self.n_out * self.k_in];
        for n in 0..self.n_out {
            for b in 0..self.n_blocks {
                let scale  = self.block_scale(n, b);
                let quants = self.block_quants(n, b);
                let base   = n * self.k_in + b * Q8_0_BLOCK_ELEMENTS;
                for k in 0..Q8_0_BLOCK_ELEMENTS {
                    out[base + k] = f32::from(quants[k] as i8) * scale;
                }
            }
        }
        out
    }
}

// ── matmul_q8_0_direct ────────────────────────────────────────────────────────

/// Direct Q8_0 weight × f32 activation GEMM — W8A8 path, zero weight dequant.
///
/// For each activation row, quantizes to INT8 per-tensor (one scale for the
/// whole row).  For each weight block, computes an INT8 dot product using the
/// block's per-block scale, then rescales to f32.
///
/// # Formula
///
/// ```text
/// out[m, n] = Σ_b  dot_i32(a_q[m, b], w_q[n, b])  *  s_a[m]  *  d[n, b]
/// ```
///
/// where `s_a[m]` is the per-tensor activation scale and `d[n, b]` is the
/// Q8_0 per-block weight scale (stored as f16 in the GGUF bytes).
///
/// # Arguments
///
/// * `input`   – `[M, K]` f32 activations
/// * `weights` – `Q8_0WeightMatrix` with `k_in == K`
///
/// # Returns
///
/// Contiguous `Tensor<f32>` of shape `[M, N]`.
///
/// # Errors
///
/// [`TensorError::InvalidShape`] if dimensions are inconsistent.
#[must_use = "returns a new tensor"]
pub fn matmul_q8_0_direct(
    input:   &Tensor<f32>,
    weights: &Q8_0WeightMatrix,
) -> Result<Tensor<f32>> {
    if input.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_q8_0_direct: input must be 2-D, got {}D (shape {:?})",
                input.ndim(), input.dims()
            ),
        });
    }
    let m  = input.dims()[0];
    let k  = input.dims()[1];
    if k != weights.k_in {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, weights.k_in],
            got:      vec![m, k],
        });
    }
    let n         = weights.n_out;
    let n_blocks  = weights.n_blocks;

    // Ensure contiguous before slicing rows.
    let input_c: Cow<Tensor<f32>> = if input.is_contiguous() {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(input.contiguous())
    };

    let input_data = input_c.as_slice();
    let mut out    = vec![0.0_f32; m * n];

    for m_i in 0..m {
        let act_row = &input_data[m_i * k..(m_i + 1) * k];

        // Quantize activation row to INT8 per-tensor.
        let (act_q, act_scale) = quantize_symmetric(act_row);

        for n_i in 0..n {
            let mut acc = 0.0_f32;

            for blk in 0..n_blocks {
                let w_scale  = weights.block_scale(n_i, blk);
                let w_quants = weights.block_quants(n_i, blk);

                // INT8 dot product — accumulate in i32 to prevent overflow.
                let base_k  = blk * Q8_0_BLOCK_ELEMENTS;
                let mut dot = 0_i32;
                for k_i in 0..Q8_0_BLOCK_ELEMENTS {
                    dot += i32::from(act_q[base_k + k_i])
                         * i32::from(w_quants[k_i] as i8);
                }

                // Rescale: activation scale × block weight scale.
                acc += dot as f32 * act_scale * w_scale;
            }

            out[m_i * n + n_i] = acc;
        }
    }

    Tensor::from_vec(out, vec![m, n])
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matmul::naive::matmul_naive;

    /// Tolerance: two rounds of quantization error (activation + weight).
    /// ~2% relative for smooth distributions, matching the int8 kernel budget.
    const REL_TOL: f32 = 0.03;

    fn max_rel_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| {
            let denom = x.abs().max(y.abs()).max(1e-3);
            (x - y).abs() / denom
        }).fold(0.0_f32, f32::max)
    }

    // ── from_raw_bytes validation ──────────────────────────────────────────

    #[test]
    fn from_raw_bytes_correct_size_accepted() {
        // n_out=2, k_in=32 → 2 rows × 1 block × 34 bytes = 68 bytes
        let data = vec![0u8; 2 * 1 * 34];
        assert!(Q8_0WeightMatrix::from_raw_bytes(data, 2, 32).is_ok());
    }

    #[test]
    fn from_raw_bytes_k_not_multiple_of_32_rejected() {
        let data = vec![0u8; 34];
        assert!(Q8_0WeightMatrix::from_raw_bytes(data, 1, 31).is_err());
    }

    #[test]
    fn from_raw_bytes_wrong_byte_length_rejected() {
        let data = vec![0u8; 33]; // wrong length
        assert!(Q8_0WeightMatrix::from_raw_bytes(data, 1, 32).is_err());
    }

    #[test]
    fn from_raw_bytes_zero_n_out_accepted() {
        let data = vec![];
        assert!(Q8_0WeightMatrix::from_raw_bytes(data, 0, 32).is_ok());
    }

    // ── from_f32_tensor ────────────────────────────────────────────────────

    #[test]
    fn from_f32_tensor_non_2d_rejected() {
        let t = Tensor::from_vec(vec![1.0f32; 64], vec![2, 4, 8]).unwrap();
        assert!(Q8_0WeightMatrix::from_f32_tensor(&t).is_err());
    }

    #[test]
    fn from_f32_tensor_k_not_multiple_of_32_rejected() {
        let t = Tensor::from_vec(vec![1.0f32; 31], vec![1, 31]).unwrap();
        assert!(Q8_0WeightMatrix::from_f32_tensor(&t).is_err());
    }

    // ── matmul_q8_0_direct: shape & basic correctness ──────────────────────

    #[test]
    fn output_shape_m_by_n() {
        let w_f32: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let wt = Tensor::from_vec(w_f32, vec![4, 32]).unwrap();
        let qw = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();
        let inp = Tensor::from_vec(vec![0.5f32; 96], vec![3, 32]).unwrap();
        let out = matmul_q8_0_direct(&inp, &qw).unwrap();
        assert_eq!(out.dims(), &[3, 4]);
    }

    #[test]
    fn zero_input_gives_zero_output() {
        let w: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 - 6.0).collect();
        let wt = Tensor::from_vec(w, vec![4, 32]).unwrap();
        let qw = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();
        let inp = Tensor::zeros(vec![2, 32]);
        let out = matmul_q8_0_direct(&inp, &qw).unwrap();
        for &v in out.as_slice() {
            assert!(v.abs() < 1e-5, "expected 0, got {v}");
        }
    }

    #[test]
    fn zero_weights_give_zero_output() {
        let wt = Tensor::zeros(vec![2, 32]);
        let qw = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();
        let inp: Vec<f32> = (0..64).map(|i| i as f32 * 0.1 - 3.0).collect();
        let input = Tensor::from_vec(inp, vec![2, 32]).unwrap();
        let out = matmul_q8_0_direct(&input, &qw).unwrap();
        for &v in out.as_slice() {
            assert!(v.abs() < 1e-4, "expected ~0, got {v}");
        }
    }

    // ── correctness: direct path matches naive + dequant ───────────────────

    #[test]
    fn matches_dequant_naive_small() {
        // [2, 32] × [4, 32]^T → [2, 4]
        let w: Vec<f32> = (0..128).map(|i| (i as f32) * 0.03 - 2.0).collect();
        let inp: Vec<f32> = (0..64).map(|i| (i as f32) * 0.05 - 1.6).collect();

        let wt  = Tensor::from_vec(w.clone(), vec![4, 32]).unwrap();
        let qw  = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();
        let inp_t = Tensor::from_vec(inp, vec![2, 32]).unwrap();

        // f32 reference via dequantize_all + naive matmul
        let w_deq = Tensor::from_vec(qw.dequantize_all(), vec![4, 32]).unwrap();
        let ref_t = matmul_naive(&inp_t, &w_deq.transpose(0, 1).unwrap()).unwrap();

        let direct = matmul_q8_0_direct(&inp_t, &qw).unwrap();
        let mre = max_rel_err(direct.as_slice(), ref_t.as_slice());
        assert!(mre < REL_TOL,
            "max rel error {mre:.4} >= tolerance {REL_TOL}");
    }

    #[test]
    fn matches_dequant_naive_multi_block() {
        // [3, 128] × [6, 128]^T → [3, 6]  (4 blocks per row)
        let w: Vec<f32>   = (0..768).map(|i| (i as f32) * 0.01 - 3.84).collect();
        let inp: Vec<f32> = (0..384).map(|i| (i as f32) * 0.02 - 3.84).collect();

        let wt    = Tensor::from_vec(w.clone(), vec![6, 128]).unwrap();
        let qw    = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();
        let inp_t = Tensor::from_vec(inp, vec![3, 128]).unwrap();

        let w_deq = Tensor::from_vec(qw.dequantize_all(), vec![6, 128]).unwrap();
        let ref_t = matmul_naive(&inp_t, &w_deq.transpose(0, 1).unwrap()).unwrap();
        let direct = matmul_q8_0_direct(&inp_t, &qw).unwrap();

        let mre = max_rel_err(direct.as_slice(), ref_t.as_slice());
        assert!(mre < REL_TOL,
            "max rel error {mre:.4} >= tolerance {REL_TOL}");
    }

    #[test]
    fn single_row_single_output_dot_product() {
        // 1×32 input all-ones, 1×32 weight all-ones → dot = 32
        let wt  = Tensor::from_vec(vec![1.0f32; 32], vec![1, 32]).unwrap();
        let qw  = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();
        let inp = Tensor::from_vec(vec![0.5f32; 32], vec![1, 32]).unwrap();
        let out = matmul_q8_0_direct(&inp, &qw).unwrap();
        assert_eq!(out.dims(), &[1, 1]);
        // Expected: 32 × 0.5 × 1.0 = 16.0; allow quantization error
        let got = out.as_slice()[0];
        assert!((got - 16.0).abs() < 1.0,
            "expected ~16.0, got {got}");
    }

    // ── non-contiguous input ───────────────────────────────────────────────

    #[test]
    fn non_contiguous_input_handled() {
        let data: Vec<f32> = (0..128).map(|i| i as f32 * 0.05 - 3.0).collect();
        let t = Tensor::from_vec(data, vec![64, 2]).unwrap();
        let inp_nc = t.transpose(0, 1).unwrap(); // [2, 64], non-contiguous
        assert!(!inp_nc.is_contiguous());

        let w: Vec<f32> = (0..192).map(|i| (i as f32) * 0.02 - 2.0).collect();
        let wt  = Tensor::from_vec(w, vec![3, 64]).unwrap();
        let qw  = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();

        let out_nc = matmul_q8_0_direct(&inp_nc, &qw).unwrap();
        let out_c  = matmul_q8_0_direct(&inp_nc.contiguous(), &qw).unwrap();
        // Results should be identical (same data, just laid out differently).
        assert_eq!(out_nc.as_slice(), out_c.as_slice());
    }

    // ── error paths ────────────────────────────────────────────────────────

    #[test]
    fn error_on_non_2d_input() {
        let wt = Tensor::from_vec(vec![0.0f32; 32], vec![1, 32]).unwrap();
        let qw = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();
        let inp = Tensor::from_vec(vec![1.0f32; 64], vec![2, 4, 8]).unwrap();
        assert!(matches!(
            matmul_q8_0_direct(&inp, &qw),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn error_on_k_mismatch() {
        let wt  = Tensor::from_vec(vec![0.0f32; 64], vec![2, 32]).unwrap(); // k_in=32
        let qw  = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();
        let inp = Tensor::from_vec(vec![1.0f32; 64], vec![2, 32]).unwrap();
        let _ = matmul_q8_0_direct(&inp, &qw).unwrap(); // correct k — should pass

        let wt2  = Tensor::from_vec(vec![0.0f32; 128], vec![2, 64]).unwrap(); // k_in=64
        let qw2  = Q8_0WeightMatrix::from_f32_tensor(&wt2).unwrap();
        let inp2 = Tensor::from_vec(vec![1.0f32; 32], vec![1, 32]).unwrap(); // k=32 ≠ 64
        assert!(matches!(
            matmul_q8_0_direct(&inp2, &qw2),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    // ── no NaN/Inf ─────────────────────────────────────────────────────────

    #[test]
    fn no_nan_or_inf_with_varied_inputs() {
        let w: Vec<f32>   = (0..1024).map(|i| (i as f32) * 0.01 - 5.12).collect();
        let inp: Vec<f32> = (0..512).map(|i| (i as f32) * 0.02 - 5.12).collect();
        let wt  = Tensor::from_vec(w, vec![32, 32]).unwrap();
        let qw  = Q8_0WeightMatrix::from_f32_tensor(&wt).unwrap();
        let inp = Tensor::from_vec(inp, vec![16, 32]).unwrap();
        let out = matmul_q8_0_direct(&inp, &qw).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(),      "NaN in output");
            assert!(!v.is_infinite(), "Inf in output");
        }
    }
}
