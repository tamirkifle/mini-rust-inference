//! Direct Q8_0 inference kernel — commit 14.2.
//!
//! Rather than dequantising the entire weight matrix to f32 upfront,
//! this module keeps weights as raw GGUF Q8_0 bytes and performs the
//! dot-product **block-by-block**:
//!
//! ```text
//! result[m, n] = Σ_b  scale_b  ×  Σ_{k in block b}  act[m,k] × q[n,k]
//! ```
//!
//! where `act[m,k]` is an f32 activation, `q[n,k]` is an `i8` weight,
//! and `scale_b` is the f16 scale stored in block b.
//!
//! # Memory advantage
//!
//! A 4 096×4 096 f32 weight matrix consumes 64 MB.  The same matrix as
//! Q8_0 consumes ~17 MB (1 byte/element + 1 f16 per 32 elements).
//! With the direct path the f32 materialisation is avoided entirely —
//! only a 32-element scratch buffer is needed per block.
//!
//! # API
//!
//! ```
//! use llm_engine::gguf::quant::q8_inference::{Q8WeightMatrix, matmul_q8_direct};
//! use llm_engine::Tensor;
//!
//! // Build from f32 weights (test/demo path — encodes them as Q8_0)
//! let weights: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 - 6.4).collect();
//! let qw = Q8WeightMatrix::from_f32(&weights, 4, 32).unwrap();
//!
//! let input = Tensor::from_vec(vec![1.0f32; 96], vec![3, 32]).unwrap();
//! let out = matmul_q8_direct(&input, &qw).unwrap();
//! assert_eq!(out.dims(), &[3, 4]);
//! ```

use std::borrow::Cow;

use crate::gguf::dequant::{Q8_0_BLOCK_ELEMENTS, Q8_0_BLOCK_SIZE};
use crate::gguf::f16_to_f32;
use crate::tensor::{Result, Tensor, TensorError};

// Local aliases for readability.
const BLOCK_ELEMS: usize = Q8_0_BLOCK_ELEMENTS; // 32
const BLOCK_BYTES: usize = Q8_0_BLOCK_SIZE;     // 34

// ── Q8WeightMatrix ────────────────────────────────────────────────────────────

/// A weight matrix stored as raw Q8_0 GGUF bytes.
///
/// Layout of `raw`:
/// ```text
/// row 0 : blocks[0 .. blocks_per_row]     (blocks_per_row × 34 bytes)
/// row 1 : blocks[blocks_per_row .. 2×bpr]
/// ...
/// row n_out-1
/// ```
/// Each 34-byte block: `[f16_scale (2 B)] [i8 × 32 (32 B)]`.
#[derive(Debug, Clone)]
pub struct Q8WeightMatrix {
    /// Raw Q8_0 bytes, row-major.
    pub(crate) raw:           Vec<u8>,
    /// Number of output channels (rows of the logical weight matrix).
    pub n_out:          usize,
    /// Inner dimension (columns); must be a multiple of 32.
    pub k_in:           usize,
    /// Number of Q8_0 blocks per weight row = `k_in / 32`.
    pub blocks_per_row: usize,
}

impl Q8WeightMatrix {
    /// Construct from pre-existing raw Q8_0 bytes.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if `k_in` is not a multiple of 32, or if
    /// `raw.len()` does not equal `n_out * (k_in / 32) * 34`.
    pub fn from_raw_bytes(raw: Vec<u8>, n_out: usize, k_in: usize) -> Result<Self> {
        if k_in % BLOCK_ELEMS != 0 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Q8WeightMatrix: k_in={k_in} must be a multiple of {BLOCK_ELEMS}"
                ),
            });
        }
        let blocks_per_row = k_in / BLOCK_ELEMS;
        let expected = n_out * blocks_per_row * BLOCK_BYTES;
        if raw.len() != expected {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Q8WeightMatrix: raw.len()={} but expected {expected} \
                     for shape [{n_out}, {k_in}]",
                    raw.len()
                ),
            });
        }
        Ok(Self { raw, n_out, k_in, blocks_per_row })
    }

    /// Encode f32 weights as Q8_0 (per-block symmetric quantization).
    ///
    /// Useful for testing and for loading non-GGUF weight sources.
    /// Each 32-element block is quantized with `scale = max_abs / 127`.
    ///
    /// # Errors
    ///
    /// [`TensorError::InvalidShape`] if `weights.len() != n_out * k_in`
    /// or `k_in % 32 != 0`.
    pub fn from_f32(weights: &[f32], n_out: usize, k_in: usize) -> Result<Self> {
        if weights.len() != n_out * k_in {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Q8WeightMatrix::from_f32: weights.len()={} != n_out*k_in={}",
                    weights.len(), n_out * k_in
                ),
            });
        }
        if k_in % BLOCK_ELEMS != 0 {
            return Err(TensorError::InvalidShape {
                reason: format!(
                    "Q8WeightMatrix::from_f32: k_in={k_in} must be a multiple of {BLOCK_ELEMS}"
                ),
            });
        }
        let blocks_per_row = k_in / BLOCK_ELEMS;
        let mut raw = Vec::with_capacity(n_out * blocks_per_row * BLOCK_BYTES);

        for n in 0..n_out {
            for b in 0..blocks_per_row {
                let start = n * k_in + b * BLOCK_ELEMS;
                let block_f32 = &weights[start..start + BLOCK_ELEMS];
                encode_q8_0_block(block_f32, &mut raw);
            }
        }
        Ok(Self { raw, n_out, k_in, blocks_per_row })
    }

    /// Return a borrowed byte slice for a single row's block `b`.
    #[inline]
    fn block_bytes(&self, row: usize, b: usize) -> &[u8] {
        let start = (row * self.blocks_per_row + b) * BLOCK_BYTES;
        &self.raw[start..start + BLOCK_BYTES]
    }

    /// Dequantize all blocks to a contiguous f32 slice `[n_out, k_in]`.
    ///
    /// Used for correctness reference comparisons.
    #[must_use]
    pub fn dequantize_all(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.n_out * self.k_in);
        for n in 0..self.n_out {
            for b in 0..self.blocks_per_row {
                let blk = self.block_bytes(n, b);
                let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for &q in &blk[2..] {
                    out.push(scale * f32::from(q as i8));
                }
            }
        }
        out
    }
}

// ── block encode helper ───────────────────────────────────────────────────────

/// Encode exactly 32 f32 values as one Q8_0 block, appending 34 bytes to `buf`.
///
/// Scale: `max_abs(block) / 127`; zero block → scale = 0 (all quants = 0).
#[inline]
fn encode_q8_0_block(block: &[f32], buf: &mut Vec<u8>) {
    debug_assert_eq!(block.len(), BLOCK_ELEMS);
    let max_abs = block.iter().map(|v| v.abs()).fold(0.0_f32, f32::max);
    let scale   = if max_abs == 0.0 { 0.0_f32 } else { max_abs / 127.0 };

    // Write f16 scale (little-endian)
    let scale_f16 = crate::gguf::quantization::f32_to_f16(scale);
    buf.extend_from_slice(&scale_f16.to_le_bytes());

    // Write 32 × i8 quants
    if scale > 0.0 {
        let inv = 1.0 / scale;
        for &v in block {
            buf.push((v * inv).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    } else {
        for _ in 0..BLOCK_ELEMS { buf.push(0); }
    }
}

// ── hot kernel helper ─────────────────────────────────────────────────────────

/// Block-level dot product: `scale_b × Σ_i  act[i] × q[i]` for one Q8_0 block.
///
/// `blk` is exactly `BLOCK_BYTES` = 34 bytes.
/// `act` is exactly `BLOCK_ELEMS` = 32 f32 values.
#[inline(always)]
fn block_dot(blk: &[u8], act: &[f32]) -> f32 {
    let scale = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
    let qs    = &blk[2..];
    let mut acc = 0.0_f32;
    for (i, &q) in qs.iter().enumerate() {
        acc += f32::from(q as i8) * act[i];
    }
    scale * acc
}

// ── public matmul API ─────────────────────────────────────────────────────────

/// Compute `output = input × Q8WeightMatrix^T` without materialising f32 weights.
///
/// `input` shape: `[M, K]`  
/// `weights` shape: `[N, K]`  
/// Output shape: `[M, N]`
///
/// # Errors
///
/// * [`TensorError::InvalidShape`]  — `input` not 2-D
/// * [`TensorError::ShapeMismatch`] — `input.dims()[1] != weights.k_in`
#[must_use = "returns a new tensor"]
pub fn matmul_q8_direct(
    input:   &Tensor<f32>,
    weights: &Q8WeightMatrix,
) -> Result<Tensor<f32>> {
    if input.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_q8_direct: input must be 2-D, got {}D (shape {:?})",
                input.ndim(), input.dims()
            ),
        });
    }
    let m   = input.dims()[0];
    let k_a = input.dims()[1];
    if k_a != weights.k_in {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, weights.k_in],
            got:      vec![m, k_a],
        });
    }

    // Ensure contiguous activation slice.
    let inp_c: Cow<Tensor<f32>> = if input.is_contiguous() {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(input.contiguous())
    };
    let inp = inp_c.as_slice();

    let n   = weights.n_out;
    let k   = weights.k_in;
    let bpr = weights.blocks_per_row;

    let mut out = vec![0.0_f32; m * n];

    for n_i in 0..n {
        let row_base = n_i * bpr;
        for m_i in 0..m {
            let act_row = &inp[m_i * k..(m_i + 1) * k];
            let mut acc = 0.0_f32;
            for b in 0..bpr {
                let blk_start = (row_base + b) * BLOCK_BYTES;
                let blk = &weights.raw[blk_start..blk_start + BLOCK_BYTES];
                let act_blk = &act_row[b * BLOCK_ELEMS..(b + 1) * BLOCK_ELEMS];
                acc += block_dot(blk, act_blk);
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
    use crate::gguf::dequant::dequantize_q8_0;
    use crate::ops::matmul::naive::matmul_naive;

    // Tolerance: Q8_0 quantization introduces ≤0.5 LSB error per element.
    // For block dot products this sums across 32 elements; we use 1e-4 absolute.
    const EPS: f32 = 1e-4;

    fn close(a: f32, b: f32) -> bool { (a - b).abs() < EPS }
    fn close_slice(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
    }

    // ── Q8WeightMatrix construction ───────────────────────────────────────

    #[test]
    fn from_f32_encodes_correct_dimensions() {
        let w: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 - 6.4).collect();
        let qw = Q8WeightMatrix::from_f32(&w, 4, 32).unwrap();
        assert_eq!(qw.n_out, 4);
        assert_eq!(qw.k_in, 32);
        assert_eq!(qw.blocks_per_row, 1);
        // 4 rows × 1 block/row × 34 bytes/block
        assert_eq!(qw.raw.len(), 4 * 34);
    }

    #[test]
    fn from_f32_rejects_k_not_multiple_of_32() {
        let w = vec![0.0f32; 33];
        assert!(Q8WeightMatrix::from_f32(&w, 1, 33).is_err());
    }

    #[test]
    fn from_f32_rejects_length_mismatch() {
        let w = vec![0.0f32; 31]; // not n_out*k_in
        assert!(Q8WeightMatrix::from_f32(&w, 1, 32).is_err());
    }

    #[test]
    fn from_raw_bytes_rejects_wrong_length() {
        let raw = vec![0u8; 33]; // should be 34 for (1, 32)
        assert!(Q8WeightMatrix::from_raw_bytes(raw, 1, 32).is_err());
    }

    #[test]
    fn from_raw_bytes_accepts_correct_layout() {
        let raw = vec![0u8; 2 * 34]; // 2 rows, each 1 block (k=32)
        let qw = Q8WeightMatrix::from_raw_bytes(raw, 2, 32).unwrap();
        assert_eq!(qw.n_out, 2);
        assert_eq!(qw.blocks_per_row, 1);
    }

    // ── dequantize_all roundtrip ───────────────────────────────────────────

    #[test]
    fn dequantize_all_matches_q8_0_dequant_fn() {
        let w: Vec<f32> = (0..256).map(|i| (i as f32) * 0.03 - 3.0).collect();
        let qw = Q8WeightMatrix::from_f32(&w, 8, 32).unwrap();

        let via_struct = qw.dequantize_all();
        let via_fn     = dequantize_q8_0(&qw.raw).unwrap();

        assert_eq!(via_struct.len(), via_fn.len());
        for (a, b) in via_struct.iter().zip(&via_fn) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    // ── matmul_q8_direct vs f32 reference ─────────────────────────────────

    #[test]
    fn output_shape_m_by_n() {
        let w: Vec<f32> = (0..192).map(|i| i as f32 * 0.02 - 1.9).collect();
        let qw    = Q8WeightMatrix::from_f32(&w, 6, 32).unwrap();
        let input = Tensor::from_vec(vec![0.5f32; 4 * 32], vec![4, 32]).unwrap();
        let out   = matmul_q8_direct(&input, &qw).unwrap();
        assert_eq!(out.dims(), &[4, 6]);
    }

    #[test]
    fn zero_input_gives_zero_output() {
        let w: Vec<f32> = (0..128).map(|i| i as f32 * 0.1 - 6.4).collect();
        let qw    = Q8WeightMatrix::from_f32(&w, 4, 32).unwrap();
        let input = Tensor::zeros(vec![3, 32]);
        let out   = matmul_q8_direct(&input, &qw).unwrap();
        for &v in out.as_slice() {
            assert!(v.abs() < EPS, "expected 0, got {v}");
        }
    }

    #[test]
    fn zero_weights_give_zero_output() {
        let qw    = Q8WeightMatrix::from_f32(&vec![0.0f32; 128], 4, 32).unwrap();
        let input = Tensor::from_vec((0..96).map(|i| i as f32 * 0.1).collect(), vec![3, 32]).unwrap();
        let out   = matmul_q8_direct(&input, &qw).unwrap();
        for &v in out.as_slice() {
            assert!(v.abs() < EPS, "expected ~0, got {v}");
        }
    }

    /// The direct Q8_0 path must match naive(input, deq_weights^T) exactly,
    /// because both perform the same block-wise computation with the same
    /// Q8_0 encoding.
    #[test]
    fn direct_matches_dequant_then_naive_single_block() {
        let w: Vec<f32> = (0..32).map(|i| i as f32 * 0.05 - 0.8).collect();
        let qw    = Q8WeightMatrix::from_f32(&w, 1, 32).unwrap();
        let input = Tensor::from_vec((0..32).map(|i| i as f32 * 0.03).collect(), vec![1, 32]).unwrap();

        let direct  = matmul_q8_direct(&input, &qw).unwrap();

        let deq_data = qw.dequantize_all();
        let w_deq = Tensor::from_vec(deq_data, vec![1, 32]).unwrap();
        let ref_out = matmul_naive(&input, &w_deq.transpose(0, 1).unwrap()).unwrap();

        assert!(close_slice(direct.as_slice(), ref_out.as_slice()),
            "direct={:?} ref={:?}", direct.as_slice(), ref_out.as_slice());
    }

    #[test]
    fn direct_matches_dequant_then_naive_multi_row() {
        // weight [6, 64], input [4, 64]
        let w: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01 - 1.9).collect();
        let qw    = Q8WeightMatrix::from_f32(&w, 6, 64).unwrap();
        let inp   = Tensor::from_vec((0..256).map(|i| i as f32 * 0.02).collect(), vec![4, 64]).unwrap();

        let direct = matmul_q8_direct(&inp, &qw).unwrap();

        let deq_data = qw.dequantize_all();
        let w_deq = Tensor::from_vec(deq_data, vec![6, 64]).unwrap();
        let ref_out = matmul_naive(&inp, &w_deq.transpose(0, 1).unwrap()).unwrap();

        assert_eq!(direct.dims(), &[4, 6]);
        assert!(close_slice(direct.as_slice(), ref_out.as_slice()),
            "max diff = {}",
            direct.as_slice().iter().zip(ref_out.as_slice())
                .map(|(a,b)| (a-b).abs()).fold(0.0f32, f32::max));
    }

    #[test]
    fn direct_matches_dequant_larger_k() {
        // weight [8, 128] — 4 blocks per row; input [3, 128]
        // Tolerance is looser than EPS: floating-point order differs between
        // block_dot (scale × Σ act[i]×q[i]) and the reference (Σ act[i]×(scale×q[i])).
        // Both are mathematically identical but FP rounding accumulates across 4 blocks.
        let w: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.003 - 1.5).collect();
        let qw  = Q8WeightMatrix::from_f32(&w, 8, 128).unwrap();
        let inp = Tensor::from_vec((0..384).map(|i| i as f32 * 0.01).collect(), vec![3, 128]).unwrap();

        let direct = matmul_q8_direct(&inp, &qw).unwrap();

        let deq_data = qw.dequantize_all();
        let w_deq   = Tensor::from_vec(deq_data, vec![8, 128]).unwrap();
        let ref_out = matmul_naive(&inp, &w_deq.transpose(0, 1).unwrap()).unwrap();

        assert_eq!(direct.dims(), &[3, 8]);
        // Relative tolerance: max |a-b| / max(|a|,|b|,1) < 0.1%
        let max_rel: f32 = direct.as_slice().iter().zip(ref_out.as_slice())
            .map(|(a, b)| (a - b).abs() / a.abs().max(b.abs()).max(1.0))
            .fold(0.0, f32::max);
        assert!(max_rel < 1e-3,
            "max relative error {max_rel:.2e} exceeds 0.1% for K=128 test");
    }

    /// Round-trip using raw GGUF-style bytes (from dequant module helper).
    #[test]
    fn from_raw_bytes_round_trip() {
        use crate::gguf::dequant::create_q8_0_block;

        let w_f32: Vec<f32> = (0..64).map(|i| i as f32 * 0.05 - 1.6).collect();

        // Build raw Q8_0 bytes block-by-block (simulates GGUF tensor bytes)
        let mut raw_bytes: Vec<u8> = Vec::new();
        for chunk in w_f32.chunks(32) {
            raw_bytes.extend_from_slice(&create_q8_0_block(chunk));
        }
        // That's 1 row of 2 blocks for n_out=1, k_in=64
        let qw = Q8WeightMatrix::from_raw_bytes(raw_bytes, 1, 64).unwrap();

        let inp    = Tensor::from_vec(vec![1.0f32; 64], vec![1, 64]).unwrap();
        let direct = matmul_q8_direct(&inp, &qw).unwrap();

        let deq_data = qw.dequantize_all();
        let w_deq   = Tensor::from_vec(deq_data, vec![1, 64]).unwrap();
        let ref_out = matmul_naive(&inp, &w_deq.transpose(0, 1).unwrap()).unwrap();

        assert!(close_slice(direct.as_slice(), ref_out.as_slice()));
    }

    // ── non-contiguous input ──────────────────────────────────────────────

    #[test]
    fn non_contiguous_input_handled() {
        let data: Vec<f32> = (0..128).map(|i| i as f32 * 0.05).collect();
        let t   = Tensor::from_vec(data, vec![64, 2]).unwrap();
        let inp_nc = t.transpose(0, 1).unwrap(); // [2, 64] non-contiguous
        assert!(!inp_nc.is_contiguous());

        let w: Vec<f32> = (0..192).map(|i| i as f32 * 0.02 - 1.9).collect();
        let qw = Q8WeightMatrix::from_f32(&w, 3, 64).unwrap();

        let direct = matmul_q8_direct(&inp_nc, &qw).unwrap();
        let cont   = matmul_q8_direct(&inp_nc.contiguous(), &qw).unwrap();
        assert_eq!(direct.as_slice(), cont.as_slice());
    }

    // ── error paths ───────────────────────────────────────────────────────

    #[test]
    fn error_non_2d_input() {
        let qw  = Q8WeightMatrix::from_f32(&vec![0.0f32; 32], 1, 32).unwrap();
        let inp = Tensor::from_vec(vec![1.0f32; 64], vec![2, 4, 8]).unwrap();
        assert!(matches!(
            matmul_q8_direct(&inp, &qw),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn error_k_mismatch() {
        let qw  = Q8WeightMatrix::from_f32(&vec![0.0f32; 64], 2, 32).unwrap(); // k=32
        let inp = Tensor::from_vec(vec![1.0f32; 64], vec![2, 32]).unwrap();
        // This should succeed (k matches)
        let _ = matmul_q8_direct(&inp, &qw).unwrap();

        // Now a real mismatch: qw.k_in=64, input k=32
        let qw2  = Q8WeightMatrix::from_f32(&vec![0.0f32; 128], 2, 64).unwrap();
        let inp2 = Tensor::from_vec(vec![1.0f32; 32], vec![1, 32]).unwrap();
        assert!(matches!(
            matmul_q8_direct(&inp2, &qw2),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }
}
