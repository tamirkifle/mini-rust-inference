//! INT4 (Q4_0) dequantized matrix multiplication.
//!
//! Rather than materialising the full f32 weight matrix, this kernel
//! dequantises Q4_0 blocks **lazily** — one 32-element block at a time —
//! immediately before the inner dot-product loop.  This halves peak working-
//! set memory versus a two-pass (dequant-then-multiply) approach.
//!
//! # Layout contract
//!
//! The `weight_q4` byte slice encodes a `[N, K]` weight matrix in row-major
//! Q4_0 order:
//!
//! ```text
//! row 0 : blocks[0..K/32]   (K/32 × 18 bytes)
//! row 1 : blocks[K/32..2K/32]
//! ...
//! row N-1
//! ```
//!
//! Each 18-byte Q4_0 block holds one f16 scale followed by 16 packed-nibble
//! bytes (two 4-bit values per byte, low nibble first, biased by +8).
//!
//! # Example
//!
//! ```
//! use llm_engine::gguf::dequant::create_q4_0_block;
//! use llm_engine::ops::matmul::q4_0::matmul_q4_0_dequant;
//! use llm_engine::Tensor;
//!
//! // 1-row weight [1, 32], packed as one Q4_0 block
//! let w_vals: Vec<f32> = (0..32).map(|i| i as f32).collect();
//! let block = create_q4_0_block(&w_vals);
//! let weight_q4: Vec<u8> = block.to_vec();
//!
//! // input [2, 32]
//! let input = Tensor::from_vec(vec![1.0f32; 64], vec![2, 32]).unwrap();
//!
//! let out = matmul_q4_0_dequant(&input, &weight_q4, 1, 32).unwrap();
//! assert_eq!(out.dims(), &[2, 1]);
//! ```

use std::borrow::Cow;

use crate::gguf::dequant::{Q4_0_BLOCK_ELEMENTS, Q4_0_BLOCK_SIZE};
use crate::gguf::f16_to_f32;
use crate::tensor::{Result, Tensor, TensorError};

// Re-export as local aliases for readability.
const BLOCK_ELEMS: usize = Q4_0_BLOCK_ELEMENTS; // 32
const BLOCK_BYTES: usize = Q4_0_BLOCK_SIZE; // 18

// ── hot inner helper ────────────────────────────────────────────────────────

/// Dequantises one Q4_0 block (exactly `BLOCK_BYTES` bytes) into `out[0..32]`.
///
/// Marked `#[inline(always)]` so the compiler can eliminate the call overhead
/// and auto-vectorise the loop when optimising.
#[inline(always)]
fn dequant_block_into(block: &[u8], out: &mut [f32; BLOCK_ELEMS]) {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..];
    for (i, &byte) in qs.iter().enumerate() {
        let q_lo = (byte & 0x0F) as i32 - 8;
        let q_hi = ((byte >> 4) & 0x0F) as i32 - 8;
        out[i * 2] = scale * q_lo as f32;
        out[i * 2 + 1] = scale * q_hi as f32;
    }
}

// ── public API ───────────────────────────────────────────────────────────────

/// Computes `output = input × weight^T` where `weight` is stored in Q4_0 format.
///
/// This is the standard linear-layer operation for inference:
/// `input` carries activations; `weight` is the pre-quantised parameter matrix.
///
/// # Arguments
///
/// * `input`     – `[M, K]` f32 activation tensor (contiguous or not; copied if non-contiguous)
/// * `weight_q4` – raw Q4_0 bytes encoding a weight matrix of logical shape `[N, K]`
/// * `n_out`     – `N`, number of output features (rows of the weight matrix)
/// * `k`         – `K`, inner dimension; must equal `input.dims()[1]` and be a multiple of 32
///
/// # Returns
///
/// New contiguous `Tensor<f32>` of shape `[M, N]`.
///
/// # Errors
///
/// * [`TensorError::InvalidShape`]  – if `input` is not 2-D, `k % 32 != 0`, or the byte
///   slice length is inconsistent with `(n_out, k)`.
/// * [`TensorError::ShapeMismatch`] – if `input.dims()[1] != k`.
#[must_use = "returns a new tensor; result is not used in-place"]
pub fn matmul_q4_0_dequant(
    input: &Tensor<f32>,
    weight_q4: &[u8],
    n_out: usize,
    k: usize,
) -> Result<Tensor<f32>> {
    // ── dimensionality ─────────────────────────────────────────────────────
    if input.ndim() != 2 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_q4_0_dequant: `input` must be 2-D, got {}D (shape {:?})",
                input.ndim(),
                input.dims()
            ),
        });
    }

    let m = input.dims()[0];
    let k_in = input.dims()[1];

    if k_in != k {
        return Err(TensorError::ShapeMismatch {
            expected: vec![m, k],
            got: vec![m, k_in],
        });
    }

    if k % BLOCK_ELEMS != 0 {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_q4_0_dequant: k={k} must be a multiple of {BLOCK_ELEMS} (Q4_0 block size)"
            ),
        });
    }

    let blocks_per_row = k / BLOCK_ELEMS;
    let expected_bytes = n_out * blocks_per_row * BLOCK_BYTES;

    if weight_q4.len() != expected_bytes {
        return Err(TensorError::InvalidShape {
            reason: format!(
                "matmul_q4_0_dequant: weight_q4 has {} bytes, expected {expected_bytes} \
                 for shape [{n_out}, {k}]",
                weight_q4.len()
            ),
        });
    }

    // ── contiguity gate ────────────────────────────────────────────────────
    let input_c: Cow<Tensor<f32>> = if input.is_contiguous() {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(input.contiguous())
    };
    let inp = input_c.as_slice();

    // ── main kernel ────────────────────────────────────────────────────────
    // Loop order: output-row (n) × input-row (m) × block (b) × element (i)
    // For each (m, n) pair we accumulate a scalar dot product over K.
    // Re-using a fixed 32-float buffer avoids any allocation inside the loop.
    let mut out = vec![0.0_f32; m * n_out];
    let mut block_buf = [0.0_f32; BLOCK_ELEMS];

    for n in 0..n_out {
        let row_byte_base = n * blocks_per_row * BLOCK_BYTES;

        for m_i in 0..m {
            let mut acc = 0.0_f32;
            let inp_row = &inp[m_i * k..(m_i + 1) * k];

            for b in 0..blocks_per_row {
                let blk_start = row_byte_base + b * BLOCK_BYTES;
                dequant_block_into(
                    &weight_q4[blk_start..blk_start + BLOCK_BYTES],
                    &mut block_buf,
                );
                let inp_blk = &inp_row[b * BLOCK_ELEMS..(b + 1) * BLOCK_ELEMS];
                for i in 0..BLOCK_ELEMS {
                    acc += inp_blk[i] * block_buf[i];
                }
            }

            out[m_i * n_out + n] = acc;
        }
    }

    Tensor::from_vec(out, vec![m, n_out])
}

// ── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::dequant::{create_q4_0_block, dequantize_q4_0};
    use crate::ops::matmul::naive::matmul_naive;

    const EPS: f32 = 1e-5;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }
    fn close_slice(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| close(*x, *y))
    }

    /// Build Q4_0 bytes for a weight matrix whose rows are given by `rows`.
    fn pack_weight(rows: &[Vec<f32>]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for row in rows {
            assert!(row.len() % 32 == 0, "row length must be multiple of 32");
            for chunk in row.chunks(32) {
                bytes.extend_from_slice(&create_q4_0_block(chunk));
            }
        }
        bytes
    }

    // ── correctness: q4_0 matmul == naive matmul on deq'd weights ───────────

    #[test]
    fn test_matches_naive_single_block() {
        // 1-row weight [1, 32], 1-row input [1, 32]
        let w_row: Vec<f32> = (0..32).map(|i| i as f32 * 0.1 - 1.5).collect();
        let weight_q4 = pack_weight(&[w_row.clone()]);

        // Build deq'd weight tensor for naive reference
        let w_deq_bytes = dequantize_q4_0(&weight_q4).unwrap();
        let w_deq = Tensor::from_vec(w_deq_bytes, vec![1, 32]).unwrap();

        let inp: Vec<f32> = (0..32).map(|i| i as f32 * 0.05).collect();
        let input = Tensor::from_vec(inp.clone(), vec![1, 32]).unwrap();

        let q4_out = matmul_q4_0_dequant(&input, &weight_q4, 1, 32).unwrap();
        let ref_out = matmul_naive(&input, &w_deq.transpose(0, 1).unwrap()).unwrap();

        assert_eq!(q4_out.dims(), &[1, 1]);
        assert!(
            close_slice(q4_out.as_slice(), ref_out.as_slice()),
            "q4={:?} ref={:?}",
            q4_out.as_slice(),
            ref_out.as_slice()
        );
    }

    #[test]
    fn test_matches_naive_multi_row_weight() {
        // weight [4, 64], input [3, 64]
        let weight_rows: Vec<Vec<f32>> = (0..4)
            .map(|n| (0..64).map(|k| ((n * 64 + k) as f32) * 0.02 - 0.5).collect())
            .collect();
        let weight_q4 = pack_weight(&weight_rows);

        let w_deq_bytes = dequantize_q4_0(&weight_q4).unwrap();
        let w_deq = Tensor::from_vec(w_deq_bytes, vec![4, 64]).unwrap();

        let inp_data: Vec<f32> = (0..192).map(|i| i as f32 * 0.01).collect();
        let input = Tensor::from_vec(inp_data, vec![3, 64]).unwrap();

        let q4_out = matmul_q4_0_dequant(&input, &weight_q4, 4, 64).unwrap();
        let ref_out = matmul_naive(&input, &w_deq.transpose(0, 1).unwrap()).unwrap();

        assert_eq!(q4_out.dims(), &[3, 4]);
        assert!(
            close_slice(q4_out.as_slice(), ref_out.as_slice()),
            "q4={:?}\nref={:?}",
            q4_out.as_slice(),
            ref_out.as_slice()
        );
    }

    #[test]
    fn test_matches_naive_zeros_input() {
        let w_row: Vec<f32> = vec![1.0f32; 32];
        let weight_q4 = pack_weight(&[w_row]);
        let input = Tensor::zeros(vec![2, 32]);

        let out = matmul_q4_0_dequant(&input, &weight_q4, 1, 32).unwrap();
        assert_eq!(out.dims(), &[2, 1]);
        for &v in out.as_slice() {
            assert!(v.abs() < EPS, "expected ~0, got {v}");
        }
    }

    #[test]
    fn test_non_contiguous_input() {
        // Transpose a [64, 2] tensor to get a non-contiguous [2, 64] input
        let data: Vec<f32> = (0..128).map(|i| i as f32 * 0.01).collect();
        let t = Tensor::from_vec(data, vec![64, 2]).unwrap();
        let input_nc = t.transpose(0, 1).unwrap(); // [2, 64] non-contiguous
        assert!(!input_nc.is_contiguous());

        let w_rows: Vec<Vec<f32>> = (0..3)
            .map(|n| (0..64).map(|k| ((n + k) as f32) * 0.03 - 0.5).collect())
            .collect();
        let weight_q4 = pack_weight(&w_rows);
        let w_deq_bytes = dequantize_q4_0(&weight_q4).unwrap();
        let w_deq = Tensor::from_vec(w_deq_bytes, vec![3, 64]).unwrap();

        let out_q4 = matmul_q4_0_dequant(&input_nc, &weight_q4, 3, 64).unwrap();
        let input_cont = input_nc.contiguous();
        let ref_out = matmul_naive(&input_cont, &w_deq.transpose(0, 1).unwrap()).unwrap();

        assert_eq!(out_q4.dims(), &[2, 3]);
        assert!(close_slice(out_q4.as_slice(), ref_out.as_slice()));
    }

    // ── error handling ───────────────────────────────────────────────────────

    #[test]
    fn test_error_non_2d_input() {
        let input = Tensor::from_vec(vec![1.0f32; 64], vec![2, 4, 8]).unwrap();
        let dummy = vec![0u8; 18]; // 1 block
        assert!(matches!(
            matmul_q4_0_dequant(&input, &dummy, 1, 32),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_error_k_not_multiple_of_32() {
        let input = Tensor::from_vec(vec![1.0f32; 33], vec![1, 33]).unwrap();
        let dummy = vec![0u8; 18];
        assert!(matches!(
            matmul_q4_0_dequant(&input, &dummy, 1, 33),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_error_k_mismatch() {
        let input = Tensor::from_vec(vec![1.0f32; 32], vec![1, 32]).unwrap();
        let dummy = vec![0u8; 18];
        // passing k=64 but input has k=32
        assert!(matches!(
            matmul_q4_0_dequant(&input, &dummy, 1, 64),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn test_error_wrong_byte_length() {
        let input = Tensor::from_vec(vec![1.0f32; 32], vec![1, 32]).unwrap();
        // correct for (n_out=1, k=32) is 18 bytes; we pass 20
        let dummy = vec![0u8; 20];
        assert!(matches!(
            matmul_q4_0_dequant(&input, &dummy, 1, 32),
            Err(TensorError::InvalidShape { .. })
        ));
    }

    #[test]
    fn test_output_shape() {
        let weight_rows: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0f32; 128]).collect();
        let weight_q4 = pack_weight(&weight_rows);
        let input = Tensor::zeros(vec![5, 128]);

        let out = matmul_q4_0_dequant(&input, &weight_q4, 8, 128).unwrap();
        assert_eq!(out.dims(), &[5, 8]);
    }

    #[test]
    fn test_larger_weight_correctness() {
        // weight [8, 128], input [4, 128] — exercise multiple blocks per row
        let weight_rows: Vec<Vec<f32>> = (0..8)
            .map(|n| (0..128).map(|k| ((n * 128 + k) as f32) * 0.01 - 0.5).collect())
            .collect();
        let weight_q4 = pack_weight(&weight_rows);
        let w_deq_bytes = dequantize_q4_0(&weight_q4).unwrap();
        let w_deq = Tensor::from_vec(w_deq_bytes, vec![8, 128]).unwrap();

        let inp_data: Vec<f32> = (0..512).map(|i| i as f32 * 0.005).collect();
        let input = Tensor::from_vec(inp_data, vec![4, 128]).unwrap();

        let q4_out = matmul_q4_0_dequant(&input, &weight_q4, 8, 128).unwrap();
        let ref_out = matmul_naive(&input, &w_deq.transpose(0, 1).unwrap()).unwrap();

        assert_eq!(q4_out.dims(), &[4, 8]);
        assert!(
            close_slice(q4_out.as_slice(), ref_out.as_slice()),
            "max diff = {}",
            q4_out
                .as_slice()
                .iter()
                .zip(ref_out.as_slice())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max)
        );
    }
}
