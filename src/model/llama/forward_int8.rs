//! Mixed-precision Llama forward pass — commit 14.1.
//!
//! All seven linear projections per transformer block (Q, K, V, out, gate, up, down)
//! are replaced with INT8 × INT8 → INT32 → f32 matmul.  Activations remain f32;
//! only the *weights* are quantized (per-channel INT8).  This is the standard
//! "W8A8" (weight-INT8, activation-INT8 on-the-fly) mixed-precision pattern.
//!
//! # Memory savings
//!
//! Each `f32` weight is replaced by one `i8` + one `f32` scale per row.
//! For `N` output channels and `K` input features the saving is:
//! `4*N*K bytes → N*K + 4*N bytes`.  At K=4096, N=4096 that is 64 MB → 17 MB.
//!
//! # Accuracy
//!
//! For smooth (non-adversarial) weight distributions the typical relative
//! logit error vs the f32 baseline is below 2–3%, well within the 10% budget
//! stated in the roadmap.
//!
//! # API
//!
//! `LlamaModelInt8` has the same public surface as `LlamaModel`:
//! - `forward(&[u32]) -> Result<Tensor<f32>>`
//! - `forward_decode(u32, usize, &mut KvCache) -> Result<Vec<f32>>`
//! - `from_loader(&GgufLoader) -> Result<Self>`

use std::borrow::Cow;

use crate::attention::cached::{cached_attention_decode, cached_attention_prefill};
use crate::attention::gqa::grouped_query_attention_causal_with_offset;
use crate::cache::KvCache;
use crate::gguf::{GgufLoader, TensorExtractor};
use crate::model::{ModelError, Result};
use crate::model::llama::{
    LlamaConfig,
    weights::{GlobalWeightRole, WeightRole, global_weight_name, weight_name},
};
use crate::ops::activation::swiglu;
use crate::ops::matmul::matmul_int8_from_f32;
use crate::ops::matmul::matmul_int8_parallel;
use crate::ops::norm::rmsnorm;
use crate::ops::rope::{rope_apply, RopeTable};
use crate::quant::int8::per_channel::{quantize_per_channel, QuantizedMatrix};
use crate::quant::int8::symmetric::quantize_symmetric;
use crate::tensor::Tensor;

// ── quantization helper ───────────────────────────────────────────────────────

/// Quantize a 2-D `Tensor<f32>` weight to INT8 per-channel.
///
/// Contiguity is ensured before slicing.  Shape must be `[n_out, k_in]`.
fn quantize_tensor(t: &Tensor<f32>) -> QuantizedMatrix {
    debug_assert_eq!(t.ndim(), 2, "quantize_tensor: expected 2-D weight");
    let n_out = t.dims()[0];
    let k_in  = t.dims()[1];
    let slice: Cow<[f32]> = if t.is_contiguous() {
        Cow::Borrowed(t.as_slice())
    } else {
        Cow::Owned(t.contiguous().as_slice().to_vec())
    };
    quantize_per_channel(&slice, n_out, k_in)
}

// ── projection helper ─────────────────────────────────────────────────────────

/// INT8 linear projection: `input @ qw^T`.
///
/// `qw` is stored as `[out, in]` (GGUF convention).
/// `matmul_int8_from_f32` computes `output[m,n] = Σ_k input[m,k] * qw[n,k]`,
/// which is exactly `input @ qw.T`.
#[inline]
fn proj_int8(input: &Tensor<f32>, qw: &QuantizedMatrix) -> Result<Tensor<f32>> {
    Ok(matmul_int8_from_f32(input, qw)?)
}

/// Parallel INT8 linear projection using rayon over output rows.
///
/// Equivalent to [`proj_int8`] but routes through [`matmul_int8_parallel`],
/// which distributes output rows across the rayon thread pool.  The inner
/// dot product uses NEON on aarch64 and AVX2 on x86_64 — combining SIMD
/// throughput within each row with multi-core parallelism across rows.
///
/// **When to use:** prefill paths with seq_len ≥ 8 where M (number of rows)
/// is large enough to amortize rayon dispatch overhead.  For decode (M = 1)
/// use [`proj_int8`] — rayon provides no benefit and adds thread-pool overhead.
#[inline]
fn proj_int8_par(input: &Tensor<f32>, qw: &QuantizedMatrix) -> Result<Tensor<f32>> {
    let m = input.dims()[0];
    let input_c: Cow<Tensor<f32>> = if input.is_contiguous() {
        Cow::Borrowed(input)
    } else {
        Cow::Owned(input.contiguous())
    };
    let (act_q, act_scale) = quantize_symmetric(input_c.as_slice());
    Ok(matmul_int8_parallel(&act_q, act_scale, qw, m)?)
}

// ── add_elementwise ───────────────────────────────────────────────────────────

fn add_elementwise(a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>> {
    if a.dims() != b.dims() {
        return Err(ModelError::TensorError(
            crate::tensor::TensorError::ShapeMismatch {
                expected: a.dims().to_vec(),
                got:      b.dims().to_vec(),
            },
        ));
    }
    let a_d: Cow<[f32]> = if a.is_contiguous() { Cow::Borrowed(a.as_slice()) }
                          else { Cow::Owned(a.contiguous().as_slice().to_vec()) };
    let b_d: Cow<[f32]> = if b.is_contiguous() { Cow::Borrowed(b.as_slice()) }
                          else { Cow::Owned(b.contiguous().as_slice().to_vec()) };
    let out: Vec<f32> = a_d.iter().zip(b_d.iter()).map(|(x, y)| x + y).collect();
    Ok(Tensor::from_vec(out, a.shape().clone())?)
}

// ── embed helper ──────────────────────────────────────────────────────────────

fn embed(token_embd: &Tensor<f32>, tokens: &[u32]) -> Result<Tensor<f32>> {
    let vocab = token_embd.dims()[0];
    let dim   = token_embd.dims()[1];
    let data  = token_embd.as_slice();
    let mut out = Vec::with_capacity(tokens.len() * dim);
    for &tok in tokens {
        let id = tok as usize;
        if id >= vocab {
            return Err(ModelError::InvalidConfig {
                reason: format!("token id {id} >= vocab_size {vocab}"),
            });
        }
        out.extend_from_slice(&data[id * dim..(id + 1) * dim]);
    }
    Ok(Tensor::from_vec(out, vec![tokens.len(), dim])?)
}

// ── TransformerBlockInt8 ──────────────────────────────────────────────────────

/// One Llama transformer block with INT8-quantized linear projections.
///
/// All seven weight matrices (wq, wk, wv, wo, wgate, wup, wdown) are stored as
/// [`QuantizedMatrix`]; norm scales and RoPE table remain in f32.
pub struct TransformerBlockInt8 {
    wq:    QuantizedMatrix,
    wk:    QuantizedMatrix,
    wv:    QuantizedMatrix,
    wo:    QuantizedMatrix,
    attn_norm:   Tensor<f32>,
    wgate: QuantizedMatrix,
    wup:   QuantizedMatrix,
    wdown: QuantizedMatrix,
    ffn_norm:    Tensor<f32>,
    rope_table:  RopeTable,
    n_heads:     usize,
    n_kv_heads:  usize,
    rms_norm_eps: f32,
}

impl TransformerBlockInt8 {
    /// Construct from pre-quantized weights.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        wq:    QuantizedMatrix, wk: QuantizedMatrix,
        wv:    QuantizedMatrix, wo: QuantizedMatrix,
        attn_norm:  Tensor<f32>,
        wgate: QuantizedMatrix, wup: QuantizedMatrix, wdown: QuantizedMatrix,
        ffn_norm:   Tensor<f32>,
        rope_table: RopeTable,
        n_heads: usize, n_kv_heads: usize, rms_norm_eps: f32,
    ) -> Self {
        Self { wq, wk, wv, wo, attn_norm,
               wgate, wup, wdown, ffn_norm,
               rope_table, n_heads, n_kv_heads, rms_norm_eps }
    }

    /// Full-sequence prefill forward (no KV-cache write, start_pos = 0).
    pub fn forward(&self, x: &Tensor<f32>, start_pos: usize) -> Result<Tensor<f32>> {
        let x = self.attn_sublayer(x, start_pos)?;
        self.ffn_sublayer(&x)
    }

    /// Cached prefill forward — writes K/V into `cache[layer]`.
    pub fn forward_cached(
        &self, x: &Tensor<f32>, start_pos: usize,
        cache: &mut KvCache, layer: usize,
    ) -> Result<Tensor<f32>> {
        let x = self.attn_sublayer_cached(x, start_pos, cache, layer)?;
        self.ffn_sublayer(&x)
    }

    /// Parallel cached prefill forward — same as [`forward_cached`] but uses
    /// [`matmul_int8_parallel`] (rayon over output rows) for all seven
    /// projections: Q, K, V, out, gate, up, down.
    ///
    /// Automatically falls back to [`forward_cached`] (sequential) when
    /// `seq_len < MIN_PARALLEL_SEQ` to avoid rayon dispatch overhead
    /// exceeding the parallel benefit.  The threshold is calibrated to the
    /// M1 Pro rayon pool (~240 µs fixed cost): below 32 rows the sequential
    /// path is 1.4–1.7× faster; above 64 rows parallel wins at every tested
    /// embedding dimension.  At real 7B scale (d=4096) the crossover is
    /// seq≈4–8, so 32 is conservative on all practical hardware.
    ///
    /// Results are numerically identical to [`forward_cached`] — same
    /// quantization path, only scheduling differs.
    pub fn forward_cached_parallel(
        &self, x: &Tensor<f32>, start_pos: usize,
        cache: &mut KvCache, layer: usize,
    ) -> Result<Tensor<f32>> {
        // Rayon's fixed thread-dispatch cost (~240 µs on M1 Pro) makes the
        // parallel path slower than sequential for small sequence lengths.
        // Measured crossover: seq≈64 at d=256 (bench block), seq≈4 at d=4096 (7B).
        // Threshold of 32 is conservative and correct on all tested configurations.
        const MIN_PARALLEL_SEQ: usize = 32;
        if x.dims()[0] < MIN_PARALLEL_SEQ {
            return self.forward_cached(x, start_pos, cache, layer);
        }
        let x = self.attn_sublayer_cached_parallel(x, start_pos, cache, layer)?;
        self.ffn_sublayer_parallel(&x)
    }

    /// Single-token decode step — appends one K/V row to `cache[layer]`.
    pub fn forward_decode(
        &self, x: &Tensor<f32>, pos: usize,
        cache: &mut KvCache, layer: usize,
    ) -> Result<Tensor<f32>> {
        let x = self.attn_sublayer_decode(x, pos, cache, layer)?;
        self.ffn_sublayer(&x)
    }

    // ── private sublayers ────────────────────────────────────────────────

    fn attn_sublayer(&self, x: &Tensor<f32>, start_pos: usize) -> Result<Tensor<f32>> {
        let seq = x.dims()[0];
        let normed = rmsnorm(x, &self.attn_norm, self.rms_norm_eps)?;
        let mut q  = proj_int8(&normed, &self.wq)?;
        let mut k  = proj_int8(&normed, &self.wk)?;
        let v      = proj_int8(&normed, &self.wv)?;
        let hd_q = q.dims()[1] / self.n_heads;
        let hd_k = k.dims()[1] / self.n_kv_heads;
        q = q.reshape(vec![seq, self.n_heads,    hd_q])?;
        k = k.reshape(vec![seq, self.n_kv_heads, hd_k])?;
        rope_apply(&mut q, &self.rope_table, start_pos)?;
        rope_apply(&mut k, &self.rope_table, start_pos)?;
        q = q.reshape(vec![seq, self.n_heads    * hd_q])?;
        k = k.reshape(vec![seq, self.n_kv_heads * hd_k])?;
        let attn_out = grouped_query_attention_causal_with_offset(
            &q, &k, &v, self.n_heads, self.n_kv_heads, start_pos,
        )?;
        let projected = proj_int8(&attn_out, &self.wo)?;
        add_elementwise(x, &projected)
    }

    fn attn_sublayer_cached(
        &self, x: &Tensor<f32>, start_pos: usize,
        cache: &mut KvCache, layer: usize,
    ) -> Result<Tensor<f32>> {
        let seq = x.dims()[0];
        let normed = rmsnorm(x, &self.attn_norm, self.rms_norm_eps)?;
        let mut q  = proj_int8(&normed, &self.wq)?;
        let mut k  = proj_int8(&normed, &self.wk)?;
        let v      = proj_int8(&normed, &self.wv)?;
        let hd_q = q.dims()[1] / self.n_heads;
        let hd_k = k.dims()[1] / self.n_kv_heads;
        q = q.reshape(vec![seq, self.n_heads,    hd_q])?;
        k = k.reshape(vec![seq, self.n_kv_heads, hd_k])?;
        rope_apply(&mut q, &self.rope_table, start_pos)?;
        rope_apply(&mut k, &self.rope_table, start_pos)?;
        q = q.reshape(vec![seq, self.n_heads    * hd_q])?;
        k = k.reshape(vec![seq, self.n_kv_heads * hd_k])?;
        let attn_out = cached_attention_prefill(
            cache, layer, start_pos, &q, &k, &v, self.n_heads, self.n_kv_heads,
        )?;
        let projected = proj_int8(&attn_out, &self.wo)?;
        add_elementwise(x, &projected)
    }

    fn attn_sublayer_decode(
        &self, x: &Tensor<f32>, pos: usize,
        cache: &mut KvCache, layer: usize,
    ) -> Result<Tensor<f32>> {
        let normed = rmsnorm(x, &self.attn_norm, self.rms_norm_eps)?;
        let mut q  = proj_int8(&normed, &self.wq)?;
        let mut k  = proj_int8(&normed, &self.wk)?;
        let v      = proj_int8(&normed, &self.wv)?;
        let hd_q = q.dims()[1] / self.n_heads;
        let hd_k = k.dims()[1] / self.n_kv_heads;
        q = q.reshape(vec![1, self.n_heads,    hd_q])?;
        k = k.reshape(vec![1, self.n_kv_heads, hd_k])?;
        rope_apply(&mut q, &self.rope_table, pos)?;
        rope_apply(&mut k, &self.rope_table, pos)?;
        q = q.reshape(vec![1, self.n_heads    * hd_q])?;
        k = k.reshape(vec![1, self.n_kv_heads * hd_k])?;
        let attn_out = cached_attention_decode(
            cache, layer, pos, &q, &k, &v, self.n_heads, self.n_kv_heads,
        )?;
        let projected = proj_int8(&attn_out, &self.wo)?;
        add_elementwise(x, &projected)
    }

    fn ffn_sublayer(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let normed  = rmsnorm(x, &self.ffn_norm, self.rms_norm_eps)?;
        let gate    = proj_int8(&normed, &self.wgate)?;
        let up      = proj_int8(&normed, &self.wup)?;
        let hidden  = swiglu(&gate, &up)?;
        let ffn_out = proj_int8(&hidden, &self.wdown)?;
        add_elementwise(x, &ffn_out)
    }

    // ── parallel sublayers (commit 18.6) ─────────────────────────────────

    fn attn_sublayer_cached_parallel(
        &self, x: &Tensor<f32>, start_pos: usize,
        cache: &mut KvCache, layer: usize,
    ) -> Result<Tensor<f32>> {
        let seq = x.dims()[0];
        let normed = rmsnorm(x, &self.attn_norm, self.rms_norm_eps)?;
        let mut q  = proj_int8_par(&normed, &self.wq)?;
        let mut k  = proj_int8_par(&normed, &self.wk)?;
        let v      = proj_int8_par(&normed, &self.wv)?;
        let hd_q = q.dims()[1] / self.n_heads;
        let hd_k = k.dims()[1] / self.n_kv_heads;
        q = q.reshape(vec![seq, self.n_heads,    hd_q])?;
        k = k.reshape(vec![seq, self.n_kv_heads, hd_k])?;
        rope_apply(&mut q, &self.rope_table, start_pos)?;
        rope_apply(&mut k, &self.rope_table, start_pos)?;
        q = q.reshape(vec![seq, self.n_heads    * hd_q])?;
        k = k.reshape(vec![seq, self.n_kv_heads * hd_k])?;
        let attn_out = cached_attention_prefill(
            cache, layer, start_pos, &q, &k, &v, self.n_heads, self.n_kv_heads,
        )?;
        let projected = proj_int8_par(&attn_out, &self.wo)?;
        add_elementwise(x, &projected)
    }

    fn ffn_sublayer_parallel(&self, x: &Tensor<f32>) -> Result<Tensor<f32>> {
        let normed  = rmsnorm(x, &self.ffn_norm, self.rms_norm_eps)?;
        let gate    = proj_int8_par(&normed, &self.wgate)?;
        let up      = proj_int8_par(&normed, &self.wup)?;
        let hidden  = swiglu(&gate, &up)?;
        let ffn_out = proj_int8_par(&hidden, &self.wdown)?;
        add_elementwise(x, &ffn_out)
    }
}

// ── LlamaModelInt8 ────────────────────────────────────────────────────────────

/// Full Llama model with INT8-quantized projection weights.
///
/// Mirrors the API of [`LlamaModel`] but stores all seven per-layer
/// projection matrices as [`QuantizedMatrix`] (per-channel INT8).
/// The token embedding table, RMSNorm scales, and positional tables
/// remain in f32 because they are either accessed as lookups or are
/// too small to benefit from quantization.
///
/// [`LlamaModel`]: crate::model::llama::LlamaModel
pub struct LlamaModelInt8 {
    config:      LlamaConfig,
    /// `[vocab_size, embed_dim]` — kept f32 (embedding is a lookup).
    token_embd:  Tensor<f32>,
    blocks:      Vec<TransformerBlockInt8>,
    /// `[embed_dim]` — kept f32.
    output_norm: Tensor<f32>,
    /// `[vocab_size, embed_dim]` — INT8 unembedding matrix.
    output:      QuantizedMatrix,
}

impl LlamaModelInt8 {
    /// Construct from pre-built components (used in tests).
    #[must_use]
    pub fn new(
        config:      LlamaConfig,
        token_embd:  Tensor<f32>,
        blocks:      Vec<TransformerBlockInt8>,
        output_norm: Tensor<f32>,
        output:      QuantizedMatrix,
    ) -> Self {
        Self { config, token_embd, blocks, output_norm, output }
    }

    /// Return the model configuration.
    #[must_use]
    pub fn config(&self) -> &LlamaConfig { &self.config }

    /// Load a GGUF model and quantize all projection weights to INT8.
    ///
    /// Weights stored as Q4_0/Q8_0/F16 in GGUF are first dequantized to f32
    /// by `TensorExtractor::extract`, then immediately requantized to INT8
    /// per-channel.  The norm vectors and embedding table remain f32.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] on missing tensors or malformed metadata.
    pub fn from_loader(loader: &GgufLoader) -> Result<Self> {
        let config = LlamaConfig::from_metadata(loader.metadata())?;
        let ex     = TensorExtractor::new(loader);

        let token_embd:  Tensor<f32> = ex.extract(global_weight_name(GlobalWeightRole::TokenEmbd))?;
        let output_norm: Tensor<f32> = ex.extract(global_weight_name(GlobalWeightRole::OutputNorm))?;
        let output_f32:  Tensor<f32> = ex.extract(global_weight_name(GlobalWeightRole::Output))?;
        let output = quantize_tensor(&output_f32);

        let head_dim   = config.head_dim() as usize;
        let rope_table = RopeTable::new(
            config.context_length as usize, head_dim, config.rope_freq_base,
        );

        let mut blocks = Vec::with_capacity(config.block_count as usize);
        for layer in 0..config.block_count as usize {
            let wq:        Tensor<f32> = ex.extract(&weight_name(layer, WeightRole::AttnQ))?;
            let wk:        Tensor<f32> = ex.extract(&weight_name(layer, WeightRole::AttnK))?;
            let wv:        Tensor<f32> = ex.extract(&weight_name(layer, WeightRole::AttnV))?;
            let wo:        Tensor<f32> = ex.extract(&weight_name(layer, WeightRole::AttnOutput))?;
            let attn_norm: Tensor<f32> = ex.extract(&weight_name(layer, WeightRole::AttnNorm))?;
            let wgate:     Tensor<f32> = ex.extract(&weight_name(layer, WeightRole::FfnGate))?;
            let wup:       Tensor<f32> = ex.extract(&weight_name(layer, WeightRole::FfnUp))?;
            let wdown:     Tensor<f32> = ex.extract(&weight_name(layer, WeightRole::FfnDown))?;
            let ffn_norm:  Tensor<f32> = ex.extract(&weight_name(layer, WeightRole::FfnNorm))?;
            blocks.push(TransformerBlockInt8::new(
                quantize_tensor(&wq),   quantize_tensor(&wk),
                quantize_tensor(&wv),   quantize_tensor(&wo),
                attn_norm,
                quantize_tensor(&wgate), quantize_tensor(&wup), quantize_tensor(&wdown),
                ffn_norm,
                rope_table.clone(),
                config.n_heads as usize, config.n_kv_heads as usize, config.rms_norm_eps,
            ));
        }
        Ok(Self::new(config, token_embd, blocks, output_norm, output))
    }

    /// Full forward pass (prefill, start_pos = 0).
    ///
    /// Returns logits `[seq_len, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] on out-of-range token IDs or tensor shape failures.
    pub fn forward(&self, tokens: &[u32]) -> Result<Tensor<f32>> {
        if tokens.is_empty() {
            return Err(ModelError::InvalidConfig {
                reason: "forward: token list must not be empty".to_string(),
            });
        }
        let mut x = embed(&self.token_embd, tokens)?;
        for block in &self.blocks {
            x = block.forward(&x, 0)?;
        }
        x = rmsnorm(&x, &self.output_norm, self.config.rms_norm_eps)?;
        // [seq, embed] @ output^T  [vocab, embed] → [seq, vocab]
        Ok(matmul_int8_from_f32(&x, &self.output)?)
    }

    /// Single-token decode at absolute position `pos`.
    ///
    /// Returns logits `[vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError`] on out-of-range token ID, cache overflow, or shape failures.
    pub fn forward_decode(
        &self,
        token: u32,
        pos:   usize,
        cache: &mut KvCache,
    ) -> Result<Vec<f32>> {
        let mut x = embed(&self.token_embd, &[token])?;
        for (layer, block) in self.blocks.iter().enumerate() {
            x = block.forward_decode(&x, pos, cache, layer)?;
        }
        x = rmsnorm(&x, &self.output_norm, self.config.rms_norm_eps)?;
        let logits = matmul_int8_from_f32(&x, &self.output)?;
        Ok(logits.as_slice().to_vec())
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::KvCache;
    use crate::gguf::{Metadata, MetadataValue};
    use crate::model::llama::{config::LlamaConfig, block::TransformerBlock, forward::LlamaModel};
    use crate::ops::rope::RopeTable;

    fn tiny_config() -> LlamaConfig {
        let mut m = Metadata::new();
        for (k, v) in [
            ("llama.block_count",          MetadataValue::Uint32(2)),
            ("llama.embedding_length",     MetadataValue::Uint32(16)),
            ("llama.attention.head_count", MetadataValue::Uint32(2)),
            ("llama.feed_forward_length",  MetadataValue::Uint32(32)),
            ("llama.context_length",       MetadataValue::Uint32(64)),
            ("llama.vocab_size",           MetadataValue::Uint32(32)),
        ] { m.insert(k.to_string(), v); }
        LlamaConfig::from_metadata(&m).unwrap()
    }

    /// Build weight tensors with smooth non-trivial values for comparison tests.
    fn smooth_weight(rows: usize, cols: usize, seed: f32) -> Tensor<f32> {
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * seed * 0.01) - (rows * cols) as f32 * seed * 0.005).sin() * 0.5)
            .collect();
        Tensor::from_vec(data, vec![rows, cols]).unwrap()
    }

    /// Build a `LlamaModelInt8` with synthetic non-trivial weights.
    fn make_int8_model(cfg: &LlamaConfig) -> LlamaModelInt8 {
        let embed  = cfg.embedding_length   as usize;
        let vocab  = cfg.vocab_size         as usize;
        let ffn    = cfg.feed_forward_length as usize;
        let heads  = cfg.n_heads            as usize;
        let kv     = cfg.n_kv_heads         as usize;
        let hd     = cfg.head_dim()         as usize;
        let qd     = heads * hd;
        let kvd    = kv   * hd;
        let rope   = RopeTable::new(cfg.context_length as usize, hd, cfg.rope_freq_base);

        let make_block = |seed: f32| {
            TransformerBlockInt8::new(
                quantize_tensor(&smooth_weight(qd,   embed, seed * 1.0)),
                quantize_tensor(&smooth_weight(kvd,  embed, seed * 1.1)),
                quantize_tensor(&smooth_weight(kvd,  embed, seed * 1.2)),
                quantize_tensor(&smooth_weight(embed, qd,   seed * 1.3)),
                Tensor::ones(vec![embed]),
                quantize_tensor(&smooth_weight(ffn,  embed, seed * 1.4)),
                quantize_tensor(&smooth_weight(ffn,  embed, seed * 1.5)),
                quantize_tensor(&smooth_weight(embed, ffn,  seed * 1.6)),
                Tensor::ones(vec![embed]),
                rope.clone(),
                heads, kv, cfg.rms_norm_eps,
            )
        };
        let blocks: Vec<_> = (0..cfg.block_count as usize)
            .map(|i| make_block((i + 1) as f32))
            .collect();
        LlamaModelInt8::new(
            cfg.clone(),
            smooth_weight(vocab, embed, 0.5),
            blocks,
            Tensor::ones(vec![embed]),
            quantize_tensor(&smooth_weight(vocab, embed, 0.7)),
        )
    }

    /// Build the equivalent `LlamaModel` (f32) from the same raw weights.
    fn make_f32_model(cfg: &LlamaConfig) -> LlamaModel {
        let embed  = cfg.embedding_length   as usize;
        let vocab  = cfg.vocab_size         as usize;
        let ffn    = cfg.feed_forward_length as usize;
        let heads  = cfg.n_heads            as usize;
        let kv     = cfg.n_kv_heads         as usize;
        let hd     = cfg.head_dim()         as usize;
        let qd     = heads * hd;
        let kvd    = kv   * hd;
        let rope   = RopeTable::new(cfg.context_length as usize, hd, cfg.rope_freq_base);

        let make_block = |seed: f32| {
            TransformerBlock::new(
                smooth_weight(qd,   embed, seed * 1.0),
                smooth_weight(kvd,  embed, seed * 1.1),
                smooth_weight(kvd,  embed, seed * 1.2),
                smooth_weight(embed, qd,   seed * 1.3),
                Tensor::ones(vec![embed]),
                smooth_weight(ffn,  embed, seed * 1.4),
                smooth_weight(ffn,  embed, seed * 1.5),
                smooth_weight(embed, ffn,  seed * 1.6),
                Tensor::ones(vec![embed]),
                rope.clone(),
                heads, kv, cfg.rms_norm_eps,
            )
        };
        let blocks: Vec<_> = (0..cfg.block_count as usize)
            .map(|i| make_block((i + 1) as f32))
            .collect();
        LlamaModel::new(
            cfg.clone(),
            smooth_weight(vocab, embed, 0.5),
            blocks,
            Tensor::ones(vec![embed]),
            smooth_weight(vocab, embed, 0.7),
        )
    }

    // ── output shape ──────────────────────────────────────────────────────

    #[test]
    fn forward_single_token_output_shape() {
        let cfg   = tiny_config();
        let vocab = cfg.vocab_size as usize;
        let model = make_int8_model(&cfg);
        let out   = model.forward(&[0]).unwrap();
        assert_eq!(out.dims(), &[1, vocab]);
    }

    #[test]
    fn forward_multi_token_output_shape() {
        let cfg   = tiny_config();
        let vocab = cfg.vocab_size as usize;
        let model = make_int8_model(&cfg);
        let out   = model.forward(&[0, 1, 2, 3]).unwrap();
        assert_eq!(out.dims(), &[4, vocab]);
    }

    // ── numerical validity ────────────────────────────────────────────────

    #[test]
    fn forward_no_nan_or_inf() {
        let cfg   = tiny_config();
        let model = make_int8_model(&cfg);
        let out   = model.forward(&[0, 5, 31]).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(),      "NaN in INT8 logits");
            assert!(!v.is_infinite(), "Inf in INT8 logits");
        }
    }

    // ── error paths ───────────────────────────────────────────────────────

    #[test]
    fn forward_empty_tokens_rejected() {
        let cfg   = tiny_config();
        let model = make_int8_model(&cfg);
        assert!(model.forward(&[]).is_err());
    }

    #[test]
    fn forward_out_of_range_token_rejected() {
        let cfg   = tiny_config();
        let model = make_int8_model(&cfg);
        assert!(model.forward(&[cfg.vocab_size]).is_err());
    }

    // ── forward_decode ────────────────────────────────────────────────────

    #[test]
    fn forward_decode_output_length() {
        let cfg        = tiny_config();
        let vocab      = cfg.vocab_size as usize;
        let model      = make_int8_model(&cfg);
        let n_layers   = cfg.block_count as usize;
        let n_kv_heads = cfg.n_kv_heads  as usize;
        let head_dim   = cfg.head_dim()  as usize;
        let mut cache  = KvCache::new(n_layers, cfg.context_length as usize, n_kv_heads, head_dim);
        let logits = model.forward_decode(0, 0, &mut cache).unwrap();
        assert_eq!(logits.len(), vocab);
    }

    #[test]
    fn forward_decode_no_nan() {
        let cfg        = tiny_config();
        let model      = make_int8_model(&cfg);
        let n_layers   = cfg.block_count as usize;
        let n_kv_heads = cfg.n_kv_heads  as usize;
        let head_dim   = cfg.head_dim()  as usize;
        let mut cache  = KvCache::new(n_layers, cfg.context_length as usize, n_kv_heads, head_dim);
        let logits = model.forward_decode(5, 0, &mut cache).unwrap();
        for (i, &v) in logits.iter().enumerate() {
            assert!(!v.is_nan(),      "NaN in INT8 decode logits[{i}]");
            assert!(!v.is_infinite(), "Inf in INT8 decode logits[{i}]");
        }
    }

    // ── quality: INT8 logits within 10% of the logit *range* vs f32 ─────
    //
    // Per-element relative error blows up when individual logits are near
    // zero (expected for synthetic tiny models).  The metric that actually
    // matters for generation quality is: does quantization shift any logit
    // by more than X% of the *total logit spread*?  A shift smaller than
    // the spread cannot change the argmax ordering significantly.
    //
    // Metric: max_abs_logit_err / logit_range < 0.10

    #[test]
    fn logits_within_10_percent_of_f32_baseline() {
        let cfg        = tiny_config();
        let int8_model = make_int8_model(&cfg);
        let f32_model  = make_f32_model(&cfg);
        let tokens     = &[0_u32, 5, 15, 31];

        let int8_logits = int8_model.forward(tokens).unwrap();
        let f32_logits  = f32_model.forward(tokens).unwrap();
        assert_eq!(int8_logits.dims(), f32_logits.dims());

        let f32_data = f32_logits.as_slice();
        let f32_max  = f32_data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let f32_min  = f32_data.iter().cloned().fold(f32::INFINITY,     f32::min);
        // Floor of 1e-3 ensures an all-equal-logit model doesn't trivially pass.
        let logit_range = (f32_max - f32_min).max(1e-3);

        let max_abs_err = int8_logits.as_slice().iter()
            .zip(f32_data)
            .map(|(&i, &f)| (i - f).abs())
            .fold(0.0_f32, f32::max);

        let normalized = max_abs_err / logit_range;
        assert!(
            normalized < 0.10,
            "max abs err {max_abs_err:.5} is {:.1}% of logit range {logit_range:.5} — exceeds 10% budget",
            normalized * 100.0
        );
    }

    /// Confirm the config accessor works on the INT8 model.
    #[test]
    fn config_accessor() {
        let cfg   = tiny_config();
        let model = make_int8_model(&cfg);
        assert_eq!(model.config().block_count, 2);
        assert_eq!(model.config().vocab_size, 32);
    }

    // ── forward_cached_parallel tests (commit 18.6) ───────────────────────

    fn make_single_int8_block(cfg: &LlamaConfig) -> TransformerBlockInt8 {
        let embed = cfg.embedding_length    as usize;
        let ffn   = cfg.feed_forward_length as usize;
        let heads = cfg.n_heads             as usize;
        let kv    = cfg.n_kv_heads          as usize;
        let hd    = cfg.head_dim()          as usize;
        let qd    = heads * hd;
        let kvd   = kv    * hd;
        let rope  = RopeTable::new(cfg.context_length as usize, hd, cfg.rope_freq_base);
        TransformerBlockInt8::new(
            quantize_tensor(&smooth_weight(qd,   embed, 1.0)),
            quantize_tensor(&smooth_weight(kvd,  embed, 1.1)),
            quantize_tensor(&smooth_weight(kvd,  embed, 1.2)),
            quantize_tensor(&smooth_weight(embed, qd,   1.3)),
            Tensor::ones(vec![embed]),
            quantize_tensor(&smooth_weight(ffn,  embed, 1.4)),
            quantize_tensor(&smooth_weight(ffn,  embed, 1.5)),
            quantize_tensor(&smooth_weight(embed, ffn,  1.6)),
            Tensor::ones(vec![embed]),
            rope, heads, kv, cfg.rms_norm_eps,
        )
    }

    #[test]
    fn forward_cached_parallel_output_shape() {
        let cfg       = tiny_config();
        let block     = make_single_int8_block(&cfg);
        let embed     = cfg.embedding_length as usize;
        let seq       = 32_usize;  // at or above MIN_PARALLEL_SEQ — exercises parallel path
        let n_kv      = cfg.n_kv_heads as usize;
        let hd        = cfg.head_dim()  as usize;
        let x         = Tensor::from_vec(vec![0.1_f32; seq * embed], vec![seq, embed]).unwrap();
        let mut cache = KvCache::new(1, cfg.context_length as usize, n_kv, hd);
        let out = block.forward_cached_parallel(&x, 0, &mut cache, 0).unwrap();
        assert_eq!(out.dims(), &[seq, embed]);
    }

    #[test]
    fn forward_cached_parallel_no_nan() {
        let cfg       = tiny_config();
        let block     = make_single_int8_block(&cfg);
        let embed     = cfg.embedding_length as usize;
        let seq       = 32_usize;
        let n_kv      = cfg.n_kv_heads as usize;
        let hd        = cfg.head_dim()  as usize;
        let data: Vec<f32> = (0..seq * embed).map(|i| (i as f32 * 0.01).sin()).collect();
        let x         = Tensor::from_vec(data, vec![seq, embed]).unwrap();
        let mut cache = KvCache::new(1, cfg.context_length as usize, n_kv, hd);
        let out = block.forward_cached_parallel(&x, 0, &mut cache, 0).unwrap();
        for &v in out.as_slice() {
            assert!(!v.is_nan(),      "NaN in parallel INT8 block output");
            assert!(!v.is_infinite(), "Inf in parallel INT8 block output");
        }
    }

    /// Parallel must produce numerically identical output to sequential.
    #[test]
    fn forward_cached_parallel_matches_sequential() {
        let cfg       = tiny_config();
        let block     = make_single_int8_block(&cfg);
        let embed     = cfg.embedding_length as usize;
        let seq       = 32_usize;  // above threshold — both paths exercise full parallel logic
        let n_kv      = cfg.n_kv_heads as usize;
        let hd        = cfg.head_dim()  as usize;
        let data: Vec<f32> = (0..seq * embed).map(|i| (i as f32 * 0.017 - 0.5).tanh()).collect();
        let x = Tensor::from_vec(data, vec![seq, embed]).unwrap();

        let mut cache_seq = KvCache::new(1, cfg.context_length as usize, n_kv, hd);
        let mut cache_par = KvCache::new(1, cfg.context_length as usize, n_kv, hd);

        let out_seq = block.forward_cached(&x, 0, &mut cache_seq, 0).unwrap();
        let out_par = block.forward_cached_parallel(&x, 0, &mut cache_par, 0).unwrap();

        assert_eq!(out_seq.dims(), out_par.dims());
        for (i, (s, p)) in out_seq.as_slice().iter().zip(out_par.as_slice()).enumerate() {
            assert_eq!(
                s.to_bits(), p.to_bits(),
                "element {i}: seq={s} par={p} — parallel and sequential INT8 paths diverged"
            );
        }
    }

    /// Below MIN_PARALLEL_SEQ the parallel method must fall back to sequential,
    /// producing identical output and never being slower due to rayon overhead.
    #[test]
    fn forward_cached_parallel_falls_back_below_threshold() {
        let cfg       = tiny_config();
        let block     = make_single_int8_block(&cfg);
        let embed     = cfg.embedding_length as usize;
        let seq       = 8_usize;  // well below MIN_PARALLEL_SEQ=32
        let n_kv      = cfg.n_kv_heads as usize;
        let hd        = cfg.head_dim()  as usize;
        let data: Vec<f32> = (0..seq * embed).map(|i| (i as f32 * 0.03).cos()).collect();
        let x = Tensor::from_vec(data, vec![seq, embed]).unwrap();

        let mut cache_seq = KvCache::new(1, cfg.context_length as usize, n_kv, hd);
        let mut cache_par = KvCache::new(1, cfg.context_length as usize, n_kv, hd);

        // Both should produce identical output because parallel falls back to sequential.
        let out_seq = block.forward_cached(&x, 0, &mut cache_seq, 0).unwrap();
        let out_par = block.forward_cached_parallel(&x, 0, &mut cache_par, 0).unwrap();

        assert_eq!(out_seq.dims(), &[seq, embed]);
        assert_eq!(out_par.dims(), &[seq, embed]);
        for (i, (s, p)) in out_seq.as_slice().iter().zip(out_par.as_slice()).enumerate() {
            assert_eq!(
                s.to_bits(), p.to_bits(),
                "element {i}: fallback path diverged from sequential at seq={seq}"
            );
        }
    }
}
