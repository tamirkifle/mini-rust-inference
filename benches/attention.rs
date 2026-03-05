//! Criterion micro-benchmarks for attention, normalization, and softmax — commit 17.1.
//!
//! Covers:
//!   - Scaled dot-product attention (SDPA) at various sequence lengths
//!   - Grouped-query attention (GQA) causal — Llama-2 style config
//!   - RMSNorm: scalar vs SIMD at LLM-scale hidden widths
//!   - Softmax: scalar vs SIMD over attention logits
//!
//! GQA config used: n_heads=8, n_kv_heads=2, d_k=64 — proportional to
//! Llama-2 70B's 64-head / 8-kv-head configuration.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use llm_engine::attention::{
    grouped_query_attention_causal, scaled_dot_product_attention,
};
use llm_engine::ops::norm::{rmsnorm, rmsnorm_simd};
use llm_engine::ops::softmax::softmax;
use llm_engine::ops::softmax_simd::softmax_simd;
use llm_engine::tensor::Tensor;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_f32(rows: usize, cols: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32) * 0.01 - (rows * cols / 2) as f32 * 0.01)
        .collect();
    Tensor::from_vec(data, vec![rows, cols]).unwrap()
}

fn make_ones(n: usize) -> Tensor<f32> {
    Tensor::ones(vec![n])
}

// ── SDPA ──────────────────────────────────────────────────────────────────────

fn bench_sdpa(c: &mut Criterion) {
    let mut group = c.benchmark_group("sdpa");

    // d_k = 64 (standard per-head dimension)
    let d_k = 64_usize;

    for &seq in &[32usize, 128, 512] {
        let q = make_f32(seq, d_k);
        let k = make_f32(seq, d_k);
        let v = make_f32(seq, d_k);

        // Throughput expressed as Q·K^T entries computed
        group.throughput(Throughput::Elements((seq * seq) as u64));

        group.bench_with_input(BenchmarkId::new("seq", seq), &seq, |bench, _| {
            bench.iter(|| scaled_dot_product_attention(&q, &k, &v).unwrap());
        });
    }

    group.finish();
}

// ── GQA causal ───────────────────────────────────────────────────────────────

fn bench_gqa(c: &mut Criterion) {
    let mut group = c.benchmark_group("gqa_causal");

    let n_heads    = 8_usize;
    let n_kv_heads = 2_usize;
    let d_k        = 64_usize;

    for &seq in &[32usize, 128, 512] {
        let q = make_f32(seq, n_heads * d_k);
        let k = make_f32(seq, n_kv_heads * d_k);
        let v = make_f32(seq, n_kv_heads * d_k);

        // Throughput: n_heads independent seq×seq attention maps
        group.throughput(Throughput::Elements((n_heads * seq * seq) as u64));

        group.bench_with_input(BenchmarkId::new("seq", seq), &seq, |bench, _| {
            bench.iter(|| {
                grouped_query_attention_causal(&q, &k, &v, n_heads, n_kv_heads).unwrap()
            });
        });
    }

    group.finish();
}

// ── RMSNorm ───────────────────────────────────────────────────────────────────

fn bench_rmsnorm(c: &mut Criterion) {
    let mut group = c.benchmark_group("rmsnorm");

    for &d in &[512usize, 2048, 4096] {
        let x = make_f32(1, d); // single vector (decode path)
        let w = make_ones(d);

        group.throughput(Throughput::Elements(d as u64));

        group.bench_with_input(BenchmarkId::new("scalar", d), &d, |bench, _| {
            bench.iter(|| rmsnorm(&x, &w, 1e-5).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("simd", d), &d, |bench, _| {
            bench.iter(|| rmsnorm_simd(&x, &w, 1e-5).unwrap());
        });
    }

    group.finish();

    // Also benchmark with a batch (prefill-like: [seq, d])
    let mut batch_group = c.benchmark_group("rmsnorm_batch");
    let d = 4096_usize;
    let w = make_ones(d);

    for &seq in &[8usize, 32, 128] {
        let x = make_f32(seq, d);
        batch_group.throughput(Throughput::Elements((seq * d) as u64));

        batch_group.bench_with_input(BenchmarkId::new("scalar_seq", seq), &seq, |bench, _| {
            bench.iter(|| rmsnorm(&x, &w, 1e-5).unwrap());
        });

        batch_group.bench_with_input(BenchmarkId::new("simd_seq", seq), &seq, |bench, _| {
            bench.iter(|| rmsnorm_simd(&x, &w, 1e-5).unwrap());
        });
    }

    batch_group.finish();
}

// ── Softmax ───────────────────────────────────────────────────────────────────

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax");

    // Attention logit row widths: seq_len tokens
    for &n in &[128usize, 512, 2048, 32768] {
        let x = make_f32(1, n);
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("scalar", n), &n, |bench, _| {
            bench.iter(|| softmax(&x).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("simd", n), &n, |bench, _| {
            bench.iter(|| softmax_simd(&x).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_sdpa, bench_gqa, bench_rmsnorm, bench_softmax);
criterion_main!(benches);
