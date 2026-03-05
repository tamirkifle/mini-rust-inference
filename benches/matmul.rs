//! Criterion micro-benchmarks for matrix multiplication kernels — commit 17.1.
//!
//! Covers:
//!   - f32 kernels: naive, blocked (32×32 tiles), AVX2/NEON, rayon-parallel
//!   - INT8 kernel: matmul_int8_from_f32 (per-channel quantized weights)
//!
//! Matrix sizes chosen to span L1-hot (128), L2-resident (512),
//! and LLC-pressure (1024) regimes, matching realistic projection
//! layer dimensions in a 7B-parameter Llama model.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use llm_engine::ops::matmul::{
    matmul_avx2, matmul_blocked, matmul_int8_from_f32, matmul_naive, matmul_parallel,
};
use llm_engine::quant::int8::per_channel::{quantize_per_channel, QuantizedMatrix};
use llm_engine::tensor::Tensor;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_f32(rows: usize, cols: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32) * 0.001 - (rows * cols / 2) as f32 * 0.001)
        .collect();
    Tensor::from_vec(data, vec![rows, cols]).unwrap()
}

fn make_qmatrix(n_out: usize, k_in: usize) -> QuantizedMatrix {
    let data: Vec<f32> = (0..n_out * k_in)
        .map(|i| (i as f32) * 0.002 - (n_out * k_in / 2) as f32 * 0.002)
        .collect();
    quantize_per_channel(&data, n_out, k_in)
}

// ── f32 kernel comparison ─────────────────────────────────────────────────────

fn bench_f32_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_f32");

    for &size in &[128usize, 512, 1024] {
        let a = make_f32(size, size);
        let b = make_f32(size, size);
        // FLOP count for an [M,K]×[K,N] GEMM: 2·M·K·N
        let flops = (2 * size * size * size) as u64;
        group.throughput(Throughput::Elements(flops));

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| matmul_naive(&a, &b).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("blocked", size), &size, |bench, _| {
            bench.iter(|| matmul_blocked(&a, &b).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("avx2", size), &size, |bench, _| {
            bench.iter(|| matmul_avx2(&a, &b).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| matmul_parallel(&a, &b).unwrap());
        });
    }

    group.finish();
}

// ── INT8 kernel ───────────────────────────────────────────────────────────────

fn bench_int8_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_int8");

    for &size in &[128usize, 512, 1024] {
        let act = make_f32(size, size);
        let wq  = make_qmatrix(size, size);
        let flops = (2 * size * size * size) as u64;
        group.throughput(Throughput::Elements(flops));

        group.bench_with_input(
            BenchmarkId::new("int8_from_f32", size),
            &size,
            |bench, _| {
                bench.iter(|| matmul_int8_from_f32(&act, &wq).unwrap());
            },
        );
    }

    group.finish();
}

// ── Llama-scale rectangular projection benchmark ──────────────────────────────
//
// In a Llama-7B forward pass the hottest matmuls are:
//   QKV gate/up projections: [seq, 4096] × [4096, 4096]  (or [4096, 11008])
// Here we benchmark a single representative projection size.

fn bench_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_projection");

    // seq_len=16 (decode batch), hidden=512 (scaled-down proxy for 4096)
    let seq_len  = 16_usize;
    let hidden   = 512_usize;
    let ffn_dim  = 1024_usize; // scaled proxy for 11008

    let act     = make_f32(seq_len, hidden);
    let wq_ff   = make_qmatrix(ffn_dim, hidden);
    let w_ff_f32 = make_f32(ffn_dim, hidden);

    group.throughput(Throughput::Elements(
        (2 * seq_len * hidden * ffn_dim) as u64,
    ));

    group.bench_function("f32_blocked", |bench| {
        bench.iter(|| matmul_blocked(&act, &w_ff_f32.transpose(0, 1).unwrap()).unwrap());
    });

    group.bench_function("int8_from_f32", |bench| {
        bench.iter(|| matmul_int8_from_f32(&act, &wq_ff).unwrap());
    });

    group.finish();
}

criterion_group!(benches, bench_f32_matmul, bench_int8_matmul, bench_projection);
criterion_main!(benches);
