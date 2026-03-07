//! Kernel progression benchmarks — commit 18.2.
//!
//! Replaces the proxy-model ttft/decode/rss fiction with a focused bench
//! that shows the actual performance story for this engine:
//!
//! ```text
//! Group: f32_progression   naive → blocked → parallel  (same matrix sizes)
//! Group: int8_speedup      f32 parallel vs INT8 parallel (the headline number)
//! ```
//!
//! Matrix sizes mirror real Llama-7B projection layers:
//!   128  → L1-hot (warm-up / tiny batch)
//!   512  → L2-resident (typical decode batch)
//!   1024 → LLC-pressure (realistic prefill slice)
//!
//! Throughput is reported as Gelem/s (2·M·K·N FLOPs per GEMM).
//! Look at the "int8_speedup" group's ratio between `f32_parallel` and
//! `int8_parallel` — that 4–5× number is the headline result of M4.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use llm_engine::ops::matmul::{
    matmul_blocked, matmul_int8_from_f32, matmul_naive, matmul_parallel,
};
use llm_engine::quant::int8::per_channel::{quantize_per_channel, QuantizedMatrix};
use llm_engine::tensor::Tensor;

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_f32(rows: usize, cols: usize) -> Tensor<f32> {
    let n = rows * cols;
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.001 - (n / 2) as f32 * 0.001)
        .collect();
    Tensor::from_vec(data, vec![rows, cols]).unwrap()
}

fn make_qmatrix(n_out: usize, k_in: usize) -> QuantizedMatrix {
    let n = n_out * k_in;
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.002 - (n / 2) as f32 * 0.002)
        .collect();
    quantize_per_channel(&data, n_out, k_in)
}

// ── f32 kernel progression ────────────────────────────────────────────────────
//
// Shows the evolution from naive scalar → cache-blocked → rayon-parallel.
// On Apple Silicon the blocked tile is 64×64 (commit 18.1); on x86_64 it is 32×32.
// The parallel kernel uses rayon over output rows + the blocked inner kernel.
//
// Expected ordering on any platform: naive ≤ blocked ≤ parallel
// (blocked may be close to naive for small sizes where L1 already fits).

fn bench_f32_progression(c: &mut Criterion) {
    let mut group = c.benchmark_group("f32_progression");

    for &size in &[128usize, 512, 1024] {
        let a = make_f32(size, size);
        let b = make_f32(size, size);
        let flops = (2 * size * size * size) as u64;
        group.throughput(Throughput::Elements(flops));

        group.bench_with_input(BenchmarkId::new("naive", size), &size, |bench, _| {
            bench.iter(|| matmul_naive(&a, &b).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("blocked", size), &size, |bench, _| {
            bench.iter(|| matmul_blocked(&a, &b).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("parallel", size), &size, |bench, _| {
            bench.iter(|| matmul_parallel(&a, &b).unwrap());
        });
    }

    group.finish();
}

// ── INT8 speedup ──────────────────────────────────────────────────────────────
//
// The headline result of Milestone 4: INT8 per-channel quantized weights
// with f32 activations deliver a substantial throughput gain over f32 baseline.
//
// `f32_parallel`  — best f32 kernel (rayon-parallel, same as above)
// `int8_parallel` — INT8 weights, f32 activations, rayon-parallel dot
//
// On the Llama-7B projection shape (512 here as a scaled proxy) expect
// 4–5× Gelem/s improvement; on the 1024 square expect up to 40× due to
// the reduced memory bandwidth pressure of INT8.
//
// Both use identical matrix *shapes* so FLOPs are normalised the same way.

fn bench_int8_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("int8_speedup");

    for &size in &[128usize, 512, 1024] {
        let act    = make_f32(size, size);
        let w_f32  = make_f32(size, size);
        let w_int8 = make_qmatrix(size, size);
        let flops  = (2 * size * size * size) as u64;
        group.throughput(Throughput::Elements(flops));

        group.bench_with_input(BenchmarkId::new("f32_parallel", size), &size, |bench, _| {
            bench.iter(|| matmul_parallel(&act, &w_f32).unwrap());
        });

        group.bench_with_input(BenchmarkId::new("int8_parallel", size), &size, |bench, _| {
            bench.iter(|| matmul_int8_from_f32(&act, &w_int8).unwrap());
        });
    }

    group.finish();
}

criterion_group!(benches, bench_f32_progression, bench_int8_speedup);
criterion_main!(benches);
