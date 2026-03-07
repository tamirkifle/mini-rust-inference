//! Memory efficiency benchmarks — commit 18.3.
//!
//! Replaces the misleading proxy-vs-Metal comparison with a concrete answer
//! to the key deployment question: **does INT8 fit Llama 7B in 8 GB RAM?**
//!
//! ## Design
//!
//! We allocate representative weight slices at Llama-7B projection shape
//! (4096 × 4096 = 16.7 M parameters per matrix) in both f32 and INT8, then
//! measure the actual RSS delta reported by the OS.  No GGUF file required.
//!
//! Two benchmark groups:
//!
//! | Group                 | What it measures                                  |
//! |-----------------------|---------------------------------------------------|
//! | `weight_footprint`    | Alloc + matmul + drop one full 7-projection layer |
//! | `memory_efficiency`   | RSS delta while holding N layers live in memory   |
//!
//! A static summary is printed before the first group showing the 7B-scale
//! extrapolation for both dtypes.
//!
//! ## llama.cpp comparison note
//!
//! The companion shell script `scripts/benchmark_comparison.sh` can still run
//! a CPU-vs-CPU latency comparison when `LLAMA_CPP_BIN` and `LLAMA_MODEL` are
//! set — see the script for usage.  That path is **not** benchmarked here
//! because our engine runs on the CPU scalar/SIMD path while llama.cpp on
//! macOS defaults to Metal GPU; comparing them as though they are the same
//! hardware backend is misleading.

use criterion::{criterion_group, criterion_main, Criterion, Throughput};

use llm_engine::memory::stats::{format_bytes, query_rss};
use llm_engine::ops::matmul::matmul_int8_from_f32;
use llm_engine::ops::matmul::matmul_parallel;
use llm_engine::quant::int8::per_channel::{quantize_per_channel, QuantizedMatrix};
use llm_engine::tensor::Tensor;

// ── Llama-7B geometry constants ───────────────────────────────────────────────

/// Hidden dimension of Llama-7B.
const D_MODEL: usize = 4096;
/// FFN intermediate dimension of Llama-7B.
const D_FFN: usize = 11008;
/// Number of transformer layers in Llama-7B.
const N_LAYERS: usize = 32;

// Projection sizes (rows × cols): Q, K, V, O, gate, up, down
const PROJ_PARAMS: &[(usize, usize)] = &[
    (D_MODEL, D_MODEL),  // Q
    (D_MODEL, D_MODEL),  // K
    (D_MODEL, D_MODEL),  // V
    (D_MODEL, D_MODEL),  // O
    (D_FFN,   D_MODEL),  // gate
    (D_FFN,   D_MODEL),  // up
    (D_MODEL, D_FFN),    // down
];

/// Total f32 bytes for all projections in one transformer layer.
fn layer_f32_bytes() -> usize {
    PROJ_PARAMS.iter().map(|(r, c)| r * c * 4).sum()
}

/// Total INT8 bytes for the same projections (1 byte/weight + scale overhead).
fn layer_int8_bytes() -> usize {
    PROJ_PARAMS.iter().map(|(r, c)| r * c).sum()
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_f32_proj(rows: usize, cols: usize) -> Tensor<f32> {
    let n = rows * cols;
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.001 - (n / 2) as f32 * 0.001)
        .collect();
    Tensor::from_vec(data, vec![rows, cols]).unwrap()
}

fn make_int8_proj(rows: usize, cols: usize) -> QuantizedMatrix {
    let n = rows * cols;
    let data: Vec<f32> = (0..n)
        .map(|i| (i as f32) * 0.001 - (n / 2) as f32 * 0.001)
        .collect();
    quantize_per_channel(&data, rows, cols)
}

/// Print the 7B-scale memory extrapolation once at bench startup.
fn print_7b_summary() {
    let f32_per_layer = layer_f32_bytes();
    let i8_per_layer  = layer_int8_bytes();
    let f32_total     = f32_per_layer * N_LAYERS;
    let i8_total      = i8_per_layer  * N_LAYERS;
    let ratio         = f32_total as f64 / i8_total as f64;

    println!("\n╔══════════════════════════════════════════════════════╗");
    println!("║          Memory Footprint — Llama-7B at Scale        ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║  dtype   per-layer     total ({N_LAYERS} layers)            ║");
    println!("║  f32     {:<12}  {:<28} ║", format_bytes(f32_per_layer), format_bytes(f32_total));
    println!("║  INT8    {:<12}  {:<28} ║", format_bytes(i8_per_layer),  format_bytes(i8_total));
    println!("║  ratio   {ratio:.1}×                                       ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!("  f32 fits in 16 GB: {}", if f32_total < 16 * 1024 * 1024 * 1024 { "yes" } else { "no" });
    println!("  INT8 fits in  8 GB: {}", if i8_total  <  8 * 1024 * 1024 * 1024 { "yes" } else { "no" });
    println!();
}

// ── Group 1: weight_footprint ─────────────────────────────────────────────────
//
// Each iter allocates all 7 projections for one transformer layer, runs a
// single representative matmul (Q projection), then drops the weights.
// Throughput::Bytes shows allocation bandwidth by dtype.

fn bench_weight_footprint(c: &mut Criterion) {
    print_7b_summary();

    let mut group = c.benchmark_group("weight_footprint");
    let act = make_f32_proj(1, D_MODEL);

    group.throughput(Throughput::Bytes(layer_f32_bytes() as u64));
    group.bench_function("f32_layer", |bench| {
        bench.iter(|| {
            let projs: Vec<Tensor<f32>> = PROJ_PARAMS
                .iter()
                .map(|&(r, c)| make_f32_proj(r, c))
                .collect();
            let out = matmul_parallel(&act, &projs[0].transpose(0, 1).unwrap()).unwrap();
            std::hint::black_box(out);
            drop(projs);
        });
    });

    group.throughput(Throughput::Bytes(layer_int8_bytes() as u64));
    group.bench_function("int8_layer", |bench| {
        bench.iter(|| {
            let projs: Vec<QuantizedMatrix> = PROJ_PARAMS
                .iter()
                .map(|&(r, c)| make_int8_proj(r, c))
                .collect();
            let out = matmul_int8_from_f32(&act, &projs[0]).unwrap();
            std::hint::black_box(out);
            drop(projs);
        });
    });

    group.finish();
}

// ── Group 2: memory_efficiency ────────────────────────────────────────────────
//
// Allocates N_BENCH_LAYERS layers of weights, queries RSS before/after, and
// prints the delta.  Uses 4 layers (not 32) to keep bench runtime reasonable;
// the ratio extrapolates linearly to full 7B scale.

const N_BENCH_LAYERS: usize = 4;

fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    let act = make_f32_proj(1, D_MODEL);

    // f32 slice
    {
        let rss_before  = query_rss();
        let f32_layers: Vec<Vec<Tensor<f32>>> = (0..N_BENCH_LAYERS)
            .map(|_| PROJ_PARAMS.iter().map(|&(r, c)| make_f32_proj(r, c)).collect())
            .collect();
        let rss_after = query_rss();
        println!(
            "[memory_efficiency] f32 {N_BENCH_LAYERS} layers: allocated {} | RSS delta {}",
            format_bytes(layer_f32_bytes() * N_BENCH_LAYERS),
            format_bytes(rss_after.saturating_sub(rss_before)),
        );

        group.throughput(Throughput::Bytes((layer_f32_bytes() * N_BENCH_LAYERS) as u64));
        group.bench_function(format!("f32_{N_BENCH_LAYERS}layers"), |bench| {
            bench.iter(|| {
                let out = matmul_parallel(&act, &f32_layers[0][0].transpose(0, 1).unwrap())
                    .unwrap();
                std::hint::black_box(out);
            });
        });
        std::hint::black_box(&f32_layers);
    }

    // INT8 slice
    {
        let rss_before  = query_rss();
        let i8_layers: Vec<Vec<QuantizedMatrix>> = (0..N_BENCH_LAYERS)
            .map(|_| PROJ_PARAMS.iter().map(|&(r, c)| make_int8_proj(r, c)).collect())
            .collect();
        let rss_after = query_rss();
        println!(
            "[memory_efficiency] INT8 {N_BENCH_LAYERS} layers: allocated {} | RSS delta {}",
            format_bytes(layer_int8_bytes() * N_BENCH_LAYERS),
            format_bytes(rss_after.saturating_sub(rss_before)),
        );

        group.throughput(Throughput::Bytes((layer_int8_bytes() * N_BENCH_LAYERS) as u64));
        group.bench_function(format!("int8_{N_BENCH_LAYERS}layers"), |bench| {
            bench.iter(|| {
                let out = matmul_int8_from_f32(&act, &i8_layers[0][0]).unwrap();
                std::hint::black_box(out);
            });
        });
        std::hint::black_box(&i8_layers);
    }

    group.finish();
}

criterion_group!(benches, bench_weight_footprint, bench_memory_efficiency);
criterion_main!(benches);
