//! Block-level INT8 prefill benchmarks — commit 18.6.
//!
//! Measures the throughput of `TransformerBlockInt8::forward_cached` (sequential)
//! vs `TransformerBlockInt8::forward_cached_parallel` (rayon + NEON/AVX2)
//! across a range of sequence lengths representing real prefill workloads.
//!
//! This is the bench that validates commit 18.6: before the fix, the INT8 model
//! path had no parallel prefill at all. After the fix, all 7 projections per
//! block use `matmul_int8_parallel` (rayon over output rows).
//!
//! ## Model geometry
//!
//! The bench block uses embed=256, heads=4, kv_heads=4, head_dim=64, ffn=512.
//! This is a ~10× scaled-down Llama-7B block (4096→256, 11008→512), small
//! enough to keep bench runtime fast while large enough that rayon dispatch
//! overhead is amortized across meaningful work.
//!
//! ## Groups
//!
//! | Group                      | What it measures                          |
//! |----------------------------|-------------------------------------------|
//! | `int8_block_prefill`       | tok/s for seq_len = 16, 64, 256           |
//! | Labels: `sequential/N`     | `forward_cached` — single-threaded INT8   |
//! | Labels: `parallel/N`       | `forward_cached_parallel` — rayon INT8    |
//!
//! Throughput reported as `Throughput::Elements(seq_len)` = tokens/second.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use llm_engine::cache::KvCache;
use llm_engine::model::llama::TransformerBlockInt8;
use llm_engine::ops::rope::RopeTable;
use llm_engine::quant::int8::per_channel::quantize_per_channel;
use llm_engine::quant::int8::per_channel::QuantizedMatrix;
use llm_engine::tensor::Tensor;

// ── bench geometry ────────────────────────────────────────────────────────────

const EMBED:    usize = 256;
const HEADS:    usize = 4;
const KV_HEADS: usize = 4;
const HEAD_DIM: usize = EMBED / HEADS;  // 64
const FFN:      usize = 512;
const CTX_LEN:  usize = 1024;fn qmatrix(rows: usize, cols: usize, seed: f32) -> QuantizedMatrix {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32 * seed * 0.01) - (rows * cols) as f32 * seed * 0.005).sin() * 0.3)
        .collect();
    quantize_per_channel(&data, rows, cols)
}

fn make_block() -> TransformerBlockInt8 {
    let qd  = HEADS    * HEAD_DIM;
    let kvd = KV_HEADS * HEAD_DIM;
    let rope = RopeTable::new(CTX_LEN, HEAD_DIM, 10_000.0);
    TransformerBlockInt8::new(
        qmatrix(qd,    EMBED, 1.0),  // wq
        qmatrix(kvd,   EMBED, 1.1),  // wk
        qmatrix(kvd,   EMBED, 1.2),  // wv
        qmatrix(EMBED, qd,   1.3),   // wo
        Tensor::ones(vec![EMBED]),   // attn_norm
        qmatrix(FFN,   EMBED, 1.4),  // wgate
        qmatrix(FFN,   EMBED, 1.5),  // wup
        qmatrix(EMBED, FFN,  1.6),   // wdown
        Tensor::ones(vec![EMBED]),   // ffn_norm
        rope,
        HEADS, KV_HEADS, 1e-5,
    )
}

fn make_input(seq_len: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..seq_len * EMBED)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    Tensor::from_vec(data, vec![seq_len, EMBED]).unwrap()
}

fn fresh_cache() -> KvCache {
    KvCache::new(1, CTX_LEN, KV_HEADS, HEAD_DIM)
}

// ── benchmark ─────────────────────────────────────────────────────────────────

fn bench_int8_block_prefill(c: &mut Criterion) {
    let block = make_block();
    let mut group = c.benchmark_group("int8_block_prefill");

    for &seq_len in &[16usize, 64, 256] {
        let x = make_input(seq_len);
        group.throughput(Throughput::Elements(seq_len as u64));

        // Sequential: all 7 projections use matmul_int8_from_f32 (single-threaded).
        group.bench_with_input(
            BenchmarkId::new("sequential", seq_len),
            &seq_len,
            |bench, _| {
                bench.iter(|| {
                    let mut cache = fresh_cache();
                    block.forward_cached(&x, 0, &mut cache, 0).unwrap()
                });
            },
        );

        // Parallel: all 7 projections use matmul_int8_parallel (rayon + NEON/AVX2).
        // This is what commit 18.6 wires in via forward_cached_parallel.
        group.bench_with_input(
            BenchmarkId::new("parallel", seq_len),
            &seq_len,
            |bench, _| {
                bench.iter(|| {
                    let mut cache = fresh_cache();
                    block.forward_cached_parallel(&x, 0, &mut cache, 0).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_int8_block_prefill);
criterion_main!(benches);
