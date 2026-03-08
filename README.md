# LLM Inference Engine

A production-quality LLM inference engine written in pure Rust — no ML frameworks,
no BLAS, no Python. Designed to run Llama-family models on consumer hardware (≤16 GB RAM).

> **Status:** Milestone 5 in progress — full inference pipeline complete through Week 18.
> 869 tests passing. NEON/AVX2 SIMD kernels, INT8 quantized inference, KV-cache, and
> chunked prefill all implemented and benchmarked on real hardware.

## Benchmark Results (Apple M1 Pro, 10 cores)

### f32 Kernel Evolution

| Kernel | 128×128 | 512×512 | 1024×1024 |
|--------|--------:|--------:|----------:|
| naive (scalar i-p-j) | 13.6 | 21.1 | 23.3 |
| blocked (cache-tiled) | 23.4 | 21.0 | 22.2 |
| **parallel (NEON + rayon)** | **55.3** | **138.1** | **157.9** |

*(Units: Gelem/s — 2·M·K·N FLOPs / elapsed, Criterion mean, release profile)*

### INT8 Quantization Payoff

| Shape | f32 parallel | INT8 | INT8 speedup |
|-------|-------------:|-----:|-------------:|
| 128×128 | 57.2 | 76.7 | 1.3× |
| 512×512 | 120.8 | 98.4 | f32 wins (multi-row) |
| 1024×1024 | 137.9 | 91.5 | f32 wins (multi-row) |
| **[1, 4096] × [4096, 11008]** *(decode token)* | **4.7** | **100.6** | **21×** |

The decode-token projection shape is the critical case for interactive generation:
single output row, memory-bandwidth limited, INT8 wins by 21× by fitting 4× more
weight data in L2 cache.

### Memory Efficiency at Llama-7B Scale

| dtype | Weights per layer | Total (32 layers) | Fits in 8 GB? |
|-------|------------------:|------------------:|:-------------:|
| f32 | 772 MB | 24.12 GB | ✗ |
| **INT8** | **193 MB** | **6.03 GB** | **✓** |

INT8 is the difference between a model that runs and one that doesn't load.


## What's Implemented

### Milestone 1 — Tensor System & GGUF Parser ✅
- Generic `Tensor<T>` / `TensorView<'a, T>` with N-dimensional shape, strides, and contiguous/strided layouts
- Zero-copy slice views; reshape, transpose, permute without allocation
- Full GGUF v2/v3 parser with memory-mapped I/O (`memmap2`) for lazy weight loading
- F32, F16→F32, Q8_0, and Q4_0 dequantization; LRU tensor cache and preload hints

### Milestone 2 — Computation Core ✅
- Naive, cache-blocked, and NEON/AVX2-accelerated GEMM; matrix-vector specialization
- Operation fusion: matmul + bias + activation in one L1-resident pass
- RMSNorm, SiLU, SwiGLU, RoPE (with frequency scaling for extended context), softmax
- Causal and sliding-window masked SDPA, multi-head attention, grouped-query attention (GQA)
- Full Llama transformer block: QKV projections → RoPE → GQA → residual → SwiGLU FFN
- SentencePiece BPE tokenizer loaded directly from GGUF metadata
- Temperature / top-k / top-p sampling; greedy generation loop

### Milestone 3 — KV-Cache & Memory ✅
- Pre-allocated flat KV-cache with read/write/clear/truncate; paged variant for dynamic allocation
- Chunked prefill: processes long prompts in fixed-size windows to bound peak memory
- Prompt cache (LRU KV snapshot store) for reuse of shared prefixes
- Tensor memory pool (free-list) and arena bump allocator for zero-allocation inference passes
- Memory-mapped weight loading; RSS tracking via `/proc/self/status` (Linux) and `task_vm_info` (macOS)
- `Session` API: isolated state per request, multi-turn `extend()`, `reset()`

### Milestone 4 — SIMD & Quantized Inference ✅
- INT8 per-tensor and per-channel weight quantization with calibration tooling
- INT8 × INT8 → INT32 → f32 GEMM; direct Q8_0 inference without dequantization
- AVX2 f32 GEMM kernel (x86_64); NEON f32 GEMM kernel (`vfmaq_f32`, aarch64)
- NEON INT8 dot product (`vmull_s8` / `vpadalq_s16`, aarch64)
- AVX2 INT8 dot product (`_mm256_maddubs_epi16`, x86_64)
- SIMD RMSNorm and softmax (normalisation pass); runtime dispatch via `OnceLock`
- Rayon row-parallel GEMM: NEON/AVX2 per thread × N cores

### Milestone 5 — Polish & Benchmarks 🔄 (Week 18 of 21)
- Criterion harness: GEMM micro-benchmarks, attention benchmarks, throughput reporting (Gelem/s)
- `InferenceMetrics` library: wall-clock, RSS delta, tok/s, ms/tok
- `bench_inference`: f32 kernel progression + INT8 speedup groups
- `bench_compare_llamacpp`: Llama-7B scale memory efficiency; actual RSS delta measurement
- NEON f32 kernel (commit 18.5): fixed the vectorization failure in `matmul_blocked`,
  delivering 5–6× parallel speedup on 10 cores vs prior 1.09×


## Project Structure

```
src/
├── tensor/              # N-dimensional tensor (shape, strides, views, ops)
├── gguf/                # GGUF v2/v3 parser, metadata, dequantization, mmap, cache
├── ops/
│   ├── matmul/          # naive, blocked, neon_f32, avx2, int8, int8_neon, int8_avx2, q4_0, q8_direct, parallel
│   ├── norm/            # RMSNorm (scalar + SIMD)
│   ├── activation/      # SiLU, SwiGLU
│   ├── rope.rs          # RoPE + scaled variants (linear, NTK-aware)
│   └── softmax*.rs      # Softmax (scalar + SIMD)
├── attention/           # SDPA, causal mask, MHA, GQA, sliding window, cached
├── model/
│   └── llama/           # LlamaConfig, weights, TransformerBlock, forward pass,
│                        # forward_int8, prefill, parallel_prefill, session
├── cache/               # KvCache, CachePosition, management, paged, prompt_cache
├── memory/              # TensorPool, Arena, MemoryTracker, stats
├── quant/               # INT8 symmetric, per-channel, calibration
├── simd/                # Dispatch, f32 primitives
├── tokenizer/           # BPE tokenizer (GGUF vocab)
├── sampling/            # Temperature, top-k, top-p, greedy
├── bench/               # InferenceMetrics, Timer, measure_generate
├── session.rs           # Session: full generate/extend/reset API
├── generate.rs          # GenerateConfig, generate()
└── config.rs            # SessionConfig

benches/
├── matmul.rs            # GEMM micro-benchmarks (naive/blocked/avx2/parallel/int8)
├── attention.rs         # SDPA, GQA, RMSNorm, softmax benchmarks
├── inference.rs         # f32_progression + int8_speedup kernel progression
└── compare_llamacpp.rs  # Memory efficiency: Llama-7B f32 vs INT8 footprint + RSS

scripts/
└── benchmark_comparison.sh   # Optional CPU-vs-CPU comparison with llama.cpp

docs/
└── Week_NN.md           # Per-week engineering notes with benchmark results
```


## Quick Start

```bash
# Build
cargo build --release

# Run all 869 tests
cargo test

# Inspect a GGUF model file
cargo run --bin llm -- model.gguf
cargo run --bin llm -- model.gguf --tensors   # dump all tensor names/shapes

# Run benchmarks
cargo bench --bench matmul       # GEMM kernel comparison
cargo bench --bench inference    # f32 progression + INT8 speedup
cargo bench --bench attention    # SDPA, GQA, RMSNorm, softmax
```

## Generate Text (API)

```rust
use std::sync::Arc;
use llm_engine::{
    config::SessionConfig,
    generate::GenerateConfig,
    model::llama::LlamaModel,
    session::Session,
    tokenizer::bpe::Tokenizer,
};

// Load model and tokenizer from GGUF
let loader = llm_engine::gguf::GgufLoader::open("llama-7b-q8_0.gguf")?;
let model  = Arc::new(LlamaModel::from_loader(&loader)?);
let tok    = Arc::new(Tokenizer::from_metadata(loader.metadata())?);

// Create a session and generate
let mut session = Session::new(
    model,
    tok,
    SessionConfig::new(GenerateConfig::greedy(200), /*chunk=*/64, /*ctx=*/2048),
);
let output = session.generate("The key insight about transformers is")?;
println!("{output}");

// Multi-turn: continue without resetting the KV-cache
let follow_up = session.extend("Can you elaborate?")?;
println!("{follow_up}");
```

## Design Goals

**No ML frameworks.** Every operation — matmul, attention, softmax, RoPE — is
implemented in Rust. The only non-trivial dependencies are `memmap2` (OS-level
memory mapping), `rayon` (thread pool), and `half` (f16 conversion).

**Memory-first.** The primary target is ≤16 GB RAM. INT8 quantization (6 GB for
Llama-7B weights vs 24 GB for f32) is not an optional optimisation — it is the
path to running the model at all on consumer hardware.

**Honest benchmarks.** Every number in this README comes from Criterion runs on
an Apple M1 Pro. The raw output is in `docs/Week_18.md`. Caveats are documented:
the INT8 vs f32 comparison depends on whether the workload is single-row decode
(INT8 wins 21×) or multi-row prefill (NEON f32 parallel wins 1.4×).

## Milestones

| Milestone | Status | Weeks |
|-----------|:------:|-------|
| M0: Project setup | ✅ | — |
| M1: Tensor system & GGUF parser | ✅ | 1–4 |
| M2: Computation core (matmul → generation loop) | ✅ | 5–8 |
| M3: KV-cache & memory management | ✅ | 9–12 |
| M4: SIMD & quantized inference | ✅ | 13–16 |
| M5: Polish & benchmarks | 🔄 | 17–21 |
| E: End-to-end Llama-7B validation | 📋 | post-M5 |

## Building

```bash
# Standard build (auto-detects NEON on aarch64, AVX2 on x86_64)
cargo build --release

# Run benches with extended sample time (for 1024×1024 matmul)
cargo bench --bench matmul -- --measurement-time 15
```

**Dependencies:** Rust stable ≥ 1.75. No system libraries required. NEON is
used unconditionally on aarch64 (mandatory per the AArch64 ABI); AVX2 is
detected at runtime on x86_64.

## License

MIT
