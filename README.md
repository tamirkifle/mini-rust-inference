# Minimal LLM Inference Engine

A from-scratch implementation of an LLM inference engine in Rust, designed to
load and run Llama-family models on consumer hardware.

> Full inference pipeline implemented and benchmarked — GGUF loading, INT8 quantized
> forward pass, KV-cache, NEON/AVX2 SIMD kernels, and text generation all working.
> 873 tests passing. CLI and end-to-end model validation still to come.

## What it does

Loads Llama-family GGUF models and runs autoregressive text generation entirely in
pure Rust — no PyTorch, no ONNX, no BLAS. The target is Llama 7B on a consumer
machine with ≤16 GB RAM.

The project exists to understand transformer architectures and systems-level
performance from first principles: every kernel (matmul, attention, softmax, RoPE)
is written by hand, profiled with Criterion, and iterated on until the numbers make
sense.

## Benchmark Results (Apple M1 Pro, 10 cores)

### f32 Kernel Evolution

| Kernel | 128×128 | 512×512 | 1024×1024 |
|--------|--------:|--------:|----------:|
| naive (scalar i-p-j) | 13.6 | 21.1 | 23.3 |
| blocked (cache-tiled, NEON) | 23.4 | 21.0 | 22.2 |
| **parallel (NEON + rayon)** | **55.3** | **138.1** | **157.9** |

*(Gelem/s — 2·M·K·N FLOPs / elapsed, Criterion mean)*

### INT8 Quantization

| Shape | f32 parallel | INT8 parallel | INT8 wins |
|-------|-------------:|--------------:|:---------:|
| 512×512 | 120.3 | 234.4 | 1.95× |
| 1024×1024 | 138.4 | 194.1 | 1.40× |
| **decode token** `[1, 4096] × [4096, 11008]` | **4.7** | **100.6** | **21×** |

The decode projection is the bottleneck for interactive generation — one token row
against a full weight matrix, memory-bandwidth limited. INT8 wins by 21× because
the weight matrix fits in L2 (512 KB INT8 vs 2 MB f32).

### Memory at Llama-7B Scale

| dtype | Per layer | Total (32 layers) | Fits in 8 GB? |
|-------|----------:|------------------:|:-------------:|
| f32 | 772 MB | 24.12 GB | ✗ |
| **INT8** | **193 MB** | **6.03 GB** | **✓** |


## Features

### Tensor System
- Generic `Tensor<T>` / `TensorView<'a, T>` with N-dimensional shape and stride abstractions
- Zero-copy views for slicing, reshaping, and transposing without allocation
- Full GGUF v2/v3 parser with memory-mapped I/O for lazy weight loading
- F32, F16, Q8_0, Q4_0 dequantization; LRU tensor cache and preload hints

### Matrix Multiplication
- Naive scalar baseline; cache-tiled blocked kernel; fused matmul+bias+activation
- NEON f32 GEMM (`vfmaq_f32`, aarch64) — fixed a vectorization failure in the tiled kernel
  that caused 5× regression vs naive; parallel now reaches 132–158 Gelem/s on 10 cores
- AVX2 f32 GEMM (`_mm256_fmadd_ps`, x86_64)
- INT8 × INT8 → INT32 → f32 with NEON (`vmull_s8/vpadalq_s16`) and AVX2 backends
- Rayon row-parallel GEMM combining SIMD throughput per thread with multi-core scaling

### Neural Network Operations
- RMSNorm (scalar + SIMD), SiLU, SwiGLU, RoPE with frequency scaling for extended context
- Numerically stable softmax (scalar + SIMD normalisation pass)
- Scaled dot-product attention, causal + sliding-window masking
- Multi-head attention, grouped-query attention (GQA), cached prefill + decode paths

### Model
- Full Llama transformer block: QKV → RoPE → GQA → residual → SwiGLU FFN
- `LlamaModel` (f32) and `LlamaModelInt8` (per-channel INT8 weights, f32 activations)
- `forward_cached_parallel` for both: rayon-parallel prefill with auto-fallback to
  sequential below 32 tokens (threshold eliminates rayon overhead on small chunks)
- SentencePiece BPE tokenizer loaded directly from GGUF metadata
- Temperature / top-k / top-p sampling; greedy preset

### Memory Management
- Pre-allocated flat KV-cache; paged variant for dynamic allocation
- Chunked prefill and prompt caching (LRU KV snapshot reuse)
- Tensor memory pool (free-list) and arena bump allocator
- `Session` API: isolated per-request state, multi-turn `extend()`, `reset()`
- RSS tracking via `/proc/self/status` (Linux) and `task_vm_info` (macOS)

### Quantization
- INT8 symmetric and per-channel weight quantization with calibration tooling
- Direct Q8_0 inference from GGUF without materialising f32 weights
- Runtime SIMD dispatch via `OnceLock` (one atomic load after first call)


## Usage

```rust
use std::sync::Arc;
use llm_engine::{config::SessionConfig, generate::GenerateConfig,
                 model::llama::LlamaModel, session::Session,
                 tokenizer::bpe::Tokenizer};

let loader = llm_engine::gguf::GgufLoader::open("llama-7b-q8_0.gguf")?;
let model  = Arc::new(LlamaModel::from_loader(&loader)?);
let tok    = Arc::new(Tokenizer::from_metadata(loader.metadata())?);

let mut session = Session::new(
    model, tok,
    SessionConfig::new(GenerateConfig::greedy(200), /*chunk=*/64, /*ctx=*/2048),
);

let output = session.generate("The key insight about transformers is")?;
println!("{output}");

// Multi-turn without resetting the KV-cache
let follow_up = session.extend("Can you elaborate?")?;
println!("{follow_up}");
```

## Building

```bash
cargo build --release
cargo test              # 873 tests

# Inspect a GGUF file
cargo run --bin llm -- model.gguf
cargo run --bin llm -- model.gguf --tensors

# Benchmarks
cargo bench --bench matmul        # GEMM kernels (f32 + INT8 sequential/parallel)
cargo bench --bench inference     # f32 progression + INT8 speedup
cargo bench --bench int8_prefill  # Block-level INT8 prefill sequential vs parallel
cargo bench --bench attention     # SDPA, GQA, RMSNorm, softmax
```


## Project Structure

```
src/
├── tensor/        # N-dimensional tensor, shape, strides, views, ops
├── gguf/          # GGUF v2/v3 parser, metadata, dequantization, mmap, cache
├── ops/matmul/    # naive, blocked, neon_f32, avx2, int8, int8_neon, int8_avx2, parallel
├── ops/           # RMSNorm, SiLU, SwiGLU, RoPE, softmax (scalar + SIMD)
├── attention/     # SDPA, causal mask, MHA, GQA, sliding window, cached
├── model/llama/   # Config, weights, TransformerBlock, forward_int8, prefill, session
├── cache/         # KvCache, paged cache, prompt cache
├── memory/        # TensorPool, Arena, MemoryTracker
├── quant/         # INT8 symmetric, per-channel, calibration
├── simd/          # Runtime dispatch, f32 primitives
├── tokenizer/     # BPE (loaded from GGUF)
├── sampling/      # Temperature, top-k, top-p
└── session.rs     # Session API

benches/
├── matmul.rs          # GEMM micro-benchmarks
├── attention.rs       # Attention + norm benchmarks
├── inference.rs       # f32 progression + INT8 speedup
└── int8_prefill.rs    # Block-level INT8 prefill throughput
```

## Roadmap

- [x] Core tensor system with N-dimensional shape and stride abstractions
- [x] GGUF v2/v3 parser with memory-mapped I/O
- [x] F32 / F16 / Q8_0 / Q4_0 dequantization
- [x] Full Llama forward pass (attention, RoPE, SwiGLU FFN)
- [x] BPE tokenizer, sampling, generation loop
- [x] KV-cache, chunked prefill, prompt caching
- [x] INT8 quantized inference (per-channel weights, f32 activations)
- [x] NEON and AVX2 SIMD kernels; rayon row-parallel GEMM
- [x] Criterion benchmarks with honest numbers
- [ ] Polished CLI, TOML config
- [ ] End-to-end Llama 7B validation on real GGUF

## License

MIT
