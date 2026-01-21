# LLM Inference Engine

A production-quality LLM inference engine built from scratch in Rust. Designed to run Llama 7B at competitive speeds on consumer hardware.

## Goals

- **Educational**: Deep understanding of transformer architectures and systems programming
- **Performant**: SIMD vectorization, INT8/INT4 quantization, cache-optimized algorithms
- **Portable**: Runs on x86 (AVX2) and ARM (NEON) without external dependencies

## Features

| Feature | Status |
|---------|--------|
| GGUF model loading | 🚧 Planned |
| Tensor operations | 🚧 Planned |
| Multi-head attention | 🚧 Planned |
| KV-cache | 🚧 Planned |
| INT8 quantization | 🚧 Planned |
| SIMD kernels (AVX2/NEON) | 🚧 Planned |
| Speculative decoding | 📋 Extension |

## Quick Start

```bash
# Build
cargo build --release

# Run inference
./target/release/llm generate \
    --model models/tinyllama-1.1b.gguf \
    --prompt "Once upon a time"

# Inspect model
./target/release/llm inspect --model models/llama-7b.gguf
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      CLI / API                          │
├─────────────────────────────────────────────────────────┤
│                   Generation Loop                       │
│              (sampling, token decode)                   │
├─────────────────────────────────────────────────────────┤
│                  Model (Llama)                          │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│    │ Embed   │→ │ Blocks  │→ │ LM Head │              │
│    └─────────┘  └────┬────┘  └─────────┘              │
│                      │                                  │
│         ┌────────────┼────────────┐                    │
│         ▼            ▼            ▼                    │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐              │
│    │  Attn   │  │   FFN   │  │  Norm   │              │
│    └─────────┘  └─────────┘  └─────────┘              │
├─────────────────────────────────────────────────────────┤
│                  Operations Layer                       │
│   matmul │ softmax │ RoPE │ RMSNorm │ SiLU            │
├─────────────────────────────────────────────────────────┤
│                  Compute Backends                       │
│        ┌──────────┬──────────┬──────────┐             │
│        │  Scalar  │   AVX2   │   NEON   │             │
│        └──────────┴──────────┴──────────┘             │
├─────────────────────────────────────────────────────────┤
│                  Memory Management                      │
│     KV-Cache │ Tensor Pool │ Memory-mapped I/O        │
├─────────────────────────────────────────────────────────┤
│                  GGUF Loader                           │
│        header │ metadata │ tensors │ quantization     │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/
├── lib.rs              # Library entry point
├── bin/
│   └── main.rs         # CLI binary
├── tensor/             # N-dimensional array abstraction
├── gguf/               # GGUF format parser
├── ops/                # Core operations (matmul, attention, etc.)
├── model/              # Model architectures
├── cache/              # KV-cache management
├── quant/              # Quantization (INT8/INT4)
└── simd/               # SIMD kernels
```

## Performance Targets

| Model | Quantization | Target Speed | Memory |
|-------|--------------|--------------|--------|
| TinyLlama 1.1B | FP32 | 20+ tok/s | ~5 GB |
| Llama 7B | INT8 | 10+ tok/s | ~8 GB |
| Llama 7B | INT4 | 15+ tok/s | ~4 GB |

*Targets for modern consumer CPU (e.g., Ryzen 7, Apple M1)*

## Building

### Requirements

- Rust 1.75+ (stable)
- ~16 GB RAM for development with 7B models

### Build Commands

```bash
# Debug build
cargo build

# Release build (optimized)
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench

# Generate documentation
cargo doc --open
```

## Resources

### Reference Implementations
- [llama2.c](https://github.com/karpathy/llama2.c) - Minimal C implementation
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - Production C++ implementation
- [GGUF Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [LLaMA](https://arxiv.org/abs/2302.13971) - LLaMA architecture details
- [RoFormer](https://arxiv.org/abs/2104.09864) - Rotary Position Embedding

## License

MIT License - see [LICENSE](LICENSE) for details.