# Architecture Overview

This document describes the high-level architecture of the LLM Inference Engine.

## Design Principles

1. **Zero-copy where possible**: Use memory-mapped I/O and borrowed views to minimize data copying
2. **Cache-aware algorithms**: Design for CPU cache hierarchy (blocking, prefetching)
3. **Separation of concerns**: Clean boundaries between parsing, computation, and memory management
4. **Progressive optimization**: Scalar baseline first, then SIMD, then quantization

## Component Overview

### GGUF Loader (`src/gguf/`)

Responsible for parsing GGUF model files and providing access to weights.

```
┌─────────────────────────────────────────┐
│              GGUF File                  │
├─────────────────────────────────────────┤
│  Header     │ magic, version, counts   │
├─────────────────────────────────────────┤
│  Metadata   │ key-value pairs          │
├─────────────────────────────────────────┤
│  TensorInfo │ names, shapes, offsets   │
├─────────────────────────────────────────┤
│  TensorData │ raw weight bytes         │
└─────────────────────────────────────────┘
```

Key design decisions:
- Memory-mapped file access for lazy loading
- On-demand dequantization (don't decompress until needed)
- Tensor caching with LRU eviction

### Tensor System (`src/tensor/`)

N-dimensional array abstraction supporting both owned and borrowed data.

```rust
// Core types
struct Shape(Vec<usize>);
struct Stride(Vec<usize>);

struct Tensor<T> {
    data: Storage<T>,  // Owned or borrowed
    shape: Shape,
    stride: Stride,
}
```

Key design decisions:
- Generic over element type `T` for future quantized types
- Stride-based views enable reshape/transpose without copying
- Explicit contiguous vs strided distinction for kernel optimization

### Operations (`src/ops/`)

Core computational kernels organized by operation type.

```
ops/
├── matmul/       # Matrix multiplication variants
│   ├── naive.rs  # Reference implementation
│   ├── blocked.rs # Cache-blocked
│   └── avx2.rs   # SIMD optimized
├── attention/    # Attention mechanisms
├── norm/         # Normalization (RMSNorm)
├── activation/   # Activation functions (SiLU)
└── rope.rs       # Rotary position embedding
```

Key design decisions:
- Trait-based dispatch allows runtime kernel selection
- Fused operations reduce memory bandwidth (matmul+bias, matmul+activation)
- Separate prefill (batched) vs decode (single token) paths

### Model (`src/model/`)

Architecture-specific forward pass implementations.

```rust
// Llama architecture
struct LlamaConfig {
    dim: usize,
    n_layers: usize,
    n_heads: usize,
    n_kv_heads: usize,  // For GQA
    vocab_size: usize,
    // ...
}

struct LlamaModel {
    config: LlamaConfig,
    weights: LlamaWeights,
}
```

Key design decisions:
- Config parsed from GGUF metadata
- Weights loaded lazily via GGUF loader
- Forward pass operates on pre-allocated buffers (no allocation during inference)

### KV-Cache (`src/cache/`)

Manages key-value cache for efficient autoregressive generation.

```
┌─────────────────────────────────────┐
│           KV Cache                  │
├─────────────────────────────────────┤
│ Layer 0  │ K: [seq, heads, dim]    │
│          │ V: [seq, heads, dim]    │
├─────────────────────────────────────┤
│ Layer 1  │ K: [seq, heads, dim]    │
│          │ V: [seq, heads, dim]    │
├─────────────────────────────────────┤
│   ...    │        ...              │
└─────────────────────────────────────┘
```

Key design decisions:
- Pre-allocated for max sequence length
- Position tracking for append-only updates
- Support for cache truncation (regeneration scenarios)

### Quantization (`src/quant/`)

Support for reduced-precision inference.

```
Quantization Hierarchy:
├── FP32 (baseline, ~4 bytes/param)
├── FP16 (~2 bytes/param)
├── INT8 (~1 byte/param)
│   ├── Symmetric (scale only)
│   └── Asymmetric (scale + zero-point)
└── INT4 (~0.5 bytes/param)
    └── Block quantization with scales
```

Key design decisions:
- Weights stored quantized, dequantize on-the-fly or compute in INT
- Per-channel quantization for weights (better accuracy)
- Dynamic quantization option for activations

### SIMD Backend (`src/simd/`)

Platform-specific vectorized kernels.

```rust
// Runtime dispatch
fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    if is_x86_feature_detected!("avx2") {
        matmul_avx2(a, b)
    } else {
        matmul_scalar(a, b)
    }
}
```

Key design decisions:
- Runtime feature detection (no compile-time target lock-in)
- Scalar fallback always available
- Separate implementations for x86 (AVX2) and ARM (NEON)

## Data Flow

### Inference Pipeline

```
Input Text
    │
    ▼
┌─────────┐
│Tokenizer│ → Token IDs
└────┬────┘
     │
     ▼
┌─────────┐
│ Embed   │ → [1, dim]
└────┬────┘
     │
     ▼ (repeat N layers)
┌─────────────────────┐
│  Transformer Block  │
│  ┌───────┐          │
│  │ Norm  │          │
│  └───┬───┘          │
│      ▼              │
│  ┌───────┐          │
│  │ Attn  │←─ KV Cache
│  └───┬───┘          │
│      ▼              │
│  ┌───────┐          │
│  │ Norm  │          │
│  └───┬───┘          │
│      ▼              │
│  ┌───────┐          │
│  │  FFN  │          │
│  └───┬───┘          │
└──────┼──────────────┘
       │
       ▼
┌─────────┐
│  Norm   │
└────┬────┘
     │
     ▼
┌─────────┐
│ LM Head │ → Logits [vocab_size]
└────┬────┘
     │
     ▼
┌─────────┐
│ Sample  │ → Next Token
└─────────┘
```

## Memory Budget

For Llama 7B with INT8 quantization:

| Component | Size |
|-----------|------|
| Weights | ~7 GB |
| KV-Cache (2K context) | ~1 GB |
| Activations | ~200 MB |
| **Total** | **~8 GB** |

## Future Extensions

- **INT4 quantization**: Further memory reduction
- **Speculative decoding**: Latency improvement via draft model
- **Continuous batching**: Throughput optimization for serving
- **GPU backend**: CUDA/Metal support