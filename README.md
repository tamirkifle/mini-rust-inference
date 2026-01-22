# Minimal LLM Inference Engine

A from-scratch implementation of an LLM inference engine in Rust, designed to load and run Llama-family models on consumer hardware.

> **Status:** Work in progress — core tensor system and GGUF parser complete, inference pipeline in development.

## Features

### Tensor System
- Generic N-dimensional tensor implementation with shape and stride abstractions
- Memory-efficient views for slicing and reshaping without copying
- Standard operations: element-wise math, matrix multiplication, broadcasting

### GGUF Parser
- Full GGUF v2/v3 format support with memory-mapped I/O for efficient large file handling
- Metadata extraction (model architecture, tokenizer config, hyperparameters)
- Tensor information parsing with automatic alignment handling

### Quantization Support
- **F32/F16**: Direct extraction and half-precision conversion
- **Q8_0**: 8-bit block quantization (32 elements/block, ~2x compression)
- **Q4_0**: 4-bit block quantization (32 elements/block, ~4x compression)

### Performance
- LRU tensor cache with configurable memory limits
- Preloading strategies (eager, lazy, selective pattern matching)
- Zero-copy memory mapping for model files

## Usage

```rust
use llm_engine::gguf::{GgufLoader, TensorExtractor};

// Load GGUF model file
let loader = GgufLoader::open("model.gguf")?;

// Inspect model metadata
println!("Architecture: {}", loader.metadata().get_str("general.architecture").unwrap());
println!("Tensors: {}", loader.tensors().len());

// Extract and dequantize tensors
let extractor = TensorExtractor::new(&loader);
let weights = extractor.extract("model.layers.0.attn.wq.weight")?;
println!("Shape: {:?}", weights.dims());
```

## Project Structure

```
src/
├── tensor/          # N-dimensional tensor implementation
│   ├── shape.rs     # Shape and stride abstractions
│   ├── view.rs      # Non-owning tensor views
│   └── ops.rs       # Tensor operations
└── gguf/            # GGUF file format support
    ├── loader.rs    # File loading and memory mapping
    ├── metadata.rs  # Key-value metadata parsing
    ├── extract.rs   # Tensor extraction API
    ├── dequant.rs   # Quantization/dequantization
    └── cache.rs     # Tensor caching layer
```

## Roadmap

- [x] Core tensor system
- [x] GGUF parser with memory-mapped I/O
- [x] F32/F16/Q8_0/Q4_0 dequantization
- [x] Tensor caching and preloading
- [ ] Tokenizer (BPE)
- [ ] Embedding and position encoding
- [ ] Attention mechanism
- [ ] Transformer blocks
- [ ] Text generation pipeline

## Building

```bash
cargo build --release
cargo test
```

## License

MIT