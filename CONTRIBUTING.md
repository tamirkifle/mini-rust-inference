# Contributing to LLM Inference Engine

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/tamiryirga/llm-inference-engine.git
   cd llm-inference-engine
   ```

2. **Install Rust** (if needed)
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```

3. **Build and test**
   ```bash
   cargo build
   cargo test
   ```

## Code Standards

### Formatting

All code must pass `rustfmt`:

```bash
cargo fmt --all -- --check
```

### Linting

All code must pass `clippy` with pedantic lints:

```bash
cargo clippy --all-targets -- -D warnings -W clippy::all
```

### Testing

All tests must pass:

```bash
cargo test
```

## Commit Guidelines

### Message Format

```
[Action] component: specific change

Optional longer description explaining why this change
was made and any important context.
```

### Actions

- `[Init]` - Project initialization
- `[Add]` - New feature or file
- `[Fix]` - Bug fix
- `[Refactor]` - Code restructuring without behavior change
- `[Perf]` - Performance optimization
- `[Docs]` - Documentation only
- `[Test]` - Test additions or fixes
- `[Config]` - Configuration changes

### Examples

```
[Add] tensor: implement shape and stride computation
[Fix] gguf: handle big-endian metadata correctly
[Perf] matmul: add AVX2 SIMD kernel
[Docs] attention: explain multi-head splitting
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make focused, atomic commits
3. Ensure CI passes (fmt, clippy, test)
4. Write clear PR description explaining changes
5. Request review

## Architecture Decisions

Major architectural decisions are documented in `docs/`. When proposing significant changes:

1. Open an issue describing the proposed change
2. Discuss tradeoffs and alternatives
3. Document the decision in `docs/` if accepted

## Performance Work

When working on performance optimizations:

1. Establish baseline benchmark first
2. Profile before optimizing
3. Document methodology and results
4. Ensure correctness tests still pass

## Questions?

Open an issue for any questions about contributing.