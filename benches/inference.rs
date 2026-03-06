//! End-to-end inference benchmarks — commit 17.2.
//!
//! Measures wall-clock latency for generation using an in-memory zero-weight
//! model (no GGUF file required), allowing the benchmark harness to focus on
//! throughput of the kernel pipeline rather than I/O.
//!
//! Three benchmark groups:
//!
//! | Group                  | What it measures                                   |
//! |------------------------|----------------------------------------------------|
//! | `ttft_proxy`           | Time-to-first-token: prefill cost at prompt length |
//! | `decode_throughput`    | Tokens/sec during autoregressive decode phase      |
//! | `rss_delta`            | RSS growth per complete generate call              |
//!
//! The tiny model has 1 block, 8-dim embeddings, 2 heads — zero-weight so
//! logits collapse to 0, sampling always picks the first non-special token.
//! Throughput numbers here reflect pure kernel overhead, not model quality.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use llm_engine::bench::metrics::{measure_generate, InferenceMetrics};
use llm_engine::config::SessionConfig;
use llm_engine::generate::GenerateConfig;
use llm_engine::gguf::{Metadata, MetadataValue};
use llm_engine::model::llama::{LlamaConfig, LlamaModel, TransformerBlock};
use llm_engine::ops::rope::RopeTable;
use llm_engine::session::Session;
use llm_engine::tensor::Tensor;
use llm_engine::tokenizer::bpe::Tokenizer;

// ── fixture helpers ───────────────────────────────────────────────────────────

fn tiny_config(embed: u32, heads: u32, ffn: u32, vocab: u32) -> LlamaConfig {
    let mut m = Metadata::new();
    for (k, v) in [
        ("llama.block_count",          MetadataValue::Uint32(2)),
        ("llama.embedding_length",     MetadataValue::Uint32(embed)),
        ("llama.attention.head_count", MetadataValue::Uint32(heads)),
        ("llama.feed_forward_length",  MetadataValue::Uint32(ffn)),
        ("llama.context_length",       MetadataValue::Uint32(512)),
        ("llama.vocab_size",           MetadataValue::Uint32(vocab)),
    ] {
        m.insert(k.to_string(), v);
    }
    LlamaConfig::from_metadata(&m).unwrap()
}

fn make_model(cfg: &LlamaConfig) -> Arc<LlamaModel> {
    let embed = cfg.embedding_length   as usize;
    let vocab = cfg.vocab_size         as usize;
    let ffn   = cfg.feed_forward_length as usize;
    let heads = cfg.n_heads            as usize;
    let kv    = cfg.n_kv_heads         as usize;
    let hd    = cfg.head_dim()         as usize;
    let rope  = RopeTable::new(cfg.context_length as usize, hd, cfg.rope_freq_base);
    let block = || TransformerBlock::new(
        Tensor::zeros(vec![heads * hd, embed]),
        Tensor::zeros(vec![kv    * hd, embed]),
        Tensor::zeros(vec![kv    * hd, embed]),
        Tensor::zeros(vec![embed, heads * hd]),
        Tensor::ones(vec![embed]),
        Tensor::zeros(vec![ffn, embed]),
        Tensor::zeros(vec![ffn, embed]),
        Tensor::zeros(vec![embed, ffn]),
        Tensor::ones(vec![embed]),
        rope.clone(), heads, kv, cfg.rms_norm_eps,
    );
    Arc::new(LlamaModel::new(
        cfg.clone(),
        Tensor::zeros(vec![vocab, embed]),
        vec![block(), block()],
        Tensor::ones(vec![embed]),
        Tensor::zeros(vec![vocab, embed]),
    ))
}

fn make_tokenizer(vocab: u32) -> Arc<Tokenizer> {
    use llm_engine::gguf::keys;

    // Build a simple vocab: <unk>, <s>, </s>, then "a"–…
    let mut vocab_strs: Vec<String> = vec![
        "<unk>".into(), "<s>".into(), "</s>".into(),
    ];
    for i in 3..vocab {
        vocab_strs.push(format!("tok{i}"));
    }
    let scores: Vec<f32> = vocab_strs.iter().enumerate().map(|(i, _)| {
        if i < 3 { f32::NEG_INFINITY } else { -(i as f32) }
    }).collect();
    let types: Vec<i32> = vocab_strs.iter().enumerate().map(|(i, _)| {
        if i == 0 { 2 } else if i < 3 { 3 } else { 1 }
    }).collect();

    let mut m = Metadata::new();
    m.insert(keys::TOKENIZER_GGML_TOKENS.to_string(),
        MetadataValue::StringArray(vocab_strs));
    m.insert(keys::TOKENIZER_GGML_SCORES.to_string(),
        MetadataValue::Float32Array(scores));
    m.insert(keys::TOKENIZER_GGML_TOKEN_TYPE.to_string(),
        MetadataValue::Int32Array(types));
    m.insert(keys::TOKENIZER_GGML_BOS_TOKEN_ID.to_string(),
        MetadataValue::Uint32(1));
    m.insert(keys::TOKENIZER_GGML_EOS_TOKEN_ID.to_string(),
        MetadataValue::Uint32(2));
    Arc::new(Tokenizer::from_metadata(&m).unwrap())
}

/// Build a Session with the given max_new_tokens limit.
fn make_session(max_new_tokens: usize) -> Session {
    let cfg   = tiny_config(32, 4, 64, 64);
    let model = make_model(&cfg);
    let tok   = make_tokenizer(64);
    let gcfg  = GenerateConfig::greedy(max_new_tokens);
    Session::new(model, tok, SessionConfig::new(gcfg, 16, 512))
}

// ── TTFT proxy benchmark ──────────────────────────────────────────────────────
//
// Generate exactly 1 new token so total latency ≈ prefill latency.
// Vary prompt length to see how prefill scales with sequence length.

fn bench_ttft_proxy(c: &mut Criterion) {
    let mut group = c.benchmark_group("ttft_proxy");

    for &prompt_tokens in &[4usize, 16, 64] {
        // Build a prompt string that encodes to approximately `prompt_tokens` tokens.
        // Each "tokN " encodes as one token in our tiny tokenizer.
        let prompt: String = (3..(3 + prompt_tokens))
            .map(|i| format!("tok{i} "))
            .collect();

        group.throughput(Throughput::Elements(prompt_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("prompt_len", prompt_tokens),
            &prompt,
            |bench, p| {
                bench.iter(|| {
                    let mut session = make_session(1);
                    session.generate(p).unwrap();
                });
            },
        );
    }

    group.finish();
}

// ── Decode throughput benchmark ───────────────────────────────────────────────
//
// Fixed short prompt, vary generation budget to measure decode throughput.

fn bench_decode_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("decode_throughput");

    let prompt = "tok3 tok4 tok5 tok6";

    for &n_tokens in &[4usize, 16, 32] {
        group.throughput(Throughput::Elements(n_tokens as u64));

        group.bench_with_input(
            BenchmarkId::new("n_tokens", n_tokens),
            &n_tokens,
            |bench, &n| {
                bench.iter(|| {
                    let mut session = make_session(n);
                    session.generate(prompt).unwrap();
                });
            },
        );
    }

    group.finish();
}

// ── RSS delta benchmark ───────────────────────────────────────────────────────
//
// Wraps generate() with measure_generate() to report RSS growth per call.
// Not a Criterion throughput benchmark — just one sample, printed as a
// custom metric for CI inspection.

fn bench_rss_delta(c: &mut Criterion) {
    let mut group = c.benchmark_group("rss_delta");

    group.bench_function("generate_10_tokens", |bench| {
        bench.iter(|| {
            let mut session = make_session(10);
            let prompt = "tok3 tok4 tok5";
            // measure_generate captures RSS before/after and returns InferenceMetrics
            let m: InferenceMetrics = measure_generate(3, 10, || {
                let _ = session.generate(prompt).unwrap();
            });
            // Use throughput as a secondary signal: black-box the result so the
            // compiler doesn't optimize away the measurement.
            std::hint::black_box(m.tokens_per_second());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_ttft_proxy, bench_decode_throughput, bench_rss_delta);
criterion_main!(benches);
