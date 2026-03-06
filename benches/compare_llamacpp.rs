//! Controlled comparison benchmark — commit 17.3.
//!
//! Benchmarks this engine under the same `N_PROMPT` / `N_GEN` settings that
//! `scripts/benchmark_comparison.sh` uses when invoking llama.cpp, enabling
//! an apples-to-apples throughput comparison.
//!
//! # Configuration (via environment variables)
//!
//! | Variable          | Default | Meaning                                    |
//! |-------------------|---------|--------------------------------------------|
//! | `BENCH_N_PROMPT`  | 64      | Prompt token count (prefill length)        |
//! | `BENCH_N_GEN`     | 128     | Tokens to generate per iteration           |
//! | `BENCH_N_WARMUP`  | 2       | Extra warm-up iterations before sampling   |
//!
//! # Throughput reporting
//!
//! Using `Throughput::Elements(N_GEN)` causes Criterion to report results as
//! `thrpt: [lower mean upper] elem/s` where `elem/s == tok/s`.
//! `scripts/benchmark_comparison.sh` greps for `thrpt:` and extracts the mean.
//!
//! # Model note
//!
//! Uses a zero-weight in-memory proxy model — no GGUF file is required.
//! This measures raw kernel throughput (matmul, attention, norm pipeline),
//! not real model quality.  For a fair quality comparison use
//! `scripts/benchmark_comparison.sh` with a real GGUF model and llama.cpp.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use llm_engine::config::SessionConfig;
use llm_engine::generate::GenerateConfig;
use llm_engine::gguf::{Metadata, MetadataValue};
use llm_engine::model::llama::{LlamaConfig, LlamaModel, TransformerBlock};
use llm_engine::ops::rope::RopeTable;
use llm_engine::session::Session;
use llm_engine::tensor::Tensor;
use llm_engine::tokenizer::bpe::Tokenizer;

// ── settings (env-var-configurable, with defaults matching the shell script) ───

fn n_prompt() -> usize {
    std::env::var("BENCH_N_PROMPT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(64)
}

fn n_gen() -> usize {
    std::env::var("BENCH_N_GEN")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128)
}

// ── fixture helpers (shared with benches/inference.rs) ────────────────────────

fn make_config(vocab: u32) -> LlamaConfig {
    let mut m = Metadata::new();
    for (k, v) in [
        ("llama.block_count",          MetadataValue::Uint32(2)),
        ("llama.embedding_length",     MetadataValue::Uint32(64)),
        ("llama.attention.head_count", MetadataValue::Uint32(8)),
        ("llama.feed_forward_length",  MetadataValue::Uint32(128)),
        ("llama.context_length",       MetadataValue::Uint32(1024)),
        ("llama.vocab_size",           MetadataValue::Uint32(vocab)),
    ] {
        m.insert(k.to_string(), v);
    }
    LlamaConfig::from_metadata(&m).unwrap()
}

fn make_model(cfg: &LlamaConfig) -> Arc<LlamaModel> {
    let embed = cfg.embedding_length    as usize;
    let vocab = cfg.vocab_size          as usize;
    let ffn   = cfg.feed_forward_length as usize;
    let heads = cfg.n_heads             as usize;
    let kv    = cfg.n_kv_heads          as usize;
    let hd    = cfg.head_dim()          as usize;
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
    let mut vocab_strs: Vec<String> = vec!["<unk>".into(), "<s>".into(), "</s>".into()];
    for i in 3..vocab {
        vocab_strs.push(format!("tok{i}"));
    }
    let scores: Vec<f32> = vocab_strs.iter().enumerate()
        .map(|(i, _)| if i < 3 { f32::NEG_INFINITY } else { -(i as f32) })
        .collect();
    let types: Vec<i32> = vocab_strs.iter().enumerate()
        .map(|(i, _)| if i == 0 { 2 } else if i < 3 { 3 } else { 1 })
        .collect();

    let mut m = Metadata::new();
    m.insert(keys::TOKENIZER_GGML_TOKENS.to_string(),      MetadataValue::StringArray(vocab_strs));
    m.insert(keys::TOKENIZER_GGML_SCORES.to_string(),      MetadataValue::Float32Array(scores));
    m.insert(keys::TOKENIZER_GGML_TOKEN_TYPE.to_string(),  MetadataValue::Int32Array(types));
    m.insert(keys::TOKENIZER_GGML_BOS_TOKEN_ID.to_string(),MetadataValue::Uint32(1));
    m.insert(keys::TOKENIZER_GGML_EOS_TOKEN_ID.to_string(),MetadataValue::Uint32(2));
    Arc::new(Tokenizer::from_metadata(&m).unwrap())
}


// ── comparison benchmark ───────────────────────────────────────────────────────

fn bench_comparison(c: &mut Criterion) {
    let vocab: u32 = 256;
    let n_prompt = n_prompt();
    let n_gen    = n_gen();

    let cfg   = make_config(vocab);
    let model = make_model(&cfg);
    let tok   = make_tokenizer(vocab);

    // Build a prompt string that encodes to n_prompt tokens.
    // Each "tokN " is one token in the synthetic vocabulary.
    let prompt: String = (3..(3 + n_prompt))
        .map(|i| format!("tok{i} "))
        .collect();

    let mut group = c.benchmark_group("comparison");

    // Throughput::Elements(n_gen) → Criterion reports elem/s ≡ tok/s
    // The shell script greps for "thrpt:" to extract the mean.
    group.throughput(Throughput::Elements(n_gen as u64));

    // Bench label encodes n_gen so the shell script knows the denominator.
    group.bench_with_input(
        BenchmarkId::new("our_engine", format!("n{n_gen}")),
        &prompt,
        |bench, p| {
            bench.iter(|| {
                let gcfg = GenerateConfig::greedy(n_gen);
                // chunk_size=64 matches a realistic prefill chunk for this proxy model
                let scfg = SessionConfig::new(gcfg, 64, 1024);
                let mut session = Session::new(
                    Arc::clone(&model),
                    Arc::clone(&tok),
                    scfg,
                );
                // reset() is called inside generate(); each iter starts fresh
                session.generate(p).unwrap();
            });
        },
    );

    group.finish();
}

// ── decode-only path (matches llama.cpp's "tg" / token generation metric) ─────

fn bench_decode_only(c: &mut Criterion) {
    let vocab: u32 = 256;
    let n_gen    = n_gen();
    // Short fixed prompt so prefill cost is negligible; this isolates decode.
    let n_prompt_short = 4_usize;

    let cfg   = make_config(vocab);
    let model = make_model(&cfg);
    let tok   = make_tokenizer(vocab);

    let short_prompt: String = (3..(3 + n_prompt_short))
        .map(|i| format!("tok{i} "))
        .collect();

    let mut group = c.benchmark_group("decode_only");
    group.throughput(Throughput::Elements(n_gen as u64));

    group.bench_with_input(
        BenchmarkId::new("our_engine", format!("n{n_gen}")),
        &short_prompt,
        |bench, p| {
            bench.iter(|| {
                let gcfg = GenerateConfig::greedy(n_gen);
                let scfg = SessionConfig::new(gcfg, 64, 1024);
                let mut session = Session::new(
                    Arc::clone(&model),
                    Arc::clone(&tok),
                    scfg,
                );
                session.generate(p).unwrap();
            });
        },
    );

    group.finish();
}

criterion_group!(benches, bench_comparison, bench_decode_only);
criterion_main!(benches);
