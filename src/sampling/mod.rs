//! Token sampling strategies — commit 8.5.
//!
//! Implements greedy, temperature, top-k, and top-p (nucleus) sampling
//! from a raw logits vector.  No external RNG dependency — uses a simple
//! xorshift64 PRNG seeded by the caller.
//!
//! # Sampling pipeline
//!
//! ```text
//! logits [vocab]
//!   │  temperature scaling  (if T > 0)
//!   ▼
//! scaled logits
//!   │  top-k filter         (if k > 0)
//!   ▼
//! filtered logits
//!   │  softmax
//!   ▼
//! probs
//!   │  top-p nucleus mask   (if p < 1.0)
//!   ▼
//! nucleus probs
//!   │  categorical draw     (or argmax if T == 0)
//!   ▼
//! token id
//! ```

/// Configuration for a single sampling step.
#[derive(Debug, Clone)]
pub struct SamplingConfig { // CHANGED
    /// Softmax temperature.  0.0 → greedy argmax; 1.0 → unscaled.
    pub temperature: f32,
    /// Keep only the top-k candidates before softmax.  0 → disabled.
    pub top_k: usize,
    /// Nucleus probability mass.  1.0 → disabled.
    pub top_p: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 0, top_p: 1.0 } // CHANGED
    }
}

impl SamplingConfig {
    /// Greedy (deterministic) preset.
    #[must_use]
    pub fn greedy() -> Self {
        Self { temperature: 0.0, top_k: 0, top_p: 1.0 } // CHANGED
    }
}

/// Minimal xorshift64 PRNG — no external deps required.
#[derive(Debug, Clone)]
pub struct SimpleRng { state: u64 } // CHANGED

impl SimpleRng {
    /// Seed with any non-zero value.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0xDEAD_BEEF_CAFE_1337 } else { seed } } // CHANGED
    }

    /// Next pseudo-random u64.
    pub fn next_u64(&mut self) -> u64 { // CHANGED
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform float in [0, 1).
    pub fn next_f32(&mut self) -> f32 { // CHANGED
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }
}

/// Sample the next token from `logits` using `config`.
///
/// `rng` is only used when `temperature > 0`; pass any value for greedy.
#[must_use]
pub fn sample(logits: &[f32], config: &SamplingConfig, rng: &mut SimpleRng) -> u32 { // CHANGED
    assert!(!logits.is_empty(), "logits must not be empty");

    // ── greedy short-circuit ───────────────────────────────────────────────
    if config.temperature <= 0.0 {
        return argmax(logits); // CHANGED
    }

    // ── temperature scaling ────────────────────────────────────────────────
    let inv_temp = 1.0 / config.temperature;
    let mut scaled: Vec<f32> = logits.iter().map(|&x| x * inv_temp).collect(); // CHANGED

    // ── top-k filter ───────────────────────────────────────────────────────
    // Zero out everything outside the top-k positions.
    if config.top_k > 0 && config.top_k < scaled.len() {
        let mut indexed: Vec<(usize, f32)> = scaled.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // descending
        let threshold = indexed[config.top_k - 1].1;
        for v in scaled.iter_mut() {
            if *v < threshold { *v = f32::NEG_INFINITY; } // CHANGED
        }
    }

    // ── softmax → probs ────────────────────────────────────────────────────
    let probs = softmax_vec(&scaled); // CHANGED

    // ── top-p nucleus ──────────────────────────────────────────────────────
    let probs = if config.top_p < 1.0 {
        nucleus_filter(probs, config.top_p) // CHANGED
    } else {
        probs
    };

    // ── categorical draw ───────────────────────────────────────────────────
    categorical_sample(&probs, rng) // CHANGED
}

// ── private helpers ───────────────────────────────────────────────────────────

/// Argmax — index of the highest value.
fn argmax(v: &[f32]) -> u32 { // CHANGED
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

/// Numerically-stable softmax over a slice → new Vec<f32>.
fn softmax_vec(v: &[f32]) -> Vec<f32> { // CHANGED
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp: Vec<f32> = v.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter_mut().for_each(|x| *x /= sum);
    exp
}

/// Zero out all tokens outside the top-p cumulative probability mass.
fn nucleus_filter(mut probs: Vec<f32>, top_p: f32) -> Vec<f32> { // CHANGED
    // Sort indices by probability descending
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Walk until cumulative mass ≥ top_p
    let mut cumsum = 0.0_f32;
    let mut cutoff_idx = indexed.len();
    for (rank, &(_, p)) in indexed.iter().enumerate() {
        cumsum += p;
        if cumsum >= top_p {
            cutoff_idx = rank + 1;
            break;
        }
    }

    // Zero out tokens ranked below the cutoff
    let keep: std::collections::HashSet<usize> =
        indexed[..cutoff_idx].iter().map(|&(i, _)| i).collect();
    for (i, p) in probs.iter_mut().enumerate() {
        if !keep.contains(&i) { *p = 0.0; }
    }

    // Re-normalise
    let sum: f32 = probs.iter().sum();
    if sum > 0.0 { probs.iter_mut().for_each(|p| *p /= sum); }
    probs
}

/// Draw a single index from a categorical distribution.
fn categorical_sample(probs: &[f32], rng: &mut SimpleRng) -> u32 { // CHANGED
    let r = rng.next_f32();
    let mut cumsum = 0.0_f32;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum { return i as u32; }
    }
    (probs.len() - 1) as u32 // fallback: last token
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_picks_argmax() { // CHANGED
        let logits = vec![0.1_f32, 5.0, 1.0, 0.5];
        let cfg = SamplingConfig::greedy();
        let mut rng = SimpleRng::new(42);
        assert_eq!(sample(&logits, &cfg, &mut rng), 1);
    }

    #[test]
    fn test_temperature_zero_is_greedy() { // CHANGED
        let logits = vec![1.0_f32, 10.0, 2.0];
        let cfg = SamplingConfig { temperature: 0.0, top_k: 0, top_p: 1.0 };
        let mut rng = SimpleRng::new(1);
        assert_eq!(sample(&logits, &cfg, &mut rng), 1);
    }

    #[test]
    fn test_top_k_restricts_candidates() { // CHANGED
        // With a spike logit at index 2 and top_k=1, must always return 2
        let logits = vec![-100.0_f32, -100.0, 100.0, -100.0];
        let cfg = SamplingConfig { temperature: 1.0, top_k: 1, top_p: 1.0 };
        let mut rng = SimpleRng::new(7);
        for _ in 0..20 {
            assert_eq!(sample(&logits, &cfg, &mut rng), 2);
        }
    }

    #[test]
    fn test_top_p_restricts_to_nucleus() { // CHANGED
        // Prob mass almost all on token 0; top_p=0.95 → only token 0 survives
        let logits = vec![100.0_f32, -100.0, -100.0];
        let cfg = SamplingConfig { temperature: 1.0, top_k: 0, top_p: 0.95 };
        let mut rng = SimpleRng::new(3);
        for _ in 0..20 {
            assert_eq!(sample(&logits, &cfg, &mut rng), 0);
        }
    }

    #[test]
    fn test_sample_always_in_vocab() { // CHANGED
        let logits: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let cfg = SamplingConfig { temperature: 1.0, top_k: 10, top_p: 0.9 };
        let mut rng = SimpleRng::new(99);
        for _ in 0..100 {
            let tok = sample(&logits, &cfg, &mut rng);
            assert!((tok as usize) < logits.len());
        }
    }

    #[test]
    fn test_rng_produces_varied_values() { // CHANGED
        let mut rng = SimpleRng::new(12345);
        let vals: Vec<f32> = (0..100).map(|_| rng.next_f32()).collect();
        let all_same = vals.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-9);
        assert!(!all_same, "RNG should not produce constant output");
        assert!(vals.iter().all(|&v| v >= 0.0 && v < 1.0));
    }

    #[test]
    fn test_softmax_vec_sums_to_one() { // CHANGED
        let logits = vec![1.0_f32, 2.0, 3.0, 4.0];
        let probs = softmax_vec(&logits);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_nucleus_filter_renormalises() { // CHANGED
        let probs = vec![0.5_f32, 0.3, 0.15, 0.05];
        let filtered = nucleus_filter(probs, 0.8);
        assert!((filtered.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}
