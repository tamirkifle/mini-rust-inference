//! SentencePiece-style BPE tokenizer implementation — commit 8.4.

use std::collections::HashMap;
use crate::gguf::Metadata;
use crate::gguf::keys;
use crate::model::{ModelError, Result};

// ── token type constants (SentencePiece / GGUF convention) ──────────────────
const TOKEN_TYPE_NORMAL:  i32 = 1;
const TOKEN_TYPE_UNKNOWN: i32 = 2;
const TOKEN_TYPE_CONTROL: i32 = 3;
const TOKEN_TYPE_BYTE:    i32 = 6;

/// SentencePiece BPE tokenizer loaded from GGUF metadata.
#[derive(Debug, Clone)]
pub struct Tokenizer { // CHANGED
    /// Vocabulary: token_id → string representation.
    vocab: Vec<String>,
    /// Reverse map: string → token_id.
    token_to_id: HashMap<String, u32>,
    /// Per-token score (higher = merge first).
    scores: Vec<f32>,
    /// Per-token type (normal, control, byte, …).
    token_type: Vec<i32>,
    /// Beginning-of-sequence token ID.
    pub bos_id: u32,
    /// End-of-sequence token ID.
    pub eos_id: u32,
}

impl Tokenizer {
    /// Load a tokenizer from GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns [`ModelError::MissingMetadataKey`] if any required key is absent.
    pub fn from_metadata(meta: &Metadata) -> Result<Self> { // CHANGED
        // ── vocab ──────────────────────────────────────────────────────────
        let vocab: Vec<String> = meta
            .get(keys::TOKENIZER_GGML_TOKENS)
            .and_then(|v| v.as_string_array())
            .ok_or(ModelError::MissingMetadataKey { key: keys::TOKENIZER_GGML_TOKENS })?
            .to_vec();

        // ── scores ─────────────────────────────────────────────────────────
        let scores: Vec<f32> = meta
            .get(keys::TOKENIZER_GGML_SCORES)
            .and_then(|v| v.as_f32_array())
            .ok_or(ModelError::MissingMetadataKey { key: keys::TOKENIZER_GGML_SCORES })?
            .to_vec();

        // ── token types ────────────────────────────────────────────────────
        let token_type: Vec<i32> = meta
            .get(keys::TOKENIZER_GGML_TOKEN_TYPE)
            .and_then(|v| v.as_i32_array())
            .ok_or(ModelError::MissingMetadataKey { key: keys::TOKENIZER_GGML_TOKEN_TYPE })?
            .to_vec();

        // ── special token IDs ──────────────────────────────────────────────
        let bos_id = meta.get_u32(keys::TOKENIZER_GGML_BOS_TOKEN_ID)
            .ok_or(ModelError::MissingMetadataKey { key: keys::TOKENIZER_GGML_BOS_TOKEN_ID })?;
        let eos_id = meta.get_u32(keys::TOKENIZER_GGML_EOS_TOKEN_ID)
            .ok_or(ModelError::MissingMetadataKey { key: keys::TOKENIZER_GGML_EOS_TOKEN_ID })?;

        // ── reverse map ────────────────────────────────────────────────────
        let token_to_id: HashMap<String, u32> = vocab.iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect(); // CHANGED

        Ok(Self { vocab, token_to_id, scores, token_type, bos_id, eos_id })
    }

    /// Vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize { self.vocab.len() } // CHANGED

    /// Token string for a given ID, or `None` if out of range.
    #[must_use]
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.vocab.get(id as usize).map(String::as_str) // CHANGED
    }

    /// Token ID for a given string, or `None` if not in vocabulary.
    #[must_use]
    pub fn token_to_id(&self, s: &str) -> Option<u32> {
        self.token_to_id.get(s).copied() // CHANGED
    }

    // ── encode ───────────────────────────────────────────────────────────────

    /// Encode `text` into a sequence of token IDs.
    ///
    /// Uses SentencePiece max-score BPE. Spaces are represented as `▁` (U+2581).
    /// Characters absent from the vocabulary fall back to individual byte tokens
    /// (`<0x00>` … `<0xFF>`).
    #[must_use]
    pub fn encode(&self, text: &str) -> Vec<u32> { // CHANGED
        if text.is_empty() { return vec![]; }

        // 1. Normalise: prepend ▁, replace every ASCII space with ▁
        let normalised = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));

        // 2. Split into Unicode scalar values → initial token list
        let mut symbols: Vec<String> = normalised
            .chars()
            .map(|c| c.to_string())
            .collect();

        // 3. Map each symbol to a token ID with byte fallback
        // (just used to check existence; we keep symbols as Strings)
        let _ = symbols.iter().map(|s| self.char_to_id(s)).collect::<Vec<_>>();

        // 4. BPE merge loop — max-score greedy
        loop {
            let mut best_score = f32::NEG_INFINITY;
            let mut best_pos   = usize::MAX;
            let mut best_tok   = String::new();

            for i in 0..symbols.len().saturating_sub(1) {
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = self.scores[id as usize];
                    if score > best_score {
                        best_score = score;
                        best_pos   = i;
                        best_tok   = merged;
                    }
                }
            }

            if best_pos == usize::MAX { break; } // no more merges

            // Apply the best merge
            symbols[best_pos] = best_tok;
            symbols.remove(best_pos + 1); // CHANGED
        }

        // 5. Map final symbols → token IDs (byte fallback for unknowns)
        symbols.iter().flat_map(|s| self.symbol_to_ids(s)).collect() // CHANGED
    }

    // ── decode ───────────────────────────────────────────────────────────────

    /// Decode a sequence of token IDs back into a UTF-8 string.
    ///
    /// Control tokens (type 3) are skipped.
    /// `▁` (U+2581) is replaced with an ASCII space.
    /// One leading space is trimmed.
    #[must_use]
    pub fn decode(&self, tokens: &[u32]) -> String { // CHANGED
        let mut out = String::new();
        for &id in tokens {
            let typ = self.token_type.get(id as usize).copied().unwrap_or(TOKEN_TYPE_NORMAL);
            if typ == TOKEN_TYPE_CONTROL { continue; }

            if let Some(s) = self.vocab.get(id as usize) {
                if typ == TOKEN_TYPE_BYTE {
                    // byte token: "<0xHH>" → actual byte
                    if let Some(b) = parse_byte_token(s) {
                        out.push(b as char);
                        continue;
                    }
                }
                out.push_str(s);
            }
        }
        // Replace ▁ with space and trim exactly one leading space
        let text = out.replace('\u{2581}', " ");
        text.strip_prefix(' ').unwrap_or(&text).to_string() // CHANGED
    }

    // ── private helpers ───────────────────────────────────────────────────────

    /// Single-character → token ID with byte fallback.
    fn char_to_id(&self, s: &str) -> Option<u32> { // CHANGED
        if let Some(&id) = self.token_to_id.get(s) {
            return Some(id);
        }
        // Try individual bytes
        let bytes = s.as_bytes();
        if bytes.len() == 1 {
            let key = format!("<0x{:02X}>", bytes[0]);
            return self.token_to_id.get(&key).copied();
        }
        None
    }

    /// Map a (possibly merged) symbol to one or more token IDs.
    ///
    /// If the symbol is in vocab, return `[id]`.
    /// Otherwise, fall back to individual byte tokens for each byte.
    fn symbol_to_ids(&self, s: &str) -> Vec<u32> { // CHANGED
        if let Some(&id) = self.token_to_id.get(s) {
            return vec![id];
        }
        s.bytes()
            .filter_map(|b| {
                let key = format!("<0x{:02X}>", b);
                self.token_to_id.get(&key).copied()
            })
            .collect()
    }
}

// ── module-level helper ───────────────────────────────────────────────────────

/// Parse a GGUF byte token string `"<0xHH>"` → byte value.
fn parse_byte_token(s: &str) -> Option<u8> { // CHANGED
    let inner = s.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(inner, 16).ok()
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::{Metadata, MetadataValue, keys};

    /// Build a minimal vocabulary sufficient for the unit tests below.
    ///
    /// Vocabulary (index → string → score → type):
    ///   0  <unk>  -inf  unknown(2)
    ///   1  <s>    -inf  control(3)   ← bos
    ///   2  </s>   -inf  control(3)   ← eos
    ///   3  <0x20>  0.0  byte(6)
    ///   4  ▁        0.0  normal(1)
    ///   5  h       -5.0  normal(1)
    ///   6  e       -5.0  normal(1)
    ///   7  l       -5.0  normal(1)
    ///   8  o       -5.0  normal(1)
    ///   9  he     -1.0  normal(1)   ← merge of h+e
    ///  10  hel    -0.5  normal(1)   ← merge of he+l
    ///  11  hell   -0.2  normal(1)   ← merge of hel+l
    ///  12  hello  -0.1  normal(1)   ← merge of hell+o
    ///  13  ▁hello  0.5  normal(1)   ← merge of ▁+hello
    fn make_tokenizer() -> Tokenizer {
        let vocab = vec![
            "<unk>", "<s>", "</s>", "<0x20>", "\u{2581}",
            "h", "e", "l", "o",
            "he", "hel", "hell", "hello", "\u{2581}hello",
        ];
        let scores: Vec<f32> = vec![
            f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY,
            0.0, 0.0,
            -5.0, -5.0, -5.0, -5.0,
            -1.0, -0.5, -0.2, -0.1, 0.5,
        ];
        let types: Vec<i32> = vec![
            2, 3, 3, 6, 1,
            1, 1, 1, 1,
            1, 1, 1, 1, 1,
        ];
        let mut meta = Metadata::new();
        meta.insert(keys::TOKENIZER_GGML_TOKENS.to_string(),
            MetadataValue::StringArray(vocab.iter().map(|s| s.to_string()).collect()));
        meta.insert(keys::TOKENIZER_GGML_SCORES.to_string(),
            MetadataValue::Float32Array(scores));
        meta.insert(keys::TOKENIZER_GGML_TOKEN_TYPE.to_string(),
            MetadataValue::Int32Array(types));
        meta.insert(keys::TOKENIZER_GGML_BOS_TOKEN_ID.to_string(),
            MetadataValue::Uint32(1));
        meta.insert(keys::TOKENIZER_GGML_EOS_TOKEN_ID.to_string(),
            MetadataValue::Uint32(2));
        Tokenizer::from_metadata(&meta).unwrap()
    }

    #[test]
    fn test_vocab_size() { // CHANGED
        let tok = make_tokenizer();
        assert_eq!(tok.vocab_size(), 14);
    }

    #[test]
    fn test_id_to_token_roundtrip() { // CHANGED
        let tok = make_tokenizer();
        assert_eq!(tok.id_to_token(5), Some("h"));
        assert_eq!(tok.id_to_token(13), Some("\u{2581}hello"));
        assert_eq!(tok.id_to_token(99), None);
    }

    #[test]
    fn test_token_to_id_roundtrip() { // CHANGED
        let tok = make_tokenizer();
        assert_eq!(tok.token_to_id("h"), Some(5));
        assert_eq!(tok.token_to_id("hello"), Some(12));
        assert_eq!(tok.token_to_id("xyz"), None);
    }

    #[test]
    fn test_encode_hello_merges_to_single_token() { // CHANGED
        let tok = make_tokenizer();
        // "hello" → prepend ▁ → "▁hello" → id 13
        let ids = tok.encode("hello");
        assert_eq!(ids, vec![13], "should fully merge to ▁hello token");
    }

    #[test]
    fn test_encode_empty_string() { // CHANGED
        let tok = make_tokenizer();
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn test_decode_skips_control_tokens() { // CHANGED
        let tok = make_tokenizer();
        // tokens: [bos=1, ▁hello=13, eos=2]
        let text = tok.decode(&[1, 13, 2]);
        assert_eq!(text, "hello"); // bos/eos (control) skipped, leading ▁ → space → trimmed
    }

    #[test]
    fn test_decode_single_token() { // CHANGED
        let tok = make_tokenizer();
        assert_eq!(tok.decode(&[13]), "hello");
    }

    #[test]
    fn test_encode_decode_roundtrip() { // CHANGED
        let tok = make_tokenizer();
        let ids = tok.encode("hello");
        let text = tok.decode(&ids);
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_bos_eos_ids() { // CHANGED
        let tok = make_tokenizer();
        assert_eq!(tok.bos_id, 1);
        assert_eq!(tok.eos_id, 2);
    }

    #[test]
    fn test_parse_byte_token_valid() { // CHANGED
        assert_eq!(parse_byte_token("<0x20>"), Some(0x20));
        assert_eq!(parse_byte_token("<0x00>"), Some(0x00));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("<0xAB>"), Some(0xAB));
    }

    #[test]
    fn test_parse_byte_token_invalid() { // CHANGED
        assert_eq!(parse_byte_token("hello"), None);
        assert_eq!(parse_byte_token("<0xGG>"), None);
    }

    #[test]
    fn test_missing_required_key() { // CHANGED
        let meta = Metadata::new(); // empty
        let result = Tokenizer::from_metadata(&meta);
        assert!(matches!(result, Err(ModelError::MissingMetadataKey { .. })));
    }
}
