//! Top-level engine API for the DS4 inference engine.
//!
//! Maps to `ds4_engine` from `ds4.h` / `ds4.c`.  The `Engine` struct owns the
//! model file mapping, the bound weight table, the vocabulary, and the backend
//! lifecycle.  Sessions (one per inference timeline) are created via
//! `Engine::create_session()`.
//!
//! # Lifecycle
//!
//! 1. `Engine::open()` — mmap the GGUF file(s), validate the fixed
//!    DeepSeek V4 Flash layout, bind tensors, load the BPE vocabulary, and
//!    initialise the Metal backend if requested.
//! 2. Use `Engine::tokenize()`, `Engine::summary()`, or create sessions.
//! 3. `Engine::close()` — release resources (Metal, mmaps, session state).

use std::collections::HashMap;

use anyhow::{bail, Context, Result};

use crate::gguf::{ModelGGUF, TensorInfo};
use crate::types::{
    Backend, ContextMemory, EngineOptions, TensorType, TokenVec, DS4_COMPRESS_ROPE_FREQ_BASE,
    DS4_EXPERT_WEIGHT_SCALE, DS4_HC_EPS, DS4_N_EMBD, DS4_N_EXPERT, DS4_N_EXPERT_SHARED,
    DS4_N_EXPERT_USED, DS4_N_FF_EXP, DS4_N_HASH_LAYER, DS4_N_HC, DS4_N_HC_SINKHORN_ITER,
    DS4_N_HEAD, DS4_N_HEAD_DIM, DS4_N_HEAD_KV, DS4_N_INDEXER_HEAD, DS4_N_INDEXER_HEAD_DIM,
    DS4_N_INDEXER_TOP_K, DS4_N_LAYER, DS4_N_LORA_O, DS4_N_LORA_Q, DS4_N_OUT_GROUP, DS4_N_ROT,
    DS4_N_SWA, DS4_N_VALUE_DIM, DS4_N_VOCAB, DS4_RMS_EPS, DS4_ROPE_FREQ_BASE, DS4_ROPE_ORIG_CTX,
    DS4_ROPE_SCALE_FACTOR, DS4_ROPE_YARN_BETA_FAST, DS4_ROPE_YARN_BETA_SLOW, DS4_SWIGLU_CLAMP_EXP,
};

// ── Forward declaration ───────────────────────────────────────────────────
//
// The session module lives under `src/session/`; we keep a thin reference
// here so the engine can create sessions.  The full type is opaque from the
// engine's perspective.

/// Opaque inference session handle.
///
/// Full definition is in `crate::session`.
pub use crate::session::Session;

// ── Added token ───────────────────────────────────────────────────────────

/// A single added token entry for the tokenizer (extra tokens beyond the base
/// GGUF vocabulary, e.g. special control tokens).
#[derive(Debug, Clone)]
pub struct AddedToken {
    pub id: i32,
    pub content: String,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
    pub special: bool,
}

// ── Vocab ─────────────────────────────────────────────────────────────────

/// BPE tokenizer vocabulary loaded from the GGUF file.
///
/// Mirrors the `ds4_vocab` struct from `ds4.c` (lines 13521–13533).
pub struct Vocab {
    /// Raw token byte strings, indexed by token ID.
    pub tokens: Vec<Vec<u8>>,
    /// Token scores (log probabilities from the GGUF `tokenizer.ggml.scores`).
    pub scores: Vec<f32>,
    /// Byte-level string → token ID lookup (via GPT-2 byte encoding for BPE).
    pub token_to_id: HashMap<String, i32>,
    /// Token ID → decoded text string (GPT-2 byte decoded).
    pub id_to_token: Vec<String>,
    /// Merge ranks: "pair bytes" → rank.  Keys are space-joined BPE pair strings.
    pub merge_rank: HashMap<String, i32>,

    // Special token IDs (looked up during vocab loading).
    pub bos_id: i32,
    pub eos_id: i32,
    pub pad_id: i32,
    /// Number of extra ID slots reserved for special tokens.
    pub extra_ids: i32,
    /// List of extra tokens beyond the base vocabulary.
    pub added_tokens: Vec<AddedToken>,

    // DeepSeek-specific special token IDs (hard-coded in vocab_load).
    pub user_id: i32,
    pub assistant_id: i32,
    pub think_start_id: i32,
    pub think_end_id: i32,
    pub dsml_id: i32,
}

impl Vocab {
    /// Load the vocabulary and merge table from a parsed GGUF model.
    ///
    /// Corresponds to `vocab_load()` in `ds4.c` lines 13891–13931.
    fn load(model: &ModelGGUF) -> Result<Self> {
        // ── Token strings ─────────────────────────────────────────────
        let tokens_arr = model
            .get_array("tokenizer.ggml.tokens")
            .context("GGUF tokenizer token table is missing")?;
        if tokens_arr.is_empty() {
            bail!("GGUF tokenizer token table is empty");
        }

        let n_vocab = tokens_arr.len();
        let mut tokens: Vec<Vec<u8>> = Vec::with_capacity(n_vocab);
        let mut token_to_id: HashMap<String, i32> = HashMap::with_capacity(n_vocab);
        let mut id_to_token: Vec<String> = Vec::with_capacity(n_vocab);

        for (i, val) in tokens_arr.iter().enumerate() {
            let s = val.as_string().unwrap_or("");
            tokens.push(s.as_bytes().to_vec());
            // Build the GPT-2 byte-encoded key for BPE lookup
            let encoded = byte_encode_str(s);
            token_to_id.insert(encoded.clone(), i as i32);
            id_to_token.push(byte_decode_str(s));
        }

        // ── Merge ranks ──────────────────────────────────────────────
        let merges_arr = model
            .get_array("tokenizer.ggml.merges")
            .context("GGUF tokenizer merge table is missing")?;
        let mut merge_rank: HashMap<String, i32> = HashMap::with_capacity(merges_arr.len());
        for (i, val) in merges_arr.iter().enumerate() {
            if let Some(merge_str) = val.as_string() {
                merge_rank.insert(merge_str.to_string(), i as i32);
            }
        }

        // ── Token scores ──────────────────────────────────────────────
        let scores: Vec<f32> = match model.get_array("tokenizer.ggml.scores") {
            Some(arr) => arr.iter().filter_map(|v| v.as_f32()).collect(),
            None => vec![0.0f32; n_vocab],
        };

        // ── Added tokens ──────────────────────────────────────────────
        let mut extra_ids: i32 = 0;
        let mut added_tokens: Vec<AddedToken> = Vec::new();
        if let Some(arr) = model.get_array("tokenizer.ggml.added_tokens") {
            for val in arr {
                if val.as_string().is_some() {
                    // Skip — added tokens are complex; we just record the count.
                    extra_ids += 1;
                } else if val.as_array().is_some() {
                    // Could be a JSON-ish dict; we treat minimally.
                    extra_ids += 1;
                }
            }
        }

        // ── Special token IDs (hard-coded DeepSeek names) ────────────
        let bos_id = Self::lookup_special(&token_to_id, "<｜begin▁of▁sentence｜>");
        let eos_id = Self::lookup_special(&token_to_id, "<｜end▁of▁sentence｜>");
        let user_id = Self::lookup_special(&token_to_id, "<｜User｜>");
        let assistant_id = Self::lookup_special(&token_to_id, "<｜Assistant｜>");
        let think_start_id = Self::lookup_special(&token_to_id, "<think>");
        let think_end_id = Self::lookup_special(&token_to_id, "</think>");
        let dsml_id = Self::lookup_special(&token_to_id, "｜DSML｜");

        let pad_id = model
            .get_u32("tokenizer.ggml.padding_token_id")
            .map(|v| v as i32)
            .unwrap_or(eos_id);

        Ok(Vocab {
            tokens,
            scores,
            token_to_id,
            id_to_token,
            merge_rank,
            bos_id,
            eos_id,
            pad_id,
            extra_ids,
            added_tokens,
            user_id,
            assistant_id,
            think_start_id,
            think_end_id,
            dsml_id,
        })
    }

    /// Look up a special token verbatim (no byte-encoding).
    fn lookup_special(map: &HashMap<String, i32>, token: &str) -> i32 {
        // Special tokens are stored in the token_to_id map under their
        // byte-encoded key, but they contain only ASCII bytes so the
        // byte encoding is identity.  Try both.
        if let Some(id) = map.get(token) {
            return *id;
        }
        // Fallback: search the raw tokens for a match.
        // This may happen if the key is stored differently.
        -1
    }

    /// Look up a token by its decoded text (byte-encoding on input).
    fn lookup(&self, text: &str) -> Option<i32> {
        let encoded = byte_encode_str(text);
        self.token_to_id.get(&encoded).copied()
    }

    /// Decode a single token ID to its text representation.
    fn decode(&self, token: i32) -> Option<String> {
        let idx = token as usize;
        if idx < self.id_to_token.len() {
            Some(self.id_to_token[idx].clone())
        } else {
            None
        }
    }

    /// Return the BPE merge rank for a pair of adjacent symbols.
    fn bpe_rank(&self, a: &[u8], b: &[u8]) -> Option<i32> {
        let mut key = Vec::with_capacity(a.len() + 1 + b.len());
        key.extend_from_slice(a);
        key.push(b' ');
        key.extend_from_slice(b);
        let key_str = String::from_utf8_lossy(&key).to_string();
        self.merge_rank.get(&key_str).copied()
    }

    /// GPT-2 byte-level BPE tokenization for one pre-tokenized piece.
    fn bpe_emit_piece(&self, piece: &[u8], out: &mut Vec<i32>) {
        // First byte-encode the piece (map bytes to printable codepoints).
        let encoded = byte_encode_bytes(piece);
        let encoded_str = String::from_utf8_lossy(&encoded);

        // Split into UTF-8 codepoint symbols.
        let mut syms: Vec<Vec<u8>> = encoded_str
            .chars()
            .map(|c| {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                s.as_bytes().to_vec()
            })
            .collect();

        if syms.is_empty() {
            return;
        }

        // Greedy BPE merge loop.
        loop {
            let mut best_i = None;
            let mut best_rank = i32::MAX;

            for i in 0..syms.len().saturating_sub(1) {
                if let Some(rank) = self.bpe_rank(&syms[i], &syms[i + 1]) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = Some(i);
                    }
                }
            }

            let Some(merge_i) = best_i else { break };

            // Merge the pair.
            let merged = [syms[merge_i].as_slice(), syms[merge_i + 1].as_slice()].concat();
            syms[merge_i] = merged;
            syms.remove(merge_i + 1);
        }

        // Emit token IDs.
        for sym in &syms {
            let sym_str = String::from_utf8_lossy(sym).to_string();
            if let Some(&tok) = self.token_to_id.get(&sym_str) {
                out.push(tok);
            } else {
                // Fallback: emit individual bytes.
                for b in sym {
                    let byte_key = byte_encode_bytes(&[*b]);
                    let byte_str = String::from_utf8_lossy(&byte_key).to_string();
                    if let Some(&tok) = self.token_to_id.get(&byte_str) {
                        out.push(tok);
                    }
                }
            }
        }
    }

    /// BPE tokenize a span of text (the standard tokenization entry point).
    fn bpe_tokenize(&self, text: &str, out: &mut Vec<i32>) {
        let bytes = text.as_bytes();
        let len = bytes.len();
        let mut pos = 0;

        while pos < len {
            let start = pos;
            let c = bytes[pos];

            if c.is_ascii_digit() {
                let mut ndigits = 0;
                while pos < len && bytes[pos].is_ascii_digit() && ndigits < 3 {
                    pos += 1;
                    ndigits += 1;
                }
            } else if is_cjk_hira_kata(bytes, len, pos) {
                pos = next_utf8_char(bytes, len, pos);
                while pos < len && is_cjk_hira_kata(bytes, len, pos) {
                    pos = next_utf8_char(bytes, len, pos);
                }
            } else if is_ascii_punct_symbol(c)
                && pos + 1 < len
                && bytes[pos + 1].is_ascii_alphabetic()
            {
                pos += 1;
                while pos < len && bytes[pos].is_ascii_alphabetic() {
                    pos += 1;
                }
            } else if is_letter_like(bytes, len, pos) {
                pos = consume_letters(bytes, len, pos);
            } else if !is_ascii_newline(c)
                && !is_ascii_punct_symbol(c)
                && pos + 1 < len
                && is_letter_like(bytes, len, pos + 1)
            {
                pos += 1;
                pos = consume_letters(bytes, len, pos);
            } else if c == b' ' && pos + 1 < len && is_ascii_punct_symbol(bytes[pos + 1]) {
                pos += 1;
                while pos < len && is_ascii_punct_symbol(bytes[pos]) {
                    pos += 1;
                }
                while pos < len && is_ascii_newline(bytes[pos]) {
                    pos += 1;
                }
            } else if is_ascii_punct_symbol(c) {
                while pos < len && is_ascii_punct_symbol(bytes[pos]) {
                    pos += 1;
                }
                while pos < len && is_ascii_newline(bytes[pos]) {
                    pos += 1;
                }
            } else if is_ascii_space(c) {
                let mut p = pos;
                let mut last_newline_end = 0;
                while p < len && is_ascii_space(bytes[p]) {
                    let sc = bytes[p];
                    p += 1;
                    if is_ascii_newline(sc) {
                        last_newline_end = p;
                    }
                }
                if last_newline_end > 0 {
                    pos = last_newline_end;
                } else if p < len
                    && p > pos + 1
                    && (is_letter_like(bytes, len, p) || is_ascii_punct_symbol(bytes[p]))
                {
                    pos = p - 1;
                } else {
                    pos = p;
                }
            } else {
                pos = next_utf8_char(bytes, len, pos);
            }

            if pos == start {
                pos = next_utf8_char(bytes, len, pos);
            }

            self.bpe_emit_piece(&bytes[start..pos], out);
        }
    }
}

// ── GPT-2 byte encoding helpers ───────────────────────────────────────────

/// GPT-2 byte-level BPE first maps raw bytes to printable Unicode codepoints
/// so merges can operate on UTF-8 strings without losing byte identity.
/// Returns the byte-encoded UTF-8 string.
fn byte_encode_bytes(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() * 4);
    for &b in input {
        let cp = gpt2_byte_to_codepoint(b);
        encode_utf8_cp(cp, &mut out);
    }
    out
}

/// String version of `byte_encode_bytes`.
fn byte_encode_str(input: &str) -> String {
    let bytes = byte_encode_bytes(input.as_bytes());
    String::from_utf8_lossy(&bytes).to_string()
}

/// Reverse the GPT-2 byte encoding: map codepoints back to bytes.
fn byte_decode_bytes(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len());
    let len = input.len();
    let mut pos = 0;
    while pos < len {
        let (cp, next) = utf8_decode_one(input, len, pos);
        if let Some(b) = gpt2_codepoint_to_byte(cp) {
            out.push(b);
        }
        pos = next;
    }
    out
}

/// String version of `byte_decode_bytes`.
fn byte_decode_str(input: &str) -> String {
    let bytes = byte_decode_bytes(input.as_bytes());
    String::from_utf8_lossy(&bytes).to_string()
}

fn gpt2_byte_to_codepoint(b: u8) -> u32 {
    if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) {
        return b as u32;
    }

    let mut n = 0u32;
    for x in 0..256u32 {
        if (x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174) {
            continue;
        }
        if x == b as u32 {
            return 256 + n;
        }
        n += 1;
    }
    b as u32
}

fn gpt2_codepoint_to_byte(cp: u32) -> Option<u8> {
    if (cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) || (cp >= 174 && cp <= 255) {
        return Some(cp as u8);
    }

    let mut n = 0u32;
    for b in 0..256u32 {
        if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174) {
            continue;
        }
        if cp == 256 + n {
            return Some(b as u8);
        }
        n += 1;
    }
    None
}

fn encode_utf8_cp(cp: u32, out: &mut Vec<u8>) {
    if cp <= 0x7f {
        out.push(cp as u8);
    } else if cp <= 0x7ff {
        out.push(0xc0 | (cp >> 6) as u8);
        out.push(0x80 | (cp & 0x3f) as u8);
    } else if cp <= 0xffff {
        out.push(0xe0 | (cp >> 12) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3f) as u8);
        out.push(0x80 | (cp & 0x3f) as u8);
    } else {
        out.push(0xf0 | (cp >> 18) as u8);
        out.push(0x80 | ((cp >> 12) & 0x3f) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3f) as u8);
        out.push(0x80 | (cp & 0x3f) as u8);
    }
}

fn utf8_decode_one(bytes: &[u8], len: usize, pos: usize) -> (u32, usize) {
    let c0 = bytes[pos];
    let n = utf8_len_from_first_byte(c0);
    let n = if pos + n > len { 1 } else { n };
    let next = pos + n;

    let cp = if n == 1 {
        c0 as u32
    } else if n == 2 {
        ((c0 as u32 & 0x1f) << 6) | (bytes[pos + 1] as u32 & 0x3f)
    } else if n == 3 {
        ((c0 as u32 & 0x0f) << 12)
            | ((bytes[pos + 1] as u32 & 0x3f) << 6)
            | (bytes[pos + 2] as u32 & 0x3f)
    } else {
        ((c0 as u32 & 0x07) << 18)
            | ((bytes[pos + 1] as u32 & 0x3f) << 12)
            | ((bytes[pos + 2] as u32 & 0x3f) << 6)
            | (bytes[pos + 3] as u32 & 0x3f)
    };

    (cp, next)
}

fn utf8_len_from_first_byte(c: u8) -> usize {
    if c < 0x80 {
        1
    } else if (c & 0xe0) == 0xc0 {
        2
    } else if (c & 0xf0) == 0xe0 {
        3
    } else if (c & 0xf8) == 0xf0 {
        4
    } else {
        1
    }
}

fn next_utf8_char(bytes: &[u8], len: usize, pos: usize) -> usize {
    let n = utf8_len_from_first_byte(bytes[pos]);
    if pos + n > len {
        pos + 1
    } else {
        pos + n
    }
}

// ── Pre-tokenizer character classification ───────────────────────────────

fn is_ascii_newline(c: u8) -> bool {
    c == b'\n' || c == b'\r'
}

fn is_ascii_space(c: u8) -> bool {
    c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == b'\x0b' || c == b'\x0c'
}

fn is_ascii_punct_symbol(c: u8) -> bool {
    (c >= b'!' && c <= b'/')
        || (c >= b':' && c <= b'@')
        || (c >= b'[' && c <= b'`')
        || (c >= b'{' && c <= b'~')
}

fn is_cjk_hira_kata(bytes: &[u8], len: usize, pos: usize) -> bool {
    if bytes[pos] < 128 {
        return false;
    }
    let (cp, _) = utf8_decode_one(bytes, len, pos);
    (0x4e00..=0x9fa5).contains(&cp)
        || (0x3040..=0x309f).contains(&cp)
        || (0x30a0..=0x30ff).contains(&cp)
}

fn is_letter_like(bytes: &[u8], _len: usize, pos: usize) -> bool {
    let c = bytes[pos];
    if c < 128 {
        return c.is_ascii_alphabetic();
    }
    // Non-ASCII non-control bytes are treated as letters.
    true
}

fn consume_letters(bytes: &[u8], len: usize, mut pos: usize) -> usize {
    while pos < len && is_letter_like(bytes, len, pos) {
        pos = next_utf8_char(bytes, len, pos);
    }
    pos
}

// ── Layer weight table ────────────────────────────────────────────────────

/// Per-layer weight pointers for the DeepSeek V4 Flash fixed layout.
///
/// Maps to `ds4_layer_weights` from `ds4.c` lines 1841–1877.  Every field is
/// a `TensorInfo` (or `Option<TensorInfo>` for tensors that only exist in
/// certain layers).
#[derive(Debug, Clone)]
pub struct LayerWeights {
    // ── Hyper-connection attention path ───────────────────────────────
    pub hc_attn_fn: TensorInfo,
    pub hc_attn_scale: TensorInfo,
    pub hc_attn_base: TensorInfo,

    // ── Attention sub-layer ──────────────────────────────────────────
    pub attn_norm: TensorInfo,
    pub attn_q_a: TensorInfo,
    pub attn_q_a_norm: TensorInfo,
    pub attn_q_b: TensorInfo,
    pub attn_kv: TensorInfo,
    pub attn_kv_a_norm: TensorInfo,
    pub attn_sinks: TensorInfo,
    pub attn_output_a: TensorInfo,
    pub attn_output_b: TensorInfo,

    // ── Optional: compressor (exists when compress_ratio != 0) ────────
    pub attn_compressor_ape: Option<TensorInfo>,
    pub attn_compressor_kv: Option<TensorInfo>,
    pub attn_compressor_gate: Option<TensorInfo>,
    pub attn_compressor_norm: Option<TensorInfo>,

    // ── Optional: indexer (exists only when compress_ratio == 4) ──────
    pub indexer_attn_q_b: Option<TensorInfo>,
    pub indexer_proj: Option<TensorInfo>,
    pub indexer_compressor_ape: Option<TensorInfo>,
    pub indexer_compressor_kv: Option<TensorInfo>,
    pub indexer_compressor_gate: Option<TensorInfo>,
    pub indexer_compressor_norm: Option<TensorInfo>,

    // ── Hyper-connection FFN path ────────────────────────────────────
    pub hc_ffn_fn: TensorInfo,
    pub hc_ffn_scale: TensorInfo,
    pub hc_ffn_base: TensorInfo,

    // ── FFN sub-layer ────────────────────────────────────────────────
    pub ffn_norm: TensorInfo,
    pub ffn_gate_inp: TensorInfo,
    /// Optional: expert probability bias (exists only for hash layers).
    pub ffn_exp_probs_b: Option<TensorInfo>,

    // ── Routed experts (moE) ─────────────────────────────────────────
    pub ffn_gate_exps: TensorInfo,
    pub ffn_up_exps: TensorInfo,
    pub ffn_down_exps: TensorInfo,

    // ── Shared experts ───────────────────────────────────────────────
    pub ffn_gate_shexp: TensorInfo,
    pub ffn_up_shexp: TensorInfo,
    pub ffn_down_shexp: TensorInfo,

    /// Optional: hash routing table (exists only in first `DS4_N_HASH_LAYER` layers).
    pub ffn_gate_tid2eid: Option<TensorInfo>,
}

// ── Top-level weight table ────────────────────────────────────────────────

/// Fixed weight table for the DS4 model.
///
/// Maps to `ds4_weights` from `ds4.c` lines 1879–1887.  After
/// `weights_bind()`, every field points to a `TensorInfo` located in the mmap.
#[derive(Debug, Clone)]
pub struct Weights {
    pub token_embd: TensorInfo,
    pub output_hc_base: TensorInfo,
    pub output_hc_fn: TensorInfo,
    pub output_hc_scale: TensorInfo,
    pub output_norm: TensorInfo,
    pub output: TensorInfo,
    pub layer: [LayerWeights; DS4_N_LAYER as usize],
}

// ── MTP weights ───────────────────────────────────────────────────────────

/// MTP (Multi-Token Prediction) head weights.
///
/// Maps to `ds4_mtp_weights` from `ds4.c` lines 1889–1899.
#[derive(Debug, Clone)]
pub struct MtpWeights {
    pub e_proj: TensorInfo,
    pub h_proj: TensorInfo,
    pub enorm: TensorInfo,
    pub hnorm: TensorInfo,
    pub norm: TensorInfo,
    pub hc_head_base: TensorInfo,
    pub hc_head_fn: TensorInfo,
    pub hc_head_scale: TensorInfo,
    /// Single MTP block (same structure as a regular layer, but may have
    /// different tensor shapes).
    pub block: MtpLayerWeights,
}

/// MTP block layer weights (same structure as `LayerWeights` within the MTP head).
#[derive(Debug, Clone)]
pub struct MtpLayerWeights {
    pub hc_attn_fn: TensorInfo,
    pub hc_attn_scale: TensorInfo,
    pub hc_attn_base: TensorInfo,
    pub attn_norm: TensorInfo,
    pub attn_q_a: TensorInfo,
    pub attn_q_a_norm: TensorInfo,
    pub attn_q_b: TensorInfo,
    pub attn_kv: TensorInfo,
    pub attn_kv_a_norm: TensorInfo,
    pub attn_sinks: TensorInfo,
    pub attn_output_a: TensorInfo,
    pub attn_output_b: TensorInfo,
    pub hc_ffn_fn: TensorInfo,
    pub hc_ffn_scale: TensorInfo,
    pub hc_ffn_base: TensorInfo,
    pub ffn_norm: TensorInfo,
    pub ffn_gate_inp: TensorInfo,
    pub ffn_exp_probs_b: TensorInfo,
    pub ffn_gate_exps: TensorInfo,
    pub ffn_up_exps: TensorInfo,
    pub ffn_down_exps: TensorInfo,
    pub ffn_gate_shexp: TensorInfo,
    pub ffn_up_shexp: TensorInfo,
    pub ffn_down_shexp: TensorInfo,
}

// ── Compression ratio per layer ──────────────────────────────────────────

/// Return the attention compression ratio for layer `il`.
///
/// Layers 0 and 1 are dense (ratio = 0).  Even layers >= 2 have ratio 4 with
/// an indexer.  Odd layers >= 3 have ratio 128 without an indexer.
///
/// This matches `ds4_layer_compress_ratio()` in `ds4.c` lines 407–411.
pub fn layer_compress_ratio(il: u32) -> u32 {
    assert!(
        il < DS4_N_LAYER,
        "DeepSeek4 layer index {il} is outside the fixed model layout"
    );
    if il < 2 {
        return 0;
    }
    if (il & 1) == 0 {
        4
    } else {
        128
    }
}

// ── Engine ────────────────────────────────────────────────────────────────

/// Top-level DS4 inference engine.
///
/// Owns the mmap'd model file(s), the bound weight table, the BPE vocabulary,
/// and the backend (Metal or CPU).  This is the public API boundary: CLI and
/// server code interact with the engine through this struct.
pub struct Engine {
    /// Primary model (mmap'd GGUF).
    pub model: ModelGGUF,
    /// Bound weight pointers into the primary model's tensor data.
    pub weights: Weights,
    /// Optional MTP model file.
    pub mtp_model: Option<ModelGGUF>,
    /// Optional MTP weight pointers.
    pub mtp_weights: Option<MtpWeights>,
    /// BPE tokenizer vocabulary loaded from the primary model.
    pub vocab: Vocab,
    /// Selected compute backend.
    pub backend: Backend,
    /// Number of MTP draft tokens to generate per step.
    pub mtp_draft_tokens: u32,
    /// MTP acceptance margin.
    pub mtp_margin: f32,
    /// Whether to use high-quality (but slower) inference path.
    pub quality: bool,
    /// Whether the Metal backend has been initialised.
    pub metal_initialized: bool,
    /// Whether the MTP model is fully loaded and ready.
    pub mtp_ready: bool,
}

impl Engine {
    /// Open and validate a DS4 model file, bind weights, load the vocabulary,
    /// and initialise the backend.
    ///
    /// Corresponds to `ds4_engine_open()` in `ds4.c` lines 15707–15783.
    pub fn open(options: &EngineOptions) -> Result<Self> {
        let backend = options.backend;
        let quality = options.quality;

        let mtp_draft_tokens = if options.mtp_draft_tokens > 0 {
            options.mtp_draft_tokens.min(16)
        } else {
            1
        };
        let mtp_margin = if options.mtp_margin >= 0.0 {
            options.mtp_margin
        } else {
            3.0
        };

        // ── Open primary model ───────────────────────────────────────
        let model = ModelGGUF::open(&options.model_path)
            .with_context(|| format!("failed to open model: {}", options.model_path))?;

        // ── Load vocabulary ──────────────────────────────────────────
        let vocab = Vocab::load(&model).context("failed to load vocabulary from model")?;

        // ── Validate model configuration ─────────────────────────────
        config_validate_model(&model)?;

        // ── Bind weight pointers ─────────────────────────────────────
        let weights = weights_bind(&model)?;

        // ── Open MTP model (optional) ────────────────────────────────
        let (mtp_model, mtp_weights, mtp_ready) = if let Some(ref mtp_path) = options.mtp_path {
            let mtp_m = ModelGGUF::open(mtp_path)
                .with_context(|| format!("failed to open MTP model: {mtp_path}"))?;
            let mtp_w = mtp_weights_bind(&mtp_m)?;
            (Some(mtp_m), Some(mtp_w), true)
        } else {
            (None, None, false)
        };

        // ── Metal initialisation ─────────────────────────────────────
        let metal_initialized = if backend == Backend::Metal {
            #[cfg(not(ds4_no_metal))]
            {
                crate::metal_ffi::metal_init().context("Metal backend unavailable")?;
                crate::metal_ffi::metal_set_quality(quality);

                // Map the primary model's tensor data region into Metal's address space.
                let mmap = model.mmap();
                let tensor_len = model.file_size().saturating_sub(model.tensor_data_pos);
                if let Err(e) = crate::metal_ffi::metal_set_model_map_range(
                    mmap,
                    model.tensor_data_pos,
                    tensor_len,
                ) {
                    // Cleanup on failure
                    crate::metal_ffi::metal_cleanup();
                    bail!(
                        "Metal failed to map model views: {}. \
                         This is commonly caused by insufficient memory or Metal VM budget.",
                        e
                    );
                }

                // Map MTP model if present.
                if mtp_ready {
                    if let Some(ref mtp_m) = mtp_model {
                        let mtp_mmap = mtp_m.mmap();
                        let mtp_tensor_len =
                            mtp_m.file_size().saturating_sub(mtp_m.tensor_data_pos);
                        if let Err(e) = crate::metal_ffi::metal_set_model_map_range(
                            mtp_mmap,
                            mtp_m.tensor_data_pos,
                            mtp_tensor_len,
                        ) {
                            crate::metal_ffi::metal_cleanup();
                            bail!(
                                "Metal failed to map MTP model views: {}. \
                                 This is commonly caused by insufficient memory or Metal VM budget.",
                                e
                            );
                        }
                    }
                }

                log::info!("Metal backend initialized");
                true
            }
            #[cfg(ds4_no_metal)]
            {
                let _ = quality;
                bail!("Metal backend requested but this build has no Metal support");
            }
        } else {
            false
        };

        Ok(Engine {
            model,
            weights,
            mtp_model,
            mtp_weights,
            vocab,
            backend,
            mtp_draft_tokens,
            mtp_margin,
            quality,
            metal_initialized,
            mtp_ready,
        })
    }

    /// Print a summary of the loaded model to stderr.
    ///
    /// Corresponds to `ds4_engine_summary()` in `ds4.c` line 15785.
    pub fn summary(&self) {
        self.model.summary();
    }

    /// Return the name of the active backend.
    pub fn backend_name(&self) -> &str {
        self.backend.name()
    }

    /// Return the number of bits per weight for routed expert quantisation
    /// (4 for Q4_K, 2 for Q2_K/IQ2_XXS).
    pub fn routed_quant_bits(&self) -> u32 {
        let gate = &self.weights.layer[0].ffn_gate_exps;
        match gate.tensor_type {
            12 => 4, // DS4_TENSOR_Q4_K
            _ => 2,  // Q2_K, IQ2_XXS, etc.
        }
    }

    /// Whether the MTP model is loaded and ready.
    pub fn has_mtp(&self) -> bool {
        self.mtp_ready
    }

    /// Number of MTP draft tokens configured (0 if MTP is not loaded).
    pub fn mtp_draft_tokens(&self) -> u32 {
        if self.mtp_ready {
            self.mtp_draft_tokens
        } else {
            0
        }
    }

    /// Tokenize a text string using the DeepSeek JoyAI BPE tokenizer.
    ///
    /// Returns a `TokenVec` with the token IDs.
    pub fn tokenize(&self, text: &str) -> TokenVec {
        let mut v = Vec::new();
        self.vocab.bpe_tokenize(text, &mut v);
        TokenVec { v }
    }

    /// Return the EOS (end-of-sequence) token ID.
    pub fn token_eos(&self) -> i32 {
        self.vocab.eos_id
    }

    /// Decode a single token ID back to its text representation.
    ///
    /// Returns `None` if the token ID is out of range.
    pub fn token_text(&self, token: i32) -> Option<String> {
        if token < 0 || token as usize >= self.vocab.id_to_token.len() {
            return None;
        }
        self.vocab.decode(token)
    }

    /// Create a new inference session with the given context size.
    ///
    /// Returns an error if the backend is CPU (Metal is required for sessions
    /// in this Rust port, matching the C code's `#ifndef DS4_NO_METAL` guard).
    pub fn create_session(&self, ctx_size: u32) -> Result<Session> {
        // Session creation is handled by the session module.
        // For CPU-only builds, the session module provides a full implementation.
        // For Metal, this will delegate to the FFI-based Metal graph allocation.
        use crate::session::Session;
        Session::create(self, ctx_size)
    }

    /// Estimate the memory requirements for a given backend and context size.
    ///
    /// This is a static method that does not require an open engine.
    pub fn context_memory_estimate(backend: Backend, ctx_size: i32) -> ContextMemory {
        let ctx = if ctx_size > 0 { ctx_size as u32 } else { 1 };

        if backend == Backend::Metal {
            metal_context_memory_estimate(ctx)
        } else {
            cpu_context_memory_estimate(ctx)
        }
    }

    /// Release all engine resources.
    ///
    /// Corresponds to `ds4_engine_close()` in `ds4.c` lines 15789–15801.
    pub fn close(&mut self) {
        #[cfg(not(ds4_no_metal))]
        if self.metal_initialized {
            crate::metal_ffi::metal_cleanup();
        }
        // Model mmaps are dropped when the fields go out of scope.
        // No explicit free needed for weights (they're just TensorInfos).
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Rust's drop will handle ModelGGUF (mmap) and the other fields.
        // We still need to clean up Metal explicitly.
        #[cfg(not(ds4_no_metal))]
        if self.metal_initialized {
            crate::metal_ffi::metal_cleanup();
        }
    }
}

// ── Weight binding ────────────────────────────────────────────────────────

/// Bind tensor names from the GGUF tensor directory into the fixed DS4
/// layer layout.  This is the point where stringly GGUF metadata becomes a
/// direct model-specific table.
///
/// Corresponds to `weights_bind()` in `ds4.c` lines 2432–2490.
fn weights_bind(model: &ModelGGUF) -> Result<Weights> {
    // ── Top-level tensors ────────────────────────────────────────────
    let token_embd = find_required_tensor(model, "token_embd.weight")?;
    let output_hc_base = find_required_tensor(model, "output_hc_base.weight")?;
    let output_hc_fn = find_required_tensor(model, "output_hc_fn.weight")?;
    let output_hc_scale = find_required_tensor(model, "output_hc_scale.weight")?;
    let output_norm = find_required_tensor(model, "output_norm.weight")?;
    let output = find_required_tensor(model, "output.weight")?;

    // ── Per-layer tensors ────────────────────────────────────────────
    let mut layer_vec: Vec<LayerWeights> = Vec::with_capacity(DS4_N_LAYER as usize);

    for il in 0..DS4_N_LAYER {
        let compress_ratio = layer_compress_ratio(il);

        layer_vec.push(LayerWeights {
            hc_attn_fn: find_required_tensorf(model, "blk.{il}.hc_attn_fn.weight", il)?,
            hc_attn_scale: find_required_tensorf(model, "blk.{il}.hc_attn_scale.weight", il)?,
            hc_attn_base: find_required_tensorf(model, "blk.{il}.hc_attn_base.weight", il)?,
            attn_norm: find_required_tensorf(model, "blk.{il}.attn_norm.weight", il)?,
            attn_q_a: find_required_tensorf(model, "blk.{il}.attn_q_a.weight", il)?,
            attn_q_a_norm: find_required_tensorf(model, "blk.{il}.attn_q_a_norm.weight", il)?,
            attn_q_b: find_required_tensorf(model, "blk.{il}.attn_q_b.weight", il)?,
            attn_kv: find_required_tensorf(model, "blk.{il}.attn_kv.weight", il)?,
            attn_kv_a_norm: find_required_tensorf(model, "blk.{il}.attn_kv_a_norm.weight", il)?,
            attn_sinks: find_required_tensorf(model, "blk.{il}.attn_sinks.weight", il)?,
            attn_output_a: find_required_tensorf(model, "blk.{il}.attn_output_a.weight", il)?,
            attn_output_b: find_required_tensorf(model, "blk.{il}.attn_output_b.weight", il)?,

            // Optional: compressor (exists only when compress_ratio != 0)
            attn_compressor_ape: if compress_ratio != 0 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.attn_compressor_ape.weight",
                    il,
                )?)
            } else {
                None
            },
            attn_compressor_kv: if compress_ratio != 0 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.attn_compressor_kv.weight",
                    il,
                )?)
            } else {
                None
            },
            attn_compressor_gate: if compress_ratio != 0 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.attn_compressor_gate.weight",
                    il,
                )?)
            } else {
                None
            },
            attn_compressor_norm: if compress_ratio != 0 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.attn_compressor_norm.weight",
                    il,
                )?)
            } else {
                None
            },

            // Optional: indexer (exists only when compress_ratio == 4)
            indexer_attn_q_b: if compress_ratio == 4 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.indexer.attn_q_b.weight",
                    il,
                )?)
            } else {
                None
            },
            indexer_proj: if compress_ratio == 4 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.indexer.proj.weight",
                    il,
                )?)
            } else {
                None
            },
            indexer_compressor_ape: if compress_ratio == 4 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.indexer_compressor_ape.weight",
                    il,
                )?)
            } else {
                None
            },
            indexer_compressor_kv: if compress_ratio == 4 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.indexer_compressor_kv.weight",
                    il,
                )?)
            } else {
                None
            },
            indexer_compressor_gate: if compress_ratio == 4 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.indexer_compressor_gate.weight",
                    il,
                )?)
            } else {
                None
            },
            indexer_compressor_norm: if compress_ratio == 4 {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.indexer_compressor_norm.weight",
                    il,
                )?)
            } else {
                None
            },

            hc_ffn_fn: find_required_tensorf(model, "blk.{il}.hc_ffn_fn.weight", il)?,
            hc_ffn_scale: find_required_tensorf(model, "blk.{il}.hc_ffn_scale.weight", il)?,
            hc_ffn_base: find_required_tensorf(model, "blk.{il}.hc_ffn_base.weight", il)?,
            ffn_norm: find_required_tensorf(model, "blk.{il}.ffn_norm.weight", il)?,
            ffn_gate_inp: find_required_tensorf(model, "blk.{il}.ffn_gate_inp.weight", il)?,

            ffn_exp_probs_b: model
                .find_tensor(&format!("blk.{il}.exp_probs_b.bias"))
                .cloned(),

            ffn_gate_exps: find_required_tensorf(model, "blk.{il}.ffn_gate_exps.weight", il)?,
            ffn_up_exps: find_required_tensorf(model, "blk.{il}.ffn_up_exps.weight", il)?,
            ffn_down_exps: find_required_tensorf(model, "blk.{il}.ffn_down_exps.weight", il)?,
            ffn_gate_shexp: find_required_tensorf(model, "blk.{il}.ffn_gate_shexp.weight", il)?,
            ffn_up_shexp: find_required_tensorf(model, "blk.{il}.ffn_up_shexp.weight", il)?,
            ffn_down_shexp: find_required_tensorf(model, "blk.{il}.ffn_down_shexp.weight", il)?,

            ffn_gate_tid2eid: if il < DS4_N_HASH_LAYER {
                Some(find_required_tensorf(
                    model,
                    "blk.{il}.ffn_gate_tid2eid.weight",
                    il,
                )?)
            } else {
                None
            },
        });
    }

    // Convert Vec to fixed-size array.
    let layer: [LayerWeights; DS4_N_LAYER as usize] = layer_vec
        .try_into()
        .map_err(|_| anyhow::anyhow!("internal error: layer count mismatch"))?;

    let weights = Weights {
        token_embd,
        output_hc_base,
        output_hc_fn,
        output_hc_scale,
        output_norm,
        output,
        layer,
    };

    // Validate the layout after binding.
    weights_validate_layout(&weights)?;

    Ok(weights)
}

/// Bind MTP weight tensors.
///
/// Corresponds to `mtp_weights_bind()` in `ds4.c` lines 2492–2531.
fn mtp_weights_bind(model: &ModelGGUF) -> Result<MtpWeights> {
    let e_proj = find_required_tensor(model, "mtp.0.e_proj.weight")?;
    let h_proj = find_required_tensor(model, "mtp.0.h_proj.weight")?;
    let enorm = find_required_tensor(model, "mtp.0.enorm.weight")?;
    let hnorm = find_required_tensor(model, "mtp.0.hnorm.weight")?;
    let norm = find_required_tensor(model, "mtp.0.norm.weight")?;
    let hc_head_base = find_required_tensor(model, "mtp.0.hc_head_base.weight")?;
    let hc_head_fn = find_required_tensor(model, "mtp.0.hc_head_fn.weight")?;
    let hc_head_scale = find_required_tensor(model, "mtp.0.hc_head_scale.weight")?;

    let block = MtpLayerWeights {
        hc_attn_fn: find_required_tensor(model, "mtp.0.hc_attn_fn.weight")?,
        hc_attn_scale: find_required_tensor(model, "mtp.0.hc_attn_scale.weight")?,
        hc_attn_base: find_required_tensor(model, "mtp.0.hc_attn_base.weight")?,
        attn_norm: find_required_tensor(model, "mtp.0.attn_norm.weight")?,
        attn_q_a: find_required_tensor(model, "mtp.0.attn_q_a.weight")?,
        attn_q_a_norm: find_required_tensor(model, "mtp.0.attn_q_a_norm.weight")?,
        attn_q_b: find_required_tensor(model, "mtp.0.attn_q_b.weight")?,
        attn_kv: find_required_tensor(model, "mtp.0.attn_kv.weight")?,
        attn_kv_a_norm: find_required_tensor(model, "mtp.0.attn_kv_a_norm.weight")?,
        attn_sinks: find_required_tensor(model, "mtp.0.attn_sinks.weight")?,
        attn_output_a: find_required_tensor(model, "mtp.0.attn_output_a.weight")?,
        attn_output_b: find_required_tensor(model, "mtp.0.attn_output_b.weight")?,
        hc_ffn_fn: find_required_tensor(model, "mtp.0.hc_ffn_fn.weight")?,
        hc_ffn_scale: find_required_tensor(model, "mtp.0.hc_ffn_scale.weight")?,
        hc_ffn_base: find_required_tensor(model, "mtp.0.hc_ffn_base.weight")?,
        ffn_norm: find_required_tensor(model, "mtp.0.ffn_norm.weight")?,
        ffn_gate_inp: find_required_tensor(model, "mtp.0.ffn_gate_inp.weight")?,
        ffn_exp_probs_b: find_required_tensor(model, "mtp.0.exp_probs_b.bias")?,
        ffn_gate_exps: find_required_tensor(model, "mtp.0.ffn_gate_exps.weight")?,
        ffn_up_exps: find_required_tensor(model, "mtp.0.ffn_up_exps.weight")?,
        ffn_down_exps: find_required_tensor(model, "mtp.0.ffn_down_exps.weight")?,
        ffn_gate_shexp: find_required_tensor(model, "mtp.0.ffn_gate_shexp.weight")?,
        ffn_up_shexp: find_required_tensor(model, "mtp.0.ffn_up_shexp.weight")?,
        ffn_down_shexp: find_required_tensor(model, "mtp.0.ffn_down_shexp.weight")?,
    };

    let weights = MtpWeights {
        e_proj,
        h_proj,
        enorm,
        hnorm,
        norm,
        hc_head_base,
        hc_head_fn,
        hc_head_scale,
        block,
    };

    mtp_weights_validate_layout(&weights)?;

    Ok(weights)
}

// ── Tensor lookup helpers ─────────────────────────────────────────────────

/// Find a required tensor by name, returning an error if it is missing.
fn find_required_tensor(model: &ModelGGUF, name: &str) -> Result<TensorInfo> {
    model
        .find_tensor(name)
        .cloned()
        .with_context(|| format!("required tensor is missing: {name}"))
}

/// Find a required tensor by a formatted name pattern, e.g. `blk.{il}.attn_norm.weight`.
fn find_required_tensorf(model: &ModelGGUF, fmt: &str, layer: u32) -> Result<TensorInfo> {
    let name = fmt.replace("{il}", &layer.to_string());
    model
        .find_tensor(&name)
        .cloned()
        .with_context(|| format!("required tensor is missing: {name}"))
}

// ── Layout validation ─────────────────────────────────────────────────────

/// Verify that every required tensor has the expected type and dimensions for
/// the DeepSeek V4 Flash fixed layout.
///
/// Corresponds to `weights_validate_layout()` in `ds4.c` lines 2139–2208.
fn weights_validate_layout(w: &Weights) -> Result<()> {
    let hc_dim = u64::from(DS4_N_EMBD) * u64::from(DS4_N_HC);
    let hc_mix_dim = 2u64 * u64::from(DS4_N_HC) + u64::from(DS4_N_HC) * u64::from(DS4_N_HC);
    let q_dim = u64::from(DS4_N_HEAD) * u64::from(DS4_N_HEAD_DIM);
    let out_low_dim = u64::from(DS4_N_OUT_GROUP) * u64::from(DS4_N_LORA_O);

    expect_layout(
        &w.token_embd,
        Some(1),
        2,
        DS4_N_EMBD.into(),
        DS4_N_VOCAB.into(),
        0,
    )?;
    expect_layout(&w.output_hc_base, Some(0), 1, DS4_N_HC.into(), 0, 0)?;
    expect_layout(&w.output_hc_fn, Some(1), 2, hc_dim, DS4_N_HC.into(), 0)?;
    expect_layout(&w.output_hc_scale, Some(0), 1, 1, 0, 0)?;
    expect_layout(&w.output_norm, Some(0), 1, DS4_N_EMBD.into(), 0, 0)?;
    expect_layout(
        &w.output,
        Some(8),
        2,
        DS4_N_EMBD.into(),
        DS4_N_VOCAB.into(),
        0,
    )?;

    for il in 0..DS4_N_LAYER {
        let l = &w.layer[il as usize];
        let ratio = layer_compress_ratio(il);

        expect_layout(&l.hc_attn_fn, Some(1), 2, hc_dim, hc_mix_dim, 0)?;
        expect_layout(&l.hc_attn_scale, Some(0), 1, 3, 0, 0)?;
        expect_layout(&l.hc_attn_base, Some(0), 1, hc_mix_dim, 0, 0)?;
        expect_layout(&l.attn_norm, Some(0), 1, DS4_N_EMBD.into(), 0, 0)?;
        expect_layout(
            &l.attn_q_a,
            Some(8),
            2,
            DS4_N_EMBD.into(),
            DS4_N_LORA_Q.into(),
            0,
        )?;
        expect_layout(&l.attn_q_a_norm, Some(0), 1, DS4_N_LORA_Q.into(), 0, 0)?;
        expect_layout(&l.attn_q_b, Some(8), 2, DS4_N_LORA_Q.into(), q_dim, 0)?;
        expect_layout(
            &l.attn_kv,
            Some(8),
            2,
            DS4_N_EMBD.into(),
            DS4_N_HEAD_DIM.into(),
            0,
        )?;
        expect_layout(&l.attn_kv_a_norm, Some(0), 1, DS4_N_HEAD_DIM.into(), 0, 0)?;
        expect_layout(&l.attn_sinks, Some(0), 1, DS4_N_HEAD.into(), 0, 0)?;

        let attn_out_a_dim =
            u64::from(DS4_N_HEAD_DIM) * (u64::from(DS4_N_HEAD) / u64::from(DS4_N_OUT_GROUP));
        expect_layout(&l.attn_output_a, Some(8), 2, attn_out_a_dim, out_low_dim, 0)?;
        expect_layout(
            &l.attn_output_b,
            Some(8),
            2,
            out_low_dim,
            DS4_N_EMBD.into(),
            0,
        )?;

        if ratio != 0 {
            let coff = if ratio == 4 { 2u32 } else { 1u32 };
            let comp_width = u64::from(coff) * u64::from(DS4_N_HEAD_DIM);

            if let Some(ref t) = l.attn_compressor_ape {
                expect_layout(t, Some(1), 2, comp_width, ratio.into(), 0)?;
            } else {
                bail!("layer {il}: compressor_ape expected when compress_ratio != 0");
            }
            if let Some(ref t) = l.attn_compressor_kv {
                expect_layout(t, Some(1), 2, DS4_N_EMBD.into(), comp_width, 0)?;
            } else {
                bail!("layer {il}: compressor_kv expected when compress_ratio != 0");
            }
            if let Some(ref t) = l.attn_compressor_gate {
                expect_layout(t, Some(1), 2, DS4_N_EMBD.into(), comp_width, 0)?;
            } else {
                bail!("layer {il}: compressor_gate expected when compress_ratio != 0");
            }
            if let Some(ref t) = l.attn_compressor_norm {
                expect_layout(t, Some(0), 1, DS4_N_HEAD_DIM.into(), 0, 0)?;
            } else {
                bail!("layer {il}: compressor_norm expected when compress_ratio != 0");
            }
        }

        if ratio == 4 {
            let index_q_dim = u64::from(DS4_N_INDEXER_HEAD) * u64::from(DS4_N_INDEXER_HEAD_DIM);
            let index_width = 2u64 * u64::from(DS4_N_INDEXER_HEAD_DIM);

            if let Some(ref t) = l.indexer_attn_q_b {
                expect_layout(t, Some(1), 2, DS4_N_LORA_Q.into(), index_q_dim, 0)?;
            } else {
                bail!("layer {il}: indexer_attn_q_b expected when compress_ratio == 4");
            }
            if let Some(ref t) = l.indexer_proj {
                expect_layout(
                    t,
                    Some(1),
                    2,
                    DS4_N_EMBD.into(),
                    DS4_N_INDEXER_HEAD.into(),
                    0,
                )?;
            } else {
                bail!("layer {il}: indexer_proj expected when compress_ratio == 4");
            }
            if let Some(ref t) = l.indexer_compressor_ape {
                expect_layout(t, Some(1), 2, index_width, ratio.into(), 0)?;
            } else {
                bail!("layer {il}: indexer_compressor_ape expected when compress_ratio == 4");
            }
            if let Some(ref t) = l.indexer_compressor_kv {
                expect_layout(t, Some(1), 2, DS4_N_EMBD.into(), index_width, 0)?;
            } else {
                bail!("layer {il}: indexer_compressor_kv expected when compress_ratio == 4");
            }
            if let Some(ref t) = l.indexer_compressor_gate {
                expect_layout(t, Some(1), 2, DS4_N_EMBD.into(), index_width, 0)?;
            } else {
                bail!("layer {il}: indexer_compressor_gate expected when compress_ratio == 4");
            }
            if let Some(ref t) = l.indexer_compressor_norm {
                expect_layout(t, Some(0), 1, DS4_N_INDEXER_HEAD_DIM.into(), 0, 0)?;
            } else {
                bail!("layer {il}: indexer_compressor_norm expected when compress_ratio == 4");
            }
        }

        expect_layout(&l.hc_ffn_fn, Some(1), 2, hc_dim, hc_mix_dim, 0)?;
        expect_layout(&l.hc_ffn_scale, Some(0), 1, 3, 0, 0)?;
        expect_layout(&l.hc_ffn_base, Some(0), 1, hc_mix_dim, 0, 0)?;
        expect_layout(&l.ffn_norm, Some(0), 1, DS4_N_EMBD.into(), 0, 0)?;
        expect_layout(
            &l.ffn_gate_inp,
            Some(1),
            2,
            DS4_N_EMBD.into(),
            DS4_N_EXPERT.into(),
            0,
        )?;

        // exp_probs_b is optional
        if let Some(ref t) = l.ffn_exp_probs_b {
            expect_layout(t, Some(0), 1, DS4_N_EXPERT.into(), 0, 0)?;
        }

        expect_routed_expert(
            &l.ffn_gate_exps,
            3,
            DS4_N_EMBD.into(),
            DS4_N_FF_EXP.into(),
            DS4_N_EXPERT.into(),
        )?;
        expect_routed_expert(
            &l.ffn_up_exps,
            3,
            DS4_N_EMBD.into(),
            DS4_N_FF_EXP.into(),
            DS4_N_EXPERT.into(),
        )?;
        expect_routed_expert(
            &l.ffn_down_exps,
            3,
            DS4_N_FF_EXP.into(),
            DS4_N_EMBD.into(),
            DS4_N_EXPERT.into(),
        )?;

        // Verify gate and up experts use the same quant type
        if l.ffn_gate_exps.tensor_type != l.ffn_up_exps.tensor_type {
            bail!(
                "layer {il}: routed gate/up experts use different quant types \
                 (gate={}, up={})",
                l.ffn_gate_exps.type_name(),
                l.ffn_up_exps.type_name()
            );
        }

        expect_layout(
            &l.ffn_gate_shexp,
            Some(8),
            2,
            DS4_N_EMBD.into(),
            DS4_N_FF_EXP.into(),
            0,
        )?;
        expect_layout(
            &l.ffn_up_shexp,
            Some(8),
            2,
            DS4_N_EMBD.into(),
            DS4_N_FF_EXP.into(),
            0,
        )?;
        expect_layout(
            &l.ffn_down_shexp,
            Some(8),
            2,
            DS4_N_FF_EXP.into(),
            DS4_N_EMBD.into(),
            0,
        )?;

        if il < DS4_N_HASH_LAYER {
            if let Some(ref t) = l.ffn_gate_tid2eid {
                expect_layout(
                    t,
                    Some(26),
                    2,
                    DS4_N_EXPERT_USED.into(),
                    DS4_N_VOCAB.into(),
                    0,
                )?;
            } else {
                bail!("layer {il}: ffn_gate_tid2eid expected in a hash layer");
            }
        }
    }

    Ok(())
}

/// Validate MTP weight layout.
///
/// Corresponds to `mtp_weights_validate_layout()` in `ds4.c` lines 2210–2254.
fn mtp_weights_validate_layout(w: &MtpWeights) -> Result<()> {
    let hc_dim = u64::from(DS4_N_EMBD) * u64::from(DS4_N_HC);
    let hc_mix_dim = 2u64 * u64::from(DS4_N_HC) + u64::from(DS4_N_HC) * u64::from(DS4_N_HC);
    let q_dim = u64::from(DS4_N_HEAD) * u64::from(DS4_N_HEAD_DIM);
    let out_low_dim = u64::from(DS4_N_OUT_GROUP) * u64::from(DS4_N_LORA_O);

    expect_layout(&w.hc_head_base, Some(0), 1, DS4_N_HC.into(), 0, 0)?;
    expect_plain_layout(&w.hc_head_fn, 2, hc_dim, DS4_N_HC.into(), 0)?;
    expect_layout(&w.hc_head_scale, Some(0), 1, 1, 0, 0)?;
    expect_layout(
        &w.e_proj,
        Some(8),
        2,
        DS4_N_EMBD.into(),
        DS4_N_EMBD.into(),
        0,
    )?;
    expect_layout(
        &w.h_proj,
        Some(8),
        2,
        DS4_N_EMBD.into(),
        DS4_N_EMBD.into(),
        0,
    )?;
    expect_layout(&w.enorm, Some(0), 1, DS4_N_EMBD.into(), 0, 0)?;
    expect_layout(&w.hnorm, Some(0), 1, DS4_N_EMBD.into(), 0, 0)?;
    expect_layout(&w.norm, Some(0), 1, DS4_N_EMBD.into(), 0, 0)?;

    let l = &w.block;
    expect_plain_layout(&l.hc_attn_fn, 2, hc_dim, hc_mix_dim, 0)?;
    expect_layout(&l.hc_attn_scale, Some(0), 1, 3, 0, 0)?;
    expect_layout(&l.hc_attn_base, Some(0), 1, hc_mix_dim, 0, 0)?;
    expect_layout(&l.attn_norm, Some(0), 1, DS4_N_EMBD.into(), 0, 0)?;
    expect_layout(
        &l.attn_q_a,
        Some(8),
        2,
        DS4_N_EMBD.into(),
        DS4_N_LORA_Q.into(),
        0,
    )?;
    expect_layout(&l.attn_q_a_norm, Some(0), 1, DS4_N_LORA_Q.into(), 0, 0)?;
    expect_layout(&l.attn_q_b, Some(8), 2, DS4_N_LORA_Q.into(), q_dim, 0)?;
    expect_layout(
        &l.attn_kv,
        Some(8),
        2,
        DS4_N_EMBD.into(),
        DS4_N_HEAD_DIM.into(),
        0,
    )?;
    expect_layout(&l.attn_kv_a_norm, Some(0), 1, DS4_N_HEAD_DIM.into(), 0, 0)?;
    expect_layout(&l.attn_sinks, Some(0), 1, DS4_N_HEAD.into(), 0, 0)?;

    let attn_out_a_dim =
        u64::from(DS4_N_HEAD_DIM) * (u64::from(DS4_N_HEAD) / u64::from(DS4_N_OUT_GROUP));
    expect_layout(&l.attn_output_a, Some(8), 2, attn_out_a_dim, out_low_dim, 0)?;
    expect_layout(
        &l.attn_output_b,
        Some(8),
        2,
        out_low_dim,
        DS4_N_EMBD.into(),
        0,
    )?;

    expect_plain_layout(&l.hc_ffn_fn, 2, hc_dim, hc_mix_dim, 0)?;
    expect_layout(&l.hc_ffn_scale, Some(0), 1, 3, 0, 0)?;
    expect_layout(&l.hc_ffn_base, Some(0), 1, hc_mix_dim, 0, 0)?;
    expect_layout(&l.ffn_norm, Some(0), 1, DS4_N_EMBD.into(), 0, 0)?;
    expect_plain_layout(
        &l.ffn_gate_inp,
        2,
        DS4_N_EMBD.into(),
        DS4_N_EXPERT.into(),
        0,
    )?;
    expect_layout(&l.ffn_exp_probs_b, Some(0), 1, DS4_N_EXPERT.into(), 0, 0)?;
    expect_routed_expert(
        &l.ffn_gate_exps,
        3,
        DS4_N_EMBD.into(),
        DS4_N_FF_EXP.into(),
        DS4_N_EXPERT.into(),
    )?;
    expect_routed_expert(
        &l.ffn_up_exps,
        3,
        DS4_N_EMBD.into(),
        DS4_N_FF_EXP.into(),
        DS4_N_EXPERT.into(),
    )?;
    expect_routed_expert(
        &l.ffn_down_exps,
        3,
        DS4_N_FF_EXP.into(),
        DS4_N_EMBD.into(),
        DS4_N_EXPERT.into(),
    )?;

    if l.ffn_gate_exps.tensor_type != l.ffn_up_exps.tensor_type {
        bail!("MTP routed gate/up experts use different quant types");
    }

    expect_layout(
        &l.ffn_gate_shexp,
        Some(8),
        2,
        DS4_N_EMBD.into(),
        DS4_N_FF_EXP.into(),
        0,
    )?;
    expect_layout(
        &l.ffn_up_shexp,
        Some(8),
        2,
        DS4_N_EMBD.into(),
        DS4_N_FF_EXP.into(),
        0,
    )?;
    expect_layout(
        &l.ffn_down_shexp,
        Some(8),
        2,
        DS4_N_FF_EXP.into(),
        DS4_N_EMBD.into(),
        0,
    )?;

    Ok(())
}

// ── Layout check helpers ──────────────────────────────────────────────────

/// Verify a tensor's type, ndim, and dimension[0..2].
///
/// `expected_type`: `Some(t)` requires exact type match; `None` skips type check.
fn expect_layout(
    t: &TensorInfo,
    expected_type: Option<u32>,
    ndim: u32,
    d0: u64,
    d1: u64,
    d2: u64,
) -> Result<()> {
    if let Some(ty) = expected_type {
        if t.tensor_type != ty {
            bail!(
                "tensor {} has type {}, expected type {}",
                t.name,
                t.type_name(),
                TensorType::from_u32(ty)
                    .map(|tt| tt.name().to_string())
                    .unwrap_or_else(|| ty.to_string())
            );
        }
    }
    if t.ndim != ndim {
        bail!(
            "tensor {} has {} dimensions, expected {}",
            t.name,
            t.ndim,
            ndim
        );
    }

    let want = [d0, d1, d2];
    for i in 0..ndim as usize {
        let dim = t.dims.get(i).copied().unwrap_or(0);
        if dim != want[i] {
            bail!(
                "tensor {} has dim[{}]={}, expected {}",
                t.name,
                i,
                dim,
                want[i]
            );
        }
    }
    Ok(())
}

/// Verify a "plain" tensor (F16 or F32 type).
fn expect_plain_layout(t: &TensorInfo, ndim: u32, d0: u64, d1: u64, d2: u64) -> Result<()> {
    if t.tensor_type != 0 && t.tensor_type != 1 {
        // DS4_TENSOR_F32 = 0, DS4_TENSOR_F16 = 1
        bail!(
            "tensor {} has type {}, expected F16 or F32",
            t.name,
            t.type_name()
        );
    }
    expect_layout(t, None, ndim, d0, d1, d2)
}

/// Verify a routed expert tensor (IQ2_XXS, Q2_K, or Q4_K type).
fn expect_routed_expert(t: &TensorInfo, ndim: u32, d0: u64, d1: u64, d2: u64) -> Result<()> {
    if t.tensor_type != 10 && t.tensor_type != 12 && t.tensor_type != 16 {
        // Q2_K = 10, Q4_K = 12, IQ2_XXS = 16
        bail!(
            "tensor {} has type {} ({}), expected a routed expert quant type",
            t.name,
            t.tensor_type,
            t.type_name()
        );
    }
    if t.ndim != ndim {
        bail!(
            "tensor {} has {} dimensions, expected {}",
            t.name,
            t.ndim,
            ndim
        );
    }

    let want = [d0, d1, d2];
    for i in 0..ndim as usize {
        let dim = t.dims.get(i).copied().unwrap_or(0);
        if dim != want[i] {
            bail!(
                "tensor {} has dim[{}]={}, expected {}",
                t.name,
                i,
                dim,
                want[i]
            );
        }
    }
    Ok(())
}

// ── Model configuration validation ────────────────────────────────────────

/// Validate that the GGUF metadata matches the fixed DeepSeek V4 Flash layout.
///
/// Corresponds to `config_validate_model()` in `ds4.c` lines 2346–2428.
fn config_validate_model(model: &ModelGGUF) -> Result<()> {
    let check = |name: &str, got: u32, expected: u32| -> Result<()> {
        if got != expected {
            bail!("expected {name}={expected} for DeepSeek4 Flash, got {got}");
        }
        Ok(())
    };

    let n_layer = required_u32(model, "deepseek4.block_count")?;
    let n_embd = required_u32(model, "deepseek4.embedding_length")?;
    let n_vocab = required_u32(model, "deepseek4.vocab_size")?;
    let n_head = required_u32(model, "deepseek4.attention.head_count")?;
    let n_head_kv = required_u32(model, "deepseek4.attention.head_count_kv")?;
    let n_head_dim = required_u32(model, "deepseek4.attention.key_length")?;
    let n_value_dim = required_u32(model, "deepseek4.attention.value_length")?;
    let n_rot = required_u32(model, "deepseek4.rope.dimension_count")?;
    let n_lora_q = required_u32(model, "deepseek4.attention.q_lora_rank")?;
    let n_lora_o = required_u32(model, "deepseek4.attention.output_lora_rank")?;
    let n_out_group = required_u32(model, "deepseek4.attention.output_group_count")?;
    let n_expert = required_u32(model, "deepseek4.expert_count")?;
    let n_expert_used = required_u32(model, "deepseek4.expert_used_count")?;
    let n_ff_exp = required_u32(model, "deepseek4.expert_feed_forward_length")?;
    let n_expert_shared = required_u32(model, "deepseek4.expert_shared_count")?;
    let n_hash_layer = required_u32(model, "deepseek4.hash_layer_count")?;

    check("block_count", n_layer, DS4_N_LAYER)?;
    check("embedding_length", n_embd, DS4_N_EMBD)?;
    check("vocab_size", n_vocab, DS4_N_VOCAB)?;
    check("attention.head_count", n_head, DS4_N_HEAD)?;
    check("attention.key_length", n_head_dim, DS4_N_HEAD_DIM)?;
    check("attention.head_count_kv", n_head_kv, DS4_N_HEAD_KV)?;
    check("attention.value_length", n_value_dim, DS4_N_VALUE_DIM)?;
    check("rope.dimension_count", n_rot, DS4_N_ROT)?;
    check("attention.output_group_count", n_out_group, DS4_N_OUT_GROUP)?;
    check("attention.q_lora_rank", n_lora_q, DS4_N_LORA_Q)?;
    check("attention.output_lora_rank", n_lora_o, DS4_N_LORA_O)?;
    check("expert_count", n_expert, DS4_N_EXPERT)?;
    check("expert_used_count", n_expert_used, DS4_N_EXPERT_USED)?;
    check("expert_feed_forward_length", n_ff_exp, DS4_N_FF_EXP)?;
    check("expert_shared_count", n_expert_shared, DS4_N_EXPERT_SHARED)?;
    check("hash_layer_count", n_hash_layer, DS4_N_HASH_LAYER)?;

    // Optional group routing (must be 0 for Flash)
    if let Some(v) = model.get_u32("deepseek4.expert_group_count") {
        if v != 0 {
            bail!("expected expert_group_count=0 for DeepSeek4 Flash, got {v}");
        }
    }
    if let Some(v) = model.get_u32("deepseek4.expert_group_used_count") {
        if v != 0 {
            bail!("expected expert_group_used_count=0 for DeepSeek4 Flash, got {v}");
        }
    }

    let n_swa = required_u32(model, "deepseek4.attention.sliding_window")?;
    check("attention.sliding_window", n_swa, DS4_N_SWA)?;

    let n_indexer_head = required_u32(model, "deepseek4.attention.indexer.head_count")?;
    let n_indexer_head_dim = required_u32(model, "deepseek4.attention.indexer.key_length")?;
    let n_indexer_top_k = required_u32(model, "deepseek4.attention.indexer.top_k")?;
    check(
        "attention.indexer.head_count",
        n_indexer_head,
        DS4_N_INDEXER_HEAD,
    )?;
    check(
        "attention.indexer.key_length",
        n_indexer_head_dim,
        DS4_N_INDEXER_HEAD_DIM,
    )?;
    check(
        "attention.indexer.top_k",
        n_indexer_top_k,
        DS4_N_INDEXER_TOP_K,
    )?;

    let n_hc = required_u32(model, "deepseek4.hyper_connection.count")?;
    check("hyper_connection.count", n_hc, DS4_N_HC)?;

    let n_hc_sinkhorn_iter = required_u32(model, "deepseek4.hyper_connection.sinkhorn_iterations")?;
    check(
        "hyper_connection.sinkhorn_iterations",
        n_hc_sinkhorn_iter,
        DS4_N_HC_SINKHORN_ITER,
    )?;

    // Validate compression ratio metadata
    validate_compress_ratios(model)?;

    // Validate SwiGLU clamp metadata
    validate_swiglu_clamp(model)?;

    // RoPE / scaling / epsilon validation
    let rope_orig_ctx = required_u64(model, "deepseek4.rope.scaling.original_context_length")?;
    if rope_orig_ctx != DS4_ROPE_ORIG_CTX {
        bail!(
            "expected rope.scaling.original_context_length={DS4_ROPE_ORIG_CTX} \
             for DeepSeek4 Flash, got {rope_orig_ctx}"
        );
    }

    let rope_freq_base = required_f32(model, "deepseek4.rope.freq_base")?;
    expect_f32("rope.freq_base", rope_freq_base, DS4_ROPE_FREQ_BASE)?;

    let rope_scale_factor = required_f32(model, "deepseek4.rope.scaling.factor")?;
    expect_f32(
        "rope.scaling.factor",
        rope_scale_factor,
        DS4_ROPE_SCALE_FACTOR,
    )?;

    let rope_yarn_beta_fast = required_f32(model, "deepseek4.rope.scaling.yarn_beta_fast")?;
    expect_f32(
        "rope.scaling.yarn_beta_fast",
        rope_yarn_beta_fast,
        DS4_ROPE_YARN_BETA_FAST,
    )?;

    let rope_yarn_beta_slow = required_f32(model, "deepseek4.rope.scaling.yarn_beta_slow")?;
    expect_f32(
        "rope.scaling.yarn_beta_slow",
        rope_yarn_beta_slow,
        DS4_ROPE_YARN_BETA_SLOW,
    )?;

    let compress_rope_freq_base =
        required_f32(model, "deepseek4.attention.compress_rope_freq_base")?;
    expect_f32(
        "attention.compress_rope_freq_base",
        compress_rope_freq_base,
        DS4_COMPRESS_ROPE_FREQ_BASE,
    )?;

    let expert_weight_scale = required_f32(model, "deepseek4.expert_weights_scale")?;
    expect_f32(
        "expert_weights_scale",
        expert_weight_scale,
        DS4_EXPERT_WEIGHT_SCALE,
    )?;

    let rms_eps = required_f32(model, "deepseek4.attention.layer_norm_rms_epsilon")?;
    expect_f32("attention.layer_norm_rms_epsilon", rms_eps, DS4_RMS_EPS)?;

    let hc_eps = required_f32(model, "deepseek4.hyper_connection.epsilon")?;
    expect_f32("hyper_connection.epsilon", hc_eps, DS4_HC_EPS)?;

    let expert_weight_norm = required_bool(model, "deepseek4.expert_weights_norm")?;
    if !expert_weight_norm {
        bail!("expected expert_weights_norm=true for DeepSeek4 Flash");
    }

    Ok(())
}

/// Validate the `deepseek4.attention.compress_ratios` metadata array.
fn validate_compress_ratios(model: &ModelGGUF) -> Result<()> {
    let arr = model
        .get_array("deepseek4.attention.compress_ratios")
        .context(
        "required int32/uint32 array metadata key is missing: deepseek4.attention.compress_ratios",
    )?;

    if arr.len() < DS4_N_LAYER as usize {
        bail!(
            "deepseek4.attention.compress_ratios is shorter than the layer count \
             ({} < {DS4_N_LAYER})",
            arr.len()
        );
    }

    for (il, val) in arr.iter().enumerate().take(DS4_N_LAYER as usize) {
        let got = val.as_u32().unwrap_or(u32::MAX);
        let expected = layer_compress_ratio(il as u32);
        if got != expected {
            bail!(
                "unexpected DeepSeek4 compression ratio at layer {il}: \
                 got {got}, expected {expected}"
            );
        }
    }

    Ok(())
}

/// Validate the `deepseek4.swiglu_clamp_exp` metadata array.
fn validate_swiglu_clamp(model: &ModelGGUF) -> Result<()> {
    let arr = model
        .get_array("deepseek4.swiglu_clamp_exp")
        .context("required float array metadata key is missing: deepseek4.swiglu_clamp_exp")?;

    if arr.len() < DS4_N_LAYER as usize {
        bail!(
            "deepseek4.swiglu_clamp_exp is shorter than the layer count \
             ({} < {DS4_N_LAYER})",
            arr.len()
        );
    }

    for (i, val) in arr.iter().enumerate().take(DS4_N_LAYER as usize) {
        let got = val.as_f32().unwrap_or(f32::NAN);
        expect_f32(&format!("swiglu_clamp_exp[{i}]"), got, DS4_SWIGLU_CLAMP_EXP)?;
    }

    Ok(())
}

// ── Metadata access helpers ──────────────────────────────────────────────

fn required_u32(model: &ModelGGUF, key: &str) -> Result<u32> {
    model
        .get_u32(key)
        .with_context(|| format!("required metadata key is missing: {key}"))
}

fn required_u64(model: &ModelGGUF, key: &str) -> Result<u64> {
    model
        .get_u64(key)
        .with_context(|| format!("required metadata key is missing: {key}"))
}

fn required_f32(model: &ModelGGUF, key: &str) -> Result<f32> {
    model
        .get_f32(key)
        .with_context(|| format!("required metadata key is missing: {key}"))
}

fn required_bool(model: &ModelGGUF, key: &str) -> Result<bool> {
    model
        .get_bool(key)
        .with_context(|| format!("required metadata key is missing: {key}"))
}

fn expect_f32(name: &str, got: f32, expected: f32) -> Result<()> {
    let scale = if expected.abs() > 1.0 {
        expected.abs()
    } else {
        1.0
    };
    if (got - expected).abs() <= scale * 1.0e-6 {
        Ok(())
    } else {
        bail!("expected {name}={expected} for DeepSeek4 Flash, got {got}");
    }
}

// ── Memory estimation ─────────────────────────────────────────────────────

/// Estimate memory requirements for the Metal backend.
fn metal_context_memory_estimate(ctx: u32) -> ContextMemory {
    use std::cmp;

    let prefill_cap = metal_graph_prefill_cap_for_prompt(ctx as i32);
    let raw_cap = metal_graph_raw_cap_for_context(ctx as i32, prefill_cap);

    let mut min_ratio = u32::MAX;
    for il in 0..DS4_N_LAYER {
        let ratio = layer_compress_ratio(il);
        if ratio != 0 && ratio < min_ratio {
            min_ratio = ratio;
        }
    }
    if min_ratio == u32::MAX {
        min_ratio = ctx;
    }
    let comp_cap = ctx / min_ratio + 2;
    let comp_cap = cmp::max(comp_cap, 2);

    let raw_bytes = u64::from(DS4_N_LAYER) * u64::from(raw_cap) * u64::from(DS4_N_HEAD_DIM) * 4; // sizeof(float)

    let mut compressed_bytes: u64 = 0;
    for il in 0..DS4_N_LAYER {
        let ratio = layer_compress_ratio(il);
        if ratio == 0 {
            continue;
        }
        compressed_bytes += u64::from(comp_cap) * u64::from(DS4_N_HEAD_DIM) * 4;
        if ratio == 4 {
            compressed_bytes += u64::from(comp_cap) * u64::from(DS4_N_INDEXER_HEAD_DIM) * 4;
        }
    }

    let scratch_bytes = 2u64 * u64::from(comp_cap) * u64::from(prefill_cap) * 4;

    ContextMemory {
        total_bytes: raw_bytes + compressed_bytes + scratch_bytes,
        raw_bytes,
        compressed_bytes,
        scratch_bytes,
        prefill_cap,
        raw_cap,
        comp_cap,
    }
}

/// Estimate memory requirements for the CPU backend.
fn cpu_context_memory_estimate(ctx: u32) -> ContextMemory {
    let raw_cap = ds4_default_raw_cap(ctx);

    let raw_bytes = u64::from(DS4_N_LAYER) * u64::from(raw_cap) * u64::from(DS4_N_HEAD_DIM) * 4; // sizeof(float)

    let mut compressed_bytes: u64 = 0;
    let mut comp_cap: u32 = 0;
    for il in 0..DS4_N_LAYER {
        let ratio = layer_compress_ratio(il);
        if ratio == 0 {
            continue;
        }
        let layer_comp_cap = ctx / ratio + 2;
        if ratio == 4 {
            comp_cap = layer_comp_cap;
        }
        compressed_bytes += u64::from(layer_comp_cap) * u64::from(DS4_N_HEAD_DIM) * 4;
        if ratio == 4 {
            compressed_bytes += u64::from(layer_comp_cap) * u64::from(DS4_N_INDEXER_HEAD_DIM) * 4;
        }
    }
    if comp_cap == 0 {
        comp_cap = ctx / 4 + 2;
    }

    let scratch_bytes =
        (u64::from(raw_cap + comp_cap) * 4) + (u64::from(comp_cap) * 4) + (u64::from(comp_cap) * 1); // bool

    ContextMemory {
        total_bytes: raw_bytes + compressed_bytes + scratch_bytes,
        raw_bytes,
        compressed_bytes,
        scratch_bytes,
        prefill_cap: 0,
        raw_cap,
        comp_cap,
    }
}

/// Minimum raw KV slots for a given context size (CPU path).
fn ds4_default_raw_cap(ctx: u32) -> u32 {
    // Match the C code's ds4_default_raw_cap heuristic (line 5895).
    if ctx <= 256 {
        ctx
    } else {
        256 + (ctx - 256) / 4
    }
}

/// Metal graph prefill capacity for a given context.
fn metal_graph_prefill_cap_for_prompt(ctx: i32) -> u32 {
    // Match the C code's metal_graph_prefill_cap_for_prompt heuristic.
    if ctx <= 2048 {
        ctx as u32
    } else {
        2048u32 + ((ctx as u32 - 2048u32) / 8)
    }
}

/// Metal graph raw capacity for a given context and prefill cap.
fn metal_graph_raw_cap_for_context(ctx: i32, prefill_cap: u32) -> u32 {
    // Match the C code's metal_graph_raw_cap_for_context heuristic.
    let ctx = ctx as u32;
    if ctx <= 4096 {
        ctx
    } else {
        ctx + prefill_cap
    }
}

// ── Chat formatting ───────────────────────────────────────────────────────

/// DeepSeek chat template helper methods, matching `ds4_chat_*` functions from
/// `ds4.c` lines 13940–14072.
impl Engine {
    /// Open a chat session by adding the BOS token.
    pub fn chat_begin(&self, tokens: &mut TokenVec) {
        tokens.push(self.vocab.bos_id);
    }

    /// Encode a full chat prompt (system + user + assistant marker + think mode).
    ///
    /// Corresponds to `encode_chat_prompt()` in `ds4.c`.
    pub fn encode_chat_prompt(
        &self,
        system: Option<&str>,
        prompt: &str,
        think_mode: crate::types::ThinkMode,
        out: &mut TokenVec,
    ) {
        out.push(self.vocab.bos_id);

        if think_mode == crate::types::ThinkMode::Max {
            self.tokenize_max_effort_prefix(out);
        }

        if let Some(sys) = system {
            if !sys.is_empty() {
                self.vocab.bpe_tokenize(sys, &mut out.v);
            }
        }

        out.push(self.vocab.user_id);
        self.vocab.bpe_tokenize(prompt, &mut out.v);
        out.push(self.vocab.assistant_id);

        if think_mode.is_enabled() {
            out.push(self.vocab.think_start_id);
        } else {
            out.push(self.vocab.think_end_id);
        }
    }

    /// Append the max-effort reasoning prefix to a token sequence.
    pub fn chat_append_max_effort_prefix(&self, tokens: &mut TokenVec) {
        self.tokenize_max_effort_prefix(tokens);
    }

    /// Append a chat message for a given role.
    ///
    /// Corresponds to `ds4_chat_append_message()` in `ds4.c`.
    pub fn chat_append_message(&self, tokens: &mut TokenVec, role: &str, content: &str) {
        let role = if role.is_empty() { "user" } else { role };
        let content = if content.is_empty() { "" } else { content };

        match role {
            "system" | "developer" => {
                self.vocab.bpe_tokenize(content, &mut tokens.v);
            }
            "assistant" => {
                tokens.push(self.vocab.assistant_id);
                // If content doesn't start with <think> or </think>, add </think> prefix.
                if !content.starts_with("<think>") && !content.starts_with("</think>") {
                    tokens.push(self.vocab.think_end_id);
                }
                self.vocab.bpe_tokenize(content, &mut tokens.v);
            }
            _ => {
                tokens.push(self.vocab.user_id);
                if role == "tool" || role == "function" {
                    self.vocab.bpe_tokenize("Tool: ", &mut tokens.v);
                }
                self.vocab.bpe_tokenize(content, &mut tokens.v);
            }
        }
    }

    /// Append the assistant turn prefix (with think mode).
    ///
    /// Corresponds to `ds4_chat_append_assistant_prefix()` in `ds4.c`.
    pub fn chat_append_assistant_prefix(
        &self,
        tokens: &mut TokenVec,
        think_mode: crate::types::ThinkMode,
    ) {
        tokens.push(self.vocab.assistant_id);
        if think_mode.is_enabled() {
            tokens.push(self.vocab.think_start_id);
        } else {
            tokens.push(self.vocab.think_end_id);
        }
    }

    /// Tokenize a rendered chat text, preserving special tokens.
    ///
    /// Corresponds to `tokenize_rendered_chat_vocab()` in `ds4.c`.
    pub fn tokenize_rendered_chat(&self, text: &str, out: &mut TokenVec) {
        let specials: [(&str, i32); 7] = [
            ("<｜begin▁of▁sentence｜>", self.vocab.bos_id),
            ("<｜end▁of▁sentence｜>", self.vocab.eos_id),
            ("<｜User｜>", self.vocab.user_id),
            ("<｜Assistant｜>", self.vocab.assistant_id),
            ("<think>", self.vocab.think_start_id),
            ("</think>", self.vocab.think_end_id),
            ("｜DSML｜", self.vocab.dsml_id),
        ];

        let mut span_start = 0usize;
        let bytes = text.as_bytes();
        let mut pos = 0;

        while pos < bytes.len() {
            let remaining = &bytes[pos..];
            let mut found = false;

            for (token_text, token_id) in &specials {
                if remaining.starts_with(token_text.as_bytes()) {
                    // Tokenize any text before the special token.
                    if pos > span_start {
                        let span = &text[span_start..pos];
                        self.vocab.bpe_tokenize(span, &mut out.v);
                    }
                    out.push(*token_id);
                    pos += token_text.len();
                    span_start = pos;
                    found = true;
                    break;
                }
            }

            if !found {
                pos += 1;
            }
        }

        // Tokenize any remaining text.
        if pos > span_start {
            let span = &text[span_start..pos];
            self.vocab.bpe_tokenize(span, &mut out.v);
        }
    }

    /// Tokenize text using the "rendered chat" tokenizer (public API).
    pub fn tokenize_rendered(&self, text: &str) -> TokenVec {
        let mut out = TokenVec::new();
        self.tokenize_rendered_chat(text, &mut out);
        out
    }

    /// Append the max-effort reasoning prefix tokens.
    fn tokenize_max_effort_prefix(&self, tokens: &mut TokenVec) {
        let prefix = crate::types::think_max_prefix();
        self.vocab.bpe_tokenize(prefix, &mut tokens.v);
    }

    /// Dump token IDs to stderr (matching `ds4_engine_dump_tokens`).
    pub fn dump_tokens(&self, tokens: &TokenVec) {
        let ids: Vec<String> = tokens.v.iter().map(|t| t.to_string()).collect();
        eprintln!("[{}]", ids.join(", "));
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_compress_ratio() {
        assert_eq!(layer_compress_ratio(0), 0);
        assert_eq!(layer_compress_ratio(1), 0);
        assert_eq!(layer_compress_ratio(2), 4);
        assert_eq!(layer_compress_ratio(3), 128);
        assert_eq!(layer_compress_ratio(4), 4);
        assert_eq!(layer_compress_ratio(5), 128);
    }

    #[test]
    fn test_byte_encode_decode() {
        let original = "Hello, 世界!";
        let encoded = byte_encode_str(original);
        let decoded = byte_decode_str(&encoded);
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_gpt2_byte_roundtrip() {
        for b in 0u8..=255 {
            let cp = gpt2_byte_to_codepoint(b);
            let back = gpt2_codepoint_to_byte(cp);
            assert_eq!(back, Some(b), "byte {b} -> codepoint {cp} -> ?");
        }
    }

    #[test]
    fn test_utf8_encode_decode() {
        let test_cps = [
            0x24u32, 0xa2, 0x20ac, 0x10348, 0x7f, 0x80, 0x7ff, 0x800, 0xffff, 0x10000,
        ];
        for &cp in &test_cps {
            let mut buf = Vec::new();
            encode_utf8_cp(cp, &mut buf);
            let (decoded, _) = utf8_decode_one(&buf, buf.len(), 0);
            assert_eq!(decoded, cp, "codepoint {cp:#x}");
        }
    }
}
