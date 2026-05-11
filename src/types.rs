//! Core type definitions for the DS4 inference engine.
//!
//! Maps to concepts from ds4.h and the fixed DeepSeek V4 Flash model shape.

// ── Fixed model shape constants ───────────────────────────────────────────

pub const DS4_N_LAYER: u32 = 43;
pub const DS4_N_EMBD: u32 = 4096;
pub const DS4_N_VOCAB: u32 = 129280;
pub const DS4_N_HEAD: u32 = 64;
pub const DS4_N_HEAD_KV: u32 = 1;
pub const DS4_N_HEAD_DIM: u32 = 512;
pub const DS4_N_VALUE_DIM: u32 = 512;
pub const DS4_N_ROT: u32 = 64;
pub const DS4_N_OUT_GROUP: u32 = 8;
pub const DS4_N_LORA_Q: u32 = 1024;
pub const DS4_N_LORA_O: u32 = 1024;
pub const DS4_N_EXPERT: u32 = 256;
pub const DS4_N_EXPERT_USED: u32 = 6;
pub const DS4_N_EXPERT_SHARED: u32 = 1;
pub const DS4_N_FF_EXP: u32 = 2048;
pub const DS4_N_HASH_LAYER: u32 = 3;
pub const DS4_N_SWA: u32 = 128;
pub const DS4_N_INDEXER_HEAD: u32 = 64;
pub const DS4_N_INDEXER_HEAD_DIM: u32 = 128;
pub const DS4_N_INDEXER_TOP_K: u32 = 512;
pub const DS4_N_HC: u32 = 4;
pub const DS4_N_HC_SINKHORN_ITER: u32 = 20;

pub const DS4_NEG_INF: f32 = -1.0e30;
pub const DS4_POS_INF: f32 = 1.0e30;
pub const DS4_RMS_EPS: f32 = 1.0e-6;
pub const DS4_HC_EPS: f32 = 1.0e-6;
pub const DS4_EXPERT_WEIGHT_SCALE: f32 = 1.5;
pub const DS4_SWIGLU_CLAMP_EXP: f32 = 10.0;
pub const DS4_ROPE_FREQ_BASE: f32 = 10000.0;
pub const DS4_ROPE_SCALE_FACTOR: f32 = 16.0;
pub const DS4_ROPE_YARN_BETA_FAST: f32 = 32.0;
pub const DS4_ROPE_YARN_BETA_SLOW: f32 = 1.0;
pub const DS4_COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
pub const DS4_ROPE_ORIG_CTX: u64 = 65536;

pub const DS4_THINK_MAX_MIN_CONTEXT: u32 = 393216;

pub const QK_K: u32 = 256;

// ── Enums ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Metal,
    Cpu,
}

impl Backend {
    pub fn from_name(name: &str) -> Self {
        match name.to_lowercase().as_str() {
            "cpu" => Backend::Cpu,
            _ => Backend::Metal,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Backend::Metal => "metal",
            Backend::Cpu => "cpu",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkMode {
    None,
    High,
    Max,
}

impl ThinkMode {
    pub fn is_enabled(&self) -> bool {
        !matches!(self, ThinkMode::None)
    }

    pub fn name(&self) -> &'static str {
        match self {
            ThinkMode::None => "none",
            ThinkMode::High => "high",
            ThinkMode::Max => "max",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogType {
    Default,
    Prefill,
    Generation,
    Kvcache,
    Tool,
    Warning,
    Timing,
    Ok,
    Error,
}

// ── Token types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TokenVec {
    pub v: Vec<i32>,
}

impl TokenVec {
    pub fn new() -> Self {
        TokenVec { v: Vec::new() }
    }

    pub fn push(&mut self, token: i32) {
        self.v.push(token);
    }

    pub fn len(&self) -> usize {
        self.v.len()
    }

    pub fn is_empty(&self) -> bool {
        self.v.is_empty()
    }

    pub fn starts_with(&self, prefix: &TokenVec) -> bool {
        if prefix.len() > self.len() {
            return false;
        }
        self.v[..prefix.len()] == prefix.v[..]
    }

    pub fn copy_from(&mut self, other: &TokenVec) {
        self.v = other.v.clone();
    }
}

impl Default for TokenVec {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct TokenScore {
    pub id: i32,
    pub logit: f32,
    pub logprob: f32,
}

// ── Backend options ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EngineOptions {
    pub model_path: String,
    pub mtp_path: Option<String>,
    pub backend: Backend,
    pub n_threads: u32,
    pub mtp_draft_tokens: u32,
    pub mtp_margin: f32,
    pub warm_weights: bool,
    pub quality: bool,
}

impl Default for EngineOptions {
    fn default() -> Self {
        EngineOptions {
            model_path: "ds4flash.gguf".to_string(),
            mtp_path: None,
            backend: Backend::Metal,
            n_threads: 0,
            mtp_draft_tokens: 1,
            mtp_margin: 3.0,
            warm_weights: false,
            quality: false,
        }
    }
}

// ── Memory report ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ContextMemory {
    pub total_bytes: u64,
    pub raw_bytes: u64,
    pub compressed_bytes: u64,
    pub scratch_bytes: u64,
    pub prefill_cap: u32,
    pub raw_cap: u32,
    pub comp_cap: u32,
}

// ── Session rewrite result ────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionRewriteResult {
    Error = -1,
    Ok = 0,
    RebuildNeeded = 1,
}

// ── Reasoning prefix ──────────────────────────────────────────────────────

pub const DS4_REASONING_EFFORT_MAX_PREFIX: &str =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
     You MUST be very thorough in your thinking and comprehensively decompose \
     the problem to resolve the root cause, rigorously stress-testing your \
     logic against all potential paths, edge cases, and adversarial scenarios.\n\
     Explicitly write out your entire deliberation process, documenting every \
     intermediate step, considered alternative, and rejected hypothesis to \
     ensure absolutely no assumption is left unchecked.\n\n";

pub fn think_max_prefix() -> &'static str {
    DS4_REASONING_EFFORT_MAX_PREFIX
}

pub fn think_max_min_context() -> u32 {
    DS4_THINK_MAX_MIN_CONTEXT
}

pub fn think_mode_for_context(mode: ThinkMode, ctx_size: i32) -> ThinkMode {
    if mode == ThinkMode::Max && (ctx_size as u32) < DS4_THINK_MAX_MIN_CONTEXT {
        ThinkMode::High
    } else {
        mode
    }
}

// ── Progress callback ─────────────────────────────────────────────────────

pub type ProgressFn = Box<dyn FnMut(&str, i32, i32) + Send>;

// ── Token emit callbacks ──────────────────────────────────────────────────

pub type TokenEmitFn = Box<dyn FnMut(i32) + Send>;

// ── GGUF tensor type constants ───────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorType {
    F32 = 0,
    F16 = 1,
    Q8_0 = 8,
    Q2K = 10,
    Q4K = 12,
    IQ2Xxs = 16,
    I32 = 26,
}

impl TensorType {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(TensorType::F32),
            1 => Some(TensorType::F16),
            8 => Some(TensorType::Q8_0),
            10 => Some(TensorType::Q2K),
            12 => Some(TensorType::Q4K),
            16 => Some(TensorType::IQ2Xxs),
            26 => Some(TensorType::I32),
            _ => None,
        }
    }

    pub fn block_elems(&self) -> u32 {
        match self {
            TensorType::F32 | TensorType::F16 | TensorType::I32 => 1,
            TensorType::Q8_0 => 32,
            TensorType::Q2K | TensorType::Q4K | TensorType::IQ2Xxs => QK_K,
        }
    }

    pub fn block_bytes(&self) -> u32 {
        match self {
            TensorType::F32 => 4,
            TensorType::F16 => 2,
            TensorType::Q8_0 => 34,
            TensorType::Q2K => 84,
            TensorType::Q4K => 144,
            TensorType::IQ2Xxs => 66,
            TensorType::I32 => 4,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            TensorType::F32 => "f32",
            TensorType::F16 => "f16",
            TensorType::Q8_0 => "q8_0",
            TensorType::Q2K => "q2_k",
            TensorType::Q4K => "q4_k",
            TensorType::IQ2Xxs => "iq2_xxs",
            TensorType::I32 => "i32",
        }
    }
}

// ── Logging ───────────────────────────────────────────────────────────────

pub fn log_color_code(t: LogType) -> &'static str {
    match t {
        LogType::Prefill | LogType::Timing => "\x1b[36m",
        LogType::Generation | LogType::Ok => "\x1b[32m",
        LogType::Kvcache => "\x1b[33m",
        LogType::Tool => "\x1b[90m",
        LogType::Warning => "\x1b[38;5;208m",
        LogType::Error => "\x1b[31m",
        _ => "",
    }
}
