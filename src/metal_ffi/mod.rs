//! FFI bindings to the Objective-C Metal runtime (`ds4_metal.m` / `ds4_metal.h`).
//!
//! The C/ObjC Metal runtime is compiled as a static library by `build.rs` and
//! linked into the final binary.  This module provides:
//!
//! 1. `extern "C"` declarations mirroring every function in `ds4_metal.h`.
//! 2. Safe Rust wrapper functions that wrap each FFI call and convert C return
//!    codes (`0` = success, non-zero = error) into `Result`.
//!
//! # Tensor lifetime
//!
//! `ds4_metal_tensor` is an opaque pointer behind a zero-sized enum; the actual
//! ObjC object is a `DS4MetalTensor` (`id<MTLBuffer>` + offset).  Tensors live
//! on the Metal device and must be freed with `ds4_metal_tensor_free` (or the
//! safe wrapper `MetalTensor::free`).

#![allow(non_camel_case_types, dead_code)]

use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

// ── Error type ────────────────────────────────────────────────────────────

/// Errors originating from the Metal FFI layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetalError {
    /// The underlying C function returned a non-zero error code.
    Code(i32),
    /// Initialisation failed (ds4_metal_init returned false/0).
    InitFailed,
    /// A null pointer was returned where a valid pointer was expected.
    NullPointer(&'static str),
    /// A resource (tensor, buffer, etc.) has already been freed.
    AlreadyFreed,
    /// The Metal backend is not available (not macOS).
    Unavailable,
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetalError::Code(c) => write!(f, "Metal FFI returned error code {}", c),
            MetalError::InitFailed => write!(f, "Metal initialisation failed"),
            MetalError::NullPointer(what) => write!(f, "null pointer returned for {}", what),
            MetalError::AlreadyFreed => write!(f, "Metal resource already freed"),
            MetalError::Unavailable => write!(f, "Metal backend not available on this platform"),
        }
    }
}

impl std::error::Error for MetalError {}

/// Convenience alias for `Result` with the Metal FFI error type.
pub type MetalResult<T> = Result<T, MetalError>;

// ── Global initialisation guard ───────────────────────────────────────────

/// Tracks whether `ds4_metal_init` has been called successfully.
static METAL_INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Returns `true` if the Metal runtime has been initialised.
pub fn is_initialized() -> bool {
    METAL_INITIALIZED.load(Ordering::Relaxed)
}

// ── Opaque tensor type ────────────────────────────────────────────────────

/// Opaque Metal device tensor.
///
/// Pointers to this zero-sized enum are the Rust-side representation of
/// `ds4_metal_tensor *` from C.  The actual ObjC object (`DS4MetalTensor`)
/// wraps an `id<MTLBuffer>` and an offset within it.
pub enum ds4_metal_tensor {}

/// Convenience alias — session code expects `MetalTensor`.
pub type MetalTensor = ds4_metal_tensor;

// ── Helpers ───────────────────────────────────────────────────────────────

/// Convert a C `int` return code (0 = success) into a `MetalResult`.
fn ck(code: i32) -> MetalResult<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(MetalError::Code(code))
    }
}

/// Convert a nullable pointer into a `NonNull`, returning `NullPointer` error
/// if it is null.
fn nonnull<T>(ptr: *mut T, what: &'static str) -> MetalResult<NonNull<T>> {
    NonNull::new(ptr).ok_or(MetalError::NullPointer(what))
}

// ═══════════════════════════════════════════════════════════════════════════
// FFI Declarations
// ═══════════════════════════════════════════════════════════════════════════

extern "C" {

    // ── Lifecycle ─────────────────────────────────────────────────────────

    fn ds4_metal_init() -> i32;
    fn ds4_metal_cleanup();

    // ── Tensor operations ─────────────────────────────────────────────────

    fn ds4_metal_tensor_alloc(bytes: u64) -> *mut ds4_metal_tensor;
    fn ds4_metal_tensor_view(
        base: *const ds4_metal_tensor,
        offset: u64,
        bytes: u64,
    ) -> *mut ds4_metal_tensor;
    fn ds4_metal_tensor_free(tensor: *mut ds4_metal_tensor);
    fn ds4_metal_tensor_bytes(tensor: *const ds4_metal_tensor) -> u64;
    fn ds4_metal_tensor_contents(tensor: *mut ds4_metal_tensor) -> *mut c_void;
    fn ds4_metal_tensor_write(
        tensor: *mut ds4_metal_tensor,
        offset: u64,
        data: *const c_void,
        bytes: u64,
    ) -> i32;
    fn ds4_metal_tensor_read(
        tensor: *const ds4_metal_tensor,
        offset: u64,
        data: *mut c_void,
        bytes: u64,
    ) -> i32;
    fn ds4_metal_tensor_copy(
        dst: *mut ds4_metal_tensor,
        dst_offset: u64,
        src: *const ds4_metal_tensor,
        src_offset: u64,
        bytes: u64,
    ) -> i32;

    // ── Command batching ──────────────────────────────────────────────────

    fn ds4_metal_begin_commands() -> i32;
    fn ds4_metal_flush_commands() -> i32;
    fn ds4_metal_end_commands() -> i32;
    fn ds4_metal_synchronize() -> i32;

    // ── Model mapping ─────────────────────────────────────────────────────

    fn ds4_metal_set_model_map(model_map: *const c_void, model_size: u64) -> i32;
    fn ds4_metal_set_model_map_range(
        model_map: *const c_void,
        model_size: u64,
        map_offset: u64,
        map_size: u64,
    ) -> i32;
    fn ds4_metal_set_quality(quality: bool);
    fn ds4_metal_print_memory_report(label: *const std::ffi::c_char);

    // ── Embeddings and Indexer Helpers ────────────────────────────────────

    fn ds4_metal_embed_token_hc_tensor(
        out_hc: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n_vocab: u32,
        token: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

    fn ds4_metal_embed_tokens_hc_tensor(
        out_hc: *mut ds4_metal_tensor,
        tokens: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n_vocab: u32,
        n_tokens: u32,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

    fn ds4_metal_indexer_score_one_tensor(
        scores: *mut ds4_metal_tensor,
        q: *const ds4_metal_tensor,
        weights: *const ds4_metal_tensor,
        index_comp: *const ds4_metal_tensor,
        n_comp: u32,
        n_head: u32,
        head_dim: u32,
        scale: f32,
    ) -> i32;

    fn ds4_metal_indexer_scores_prefill_tensor(
        scores: *mut ds4_metal_tensor,
        q: *const ds4_metal_tensor,
        weights: *const ds4_metal_tensor,
        index_comp: *const ds4_metal_tensor,
        n_comp: u32,
        n_tokens: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
    ) -> i32;

    fn ds4_metal_indexer_scores_decode_batch_tensor(
        scores: *mut ds4_metal_tensor,
        q: *const ds4_metal_tensor,
        weights: *const ds4_metal_tensor,
        index_comp: *const ds4_metal_tensor,
        n_comp: u32,
        n_tokens: u32,
        pos0: u32,
        n_head: u32,
        head_dim: u32,
        ratio: u32,
        scale: f32,
    ) -> i32;

    fn ds4_metal_indexer_topk_tensor(
        selected: *mut ds4_metal_tensor,
        scores: *const ds4_metal_tensor,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> i32;

    fn ds4_metal_dsv4_topk_mask_tensor(
        mask: *mut ds4_metal_tensor,
        topk: *const ds4_metal_tensor,
        n_comp: u32,
        n_tokens: u32,
        top_k: u32,
    ) -> i32;

    // ── Dense Projections (MatMul) ────────────────────────────────────────

    fn ds4_metal_matmul_q8_0_tensor(
        out: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const ds4_metal_tensor,
        n_tok: u64,
    ) -> i32;

    fn ds4_metal_shared_gate_up_swiglu_q8_0_tensor(
        gate: *mut ds4_metal_tensor,
        up: *mut ds4_metal_tensor,
        mid: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        gate_offset: u64,
        up_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const ds4_metal_tensor,
    ) -> i32;

    fn ds4_metal_matmul_f16_tensor(
        out: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const ds4_metal_tensor,
        n_tok: u64,
    ) -> i32;

    fn ds4_metal_matmul_f16_pair_tensor(
        out_a: *mut ds4_metal_tensor,
        out_b: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_a_offset: u64,
        weight_b_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const ds4_metal_tensor,
        n_tok: u64,
    ) -> i32;

    fn ds4_metal_matmul_f32_tensor(
        out: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const ds4_metal_tensor,
        n_tok: u64,
    ) -> i32;

    fn ds4_metal_repeat_hc_tensor(
        out: *mut ds4_metal_tensor,
        row: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

    // ── Normalisation ─────────────────────────────────────────────────────

    fn ds4_metal_rms_norm_plain_tensor(
        out: *mut ds4_metal_tensor,
        x: *const ds4_metal_tensor,
        n: u32,
        eps: f32,
    ) -> i32;

    fn ds4_metal_rms_norm_plain_rows_tensor(
        out: *mut ds4_metal_tensor,
        x: *const ds4_metal_tensor,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> i32;

    fn ds4_metal_rms_norm_weight_tensor(
        out: *mut ds4_metal_tensor,
        x: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n: u32,
        eps: f32,
    ) -> i32;

    fn ds4_metal_rms_norm_weight_rows_tensor(
        out: *mut ds4_metal_tensor,
        x: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        n: u32,
        rows: u32,
        eps: f32,
    ) -> i32;

    fn ds4_metal_dsv4_qkv_rms_norm_rows_tensor(
        q_out: *mut ds4_metal_tensor,
        q: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        q_weight_offset: u64,
        q_n: u32,
        kv_out: *mut ds4_metal_tensor,
        kv: *const ds4_metal_tensor,
        kv_weight_offset: u64,
        kv_n: u32,
        rows: u32,
        eps: f32,
    ) -> i32;

    fn ds4_metal_head_rms_norm_tensor(
        x: *mut ds4_metal_tensor,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        eps: f32,
    ) -> i32;

    // ── KV Quantisation ───────────────────────────────────────────────────

    fn ds4_metal_dsv4_fp8_kv_quantize_tensor(
        x: *mut ds4_metal_tensor,
        n_tok: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> i32;

    // ── RoPE ──────────────────────────────────────────────────────────────

    fn ds4_metal_rope_tail_tensor(
        x: *mut ds4_metal_tensor,
        n_tok: u32,
        n_head: u32,
        head_dim: u32,
        n_rot: u32,
        pos0: u32,
        n_ctx_orig: u32,
        inverse: bool,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> i32;

    // ── KV Cache ──────────────────────────────────────────────────────────

    fn ds4_metal_kv_fp8_store_raw_tensor(
        kv: *mut ds4_metal_tensor,
        raw_cache: *mut ds4_metal_tensor,
        raw_cap: u32,
        row: u32,
        head_dim: u32,
        n_rot: u32,
    ) -> i32;

    fn ds4_metal_store_raw_kv_tensor(
        raw_cache: *mut ds4_metal_tensor,
        kv: *const ds4_metal_tensor,
        raw_cap: u32,
        row: u32,
        head_dim: u32,
    ) -> i32;

    fn ds4_metal_store_raw_kv_batch_tensor(
        raw_cache: *mut ds4_metal_tensor,
        kv: *const ds4_metal_tensor,
        raw_cap: u32,
        pos0: u32,
        n_tokens: u32,
        head_dim: u32,
    ) -> i32;

    fn ds4_metal_compressor_update_tensor(
        kv_cur: *const ds4_metal_tensor,
        sc_cur: *const ds4_metal_tensor,
        state_kv: *mut ds4_metal_tensor,
        state_score: *mut ds4_metal_tensor,
        comp_cache: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        norm_offset: u64,
        norm_type: u32,
        head_dim: u32,
        ratio: u32,
        pos: u32,
        comp_row: u32,
        n_rot: u32,
        n_ctx_orig: u32,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        rms_eps: f32,
    ) -> i32;

    fn ds4_metal_compressor_store_batch_tensor(
        kv: *const ds4_metal_tensor,
        sc: *const ds4_metal_tensor,
        state_kv: *mut ds4_metal_tensor,
        state_score: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_tokens: u32,
    ) -> i32;

    fn ds4_metal_compressor_prefill_tensor(
        comp_cache: *mut ds4_metal_tensor,
        state_kv: *mut ds4_metal_tensor,
        state_score: *mut ds4_metal_tensor,
        kv: *const ds4_metal_tensor,
        sc: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        norm_offset: u64,
        norm_type: u32,
        head_dim: u32,
        ratio: u32,
        pos0: u32,
        n_tokens: u32,
        n_rot: u32,
        n_ctx_orig: u32,
        quantize_fp8: bool,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        rms_eps: f32,
    ) -> i32;

    fn ds4_metal_compressor_prefill_ratio4_replay_tensor(
        comp_cache: *mut ds4_metal_tensor,
        state_kv: *mut ds4_metal_tensor,
        state_score: *mut ds4_metal_tensor,
        kv: *const ds4_metal_tensor,
        sc: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        norm_offset: u64,
        norm_type: u32,
        head_dim: u32,
        pos0: u32,
        n_tokens: u32,
        n_rot: u32,
        n_ctx_orig: u32,
        quantize_fp8: bool,
        freq_base: f32,
        freq_scale: f32,
        ext_factor: f32,
        attn_factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        rms_eps: f32,
    ) -> i32;

    fn ds4_metal_compressor_prefill_state_ratio4_tensor(
        state_kv: *mut ds4_metal_tensor,
        state_score: *mut ds4_metal_tensor,
        kv_tail: *const ds4_metal_tensor,
        sc_tail: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        ape_offset: u64,
        ape_type: u32,
        head_dim: u32,
        pos0: u32,
    ) -> i32;

    // ── Attention ─────────────────────────────────────────────────────────

    fn ds4_metal_attention_decode_heads_tensor(
        heads: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const ds4_metal_tensor,
        raw_kv: *const ds4_metal_tensor,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        comp_kv: *const ds4_metal_tensor,
        n_comp: u32,
        comp_mask: *const ds4_metal_tensor,
        use_mask: u32,
        n_head: u32,
        head_dim: u32,
    ) -> i32;

    fn ds4_metal_attention_prefill_raw_heads_tensor(
        heads: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const ds4_metal_tensor,
        raw_kv: *const ds4_metal_tensor,
        n_tokens: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) -> i32;

    fn ds4_metal_attention_decode_raw_batch_heads_tensor(
        heads: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const ds4_metal_tensor,
        raw_kv: *const ds4_metal_tensor,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        window: u32,
        n_head: u32,
        head_dim: u32,
    ) -> i32;

    fn ds4_metal_attention_decode_mixed_batch_heads_tensor(
        heads: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const ds4_metal_tensor,
        raw_kv: *const ds4_metal_tensor,
        comp_kv: *const ds4_metal_tensor,
        comp_mask: *const ds4_metal_tensor,
        use_comp_mask: u32,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> i32;

    fn ds4_metal_attention_indexed_mixed_batch_heads_tensor(
        heads: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const ds4_metal_tensor,
        raw_kv: *const ds4_metal_tensor,
        comp_kv: *const ds4_metal_tensor,
        topk: *const ds4_metal_tensor,
        n_tokens: u32,
        pos0: u32,
        n_raw: u32,
        raw_cap: u32,
        raw_start: u32,
        n_comp: u32,
        top_k: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> i32;

    fn ds4_metal_attention_prefill_static_mixed_heads_tensor(
        heads: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const ds4_metal_tensor,
        raw_kv: *const ds4_metal_tensor,
        comp_kv: *const ds4_metal_tensor,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> i32;

    fn ds4_metal_attention_prefill_masked_mixed_heads_tensor(
        heads: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        sinks_offset: u64,
        q: *const ds4_metal_tensor,
        raw_kv: *const ds4_metal_tensor,
        comp_kv: *const ds4_metal_tensor,
        comp_mask: *const ds4_metal_tensor,
        n_tokens: u32,
        n_comp: u32,
        window: u32,
        ratio: u32,
        n_head: u32,
        head_dim: u32,
    ) -> i32;

    fn ds4_metal_attention_output_q8_batch_tensor(
        out: *mut ds4_metal_tensor,
        low: *mut ds4_metal_tensor,
        group_tmp: *mut ds4_metal_tensor,
        low_tmp: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        out_a_offset: u64,
        out_b_offset: u64,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        out_dim: u64,
        heads: *const ds4_metal_tensor,
        n_tokens: u32,
    ) -> i32;

    fn ds4_metal_attention_output_low_q8_tensor(
        low: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        out_a_offset: u64,
        group_dim: u64,
        rank: u64,
        n_groups: u32,
        heads: *const ds4_metal_tensor,
    ) -> i32;

    // ── FFN / MoE ─────────────────────────────────────────────────────────

    fn ds4_metal_swiglu_tensor(
        out: *mut ds4_metal_tensor,
        gate: *const ds4_metal_tensor,
        up: *const ds4_metal_tensor,
        n: u32,
        clamp: f32,
        weight: f32,
    ) -> i32;

    fn ds4_metal_add_tensor(
        out: *mut ds4_metal_tensor,
        a: *const ds4_metal_tensor,
        b: *const ds4_metal_tensor,
        n: u32,
    ) -> i32;

    fn ds4_metal_router_select_tensor(
        selected: *mut ds4_metal_tensor,
        weights: *mut ds4_metal_tensor,
        probs: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        bias_offset: u64,
        hash_offset: u64,
        hash_rows: u32,
        token: u32,
        n_expert_groups: u32,
        n_group_used: u32,
        has_bias: bool,
        hash_mode: bool,
        logits: *const ds4_metal_tensor,
    ) -> i32;

    fn ds4_metal_router_select_batch_tensor(
        selected: *mut ds4_metal_tensor,
        weights: *mut ds4_metal_tensor,
        probs: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        bias_offset: u64,
        hash_offset: u64,
        hash_rows: u32,
        n_expert_groups: u32,
        n_group_used: u32,
        has_bias: bool,
        hash_mode: bool,
        logits: *const ds4_metal_tensor,
        tokens: *const ds4_metal_tensor,
        n_tokens: u32,
    ) -> i32;

    fn ds4_metal_routed_moe_one_tensor(
        out: *mut ds4_metal_tensor,
        gate: *mut ds4_metal_tensor,
        up: *mut ds4_metal_tensor,
        mid: *mut ds4_metal_tensor,
        experts: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        gate_offset: u64,
        up_offset: u64,
        down_offset: u64,
        gate_type: u32,
        down_type: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        expert_in_dim: u32,
        expert_mid_dim: u32,
        out_dim: u32,
        selected: *const ds4_metal_tensor,
        weights: *const ds4_metal_tensor,
        n_expert: u32,
        clamp: f32,
        x: *const ds4_metal_tensor,
    ) -> i32;

    fn ds4_metal_routed_moe_batch_tensor(
        out: *mut ds4_metal_tensor,
        gate: *mut ds4_metal_tensor,
        up: *mut ds4_metal_tensor,
        mid: *mut ds4_metal_tensor,
        experts: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        gate_offset: u64,
        up_offset: u64,
        down_offset: u64,
        gate_type: u32,
        down_type: u32,
        gate_expert_bytes: u64,
        gate_row_bytes: u64,
        down_expert_bytes: u64,
        down_row_bytes: u64,
        expert_in_dim: u32,
        expert_mid_dim: u32,
        out_dim: u32,
        selected: *const ds4_metal_tensor,
        weights: *const ds4_metal_tensor,
        n_expert: u32,
        clamp: f32,
        x: *const ds4_metal_tensor,
        n_tokens: u32,
    ) -> i32;

    // ── Hyper-Connection Kernels ──────────────────────────────────────────

    fn ds4_metal_hc_split_sinkhorn_tensor(
        out: *mut ds4_metal_tensor,
        mix: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        scale_offset: u64,
        base_offset: u64,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> i32;

    fn ds4_metal_hc_weighted_sum_tensor(
        out: *mut ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        weights: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

    fn ds4_metal_hc_weighted_sum_split_tensor(
        out: *mut ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        split: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

    fn ds4_metal_hc_split_weighted_sum_tensor(
        out: *mut ds4_metal_tensor,
        split: *mut ds4_metal_tensor,
        mix: *const ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        scale_offset: u64,
        base_offset: u64,
        n_embd: u32,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
    ) -> i32;

    fn ds4_metal_hc_split_weighted_sum_norm_tensor(
        out: *mut ds4_metal_tensor,
        norm_out: *mut ds4_metal_tensor,
        split: *mut ds4_metal_tensor,
        mix: *const ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        scale_offset: u64,
        base_offset: u64,
        norm_weight_offset: u64,
        n_embd: u32,
        n_hc: u32,
        sinkhorn_iters: u32,
        eps: f32,
        norm_eps: f32,
    ) -> i32;

    fn ds4_metal_output_hc_weights_tensor(
        out: *mut ds4_metal_tensor,
        pre: *const ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        scale_offset: u64,
        base_offset: u64,
        n_hc: u32,
        eps: f32,
    ) -> i32;

    fn ds4_metal_hc_expand_tensor(
        out_hc: *mut ds4_metal_tensor,
        block_out: *const ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        post: *const ds4_metal_tensor,
        comb: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

    fn ds4_metal_hc_expand_split_tensor(
        out_hc: *mut ds4_metal_tensor,
        block_out: *const ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        split: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

    fn ds4_metal_hc_expand_add_split_tensor(
        out_hc: *mut ds4_metal_tensor,
        block_out: *const ds4_metal_tensor,
        block_add: *const ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        split: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

    fn ds4_metal_shared_down_hc_expand_q8_0_tensor(
        out_hc: *mut ds4_metal_tensor,
        shared_out: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        shared_mid: *const ds4_metal_tensor,
        routed_out: *const ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        split: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

    fn ds4_metal_matmul_q8_0_hc_expand_tensor(
        out_hc: *mut ds4_metal_tensor,
        block_out: *mut ds4_metal_tensor,
        model_map: *const c_void,
        model_size: u64,
        weight_offset: u64,
        in_dim: u64,
        out_dim: u64,
        x: *const ds4_metal_tensor,
        residual_hc: *const ds4_metal_tensor,
        split: *const ds4_metal_tensor,
        n_embd: u32,
        n_hc: u32,
    ) -> i32;

} // extern "C"

// ═══════════════════════════════════════════════════════════════════════════
// Safe Wrapper Functions
// ═══════════════════════════════════════════════════════════════════════════

// ── Lifecycle ─────────────────────────────────────────────────────────────

/// Initialise the Metal device, command queue, and shader library.
pub fn metal_init() -> MetalResult<()> {
    // Safety: ds4_metal_init is safe to call; it returns 0 on failure, 1 on
    // success (not 0). The convention is 0=success for most Metal functions,
    // but init returns a truthy value.
    let ret = unsafe { ds4_metal_init() };
    if ret == 0 {
        Err(MetalError::InitFailed)
    } else {
        METAL_INITIALIZED.store(true, Ordering::Relaxed);
        Ok(())
    }
}

/// Tear down the Metal runtime and free all resources.
pub fn metal_cleanup() {
    if METAL_INITIALIZED.swap(false, Ordering::Relaxed) {
        unsafe { ds4_metal_cleanup() }
    }
}

// ── Tensor operations ─────────────────────────────────────────────────────

/// Allocate a GPU-side tensor of `bytes` contiguous device memory.
///
/// Returns `None` if allocation failed (out of memory or Metal not initialised).
pub fn metal_tensor_alloc(bytes: u64) -> MetalResult<NonNull<ds4_metal_tensor>> {
    nonnull(
        unsafe { ds4_metal_tensor_alloc(bytes) },
        "ds4_metal_tensor_alloc",
    )
}

/// Create a view into an existing tensor starting at `offset` with `bytes` length.
///
/// The view shares the underlying `MTLBuffer` and does not own its memory.
pub fn metal_tensor_view(
    base: &ds4_metal_tensor,
    offset: u64,
    bytes: u64,
) -> MetalResult<NonNull<ds4_metal_tensor>> {
    nonnull(
        unsafe { ds4_metal_tensor_view(base as *const ds4_metal_tensor, offset, bytes) },
        "ds4_metal_tensor_view",
    )
}

/// Free a GPU tensor allocated with `metal_tensor_alloc`.
///
/// # Safety
/// The tensor must not be used after this call.
pub unsafe fn metal_tensor_free(tensor: *mut ds4_metal_tensor) {
    if !tensor.is_null() {
        ds4_metal_tensor_free(tensor)
    }
}

/// Return the byte size of a GPU tensor.
pub fn metal_tensor_bytes(tensor: &ds4_metal_tensor) -> u64 {
    unsafe { ds4_metal_tensor_bytes(tensor as *const ds4_metal_tensor) }
}

/// Return a CPU-mapped pointer to the tensor's contents.
///
/// # Safety
/// The returned pointer is only valid while the tensor is alive and until
/// command encoding that touches the buffer completes.
pub unsafe fn metal_tensor_contents(tensor: *mut ds4_metal_tensor) -> MetalResult<NonNull<c_void>> {
    let ptr = ds4_metal_tensor_contents(tensor);
    nonnull(ptr, "ds4_metal_tensor_contents")
}

/// Write `bytes` of `data` into the tensor at `offset`.
pub fn metal_tensor_write(tensor: &ds4_metal_tensor, offset: u64, data: &[u8]) -> MetalResult<()> {
    let bytes = data.len() as u64;
    ck(unsafe {
        ds4_metal_tensor_write(
            tensor as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            offset,
            data.as_ptr() as *const c_void,
            bytes,
        )
    })
}

/// Read `bytes` from the tensor at `offset` into `data`.
pub fn metal_tensor_read(
    tensor: &ds4_metal_tensor,
    offset: u64,
    data: &mut [u8],
) -> MetalResult<()> {
    let bytes = data.len() as u64;
    ck(unsafe {
        ds4_metal_tensor_read(
            tensor as *const ds4_metal_tensor,
            offset,
            data.as_mut_ptr() as *mut c_void,
            bytes,
        )
    })
}

/// Copy `bytes` from `src` at `src_offset` to `dst` at `dst_offset`.
pub fn metal_tensor_copy(
    dst: &ds4_metal_tensor,
    dst_offset: u64,
    src: &ds4_metal_tensor,
    src_offset: u64,
    bytes: u64,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_tensor_copy(
            dst as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            dst_offset,
            src as *const ds4_metal_tensor,
            src_offset,
            bytes,
        )
    })
}

// ── Command batching ──────────────────────────────────────────────────────

/// Begin encoding a new command batch.
pub fn metal_begin_commands() -> MetalResult<()> {
    ck(unsafe { ds4_metal_begin_commands() })
}

/// Flush the current command batch to the GPU without waiting.
pub fn metal_flush_commands() -> MetalResult<()> {
    ck(unsafe { ds4_metal_flush_commands() })
}

/// End the current command batch and commit it for execution.
pub fn metal_end_commands() -> MetalResult<()> {
    ck(unsafe { ds4_metal_end_commands() })
}

/// Wait for all pending GPU commands to complete.
pub fn metal_synchronize() -> MetalResult<()> {
    ck(unsafe { ds4_metal_synchronize() })
}

// ── Model mapping ─────────────────────────────────────────────────────────

/// Map the entire model GGUF data into GPU-visible memory.
pub fn metal_set_model_map(model_map: &[u8]) -> MetalResult<()> {
    let ptr = model_map.as_ptr() as *const c_void;
    let size = model_map.len() as u64;
    ck(unsafe { ds4_metal_set_model_map(ptr, size) })
}

/// Map a sub-range of the model GGUF data into GPU-visible memory.
pub fn metal_set_model_map_range(
    model_map: &[u8],
    map_offset: u64,
    map_size: u64,
) -> MetalResult<()> {
    let ptr = model_map.as_ptr() as *const c_void;
    let size = model_map.len() as u64;
    ck(unsafe { ds4_metal_set_model_map_range(ptr, size, map_offset, map_size) })
}

/// Set quality mode (extra precision in kernels).
pub fn metal_set_quality(quality: bool) {
    unsafe { ds4_metal_set_quality(quality) }
}

/// Print the Metal memory report to stderr.
pub fn metal_print_memory_report(label: &str) {
    let c_label = std::ffi::CString::new(label).expect("CString::new failed");
    unsafe { ds4_metal_print_memory_report(c_label.as_ptr()) }
}

// ── Embeddings and Indexer Helpers ────────────────────────────────────────

/// Embed a single token into HC state via lookup from the model weights.
pub fn metal_embed_token_hc_tensor(
    out_hc: &ds4_metal_tensor,
    model_map: &[u8],
    weight_offset: u64,
    n_vocab: u32,
    token: u32,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_embed_token_hc_tensor(
            out_hc as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_offset,
            n_vocab,
            token,
            n_embd,
            n_hc,
        )
    })
}

/// Embed tokens into HC state from a token ID tensor.
pub fn metal_embed_tokens_hc_tensor(
    out_hc: &ds4_metal_tensor,
    tokens: &ds4_metal_tensor,
    model_map: &[u8],
    weight_offset: u64,
    n_vocab: u32,
    n_tokens: u32,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_embed_tokens_hc_tensor(
            out_hc as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            tokens as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_offset,
            n_vocab,
            n_tokens,
            n_embd,
            n_hc,
        )
    })
}

/// Score a single query against the indexer compressor state.
pub fn metal_indexer_score_one_tensor(
    scores: &ds4_metal_tensor,
    q: &ds4_metal_tensor,
    weights: &ds4_metal_tensor,
    index_comp: &ds4_metal_tensor,
    n_comp: u32,
    n_head: u32,
    head_dim: u32,
    scale: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_indexer_score_one_tensor(
            scores as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            q as *const ds4_metal_tensor,
            weights as *const ds4_metal_tensor,
            index_comp as *const ds4_metal_tensor,
            n_comp,
            n_head,
            head_dim,
            scale,
        )
    })
}

/// Score all queries during prefill against the indexer.
pub fn metal_indexer_scores_prefill_tensor(
    scores: &ds4_metal_tensor,
    q: &ds4_metal_tensor,
    weights: &ds4_metal_tensor,
    index_comp: &ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    n_head: u32,
    head_dim: u32,
    ratio: u32,
    scale: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_indexer_scores_prefill_tensor(
            scores as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            q as *const ds4_metal_tensor,
            weights as *const ds4_metal_tensor,
            index_comp as *const ds4_metal_tensor,
            n_comp,
            n_tokens,
            n_head,
            head_dim,
            ratio,
            scale,
        )
    })
}

/// Score queries during batched decode against the indexer.
pub fn metal_indexer_scores_decode_batch_tensor(
    scores: &ds4_metal_tensor,
    q: &ds4_metal_tensor,
    weights: &ds4_metal_tensor,
    index_comp: &ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    pos0: u32,
    n_head: u32,
    head_dim: u32,
    ratio: u32,
    scale: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_indexer_scores_decode_batch_tensor(
            scores as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            q as *const ds4_metal_tensor,
            weights as *const ds4_metal_tensor,
            index_comp as *const ds4_metal_tensor,
            n_comp,
            n_tokens,
            pos0,
            n_head,
            head_dim,
            ratio,
            scale,
        )
    })
}

/// Top-k selection from indexer scores.
pub fn metal_indexer_topk_tensor(
    selected: &ds4_metal_tensor,
    scores: &ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_indexer_topk_tensor(
            selected as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            scores as *const ds4_metal_tensor,
            n_comp,
            n_tokens,
            top_k,
        )
    })
}

/// Build a top-k mask from the selected indices.
pub fn metal_dsv4_topk_mask_tensor(
    mask: &ds4_metal_tensor,
    topk: &ds4_metal_tensor,
    n_comp: u32,
    n_tokens: u32,
    top_k: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_dsv4_topk_mask_tensor(
            mask as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            topk as *const ds4_metal_tensor,
            n_comp,
            n_tokens,
            top_k,
        )
    })
}

// ── Dense Projections ─────────────────────────────────────────────────────

/// Q8_0 quantised matrix-vector multiply.
pub fn metal_matmul_q8_0_tensor(
    out: &ds4_metal_tensor,
    model_map: &[u8],
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: &ds4_metal_tensor,
    n_tok: u64,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_matmul_q8_0_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_offset,
            in_dim,
            out_dim,
            x as *const ds4_metal_tensor,
            n_tok,
        )
    })
}

/// Shared gate+up projection with fused SwiGLU, Q8_0 weights.
pub fn metal_shared_gate_up_swiglu_q8_0_tensor(
    gate: &ds4_metal_tensor,
    up: &ds4_metal_tensor,
    mid: &ds4_metal_tensor,
    model_map: &[u8],
    gate_offset: u64,
    up_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: &ds4_metal_tensor,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_shared_gate_up_swiglu_q8_0_tensor(
            gate as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            up as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            mid as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            gate_offset,
            up_offset,
            in_dim,
            out_dim,
            x as *const ds4_metal_tensor,
        )
    })
}

/// F16 matrix-vector multiply.
pub fn metal_matmul_f16_tensor(
    out: &ds4_metal_tensor,
    model_map: &[u8],
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: &ds4_metal_tensor,
    n_tok: u64,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_matmul_f16_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_offset,
            in_dim,
            out_dim,
            x as *const ds4_metal_tensor,
            n_tok,
        )
    })
}

/// F16 matrix-vector multiply producing two output projections from two weight matrices.
pub fn metal_matmul_f16_pair_tensor(
    out_a: &ds4_metal_tensor,
    out_b: &ds4_metal_tensor,
    model_map: &[u8],
    weight_a_offset: u64,
    weight_b_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: &ds4_metal_tensor,
    n_tok: u64,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_matmul_f16_pair_tensor(
            out_a as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            out_b as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_a_offset,
            weight_b_offset,
            in_dim,
            out_dim,
            x as *const ds4_metal_tensor,
            n_tok,
        )
    })
}

/// F32 matrix-vector multiply.
pub fn metal_matmul_f32_tensor(
    out: &ds4_metal_tensor,
    model_map: &[u8],
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: &ds4_metal_tensor,
    n_tok: u64,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_matmul_f32_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_offset,
            in_dim,
            out_dim,
            x as *const ds4_metal_tensor,
            n_tok,
        )
    })
}

/// Repeat a single HC row across the HC dimension.
pub fn metal_repeat_hc_tensor(
    out: &ds4_metal_tensor,
    row: &ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_repeat_hc_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            row as *const ds4_metal_tensor,
            n_embd,
            n_hc,
        )
    })
}

// ── Normalisation ─────────────────────────────────────────────────────────

/// Plain RMS normalisation (single row).
pub fn metal_rms_norm_plain_tensor(
    out: &ds4_metal_tensor,
    x: &ds4_metal_tensor,
    n: u32,
    eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_rms_norm_plain_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            x as *const ds4_metal_tensor,
            n,
            eps,
        )
    })
}

/// Plain RMS normalisation over multiple rows.
pub fn metal_rms_norm_plain_rows_tensor(
    out: &ds4_metal_tensor,
    x: &ds4_metal_tensor,
    n: u32,
    rows: u32,
    eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_rms_norm_plain_rows_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            x as *const ds4_metal_tensor,
            n,
            rows,
            eps,
        )
    })
}

/// RMS normalisation with learned weight (single row).
pub fn metal_rms_norm_weight_tensor(
    out: &ds4_metal_tensor,
    x: &ds4_metal_tensor,
    model_map: &[u8],
    weight_offset: u64,
    n: u32,
    eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_rms_norm_weight_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            x as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_offset,
            n,
            eps,
        )
    })
}

/// RMS normalisation with learned weight over multiple rows.
pub fn metal_rms_norm_weight_rows_tensor(
    out: &ds4_metal_tensor,
    x: &ds4_metal_tensor,
    model_map: &[u8],
    weight_offset: u64,
    n: u32,
    rows: u32,
    eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_rms_norm_weight_rows_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            x as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_offset,
            n,
            rows,
            eps,
        )
    })
}

/// DSv4 fused Q/KV RMS normalisation over rows.
pub fn metal_dsv4_qkv_rms_norm_rows_tensor(
    q_out: &ds4_metal_tensor,
    q: &ds4_metal_tensor,
    model_map: &[u8],
    q_weight_offset: u64,
    q_n: u32,
    kv_out: &ds4_metal_tensor,
    kv: &ds4_metal_tensor,
    kv_weight_offset: u64,
    kv_n: u32,
    rows: u32,
    eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_dsv4_qkv_rms_norm_rows_tensor(
            q_out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            q as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            q_weight_offset,
            q_n,
            kv_out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            kv as *const ds4_metal_tensor,
            kv_weight_offset,
            kv_n,
            rows,
            eps,
        )
    })
}

/// Head-wise RMS normalisation (in-place on `x`).
pub fn metal_head_rms_norm_tensor(
    x: &ds4_metal_tensor,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_head_rms_norm_tensor(
            x as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            n_tok,
            n_head,
            head_dim,
            eps,
        )
    })
}

// ── KV Quantisation ───────────────────────────────────────────────────────

/// Quantise KV activations to FP8 (in-place on `x`).
pub fn metal_dsv4_fp8_kv_quantize_tensor(
    x: &ds4_metal_tensor,
    n_tok: u32,
    head_dim: u32,
    n_rot: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_dsv4_fp8_kv_quantize_tensor(
            x as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            n_tok,
            head_dim,
            n_rot,
        )
    })
}

// ── RoPE ──────────────────────────────────────────────────────────────────

/// Apply tail-only Rotary Position Embedding.
#[allow(clippy::too_many_arguments)]
pub fn metal_rope_tail_tensor(
    x: &ds4_metal_tensor,
    n_tok: u32,
    n_head: u32,
    head_dim: u32,
    n_rot: u32,
    pos0: u32,
    n_ctx_orig: u32,
    inverse: bool,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_rope_tail_tensor(
            x as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            n_tok,
            n_head,
            head_dim,
            n_rot,
            pos0,
            n_ctx_orig,
            inverse,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
        )
    })
}

// ── KV Cache ──────────────────────────────────────────────────────────────

/// Fused FP8 quantise + store KV row into the raw attention cache.
pub fn metal_kv_fp8_store_raw_tensor(
    kv: &ds4_metal_tensor,
    raw_cache: &ds4_metal_tensor,
    raw_cap: u32,
    row: u32,
    head_dim: u32,
    n_rot: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_kv_fp8_store_raw_tensor(
            kv as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            raw_cache as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            raw_cap,
            row,
            head_dim,
            n_rot,
        )
    })
}

/// Store a raw KV row into the attention cache (F16 reference path).
pub fn metal_store_raw_kv_tensor(
    raw_cache: &ds4_metal_tensor,
    kv: &ds4_metal_tensor,
    raw_cap: u32,
    row: u32,
    head_dim: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_store_raw_kv_tensor(
            raw_cache as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            kv as *const ds4_metal_tensor,
            raw_cap,
            row,
            head_dim,
        )
    })
}

/// Store a batch of raw KV rows into the attention cache.
pub fn metal_store_raw_kv_batch_tensor(
    raw_cache: &ds4_metal_tensor,
    kv: &ds4_metal_tensor,
    raw_cap: u32,
    pos0: u32,
    n_tokens: u32,
    head_dim: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_store_raw_kv_batch_tensor(
            raw_cache as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            kv as *const ds4_metal_tensor,
            raw_cap,
            pos0,
            n_tokens,
            head_dim,
        )
    })
}

/// Update the compressor state with a single new KV/score row.
#[allow(clippy::too_many_arguments)]
pub fn metal_compressor_update_tensor(
    kv_cur: &ds4_metal_tensor,
    sc_cur: &ds4_metal_tensor,
    state_kv: &ds4_metal_tensor,
    state_score: &ds4_metal_tensor,
    comp_cache: &ds4_metal_tensor,
    model_map: &[u8],
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64,
    norm_type: u32,
    head_dim: u32,
    ratio: u32,
    pos: u32,
    comp_row: u32,
    n_rot: u32,
    n_ctx_orig: u32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    rms_eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_compressor_update_tensor(
            kv_cur as *const ds4_metal_tensor,
            sc_cur as *const ds4_metal_tensor,
            state_kv as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            state_score as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            comp_cache as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            ape_offset,
            ape_type,
            norm_offset,
            norm_type,
            head_dim,
            ratio,
            pos,
            comp_row,
            n_rot,
            n_ctx_orig,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            rms_eps,
        )
    })
}

/// Store a batch of KV/score rows into the compressor rolling state.
pub fn metal_compressor_store_batch_tensor(
    kv: &ds4_metal_tensor,
    sc: &ds4_metal_tensor,
    state_kv: &ds4_metal_tensor,
    state_score: &ds4_metal_tensor,
    model_map: &[u8],
    ape_offset: u64,
    ape_type: u32,
    head_dim: u32,
    ratio: u32,
    pos0: u32,
    n_tokens: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_compressor_store_batch_tensor(
            kv as *const ds4_metal_tensor,
            sc as *const ds4_metal_tensor,
            state_kv as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            state_score as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            ape_offset,
            ape_type,
            head_dim,
            ratio,
            pos0,
            n_tokens,
        )
    })
}

/// Compressor prefill: process a batch of KV/score rows and produce compressed
/// cache entries.
#[allow(clippy::too_many_arguments)]
pub fn metal_compressor_prefill_tensor(
    comp_cache: &ds4_metal_tensor,
    state_kv: &ds4_metal_tensor,
    state_score: &ds4_metal_tensor,
    kv: &ds4_metal_tensor,
    sc: &ds4_metal_tensor,
    model_map: &[u8],
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64,
    norm_type: u32,
    head_dim: u32,
    ratio: u32,
    pos0: u32,
    n_tokens: u32,
    n_rot: u32,
    n_ctx_orig: u32,
    quantize_fp8: bool,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    rms_eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_compressor_prefill_tensor(
            comp_cache as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            state_kv as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            state_score as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            kv as *const ds4_metal_tensor,
            sc as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            ape_offset,
            ape_type,
            norm_offset,
            norm_type,
            head_dim,
            ratio,
            pos0,
            n_tokens,
            n_rot,
            n_ctx_orig,
            quantize_fp8,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            rms_eps,
        )
    })
}

/// Compressor prefill replay variant (for ratio-4 replay).
#[allow(clippy::too_many_arguments)]
pub fn metal_compressor_prefill_ratio4_replay_tensor(
    comp_cache: &ds4_metal_tensor,
    state_kv: &ds4_metal_tensor,
    state_score: &ds4_metal_tensor,
    kv: &ds4_metal_tensor,
    sc: &ds4_metal_tensor,
    model_map: &[u8],
    ape_offset: u64,
    ape_type: u32,
    norm_offset: u64,
    norm_type: u32,
    head_dim: u32,
    pos0: u32,
    n_tokens: u32,
    n_rot: u32,
    n_ctx_orig: u32,
    quantize_fp8: bool,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    rms_eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_compressor_prefill_ratio4_replay_tensor(
            comp_cache as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            state_kv as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            state_score as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            kv as *const ds4_metal_tensor,
            sc as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            ape_offset,
            ape_type,
            norm_offset,
            norm_type,
            head_dim,
            pos0,
            n_tokens,
            n_rot,
            n_ctx_orig,
            quantize_fp8,
            freq_base,
            freq_scale,
            ext_factor,
            attn_factor,
            beta_fast,
            beta_slow,
            rms_eps,
        )
    })
}

/// Compressor state initialisation for ratio-4 prefill.
pub fn metal_compressor_prefill_state_ratio4_tensor(
    state_kv: &ds4_metal_tensor,
    state_score: &ds4_metal_tensor,
    kv_tail: &ds4_metal_tensor,
    sc_tail: &ds4_metal_tensor,
    model_map: &[u8],
    ape_offset: u64,
    ape_type: u32,
    head_dim: u32,
    pos0: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_compressor_prefill_state_ratio4_tensor(
            state_kv as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            state_score as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            kv_tail as *const ds4_metal_tensor,
            sc_tail as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            ape_offset,
            ape_type,
            head_dim,
            pos0,
        )
    })
}

// ── Attention ─────────────────────────────────────────────────────────────

/// Decode step: compute attention heads from raw + compressed KV with optional
/// indexer mask.
#[allow(clippy::too_many_arguments)]
pub fn metal_attention_decode_heads_tensor(
    heads: &ds4_metal_tensor,
    model_map: &[u8],
    sinks_offset: u64,
    q: &ds4_metal_tensor,
    raw_kv: &ds4_metal_tensor,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    comp_kv: &ds4_metal_tensor,
    n_comp: u32,
    comp_mask: &ds4_metal_tensor,
    use_mask: u32,
    n_head: u32,
    head_dim: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_attention_decode_heads_tensor(
            heads as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            sinks_offset,
            q as *const ds4_metal_tensor,
            raw_kv as *const ds4_metal_tensor,
            n_raw,
            raw_cap,
            raw_start,
            comp_kv as *const ds4_metal_tensor,
            n_comp,
            comp_mask as *const ds4_metal_tensor,
            use_mask,
            n_head,
            head_dim,
        )
    })
}

/// Prefill: compute attention heads from raw KV only (SWA window).
pub fn metal_attention_prefill_raw_heads_tensor(
    heads: &ds4_metal_tensor,
    model_map: &[u8],
    sinks_offset: u64,
    q: &ds4_metal_tensor,
    raw_kv: &ds4_metal_tensor,
    n_tokens: u32,
    window: u32,
    n_head: u32,
    head_dim: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_attention_prefill_raw_heads_tensor(
            heads as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            sinks_offset,
            q as *const ds4_metal_tensor,
            raw_kv as *const ds4_metal_tensor,
            n_tokens,
            window,
            n_head,
            head_dim,
        )
    })
}

/// Batched decode: compute attention heads from raw KV only.
#[allow(clippy::too_many_arguments)]
pub fn metal_attention_decode_raw_batch_heads_tensor(
    heads: &ds4_metal_tensor,
    model_map: &[u8],
    sinks_offset: u64,
    q: &ds4_metal_tensor,
    raw_kv: &ds4_metal_tensor,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    window: u32,
    n_head: u32,
    head_dim: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_attention_decode_raw_batch_heads_tensor(
            heads as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            sinks_offset,
            q as *const ds4_metal_tensor,
            raw_kv as *const ds4_metal_tensor,
            n_tokens,
            pos0,
            n_raw,
            raw_cap,
            raw_start,
            window,
            n_head,
            head_dim,
        )
    })
}

/// Batched decode: compute attention heads from raw + compressed KV with
/// optional mask.
#[allow(clippy::too_many_arguments)]
pub fn metal_attention_decode_mixed_batch_heads_tensor(
    heads: &ds4_metal_tensor,
    model_map: &[u8],
    sinks_offset: u64,
    q: &ds4_metal_tensor,
    raw_kv: &ds4_metal_tensor,
    comp_kv: &ds4_metal_tensor,
    comp_mask: &ds4_metal_tensor,
    use_comp_mask: u32,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_attention_decode_mixed_batch_heads_tensor(
            heads as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            sinks_offset,
            q as *const ds4_metal_tensor,
            raw_kv as *const ds4_metal_tensor,
            comp_kv as *const ds4_metal_tensor,
            comp_mask as *const ds4_metal_tensor,
            use_comp_mask,
            n_tokens,
            pos0,
            n_raw,
            raw_cap,
            raw_start,
            n_comp,
            window,
            ratio,
            n_head,
            head_dim,
        )
    })
}

/// Batched decode: indexed mixed attention with top-k selection.
#[allow(clippy::too_many_arguments)]
pub fn metal_attention_indexed_mixed_batch_heads_tensor(
    heads: &ds4_metal_tensor,
    model_map: &[u8],
    sinks_offset: u64,
    q: &ds4_metal_tensor,
    raw_kv: &ds4_metal_tensor,
    comp_kv: &ds4_metal_tensor,
    topk: &ds4_metal_tensor,
    n_tokens: u32,
    pos0: u32,
    n_raw: u32,
    raw_cap: u32,
    raw_start: u32,
    n_comp: u32,
    top_k: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_attention_indexed_mixed_batch_heads_tensor(
            heads as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            sinks_offset,
            q as *const ds4_metal_tensor,
            raw_kv as *const ds4_metal_tensor,
            comp_kv as *const ds4_metal_tensor,
            topk as *const ds4_metal_tensor,
            n_tokens,
            pos0,
            n_raw,
            raw_cap,
            raw_start,
            n_comp,
            top_k,
            window,
            ratio,
            n_head,
            head_dim,
        )
    })
}

/// Prefill: static mixed attention with raw + compressed KV.
pub fn metal_attention_prefill_static_mixed_heads_tensor(
    heads: &ds4_metal_tensor,
    model_map: &[u8],
    sinks_offset: u64,
    q: &ds4_metal_tensor,
    raw_kv: &ds4_metal_tensor,
    comp_kv: &ds4_metal_tensor,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_attention_prefill_static_mixed_heads_tensor(
            heads as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            sinks_offset,
            q as *const ds4_metal_tensor,
            raw_kv as *const ds4_metal_tensor,
            comp_kv as *const ds4_metal_tensor,
            n_tokens,
            n_comp,
            window,
            ratio,
            n_head,
            head_dim,
        )
    })
}

/// Prefill: masked mixed attention with explicit compression mask.
#[allow(clippy::too_many_arguments)]
pub fn metal_attention_prefill_masked_mixed_heads_tensor(
    heads: &ds4_metal_tensor,
    model_map: &[u8],
    sinks_offset: u64,
    q: &ds4_metal_tensor,
    raw_kv: &ds4_metal_tensor,
    comp_kv: &ds4_metal_tensor,
    comp_mask: &ds4_metal_tensor,
    n_tokens: u32,
    n_comp: u32,
    window: u32,
    ratio: u32,
    n_head: u32,
    head_dim: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_attention_prefill_masked_mixed_heads_tensor(
            heads as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            sinks_offset,
            q as *const ds4_metal_tensor,
            raw_kv as *const ds4_metal_tensor,
            comp_kv as *const ds4_metal_tensor,
            comp_mask as *const ds4_metal_tensor,
            n_tokens,
            n_comp,
            window,
            ratio,
            n_head,
            head_dim,
        )
    })
}

/// Attention output projection (Q8 batch variant with low-rank).
#[allow(clippy::too_many_arguments)]
pub fn metal_attention_output_q8_batch_tensor(
    out: &ds4_metal_tensor,
    low: &ds4_metal_tensor,
    group_tmp: &ds4_metal_tensor,
    low_tmp: &ds4_metal_tensor,
    model_map: &[u8],
    out_a_offset: u64,
    out_b_offset: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    out_dim: u64,
    heads: &ds4_metal_tensor,
    n_tokens: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_attention_output_q8_batch_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            low as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            group_tmp as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            low_tmp as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            out_a_offset,
            out_b_offset,
            group_dim,
            rank,
            n_groups,
            out_dim,
            heads as *const ds4_metal_tensor,
            n_tokens,
        )
    })
}

/// Attention output low-rank Q8 projection.
pub fn metal_attention_output_low_q8_tensor(
    low: &ds4_metal_tensor,
    model_map: &[u8],
    out_a_offset: u64,
    group_dim: u64,
    rank: u64,
    n_groups: u32,
    heads: &ds4_metal_tensor,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_attention_output_low_q8_tensor(
            low as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            out_a_offset,
            group_dim,
            rank,
            n_groups,
            heads as *const ds4_metal_tensor,
        )
    })
}

// ── FFN / MoE ─────────────────────────────────────────────────────────────

/// SwiGLU activation: `out = (gate * sigmoid(gate * clamp)) * up * weight`.
pub fn metal_swiglu_tensor(
    out: &ds4_metal_tensor,
    gate: &ds4_metal_tensor,
    up: &ds4_metal_tensor,
    n: u32,
    clamp: f32,
    weight: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_swiglu_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            gate as *const ds4_metal_tensor,
            up as *const ds4_metal_tensor,
            n,
            clamp,
            weight,
        )
    })
}

/// Element-wise addition: `out = a + b`.
pub fn metal_add_tensor(
    out: &ds4_metal_tensor,
    a: &ds4_metal_tensor,
    b: &ds4_metal_tensor,
    n: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_add_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            a as *const ds4_metal_tensor,
            b as *const ds4_metal_tensor,
            n,
        )
    })
}

/// Router: select experts and compute weights for a single token.
#[allow(clippy::too_many_arguments)]
pub fn metal_router_select_tensor(
    selected: &ds4_metal_tensor,
    weights: &ds4_metal_tensor,
    probs: &ds4_metal_tensor,
    model_map: &[u8],
    bias_offset: u64,
    hash_offset: u64,
    hash_rows: u32,
    token: u32,
    n_expert_groups: u32,
    n_group_used: u32,
    has_bias: bool,
    hash_mode: bool,
    logits: &ds4_metal_tensor,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_router_select_tensor(
            selected as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            weights as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            probs as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            bias_offset,
            hash_offset,
            hash_rows,
            token,
            n_expert_groups,
            n_group_used,
            has_bias,
            hash_mode,
            logits as *const ds4_metal_tensor,
        )
    })
}

/// Router: select experts and compute weights for batched tokens.
#[allow(clippy::too_many_arguments)]
pub fn metal_router_select_batch_tensor(
    selected: &ds4_metal_tensor,
    weights: &ds4_metal_tensor,
    probs: &ds4_metal_tensor,
    model_map: &[u8],
    bias_offset: u64,
    hash_offset: u64,
    hash_rows: u32,
    n_expert_groups: u32,
    n_group_used: u32,
    has_bias: bool,
    hash_mode: bool,
    logits: &ds4_metal_tensor,
    tokens: &ds4_metal_tensor,
    n_tokens: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_router_select_batch_tensor(
            selected as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            weights as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            probs as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            bias_offset,
            hash_offset,
            hash_rows,
            n_expert_groups,
            n_group_used,
            has_bias,
            hash_mode,
            logits as *const ds4_metal_tensor,
            tokens as *const ds4_metal_tensor,
            n_tokens,
        )
    })
}

/// Routed MoE: single token, one expert with gate/up/down projections.
#[allow(clippy::too_many_arguments)]
pub fn metal_routed_moe_one_tensor(
    out: &ds4_metal_tensor,
    gate: &ds4_metal_tensor,
    up: &ds4_metal_tensor,
    mid: &ds4_metal_tensor,
    experts: &ds4_metal_tensor,
    model_map: &[u8],
    gate_offset: u64,
    up_offset: u64,
    down_offset: u64,
    gate_type: u32,
    down_type: u32,
    gate_expert_bytes: u64,
    gate_row_bytes: u64,
    down_expert_bytes: u64,
    down_row_bytes: u64,
    expert_in_dim: u32,
    expert_mid_dim: u32,
    out_dim: u32,
    selected: &ds4_metal_tensor,
    weights: &ds4_metal_tensor,
    n_expert: u32,
    clamp: f32,
    x: &ds4_metal_tensor,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_routed_moe_one_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            gate as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            up as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            mid as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            experts as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            gate_offset,
            up_offset,
            down_offset,
            gate_type,
            down_type,
            gate_expert_bytes,
            gate_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            out_dim,
            selected as *const ds4_metal_tensor,
            weights as *const ds4_metal_tensor,
            n_expert,
            clamp,
            x as *const ds4_metal_tensor,
        )
    })
}

/// Routed MoE: batched tokens with per-token routing.
#[allow(clippy::too_many_arguments)]
pub fn metal_routed_moe_batch_tensor(
    out: &ds4_metal_tensor,
    gate: &ds4_metal_tensor,
    up: &ds4_metal_tensor,
    mid: &ds4_metal_tensor,
    experts: &ds4_metal_tensor,
    model_map: &[u8],
    gate_offset: u64,
    up_offset: u64,
    down_offset: u64,
    gate_type: u32,
    down_type: u32,
    gate_expert_bytes: u64,
    gate_row_bytes: u64,
    down_expert_bytes: u64,
    down_row_bytes: u64,
    expert_in_dim: u32,
    expert_mid_dim: u32,
    out_dim: u32,
    selected: &ds4_metal_tensor,
    weights: &ds4_metal_tensor,
    n_expert: u32,
    clamp: f32,
    x: &ds4_metal_tensor,
    n_tokens: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_routed_moe_batch_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            gate as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            up as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            mid as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            experts as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            gate_offset,
            up_offset,
            down_offset,
            gate_type,
            down_type,
            gate_expert_bytes,
            gate_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            expert_in_dim,
            expert_mid_dim,
            out_dim,
            selected as *const ds4_metal_tensor,
            weights as *const ds4_metal_tensor,
            n_expert,
            clamp,
            x as *const ds4_metal_tensor,
            n_tokens,
        )
    })
}

// ── Hyper-Connection Kernels ──────────────────────────────────────────────

/// HC sinkhorn split: compute routing weights from the mixer.
pub fn metal_hc_split_sinkhorn_tensor(
    out: &ds4_metal_tensor,
    mix: &ds4_metal_tensor,
    model_map: &[u8],
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_hc_split_sinkhorn_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            mix as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            scale_offset,
            base_offset,
            n_hc,
            sinkhorn_iters,
            eps,
        )
    })
}

/// HC weighted sum: reduce four HC streams into one sublayer row.
pub fn metal_hc_weighted_sum_tensor(
    out: &ds4_metal_tensor,
    residual_hc: &ds4_metal_tensor,
    weights: &ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_hc_weighted_sum_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            residual_hc as *const ds4_metal_tensor,
            weights as *const ds4_metal_tensor,
            n_embd,
            n_hc,
        )
    })
}

/// HC weighted sum with pre-computed split weights.
pub fn metal_hc_weighted_sum_split_tensor(
    out: &ds4_metal_tensor,
    residual_hc: &ds4_metal_tensor,
    split: &ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_hc_weighted_sum_split_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            residual_hc as *const ds4_metal_tensor,
            split as *const ds4_metal_tensor,
            n_embd,
            n_hc,
        )
    })
}

/// HC fused split + weighted sum (decode-optimised).
#[allow(clippy::too_many_arguments)]
pub fn metal_hc_split_weighted_sum_tensor(
    out: &ds4_metal_tensor,
    split: &ds4_metal_tensor,
    mix: &ds4_metal_tensor,
    residual_hc: &ds4_metal_tensor,
    model_map: &[u8],
    scale_offset: u64,
    base_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_hc_split_weighted_sum_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            split as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            mix as *const ds4_metal_tensor,
            residual_hc as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            scale_offset,
            base_offset,
            n_embd,
            n_hc,
            sinkhorn_iters,
            eps,
        )
    })
}

/// HC fused split + weighted sum with RMS norm on the active row.
#[allow(clippy::too_many_arguments)]
pub fn metal_hc_split_weighted_sum_norm_tensor(
    out: &ds4_metal_tensor,
    norm_out: &ds4_metal_tensor,
    split: &ds4_metal_tensor,
    mix: &ds4_metal_tensor,
    residual_hc: &ds4_metal_tensor,
    model_map: &[u8],
    scale_offset: u64,
    base_offset: u64,
    norm_weight_offset: u64,
    n_embd: u32,
    n_hc: u32,
    sinkhorn_iters: u32,
    eps: f32,
    norm_eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_hc_split_weighted_sum_norm_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            norm_out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            split as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            mix as *const ds4_metal_tensor,
            residual_hc as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            scale_offset,
            base_offset,
            norm_weight_offset,
            n_embd,
            n_hc,
            sinkhorn_iters,
            eps,
            norm_eps,
        )
    })
}

/// HC output weights: compute per-stream output weights from the pre-activation.
pub fn metal_output_hc_weights_tensor(
    out: &ds4_metal_tensor,
    pre: &ds4_metal_tensor,
    model_map: &[u8],
    scale_offset: u64,
    base_offset: u64,
    n_hc: u32,
    eps: f32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_output_hc_weights_tensor(
            out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            pre as *const ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            scale_offset,
            base_offset,
            n_hc,
            eps,
        )
    })
}

/// HC expand: expand a sublayer output back into the four HC streams.
pub fn metal_hc_expand_tensor(
    out_hc: &ds4_metal_tensor,
    block_out: &ds4_metal_tensor,
    residual_hc: &ds4_metal_tensor,
    post: &ds4_metal_tensor,
    comb: &ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_hc_expand_tensor(
            out_hc as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            block_out as *const ds4_metal_tensor,
            residual_hc as *const ds4_metal_tensor,
            post as *const ds4_metal_tensor,
            comb as *const ds4_metal_tensor,
            n_embd,
            n_hc,
        )
    })
}

/// HC expand with pre-computed split weights.
pub fn metal_hc_expand_split_tensor(
    out_hc: &ds4_metal_tensor,
    block_out: &ds4_metal_tensor,
    residual_hc: &ds4_metal_tensor,
    split: &ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_hc_expand_split_tensor(
            out_hc as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            block_out as *const ds4_metal_tensor,
            residual_hc as *const ds4_metal_tensor,
            split as *const ds4_metal_tensor,
            n_embd,
            n_hc,
        )
    })
}

/// HC expand with an additional block add term.
pub fn metal_hc_expand_add_split_tensor(
    out_hc: &ds4_metal_tensor,
    block_out: &ds4_metal_tensor,
    block_add: &ds4_metal_tensor,
    residual_hc: &ds4_metal_tensor,
    split: &ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_hc_expand_add_split_tensor(
            out_hc as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            block_out as *const ds4_metal_tensor,
            block_add as *const ds4_metal_tensor,
            residual_hc as *const ds4_metal_tensor,
            split as *const ds4_metal_tensor,
            n_embd,
            n_hc,
        )
    })
}

/// Shared FFN down projection + HC expand (Q8_0), with routed expert
/// contribution.
#[allow(clippy::too_many_arguments)]
pub fn metal_shared_down_hc_expand_q8_0_tensor(
    out_hc: &ds4_metal_tensor,
    shared_out: &ds4_metal_tensor,
    model_map: &[u8],
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    shared_mid: &ds4_metal_tensor,
    routed_out: &ds4_metal_tensor,
    residual_hc: &ds4_metal_tensor,
    split: &ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_shared_down_hc_expand_q8_0_tensor(
            out_hc as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            shared_out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_offset,
            in_dim,
            out_dim,
            shared_mid as *const ds4_metal_tensor,
            routed_out as *const ds4_metal_tensor,
            residual_hc as *const ds4_metal_tensor,
            split as *const ds4_metal_tensor,
            n_embd,
            n_hc,
        )
    })
}

/// Q8_0 matmul + HC expand fused kernel.
#[allow(clippy::too_many_arguments)]
pub fn metal_matmul_q8_0_hc_expand_tensor(
    out_hc: &ds4_metal_tensor,
    block_out: &ds4_metal_tensor,
    model_map: &[u8],
    weight_offset: u64,
    in_dim: u64,
    out_dim: u64,
    x: &ds4_metal_tensor,
    residual_hc: &ds4_metal_tensor,
    split: &ds4_metal_tensor,
    n_embd: u32,
    n_hc: u32,
) -> MetalResult<()> {
    ck(unsafe {
        ds4_metal_matmul_q8_0_hc_expand_tensor(
            out_hc as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            block_out as *const ds4_metal_tensor as *mut ds4_metal_tensor,
            model_map.as_ptr() as *const c_void,
            model_map.len() as u64,
            weight_offset,
            in_dim,
            out_dim,
            x as *const ds4_metal_tensor,
            residual_hc as *const ds4_metal_tensor,
            split as *const ds4_metal_tensor,
            n_embd,
            n_hc,
        )
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_not_initialized_by_default() {
        assert!(!is_initialized());
    }

    #[test]
    fn test_metal_tensor_zero_sized() {
        // Verify the opaque type has zero size (it's a ZST).
        assert_eq!(std::mem::size_of::<ds4_metal_tensor>(), 0);
    }
}
