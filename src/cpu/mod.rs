//! CPU reference implementations of all math kernels used by the DS4 inference
//! engine.  This module maps to the CPU reference path from `ds4.c` and is the
//! fallback / debug path when Metal is unavailable or when running unit tests.
//!
//! Layout
//! ------
//! - Float / half helpers
//! - RMS norm family
//! - Activations (SiLU, softplus, sigmoid, SwiGLU)
//! - Basic BLAS (dot, axpy, scale)
//! - Quantization (Q8_0)
//! - Matrix-vector (f16, Q8_0, Q2_K, IQ2_XXS)
//! - RoPE
//! - FP8 KV quantization
//! - HC (head control) stream helpers
//! - Top-K
//! - Softmax
//! - Debug printing
//!
//! ARM NEON intrinsics are used on aarch64 where they materially improve
//! throughput (dot product, quantised matvec paths).  The scalar fallback is
//! correct and kept for non-ARM targets and cross-checking.

#![allow(non_snake_case)]
#![allow(dead_code)]

use crate::types::{DS4_NEG_INF, DS4_POS_INF, QK_K};

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

// ── QK_K block count helper ───────────────────────────────────────────────

/// Number of QK_K-sized blocks needed to cover `in_dim` elements.
#[inline(always)]
pub fn q8k_blocks(in_dim: u64) -> u64 {
    (in_dim + QK_K as u64 - 1) / QK_K as u64
}

// ── GGUF quantised block structs ──────────────────────────────────────────

/// Q2_K block – 84 bytes, 256 elements.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct block_q2_K {
    pub scales: [u8; QK_K as usize / 16], // 16
    pub qs: [u8; QK_K as usize / 4],      // 64
    pub d: u16,                           // scale
    pub dmin: u16,                        // min scale
}

/// Q4_K block – 144 bytes, 256 elements.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct block_q4_K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; 12],
    pub qs: [u8; QK_K as usize / 2], // 128
}

/// Q8_K block – 288 bytes, 256 elements.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct block_q8_K {
    pub d: f32,
    pub qs: [i8; QK_K as usize],          // 256
    pub bsums: [i16; QK_K as usize / 16], // 16
}

/// IQ2_XXS block – 66 bytes, 256 elements.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct block_iq2_xxs {
    pub d: u16,
    pub qs: [u16; QK_K as usize / 8], // 32
}

// ── IQ2_XXS lookup tables ─────────────────────────────────────────────────

static KMASK_IQ2XS: [u8; 8] = [1, 2, 4, 8, 16, 32, 64, 128];

static KSIGNS_IQ2XS: [u8; 128] = [
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149,
    150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170,
    43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63,
    192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212,
    85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232,
    105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123,
    252, 125, 126, 255,
];

static IQ2XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

/// Precomputed sign masks for IQ2_XXS: 128 sign patterns, each 8 bytes.
static IQ2XXS_SIGNS: [[i8; 8]; 128] = {
    let mut signs: [[i8; 8]; 128] = [[0i8; 8]; 128];
    let mut s: usize = 0;
    while s < 128 {
        let k = KSIGNS_IQ2XS[s];
        let mut j: usize = 0;
        while j < 8 {
            signs[s][j] = if (k & KMASK_IQ2XS[j]) != 0 { -1i8 } else { 1i8 };
            j += 1;
        }
        s += 1;
    }
    signs
};

/// Precomputed signed grid for IQ2_XXS: `[grid_index][sign_index][8]`.
static IQ2XXS_SIGNED_GRID: [[[i8; 8]; 128]; 256] = {
    let mut sg = [[[0i8; 8]; 128]; 256];
    let mut g: usize = 0;
    while g < 256 {
        let grid_bytes = IQ2XXS_GRID[g].to_ne_bytes(); // host-order bytes
        let mut s: usize = 0;
        while s < 128 {
            let k = KSIGNS_IQ2XS[s];
            let mut j: usize = 0;
            while j < 8 {
                let v = grid_bytes[j] as i8;
                sg[g][s][j] = if (k & KMASK_IQ2XS[j]) != 0 { -v } else { v };
                j += 1;
            }
            s += 1;
        }
        g += 1;
    }
    sg
};

// ── Float / half-precision helpers ────────────────────────────────────────

/// Convert an IEEE 754 binary16 (stored in a `u16`) to `f32`.
pub fn f16_to_f32(h: u16) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        // NEON: use the vcvt hardware instruction.
        // We need to construct a float16x4_t from one lane, convert, extract.
        unsafe {
            // vdup_n_u16 + vreinterpret + vcvt_f32_f16
            let hv: float16x4_t = vreinterpret_f16_u16(vdup_n_u16(h));
            let fv: float32x4_t = vcvt_f32_f16(hv);
            vgetq_lane_f32::<0>(fv)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let sign: u32 = ((h as u32) & 0x8000) << 16;
        let exp: u32 = ((h >> 10) & 0x1f) as u32;
        let mant: u32 = (h & 0x03ff) as u32;

        let bits: u32;
        if exp == 0 {
            if mant == 0 {
                bits = sign;
            } else {
                // Subnormal: normalize
                let mut e = 1i32;
                let mut m = mant;
                while (m & 0x0400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                m &= 0x03ff;
                bits = sign | (((e + 127 - 15) as u32) << 23) | (m << 13);
            }
        } else if exp == 31 {
            bits = sign | 0x7f800000u32 | (mant << 13);
        } else {
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
        f32::from_bits(bits)
    }
}

/// Convert an `f32` to IEEE 754 binary16 (stored in a `u16`).
pub fn f32_to_f16(f: f32) -> u16 {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let fv: float32x4_t = vdupq_n_f32(f);
            let hv: float16x4_t = vcvt_f16_f32(fv);
            vget_lane_u16::<0>(vreinterpret_u16_f16(hv))
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let bits: u32 = f.to_bits();
        let sign: u16 = ((bits >> 16) & 0x8000) as u16;
        let exp: i32 = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant: u32 = bits & 0x7fffffu;

        if exp <= 0 {
            if exp < -10 {
                return sign;
            }
            let m: u32 = mant | 0x800000u32;
            let shift: u32 = (14 - exp) as u32;
            let half_mant: u32 = m >> shift;
            let round_bit: u32 = (m >> (shift - 1)) & 1u32;
            let sticky: u32 = m & ((1u32 << (shift - 1)) - 1);
            let hm = if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
                half_mant + 1
            } else {
                half_mant
            };
            return sign | (hm as u16);
        }

        if exp >= 31 {
            if ((bits >> 23) & 0xff) == 0xff && mant != 0 {
                return sign | 0x7e00u16;
            }
            return sign | 0x7c00u16;
        }

        let mut half: u32 = sign as u32 | ((exp as u32) << 10) | (mant >> 13);
        let round: u32 = mant & 0x1fffu;
        if round > 0x1000 || (round == 0x1000 && (half & 1) != 0) {
            half += 1;
        }
        half as u16
    }
}

/// Round-trip every element of `x` through f16 conversion (in-place).
pub fn f16_round_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = f16_to_f32(f32_to_f16(*v));
    }
}

// ── RMS norm ──────────────────────────────────────────────────────────────

/// RMS normalisation **without** learned per-channel scale.
///
/// `out[i] = x[i] / sqrt(mean(x²) + eps)`
pub fn rms_norm_no_weight(out: &mut [f32], x: &[f32], eps: f32) {
    let n = x.len();
    let mut ss: f64 = 0.0;
    for i in 0..n {
        ss += (x[i] as f64) * (x[i] as f64);
    }
    let scale = 1.0 / ((ss / n as f64) as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale;
    }
}

/// RMS normalisation **with** learned per-channel scale.
///
/// `out[i] = x[i] * weight[i] / sqrt(mean(x²) + eps)`
pub fn rms_norm_weight(out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut ss: f64 = 0.0;
    for i in 0..n {
        ss += (x[i] as f64) * (x[i] as f64);
    }
    let scale = 1.0 / ((ss / n as f64) as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = x[i] * scale * weight[i];
    }
}

/// Per-head RMS normalisation applied in-place.
///
/// Each contiguous block of `head_dim` elements is normalised independently.
pub fn head_rms_norm_inplace(x: &mut [f32], n_head: u32, head_dim: u32, eps: f32) {
    let hd = head_dim as usize;
    for h in 0..n_head as usize {
        let off = h * hd;
        let head = &mut x[off..off + hd];
        let mut ss: f64 = 0.0;
        for i in 0..hd {
            ss += (head[i] as f64) * (head[i] as f64);
        }
        let scale = 1.0 / ((ss / hd as f64) as f32 + eps).sqrt();
        for i in 0..hd {
            head[i] *= scale;
        }
    }
}

// ── Activations ───────────────────────────────────────────────────────────

/// Sigmoid implemented in a numerically stable way.
pub fn sigmoid_stable(x: f32) -> f32 {
    if x >= 0.0 {
        let e = (-x).exp();
        1.0 / (1.0 + e)
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// SiLU (Swish-1) activation: `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x * sigmoid_stable(x)
}

/// Stable softplus: `log(1 + exp(x))`, avoiding overflow for large |x|.
pub fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// SwiGLU fusion: `out[i] = silu(gate[i]) * up[i]`.
pub fn swiglu(out: &mut [f32], gate: &[f32], up: &[f32]) {
    let n = out.len();
    for i in 0..n {
        out[i] = silu(gate[i]) * up[i];
    }
}

// ── Basic BLAS ────────────────────────────────────────────────────────────

/// Dot product between two `f32` slices.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let mut i = 0usize;
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            while i + 8 <= n {
                acc0 = vfmaq_f32(
                    acc0,
                    vld1q_f32(a.as_ptr().add(i)),
                    vld1q_f32(b.as_ptr().add(i)),
                );
                acc1 = vfmaq_f32(
                    acc1,
                    vld1q_f32(a.as_ptr().add(i + 4)),
                    vld1q_f32(b.as_ptr().add(i + 4)),
                );
                i += 8;
            }
            let mut acc = vaddvq_f32(vaddq_f32(acc0, acc1));
            while i < n {
                acc += a[i] * b[i];
                i += 1;
            }
            acc
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut acc = 0.0f32;
        for i in 0..n {
            acc += a[i] * b[i];
        }
        acc
    }
}

/// `y += a * x` (AXPY).
pub fn axpy_f32(y: &mut [f32], x: &[f32], a: f32) {
    let n = y.len();
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let av = vdupq_n_f32(a);
            let mut i = 0usize;
            while i + 8 <= n {
                let yy0 = vld1q_f32(y.as_ptr().add(i));
                let yy1 = vld1q_f32(y.as_ptr().add(i + 4));
                let xx0 = vld1q_f32(x.as_ptr().add(i));
                let xx1 = vld1q_f32(x.as_ptr().add(i + 4));
                vst1q_f32(y.as_mut_ptr().add(i), vfmaq_f32(yy0, av, xx0));
                vst1q_f32(y.as_mut_ptr().add(i + 4), vfmaq_f32(yy1, av, xx1));
                i += 8;
            }
            while i < n {
                y[i] += a * x[i];
                i += 1;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            y[i] += a * x[i];
        }
    }
}

/// `x *= a` (scale in-place).
pub fn scale_f32(x: &mut [f32], a: f32) {
    let n = x.len();
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let av = vdupq_n_f32(a);
            let mut i = 0usize;
            while i + 8 <= n {
                vst1q_f32(
                    x.as_mut_ptr().add(i),
                    vmulq_f32(vld1q_f32(x.as_ptr().add(i)), av),
                );
                vst1q_f32(
                    x.as_mut_ptr().add(i + 4),
                    vmulq_f32(vld1q_f32(x.as_ptr().add(i + 4)), av),
                );
                i += 8;
            }
            while i < n {
                x[i] *= a;
                i += 1;
            }
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        for i in 0..n {
            x[i] *= a;
        }
    }
}

// ── Q8_0 quantisation ─────────────────────────────────────────────────────

/// Q8_0 quantisation: split `x` into blocks of 32 elements, find the
/// absolute-maximum scale per block, and store the quantised i8 values
/// together with the per-block `f32` scale factors.
///
/// Returns `(xq, xscale)` where `xq` has length `blocks * 32` and
/// `xscale` has length `blocks`.
pub fn quantize_q8_0(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n = x.len();
    let blocks = (n + 31) / 32;
    let mut xq = vec![0i8; blocks * 32];
    let mut xscale = vec![0.0f32; blocks];

    for b in 0..blocks {
        let i0 = b * 32;
        let bn = if n - i0 < 32 { n - i0 } else { 32 };
        let chunk = &x[i0..i0 + bn];

        let mut amax = 0.0f32;
        for &v in chunk.iter() {
            let ax = v.abs();
            if ax > amax {
                amax = ax;
            }
        }

        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        xscale[b] = d;

        for (j, &v) in chunk.iter().enumerate() {
            let vi = (v * id).round() as i32;
            let clamped = if vi > 127 {
                127
            } else if vi < -128 {
                -128
            } else {
                vi
            };
            xq[i0 + j] = clamped as i8;
        }
        // Zero-pad trailing lanes in the last block
        for j in bn..32 {
            xq[i0 + j] = 0;
        }
    }

    (xq, xscale)
}

// ── Q8_0 dot product (row from weight tensor × quantised activation) ──────

/// Inner product between a Q8_0 weight row (stored as packed `u8` — first 2
/// bytes are the f16 scale, then 32 × i8 values) and a pre-quantised activation
/// vector `(xq, xscale)`.
///
/// `row` layout per block: `[ f16_scale (2 bytes) | qs[0..32] (32 bytes) ]`
/// repeated for each 32-element block of `in_dim`.
/// NEON helper: dot product of two 16×i8 vectors using stable intrinsics.
/// Returns the sum of all 32 partial products as an i32.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_dot_i8x16(a: int8x16_t, b: int8x16_t) -> i32 {
    // vmull_s8 gives 16×i16 partial products (widening multiply)
    let p0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    let p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    // vpaddlq_s16 extends to i32 pairwise sums
    let s0 = vpaddlq_s16(p0);
    let s1 = vpaddlq_s16(p1);
    vaddvq_s32(vaddq_s32(s0, s1))
}

pub fn dot_q8_0_row(row: &[u8], xq: &[i8], xscale: &[f32], in_dim: u64) -> f32 {
    let blocks = (in_dim + 31) / 32;

    #[cfg(target_arch = "aarch64")]
    {
        if (in_dim & 31) == 0 {
            unsafe {
                let mut accv0 = vdupq_n_f32(0.0);
                let mut accv1 = vdupq_n_f32(0.0);
                let mut b = 0u64;

                while b + 1 < blocks {
                    // Two blocks at a time
                    let sb0_bytes: &[u8; 2] = row[(b * 34) as usize..][..2].try_into().unwrap();
                    let sb1_bytes: &[u8; 2] =
                        row[((b + 1) * 34) as usize..][..2].try_into().unwrap();
                    let scale_bits0 = u16::from_le_bytes(*sb0_bytes);
                    let scale_bits1 = u16::from_le_bytes(*sb1_bytes);

                    let qs0_ptr = row[(b * 34 + 2) as usize..].as_ptr() as *const i8;
                    let qs1_ptr = row[((b + 1) * 34 + 2) as usize..].as_ptr() as *const i8;
                    let xq0_ptr = xq.as_ptr().add((b * 32) as usize);
                    let xq1_ptr = xq.as_ptr().add(((b + 1) * 32) as usize);

                    let qs0_0 = vld1q_s8(qs0_ptr);
                    let qs0_1 = vld1q_s8(qs0_ptr.add(16));
                    let xq0_0 = vld1q_s8(xq0_ptr);
                    let xq0_1 = vld1q_s8(xq0_ptr.add(16));
                    let dot0 = neon_dot_i8x16(qs0_0, xq0_0) + neon_dot_i8x16(qs0_1, xq0_1);
                    let fdot0 = dot0 as f32;

                    let qs1_0 = vld1q_s8(qs1_ptr);
                    let qs1_1 = vld1q_s8(qs1_ptr.add(16));
                    let xq1_0 = vld1q_s8(xq1_ptr);
                    let xq1_1 = vld1q_s8(xq1_ptr.add(16));
                    let dot1 = neon_dot_i8x16(qs1_0, xq1_0) + neon_dot_i8x16(qs1_1, xq1_1);
                    let fdot1 = dot1 as f32;

                    let s0 = f16_to_f32(scale_bits0) * xscale[b as usize];
                    let s1 = f16_to_f32(scale_bits1) * xscale[(b + 1) as usize];
                    accv0 = vfmaq_n_f32(accv0, vdupq_n_f32(fdot0), s0);
                    accv1 = vfmaq_n_f32(accv1, vdupq_n_f32(fdot1), s1);

                    b += 2;
                }

                if b < blocks {
                    let sb_bytes: &[u8; 2] = row[(b * 34) as usize..][..2].try_into().unwrap();
                    let scale_bits = u16::from_le_bytes(*sb_bytes);
                    let qs_ptr = row[(b * 34 + 2) as usize..].as_ptr() as *const i8;
                    let xq_ptr = xq.as_ptr().add((b * 32) as usize);

                    let qs_0 = vld1q_s8(qs_ptr);
                    let qs_1 = vld1q_s8(qs_ptr.add(16));
                    let xq_0 = vld1q_s8(xq_ptr);
                    let xq_1 = vld1q_s8(xq_ptr.add(16));
                    let dot = neon_dot_i8x16(qs_0, xq_0) + neon_dot_i8x16(qs_1, xq_1);
                    let fdot = dot as f32;

                    let s = f16_to_f32(scale_bits) * xscale[b as usize];
                    accv0 = vfmaq_n_f32(accv0, vdupq_n_f32(fdot), s);
                }

                vaddvq_f32(vaddq_f32(accv0, accv1))
            }
        } else {
            // Fallback for unaligned dims
            scalar_dot_q8_0_row(row, xq, xscale, in_dim, blocks)
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        scalar_dot_q8_0_row(row, xq, xscale, in_dim, blocks)
    }
}

/// Scalar fallback for `dot_q8_0_row`.
fn scalar_dot_q8_0_row(row: &[u8], xq: &[i8], xscale: &[f32], in_dim: u64, blocks: u64) -> f32 {
    let mut acc = 0.0f32;
    for b in 0..blocks {
        let sb_bytes: &[u8; 2] = row[(b * 34) as usize..][..2].try_into().unwrap();
        let scale_bits = u16::from_le_bytes(*sb_bytes);
        let qs_ptr = row[(b * 34 + 2) as usize..].as_ptr() as *const i8;

        let i0 = (b * 32) as usize;
        let n = if in_dim as usize - i0 < 32 {
            in_dim as usize - i0
        } else {
            32
        };

        let mut dot = 0i32;
        // SAFETY: the pointers and lengths are in-bounds by construction.
        unsafe {
            for j in 0..n {
                dot += (*qs_ptr.add(j) as i32) * (*xq.as_ptr().add(i0 + j) as i32);
            }
        }
        acc += f16_to_f32(scale_bits) * xscale[b as usize] * (dot as f32);
    }
    acc
}

// ── F16 matrix-vector multiply ────────────────────────────────────────────

/// Dense F16 matrix-vector multiply.
///
/// `out[o] = sum_j f16_to_f32(data[o * in_dim + j]) * x[j]`
pub fn matvec_f16(out: &mut [f32], data: &[u16], x: &[f32], in_dim: u64) {
    let out_dim = out.len();
    for o in 0..out_dim {
        let row = &data[(o as u64 * in_dim) as usize..][..in_dim as usize];
        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                let mut i = 0usize;
                let mut acc0 = vdupq_n_f32(0.0);
                let mut acc1 = vdupq_n_f32(0.0);
                let n = in_dim as usize;
                while i + 8 <= n {
                    let hv = vreinterpretq_f16_u16(vld1q_u16(row.as_ptr().add(i)));
                    let h0 = vcvt_f32_f16(vget_low_f16(hv));
                    let h1 = vcvt_f32_f16(vget_high_f16(hv));
                    let x0 = vld1q_f32(x.as_ptr().add(i));
                    let x1 = vld1q_f32(x.as_ptr().add(i + 4));
                    acc0 = vfmaq_f32(acc0, h0, x0);
                    acc1 = vfmaq_f32(acc1, h1, x1);
                    i += 8;
                }
                let mut acc = vaddvq_f32(vaddq_f32(acc0, acc1));
                while i < n {
                    acc += f16_to_f32(row[i]) * x[i];
                    i += 1;
                }
                out[o] = acc;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut acc = 0.0f32;
            for i in 0..in_dim as usize {
                acc += f16_to_f32(row[i]) * x[i];
            }
            out[o] = acc;
        }
    }
}

// ── Q8_0 matrix-vector multiply ───────────────────────────────────────────

/// Q8_0 matrix-vector multiply with pre-quantised activation.
///
/// `data` is the weight matrix in Q8_0 packed format (34 bytes per block per
/// output row).  `xq` and `xscale` are the pre-quantised input vector.
pub fn matvec_q8_0(out: &mut [f32], data: &[u8], xq: &[i8], xscale: &[f32], in_dim: u64) {
    let out_dim = out.len();
    let blocks = (in_dim + 31) / 32;
    let row_stride = (blocks * 34) as usize;

    for o in 0..out_dim {
        let row = &data[o * row_stride..][..row_stride];
        out[o] = dot_q8_0_row(row, xq, xscale, in_dim);
    }
}

// ── RoPE (Rotary Position Embedding) ──────────────────────────────────────

fn rope_yarn_ramp(low: f32, high: f32, i0: i32) -> f32 {
    let y = ((i0 / 2) as f32 - low) / (0.001f32).max(high - low);
    1.0 - (1.0f32).min((0.0f32).max(y))
}

fn rope_yarn_corr_dim(n_dims: i32, n_ctx_orig: u64, n_rot: f32, base: f32) -> f32 {
    (n_dims as f32) * ((n_ctx_orig as f32) / (n_rot * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * base.ln())
}

fn rope_yarn_corr_dims(
    n_dims: i32,
    n_ctx_orig: u64,
    freq_base: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> [f32; 2] {
    let start = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base).floor();
    let end = rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base).ceil();
    [start.max(0.0), end.min((n_dims - 1) as f32)]
}

/// Apply / inverse RoPE to the tail of each attention head.
///
/// The rotary embedding is applied only to the last `n_rot` dimensions of
/// each head (the "tail"); the first `head_dim - n_rot` dimensions are left
/// unchanged.  This matches DeepSeek V4's "tail RoPE" convention.
///
/// `inverse=true` rotates backwards (used when de-rotating the attention
/// output before the grouped output projection).
pub fn rope_tail_layer(
    x: &mut [f32],
    n_head: u32,
    head_dim: u32,
    n_rot: u32,
    pos: u32,
    il: u32,
    inverse: bool,
) {
    // Constants from the engine configuration (hard-coded for DS4)
    const DS4_ROPE_FREQ_BASE: f32 = 10000.0;
    const DS4_ROPE_SCALE_FACTOR: f32 = 16.0;
    const DS4_ROPE_YARN_BETA_FAST: f32 = 32.0;
    const DS4_ROPE_YARN_BETA_SLOW: f32 = 1.0;
    const DS4_COMPRESS_ROPE_FREQ_BASE: f32 = 160000.0;
    const DS4_ROPE_ORIG_CTX: u64 = 65536;
    const DS4_LAYER_COMPRESS_RATIO_TABLE: [u32; 43] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    ];
    // HACK for small models / testing: fallback to 0 if il is out of range
    let compress_ratio = if (il as usize) < DS4_LAYER_COMPRESS_RATIO_TABLE.len() {
        DS4_LAYER_COMPRESS_RATIO_TABLE[il as usize]
    } else {
        0
    };
    let compressed = compress_ratio != 0;

    let freq_base = if compressed && DS4_COMPRESS_ROPE_FREQ_BASE > 0.0 {
        DS4_COMPRESS_ROPE_FREQ_BASE
    } else {
        DS4_ROPE_FREQ_BASE
    };

    let freq_scale = if !compressed || DS4_ROPE_SCALE_FACTOR <= 0.0 {
        1.0
    } else {
        1.0 / DS4_ROPE_SCALE_FACTOR
    };

    let ext_factor = if compressed && DS4_ROPE_SCALE_FACTOR > 1.0 {
        1.0
    } else {
        0.0
    };

    let mut attn_factor = 1.0;
    if ext_factor != 0.0 && freq_scale > 0.0 {
        attn_factor /= 1.0 + 0.1 * (1.0 / freq_scale).ln();
    }

    let n_nope = (head_dim - n_rot) as usize;
    let theta_scale = freq_base.powf(-2.0 / n_rot as f32);
    let sin_sign = if inverse { -1.0 } else { 1.0 };

    let corr_dims = if ext_factor != 0.0 {
        rope_yarn_corr_dims(
            n_rot as i32,
            if compressed { DS4_ROPE_ORIG_CTX } else { 0 },
            freq_base,
            DS4_ROPE_YARN_BETA_FAST,
            DS4_ROPE_YARN_BETA_SLOW,
        )
    } else {
        [0.0, 0.0]
    };

    let hd = head_dim as usize;
    for h in 0..n_head as usize {
        let tail_off = h * hd + n_nope;
        let tail = &mut x[tail_off..tail_off + n_rot as usize];
        let mut theta_extrap = pos as f32;

        for i in (0..n_rot as usize).step_by(2) {
            let theta_interp = freq_scale * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = attn_factor;

            if ext_factor != 0.0 {
                let ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i as i32) * ext_factor;
                theta = theta_interp * (1.0 - ramp_mix) + theta_extrap * ramp_mix;
                mscale *= 1.0 + 0.1 * (1.0 / freq_scale).ln();
            }

            let cos_t = theta.cos() * mscale;
            let sin_t = sin_sign * theta.sin() * mscale;

            let x0 = tail[i];
            let x1 = tail[i + 1];
            tail[i] = x0 * cos_t - x1 * sin_t;
            tail[i + 1] = x0 * sin_t + x1 * cos_t;

            theta_extrap *= theta_scale;
        }
    }
}

// ── FP8 KV quantisation (E4M3 round-trip) ─────────────────────────────────

/// E4M3 value lookup for a 7-bit index (1 sign + 3 exponent + 3 mantissa).
fn dsv4_e4m3fn_value(i: usize) -> f32 {
    const EXP_SCALE: [f32; 16] = [
        0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
        128.0, 256.0,
    ];
    let exp = (i >> 3) & 0x0f;
    let mant = i & 0x07;
    if exp == 0 {
        mant as f32 * 0.001953125
    } else {
        (1.0 + mant as f32 * 0.125) * EXP_SCALE[exp]
    }
}

/// Dequantise a value through the nearest E4M3 representation.
fn dsv4_e4m3fn_dequant(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(448.0);

    let mut lo = 0usize;
    let mut hi = 126usize;
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if dsv4_e4m3fn_value(mid) <= ax {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    let mut best = lo;
    if best < 126 {
        let best_diff = (ax - dsv4_e4m3fn_value(best)).abs();
        let next_diff = (ax - dsv4_e4m3fn_value(best + 1)).abs();
        if next_diff < best_diff
            || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)
        {
            best += 1;
        }
    }

    sign * dsv4_e4m3fn_value(best)
}

/// Apply FP8 (E4M3) round-trip quantisation to the non-RoPE part of each head.
///
/// The first `n_rot` dimensions of each head (the RoPE tail) are left
/// untouched.  The remaining `head_dim - n_rot` dimensions are processed in
/// blocks of 64 with per-block scaling.
pub fn dsv4_fp8_kv_quantize_row_inplace(x: &mut [f32], head_dim: u32, n_rot: u32) {
    let n_nope = (head_dim - n_rot) as usize;
    for off in (0..n_nope).step_by(64) {
        // Find block max
        let block_end = (off + 64).min(n_nope);
        let block = &x[off..block_end];
        let mut amax = 0.0f32;
        for &v in block.iter() {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }

        if amax < 1.0e-4 {
            amax = 1.0e-4;
        }
        let scale = (2.0f32).powf((amax / 448.0).log2().ceil());
        for v in x[off..block_end].iter_mut() {
            let scaled = *v / scale;
            let clamped = if scaled > 448.0 {
                448.0
            } else if scaled < -448.0 {
                -448.0
            } else {
                scaled
            };
            *v = dsv4_e4m3fn_dequant(clamped) * scale;
        }
    }
}

// ── HC (Head Control) stream helpers ──────────────────────────────────────

/// HC Sinkhorn transform: decode the control projection into pre-weights,
/// post-gates, and a doubly-normalised combine matrix.
///
/// `out` has size `2 * n_hc + n_hc * n_hc`.
/// - `out[0..n_hc]` — HC pre weights (sigmoid + eps)
/// - `out[n_hc..2*n_hc]` — HC post gates (2 × sigmoid)
/// - `out[2*n_hc..]` — combine matrix, row-major, Sinkhorn-normalised
pub fn hc_split_sinkhorn(
    out: &mut [f32],
    mix: &[f32],
    scale: &[f32],
    base: &[f32],
    n_hc: i32,
    iters: i32,
    eps: f32,
) {
    let nh = n_hc as usize;
    let pre_scale = scale[0];
    let post_scale = scale[1];
    let comb_scale = scale[2];

    // Pre-weights: sigmoid(mix[i] * pre_scale + base[i]) + eps
    for i in 0..nh {
        let z = mix[i] * pre_scale + base[i];
        out[i] = 1.0 / (1.0 + (-z).exp()) + eps;
    }

    // Post-gates: 2 × sigmoid(mix[off] * post_scale + base[off])
    for i in 0..nh {
        let off = nh + i;
        let z = mix[off] * post_scale + base[off];
        out[off] = 2.0 / (1.0 + (-z).exp());
    }

    // Combine matrix: row-normalise, column-normalise, then iterate Sinkhorn
    let mut c = vec![0.0f32; nh * nh];

    // Initial row-wise softmax
    for dst in 0..nh {
        let mut row_max = DS4_NEG_INF;
        for src in 0..nh {
            let idx = src + dst * nh;
            let off = 2 * nh + idx;
            let v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            if v > row_max {
                row_max = v;
            }
        }

        let mut row_sum = 0.0f32;
        for src in 0..nh {
            let idx = src + dst * nh;
            let v = (c[idx] - row_max).exp();
            c[idx] = v;
            row_sum += v;
        }

        let inv = 1.0 / row_sum;
        for src in 0..nh {
            let idx = src + dst * nh;
            c[idx] = c[idx] * inv + eps;
        }
    }

    // First column normalisation
    for src in 0..nh {
        let mut sum = 0.0f32;
        for dst in 0..nh {
            sum += c[src + dst * nh];
        }
        let inv = 1.0 / (sum + eps);
        for dst in 0..nh {
            c[src + dst * nh] *= inv;
        }
    }

    // Sinkhorn iterations
    for _iter in 1..iters {
        // Row normalise
        for dst in 0..nh {
            let mut sum = 0.0f32;
            for src in 0..nh {
                sum += c[src + dst * nh];
            }
            let inv = 1.0 / (sum + eps);
            for src in 0..nh {
                c[src + dst * nh] *= inv;
            }
        }
        // Column normalise
        for src in 0..nh {
            let mut sum = 0.0f32;
            for dst in 0..nh {
                sum += c[src + dst * nh];
            }
            let inv = 1.0 / (sum + eps);
            for dst in 0..nh {
                c[src + dst * nh] *= inv;
            }
        }
    }

    // Copy combine matrix to output
    let out_off = 2 * nh;
    for i in 0..nh * nh {
        out[out_off + i] = c[i];
    }
}

/// Weighted HC sum: reduce `n_hc` streams (each `n_embd` wide) into a single
/// plain embedding by weighting with the HC pre-weights.
pub fn hc_weighted_sum(out: &mut [f32], x_hc: &[f32], weights: &[f32], n_embd: u32, n_hc: u32) {
    let ne = n_embd as usize;
    let nh = n_hc as usize;
    for d in 0..ne {
        let mut acc = 0.0f32;
        for h in 0..nh {
            acc += x_hc[h * ne + d] * weights[h];
        }
        out[d] = acc;
    }
}

/// HC post step: inject the new block output and mix the previous HC streams
/// through the learned combine matrix and post gates.
///
/// `out_hc[dst * n_embd + d] = block_out[d] * post[dst] + sum_src comb[dst + src * n_hc] * residual_hc[src * n_embd + d]`
pub fn hc_post_one(
    out_hc: &mut [f32],
    block_out: &[f32],
    residual_hc: &[f32],
    post: &[f32],
    comb: &[f32],
    n_embd: u32,
    n_hc: u32,
) {
    let ne = n_embd as usize;
    let nh = n_hc as usize;
    for dst in 0..nh {
        for d in 0..ne {
            let mut acc = block_out[d] * post[dst];
            for src in 0..nh {
                // comb is addressed as [dst_hc, src_hc] in column-major (same as C)
                acc += comb[dst + src * nh] * residual_hc[src * ne + d];
            }
            out_hc[dst * ne + d] = acc;
        }
    }
}

// ── Top-K descending ──────────────────────────────────────────────────────

/// Find the indices of the `k` largest elements in `score`, returned in
/// descending order of value.
pub fn topk_desc(score: &[f32], k: usize) -> Vec<usize> {
    let n = score.len();
    let mut idx = vec![usize::MAX; k];

    for i in 0..n {
        for j in 0..k {
            if idx[j] == usize::MAX || score[i] > score[idx[j]] {
                // Shift remaining right
                for m in (j + 1..k).rev() {
                    idx[m] = idx[m - 1];
                }
                idx[j] = i;
                break;
            }
        }
    }

    idx
}

// ── Q8_K quantisation ─────────────────────────────────────────────────────

/// Quantise a QK_K-aligned float vector to `block_q8_K` blocks.
///
/// `x.len()` must be a multiple of `QK_K`.
pub fn ds4_quantize_row_q8_K(x: &[f32]) -> Vec<block_q8_K> {
    let nb = x.len() / QK_K as usize;
    let mut y = Vec::with_capacity(nb);

    for b in 0..nb {
        let base = b * QK_K as usize;
        let chunk = &x[base..base + QK_K as usize];

        let mut max_val = 0.0f32;
        let mut amax = 0.0f32;
        for &v in chunk.iter() {
            let av = v.abs();
            if av > amax {
                amax = av;
                max_val = v;
            }
        }

        let mut block = block_q8_K {
            d: 0.0,
            qs: [0i8; QK_K as usize],
            bsums: [0i16; QK_K as usize / 16],
        };

        if amax == 0.0 {
            y.push(block);
            continue;
        }

        let iscale = -127.0 / max_val;
        for j in 0..QK_K as usize {
            let v = (iscale * chunk[j]).round() as i32;
            block.qs[j] = if v > 127 {
                127
            } else if v < -128 {
                -128
            } else {
                v as i8
            };
        }

        for j in 0..(QK_K as usize / 16) {
            let mut sum = 0i32;
            for i in 0..16 {
                sum += block.qs[j * 16 + i] as i32;
            }
            block.bsums[j] = sum as i16;
        }

        block.d = 1.0 / iscale;
        y.push(block);
    }

    y
}

// ── Q2_K dot product ──────────────────────────────────────────────────────

/// Dot product between a Q2_K weight vector and a Q8_K activation vector.
///
/// Both must have `QK_K` elements per block and share the same block count.
/// NEON helper: dot product of 16 Q2 values (2-bit packed) with 16 Q8 values.
#[cfg(target_arch = "aarch64")]
unsafe fn neon_dot_q2_16(q2bits: uint8x16_t, q8: int8x16_t, shift: i32) -> i32 {
    let shifted = match shift {
        0 => q2bits,
        2 => vshrq_n_u8(q2bits, 2),
        4 => vshrq_n_u8(q2bits, 4),
        _ => vshrq_n_u8(q2bits, 6),
    };
    let vals_u = vandq_u8(shifted, vdupq_n_u8(3));
    let vals = vreinterpretq_s8_u8(vals_u);
    // Use the same widening-multiply approach as neon_dot_i8x16
    let p0 = vmull_s8(vget_low_s8(q8), vget_low_s8(vals));
    let p1 = vmull_s8(vget_high_s8(q8), vget_high_s8(vals));
    let s0 = vpaddlq_s16(p0);
    let s1 = vpaddlq_s16(p1);
    vaddvq_s32(vaddq_s32(s0, s1))
}

pub fn ds4_vec_dot_q2_K_q8_K(x: &[block_q2_K], y: &[block_q8_K]) -> f32 {
    let nb = x.len().min(y.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let m4 = vdupq_n_u8(0x0f);
            let mut sum = 0.0f32;

            for i in 0..nb {
                let d_val = y[i].d * f16_to_f32(x[i].d);
                let dmin_val = -y[i].d * f16_to_f32(x[i].dmin);

                let q2 = &x[i].qs;
                let q8 = &y[i].qs;
                let sc = &x[i].scales;

                let mins_and_scales = vld1q_u8(sc.as_ptr());
                let scales = vandq_u8(mins_and_scales, m4);

                let mut scale_lanes: [u8; 16] = [0u8; 16];
                vst1q_u8(scale_lanes.as_mut_ptr(), scales);

                let mins = vshrq_n_u8(mins_and_scales, 4);
                let q8sums = vld1q_s16_x2(y[i].bsums.as_ptr() as *const i16);
                let mins_lo = vmovl_u8(vget_low_u8(mins));
                let mins_hi = vmovl_u8(vget_high_u8(mins));
                let mins16_0 = vreinterpretq_s16_u16(mins_lo);
                let mins16_1 = vreinterpretq_s16_u16(mins_hi);

                let s0 = vaddq_s32(
                    vmull_s16(vget_low_s16(mins16_0), vget_low_s16(q8sums.0)),
                    vmull_s16(vget_high_s16(mins16_0), vget_high_s16(q8sums.0)),
                );
                let s1 = vaddq_s32(
                    vmull_s16(vget_low_s16(mins16_1), vget_low_s16(q8sums.1)),
                    vmull_s16(vget_high_s16(mins16_1), vget_high_s16(q8sums.1)),
                );
                sum += dmin_val * (vaddvq_s32(vaddq_s32(s0, s1)) as f32);

                let mut isum = 0i32;

                // Process two 128-bit halves (QK_K / 128 = 2)
                for chunk_idx in 0..2 {
                    let q2_base = q2.as_ptr().add(chunk_idx * 32);
                    let q8_base = q8.as_ptr().add(chunk_idx * 64);

                    let q2bits = vld1q_u8_x2(q2_base);

                    // Shift = 0
                    let q8bytes = vld1q_s8_x2(q8_base);
                    let q8b_0 = q8bytes.0;
                    let q8b_1 = q8bytes.1;
                    isum += neon_dot_q2_16(q2bits.0, q8b_0, 0) * scale_lanes[chunk_idx * 8] as i32;
                    isum +=
                        neon_dot_q2_16(q2bits.1, q8b_1, 0) * scale_lanes[chunk_idx * 8 + 1] as i32;

                    // Shift = 2
                    let q8bytes2 = vld1q_s8_x2(q8_base.add(32));
                    let q8b2_0 = q8bytes2.0;
                    let q8b2_1 = q8bytes2.1;
                    isum +=
                        neon_dot_q2_16(q2bits.0, q8b2_0, 2) * scale_lanes[chunk_idx * 8 + 2] as i32;
                    isum +=
                        neon_dot_q2_16(q2bits.1, q8b2_1, 2) * scale_lanes[chunk_idx * 8 + 3] as i32;

                    // Shift = 4
                    let q8bytes3 = vld1q_s8_x2(q8_base.add(64));
                    let q8b3_0 = q8bytes3.0;
                    let q8b3_1 = q8bytes3.1;
                    isum +=
                        neon_dot_q2_16(q2bits.0, q8b3_0, 4) * scale_lanes[chunk_idx * 8 + 4] as i32;
                    isum +=
                        neon_dot_q2_16(q2bits.1, q8b3_1, 4) * scale_lanes[chunk_idx * 8 + 5] as i32;

                    // Shift = 6
                    let q8bytes4 = vld1q_s8_x2(q8_base.add(96));
                    let q8b4_0 = q8bytes4.0;
                    let q8b4_1 = q8bytes4.1;
                    isum +=
                        neon_dot_q2_16(q2bits.0, q8b4_0, 6) * scale_lanes[chunk_idx * 8 + 6] as i32;
                    isum +=
                        neon_dot_q2_16(q2bits.1, q8b4_1, 6) * scale_lanes[chunk_idx * 8 + 7] as i32;
                }

                sum += d_val * isum as f32;
            }

            sum
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sumf = 0.0f32;
        for i in 0..nb {
            let q2 = &x[i].qs;
            let q8 = &y[i].qs;
            let sc = &x[i].scales;

            // dmin contribution: bsums * (scales >> 4)
            let mut summs = 0i32;
            for j in 0..16 {
                summs += (y[i].bsums[j] as i32) * ((sc[j] >> 4) as i32);
            }

            let dall = y[i].d * f16_to_f32(x[i].d);
            let dmin = y[i].d * f16_to_f32(x[i].dmin);

            let mut isum = 0i32;
            let mut is = 0usize;
            for _k in 0..(QK_K as usize / 128) {
                let mut shift = 0i32;
                for _j in 0..4 {
                    let d0 = (sc[is] & 0x0f) as i32;
                    isum += d0 * dot_q2_16_scalar(&q2[is * 16..], &q8[is * 16..], shift);
                    is += 1;

                    let d1 = (sc[is] & 0x0f) as i32;
                    isum += d1 * dot_q2_16_scalar(&q2[(is) * 16..], &q8[(is) * 16..], shift);
                    is += 1;

                    shift += 2;
                }
            }
            sumf += dall * isum as f32 - dmin * summs as f32;
        }
        sumf
    }
}

/// Scalar helper: dot product of 16 Q2 values (packed 4-bit) with 16 Q8 values.
fn dot_q2_16_scalar(q2: &[u8], q8: &[i8], shift: i32) -> i32 {
    let mut sum = 0i32;
    for i in 0..16 {
        sum += (q8[i] as i32) * (((q2[i] as i32) >> shift) & 3);
    }
    sum
}

// ── IQ2_XXS dot product ───────────────────────────────────────────────────

/// Dot product between an IQ2_XXS weight vector and a Q8_K activation vector.
///
/// The non-NEON scalar path walks the packed index pairs and looks up the
/// precomputed signed grid values.
pub fn ds4_vec_dot_iq2_xxs_q8_K(x: &[block_iq2_xxs], y: &[block_q8_K]) -> f32 {
    let nb = x.len().min(y.len());

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            let mut sumf = 0.0f32;
            for i in 0..nb {
                let d = f16_to_f32(x[i].d) * y[i].d;
                let q2 = &x[i].qs;
                let q8 = &y[i].qs;
                let mut sumf1 = 0.0f32;
                let mut sumf2 = 0.0f32;

                let mut ib32 = 0usize;
                while ib32 < (QK_K as usize / 32) {
                    // Process 2 blocks at a time (64 Q8 values)
                    let q8b = vld1q_s8_x4(q8.as_ptr().add(ib32 * 16));
                    let q8b_0 = q8b.0;
                    let q8b_1 = q8b.1;
                    let q8b_2 = q8b.2;
                    let q8b_3 = q8b.3;

                    let mut aux32 = [0u32; 4];
                    // Copy 8 u16 into 4 u32
                    for k in 0..4 {
                        aux32[k] = (q2[ib32 / 16 * 8 + k * 2] as u32)
                            | ((q2[ib32 / 16 * 8 + k * 2 + 1] as u32) << 16);
                    }
                    let aux8: &[u8; 16] = std::mem::transmute(&aux32);

                    let q2u0 = vcombine_s8(
                        vld1_s8(IQ2XXS_GRID[aux8[0] as usize].to_ne_bytes().as_ptr() as *const i8),
                        vld1_s8(IQ2XXS_GRID[aux8[1] as usize].to_ne_bytes().as_ptr() as *const i8),
                    );
                    let q2u1 = vcombine_s8(
                        vld1_s8(IQ2XXS_GRID[aux8[2] as usize].to_ne_bytes().as_ptr() as *const i8),
                        vld1_s8(IQ2XXS_GRID[aux8[3] as usize].to_ne_bytes().as_ptr() as *const i8),
                    );
                    let q2u2 = vcombine_s8(
                        vld1_s8(IQ2XXS_GRID[aux8[8] as usize].to_ne_bytes().as_ptr() as *const i8),
                        vld1_s8(IQ2XXS_GRID[aux8[9] as usize].to_ne_bytes().as_ptr() as *const i8),
                    );
                    let q2u3 = vcombine_s8(
                        vld1_s8(IQ2XXS_GRID[aux8[10] as usize].to_ne_bytes().as_ptr() as *const i8),
                        vld1_s8(IQ2XXS_GRID[aux8[11] as usize].to_ne_bytes().as_ptr() as *const i8),
                    );

                    let sgn0 = vcombine_s8(
                        vld1_s8(IQ2XXS_SIGNS[((aux32[1] >> 0) & 127) as usize].as_ptr()),
                        vld1_s8(IQ2XXS_SIGNS[((aux32[1] >> 7) & 127) as usize].as_ptr()),
                    );
                    let sgn1 = vcombine_s8(
                        vld1_s8(IQ2XXS_SIGNS[((aux32[1] >> 14) & 127) as usize].as_ptr()),
                        vld1_s8(IQ2XXS_SIGNS[((aux32[1] >> 21) & 127) as usize].as_ptr()),
                    );
                    let sgn2 = vcombine_s8(
                        vld1_s8(IQ2XXS_SIGNS[((aux32[3] >> 0) & 127) as usize].as_ptr()),
                        vld1_s8(IQ2XXS_SIGNS[((aux32[3] >> 7) & 127) as usize].as_ptr()),
                    );
                    let sgn3 = vcombine_s8(
                        vld1_s8(IQ2XXS_SIGNS[((aux32[3] >> 14) & 127) as usize].as_ptr()),
                        vld1_s8(IQ2XXS_SIGNS[((aux32[3] >> 21) & 127) as usize].as_ptr()),
                    );

                    let q2u0 = vmulq_s8(q2u0, sgn0);
                    let q2u1 = vmulq_s8(q2u1, sgn1);
                    let q2u2 = vmulq_s8(q2u2, sgn2);
                    let q2u3 = vmulq_s8(q2u3, sgn3);

                    let p1_0 = neon_dot_i8x16(q2u0, q8b_0);
                    let p1_1 = neon_dot_i8x16(q2u1, q8b_1);
                    let p2_0 = neon_dot_i8x16(q2u2, q8b_2);
                    let p2_1 = neon_dot_i8x16(q2u3, q8b_3);

                    let ls1 = 0.5 + ((aux32[1] >> 28) as f32);
                    let ls2 = 0.5 + ((aux32[3] >> 28) as f32);

                    sumf1 += (p1_0 + p1_1) as f32 * ls1;
                    sumf2 += (p2_0 + p2_1) as f32 * ls2;

                    ib32 += 2;
                }

                sumf += d * (sumf1 + sumf2);
            }

            0.25 * sumf
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut aux32 = [0u32; 2];
        let mut sumf = 0.0f32;

        for i in 0..nb {
            let d = f16_to_f32(x[i].d) * y[i].d;
            let q2 = &x[i].qs;
            let q8 = &y[i].qs;
            let mut bsum = 0i32;

            for ib32 in 0..(QK_K as usize / 32) {
                aux32[0] = q2[ib32 * 4] as u32 | ((q2[ib32 * 4 + 1] as u32) << 16);
                aux32[1] = q2[ib32 * 4 + 2] as u32 | ((q2[ib32 * 4 + 3] as u32) << 16);

                let ls = (2 * (aux32[1] >> 28) + 1) as i32;
                let aux8: &[u8; 8] = std::mem::transmute(&aux32);

                let mut sumi = 0i32;
                for l in (0..4).step_by(2) {
                    let sign_idx0 = ((aux32[1] >> (7 * l)) & 127) as usize;
                    let sign_idx1 = ((aux32[1] >> (7 * (l + 1))) & 127) as usize;
                    let g0 = aux8[l] as usize;
                    let g1 = aux8[l + 1] as usize;

                    sumi += dot_iq2_pair_16_scalar(
                        &IQ2XXS_SIGNED_GRID[g0][sign_idx0],
                        &IQ2XXS_SIGNED_GRID[g1][sign_idx1],
                        &q8[ib32 * 16 + l * 8..],
                    );
                }
                bsum += sumi * ls;
            }
            sumf += d * bsum as f32;
        }

        0.125 * sumf
    }
}

/// Scalar helper: dot product of 16 signed grid values (two 8-element groups)
/// with 16 Q8 values.
fn dot_iq2_pair_16_scalar(grid0: &[i8; 8], grid1: &[i8; 8], q8: &[i8]) -> i32 {
    let mut sum = 0i32;
    for i in 0..8 {
        sum += (grid0[i] as i32) * (q8[i] as i32);
    }
    for i in 0..8 {
        sum += (grid1[i] as i32) * (q8[8 + i] as i32);
    }
    sum
}

// ── Softmax ───────────────────────────────────────────────────────────────

/// Standard softmax in-place over the slice `x`.
///
/// `out[i] = exp(x[i] - max) / sum(exp(x - max))`
pub fn softmax(out: &mut [f32], x: &[f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }

    // Find max for numerical stability
    let mut maxv = DS4_NEG_INF;
    for &v in x.iter() {
        if v > maxv {
            maxv = v;
        }
    }

    // Compute exp and sum
    let mut sum = 0.0f32;
    for i in 0..n {
        let e = (x[i] - maxv).exp();
        out[i] = e;
        sum += e;
    }

    // Normalise
    let inv = 1.0 / sum;
    for i in 0..n {
        out[i] *= inv;
    }
}

// ── Debug printing ────────────────────────────────────────────────────────

/// Print a summary of vector statistics (min, max, RMS) to stdout.
pub fn print_vec_stats(name: &str, x: &[f32]) {
    let n = x.len();
    if n == 0 {
        println!("{}: (empty)", name);
        return;
    }

    let mut minv = DS4_POS_INF;
    let mut maxv = DS4_NEG_INF;
    let mut ss = 0.0f64;

    for &v in x.iter() {
        if v < minv {
            minv = v;
        }
        if v > maxv {
            maxv = v;
        }
        ss += (v as f64) * (v as f64);
    }

    let rms = (ss / n as f64).sqrt();
    println!("{}: min={} max={} rms={}", name, minv, maxv, rms);
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_roundtrip() {
        let vals: Vec<f32> = vec![0.0, 1.0, -1.0, 0.5, -0.5, 65504.0, 1.0e-8];
        for &v in &vals {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            // Half precision is ~5 decimal digits
            let diff = (v - back).abs();
            let rel = diff / v.abs().max(1.0e-10);
            assert!(
                rel < 1.0e-3 || diff < 1.0e-5,
                "f16 roundtrip failed for {}",
                v
            );
        }
    }

    #[test]
    fn test_softmax() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut out = vec![0.0f32; 5];
        softmax(&mut out, &x);
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1.0e-6);
        assert!(out[4] > out[3]);
    }

    #[test]
    fn test_silu() {
        assert!((silu(0.0) - 0.0).abs() < 1.0e-7);
        assert!(silu(1.0) > 0.5);
        assert!(silu(1.0) < 1.0);
    }

    #[test]
    fn test_dot_f32() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b: Vec<f32> = vec![4.0, 5.0, 6.0];
        let d = dot_f32(&a, &b);
        assert!((d - 32.0).abs() < 1.0e-6);
    }

    #[test]
    fn test_axpy_f32() {
        let mut y: Vec<f32> = vec![1.0, 2.0, 3.0];
        let x: Vec<f32> = vec![0.5, 0.5, 0.5];
        axpy_f32(&mut y, &x, 2.0);
        assert!((y[0] - 2.0).abs() < 1.0e-6);
        assert!((y[1] - 3.0).abs() < 1.0e-6);
        assert!((y[2] - 4.0).abs() < 1.0e-6);
    }

    #[test]
    fn test_scale_f32() {
        let mut x: Vec<f32> = vec![1.0, 2.0, 3.0];
        scale_f32(&mut x, 0.5);
        assert!((x[0] - 0.5).abs() < 1.0e-6);
        assert!((x[2] - 1.5).abs() < 1.0e-6);
    }

    #[test]
    fn test_rms_norm_no_weight() {
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut out = vec![0.0f32; 4];
        rms_norm_no_weight(&mut out, &x, 1.0e-6);
        let sum_sq: f32 = out.iter().map(|v| v * v).sum();
        let mean_sq = sum_sq / 4.0;
        assert!((mean_sq - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_quantize_q8_0_roundtrip() {
        let x: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let (xq, xscale) = quantize_q8_0(&x);
        // Dequantize and check
        let mut err = 0.0f32;
        for i in 0..32 {
            let deq = (xq[i] as f32) * xscale[0];
            err += (x[i] - deq).abs();
        }
        assert!(err < 1.0, "Q8_0 round-trip error too large: {}", err);
    }

    #[test]
    fn test_topk_desc() {
        let scores: Vec<f32> = vec![0.1, 0.8, 0.3, 0.9, 0.2];
        let idx = topk_desc(&scores, 3);
        assert_eq!(idx.len(), 3);
        assert_eq!(idx[0], 3); // 0.9
        assert_eq!(idx[1], 1); // 0.8
        assert_eq!(idx[2], 2); // 0.3
    }

    #[test]
    fn test_swiglu() {
        let gate: Vec<f32> = vec![1.0, 2.0, 3.0];
        let up: Vec<f32> = vec![0.5, 0.5, 0.5];
        let mut out = vec![0.0f32; 3];
        swiglu(&mut out, &gate, &up);
        for i in 0..3 {
            assert!((out[i] - silu(gate[i]) * up[i]).abs() < 1.0e-6);
        }
    }

    #[test]
    fn test_head_rms_norm_inplace() {
        let n_head = 2u32;
        let head_dim = 4u32;
        let mut x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        head_rms_norm_inplace(&mut x, n_head, head_dim, 1.0e-6);
        for h in 0..2 {
            let off = h as usize * head_dim as usize;
            let sum_sq: f32 = x[off..off + head_dim as usize].iter().map(|v| v * v).sum();
            let mean_sq = sum_sq / head_dim as f32;
            assert!(
                (mean_sq - 1.0).abs() < 0.01,
                "head {} RMS={}",
                h,
                mean_sq.sqrt()
            );
        }
    }

    #[test]
    fn test_sigmoid_stable() {
        assert!((sigmoid_stable(0.0) - 0.5).abs() < 1.0e-7);
        assert!(sigmoid_stable(100.0) > 0.9999);
        assert!(sigmoid_stable(-100.0) < 0.0001);
    }

    #[test]
    fn test_softplus_stable() {
        assert!((softplus_stable(0.0) - 0.693147).abs() < 1.0e-4);
        assert!((softplus_stable(100.0) - 100.0).abs() < 1.0);
        assert!((softplus_stable(-100.0)).abs() < 1.0e-40);
    }

    #[test]
    fn test_q8k_blocks() {
        assert_eq!(q8k_blocks(256), 1);
        assert_eq!(q8k_blocks(257), 2);
        assert_eq!(q8k_blocks(512), 2);
        assert_eq!(q8k_blocks(0), 0);
    }

    #[test]
    fn test_hc_weighted_sum() {
        let n_embd = 4u32;
        let n_hc = 3u32;
        let x_hc: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let weights: Vec<f32> = vec![1.0, 0.0, 0.0];
        let mut out = vec![0.0f32; n_embd as usize];
        hc_weighted_sum(&mut out, &x_hc, &weights, n_embd, n_hc);
        for d in 0..4 {
            assert!((out[d] - d as f32).abs() < 1.0e-6);
        }
    }

    #[test]
    fn test_hc_post_one() {
        let n_embd = 2u32;
        let n_hc = 2u32;
        let mut out_hc = vec![0.0f32; 4];
        let block_out: Vec<f32> = vec![1.0, 2.0];
        let residual_hc: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4];
        let post: Vec<f32> = vec![0.5, 1.5];
        let comb: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4]; // [dst,src] = dst + src*nh
        hc_post_one(
            &mut out_hc,
            &block_out,
            &residual_hc,
            &post,
            &comb,
            n_embd,
            n_hc,
        );
        // Just verify no panics and values are finite
        for &v in out_hc.iter() {
            assert!(v.is_finite());
        }
    }
}
