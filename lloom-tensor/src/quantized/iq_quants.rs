//! IQ-family and Q2_0 GGUF block types: byte-exact dequant (`to_float`) ported
//! from llama.cpp `ggml-quants.c` / `ggml-common.h` (the qwen4exp fork that
//! loads GSQ-RCO Flash-Next checkpoints).
//!
//! These are the storage types a non-uniform GSQ-RCO GGUF assigns per tensor
//! (IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, and the fork's own Q2_0, type code 42).
//! Only `to_float` (dequant) is implemented — lloom reads these checkpoints, it
//! does not quantize them, so `from_float`/`vec_dot` on the CPU path bail. The
//! Metal decode path runs the matmul through the existing `kernel_mul_mv_iq*`
//! kernels with the weights in their native block layout (dequant-on-the-fly);
//! the CPU `to_float` here is the reference dequant for the no-metal build and
//! the parity tests.
//!
//! Correctness is byte-exact against the reference (`dequantize_row_*`); a
//! wrong bit here is the silent-corruption class, so the code mirrors the C line
//! for line rather than being "cleaned up".

use super::iq_grids::{
    IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS, KSIGNS_IQ2XS,
    KVALUES_IQ4NL,
};
use super::k_quants::{GgmlType, QK_K};
use super::GgmlDType;
use half::f16;

pub const QK2_0: usize = 64;
pub const QK4_NL: usize = 32;

#[inline]
fn h2f(h: f16) -> f32 {
    h.to_f32()
}

/// A grid entry (`uint64_t` / `uint32_t`) reinterpreted as 8 / 4 signed-int8
/// codebook values, exactly as the C casts `(const uint8_t *)(grid + idx)` and
/// reads `grid[j]` — the bytes are stored little-endian and read as `int8`.
#[inline]
fn grid8_u64(v: u64) -> [i8; 8] {
    let b = v.to_le_bytes();
    [
        b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8, b[4] as i8, b[5] as i8, b[6] as i8,
        b[7] as i8,
    ]
}

#[inline]
fn grid8_u32(v: u32) -> [i8; 4] {
    let b = v.to_le_bytes();
    [b[0] as i8, b[1] as i8, b[2] as i8, b[3] as i8]
}

#[inline]
fn sign_mul(val: f32, signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -val
    } else {
        val
    }
}

fn unimpl_from_float(dtype: GgmlDType) -> ! {
    panic!("from_float is not implemented for read-only quant type {dtype:?}")
}
fn unimpl_vec_dot(dtype: GgmlDType) -> ! {
    panic!("cpu vec_dot is not implemented for quant type {dtype:?}; matmul runs on Metal")
}

// ---------------------------------------------------------------------------
// Q2_0 (fork type code 42): block of 64, one f16 scale, 2 bits/weight.
// dequant: w = (q - 1) * d, q in {0,1,2,3} -> {-1,0,1,2}. (ggml-quants.c:439)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ2_0 {
    pub d: f16,
    pub qs: [u8; QK2_0 / 4],
}
const _: () = assert!(std::mem::size_of::<BlockQ2_0>() == 2 + QK2_0 / 4);

impl GgmlType for BlockQ2_0 {
    const DTYPE: GgmlDType = GgmlDType::Q2_0;
    const BLCK_SIZE: usize = QK2_0;
    type VecDotType = super::k_quants::BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let qk = QK2_0;
        for (i, x) in xs.iter().enumerate() {
            let d = h2f(x.d);
            for j in 0..qk {
                let byte_index = j / 4;
                let bit_offset = (j % 4) * 2;
                let q = (x.qs[byte_index] >> bit_offset) & 0x03;
                ys[i * qk + j] = (q as i32 - 1) as f32 * d;
            }
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimpl_from_float(Self::DTYPE)
    }
    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
}

// ---------------------------------------------------------------------------
// IQ2_XXS: 2.0625 bpw. (ggml-quants.c:2488)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ2xxs {
    pub d: f16,
    pub qs: [u16; QK_K / 8],
}
const _: () = assert!(std::mem::size_of::<BlockIQ2xxs>() == 2 + (QK_K / 8) * 2);

impl GgmlType for BlockIQ2xxs {
    const DTYPE: GgmlDType = GgmlDType::IQ2XXS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = super::k_quants::BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let mut yi = 0usize;
        for x in xs {
            let d = h2f(x.d);
            // qs is QK_K/8 u16 = QK_K/4 bytes; each ib32 consumes 4 u16 (aux32[0..2]).
            for ib32 in 0..(QK_K / 32) {
                let a0 = x.qs[4 * ib32] as u32 | ((x.qs[4 * ib32 + 1] as u32) << 16);
                let a1 = x.qs[4 * ib32 + 2] as u32 | ((x.qs[4 * ib32 + 3] as u32) << 16);
                let aux8 = a0.to_le_bytes();
                let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let grid = grid8_u64(IQ2XXS_GRID[aux8[l] as usize]);
                    let signs = KSIGNS_IQ2XS[((a1 >> (7 * l)) & 127) as usize];
                    for j in 0..8 {
                        ys[yi + j] = sign_mul(db * grid[j] as f32, signs, j);
                    }
                    yi += 8;
                }
            }
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimpl_from_float(Self::DTYPE)
    }
    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
}

// ---------------------------------------------------------------------------
// IQ2_XS: 2.3125 bpw. (ggml-quants.c:2516)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ2xs {
    pub d: f16,
    pub qs: [u16; QK_K / 8],
    pub scales: [u8; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIQ2xs>() == 2 + (QK_K / 8) * 2 + QK_K / 32);

impl GgmlType for BlockIQ2xs {
    const DTYPE: GgmlDType = GgmlDType::IQ2XS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = super::k_quants::BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let mut yi = 0usize;
        for x in xs {
            let d = h2f(x.d);
            for ib32 in 0..(QK_K / 32) {
                let db = [
                    d * (0.5 + (x.scales[ib32] & 0xf) as f32) * 0.25,
                    d * (0.5 + (x.scales[ib32] >> 4) as f32) * 0.25,
                ];
                for l in 0..4 {
                    let q = x.qs[4 * ib32 + l];
                    let grid = grid8_u64(IQ2XS_GRID[(q & 511) as usize]);
                    let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                    for j in 0..8 {
                        ys[yi + j] = sign_mul(db[l / 2] * grid[j] as f32, signs, j);
                    }
                    yi += 8;
                }
            }
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimpl_from_float(Self::DTYPE)
    }
    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
}

// ---------------------------------------------------------------------------
// IQ2_S: 2.5625 bpw. (ggml-quants.c:2543)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ2s {
    pub d: f16,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub scales: [u8; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIQ2s>() == 2 + QK_K / 4 + QK_K / 16);

impl GgmlType for BlockIQ2s {
    const DTYPE: GgmlDType = GgmlDType::IQ2S;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = super::k_quants::BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let mut yi = 0usize;
        for x in xs {
            let d = h2f(x.d);
            // signs live in the second half of qs (qs + QK_K/8), advancing by 4
            // per ib32; qs (grid indices) advances by 4 per ib32 from the start.
            for ib32 in 0..(QK_K / 32) {
                let db = [
                    d * (0.5 + (x.scales[ib32] & 0xf) as f32) * 0.25,
                    d * (0.5 + (x.scales[ib32] >> 4) as f32) * 0.25,
                ];
                let qs_base = 4 * ib32;
                let signs_base = QK_K / 8 + 4 * ib32;
                for l in 0..4 {
                    let dl = db[l / 2];
                    let idx = x.qs[qs_base + l] as usize
                        | (((x.qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                    let grid = grid8_u64(IQ2S_GRID[idx]);
                    let signs = x.qs[signs_base + l];
                    for j in 0..8 {
                        ys[yi + j] = sign_mul(dl * grid[j] as f32, signs, j);
                    }
                    yi += 8;
                }
            }
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimpl_from_float(Self::DTYPE)
    }
    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
}

// ---------------------------------------------------------------------------
// IQ3_XXS: 3.0625 bpw. (ggml-quants.c:2575)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ3xxs {
    pub d: f16,
    pub qs: [u8; 3 * QK_K / 8],
}
const _: () = assert!(std::mem::size_of::<BlockIQ3xxs>() == 2 + 3 * (QK_K / 8));

impl GgmlType for BlockIQ3xxs {
    const DTYPE: GgmlDType = GgmlDType::IQ3XXS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = super::k_quants::BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let mut yi = 0usize;
        for x in xs {
            let d = h2f(x.d);
            // qs: first QK_K/4 bytes are grid indices, then scales_and_signs.
            let sas_off = QK_K / 4;
            for ib32 in 0..(QK_K / 32) {
                let qs_base = 8 * ib32;
                let aux32 = u32::from_le_bytes([
                    x.qs[sas_off + 4 * ib32],
                    x.qs[sas_off + 4 * ib32 + 1],
                    x.qs[sas_off + 4 * ib32 + 2],
                    x.qs[sas_off + 4 * ib32 + 3],
                ]);
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                    let grid1 = grid8_u32(IQ3XXS_GRID[x.qs[qs_base + 2 * l] as usize]);
                    let grid2 = grid8_u32(IQ3XXS_GRID[x.qs[qs_base + 2 * l + 1] as usize]);
                    for j in 0..4 {
                        ys[yi + j] = sign_mul(db * grid1[j] as f32, signs, j);
                        ys[yi + j + 4] = sign_mul(db * grid2[j] as f32, signs, j + 4);
                    }
                    yi += 8;
                }
            }
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimpl_from_float(Self::DTYPE)
    }
    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
}

// ---------------------------------------------------------------------------
// IQ3_S: 3.4375 bpw. (ggml-quants.c:2607)
// ---------------------------------------------------------------------------

const IQ3S_N_SCALE: usize = QK_K / 64;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ3s {
    pub d: f16,
    pub qs: [u8; QK_K / 4],
    pub qh: [u8; QK_K / 32],
    pub signs: [u8; QK_K / 8],
    pub scales: [u8; IQ3S_N_SCALE],
}
const _: () = assert!(std::mem::size_of::<BlockIQ3s>() == 2 + 13 * (QK_K / 32) + IQ3S_N_SCALE);

impl GgmlType for BlockIQ3s {
    const DTYPE: GgmlDType = GgmlDType::IQ3S;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = super::k_quants::BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let mut yi = 0usize;
        for x in xs {
            let d = h2f(x.d);
            let mut qs_off = 0usize;
            let mut qh_off = 0usize;
            let mut signs_off = 0usize;
            // ib32 steps by 2; each iteration handles two 32-blocks (db1, db2).
            let mut ib32 = 0usize;
            while ib32 < QK_K / 32 {
                let db1 = d * (1 + 2 * (x.scales[ib32 / 2] & 0xf) as i32) as f32;
                let db2 = d * (1 + 2 * (x.scales[ib32 / 2] >> 4) as i32) as f32;
                // first 32-block: qh[0]
                for l in 0..4 {
                    let g1 = x.qs[qs_off + 2 * l] as usize
                        | (((x.qh[qh_off] as usize) << (8 - 2 * l)) & 256);
                    let g2 = x.qs[qs_off + 2 * l + 1] as usize
                        | (((x.qh[qh_off] as usize) << (7 - 2 * l)) & 256);
                    let grid1 = grid8_u32(IQ3S_GRID[g1]);
                    let grid2 = grid8_u32(IQ3S_GRID[g2]);
                    let signs = x.signs[signs_off + l];
                    for j in 0..4 {
                        ys[yi + j] = sign_mul(db1 * grid1[j] as f32, signs, j);
                        ys[yi + j + 4] = sign_mul(db1 * grid2[j] as f32, signs, j + 4);
                    }
                    yi += 8;
                }
                qs_off += 8;
                signs_off += 4;
                // second 32-block: qh[1]
                for l in 0..4 {
                    let g1 = x.qs[qs_off + 2 * l] as usize
                        | (((x.qh[qh_off + 1] as usize) << (8 - 2 * l)) & 256);
                    let g2 = x.qs[qs_off + 2 * l + 1] as usize
                        | (((x.qh[qh_off + 1] as usize) << (7 - 2 * l)) & 256);
                    let grid1 = grid8_u32(IQ3S_GRID[g1]);
                    let grid2 = grid8_u32(IQ3S_GRID[g2]);
                    let signs = x.signs[signs_off + l];
                    for j in 0..4 {
                        ys[yi + j] = sign_mul(db2 * grid1[j] as f32, signs, j);
                        ys[yi + j + 4] = sign_mul(db2 * grid2[j] as f32, signs, j + 4);
                    }
                    yi += 8;
                }
                qh_off += 2;
                qs_off += 8;
                signs_off += 4;
                ib32 += 2;
            }
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimpl_from_float(Self::DTYPE)
    }
    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
}

// ---------------------------------------------------------------------------
// IQ4_NL: block of 32, non-linear codebook. (ggml-quants.c:2725)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ4nl {
    pub d: f16,
    pub qs: [u8; QK4_NL / 2],
}
const _: () = assert!(std::mem::size_of::<BlockIQ4nl>() == 2 + QK4_NL / 2);

impl GgmlType for BlockIQ4nl {
    const DTYPE: GgmlDType = GgmlDType::IQ4NL;
    const BLCK_SIZE: usize = QK4_NL;
    type VecDotType = super::k_quants::BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let qk = QK4_NL;
        for (i, x) in xs.iter().enumerate() {
            let d = h2f(x.d);
            for j in 0..(qk / 2) {
                ys[i * qk + j] = d * KVALUES_IQ4NL[(x.qs[j] & 0xf) as usize] as f32;
                ys[i * qk + j + qk / 2] = d * KVALUES_IQ4NL[(x.qs[j] >> 4) as usize] as f32;
            }
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimpl_from_float(Self::DTYPE)
    }
    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
}

// ---------------------------------------------------------------------------
// IQ4_XS: super-block of 256, non-linear codebook. (ggml-quants.c:2743)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct BlockIQ4xs {
    pub d: f16,
    pub scales_h: u16,
    pub scales_l: [u8; QK_K / 64],
    pub qs: [u8; QK_K / 2],
}
const _: () = assert!(std::mem::size_of::<BlockIQ4xs>() == 2 + 2 + QK_K / 64 + QK_K / 2);

impl GgmlType for BlockIQ4xs {
    const DTYPE: GgmlDType = GgmlDType::IQ4XS;
    const BLCK_SIZE: usize = QK_K;
    type VecDotType = super::k_quants::BlockQ8K;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let mut yi = 0usize;
        for x in xs {
            let d = h2f(x.d);
            let mut qs_off = 0usize;
            for ib in 0..(QK_K / 32) {
                let ls = (((x.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32)
                    | ((((x.scales_h >> (2 * ib)) & 3) as i32) << 4);
                let dl = d * (ls - 32) as f32;
                for j in 0..16 {
                    ys[yi + j] = dl * KVALUES_IQ4NL[(x.qs[qs_off + j] & 0xf) as usize] as f32;
                    ys[yi + j + 16] = dl * KVALUES_IQ4NL[(x.qs[qs_off + j] >> 4) as usize] as f32;
                }
                yi += 32;
                qs_off += 16;
            }
        }
    }
    fn from_float(_xs: &[f32], _ys: &mut [Self]) {
        unimpl_from_float(Self::DTYPE)
    }
    fn vec_dot(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
    fn vec_dot_unopt(_n: usize, _xs: &[Self], _ys: &[Self::VecDotType]) -> f32 {
        unimpl_vec_dot(Self::DTYPE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Block byte sizes match the ggml `static_assert`s exactly (the
    /// silent-corruption guard: a wrong struct layout mis-strides the whole
    /// tensor). Values from ggml-common.h.
    #[test]
    fn block_sizes_match_ggml() {
        assert_eq!(std::mem::size_of::<BlockQ2_0>(), 18); // 2 + 64/4
        assert_eq!(std::mem::size_of::<BlockIQ2xxs>(), 66); // 2 + 32*2
        assert_eq!(std::mem::size_of::<BlockIQ2xs>(), 74); // 2 + 32*2 + 8
        assert_eq!(std::mem::size_of::<BlockIQ2s>(), 82); // 2 + 64 + 16
        assert_eq!(std::mem::size_of::<BlockIQ3xxs>(), 98); // 2 + 96
        assert_eq!(std::mem::size_of::<BlockIQ3s>(), 110); // 2 + 13*8 + 4
        assert_eq!(std::mem::size_of::<BlockIQ4nl>(), 18); // 2 + 32/2
        assert_eq!(std::mem::size_of::<BlockIQ4xs>(), 136); // 2 + 2 + 4 + 128
    }

    /// Q2_0 dequant is `w = (q-1)*d`, q in {0,1,2,3}. Hand-verified: with d=2.0
    /// and a qs byte 0b11_10_01_00 (= 0xE4), the low-to-high 2-bit codes are
    /// 0,1,2,3 → weights -1*d, 0*d, 1*d, 2*d = -2, 0, 2, 4.
    #[test]
    fn q2_0_dequant_hand_verified() {
        let d = f16::from_f32(2.0);
        let mut qs = [0u8; QK2_0 / 4];
        qs[0] = 0b11_10_01_00; // j0=00, j1=01, j2=10, j3=11
        let blk = BlockQ2_0 { d, qs };
        let mut ys = vec![0f32; QK2_0];
        BlockQ2_0::to_float(&[blk], &mut ys);
        assert_eq!(ys[0], -2.0);
        assert_eq!(ys[1], 0.0);
        assert_eq!(ys[2], 2.0);
        assert_eq!(ys[3], 4.0);
        // remaining codes are 0 → weight = (0-1)*d = -d
        assert_eq!(ys[4], -2.0);
    }

    /// IQ4_NL maps each nibble through the fixed `kvalues_iq4nl` codebook and
    /// scales by d; low nibble → first half, high nibble → second half.
    #[test]
    fn iq4_nl_dequant_hand_verified() {
        let d = f16::from_f32(0.5);
        let mut qs = [0u8; QK4_NL / 2];
        // qs[0] low nibble = 0 (kvalues[0]=-127), high nibble = 15 (kvalues[15]=113)
        qs[0] = 0xF0;
        let blk = BlockIQ4nl { d, qs };
        let mut ys = vec![0f32; QK4_NL];
        BlockIQ4nl::to_float(&[blk], &mut ys);
        assert_eq!(ys[0], 0.5 * -127.0); // low nibble → position 0
        assert_eq!(ys[QK4_NL / 2], 0.5 * 113.0); // high nibble → position 16
    }

    /// The grids are the exact sizes the ggml declarations use — a truncated or
    /// over-long grid would index-out or mis-align.
    #[test]
    fn grid_table_sizes() {
        assert_eq!(IQ2XXS_GRID.len(), 256);
        assert_eq!(IQ2XS_GRID.len(), 512);
        assert_eq!(IQ2S_GRID.len(), 1024);
        assert_eq!(IQ3XXS_GRID.len(), 256);
        assert_eq!(IQ3S_GRID.len(), 512);
        assert_eq!(KSIGNS_IQ2XS.len(), 128);
        assert_eq!(KMASK_IQ2XS.len(), 8);
        assert_eq!(KVALUES_IQ4NL.len(), 16);
    }
}
