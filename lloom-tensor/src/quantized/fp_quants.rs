//! Block formats whose codes are small floats: MXFP4 (ggml's 4-bit E2M1
//! codes under a power-of-two E8M0 scale) and FP8 E4M3 (8-bit E4M3FN codes
//! under a bf16 scale).
//!
//! MXFP4 is ggml's block (`ggml-common.h` `block_mxfp4`, type 39) and its CPU
//! arithmetic is mirrored line for line from `ggml-quants.c`
//! (`dequantize_row_mxfp4`, `quantize_row_mxfp4_ref`) and the generic
//! `ggml_vec_dot_mxfp4_q8_0`: a GGUF written by llama.cpp reads here bit for
//! bit, and one written here reads there.
//!
//! FP8 E4M3 has no ggml type. The block is Q8_0's shape with the codes
//! decoded as E4M3FN and the scale stored as bf16, which carries a
//! block-scaled FP8 checkpoint's own weights without re-quantising them: an
//! E4M3 code and a bf16 scale are both exact in f32, and so is their product
//! (`docs/invariants.md` §15.2a's storage axis).
//!
//! Both are full `GgmlType`s: `to_float` is the reference every kernel's
//! parity is measured against, `from_float` the quantizer the tiny
//! checkpoints and the tests use, `vec_dot` the CPU matmul against a Q8_0
//! activation (what ggml pairs MXFP4 with).

use super::k_quants::{BlockQ8_0, GgmlType};
use super::GgmlDType;
use half::bf16;

pub const QK_MXFP4: usize = 32;
pub const QK_FP8: usize = 32;

/// The E2M1 values, doubled so they are integers (`kvalues_mxfp4`); the
/// scale is halved to match ([`e8m0_to_f32_half`]).
pub const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Half the value of an E8M0 scale byte, `2^(x - 128)`
/// (`ggml_e8m0_to_fp32_half`): the two smallest are f32 subnormals.
#[inline]
pub fn e8m0_to_f32_half(x: u8) -> f32 {
    let bits = if x < 2 {
        0x0020_0000u32 << x
    } else {
        ((x as u32) - 1) << 23
    };
    f32::from_bits(bits)
}

/// An E4M3FN code's value: bias 7, no infinities, `S.1111.111` NaN, largest
/// finite ±448, subnormals `m · 2⁻⁹`. Exact in f32.
#[inline]
pub fn e4m3_to_f32(b: u8) -> f32 {
    let e = ((b >> 3) & 0x0F) as i32;
    let m = (b & 0x07) as f32;
    let mag = if e == 0x0F && b & 0x07 == 0x07 {
        f32::NAN
    } else if e == 0 {
        m * (1.0 / 512.0)
    } else {
        (8.0 + m) * f32::from_bits(((e - 10 + 127) as u32) << 23)
    };
    if b & 0x80 != 0 {
        -mag
    } else {
        mag
    }
}

/// The E4M3FN code nearest `x`, ties to even, saturating at ±448; NaN maps
/// to the positive NaN code.
pub fn f32_to_e4m3(x: f32) -> u8 {
    if x.is_nan() {
        return 0x7F;
    }
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let a = x.abs();
    if a >= 448.0 {
        return sign | 0x7E;
    }
    // Below the smallest normal, 2⁻⁶, the codes are a uniform grid of
    // 2⁻⁹; the step to 8 · 2⁻⁹ lands on code 0x08, the smallest normal.
    if a < 1.0 / 64.0 {
        return sign | (a * 512.0).round_ties_even() as u8;
    }
    let e = ((a.to_bits() >> 23) & 0xFF) as i32 - 127;
    let frac = a / f32::from_bits(((e + 127) as u32) << 23);
    let mut m = ((frac - 1.0) * 8.0).round_ties_even() as i32;
    let mut e = e;
    if m == 8 {
        m = 0;
        e += 1;
    }
    sign | (((e + 7) as u8) << 3) | m as u8
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockMxfp4 {
    /// The E8M0 scale.
    pub e: u8,
    /// Element `j` in the low nibble of byte `j`, element `j + 16` in its
    /// high nibble.
    pub qs: [u8; QK_MXFP4 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockMxfp4>() == 1 + QK_MXFP4 / 2);

fn best_index_mxfp4(x: f32, e: f32) -> u8 {
    let mut best = 0usize;
    let mut best_err = (KVALUES_MXFP4[0] as f32 * e - x).abs();
    for (i, &v) in KVALUES_MXFP4.iter().enumerate().skip(1) {
        let err = (v as f32 * e - x).abs();
        if err < best_err {
            best = i;
            best_err = err;
        }
    }
    best as u8
}

impl GgmlType for BlockMxfp4 {
    const DTYPE: GgmlDType = GgmlDType::MXFP4;
    const BLCK_SIZE: usize = QK_MXFP4;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let qk = QK_MXFP4;
        for (i, x) in xs.iter().enumerate() {
            let d = e8m0_to_f32_half(x.e);
            for j in 0..qk / 2 {
                let x0 = KVALUES_MXFP4[(x.qs[j] & 0x0F) as usize];
                let x1 = KVALUES_MXFP4[(x.qs[j] >> 4) as usize];
                ys[i * qk + j] = x0 as f32 * d;
                ys[i * qk + j + qk / 2] = x1 as f32 * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let qk = QK_MXFP4;
        for (i, y) in ys.iter_mut().enumerate() {
            let x = &xs[i * qk..(i + 1) * qk];
            let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
            let e = if amax > 0.0 {
                (amax.log2().floor() - 2.0 + 127.0) as u8
            } else {
                0
            };
            let d = e8m0_to_f32_half(e);
            y.e = e;
            for j in 0..qk / 2 {
                let x0 = best_index_mxfp4(x[j], d);
                let x1 = best_index_mxfp4(x[qk / 2 + j], d);
                y.qs[j] = x0 | (x1 << 4);
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK_MXFP4), "vec_dot_mxfp4_q8_0: {n}");
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()).take(n / QK_MXFP4) {
            let d = y.d.to_f32() * e8m0_to_f32_half(x.e);
            let (mut sumi1, mut sumi2) = (0i32, 0i32);
            for j in 0..QK_MXFP4 / 2 {
                sumi1 += y.qs[j] as i32 * KVALUES_MXFP4[(x.qs[j] & 0x0F) as usize] as i32;
                sumi2 +=
                    y.qs[j + QK_MXFP4 / 2] as i32 * KVALUES_MXFP4[(x.qs[j] >> 4) as usize] as i32;
            }
            sumf += d * (sumi1 + sumi2) as f32;
        }
        sumf
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlockFp8E4M3 {
    /// The block's scale.
    pub d: bf16,
    /// E4M3FN codes.
    pub qs: [u8; QK_FP8],
}
const _: () = assert!(std::mem::size_of::<BlockFp8E4M3>() == 2 + QK_FP8);

impl GgmlType for BlockFp8E4M3 {
    const DTYPE: GgmlDType = GgmlDType::FP8E4M3;
    const BLCK_SIZE: usize = QK_FP8;
    type VecDotType = BlockQ8_0;

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (i, x) in xs.iter().enumerate() {
            let d = x.d.to_f32();
            for j in 0..QK_FP8 {
                ys[i * QK_FP8 + j] = e4m3_to_f32(x.qs[j]) * d;
            }
        }
    }

    /// The scale maps the block's largest magnitude to E4M3's largest
    /// finite value, 448; bf16's rounding of it may push a code past 448,
    /// which saturates.
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (i, y) in ys.iter_mut().enumerate() {
            let x = &xs[i * QK_FP8..(i + 1) * QK_FP8];
            let amax = x.iter().fold(0f32, |a, v| a.max(v.abs()));
            let d = bf16::from_f32(amax / 448.0);
            let id = if d.to_f32() != 0.0 {
                1.0 / d.to_f32()
            } else {
                0.0
            };
            y.d = d;
            for (q, &v) in y.qs.iter_mut().zip(x) {
                *q = f32_to_e4m3(v * id);
            }
        }
    }

    fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        Self::vec_dot_unopt(n, xs, ys)
    }

    fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
        debug_assert!(n.is_multiple_of(QK_FP8), "vec_dot_fp8_e4m3_q8_0: {n}");
        let mut sumf = 0f32;
        for (x, y) in xs.iter().zip(ys.iter()).take(n / QK_FP8) {
            let mut s = 0f32;
            for j in 0..QK_FP8 {
                s += e4m3_to_f32(x.qs[j]) * y.qs[j] as f32;
            }
            sumf += s * x.d.to_f32() * y.d.to_f32();
        }
        sumf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every E4M3FN code against the format's definition computed in f64:
    /// sign, exponent bias 7, the subnormal grid, the two NaNs, ±448.
    #[test]
    fn every_e4m3_code_decodes_to_its_definition() {
        for b in 0..=255u8 {
            let (s, e, m) = (b >> 7, (b >> 3) & 0xF, b & 7);
            let got = e4m3_to_f32(b);
            if e == 15 && m == 7 {
                assert!(got.is_nan(), "{b:#04x}");
                continue;
            }
            let mag = if e == 0 {
                m as f64 / 8.0 * 2f64.powi(-6)
            } else {
                (1.0 + m as f64 / 8.0) * 2f64.powi(e as i32 - 7)
            };
            let want = if s == 1 { -mag } else { mag };
            assert_eq!(got as f64, want, "{b:#04x}");
            assert_eq!(got.is_sign_negative(), s == 1, "{b:#04x}: sign of {got}");
        }
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
    }

    /// The encoder inverts the decoder on every finite code, rounds each
    /// midpoint between two neighbours to the even code, and saturates.
    #[test]
    fn e4m3_encoding_round_trips_and_ties_to_even() {
        for b in 0..=255u8 {
            let v = e4m3_to_f32(b);
            if v.is_nan() {
                continue;
            }
            let back = f32_to_e4m3(v);
            assert_eq!(e4m3_to_f32(back), v, "{b:#04x} -> {v} -> {back:#04x}");
        }
        for b in 0..0x7Eu8 {
            let (lo, hi) = (e4m3_to_f32(b), e4m3_to_f32(b + 1));
            let mid = (lo + hi) / 2.0;
            let even = if b % 2 == 0 { b } else { b + 1 };
            assert_eq!(
                f32_to_e4m3(mid),
                even,
                "midpoint of {b:#04x}, {:#04x}",
                b + 1
            );
            assert_eq!(f32_to_e4m3(-mid), 0x80 | even);
        }
        assert_eq!(f32_to_e4m3(1e9), 0x7E);
        assert_eq!(f32_to_e4m3(-1e9), 0xFE);
        assert_eq!(f32_to_e4m3(f32::NAN), 0x7F);
    }

    /// `ggml_e8m0_to_fp32_half` at its three regimes.
    #[test]
    fn e8m0_half_is_two_to_the_x_minus_128() {
        for x in 0..=254u8 {
            assert_eq!(e8m0_to_f32_half(x) as f64, 2f64.powi(x as i32 - 128), "{x}");
        }
    }

    /// A block quantized and dequantized stays within half a step of the
    /// code grid at its scale, for both formats.
    #[test]
    fn both_quantizers_round_trip_within_a_step() {
        let xs: Vec<f32> = (0..128)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013 * (1 + i / 32) as f32)
            .collect();
        let mut m = vec![BlockMxfp4 { e: 0, qs: [0; 16] }; 4];
        BlockMxfp4::from_float(&xs, &mut m);
        let mut back = vec![0f32; 128];
        BlockMxfp4::to_float(&m, &mut back);
        for (b, blk) in m.iter().enumerate() {
            let step = 2.0 * e8m0_to_f32_half(blk.e) * 2.0;
            for j in 0..32 {
                let (x, y) = (xs[b * 32 + j], back[b * 32 + j]);
                assert!((x - y).abs() <= step, "mxfp4 {x} -> {y} (step {step})");
            }
        }
        let mut f = vec![
            BlockFp8E4M3 {
                d: bf16::ZERO,
                qs: [0; 32]
            };
            4
        ];
        BlockFp8E4M3::from_float(&xs, &mut f);
        BlockFp8E4M3::to_float(&f, &mut back);
        for (x, y) in xs.iter().zip(&back) {
            assert!((x - y).abs() <= x.abs() / 16.0 + 1e-6, "fp8 {x} -> {y}");
        }
    }

    /// A block-scaled FP8 checkpoint (E4M3 codes `[n, k]`, a bf16 scale per
    /// 128 × 128 tile: Qwen3.8-Flash-Next-FP8's routed experts) carries into
    /// this block exactly: each 32-element row segment takes its tile's
    /// scale and its codes unchanged, and the block's `to_float` is the
    /// checkpoint's `code · scale` bit for bit — K need only be a multiple of
    /// 32, so a partial last tile (K = 320 is 2.5 tiles) packs as well.
    #[test]
    fn a_block_scaled_checkpoint_packs_without_requantising() {
        let (n, k, tile) = (160usize, 320usize, 128usize);
        let codes: Vec<u8> = (0..n * k)
            .map(|i| {
                let b = (i * 37 + i / 7) as u8;
                if b & 0x7F == 0x7F {
                    b ^ 1
                } else {
                    b
                }
            })
            .collect();
        let (tn, tk) = (n.div_ceil(tile), k.div_ceil(tile));
        let scales: Vec<bf16> = (0..tn * tk)
            .map(|t| bf16::from_f32(1.37e-4 * (1.0 + t as f32 * 0.31)))
            .collect();
        let blocks: Vec<BlockFp8E4M3> = (0..n)
            .flat_map(|r| (0..k / QK_FP8).map(move |b| (r, b)))
            .map(|(r, b)| {
                let mut qs = [0u8; QK_FP8];
                qs.copy_from_slice(&codes[r * k + b * QK_FP8..r * k + (b + 1) * QK_FP8]);
                BlockFp8E4M3 {
                    d: scales[(r / tile) * tk + (b * QK_FP8) / tile],
                    qs,
                }
            })
            .collect();
        let mut got = vec![0f32; n * k];
        BlockFp8E4M3::to_float(&blocks, &mut got);
        for r in 0..n {
            for c in 0..k {
                let want =
                    e4m3_to_f32(codes[r * k + c]) * scales[(r / tile) * tk + c / tile].to_f32();
                assert_eq!(got[r * k + c].to_bits(), want.to_bits(), "[{r}, {c}]");
            }
        }
    }

    /// The CPU dot against a Q8_0 activation equals the dequantized weight
    /// dotted with the dequantized activation, to f32 rounding.
    #[test]
    fn vec_dot_is_the_dot_of_the_dequantized_operands() {
        let w: Vec<f32> = (0..64)
            .map(|i| ((i * 13 % 29) as f32 - 14.0) * 0.07)
            .collect();
        let a: Vec<f32> = (0..64)
            .map(|i| ((i * 7 % 23) as f32 - 11.0) * 0.11)
            .collect();
        let mut q8 = vec![BlockQ8_0::zeros(); 2];
        BlockQ8_0::from_float(&a, &mut q8);
        let mut a_back = vec![0f32; 64];
        BlockQ8_0::to_float(&q8, &mut a_back);
        let check = |dot: f32, w_back: &[f32], name: &str| {
            let want: f64 = w_back
                .iter()
                .zip(&a_back)
                .map(|(x, y)| *x as f64 * *y as f64)
                .sum();
            assert!((dot as f64 - want).abs() < 1e-4, "{name}: {dot} vs {want}");
        };
        let mut m = vec![BlockMxfp4 { e: 0, qs: [0; 16] }; 2];
        BlockMxfp4::from_float(&w, &mut m);
        let mut mb = vec![0f32; 64];
        BlockMxfp4::to_float(&m, &mut mb);
        check(BlockMxfp4::vec_dot(64, &m, &q8), &mb, "mxfp4");
        let mut f = vec![
            BlockFp8E4M3 {
                d: bf16::ZERO,
                qs: [0; 32]
            };
            2
        ];
        BlockFp8E4M3::from_float(&w, &mut f);
        let mut fb = vec![0f32; 64];
        BlockFp8E4M3::to_float(&f, &mut fb);
        check(BlockFp8E4M3::vec_dot(64, &f, &q8), &fb, "fp8");
    }
}
