//! Quantized weights synthesized as packed blocks (`docs/porting-models.md`
//! §21.8, INV-4): every byte of a block drawn from the tensor's key, then
//! each format's float scale fields overwritten with bounded finite values
//! and the whole tensor scaled so its dequantized magnitude is the law's.
//! Every GGUF kind is covered, including the IQ family and Q2_0, which have
//! no quantizer in the tree; the reference for what a block means is the
//! format's own `to_float`.

use super::init::{Key, Role, ValueLaw};
use lloom_tensor::quantized::{GgmlDType, QStorage, QTensor};
use lloom_tensor::{bail, Device, Result, Shape};
use std::borrow::Cow;

/// The float fields of one block of `dtype`: the byte offsets of its f16
/// scales, and of its f32 scales. Everything else in a block is integer
/// codes, indices into a grid, or sign bits.
pub fn scale_fields(dtype: GgmlDType) -> (&'static [usize], &'static [usize]) {
    use GgmlDType::*;
    match dtype {
        Q4_0 | Q5_0 | Q8_0 => (&[0], &[]),
        Q4_1 | Q5_1 => (&[0, 2], &[]),
        // `s` (offset 2) is `d · Σqs`, a cache for the dot product: it is
        // recomputed, not drawn (`fix_derived`).
        Q8_1 => (&[0], &[]),
        Q2K => (&[80, 82], &[]),
        Q3K => (&[108], &[]),
        Q4K | Q5K => (&[0, 2], &[]),
        Q6K => (&[208], &[]),
        // `bsums` are Σqs per 16, recomputed (`fix_derived`).
        Q8K => (&[], &[0]),
        IQ2XXS | IQ2XS | IQ2S | IQ3XXS | IQ3S | IQ4NL | IQ4XS | Q2_0 => (&[0], &[]),
        F32 | F16 | BF16 => (&[], &[]),
    }
}

/// Whether a kind is stored as packed blocks (rather than as plain floats).
pub fn is_packed(dtype: GgmlDType) -> bool {
    !matches!(dtype, GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16)
}

/// Every kind the GGUF reader accepts, in type-code order.
pub const ALL_KINDS: [GgmlDType; 23] = [
    GgmlDType::F32,
    GgmlDType::F16,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q8_1,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8K,
    GgmlDType::IQ2XXS,
    GgmlDType::IQ2XS,
    GgmlDType::IQ3XXS,
    GgmlDType::IQ4NL,
    GgmlDType::IQ3S,
    GgmlDType::IQ2S,
    GgmlDType::IQ4XS,
    GgmlDType::BF16,
    GgmlDType::Q2_0,
];

fn get_f16(b: &[u8], at: usize) -> f32 {
    half::f16::from_le_bytes([b[at], b[at + 1]]).to_f32()
}
fn put_f16(b: &mut [u8], at: usize, v: f32) {
    b[at..at + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
}
fn get_f32(b: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}
fn put_f32(b: &mut [u8], at: usize, v: f32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Recompute the fields that cache a function of the codes: Q8_1's
/// `s = d · Σqs`, Q8_K's `bsums` (Σqs per 16).
fn fix_derived(dtype: GgmlDType, block: &mut [u8]) {
    match dtype {
        GgmlDType::Q8_1 => {
            let d = get_f16(block, 0);
            let sum: i32 = block[4..36].iter().map(|&q| q as i8 as i32).sum();
            put_f16(block, 2, d * sum as f32);
        }
        GgmlDType::Q8K => {
            for g in 0..16 {
                let s: i32 = block[4 + g * 16..4 + (g + 1) * 16]
                    .iter()
                    .map(|&q| q as i8 as i32)
                    .sum();
                let at = 260 + g * 2;
                block[at..at + 2].copy_from_slice(&(s as i16).to_le_bytes());
            }
        }
        _ => {}
    }
}

/// The packed bytes of a `dtype` tensor of `shape`: codes from `key`, each
/// scale in `[0.5, 1)` of its final magnitude, the whole tensor scaled so
/// its largest dequantized magnitude is `amplitude`.
pub fn synthesize_bytes(
    dtype: GgmlDType,
    key: Key,
    shape: &Shape,
    amplitude: f32,
) -> Result<Vec<u8>> {
    if !is_packed(dtype) {
        bail!("{dtype:?} is not a packed kind")
    }
    let n = shape.elem_count();
    let bs = dtype.block_size();
    if !n.is_multiple_of(bs) || shape.dims().last().is_none_or(|d| !d.is_multiple_of(bs)) {
        bail!("{dtype:?}: shape {shape:?} is not a whole number of {bs}-element blocks per row")
    }
    let ts = dtype.type_size();
    let blocks = n / bs;
    let mut bytes = vec![0u8; blocks * ts];
    for (w, chunk) in bytes.chunks_mut(8).enumerate() {
        let v = key.bits(w as u64).to_le_bytes();
        chunk.copy_from_slice(&v[..chunk.len()]);
    }
    let (f16s, f32s) = scale_fields(dtype);
    // The scales draw from a stream of their own, so they do not repeat the
    // codes they sit beside.
    let skey = Key::new(key.bits(u64::MAX), "scales");
    for (b, block) in bytes.chunks_mut(ts).enumerate() {
        for (j, &at) in f16s.iter().enumerate() {
            let u = skey.unit((b * 4 + j) as u64);
            put_f16(block, at, 0.75 + 0.25 * u);
        }
        for (j, &at) in f32s.iter().enumerate() {
            let u = skey.unit((b * 4 + j) as u64);
            put_f32(block, at, 0.75 + 0.25 * u);
        }
        fix_derived(dtype, block);
    }
    // Scale to the amplitude: every dequantized value is linear in the
    // block's float fields jointly (codes times scales, less a min times a
    // scale), so scaling all of them scales every value.
    let amax = max_abs(&dequant_cpu(&bytes, dtype, shape)?);
    if !amax.is_finite() || amax == 0.0 {
        bail!("{dtype:?}: synthesized blocks dequantize to max |v| = {amax}")
    }
    let k = amplitude / amax;
    for block in bytes.chunks_mut(ts) {
        for &at in f16s {
            let v = get_f16(block, at);
            put_f16(block, at, v * k);
        }
        for &at in f32s {
            let v = get_f32(block, at);
            put_f32(block, at, v * k);
        }
        fix_derived(dtype, block);
    }
    Ok(bytes)
}

/// `bytes` of `dtype` dequantized by the format's own `to_float`.
pub fn dequant_cpu(bytes: &[u8], dtype: GgmlDType, shape: &Shape) -> Result<Vec<f32>> {
    let q = QTensor::new(
        QStorage::from_data(Cow::Borrowed(bytes), &Device::Cpu, dtype)?,
        shape.clone(),
    )?;
    q.dequantize(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0f32, |a, x| {
        if x.is_finite() {
            a.max(x.abs())
        } else {
            f32::NAN
        }
    })
}

/// The largest magnitude a law gives a tensor of `role`: what a packed
/// tensor is scaled to, so a quantized weight sits at the float law's scale.
pub fn amplitude(law: ValueLaw, role: Role) -> f32 {
    match (law, role) {
        (ValueLaw::RampMod17, _) => 0.25,
        (_, Role::Const(c)) => c.abs().max(1e-3),
        (_, Role::Range { lo, hi }) => lo.abs().max(hi.abs()),
        (ValueLaw::HashUniform { gain }, Role::Matrix { fan_in }) => {
            gain / (fan_in.max(1) as f32).sqrt()
        }
        (ValueLaw::Uniform { half }, Role::Matrix { .. }) => half,
        (ValueLaw::RampSin, Role::Matrix { .. }) => 0.08,
    }
}

/// How a packed kind is produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Synthesis {
    /// Random blocks with bounded scales (every kind).
    #[default]
    Packed,
    /// The law's floats through the format's quantizer (kinds that have one).
    Quantized,
}

/// One weight of `dtype`: floats from the law for F32 / F16 / BF16; packed
/// blocks, or the law's floats quantized, for the rest. Built on the CPU and
/// moved to `dev` as bytes, so it is the same tensor on every device.
#[allow(clippy::too_many_arguments)]
pub fn qtensor(
    law: ValueLaw,
    seed: u64,
    name: &str,
    role: Role,
    shape: &Shape,
    dtype: GgmlDType,
    how: Synthesis,
    dev: &Device,
) -> Result<QTensor> {
    let key = Key::new(seed, name);
    let bytes: Vec<u8> = match dtype {
        GgmlDType::F32 => law
            .fill(seed, name, role, shape.elem_count())
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect(),
        GgmlDType::F16 => law
            .fill(seed, name, role, shape.elem_count())
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect(),
        GgmlDType::BF16 => law
            .fill(seed, name, role, shape.elem_count())
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect(),
        _ => match how {
            Synthesis::Packed => synthesize_bytes(dtype, key, shape, amplitude(law, role))?,
            Synthesis::Quantized => {
                if dtype.is_read_only() {
                    bail!("{dtype:?} has no quantizer in the tree; synthesize it packed")
                }
                let t = law.tensor(seed, name, role, shape, &Device::Cpu)?;
                QTensor::quantize(&t, dtype)?.data()?.into_owned()
            }
        },
    };
    QTensor::new(
        QStorage::from_data(Cow::Owned(bytes), dev, dtype)?,
        shape.clone(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// P-c on the CPU: for every packed kind, random bits outside the scale
    /// fields dequantize to finite values at the requested amplitude, and the
    /// synthesis is keyed (same key, same bytes; another key, other bytes).
    #[test]
    fn every_packed_kind_synthesizes_finite_bounded_blocks() {
        for dtype in ALL_KINDS.into_iter().filter(|d| is_packed(*d)) {
            let shape = Shape::from((3, 2 * dtype.block_size().max(256)));
            let key = Key::new(11, "blk.0.ffn_down_exps.weight");
            let a = synthesize_bytes(dtype, key, &shape, 0.05).unwrap();
            let b = synthesize_bytes(dtype, key, &shape, 0.05).unwrap();
            assert_eq!(a, b, "{dtype:?}: keyed");
            let c = synthesize_bytes(
                dtype,
                Key::new(12, "blk.0.ffn_down_exps.weight"),
                &shape,
                0.05,
            )
            .unwrap();
            assert_ne!(a, c, "{dtype:?}: another seed");
            let v = dequant_cpu(&a, dtype, &shape).unwrap();
            assert!(v.iter().all(|x| x.is_finite()), "{dtype:?}: non-finite");
            let amax = v.iter().fold(0f32, |m, x| m.max(x.abs()));
            // f16 rounding of the rescaled scales moves the peak by < 0.1 %.
            assert!((amax - 0.05).abs() < 0.05 * 2e-3, "{dtype:?}: amax {amax}");
            let distinct = v
                .iter()
                .map(|x| x.to_bits())
                .collect::<std::collections::BTreeSet<_>>();
            assert!(
                distinct.len() > 8,
                "{dtype:?}: {} distinct values",
                distinct.len()
            );
        }
    }

    /// The derived fields agree with the codes after synthesis.
    #[test]
    fn the_cached_sums_are_recomputed() {
        let shape = Shape::from((1, 256));
        let q8k = synthesize_bytes(GgmlDType::Q8K, Key::new(1, "x"), &shape, 1.0).unwrap();
        for g in 0..16 {
            let s: i32 = q8k[4 + g * 16..4 + (g + 1) * 16]
                .iter()
                .map(|&q| q as i8 as i32)
                .sum();
            assert_eq!(
                i16::from_le_bytes([q8k[260 + 2 * g], q8k[261 + 2 * g]]) as i32,
                s
            );
        }
        let q81 = synthesize_bytes(
            GgmlDType::Q8_1,
            Key::new(1, "x"),
            &Shape::from((1, 32)),
            1.0,
        )
        .unwrap();
        let d = get_f16(&q81, 0);
        let sum: i32 = q81[4..36].iter().map(|&q| q as i8 as i32).sum();
        assert_eq!(
            get_f16(&q81, 2),
            half::f16::from_f32(d * sum as f32).to_f32()
        );
    }

    /// The float kinds come from the law; a quantized arm exists where the
    /// format has a quantizer and is refused by name where it does not.
    #[test]
    fn float_kinds_follow_the_law_and_the_quantized_arm_is_named() {
        let shape = Shape::from((4, 32));
        let role = Role::Matrix { fan_in: 32 };
        let law = ValueLaw::DEFAULT;
        let want = law.fill(5, "w", role, 128);
        let f32q = qtensor(
            law,
            5,
            "w",
            role,
            &shape,
            GgmlDType::F32,
            Synthesis::Packed,
            &Device::Cpu,
        )
        .unwrap();
        let got = f32q
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(got, want);
        let q80 = qtensor(
            law,
            5,
            "w",
            role,
            &shape,
            GgmlDType::Q8_0,
            Synthesis::Quantized,
            &Device::Cpu,
        )
        .unwrap();
        let back = q80
            .dequantize(&Device::Cpu)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let err = back
            .iter()
            .zip(&want)
            .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(err < 2e-3, "{err}");
        let e = qtensor(
            law,
            5,
            "w",
            role,
            &Shape::from((1, 256)),
            GgmlDType::IQ2XXS,
            Synthesis::Quantized,
            &Device::Cpu,
        )
        .unwrap_err()
        .to_string();
        assert!(e.contains("no quantizer"), "{e}");
    }
}
