//! The routed SwiGLU on the gate|up launch's store
//! (`QTensor::indexed_moe_forward_glu_as`, `kernels.fusion.routed_glu`;
//! `docs/expert-streaming.md` §20.15f), against the path it replaces: the
//! one-launch pair (`indexed_moe_forward_pair_as`) then `silu_mul`, the
//! separate launch, bit for bit. Every routed kind the pair is held at, f32
//! and f16 activations, a row count on and off every kind's tile (64, 37), one
//! row and nine (a decode step, and a plain pair over several planes), with
//! the dispatch census showing one GLU launch and no `silu_mul` on the arm,
//! and the pair plus `silu_mul` on the reference. Each test prints the Metal
//! API it ran on (`LLOOM_METAL_API`).
use super::kernels::silu_mul;
use lloom_tensor::kernels::metal::profile;
use lloom_tensor::quantized::{GgmlDType, MoeArm, QStorage, QTensor};
use lloom_tensor::{DType, Device, Tensor};
use std::borrow::Cow;

/// The checkpoints' gate|up kinds, their `down` kinds, the tiny model's Q8_0,
/// the K-quants, the legacy blocks, and the dense f32 / f16 experts (f32 only).
const KINDS: [GgmlDType; 17] = [
    GgmlDType::IQ2XS,
    GgmlDType::IQ2XXS,
    GgmlDType::IQ2S,
    GgmlDType::IQ3XXS,
    GgmlDType::IQ3S,
    GgmlDType::IQ4NL,
    GgmlDType::IQ4XS,
    GgmlDType::Q2_0,
    GgmlDType::Q8_0,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::F32,
    GgmlDType::F16,
];

const E: usize = 6;
const K: usize = 256;
const TOPK: usize = 3;

fn lcg(seed: u32) -> impl FnMut() -> u8 {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
    move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8
    }
}

fn dense(dtype: GgmlDType) -> bool {
    matches!(dtype, GgmlDType::F32 | GgmlDType::F16)
}

/// A stacked `[E, n, K]` weight of `dtype` on `dev`: codebook kinds as LCG
/// bytes with a sane f16 scale at each block's head (every pattern is a
/// valid block), the others a smooth f32 weight quantized on the CPU.
fn weight(dtype: GgmlDType, n: usize, seed: u32, dev: &Device) -> QTensor {
    let raw: Vec<u8> = match dtype {
        GgmlDType::IQ2XS
        | GgmlDType::IQ2XXS
        | GgmlDType::IQ2S
        | GgmlDType::IQ3XXS
        | GgmlDType::IQ3S
        | GgmlDType::IQ4NL
        | GgmlDType::IQ4XS
        | GgmlDType::Q2_0 => {
            let (bs, ts) = (dtype.block_size(), dtype.type_size());
            let blocks = E * n * K / bs;
            let mut next = lcg(seed);
            let mut raw: Vec<u8> = (0..blocks * ts).map(|_| next()).collect();
            for i in 0..blocks {
                let d = half::f16::from_f32(0.02 + 0.015 * ((i as f32) * 0.37).sin());
                raw[i * ts..i * ts + 2].copy_from_slice(&d.to_le_bytes());
            }
            raw
        }
        _ => {
            let w: Vec<f32> = (0..E * n * K)
                .map(|i| {
                    let i = i as f32 + seed as f32;
                    (i * 0.0037).sin() * 0.5 + (i * 0.011).cos() * 0.25
                })
                .collect();
            let w = Tensor::from_vec(w, (E, n, K), &Device::Cpu).unwrap();
            QTensor::quantize(&w, dtype)
                .unwrap()
                .data()
                .unwrap()
                .into_owned()
        }
    };
    QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), dev, dtype).unwrap(),
        (E, n, K),
    )
    .unwrap()
}

fn activations(rows: usize, act: DType, dev: &Device) -> Tensor {
    let data: Vec<f32> = (0..rows * K)
        .map(|i| ((i as f32) * 0.013).sin() * 0.8 + ((i as f32) * 0.0071).cos() * 0.3)
        .collect();
    Tensor::from_vec(data, (rows, 1, K), dev)
        .unwrap()
        .to_dtype(act)
        .unwrap()
}

fn ids(rows: usize, dev: &Device) -> Tensor {
    let v: Vec<u32> = (0..rows * TOPK)
        .map(|i| ((i * 5 + i / TOPK) % E) as u32)
        .collect();
    Tensor::from_vec(v, (rows, TOPK), dev).unwrap()
}

fn bits(t: &Tensor) -> Vec<u32> {
    t.flatten_all()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn metal() -> Option<Device> {
    std::env::set_var("CANDLE_METAL_PROFILE", "1");
    let dev = Device::new_metal(0).ok()?;
    assert!(profile::enabled(), "the census needs the profile switch");
    api_line(&dev);
    Some(dev)
}

/// The queue this device installed, beside the one `LLOOM_METAL_API` asked
/// for; a request the device did not take fails here rather than passing on
/// the other API.
fn api_line(dev: &Device) {
    let asked = std::env::var("LLOOM_METAL_API").unwrap_or_else(|_| "default".into());
    let actual = dev
        .as_metal_device()
        .unwrap()
        .commands()
        .actual_command_api();
    eprintln!("ROUTED_GLU api requested={asked} actual={actual}");
    if asked == "metal3" || asked == "metal4" {
        assert_eq!(
            actual, asked,
            "the device did not take the requested Metal API"
        );
    }
}

/// `(label, launches)` of every dispatch encoded while `f` runs.
fn census<T>(dev: &Device, f: impl FnOnce() -> T) -> (T, Vec<(String, u64)>) {
    dev.synchronize().unwrap();
    let w = dev
        .as_metal_device()
        .unwrap()
        .dispatch_census()
        .open()
        .unwrap();
    let out = f();
    dev.synchronize().unwrap();
    let c = w
        .close()
        .into_iter()
        .map(|(l, n)| (l.to_string(), n))
        .collect();
    (out, c)
}

fn launches(c: &[(String, u64)], pred: impl Fn(&str) -> bool) -> u64 {
    c.iter().filter(|(l, _)| pred(l)).map(|(_, n)| *n).sum()
}

/// One kind, activation dtype, `n` and row count: the GLU launch's output is
/// `silu_mul` over the pair's halves bit for bit, it is not the gate half
/// (the epilogue ran), it is the same over two runs, and the census reads one
/// `_glu` launch with no `silu_mul` against the pair's one `mul_mv_id2` and
/// one `silu_mul`.
fn glu_is_the_pair_then_silu_mul(
    dev: &Device,
    dtype: GgmlDType,
    act: DType,
    n: usize,
    rows: usize,
) {
    let g = weight(dtype, n, 17, dev);
    let u = weight(dtype, n, 23, dev);
    let x = activations(rows, act, dev);
    let ids = ids(rows, dev);
    let what = format!("{dtype:?} {act:?} n={n} rows={rows}");
    let ((pg, pu, want), reference) = census(dev, || {
        let (pg, pu) = g
            .indexed_moe_forward_pair_as(&u, &x, &ids, MoeArm::MatVec, act)
            .unwrap();
        let want = silu_mul(&pg, &pu)
            .expect("silu_mul serves the pair")
            .unwrap();
        (pg, pu, want)
    });
    assert_ne!(bits(&pg), bits(&pu), "{what}: the two stacks must differ");
    let (got, fused) = census(dev, || {
        g.indexed_moe_forward_glu_as(&u, &x, &ids, MoeArm::MatVec, act)
            .unwrap()
            .unwrap_or_else(|| panic!("{what}: the GLU launch declined a plain pair"))
    });
    assert_eq!(got.dims(), &[rows, TOPK, n], "{what}: shape");
    assert_eq!(got.dtype(), want.dtype(), "{what}: store dtype");
    assert_eq!(
        bits(&got),
        bits(&want),
        "{what}: not the pair then silu_mul"
    );
    assert_ne!(
        bits(&got),
        bits(&pg),
        "{what}: the store is the gate's, no GLU ran"
    );
    let again = g
        .indexed_moe_forward_glu_as(&u, &x, &ids, MoeArm::MatVec, act)
        .unwrap()
        .unwrap();
    assert_eq!(bits(&again), bits(&got), "{what}: not bit-stable");

    let is_glu = |l: &str| l.starts_with("kernel_mul_mv_id2_") && l.ends_with("_glu");
    let is_silu_mul = |l: &str| l.starts_with("silu_mul_");
    let is_pair = |l: &str| l.starts_with("kernel_mul_mv_id2_") && !l.ends_with("_glu");
    assert_eq!(
        (
            launches(&fused, is_glu),
            launches(&fused, is_pair),
            launches(&fused, is_silu_mul)
        ),
        (1, 0, 0),
        "{what}: the arm's census {fused:?}"
    );
    assert_eq!(
        (
            launches(&reference, is_glu),
            launches(&reference, is_pair),
            launches(&reference, is_silu_mul)
        ),
        (0, 1, 1),
        "{what}: the reference's census {reference:?}"
    );
}

#[test]
fn the_routed_glu_is_the_pair_then_silu_mul_at_f32() {
    let Some(dev) = metal() else { return };
    for dtype in KINDS {
        for n in [64, 37] {
            for rows in [1, 9] {
                glu_is_the_pair_then_silu_mul(&dev, dtype, DType::F32, n, rows);
            }
        }
    }
}

#[test]
fn the_routed_glu_is_the_pair_then_silu_mul_at_f16() {
    let Some(dev) = metal() else { return };
    for dtype in KINDS.into_iter().filter(|d| !dense(*d)) {
        for n in [64, 37] {
            for rows in [1, 9] {
                glu_is_the_pair_then_silu_mul(&dev, dtype, DType::F16, n, rows);
            }
        }
    }
}

/// The launch serves the plain pair only: the GEMM arm, an f32 activation
/// computed in f16 (the fold), dense f16 activations, two stacks of different
/// kinds, and every row count the plain pair kernel does not take
/// (`mv_id2_glu_applies`: the row axis, the union) decline, and the caller's
/// pair + `silu_mul` runs. At one row it serves.
#[test]
fn the_routed_glu_declines_where_the_pair_is_not_its_plain_launch() {
    let Some(dev) = metal() else { return };
    let n = 64;
    let q = weight(GgmlDType::Q8_0, n, 17, &dev);
    let q2 = weight(GgmlDType::Q8_0, n, 23, &dev);
    let f = weight(GgmlDType::F32, n, 23, &dev);
    let x = activations(1, DType::F32, &dev);
    let i1 = ids(1, &dev);
    let glu = |a: &QTensor, b: &QTensor, x: &Tensor, ids: &Tensor, arm, act| {
        a.indexed_moe_forward_glu_as(b, x, ids, arm, act).unwrap()
    };
    assert!(
        glu(&q, &q2, &x, &i1, MoeArm::MatVec, DType::F32).is_some(),
        "one row, f32"
    );
    assert!(
        glu(&q, &q2, &x, &i1, MoeArm::Gemm, DType::F32).is_none(),
        "the GEMM arm"
    );
    assert!(
        glu(&q, &q2, &x, &i1, MoeArm::MatVec, DType::F16).is_none(),
        "the fold"
    );
    assert!(
        glu(&q, &f, &x, &i1, MoeArm::MatVec, DType::F32).is_none(),
        "two kinds"
    );
    let fw = weight(GgmlDType::F32, n, 17, &dev);
    let x16 = activations(1, DType::F16, &dev);
    assert!(
        glu(&fw, &f, &x16, &i1, MoeArm::MatVec, DType::F16).is_none(),
        "dense at f16"
    );
    for dtype in KINDS {
        let (a, b) = (weight(dtype, n, 17, &dev), weight(dtype, n, 23, &dev));
        for rows in 1..=10 {
            let x = activations(rows, DType::F32, &dev);
            let ids = ids(rows, &dev);
            let served = glu(&a, &b, &x, &ids, MoeArm::MatVec, DType::F32).is_some();
            let plain = lloom_tensor::kernels::mv_id2_glu_applies(
                dtype.into(),
                lloom_tensor::kernels::GemvTriple::F32F32F32,
                rows,
            );
            assert_eq!(served, plain, "{dtype:?} rows={rows}");
        }
    }
}
