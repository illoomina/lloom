//! The routed matvec's union form (`kernels.quant_matvec.routed_union`,
//! `LLOOM_QMV_ID_UNION`): over two to eight rows, the threadgroup of the
//! first plane naming a slot runs every plane naming it, so a slot the rows
//! share is read once. It is the routed row axis at capacity 1
//! (`kernel_mul_mv_id{,2}_<stem>_<triple>_r1`, `docs/tuning.md` §13.5w), and
//! under `deterministic` (pinned here: the union is the one routed form the
//! contract admits) every plane must be the per-plane launch's, bit for bit —
//! the single launch, the f16 store and the gate|up pair — and
//! within the dense reference's tolerance (`moe_routed_metal`'s, which pins
//! the incumbent). The slot ids repeat across rows and within a row (a missing
//! expert maps to slot 0 on the streamed tier's device plan), so the union
//! runs more than one plane in a threadgroup on every shape.
//!
//! Its own process (nextest's): the contract and the arm are process-wide
//! (`set_qmv_id_union`).
#![cfg(feature = "metal")]

use half::f16;
use lloom_tensor::quantized::{GgmlDType, PairOrder, QStorage, QTensor};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::borrow::Cow;

fn device() -> Result<Device> {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "deterministic".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(!lloom_tensor::tuning::contract().is_fast());
    Device::new_metal(0)
}

/// The kernels a closure ran whose names carry the union's suffix.
fn union_kernels(dev: &Device, f: impl FnOnce() -> Result<()>) -> Result<u64> {
    dev.synchronize()?;
    let window = dev
        .as_metal_device()?
        .dispatch_census()
        .open()
        .expect("a window");
    f()?;
    dev.synchronize()?;
    Ok(window
        .close()
        .iter()
        .filter(|(l, _)| l.starts_with("kernel_mul_mv_id") && l.ends_with("_r1"))
        .map(|(_, n)| *n)
        .sum())
}

fn lcg(seed: u32) -> impl FnMut() -> u8 {
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(12345);
    move || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 24) as u8
    }
}

/// A stacked `[e, n, k]` weight of `dtype` on `dev` and its CPU dequant: a
/// quantizable kind from a smooth f32 weight, a codebook kind from LCG bytes
/// with a sane f16 scale at the head of every block.
fn stacked(
    dtype: GgmlDType,
    (e, n, k): (usize, usize, usize),
    seed: u32,
    dev: &Device,
) -> Result<(QTensor, Tensor)> {
    let raw: Vec<u8> = match dtype {
        GgmlDType::IQ2XXS
        | GgmlDType::IQ2XS
        | GgmlDType::IQ2S
        | GgmlDType::IQ3XXS
        | GgmlDType::IQ3S
        | GgmlDType::IQ4XS
        | GgmlDType::IQ4NL
        | GgmlDType::Q2_0 => {
            let (bs, ts) = (dtype.block_size(), dtype.type_size());
            let blocks = e * n * k / bs;
            let mut next = lcg(seed);
            let mut raw: Vec<u8> = (0..blocks * ts).map(|_| next()).collect();
            for i in 0..blocks {
                let d = f16::from_f32(0.01 + 0.006 * ((i as f32) * 0.37).sin());
                raw[i * ts..i * ts + 2].copy_from_slice(&d.to_le_bytes());
            }
            raw
        }
        _ => {
            let w: Vec<f32> = (0..e * n * k)
                .map(|i| {
                    let i = i as f32 + seed as f32;
                    (i * 0.0037).sin() * 0.5 + (i * 0.011).cos() * 0.25
                })
                .collect();
            let w = Tensor::from_vec(w, (e, n, k), &Device::Cpu)?;
            QTensor::quantize(&w, dtype)?.data()?.into_owned()
        }
    };
    let q = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), dev, dtype)?,
        (e, n, k),
    )?;
    let cpu = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), &Device::Cpu, dtype)?,
        (e, n, k),
    )?;
    Ok((q, cpu.dequantize(&Device::Cpu)?))
}

fn activations((batch, in_dim1, k): (usize, usize, usize), seed: f32) -> Result<Tensor> {
    let data: Vec<f32> = (0..batch * in_dim1 * k)
        .map(|i| ((i as f32) * 0.013 + seed).sin() * 0.8 + ((i as f32) * 0.0071).cos() * 0.3)
        .collect();
    Tensor::from_vec(data, (batch, in_dim1, k), &Device::Cpu)
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    let t = t
        .to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?;
    Ok(t.to_vec1::<f32>()?.into_iter().map(f32::to_bits).collect())
}

/// `out[b, t] = x[b, t or 0] · W[ids[b, t]]ᵀ`, f32, on the CPU.
fn dense(x: &Tensor, w: &Tensor, ids: &[u32], topk: usize) -> Result<Vec<f32>> {
    let (batch, in_dim1, _) = x.dims3()?;
    let mut out = Vec::new();
    for b in 0..batch {
        for t in 0..topk {
            let xr = x
                .get(b)?
                .get(if in_dim1 == 1 { 0 } else { t })?
                .to_dtype(DType::F32)?;
            let we = w.get(ids[b * topk + t] as usize)?;
            out.extend(
                we.matmul(&xr.unsqueeze(1)?)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
        }
    }
    Ok(out)
}

fn rel_err(got: &[f32], want: &[f32]) -> f32 {
    let num: f32 = got.iter().zip(want).map(|(a, b)| (a - b).abs()).sum();
    let den: f32 = want.iter().map(|b| b.abs()).sum::<f32>().max(1e-12);
    num / den
}

/// The slot ids of four rows at top-3 over six experts: slot 5 named by
/// every row, slot 0 twice in row 2, slot 2 by rows 0 and 3.
const IDS: [u32; 12] = [5, 2, 1, 4, 5, 3, 0, 5, 0, 2, 5, 4];
const SHAPE: (usize, usize, usize) = (6, 40, 256);

/// One kind at one activation dtype: the single launch on the gate shape
/// (`in_dim1 = 1`) and the down shape (`in_dim1 = topk`), and the pair.
fn union_is_the_incumbent(dtype: GgmlDType, act: DType) -> Result<()> {
    let dev = device()?;
    let (e, n, k) = SHAPE;
    let (batch, topk) = (4usize, 3usize);
    let (wg, refg) = stacked(dtype, (e, n, k), 11, &dev)?;
    let (wu, refu) = stacked(dtype, (e, n, k), 29, &dev)?;
    let ids = Tensor::from_vec(IDS.to_vec(), (batch, topk), &Device::Cpu)?.to_device(&dev)?;
    for in_dim1 in [1, topk] {
        let x = activations((batch, in_dim1, k), 0.3 + in_dim1 as f32)?.to_dtype(act)?;
        let xd = x.to_device(&dev)?;
        lloom_tensor::kernels::set_qmv_id_union(false);
        let before = lloom_tensor::kernels::qmv_id_union_launches();
        let inc = wg.indexed_moe_forward(&xd, &ids)?;
        let (inc_g, inc_u) = wg.indexed_moe_forward_pair(&wu, &xd, &ids)?;
        assert_eq!(
            lloom_tensor::kernels::qmv_id_union_launches(),
            before,
            "the incumbent took the union"
        );
        lloom_tensor::kernels::set_qmv_id_union(true);
        let mut out = None;
        let in_census = union_kernels(&dev, || {
            out = Some((
                wg.indexed_moe_forward(&xd, &ids)?,
                wg.indexed_moe_forward_pair(&wu, &xd, &ids)?,
            ));
            Ok(())
        })?;
        let (uni, (uni_g, uni_u)) = out.unwrap();
        let launched = lloom_tensor::kernels::qmv_id_union_launches() - before;
        lloom_tensor::kernels::set_qmv_id_union(false);
        // Non-vacuity: the single launch is the union's, and so is the pair
        // on every grid order (the row kernel decodes each order's grid);
        // `PairOrder::Split` is TWO launches, one per half.
        let pair_unions = match PairOrder::configured() {
            PairOrder::Split => 2,
            _ => 1,
        };
        assert_eq!(
            in_census,
            launched,
            "{dtype:?} {act:?} in_dim1={in_dim1}: the counter and the census's `_r1` kernels disagree"
        );
        assert_eq!(
            launched,
            1 + pair_unions,
            "{dtype:?} {act:?} in_dim1={in_dim1}: {launched} union launches of {} on the {} grid",
            1 + pair_unions,
            PairOrder::configured().name()
        );
        let tag = format!("{dtype:?} {act:?} in_dim1={in_dim1}");
        for (what, a, b) in [
            ("single", &inc, &uni),
            ("pair gate", &inc_g, &uni_g),
            ("pair up", &inc_u, &uni_u),
        ] {
            let (a, b) = (bits(a)?, bits(b)?);
            if let Some(i) = (0..a.len()).find(|&i| a[i] != b[i]) {
                panic!(
                    "{tag} {what}: element {i} (plane {}) is {} under the union, {} under the incumbent",
                    i / n,
                    f32::from_bits(b[i]),
                    f32::from_bits(a[i])
                );
            }
        }
        // The absolute reference, as `moe_routed_metal` bounds the incumbent.
        let xr = x.to_dtype(DType::F32)?;
        let tol = if act == DType::F16 { 4e-3 } else { 1e-4 };
        for (what, got, w) in [
            ("single", &uni, &refg),
            ("pair gate", &uni_g, &refg),
            ("pair up", &uni_u, &refu),
        ] {
            let got: Vec<f32> = got
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let err = rel_err(&got, &dense(&xr, w, &IDS, topk)?);
            assert!(
                err < tol,
                "{tag} {what}: mean rel err {err} against the dense reference"
            );
        }
        println!("UNION {tag}: single, pair bit-identical to the incumbent");
    }
    Ok(())
}

/// One launch of one row is never the union's: nothing to share.
#[test]
fn a_one_row_launch_is_the_incumbents() -> Result<()> {
    let dev = device()?;
    let (w, _) = stacked(GgmlDType::Q4K, SHAPE, 3, &dev)?;
    let ids = Tensor::from_vec(vec![5u32, 2, 1], (1, 3), &Device::Cpu)?.to_device(&dev)?;
    let x = activations((1, 1, SHAPE.2), 0.1)?.to_device(&dev)?;
    lloom_tensor::kernels::set_qmv_id_union(true);
    let before = lloom_tensor::kernels::qmv_id_union_launches();
    w.indexed_moe_forward(&x, &ids)?;
    let launched = lloom_tensor::kernels::qmv_id_union_launches() - before;
    lloom_tensor::kernels::set_qmv_id_union(false);
    assert_eq!(launched, 0, "a one-row launch took the union form");
    Ok(())
}

/// Past `MV_ID_UNION_MAX_ROWS` rows a launch is the incumbent's: over a
/// prefill tile's rows a slot's threadgroup would run most of the planes
/// one after another.
#[test]
fn a_launch_past_the_union_cap_is_the_incumbents() -> Result<()> {
    let dev = device()?;
    let (w, _) = stacked(GgmlDType::Q4K, SHAPE, 3, &dev)?;
    let rows = lloom_tensor::kernels::MV_ID_UNION_MAX_ROWS + 1;
    let ids: Vec<u32> = (0..rows * 3).map(|i| (i % 6) as u32).collect();
    let ids = Tensor::from_vec(ids, (rows, 3), &Device::Cpu)?.to_device(&dev)?;
    let x = activations((rows, 1, SHAPE.2), 0.1)?.to_device(&dev)?;
    lloom_tensor::kernels::set_qmv_id_union(true);
    let before = lloom_tensor::kernels::qmv_id_union_launches();
    w.indexed_moe_forward(&x, &ids)?;
    let at_cap = {
        let ids = ids.narrow(0, 0, rows - 1)?;
        let x = x.narrow(0, 0, rows - 1)?.contiguous()?;
        let b = lloom_tensor::kernels::qmv_id_union_launches();
        w.indexed_moe_forward(&x, &ids)?;
        lloom_tensor::kernels::qmv_id_union_launches() - b
    };
    let past = lloom_tensor::kernels::qmv_id_union_launches() - before - at_cap;
    lloom_tensor::kernels::set_qmv_id_union(false);
    assert_eq!(past, 0, "a launch of {rows} rows took the union form");
    assert_eq!(
        at_cap,
        1,
        "a launch of {} rows did not take the union form",
        rows - 1
    );
    Ok(())
}

#[test]
fn union_q4k() -> Result<()> {
    union_is_the_incumbent(GgmlDType::Q4K, DType::F32)?;
    union_is_the_incumbent(GgmlDType::Q4K, DType::F16)
}

#[test]
fn union_q8_0() -> Result<()> {
    union_is_the_incumbent(GgmlDType::Q8_0, DType::F32)?;
    union_is_the_incumbent(GgmlDType::Q8_0, DType::F16)
}

#[test]
fn union_iq_kinds_the_checkpoint_routes() -> Result<()> {
    for dtype in [
        GgmlDType::IQ2XXS,
        GgmlDType::IQ2XS,
        GgmlDType::IQ2S,
        GgmlDType::IQ3XXS,
        GgmlDType::IQ3S,
        GgmlDType::IQ4NL,
        GgmlDType::IQ4XS,
        GgmlDType::Q2_0,
    ] {
        union_is_the_incumbent(dtype, DType::F32)?;
        union_is_the_incumbent(dtype, DType::F16)?;
    }
    Ok(())
}
