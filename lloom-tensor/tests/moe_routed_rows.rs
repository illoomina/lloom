//! The routed matmul-vec on the row axis (`kernel_mul_mv_id{,2}_<stem>_<triple>_r<N>`,
//! `docs/tuning.md` §13.5v): over two or more rows under `fast`, the
//! threadgroup of the first plane naming a slot computes every plane naming
//! it, `N` at a time through one core call, so a slot the rows share is read
//! once. Every plane — the single launch, the f16 store and the gate|up pair
//! — is checked against the dense reference (`moe_routed_metal`'s bound) and
//! against the per-plane launch with the axis off (each plane the one-row
//! core's sum in its order: 1e-4 relative, the bit agreement printed), with
//! the census showing the kernel at the capacity the dispatch names. The
//! slot ids repeat across rows and within a row (a missing expert maps to
//! slot 0 on the streamed tier's device plan), so a slot's planes exceed one
//! core call's rows on every shape the capacity-2 kernel sees.
//!
//! Its own binary: the contract and the axis are process-wide.
#![cfg(feature = "metal")]

use half::f16;
use lloom_tensor::quantized::{GgmlDType, QStorage, QTensor};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::borrow::Cow;

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        // Every kind at every capacity, whatever its measured ceiling.
        t.qmv.rows_max = Some(8);
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(lloom_tensor::tuning::contract().is_fast());
    Device::new_metal(0).expect("a Metal device")
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

fn values(t: &Tensor) -> Result<Vec<f32>> {
    t.to_device(&Device::Cpu)?
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()
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

/// The worst element against the per-plane launch, relative to its scale,
/// and how many elements agree bit for bit.
fn against_incumbent(got: &[f32], inc: &[f32]) -> (f32, usize) {
    let scale = inc.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
    let worst = got
        .iter()
        .zip(inc)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs() / scale));
    let same = got
        .iter()
        .zip(inc)
        .filter(|(a, b)| a.to_bits() == b.to_bits())
        .count();
    (worst, same)
}

fn census(dev: &Device, f: impl FnOnce() -> Result<()>) -> Result<Vec<(String, u64)>> {
    dev.synchronize()?;
    let window = dev
        .as_metal_device()?
        .dispatch_census()
        .open()
        .expect("a census window on this test's device");
    f()?;
    dev.synchronize()?;
    Ok(window.close())
}

/// The slot ids of four rows at top-3 over six experts: slot 5 named by
/// every row, slot 0 twice in row 2, slot 2 by rows 0 and 3.
const IDS: [u32; 12] = [5, 2, 1, 4, 5, 3, 0, 5, 0, 2, 5, 4];
const SHAPE: (usize, usize, usize) = (6, 40, 256);

/// One kind at one activation dtype: the single launch on the gate shape
/// (`in_dim1 = 1`) and the down shape (`in_dim1 = topk`), and the pair, at
/// four rows (capacity 4) and at two (capacity 2, where slot 5's four planes
/// take two core calls).
fn rows_are_the_incumbent(dev: &Device, dtype: GgmlDType, act: DType) -> Result<()> {
    let (e, n, k) = SHAPE;
    let topk = 3usize;
    let (wg, refg) = stacked(dtype, (e, n, k), 11, dev)?;
    let (wu, refu) = stacked(dtype, (e, n, k), 29, dev)?;
    for batch in [4usize, 2] {
        let ids_v: Vec<u32> = IDS[..batch * topk].to_vec();
        let ids = Tensor::from_vec(ids_v.clone(), (batch, topk), &Device::Cpu)?.to_device(dev)?;
        let cap = lloom_tensor::kernels::mv_rows_capacity(dtype.into(), batch, true);
        assert!(cap > 1, "{dtype:?}: no capacity at {batch} rows under fast");
        for in_dim1 in [1, topk] {
            let x = activations((batch, in_dim1, k), 0.3 + in_dim1 as f32)?.to_dtype(act)?;
            let xd = x.to_device(dev)?;
            lloom_tensor::kernels::set_qmv_rows(false);
            let inc = wg.indexed_moe_forward(&xd, &ids)?;
            let (inc_g, inc_u) = wg.indexed_moe_forward_pair(&wu, &xd, &ids)?;
            lloom_tensor::kernels::set_qmv_rows(true);
            let before = lloom_tensor::kernels::qmv_id_rows_launches();
            let mut got = None;
            let c = census(dev, || {
                got = Some((
                    wg.indexed_moe_forward(&xd, &ids)?,
                    wg.indexed_moe_forward_pair(&wu, &xd, &ids)?,
                ));
                Ok(())
            })?;
            let (uni, (uni_g, uni_u)) = got.unwrap();
            let launched = lloom_tensor::kernels::qmv_id_rows_launches() - before;
            let suffix = format!("_r{cap}");
            let at_cap: u64 = c
                .iter()
                .filter(|(l, _)| l.starts_with("kernel_mul_mv_id") && l.ends_with(&suffix))
                .map(|(_, n)| *n)
                .sum();
            let tag = format!("{dtype:?} {act:?} batch={batch} in_dim1={in_dim1}");
            assert!(
                launched >= 2 && at_cap == launched,
                "{tag}: {launched} launches counted, {at_cap} at {suffix} in the census ({c:?})"
            );
            for (what, a, b) in [
                ("single", &uni, &inc),
                ("pair gate", &uni_g, &inc_g),
                ("pair up", &uni_u, &inc_u),
            ] {
                let (worst, same) = against_incumbent(&values(a)?, &values(b)?);
                println!("ROWS_ID {tag} {what}: worst_rel_vs_per_plane={worst:e} bit_equal={same}/{}", n * batch * topk);
                assert!(worst < 1e-4, "{tag} {what}: {worst:e} off the per-plane launch");
            }
            // The absolute reference, as `moe_routed_metal` bounds the incumbent.
            let xr = x.to_dtype(DType::F32)?;
            let tol = if act == DType::F16 { 4e-3 } else { 1e-4 };
            for (what, got, w) in [
                ("single", &uni, &refg),
                ("pair gate", &uni_g, &refg),
                ("pair up", &uni_u, &refu),
            ] {
                let err = rel_err(&values(got)?, &dense(&xr, w, &ids_v, topk)?);
                assert!(
                    err < tol,
                    "{tag} {what}: mean rel err {err} against the dense reference"
                );
            }
        }
    }
    Ok(())
}

/// One launch of one row never takes the axis: nothing to share. One test
/// over every kind: the axis is process-wide and the census a per-device
/// window.
#[test]
fn the_routed_row_axis_is_each_plane_the_one_row_cores_sum() -> Result<()> {
    let dev = device();
    assert!(lloom_tensor::kernels::metal::profile::enabled());
    let (w, _) = stacked(GgmlDType::Q4K, SHAPE, 3, &dev)?;
    let ids = Tensor::from_vec(vec![5u32, 2, 1], (1, 3), &Device::Cpu)?.to_device(&dev)?;
    let x = activations((1, 1, SHAPE.2), 0.1)?.to_device(&dev)?;
    let before = lloom_tensor::kernels::qmv_id_rows_launches();
    w.indexed_moe_forward(&x, &ids)?;
    assert_eq!(
        lloom_tensor::kernels::qmv_id_rows_launches() - before,
        0,
        "a one-row launch took the row axis"
    );
    for dtype in [
        GgmlDType::Q4K,
        GgmlDType::Q8_0,
        GgmlDType::IQ2XXS,
        GgmlDType::IQ2XS,
        GgmlDType::IQ2S,
        GgmlDType::IQ3XXS,
        GgmlDType::IQ3S,
        GgmlDType::IQ4NL,
        GgmlDType::IQ4XS,
        GgmlDType::Q2_0,
    ] {
        rows_are_the_incumbent(&dev, dtype, DType::F32)?;
        rows_are_the_incumbent(&dev, dtype, DType::F16)?;
    }
    Ok(())
}
