//! The row axis of the quantized matmul-vec under `fast`: `m` activation
//! rows (2..=8) share one weight read (`kernel_mul_mv_<stem>_<triple>_r<N>`,
//! the core at capacity `N`). Every output row is checked against the CPU
//! dequant reference, at a row count that fills the last threadgroup
//! (`n = 40`) and one that does not (`n = 37`), the dispatch census shows
//! the kernel at the capacity the dispatch names ran once, and the same call
//! twice gives the same bits. A parity test the per-row launches pass is not
//! a test of this axis; the census is what makes it one.
//!
//! The mutation harness `design/measurements/issue-942-raw/mutate-rows.sh`
//! breaks a row's index, the row loop's bound and the dispatch's grid and
//! shows this binary red each time.
//!
//! Its own binary: the contract is read once per process.
#![cfg(feature = "metal")]

mod rows_identity;

use lloom_tensor::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use lloom_tensor::{Device, Module, Result, Tensor};
use rows_identity::KINDS;
use std::borrow::Cow;

const K: usize = 512;

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        // Every kind at every capacity, whatever its measured ceiling.
        t.qmv.rows_max = Some(8);
        t.instrument.profile = true;
        // Rows past the ceiling stay on the matmul-vec (the mm kernel takes
        // over at 9 by default): the grouping check below needs 11.
        t.qmv.mm_from = Some(12);
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(lloom_tensor::tuning::contract().is_fast());
    assert!(lloom_tensor::tuning::tuning().qmv.rows, "the axis is on by default");
    Device::new_metal(0).expect("a Metal device")
}

/// The raw blocks of an `[n, K]` weight: the CPU quantizer's output for the
/// kinds with a quantizer, LCG bytes with a sane f16 scale at offset 0 of
/// every block for the codebook kinds (`iq_metal`'s fixture).
fn raw_weight(dtype: GgmlDType, n: usize, seed: u32) -> Result<Vec<u8>> {
    if matches!(
        dtype,
        GgmlDType::Q6K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q8_0 | GgmlDType::Q4_0
    ) {
        let w: Vec<f32> = (0..n * K)
            .map(|i| (i as f32 * 0.61 + seed as f32).sin() * 0.05)
            .collect();
        let w = Tensor::from_vec(w, (n, K), &Device::Cpu)?;
        return QTensor::quantize(&w, dtype)?.data().map(|d| d.into_owned());
    }
    let (bs, ts) = (dtype.block_size(), dtype.type_size());
    let n_blocks = n * K / bs;
    let mut state = seed.wrapping_mul(2654435761).wrapping_add(777);
    let mut raw = vec![0u8; n_blocks * ts];
    for b in raw.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (state >> 24) as u8;
    }
    for i in 0..n_blocks {
        let d = half::f16::from_f32(0.02 + 0.015 * ((i as f32) * 0.37).sin());
        raw[i * ts..i * ts + 2].copy_from_slice(&d.to_le_bytes());
    }
    Ok(raw)
}

/// The weight on `dev` and its CPU f32 dequant, from the same bytes.
fn weight(dtype: GgmlDType, n: usize, seed: u32, dev: &Device) -> Result<(QMatMul, Tensor)> {
    let raw = raw_weight(dtype, n, seed)?;
    let on_dev = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), dev, dtype)?,
        (n, K),
    )?;
    let on_cpu = QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), &Device::Cpu, dtype)?,
        (n, K),
    )?;
    Ok((
        QMatMul::from_qtensor(on_dev)?,
        on_cpu.dequantize(&Device::Cpu)?,
    ))
}

fn activations(m: usize) -> Result<Tensor> {
    let data = (0..m * K)
        .map(|i| ((i as f32) * 0.013).sin() * 0.8 + ((i as f32) * 0.0071).cos() * 0.3)
        .collect::<Vec<_>>();
    Tensor::from_vec(data, (m, K), &Device::Cpu)
}

/// The worst `|got - want|` over every element relative to the reference's
/// largest magnitude; a NaN anywhere reads as NaN (a NaN-dropping fold
/// would read a poisoned row as agreement).
fn worst_rel(got: &Tensor, want: &Tensor) -> Result<f32> {
    let got = got
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let want = want.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(got.len(), want.len());
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
    Ok(got.iter().zip(&want).fold(0f32, |m, (g, w)| {
        let d = (g - w).abs() / scale;
        if d.is_nan() || m.is_nan() {
            f32::NAN
        } else {
            m.max(d)
        }
    }))
}

fn launches(dev: &Device, f: impl FnOnce()) -> Vec<(String, u64)> {
    dev.synchronize().unwrap();
    let window = dev
        .as_metal_device()
        .unwrap()
        .dispatch_census()
        .open()
        .expect("a census window on this test's device");
    f();
    dev.synchronize().unwrap();
    window.close()
}

fn count(census: &[(String, u64)], pred: impl Fn(&str) -> bool) -> u64 {
    census
        .iter()
        .filter(|(l, _)| pred(l))
        .map(|(_, n)| *n)
        .sum()
}

/// Every output row of the row axis against the CPU reference, `m` 2..=8,
/// the kernel at the dispatch's capacity shown to have run (one launch per
/// call), the same call twice giving the same bits, and the engagement
/// counter reading one launch per call; then a launch past the kind's
/// ceiling, whose rows past it form a second group on the grid rather than
/// being dropped. One test: the census is a per-device window.
#[test]
fn the_row_axis_matches_the_cpu_reference_at_every_row() -> Result<()> {
    let dev = device();
    assert!(
        lloom_tensor::kernels::metal::profile::enabled(),
        "the census decides which kernels ran"
    );
    let before = lloom_tensor::kernels::qmv_rows_launches();
    let mut failures = Vec::new();
    let mut calls = 0u64;
    for dtype in KINDS {
        for n in [37usize, 40] {
            let (w, w_ref) = weight(dtype, n, 7, &dev)?;
            for m in 2..=8usize {
                let x = activations(m)?;
                let want = x.matmul(&w_ref.t()?)?;
                let xd = x.to_device(&dev)?;
                let mut got = None;
                let census = launches(&dev, || got = Some(w.forward(&xd).unwrap()));
                calls += 1;
                let got = got.unwrap();
                let err = worst_rel(&got, &want)?;
                let cap = lloom_tensor::kernels::mv_rows_capacity(dtype.into(), m, false);
                let suffix = format!("_r{cap}");
                let rows = count(&census, |l| l.contains("kernel_mul_mv_") && l.ends_with(&suffix));
                let again = w.forward(&xd)?.to_device(&Device::Cpu)?;
                let same = got
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    == again.flatten_all()?.to_vec1::<f32>()?;
                calls += 1;
                println!("ROWS {dtype:?} n={n} m={m} worst_rel={err:e} {suffix}_launches={rows} repeat_identical={same}");
                if err.is_nan() || err >= 2e-3 {
                    failures.push(format!("{dtype:?} n={n} m={m}: worst rel err {err}"));
                }
                if cap < 2 || rows != 1 {
                    failures.push(format!(
                        "{dtype:?} n={n} m={m}: {rows} {suffix} launches ({census:?})"
                    ));
                }
                if !same {
                    failures.push(format!("{dtype:?} n={n} m={m}: two calls disagree"));
                }
            }
        }
    }
    let counted = lloom_tensor::kernels::qmv_rows_launches() - before;
    if counted != calls {
        failures.push(format!("the counter read {counted} launches for {calls} calls"));
    }
    // Past the ceiling: a second group.
    let dtype = GgmlDType::Q4K;
    let ceiling = lloom_tensor::tuning::tuning().qmv.rows_max.expect("the override this binary sets");
    let m = ceiling + 3;
    let (w, w_ref) = weight(dtype, 40, 3, &dev)?;
    let x = activations(m)?;
    let want = x.matmul(&w_ref.t()?)?;
    let xd = x.to_device(&dev)?;
    let mut got = None;
    let census = launches(&dev, || got = Some(w.forward(&xd).unwrap()));
    let err = worst_rel(&got.unwrap(), &want)?;
    println!("GROUPS {dtype:?} ceiling={ceiling} m={m} worst_rel={err:e} census={census:?}");
    if err.is_nan() || err >= 2e-3 {
        failures.push(format!("{dtype:?} m={m} past the ceiling {ceiling}: worst rel err {err}"));
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}
