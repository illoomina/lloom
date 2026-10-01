//! The small-batch quantized matmul-vec (`LLOOM_QMV_ROWS=on`): `m`
//! activation rows (2..=8) share one weight read (`kernel_mul_mv_*_rows_impl`,
//! `NR1` rows a threadgroup) for the five stem types it is built for. Every
//! output row is checked against the CPU dequant reference, at a row count
//! that fills the last threadgroup (`n = 40`) and one that does not (`n = 37`),
//! and the dispatch census shows the rows kernels ran — a parity test the
//! per-row launches pass is not a test of this arm. Each rows kernel indexes
//! its rows for its own threadgroup geometry, so a one-row core whose
//! geometry is tuned (`q6k_nr0`, `iq_nsg`) must not lend it its grid.
//!
//! Its own binary: the arm's switch is read once per process, installed here
//! before any tensor exists. The default (the per-row launches, bit-identical
//! to `m` one-row launches) is `qmatmul_rows_default`.
#![cfg(feature = "metal")]

use half::f16;
use lloom_tensor::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use lloom_tensor::{Device, Module, Result, Tensor};
use std::borrow::Cow;

const TYPES: [GgmlDType; 5] = [
    GgmlDType::Q6K,
    GgmlDType::Q4K,
    GgmlDType::IQ4XS,
    GgmlDType::IQ4NL,
    GgmlDType::IQ3S,
];
const K: usize = 512;

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.qmv.rows = true;
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(
        lloom_tensor::tuning::tuning().qmv.rows,
        "the switch was read before this binary installed it"
    );
    Device::new_metal(0).expect("a Metal device")
}

/// The raw blocks of an `[n, K]` weight: LCG bytes with a sane f16 scale at
/// offset 0 of every block for the codebook types (`iq_metal`'s fixture),
/// the CPU quantizer's output for the K-quants.
fn raw_weight(dtype: GgmlDType, n: usize, seed: u32) -> Result<Vec<u8>> {
    if matches!(dtype, GgmlDType::Q6K | GgmlDType::Q4K) {
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
        let d = f16::from_f32(0.02 + 0.015 * ((i as f32) * 0.37).sin());
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

/// Every output row of the rows arm against the CPU reference, `m` 2..=8,
/// the rows kernels shown to have run (one launch per call, `_r2` at two
/// rows and `_r4` above), and the same call twice giving the same bits.
#[test]
fn the_rows_arm_matches_the_cpu_reference_at_every_row() -> Result<()> {
    let dev = device();
    assert!(
        lloom_tensor::kernels::metal::profile::enabled(),
        "the census decides which kernels ran"
    );
    let mut failures = Vec::new();
    for dtype in TYPES {
        for n in [37usize, 40] {
            let (w, w_ref) = weight(dtype, n, 7, &dev)?;
            for m in 2..=8usize {
                let x = activations(m)?;
                let want = x.matmul(&w_ref.t()?)?;
                let xd = x.to_device(&dev)?;
                let mut got = None;
                let census = launches(&dev, || got = Some(w.forward(&xd).unwrap()));
                let got = got.unwrap();
                let err = worst_rel(&got, &want)?;
                let nr1 = if m <= 2 { "_r2" } else { "_r4" };
                let rows = count(&census, |l| l.contains("kernel_mul_mv_") && l.contains(nr1));
                let again = w.forward(&xd)?.to_device(&Device::Cpu)?;
                let same = got
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    == again.flatten_all()?.to_vec1::<f32>()?;
                println!("ROWS {dtype:?} n={n} m={m} worst_rel={err:e} {nr1}_launches={rows} repeat_identical={same}");
                if !(err < 2e-3) {
                    failures.push(format!("{dtype:?} n={n} m={m}: worst rel err {err}"));
                }
                if rows != 1 {
                    failures.push(format!(
                        "{dtype:?} n={n} m={m}: {rows} {nr1} launches ({census:?})"
                    ));
                }
                if !same {
                    failures.push(format!("{dtype:?} n={n} m={m}: two calls disagree"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

/// The arm is a different contraction of the same sum, not a different sum:
/// each row against the one-row core's result for that row, far inside the
/// reference bound above.
#[test]
fn the_rows_arm_is_a_reordering_of_the_one_row_core() -> Result<()> {
    let dev = device();
    let mut failures = Vec::new();
    for dtype in TYPES {
        let (w, _) = weight(dtype, 40, 11, &dev)?;
        let m = 4;
        let x = activations(m)?.to_device(&dev)?;
        let rows = w.forward(&x)?;
        for r in 0..m {
            let one = w.forward(&x.narrow(0, r, 1)?.contiguous()?)?;
            let err = worst_rel(&rows.narrow(0, r, 1)?, &one.to_device(&Device::Cpu)?)?;
            println!("REORDER {dtype:?} row={r} worst_rel_vs_one_row={err:e}");
            if !(err < 1e-5) {
                failures.push(format!("{dtype:?} row {r}: {err} against the one-row core"));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
