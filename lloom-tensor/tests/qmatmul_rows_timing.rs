//! The row axis timed, per kind and capacity, at the 27B's `attn_qkv` shape
//! (`[10240 × 2560]`): one launch of `m = N` rows at capacity `N` against
//! `N` one-row launches and against the per-row launch of `N` rows (the
//! `deterministic` form) — the ceiling each kind's `rows_max` records is
//! the largest capacity that reads faster per row than the per-row launch.
//! A measurement, ignored by default; run by hand under `lloom-arb measure`
//! with `--ignored --nocapture` on a `fast` process, the record in
//! `design/measurements/issue-942-rows-axis.md`. The codebook kinds have no
//! in-tree quantizer, so their weights are LCG bytes (the timing does not
//! read the values).
#![cfg(feature = "metal")]
use half::f16;
use lloom_tensor::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use lloom_tensor::{Device, Module, Tensor};
use std::borrow::Cow;
use std::time::Instant;

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(lloom_tensor::tuning::contract().is_fast());
    Device::new_metal(0).unwrap()
}

fn weight(dtype: GgmlDType, (n, k): (usize, usize), dev: &Device) -> QMatMul {
    let (bs, ts) = (dtype.block_size(), dtype.type_size());
    let blocks = n * k / bs;
    let mut state = 77u32;
    let mut raw = vec![0u8; blocks * ts];
    for b in raw.iter_mut() {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        *b = (state >> 24) as u8;
    }
    for i in 0..blocks {
        let d = f16::from_f32(0.01 + 0.006 * ((i as f32) * 0.37).sin());
        raw[i * ts..i * ts + 2].copy_from_slice(&d.to_le_bytes());
    }
    QMatMul::from_qtensor(
        QTensor::new(
            QStorage::from_data(Cow::Borrowed(&raw), dev, dtype).unwrap(),
            (n, k),
        )
        .unwrap(),
    )
    .unwrap()
}

/// Median of `reps` timings of `iters` launches, ms per launch.
fn time(dev: &Device, mut f: impl FnMut(), iters: usize, reps: usize) -> f64 {
    for _ in 0..3 {
        f();
    }
    dev.synchronize().unwrap();
    let mut ms = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t = Instant::now();
        for _ in 0..iters {
            f();
        }
        dev.synchronize().unwrap();
        ms.push(t.elapsed().as_secs_f64() * 1e3 / iters as f64);
    }
    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ms[reps / 2]
}

#[test]
#[ignore]
fn every_kind_at_every_capacity() {
    let dev = device();
    let (n, k) = (10240usize, 2560usize);
    let kinds = [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::IQ2XXS,
        GgmlDType::IQ2XS,
        GgmlDType::IQ2S,
        GgmlDType::IQ3XXS,
        GgmlDType::IQ3S,
        GgmlDType::IQ4NL,
        GgmlDType::IQ4XS,
        GgmlDType::Q2_0,
        GgmlDType::MXFP4,
        GgmlDType::FP8E4M3,
    ];
    println!("TIMING shape=[{n} x {k}] contract=fast iters=50 reps=7 median ms per launch");
    println!("| kind | m=1 | m | per-row (axis off) | axis _rN | per-row/row ms | axis/row ms | axis vs per-row |");
    for dtype in kinds {
        let q = weight(dtype, (n, k), &dev);
        let x1 = Tensor::randn(0f32, 1.0, (1, k), &dev).unwrap();
        lloom_tensor::kernels::set_qmv_rows(false);
        let m1 = time(&dev, || { q.forward(&x1).unwrap(); }, 50, 7);
        for m in [2usize, 3, 4, 5, 6, 8] {
            let xm = Tensor::randn(0f32, 1.0, (m, k), &dev).unwrap();
            lloom_tensor::kernels::set_qmv_rows(false);
            let per_row = time(&dev, || { q.forward(&xm).unwrap(); }, 50, 7);
            lloom_tensor::kernels::set_qmv_rows(true);
            let cap = lloom_tensor::kernels::mv_rows_capacity(dtype.into(), m, false);
            let axis = time(&dev, || { q.forward(&xm).unwrap(); }, 50, 7);
            println!(
                "| {dtype:?} | {m1:.4} | {m} | {per_row:.4} | _r{cap} {axis:.4} | {:.4} | {:.4} | {:+.1}% |",
                per_row / m as f64,
                axis / m as f64,
                (axis / per_row - 1.0) * 100.0
            );
        }
        lloom_tensor::kernels::set_qmv_rows(true);
    }
}
