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
use lloom_tensor::quantized::{GgmlDType, MoeArm, QMatMul, QStorage, QTensor};
use lloom_tensor::{Device, Module, Tensor};
use std::borrow::Cow;
use std::time::Instant;

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        // Every capacity of every kind: the measurement the ceilings come from.
        t.qmv.rows_max = Some(8);
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
    // `QMM_SHAPE=n,k` times another stem: the 27B's `ffn_up` is 17408,5120,
    // its `ffn_down` 5120,17408.
    let (n, k) = std::env::var("QMM_SHAPE")
        .ok()
        .and_then(|v| {
            let mut it = v.split(',').map(|s| s.trim().parse::<usize>().ok());
            Some((it.next()??, it.next()??))
        })
        .unwrap_or((10240usize, 2560usize));
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
    // `QMM_KINDS=Q4K,Q6K` times a subset.
    let only: Vec<String> = std::env::var("QMM_KINDS")
        .map(|v| v.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();
    // `QMM_TABLE=1`: the kind table's ceilings instead of 8 for every kind
    // (the policy as shipped, so the capacity column is what a run takes).
    if std::env::var("QMM_TABLE").is_ok() {
        lloom_tensor::kernels::set_qmv_rows_max(None);
    }
    println!(
        "TIMING shape=[{n} x {k}] contract=fast table={} iters=50 reps=7 median ms per launch",
        std::env::var("QMM_TABLE").is_ok()
    );
    println!("| kind | m=1 | m | per-row (axis off) | axis _rN | per-row/row ms | axis/row ms | axis vs per-row |");
    for dtype in kinds {
        if !only.is_empty() && !only.iter().any(|s| *s == format!("{dtype:?}")) {
            continue;
        }
        let q = weight(dtype, (n, k), &dev);
        let x1 = Tensor::randn(0f32, 1.0, (1, k), &dev).unwrap();
        lloom_tensor::kernels::set_qmv_rows(false);
        let m1 = time(
            &dev,
            || {
                q.forward(&x1).unwrap();
            },
            50,
            7,
        );
        for m in [2usize, 3, 4, 5, 6, 8] {
            let xm = Tensor::randn(0f32, 1.0, (m, k), &dev).unwrap();
            lloom_tensor::kernels::set_qmv_rows(false);
            let per_row = time(
                &dev,
                || {
                    q.forward(&xm).unwrap();
                },
                50,
                7,
            );
            lloom_tensor::kernels::set_qmv_rows(true);
            let cap = lloom_tensor::kernels::mv_rows_capacity(dtype.into(), m, false);
            let axis = time(
                &dev,
                || {
                    q.forward(&xm).unwrap();
                },
                50,
                7,
            );
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

/// The routed matmul-vec on the row axis, per kind: `batch` rows at top-8
/// over 64 experts of `[512 × 2048]` (Qwen3.6-35B-A3B's gate/up expert),
/// the ids from a fixed LCG (rows share a slot where the draw repeats, as a
/// window's rows do when they route alike — and `QMM_SHARE=1` makes every
/// row name the first row's slots, the window's best case), against the
/// per-plane launch with the axis off. Ignored; under `lloom-arb measure`.
#[test]
#[ignore]
fn every_routed_kind_at_every_capacity() {
    let dev = device();
    let (e, n, k, topk) = (64usize, 512usize, 2048usize, 8usize);
    let share = std::env::var("QMM_SHARE").is_ok();
    let kinds = [
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
        GgmlDType::Q8_0,
        GgmlDType::IQ2XXS,
        GgmlDType::IQ2XS,
        GgmlDType::IQ2S,
        GgmlDType::IQ3XXS,
        GgmlDType::IQ3S,
        GgmlDType::IQ4XS,
        GgmlDType::Q2_0,
    ];
    println!("TIMING_ID experts={e} [{n} x {k}] topk={topk} share={share} contract=fast iters=50 reps=7 median ms per launch");
    println!(
        "| kind | batch | slots named | per-plane (axis off) | axis _rN | axis vs per-plane |"
    );
    for dtype in kinds {
        let (bs, ts) = (dtype.block_size(), dtype.type_size());
        let blocks = e * n * k / bs;
        let mut state = 99u32;
        let mut raw = vec![0u8; blocks * ts];
        for b in raw.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (state >> 24) as u8;
        }
        for i in 0..blocks {
            let d = f16::from_f32(0.01 + 0.006 * ((i as f32) * 0.37).sin());
            raw[i * ts..i * ts + 2].copy_from_slice(&d.to_le_bytes());
        }
        let w = QTensor::new(
            QStorage::from_data(Cow::Borrowed(&raw), &dev, dtype).unwrap(),
            (e, n, k),
        )
        .unwrap();
        for batch in [2usize, 3, 4, 6, 8] {
            let mut s = 7u32;
            let mut ids = Vec::with_capacity(batch * topk);
            for r in 0..batch {
                for t in 0..topk {
                    if share && r > 0 {
                        ids.push(ids[t]);
                        continue;
                    }
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    ids.push((s >> 16) % e as u32);
                }
            }
            let mut named = ids.clone();
            named.sort_unstable();
            named.dedup();
            let ids_t = Tensor::from_vec(ids, (batch, topk), &dev).unwrap();
            let x = Tensor::randn(0f32, 1.0, (batch, 1, k), &dev).unwrap();
            lloom_tensor::kernels::set_qmv_rows(false);
            let per_plane = time(
                &dev,
                || {
                    w.indexed_moe_forward_as(&x, &ids_t, MoeArm::MatVec, lloom_tensor::DType::F32)
                        .unwrap();
                },
                50,
                7,
            );
            lloom_tensor::kernels::set_qmv_rows(true);
            let cap = lloom_tensor::kernels::mv_rows_capacity(dtype.into(), batch, true);
            let axis = time(
                &dev,
                || {
                    w.indexed_moe_forward_as(&x, &ids_t, MoeArm::MatVec, lloom_tensor::DType::F32)
                        .unwrap();
                },
                50,
                7,
            );
            println!(
                "| {dtype:?} | {batch} | {} of {} | {per_plane:.4} | _r{cap} {axis:.4} | {:+.1}% |",
                named.len(),
                batch * topk,
                (axis / per_plane - 1.0) * 100.0
            );
        }
    }
}
