//! The body `qmatmul_rows_default` and `qmatmul_rows_under_fast` share: an
//! `m`-row quantized matmul-vec (2..=8 rows) against a one-row launch of
//! each of its rows, with the census read for the row axis
//! (`kernel_mul_mv_*_r<N>`). Under `deterministic` no rows kernel runs and
//! every row is the one-row launch's bits; under `fast` the rows kernel at
//! the capacity the dispatch names runs, and each row is the one-row core's
//! sum in the one-row core's order — reported bit for bit, bounded at 1e-5
//! relative (the compiler's contraction is the one thing the row loop may
//! move).

use lloom_tensor::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use lloom_tensor::{Device, Module, Result, Tensor};
use std::borrow::Cow;

const K: usize = 512;

/// The five kinds the Flash-Next checkpoint decodes through plus the
/// 27B's Q4_K_M stems' second kind: every core family with a row loop of its
/// own shape (the K-quants, the codebook cores, the byte-code core).
pub const KINDS: [GgmlDType; 8] = [
    GgmlDType::Q6K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q8_0,
    GgmlDType::Q4_0,
    GgmlDType::IQ4XS,
    GgmlDType::IQ4NL,
    GgmlDType::IQ3S,
];

pub fn weight(dtype: GgmlDType, n: usize, dev: &Device) -> Result<QMatMul> {
    let raw: Vec<u8> = if matches!(
        dtype,
        GgmlDType::Q6K | GgmlDType::Q4K | GgmlDType::Q5K | GgmlDType::Q8_0 | GgmlDType::Q4_0
    ) {
        let w: Vec<f32> = (0..n * K).map(|i| (i as f32 * 0.61).sin() * 0.05).collect();
        let w = Tensor::from_vec(w, (n, K), &Device::Cpu)?;
        QTensor::quantize(&w, dtype)?.data()?.into_owned()
    } else {
        let (bs, ts) = (dtype.block_size(), dtype.type_size());
        let mut state = 12345u32;
        let mut raw = vec![0u8; n * K / bs * ts];
        for b in raw.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *b = (state >> 24) as u8;
        }
        for i in 0..n * K / bs {
            let d = half::f16::from_f32(0.02 + 0.015 * ((i as f32) * 0.37).sin());
            raw[i * ts..i * ts + 2].copy_from_slice(&d.to_le_bytes());
        }
        raw
    };
    QMatMul::from_qtensor(QTensor::new(
        QStorage::from_data(Cow::Borrowed(&raw), dev, dtype)?,
        (n, K),
    )?)
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

/// Every row of an `m`-row launch (2..=8) against a one-row launch of that
/// row. `expect_rows`: the row axis is expected in the census (`fast`) —
/// then the rows are the one-row core's to 1e-5 and the bit agreement is
/// printed; otherwise no rows kernel may run and the bits must be equal.
pub fn a_row_is_the_same_bits_whatever_rows_share_its_launch(
    dev: &Device,
    expect_rows: bool,
) -> Result<()> {
    let dev = dev.clone();
    assert!(lloom_tensor::kernels::metal::profile::enabled());
    let mut bit_equal = 0usize;
    let mut rows_checked = 0usize;
    for dtype in KINDS {
        let w = weight(dtype, 40, &dev)?;
        for m in 2..=8usize {
            let data: Vec<f32> = (0..m * K).map(|i| ((i as f32) * 0.013).sin()).collect();
            let x = Tensor::from_vec(data, (m, K), &dev)?;
            dev.synchronize()?;
            let window = dev
                .as_metal_device()?
                .dispatch_census()
                .open()
                .expect("a window");
            let rows = w.forward(&x)?;
            dev.synchronize()?;
            let census = window.close();
            let cap = lloom_tensor::kernels::mv_rows_capacity(dtype.into(), m, false);
            let rows_kernels: u64 = census
                .iter()
                .filter(|(l, _)| l.contains("kernel_mul_mv_") && l.contains("_r"))
                .map(|(_, n)| *n)
                .sum();
            if expect_rows {
                assert!(cap > 1, "{dtype:?} m={m}: the dispatch names no capacity under fast");
                let at_cap = census
                    .iter()
                    .filter(|(l, _)| l.ends_with(&format!("_r{cap}")))
                    .map(|(_, n)| *n)
                    .sum::<u64>();
                assert_eq!(
                    at_cap, 1,
                    "{dtype:?} m={m}: the rows kernel at capacity {cap} did not run once ({census:?})"
                );
            } else {
                assert_eq!(cap, 1, "{dtype:?} m={m}: a capacity under deterministic");
                assert_eq!(
                    rows_kernels, 0,
                    "{dtype:?} m={m}: a rows kernel ran under deterministic ({census:?})"
                );
            }
            for r in 0..m {
                let one = w.forward(&x.narrow(0, r, 1)?.contiguous()?)?;
                let a = bits(&rows.narrow(0, r, 1)?)?;
                let b = bits(&one)?;
                rows_checked += 1;
                if a == b {
                    bit_equal += 1;
                } else if expect_rows {
                    let av: Vec<f32> = a.iter().map(|v| f32::from_bits(*v)).collect();
                    let bv: Vec<f32> = b.iter().map(|v| f32::from_bits(*v)).collect();
                    let scale = bv.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
                    let worst = av
                        .iter()
                        .zip(&bv)
                        .fold(0f32, |m, (p, q)| m.max((p - q).abs() / scale));
                    assert!(
                        worst < 1e-5,
                        "{dtype:?} m={m} row {r}: {worst:e} off the one-row core"
                    );
                } else {
                    panic!("{dtype:?} m={m} row {r}: the batch moved a row's bits");
                }
            }
        }
    }
    println!(
        "ROWS_IDENTITY expect_rows={expect_rows} rows_checked={rows_checked} bit_equal={bit_equal}"
    );
    Ok(())
}
