//! The deterministic twin of `qmatmul_rows_fast`: under the default
//! contract an `m`-row quantized matmul-vec (2..=8 rows) is one launch with
//! the rows on the grid's second axis, and each row is bit-identical to a
//! one-row launch of that row — a sequence's logits do not depend on how
//! many rows share its batch. The rows kernels must not run.
//!
//! Its own binary: the contract is read once per process, and this one
//! never installs `fast`.
#![cfg(feature = "metal")]

use lloom_tensor::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use lloom_tensor::{Device, Module, Result, Tensor};
use std::borrow::Cow;

const K: usize = 512;

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(!lloom_tensor::tuning::contract().is_fast());
    Device::new_metal(0).expect("a Metal device")
}

fn weight(dtype: GgmlDType, n: usize, dev: &Device) -> Result<QMatMul> {
    let raw: Vec<u8> = if matches!(dtype, GgmlDType::Q6K | GgmlDType::Q4K) {
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
    QMatMul::from_qtensor(QTensor::new(QStorage::from_data(Cow::Borrowed(&raw), dev, dtype)?, (n, K))?)
}

#[test]
fn a_row_is_the_same_bits_whatever_rows_share_its_launch() -> Result<()> {
    let dev = device();
    assert!(lloom_tensor::kernels::metal::profile::enabled());
    for dtype in [GgmlDType::Q6K, GgmlDType::Q4K, GgmlDType::IQ4XS, GgmlDType::IQ4NL, GgmlDType::IQ3S] {
        let w = weight(dtype, 40, &dev)?;
        for m in 2..=8usize {
            let data: Vec<f32> = (0..m * K).map(|i| ((i as f32) * 0.013).sin()).collect();
            let x = Tensor::from_vec(data, (m, K), &dev)?;
            dev.synchronize()?;
            let window = dev.as_metal_device()?.dispatch_census().open().expect("a window");
            let rows = w.forward(&x)?;
            dev.synchronize()?;
            let census = window.close();
            let rows_kernels: u64 = census
                .iter()
                .filter(|(l, _)| l.contains("_r2") || l.contains("_r4"))
                .map(|(_, n)| *n)
                .sum();
            assert_eq!(rows_kernels, 0, "{dtype:?} m={m}: a rows kernel ran under deterministic ({census:?})");
            for r in 0..m {
                let one = w.forward(&x.narrow(0, r, 1)?.contiguous()?)?;
                let a: Vec<u32> = rows.narrow(0, r, 1)?.flatten_all()?.to_vec1::<f32>()?.iter().map(|v| v.to_bits()).collect();
                let b: Vec<u32> = one.flatten_all()?.to_vec1::<f32>()?.iter().map(|v| v.to_bits()).collect();
                assert_eq!(a, b, "{dtype:?} m={m} row {r}: the batch moved a row's bits");
            }
        }
    }
    Ok(())
}
