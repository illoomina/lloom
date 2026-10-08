//! The row axis under `fast` with the kind table's measured ceilings
//! (`QuantKind::rows_max`, `rows_max_routed`; `docs/tuning.md` §13.5v): a
//! kind takes the axis exactly where its ceiling says, the capacity the one
//! the dispatch names, and nowhere else. Absolute references, not a shape:
//! Q2_K at 8 rows runs `_r8`, Q8_0 at 2 rows `_r2` and at 5 rows `_r2` in
//! groups, Q4_K and IQ3_XXS no rows kernel at any row count, and the routed
//! launch of every kind none.
//!
//! Its own binary: the contract and the ceilings are read once per process.
#![cfg(feature = "metal")]

#[path = "rows_identity/mod.rs"]
mod rows_identity;

use lloom_tensor::quantized::{GgmlDType, QStorage, QTensor};
use lloom_tensor::{Device, Module, Result, Tensor};
use std::borrow::Cow;

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(lloom_tensor::tuning::contract().is_fast());
    assert_eq!(
        lloom_tensor::tuning::tuning().qmv.rows_max,
        None,
        "the table's ceilings"
    );
    Device::new_metal(0).expect("a Metal device")
}

fn rows_kernels(dev: &Device, f: impl FnOnce() -> Result<()>) -> Result<Vec<(String, u64)>> {
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
        .into_iter()
        .filter(|(l, _)| l.starts_with("kernel_mul_mv_") && l.contains("_r"))
        .collect())
}

/// One test: the census is a per-device window.
#[test]
fn a_kind_takes_the_axis_where_its_ceiling_says() -> Result<()> {
    let dev = device();
    let k = 512usize;
    let cases: [(GgmlDType, usize, Option<&str>); 9] = [
        (GgmlDType::Q2K, 2, Some("kernel_mul_mv_q2_K_f32_r2")),
        (GgmlDType::Q2K, 3, Some("kernel_mul_mv_q2_K_f32_r4")),
        (GgmlDType::Q2K, 8, Some("kernel_mul_mv_q2_K_f32_r8")),
        (GgmlDType::Q8_0, 2, Some("kernel_mul_mv_q8_0_f32_r2")),
        (GgmlDType::Q8_0, 5, Some("kernel_mul_mv_q8_0_f32_r2")),
        (GgmlDType::Q4K, 2, None),
        (GgmlDType::Q4K, 8, None),
        (GgmlDType::IQ3XXS, 4, None),
        (GgmlDType::Q6K, 4, None),
    ];
    let mut failures = Vec::new();
    for (dtype, m, want) in cases {
        let w = rows_identity::weight(dtype, 40, &dev)?;
        let x = Tensor::randn(0f32, 1.0, (m, k), &dev)?;
        let ran = rows_kernels(&dev, || w.forward(&x).map(|_| ()))?;
        let expect: Vec<(String, u64)> = want.map(|n| vec![(n.to_string(), 1)]).unwrap_or_default();
        println!("CEILING {dtype:?} m={m} ran={ran:?}");
        if ran != expect {
            failures.push(format!(
                "{dtype:?} m={m}: ran {ran:?}, the table says {expect:?}"
            ));
        }
    }
    // The routed launch: no kind's routed ceiling admits the axis.
    for dtype in [GgmlDType::Q2K, GgmlDType::Q8_0, GgmlDType::Q4K] {
        let (e, n) = (4usize, 32usize);
        let raw = QTensor::quantize(
            &Tensor::randn(0f32, 0.05, (e, n, 256), &Device::Cpu)?,
            dtype,
        )?
        .data()?
        .into_owned();
        let q = QTensor::new(
            QStorage::from_data(Cow::Borrowed(&raw), &dev, dtype)?,
            (e, n, 256),
        )?;
        let ids = Tensor::from_vec(vec![1u32, 2, 1, 3], (2, 2), &dev)?;
        let x = Tensor::randn(0f32, 1.0, (2, 1, 256), &dev)?;
        let ran = rows_kernels(&dev, || q.indexed_moe_forward(&x, &ids).map(|_| ()))?;
        println!("CEILING routed {dtype:?} ran={ran:?}");
        if !ran.is_empty() {
            failures.push(format!(
                "routed {dtype:?}: ran {ran:?}, no routed ceiling admits the axis"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}
