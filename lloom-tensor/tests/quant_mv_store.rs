//! The quantized matvec with a block on its store (`MvStore`,
//! `Source::QuantizedStore`) against the pair it replaces, on the device:
//! every instantiation the store library carries is a cell — read from the
//! COMPILED library (`Library::function_names`), never from a list kept here
//! — and each cell runs through the dispatch production runs
//! (`QMatMul::forward_silu`, `QMatMul::forward_mul_into`), its own kernel
//! shown in the dispatch census.
//!
//! The reference is the composed pair on the same device over the same
//! bytes: the plain matvec (the same core, the `f32` store), then the
//! activation (`silu`) or the product with the gate (`silu(Wg·x) · (Wu·x)`,
//! the SwiGLU `silu_mul` computes). The block changes only what the store
//! does with the core's sum, in the pair's own arithmetic, so the cells are
//! held to the pair BIT FOR BIT, and to themselves run to run.
#![cfg(feature = "metal")]

mod quant_parity;

use lloom_tensor::kernels::source::Source;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Device, Module, Result, Tensor};
use std::collections::BTreeSet;
use std::sync::Arc;

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "deterministic".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    Device::new_metal(0).expect("a Metal device")
}

fn bits(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

/// The labels `f` launched.
fn launched<T>(dev: &Device, f: impl FnOnce() -> Result<T>) -> Result<(T, BTreeSet<String>)> {
    dev.synchronize()?;
    let window = dev
        .as_metal_device()?
        .dispatch_census()
        .open()
        .map_err(|e| lloom_tensor::Error::Msg(format!("census: {e:?}")))?;
    let r = f()?;
    dev.synchronize()?;
    Ok((r, window.close().into_iter().map(|(l, _)| l).collect()))
}

#[test]
fn every_store_block_is_its_pair_bit_for_bit() -> Result<()> {
    let dev = device();
    let md = dev.as_metal_device()?;
    let names: BTreeSet<String> = md
        .kernels()
        .load_library(md.device(), Source::QuantizedStore)
        .map_err(|e| lloom_tensor::Error::Msg(format!("{e}")))?
        .function_names()
        .into_iter()
        .filter(|n| {
            n.starts_with("kernel_mul_mv_") && (n.ends_with("_f32_silu") || n.ends_with("_f32_mul"))
        })
        .collect();
    // Every kind with a genericized core, two blocks each.
    let expect: BTreeSet<String> = lloom_tensor::kernels::mv_store_instantiations()
        .into_iter()
        .map(|(stem, _, suffix)| format!("kernel_mul_mv_{stem}_f32{suffix}"))
        .collect();
    assert_eq!(
        names, expect,
        "the library carries the instantiations the kinds table names"
    );
    assert_eq!(names.len(), 2 * 18);
    let (n, k) = (96usize, 512usize);
    let mut failures = Vec::new();
    let mut ran = 0;
    for d in quant_parity::ALL {
        let stem = quant_parity::stem(d);
        let silu = format!("kernel_mul_mv_{stem}_f32_silu");
        if !names.contains(&silu) {
            continue;
        }
        let mul = format!("kernel_mul_mv_{stem}_f32_mul");
        let wg = quant_parity::weight(&dev, d, &[n, k], 11, 0.5)?;
        let wu = quant_parity::weight(&dev, d, &[n, k], 23, 0.5)?;
        let (x, _) = quant_parity::activations(&dev, &[1, k], quant_parity::Act::F32, 7)?;
        let g = QMatMul::QTensor(Arc::new(wg.q));
        let u = QMatMul::QTensor(Arc::new(wu.q));
        // The pair: the plain matvec, then the SiLU; then the product.
        let gate_ref = g.forward(&x)?.silu()?;
        let swiglu_ref = (&gate_ref * u.forward(&x)?)?;
        // The SiLU on the store.
        let (gate, labels) = launched(&dev, || {
            g.forward_silu(&x)
                .expect("a quantized kind with a core is served")
        })?;
        if !labels.contains(&silu) {
            failures.push(format!("{silu}: not launched ({labels:?})"));
        }
        if bits(&gate)? != bits(&gate_ref)? {
            failures.push(format!("{silu}: differs from silu(W·x)"));
        }
        // The product multiplied into the gate's buffer.
        let ((), labels) = launched(&dev, || {
            u.forward_mul_into(&x, &gate)
                .expect("a quantized kind with a core is served")
        })?;
        if !labels.contains(&mul) {
            failures.push(format!("{mul}: not launched ({labels:?})"));
        }
        if bits(&gate)? != bits(&swiglu_ref)? {
            let (a, b) = (
                gate.flatten_all()?.to_vec1::<f32>()?,
                swiglu_ref.flatten_all()?.to_vec1::<f32>()?,
            );
            let differ = a
                .iter()
                .zip(&b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            failures.push(format!(
                "{mul}: {differ} of {n} differ from silu(Wg·x)·(Wu·x)"
            ));
        }
        // Run to run.
        let again = g.forward_silu(&x).unwrap()?;
        u.forward_mul_into(&x, &again).unwrap()?;
        if bits(&again)? != bits(&gate)? {
            failures.push(format!("{stem}: not bit-stable run to run"));
        }
        ran += 1;
    }
    eprintln!("{ran} kinds, {} cells", 2 * ran);
    assert_eq!(ran, 18, "every kind the library carries ran");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

/// What the blocks do not serve: a dense weight, a multi-row activation.
#[test]
fn the_store_blocks_decline_what_they_do_not_serve() -> Result<()> {
    let dev = device();
    let w = quant_parity::weight(
        &dev,
        lloom_tensor::quantized::GgmlDType::F16,
        &[64, 256],
        3,
        0.5,
    )?;
    let dense = QMatMul::QTensor(Arc::new(w.q));
    let (x, _) = quant_parity::activations(&dev, &[1, 256], quant_parity::Act::F32, 5)?;
    assert!(
        dense.forward_silu(&x).is_none(),
        "a dense kind has no genericized core"
    );
    let w = quant_parity::weight(
        &dev,
        lloom_tensor::quantized::GgmlDType::Q4K,
        &[64, 256],
        3,
        0.5,
    )?;
    let q = QMatMul::QTensor(Arc::new(w.q));
    let (x2, _) = quant_parity::activations(&dev, &[2, 256], quant_parity::Act::F32, 5)?;
    assert!(
        q.forward_silu(&x2).is_none(),
        "two rows are not the one-row store"
    );
    let gate = Tensor::zeros(63, lloom_tensor::DType::F32, &dev)?;
    assert!(
        q.forward_mul_into(&x, &gate).is_none(),
        "a gate of another extent"
    );
    Ok(())
}
