//! Whether a row of a dense f16 `x · Wᵀ` keeps its bits when the number of
//! rows beside it changes — the property a batched decode's row needs for its
//! logits not to depend on who else is in the batch (§2.3.3 #7) — against the
//! kernel the live `m` selects (`mlx_gemm::select_tile_config`: the GEMV at
//! `m = 1`, the 16-row tile below `skinny_max`, a 64-row tile from it). The
//! dispatch census names the kernel each `m` ran, so the comparison is between
//! kernels shown to differ, not assumed to.
//!
//! At LFM2's decode shapes (hidden 2048 → 8192 and back), in the two layouts a
//! batched step takes (`[m, k]` and the rows-as-batch `[m, 1, k]`): row 0 of
//! every `m` from 2 to 128 is the `m = 2` row bit for bit, on the 16-row and
//! the 64-row tiles alike, and the GEMV's row at `m = 1` is not. So the tile
//! the batch selects costs no batch invariance; the `m = 1` route is the one
//! seam (`architecture.md` §2.3.5b).
//!
//! The fast contract retains the batch-dependent GEMM selection characterized
//! here; deterministic row identity is covered by `f16_rows_identity`.
//! Run with process isolation: the census switch is read once per process.
#![cfg(feature = "metal")]

use lloom_tensor::{DType, Device, Result, Tensor};

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(lloom_tensor::kernels::metal::profile::enabled());
    Device::new_metal(0).expect("a Metal device")
}

fn row0_bits(t: &Tensor) -> Result<Vec<u32>> {
    let t = t.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let n = t.dim(t.rank() - 1)?;
    Ok(t.reshape(((), n))?
        .get(0)?
        .to_vec1::<f32>()?
        .iter()
        .map(|v| v.to_bits())
        .collect())
}

/// The matmul kernels `f` dispatched on `dev`, by label.
fn kernels(dev: &Device, f: impl FnOnce() -> Result<Tensor>) -> Result<(Tensor, Vec<String>)> {
    dev.synchronize()?;
    let window = dev
        .as_metal_device()?
        .dispatch_census()
        .open()
        .expect("a census window on this test's device");
    let y = f()?;
    dev.synchronize()?;
    let mut labels: Vec<String> = window
        .close()
        .into_iter()
        .map(|(l, _)| l)
        .filter(|l| l.starts_with("gemm") || l.starts_with("gemv"))
        .collect();
    labels.sort();
    Ok((y, labels))
}

#[test]
fn a_rows_bits_by_the_kernel_its_batch_selects() -> Result<()> {
    let dev = device();
    let ms = [1usize, 2, 3, 8, 16, 31, 32, 33, 40, 64, 128];
    // The boundary the shipped selection draws (`skinny_max`'s default,
    // 33): written here rather than read from the knob, so a moved boundary
    // fails this test instead of moving its expectation with it.
    const SKINNY: usize = 33;
    let mut failures = Vec::new();
    for (n, k) in [(8192usize, 2048usize), (2048, 8192)] {
        let w = Tensor::randn(0f32, 0.02, (n, k), &dev)?.to_dtype(DType::F16)?;
        let x = Tensor::randn(0f32, 1.0, (128, k), &dev)?.to_dtype(DType::F16)?;
        for rank3 in [false, true] {
            let mut by_m = Vec::new();
            for &m in &ms {
                let xm = x.narrow(0, 0, m)?;
                let (y, labels) = kernels(&dev, || {
                    if rank3 {
                        xm.reshape((m, 1, k))?.broadcast_matmul(&w.t()?)
                    } else {
                        xm.matmul(&w.t()?)
                    }
                })?;
                by_m.push((m, row0_bits(&y)?, labels));
            }
            let reference = by_m.iter().find(|(m, _, _)| *m == 2).unwrap().1.clone();
            for (m, bits, labels) in &by_m {
                let same = *bits == reference;
                println!(
                    "ROWINV n={n} k={k} rank3={rank3} m={m} row0_vs_m2={} kernels={labels:?}",
                    if same { "=" } else { "≠" }
                );
                let gemv = labels.iter().any(|l| l.starts_with("gemv"));
                let tile16 = labels.iter().any(|l| l.contains("_16_32_16_"));
                let tile64 = labels.iter().any(|l| l.contains("_64_"));
                let ok = match *m {
                    1 => gemv && !same,
                    m if m < SKINNY => tile16 && same,
                    _ => tile64 && same,
                };
                if !ok {
                    failures.push(format!(
                        "n={n} k={k} rank3={rank3} m={m}: same={same} kernels={labels:?}"
                    ));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "expected the GEMV to differ at m = 1 and every tile to keep row 0's bits:\n{}",
        failures.join("\n")
    );
    Ok(())
}
