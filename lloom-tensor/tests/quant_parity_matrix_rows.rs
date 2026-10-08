//! The quantized-kernel parity matrix (`quant_parity/mod.rs`) for the
//! small-batch rows kernels (`kernel_mul_mv_*_f32_r{2,4}`): their switch
//! (`tuning.qmv.rows`) is read once per process, so they have this binary.
#![cfg(feature = "metal")]

#[path = "quant_parity/mod.rs"]
mod quant_parity;

use lloom_tensor::{Device, Result};
use quant_parity::{Entry, Matrix};

const ENTRIES: [Entry; 1] = [Entry::Rows];

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "deterministic".into();
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

#[test]
fn every_rows_kernel_matches_the_cpu_reference() -> Result<()> {
    let mut m = Matrix::new(device())?;
    m.run(&ENTRIES)?;
    m.check_coverage(&ENTRIES);
    let fails = m.report(&ENTRIES);
    assert!(
        fails.is_empty(),
        "{} failures:\n{}",
        fails.len(),
        fails.join("\n")
    );
    Ok(())
}
