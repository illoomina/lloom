//! The quantized-kernel parity matrix (`quant_parity/mod.rs`) under
//! `deterministic`: every instantiated (format × entry point × activation
//! dtype) except the row axis's kernels, which `quant_parity_matrix_rows`
//! runs under `fast` (the contract is read once per process).
//!
//! One test: the dispatch census is a per-device window, and two tests on
//! parallel threads would count each other's launches.
#![cfg(feature = "metal")]

use super::quant_parity;

use lloom_tensor::{Device, Result};
use quant_parity::{Entry, Matrix};

const ENTRIES: [Entry; 12] = [
    Entry::MulMv,
    Entry::MulMvId,
    Entry::MulMvId2,
    Entry::MulMm,
    Entry::DequantMm,
    Entry::MmId32,
    Entry::MmId16,
    Entry::GetRows,
    Entry::DequantRowsF16,
    Entry::KvSdpa,
    Entry::KvSdpaSel,
    Entry::KvDequant,
];

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "deterministic".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(
        !lloom_tensor::tuning::contract().is_fast(),
        "the row axis is quant_parity_matrix_rows' binary"
    );
    Device::new_metal(0).expect("a Metal device")
}

#[test]
fn every_instantiated_quant_decode_matches_the_cpu_reference() -> Result<()> {
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
