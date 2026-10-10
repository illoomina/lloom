//! The quantized-kernel parity matrix (`quant_parity/mod.rs`) on the row
//! axis: every kind's plain matmul-vec and routed single and pair at every
//! capacity (`kernel_mul_mv{,_id,_id2}_<stem>_<triple>_r{2,4,8}`), under
//! the `fast` contract that takes them — read once per process, so they
//! have this binary.
#![cfg(feature = "metal")]

#[path = "quant_parity/mod.rs"]
mod quant_parity;

use lloom_tensor::{Device, Result};
use quant_parity::{Entry, Matrix};

const ENTRIES: [Entry; 3] = [Entry::Rows, Entry::RowsId, Entry::RowsId2];

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        // Every kind at every capacity, whatever its measured ceiling.
        t.qmv.rows_max = Some(8);
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(
        lloom_tensor::tuning::contract().is_fast(),
        "the contract was read before this binary installed it"
    );
    assert!(lloom_tensor::tuning::tuning().qmv.rows);
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
