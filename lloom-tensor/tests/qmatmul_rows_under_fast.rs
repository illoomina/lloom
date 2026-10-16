//! The `fast` contract takes the row axis: an `m`-row quantized matmul-vec
//! (2..=8 rows) runs the core at the capacity the dispatch names (`_r2`,
//! `_r4`, `_r8`), every row's sum the one-row core's in the one-row core's
//! order (`docs/tuning.md` §13.5w). Each row is held to 1e-5 of the one-row
//! launch and the bit agreement is printed, not asserted: the row loop may
//! move the compiler's contraction and nothing else.
//!
//! Its own binary: the contract is read once per process.
#![cfg(feature = "metal")]

use super::rows_identity;

use lloom_tensor::{Device, Result};

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
    assert!(lloom_tensor::tuning::contract().is_fast());
    Device::new_metal(0).expect("a Metal device")
}

/// One test: the axis is a process-wide switch, and the census a per-device
/// window.
#[test]
fn a_row_is_the_one_row_cores_sum_at_every_capacity_and_the_per_row_launch_with_the_axis_off(
) -> Result<()> {
    let dev = device();
    rows_identity::a_row_is_the_same_bits_whatever_rows_share_its_launch(&dev, true)?;
    lloom_tensor::kernels::set_qmv_rows(false);
    let off = rows_identity::a_row_is_the_same_bits_whatever_rows_share_its_launch(&dev, false);
    lloom_tensor::kernels::set_qmv_rows(true);
    off
}
