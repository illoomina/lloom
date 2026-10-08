//! Under the `deterministic` contract (pinned here) an `m`-row quantized
//! matmul-vec (2..=8 rows) is one launch of the one-row core with the rows
//! on the grid's second axis, and each row is bit-identical to a one-row
//! launch of that row: a sequence's logits do not depend on how many rows
//! share its batch. No kernel on the row axis (`_r<N>`) may run.
//!
//! Its own binary: the contract is read once per process.
#![cfg(feature = "metal")]

mod rows_identity;

use lloom_tensor::{Device, Result};

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "deterministic".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(!lloom_tensor::tuning::contract().is_fast());
    assert!(
        lloom_tensor::tuning::tuning().qmv.rows,
        "the axis is on by default; it is the contract that keeps it off here"
    );
    Device::new_metal(0).expect("a Metal device")
}

#[test]
fn a_row_is_the_same_bits_whatever_rows_share_its_launch() -> Result<()> {
    rows_identity::a_row_is_the_same_bits_whatever_rows_share_its_launch(&device(), false)
}
