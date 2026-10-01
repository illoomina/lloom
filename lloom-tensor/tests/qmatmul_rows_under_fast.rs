//! The `fast` contract does not take the rows kernels: measured slower than
//! the per-row launches on the real checkpoint (`architecture.md` §2.3.5b),
//! so under `fast` an `m`-row launch is still the per-row one, each row's
//! bits a one-row launch's. Adding the rows arm to `fast` is a decision with
//! a measurement behind it, and this test is where it shows.
//!
//! Its own binary: the contract is read once per process.
#![cfg(feature = "metal")]

mod rows_identity;

use lloom_tensor::{Device, Result};

fn device() -> Device {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let mut t = lloom_tensor::tuning::Tuning::default();
        t.contract.profile = "fast".into();
        t.instrument.profile = true;
        let _ = lloom_tensor::tuning::install(t);
    });
    assert!(lloom_tensor::tuning::contract().is_fast());
    Device::new_metal(0).expect("a Metal device")
}

#[test]
fn a_row_is_the_same_bits_whatever_rows_share_its_launch() -> Result<()> {
    rows_identity::a_row_is_the_same_bits_whatever_rows_share_its_launch(&device())
}
