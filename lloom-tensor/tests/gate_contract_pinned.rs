//! A pinned gate runs `deterministic` whatever the library default is, and
//! the pin reaches the tuning every site reads: `contract()` and
//! `tuning().contract.profile` agree with it. The refused half is
//! `gate_contract_unpinned`.
//!
//! Its own binary: the pin is process-wide.

use lloom_tensor::tuning::{self, Contract};

#[test]
fn a_pinned_gate_runs_deterministic() {
    if std::env::var_os("LLOOM_PROFILE").is_some() {
        eprintln!("LLOOM_PROFILE is set; it wins over the pin by design");
        return;
    }
    assert_eq!(
        tuning::pin_contract(Contract::Deterministic),
        Ok(Contract::Deterministic)
    );
    assert_eq!(
        tuning::gate_contract("this test"),
        Ok(Contract::Deterministic)
    );
    assert_eq!(tuning::contract(), Contract::Deterministic);
    assert_eq!(tuning::tuning().contract.profile, "deterministic");
    assert!(
        tuning::pin_contract(Contract::Fast).is_err(),
        "a second pin after the tuning was fixed must be refused, not ignored"
    );
}
