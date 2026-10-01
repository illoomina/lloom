//! A gate that never stated its contract is refused: with no pin and no
//! `LLOOM_PROFILE`, `gate_contract` errs and names the default it would have
//! run (`fast`), so a harness that lost its pin stops instead of decoding a
//! digest under the wrong contract (`architecture.md` §2.3.5b). The pinned
//! half is `gate_contract_pinned`.
//!
//! Its own binary: the pin is process-wide.

#[test]
fn a_gate_with_no_stated_contract_is_refused() {
    if std::env::var_os("LLOOM_PROFILE").is_some() {
        eprintln!("LLOOM_PROFILE is set; the unpinned case is not observable here");
        return;
    }
    let err = lloom_tensor::tuning::gate_contract("this test").unwrap_err();
    assert!(err.contains("inherited the default (fast)"), "{err}");
    assert!(
        lloom_tensor::tuning::contract().is_fast(),
        "the library default is fast"
    );
}
