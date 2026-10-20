//! One integration-test binary for the crate (lloom #939): every
//! `tests/*.rs` file is a module here, so the suite links the engine once
//! instead of once per file, and a new test file is one `mod` line. The
//! files keep their names and their `#![cfg(feature = ...)]` gates. Each
//! test still runs in its own process under `cargo nextest run` (the gate's
//! runner); under plain `cargo test` the tests share a process, and a test
//! that installs process-global state fails by name when another test did
//! (`lloom_tensor::isolation`).
//!
//! Its own binary, because a test binary carries at most one
//! `#[global_allocator]`: (none in this crate).
//!
//! `every_test_file_is_a_module` below reads this directory and this file,
//! so a file that is not a module here fails the suite instead of silently
//! not running.
mod accepted_frontier;
mod attention_purpose;
mod checkpoint_policy;
mod failure_next_request;
mod load_warnings;
mod machine_gate;
mod qwen4exp_served_auto;
mod recovery_frontier_served;
mod sampling_host_contract;
mod tiny_chat_e2e;
mod tiny_templates;
mod turn_reuse;

/// Every `tests/*.rs` beside this file is declared above (or is one of the
/// files that owns its binary), read from the directory at test time.
#[test]
fn every_test_file_is_a_module() {
    let own: &[&str] = &[];
    let dir = format!("{}/tests", lloom_manifest_dir());
    let me = include_str!("main.rs");
    let mut missing = Vec::new();
    let mut files = 0;
    for entry in std::fs::read_dir(dir).expect("tests/ is readable") {
        let name = entry
            .expect("an entry")
            .file_name()
            .into_string()
            .expect("utf-8");
        let Some(stem) = name.strip_suffix(".rs") else {
            continue;
        };
        if stem == "main" || own.contains(&stem) {
            continue;
        }
        files += 1;
        if !me.lines().any(|l| l == format!("mod {stem};")) {
            missing.push(stem.to_string());
        }
    }
    assert!(files > 0, "no test files beside main.rs");
    assert!(
        missing.is_empty(),
        "test files that are not modules of tests/main.rs (add `mod <name>;`): {missing:?}"
    );
}

/// The crate's manifest directory **at run time** — what cargo and nextest
/// export to the test process — falling back to the build-time path. A test
/// binary archived on one host and run on another (`cargo nextest archive` /
/// `--workspace-remap`) reads its fixtures where the archive was unpacked;
/// `env!("CARGO_MANIFEST_DIR")` would bake the building host's path in
/// (measured 2026-10-13: `every_test_file_is_a_module` panicked with NotFound
/// on silverberry from a Mac-built archive).
fn lloom_manifest_dir() -> String {
    std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| env!("CARGO_MANIFEST_DIR").to_string())
}

mod sampling_admission;
