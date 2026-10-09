//! The machine gate end to end (`memory.md` §9.10): every family's tiny
//! checkpoint through the shipped `lloom chat`, with the machine's available
//! memory and the registry injected (`memory.machine.available_gb`,
//! `memory.machine.registry`), so no test depends on — or exhausts — the
//! memory of the machine it runs on.
//!
//! Per family and device:
//! - with room, the load prints its predicted peak and the machine's
//!   figures, reserves, decodes, and leaves the registry empty on exit;
//! - with the injected available memory under the reserve, it is refused
//!   with the arithmetic and decodes nothing;
//! - with another process's reservation standing in the registry (held by
//!   this test, not yet allocated), the same load is refused naming it —
//!   the case two processes both seeing the memory free would race on —
//!   and admitted once that reservation is released.
#![cfg(feature = "serve")]

use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::models::qwen3_5::testing::{Qwen3_5MoeTiny, Qwen3_5Tiny};
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, Weights};
use lloom_tensor::machine::Registry;
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lloom-machine-gate-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write<F: TinyFamily>(dir: &Path) -> PathBuf {
    let mut cfg = F::template();
    cfg.set_axis("vocab_size", F::tokenizer().vocab_size());
    cfg.derive();
    TinyCheckpoint::<F>::new(cfg, Weights::default())
        .write(dir)
        .expect("the tiny checkpoint writes")
}

/// One `lloom chat --raw` turn against `registry` with `available_gb`
/// injected and a 4 GB reserve: (exit ok, stderr).
fn chat(model: &Path, extra: &[&str], registry: &Path, available_gb: f64) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(model)
        .args(["--raw", "-p", "Hi", "-n", "4"])
        .args(extra)
        .arg("--set")
        .arg(format!("memory.machine.registry={}", registry.display()))
        .arg("--set")
        .arg(format!("memory.machine.available_gb={available_gb}"))
        .args(["--set", "memory.machine.reserve_gb=4"])
        .output()
        .expect("lloom runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

/// The three arms on one family and device.
fn gate_arms(family: &str, model: &Path, extra: &[&str]) {
    let registry = scratch(&format!("{family}-registry"));
    let what = format!("{family} {extra:?}");

    // Room: admitted, reported, released at exit.
    let (ok, err) = chat(model, extra, &registry, 100.0);
    assert!(ok, "{what}: admitted with room, but:\n{err}");
    let peak = err
        .lines()
        .find(|l| l.starts_with("[machine: peak "))
        .unwrap_or_else(|| panic!("{what}: no peak line:\n{err}"));
    assert!(peak.contains("weights"), "{what}: {peak}");
    let verdict = err
        .lines()
        .find(|l| l.starts_with("[machine: need="))
        .unwrap_or_else(|| panic!("{what}: no verdict line:\n{err}"));
    for want in ["available=100.00 GB (injected)", "reserve=4.00 GB", "fits"] {
        assert!(verdict.contains(want), "{what}: {want:?} not in {verdict}");
    }
    assert!(err.contains("[turn 1:"), "{what}: no turn:\n{err}");
    assert!(
        Registry::open(&registry)
            .unwrap()
            .entries()
            .unwrap()
            .is_empty(),
        "{what}: the reservation outlived the process"
    );

    // Under the reserve: refused with the arithmetic, nothing decoded.
    let (ok, err) = chat(model, extra, &registry, 3.0);
    assert!(
        !ok,
        "{what}: admitted with 3 GB available under a 4 GB reserve:\n{err}"
    );
    for want in [
        "the machine cannot hold this now",
        "available  3.00 GB (injected",
        "reserve    4.00 GB",
        "Nothing was allocated",
        "--force",
    ] {
        assert!(
            err.contains(want),
            "{what}: {want:?} not in the refusal:\n{err}"
        );
    }
    assert!(
        !err.contains("[turn 1:"),
        "{what}: decoded after a refusal:\n{err}"
    );

    // Another process's reservation, not yet allocated: 100 GB available,
    // 4 kept, 99.9 reserved by the stand-in (this test process, which holds
    // far less than the 3.9 GB that would leave room) — no room, whatever
    // the tiny peak. Refused naming the stand-in; admitted once it drops.
    let reg = Registry::open(&registry).unwrap();
    let stand_in = reg
        .reserve_unchecked("stand-in for a first load", 99_900_000_000)
        .unwrap();
    let (ok, err) = chat(model, extra, &registry, 100.0);
    assert!(!ok, "{what}: admitted beside a 99.9 GB reservation:\n{err}");
    let pid = format!("pid {}", std::process::id());
    for want in [
        "stand-in for a first load",
        pid.as_str(),
        "reserved 99.90 GB",
    ] {
        assert!(
            err.contains(want),
            "{what}: {want:?} not in the refusal:\n{err}"
        );
    }
    assert!(
        !err.contains("[turn 1:"),
        "{what}: decoded after a refusal:\n{err}"
    );
    drop(stand_in);
    let (ok, err) = chat(model, extra, &registry, 100.0);
    assert!(
        ok,
        "{what}: still refused after the stand-in released:\n{err}"
    );
    std::fs::remove_dir_all(&registry).ok();
}

#[test]
fn lfm2_tiny_is_admitted_on_the_machine_and_refused_without_room() {
    let dir = scratch("lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    gate_arms("lfm2", &model, &["--cpu"]);
    #[cfg(feature = "metal")]
    gate_arms("lfm2", &model, &[]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn qwen3_5_tiny_is_admitted_on_the_machine_and_refused_without_room() {
    for (tag, model) in [
        ("qwen3_5", write::<Qwen3_5Tiny> as fn(&Path) -> PathBuf),
        ("qwen3_5_moe", write::<Qwen3_5MoeTiny>),
    ] {
        let dir = scratch(tag);
        let model = model(&dir);
        let tok = dir.join("tokenizer.json");
        let tok = tok.to_str().unwrap();
        gate_arms(tag, &model, &["--tokenizer", tok, "--cpu"]);
        #[cfg(feature = "metal")]
        gate_arms(tag, &model, &["--tokenizer", tok]);
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn qwen4exp_tiny_is_admitted_on_the_machine_and_refused_without_room() {
    let dir = scratch("qwen4exp");
    let model = write::<Qwen4ExpTiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let tok = tok.to_str().unwrap();
    gate_arms(
        "qwen4exp",
        &model,
        &["--tokenizer", tok, "--experts", "resident", "--cpu"],
    );
    // `deterministic`: the tiny's f32 expert weights have no arm under
    // `fast`'s f16 routed activations on Metal, which is not this gate's.
    #[cfg(feature = "metal")]
    for experts in ["resident", "stream"] {
        gate_arms(
            "qwen4exp",
            &model,
            &[
                "--tokenizer",
                tok,
                "--experts",
                experts,
                "--profile",
                "deterministic",
            ],
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// `--force` loads what the machine refuses, and still records the
/// reservation for the other processes to see.
#[test]
fn force_loads_what_the_machine_refuses_and_says_so() {
    let dir = scratch("force");
    let model = write::<Lfm2Tiny>(&dir);
    let registry = scratch("force-registry");
    let (ok, err) = chat(&model, &["--cpu", "--force"], &registry, 3.0);
    assert!(ok, "--force did not load:\n{err}");
    assert!(
        err.contains("FORCED"),
        "the line does not say it was forced:\n{err}"
    );
    assert!(err.contains("[turn 1:"), "{err}");
    std::fs::remove_dir_all(&registry).ok();
    std::fs::remove_dir_all(&dir).ok();
}

/// `serve` is admitted at startup through the same load: refused with the
/// arithmetic, it exits before binding the socket.
#[test]
fn serve_refuses_at_startup() {
    let dir = scratch("serve");
    let model = write::<Lfm2Tiny>(&dir);
    let registry = scratch("serve-registry");
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["serve", "--model"])
        .arg(&model)
        .args(["--cpu", "--port", "0"])
        .arg("--set")
        .arg(format!("memory.machine.registry={}", registry.display()))
        .args([
            "--set",
            "memory.machine.available_gb=3",
            "--set",
            "memory.machine.reserve_gb=4",
        ])
        .output()
        .expect("lloom runs");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "serve started without room:\n{err}");
    assert!(
        err.contains("the machine cannot hold this now"),
        "no refusal:\n{err}"
    );
    std::fs::remove_dir_all(&registry).ok();
    std::fs::remove_dir_all(&dir).ok();
}
