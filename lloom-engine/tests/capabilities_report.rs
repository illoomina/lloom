//! The capability report follows the load's arms. Qwen3.5's EOS gate is an
//! arm (`models.qwen3_5.eos_gate`), on by default for chat and serve on
//! Metal, so the `[capabilities: …]` line a loaded engine prints — the same
//! `Capabilities` `/health` serves — says what the family resolved, not the
//! family's row with the arm off.
#![cfg(feature = "serve")]

use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, Weights};
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lloom-capabilities-{}-{tag}", std::process::id()));
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

/// One `lloom chat` turn; its `[capabilities: …]` line and its turn line.
fn report(model: &Path, tok: &Path, extra: &[&str]) -> (String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(model)
        .arg("--tokenizer")
        .arg(tok)
        .args(["--raw", "-p", "Hi", "-n", "4", "-v"])
        .args(extra)
        .output()
        .expect("lloom runs");
    let err = String::from_utf8_lossy(&out.stderr).to_string();
    let all = format!("{err}{}", String::from_utf8_lossy(&out.stdout));
    assert!(out.status.success(), "{extra:?}: the turn ran:\n{all}");
    let line = |p: &str| {
        all.lines()
            .find(|l| l.starts_with(p))
            .unwrap_or_else(|| panic!("{extra:?}: no {p} line:\n{all}"))
            .to_string()
    };
    (line("[capabilities: "), line("[turn 1:"))
}

/// The gate's word in a line: `eos_gate=<word>`.
fn gate(line: &str) -> &str {
    line.split_whitespace()
        .find_map(|w| w.strip_prefix("eos_gate="))
        .unwrap_or_else(|| panic!("no eos_gate= in {line}"))
        .trim_end_matches(']')
}

/// On Metal: the front end's default turns the arm on, and the report says
/// so; turned off, it says that. The CPU keeps the library's value, off.
#[test]
fn qwen3_5s_capability_line_reports_the_gate_the_load_resolved() {
    // The tiny Qwen's head dim has no fused attention path; the reference
    // purpose attends through the composed oracle (`machine_gate.rs`).
    std::env::set_var("LLOOM_EXECUTION__PURPOSE", "reference");
    let dir = scratch("qwen3_5");
    let model = write::<Qwen3_5Tiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let mut arms: Vec<(Vec<&str>, &str)> = vec![(vec!["--cpu"], "false")];
    if cfg!(feature = "metal") {
        arms.push((vec![], "true"));
        arms.push((vec!["--set", "models.qwen3_5.eos_gate=false"], "false"));
    }
    for (extra, want) in arms {
        let (caps, turn) = report(&model, &tok, &extra);
        assert_eq!(
            gate(&caps),
            want,
            "{extra:?}: the capability line reports the arm the load resolved\n{caps}\n{turn}"
        );
        if want == "false" {
            assert_eq!(gate(&turn), "off", "{extra:?}: no gate rode the run\n{turn}");
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}
