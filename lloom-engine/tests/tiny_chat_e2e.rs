//! `lloom chat --model <tiny>` end to end on both families
//! (`docs/porting-models.md` §21.8, #761): each family's tiny checkpoint is
//! written to real files — safetensors + `config.json` for LFM2, one GGUF
//! for Qwen4-exp, each with the byte-level tokenizer and the family's own
//! chat template — and the shipped binary loads it through family
//! detection, renders the template, prefills and decodes.
//!
//! What is asserted, per family and device:
//! - the process exits 0 and reports its turn;
//! - the prompt it encoded is exactly the ids the tokenizer is built to give
//!   (`prompt_ids_sha256` against ids computed here, not by the engine);
//! - it generated tokens, and a second run generated the same ones.
#![cfg(feature = "serve")]

use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, Weights};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::Command;

const PROMPT: &str = "Hi! \"The capital of France\" is";
const MAX_TOKENS: &str = "8";

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("lloom-tiny-e2e-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The family's template, widened to the tokenizer's ids, written to `dir`.
fn write<F: TinyFamily>(dir: &Path) -> PathBuf {
    let mut cfg = F::template();
    cfg.set_axis("vocab_size", F::tokenizer().vocab_size());
    cfg.derive();
    TinyCheckpoint::<F>::new(cfg, Weights::default())
        .write(dir)
        .expect("the tiny checkpoint writes")
}

/// `lloom-determinism`'s digest: SHA-256 over the ids as little-endian u32.
fn sha(ids: &[u32]) -> String {
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    format!("{:x}", h.finalize())
}

/// One `lloom chat --raw` turn; the `[turn 1: …]` line's fields.
fn chat(model: &Path, extra: &[&str]) -> std::collections::BTreeMap<String, String> {
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(model)
        .args(["--raw", "-p", PROMPT, "-n", MAX_TOKENS])
        .args(extra)
        .output()
        .expect("lloom runs");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "lloom chat {model:?} {extra:?} exited {:?}:\n{stderr}",
        out.status
    );
    let line = stderr
        .lines()
        .find(|l| l.starts_with("[turn 1:"))
        .unwrap_or_else(|| panic!("no turn line:\n{stderr}"));
    eprintln!("{line}");
    line.trim_start_matches("[turn 1:")
        .trim_end_matches(']')
        .split_whitespace()
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn check(model: &Path, extra: &[&str], want_prompt: &[u32]) {
    let a = chat(model, extra);
    assert_eq!(
        a["prompt_ids_sha256"],
        sha(want_prompt),
        "the engine encoded something other than {want_prompt:?}"
    );
    assert_eq!(a["prompt_tokens"], want_prompt.len().to_string());
    let n: usize = a["completion_tokens"].parse().unwrap();
    assert!(n > 0, "nothing generated: {a:?}");
    let b = chat(model, extra);
    assert_eq!(
        a["tokens_sha256"], b["tokens_sha256"],
        "two runs, two answers"
    );
}

fn bytes(s: &str) -> Vec<u32> {
    s.bytes().map(u32::from).collect()
}

/// LFM2: the raw prompt is the bytes behind the checkpoint's BOS
/// (`<|startoftext|>`), which the engine prepends.
#[test]
fn lfm2_tiny_chats_on_the_cpu_and_metal() {
    let dir = scratch("lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let bos = Lfm2Tiny::tokenizer().id("<|startoftext|>").unwrap();
    let mut want = vec![bos];
    want.extend(bytes(PROMPT));
    check(&model, &["--cpu"], &want);
    #[cfg(feature = "metal")]
    check(&model, &[], &want);
    std::fs::remove_dir_all(&dir).ok();
}

/// Qwen4-exp: no BOS; the tokenizer sits beside the GGUF.
#[test]
fn qwen4exp_tiny_chats_on_the_cpu_and_metal() {
    let dir = scratch("qwen4exp");
    let model = write::<Qwen4ExpTiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let tok = tok.to_str().unwrap();
    let want = bytes(PROMPT);
    let base = ["--tokenizer", tok, "--experts", "resident"];
    let mut cpu = base.to_vec();
    cpu.push("--cpu");
    check(&model, &cpu, &want);
    #[cfg(feature = "metal")]
    check(&model, &base, &want);
    std::fs::remove_dir_all(&dir).ok();
}
