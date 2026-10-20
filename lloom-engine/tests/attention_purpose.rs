//! The execution purpose and the one attention policy, end to end through
//! `lloom chat` on Metal (`docs/dependency-ordered-reuse-design.md` §8,
//! §13.6): a production load dispatches every attention through a prepared,
//! bounded-memory route and refuses at load any configuration under which an
//! attention takes the composed scores path, whatever option selected it; a
//! load that declares `execution.purpose = reference` admits the composed
//! oracle and says so.
//!
//! The fixtures: each tiny Qwen at head dim 32, which its fused route serves
//! (the production fixture), and at the template's head dim 16, which no
//! fused path serves (the composed oracle's fixture); the tiny LFM2 under
//! `--attn generic`, the composed reference arm. The CPU is not driven: off
//! Metal the composed attention is the backend's own and every purpose
//! admits it.
#![cfg(all(feature = "serve", feature = "metal"))]

use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, Weights};
use std::path::{Path, PathBuf};
use std::process::Command;

const PROMPT: &str = "Hi! \"The capital of France\" is";
const PURPOSE_ENV: &str = "LLOOM_EXECUTION__PURPOSE";
const REFERENCE_NOTE: &str = "execution purpose `reference`";
const PRODUCTION_REFUSAL: &str = "a production load on Metal dispatches only through a prepared, \
                                  bounded-memory attention route";

fn scratch(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "lloom-attention-purpose-{}-{tag}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// The family's template, widened to the tokenizer's ids, at `head_dim`
/// when given.
fn write<F: TinyFamily>(dir: &Path, head_dim: Option<usize>) -> PathBuf {
    let mut cfg = F::template();
    cfg.set_axis("vocab_size", F::tokenizer().vocab_size());
    if let Some(hd) = head_dim {
        assert!(cfg.set_axis("head_dim", hd), "head_dim is an axis");
    }
    cfg.derive();
    TinyCheckpoint::<F>::new(cfg, Weights::default())
        .write(dir)
        .expect("the tiny checkpoint writes")
}

/// One `lloom chat --raw` turn on the default device (Metal), with
/// `purpose` as the process's `execution.purpose` when given: whether it
/// exited 0, and its stderr.
fn chat(model: &Path, extra: &[&str], purpose: Option<&str>) -> (bool, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_lloom"));
    cmd.args(["chat", "--model"])
        .arg(model)
        .args(["--raw", "-p", PROMPT, "-n", "4"])
        .args(extra)
        .env_remove(PURPOSE_ENV);
    if let Some(p) = purpose {
        cmd.env(PURPOSE_ENV, p);
    }
    let out = cmd.output().expect("lloom runs");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// The `tokens_sha256` of the `[turn 1: …]` line.
fn tokens(stderr: &str) -> String {
    let line = stderr
        .lines()
        .find(|l| l.starts_with("[turn 1:"))
        .unwrap_or_else(|| panic!("no turn line:\n{stderr}"));
    line.split_whitespace()
        .find_map(|kv| kv.strip_prefix("tokens_sha256="))
        .unwrap_or_else(|| panic!("no tokens_sha256: {line}"))
        .to_string()
}

/// The GGUF families' load flags: their tokenizer, resident experts.
fn gguf_args(dir: &Path) -> Vec<String> {
    vec![
        "--tokenizer".into(),
        dir.join("tokenizer.json").display().to_string(),
        "--experts".into(),
        "resident".into(),
    ]
}

/// Decision 40's production-route fixture: each tiny Qwen at head dim 32
/// chats on Metal under the default purpose, production, twice to the same
/// tokens, and no reference note is printed. A production layer refuses the
/// composed attention at every call, so a completed turn is one no call of
/// which took it.
#[test]
fn a_qwen_at_a_covered_head_dim_chats_on_the_production_route() {
    for (tag, model) in [
        (
            "qwen4exp",
            write::<Qwen4ExpTiny> as fn(&Path, Option<usize>) -> PathBuf,
        ),
        ("qwen3_5", write::<Qwen3_5Tiny>),
    ] {
        let dir = scratch(&format!("{tag}-32"));
        let ckpt = model(&dir, Some(32));
        let args = gguf_args(&dir);
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let (ok, a) = chat(&ckpt, &args, None);
        assert!(ok, "{tag} at head dim 32 under production:\n{a}");
        assert!(!a.contains(REFERENCE_NOTE), "{tag}:\n{a}");
        let (ok, b) = chat(&ckpt, &args, None);
        assert!(ok, "{tag}, second run:\n{b}");
        assert_eq!(tokens(&a), tokens(&b), "{tag}: two runs, two answers");
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// Decision 41, one policy for every option: under production the load is
/// refused when the configuration puts an attention on the composed scores
/// path — a head dim no fused path serves, `kernels.fusion.disabled` at a
/// head dim one does, LFM2's `--attn generic` — and the refusal names the
/// route (the composed scores path) beside what selected it. The same
/// configurations load under the reference purpose, which says so on its
/// stderr. A fused route under the same option stays admitted: LFM2 at
/// `--attn sdpa`, the Qwen at head dim 32 without the option.
#[test]
fn production_refuses_the_composed_route_whatever_selects_it() {
    let dir = scratch("refusals");
    let qwen16 = write::<Qwen4ExpTiny>(&dir.join("qwen16"), None);
    let qwen32 = write::<Qwen4ExpTiny>(&dir.join("qwen32"), Some(32));
    let lfm2 = write::<Lfm2Tiny>(&dir.join("lfm2"), None);
    let q16 = gguf_args(&dir.join("qwen16"));
    let q32 = gguf_args(&dir.join("qwen32"));
    let mut q32_off = q32.clone();
    q32_off.push("--fused-off".into());
    // chat's Metal default KV layout is paged, which only FlashDecoding
    // reads; the LFM2 arms below name the attention, so they name the
    // strided layout with it.
    let generic: Vec<String> = ["--attn", "generic", "--kv-append", "cat"]
        .map(String::from)
        .to_vec();

    let cases: [(&str, &Path, &Vec<String>, &str); 3] = [
        ("qwen head dim 16", &qwen16, &q16, "head dim 16"),
        ("qwen fused off", &qwen32, &q32_off, "qwen4exp.fused_off"),
        (
            "lfm2 --attn generic",
            &lfm2,
            &generic,
            "the composed reference attention",
        ),
    ];
    for (name, model, args, selected_by) in cases {
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let (ok, refused) = chat(model, &args, None);
        assert!(!ok, "{name} loaded under production:\n{refused}");
        let line = refused
            .lines()
            .find(|l| l.contains("composed scores path"))
            .unwrap_or("");
        eprintln!("{name}: refused: {line}");
        assert!(
            refused.contains(selected_by)
                && refused.contains("composed scores path")
                && refused.contains(PRODUCTION_REFUSAL),
            "{name}: refused without naming the route and its cause:\n{refused}"
        );
        let (ok, reference) = chat(model, &args, Some("reference"));
        assert!(ok, "{name} under the reference purpose:\n{reference}");
        assert!(reference.contains(REFERENCE_NOTE), "{name}:\n{reference}");
        eprintln!("{name}: reference: loaded, {}", tokens(&reference));
    }

    let sdpa = ["--attn", "sdpa", "--kv-append", "cat"];
    let (ok, served) = chat(&lfm2, &sdpa, None);
    assert!(ok, "lfm2 --attn sdpa under production:\n{served}");
    let q32: Vec<&str> = q32.iter().map(String::as_str).collect();
    let (ok, served) = chat(&qwen32, &q32, Some("production"));
    assert!(ok, "qwen head dim 32 under production:\n{served}");
    std::fs::remove_dir_all(&dir).ok();
}
