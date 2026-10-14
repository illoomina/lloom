//! `lloom chat --model <tiny>` end to end on both families
//! (`docs/porting-models.md` §21.8, #761): each family's tiny checkpoint is
//! written to real files — safetensors + `config.json` for LFM2, one GGUF
//! for Qwen4-exp, each with the byte-level tokenizer and the family's tiny
//! chat template (`tiny_templates.rs`) — and the shipped binary loads it
//! through family detection, renders the template, prefills and decodes.
//!
//! What is asserted, per family and device:
//! - the process exits 0 and reports its turn;
//! - the prompt it encoded is exactly the ids the tokenizer is built to give
//!   (`prompt_ids_sha256` against ids computed here, not by the engine);
//! - it generated tokens, and a second run generated the same ones.
//!
//! And the thinking effort (`--reasoning-effort`, the `reasoning_effort`
//! request field), through the template: on Qwen4-exp each value's prompt is
//! the ids of the rendering written out here, and no value is `xhigh`'s; an
//! effort the template would not act on (LFM2's reads none; Qwen4-exp's with
//! thinking off has no effect) is refused by `chat` and a 400 on `serve`.
//! The thinking switch (`--think` / `--no-think`, `enable_thinking`): Qwen4-exp
//! through its template's switch, LFM2 by the engine closing the block its
//! template opens. And `serve`'s request surface: top-level `tools`, and a
//! template that fails on the request's data is a 400.
#![cfg(feature = "serve")]

use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::models::qwen3_5::testing::{Qwen3_5MoeTiny, Qwen3_5Tiny};
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::testing::tiny::{TinyCheckpoint, TinyConfig, TinyFamily, TinyTokenizer, Weights};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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
    chat_with_stderr(model, extra).0
}

/// [`chat`], with the process's stderr beside the turn's fields.
fn chat_with_stderr(
    model: &Path,
    extra: &[&str],
) -> (std::collections::BTreeMap<String, String>, String) {
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
    let fields = line
        .trim_start_matches("[turn 1:")
        .trim_end_matches(']')
        .split_whitespace()
        .filter_map(|kv| kv.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    (fields, stderr.to_string())
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

/// Qwen3.5, both variants: no BOS; the GGUF is detected by its header's
/// architecture (the family claims `qwen35` / `qwen35moe`, not every GGUF),
/// the tokenizer sits beside it.
#[test]
fn qwen3_5_tiny_chats_on_the_cpu_and_metal() {
    let dir = scratch("qwen3_5");
    let model = write::<Qwen3_5Tiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let tok = tok.to_str().unwrap();
    let want = bytes(PROMPT);
    let base = ["--tokenizer", tok];
    let mut cpu = base.to_vec();
    cpu.push("--cpu");
    check(&model, &cpu, &want);
    #[cfg(feature = "metal")]
    check(&model, &base, &want);
    std::fs::remove_dir_all(&dir).ok();

    let dir = scratch("qwen3_5_moe");
    let model = write::<Qwen3_5MoeTiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let tok = tok.to_str().unwrap();
    let base = ["--tokenizer", tok];
    let mut cpu = base.to_vec();
    cpu.push("--cpu");
    check(&model, &cpu, &want);
    #[cfg(feature = "metal")]
    check(&model, &base, &want);
    std::fs::remove_dir_all(&dir).ok();
}

/// Three turns of `lloom chat` on `model` with `extra` flags: the three
/// `[turn N: …]` lines' fields, and the process's stderr.
fn three_turns(
    model: &Path,
    tok: &str,
    extra: &[&str],
) -> (Vec<std::collections::BTreeMap<String, String>>, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(model)
        .args(["--tokenizer", tok, "-n", "4", "--profile", "deterministic"])
        // The blocking readback: under a GPU stall the ring arm's streamed
        // text was measured short of its tokens (a reply two ids shorter,
        // and the next prompt with it; `measurements/issue-940.md` §2) —
        // no reuse causes it, and it would read as one here.
        .args(["--readback", "blocking"])
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("lloom runs");
    {
        let mut stdin = child.stdin.take().unwrap();
        for line in ["first question", "second one", "third"] {
            writeln!(stdin, "{line}").unwrap();
        }
    }
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    assert!(out.status.success(), "lloom chat {extra:?}:\n{stderr}");
    let turns = stderr
        .lines()
        .filter(|l| l.starts_with("[turn ") && l[6..].starts_with(|c: char| c.is_ascii_digit()))
        .map(|l| {
            l.split_whitespace()
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.to_string(), v.trim_end_matches(']').to_string()))
                .collect()
        })
        .collect::<Vec<_>>();
    assert_eq!(turns.len(), 3, "{stderr}");
    (turns, stderr)
}

/// The reuse of three turns against their `--no-reuse` run, on `model`
/// with `arms`: turns 2 and 3 reuse what the template keeps — `Extend` when
/// the previous turn's prompt and reply are all in the next prompt (the
/// reply's sampled ids), else `Rewound` to no earlier than the previous
/// turn's checkpoint, which stands where the next render diverges — and
/// every turn's tokens are the fresh run's.
fn assert_three_turns_reuse(model: &Path, tok: &str, arms: Vec<Vec<&str>>) {
    for arm in arms {
        let (reused, stderr) = three_turns(model, tok, &arm);
        assert!(
            stderr.contains("[checkpoint ring: 2 ×"),
            "{arm:?}: the load reports the checkpoint ring:\n{stderr}"
        );
        let mut fresh_arm = arm.clone();
        fresh_arm.push("--no-reuse");
        let (fresh, fresh_stderr) = three_turns(model, tok, &fresh_arm);
        assert_eq!(reused[0]["reuse"], "Fresh", "{arm:?}:\n{stderr}");
        for t in 1..3 {
            let prompt: usize = reused[t]["prompt_tokens"].parse().unwrap();
            let prev: usize = reused[t - 1]["prompt_tokens"].parse().unwrap();
            let reply: usize = reused[t - 1]["completion_tokens"].parse().unwrap();
            let kept: usize = reused[t]["reused"].parse().unwrap();
            let prefilled: usize = reused[t]["prefilled"].parse().unwrap();
            // The session log is the previous prompt and its reply. All of
            // it in the next prompt extends; else the cache goes back to
            // where they part, never before the previous turn's checkpoint
            // (`gen_prompt` ids before its prompt's end, the load-time
            // constant: where the next render diverges).
            let log = prev + reply;
            let gen: usize = reused[t - 1]["gen_prompt"].parse().unwrap();
            let ok = if kept >= log {
                reused[t]["reuse"] == "Extend"
            } else {
                reused[t]["reuse"] == format!("Rewound({})", log - kept) && kept + gen >= prev
            };
            assert!(
                ok,
                "{arm:?} turn {}: {} reused={kept} of the previous {prev} (the re-render restores \
                 to the previous turn's checkpoint: {reply} reply tokens and a short tail dropped, \
                 or extends the log):\n{stderr}",
                t + 1,
                reused[t]["reuse"]
            );
            assert_eq!(
                kept + prefilled,
                prompt,
                "{arm:?} turn {}:\n{stderr}",
                t + 1
            );
        }
        for t in 0..3 {
            assert_eq!(
                fresh[t]["reuse"], "Fresh",
                "{arm:?}: --no-reuse prefills every turn whole"
            );
            assert_eq!(
                reused[t]["tokens_sha256"],
                fresh[t]["tokens_sha256"],
                "{arm:?} turn {}: the reused turn's tokens are the fresh run's:\n{stderr}",
                t + 1
            );
            assert_eq!(
                reused[t]["prompt_ids_sha256"],
                fresh[t]["prompt_ids_sha256"],
                "{arm:?} turn {}: the same conversation's prompt\n--- reused:\n{stderr}\n--- fresh:\n{fresh_stderr}",
                t + 1
            );
            assert_eq!(
                reused[t]["renders"],
                "1",
                "{arm:?} turn {}: one render a request, the conversation's own:\n{stderr}",
                t + 1
            );
        }
    }
}

/// Turn reuse on the generic decoder (issue #940): three turns on stdin
/// through the tiny Qwen3.5's template, both think modes, on the CPU and on
/// Metal. Turn 1 prefills whole and leaves its checkpoint `tail` tokens
/// before the end (the generation prompt's `<think>\n`, or the closed
/// block with thinking off, which the re-render drops); turns 2 and 3 are
/// `Rewound(reply + tail)` to it, prefilling the changed suffix only; and
/// every turn's tokens are the `--no-reuse` run's (a fresh prefill of the
/// same conversation). The load reports the ring.
#[test]
fn qwen3_5_tiny_reuses_the_conversation_across_turns() {
    let dir = scratch("qwen3_5_turns");
    let model = write::<Qwen3_5Tiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let tok = tok.to_str().unwrap().to_string();
    let mut arms: Vec<Vec<&str>> = vec![vec!["--cpu"], vec!["--cpu", "--no-think"]];
    if cfg!(feature = "metal") {
        arms.push(vec![]);
        arms.push(vec!["--no-think"]);
    }
    assert_three_turns_reuse(&model, &tok, arms);
    std::fs::remove_dir_all(&dir).ok();
}

/// Turn reuse on Qwen4-exp's own state (issue #940, review 1): the same
/// three turns on the tiny resident model, thinking unset and off, on the
/// CPU, beside `--no-reuse`.
#[test]
fn qwen4exp_tiny_reuses_the_conversation_across_turns() {
    let dir = scratch("qwen4exp_turns");
    let model = write::<Qwen4ExpTiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let tok = tok.to_str().unwrap().to_string();
    let arms: Vec<Vec<&str>> = vec![
        vec!["--experts", "resident", "--cpu"],
        vec!["--experts", "resident", "--cpu", "--no-think"],
    ];
    // Not on Metal: the tiny Qwen4-exp's routed f16 pair has no arm there
    // (`qwen4exp_tiny_chats_on_the_cpu_and_metal` is red on dev for it,
    // `measurements/issue-932.md` §1); the restore itself is held on Metal
    // by `lloom/tests/qwen4exp_turn_checkpoint.rs`.
    assert_three_turns_reuse(&model, &tok, arms);
    std::fs::remove_dir_all(&dir).ok();
}

/// Qwen3.5 through the speculative body: a tiny checkpoint that keeps its
/// MTP block (`block_count` = `n_layers` + 1, `nextn_predict_layers` 1,
/// unsloth's Qwen3.8-27B shape) served with `--speculate mtp` emits the
/// greedy run's tokens, at `--spec-tokens` 1 and 2, on the CPU and on
/// Metal; the turn reports its rounds. A checkpoint without the block
/// refuses the method by name.
#[test]
fn qwen3_5_tiny_mtp_is_byte_identical_to_greedy_through_chat() {
    let dir = scratch("qwen3_5_mtp");
    let mut cfg = Qwen3_5Tiny::template();
    TinyConfig::set_axis(
        &mut cfg,
        "vocab_size",
        Qwen3_5Tiny::tokenizer().vocab_size(),
    );
    TinyConfig::derive(&mut cfg);
    cfg.n_nextn_layers = 1;
    let model = TinyCheckpoint::<Qwen3_5Tiny>::new(cfg, Weights::default())
        .write(&dir)
        .expect("the tiny checkpoint writes");
    let tok = dir.join("tokenizer.json");
    let tok = tok.to_str().unwrap();
    let want = bytes(PROMPT);
    let mut devices: Vec<&[&str]> = vec![&["--cpu"]];
    if cfg!(feature = "metal") {
        devices.push(&[]);
    }
    for dev in devices {
        let mut base = vec![
            "--tokenizer",
            tok,
            "--sampling-defaults",
            "greedy",
            "-n",
            "12",
        ];
        base.extend_from_slice(dev);
        let greedy = chat(&model, &base);
        assert_eq!(greedy["prompt_ids_sha256"], sha(&want));
        assert_eq!(greedy["completion_tokens"], "12");
        for k in ["1", "2"] {
            let mut args = base.clone();
            args.extend_from_slice(&["--speculate", "mtp", "--spec-tokens", k]);
            let (spec, stderr) = chat_with_stderr(&model, &args);
            assert_eq!(
                spec["tokens_sha256"], greedy["tokens_sha256"],
                "{dev:?} k={k}: the speculative body's tokens are not greedy's"
            );
            assert_eq!(spec["completion_tokens"], "12");
            // The acceptance line (`report::print_rounds`): rounds and the
            // tokens they kept, the bonus counted.
            let accept = stderr
                .lines()
                .find(|l| l.starts_with("DSPARK_ACCEPT "))
                .unwrap_or_else(|| panic!("{dev:?} k={k}: no acceptance line:\n{stderr}"));
            eprintln!("{accept}");
            let field = |name: &str| -> usize {
                accept
                    .split_whitespace()
                    .find_map(|kv| kv.strip_prefix(&format!("{name}=")))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or_else(|| panic!("no {name} on {accept}"))
            };
            let (rounds, tokens) = (field("rounds"), field("tokens"));
            assert!((1..=12).contains(&rounds), "{dev:?} k={k}: {rounds} rounds");
            // The first token is the prefill's, before round 0.
            assert!(
                tokens + 1 >= 12,
                "{dev:?} k={k}: the rounds kept {tokens} tokens for 12 emitted"
            );
        }
    }
    // The logits digest (`--logits-digest`): the same tokens as the run
    // without it, a digest the same on a second run — and under the
    // speculative body the same digest: on the CPU the generic decoder's
    // window runs position by position through the decode step, so the rows
    // a round keeps are the K=1 steps' logits, bit for bit.
    let mut args = vec![
        "--tokenizer",
        tok,
        "--sampling-defaults",
        "greedy",
        "-n",
        "12",
        "--cpu",
    ];
    let greedy = chat(&model, &args);
    assert_eq!(greedy["logits_sha256"], "-");
    args.push("--logits-digest");
    let a = chat(&model, &args);
    let b = chat(&model, &args);
    assert_eq!(a["tokens_sha256"], greedy["tokens_sha256"]);
    assert_eq!(a["logits_sha256"].len(), 64, "{a:?}");
    assert_eq!(a["logits_sha256"], b["logits_sha256"]);
    assert_eq!(
        a["readback"], "blocking",
        "the digest reads the logits on the host"
    );
    for k in ["1", "2"] {
        let mut spec_args = args.clone();
        spec_args.extend_from_slice(&["--speculate", "mtp", "--spec-tokens", k]);
        let s = chat(&model, &spec_args);
        assert_eq!(s["tokens_sha256"], greedy["tokens_sha256"], "k={k}");
        assert_eq!(
            s["logits_sha256"], a["logits_sha256"],
            "k={k}: the speculative body's digest over the kept rows is not the K=1 run's"
        );
    }
    std::fs::remove_dir_all(&dir).ok();

    // Without the block, the method is refused by name.
    let dir = scratch("qwen3_5_no_mtp");
    let model = write::<Qwen3_5Tiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(&model)
        .args(["--raw", "-p", PROMPT, "-n", "4", "--cpu", "--tokenizer"])
        .arg(&tok)
        .args(["--speculate", "mtp"])
        .output()
        .expect("lloom runs");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "a checkpoint without the block served MTP:\n{stderr}"
    );
    assert!(stderr.contains("no MTP block"), "{stderr}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The tiny tokenizer's ids for `s`, computed here: an added token where one
/// starts, a byte everywhere else.
fn encode(tok: &TinyTokenizer, s: &str) -> Vec<u32> {
    let mut ids = Vec::new();
    let mut rest = s;
    while !rest.is_empty() {
        let added = tok
            .added
            .iter()
            .filter(|(c, _)| rest.starts_with(c.as_str()))
            .max_by_key(|(c, _)| c.len());
        match added {
            Some((c, _)) => {
                ids.push(tok.id(c).unwrap());
                rest = &rest[c.len()..];
            }
            None => {
                ids.push(u32::from(rest.as_bytes()[0]));
                rest = &rest[1..];
            }
        }
    }
    ids
}

const LOW: &str = "Thinking effort: low. Think briefly.";
const XHIGH: &str = "Thinking effort: xhigh. Think the problem through.";
const REJECTED: &str =
    "tiny qwen4exp template: reasoning_effort high is not one of low, medium, xhigh";

/// Qwen4-exp's rendering of one user message with thinking on: the effort's
/// instruction as the system turn (`medium` has none), then the open think
/// block.
fn qwen_thinking(instruction: Option<&str>) -> String {
    let system = instruction
        .map(|i| format!("<|im_start|>system\n{i}<|im_end|>\n"))
        .unwrap_or_default();
    format!("{system}<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n")
}

/// Qwen4-exp's rendering with thinking off: no instruction, an empty think block.
fn qwen_not_thinking() -> String {
    format!("<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

/// One templated `lloom chat` turn: its exit status, stderr, and the turn
/// line's fields when there is one.
fn chat_templated(
    model: &Path,
    extra: &[&str],
) -> (bool, String, std::collections::BTreeMap<String, String>) {
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(model)
        .args(["-p", PROMPT, "-n", "4"])
        .args(extra)
        .output()
        .expect("lloom runs");
    let stderr = String::from_utf8_lossy(&out.stderr).to_string();
    let fields = stderr
        .lines()
        .find(|l| l.starts_with("[turn 1:"))
        .map(|line| {
            line.trim_start_matches("[turn 1:")
                .trim_end_matches(']')
                .split_whitespace()
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect()
        })
        .unwrap_or_default();
    (out.status.success(), stderr, fields)
}

fn qwen_tiny(tag: &str) -> (PathBuf, PathBuf, Vec<String>) {
    let dir = scratch(tag);
    let model = write::<Qwen4ExpTiny>(&dir);
    let tok = dir.join("tokenizer.json").to_str().unwrap().to_string();
    let args = vec![
        "--tokenizer".into(),
        tok,
        "--experts".into(),
        "resident".into(),
        "--cpu".into(),
    ];
    (dir, model, args)
}

/// `lloom chat --think --reasoning-effort E` on the tiny Qwen4-exp: the
/// prompt is the rendering with E's instruction, the turn line names E, and
/// no flag is the template's default — byte-identical to `xhigh`.
#[test]
fn qwen4exp_chat_renders_each_reasoning_effort() {
    let (dir, model, base) = qwen_tiny("effort-chat");
    let tok = Qwen4ExpTiny::tokenizer();
    let mut seen = std::collections::BTreeMap::new();
    for (effort, instruction) in [
        (Some("low"), Some(LOW)),
        (Some("medium"), None),
        (Some("xhigh"), Some(XHIGH)),
        (None, Some(XHIGH)),
    ] {
        let mut args: Vec<&str> = base.iter().map(String::as_str).collect();
        args.push("--think");
        if let Some(e) = effort {
            args.extend(["--reasoning-effort", e]);
        }
        let (ok, stderr, f) = chat_templated(&model, &args);
        assert!(ok, "{args:?}:\n{stderr}");
        let want = encode(&tok, &qwen_thinking(instruction));
        assert_eq!(
            f["prompt_ids_sha256"],
            sha(&want),
            "{effort:?}: the prompt is not the rendering"
        );
        assert_eq!(f["prompt_tokens"], want.len().to_string());
        assert_eq!(f["reasoning_effort"], effort.unwrap_or("default"));
        assert!(f["completion_tokens"].parse::<usize>().unwrap() > 0);
        seen.insert(effort.unwrap_or("default"), f["prompt_ids_sha256"].clone());
    }
    assert_ne!(seen["low"], seen["xhigh"]);
    assert_ne!(seen["medium"], seen["xhigh"]);
    assert_eq!(seen["default"], seen["xhigh"]);
    std::fs::remove_dir_all(&dir).ok();
}

/// The refusals, before any turn and by name: a value the template rejects
/// (its own message), an effort the template ignores with thinking off, an
/// effort on LFM2 (whose template does not read it), and `--raw`.
#[test]
fn chat_refuses_an_effort_the_template_rejects_or_does_not_act_on() {
    let (dir, model, base) = qwen_tiny("effort-refuse");
    let with = |extra: &[&str]| {
        let mut a: Vec<&str> = base.iter().map(String::as_str).collect();
        a.extend(extra);
        chat_templated(&model, &a)
    };
    let (ok, stderr, f) = with(&["--think", "--reasoning-effort", "high"]);
    assert!(!ok && f.is_empty(), "{stderr}");
    assert!(
        stderr.contains(&format!(
            "--reasoning-effort high: the chat template rejected the request: {REJECTED}"
        )),
        "{stderr}"
    );
    let (ok, stderr, f) = with(&["--reasoning-effort", "low"]);
    assert!(!ok && f.is_empty(), "{stderr}");
    assert!(stderr.contains(
        "--reasoning-effort low: qwen4exp's chat template renders the same prompt for any reasoning_effort \
         with enable_thinking=false (add --think)"
    ), "{stderr}");
    let (ok, stderr, _) = with(&["--raw", "--reasoning-effort", "low"]);
    assert!(!ok, "{stderr}");
    assert!(stderr.contains("--raw renders no template"), "{stderr}");
    std::fs::remove_dir_all(&dir).ok();

    let dir = scratch("effort-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let (ok, stderr, f) = chat_templated(&model, &["--cpu", "--reasoning-effort", "low"]);
    assert!(!ok && f.is_empty(), "{stderr}");
    assert!(
        stderr.contains("--reasoning-effort: lfm2's chat template does not read reasoning_effort"),
        "{stderr}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// A `lloom serve` process on a tiny checkpoint, driven over its socket.
struct Server {
    child: std::process::Child,
    port: u16,
}

impl Server {
    fn start(model: &Path, extra: &[&str]) -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_lloom"))
            .args(["serve", "--model"])
            .arg(model)
            .args(extra)
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("lloom serve runs");
        let mut line = String::new();
        std::io::BufReader::new(child.stdout.take().unwrap())
            .read_line(&mut line)
            .unwrap();
        let ready: Value = serde_json::from_str(&line).unwrap_or_else(|e| panic!("{line:?}: {e}"));
        assert_eq!(ready["kind"], "ready");
        Self {
            child,
            port: ready["port"].as_u64().unwrap() as u16,
        }
    }

    /// One request: the status and the JSON body.
    fn call(&self, method: &str, path: &str, body: &Value) -> (u16, Value) {
        let mut s = std::net::TcpStream::connect(("127.0.0.1", self.port)).unwrap();
        let body = body.to_string();
        write!(
            s,
            "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut response = String::new();
        s.read_to_string(&mut response).unwrap();
        let (head, body) = response.split_once("\r\n\r\n").unwrap();
        let status = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (
            status,
            serde_json::from_str(body).unwrap_or_else(|e| panic!("{body:?}: {e}")),
        )
    }

    fn chat(&self, fields: Value) -> (u16, Value) {
        let mut req = json!({
            "messages": [{ "role": "user", "content": PROMPT }],
            "max_tokens": 4,
        });
        for (k, v) in fields.as_object().unwrap() {
            req[k] = v.clone();
        }
        self.call("POST", "/v1/chat/completions", &req)
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.call("POST", "/shutdown", &json!({}));
        let _ = self.child.wait();
    }
}

/// `lloom serve` on the tiny Qwen4-exp: the request's `reasoning_effort`
/// (either spelling) is the prompt's, and `timings` says so with the
/// prompt's ids; a value the template rejects is a 400 carrying its message;
/// an effort the template ignores (thinking off) is served, listed as ignored.
#[test]
fn qwen4exp_serve_takes_the_reasoning_effort_request_field() {
    let (dir, model, base) = qwen_tiny("effort-serve");
    let tok = Qwen4ExpTiny::tokenizer();
    let args: Vec<&str> = base.iter().map(String::as_str).collect();
    let server = Server::start(&model, &args);
    let think = json!({ "enable_thinking": true });

    let (_, health) = server.call("GET", "/health", &json!({}));
    assert_eq!(
        health["chat_template_reads"],
        json!({"enable_thinking": true, "reasoning_effort": true})
    );

    let (status, low) =
        server.chat(json!({ "reasoning_effort": "low", "chat_template_kwargs": think }));
    assert_eq!(status, 200, "{low}");
    assert_eq!(low["timings"]["reasoning_effort"], "low");
    assert_eq!(
        low["timings"]["prompt_ids_sha256"],
        sha(&encode(&tok, &qwen_thinking(Some(LOW))))
    );
    assert_eq!(low["timings"]["thinking"], "on");
    assert_eq!(low["timings"]["thinking_by"], "template");

    let (status, xhigh) = server.chat(json!({
        "chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": "xhigh" },
    }));
    assert_eq!(status, 200, "{xhigh}");
    assert_eq!(xhigh["timings"]["reasoning_effort"], "xhigh");
    let want = sha(&encode(&tok, &qwen_thinking(Some(XHIGH))));
    assert_eq!(xhigh["timings"]["prompt_ids_sha256"], want);
    let (status, default) = server.chat(json!({ "chat_template_kwargs": think }));
    assert_eq!(status, 200, "{default}");
    assert_eq!(default["timings"]["reasoning_effort"], Value::Null);
    assert_eq!(
        default["timings"]["prompt_ids_sha256"], want,
        "no effort is the template's xhigh"
    );

    let (status, err) =
        server.chat(json!({ "reasoning_effort": "high", "chat_template_kwargs": think }));
    assert_eq!(status, 400, "{err}");
    assert_eq!(
        err["error"]["message"],
        format!("the chat template rejected the request: {REJECTED}")
    );
    let (status, err) = server.chat(json!({
        "reasoning_effort": "low",
        "chat_template_kwargs": { "enable_thinking": true, "reasoning_effort": "xhigh" },
    }));
    assert_eq!(status, 400, "{err}");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("disagree"),
        "{err}"
    );

    let (status, off) =
        server.chat(json!({ "chat_template_kwargs": { "enable_thinking": false } }));
    assert_eq!(status, 200, "{off}");
    assert_eq!(
        off["timings"]["prompt_ids_sha256"],
        sha(&encode(&tok, &qwen_not_thinking()))
    );
    assert_eq!(off["timings"]["thinking"], "off");
    assert_eq!(off["timings"]["thinking_by"], "template");
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// An effort the template would not act on is a 400 on `serve`, by either
/// spelling, naming the family and why: LFM2's template does not read it;
/// Qwen4-exp's renders the same prompt for any effort with thinking off. The
/// same requests without the effort are served.
#[test]
fn serve_refuses_an_effort_the_template_does_not_act_on() {
    let dir = scratch("effort-serve-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let server = Server::start(&model, &["--cpu"]);
    let (_, health) = server.call("GET", "/health", &json!({}));
    assert_eq!(
        health["chat_template_reads"],
        json!({"enable_thinking": false, "reasoning_effort": false})
    );
    let not_read = "reasoning_effort=\"low\": lfm2's chat template does not read reasoning_effort";
    for req in [
        json!({ "reasoning_effort": "low" }),
        json!({ "chat_template_kwargs": { "reasoning_effort": "low" } }),
    ] {
        let (status, err) = server.chat(req.clone());
        assert_eq!(status, 400, "{req}: {err}");
        assert_eq!(err["error"]["message"], not_read, "{req}");
    }
    let (status, without) = server.chat(json!({ "reasoning_effort": null }));
    assert_eq!(status, 200, "{without}");
    assert_eq!(without["timings"]["reasoning_effort"], Value::Null);
    drop(server);
    std::fs::remove_dir_all(&dir).ok();

    let (dir, model, base) = qwen_tiny("effort-serve-off");
    let args: Vec<&str> = base.iter().map(String::as_str).collect();
    let server = Server::start(&model, &args);
    let no_effect = "reasoning_effort=\"low\": qwen4exp's chat template renders the same prompt \
                     for any reasoning_effort with enable_thinking=false";
    for req in [
        json!({ "reasoning_effort": "low", "enable_thinking": false }),
        json!({ "chat_template_kwargs": { "reasoning_effort": "low", "enable_thinking": false } }),
    ] {
        let (status, err) = server.chat(req.clone());
        assert_eq!(status, 400, "{req}: {err}");
        assert_eq!(err["error"]["message"], no_effect, "{req}");
    }
    let (status, ok) = server.chat(json!({ "enable_thinking": false }));
    assert_eq!(status, 200, "{ok}");
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// The ids a tiny family gives an added token.
fn added_id(tok: &TinyTokenizer, content: &str) -> u32 {
    tok.id(content)
        .unwrap_or_else(|| panic!("{content} is not an added token"))
}

/// The decision scoring on `lloom serve`: `logprobs` with `options` scores
/// every option's first token under one forward on the tiny Qwen3.5 (the
/// decision models' family) — the chosen option carries the highest
/// log-probability, every option's logprob is that token's in the full
/// distribution (the top-k block agrees where it lists it), an option
/// encoding to several tokens reports the count, two options sharing a
/// first token are flagged, and the shapes a decision cannot take are 400s.
#[test]
fn serve_scores_decision_options_from_one_forward() {
    let dir = scratch("decide-qwen3_5");
    let model = write::<Qwen3_5Tiny>(&dir);
    let tok = dir.join("tokenizer.json");
    let server = Server::start(&model, &["--cpu", "--tokenizer", tok.to_str().unwrap()]);
    let req = |options: Value, top: u64| {
        json!({
            "max_tokens": 1,
            "logprobs": true,
            "top_logprobs": top,
            "options": options,
        })
    };
    let (status, body) = server.chat(req(json!(["A", "B", "yes!", "no"]), 64));
    assert_eq!(status, 200, "{body}");
    let d = &body["choices"][0]["decision"];
    let opts = d["options"].as_array().unwrap();
    assert_eq!(opts.len(), 4, "{d}");
    let lps: Vec<f64> = opts
        .iter()
        .map(|o| o["logprob"].as_f64().unwrap())
        .collect();
    let choice = d["choice"].as_u64().unwrap() as usize;
    let best = lps.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_eq!(lps[choice], best, "{d}");
    assert_eq!(d["text"], opts[choice]["text"]);
    assert!(lps.iter().all(|lp| lp.is_finite() && *lp <= 0.0), "{d}");
    assert_eq!(
        opts[2]["tokens"], 4,
        "`yes!` is four byte tokens on the tiny tokenizer: {d}"
    );
    assert_eq!(opts[0]["tokens"], 1);
    assert_eq!(d["ambiguous"], json!([]));
    // The top-k block and the decision agree on a token both list.
    let top = body["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        .as_array()
        .unwrap();
    for o in opts {
        if let Some(t) = top.iter().find(|t| t["token"] == o["token"]) {
            assert_eq!(t["logprob"], o["logprob"], "{o} vs {t}");
        }
    }
    // Two options with one first token are flagged.
    let (status, body) = server.chat(req(json!(["yes", "yes!"]), 1));
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["choices"][0]["decision"]["ambiguous"], json!([0, 1]));
    // The shapes a decision cannot take.
    for (options, why) in [
        (json!(["A"]), "at least two"),
        (json!(["A", ""]), "non-empty"),
        (json!("A"), "array"),
    ] {
        let (status, body) = server.chat(req(options, 1));
        assert_eq!(status, 400, "{body}");
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains(why),
            "{body}"
        );
    }
    // Without options the block is absent and the reply is as before.
    let (status, body) = server.chat(json!({ "max_tokens": 1, "logprobs": true }));
    assert_eq!(status, 200, "{body}");
    assert!(body["choices"][0]["decision"].is_null());
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// A `lloom serve` chat request's generated ids and `timings`.
fn served_ids(server: &Server, mut fields: Value) -> (Vec<u32>, Value) {
    fields["return_token_ids"] = json!(true);
    let (status, body) = server.chat(fields);
    assert_eq!(status, 200, "{body}");
    let ids = body["timings"]["tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.as_u64().unwrap() as u32)
        .collect();
    (ids, body["timings"].clone())
}

/// The thinking budget through `lloom serve` on both tiny families, on the
/// CPU and on Metal, on each family's default readback arm and a ring arm
/// beside it. Each prompt leaves a block open (Qwen4-exp with thinking on;
/// LFM2's template always opens one). The model is kept from closing the
/// block itself (a bias of -100 on the end marker and the tool-call start),
/// so at the budget of 3 — counted, as vLLM counts it, from the start marker,
/// the prompt's own tokens after it included — the forced tail (the message's
/// bytes, then the end marker) is fed, and the ids before it are the
/// unbudgeted run's. A budget
/// past `max_tokens` changes nothing. With the tool-call start biased up
/// instead, the model calls a tool at once, which closes the block, and
/// nothing is forced: every id is the unbudgeted run's. A budget of 0 (no
/// bias, so the default arm's fused publish runs after the tail) ends the
/// block at the first token.
#[test]
fn thinking_budget_forces_the_end_marker_through_serve() {
    struct Family {
        name: &'static str,
        tok: TinyTokenizer,
        model: PathBuf,
        args: Vec<String>,
        kwargs: Value,
        tool: &'static str,
        /// Tokens of the block the prompt holds after `<think>`, which the
        /// budget counts: Qwen4-exp's prompt ends `<think>\n`, LFM2's `<think>`.
        prior: usize,
    }
    let q_dir = scratch("budget-qwen");
    let l_dir = scratch("budget-lfm2");
    let q_model = write::<Qwen4ExpTiny>(&q_dir);
    let families = [
        Family {
            name: "qwen4exp",
            tok: Qwen4ExpTiny::tokenizer(),
            args: vec![
                "--tokenizer".into(),
                q_dir.join("tokenizer.json").to_str().unwrap().into(),
                "--experts".into(),
                "resident".into(),
            ],
            model: q_model,
            kwargs: json!({ "enable_thinking": true }),
            tool: "<tool_call>",
            prior: 1,
        },
        Family {
            name: "lfm2",
            tok: Lfm2Tiny::tokenizer(),
            model: write::<Lfm2Tiny>(&l_dir),
            args: vec![],
            kwargs: json!({}),
            tool: "<|tool_call_start|>",
            prior: 0,
        },
    ];
    let mut devices = vec![true];
    if cfg!(feature = "metal") {
        devices.push(false);
    }
    let message = "Enough.";
    for f in &families {
        let end = added_id(&f.tok, "</think>");
        let tool = added_id(&f.tok, f.tool);
        let mut tail = bytes(message);
        tail.push(end);
        for &cpu in &devices {
            for arm in [None, Some("ring-la0")] {
                let mut args: Vec<&str> = f.args.iter().map(String::as_str).collect();
                if cpu {
                    args.push("--cpu");
                }
                if let Some(a) = arm {
                    args.extend(["--readback", a]);
                }
                let at = format!("{} cpu={cpu} {arm:?}", f.name);
                let server = Server::start(&f.model, &args);
                let (_, health) = server.call("GET", "/health", &json!({}));
                assert_eq!(
                    health["reasoning_tags"],
                    json!({ "start": "<think>", "end": "</think>", "tool_call": f.tool }),
                    "{at}"
                );
                let req = |bias: Value, extra: Value| {
                    let mut r = json!({
                        "max_tokens": 16,
                        "ignore_eos": true,
                        "chat_template_kwargs": f.kwargs,
                        "logit_bias": bias,
                    });
                    for (k, v) in extra.as_object().unwrap() {
                        r[k] = v.clone();
                    }
                    r
                };
                let closed = json!({ end.to_string(): -100, tool.to_string(): -100 });
                let (plain, t) = served_ids(&server, req(closed.clone(), json!({})));
                assert_eq!(t["thinking_budget"], Value::Null, "{at}");
                assert_eq!(plain.len(), 16, "{at}: {plain:?}");

                let (got, t) = served_ids(
                    &server,
                    req(
                        closed.clone(),
                        json!({ "thinking_token_budget": 3, "thinking_budget_message": message }),
                    ),
                );
                let cut = 3 - f.prior;
                assert_eq!(got[..cut], plain[..cut], "{at}: before the budget");
                assert_eq!(
                    got[cut..cut + tail.len()],
                    tail[..],
                    "{at}: the forced tail"
                );
                assert_eq!(t["thinking_budget"], 3, "{at}");
                assert_eq!(t["thinking_budget_fired"], true, "{at}");
                assert_eq!(t["thinking_tokens"], 3 + bytes(message).len(), "{at}");
                assert_eq!(t["thinking_budget_ignored"], Value::Null, "{at}");

                let (got, t) = served_ids(
                    &server,
                    req(closed.clone(), json!({ "thinking_token_budget": 100 })),
                );
                assert_eq!(got, plain, "{at}: a budget past max_tokens");
                assert_eq!(t["thinking_budget_fired"], false, "{at}");
                assert_eq!(t["thinking_tokens"], 16 + f.prior, "{at}");

                let calls = json!({ end.to_string(): -100, tool.to_string(): 100 });
                let (plain_call, _) = served_ids(&server, req(calls.clone(), json!({})));
                assert_eq!(plain_call[0], tool, "{at}: the bias did not call a tool");
                let (got, t) =
                    served_ids(&server, req(calls, json!({ "thinking_token_budget": 3 })));
                assert_eq!(got, plain_call, "{at}: a tool call ends the block");
                assert_eq!(t["thinking_budget_fired"], false, "{at}");
                assert_eq!(t["thinking_tokens"], f.prior, "{at}");

                let (got, t) = served_ids(
                    &server,
                    json!({
                        "max_tokens": 6,
                        "ignore_eos": true,
                        "chat_template_kwargs": f.kwargs,
                        "thinking_token_budget": 0,
                    }),
                );
                assert_eq!(got[0], end, "{at}: a budget of 0");
                assert_eq!(t["thinking_budget_fired"], true, "{at}");
                assert_eq!(t["thinking_tokens"], f.prior, "{at}");
                drop(server);
            }
        }
    }
    std::fs::remove_dir_all(&q_dir).ok();
    std::fs::remove_dir_all(&l_dir).ok();
}

/// Where the thinking budget cannot apply: `lloom serve` serves the request
/// without it and says why (Qwen4-exp with thinking off, whose prompt closes
/// its block), and `lloom chat` refuses the flag by name before the first
/// turn — with the hint on Qwen4-exp, and on a template that writes no
/// thinking block at all (the tiny Qwen4-exp checkpoint with a plain
/// template beside it). With `--think` the chat runs it and says so.
#[test]
fn thinking_budget_is_ignored_or_refused_where_no_block_is_open() {
    let (dir, model, base) = qwen_tiny("budget-refuse");
    let args: Vec<&str> = base.iter().map(String::as_str).collect();
    let server = Server::start(&model, &args);
    let off = json!({ "enable_thinking": false });
    let (status, plain) =
        server.chat(json!({ "chat_template_kwargs": off, "return_token_ids": true }));
    assert_eq!(status, 200, "{plain}");
    let (status, with) = server.chat(json!({ "chat_template_kwargs": off,
        "thinking_token_budget": 2, "return_token_ids": true }));
    assert_eq!(status, 200, "{with}");
    assert_eq!(with["timings"]["tokens"], plain["timings"]["tokens"]);
    assert_eq!(with["timings"]["thinking_budget"], 2);
    assert_eq!(
        with["timings"]["thinking_budget_ignored"],
        "the prompt leaves no thinking block open"
    );
    assert_eq!(with["timings"]["thinking_budget_fired"], Value::Null);
    let (status, err) = server.chat(json!({ "thinking_token_budget": -2 }));
    assert_eq!(status, 400, "{err}");
    drop(server);

    // A server whose requests run the batch scheduler serves the budget
    // ignored, by that reason, rather than failing the request.
    let mut batched: Vec<&str> = base.iter().map(String::as_str).collect();
    batched.extend(["--batch", "2"]);
    let server = Server::start(&model, &batched);
    let on = json!({ "enable_thinking": true });
    let (status, plain) =
        server.chat(json!({ "chat_template_kwargs": on, "return_token_ids": true }));
    assert_eq!(status, 200, "{plain}");
    let (status, with) = server.chat(json!({ "chat_template_kwargs": on,
        "thinking_token_budget": 0, "return_token_ids": true }));
    assert_eq!(status, 200, "{with}");
    assert_eq!(with["timings"]["tokens"], plain["timings"]["tokens"]);
    assert!(
        with["timings"]["thinking_budget_ignored"]
            .as_str()
            .unwrap()
            .contains("the batch scheduler (--batch N)"),
        "{with}"
    );
    drop(server);

    let with_args = |extra: &[&str]| {
        let mut a: Vec<&str> = base.iter().map(String::as_str).collect();
        a.extend(extra);
        chat_templated(&model, &a)
    };
    let (ok, stderr, f) = with_args(&["--thinking-budget", "4"]);
    assert!(!ok && f.is_empty(), "{stderr}");
    assert!(
        stderr.contains(
            "--thinking-budget 4: qwen4exp: the prompt leaves no thinking block open (add --think)"
        ),
        "{stderr}"
    );
    let (ok, stderr, f) = with_args(&["--think", "--thinking-budget", "4"]);
    assert!(ok, "{stderr}");
    assert_eq!(f["thinking_budget"], "4");
    assert_ne!(f["budget_fired"], "none", "{stderr}");
    std::fs::write(
        dir.join("chat_template.jinja"),
        "{%- for m in messages -%}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{%- endfor -%}<|im_start|>assistant\n",
    )
    .unwrap();
    let (ok, stderr, f) = with_args(&["--thinking-budget", "4"]);
    assert!(!ok && f.is_empty(), "{stderr}");
    assert!(
        stderr
            .contains("--thinking-budget 4: qwen4exp: the chat template writes no thinking block"),
        "{stderr}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The prompt budget on a raw prompt (`lloom chat --raw`): over it the turn
/// is refused by default, with the prompt's length and the budget; under
/// `--prompt-budget-mode truncate` the prompt is its last tokens, BOS kept
/// on LFM2, and the prefill in tiles of 4 over the 8 kept tokens (the cut on
/// a tile boundary) decodes exactly what the kept text given as the prompt
/// decodes.
#[test]
fn prompt_budget_refuses_or_keeps_the_last_tokens() {
    let l_dir = scratch("prompt-budget-lfm2");
    let l_model = write::<Lfm2Tiny>(&l_dir);
    let (q_dir, q_model, q_base) = qwen_tiny("prompt-budget-qwen");
    let bos = Lfm2Tiny::tokenizer().id("<|startoftext|>").unwrap();
    let cases: [(&Path, Vec<&str>, Option<u32>); 2] = [
        (&l_model, vec!["--cpu"], Some(bos)),
        (&q_model, q_base.iter().map(String::as_str).collect(), None),
    ];
    for (model, base, bos) in cases {
        let run = |prompt: &str, extra: &[&str]| {
            let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
                .args(["chat", "--model"])
                .arg(model)
                .args([
                    "--raw",
                    "-p",
                    prompt,
                    "-n",
                    MAX_TOKENS,
                    "--prefill-tile",
                    "4",
                ])
                .args(&base)
                .args(extra)
                .output()
                .expect("lloom runs");
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            let fields: std::collections::BTreeMap<String, String> = stderr
                .lines()
                .find(|l| l.starts_with("[turn 1:"))
                .map(|line| {
                    line.trim_start_matches("[turn 1:")
                        .trim_end_matches(']')
                        .split_whitespace()
                        .filter_map(|kv| kv.split_once('='))
                        .map(|(k, v)| (k.to_string(), v.to_string()))
                        .collect()
                })
                .unwrap_or_default();
            (out.status.success(), stderr, fields)
        };
        let full = bos.into_iter().chain(bytes(PROMPT)).collect::<Vec<u32>>();
        let (ok, stderr, _) = run(PROMPT, &["--prompt-budget", "8"]);
        assert!(!ok, "{stderr}");
        assert!(
            stderr.contains(&format!(
                "the prompt is {} tokens, over the prompt budget of 8",
                full.len()
            )),
            "{stderr}"
        );
        let (ok, stderr, cut) = run(
            PROMPT,
            &["--prompt-budget", "8", "--prompt-budget-mode", "truncate"],
        );
        assert!(ok, "{stderr}");
        let kept_bytes = 8 - usize::from(bos.is_some());
        let tail = &PROMPT[PROMPT.len() - kept_bytes..];
        let want: Vec<u32> = bos.into_iter().chain(bytes(tail)).collect();
        assert_eq!(cut["prompt_tokens"], "8");
        assert_eq!(cut["prompt_ids_sha256"], sha(&want), "{model:?}");
        assert_eq!(cut["prompt_dropped"], (full.len() - 8).to_string());
        assert_eq!(cut["prompt_budget"], "8");
        let (ok, stderr, direct) = run(tail, &[]);
        assert!(ok, "{stderr}");
        assert_eq!(direct["prompt_ids_sha256"], cut["prompt_ids_sha256"]);
        assert_eq!(
            direct["tokens_sha256"], cut["tokens_sha256"],
            "{model:?}: the truncated prompt decodes other than the same ids given whole"
        );
        assert_eq!(direct["prompt_dropped"], "0");
    }
    std::fs::remove_dir_all(&l_dir).ok();
    std::fs::remove_dir_all(&q_dir).ok();
}

/// The prompt budget on a conversation, and what it does to the cache.
///
/// `lloom chat` (three turns on stdin, a budget the second turn's
/// conversation exceeds): the truncated turn drops the oldest turns whole,
/// is reported, and does not reuse the cache (its prompt no longer starts
/// with what the cache holds); the conversation itself was shortened, so the
/// turn after it fits without dropping anything again.
///
/// `lloom serve`: a request over `truncate_prompt_tokens` renders the kept
/// turns only — its prompt is exactly the rendering of the conversation
/// without them, which a second server answers identically — and says what
/// it dropped.
#[test]
fn prompt_budget_drops_whole_turns_and_the_cache_follows() {
    let dir = scratch("prompt-budget-conv");
    let model = write::<Lfm2Tiny>(&dir);
    let mut child = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(&model)
        .args(["--cpu", "-n", "4", "--prompt-budget", "88"])
        .args(["--prompt-budget-mode", "truncate"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("lloom runs");
    {
        let mut stdin = child.stdin.take().unwrap();
        for line in [
            "a first question here",
            "a second one, longer still",
            "third",
        ] {
            writeln!(stdin, "{line}").unwrap();
        }
    }
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let turns: Vec<std::collections::BTreeMap<String, String>> = stderr
        .lines()
        .filter(|l| l.starts_with("[turn "))
        .map(|l| {
            l.split_whitespace()
                .filter_map(|kv| kv.split_once('='))
                .map(|(k, v)| (k.to_string(), v.trim_end_matches(']').to_string()))
                .collect()
        })
        .collect();
    assert_eq!(turns.len(), 3, "{stderr}");
    let dropped: Vec<usize> = turns
        .iter()
        .map(|t| t["prompt_dropped"].parse().unwrap())
        .collect();
    let first_cut = dropped
        .iter()
        .position(|&d| d > 0)
        .unwrap_or_else(|| panic!("no turn was truncated at a budget of 88:\n{stderr}"));
    assert_eq!(first_cut, 1, "turn 1 fits and turn 2 does not:\n{stderr}");
    assert!(
        !["Extend", "Resume"].contains(&turns[1]["reuse"].as_str()),
        "the truncated turn reused a cache its prompt no longer starts with:\n{stderr}"
    );
    for t in &turns {
        assert!(
            t["prompt_tokens"].parse::<usize>().unwrap() <= 88,
            "{stderr}"
        );
    }
    assert_eq!(dropped[2], 0, "the third turn dropped again:\n{stderr}");

    let tok = Lfm2Tiny::tokenizer();
    let conv = json!([
        { "role": "user", "content": "a first question here" },
        { "role": "assistant", "content": "an answer" },
        { "role": "user", "content": PROMPT },
    ]);
    let kept = json!([{ "role": "user", "content": PROMPT }]);
    let a = Server::start(&model, &["--cpu"]);
    let (status, full) = a.call(
        "POST",
        "/v1/chat/completions",
        &json!({ "messages": conv, "max_tokens": 4 }),
    );
    assert_eq!(status, 200, "{full}");
    let (status, cut) = a.call(
        "POST",
        "/v1/chat/completions",
        &json!({ "messages": conv, "max_tokens": 4, "truncate_prompt_tokens": 60,
                 "return_token_ids": true }),
    );
    assert_eq!(status, 200, "{cut}");
    let bos = tok.id("<|startoftext|>").unwrap();
    let mut want = vec![bos];
    want.extend(encode(
        &tok,
        &format!("<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>"),
    ));
    assert!(
        want.len() <= 60,
        "the fixture's kept turn is over the budget"
    );
    assert_eq!(cut["timings"]["prompt_ids_sha256"], sha(&want));
    assert_eq!(cut["usage"]["prompt_tokens"], want.len());
    assert_eq!(cut["timings"]["prompt_messages_dropped"], 2);
    assert_eq!(
        cut["timings"]["prompt_tokens_dropped"],
        full["usage"]["prompt_tokens"].as_u64().unwrap() - want.len() as u64
    );
    assert_eq!(cut["timings"]["prompt_budget"], 60);
    let (status, err) = a.call(
        "POST",
        "/v1/chat/completions",
        &json!({ "messages": conv, "max_tokens": 4, "truncate_prompt_tokens": 5 }),
    );
    assert_eq!(status, 400, "{err}");
    drop(a);
    let b = Server::start(&model, &["--cpu"]);
    let (status, alone) = b.call(
        "POST",
        "/v1/chat/completions",
        &json!({ "messages": kept, "max_tokens": 4, "return_token_ids": true }),
    );
    assert_eq!(status, 200, "{alone}");
    assert_eq!(alone["timings"]["prompt_ids_sha256"], sha(&want));
    assert_eq!(alone["timings"]["tokens"], cut["timings"]["tokens"]);
    drop(b);
    std::fs::remove_dir_all(&dir).ok();
}

/// The tiny LFM2's rendering of one user message: its template opens the
/// thinking block on every turn.
fn lfm2_rendered() -> String {
    format!("<|startoftext|><|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>")
}

/// A template that writes a thinking block for a past turn's reasoning (so
/// the family has reasoning tags) but opens none in its generation prompt,
/// and reads no `enable_thinking`: as LFM2.5-VL-3B's does.
const NO_OPEN_BLOCK: &str = "{{- bos_token -}}{%- for m in messages -%}\
    <|im_start|>{{ m.role ~ '\\n' }}\
    {% if m.reasoning_content %}<think>{{ m.reasoning_content }}</think>{% endif %}\
    {{ m.content }}<|im_end|>{{ '\\n' }}{%- endfor -%}<|im_start|>assistant{{ '\\n' }}";

/// The directory a written tiny checkpoint reads its chat template from.
fn template_dir<'a>(model: &'a Path, dir: &'a Path) -> &'a Path {
    if model.is_dir() {
        model
    } else {
        dir
    }
}

/// `lloom chat --think` / `--no-think`.
///
/// LFM2: its template opens the block on every turn, so no flag and `--think`
/// both run its rendering, thinking; `--no-think` runs that rendering and the
/// end marker, closed by the engine, and decodes exactly what that prompt
/// given raw decodes. Qwen4-exp: its template's switch, `--no-think` being
/// the chat's default. A template that opens no block and has no switch
/// refuses `--think` and runs `--no-think` as rendered. `--raw` refuses both,
/// and a thinking budget under `--no-think` is refused with the hint.
#[test]
fn chat_switches_thinking_by_the_template_or_by_closing_the_block() {
    let dir = scratch("think-chat-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let tok = Lfm2Tiny::tokenizer();
    let on = encode(&tok, &lfm2_rendered());
    let mut off = on.clone();
    off.push(added_id(&tok, "</think>"));
    let run = |extra: &[&str]| {
        let mut a = vec!["--cpu"];
        a.extend(extra);
        chat_templated(&model, &a)
    };
    let mut no_think = None;
    for (flags, want, think, by) in [
        (&[][..], &on, "on", "template"),
        (&["--think"][..], &on, "on", "template"),
        (&["--no-think"][..], &off, "off", "engine"),
    ] {
        let (ok, stderr, f) = run(flags);
        assert!(ok, "{flags:?}:\n{stderr}");
        assert_eq!(f["prompt_ids_sha256"], sha(want), "{flags:?}: the prompt");
        assert_eq!(f["prompt_tokens"], want.len().to_string(), "{flags:?}");
        assert_eq!((f["think"].as_str(), f["think_by"].as_str()), (think, by));
        assert!(f["completion_tokens"].parse::<usize>().unwrap() > 0);
        if flags == ["--no-think"] {
            no_think = Some(f);
        }
    }
    let no_think = no_think.unwrap();
    let raw_text = format!(
        "{}</think>",
        lfm2_rendered().trim_start_matches("<|startoftext|>")
    );
    let out = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(&model)
        .args(["--cpu", "--raw", "-n", "4", "-p", &raw_text])
        .output()
        .expect("lloom runs");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let line = stderr.lines().find(|l| l.starts_with("[turn 1:")).unwrap();
    assert!(
        line.contains(&format!("prompt_ids_sha256={}", sha(&off))),
        "the raw prompt is not --no-think's: {line}"
    );
    assert!(
        line.contains(&format!("tokens_sha256={} ", no_think["tokens_sha256"])),
        "--no-think decoded other than its prompt given raw: {line}"
    );

    let (ok, stderr, _) = chat_templated(&model, &["--cpu", "--raw", "--no-think"]);
    assert!(!ok, "{stderr}");
    assert!(stderr.contains("--raw renders no template"), "{stderr}");
    let (ok, stderr, _) = run(&["--no-think", "--thinking-budget", "4"]);
    assert!(!ok, "{stderr}");
    assert!(
        stderr.contains(
            "--thinking-budget 4: lfm2: the prompt leaves no thinking block open (drop --no-think)"
        ),
        "{stderr}"
    );

    std::fs::write(
        template_dir(&model, &dir).join("chat_template.jinja"),
        NO_OPEN_BLOCK,
    )
    .unwrap();
    let plain = encode(
        &tok,
        &format!("<|startoftext|><|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n"),
    );
    let (ok, stderr, f) = run(&["--think"]);
    assert!(!ok && f.is_empty(), "{stderr}");
    assert!(
        stderr.contains("--think: lfm2: the chat template leaves no thinking block open"),
        "{stderr}"
    );
    let (ok, stderr, f) = run(&["--no-think"]);
    assert!(ok, "{stderr}");
    assert_eq!(f["prompt_ids_sha256"], sha(&plain));
    assert_eq!(
        (f["think"].as_str(), f["think_by"].as_str()),
        ("off", "template")
    );
    std::fs::remove_dir_all(&dir).ok();

    let (dir, model, base) = qwen_tiny("think-chat-qwen");
    let tok = Qwen4ExpTiny::tokenizer();
    for (flags, text, think) in [
        (&[][..], qwen_not_thinking(), "off"),
        (&["--no-think"][..], qwen_not_thinking(), "off"),
        (&["--think"][..], qwen_thinking(Some(XHIGH)), "on"),
    ] {
        let mut a: Vec<&str> = base.iter().map(String::as_str).collect();
        a.extend(flags);
        let (ok, stderr, f) = chat_templated(&model, &a);
        assert!(ok, "{flags:?}:\n{stderr}");
        assert_eq!(
            f["prompt_ids_sha256"],
            sha(&encode(&tok, &text)),
            "{flags:?}"
        );
        assert_eq!(
            (f["think"].as_str(), f["think_by"].as_str()),
            (think, "template")
        );
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// `serve`'s `enable_thinking`, top-level or in `chat_template_kwargs`.
/// LFM2: off is the rendering and the end marker, closed by the engine, and
/// `timings` says so; on and no field are the rendering; the two spellings
/// must agree and must be booleans. A thinking budget on an off prompt is
/// served without it. The same off prompt under the batch scheduler, where
/// no forced token can be fed. Qwen4-exp: the top-level field reaches its
/// template's switch. A template that opens no block: on is a 400.
#[test]
fn serve_switches_thinking_by_the_template_or_by_closing_the_block() {
    let dir = scratch("think-serve-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let tok = Lfm2Tiny::tokenizer();
    let on = encode(&tok, &lfm2_rendered());
    let mut off = on.clone();
    off.push(added_id(&tok, "</think>"));
    let server = Server::start(&model, &["--cpu"]);
    for (req, want, think, by) in [
        (json!({}), &on, "on", "template"),
        (json!({ "enable_thinking": true }), &on, "on", "template"),
        (json!({ "enable_thinking": false }), &off, "off", "engine"),
        (
            json!({ "chat_template_kwargs": { "enable_thinking": false } }),
            &off,
            "off",
            "engine",
        ),
        (
            json!({ "enable_thinking": false, "chat_template_kwargs": { "enable_thinking": false } }),
            &off,
            "off",
            "engine",
        ),
    ] {
        let (status, body) = server.chat(req.clone());
        assert_eq!(status, 200, "{req}: {body}");
        let t = &body["timings"];
        assert_eq!(t["prompt_ids_sha256"], sha(want), "{req}: the prompt");
        assert_eq!(body["usage"]["prompt_tokens"], want.len(), "{req}");
        assert_eq!(
            (t["thinking"].as_str(), t["thinking_by"].as_str()),
            (Some(think), Some(by))
        );
    }
    let (status, err) = server.chat(json!({
        "enable_thinking": true, "chat_template_kwargs": { "enable_thinking": false },
    }));
    assert_eq!(status, 400, "{err}");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("disagree"),
        "{err}"
    );
    let (status, err) = server.chat(json!({ "enable_thinking": "no" }));
    assert_eq!(status, 400, "{err}");
    assert_eq!(
        err["error"]["message"],
        "'enable_thinking' must be true or false, not \"no\""
    );
    let (status, body) =
        server.chat(json!({ "enable_thinking": false, "thinking_token_budget": 2 }));
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["timings"]["thinking_budget_ignored"],
        "the prompt leaves no thinking block open"
    );
    assert_eq!(body["timings"]["prompt_ids_sha256"], sha(&off));
    drop(server);

    // LFM2's shared pool is Metal-only; the private rows run on the CPU.
    let mut batched = vec![vec!["--cpu", "--batch", "2", "--rows", "private"]];
    if cfg!(feature = "metal") {
        batched.push(vec!["--batch", "2"]);
    }
    for args in batched {
        let server = Server::start(&model, &args);
        let (status, body) = server.chat(json!({ "enable_thinking": false }));
        assert_eq!(status, 200, "{args:?}: {body}");
        assert_eq!(body["timings"]["prompt_ids_sha256"], sha(&off), "{args:?}");
        assert_eq!(body["timings"]["thinking_by"], "engine", "{args:?}");
        drop(server);
    }

    std::fs::write(
        template_dir(&model, &dir).join("chat_template.jinja"),
        NO_OPEN_BLOCK,
    )
    .unwrap();
    let server = Server::start(&model, &["--cpu"]);
    let (status, err) = server.chat(json!({ "enable_thinking": true }));
    assert_eq!(status, 400, "{err}");
    assert_eq!(
        err["error"]["message"],
        "enable_thinking=true: lfm2: the chat template leaves no thinking block open for this \
         prompt and has no enable_thinking switch that opens one"
    );
    let (status, body) = server.chat(json!({ "enable_thinking": false }));
    assert_eq!(status, 200, "{body}");
    assert_eq!(body["timings"]["thinking"], "off");
    drop(server);
    std::fs::remove_dir_all(&dir).ok();

    let (dir, model, base) = qwen_tiny("think-serve-qwen");
    let args: Vec<&str> = base.iter().map(String::as_str).collect();
    let server = Server::start(&model, &args);
    let qtok = Qwen4ExpTiny::tokenizer();
    let (status, body) = server.chat(json!({ "enable_thinking": false }));
    assert_eq!(status, 200, "{body}");
    assert_eq!(
        body["timings"]["prompt_ids_sha256"],
        sha(&encode(&qtok, &qwen_not_thinking())),
        "the top-level field reaches the template's switch"
    );
    assert_eq!(body["timings"]["thinking_by"], "template");
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// A `/eval` stream's events (`application/x-ndjson`, chunked).
fn eval_events(server: &Server, body: &Value) -> Vec<Value> {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    let body = body.to_string();
    write!(
        s,
        "POST /eval HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut response = Vec::new();
    s.read_to_end(&mut response).unwrap();
    let response = String::from_utf8(response).unwrap();
    let (head, mut rest) = response.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("HTTP/1.1 200"), "{head}\n{rest}");
    let mut text = String::new();
    while let Some((size, after)) = rest.split_once("\r\n") {
        let n = usize::from_str_radix(size.trim(), 16).unwrap();
        if n == 0 {
            break;
        }
        text.push_str(&after[..n]);
        rest = &after[n + 2..];
    }
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

/// A streamed chat completion (`"stream": true`, SSE): the deltas, joined,
/// are the non-streamed reply, the first chunk names the role, the last one
/// carries the finish reason, and the stream ends with `[DONE]`. The deltas
/// are batched by the writer thread (`StreamWriter::start_with_deltas`), so
/// their count is not the token count; their text is the reply's.
#[test]
fn a_streamed_chat_is_the_unstreamed_reply_in_order() {
    let dir = scratch("sse-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let server = Server::start(&model, &["--cpu"]);
    let (status, plain) = server.chat(json!({ "max_tokens": 24 }));
    assert_eq!(status, 200, "{plain}");
    let want = plain["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    let req = json!({
        "messages": [{ "role": "user", "content": PROMPT }],
        "max_tokens": 24,
        "stream": true,
    });
    let mut s = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    let body = req.to_string();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut response = Vec::new();
    s.read_to_end(&mut response).unwrap();
    let response = String::from_utf8(response).unwrap();
    let (head, mut rest) = response.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("HTTP/1.1 200"), "{head}\n{rest}");
    let mut text = String::new();
    while let Some((size, after)) = rest.split_once("\r\n") {
        let n = usize::from_str_radix(size.trim(), 16).unwrap();
        if n == 0 {
            break;
        }
        text.push_str(&after[..n]);
        rest = &after[n + 2..];
    }
    let events: Vec<&str> = text
        .split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .collect();
    assert_eq!(events.last(), Some(&"[DONE]"), "{text}");
    let chunks: Vec<Value> = events[..events.len() - 1]
        .iter()
        .map(|e| serde_json::from_str(e).unwrap_or_else(|err| panic!("{e:?}: {err}")))
        .collect();
    assert_eq!(
        chunks[0]["choices"][0]["delta"]["role"], "assistant",
        "{text}"
    );
    let last = chunks.last().unwrap();
    assert!(last["choices"][0]["finish_reason"].is_string(), "{last}");
    let got: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(got, want, "the streamed deltas, joined, are the reply");
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// `/eval`'s `enable_thinking` on LFM2: `false` runs the rendering closed by
/// the end marker, the prompt it reports says so, and a sample decodes; a
/// value that is not a boolean is a 400.
#[test]
fn eval_switches_thinking_off_on_lfm2() {
    let dir = scratch("think-eval-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let server = Server::start(&model, &["--cpu"]);
    let sample = json!([{ "id": "a", "messages": [{ "role": "user", "content": PROMPT }] }]);
    for (want, ends) in [(Some(false), "<think></think>"), (None, "<think>")] {
        let mut req = json!({ "samples": sample, "max_tokens": 4 });
        if let Some(w) = want {
            req["enable_thinking"] = json!(w);
        }
        let events = eval_events(&server, &req);
        let start = events
            .iter()
            .find(|e| e["kind"] == "eval_sample_start")
            .unwrap();
        let prompt = start["prompt"].as_str().unwrap();
        assert!(
            prompt.ends_with(&format!("<|im_start|>assistant\n{ends}")),
            "{want:?}: {prompt:?}"
        );
        let done = events
            .iter()
            .find(|e| e["kind"] == "eval_sample_done")
            .unwrap();
        assert!(
            done["completion_tokens"].as_u64().unwrap() > 0,
            "{events:?}"
        );
        assert!(
            events.iter().any(|e| e["kind"] == "eval_done"),
            "{events:?}"
        );
    }
    let (status, err) = server.call(
        "POST",
        "/eval",
        &json!({ "samples": sample, "max_tokens": 4, "enable_thinking": 1 }),
    );
    assert_eq!(status, 400, "{err}");
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// `serve` folds OpenAI's top-level `tools` into the template's `tools`: the
/// prompt is the template's rendering with the list (computed here through
/// the same template), whichever spelling carries it, and differs from the
/// prompt without it; the two spellings must agree.
#[test]
fn serve_folds_the_top_level_tools_field() {
    let dir = scratch("tools-serve-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let tok = Lfm2Tiny::tokenizer();
    let tools = json!([{ "type": "function", "function": {
        "name": "get_weather", "description": "The weather in a city.",
        "parameters": { "type": "object", "properties": { "city": { "type": "string" } } },
    } }]);
    let messages = json!([{ "role": "user", "content": PROMPT }]);
    let mut kw = serde_json::Map::new();
    kw.insert("tools".into(), tools.clone());
    let text = lloom_engine::app::template::Template::compile(tok.chat_template)
        .unwrap()
        .render(&messages, "<|startoftext|>", &kw)
        .unwrap();
    assert!(text.contains("get_weather"), "{text}");
    let want = sha(&encode(&tok, &text));
    let server = Server::start(&model, &["--cpu"]);
    let (_, plain) = server.chat(json!({}));
    for req in [
        json!({ "tools": tools }),
        json!({ "chat_template_kwargs": { "tools": tools } }),
        json!({ "tools": tools, "chat_template_kwargs": { "tools": tools } }),
    ] {
        let (status, body) = server.chat(req.clone());
        assert_eq!(status, 200, "{req}: {body}");
        assert_eq!(body["timings"]["prompt_ids_sha256"], want, "{req}");
        assert_ne!(
            body["timings"]["prompt_ids_sha256"],
            plain["timings"]["prompt_ids_sha256"]
        );
    }
    let (status, err) =
        server.chat(json!({ "tools": tools, "chat_template_kwargs": { "tools": [] } }));
    assert_eq!(status, 400, "{err}");
    assert!(
        err["error"]["message"]
            .as_str()
            .unwrap()
            .contains("disagree"),
        "{err}"
    );
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// A template failing on the request's data outside `raise_exception` is the
/// client's 400, with the template's error: a message content the tiny
/// templates cannot iterate (a number), on both families.
#[test]
fn serve_answers_a_template_failure_on_client_data_with_a_400() {
    let dir = scratch("render-400-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let bad = json!({ "messages": [{ "role": "user", "content": 5 }], "max_tokens": 2 });
    let server = Server::start(&model, &["--cpu"]);
    let (status, err) = server.call("POST", "/v1/chat/completions", &bad);
    assert_eq!(status, 400, "{err}");
    let m = err["error"]["message"].as_str().unwrap();
    assert!(
        m.starts_with("the chat template cannot render the request: "),
        "{m}"
    );
    let (status, ok) = server.chat(json!({}));
    assert_eq!(status, 200, "the server serves after it: {ok}");
    drop(server);
    std::fs::remove_dir_all(&dir).ok();

    let (dir, model, base) = qwen_tiny("render-400-qwen");
    let args: Vec<&str> = base.iter().map(String::as_str).collect();
    let server = Server::start(&model, &args);
    let (status, err) = server.call("POST", "/v1/chat/completions", &bad);
    assert_eq!(status, 400, "{err}");
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// A `/v1/systemone` body the tiny byte-level tokenizer can answer: one
/// choice between `A` and `B` (single byte tokens; `yes`/`no` are not).
fn tiny_decision() -> Value {
    json!({
        "state": "The loom threads warp and weft.",
        "questions": {"pick": {"type": "choice", "instructions": "Which one?",
                               "criteria": {"A": "the first", "B": "the second"}}}
    })
}

fn assert_tiny_answer(response: &Value) {
    let answer = &response["answers"]["pick"];
    let probabilities = answer["probabilities"].as_object().unwrap();
    let total: f64 = probabilities.values().map(|p| p.as_f64().unwrap()).sum();
    assert!((total - 1.0).abs() < 1e-9, "{response}");
    assert!(
        matches!(answer["choice"].as_str(), Some("A" | "B")),
        "{response}"
    );
    assert_eq!(response["usage"]["output_tokens"], 0, "{response}");
}

/// `lloom chat -p "/decide {…}"` on a plain tiny LFM2 with the shipped d1
/// template selected (`decision.template = lfm2-d1`, installable on the
/// default family now that it has an executor): one answer on stdout, and
/// the process ends: a command line given as `--prompt` is read once, not
/// again for a turn it never ran.
#[test]
fn lfm2_tiny_chat_answers_one_decide_line_and_exits() {
    let dir = scratch("decide-chat-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let line = format!("/decide {}", tiny_decision());
    let mut child = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["chat", "--model"])
        .arg(&model)
        .args(["--cpu", "--decision-template", "lfm2-d1", "-p", &line])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("lloom runs");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if std::time::Instant::now() > deadline {
            child.kill().ok();
            child.wait().ok();
            panic!("`lloom chat -p \"/decide …\"` did not exit: the prompt was read again");
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    let mut stdout = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    let mut stderr = String::new();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert!(status.success(), "exited {status:?}:\n{stderr}");
    assert!(
        stderr.contains("[decision template: lfm2-d1 (engine-shipped, sha256 b47201d7"),
        "{stderr}"
    );
    let lines: Vec<&str> = stdout.lines().filter(|l| l.starts_with('{')).collect();
    assert_eq!(lines.len(), 1, "one answer, once:\n{stdout}");
    assert_tiny_answer(&serde_json::from_str(lines[0]).unwrap());
    std::fs::remove_dir_all(&dir).ok();
}

/// `lloom serve` on the same plain tiny LFM2 with the d1 template selected:
/// a decision, a chat, the decision again on the one loaded model (equal,
/// on the CPU's deterministic arithmetic); the response names the model
/// first, and `/v1/models` does not advertise decisions for a selected
/// (not native) template.
#[test]
fn lfm2_tiny_serves_decisions_and_chat_on_one_model() {
    let dir = scratch("decide-serve-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let server = Server::start(&model, &["--cpu", "--decision-template", "lfm2-d1"]);
    let (status, first) = server.call("POST", "/v1/systemone", &tiny_decision());
    assert_eq!(status, 200, "{first}");
    assert_tiny_answer(&first);
    assert!(first["model"].is_string(), "{first}");
    let (status, body) = server.chat(json!({}));
    assert_eq!(status, 200, "{body}");
    let (status, again) = server.call("POST", "/v1/systemone", &tiny_decision());
    assert_eq!(status, 200, "{again}");
    assert_eq!(
        first, again,
        "a chat between two decisions moved the second"
    );
    let (status, models) = server.call("GET", "/v1/models", &json!({}));
    assert_eq!(status, 200, "{models}");
    assert_eq!(
        models["data"][0]["architecture"]["output_modalities"],
        json!(["text"])
    );
    // A text checkpoint builds no tower: no image input is listed.
    assert_eq!(
        models["data"][0]["architecture"]["input_modalities"],
        json!(["text"])
    );
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// A decision over its own state (`n` picks the wording).
fn tiny_decision_over(n: usize) -> Value {
    let mut body = tiny_decision();
    body["state"] = json!(format!(
        "Thread {n}: the loom threads warp and weft, row {n}."
    ));
    body
}

/// `lloom serve` on the tiny LFM2 with eight requests started at once — six
/// decisions over different states and two chats, three bursts — answers
/// every one: requests that find the model busy wait their turn rather than
/// being refused. Each decision equals its answer alone: bit for bit under
/// `deterministic`, within F32 tolerance under `fast`, where decisions that
/// waited together are read as one tree.
#[test]
fn lfm2_tiny_serve_queues_concurrent_requests_and_answers_each_as_alone() {
    let dir = scratch("decide-concurrent-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    for profile in ["deterministic", "fast"] {
        let server = Server::start(
            &model,
            &[
                "--cpu",
                "--decision-template",
                "lfm2-d1",
                "--profile",
                profile,
            ],
        );
        let alone: Vec<Value> = (0..6)
            .map(|n| {
                let (status, body) = server.call("POST", "/v1/systemone", &tiny_decision_over(n));
                assert_eq!(status, 200, "{body}");
                body
            })
            .collect();
        for burst in 0..3 {
            let gate = std::sync::Barrier::new(8);
            let replies: Vec<(usize, u16, Value)> = std::thread::scope(|scope| {
                let handles: Vec<_> = (0..8)
                    .map(|i| {
                        let (server, gate) = (&server, &gate);
                        scope.spawn(move || {
                            gate.wait();
                            let (status, body) = if i < 6 {
                                server.call("POST", "/v1/systemone", &tiny_decision_over(i))
                            } else {
                                server.chat(json!({"max_tokens": 16}))
                            };
                            (i, status, body)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for (i, status, body) in replies {
                assert_eq!(status, 200, "{profile} burst {burst} request {i}: {body}");
                if i >= 6 {
                    continue;
                }
                let (got, want) = (&body["answers"]["pick"], &alone[i]["answers"]["pick"]);
                assert_eq!(body["usage"], alone[i]["usage"], "{profile} request {i}");
                assert_eq!(got["choice"], want["choice"], "{profile} request {i}");
                if profile == "deterministic" {
                    assert_eq!(body, alone[i], "{profile} burst {burst} request {i} moved");
                } else {
                    for label in ["A", "B"] {
                        let (g, w) = (
                            got["probabilities"][label].as_f64().unwrap(),
                            want["probabilities"][label].as_f64().unwrap(),
                        );
                        assert!(
                            (g - w).abs() < 1e-5,
                            "{profile} request {i} {label}: {g} != {w}"
                        );
                    }
                }
            }
        }
        drop(server);
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// The process's front-end counters (`/health`): template renders and
/// tokenization passes so far.
fn passes(server: &Server) -> (u64, u64) {
    let (status, h) = server.call("GET", "/health", &json!({}));
    assert_eq!(status, 200, "{h}");
    (
        h["template_renders"].as_u64().unwrap(),
        h["tokenizations"].as_u64().unwrap(),
    )
}

/// The request front end through `lloom serve`, on the tiny Qwen4-exp with
/// thinking off (its template keeps the generation prompt in the history):
/// each chat request renders the template once and tokenizes the result
/// once, whatever its history; the second turn carries the first reply's
/// sampled ids, so the session extends — it reuses the first prompt and the
/// whole reply.
#[test]
fn serve_renders_once_and_the_next_turn_extends_the_sampled_reply() {
    let (dir, model, base) = qwen_tiny("front-once");
    let mut args: Vec<&str> = base.iter().map(String::as_str).collect();
    args.extend(["--prefix-reuse", "true"]);
    let server = Server::start(&model, &args);
    // `a` every step: a reply the template renders back verbatim.
    let common = json!({ "enable_thinking": false, "temperature": 0, "max_tokens": 6,
                         "logit_bias": { "97": 100 } });
    let with = |messages: Value| {
        let mut r = common.clone();
        r["messages"] = messages;
        r
    };
    let before = passes(&server);
    let (status, one) = server.chat(with(json!([{ "role": "user", "content": PROMPT }])));
    assert_eq!(status, 200, "{one}");
    let after = passes(&server);
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (1, 1),
        "turn 1: {one}"
    );
    let reply = one["choices"][0]["message"]["content"].as_str().unwrap();
    assert_eq!(reply, "aaaaaa", "the bias holds");
    let p1 = one["usage"]["prompt_tokens"].as_u64().unwrap();
    let r1 = one["usage"]["completion_tokens"].as_u64().unwrap();

    let before = passes(&server);
    let (status, two) = server.chat(with(json!([
        { "role": "user", "content": PROMPT },
        { "role": "assistant", "content": reply },
        { "role": "user", "content": "And Italy?" },
    ])));
    assert_eq!(status, 200, "{two}");
    let after = passes(&server);
    assert_eq!(
        (after.0 - before.0, after.1 - before.1),
        (1, 1),
        "turn 2: {two}"
    );
    let t = &two["timings"];
    assert_eq!(t["renders"], 1, "{t}");
    assert_eq!(t["history_matched"], 1, "{t}");
    assert_eq!(t["history_spliced_tokens"], r1, "{t}");
    assert_eq!(t["reuse"], "Extend", "{t}");
    assert_eq!(
        t["reused"],
        p1 + r1,
        "the first prompt and the whole reply: {t}"
    );
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// A reply of ours is the logged one only in the conversation it answered,
/// through `lloom serve`, streamed or not: two conversations get the same
/// text back (the bias), and each continuation matches its own reply; a
/// history that never got that text from us carries it unverified and
/// renders as sent.
#[test]
fn serve_takes_a_logged_reply_only_in_the_conversation_it_answered() {
    let (dir, model, base) = qwen_tiny("front-scope");
    let mut args: Vec<&str> = base.iter().map(String::as_str).collect();
    args.extend(["--prefix-reuse", "true"]);
    let server = Server::start(&model, &args);
    let common = json!({ "enable_thinking": false, "temperature": 0, "max_tokens": 6,
                         "logit_bias": { "97": 100 } });
    let with = |messages: Value, stream: bool| {
        let mut r = common.clone();
        r["messages"] = messages;
        r["stream"] = json!(stream);
        r
    };
    let mut replies = Vec::new();
    for (question, stream) in [(PROMPT, false), ("And the other one?", true)] {
        let body = with(json!([{ "role": "user", "content": question }]), stream);
        if stream {
            let events = sse(&server, &body);
            let text: String = events
                .iter()
                .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
                .collect();
            replies.push((question, text));
        } else {
            let (status, one) = server.call("POST", "/v1/chat/completions", &body);
            assert_eq!(status, 200, "{one}");
            let text = one["choices"][0]["message"]["content"].as_str().unwrap();
            replies.push((question, text.to_string()));
        }
    }
    assert_eq!(replies[0].1, "aaaaaa", "the bias holds");
    assert_eq!(replies[0].1, replies[1].1, "one text, two conversations");
    for (question, reply) in &replies {
        let (status, two) = server.call(
            "POST",
            "/v1/chat/completions",
            &with(
                json!([
                    { "role": "user", "content": question },
                    { "role": "assistant", "content": reply },
                    { "role": "user", "content": "Again?" },
                ]),
                false,
            ),
        );
        assert_eq!(status, 200, "{two}");
        let t = &two["timings"];
        assert_eq!(
            (&t["history_matched"], &t["history_unverified"]),
            (&json!(1), &json!(0)),
            "{question}: {t}"
        );
    }
    let (status, foreign) = server.call(
        "POST",
        "/v1/chat/completions",
        &with(
            json!([
                { "role": "user", "content": "Never asked." },
                { "role": "assistant", "content": replies[0].1 },
                { "role": "user", "content": "Again?" },
            ]),
            false,
        ),
    );
    assert_eq!(status, 200, "{foreign}");
    let t = &foreign["timings"];
    assert_eq!(
        (
            &t["history_matched"],
            &t["history_unverified"],
            &t["history_spliced_tokens"]
        ),
        (&json!(0), &json!(1), &json!(0)),
        "{t}"
    );
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// The external contract (decision 33, #1008): a response is text unless
/// the request asks for ids (`return_token_ids`) or logprobs (`logprobs`,
/// bounded at `MAX_TOP_LOGPROBS` candidates). The ids returned are the
/// session's own: the next turn's prompt carries them where the reply sits.
/// `/health` says whether logprobs are served, and counts every turn's
/// reuse by outcome; a batched server refuses per-token logprobs by name.
#[test]
fn serve_returns_text_unless_ids_or_logprobs_are_asked() {
    let (dir, model, base) = qwen_tiny("external-contract");
    let mut args: Vec<&str> = base.iter().map(String::as_str).collect();
    args.extend(["--prefix-reuse", "true"]);
    let server = Server::start(&model, &args);
    let common = json!({ "enable_thinking": false, "temperature": 0, "max_tokens": 6,
                         "logit_bias": { "97": 100 } });
    let with = |messages: Value, extra: Value| {
        let mut r = common.clone();
        r["messages"] = messages;
        for (k, v) in extra.as_object().unwrap() {
            r[k] = v.clone();
        }
        r
    };
    let first = json!([{ "role": "user", "content": PROMPT }]);
    let (status, plain) = server.call(
        "POST",
        "/v1/chat/completions",
        &with(first.clone(), json!({})),
    );
    assert_eq!(status, 200, "{plain}");
    let choice = &plain["choices"][0];
    assert!(choice["message"]["content"].is_string(), "{plain}");
    for absent in [
        &plain["prompt_token_ids"],
        &plain["prompt_sampled_logprobs"],
        &choice["token_ids"],
        &choice["token_logprobs"],
        &choice["logprobs"],
        &plain["timings"]["tokens"],
    ] {
        assert!(absent.is_null(), "text only, by default: {plain}");
    }

    let (status, one) = server.call(
        "POST",
        "/v1/chat/completions",
        &with(first, json!({ "return_token_ids": true })),
    );
    assert_eq!(status, 200, "{one}");
    let ids = one["choices"][0]["token_ids"].as_array().unwrap().clone();
    assert_eq!(one["timings"]["tokens"], json!(ids));
    assert!(
        one["choices"][0]["token_logprobs"].is_null(),
        "logprobs not asked: {one}"
    );
    let p1 = one["prompt_token_ids"].as_array().unwrap().len();
    let reply = one["choices"][0]["message"]["content"].as_str().unwrap();
    let (status, two) = server.call(
        "POST",
        "/v1/chat/completions",
        &with(
            json!([
                { "role": "user", "content": PROMPT },
                { "role": "assistant", "content": reply },
                { "role": "user", "content": "Again?" },
            ]),
            json!({ "return_token_ids": true }),
        ),
    );
    assert_eq!(status, 200, "{two}");
    let prompt2 = two["prompt_token_ids"].as_array().unwrap();
    assert_eq!(
        &prompt2[p1..p1 + ids.len()],
        &ids[..],
        "the session's ids: {two}"
    );
    let t = &two["timings"]["turn"];
    assert_eq!(t["reuse_policy"], "on", "{t}");
    assert_eq!(t["outcome"], "extend", "{t}");
    assert_eq!(t["boundary"], json!(p1 + ids.len()), "{t}");
    assert_eq!(t["reused"], t["boundary"], "{t}");

    let (status, lp) = server.call(
        "POST",
        "/v1/chat/completions",
        &with(
            json!([{ "role": "user", "content": "Other?" }]),
            json!({ "logprobs": true, "top_logprobs": 3 }),
        ),
    );
    assert_eq!(status, 200, "{lp}");
    let content = lp["choices"][0]["logprobs"]["content"].as_array().unwrap();
    assert_eq!(
        content.len(),
        lp["usage"]["completion_tokens"].as_u64().unwrap() as usize
    );
    for c in content {
        assert_eq!(c["top_logprobs"].as_array().unwrap().len(), 3, "{c}");
    }
    assert!(
        lp["choices"][0]["token_ids"].is_null(),
        "ids not asked: {lp}"
    );
    for (extra, says) in [
        (
            json!({ "logprobs": true, "top_logprobs": 65 }),
            "from 0 to 64",
        ),
        (json!({ "top_logprobs": 2 }), "set logprobs to true"),
    ] {
        let (status, err) = server.call(
            "POST",
            "/v1/chat/completions",
            &with(json!([{ "role": "user", "content": "Other?" }]), extra),
        );
        assert_eq!(status, 400, "{err}");
        assert!(
            err["error"]["message"].as_str().unwrap().contains(says),
            "{err}"
        );
    }

    let (status, h) = server.call("GET", "/health", &json!({}));
    assert_eq!(status, 200, "{h}");
    assert_eq!(h["logprobs"]["served"], true, "{h}");
    assert_eq!(h["logprobs"]["top_max"], 64, "{h}");
    let reuse = &h["reuse"];
    assert_eq!(reuse["reuse_policy"], "on", "{reuse}");
    assert_eq!(reuse["turns"], 4, "four generating requests: {reuse}");
    assert_eq!(reuse["outcome"]["extend"], 1, "{reuse}");
    let counted: u64 = reuse["outcome"]
        .as_object()
        .unwrap()
        .values()
        .map(|n| n.as_u64().unwrap())
        .sum();
    assert_eq!(counted, 4, "{reuse}");
    assert_eq!(
        reuse["last"], lp["timings"]["turn"],
        "the last turn's own report"
    );
    drop(server);

    // Reuse off (serve's default): every turn cold by policy.
    let server = Server::start(&model, &base.iter().map(String::as_str).collect::<Vec<_>>());
    let (status, cold) = server.call(
        "POST",
        "/v1/chat/completions",
        &with(json!([{ "role": "user", "content": PROMPT }]), json!({})),
    );
    assert_eq!(status, 200, "{cold}");
    let t = &cold["timings"]["turn"];
    assert_eq!(
        (&t["reuse_policy"], &t["reason"]),
        (&json!("off"), &json!("policy-off")),
        "{t}"
    );
    drop(server);

    // A batched server refuses per-token logprobs by name, and says so.
    let mut batched: Vec<&str> = base.iter().map(String::as_str).collect();
    batched.extend(["--batch", "2"]);
    let server = Server::start(&model, &batched);
    let (status, h) = server.call("GET", "/health", &json!({}));
    assert_eq!(status, 200, "{h}");
    assert_eq!(h["logprobs"]["served"], false, "{h}");
    let why = h["logprobs"]["refused"].as_str().unwrap().to_string();
    assert!(why.contains("the batch scheduler"), "{why}");
    let (status, err) = server.call(
        "POST",
        "/v1/chat/completions",
        &with(
            json!([{ "role": "user", "content": PROMPT }]),
            json!({ "logprobs": true }),
        ),
    );
    assert_eq!(status, 400, "{err}");
    assert!(
        err["error"]["message"].as_str().unwrap().contains(&why),
        "{err}"
    );
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// A stop string never reaches a streaming client: the streamed deltas,
/// joined, are the unstreamed reply cut before the stop, and carry no part
/// of it (`TextStream::publish` holds what may begin one).
#[test]
fn a_streamed_reply_never_carries_its_stop_string() {
    let (dir, model, base) = qwen_tiny("stream-stop");
    let server = Server::start(&model, &base.iter().map(String::as_str).collect::<Vec<_>>());
    let req = |stop: Option<&str>, stream: bool| {
        json!({ "messages": [{ "role": "user", "content": PROMPT }], "temperature": 0,
                "max_tokens": 16, "ignore_eos": true, "enable_thinking": false,
                "stop": stop, "stream": stream })
    };
    let (status, full) = server.call("POST", "/v1/chat/completions", &req(None, false));
    assert_eq!(status, 200, "{full}");
    let text = full["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    let chars: Vec<char> = text.chars().collect();
    assert!(chars.len() >= 6, "a reply long enough to cut: {text:?}");
    let stop: String = chars[3..5].iter().collect();
    let cut = &text[..text.find(&stop).unwrap()];
    let (status, unstreamed) =
        server.call("POST", "/v1/chat/completions", &req(Some(&stop), false));
    assert_eq!(status, 200, "{unstreamed}");
    assert_eq!(
        unstreamed["choices"][0]["message"]["content"], cut,
        "{unstreamed}"
    );
    let events = sse(&server, &req(Some(&stop), true));
    let streamed: String = events
        .iter()
        .filter_map(|e| e["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(streamed, cut, "stop {stop:?} in {text:?}");
    assert!(!streamed.contains(&stop));
    let last = events.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop", "{last}");
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// A streamed request whose client goes away publishes nothing into the
/// next request: the next reply is a fresh server's, id for id.
#[test]
fn a_cancelled_stream_leaves_nothing_in_the_next_reply() {
    let (dir, model, base) = qwen_tiny("stream-cancel");
    let args: Vec<&str> = base.iter().map(String::as_str).collect();
    let next = json!({ "messages": [{ "role": "user", "content": "Something else?" }],
                       "temperature": 0, "max_tokens": 8, "enable_thinking": false,
                       "return_token_ids": true });
    let reference = {
        let fresh = Server::start(&model, &args);
        let (status, r) = fresh.call("POST", "/v1/chat/completions", &next);
        assert_eq!(status, 200, "{r}");
        r
    };
    let server = Server::start(&model, &args);
    {
        let mut s = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
        let body = json!({ "messages": [{ "role": "user", "content": PROMPT }], "temperature": 0,
                           "max_tokens": 4000, "ignore_eos": true, "stream": true,
                           "enable_thinking": false })
        .to_string();
        write!(
            s,
            "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
        let mut first = [0u8; 256];
        let n = s.read(&mut first).unwrap();
        assert!(n > 0);
        // Gone, mid-stream.
    }
    let (status, after) = server.call("POST", "/v1/chat/completions", &next);
    assert_eq!(status, 200, "{after}");
    assert_eq!(
        after["choices"][0]["token_ids"], reference["choices"][0]["token_ids"],
        "{after}"
    );
    assert_eq!(
        after["choices"][0]["message"]["content"],
        reference["choices"][0]["message"]["content"]
    );
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// The same session continued after each way a reply stops, through the
/// front end (`--prefix-reuse true`): the history carries the text the
/// client got, and the turn report says what the session did with it.
/// A length stop extends. A stop string leaves in the log every id the
/// loop produced (the stop's included), so the next turn leaves the log
/// where the delivered text ends and goes back to there (the window or a
/// turn checkpoint) — what decision 42 prices.
#[test]
fn the_next_turn_continues_what_each_stop_left() {
    let (dir, model, base) = qwen_tiny("stop-causes");
    let mut args: Vec<&str> = base.iter().map(String::as_str).collect();
    args.extend(["--prefix-reuse", "true"]);
    let server = Server::start(&model, &args);
    let turn = |messages: Value, extra: Value| {
        // Letters only (the tiny model's own replies are lone bytes no
        // whole-character cut can end on): `a`, `b` and `c` biased alike,
        // the model's logits choosing among them.
        let mut r = json!({ "messages": messages, "temperature": 0, "enable_thinking": false,
                            "return_token_ids": true, "ignore_eos": true,
                            "logit_bias": { "97": 100, "98": 100, "99": 100 } });
        for (k, v) in extra.as_object().unwrap() {
            r[k] = v.clone();
        }
        let (status, body) = server.call("POST", "/v1/chat/completions", &r);
        assert_eq!(status, 200, "{body}");
        body
    };
    let probe = turn(
        json!([{ "role": "user", "content": "Probe." }]),
        json!({ "max_tokens": 16 }),
    );
    let text = probe["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    // A stop that first occurs a few letters in, so the delivered text is
    // not empty.
    let stop = (2..=6)
        .flat_map(|w| (3..text.len().saturating_sub(w)).map(move |i| (i, w)))
        .map(|(i, w)| (i, text[i..i + w].to_string()))
        .find(|(i, s)| text.find(s.as_str()) == Some(*i))
        .map(|(_, s)| s)
        .unwrap_or_else(|| panic!("no stop first occurring past the third letter: {text:?}"));
    for (cause, extra) in [
        ("length", json!({ "max_tokens": 6 })),
        ("stop", json!({ "max_tokens": 16, "stop": stop })),
    ] {
        let q = json!([{ "role": "user", "content": "Probe." }]);
        let one = turn(q.clone(), extra);
        let delivered = one["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_string();
        let p1 = one["prompt_token_ids"].as_array().unwrap().len();
        let produced = one["choices"][0]["token_ids"].as_array().unwrap().len();
        let two = turn(
            json!([q[0], { "role": "assistant", "content": delivered },
                   { "role": "user", "content": "Again." }]),
            json!({ "max_tokens": 4 }),
        );
        let t = &two["timings"]["turn"];
        eprintln!("STOP-CAUSE {cause}: delivered {delivered:?} produced {produced} turn {t}");
        assert_eq!(
            two["timings"]["history_matched"], 1,
            "{cause}: the delivered reply is ours: {two}"
        );
        assert_eq!(
            t["boundary"],
            json!(p1 + produced),
            "{cause}: the log is every id produced: {t}"
        );
        match cause {
            "length" => assert_eq!(t["outcome"], "extend", "{t}"),
            _ => {
                assert!(
                    matches!(t["outcome"].as_str(), Some("rewind" | "restore")),
                    "{t}"
                );
                assert_eq!(t["reason"], "diverged", "{t}");
                let at = t["diverged_at"].as_u64().unwrap() as usize;
                assert!(at >= p1 && at < p1 + produced, "inside the reply: {t}");
            }
        }
    }
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// A shutdown asked for while a request decodes: the process exits cleanly
/// once the request's thread is done with the model, and the streaming
/// client gets a complete reply or a cut connection, never a panic.
#[test]
fn shutdown_waits_for_the_request_that_holds_the_model() {
    let (dir, model, base) = qwen_tiny("shutdown-busy");
    let mut child = Command::new(env!("CARGO_BIN_EXE_lloom"))
        .args(["serve", "--model"])
        .arg(&model)
        .args(&base)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("lloom serve runs");
    let mut line = String::new();
    std::io::BufReader::new(child.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let ready: Value = serde_json::from_str(&line).unwrap();
    let port = ready["port"].as_u64().unwrap() as u16;
    let mut s = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    let body = json!({ "messages": [{ "role": "user", "content": PROMPT }], "temperature": 0,
                       "max_tokens": 3000, "ignore_eos": true, "stream": true,
                       "enable_thinking": false })
    .to_string();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut first = [0u8; 256];
    assert!(s.read(&mut first).unwrap() > 0, "the stream started");
    let mut c = std::net::TcpStream::connect(("127.0.0.1", port)).unwrap();
    write!(
        c,
        "POST /shutdown HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{{}}"
    )
    .unwrap();
    let mut rest = Vec::new();
    let _ = s.read_to_end(&mut rest);
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
    assert!(stderr.contains("[shutdown requested]"), "{stderr}");
    std::fs::remove_dir_all(&dir).ok();
}

/// The SSE events of a streamed chat request, `[DONE]` excluded.
fn sse(server: &Server, body: &Value) -> Vec<Value> {
    let mut s = std::net::TcpStream::connect(("127.0.0.1", server.port)).unwrap();
    let body = body.to_string();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut response = String::new();
    s.read_to_string(&mut response).unwrap();
    let (head, mut rest) = response.split_once("\r\n\r\n").unwrap();
    assert!(head.starts_with("HTTP/1.1 200"), "{head}\n{rest}");
    let mut text = String::new();
    while let Some((size, after)) = rest.split_once("\r\n") {
        let n = usize::from_str_radix(size.trim(), 16).unwrap();
        if n == 0 {
            break;
        }
        text.push_str(&after[..n]);
        rest = &after[n + 2..];
    }
    text.split("\n\n")
        .filter_map(|e| e.strip_prefix("data: "))
        .filter(|e| *e != "[DONE]")
        .map(|e| serde_json::from_str(e).unwrap_or_else(|err| panic!("{e:?}: {err}")))
        .collect()
}

/// The token-level API: ids in (`prompt_token_ids`, on both routes) are the
/// ids the engine ran, and the ids out (`return_token_ids`) are the run's,
/// with their log-probabilities (`logprobs`); greedy, each is its row's
/// maximum. With `session.keep_logprobs = sampled`, the next turn's prompt
/// carries the first reply's ids and the logprobs they were sampled at.
#[test]
fn the_token_api_runs_the_ids_sent_and_returns_the_ids_and_logprobs_run() {
    let (dir, model, base) = qwen_tiny("token-api");
    let mut args: Vec<&str> = base.iter().map(String::as_str).collect();
    args.extend(["--keep-logprobs", "sampled"]);
    let server = Server::start(&model, &args);
    let tok = Qwen4ExpTiny::tokenizer();
    let ids = encode(&tok, &qwen_not_thinking());
    let (status, c) = server.call(
        "POST",
        "/v1/completions",
        &json!({ "prompt_token_ids": ids, "max_tokens": 5, "temperature": 0,
                 "logprobs": 2, "return_token_ids": true, "ignore_eos": true }),
    );
    assert_eq!(status, 200, "{c}");
    assert_eq!(c["timings"]["prompt_ids_sha256"], sha(&ids), "{c}");
    assert_eq!(c["prompt_token_ids"], json!(ids));
    let out = &c["choices"][0];
    assert_eq!(out["token_ids"], c["timings"]["tokens"]);
    let n = out["token_ids"].as_array().unwrap().len();
    assert_eq!(n, 5);
    let lp = &out["logprobs"];
    let sampled = lp["token_logprobs"].as_array().unwrap();
    assert_eq!(sampled.len(), n);
    for (i, top) in lp["top_ids"].as_array().unwrap().iter().enumerate() {
        let top = top.as_array().unwrap();
        assert_eq!(top.len(), 2, "{lp}");
        let (first, second) = (top[0][1].as_f64().unwrap(), top[1][1].as_f64().unwrap());
        assert!(first >= second, "most likely first: {lp}");
        let s = sampled[i].as_f64().unwrap();
        assert!(s <= 0.0, "{lp}");
        assert_eq!(
            top[0][0], out["token_ids"][i],
            "greedy draws the row's maximum: {lp}"
        );
        assert!(
            (s - first).abs() < 1e-6,
            "position {i}: {s} vs {first}: {lp}"
        );
    }

    let (status, chat) = server.call(
        "POST",
        "/v1/chat/completions",
        &json!({ "prompt_token_ids": ids, "max_tokens": 3, "temperature": 0, "ignore_eos": true }),
    );
    assert_eq!(status, 200, "{chat}");
    assert_eq!(chat["timings"]["prompt_ids_sha256"], sha(&ids));

    let first = json!([{ "role": "user", "content": PROMPT }]);
    // The session keeps logprobs, but a request that asks only for ids gets
    // none of them (decision 33: each is asked for).
    let (status, ids_only) = server.call(
        "POST",
        "/v1/chat/completions",
        &json!({ "messages": [{ "role": "user", "content": "Just ids?" }],
                 "enable_thinking": false, "temperature": 0, "max_tokens": 3,
                 "return_token_ids": true }),
    );
    assert_eq!(status, 200, "{ids_only}");
    assert!(ids_only["choices"][0]["token_ids"].is_array(), "{ids_only}");
    assert!(
        ids_only["choices"][0]["token_logprobs"].is_null(),
        "{ids_only}"
    );
    assert!(ids_only["prompt_sampled_logprobs"].is_null(), "{ids_only}");
    let req = |messages: Value| {
        json!({ "messages": messages, "enable_thinking": false, "temperature": 0,
                "max_tokens": 4, "logit_bias": { "98": 100 }, "return_token_ids": true,
                "logprobs": true })
    };
    let (status, one) = server.call("POST", "/v1/chat/completions", &req(first));
    assert_eq!(status, 200, "{one}");
    let reply = one["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_string();
    let reply_ids = one["choices"][0]["token_ids"].as_array().unwrap().clone();
    let reply_lp = one["choices"][0]["token_logprobs"]
        .as_array()
        .unwrap()
        .clone();
    assert_eq!(
        reply_lp.len(),
        reply_ids.len(),
        "the session's capture: {one}"
    );
    let p1 = one["prompt_token_ids"].as_array().unwrap().len();
    let (status, two) = server.call(
        "POST",
        "/v1/chat/completions",
        &req(json!([
            { "role": "user", "content": PROMPT },
            { "role": "assistant", "content": reply },
            { "role": "user", "content": "And Italy?" },
        ])),
    );
    assert_eq!(status, 200, "{two}");
    let ids2 = two["prompt_token_ids"].as_array().unwrap();
    let lp2 = two["prompt_sampled_logprobs"].as_array().unwrap();
    assert_eq!(lp2.len(), ids2.len());
    let span = p1..p1 + reply_ids.len();
    assert_eq!(&ids2[span.clone()], &reply_ids[..], "{two}");
    assert_eq!(&lp2[span.clone()], &reply_lp[..], "{two}");
    assert!(lp2[..p1].iter().all(Value::is_null));
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}
