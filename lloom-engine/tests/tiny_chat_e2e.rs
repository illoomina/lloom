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
//! the ids of the rendering written out here, and no value is `xhigh`'s; on
//! LFM2, whose template does not read it, `chat` refuses the flag and
//! `serve` renders without it and says so.
#![cfg(feature = "serve")]

use lloom::models::lfm2_testing::Lfm2Tiny;
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
    assert_eq!(low["timings"]["template_kwargs_ignored"], json!([]));

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

    let (status, off) = server.chat(json!({
        "reasoning_effort": "low", "chat_template_kwargs": { "enable_thinking": false },
    }));
    assert_eq!(status, 200, "{off}");
    assert_eq!(off["timings"]["reasoning_effort"], Value::Null);
    assert_eq!(
        off["timings"]["template_kwargs_ignored"][0]["name"],
        "reasoning_effort"
    );
    assert_eq!(off["timings"]["template_kwargs_ignored"][0]["value"], "low");
    assert_eq!(
        off["timings"]["prompt_ids_sha256"],
        sha(&encode(&tok, &qwen_not_thinking()))
    );
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}

/// `lloom serve` on the tiny LFM2: clients send `reasoning_effort` blindly,
/// so it is served — rendered without it, byte-identical to a request with
/// none, and listed as unread.
#[test]
fn lfm2_serve_ignores_the_reasoning_effort_it_cannot_read() {
    let dir = scratch("effort-serve-lfm2");
    let model = write::<Lfm2Tiny>(&dir);
    let server = Server::start(&model, &["--cpu"]);
    let (_, health) = server.call("GET", "/health", &json!({}));
    assert_eq!(
        health["chat_template_reads"],
        json!({"enable_thinking": false, "reasoning_effort": false})
    );
    let (status, with) = server.chat(json!({ "reasoning_effort": "low" }));
    assert_eq!(status, 200, "{with}");
    let (status, without) = server.chat(json!({}));
    assert_eq!(status, 200, "{without}");
    assert_eq!(
        with["timings"]["prompt_ids_sha256"],
        without["timings"]["prompt_ids_sha256"]
    );
    assert_eq!(with["timings"]["reasoning_effort"], Value::Null);
    assert_eq!(
        with["timings"]["template_kwargs_ignored"],
        json!([{ "name": "reasoning_effort", "value": "low", "reason": "is not read by the chat template" }])
    );
    assert_eq!(without["timings"]["template_kwargs_ignored"], json!([]));
    drop(server);
    std::fs::remove_dir_all(&dir).ok();
}
