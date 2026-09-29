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

const LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";
const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";

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
    assert!(stderr.contains(
        "--reasoning-effort high: the chat template rejected the request: Unexpected reasoning effort high. \
         Supported types are xhigh (default), medium, and low."
    ), "{stderr}");
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
        "the chat template rejected the request: Unexpected reasoning effort high. \
         Supported types are xhigh (default), medium, and low."
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

/// The ids a tiny family gives an added token.
fn added_id(tok: &TinyTokenizer, content: &str) -> u32 {
    tok.id(content)
        .unwrap_or_else(|| panic!("{content} is not an added token"))
}

/// A `lloom serve` chat request's generated ids and `timings`.
fn served_ids(server: &Server, fields: Value) -> (Vec<u32>, Value) {
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
    let (status, plain) = server.chat(json!({ "chat_template_kwargs": off }));
    assert_eq!(status, 200, "{plain}");
    let (status, with) =
        server.chat(json!({ "chat_template_kwargs": off, "thinking_token_budget": 2 }));
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
    let (status, plain) = server.chat(json!({ "chat_template_kwargs": on }));
    assert_eq!(status, 200, "{plain}");
    let (status, with) =
        server.chat(json!({ "chat_template_kwargs": on, "thinking_token_budget": 0 }));
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
        &json!({ "messages": conv, "max_tokens": 4, "truncate_prompt_tokens": 60 }),
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
        &json!({ "messages": kept, "max_tokens": 4 }),
    );
    assert_eq!(status, 200, "{alone}");
    assert_eq!(alone["timings"]["prompt_ids_sha256"], sha(&want));
    assert_eq!(alone["timings"]["tokens"], cut["timings"]["tokens"]);
    drop(b);
    std::fs::remove_dir_all(&dir).ok();
}
