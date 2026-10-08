//! `lloom-serve` — the engine as a process a client can drive (issue #695).
//!
//! One binary, two contracts on one loopback port:
//!
//! - the **OpenAI-compatible** surface a generic client speaks —
//!   `/v1/completions` (with `ignore_eos` and a `usage` block),
//!   `/v1/chat/completions` (streaming SSE, the checkpoint's own chat
//!   template, `logprobs`), `/v1/models`, `/tokenize`;
//! - the **benchmark sidecar** contract LiquidAI's `pipette` runners drive an
//!   engine through (`Liquid4All/pipette-clients`, the MLX sidecar's shape):
//!   `/prefill_throughput`, `/decode_throughput`, `/end_to_end_latency`,
//!   `/max_memory_usage`, `/eval` + `/eval/abort`, `/health`, `/shutdown`,
//!   and the `{"kind":"ready",…}` marker on stdout once the model is loaded.
//!
//! Two serving shapes (`--batch`, issue #148). At `--batch 1` the engine's
//! one sequence sits behind a mutex and the decode loop runs on the request's
//! thread — the shape a B=1 benchmark wants, every cell one cold request
//! against an idle engine, at the crown path's throughput. At `--batch N`
//! the continuous-batching scheduler runs on its own thread: requests from
//! any number of clients join the running batch and leave it at their EOS
//! (one `m = B` weight sweep per step for every live row), and the benchmark
//! cells take a `batch` field to run `B` rows as one closed batch with the
//! engine to themselves. `--dspark` runs the speculative body in either
//! shape.
//!
//! Every timing this binary reports is the engine's own bracket
//! (`DecodeStats`, or a `synchronize()`-closed window) and says which; the
//! runner brackets the HTTP request itself for the latency cells.

use super::budgets::{self, Dropped, PromptMode};
use super::common::{ms, MemSnapshot, NEVER_EOS};
use super::served::Served;
use super::template::{
    request_kwargs, Rejected, Unrenderable, ENABLE_THINKING, ENGINE_KWARGS, NOT_READ,
    REASONING_EFFORT, REQUEST_FIELDS,
};
use super::thinking::{self, Thinking};
use super::vision::{expand_image_tokens, split_image_parts, DecodedImage};
use crate::sink::{DeltaBuffer, DRAIN_PERIOD};
use crate::{DecodeParams, Decoded, Stop};
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

/// The seed passage the throughput cells tile into their prompt when no
/// `--prompt-seed-file` and no seed environment variable is given.
const DEFAULT_SEED_TEXT: &str = "The loom threads warp and weft into cloth the way a decoder \
threads its weights and its cache into text: one pass over the whole working set per token, \
the same pattern every time, only the position moving. A benchmark that wants a prompt of an \
exact length repeats a passage like this one until the count is met, because the count is the \
work and the words are not. ";

/// The serving flags: the option layers and the tree's flags, the socket,
/// the batch and the prefix reuse.
#[derive(Parser)]
#[command(
    name = "lloom serve",
    about = "Serve a checkpoint (LFM2 or Qwen4-exp) over an OpenAI-compatible endpoint and the pipette benchmark sidecar contract",
    after_help = super::OPTIONS_HELP
)]
pub struct ServeArgs {
    #[command(flatten)]
    pub common: super::CommonArgs,
    #[command(flatten)]
    pub server: crate::options::ServerConfigArgs,
    #[command(flatten)]
    pub scheduler: crate::options::SchedulerConfigArgs,
    #[command(flatten)]
    pub prefix: crate::options::PrefixConfigArgs,
    #[command(flatten)]
    pub report: crate::options::ReportConfigArgs,
}

impl ServeArgs {
    /// `serve`'s own flags, as the top layer.
    pub fn front(&self, o: &mut crate::EngineOptions) -> std::result::Result<(), String> {
        self.server.apply(&mut o.server)?;
        self.scheduler.apply(&mut o.scheduler)?;
        self.prefix.apply(&mut o.prefix)?;
        self.report.apply(&mut o.report)
    }
}

/// The prompt seed for the throughput cells: the file, the pipette's env
/// seed, then the built-in passage.
fn seed_text(file: Option<&std::path::Path>) -> Result<String> {
    Ok(match file {
        Some(p) => std::fs::read_to_string(p)
            .with_context(|| format!("reading the prompt seed {}", p.display()))?,
        None => std::env::var("PIPETTE_PROMPT_SEED_TEXT")
            .or_else(|_| std::env::var("PIPETTE_MLX_PROMPT_SEED_TEXT"))
            .unwrap_or_else(|_| DEFAULT_SEED_TEXT.to_string()),
    })
}

/// State shared by every request thread.
struct Shared {
    engine: Box<dyn Served>,
    /// Sample ids an `/eval/abort` asked the running `/eval` to stop early.
    aborts: Mutex<HashSet<String>>,
    /// Bumped after every insert into `aborts`: the decode thread's
    /// per-fragment check reads this atomic and takes the lock only when it
    /// moved, not every fragment.
    abort_epoch: AtomicU64,
    server: Server,
    model_name: String,
    /// The sampling defaults a request that names none decodes with (the
    /// `sampling` group; `--generation-config`).
    sampling: crate::SamplingConfig,
}

pub fn run(args: ServeArgs) -> Result<()> {
    let mut options = super::resolve_options(&args.common, &|o| args.front(o))?;
    let seed_text = seed_text(options.server.pipette.prompt_seed_file.as_deref())?;
    let batched = options.scheduler.max_sequences > 1;
    // The CHUNK defaults to 512 at `N > 1` (the interleaved prefill), whole
    // prompts at `N = 1`.
    if batched && options.decode.prefill.chunk == 0 {
        options.decode.prefill.chunk = 512;
    }
    let load_start = Instant::now();
    let shape = super::Shape::of(&options, false, seed_text);
    let reuse_prefix = shape.reuse_prefix;
    let (engine, model_path) = super::load_engine(&options.load, shape)?;
    let model_name = engine.model_name().to_string();
    let mut sampling = options.sampling.clone();
    sampling.apply_checkpoint(engine.generation_config().as_ref());
    sampling
        .check()
        .map_err(|e| anyhow::anyhow!("the checkpoint's generation_config.json: {e}"))?;
    eprintln!(
        "[loaded {} from {} in {:.2}s on {:?}; kv_capacity={} readback={} reuse_prefix={} {}]",
        model_name,
        model_path.display(),
        load_start.elapsed().as_secs_f64(),
        engine.device(),
        engine.kv_capacity(),
        engine.readback().name(),
        reuse_prefix,
        engine.describe(),
    );
    eprintln!("[capabilities: {}]", engine.capabilities().line());
    eprintln!("[{}]", MemSnapshot::read(engine.device()).json());
    let (host, port) = (options.server.host.clone(), options.server.port);

    let server = Server::http((host.as_str(), port))
        .map_err(|e| anyhow::anyhow!("binding {host}:{port}: {e}"))?;
    let port = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        #[allow(unreachable_patterns)]
        _ => port,
    };
    let shared = Arc::new(Shared {
        engine,
        aborts: Mutex::new(HashSet::new()),
        abort_epoch: AtomicU64::new(0),
        server,
        model_name,
        sampling,
    });

    // The ready marker the sidecar runner waits for, on stdout and nothing
    // else on stdout before it. Everything the operator reads goes to stderr.
    println!("{}", json!({"kind": "ready", "host": host, "port": port}));
    std::io::stdout().flush().ok();
    eprintln!("[serving on http://{host}:{port}]");

    loop {
        let request = match shared.server.recv() {
            Ok(r) => r,
            Err(_) => break,
        };
        let path = request.url().split('?').next().unwrap_or("").to_string();
        let method = request.method().clone();
        match (method, path.as_str()) {
            (Method::Get, "/health") => {
                // The memory gauges, read once: what the process holds now,
                // beside the family's own state.
                let mut body = shared.engine.health();
                body["mem"] = MemSnapshot::read(shared.engine.device()).json();
                body["capabilities"] = shared.engine.capabilities().json();
                body["load_warnings"] = load_warnings_json(shared.engine.as_ref());
                body["chat_template_reads"] = ENGINE_KWARGS
                    .iter()
                    .map(|&k| (k.to_string(), json!(shared.engine.template_reads(k))))
                    .collect::<Map<String, Value>>()
                    .into();
                // The thinking budget's markers, as the template writes them.
                let engine = shared.engine.as_ref();
                body["reasoning_tags"] = engine.think_tags().map_or(Value::Null, |t| {
                    json!({
                        "start": engine.token_str(t.start),
                        "end": engine.token_str(t.end),
                        "tool_call": t.tool_call.map(|id| engine.token_str(id)),
                    })
                });
                let _ = respond_json(request, 200, &body);
            }
            (Method::Get, "/v1/models") => {
                let _ = respond_json(
                    request,
                    200,
                    &models_body(&shared.model_name, shared.engine.as_ref()),
                );
            }
            (Method::Post, "/eval/abort") => {
                let _ = handle_abort(&shared, request);
            }
            (Method::Post, "/shutdown") => {
                let _ = respond_json(request, 200, &json!({}));
                eprintln!("[shutdown requested]");
                break;
            }
            (Method::Post, _) => {
                let shared = Arc::clone(&shared);
                std::thread::spawn(move || {
                    let path = path;
                    let started = Instant::now();
                    match handle_model_request(&shared, &path, request) {
                        Ok(()) => {}
                        Err(e) => eprintln!(
                            "[{path} failed after {:.1} ms: {e:#}]",
                            ms(started.elapsed())
                        ),
                    }
                });
            }
            _ => {
                let _ = respond_error(request, 404, &format!("unknown endpoint: {path}"));
            }
        }
    }
    // A request thread still decoding holds the sequence, and the batch
    // thread holds the pool; wait for both so the model is not dropped under
    // a running command buffer.
    shared.engine.shutdown();
    Ok(())
}

// ---- routing ---------------------------------------------------------------

fn handle_model_request(shared: &Shared, path: &str, mut request: Request) -> Result<()> {
    let mut body = String::new();
    request
        .as_reader()
        .read_to_string(&mut body)
        .context("reading the request body")?;
    let value: Value = if body.trim().is_empty() {
        json!({})
    } else {
        match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => {
                respond_error(request, 400, &format!("request body is not JSON: {e}"))?;
                return Ok(());
            }
        }
    };
    let result = match path {
        "/tokenize" => tokenize(shared, &value).map(Reply::Json),
        "/v1/completions" => completions(shared, &value, request.headers()),
        "/v1/chat/completions" => chat_completions(shared, &value),
        "/prefill_throughput" => prefill_throughput(shared, &value).map(Reply::Json),
        "/decode_throughput" => decode_throughput(shared, &value).map(Reply::Json),
        "/max_memory_usage" => max_memory_usage(shared, &value).map(Reply::Json),
        "/end_to_end_latency" => end_to_end_latency(shared, &value).map(Reply::Json),
        "/eval" => Ok(Reply::Eval(value)),
        _ => Err(Failure::NotFound(format!("unknown endpoint: {path}"))),
    };
    match result {
        Ok(Reply::Json(v)) => respond_json(request, 200, &v),
        Ok(Reply::Stream(stream)) => stream_response(shared, request, *stream),
        Ok(Reply::Eval(v)) => eval(shared, request, &v),
        Err(Failure::BadRequest(m)) => respond_error(request, 400, &m),
        Err(Failure::NotFound(m)) => respond_error(request, 404, &m),
        Err(Failure::Internal(e)) => respond_error(request, 500, &format!("{e:#}")),
    }
}

enum Reply {
    Json(Value),
    Stream(Box<StreamingRun>),
    Eval(Value),
}

/// A request whose body is produced while the decode runs.
struct StreamingRun {
    kind: StreamKind,
    prompt_ids: Vec<u32>,
    facts: PromptFacts,
    /// The chat's images, in the order their `<image>` positions appear.
    images: Vec<DecodedImage>,
    params: DecodeParams,
    id: String,
    model: String,
}

/// One decode of `prompt_ids`, through the image path when the request
/// carries images.
fn run_decode(
    engine: &dyn Served,
    prompt_ids: &[u32],
    images: &[DecodedImage],
    params: &DecodeParams,
    on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
) -> Result<Decoded> {
    if images.is_empty() {
        engine.complete_until(prompt_ids, params, on_fragment)
    } else {
        engine.complete_images_until(prompt_ids, images, params, on_fragment)
    }
}

#[derive(Clone, Copy)]
enum StreamKind {
    Completion,
    Chat,
}

enum Failure {
    BadRequest(String),
    NotFound(String),
    Internal(anyhow::Error),
}

impl From<anyhow::Error> for Failure {
    fn from(e: anyhow::Error) -> Self {
        Failure::Internal(e)
    }
}

fn bad(msg: impl Into<String>) -> Failure {
    Failure::BadRequest(msg.into())
}

type Handled<T> = std::result::Result<T, Failure>;

// ---- /tokenize ---------------------------------------------------------------

/// Both spellings: the MLX sidecar's `{prompt}` (always with special tokens)
/// and vLLM's `{model, prompt, add_special_tokens}`; llama-server's
/// `{content, add_special}` too. All report `tokens` and `count`.
fn tokenize(shared: &Shared, v: &Value) -> Handled<Value> {
    let text = v
        .get("prompt")
        .or_else(|| v.get("content"))
        .and_then(Value::as_str)
        .ok_or_else(|| bad("field 'prompt' must be a string"))?;
    let add_special = v
        .get("add_special_tokens")
        .or_else(|| v.get("add_special"))
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let ids = shared.engine.encode(text, add_special)?;
    Ok(json!({ "tokens": ids, "count": ids.len() }))
}

// ---- /v1/completions ---------------------------------------------------------

fn completions(shared: &Shared, v: &Value, _headers: &[Header]) -> Handled<Reply> {
    let model = shared.model_name.clone();
    let params = sampling_params(shared, v)?;
    let stream = v.get("stream").and_then(Value::as_bool).unwrap_or(false);
    if v.get("n").and_then(Value::as_u64).unwrap_or(1) != 1 {
        return Err(bad("n > 1 is not supported"));
    }
    if v.get("logprobs").map(|l| !l.is_null()).unwrap_or(false) {
        return Err(bad("logprobs is not supported on /v1/completions"));
    }
    let prompt_ids = {
        let engine = shared.engine.as_ref();
        match v.get("prompt") {
            Some(Value::String(s)) => engine.encode(s, true)?,
            Some(Value::Array(items)) if items.iter().all(Value::is_u64) => items
                .iter()
                .filter_map(Value::as_u64)
                .map(|t| t as u32)
                .collect(),
            Some(Value::Array(items)) if items.len() == 1 && items[0].is_string() => {
                engine.encode(items[0].as_str().unwrap_or(""), true)?
            }
            _ => {
                return Err(bad(
                    "field 'prompt' must be a string or an array of token ids",
                ))
            }
        }
    };
    if prompt_ids.is_empty() {
        return Err(bad("empty prompt"));
    }
    let mut params = params;
    let wanted = request_budgets(shared, v, params.max_tokens)?;
    let (prompt_ids, dropped) = match wanted.prompt {
        Some((k, mode)) if prompt_ids.len() > k => {
            if mode == PromptMode::Refuse {
                return Err(bad(budgets::over_budget(
                    prompt_ids.len(),
                    k,
                    "truncate_prompt_tokens keeps the last tokens instead",
                )));
            }
            budgets::truncate_raw(&prompt_ids, k, shared.engine.bos_id())
        }
        _ => (prompt_ids, Dropped::default()),
    };
    let id = format!("cmpl-{}", short_id());
    let mut facts = PromptFacts::of(&prompt_ids);
    facts.apply_budgets(
        shared.engine.as_ref(),
        "/v1/completions",
        &wanted,
        dropped,
        &prompt_ids,
        &mut params,
    )?;
    if stream {
        return Ok(Reply::Stream(Box::new(StreamingRun {
            kind: StreamKind::Completion,
            prompt_ids,
            facts,
            images: Vec::new(),
            params,
            id,
            model,
        })));
    }
    let started = Instant::now();
    let decoded = shared.engine.complete(&prompt_ids, &params, &mut |_| {})?;
    let elapsed = started.elapsed();
    report(
        "/v1/completions",
        prompt_ids.len(),
        &facts,
        &decoded,
        ms(elapsed),
    );
    Ok(Reply::Json(json!({
        "id": id,
        "object": "text_completion",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "text": decoded.text,
            "logprobs": null,
            "finish_reason": finish_reason(decoded.stop),
        }],
        "usage": usage(prompt_ids.len(), decoded.tokens.len()),
        "timings": timings(prompt_ids.len(), &facts, &decoded, elapsed),
    })))
}

// ---- /v1/chat/completions ----------------------------------------------------

fn chat_completions(shared: &Shared, v: &Value) -> Handled<Reply> {
    let model = shared.model_name.clone();
    let messages = v
        .get("messages")
        .filter(|m| m.is_array())
        .ok_or_else(|| bad("field 'messages' must be an array"))?;
    if messages.as_array().map(Vec::is_empty).unwrap_or(true) {
        return Err(bad("messages must not be empty"));
    }
    if v.get("guided_choice").is_some() || v.get("guided_json").is_some() {
        // Constrained decoding is not built; a 4xx is what a client that can
        // fall back to `logprobs` looks for.
        return Err(bad("guided decoding is not supported"));
    }
    let mut params = sampling_params(shared, v)?;
    let wanted = request_budgets(shared, v, params.max_tokens)?;
    let stream = v.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let want_logprobs = v.get("logprobs").and_then(Value::as_bool).unwrap_or(false);
    let mut kwargs = request_kwargs(v, &REQUEST_FIELDS).map_err(bad)?;
    let want = kwargs.get(ENABLE_THINKING).and_then(Value::as_bool);
    thinking::set_kwarg(shared.engine.as_ref(), want, &mut kwargs);

    // OpenAI `image_url` parts become the template's `{"type": "image"}`;
    // string content passes through untouched.
    let (messages, images) = split_image_parts(messages).map_err(|e| bad(format!("{e:#}")))?;
    let (prompt_ids, facts) = {
        let engine = shared.engine.as_ref();
        let (rendered, mut facts) =
            render_request(engine, "/v1/chat/completions", &messages, &kwargs)?;
        let rendered = if images.is_empty() {
            rendered
        } else {
            let expansions = images
                .iter()
                .map(|img| engine.image_placeholder(img))
                .collect::<Result<Vec<_>>>()
                .map_err(|e| bad(format!("{e:#}")))?;
            expand_image_tokens(&rendered, &expansions).map_err(|e| bad(format!("{e:#}")))?
        };
        let mut ids = engine.encode(&rendered, true)?;
        if let Some(bos) = engine.bos_id() {
            if ids.iter().filter(|&&t| t == bos).count() > 1 {
                eprintln!("[chat: the rendered template carries BOS twice; check bos_token]");
            }
        }
        let tags = engine.think_tags();
        facts.thinking = Some(apply_thinking(engine, want, tags.as_ref(), &mut ids)?);
        // The prompt budget, before the engine sees the prompt: the oldest
        // turns go whole, so what runs is the template's own rendering.
        let mut dropped = Dropped::default();
        if let Some((k, mode)) = wanted.prompt.filter(|&(k, _)| ids.len() > k) {
            if mode == PromptMode::Refuse {
                return Err(bad(budgets::over_budget(
                    ids.len(),
                    k,
                    "truncate_prompt_tokens drops the oldest turns instead",
                )));
            }
            if !images.is_empty() {
                return Err(bad(
                    "truncate_prompt_tokens with an image is not supported: truncating would \
                     drop turns without their images",
                ));
            }
            let arr = messages.as_array().cloned().unwrap_or_default();
            let fitted = budgets::fit_messages(&arr, k, ids.len(), |ms| {
                let mut ids = engine.encode(&engine.render_chat(&json!(ms), &kwargs)?, true)?;
                let _ = thinking::apply(want, tags.as_ref(), &mut ids);
                Ok(ids)
            })
            .map_err(template_failure)?;
            let Some(fitted) = fitted else {
                return Err(bad(budgets::over_budget(
                    ids.len(),
                    k,
                    "the last user turn alone does not fit",
                )));
            };
            (ids, dropped) = (fitted.ids, fitted.dropped);
        }
        facts.ids_sha256 = super::chat::tokens_sha256(&ids);
        facts.apply_budgets(
            engine,
            "/v1/chat/completions",
            &wanted,
            dropped,
            &ids,
            &mut params,
        )?;
        (ids, facts)
    };
    let id = format!("chatcmpl-{}", short_id());

    if want_logprobs {
        if stream {
            return Err(bad("logprobs with stream is not supported"));
        }
        if !images.is_empty() {
            return Err(bad("logprobs with an image is not supported"));
        }
        let top_k = v
            .get("top_logprobs")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .clamp(1, 64) as usize;
        if params.max_tokens != 1 {
            return Err(bad("logprobs is supported for max_tokens = 1 only"));
        }
        // The decision scoring: `options` (strings), each scored by the
        // log-probability of its first token under the same forward — a
        // decision model's (JEV, clef) one-forward choice. An option is a
        // token or the tokens it encodes to, scored at its first.
        let options: Vec<String> = match v.get("options") {
            None | Some(Value::Null) => Vec::new(),
            Some(Value::Array(a)) => {
                let mut out = Vec::with_capacity(a.len());
                for o in a {
                    match o.as_str() {
                        Some(s) if !s.is_empty() => out.push(s.to_string()),
                        _ => return Err(bad("options: every option is a non-empty string")),
                    }
                }
                if out.len() < 2 {
                    return Err(bad("options: a decision takes at least two"));
                }
                out
            }
            Some(_) => return Err(bad("options: an array of strings")),
        };
        // One forward: the argmax is the token, and its distribution is the
        // logprobs block. Temperature does not apply on this path. With
        // options the whole distribution comes back, so each option's first
        // token is scored wherever it ranks.
        let engine = shared.engine.as_ref();
        let started = Instant::now();
        let want = if options.is_empty() {
            top_k
        } else {
            usize::MAX
        };
        let (argmax, all) = engine.next_token_logprobs(&prompt_ids, want)?;
        let top = &all[..top_k.min(all.len())];
        let content = engine.token_str(argmax);
        let token_lp = top
            .iter()
            .find(|(t, _)| *t == argmax)
            .map(|(_, lp)| *lp)
            .unwrap_or(f32::NEG_INFINITY);
        let top_entries: Vec<Value> = top
            .iter()
            .map(|(t, lp)| json!({ "token": engine.token_str(*t), "logprob": lp }))
            .collect();
        let decision = if options.is_empty() {
            Value::Null
        } else {
            let mut scored = Vec::with_capacity(options.len());
            let mut best: Option<(usize, f32)> = None;
            for (i, text) in options.iter().enumerate() {
                let ids = engine.encode(text, false)?;
                let Some(&first) = ids.first() else {
                    return Err(bad(format!("options[{i}] encodes to no token")));
                };
                let (rank, lp) = all
                    .iter()
                    .enumerate()
                    .find(|(_, (t, _))| *t == first)
                    .map(|(r, (_, lp))| (r, *lp))
                    .unwrap_or((all.len(), f32::NEG_INFINITY));
                if best.is_none_or(|(_, b)| lp > b) {
                    best = Some((i, lp));
                }
                scored.push(json!({
                    "text": text,
                    "token": engine.token_str(first),
                    "token_id": first,
                    "tokens": ids.len(),
                    "logprob": lp,
                    "rank": rank,
                }));
            }
            let firsts: Vec<u64> = scored
                .iter()
                .map(|o| o["token_id"].as_u64().unwrap_or(0))
                .collect();
            let mut ambiguous: Vec<usize> = Vec::new();
            for (i, t) in firsts.iter().enumerate() {
                if firsts.iter().enumerate().any(|(j, u)| j != i && u == t) {
                    ambiguous.push(i);
                }
            }
            let (choice, _) = best.expect("at least two options");
            json!({
                "choice": choice,
                "text": options[choice],
                "options": scored,
                "ambiguous": ambiguous,
            })
        };
        eprintln!(
            "[/v1/chat/completions logprobs: prompt_tokens={} top_logprobs={top_k} options={} {:.1} ms]",
            prompt_ids.len(),
            options.len(),
            ms(started.elapsed())
        );
        return Ok(Reply::Json(json!({
            "id": id,
            "object": "chat.completion",
            "created": now(),
            "model": model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "logprobs": { "content": [{
                    "token": content,
                    "logprob": token_lp,
                    "top_logprobs": top_entries,
                }] },
                "decision": decision,
                "finish_reason": "length",
            }],
            "usage": usage(prompt_ids.len(), 1),
        })));
    }

    if stream {
        return Ok(Reply::Stream(Box::new(StreamingRun {
            kind: StreamKind::Chat,
            prompt_ids,
            facts,
            images,
            params,
            id,
            model,
        })));
    }
    let started = Instant::now();
    let decoded = run_decode(
        shared.engine.as_ref(),
        &prompt_ids,
        &images,
        &params,
        &mut |_| true,
    )?;
    let elapsed = started.elapsed();
    report(
        "/v1/chat/completions",
        prompt_ids.len(),
        &facts,
        &decoded,
        ms(elapsed),
    );
    let mut body = json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": { "role": "assistant", "content": decoded.text },
            "logprobs": null,
            "finish_reason": finish_reason(decoded.stop),
        }],
        "usage": usage(prompt_ids.len(), decoded.tokens.len()),
        "timings": timings(prompt_ids.len(), &facts, &decoded, elapsed),
    });
    attach_vision(shared, &mut body);
    Ok(Reply::Json(body))
}

/// The token budgets a request asks for, over the process's defaults
/// (`--thinking-budget`, `--prompt-budget`): vLLM's `thinking_token_budget`
/// (`-1` is none) with `thinking_budget_message`, and its
/// `truncate_prompt_tokens` (`-1` is the longest prompt the engine takes
/// beside `max_tokens`), which asks for truncation whatever the process's
/// mode.
struct Budgets {
    thinking: Option<usize>,
    message: Option<String>,
    prompt: Option<(usize, PromptMode)>,
}

fn request_budgets(shared: &Shared, v: &Value, max_tokens: usize) -> Handled<Budgets> {
    let signed = |key: &str| -> Handled<Option<i64>> {
        match v.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(n) => n
                .as_i64()
                .map(Some)
                .ok_or_else(|| bad(format!("field '{key}' must be an integer"))),
        }
    };
    let thinking = match signed("thinking_token_budget")? {
        None => shared.sampling.thinking_budget,
        Some(-1) => None,
        Some(n) if n >= 0 => Some(n as usize),
        Some(n) => {
            return Err(bad(format!(
                "field 'thinking_token_budget' must be -1 (none) or at least 0, not {n}"
            )))
        }
    };
    let message = match v.get("thinking_budget_message") {
        None | Some(Value::Null) => shared.sampling.thinking_budget_message.clone(),
        Some(Value::String(s)) => Some(s.clone()),
        Some(_) => return Err(bad("field 'thinking_budget_message' must be a string")),
    };
    let prompt = match signed("truncate_prompt_tokens")? {
        None => shared
            .sampling
            .prompt_budget
            .map(|k| (k, PromptMode::parse(&shared.sampling.prompt_budget_mode))),
        Some(-1) => Some((
            shared
                .engine
                .kv_capacity()
                .saturating_sub(max_tokens)
                .max(1),
            PromptMode::Truncate,
        )),
        Some(k) if k >= 1 => Some((k as usize, PromptMode::Truncate)),
        Some(k) => {
            return Err(bad(format!(
                "field 'truncate_prompt_tokens' must be -1 or at least 1, not {k}"
            )))
        }
    };
    Ok(Budgets {
        thinking,
        message,
        prompt,
    })
}

/// The sampling knobs both OpenAI endpoints share.
fn sampling_params(shared: &Shared, v: &Value) -> Handled<DecodeParams> {
    let max_tokens = v
        .get("max_completion_tokens")
        .or_else(|| v.get("max_tokens"))
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(shared.sampling.max_tokens);
    if max_tokens == 0 {
        return Err(bad("max_tokens must be positive"));
    }
    let temperature = match v.get("temperature") {
        None | Some(Value::Null) => shared.sampling.temperature,
        Some(t) => Some(
            t.as_f64()
                .ok_or_else(|| bad("field 'temperature' must be a number"))?,
        ),
    };
    let top_k = match v.get("top_k") {
        None | Some(Value::Null) => shared.sampling.top_k,
        Some(k) => Some(
            k.as_u64()
                .ok_or_else(|| bad("field 'top_k' must be a positive integer"))?
                as usize,
        ),
    };
    let gumbel = v
        .get("gumbel")
        .and_then(Value::as_bool)
        .unwrap_or(shared.sampling.gumbel);
    let top_p = match v.get("top_p") {
        None | Some(Value::Null) => shared.sampling.top_p,
        Some(p) => {
            let p = p
                .as_f64()
                .ok_or_else(|| bad("field 'top_p' must be a number"))?;
            // `1.0` is the OpenAI default and means no nucleus.
            (p > 0.0 && p < 1.0).then_some(p)
        }
    };
    // A sampled request with no seed draws a fresh one, so repeats are
    // independent; a seeded one reproduces.
    let seed = v
        .get("seed")
        .and_then(Value::as_u64)
        .unwrap_or_else(|| shared.sampling.seed_or_random());
    let ignore_eos = v
        .get("ignore_eos")
        .and_then(Value::as_bool)
        .unwrap_or(shared.sampling.ignore_eos);
    let stop_strings: Vec<String> = match v.get("stop") {
        None | Some(Value::Null) => shared.sampling.stop.clone(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        Some(_) => return Err(bad("field 'stop' must be a string or an array of strings")),
    };
    // vLLM's other draw-changing fields: refused by name, never ignored.
    for key in [
        "min_tokens",
        "stop_token_ids",
        "bad_words",
        "allowed_token_ids",
    ] {
        let neutral = match v.get(key) {
            None | Some(Value::Null) => true,
            Some(Value::Number(n)) => n.as_f64() == Some(0.0),
            Some(Value::Array(a)) => a.is_empty(),
            Some(_) => false,
        };
        if !neutral {
            return Err(bad(format!("field '{key}' is not supported")));
        }
    }
    let processors = request_processors(&shared.sampling, v)?;
    let (eos_ids, readback) = {
        let engine = shared.engine.as_ref();
        let eos = if ignore_eos {
            NEVER_EOS.to_vec()
        } else {
            engine.eos_ids().to_vec()
        };
        (eos, engine.readback())
    };
    Ok(DecodeParams {
        seed,
        temperature,
        top_p,
        top_k,
        gumbel,
        processors,
        max_tokens,
        eos_ids,
        stop_strings,
        readback,
        logits_digest: false,
        prefill_tile: shared.engine.prefill_tile(),
        thinking: None,
    })
}

/// The request's logits processors (vLLM's `repetition_penalty` and `min_p`,
/// OpenAI's `presence_penalty`, `frequency_penalty` and `logit_bias`) over the
/// process's defaults, each refused by name when malformed or out of range.
fn request_processors(
    defaults: &crate::SamplingConfig,
    v: &Value,
) -> Handled<lloom::generation::Processors> {
    let mut cfg = defaults.clone();
    for (key, slot) in [
        ("repetition_penalty", &mut cfg.repetition_penalty),
        ("presence_penalty", &mut cfg.presence_penalty),
        ("frequency_penalty", &mut cfg.frequency_penalty),
        ("min_p", &mut cfg.min_p),
    ] {
        match v.get(key) {
            None | Some(Value::Null) => {}
            Some(x) => {
                *slot = Some(
                    x.as_f64()
                        .ok_or_else(|| bad(format!("field '{key}' must be a number")))?,
                )
            }
        }
    }
    match v.get("logit_bias") {
        None | Some(Value::Null) => {}
        Some(Value::Object(map)) => {
            let mut bias = std::collections::BTreeMap::new();
            for (id, b) in map {
                let id: u32 = id.trim().parse().map_err(|_| {
                    bad(format!("field 'logit_bias': key `{id}` is not a token id"))
                })?;
                let b = b.as_f64().ok_or_else(|| {
                    bad(format!(
                        "field 'logit_bias': the bias for token {id} must be a number"
                    ))
                })?;
                bias.insert(id, b);
            }
            cfg.logit_bias = bias;
        }
        Some(_) => {
            return Err(bad(
                "field 'logit_bias' must be an object mapping token ids to biases",
            ))
        }
    }
    let processors = cfg.processors();
    processors.validate().map_err(bad)?;
    Ok(processors)
}

// ---- streaming (SSE) ---------------------------------------------------------

/// Run the decode on this thread and write the OpenAI SSE stream as fragments
/// arrive: chunked transfer through the request's raw writer, so a token is on
/// the wire when the loop's commit drains it.
fn stream_response(shared: &Shared, request: Request, run: StreamingRun) -> Result<()> {
    let endpoint = match run.kind {
        StreamKind::Completion => "/v1/completions",
        StreamKind::Chat => "/v1/chat/completions",
    };
    let created = now();
    // Each drained batch of text becomes one chunk, wrapped on the writer's
    // thread (`StreamWriter::start_with_deltas`).
    let wrap = {
        let (id, model, kind) = (run.id.clone(), run.model.clone(), run.kind);
        move |text: &str| -> String {
            let chunk = match kind {
                StreamKind::Completion => json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{ "index": 0, "text": text, "logprobs": null, "finish_reason": null }],
                }),
                StreamKind::Chat => json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{ "index": 0, "delta": { "content": text }, "finish_reason": null }],
                }),
            };
            format!("data: {chunk}\n\n")
        }
    };
    let sw =
        StreamWriter::start_with_deltas(ChunkedWriter::start(request, "text/event-stream")?, wrap);
    if let StreamKind::Chat = run.kind {
        sw.event(&json!({
            "id": run.id, "object": "chat.completion.chunk", "created": created,
            "model": run.model,
            "choices": [{ "index": 0, "delta": { "role": "assistant" }, "finish_reason": null }],
        }));
    }
    let started = Instant::now();
    let decoded = {
        let engine = shared.engine.as_ref();
        // The callback runs on the decode thread: it hands the text over;
        // the chunk is built and written on the writer thread. A client that
        // went away stops the run at the next drained batch.
        run_decode(
            engine,
            &run.prompt_ids,
            &run.images,
            &run.params,
            &mut |frag| {
                if sw.client_gone() {
                    return false;
                }
                sw.delta(frag);
                true
            },
        )
    };
    let (mut w, write_err) = sw.finish()?;
    let decoded = decoded?;
    if let Some(e) = write_err {
        anyhow::bail!("client went away mid-stream: {e}");
    }
    let elapsed = started.elapsed();
    report(
        endpoint,
        run.prompt_ids.len(),
        &run.facts,
        &decoded,
        ms(elapsed),
    );
    let usage = usage(run.prompt_ids.len(), decoded.tokens.len());
    let last = match run.kind {
        StreamKind::Completion => json!({
            "id": run.id, "object": "text_completion", "created": created, "model": run.model,
            "choices": [{ "index": 0, "text": "", "logprobs": null,
                          "finish_reason": finish_reason(decoded.stop) }],
            "usage": usage,
            "timings": timings(run.prompt_ids.len(), &run.facts, &decoded, elapsed),
        }),
        StreamKind::Chat => json!({
            "id": run.id, "object": "chat.completion.chunk", "created": created, "model": run.model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason(decoded.stop) }],
            "usage": usage,
            "timings": timings(run.prompt_ids.len(), &run.facts, &decoded, elapsed),
        }),
    };
    let mut last = last;
    if let StreamKind::Chat = run.kind {
        attach_vision(shared, &mut last);
    }
    w.event(&last)?;
    w.raw("data: [DONE]\n\n")?;
    w.finish()
}

// ---- the pipette sidecar contract -------------------------------------------

fn field_usize(v: &Value, name: &str) -> Handled<usize> {
    match v.get(name) {
        Some(n) => n
            .as_u64()
            .filter(|&n| n > 0)
            .map(|n| n as usize)
            .ok_or_else(|| bad(format!("field '{name}' must be a positive integer"))),
        None => Err(bad(format!("field '{name}' must be a positive integer"))),
    }
}

/// The `batch` a sidecar cell asks for: `B` rows in one closed batch
/// (`1`, the default, is the single-request cell).
fn field_batch(shared: &Shared, v: &Value) -> Handled<usize> {
    let batch = match v.get("batch") {
        None | Some(Value::Null) => 1,
        Some(n) => n
            .as_u64()
            .filter(|&n| n > 0)
            .map(|n| n as usize)
            .ok_or_else(|| bad("field 'batch' must be a positive integer"))?,
    };
    shared
        .engine
        .check_batch(batch)
        .map_err(|e| bad(e.to_string()))?;
    Ok(batch)
}

fn prefill_throughput(shared: &Shared, v: &Value) -> Handled<Value> {
    let prompt_tokens = field_usize(v, "prompt_tokens")?;
    let batch = field_batch(shared, v)?;
    let t = shared.engine.prefill_throughput(prompt_tokens, batch)?;
    let total = t.prompt_tokens * t.batch;
    let prompt_tps = total as f64 / (t.prefill_ms / 1e3);
    eprintln!(
        "[/prefill_throughput: prompt_tokens={} batch={} ({} forwards) prefill {:.2} ms = {prompt_tps:.1} tok/s aggregate; {}]",
        t.prompt_tokens, t.batch, t.forwards, t.prefill_ms, t.mem.line()
    );
    Ok(json!({
        "prompt_tps": prompt_tps,
        "prompt_tokens": t.prompt_tokens,
        "batch": t.batch,
        "batch_prompt_tokens": total,
        "prefill_ms": t.prefill_ms,
        "forwards": t.forwards,
        "mem": t.mem.json(),
    }))
}

fn decode_throughput(shared: &Shared, v: &Value) -> Handled<Value> {
    let prompt_tokens = field_usize(v, "prompt_tokens")?;
    let decode_tokens = field_usize(v, "decode_tokens")?;
    let batch = field_batch(shared, v)?;
    let t = shared
        .engine
        .decode_throughput(prompt_tokens, decode_tokens, batch)?;
    let total: usize = t.row_tokens.iter().sum();
    let generation_tps = total as f64 / (t.decode_ms / 1e3);
    let per_row_tps = generation_tps / t.batch as f64;
    eprintln!(
        "[/decode_throughput: prompt_tokens={} decode_tokens={} batch={} context {:.2} ms, decode {:.2} ms \
         = {generation_tps:.1} tok/s aggregate ({per_row_tps:.1} per row), readback={}, bracket: {}; {}]",
        t.prompt_tokens,
        t.decode_tokens,
        t.batch,
        t.prefill_ms,
        t.decode_ms,
        t.readback.name(),
        t.bracket,
        t.mem.line(),
    );
    if let Some(p) = &t.profile {
        let steps = if t.row_rounds.is_empty() {
            t.decode_tokens.max(1) as f64
        } else {
            t.row_rounds.iter().copied().max().unwrap_or(1).max(1) as f64
        };
        eprintln!(
            "[/decode_throughput profile: gpu_busy {:.2} ms over the bracket ({:.3} ms per step/round of {steps:.0}), span {:.2} ms, {} dispatches ({:.1} per step), {} command buffers, {} encoders; top kernels: {}]",
            p.gpu_busy_ms,
            p.gpu_busy_ms / steps,
            p.gpu_span_ms,
            p.dispatches,
            p.dispatches as f64 / steps,
            p.command_buffers,
            p.encoders,
            p.by_label
                .iter()
                .take(12)
                .map(|(k, n)| format!("{k}={n}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    if !t.row_rounds.is_empty() {
        // The round's phases, per round, on the row that ran the most rounds
        // (the batch's longest-lived row: its rounds are the batch's).
        let (r, &rounds) = t
            .row_rounds
            .iter()
            .enumerate()
            .max_by_key(|(_, &n)| n)
            .expect("non-empty");
        let per = |v: &Vec<f64>| v[r] / rounds.max(1) as f64;
        eprintln!(
            "[/decode_throughput rounds: row {r} ran {rounds} rounds at {:.2} accepted; per round draft {:.2} ms, verify {:.2} ms, observe {:.2} ms]",
            t.row_accepted[r],
            per(&t.row_draft_ms),
            per(&t.row_verify_ms),
            per(&t.row_observe_ms),
        );
    }
    Ok(json!({
        "generation_tps": generation_tps,
        "per_row_tps": per_row_tps,
        "decode_tokens": t.decode_tokens,
        "prompt_tokens": t.prompt_tokens,
        "batch": t.batch,
        "batch_decode_tokens": total,
        "decode_ms": t.decode_ms,
        "context_ms": t.prefill_ms,
        "row_tokens": t.row_tokens,
        "row_rounds": t.row_rounds,
        "row_accepted": t.row_accepted,
        "row_draft_ms": t.row_draft_ms,
        "row_verify_ms": t.row_verify_ms,
        "row_observe_ms": t.row_observe_ms,
        "profile": t.profile.as_ref().map(|p| p.json()),
        "readback": t.readback.name(),
        "bracket": t.bracket,
        "mem": t.mem.json(),
    }))
}

/// `B` rows through the request path at once — the shape `B` clients make —
/// each row a full request with `params`, run on `B` threads; the wall
/// closes when the last row lands. Returns the rows in order.
fn concurrent_rows(
    shared: &Shared,
    prompts: Vec<Vec<u32>>,
    params: &DecodeParams,
) -> Result<Vec<Decoded>> {
    std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .iter()
            .map(|ids| {
                let params = params.clone();
                scope.spawn(move || shared.engine.complete(ids, &params, &mut |_| {}))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| {
                h.join()
                    .map_err(|_| anyhow::anyhow!("a request thread panicked"))?
            })
            .collect()
    })
}

fn max_memory_usage(shared: &Shared, v: &Value) -> Handled<Value> {
    let prompt_tokens = field_usize(v, "prompt_tokens")?;
    let decode_tokens = field_usize(v, "decode_tokens")?;
    let batch = field_batch(shared, v)?;
    let engine = shared.engine.as_ref();
    let params = DecodeParams {
        max_tokens: decode_tokens,
        eos_ids: NEVER_EOS.to_vec(),
        readback: engine.readback(),
        prefill_tile: engine.prefill_tile(),
        ..DecodeParams::default()
    };
    let prompts: Vec<Vec<u32>> = (0..batch)
        .map(|r| engine.seed_prompt_row(prompt_tokens, r))
        .collect();
    let decoded = concurrent_rows(shared, prompts, &params)?;
    let completion: usize = decoded.iter().map(|d| d.tokens.len()).sum();
    eprintln!(
        "[/max_memory_usage: prompt_tokens={} batch={batch} completion_tokens={completion}]",
        prompt_tokens
    );
    Ok(json!({
        "prompt_tokens": prompt_tokens,
        "batch": batch,
        "completion_tokens": completion / batch.max(1),
        "batch_completion_tokens": completion,
    }))
}

/// Tokenize inside the bracket, as a client-visible request would: the
/// runner times the HTTP request; `total_ms` is the same window one layer in.
/// At `batch > 1` the same prompt is sent as `B` concurrent requests and the
/// window closes when the last one lands (`per_row_ms` carries each).
fn end_to_end_latency(shared: &Shared, v: &Value) -> Handled<Value> {
    let prompt = v
        .get("prompt")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("field 'prompt' must be a string"))?;
    let decode_tokens = field_usize(v, "decode_tokens")?;
    let batch = field_batch(shared, v)?;
    let engine = shared.engine.as_ref();
    let started = Instant::now();
    let ids = engine.encode(prompt, true)?;
    let params = DecodeParams {
        max_tokens: decode_tokens,
        eos_ids: NEVER_EOS.to_vec(),
        readback: engine.readback(),
        prefill_tile: engine.prefill_tile(),
        ..DecodeParams::default()
    };
    if batch == 1 {
        let decoded = engine.complete(&ids, &params, &mut |_| {})?;
        let total_ms = ms(started.elapsed());
        report(
            "/end_to_end_latency",
            ids.len(),
            &PromptFacts::of(&ids),
            &decoded,
            total_ms,
        );
        return Ok(json!({
            "total_ms": total_ms,
            "prompt_tokens": ids.len(),
            "completion_tokens": decoded.tokens.len(),
            "batch": 1,
        }));
    }
    let prompts: Vec<Vec<u32>> = (0..batch).map(|_| ids.clone()).collect();
    let (decoded, per_row_ms): (Vec<Decoded>, Vec<f64>) = std::thread::scope(|scope| {
        let handles: Vec<_> = prompts
            .iter()
            .map(|p| {
                let params = params.clone();
                scope.spawn(move || {
                    let t = Instant::now();
                    let d = engine.complete(p, &params, &mut |_| {});
                    (d, ms(t.elapsed()))
                })
            })
            .collect();
        let mut ds = Vec::new();
        let mut ts = Vec::new();
        for h in handles {
            let (d, t) = h
                .join()
                .map_err(|_| anyhow::anyhow!("a request thread panicked"))?;
            ds.push(d?);
            ts.push(t);
        }
        Ok::<_, anyhow::Error>((ds, ts))
    })?;
    let total_ms = ms(started.elapsed());
    let facts = PromptFacts::of(&ids);
    for (i, d) in decoded.iter().enumerate() {
        report(
            &format!("/end_to_end_latency row {i}/{batch}"),
            ids.len(),
            &facts,
            d,
            per_row_ms[i],
        );
    }
    let completion: Vec<usize> = decoded.iter().map(|d| d.tokens.len()).collect();
    if completion.iter().any(|&c| c != decode_tokens) {
        return Err(Failure::Internal(anyhow::anyhow!(
            "rows produced {completion:?} tokens, expected {decode_tokens} each"
        )));
    }
    Ok(json!({
        "total_ms": total_ms,
        "prompt_tokens": ids.len(),
        "completion_tokens": decode_tokens,
        "batch": batch,
        "batch_completion_tokens": completion.iter().sum::<usize>(),
        "per_row_ms": per_row_ms,
        "readback": decoded.first().map(|d| d.readback.name()),
    }))
}

/// The eval stream: one JSONL event per sample start / chunk / done, then
/// `eval_done`. The chat template is the checkpoint's; a sample an
/// `/eval/abort` names is stopped at the next drained batch and its partial
/// completion reported with `stopped_early`.
///
/// Under `--batch N` the samples run `N` at a time — `N` worker threads
/// pulling from the list, each sample a request through the scheduler, so
/// a sample joins the batch the moment one leaves it — and their events
/// interleave on the stream, keyed by `sample_id`. At `--batch 1` one
/// worker runs them in order, as before.
fn eval(shared: &Shared, request: Request, v: &Value) -> Result<()> {
    let samples = match v.get("samples").and_then(Value::as_array) {
        Some(s) => s.clone(),
        None => {
            respond_error(request, 400, "field 'samples' must be a list")?;
            return Ok(());
        }
    };
    let max_tokens = match field_usize(v, "max_tokens") {
        Ok(n) => n,
        Err(_) => {
            respond_error(
                request,
                400,
                "field 'max_tokens' must be a positive integer",
            )?;
            return Ok(());
        }
    };
    let done: HashSet<String> = v
        .get("completions_done_ids")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default();
    let temperature = v.get("temperature").and_then(Value::as_f64).unwrap_or(0.0);
    if !(0.0..=2.0).contains(&temperature) {
        respond_error(
            request,
            400,
            "field 'temperature' must be a number in [0.0, 2.0]",
        )?;
        return Ok(());
    }
    let want = match v.get(ENABLE_THINKING) {
        None | Some(Value::Null) => None,
        Some(Value::Bool(b)) => Some(*b),
        Some(_) => {
            respond_error(
                request,
                400,
                "field 'enable_thinking' must be true or false",
            )?;
            return Ok(());
        }
    };
    let mut kwargs = Map::new();
    thinking::set_kwarg(shared.engine.as_ref(), want, &mut kwargs);
    let tags = shared.engine.think_tags();

    let sw = StreamWriter::start(ChunkedWriter::start(request, "application/x-ndjson")?);
    let engine = shared.engine.as_ref();
    let total = samples.len();
    let workers = engine.batch().max(1).min(total.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let run = std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let (sw, samples, done, kwargs, next, tags) =
                    (&sw, &samples, &done, &kwargs, &next, &tags);
                scope.spawn(move || -> Result<()> {
                    loop {
                        let i = next.fetch_add(1, Ordering::AcqRel);
                        if i >= samples.len() || sw.client_gone() {
                            return Ok(());
                        }
                        let sample = &samples[i];
                        let sample_id = sample
                            .get("id")
                            .and_then(Value::as_str)
                            .context("eval sample missing id")?
                            .to_string();
                        if done.contains(&sample_id) {
                            eprintln!(
                                "[eval {}/{total} id={sample_id} skipped, checkpointed]",
                                i + 1
                            );
                            continue;
                        }
                        let messages = sample
                            .get("messages")
                            .context("eval sample missing messages")?;
                        let mut prompt = engine.render_chat(messages, kwargs)?;
                        let mut ids = engine.encode(&prompt, true)?;
                        let think =
                            thinking::apply(want, tags.as_ref(), &mut ids).map_err(|why| {
                                anyhow::anyhow!(
                                    "enable_thinking=true: {}: {why}",
                                    engine.capabilities().family
                                )
                            })?;
                        if think.by == thinking::By::Engine {
                            prompt.push_str(&engine.token_str(ids[ids.len() - 1]));
                        }
                        sw.event_line(&json!({
                            "kind": "eval_sample_start", "sample_id": sample_id, "prompt": prompt,
                        }));
                        let params = DecodeParams {
                            seed: rand::random::<u64>(),
                            temperature: (temperature > 0.0).then_some(temperature),
                            top_p: None,
                            top_k: None,
                            gumbel: false,
                            processors: Default::default(),
                            max_tokens,
                            eos_ids: engine.eos_ids().to_vec(),
                            stop_strings: Vec::new(),
                            readback: engine.readback(),
                            logits_digest: false,
                            prefill_tile: engine.prefill_tile(),
                            thinking: None,
                        };
                        let mut pending = String::new();
                        let started = Instant::now();
                        let aborts = &shared.aborts;
                        let epoch = &shared.abort_epoch;
                        // The set is consulted on the first fragment and
                        // again only after an abort was posted (`abort_epoch`).
                        let mut seen = u64::MAX;
                        let decoded = {
                            let sid = sample_id.clone();
                            // An abort ends the run through the decode seam's own hook,
                            // checked at every drained batch; so does a client that went
                            // away (the writer thread latches the failed write).
                            engine.complete_until(&ids, &params, &mut |frag| {
                                pending.push_str(frag);
                                if pending.len() >= EVAL_CHUNK_FLUSH_BYTES {
                                    sw.event_line(&json!({
                                        "kind": "eval_sample_chunk", "sample_id": sid, "delta": pending,
                                    }));
                                    pending.clear();
                                }
                                if sw.client_gone() {
                                    return false;
                                }
                                let now = epoch.load(Ordering::Acquire);
                                if now != seen {
                                    seen = now;
                                    return !lock(aborts).contains(&sid);
                                }
                                true
                            })
                        };
                        let decoded = decoded?;
                        if sw.client_gone() {
                            anyhow::bail!("client went away mid-eval");
                        }
                        let completion = decoded.text.clone();
                        let stopped_early = decoded.stop == Stop::Cancelled;
                        if stopped_early {
                            lock(aborts).remove(&sample_id);
                        }
                        eprintln!(
                            "[eval {}/{total} id={sample_id} {} tokens {:.1} ms {} readback={}]",
                            i + 1,
                            decoded.tokens.len(),
                            ms(started.elapsed()),
                            if stopped_early {
                                "stopped_early"
                            } else {
                                "done"
                            },
                            decoded.readback.name(),
                        );
                        sw.event_line(&json!({
                            "kind": "eval_sample_done", "sample_id": sample_id,
                            "completion": completion, "stopped_early": stopped_early,
                            "completion_tokens": decoded.tokens.len(),
                            "finish_reason": finish_reason(decoded.stop),
                        }));
                    }
                })
            })
            .collect();
        let mut first_err = None;
        for h in handles {
            match h.join() {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    first_err.get_or_insert(e);
                }
                Err(_) => {
                    first_err.get_or_insert(anyhow::anyhow!("an eval worker panicked"));
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => {
                sw.event_line(&json!({ "kind": "eval_done" }));
                Ok(())
            }
        }
    });
    if let Err(e) = &run {
        sw.event_line(&json!({ "kind": "eval_error", "error": format!("{e:#}") }));
    }
    let (w, write_err) = sw.finish()?;
    if let Some(e) = write_err {
        eprintln!("[eval: client went away: {e}]");
    }
    w.finish()?;
    run
}

/// Bytes of completion text buffered before an `eval_sample_chunk` goes out.
const EVAL_CHUNK_FLUSH_BYTES: usize = 256;

fn handle_abort(shared: &Shared, mut request: Request) -> Result<()> {
    let mut body = String::new();
    request.as_reader().read_to_string(&mut body).ok();
    let v: Value = serde_json::from_str(&body).unwrap_or(json!({}));
    match v.get("sample_id").and_then(Value::as_str) {
        Some(id) => {
            lock(&shared.aborts).insert(id.to_string());
            shared.abort_epoch.fetch_add(1, Ordering::Release);
            eprintln!("[eval abort requested for {id}]");
            respond_json(request, 200, &json!({ "sample_id": id }))
        }
        None => respond_error(request, 400, "field 'sample_id' must be a string"),
    }
}

// ---- responses ---------------------------------------------------------------

fn respond_json(request: Request, status: u16, body: &Value) -> Result<()> {
    let text = body.to_string();
    let response = Response::from_string(text)
        .with_status_code(StatusCode(status))
        .with_header(header("Content-Type", "application/json"));
    request.respond(response).context("writing the response")
}

fn respond_error(request: Request, status: u16, message: &str) -> Result<()> {
    let kind = match status {
        400 => "invalid_request_error",
        404 => "not_found_error",
        _ => "server_error",
    };
    respond_json(
        request,
        status,
        &json!({ "error": { "message": message, "type": kind, "code": status } }),
    )
}

fn header(name: &str, value: &str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("static header")
}

/// A chunked HTTP/1.1 response written by hand through the request's raw
/// writer, so the body can be produced on the decoding thread as the loop
/// drains tokens.
struct ChunkedWriter {
    w: Box<dyn Write + Send + 'static>,
}

impl ChunkedWriter {
    fn start(request: Request, content_type: &str) -> Result<Self> {
        let mut w = request.into_writer();
        write!(
            w,
            "HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nCache-Control: no-cache\r\n\
             Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )?;
        w.flush()?;
        Ok(Self { w })
    }

    fn raw(&mut self, s: &str) -> std::io::Result<()> {
        write!(self.w, "{:x}\r\n{s}\r\n", s.len())?;
        self.w.flush()
    }

    /// One SSE event.
    fn event(&mut self, v: &Value) -> std::io::Result<()> {
        self.raw(&format!("data: {v}\n\n"))
    }

    /// One JSONL line.
    fn finish(mut self) -> Result<()> {
        self.w.write_all(b"0\r\n\r\n")?;
        self.w.flush()?;
        Ok(())
    }
}

// ---- small helpers -----------------------------------------------------------

/// A [`ChunkedWriter`] driven from its own thread, so the thread running the
/// decode never blocks on the socket (issue #702 reopened: `lfm2-chat` moved
/// its print off the hot thread in `352e8129`; this is the same seam for the
/// server). Payloads are complete wire fragments (`data: …\n\n`, or a JSONL
/// line) queued through an unbounded channel — the queue is the slack a slow
/// client gets, the ring's own slack window being the other half. A write
/// error is latched and reported by [`Self::client_gone`], which the decode's
/// callback reads to stop the run; [`Self::finish`] drains the queue, joins
/// the thread and hands the writer back for the trailer.
///
/// **The per-token text does not go through the channel** when the writer
/// is started with [`Self::start_with_deltas`]: a `send` to the parked writer
/// is a wake — a syscall — per fragment, and the payload a `format!` and a
/// `json!` built on the decode thread. The decode thread appends the bare
/// text to a [`DeltaBuffer`] instead (`sink.rs`: no wait, no syscall, no
/// allocation once grown), and this thread takes what accumulated every
/// [`DRAIN_PERIOD`] and wraps it in one event — a delta of several tokens,
/// which the OpenAI stream allows.
struct StreamWriter {
    tx: Option<std::sync::mpsc::Sender<String>>,
    gone: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<(ChunkedWriter, Option<std::io::Error>)>>,
    deltas: Option<Arc<DeltaBuffer>>,
    /// The deltas' wire wrapping, shared with the thread: [`Self::send`]
    /// wraps what is pending itself so a control event keeps its place.
    wrap: Option<Wrap>,
    /// The producer's spill (`DeltaBuffer::push`); only the producer locks it.
    spill: Mutex<String>,
}

type Wrap = Arc<dyn Fn(&str) -> String + Send + Sync>;

impl StreamWriter {
    fn start(w: ChunkedWriter) -> Self {
        Self::spawn(w, None)
    }

    /// [`Self::start`] with a delta buffer: [`Self::delta`] hands text over,
    /// and `wrap` turns each drained batch into its wire event on this
    /// writer's thread.
    fn start_with_deltas(
        w: ChunkedWriter,
        wrap: impl Fn(&str) -> String + Send + Sync + 'static,
    ) -> Self {
        Self::spawn(w, Some(Arc::new(wrap)))
    }

    fn spawn(mut w: ChunkedWriter, wrap: Option<Wrap>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let gone = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&gone);
        let deltas = wrap.as_ref().map(|_| DeltaBuffer::new());
        let buf = deltas.clone();
        let thread_wrap = wrap.clone();
        let thread = std::thread::spawn(move || {
            let mut err: Option<std::io::Error> = None;
            let mut local = String::with_capacity(4096);
            let write = |w: &mut ChunkedWriter, payload: &str, err: &mut Option<std::io::Error>| {
                if err.is_some() {
                    return;
                }
                if let Err(e) = w.raw(payload) {
                    flag.store(true, Ordering::Release);
                    *err = Some(e);
                }
            };
            // The text that accumulated, as one event. (Text pushed before a
            // control event reaches the channel ahead of it: `send` takes it.)
            let mut drain = |w: &mut ChunkedWriter, err: &mut Option<std::io::Error>| {
                if let (Some(b), Some(wrap)) = (&buf, &thread_wrap) {
                    if b.take_into(&mut local) {
                        write(w, &wrap(&local), err);
                    }
                }
            };
            loop {
                let next = if buf.is_some() {
                    rx.recv_timeout(DRAIN_PERIOD)
                } else {
                    rx.recv()
                        .map_err(|_| std::sync::mpsc::RecvTimeoutError::Disconnected)
                };
                match next {
                    Ok(payload) => write(&mut w, &payload, &mut err),
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => drain(&mut w, &mut err),
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        drain(&mut w, &mut err);
                        break;
                    }
                }
            }
            (w, err)
        });
        Self {
            tx: Some(tx),
            gone,
            thread: Some(thread),
            deltas,
            wrap,
            spill: Mutex::new(String::new()),
        }
    }

    /// Hand text to the writer (the decode thread's per-fragment call on a
    /// writer started with deltas): no wait, no syscall, no allocation.
    fn delta(&self, text: &str) {
        if let Some(b) = &self.deltas {
            b.push(&mut lock(&self.spill), text);
        }
    }

    /// Everything handed over so far, wrapped, queued ahead of what comes
    /// next on the channel. Off the per-token path (a control event).
    fn queue_pending(&self) {
        if let (Some(b), Some(wrap), Some(tx)) = (&self.deltas, &self.wrap, &self.tx) {
            b.flush(&mut lock(&self.spill));
            let mut text = String::new();
            if b.take_into(&mut text) {
                let _ = tx.send(wrap(&text));
            }
        }
    }

    /// Queue one SSE event.
    fn event(&self, v: &Value) {
        self.send(format!("data: {v}\n\n"));
    }

    /// Queue one JSONL line.
    fn event_line(&self, v: &Value) {
        self.send(format!("{v}\n"));
    }

    fn send(&self, payload: String) {
        self.queue_pending();
        if let Some(tx) = &self.tx {
            // A closed receiver means the thread is gone, which `gone` reports.
            let _ = tx.send(payload);
        }
    }

    /// The client stopped reading: a write failed on the writer thread.
    fn client_gone(&self) -> bool {
        self.gone.load(Ordering::Acquire)
    }

    /// Drain what is queued, stop the thread, and return the writer with the
    /// first write error, if any.
    fn finish(mut self) -> Result<(ChunkedWriter, Option<std::io::Error>)> {
        if let Some(b) = &self.deltas {
            b.flush(&mut lock(&self.spill));
        }
        drop(self.tx.take());
        let handle = self
            .thread
            .take()
            .context("stream writer already finished")?;
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("the stream writer thread panicked"))
    }
}

impl Drop for StreamWriter {
    fn drop(&mut self) {
        if let Some(b) = &self.deltas {
            b.flush(&mut lock(&self.spill));
        }
        drop(self.tx.take());
        if let Some(h) = self.thread.take() {
            let _ = h.join();
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn short_id() -> String {
    format!("{:016x}", rand::random::<u64>())
}

/// The per-request vision report (`Served::vision_report`: the model's
/// tensors, whether the tower is loaded, the image census) as a `vision`
/// field beside `timings`, and on stderr. Nothing for a family without vision.
fn attach_vision(shared: &Shared, body: &mut Value) {
    let report = shared.engine.vision_report();
    if !report.is_null() {
        eprintln!("[vision: {report}]");
        body["vision"] = report;
    }
}

fn finish_reason(stop: Stop) -> &'static str {
    match stop {
        Stop::Eos | Stop::StopString => "stop",
        Stop::Length => "length",
        Stop::Cancelled => "cancelled",
    }
}

fn usage(prompt: usize, completion: usize) -> Value {
    json!({
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    })
}

/// The engine's own brackets beside the request's wall clock: the prompt's
/// submit time, the per-step host clock and the arm that ran. `predicted_ms`
/// is the sum of the loop's steps — the decode window, prefill excluded.
fn timings(prompt: usize, facts: &PromptFacts, d: &Decoded, elapsed: std::time::Duration) -> Value {
    let st = &d.stats;
    // Release builds compile the per-step clock out (`step_ns` is empty, since
    // `6c33cd89`), so the decode's window is the request's wall minus the
    // prompt's submit — the same coarse figure `lfm2-chat` reports.
    let predicted_ms = if st.step_ns.is_empty() {
        (ms(elapsed) - st.prefill_ns as f64 / 1e6).max(0.0)
    } else {
        st.step_ns.iter().sum::<u64>() as f64 / 1e6
    };
    let n = d.tokens.len();
    json!({
        "prompt_n": prompt,
        "prompt_submit_ms": st.prefill_ns as f64 / 1e6,
        "predicted_n": n,
        "predicted_ms": predicted_ms,
        "predicted_per_second": if predicted_ms > 0.0 { n as f64 / (predicted_ms / 1e3) } else { 0.0 },
        "total_ms": ms(elapsed),
        "readback": d.readback.name(),
        "reuse": format!("{:?}", d.reuse),
        "prefilled": d.prefilled,
        "commits": st.commits,
        "ring_publishes": st.readback.ring_publishes,
        "poll_fallbacks": st.readback.poll_fallbacks,
        "publish_parks": st.readback.publish_parks,
        "overrun": st.overrun,
        "eos_gate": super::common::eos_gate_name(st.eos_gate),
        "host_fill": super::common::host_fill_name(st.host_feed.as_ref()),
        // The generated ids, so a gate can compare a run against a recorded
        // token digest without a second path to the model.
        "tokens": d.tokens,
        "prompt_ids_sha256": facts.ids_sha256,
        // The template's variable as the prompt reflects it; null for its
        // default (a value the template would not act on is a 400).
        "reasoning_effort": facts.reasoning_effort,
        // Whether the prompt leaves a thinking block open, and who decided:
        // the template (its switch or its own behaviour) or the engine,
        // which closed the block the template opened.
        "thinking": facts.thinking.map(Thinking::name),
        "thinking_by": facts.thinking.map(|t| t.by.name()),
        // The token budgets: what was asked, whether the thinking one ran
        // (`thinking_budget_ignored` says why not) and what it did, and what
        // the prompt one dropped before the prefill.
        "thinking_budget": facts.thinking_budget,
        "thinking_budget_ignored": facts.thinking_ignored,
        "thinking_tokens": d.thinking.map(|t| t.thought),
        "thinking_budget_fired": d.thinking.map(|t| t.fired),
        "prompt_budget": facts.prompt_budget,
        "prompt_tokens_dropped": facts.dropped.tokens,
        "prompt_messages_dropped": facts.dropped.messages,
    })
}

fn report(endpoint: &str, prompt: usize, facts: &PromptFacts, d: &Decoded, total_ms: f64) {
    let st = &d.stats;
    let mut steps: Vec<u64> = st.step_ns.clone();
    steps.sort_unstable();
    let steady = if steps.is_empty() {
        0.0
    } else {
        1e9 / steps[steps.len() / 2] as f64
    };
    eprintln!(
        "[{endpoint}: prompt_tokens={prompt} completion_tokens={} stop={:?} total {total_ms:.1} ms, \
         {steady:.1} tok/s steady (median step), readback={} reuse={:?} prefilled={} \
         commits={} poll_fallbacks={} overrun={} eos_gate={} host_fill={} prompt_ids_sha256={} \
         reasoning_effort={} think={} think_by={} thinking_budget={} thought={} budget_fired={} \
         prompt_budget={} prompt_dropped={}]",
        d.tokens.len(),
        d.stop,
        d.readback.name(),
        d.reuse,
        d.prefilled,
        st.commits,
        st.readback.poll_fallbacks,
        st.overrun,
        super::common::eos_gate_name(st.eos_gate),
        super::common::host_fill_name(st.host_feed.as_ref()),
        facts.ids_sha256,
        facts.reasoning_effort.as_str().unwrap_or("default"),
        facts.thinking.map_or("none", Thinking::name),
        facts.thinking.map_or("none", |t| t.by.name()),
        facts
            .thinking_budget
            .map_or_else(|| "none".into(), |n| n.to_string()),
        d.thinking
            .map_or_else(|| "none".into(), |t| t.thought.to_string()),
        d.thinking
            .map_or_else(|| "none".into(), |t| t.fired.to_string()),
        facts
            .prompt_budget
            .map_or_else(|| "none".into(), |n| n.to_string()),
        facts.dropped.tokens,
    );
}

/// What a request's prompt was: the digest of its ids (`lloom chat`'s
/// `prompt_ids_sha256`) and the engine's template variables as the prompt
/// reflects them.
#[derive(Default)]
struct PromptFacts {
    ids_sha256: String,
    reasoning_effort: Value,
    /// The prompt's thinking as it runs; `None` where no template rendered it.
    thinking: Option<Thinking>,
    /// The thinking budget the request asked for, and why it was ignored
    /// (a template or a prompt with no open block) when it was.
    thinking_budget: Option<usize>,
    thinking_ignored: Option<&'static str>,
    /// The prompt budget, and what it dropped.
    prompt_budget: Option<usize>,
    dropped: Dropped,
}

impl PromptFacts {
    fn of(ids: &[u32]) -> Self {
        Self {
            ids_sha256: super::chat::tokens_sha256(ids),
            ..Self::default()
        }
    }

    /// Record the request's budgets against the prompt that runs, and arm
    /// the thinking budget on `params` where the prompt leaves a block open.
    /// Where it does not, the request is served without it and says so —
    /// clients send the field whatever the model, as with `reasoning_effort`.
    fn apply_budgets(
        &mut self,
        engine: &dyn Served,
        endpoint: &str,
        wanted: &Budgets,
        dropped: Dropped,
        ids: &[u32],
        params: &mut DecodeParams,
    ) -> Handled<()> {
        self.prompt_budget = wanted.prompt.map(|(k, _)| k);
        self.dropped = dropped;
        let Some(n) = wanted.thinking else {
            return Ok(());
        };
        self.thinking_budget = Some(n);
        let message = match &wanted.message {
            Some(m) => engine.encode(m, false)?,
            None => Vec::new(),
        };
        let refused = engine.thinking_budget_refused();
        match budgets::thinking(refused, engine.think_tags().as_ref(), ids, n, &message)
            .map_err(|e| bad(format!("{e:#}")))?
        {
            Ok(b) => params.thinking = Some(b),
            Err(why) => {
                eprintln!(
                    "[{endpoint}: {}: thinking_token_budget={n} ignored: {why}]",
                    engine.capabilities().family
                );
                self.thinking_ignored = Some(why);
            }
        }
        Ok(())
    }
}

/// A chat request's prompt through the checkpoint's template. A value the
/// template rejects, data it fails on, and a `reasoning_effort` it would not
/// act on (not read, or no effect under the other variables) are the
/// client's 400: an effort that is not available is refused, whichever
/// spelling carried it.
fn render_request(
    engine: &dyn Served,
    endpoint: &str,
    messages: &Value,
    kwargs: &Map<String, Value>,
) -> Handled<(String, PromptFacts)> {
    let rendered = engine
        .render_checked(messages, kwargs, &[REASONING_EFFORT])
        .map_err(template_failure)?;
    let family = engine.capabilities().family;
    if let Some(i) = rendered.ignored.first() {
        let why = if i.reason == NOT_READ {
            format!("{family}'s chat template does not read {REASONING_EFFORT}")
        } else {
            format!(
                "{family}'s chat template renders the same prompt for any {REASONING_EFFORT} with \
                 {ENABLE_THINKING}={}",
                kwargs.get(ENABLE_THINKING).unwrap_or(&Value::Null)
            )
        };
        eprintln!(
            "[{endpoint}: {REASONING_EFFORT}={} refused: {why}]",
            i.value
        );
        return Err(bad(format!("{REASONING_EFFORT}={}: {why}", i.value)));
    }
    let facts = PromptFacts {
        reasoning_effort: kwargs.get(REASONING_EFFORT).cloned().unwrap_or(Value::Null),
        ..PromptFacts::default()
    };
    Ok((rendered.text, facts))
}

/// A render failure: the client's 400 when the template refused or failed
/// on the request's own data, the engine's 500 otherwise.
fn template_failure(e: anyhow::Error) -> Failure {
    if let Some(r) = e.downcast_ref::<Rejected>() {
        bad(r.to_string())
    } else if let Some(r) = e.downcast_ref::<Unrenderable>() {
        bad(r.to_string())
    } else {
        Failure::Internal(e)
    }
}

/// A request's `enable_thinking` on its encoded prompt (`thinking::apply`);
/// thinking on where the prompt cannot think is the client's 400.
fn apply_thinking(
    engine: &dyn Served,
    want: Option<bool>,
    tags: Option<&super::template::ThinkTags>,
    ids: &mut Vec<u32>,
) -> Handled<Thinking> {
    thinking::apply(want, tags, ids).map_err(|why| {
        bad(format!(
            "{ENABLE_THINKING}=true: {}: {why}",
            engine.capabilities().family
        ))
    })
}

fn models_body(model_name: &str, engine: &dyn Served) -> Value {
    json!({
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": now(),
            "owned_by": "lloom",
            "load_warnings": load_warnings_json(engine),
        }],
    })
}

/// The load's degradations as `[{kind, message}]` (`Served::load_warnings`).
fn load_warnings_json(engine: &dyn Served) -> Value {
    engine
        .load_warnings()
        .iter()
        .map(|w| json!({ "kind": w.kind.name(), "message": w.message }))
        .collect()
}

// ---- `lloom bench` ---------------------------------------------------------------

/// `lloom bench`: the benchmark cells the pipette sidecar drives over HTTP
/// (`/prefill_throughput`, `/decode_throughput`, `/end_to_end_latency`,
/// `/max_memory_usage`), run in-process on the same engine and the same
/// option groups — one `BENCH` line per cell per run, the resolved arms last.
/// `--mode reuse | concurrent` runs a multi-sequence measurement mode instead
/// (`modes.rs`, LFM2): its per-sequence lines and one `RESULT` line.
#[derive(Parser)]
#[command(
    name = "lloom bench",
    about = "The benchmark cells (the pipette sidecar's) in-process, both families; or a multi-sequence mode",
    after_help = super::OPTIONS_HELP
)]
pub struct BenchArgs {
    #[command(flatten)]
    pub common: super::CommonArgs,
    #[command(flatten)]
    pub bench: crate::options::BenchConfigArgs,
    #[command(flatten)]
    pub scheduler: crate::options::SchedulerConfigArgs,
    #[command(flatten)]
    pub prefix: crate::options::PrefixConfigArgs,
    #[command(flatten)]
    pub report: crate::options::ReportConfigArgs,
    #[command(flatten)]
    pub pipette: crate::options::PipetteConfigArgs,
}

impl BenchArgs {
    /// `bench`'s own flags, as the top layer.
    pub fn front(&self, o: &mut crate::EngineOptions) -> std::result::Result<(), String> {
        self.bench.apply(&mut o.bench)?;
        self.scheduler.apply(&mut o.scheduler)?;
        self.prefix.apply(&mut o.prefix)?;
        self.report.apply(&mut o.report)?;
        self.pipette.apply(&mut o.server.pipette)
    }
}

/// One cell of `--cells`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cell {
    Prefill { p: usize },
    Decode { p: usize, d: usize },
    EndToEnd { p: usize, d: usize },
    Memory { p: usize, d: usize },
}

impl Cell {
    pub fn parse_list(s: &str) -> Result<Vec<Cell>> {
        s.split(',')
            .map(str::trim)
            .filter(|c| !c.is_empty())
            .map(Cell::parse)
            .collect()
    }

    pub fn parse(s: &str) -> Result<Cell> {
        let parts: Vec<&str> = s.split(':').collect();
        let num = |i: usize| -> Result<usize> {
            parts
                .get(i)
                .ok_or_else(|| anyhow::anyhow!("cell `{s}`: missing a size"))?
                .parse::<usize>()
                .with_context(|| format!("cell `{s}`: not a size"))
        };
        Ok(match parts[0] {
            "prefill" => Cell::Prefill { p: num(1)? },
            "decode" => Cell::Decode {
                p: num(1)?,
                d: num(2)?,
            },
            "e2e" | "end_to_end" => Cell::EndToEnd {
                p: num(1)?,
                d: num(2)?,
            },
            "memory" | "max_memory" => Cell::Memory {
                p: num(1)?,
                d: num(2)?,
            },
            other => anyhow::bail!("cell `{other}`: prefill:P | decode:P:D | e2e:P:D | memory:P:D"),
        })
    }

    pub fn name(&self) -> String {
        match self {
            Cell::Prefill { p } => format!("prefill_throughput_{p}"),
            Cell::Decode { p, d } => format!("decode_throughput_{p}_{d}"),
            Cell::EndToEnd { p, d } => format!("end_to_end_latency_{p}_{d}"),
            Cell::Memory { p, d } => format!("max_memory_usage_{p}_{d}"),
        }
    }
}

pub fn bench(args: BenchArgs) -> Result<()> {
    let mut options = super::resolve_options(&args.common, &|o| args.front(o))?;
    super::report::arm_trace();
    let mode = match options.bench.mode.as_str() {
        "reuse" => Some(super::modes::Mode::Reuse),
        "concurrent" => Some(super::modes::Mode::Concurrent),
        _ => None,
    };
    if let Some(mode) = mode {
        return bench_mode(&args, mode, options);
    }
    let cells = Cell::parse_list(&options.bench.cells)?;
    if cells.is_empty() {
        anyhow::bail!("--cells names no cell");
    }
    let seed_text = seed_text(options.server.pipette.prompt_seed_file.as_deref())?;
    let batched = options.scheduler.max_sequences > 1;
    if batched && options.decode.prefill.chunk == 0 {
        options.decode.prefill.chunk = 512;
    }
    let load_start = Instant::now();
    let shape = super::Shape::of(&options, false, seed_text);
    let (engine, model_path) = super::load_engine(&options.load, shape)?;
    let model_name = engine.model_name().to_string();
    let mut sampling = options.sampling.clone();
    sampling.apply_checkpoint(engine.generation_config().as_ref());
    sampling
        .check()
        .map_err(|e| anyhow::anyhow!("the checkpoint's generation_config.json: {e}"))?;
    eprintln!(
        "[loaded {} from {} in {:.2}s on {:?}; kv_capacity={} readback={} {}]",
        model_name,
        model_path.display(),
        load_start.elapsed().as_secs_f64(),
        engine.device(),
        engine.kv_capacity(),
        engine.readback().name(),
        engine.describe(),
    );
    eprintln!("[capabilities: {}]", engine.capabilities().line());
    // The cells are the server's handlers; they read the engine through
    // `Shared`, whose listener is unused here (a free loopback port).
    let server = Server::http("127.0.0.1:0").map_err(|e| anyhow::anyhow!("binding: {e}"))?;
    let shared = Shared {
        engine,
        aborts: Mutex::new(HashSet::new()),
        abort_epoch: AtomicU64::new(0),
        server,
        model_name: model_name.clone(),
        sampling,
    };
    let batch = options.scheduler.max_sequences.max(1);
    for run in 0..options.bench.runs.max(1) {
        for cell in &cells {
            let started = Instant::now();
            let out = match cell {
                Cell::Prefill { p } => {
                    prefill_throughput(&shared, &json!({ "prompt_tokens": p, "batch": batch }))
                }
                Cell::Decode { p, d } => decode_throughput(
                    &shared,
                    &json!({ "prompt_tokens": p, "decode_tokens": d, "batch": batch }),
                ),
                Cell::Memory { p, d } => max_memory_usage(
                    &shared,
                    &json!({ "prompt_tokens": p, "decode_tokens": d, "batch": batch }),
                ),
                Cell::EndToEnd { p, d } => {
                    // The seed tiled to `p` tokens, as text: what the sidecar
                    // posts as `prompt`.
                    let ids = shared.engine.seed_prompt_row(*p, 0);
                    let prompt: String = ids.iter().map(|&t| shared.engine.token_str(t)).collect();
                    end_to_end_latency(
                        &shared,
                        &json!({ "prompt": prompt, "decode_tokens": d, "batch": batch }),
                    )
                }
            };
            let value = match out {
                Ok(v) => v,
                Err(Failure::BadRequest(m)) | Err(Failure::NotFound(m)) => {
                    anyhow::bail!("{}: {m}", cell.name())
                }
                Err(Failure::Internal(e)) => return Err(e.context(cell.name())),
            };
            let mut fields: Vec<String> = Vec::new();
            if let Some(map) = value.as_object() {
                for (k, v) in map {
                    match v {
                        Value::Number(n) => fields.push(format!("{k}={n}")),
                        Value::String(s) => fields.push(format!("{k}={s}")),
                        Value::Bool(b) => fields.push(format!("{k}={b}")),
                        _ => {}
                    }
                }
            }
            println!(
                "BENCH run={run} cell={} wall_ms={:.1} {}",
                cell.name(),
                ms(started.elapsed()),
                fields.join(" ")
            );
        }
    }
    println!(
        "BENCH arms model={} readback={} batch={} {} capabilities: {}",
        model_name,
        shared.engine.readback().name(),
        batch,
        shared.engine.describe(),
        shared.engine.capabilities().line(),
    );
    shared.engine.shutdown();
    Ok(())
}

/// `lloom bench --mode reuse | concurrent`: resolve the mode's knobs, load
/// the engine at one sequence on the arms the mode needs, and run it.
fn bench_mode(
    args: &BenchArgs,
    mode: super::modes::Mode,
    mut options: crate::EngineOptions,
) -> Result<()> {
    let spec = resolve_mode(args, mode, &mut options)?;
    let load_start = Instant::now();
    let mut shape = super::Shape::of(&options, false, DEFAULT_SEED_TEXT.to_string());
    shape.reuse_prefix = false;
    shape.batch = 1;
    shape.mem_sample = false;
    shape.rows = "auto".into();
    shape.share_prefix = false;
    let mem_sample = options.report.memory_sample;
    let (engine, model_path) = super::load_engine(&options.load, shape)?;
    eprintln!(
        "[loaded {} from {} in {:.2}s on {:?}; {}]",
        engine.model_name(),
        model_path.display(),
        load_start.elapsed().as_secs_f64(),
        engine.device(),
        engine.describe(),
    );
    let gauge = super::common::MemGauge::start(engine.device(), mem_sample);
    let out = engine.run_mode(&spec);
    let mem = gauge.finish(engine.device());
    if mem_sample {
        super::report::print_mempeak(&mem);
    }
    #[cfg(feature = "metal")]
    if let (Some(i), lloom_tensor::Device::Metal(dev)) = (engine.installed(), engine.device()) {
        if i.pass_plans {
            super::report::print_pass_plans(dev, i.pass_widths);
        }
    }
    engine.shutdown();
    out
}

/// A mode's knobs from the option groups and `lloom bench`'s own flags, with
/// the refusals made before anything loads: a flag the mode does not read is
/// refused rather than worn by a run that ignored it (§7.1a-i). Under `--rows
/// pool*` the model group's preset becomes the pool's (every sequence is a
/// slot of one paged pool, which the `auto` rule does not pick at `--batch
/// 1`).
fn resolve_mode(
    args: &BenchArgs,
    mode: super::modes::Mode,
    options: &mut crate::EngineOptions,
) -> Result<super::modes::ModeSpec> {
    let b = options.bench.clone();
    if options.speculative.method == "auto" {
        anyhow::bail!(
            "speculative.method = auto is resolved when the model loads; lloom bench's modes \
             are chosen before that, so name the method (dspark | mtp | none)"
        );
    }
    let dspark = options.speculative.method == "dspark";
    let decode = &options.decode;
    // The flag only: the env alias is process-wide, set once for a gate
    // script's every arm, most of which have no rows gate to read it.
    let rows_gate = mode == super::modes::Mode::Concurrent && b.batched;
    if args.bench.fail_on_gate.is_some() && !rows_gate {
        anyhow::bail!(
            "--fail-on-gate is read by the rows gate, `--mode concurrent --batched`; \
             --mode {}{} prints no gate verdict",
            mode.name(),
            if mode == super::modes::Mode::Concurrent {
                " without --batched"
            } else {
                ""
            }
        );
    }
    if options.scheduler.max_sequences > 1 {
        anyhow::bail!(
            "--mode {} drives its own sequences (`--sequences`); --batch starts the serving \
             scheduler, which is `lloom serve --batch N` / the cells",
            mode.name()
        );
    }
    if decode.prefill.spark.is_some() {
        anyhow::bail!(
            "--spark is not read by --mode {} (its drivers prefill per sequence); it is honoured \
             by `lloom chat` and the cells. Use --prefill-ubatch for the modes",
            mode.name()
        );
    }
    if b.batched && dspark {
        anyhow::bail!(
            "--batched drives the K=1 batched-forward body; a batched verify is not built \
             (speculative-pipeline-scheduler-design.md §4b). Pick one of --batched / --dspark"
        );
    }
    if options.executor.replay || options.executor.arena {
        anyhow::bail!(
            "--replay/--arena record one sequence's step; they do not compose with --mode {} \
             (the modes drive many)",
            mode.name()
        );
    }
    if dspark && decode.prefill.chunk > 0 {
        anyhow::bail!(
            "--prefill-chunk is not read by --dspark (its seed is `Session::prefill_streamed`, the \
             spark-then-remainder ramp); run it on the K=1 interleave or --batched"
        );
    }
    let rows = super::lfm2::RowsArm::parse(
        if options.scheduler.rows == "auto" {
            "private"
        } else {
            &options.scheduler.rows
        },
        true,
    )?;
    if rows.pooled() {
        let want = if dspark { "pool-spec" } else { "pool" };
        match options.model.cache.as_str() {
            "auto" => options.model.cache = want.into(),
            c if c == want => {}
            other => anyhow::bail!(
                "--rows {} makes every sequence a slot of one paged pool; --cache {other} is not \
                 its preset ({want})",
                rows.name()
            ),
        }
    }
    let decode = &options.decode;
    let readback = match decode.readback.spelling() {
        "auto" => {
            eprintln!(
                "[mode: --readback auto resolves per request; the modes take one arm per \
                 sequence: ring-nonblocking]"
            );
            "ring-nonblocking".to_string()
        }
        s => s.to_string(),
    };
    Ok(super::modes::ModeSpec {
        mode,
        max_tokens: options.sampling.max_tokens,
        readback,
        lookahead: decode.readback.lookahead,
        commit_budget: decode.readback.commit_budget,
        sequences: b.sequences,
        chunk: b.chunk,
        prefill_ubatch: decode.prefill.ubatch,
        // The speculative drivers' seed does not read the tile.
        prefill_tile: if dspark { 0 } else { decode.prefill.tile },
        prefill_chunk: decode.prefill.chunk,
        pass_widths: decode.prefill.pass_widths,
        stagger: b.stagger,
        long_prompt_tokens: b.long_prompt_tokens,
        dspark,
        batched: b.batched,
        rows,
        check_independence: b.independence_check,
        skip_solo: b.skip_solo,
        fail_on_gate: b.fail_on_gate,
        first: b.first.clone(),
        timeline: b.timeline.clone(),
    })
}

#[cfg(test)]
mod bench_mode_tests {
    use super::{resolve_mode, BenchArgs};
    use crate::app::modes::Mode;
    use clap::Parser;

    fn resolve(
        argv: &[&str],
    ) -> anyhow::Result<(super::super::modes::ModeSpec, crate::EngineOptions)> {
        let mut full = vec!["lloom bench"];
        full.extend_from_slice(argv);
        let args = BenchArgs::try_parse_from(full)?;
        // Without the env layer: `modes.rs`'s env tests set variables, and a
        // process-wide read here would race them.
        let front = |o: &mut crate::EngineOptions| args.front(o);
        let (mut o, _) = super::super::resolve_layers(&args.common, &front, &|_| None, false)?;
        let mode = match o.bench.mode.as_str() {
            "reuse" => Mode::Reuse,
            _ => Mode::Concurrent,
        };
        let spec = resolve_mode(&args, mode, &mut o)?;
        Ok((spec, o))
    }

    /// `--fail-on-gate` reaches the rows gate, off by default; a mode with
    /// no rows gate refuses the flag rather than wear it.
    #[test]
    fn the_gate_opt_in_reaches_the_rows_gate_only() {
        let (s, _) = resolve(&["--mode", "concurrent", "--batched"]).unwrap();
        assert!(!s.fail_on_gate, "a failing gate exits 0 by default");
        let (s, _) = resolve(&["--mode", "concurrent", "--batched", "--fail-on-gate"]).unwrap();
        assert!(s.fail_on_gate);
        let (s, _) = resolve(&[
            "--mode",
            "concurrent",
            "--batched",
            "--skip-solo",
            "--fail-on-gate",
        ])
        .unwrap();
        assert!(
            s.fail_on_gate,
            "the re-batch still carries a gate without the solo arm"
        );
        for (argv, want) in [
            (
                &["--mode", "concurrent", "--fail-on-gate"][..],
                "--mode concurrent without --batched prints no gate verdict",
            ),
            (
                &["--mode", "reuse", "--fail-on-gate=false"],
                "--mode reuse prints no gate verdict",
            ),
        ] {
            let e = refusal(argv);
            assert!(e.contains(want), "{argv:?}: {e}");
        }
    }

    fn refusal(argv: &[&str]) -> String {
        match resolve(argv) {
            Ok(_) => panic!("{argv:?}: accepted"),
            Err(e) => e.to_string(),
        }
    }

    /// The knobs reach the mode from the groups and the mode's own flags.
    #[test]
    fn the_knobs_reach_the_mode() {
        let (s, _) = resolve(&[
            "--mode",
            "concurrent",
            "--batched",
            "--sequences",
            "4",
            "--stagger",
            "8",
            "--long-prompt-tokens",
            "300",
            "--prefill-chunk",
            "64",
            "--prefill-tile",
            "0",
            "--readback",
            "ring-hostdrain",
            "--lookahead",
            "0",
            "--commit-budget",
            "steps:4",
            "-n",
            "48",
            "--first",
            "batched",
            "--skip-solo",
            "--no-independence-check",
            "--pass-widths",
            "ladder",
        ])
        .unwrap();
        assert_eq!(s.mode, Mode::Concurrent);
        assert!(s.batched && s.skip_solo && !s.check_independence);
        assert_eq!((s.sequences, s.stagger, s.long_prompt_tokens), (4, 8, 300));
        assert_eq!((s.prefill_chunk, s.prefill_tile, s.max_tokens), (64, 0, 48));
        assert_eq!((s.readback.as_str(), s.lookahead), ("ring-hostdrain", 0));
        assert_eq!(s.commit_budget.period(), 4);
        assert_eq!(s.first, "batched");
        assert_eq!(s.pass_widths, lloom::nn::PassWidths::Ladder);
        assert_eq!(
            s.rows,
            crate::app::lfm2::RowsArm::Private,
            "auto is a cache each"
        );
    }

    /// `auto` is resolved per request; a mode takes one arm per sequence.
    #[test]
    fn auto_readback_is_the_nonblocking_ring() {
        let (s, _) = resolve(&["--mode", "concurrent"]).unwrap();
        assert_eq!(s.readback, "ring-nonblocking");
        let (s, _) = resolve(&["--mode", "reuse", "--readback", "blocking"]).unwrap();
        assert_eq!(s.readback, "blocking");
    }

    /// Every sequence a slot of one pool needs the pool's preset: `auto`
    /// takes it (the speculative one under `--dspark`), a named other preset
    /// is refused by name.
    #[test]
    fn the_pool_rows_take_the_pool_preset() {
        let (s, o) = resolve(&["--mode", "concurrent", "--batched", "--rows", "pool"]).unwrap();
        assert_eq!(s.rows, crate::app::lfm2::RowsArm::Pool);
        assert_eq!(o.model.cache, "pool");
        let (_, o) =
            resolve(&["--mode", "concurrent", "--dspark", "--rows", "pool-per-row"]).unwrap();
        assert_eq!(o.model.cache, "pool-spec");
        let e = refusal(&["--mode", "concurrent", "--rows", "pool", "--cache", "crown"]);
        assert!(e.contains("--cache crown is not its preset (pool)"), "{e}");
    }

    /// What a mode does not read is refused, not worn.
    #[test]
    fn the_unread_knobs_are_refused() {
        for (argv, want) in [
            (
                &["--mode", "concurrent", "--batch", "2"][..],
                "--batch starts the serving scheduler",
            ),
            (
                &["--mode", "concurrent", "--spark", "16"],
                "--spark is not read",
            ),
            (
                &["--mode", "concurrent", "--batched", "--dspark"],
                "a batched verify is not built",
            ),
            (
                &["--mode", "reuse", "--replay"],
                "do not compose with --mode reuse",
            ),
            (
                &["--mode", "concurrent", "--arena"],
                "do not compose with --mode concurrent",
            ),
            (
                &["--mode", "concurrent", "--dspark", "--prefill-chunk", "64"],
                "--prefill-chunk is not read by --dspark",
            ),
        ] {
            let e = refusal(argv);
            assert!(e.contains(want), "{argv:?}: {e}");
        }
        let (s, _) =
            resolve(&["--mode", "concurrent", "--dspark", "--prefill-tile", "256"]).unwrap();
        assert_eq!(s.prefill_tile, 0, "the speculative seed reads no tile");
    }
}

#[cfg(test)]
mod bench_cells_tests {
    use super::Cell;

    #[test]
    fn the_cells_parse_and_name_themselves() {
        let cells =
            Cell::parse_list("prefill:512, decode:512:100,e2e:8:64,memory:512:100").unwrap();
        assert_eq!(
            cells.iter().map(Cell::name).collect::<Vec<_>>(),
            vec![
                "prefill_throughput_512",
                "decode_throughput_512_100",
                "end_to_end_latency_8_64",
                "max_memory_usage_512_100"
            ]
        );
        assert!(Cell::parse("decode:512").is_err(), "a decode cell needs D");
        assert!(Cell::parse("ttft:8").is_err());
        assert!(Cell::parse("prefill:many").is_err());
    }
}

#[cfg(test)]
mod request_processors_tests {
    use super::{request_processors, Failure};
    use serde_json::json;

    fn refusal(v: serde_json::Value) -> String {
        match request_processors(&crate::SamplingConfig::default(), &v) {
            Err(Failure::BadRequest(m)) => m,
            Err(_) => panic!("{v}: not a 400"),
            Ok(p) => panic!("{v}: accepted as {p:?}"),
        }
    }

    /// The OpenAI / vLLM fields land on the processors; an absent or null
    /// field keeps the process's default, which is neutral unless set.
    #[test]
    fn the_request_fields_reach_the_processors() {
        let v = json!({
            "repetition_penalty": 1.1, "presence_penalty": 0.5, "frequency_penalty": -0.25,
            "min_p": 0.05, "logit_bias": {"42": -100, "7": 2.5},
        });
        let p = request_processors(&crate::SamplingConfig::default(), &v)
            .ok()
            .unwrap();
        assert_eq!(p.repetition_penalty, 1.1);
        assert_eq!(p.presence_penalty, 0.5);
        assert_eq!(p.frequency_penalty, -0.25);
        assert_eq!(p.min_p, 0.05);
        assert_eq!(p.logit_bias, vec![(7, 2.5), (42, -100.0)]);
        let none = request_processors(&crate::SamplingConfig::default(), &json!({"min_p": null}))
            .ok()
            .unwrap();
        assert!(!none.is_active());
        let defaults = crate::SamplingConfig {
            repetition_penalty: Some(1.2),
            ..Default::default()
        };
        let kept = request_processors(&defaults, &json!({})).ok().unwrap();
        assert_eq!(kept.repetition_penalty, 1.2, "the process's default stands");
    }

    #[test]
    fn a_malformed_or_out_of_range_field_is_refused_by_name() {
        assert!(refusal(json!({"presence_penalty": 3}))
            .starts_with("presence_penalty must be in [-2, 2]"));
        assert!(refusal(json!({"frequency_penalty": "x"}))
            .contains("'frequency_penalty' must be a number"));
        assert!(refusal(json!({"repetition_penalty": 0})).starts_with("repetition_penalty must be"));
        assert!(refusal(json!({"min_p": 1.5})).starts_with("min_p must be in [0, 1]"));
        assert!(refusal(json!({"logit_bias": {"x": 1}})).contains("key `x` is not a token id"));
        assert!(refusal(json!({"logit_bias": {"3": 500}})).starts_with("logit_bias for token 3"));
        assert!(refusal(json!({"logit_bias": [1, 2]})).contains("must be an object"));
    }
}

#[cfg(test)]
mod stream_writer_tests {
    use super::{ChunkedWriter, StreamWriter};
    use std::io::Write;
    use std::sync::{Arc, Mutex};

    /// A writer into shared memory, so the test reads what the thread wrote.
    #[derive(Clone, Default)]
    struct Mem(Arc<Mutex<Vec<u8>>>);
    impl Write for Mem {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// The wire keeps the producer's order: a control event queued before
    /// text stays before it, text handed over before a control event goes
    /// out ahead of it (`send` takes it), and text handed over just before
    /// the run ends is written (`finish` flushes the spill; the writer drains
    /// once more on its way out, whatever its tick was doing).
    #[test]
    fn text_pushed_at_the_end_is_written_after_what_preceded_it() {
        let mem = Mem::default();
        let w = ChunkedWriter {
            w: Box::new(mem.clone()),
        };
        let sw = StreamWriter::start_with_deltas(w, |t: &str| format!("<{t}>"));
        sw.send("ctl".to_string());
        sw.delta("a");
        sw.delta("b");
        sw.send("mid".to_string());
        sw.delta("c");
        let (_w, err) = sw.finish().unwrap();
        assert!(err.is_none());
        let out = String::from_utf8(mem.0.lock().unwrap().clone()).unwrap();
        let body: String = out
            .split("\r\n")
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, s)| s)
            .collect();
        assert_eq!(body, "ctl<ab>mid<c>", "{out:?}");
    }
}
