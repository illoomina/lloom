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

use super::common::{ms, MemSnapshot, NEVER_EOS};
use super::served::Served;
use crate::{DecodeParams, Decoded, ReadbackArm, Stop};
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
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

/// The serving flags. The model flags (`--model`, the family's own knobs)
/// are [`super::ModelArgs`], shared with `lloom chat`.
#[derive(Parser)]
#[command(
    name = "lloom serve",
    about = "Serve a checkpoint (LFM2 or Qwen4-exp) over an OpenAI-compatible endpoint and the pipette benchmark sidecar contract"
)]
pub struct ServeArgs {
    #[command(flatten)]
    pub model: super::ModelArgs,

    /// Bind address. Loopback: this is a sidecar, not a public server.
    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    /// Port; 0 picks a free one. The bound port is in the ready marker.
    #[arg(long, default_value_t = 0)]
    port: u16,

    /// The option layers: `--config`, `--decode-config`, `--set`, and the
    /// generated flags (`--readback`, `--lookahead`, `--commit-budget`,
    /// `--prefill-tile`, `--prefill-chunk`, …).
    #[command(flatten)]
    pub common: super::CommonArgs,

    /// Reuse the cached prefix across requests (`Session::prepare`). Off by
    /// default: a benchmark measures a cold prefill per request.
    #[arg(long)]
    reuse_prefix: bool,

    /// A file whose text is tiled into the throughput cells' prompts. Falls
    /// back to `PIPETTE_PROMPT_SEED_TEXT` / `PIPETTE_MLX_PROMPT_SEED_TEXT`, then
    /// a built-in passage.
    #[arg(long)]
    prompt_seed_file: Option<PathBuf>,

    /// Default `max_tokens` for a request that names none.
    #[arg(long, default_value_t = 512)]
    max_tokens_default: usize,

    /// The batch width: how many requests decode together, one weight sweep
    /// per step for all of them (issue #148). `1` is the single-sequence
    /// crown loop; `N > 1` starts the continuous-batching scheduler with `N`
    /// sequences and lets the benchmark cells take `batch` up to `N`.
    #[arg(long, default_value_t = 1)]
    batch: usize,

    /// Sample the memory gauges at 10 ms around every benchmark cell and
    /// report the peaks (`mem` on the response). A gauge run — it perturbs
    /// the timing it sits beside — so off by default; `/health` reads the
    /// gauges once at any time.
    #[arg(long)]
    mem_sample: bool,

    /// How the rows of `--batch N` hold their state: `pool` (one shared
    /// paged pool, the attention and conv of the batch in one dispatch per
    /// kernel — the default at `N > 1`, under `--dspark` too, where it is
    /// the continuous speculative loop), `pool-per-row` (the same pool, the
    /// per-row loop: the A/B arm), `private` (a crown cache per row; under
    /// `--dspark` the static pipelined driver).
    #[arg(long, default_value = "auto")]
    rows: String,

    /// Share a request's prompt prefix from another row's cache, by page —
    /// an idle slot's or a live row's; a request whose prefix a live row is
    /// still feeding waits for it (`batched-decode-design.md` §12.3). The
    /// `pool` arms only. Off by default: the benchmark cells feed one prompt
    /// to every row and would otherwise measure the share.
    #[arg(long)]
    share_prefix: bool,
}

/// State shared by every request thread.
struct Shared {
    engine: Box<dyn Served>,
    /// Sample ids an `/eval/abort` asked the running `/eval` to stop early.
    aborts: Mutex<HashSet<String>>,
    server: Server,
    model_name: String,
    max_tokens_default: usize,
}

pub fn run(args: ServeArgs) -> Result<()> {
    let mut options = super::resolve_options(&args.common)?;
    let seed_text = match &args.prompt_seed_file {
        Some(p) => std::fs::read_to_string(p)
            .with_context(|| format!("reading the prompt seed {}", p.display()))?,
        None => std::env::var("PIPETTE_PROMPT_SEED_TEXT")
            .or_else(|_| std::env::var("PIPETTE_MLX_PROMPT_SEED_TEXT"))
            .unwrap_or_else(|_| DEFAULT_SEED_TEXT.to_string()),
    };

    let batched = args.batch > 1;
    // The CHUNK defaults to 512 at `N > 1` (the interleaved prefill), whole
    // prompts at `N = 1`.
    if batched && options.decode.prefill.chunk == 0 {
        options.decode.prefill.chunk = 512;
    }
    let load_start = Instant::now();
    let (engine, model_path) = super::load_engine(
        &args.model,
        super::Shape {
            decode: options.decode.clone(),
            reuse_prefix: args.reuse_prefix,
            seed_text,
            batch: args.batch,
            mem_sample: args.mem_sample,
            rows: args.rows.clone(),
            share_prefix: args.share_prefix,
        },
    )?;
    let model_name = engine.model_name().to_string();
    eprintln!(
        "[loaded {} from {} in {:.2}s on {:?}; kv_capacity={} readback={} reuse_prefix={} {}]",
        model_name,
        model_path.display(),
        load_start.elapsed().as_secs_f64(),
        engine.device(),
        engine.kv_capacity(),
        engine.readback().name(),
        args.reuse_prefix,
        engine.describe(),
    );
    eprintln!("[{}]", MemSnapshot::read(engine.device()).json());

    let server = Server::http((args.host.as_str(), args.port))
        .map_err(|e| anyhow::anyhow!("binding {}:{}: {e}", args.host, args.port))?;
    let port = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        #[allow(unreachable_patterns)]
        _ => args.port,
    };
    let shared = Arc::new(Shared {
        engine,
        aborts: Mutex::new(HashSet::new()),
        server,
        model_name,
        max_tokens_default: args.max_tokens_default,
    });

    // The ready marker the sidecar runner waits for, on stdout and nothing
    // else on stdout before it. Everything the operator reads goes to stderr.
    println!(
        "{}",
        json!({"kind": "ready", "host": args.host, "port": port})
    );
    std::io::stdout().flush().ok();
    eprintln!("[serving on http://{}:{port}]", args.host);

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
                let _ = respond_json(request, 200, &body);
            }
            (Method::Get, "/v1/models") => {
                let _ = respond_json(request, 200, &models_body(&shared.model_name));
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
        Ok(Reply::Stream(stream)) => stream_response(shared, request, stream),
        Ok(Reply::Eval(v)) => eval(shared, request, &v),
        Err(Failure::BadRequest(m)) => respond_error(request, 400, &m),
        Err(Failure::NotFound(m)) => respond_error(request, 404, &m),
        Err(Failure::Internal(e)) => respond_error(request, 500, &format!("{e:#}")),
    }
}

enum Reply {
    Json(Value),
    Stream(StreamingRun),
    Eval(Value),
}

/// A request whose body is produced while the decode runs.
struct StreamingRun {
    kind: StreamKind,
    prompt_ids: Vec<u32>,
    params: DecodeParams,
    id: String,
    model: String,
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
    let id = format!("cmpl-{}", short_id());
    if stream {
        return Ok(Reply::Stream(StreamingRun {
            kind: StreamKind::Completion,
            prompt_ids,
            params,
            id,
            model,
        }));
    }
    let started = Instant::now();
    let decoded = shared.engine.complete(&prompt_ids, &params, &mut |_| {})?;
    let elapsed = started.elapsed();
    report("/v1/completions", prompt_ids.len(), &decoded, ms(elapsed));
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
        "timings": timings(prompt_ids.len(), &decoded, elapsed),
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
    let params = sampling_params(shared, v)?;
    let stream = v.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let want_logprobs = v.get("logprobs").and_then(Value::as_bool).unwrap_or(false);
    let kwargs = v.get("chat_template_kwargs");

    let prompt_ids = {
        let engine = shared.engine.as_ref();
        let rendered = engine.render_chat(messages, kwargs)?;
        let ids = engine.encode(&rendered, true)?;
        if let Some(bos) = engine.bos_id() {
            if ids.iter().filter(|&&t| t == bos).count() > 1 {
                eprintln!("[chat: the rendered template carries BOS twice; check bos_token]");
            }
        }
        ids
    };
    let id = format!("chatcmpl-{}", short_id());

    if want_logprobs {
        if stream {
            return Err(bad("logprobs with stream is not supported"));
        }
        let top_k = v
            .get("top_logprobs")
            .and_then(Value::as_u64)
            .unwrap_or(1)
            .clamp(1, 64) as usize;
        if params.max_tokens != 1 {
            return Err(bad("logprobs is supported for max_tokens = 1 only"));
        }
        // One forward: the argmax is the token, and its distribution is the
        // logprobs block. Temperature does not apply on this path.
        let engine = shared.engine.as_ref();
        let started = Instant::now();
        let (argmax, top) = engine.next_token_logprobs(&prompt_ids, top_k)?;
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
        eprintln!(
            "[/v1/chat/completions logprobs: prompt_tokens={} top_logprobs={top_k} {:.1} ms]",
            prompt_ids.len(),
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
                "finish_reason": "length",
            }],
            "usage": usage(prompt_ids.len(), 1),
        })));
    }

    if stream {
        return Ok(Reply::Stream(StreamingRun {
            kind: StreamKind::Chat,
            prompt_ids,
            params,
            id,
            model,
        }));
    }
    let started = Instant::now();
    let decoded = shared.engine.complete(&prompt_ids, &params, &mut |_| {})?;
    let elapsed = started.elapsed();
    report(
        "/v1/chat/completions",
        prompt_ids.len(),
        &decoded,
        ms(elapsed),
    );
    Ok(Reply::Json(json!({
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
        "timings": timings(prompt_ids.len(), &decoded, elapsed),
    })))
}

/// The sampling knobs both OpenAI endpoints share.
fn sampling_params(shared: &Shared, v: &Value) -> Handled<DecodeParams> {
    let max_tokens = v
        .get("max_completion_tokens")
        .or_else(|| v.get("max_tokens"))
        .and_then(Value::as_u64)
        .map(|n| n as usize)
        .unwrap_or(shared.max_tokens_default);
    if max_tokens == 0 {
        return Err(bad("max_tokens must be positive"));
    }
    let temperature = match v.get("temperature") {
        None | Some(Value::Null) => None,
        Some(t) => Some(
            t.as_f64()
                .ok_or_else(|| bad("field 'temperature' must be a number"))?,
        ),
    };
    let top_p = match v.get("top_p") {
        None | Some(Value::Null) => None,
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
        .unwrap_or_else(rand::random::<u64>);
    let ignore_eos = v
        .get("ignore_eos")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let stop_strings: Vec<String> = match v.get("stop") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        Some(_) => return Err(bad("field 'stop' must be a string or an array of strings")),
    };
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
        max_tokens,
        eos_ids,
        stop_strings,
        readback,
        prefill_tile: shared.engine.prefill_tile(),
    })
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
    let sw = StreamWriter::start(ChunkedWriter::start(request, "text/event-stream")?);
    let created = now();
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
        let (id, model, kind) = (run.id.clone(), run.model.clone(), run.kind);
        // The callback runs on the decode thread: it serialises the chunk and
        // queues it; the socket write happens on the writer thread. A client
        // that went away stops the run at the next drained batch.
        engine.complete_until(&run.prompt_ids, &run.params, &mut |frag| {
            if sw.client_gone() {
                return false;
            }
            let chunk = match kind {
                StreamKind::Completion => json!({
                    "id": id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{ "index": 0, "text": frag, "logprobs": null, "finish_reason": null }],
                }),
                StreamKind::Chat => json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                    "choices": [{ "index": 0, "delta": { "content": frag }, "finish_reason": null }],
                }),
            };
            sw.event(&chunk);
            true
        })
    };
    let (mut w, write_err) = sw.finish()?;
    let decoded = decoded?;
    if let Some(e) = write_err {
        anyhow::bail!("client went away mid-stream: {e}");
    }
    let elapsed = started.elapsed();
    report(endpoint, run.prompt_ids.len(), &decoded, ms(elapsed));
    let usage = usage(run.prompt_ids.len(), decoded.tokens.len());
    let last = match run.kind {
        StreamKind::Completion => json!({
            "id": run.id, "object": "text_completion", "created": created, "model": run.model,
            "choices": [{ "index": 0, "text": "", "logprobs": null,
                          "finish_reason": finish_reason(decoded.stop) }],
            "usage": usage,
            "timings": timings(run.prompt_ids.len(), &decoded, elapsed),
        }),
        StreamKind::Chat => json!({
            "id": run.id, "object": "chat.completion.chunk", "created": created, "model": run.model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason(decoded.stop) }],
            "usage": usage,
            "timings": timings(run.prompt_ids.len(), &decoded, elapsed),
        }),
    };
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
        report("/end_to_end_latency", ids.len(), &decoded, total_ms);
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
    for (i, d) in decoded.iter().enumerate() {
        report(
            &format!("/end_to_end_latency row {i}/{batch}"),
            ids.len(),
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
    let enable_thinking = v.get("enable_thinking").cloned().unwrap_or(Value::Null);
    let kwargs = if enable_thinking.is_null() {
        None
    } else {
        Some(json!({ "enable_thinking": enable_thinking }))
    };

    let sw = StreamWriter::start(ChunkedWriter::start(request, "application/x-ndjson")?);
    let engine = shared.engine.as_ref();
    let total = samples.len();
    let workers = engine.batch().max(1).min(total.max(1));
    let next = std::sync::atomic::AtomicUsize::new(0);
    let run = std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let (sw, samples, done, kwargs, next) = (&sw, &samples, &done, &kwargs, &next);
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
                        let prompt = engine.render_chat(messages, kwargs.as_ref())?;
                        let ids = engine.encode(&prompt, true)?;
                        sw.event_line(&json!({
                            "kind": "eval_sample_start", "sample_id": sample_id, "prompt": prompt,
                        }));
                        let params = DecodeParams {
                            seed: rand::random::<u64>(),
                            temperature: (temperature > 0.0).then_some(temperature),
                            top_p: None,
                            max_tokens,
                            eos_ids: engine.eos_ids().to_vec(),
                            stop_strings: Vec::new(),
                            readback: engine.readback(),
                            prefill_tile: engine.prefill_tile(),
                        };
                        let mut pending = String::new();
                        let started = Instant::now();
                        let aborts = &shared.aborts;
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
                                !sw.client_gone() && !lock(aborts).contains(&sid)
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
struct StreamWriter {
    tx: Option<std::sync::mpsc::Sender<String>>,
    gone: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<(ChunkedWriter, Option<std::io::Error>)>>,
}

impl StreamWriter {
    fn start(mut w: ChunkedWriter) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let gone = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&gone);
        let thread = std::thread::spawn(move || {
            let mut err = None;
            for payload in rx {
                if err.is_some() {
                    continue;
                }
                if let Err(e) = w.raw(&payload) {
                    flag.store(true, Ordering::Release);
                    err = Some(e);
                }
            }
            (w, err)
        });
        Self {
            tx: Some(tx),
            gone,
            thread: Some(thread),
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
fn timings(prompt: usize, d: &Decoded, elapsed: std::time::Duration) -> Value {
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
        "overrun": st.overrun,
        // The generated ids, so a gate can compare a run against a recorded
        // token digest without a second path to the model.
        "tokens": d.tokens,
    })
}

fn report(endpoint: &str, prompt: usize, d: &Decoded, total_ms: f64) {
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
         commits={} poll_fallbacks={} overrun={}]",
        d.tokens.len(),
        d.stop,
        d.readback.name(),
        d.reuse,
        d.prefilled,
        st.commits,
        st.readback.poll_fallbacks,
        st.overrun,
    );
}

fn models_body(model_name: &str) -> Value {
    json!({
        "object": "list",
        "data": [{ "id": model_name, "object": "model", "created": now(), "owned_by": "lloom" }],
    })
}
