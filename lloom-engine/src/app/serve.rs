//! `lloom-serve` — the engine as a process a client can drive (issue #695).
//!
//! One binary, two contracts on one loopback port:
//!
//! - the **OpenAI-compatible** surface a generic client speaks —
//!   `/v1/completions` (with `ignore_eos`, a `usage` block and per-token
//!   `logprobs`), `/v1/chat/completions` (streaming SSE, the checkpoint's own
//!   chat template through the request front end, `front.rs`, `logprobs`),
//!   `/v1/models`, `/tokenize`; on both routes the token-level fields
//!   `prompt_token_ids` (ids in, no template) and `return_token_ids`;
//!
//! On the chat route a reply's reasoning comes back **apart from its answer**:
//! `message.reasoning_content` and `message.content` (the answer only), and on
//! the stream `delta.reasoning_content` and `delta.content` (decision 51 Q2,
//! `sequence-reuse-design.md` §2.3d). A request that wants the whole reply in
//! `content`, as before, sends `reasoning_in_content: true` — an opt-out.
//! `/v1/completions` is unchanged: its text is the reply.
//!
//! `usage.completion_tokens` counts the tokens of the completion the client
//! **received** — the accepted frontier (decision 51 Q4, the standard
//! semantics). The count the run generated is in `timings.tokens` and
//! `timings.frontier`, which differ past a stop string on the adaptive commit.
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
use super::front::{self, Front};
use super::served::Served;
use super::template::{
    request_kwargs, Rejected, Unrenderable, ENABLE_THINKING, ENGINE_KWARGS, REASONING_EFFORT,
    REQUEST_FIELDS,
};
use super::thinking::{self, Thinking};
use super::vision::{expand_image_tokens, split_image_parts, DecodedImage};
use crate::frontend::Frontend;
use crate::sink::{DeltaBuffer, DRAIN_PERIOD};
use crate::{DecodeParams, Decoded, Stop};
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Map, Value};
use std::collections::HashSet;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

mod reasoning;
mod turns;

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
    pub session: crate::options::SessionConfigArgs,
    #[command(flatten)]
    pub report: crate::options::ReportConfigArgs,
}

impl ServeArgs {
    /// `serve`'s own flags, as the top layer.
    pub fn front(&self, o: &mut crate::EngineOptions) -> std::result::Result<(), String> {
        self.server.apply(&mut o.server)?;
        self.scheduler.apply(&mut o.scheduler)?;
        self.prefix.apply(&mut o.prefix)?;
        self.session.apply(&mut o.session)?;
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
    /// What `/v1/systemone` answers with, resolved at load.
    decisions: super::decisions::Decisions,
    /// Request threads queue here for the family's single request slot.
    turns: turns::Turns,
    /// Decisions that wait together join one group (`scheduler.decision_pack`,
    /// and the `fast` contract).
    decision_pack: bool,
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
    /// The request front end: the template constants, the session log.
    front: Front,
    /// What every request reused of its session, counted (`/health`).
    reuse: parking_lot::Mutex<crate::turn::Tally>,
}

/// The largest `/v1/systemone` body a decision-only owner reads: eight
/// images or one clip, base64-encoded, with room for the state.
const DECISION_BODY_BYTES: usize = 64 << 20;

/// How long a server's in-flight requests get to answer once serving stops,
/// before their connections are cut.
const HTTP_SHUTDOWN_BUDGET: std::time::Duration = std::time::Duration::from_secs(5);

/// Stops `server` within [`HTTP_SHUTDOWN_BUDGET`] and says so when it had to
/// cut a connection or leave a thread to the retirement service.
fn shutdown_http(server: &Server) {
    let report = server.shutdown(Instant::now() + HTTP_SHUTDOWN_BUDGET);
    if !report.completed && !report.already_shut_down {
        eprintln!(
            "[http shutdown: cut {} connection(s) at the {}s budget, {} thread(s) left to the retirement service]",
            report.cut.len(),
            HTTP_SHUTDOWN_BUDGET.as_secs(),
            report.threads_retiring,
        );
    }
}

/// Joins tiny_http's retirement service before the process exits.
pub(crate) fn join_http_retirement() {
    let joined = tiny_http::join_retirement(Instant::now() + HTTP_SHUTDOWN_BUDGET);
    if !joined.completed {
        eprintln!(
            "[http retirement: {} server(s) and {} thread(s) still retiring at exit]",
            joined.pending_servers, joined.threads_retiring,
        );
    }
}

fn listener<A: std::net::ToSocketAddrs>(
    addr: A,
    policy: Option<tiny_http::TransportPolicy>,
) -> std::result::Result<Server, Box<dyn std::error::Error + Send + Sync>> {
    match policy {
        Some(policy) => Server::http_with_policy(addr, policy),
        None => Server::http(addr),
    }
}

pub fn run(args: ServeArgs) -> Result<()> {
    run_configured(args, None)
}

/// Explicit HTTP transport policy; no model/native cancellation is implied.
/// Validate cheap limits before constructing a model. Defaults remain unchanged.
pub fn run_with_transport_policy(
    args: ServeArgs,
    policy: tiny_http::TransportPolicy,
) -> Result<()> {
    policy.validate()?;
    run_configured(args, Some(policy))
}
fn run_configured(args: ServeArgs, transport: Option<tiny_http::TransportPolicy>) -> Result<()> {
    let options = super::resolve_options(&args.common, &|o| args.front(o), Frontend::Serve)?;
    let seed_text = seed_text(options.server.pipette.prompt_seed_file.as_deref())?;
    let load_start = Instant::now();
    let shape = super::Shape::of(&options, Frontend::Serve, seed_text);
    let reuse_prefix = shape.reuse_prefix;
    // A non-generating decision model is served by the decision-only listener.
    if let Some(endpoint) = super::omni::serving::load(&options.load, &shape)? {
        return run_decisions_configured(options, endpoint, DECISION_BODY_BYTES, transport, true);
    }
    let (engine, model_path) = super::load_engine(&options.load, shape)?;
    run_owner(
        options,
        engine,
        model_path,
        Some((load_start, reuse_prefix)),
        transport,
    )
}

/// Serve an already-loaded owner without creating a device, tokenizer, model or
/// admission domain. The caller creates its device, configures the same process
/// account, and completes the genuine load before entering this frontend.
///
/// Only `options.server.host`, `options.server.port`, `options.sampling` and
/// `options.session` (the request front end's) are honored; the remaining
/// options do not reconfigure this owner. Shape, decode,
/// cache, seed, prefix reuse, batch and memory policy are the already-loaded
/// owner's contract, not inferred from these options. `model_path` is a caller-
/// supplied provenance label, never opened or used to load another model.
/// This route does not activate default registration or complete admission.
pub fn run_loaded(
    options: crate::EngineOptions,
    engine: Box<dyn Served>,
    model_path: std::path::PathBuf,
) -> Result<()> {
    run_owner(options, engine, model_path, None, None)
}
/// Supplied SAME owner, with explicit plain-HTTP I/O/disposal limits. See
/// tiny_http::TransportPolicy for pipeline timeouts; not an application timeout.
pub fn run_loaded_with_transport_policy(
    options: crate::EngineOptions,
    engine: Box<dyn Served>,
    model_path: std::path::PathBuf,
    policy: tiny_http::TransportPolicy,
) -> Result<()> {
    policy.validate()?;
    run_owner(options, engine, model_path, None, Some(policy))
}

/// Startup provenance distinguishes configured options from the loaded owner's
/// actual description. Supplied owners do not install the option tree.
fn ready_configuration(options: &crate::EngineOptions, installed: bool, owner: Value) -> Value {
    json!({
        "schema_version": 1,
        "frontend": "serve",
        "options_installed": installed,
        "configured_options": crate::option_tree::index().flat(options),
        "owner": owner,
        "request_resolution": "readback and sampling may resolve per request; read response timings",
    })
}

fn run_owner(
    options: crate::EngineOptions,
    engine: Box<dyn Served>,
    model_path: std::path::PathBuf,
    load_facts: Option<(Instant, bool)>,
    transport: Option<tiny_http::TransportPolicy>,
) -> Result<()> {
    let model_name = engine.model_name().to_string();
    let mut sampling = options.sampling.clone();
    sampling.apply_checkpoint(engine.generation_config_checked()?.as_ref());
    sampling
        .check()
        .map_err(|e| anyhow::anyhow!("the checkpoint's generation_config.json: {e}"))?;
    let description = engine.describe_checked()?;
    let logprobs = logprobs_line(engine.as_ref());
    if let Some((load_start, reuse_prefix)) = load_facts {
        eprintln!(
            "[loaded {} from {} in {:.2}s on {:?}; kv_capacity={} readback={} reuse_prefix={} logprobs={logprobs} {}]",
            model_name,
            model_path.display(),
            load_start.elapsed().as_secs_f64(),
            engine.device(),
            engine.kv_capacity(),
            engine.readback().name(),
            reuse_prefix,
            description,
        );
    } else {
        eprintln!(
            "[serving existing {} from {} on {:?}; kv_capacity={} readback={} reuse_prefix={} logprobs={logprobs} {}]",
            model_name,
            model_path.display(),
            engine.device(),
            engine.kv_capacity(),
            engine.readback().name(),
            crate::turn::policy_name(engine.reuse_prefix()),
            description,
        );
    }
    eprintln!("[capabilities: {}]", engine.capabilities().line());
    // A loaded owner's path is a provenance label, never opened.
    let checkpoint = load_facts.is_some().then_some(model_path.as_path());
    let decisions = super::decisions::Decisions::resolve(
        options.load.decision_template.as_deref(),
        engine.as_ref(),
        checkpoint,
    )?;
    if let Some(line) = decisions.describe() {
        eprintln!("[{line}]");
    }
    eprintln!("[{}]", MemSnapshot::read(engine.device()).json());
    let front = Front::new(engine.as_ref(), front::SessionOptions::of(&options.session));
    eprintln!("[session: {}]", front.line());
    if let Some(why) = front.persist_unsupported() {
        anyhow::bail!(
            "session.persist_thinking: {}: {why}",
            engine.capabilities().family
        );
    }
    if let (Some(why), true) = (
        front::logprobs_refused(engine.as_ref()),
        front.options().keep != front::KeepLogprobs::None,
    ) {
        anyhow::bail!("session.keep_logprobs: {why}");
    }
    let configuration = ready_configuration(
        &options,
        load_facts.is_some(),
        json!({
            "description": description,
            "readback": engine.readback().name(),
            "kv_capacity": engine.kv_capacity(),
            "batch": engine.batch(),
            "prefill_tile": engine.prefill_tile(),
            "prefix_reuse": engine.reuse_prefix(),
            "capabilities": engine.capabilities().json(),
        }),
    );
    let (host, port) = (options.server.host.clone(), options.server.port);

    let server = listener((host.as_str(), port), transport)
        .map_err(|e| anyhow::anyhow!("binding {host}:{port}: {e}"))?;
    let port = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        #[allow(unreachable_patterns)]
        _ => port,
    };
    let shared = Arc::new(Shared {
        engine,
        decisions,
        turns: turns::Turns::default(),
        decision_pack: options.scheduler.decision_pack
            && lloom_tensor::tuning::contract().is_fast(),
        aborts: Mutex::new(HashSet::new()),
        abort_epoch: AtomicU64::new(0),
        server,
        model_name,
        sampling,
        front,
        reuse: parking_lot::Mutex::new(crate::turn::Tally::default()),
    });

    // The ready marker the sidecar runner waits for, on stdout and nothing
    // else on stdout before it. Everything the operator reads goes to stderr.
    println!(
        "{}",
        json!({"kind": "ready", "host": host, "port": port, "configuration": configuration})
    );
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
                let result = health_body(shared.engine.as_ref(), &shared.front).map(|mut b| {
                    b["reuse"] = shared.reuse.lock().json(shared.engine.reuse_prefix());
                    b
                });
                let _ = respond_metadata(request, result);
            }
            (Method::Get, "/v1/models") => {
                let result = models_body(
                    &shared.model_name,
                    shared.engine.as_ref(),
                    &shared.decisions,
                );
                let _ = respond_metadata(request, result);
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
    // In-flight requests answer within the budget; then their connections
    // are cut. A request thread still decoding holds the sequence, and the
    // batch thread holds the pool; wait for both so the model is not dropped
    // under a running command buffer.
    shutdown_http(&shared.server);
    shared.engine.shutdown();
    Ok(())
}

/// Serve a decision-only owner using the SAME ordered dispatcher/status mapper.
/// This loop spawns no request worker threads and invents no generation/KV/seed API.
/// Only host/port are honored from options. max_body_bytes bounds the bytes
/// retained by this application reader, NOT dependency header/body allocations,
/// unread-body Drop draining, read time, or complete model prediction. tiny_http
/// owns listener/connection threads and may drain unread Content-Length with a
/// remaining-body-sized allocation; this is not a hostile-input transport bound.
/// Socket/parsing/serialization/metadata buffers remain enrollment obligations.
/// Default family registration is unchanged; encoded-media capability is the
/// installed endpoint's actual contract, not inferred from its model name.
pub fn run_decisions_loaded(
    options: crate::EngineOptions,
    endpoint: Box<dyn super::decision_endpoint::DecisionEndpoint>,
    max_body_bytes: usize,
) -> Result<()> {
    run_decisions_configured(options, endpoint, max_body_bytes, None, false)
}
/// Same narrow endpoint and dispatcher, with explicit dependency disposal/I/O
/// limits. Header/JSON/queue allocations and thread joining remain separate.
pub fn run_decisions_loaded_with_transport_policy(
    options: crate::EngineOptions,
    endpoint: Box<dyn super::decision_endpoint::DecisionEndpoint>,
    max_body_bytes: usize,
    policy: tiny_http::TransportPolicy,
) -> Result<()> {
    policy.validate()?;
    run_decisions_configured(options, endpoint, max_body_bytes, Some(policy), false)
}
fn run_decisions_configured(
    options: crate::EngineOptions,
    endpoint: Box<dyn super::decision_endpoint::DecisionEndpoint>,
    max_body_bytes: usize,
    transport: Option<tiny_http::TransportPolicy>,
    options_installed: bool,
) -> Result<()> {
    let read_bound = max_body_bytes
        .checked_add(1)
        .filter(|_| max_body_bytes != 0)
        .ok_or_else(|| {
            anyhow::anyhow!("decision transport body bound must be finite and nonzero")
        })?;
    let configuration = ready_configuration(
        &options,
        options_installed,
        json!({
            "generation": false,
            "decisions": true,
            "model": endpoint.model_name(),
            "input_modalities": endpoint.input_modalities(),
        }),
    );
    let host = options.server.host;
    let server = listener((host.as_str(), options.server.port), transport)
        .map_err(|error| anyhow::anyhow!("binding decision listener: {error}"))?;
    let port = match server.server_addr() {
        tiny_http::ListenAddr::IP(addr) => addr.port(),
        #[allow(unreachable_patterns)]
        _ => options.server.port,
    };
    println!(
        "{}",
        json!({"kind":"ready","host":host,"port":port,"generation":false,"decisions":true,"configuration":configuration})
    );
    std::io::stdout().flush()?;
    loop {
        let mut request = match server.recv() {
            Ok(request) => request,
            Err(_) => break,
        };
        let path = request.url().split('?').next().unwrap_or("").to_owned();
        match (request.method().clone(), path.as_str()) {
            (Method::Get, "/health") => {
                let _ = respond_metadata(request, endpoint.health_checked());
            }
            (Method::Get, "/v1/models") => {
                // Caller provenance label; outer metadata formatting is not
                // silently claimed paid by the model's inner request pass.
                let body = json!({"object":"list","data":[{"id":endpoint.model_name(),"object":"model","generation":false,"decisions":true,
                    "architecture":{"input_modalities":endpoint.input_modalities(),"output_modalities":["decisions"]}}]});
                let _ = respond_json(request, 200, &body);
            }
            (Method::Post, "/shutdown") => {
                let _ = respond_json(request, 200, &json!({}));
                break;
            }
            (Method::Post, "/v1/systemone") => {
                let mut body = Vec::new();
                let read = request
                    .as_reader()
                    .take(read_bound as u64)
                    .read_to_end(&mut body);
                match read {
                    Err(error) => {
                        let _ =
                            respond_error(request, 400, &format!("reading request body: {error}"));
                    }
                    Ok(_) if body.len() > max_body_bytes => {
                        let _ = respond_error(
                            request,
                            413,
                            "decision request body exceeds supplied byte limit",
                        );
                    }
                    Ok(_) => {
                        let result = dispatch_systemone_with(&path, &body, |body| {
                            endpoint
                                .systemone(body)
                                .map(|response| response.with_model(endpoint.model_name()))
                        })
                        .expect("SystemOne route selected above");
                        let _ = respond_systemone(request, result);
                    }
                }
            }
            (Method::Post, "/v1/completions" | "/v1/chat/completions") => {
                let _ = respond_error(
                    request,
                    501,
                    "this decision-only owner does not generate tokens",
                );
            }
            _ => {
                let _ = respond_error(request, 404, &format!("unknown decision endpoint: {path}"));
            }
        }
    }
    // Our loop spawned no request threads and its last synchronous call
    // returned; the server's own threads and connections retire here.
    // Worker-stop only: no native DATA/callback/retirement certificate.
    shutdown_http(&server);
    endpoint.shutdown();
    Ok(())
}

// ---- routing ---------------------------------------------------------------

fn handle_model_request(shared: &Shared, path: &str, mut request: Request) -> Result<()> {
    let mut body = String::new();
    request
        .as_reader()
        .read_to_string(&mut body)
        .context("reading the request body")?;
    // This branch consumes the original bytes exactly once. Parsing through
    // Value first would reorder question/options/legend objects.
    // Decisions that wait together are answered as one group; under `fast`
    // the family packs them (`turns`).
    if let Some(result) = dispatch_systemone_with(path, body.as_bytes(), |body| {
        shared
            .turns
            .decide(body, shared.decision_pack, |bodies| {
                shared
                    .decisions
                    .decide_batch(shared.engine.as_ref(), bodies)
            })
            .map(|response| response.with_model(&shared.model_name))
    }) {
        return respond_systemone(request, result);
    }
    // A single-sequence engine's model requests take the request slot one at
    // a time; a batched engine schedules its own rows.
    let _turn = (path != "/tokenize" && shared.engine.batch() <= 1).then(|| shared.turns.take());
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

/// The actual ordered route: return None without parsing other endpoints.
#[cfg(test)]
fn dispatch_systemone(
    engine: &dyn Served,
    path: &str,
    body: &[u8],
) -> Option<std::result::Result<String, crate::decision::SystemOneError>> {
    dispatch_systemone_with(path, body, |body| engine.systemone(body))
}

fn respond_systemone(
    request: Request,
    result: std::result::Result<String, crate::decision::SystemOneError>,
) -> Result<()> {
    use crate::decision::SystemOneError;
    match result {
        Ok(text) => respond_json_text(request, 200, text),
        Err(SystemOneError::InvalidRequest(message)) => respond_error(request, 400, &message),
        Err(SystemOneError::Unsupported(message)) => respond_error(request, 501, &message),
        Err(SystemOneError::Execution(message)) => respond_error(request, 500, &message),
    }
}

/// Shared actual ordered parser/invocation/serializer for both installed shapes.
fn dispatch_systemone_with(
    path: &str,
    body: &[u8],
    execute: impl FnOnce(
        &crate::decision::OrderedJson,
    ) -> std::result::Result<
        crate::decision::SystemOneResponse,
        crate::decision::SystemOneError,
    >,
) -> Option<std::result::Result<String, crate::decision::SystemOneError>> {
    use crate::decision::{OrderedJson, SystemOneError};
    if path != "/v1/systemone" {
        return None;
    }
    Some((|| {
        let ordered = OrderedJson::from_slice(body).map_err(|e| {
            SystemOneError::InvalidRequest(format!("request body is not JSON: {e}"))
        })?;
        let response = execute(&ordered)?;
        serde_json::to_string(&response).map_err(|e| SystemOneError::Execution(e.to_string()))
    })())
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
    /// A chat prompt the front end built: its reply goes in the session log
    /// against the conversation it answers.
    record: Option<front::Origin>,
    /// `return_token_ids`: the final event carries the ids.
    return_ids: bool,
    /// `reasoning_in_content`: the deltas carry the whole reply in `content`,
    /// as before decision 51 Q2, rather than splitting the reasoning out.
    reasoning_in_content: bool,
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

fn append_fields(to: &mut serde_json::Map<String, Value>, fields: serde_json::Map<String, Value>) {
    for (key, value) in fields {
        let previous = to.get(&key).and_then(Value::as_str).unwrap_or("");
        to.insert(
            key,
            Value::String(format!("{previous}{}", value.as_str().unwrap_or(""))),
        );
    }
}

fn run_decode_observed(
    engine: &dyn Served,
    prompt_ids: &[u32],
    images: &[DecodedImage],
    params: &DecodeParams,
    split: Option<&Arc<parking_lot::Mutex<reasoning::Reasoning>>>,
    on_fragment: &mut (dyn FnMut(&str) -> bool + Send),
) -> Result<Decoded> {
    if let Some(split) = split {
        engine.complete_observed(
            prompt_ids,
            images,
            params,
            on_fragment,
            &mut |ids, text, from, decode| split.lock().observe(ids, text, from, decode),
        )
    } else {
        run_decode(engine, prompt_ids, images, params, on_fragment)
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
    // OpenAI's completions `logprobs`: the number of most likely tokens per
    // position beside the sampled one's (0 is the sampled one alone).
    let want_logprobs = match v.get("logprobs") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => None,
        Some(Value::Bool(true)) => Some(0),
        Some(n) => Some(
            n.as_u64()
                .filter(|&n| n as usize <= lloom::nn::MAX_TOP_LOGPROBS)
                .ok_or_else(|| {
                    bad(format!(
                        "field 'logprobs' must be an integer from 0 to {}",
                        lloom::nn::MAX_TOP_LOGPROBS
                    ))
                })? as usize,
        ),
    };
    if want_logprobs.is_some() {
        if stream {
            return Err(bad("logprobs with stream is not supported"));
        }
        if let Some(why) = front::logprobs_refused(shared.engine.as_ref()) {
            return Err(bad(format!("logprobs: {why}")));
        }
    }
    let return_ids = v
        .get("return_token_ids")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let prompt_ids = match field_token_ids(v)? {
        Some(ids) => {
            if v.get("prompt").is_some_and(|p| !p.is_null()) {
                return Err(bad("prompt and prompt_token_ids are two prompts; send one"));
            }
            ids
        }
        None => {
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
        }
    };
    if prompt_ids.is_empty() {
        return Err(bad("empty prompt"));
    }
    let mut params = params;
    params.logprobs = logprob_request(want_logprobs, None);
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
            record: None,
            return_ids,
            // `/v1/completions` is not a chat route: its text is the reply.
            reasoning_in_content: true,
        })));
    }
    let started = Instant::now();
    let decoded = shared.engine.complete(&prompt_ids, &params, &mut |_| {})?;
    let elapsed = started.elapsed();
    report(
        shared,
        "/v1/completions",
        prompt_ids.len(),
        &facts,
        &decoded,
        ms(elapsed),
    );
    let mut body = json!({
        "id": id,
        "object": "text_completion",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "text": decoded.text,
            "logprobs": match want_logprobs {
                Some(_) => completion_logprobs(shared.engine.as_ref(), &decoded)?,
                None => Value::Null,
            },
            "finish_reason": finish_reason(decoded.stop),
        }],
        "usage": run_usage(shared.engine.as_ref(), prompt_ids.len(), &decoded),
        "timings": timings(shared.engine.as_ref(), prompt_ids.len(), &facts, &decoded, elapsed),
    });
    if return_ids {
        body["prompt_token_ids"] = json!(prompt_ids);
        body["choices"][0]["token_ids"] = json!(decoded.tokens);
        body["timings"]["tokens"] = json!(decoded.tokens);
    }
    Ok(Reply::Json(body))
}

/// OpenAI's completions `logprobs`: per delivered id its token, its
/// log-probability, the top tokens asked for, and its text's offset.
fn completion_logprobs(engine: &dyn Served, d: &Decoded) -> Result<Value> {
    let Some(lp) = &d.logprobs else {
        return Ok(Value::Null);
    };
    let n = delivered_ids(engine, d);
    let tokens = d.tokens[..n]
        .iter()
        .map(|&t| engine.token_str_checked(t))
        .collect::<Result<Vec<String>>>()?;
    let mut offset = 0usize;
    let offsets: Vec<usize> = tokens
        .iter()
        .map(|t| {
            let o = offset;
            offset += t.len();
            o
        })
        .collect();
    let top = lp.top[..n.min(lp.top.len())]
        .iter()
        .map(|t| {
            Ok(Value::Object(
                t.iter()
                    .map(|&(id, l)| Ok((engine.token_str_checked(id)?, json!(l))))
                    .collect::<Result<_>>()?,
            ))
        })
        .collect::<Result<Vec<Value>>>()?;
    // OpenAI's `top_logprobs` keys by the token's text, and two ids can
    // share one (the halves of a character); `top_ids` keeps the ids.
    let top_ids: Vec<Value> = lp.top[..n.min(lp.top.len())]
        .iter()
        .map(|t| json!(t.iter().map(|&(id, l)| json!([id, l])).collect::<Vec<_>>()))
        .collect();
    Ok(json!({
        "tokens": tokens,
        "token_logprobs": &lp.sampled[..n.min(lp.sampled.len())],
        "top_logprobs": top,
        "top_ids": top_ids,
        "text_offset": offsets,
    }))
}

// ---- /v1/chat/completions ----------------------------------------------------

fn chat_completions(shared: &Shared, v: &Value) -> Handled<Reply> {
    let model = shared.model_name.clone();
    let no_messages = json!([]);
    let messages = match (v.get("messages"), v.get("prompt_token_ids")) {
        (None | Some(Value::Null), Some(ids)) if !ids.is_null() => &no_messages,
        (Some(_), Some(ids)) if !ids.is_null() => {
            return Err(bad(
                "messages and prompt_token_ids are two prompts; send one",
            ))
        }
        (m, _) => {
            let m = m
                .filter(|m| m.is_array())
                .ok_or_else(|| bad("field 'messages' must be an array"))?;
            if m.as_array().map(Vec::is_empty).unwrap_or(true) {
                return Err(bad("messages must not be empty"));
            }
            m
        }
    };
    if v.get("guided_choice").is_some() || v.get("guided_json").is_some() {
        // Constrained decoding is not built; a 4xx is what a client that can
        // fall back to `logprobs` looks for.
        return Err(bad("guided decoding is not supported"));
    }
    let mut params = sampling_params(shared, v)?;
    let wanted = request_budgets(shared, v, params.max_tokens)?;
    let stream = v.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let want_logprobs = match v.get("logprobs") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(_) => return Err(bad("field 'logprobs' must be a boolean")),
    };
    let mut kwargs = request_kwargs(v, &REQUEST_FIELDS).map_err(bad)?;
    let want = kwargs.get(ENABLE_THINKING).and_then(Value::as_bool);
    thinking::set_kwarg(shared.engine.as_ref(), want, &mut kwargs);

    // OpenAI `image_url` parts become the template's `{"type": "image"}`;
    // string content passes through untouched.
    let (messages, images) = split_image_parts(messages).map_err(|e| bad(format!("{e:#}")))?;
    let token_ids = field_token_ids(v)?;
    let (prompt_ids, facts, prompt) = {
        let engine = shared.engine.as_ref();
        if let Some(ids) = token_ids {
            // The token-level API: the ids run as sent, no template.
            if !images.is_empty() {
                return Err(bad("prompt_token_ids with an image is not supported"));
            }
            let mut facts = PromptFacts::of(&ids);
            facts.apply_budgets(
                engine,
                "/v1/chat/completions",
                &wanted,
                Dropped::default(),
                &ids,
                &mut params,
            )?;
            (ids, facts, None)
        } else {
            let expansions = images
                .iter()
                .map(|img| engine.image_placeholder(img))
                .collect::<Result<Vec<_>>>()
                .map_err(|e| bad(format!("{e:#}")))?;
            let expand = |r: &str| -> Result<String> {
                expand_image_tokens(r, &expansions)
                    .map_err(|e| anyhow::Error::new(front::Refused(format!("{e:#}"))))
            };
            let expand_ref: front::Expand<'_> = &expand;
            let front_start = Instant::now();
            let prompt = shared
                .front
                .prompt(
                    engine,
                    front::Request {
                        messages: &messages,
                        kwargs: kwargs.clone(),
                        want,
                        add_bos: true,
                        expand: (!images.is_empty()).then_some(expand_ref),
                        history: v.get("messages"),
                    },
                )
                .map_err(|e| front_failure("/v1/chat/completions", e))?;
            let mut facts = PromptFacts {
                reasoning_effort: kwargs.get(REASONING_EFFORT).cloned().unwrap_or(Value::Null),
                thinking: prompt.thinking,
                renders: prompt.renders,
                spliced: prompt.spliced,
                front_us: Some(front_start.elapsed().as_micros() as u64),
                ..PromptFacts::default()
            };
            let mut ids = prompt.ids.clone();
            if let Some(bos) = engine.bos_id() {
                if ids.iter().filter(|&&t| t == bos).count() > 1 {
                    eprintln!("[chat: the rendered template carries BOS twice; check bos_token]");
                }
            }
            // A suffix property: dropping the oldest turns below leaves it.
            params.turn_tail = prompt.turn_tail;
            // The prompt budget, before the engine sees the prompt: the oldest
            // turns go whole, as spans of the ids the one render gave.
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
                let fitted = match shared.front.fit(&prompt, &arr, k) {
                    Some(f) => f,
                    None => {
                        // Spans unknown: the shorter conversations rendered,
                        // counted on the request's `renders`.
                        let before = engine.template_renders();
                        let f = budgets::fit_messages(&arr, k, ids.len(), |ms| {
                            let mut ids = engine
                                .encode(&engine.render_chat(&json!(ms), &prompt.kwargs)?, true)?;
                            let _ = thinking::apply(want, engine.think_tags().as_ref(), &mut ids);
                            Ok(ids)
                        })
                        .map_err(template_failure)?
                        .map(|f| front::Fitted {
                            messages: f.messages,
                            ids: f.ids,
                            dropped: f.dropped,
                        });
                        facts.renders += engine.template_renders() - before;
                        f
                    }
                };
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
            (ids, facts, Some(prompt))
        }
    };
    let id = format!("chatcmpl-{}", short_id());

    if want_logprobs {
        if stream {
            return Err(bad("logprobs with stream is not supported"));
        }
        if !images.is_empty() {
            return Err(bad("logprobs with an image is not supported"));
        }
    }
    // Candidates beside each drawn id, asked with `logprobs`: up to lloom's
    // `MAX_TOP_LOGPROBS`, an extension past the 20 OpenAI's chat API
    // documents (no strict-compatibility bound is applied here).
    let top_logprobs = match v.get("top_logprobs") {
        None | Some(Value::Null) => None,
        Some(n) => Some(
            n.as_u64()
                .filter(|&n| n as usize <= lloom::nn::MAX_TOP_LOGPROBS)
                .ok_or_else(|| {
                    bad(format!(
                        "field 'top_logprobs' must be an integer from 0 to {}",
                        lloom::nn::MAX_TOP_LOGPROBS
                    ))
                })? as usize,
        ),
    };
    if top_logprobs.is_some() && !want_logprobs {
        return Err(bad("top_logprobs asks for logprobs: set logprobs to true"));
    }
    let top_k = top_logprobs.unwrap_or(0);
    // Past one token, each returned id carries its log-probability (and the
    // top `top_logprobs`), from the run's own steps.
    let return_ids = v
        .get("return_token_ids")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // Decision 51 Q2: a reply's reasoning comes back in `reasoning_content`
    // and `content` is the answer alone. A client that wants the whole reply
    // in `content`, as before, asks for it per request — opt-out, not opt-in.
    let reasoning_in_content = v
        .get("reasoning_in_content")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    params.logprobs = logprob_request(
        want_logprobs.then_some(top_logprobs.unwrap_or(0)),
        prompt.as_ref().and(shared.front.options().keep.capture()),
    );
    if want_logprobs && params.max_tokens > 1 {
        if let Some(why) = front::logprobs_refused(shared.engine.as_ref()) {
            return Err(bad(format!("logprobs: {why}")));
        }
    }
    if want_logprobs && params.max_tokens == 1 {
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
        // The drawn id's own log-probability needs its row's head even when
        // no candidate beside it is asked (`top_logprobs` 0).
        let want = if options.is_empty() {
            top_k.max(1)
        } else {
            usize::MAX
        };
        let (argmax, all) = engine.next_token_logprobs(&prompt_ids, want)?;
        let top = &all[..top_k.min(all.len())];
        let content = engine.token_str_checked(argmax)?;
        let token_lp = all
            .iter()
            .find(|(t, _)| *t == argmax)
            .map(|(_, lp)| *lp)
            .unwrap_or(f32::NEG_INFINITY);
        let top_entries: Vec<Value> = top
            .iter()
            .map(|(t, lp)| Ok(json!({ "token": engine.token_str_checked(*t)?, "logprob": lp })))
            .collect::<Result<_>>()?;
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
                    "token": engine.token_str_checked(first)?,
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
            record: prompt.as_ref().map(|p| p.origin),
            return_ids,
            reasoning_in_content,
        })));
    }
    let split =
        reasoning::Reasoning::for_chat(shared.engine.as_ref(), &prompt_ids, reasoning_in_content)?
            .map(|p| Arc::new(parking_lot::Mutex::new(p)));
    let mut routed = serde_json::Map::<String, Value>::new();
    let started = Instant::now();
    let decoded = run_decode_observed(
        shared.engine.as_ref(),
        &prompt_ids,
        &images,
        &params,
        split.as_ref(),
        &mut |text| {
            if let Some(p) = &split {
                append_fields(&mut routed, p.lock().push(text));
            }
            true
        },
    )?;
    if let Some(p) = &split {
        append_fields(&mut routed, p.lock().finish());
    }
    let elapsed = started.elapsed();
    let record_start = Instant::now();
    let frontier = prompt.as_ref().map(|p| {
        shared
            .front
            .record_frontier(shared.engine.as_ref(), p.origin, &decoded)
    });
    let record_us = frontier.map(|_| record_start.elapsed().as_micros() as u64);
    let recorded = frontier.is_some_and(|f| f.logged);
    report(
        shared,
        "/v1/chat/completions",
        prompt_ids.len(),
        &facts,
        &decoded,
        ms(elapsed),
    );
    let engine = shared.engine.as_ref();
    let mut message = json!({"role": "assistant", "content": decoded.text});
    if split.is_some() {
        message["content"] = routed
            .remove("content")
            .unwrap_or(Value::String(String::new()));
        if let Some(reasoning) = routed.remove("reasoning_content") {
            message["reasoning_content"] = reasoning;
        }
    }
    let mut body = json!({
        "id": id,
        "object": "chat.completion",
        "created": now(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "logprobs": if want_logprobs { chat_logprobs(engine, &decoded)? } else { Value::Null },
            "finish_reason": finish_reason(decoded.stop),
        }],
        "usage": run_usage(shared.engine.as_ref(), prompt_ids.len(), &decoded),
        "timings": timings(shared.engine.as_ref(), prompt_ids.len(), &facts, &decoded, elapsed),
    });
    body["timings"]["logged"] = json!(recorded);
    // What of the reply is the accepted history (decision 42): the whole
    // ids its delivered text begins with, and the text past them.
    body["timings"]["frontier"] = frontier.map_or(Value::Null, |f| f.json());
    body["timings"]["front_record_us"] = json!(record_us);
    // Ids only when asked (`return_token_ids`), and their logprobs only when
    // logprobs are asked too: the conversation the response returns is its
    // text (`sequence-reuse-design.md` §2.3d).
    if return_ids {
        body["prompt_token_ids"] = json!(prompt_ids);
        body["choices"][0]["token_ids"] = json!(decoded.tokens);
        body["timings"]["tokens"] = json!(decoded.tokens);
        if want_logprobs {
            if let Some(lp) = prompt.as_ref().and_then(|p| p.logprobs.as_ref()) {
                if lp.len() == prompt_ids.len() {
                    body["prompt_sampled_logprobs"] = json!(lp);
                }
            }
            if let Some(lp) = &decoded.logprobs {
                body["choices"][0]["token_logprobs"] = json!(lp.sampled);
            }
        }
    }
    attach_vision(shared, &mut body)?;
    Ok(Reply::Json(body))
}

/// The ids a request sent in place of a prompt (`prompt_token_ids`, the
/// token-level API): run as sent.
fn field_token_ids(v: &Value) -> Handled<Option<Vec<u32>>> {
    match v.get("prompt_token_ids") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Array(a)) if !a.is_empty() => a
            .iter()
            .map(|t| {
                t.as_u64()
                    .and_then(|t| u32::try_from(t).ok())
                    .ok_or_else(|| bad("prompt_token_ids: every entry is a token id"))
            })
            .collect::<Handled<Vec<u32>>>()
            .map(Some),
        Some(_) => Err(bad(
            "prompt_token_ids must be a non-empty array of token ids",
        )),
    }
}

/// What a run must capture: the request's `top` logprobs (`Some(0)` for the
/// sampled one alone) and what the session log keeps.
fn logprob_request(
    request: Option<usize>,
    keep: Option<crate::LogprobRequest>,
) -> Option<crate::LogprobRequest> {
    match (request, keep) {
        (None, None) => None,
        (r, k) => Some(crate::LogprobRequest {
            top: r.unwrap_or(0),
            rows: k.is_some_and(|k| k.rows),
        }),
    }
}

/// The returned ids a logprobs block lists beside the text: every id, unless
/// the server withheld text from the reply (a stop string, a fragment the
/// consumer refused); then the ids whose text the delivered text is — the
/// accepted frontier (`front::accepted`) — so the withheld text does not
/// reach the client through its tokens. `token_ids` / `token_logprobs`
/// carry every id either way. Every id on a family with no text side.
fn delivered_ids(engine: &dyn Served, d: &Decoded) -> usize {
    match (d.stop, engine.text()) {
        (Stop::StopString | Stop::Cancelled, Some(t)) => {
            front::accepted(t, &d.tokens, &d.text).map_or(0, |(n, _)| n)
        }
        _ => d.tokens.len(),
    }
}

/// OpenAI's chat `logprobs.content`: per delivered id, its token, its
/// log-probability and the `top_logprobs` asked for.
fn chat_logprobs(engine: &dyn Served, d: &Decoded) -> Result<Value> {
    let Some(lp) = &d.logprobs else {
        return Ok(Value::Null);
    };
    let n = delivered_ids(engine, d);
    let content = d.tokens[..n]
        .iter()
        .zip(&lp.sampled)
        .enumerate()
        .map(|(i, (&id, &l))| {
            let top = match lp.top.get(i) {
                Some(t) => t
                    .iter()
                    .map(|&(t, l)| {
                        Ok(json!({ "token": engine.token_str_checked(t)?, "id": t, "logprob": l }))
                    })
                    .collect::<Result<Vec<Value>>>()?,
                None => Vec::new(),
            };
            Ok(json!({
                "token": engine.token_str_checked(id)?,
                "id": id,
                "logprob": l,
                "top_logprobs": top,
            }))
        })
        .collect::<Result<Vec<Value>>>()?;
    Ok(json!({ "content": content }))
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
    // `generation.sampling.method`, by its key's spelling or as the
    // `gumbel` switch; set both ways they must agree.
    let method = match v.get("sampling_method") {
        None | Some(Value::Null) => None,
        Some(Value::String(m)) if m == "multinomial" => Some(false),
        Some(Value::String(m)) if m == "gumbel" => Some(true),
        Some(_) => {
            return Err(bad(
                "field 'sampling_method' must be \"multinomial\" or \"gumbel\"",
            ))
        }
    };
    let switch = match v.get("gumbel") {
        None | Some(Value::Null) => None,
        Some(b) => Some(
            b.as_bool()
                .ok_or_else(|| bad("field 'gumbel' must be a boolean"))?,
        ),
    };
    let gumbel = match (method, switch) {
        (Some(m), Some(g)) if m != g => {
            return Err(bad("fields 'sampling_method' and 'gumbel' disagree"))
        }
        (Some(m), _) => m,
        (None, Some(g)) => g,
        (None, None) => shared.sampling.gumbel,
    };
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
        turn_tail: 0,
        no_checkpoint: false,
        thinking: None,
        logprobs: None,
        cold: false,
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
    stream_response_with_writer(shared, run, || {
        ChunkedWriter::start(request, "text/event-stream")
    })
}

// The HTTP caller and deterministic I/O-failure controls use the same decode,
// parser finalization, frontier recording and trailer path. Delay opening the
// writer until the fallible reasoning metadata has been checked.
fn stream_response_with_writer(
    shared: &Shared,
    run: StreamingRun,
    start_writer: impl FnOnce() -> Result<ChunkedWriter>,
) -> Result<()> {
    let endpoint = match run.kind {
        StreamKind::Completion => "/v1/completions",
        StreamKind::Chat => "/v1/chat/completions",
    };
    let created = now();
    let split = match run.kind {
        StreamKind::Chat => reasoning::Reasoning::for_chat(
            shared.engine.as_ref(),
            &run.prompt_ids,
            run.reasoning_in_content,
        )?,
        StreamKind::Completion => None,
    }
    .map(|p| Arc::new(parking_lot::Mutex::new(p)));
    // Each drained batch of text becomes one chunk, wrapped on the writer's
    // thread (the counted adapter defers unresolved reasoning prefixes).
    let wrap = {
        let (id, model, kind) = (run.id.clone(), run.model.clone(), run.kind);
        let split = split.clone();
        move |text: &str| -> (String, usize) {
            let (chunk, resolved) = match kind {
                StreamKind::Completion => (
                    json!({
                        "id": id, "object": "text_completion", "created": created, "model": model,
                        "choices": [{ "index": 0, "text": text, "logprobs": null, "finish_reason": null }],
                    }),
                    text.len(),
                ),
                StreamKind::Chat => {
                    let (delta, resolved) = match &split {
                        Some(p) => p.lock().push_counted(text),
                        None => (
                            serde_json::Map::from_iter([("content".into(), json!(text))]),
                            text.len(),
                        ),
                    };
                    (
                        json!({
                            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [{ "index": 0, "delta": delta, "finish_reason": null }],
                        }),
                        resolved,
                    )
                }
            };
            (format!("data: {chunk}\n\n"), resolved)
        }
    };
    let writer = start_writer()?;
    let sw = if split.is_some() {
        StreamWriter::start_with_counted_deltas(writer, wrap)
    } else {
        StreamWriter::start_with_deltas(writer, move |text| wrap(text).0)
    };
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
        run_decode_observed(
            engine,
            &run.prompt_ids,
            &run.images,
            &run.params,
            split.as_ref(),
            &mut |frag| {
                if sw.client_gone() {
                    return false;
                }
                sw.delta(frag);
                true
            },
        )
    };
    let written = sw.written();
    let (mut w, mut write_err) = sw.finish()?;
    let decoded = match decoded {
        Ok(decoded) => decoded,
        Err(error) => return finish_stream_error(w, error),
    };
    if write_err.is_none() {
        if let Some(p) = split {
            let (delta, resolved) = p.lock().finish_counted();
            if !delta.is_empty() {
                match w.event(&json!({
                    "id": run.id, "object": "chat.completion.chunk", "created": created, "model": run.model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": null}],
                })) {
                    Ok(()) => {
                        written.fetch_add(resolved, Ordering::Relaxed);
                    }
                    Err(error) => {
                        write_err = Some(error);
                    }
                }
            }
        }
    }
    // The accepted history is what the server handed the writer; of that,
    // what it knows it wrote, and whether a write failed (what the client
    // received is not known either way).
    let record_start = Instant::now();
    let frontier = run.record.map(|origin| {
        shared
            .front
            .record_frontier(shared.engine.as_ref(), origin, &decoded)
    });
    let record_us = frontier.map(|_| record_start.elapsed().as_micros() as u64);
    let mut frontier_json = frontier.map_or(Value::Null, |f| f.json());
    if frontier_json.is_object() {
        frontier_json["written_bytes"] = json!(written.load(Ordering::Relaxed));
        frontier_json["write_failed"] = json!(write_err.is_some());
    }
    if let Some(e) = write_err {
        anyhow::bail!("client went away mid-stream: {e}; frontier {frontier_json}");
    }
    let elapsed = started.elapsed();
    report(
        shared,
        endpoint,
        run.prompt_ids.len(),
        &run.facts,
        &decoded,
        ms(elapsed),
    );
    let usage = run_usage(shared.engine.as_ref(), run.prompt_ids.len(), &decoded);
    let last = match run.kind {
        StreamKind::Completion => json!({
            "id": run.id, "object": "text_completion", "created": created, "model": run.model,
            "choices": [{ "index": 0, "text": "", "logprobs": null,
                          "finish_reason": finish_reason(decoded.stop) }],
            "usage": usage,
            "timings": timings(shared.engine.as_ref(), run.prompt_ids.len(), &run.facts, &decoded, elapsed),
        }),
        StreamKind::Chat => json!({
            "id": run.id, "object": "chat.completion.chunk", "created": created, "model": run.model,
            "choices": [{ "index": 0, "delta": {}, "finish_reason": finish_reason(decoded.stop) }],
            "usage": usage,
            "timings": timings(shared.engine.as_ref(), run.prompt_ids.len(), &run.facts, &decoded, elapsed),
        }),
    };
    let mut last = last;
    last["timings"]["frontier"] = frontier_json;
    last["timings"]["front_record_us"] = json!(record_us);
    if run.return_ids {
        last["prompt_token_ids"] = json!(run.prompt_ids);
        last["choices"][0]["token_ids"] = json!(decoded.tokens);
        last["timings"]["tokens"] = json!(decoded.tokens);
    }
    write_stream_trailer(w, &mut last, |body| {
        if let StreamKind::Chat = run.kind {
            attach_vision(shared, body)?;
        }
        Ok(())
    })
}

/// Post-header completion: checked metadata refusal propagates as a stream
/// error; neither a successful final event nor [DONE] is written. The status
/// already sent by ChunkedWriter::start cannot become a retroactive HTTP 500.
fn write_stream_trailer(
    mut w: ChunkedWriter,
    last: &mut Value,
    prepare: impl FnOnce(&mut Value) -> Result<()>,
) -> Result<()> {
    if let Err(error) = prepare(last) {
        return finish_stream_error(w, error);
    }
    w.event(last)?;
    w.raw("data: [DONE]\n\n")?;
    w.finish()
}

/// Headers are already on the wire: report a failed run as an error event,
/// never a success trailer or [DONE], then terminate the chunked body. Keep
/// the original failure even if the disconnected transport cannot carry it.
fn finish_stream_error(mut w: ChunkedWriter, error: anyhow::Error) -> Result<()> {
    let message = error.to_string();
    let transport = (|| -> Result<()> {
        w.event(&json!({"error": {"message": message, "type": "server_error"}}))?;
        w.finish()
    })();
    match transport {
        Ok(()) => Err(error),
        Err(transport) => Err(error.context(format!("finishing failed stream: {transport:#}"))),
    }
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
        eprintln!("[/decode_throughput profile: {}]", p.summary(steps));
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
        turn_tail: 0,
        no_checkpoint: false,
        cold: true,
        ..DecodeParams::default()
    };
    let prompts: Vec<Vec<u32>> = (0..batch)
        .map(|r| engine.seed_prompt_row_checked(prompt_tokens, r))
        .collect::<Result<_>>()?;
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
        turn_tail: 0,
        no_checkpoint: false,
        cold: true,
        ..DecodeParams::default()
    };
    if batch == 1 {
        let decoded = engine.complete(&ids, &params, &mut |_| {})?;
        let total_ms = ms(started.elapsed());
        report(
            shared,
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
            shared,
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
                        let built = shared
                            .front
                            .prompt(
                                engine,
                                front::Request {
                                    messages,
                                    kwargs: kwargs.clone(),
                                    want,
                                    add_bos: true,
                                    expand: None,
                                    history: None,
                                },
                            )
?;
                        let ids = built.ids;
                        sw.event_line(&json!({
                            "kind": "eval_sample_start", "sample_id": sample_id, "prompt": built.text,
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
                            turn_tail: 0,
                            no_checkpoint: false,
                            thinking: None,
                            logprobs: None,
                            cold: false,
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
                        let policy = engine.reuse_prefix();
                        let turn = decoded.turn.with_policy(policy);
                        shared.reuse.lock().note(&turn);
                        eprintln!(
                            "[eval {}/{total} id={sample_id} {} tokens {:.1} ms {} readback={} reuse_policy={} reused={} {}]",
                            i + 1,
                            decoded.tokens.len(),
                            ms(started.elapsed()),
                            if stopped_early {
                                "stopped_early"
                            } else {
                                "done"
                            },
                            decoded.readback.name(),
                            crate::turn::policy_name(policy),
                            turn.reused,
                            turn.fields(),
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
    respond_json_text(request, status, body.to_string())
}

fn respond_json_text(request: Request, status: u16, text: String) -> Result<()> {
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
    tx: Option<std::sync::mpsc::Sender<(String, usize)>>,
    gone: Arc<AtomicBool>,
    /// Raw text bytes resolved by successfully written events. A routing
    /// wrapper can defer credit for held marker prefixes until a later event;
    /// deliberately consumed complete markers still count as raw text.
    /// Never a claim about what a client received or JSON/wire byte length.
    written: Arc<AtomicUsize>,
    thread: Option<std::thread::JoinHandle<(ChunkedWriter, Option<std::io::Error>)>>,
    deltas: Option<Arc<DeltaBuffer>>,
    /// The deltas' wire wrapping, shared with the thread: [`Self::send`]
    /// wraps what is pending itself so a control event keeps its place.
    wrap: Option<Wrap>,
    /// Off-token ordering only: extraction, stateful wrapping and enqueue.
    /// Never held during socket I/O or acquired by `delta`.
    order: Arc<parking_lot::Mutex<()>>,
    /// The producer's spill (`DeltaBuffer::push`); only the producer locks it.
    spill: parking_lot::Mutex<String>,
}

type Wrap = Arc<dyn Fn(&str) -> (String, usize) + Send + Sync>;

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
        Self::start_with_counted_deltas(w, move |text| (wrap(text), text.len()))
    }

    /// A routing wrapper reports raw bytes resolved by this event, which can
    /// include a prefix held from an earlier batch. Credit follows successful
    /// I/O, not parser mutation; a failed event resolves no written bytes.
    fn start_with_counted_deltas(
        w: ChunkedWriter,
        wrap: impl Fn(&str) -> (String, usize) + Send + Sync + 'static,
    ) -> Self {
        Self::spawn(w, Some(Arc::new(wrap)))
    }

    fn spawn(mut w: ChunkedWriter, wrap: Option<Wrap>) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<(String, usize)>();
        let gone = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&gone);
        let written = Arc::new(AtomicUsize::new(0));
        let wrote = Arc::clone(&written);
        let deltas = wrap.as_ref().map(|_| DeltaBuffer::new());
        let buf = deltas.clone();
        let thread_wrap = wrap.clone();
        let order = Arc::new(parking_lot::Mutex::new(()));
        let thread_order = Arc::clone(&order);
        let thread = std::thread::spawn(move || {
            let mut err: Option<std::io::Error> = None;
            let mut local = String::with_capacity(4096);
            let write = |w: &mut ChunkedWriter,
                         payload: &str,
                         text: usize,
                         err: &mut Option<std::io::Error>| {
                if err.is_some() {
                    return;
                }
                match w.raw(payload) {
                    Ok(()) => {
                        wrote.fetch_add(text, Ordering::Relaxed);
                    }
                    Err(e) => {
                        flag.store(true, Ordering::Release);
                        *err = Some(e);
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
                let (work, closed) = match next {
                    Ok(work) => (Some(work), false),
                    Err(_) => Self::pending_after_timeout(
                        &rx,
                        buf.as_ref(),
                        thread_wrap.as_ref(),
                        &thread_order,
                        &mut local,
                    ),
                };
                // Selection/wrapping is finished. The sole writer performs
                // network I/O without the off-token ordering guard.
                if let Some((payload, resolved)) = work {
                    write(&mut w, &payload, resolved, &mut err);
                }
                if closed {
                    break;
                }
            }
            (w, err)
        });
        Self {
            tx: Some(tx),
            gone,
            written,
            thread: Some(thread),
            deltas,
            wrap,
            order,
            spill: parking_lot::Mutex::new(String::new()),
        }
    }

    /// A timeout is not a claim that the channel is still empty. Under the
    /// same off-token guard used by send, prefer an already queued event over
    /// bytes appended after that event; then select/wrap at most one batch.
    /// The guard is dropped before the caller performs any network I/O.
    fn pending_after_timeout(
        rx: &std::sync::mpsc::Receiver<(String, usize)>,
        buf: Option<&Arc<DeltaBuffer>>,
        wrap: Option<&Wrap>,
        order: &parking_lot::Mutex<()>,
        local: &mut String,
    ) -> (Option<(String, usize)>, bool) {
        let _guard = order.lock();
        let closed = match rx.try_recv() {
            Ok(work) => return (Some(work), false),
            Err(std::sync::mpsc::TryRecvError::Empty) => false,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => true,
        };
        let work = match (buf, wrap) {
            (Some(buf), Some(wrap)) if buf.take_into(local) => Some(wrap(local)),
            _ => None,
        };
        (work, closed)
    }

    /// Hand text to the writer (the decode thread's per-fragment call on a
    /// writer started with deltas): no wait, no syscall, no allocation.
    fn delta(&self, text: &str) {
        if let Some(b) = &self.deltas {
            b.push(&mut self.spill.lock(), text);
        }
    }

    /// Everything handed over so far, wrapped, queued ahead of what comes
    /// next on the channel. Caller `send` holds `order` through both enqueues.
    /// Off the per-token path (a control event).
    fn queue_pending(&self) {
        if let (Some(b), Some(wrap), Some(tx)) = (&self.deltas, &self.wrap, &self.tx) {
            b.flush(&mut self.spill.lock());
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
        let _guard = self.order.lock();
        self.queue_pending();
        if let Some(tx) = &self.tx {
            // A closed receiver means the thread is gone, which `gone` reports.
            let _ = tx.send((payload, 0));
        }
    }

    /// Raw resolved bytes written so far. Joining [`Self::finish`] settles
    /// queued events; a caller's explicit final routing flush credits this
    /// same counter only after its own successful write.
    fn written(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.written)
    }

    /// The client stopped reading: a write failed on the writer thread.
    fn client_gone(&self) -> bool {
        self.gone.load(Ordering::Acquire)
    }

    /// Drain what is queued, stop the thread, and return the writer with the
    /// first write error, if any.
    fn finish(mut self) -> Result<(ChunkedWriter, Option<std::io::Error>)> {
        if let Some(b) = &self.deltas {
            b.flush(&mut self.spill.lock());
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
            b.flush(&mut self.spill.lock());
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
fn attach_vision(shared: &Shared, body: &mut Value) -> Result<()> {
    let report = shared.engine.vision_report_checked()?;
    if !report.is_null() {
        eprintln!("[vision: {report}]");
        body["vision"] = report;
    }
    Ok(())
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

/// `usage` for a run: `completion_tokens` is the tokens of the completion the
/// client received — the accepted frontier (decision 51 Q4, the standard
/// semantics), which is what [`delivered_ids`] already computes for the
/// logprobs block. The count the run generated stays in `timings`
/// (`timings.tokens`, `timings.frontier.generated`): past a stop string on the
/// adaptive commit with a slow consumer the two differ by up to the backlog,
/// and those ids are text the client never received.
fn run_usage(engine: &dyn Served, prompt: usize, d: &Decoded) -> Value {
    usage(prompt, delivered_ids(engine, d))
}

/// The engine's own brackets beside the request's wall clock: the prompt's
/// submit time, the per-step host clock and the arm that ran. `predicted_ms`
/// is the sum of the loop's steps — the decode window, prefill excluded.
fn timings(
    engine: &dyn Served,
    prompt: usize,
    facts: &PromptFacts,
    d: &Decoded,
    elapsed: std::time::Duration,
) -> Value {
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
    let mut t = json!({
        "prompt_n": prompt,
        "prompt_submit_ms": st.prefill_ns as f64 / 1e6,
        "predicted_n": n,
        "predicted_ms": predicted_ms,
        "predicted_per_second": if predicted_ms > 0.0 { n as f64 / (predicted_ms / 1e3) } else { 0.0 },
        "total_ms": ms(elapsed),
        "readback": d.readback.name(),
        "readback_note": d.readback_note,
        "reuse": format!("{:?}", d.reuse),
        "prefilled": d.prefilled,
        "commits": st.commits,
        "ring_publishes": st.readback.ring_publishes,
        "poll_fallbacks": st.readback.poll_fallbacks,
        "publish_parks": st.readback.publish_parks,
        "overrun": st.overrun,
        // The prompt's pass shapes: forwards the prefill took, at the tile.
        "prefill_units": st.prefill_units,
        "prefill_tile": engine.prefill_tile(),
        "eos_gate": super::common::eos_gate_name(st.eos_gate),
        "host_fill": super::common::host_fill_name(st.host_feed.as_ref()),
        "prompt_ids_sha256": facts.ids_sha256,
        // The template's variable as the prompt reflects it; null for its
        // default. Neutral values can be ignored when the template has no control.
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
        // The front end: renders, and the replies of ours the history carried
        // (matched by text) with the ids they ran as sampled.
        "renders": facts.renders,
        "history_replies": facts.spliced.replies,
        "history_matched": facts.spliced.matched,
        "history_restored": facts.spliced.restored,
        "history_reply_tokens": facts.spliced.reply_tokens,
        "history_spliced_tokens": facts.spliced.spliced_tokens,
        // Assistant messages taken as sent: their text names several
        // replies of this conversation, or a reply of another one only; and
        // reasoning the log would restore, kept as the message gave it.
        "history_ambiguous": facts.spliced.ambiguous,
        "history_unverified": facts.spliced.unverified,
        "history_declined": facts.spliced.declined,
        "reused": d.reused,
        // What the request reused of the session, and why not more: the
        // accepted boundary it found, how the state there was made ready,
        // and the ids kept, replayed and newly prefilled.
        "turn": d.turn.with_policy(engine.reuse_prefix()).json(engine.reuse_prefix()),
    });
    // Matched replies whose run ended inside their reasoning block, and those
    // given to the template as that reasoning.
    t["history_open_reasoning"] = json!(facts.spliced.open);
    t["history_open_apart"] = json!(facts.spliced.open_apart);
    // The turn checkpoint this request's prefill was asked to leave.
    t["checkpoint"] = json!(d.checkpoint_name());
    t["front_prompt_us"] = json!(facts.front_us);
    t
}

/// The request line's steady rate: one over the median step. The per-step
/// clock is the `profile` build's (`nn::decode`), so a build without it has no
/// steps to take a median of and says so rather than printing a measured zero.
fn steady_rate_field(step_ns: &[u64]) -> String {
    let mut steps = step_ns.to_vec();
    steps.sort_unstable();
    match steps.get(steps.len() / 2) {
        Some(&median) => format!("{:.1} tok/s steady (median step)", 1e9 / median as f64),
        None => "steady rate unavailable (per-step clock not in this build)".to_string(),
    }
}

fn report(
    shared: &Shared,
    endpoint: &str,
    prompt: usize,
    facts: &PromptFacts,
    d: &Decoded,
    total_ms: f64,
) {
    let policy = shared.engine.reuse_prefix();
    let turn = d.turn.with_policy(policy);
    shared.reuse.lock().note(&turn);
    let st = &d.stats;
    let steady = steady_rate_field(&st.step_ns);
    eprintln!(
        "[{endpoint}: prompt_tokens={prompt} completion_tokens={} stop={:?} total {total_ms:.1} ms, \
         {steady}, readback={} reuse={:?} prefilled={} \
         commits={} poll_fallbacks={} overrun={} eos_gate={} host_fill={} prompt_ids_sha256={} \
         reasoning_effort={} think={} think_by={} thinking_budget={} thought={} budget_fired={} \
         prompt_budget={} prompt_dropped={} reused={} renders={} history={}/{} spliced={}/{} \
         unverified={} ambiguous={} declined={} reuse_policy={} {} checkpoint={}]",
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
        d.reused,
        facts.renders,
        facts.spliced.matched,
        facts.spliced.replies,
        facts.spliced.spliced_tokens,
        facts.spliced.reply_tokens,
        facts.spliced.unverified,
        facts.spliced.ambiguous,
        facts.spliced.declined,
        crate::turn::policy_name(policy),
        turn.fields(),
        d.checkpoint_name(),
    );
    if let Some(note) = &d.readback_note {
        eprintln!("[{endpoint}: warning: {note}]");
    }
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
    /// The template renders the prompt took (1, or more where the budget's
    /// fit could not cut spans).
    renders: u64,
    /// The replies of ours the history carried, and what of them ran as
    /// sampled.
    spliced: front::Spliced,
    /// The front end's own time for the prompt (scopes, the log's lookup,
    /// the one render, the one tokenization, the alignment): a request's,
    /// never a token's.
    front_us: Option<u64>,
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

/// A front-end failure: the client's 400 when the front end refused the
/// request (an effort the template would not act on, thinking on where the
/// prompt cannot think, an image it cannot expand) or the template refused
/// or failed on its data; the engine's 500 otherwise.
fn front_failure(endpoint: &str, e: anyhow::Error) -> Failure {
    if let Some(r) = e.downcast_ref::<front::Refused>() {
        eprintln!("[{endpoint}: refused: {r}]");
        return bad(r.to_string());
    }
    template_failure(e)
}

/// Propagate admitted metadata refusal as an HTTP error, never successful
/// empty metadata. Outer response assembly/serialization enrollment stays open.
fn respond_metadata(request: Request, result: Result<Value>) -> Result<()> {
    match result {
        Ok(body) => respond_json(request, 200, &body),
        Err(error) => respond_error(request, 500, &format!("{error:#}")),
    }
}

/// What this process's logprobs surface accepts (decision 48): per route,
/// the most candidates beside each id — lloom's `MAX_TOP_LOGPROBS`, an
/// extension past the 0–20 OpenAI's chat API documents — and where it is
/// refused. A generating request's per-token logprobs are refused where
/// the decode body reads no row (`front::logprobs_refused`), and with
/// `stream`; on the chat route with an image too. The one-forward decision
/// scoring (`max_tokens: 1`) reads its row whatever the body.
fn logprobs_capability(engine: &dyn Served) -> Value {
    let top = lloom::nn::MAX_TOP_LOGPROBS;
    let refused = front::logprobs_refused(engine);
    let generating = match &refused {
        None => json!({ "served": true, "top_max": top, "stream": false }),
        Some(why) => json!({ "served": false, "refused": why }),
    };
    let mut chat = generating.clone();
    chat["images"] = json!(false);
    let mut body = json!({
        "served": refused.is_none(),
        "top_max": top,
        "openai_documented_top_max": OPENAI_TOP_LOGPROBS,
        "routes": {
            "/v1/chat/completions": chat,
            "/v1/completions": generating,
            "decision": { "served": true, "top_max": top, "options": "every candidate" },
        },
    });
    if let Some(why) = refused {
        body["refused"] = json!(why);
    }
    body
}

/// The `top_logprobs` bound OpenAI's chat API documents: reported beside
/// lloom's own, never applied (no strict-compatibility adapter is built).
const OPENAI_TOP_LOGPROBS: usize = 20;

/// [`logprobs_capability`] as the load line's `logprobs=` field.
fn logprobs_line(engine: &dyn Served) -> String {
    let top = lloom::nn::MAX_TOP_LOGPROBS;
    match front::logprobs_refused(engine) {
        None => format!("served(top<={top},stream=refused) decision_logprobs=top<={top}"),
        Some(why) => format!("refused({why}) decision_logprobs=top<={top}"),
    }
}

fn health_body(engine: &dyn Served, front: &Front) -> Result<Value> {
    let mut body = engine.health_checked()?;
    body["mem"] = MemSnapshot::read(engine.device()).json();
    body["capabilities"] = engine.capabilities().json();
    body["load_warnings"] = load_warnings_json(engine)?;
    // The request front end's passes so far: a chat request renders its
    // template once and tokenizes the result once.
    body["template_renders"] = json!(engine.template_renders());
    body["tokenizations"] = json!(front.tokenizations());
    body["session_log"] = json!(front.logged());
    // The history policy as resolved for this template: past reasoning kept
    // unless stripped on request, an unfinished reply's rendering.
    body["session"] = front.session_json();
    body["logprobs"] = logprobs_capability(engine);
    body["chat_template_reads"] = ENGINE_KWARGS
        .iter()
        .map(|&k| (k.to_string(), json!(engine.template_reads(k))))
        .collect::<Map<String, Value>>()
        .into();
    body["reasoning_tags"] = match engine.think_tags() {
        Some(t) => json!({
            "start": engine.token_str_checked(t.start)?,
            "end": engine.token_str_checked(t.end)?,
            "tool_call": t.tool_call.map(|id| engine.token_str_checked(id)).transpose()?,
        }),
        None => Value::Null,
    };
    Ok(body)
}

fn models_body(
    model_name: &str,
    engine: &dyn Served,
    decisions: &super::decisions::Decisions,
) -> Result<Value> {
    let capabilities = engine.capabilities();
    let mut input = vec!["text"];
    if capabilities.vision {
        input.push("image");
    }
    Ok(json!({
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": now(),
            "owned_by": "lloom",
            "architecture": {
                "input_modalities": input,
                "output_modalities": decisions.output_modalities(capabilities.generation),
            },
            "load_warnings": load_warnings_json(engine)?,
        }],
    }))
}

/// The load's degradations as `[{kind, message}]` (`Served::load_warnings`).
fn load_warnings_json(engine: &dyn Served) -> Result<Value> {
    Ok(engine
        .load_warnings_checked()?
        .iter()
        .map(|w| json!({ "kind": w.kind.name(), "message": w.message }))
        .collect())
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
    let options = super::resolve_options(&args.common, &|o| args.front(o), Frontend::Bench)?;
    super::report::arm_trace()?;
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
    let load_start = Instant::now();
    let shape = super::Shape::of(&options, Frontend::Bench, seed_text);
    let (engine, model_path) = super::load_engine(&options.load, shape)?;
    let model_name = engine.model_name().to_string();
    let mut sampling = options.sampling.clone();
    sampling.apply_checkpoint(engine.generation_config_checked()?.as_ref());
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
        engine.describe_checked()?,
    );
    eprintln!("[capabilities: {}]", engine.capabilities().line());
    // The cells are the server's handlers; they read the engine through
    // `Shared`, whose listener is unused here (a free loopback port).
    let server = Server::http("127.0.0.1:0").map_err(|e| anyhow::anyhow!("binding: {e}"))?;
    let front = Front::new(engine.as_ref(), front::SessionOptions::of(&options.session));
    let shared = Shared {
        engine,
        decisions: super::decisions::Decisions::Refused(
            "the benchmark cells do not take decisions".into(),
        ),
        turns: turns::Turns::default(),
        decision_pack: false,
        aborts: Mutex::new(HashSet::new()),
        abort_epoch: AtomicU64::new(0),
        server,
        model_name: model_name.clone(),
        sampling,
        front,
        reuse: parking_lot::Mutex::new(crate::turn::Tally::default()),
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
                    let ids = shared.engine.seed_prompt_row_checked(*p, 0)?;
                    let prompt: String = ids
                        .iter()
                        .map(|&t| shared.engine.token_str_checked(t))
                        .collect::<Result<Vec<_>>>()?
                        .concat();
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
        shared.engine.describe_checked()?,
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
    let mut shape = super::Shape::of(&options, Frontend::Bench, DEFAULT_SEED_TEXT.to_string());
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
        engine.describe_checked()?,
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
        let (mut o, _) = super::super::resolve_layers(
            &args.common,
            &front,
            &|_| None,
            false,
            Some(crate::frontend::Frontend::Bench),
        )?;
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
mod steady_rate_tests {
    use super::steady_rate_field;

    #[test]
    fn no_recorded_step_is_unavailable_not_a_zero_rate() {
        assert_eq!(
            steady_rate_field(&[]),
            "steady rate unavailable (per-step clock not in this build)"
        );
    }

    #[test]
    fn recorded_steps_give_one_over_the_median_step() {
        assert_eq!(
            steady_rate_field(&[4_000_000, 1_000_000, 2_000_000]),
            "500.0 tok/s steady (median step)"
        );
        assert_eq!(
            steady_rate_field(&[1_000_000, 3_000_000]),
            "333.3 tok/s steady (median step)"
        );
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

    /// A writer whose writes wait on a gate (a client that stops reading),
    /// or fail (a client that went away).
    #[derive(Clone, Default)]
    struct Gate {
        open: Arc<(parking_lot::Mutex<bool>, parking_lot::Condvar)>,
        written: Arc<Mutex<Vec<u8>>>,
        writes: Arc<std::sync::atomic::AtomicUsize>,
        fail: bool,
    }
    impl Write for Gate {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            if self.fail {
                return Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "gone"));
            }
            self.writes
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let (m, c) = &*self.open;
            let mut open = m.lock();
            while !*open {
                c.wait(&mut open);
            }
            self.written.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn body(out: &[u8]) -> String {
        String::from_utf8(out.to_vec())
            .unwrap()
            .split("\r\n")
            .enumerate()
            .filter(|(i, _)| i % 2 == 1)
            .map(|(_, s)| s.to_string())
            .collect()
    }

    /// A client that stops reading: the decode thread's hand-over never
    /// waits on it, and what waits is the text handed over since the last
    /// write — at most what was pushed, each byte once (no cumulative
    /// snapshots), every byte delivered in order once the client reads.
    /// The control channel is the other queue, and it is unbounded: what
    /// this asserts is the text's backlog, not the channel's.
    #[test]
    fn a_client_that_stops_reading_holds_back_at_most_what_was_pushed() {
        let gate = Gate::default();
        let w = ChunkedWriter {
            w: Box::new(gate.clone()),
        };
        let sw = StreamWriter::start_with_deltas(w, |t: &str| t.to_string());
        sw.send("start".to_string());
        while gate.writes.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        let mut pushed = String::new();
        for i in 0..2000 {
            let frag = format!("{i:04},");
            pushed.push_str(&frag);
            sw.delta(&frag);
        }
        let backlog = sw.deltas.as_ref().unwrap().pending_bytes() + sw.spill.lock().len();
        assert_eq!(
            backlog,
            pushed.len(),
            "nothing taken while the writer waits, nothing twice"
        );
        {
            let (m, c) = &*gate.open;
            *m.lock() = true;
            c.notify_all();
        }
        let (_w, err) = sw.finish().unwrap();
        assert!(err.is_none());
        assert_eq!(
            body(&gate.written.lock().unwrap()),
            format!("start{pushed}")
        );
    }

    /// A client that went away: the failed write is latched, the producer
    /// reads it (`client_gone`), and `finish` returns the error.
    #[test]
    fn a_client_that_went_away_is_reported_to_the_producer() {
        let gate = Gate {
            fail: true,
            ..Gate::default()
        };
        let w = ChunkedWriter {
            w: Box::new(gate.clone()),
        };
        let sw = StreamWriter::start_with_deltas(w, |t: &str| t.to_string());
        sw.delta("a");
        let t = std::time::Instant::now();
        while !sw.client_gone() {
            assert!(
                t.elapsed() < std::time::Duration::from_secs(10),
                "never latched"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let (_w, err) = sw.finish().unwrap();
        assert_eq!(err.map(|e| e.kind()), Some(std::io::ErrorKind::BrokenPipe));
    }
}

#[cfg(test)]
mod systemone_dispatch_tests {
    use super::dispatch_systemone;
    use crate::app::served::tests::NoVision;
    use crate::decision::{
        parse_systemone_questions, NoulAliases, OrderedJson, SystemOneError, SystemOneResponse,
    };
    use std::sync::atomic::Ordering;

    fn answer(body: &OrderedJson) -> std::result::Result<SystemOneResponse, SystemOneError> {
        assert_eq!(
            serde_json::to_string(body.get("state").unwrap()).unwrap(),
            r#"{"z":"雪\n","a":false}"#
        );
        let questions = parse_systemone_questions(body, 255, NoulAliases::TrueFalseOnly)
            .map_err(|e| SystemOneError::InvalidRequest(e.to_string()))?;
        let answers = questions
            .into_iter()
            .map(|(id, question)| {
                question
                    .systemone_answer(&[0.25, 0.75])
                    .map(|answer| (id, answer))
                    .map_err(|e| SystemOneError::Execution(e.to_string()))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        SystemOneResponse::new(answers, 123)
    }

    #[test]
    fn actual_narrow_dispatch_keeps_incumbent_bytes_and_refusal_classification() {
        use super::dispatch_systemone_with;
        let engine = NoVision::new(Some(answer));
        let body=br#"{"state":{"z":"\u96ea\n","a":false},"questions":{"z":{"type":"score","instructions":"rate","criteria":[{"z":null,"a":false},17]},"a":{"type":"choice","instructions":"pick","criteria":{"Z":null,"A":true}}}}"#;
        let incumbent = dispatch_systemone(&engine, "/v1/systemone", body)
            .unwrap()
            .unwrap();
        let mut calls = 0;
        let narrow = dispatch_systemone_with("/v1/systemone", body, |body| {
            calls += 1;
            answer(body)
        })
        .unwrap()
        .unwrap();
        assert_eq!(narrow, incumbent);
        assert_eq!(calls, 1);
        assert!(
            narrow.contains(r#""legend":{"0":{"z":null,"a":false},"1":17}},"a":{"type":"choice""#)
        );
        assert!(narrow.ends_with(r#""usage":{"input_tokens":123,"output_tokens":0}}"#));
        assert!(
            dispatch_systemone_with("/v1/chat/completions", b"not json", |_| panic!(
                "unrelated route called endpoint"
            ))
            .is_none()
        );
        assert!(matches!(
            dispatch_systemone_with("/v1/systemone", b"not json", |_| panic!(
                "invalid JSON called endpoint"
            )),
            Some(Err(SystemOneError::InvalidRequest(_)))
        ));
        assert!(
            matches!(dispatch_systemone_with("/v1/systemone",b"{}",|_|Err(SystemOneError::Unsupported("not installed".into()))),Some(Err(SystemOneError::Unsupported(message))) if message=="not installed")
        );
        assert!(
            matches!(dispatch_systemone_with("/v1/systemone",b"{}",|_|Err(SystemOneError::Execution("refused admission".into()))),Some(Err(SystemOneError::Execution(message))) if message=="refused admission")
        );
    }

    #[test]
    fn actual_ordered_route_calls_loaded_executor_once_and_serializes_without_value() {
        let engine = NoVision::new(Some(answer));
        let body = br#"{"state":{"z":"\u96ea\n","a":false},"questions":{"z":{"type":"score","instructions":"rate","criteria":[{"z":null,"a":false},17]},"a":{"type":"choice","instructions":"pick","criteria":{"Z":null,"A":true}}}}"#;
        let text = dispatch_systemone(&engine, "/v1/systemone", body)
            .unwrap()
            .unwrap();
        assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
        assert!(
            text.starts_with(r#"{"answers":{"z":{"type":"score""#),
            "{text}"
        );
        assert!(
            text.contains(r#""legend":{"0":{"z":null,"a":false},"1":17}},"a":{"type":"choice""#),
            "{text}"
        );
        assert!(
            text.contains(r#""probabilities":{"Z":0.25,"A":0.75}"#),
            "{text}"
        );
        assert!(
            text.ends_with(r#""usage":{"input_tokens":123,"output_tokens":0}}"#),
            "{text}"
        );
    }

    #[test]
    fn unrelated_routes_and_invalid_json_do_not_enter_executor() {
        let engine = NoVision::new(Some(answer));
        assert!(dispatch_systemone(&engine, "/v1/chat/completions", b"not json").is_none());
        assert!(matches!(
            dispatch_systemone(&engine, "/v1/systemone", b"not json"),
            Some(Err(SystemOneError::InvalidRequest(_)))
        ));
        assert_eq!(engine.calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn missing_executor_stays_explicit_and_duplicate_answers_are_refused() {
        let engine = NoVision::new(None);
        assert!(matches!(
            dispatch_systemone(&engine, "/v1/systemone", b"{}"),
            Some(Err(SystemOneError::Unsupported(_)))
        ));
        assert_eq!(engine.calls.load(Ordering::Relaxed), 1);
        let answer = crate::decision::Answer::Noul { probability: 0.5 };
        assert!(SystemOneResponse::new(vec![], 0).is_err());
        assert!(SystemOneResponse::new(
            vec![("q".into(), answer.clone()), ("q".into(), answer)],
            0
        )
        .is_err());
    }
}

#[cfg(test)]
mod metadata_consumer_tests {
    use super::{health_body, models_body, Front};
    use crate::app::served::tests::NoVision;
    use std::sync::atomic::Ordering;

    #[test]
    fn actual_metadata_bodies_propagate_refusal_without_legacy_fallback() {
        let engine = NoVision::refusing_metadata();
        assert_eq!(
            health_body(&engine, &Front::new(&engine, Default::default()))
                .unwrap_err()
                .to_string(),
            "bounded metadata health refusal",
            "actual health body must propagate refusal, not successful metadata",
        );
        assert_eq!(
            models_body(
                "bounded",
                &engine,
                &crate::app::decisions::Decisions::Refused(String::new())
            )
            .unwrap_err()
            .to_string(),
            "bounded metadata warning refusal",
            "actual model card must propagate warning refusal, not an empty list",
        );
        assert_eq!(engine.metadata_calls.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn actual_model_card_reads_current_warnings_on_each_request() {
        let engine = NoVision::dynamic_metadata();
        engine.warning_epoch.store(1, Ordering::Relaxed);
        let first = models_body(
            "bounded",
            &engine,
            &crate::app::decisions::Decisions::Refused(String::new()),
        )
        .unwrap();
        engine.warning_epoch.store(2, Ordering::Relaxed);
        let second = models_body(
            "bounded",
            &engine,
            &crate::app::decisions::Decisions::Refused(String::new()),
        )
        .unwrap();
        let warnings =
            |body: &serde_json::Value| body["data"][0]["load_warnings"].as_array().unwrap().clone();
        assert_eq!(warnings(&first).len(), 1);
        assert_eq!(warnings(&second).len(), 1);
        assert_eq!(warnings(&first)[0]["kind"], "emulated_type");
        assert_eq!(
            warnings(&first)[0]["message"],
            "bounded dynamic warning epoch 1"
        );
        assert_eq!(
            warnings(&second)[0]["message"],
            "bounded dynamic warning epoch 2",
            "actual model card must read current warnings, not a cold snapshot"
        );
        assert_eq!(engine.metadata_calls.load(Ordering::Relaxed), 2);
    }
}

#[cfg(all(test, feature = "metal"))]
#[path = "serve_loaded_tests.rs"]
mod loaded_owner_tests;

#[cfg(test)]
mod checked_trailer_tests {
    use super::{write_stream_trailer, ChunkedWriter};
    use crate::app::served::{tests::NoVision, Served};
    use parking_lot::Mutex;
    use std::io::Write;
    use std::sync::Arc;

    #[derive(Clone, Default)]
    struct Mem(Arc<Mutex<Vec<u8>>>);
    impl Write for Mem {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.lock().extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    // The actual post-decode trailer helper, with an in-memory writer, not a
    // socket/HTTP dispatch or Loaded-owner admission control. Headers are
    // already sent: refusal returns Err, not a retroactive HTTP 500.
    #[test]
    fn checked_vision_refusal_after_headers_emits_no_success_trailer() {
        let engine = NoVision::refusing_metadata();
        let mem = Mem::default();
        let headers = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n";
        mem.0.lock().extend_from_slice(headers);
        let mut writer = ChunkedWriter {
            w: Box::new(mem.clone()),
        };
        writer.raw("data: prior-fragment\n\n").unwrap();
        let before = mem.0.lock().clone();
        let mut last = serde_json::json!({"choices":[{"finish_reason":"stop"}]});
        let error = write_stream_trailer(writer, &mut last, |body| {
            let report = engine.vision_report_checked()?;
            body["vision"] = report;
            Ok(())
        })
        .unwrap_err();
        assert_eq!(error.to_string(), "bounded metadata vision refusal");
        let bytes = mem.0.lock().clone();
        assert!(bytes.starts_with(&before), "prior fragments changed");
        let tail = String::from_utf8_lossy(&bytes[before.len()..]);
        assert!(!tail.contains("[DONE]"), "error masquerades as success");
        assert!(
            !tail.contains("finish_reason"),
            "error wrote success metadata"
        );
        assert!(tail.contains("error"), "error not delivered to the client");
        assert!(tail.ends_with("0\r\n\r\n"), "error left chunked body open");
        assert!(
            last.get("vision").is_none(),
            "refusal fabricated vision metadata"
        );
    }

    #[test]
    fn post_header_error_has_bounded_socket_eof_and_no_success_done() {
        use std::io::Read;
        use std::net::{TcpListener, TcpStream};
        use std::time::Duration;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut client = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        let (server, _) = listener.accept().unwrap();
        let worker = std::thread::spawn(move || {
            let writer = ChunkedWriter {
                w: Box::new(server),
            };
            let mut success = serde_json::json!({"finish_reason":"stop"});
            write_stream_trailer(writer, &mut success, |_| {
                anyhow::bail!("socket test refusal")
            })
            .unwrap_err()
            .to_string()
        });
        let mut bytes = Vec::new();
        client
            .read_to_end(&mut bytes)
            .expect("failed stream must reach EOF, not time out");
        assert_eq!(worker.join().unwrap(), "socket test refusal");
        let text = String::from_utf8(bytes).unwrap();
        assert!(
            text.contains("socket test refusal"),
            "missing error event: {text}"
        );
        assert!(!text.contains("finish_reason") && !text.contains("[DONE]"));
        assert!(
            text.ends_with("0\r\n\r\n"),
            "missing HTTP body terminator: {text}"
        );
    }

    /// Exercise the actual post-header decode-error branch, not only the
    /// trailer helper. Keep the server alive while the client awaits the
    /// HTTP chunk terminator so dropping the fixture cannot fake completion.
    #[test]
    fn failed_decode_after_role_chunk_terminates_the_actual_stream_response() {
        use super::*;
        use std::io::{Read, Write};
        use std::net::TcpStream;
        use std::time::Duration;
        let server = Server::http("127.0.0.1:0").unwrap();
        let tiny_http::ListenAddr::IP(address) = server.server_addr() else {
            panic!("IP server")
        };
        let mut client = TcpStream::connect(address).unwrap();
        client
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        client.write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n").unwrap();
        let request = server
            .recv_timeout(Duration::from_secs(2))
            .unwrap()
            .unwrap();
        let engine = Box::new(NoVision::new(None));
        let front = Front::new(engine.as_ref(), Default::default());
        let shared = Shared {
            engine,
            decisions: super::super::decisions::Decisions::Refused("test".into()),
            turns: turns::Turns::default(),
            decision_pack: false,
            aborts: std::sync::Mutex::new(HashSet::new()),
            abort_epoch: AtomicU64::new(0),
            server,
            model_name: "stub".into(),
            sampling: Default::default(),
            front,
            reuse: parking_lot::Mutex::new(Default::default()),
        };
        let run = StreamingRun {
            kind: StreamKind::Chat,
            prompt_ids: vec![1],
            facts: Default::default(),
            // The fixture refuses image decode at the real Served seam,
            // after stream_response has written its role/header chunk.
            images: vec![DecodedImage {
                height: 1,
                width: 1,
                rgb: vec![0; 3],
                key: [0; 32],
            }],
            params: DecodeParams::default(),
            id: "failure".into(),
            model: "stub".into(),
            record: None,
            return_ids: false,
            reasoning_in_content: false,
        };
        let (release, hold) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            let error = stream_response(&shared, request, run)
                .unwrap_err()
                .to_string();
            let _ = hold.recv_timeout(Duration::from_secs(5));
            error
        });
        let mut wire = Vec::new();
        let mut byte = [0u8; 1];
        while !wire.ends_with(b"0\r\n\r\n") && wire.len() < 16 * 1024 {
            match client.read(&mut byte) {
                Ok(0) => break,
                Ok(_) => wire.push(byte[0]),
                Err(e) => panic!(
                    "stream did not terminate while server stayed alive: {e}; {:?}",
                    String::from_utf8_lossy(&wire)
                ),
            }
        }
        release.send(()).unwrap();
        let error = worker.join().unwrap();
        let text = String::from_utf8(wire).unwrap();
        assert!(error.contains("does not take image input"), "{error}");
        assert!(
            text.contains("\"role\":\"assistant\""),
            "missing initial role: {text}"
        );
        assert!(
            text.contains("does not take image input"),
            "missing decode error: {text}"
        );
        assert!(
            text.ends_with("0\r\n\r\n"),
            "missing HTTP terminator: {text}"
        );
        assert!(!text.contains("[DONE]") && !text.contains("\"finish_reason\":\"stop\""));
    }

    #[test]
    fn successful_checked_trailer_preserves_event_done_and_chunk_termination() {
        let mem = Mem::default();
        let writer = ChunkedWriter {
            w: Box::new(mem.clone()),
        };
        let mut last = serde_json::json!({"ok":true});
        write_stream_trailer(writer, &mut last, |_| Ok(())).unwrap();
        let bytes = mem.0.lock().clone();
        let text = String::from_utf8(bytes).unwrap();
        let event = text
            .find("data: {\"ok\":true}\n\n")
            .expect("success lost final event");
        let done = text.find("data: [DONE]\n\n").expect("success lost [DONE]");
        assert!(event < done, "trailer event reordered after [DONE]");
        assert!(
            text.ends_with("0\r\n\r\n"),
            "success lost chunk termination"
        );
    }
}

#[cfg(test)]
mod logprob_request_tests {
    use super::logprob_request;
    use crate::LogprobRequest;

    /// A request that asks no logprobs, on a session that keeps none,
    /// captures nothing: the run reads no row and takes no blocking arm for
    /// them. Asked 0 is the drawn id's log-probability alone.
    #[test]
    fn a_text_only_request_captures_no_row() {
        assert_eq!(logprob_request(None, None), None);
        assert_eq!(
            logprob_request(Some(0), None),
            Some(LogprobRequest {
                top: 0,
                rows: false
            })
        );
        assert_eq!(
            logprob_request(None, Some(LogprobRequest { top: 0, rows: true })),
            Some(LogprobRequest { top: 0, rows: true }),
            "the session's own keep, never serialised unless asked"
        );
    }
}

#[cfg(test)]
mod ready_configuration_tests {
    use super::ready_configuration;
    use serde_json::json;

    #[test]
    fn pipette_cell_overrides_use_the_same_serve_option_layers() {
        use clap::Parser;
        let args = super::ServeArgs::try_parse_from([
            "lloom serve",
            "--kernels-config",
            r#"{"attention":{"flash":{"partial_block":8}}}"#,
            "--speculative-config",
            r#"{"method":"auto"}"#,
            "--set",
            "kernels.attention.flash.partial_block=32",
            "--speculate",
            "none",
        ])
        .unwrap();
        let (options, _) = crate::app::resolve_layers(
            &args.common,
            &|options| args.front(options),
            &|_| None,
            false,
            Some(crate::frontend::Frontend::Serve),
        )
        .unwrap();
        let flat = crate::option_tree::index().flat(&options);
        assert_eq!(flat["kernels.attention.flash.partial_block"], 32);
        assert_eq!(flat["speculative.method"], "none");
        assert_eq!(flat["context.prefix.reuse"], true);
        assert!(!options.executor.arena && !options.executor.replay);
    }

    #[test]
    fn configured_options_are_not_claimed_as_owner_resolution() {
        let mut options = crate::EngineOptions::default();
        crate::option_tree::index()
            .set(&mut options, "kernels.attention.flash.partial_block", "8")
            .unwrap();
        let report = ready_configuration(&options, true, json!({"readback": "blocking"}));
        assert_eq!(report["schema_version"], 1);
        assert_eq!(
            report["configured_options"]["kernels.attention.flash.partial_block"],
            8
        );
        assert_eq!(report["owner"]["readback"], "blocking");
        assert_eq!(report["options_installed"], true);
        let supplied = ready_configuration(&options, false, json!({"generation": false}));
        assert_eq!(supplied["options_installed"], false);
        assert_eq!(supplied["owner"]["generation"], false);
        assert!(supplied["owner"].get("readback").is_none());
    }
}

#[cfg(test)]
#[path = "serve_stream_tail_tests.rs"]
mod stream_tail_tests;
