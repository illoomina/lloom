//! `lloom chat` — a streaming REPL on the engine: the conversation is
//! rendered through the checkpoint's own chat template each turn and the
//! session reuses the prefix its cache already holds (`Session::prepare`),
//! so turn N pays for its own tokens. One line on stderr per turn says what
//! ran: tokens, the steady rate, the readback arm, what was reused.
//!
//! Images (`--image`, or `/image PATH` for the next message) ride in the
//! message as the template's image parts; each `<image>` the template renders
//! expands to the image's tiles (`vision.rs`). A `[vision: …]` line per turn
//! says what the process holds — the model's tensors, whether the tower is
//! loaded — and, under `CANDLE_METAL_PROFILE=1`, the image's dispatch census.
//!
//! After each turn the run reports (`report.rs`): what the executor arms
//! engaged, the speculative rounds, the profile and trace a variable asked
//! for, the memory peaks under `--mem-sample`, and under `--verbose` the
//! once-per-run diagnostics (the first token's split, the visible cadence,
//! the hot thread, the churn, the lock contention, the KV slab).

use super::budgets::{self, PromptMode};
use super::common::{eos_gate_name, host_fill_name, ms, MemSnapshot, NEVER_EOS};
use super::front::{self, Front};
use super::report;
use super::served::Served;
use super::template::{Rejected, Unrenderable, ENABLE_THINKING, NOT_READ, REASONING_EFFORT};
use super::thinking::{self, Thinking};
use super::vision::{expand_image_tokens, DecodedImage};
use super::{load_engine, Shape};
use crate::frontend::Frontend;
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Map, Value};
use std::io::{BufRead, Write};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "lloom chat",
    about = "Chat with a checkpoint on the lloom engine",
    after_help = super::OPTIONS_HELP
)]
pub struct ChatArgs {
    /// The option layers (`--config`, `--set`, `--<group>-config`) and the
    /// flags generated from the tree. `--readback auto` is `lloom serve`'s
    /// on both families: the flow on LFM2 (the adaptive commit, the reader,
    /// the EOS gate — the forwards past a turn's EOS write nothing, so the
    /// next turn extends the cache) and the blocking arm on Qwen4-exp.
    #[command(flatten)]
    pub common: super::CommonArgs,

    #[command(flatten)]
    pub prefix: crate::options::PrefixConfigArgs,

    #[command(flatten)]
    pub reasoning: crate::options::ReasoningConfigArgs,

    #[command(flatten)]
    pub prompt_options: crate::options::PromptConfigArgs,

    #[command(flatten)]
    pub session: crate::options::SessionConfigArgs,

    #[command(flatten)]
    pub report: crate::options::ReportConfigArgs,

    /// A system message for the conversation.
    #[arg(long, help_heading = SESSION)]
    pub system: Option<String>,

    /// One turn from the command line instead of the REPL, then exit.
    #[arg(long, short = 'p', help_heading = SESSION)]
    pub prompt: Option<String>,

    /// The prompt is already rendered: no chat template, the text encoded as
    /// given with the checkpoint's BOS — what a benchmark runner feeds, so a
    /// reply can be compared token for token.
    #[arg(long, help_heading = SESSION)]
    pub raw: bool,

    /// An image for the first message (PNG or JPEG), repeatable; in the REPL,
    /// `/image PATH` attaches one to the next message. A family or
    /// checkpoint without a vision tower refuses it by name.
    #[arg(long, value_name = "PATH", help_heading = SESSION)]
    pub image: Vec<std::path::PathBuf>,
}

const SESSION: &str = "Session";

impl ChatArgs {
    /// The chat's own flags, as the top layer.
    pub fn front(&self, o: &mut crate::EngineOptions) -> std::result::Result<(), String> {
        self.prefix.apply(&mut o.prefix)?;
        self.reasoning.apply(&mut o.reasoning)?;
        self.prompt_options.apply(&mut o.prompt)?;
        self.session.apply(&mut o.session)?;
        self.report.apply(&mut o.report)
    }
}

/// A turn-line value, `none` when unset.
fn opt(v: Option<impl std::fmt::Display>) -> String {
    v.map_or_else(|| "none".to_string(), |v| v.to_string())
}

/// The reply's token digest in `lfm2-determinism`'s definition: SHA-256 over
/// the generated ids as little-endian `u32`, EOS excluded — so a turn is
/// comparable to the harness's `tokens_sha256` on the same prompt and arm.
pub(super) fn tokens_sha256(ids: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// Refuses a template variable the flags set that the checkpoint's template
/// would not act on, and a value it rejects — on a one-message conversation,
/// before the first turn, so the REPL does not find out at its first line.
fn check_kwargs(
    engine: &dyn Served,
    system: Option<&str>,
    kwargs: &Map<String, Value>,
) -> Result<()> {
    let Some(effort) = kwargs.get(REASONING_EFFORT).and_then(Value::as_str) else {
        return Ok(());
    };
    let mut probe = Vec::new();
    if let Some(s) = system {
        probe.push(json!({ "role": "system", "content": s }));
    }
    probe.push(json!({ "role": "user", "content": "." }));
    let rendered = engine
        .render_checked(&json!(probe), kwargs, &[REASONING_EFFORT])
        .map_err(|e| {
            if let Some(r) = e.downcast_ref::<Rejected>() {
                anyhow::anyhow!("--reasoning-effort {effort}: {r}")
            } else if let Some(r) = e.downcast_ref::<Unrenderable>() {
                anyhow::anyhow!("--reasoning-effort {effort}: {r}")
            } else {
                e
            }
        })?;
    if let Some(ignored) = rendered.ignored.first() {
        let family = engine.capabilities().family;
        if ignored.reason == NOT_READ {
            anyhow::bail!(
                "--reasoning-effort: {family}'s chat template does not read {REASONING_EFFORT}"
            );
        }
        anyhow::bail!(
            "--reasoning-effort {effort}: {family}'s chat template renders the same prompt for any \
             {REASONING_EFFORT} with {ENABLE_THINKING}={}{}",
            kwargs.get(ENABLE_THINKING).unwrap_or(&Value::Null),
            if kwargs.get(ENABLE_THINKING) == Some(&json!(false)) {
                " (add --think)"
            } else {
                ""
            },
        );
    }
    Ok(())
}

/// Refuses `--think` before the first turn where the prompt cannot think: a
/// one-message probe conversation's prompt leaves no block open and the
/// template has no switch that opens one.
fn check_thinking(
    engine: &dyn Served,
    system: Option<&str>,
    kwargs: &Map<String, Value>,
    want: Option<bool>,
) -> Result<()> {
    let mut probe = Vec::new();
    if let Some(s) = system {
        probe.push(json!({ "role": "system", "content": s }));
    }
    probe.push(json!({ "role": "user", "content": "." }));
    let mut ids = engine.encode(&engine.render_chat(&json!(probe), kwargs)?, true)?;
    if let Err(why) = thinking::apply(want, engine.think_tags().as_ref(), &mut ids) {
        anyhow::bail!("--think: {}: {why}", engine.capabilities().family);
    }
    Ok(())
}

/// Refuses `--thinking-budget` before the first turn where it cannot apply:
/// a template that writes no thinking block, or a generation prompt that
/// leaves none open (Qwen4-exp without `--think`), on a one-message probe
/// conversation. A `--raw` prompt is checked at its turn instead.
fn check_thinking_budget(
    engine: &dyn Served,
    system: Option<&str>,
    kwargs: &Map<String, Value>,
    want: Option<bool>,
    budget: usize,
) -> Result<()> {
    let mut probe = Vec::new();
    if let Some(s) = system {
        probe.push(json!({ "role": "system", "content": s }));
    }
    probe.push(json!({ "role": "user", "content": "." }));
    let mut ids = engine.encode(&engine.render_chat(&json!(probe), kwargs)?, true)?;
    let _ = thinking::apply(want, engine.think_tags().as_ref(), &mut ids);
    if let Err(why) = budgets::thinking(
        engine.thinking_budget_refused(),
        engine.think_tags().as_ref(),
        &ids,
        budget,
        &[],
    )? {
        let family = engine.capabilities().family;
        let hint = if why != budgets::NOT_THINKING {
            ""
        } else if want == Some(false) {
            " (drop --no-think)"
        } else if kwargs.get(ENABLE_THINKING) == Some(&json!(false)) {
            " (add --think)"
        } else {
            ""
        };
        anyhow::bail!("--thinking-budget {budget}: {family}: {why}{hint}");
    }
    Ok(())
}

/// One `/decide` line: a `/v1/systemone` request body inline (`{…}`) or
/// in a file, answered as `serve` answers it. The conversation is untouched:
/// a decision reads a fresh sequence, never the chat's.
fn decide(
    decisions: &super::decisions::Decisions,
    engine: &dyn Served,
    request: &str,
) -> Result<String> {
    let text = if request.starts_with('{') {
        request.to_string()
    } else {
        std::fs::read_to_string(request).with_context(|| format!("reading {request}"))?
    };
    let body = crate::decision::OrderedJson::from_slice(text.as_bytes())
        .map_err(|e| anyhow::anyhow!("the request is not JSON: {e}"))?;
    let response = decisions.decide(engine, &body)?;
    Ok(serde_json::to_string(&response)?)
}

pub fn run(args: ChatArgs) -> Result<()> {
    let options = super::resolve_options(&args.common, &|o| args.front(o), Frontend::Chat)?;
    let want = options.reasoning.enabled;
    let reasoning_effort = options.reasoning.effort.clone();
    let add_bos = options.prompt.add_bos;
    let verbose = options.report.verbose;
    let mem_sample = options.report.memory_sample;
    if args.raw && reasoning_effort.is_some() {
        anyhow::bail!(
            "--reasoning-effort sets a chat-template variable; --raw renders no template"
        );
    }
    if args.raw && want.is_some() {
        anyhow::bail!(
            "--think / --no-think act on the chat template's prompt; --raw renders no template"
        );
    }
    // After the tuning install: the trace switch is a tuning knob, and a knob
    // read before the install cannot be installed.
    report::arm_trace();
    // The chat's own long turn is its front-end default (`crate::frontend`).
    let mut sampling = options.sampling.clone();
    let t = Instant::now();
    let mut shape = Shape::of(
        &options,
        Frontend::Chat,
        "The loom threads warp and weft into cloth. ".to_string(),
    );
    // One sequence; the memory probe is the turn's own (below).
    shape.batch = 1;
    shape.rows = "auto".to_string();
    shape.share_prefix = false;
    shape.mem_sample = false;
    let reuse_prefix = shape.reuse_prefix;
    let (engine, path) = load_engine(&options.load, shape)?;
    eprintln!(
        "[loaded {} from {} in {:.1}s on {:?}; readback={} reuse_prefix={} {}; load_warnings={}]",
        engine.model_name(),
        path.display(),
        t.elapsed().as_secs_f64(),
        engine.device(),
        engine.readback().name(),
        reuse_prefix,
        engine.describe_checked()?,
        lloom::load_warning::LoadWarning::line(&engine.load_warnings_checked()?),
    );
    eprintln!("[capabilities: {}]", engine.capabilities().line());
    // `/decide` answers `/v1/systemone` requests on the same loaded model,
    // with the binding `serve` would resolve.
    let decisions = super::decisions::Decisions::resolve(
        options.load.decision_template.as_deref(),
        engine.as_ref(),
        Some(path.as_path()),
    )?;
    if let Some(line) = decisions.describe() {
        eprintln!("[{line}]");
    }
    sampling.apply_checkpoint(engine.generation_config_checked()?.as_ref());
    sampling
        .check()
        .map_err(|e| anyhow::anyhow!("the checkpoint's generation_config.json: {e}"))?;
    // A template with the switch renders thinking off unless asked, so the
    // reply is the answer; one without it renders as it does.
    let mut kwargs = Map::new();
    thinking::set_kwarg(engine.as_ref(), Some(want.unwrap_or(false)), &mut kwargs);
    if let Some(effort) = &reasoning_effort {
        kwargs.insert(REASONING_EFFORT.into(), json!(effort));
    }
    check_kwargs(engine.as_ref(), args.system.as_deref(), &kwargs)?;
    let front = Front::new(engine.as_ref(), front::SessionOptions::of(&options.session));
    eprintln!("[session: {}]", front.line());
    if let Some(why) = front.persist_unsupported() {
        anyhow::bail!(
            "--persist-thinking: {}: {why}",
            engine.capabilities().family
        );
    }
    if let (Some(why), true) = (
        front::logprobs_refused(engine.as_ref()),
        front.options().keep != front::KeepLogprobs::None,
    ) {
        anyhow::bail!("--keep-logprobs: {why}");
    }
    if !args.raw {
        check_thinking(engine.as_ref(), args.system.as_deref(), &kwargs, want)?;
    }
    if let (Some(n), false) = (sampling.thinking_budget, args.raw) {
        check_thinking_budget(engine.as_ref(), args.system.as_deref(), &kwargs, want, n)?;
    }
    let budget_message = match &sampling.thinking_budget_message {
        Some(m) => engine.encode(m, false)?,
        None => Vec::new(),
    };
    let prompt_mode = PromptMode::parse(&sampling.prompt_budget_mode);
    let effort = reasoning_effort.as_deref().unwrap_or("default");
    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = &args.system {
        messages.push(json!({ "role": "system", "content": sys }));
    }
    if args.raw && !args.image.is_empty() {
        anyhow::bail!("--raw takes a rendered prompt; an image goes through the chat template");
    }
    // Every image of the conversation, in order, and those waiting for the
    // next message.
    let mut images: Vec<DecodedImage> = Vec::new();
    let mut pending: Vec<DecodedImage> = Vec::new();
    for path in &args.image {
        let img = DecodedImage::open(path)?;
        engine.image_placeholder(&img)?;
        pending.push(img);
    }
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut turn = 0usize;
    // `--prompt` is one line, read once: a command line (`/decide`, `/image`)
    // that runs no turn must not be read again.
    let mut prompted = false;
    loop {
        let user = match &args.prompt {
            Some(_) if prompted => break,
            Some(p) => {
                prompted = true;
                p.clone()
            }
            None => {
                print!("> ");
                std::io::stdout().flush()?;
                match lines.next() {
                    Some(l) => l?,
                    None => break,
                }
            }
        };
        // A raw prompt is bytes to encode as given (its trailing newline is
        // the template's); a typed line is trimmed.
        let user = if args.raw {
            user
        } else {
            user.trim().to_string()
        };
        if user.trim().is_empty() {
            continue;
        }
        if user == "/quit" || user == "/exit" {
            break;
        }
        if let Some(request) = user.strip_prefix("/decide ") {
            match decide(&decisions, engine.as_ref(), request.trim()) {
                Ok(response) => println!("{response}"),
                Err(e) => eprintln!("[decision refused: {e:#}]"),
            }
            continue;
        }
        if let Some(path) = user.strip_prefix("/image ") {
            if args.raw {
                eprintln!(
                    "[--raw takes a rendered prompt; an image goes through the chat template]"
                );
                continue;
            }
            match DecodedImage::open(std::path::Path::new(path.trim()))
                .and_then(|img| engine.image_placeholder(&img).map(|_| img))
            {
                Ok(img) => {
                    eprintln!(
                        "[image {}x{} attached to your next message]",
                        img.width, img.height
                    );
                    pending.push(img);
                }
                Err(e) => eprintln!("[image refused: {e:#}]"),
            }
            continue;
        }
        turn += 1;
        let renders_before = engine.template_renders();
        if pending.is_empty() {
            messages.push(json!({ "role": "user", "content": user }));
        } else {
            let mut parts: Vec<Value> =
                pending.iter().map(|_| json!({ "type": "image" })).collect();
            parts.push(json!({ "type": "text", "text": user }));
            messages.push(json!({ "role": "user", "content": parts }));
            images.append(&mut pending);
        }
        // One render and one tokenization: the replies of ours the history
        // carries take the ids the model sampled (`front.rs`).
        let built = if args.raw {
            None
        } else {
            let expansions = images
                .iter()
                .map(|img| engine.image_placeholder(img))
                .collect::<Result<Vec<_>>>()?;
            let expand = |r: &str| expand_image_tokens(r, &expansions);
            let expand_ref: front::Expand<'_> = &expand;
            let conv = json!(messages);
            Some(
                front
                    .prompt(
                        engine.as_ref(),
                        front::Request {
                            messages: &conv,
                            kwargs: kwargs.clone(),
                            want,
                            add_bos,
                            expand: (!images.is_empty()).then_some(expand_ref),
                            history: None,
                        },
                    )
                    .map_err(|e| {
                        if let Some(r) = e.downcast_ref::<front::Refused>() {
                            return anyhow::anyhow!("--think: {r}");
                        }
                        e
                    })?,
            )
        };
        let mut ids = match &built {
            Some(p) => p.ids.clone(),
            None => engine.encode(&user, add_bos)?,
        };
        let tags = engine.think_tags();
        let think: Option<Thinking> = built.as_ref().and_then(|p| p.thinking);
        // The prompt budget, before the prompt reaches the engine: truncated,
        // the conversation itself loses its oldest turns, so the next turn
        // extends the cache this one leaves.
        let mut dropped = budgets::Dropped::default();
        if let Some(k) = sampling.prompt_budget.filter(|&k| ids.len() > k) {
            if prompt_mode == PromptMode::Refuse {
                anyhow::bail!(budgets::over_budget(
                    ids.len(),
                    k,
                    "--prompt-budget-mode truncate drops the oldest turns instead"
                ));
            }
            match &built {
                None => (ids, dropped) = budgets::truncate_raw(&ids, k, engine.bos_id()),
                Some(p) => {
                    if !images.is_empty() {
                        anyhow::bail!(
                            "--prompt-budget {k}: the conversation holds an image, and truncating it \
                             would drop turns without their images"
                        );
                    }
                    let fitted = match front.fit(p, &messages, k) {
                        Some(f) => f,
                        None => budgets::fit_messages(&messages, k, ids.len(), |ms| {
                            let mut ids = engine
                                .encode(&engine.render_chat(&json!(ms), &p.kwargs)?, add_bos)?;
                            thinking::apply(want, tags.as_ref(), &mut ids)
                                .map_err(|why| anyhow::anyhow!("--think: {why}"))?;
                            Ok(ids)
                        })?
                        .map(|f| front::Fitted {
                            messages: f.messages,
                            ids: f.ids,
                            dropped: f.dropped,
                        }),
                    };
                    let Some(fitted) = fitted else {
                        anyhow::bail!(budgets::over_budget(
                            ids.len(),
                            k,
                            "the last turn alone does not fit"
                        ));
                    };
                    (messages, ids, dropped) = (fitted.messages, fitted.ids, fitted.dropped);
                }
            }
        }
        let mut params =
            sampling.params(engine.eos_ids(), engine.readback(), engine.prefill_tile());
        params.logits_digest = options.decode.logits_digest;
        params.logprobs = front.options().keep.capture();
        if let Some(p) = &built {
            params.turn_tail = p.turn_tail;
        }
        if let Some(n) = sampling.thinking_budget {
            let refused = engine.thinking_budget_refused();
            params.thinking =
                match budgets::thinking(refused, tags.as_ref(), &ids, n, &budget_message)? {
                    Ok(b) => Some(b),
                    Err(why) => anyhow::bail!("--thinking-budget {n}: {why}"),
                };
        }
        let probe = report::RunProbe::start(engine.device(), mem_sample);
        let started = Instant::now();
        let mut first_at: Option<f64> = None;
        let mut first_device_allocated: Option<u64> = None;
        // One stamp a fragment at most, sized up front: the push never
        // allocates on the decode thread.
        let mut stamps: Vec<f64> =
            Vec::with_capacity(if verbose { params.max_tokens + 1 } else { 0 });
        let device = engine.device().clone();
        // The terminal write happens on the sink's thread: this callback runs
        // on the decode thread, where a `write(2)` and the stdout lock per
        // fragment would sit between the GPU's steps (`sink.rs`).
        let mut out = crate::sink::FragmentSink::start((), |(), text: &str| {
            let mut so = std::io::stdout().lock();
            let _ = so.write_all(text.as_bytes());
            let _ = so.flush();
            true
        });
        let mut on_fragment = |frag: &str| {
            if first_at.is_none() {
                first_at = Some(ms(started.elapsed()));
                if verbose {
                    first_device_allocated = Some(MemSnapshot::read(&device).device_allocated);
                }
            }
            out.push(frag);
            if verbose {
                stamps.push(ms(started.elapsed()));
            }
            true
        };
        let decoded = if images.is_empty() {
            engine.complete_until(&ids, &params, &mut on_fragment)
        } else {
            engine.complete_images_until(&ids, &images, &params, &mut on_fragment)
        };
        let elapsed = started.elapsed();
        out.finish();
        let decoded = decoded.context("decoding the turn")?;
        println!();
        if let Some(p) = &built {
            // The next turn sends `messages` back, which the prompt budget
            // may have cut: the reply answers that history.
            let origin = if dropped.messages > 0 {
                front.origin(&json!(messages), &p.kwargs, p.thinking)
            } else {
                p.origin
            };
            front.record(engine.as_ref(), origin, &decoded);
        }
        messages.push(json!({ "role": "assistant", "content": decoded.text }));
        let st = &decoded.stats;
        let wall = ms(elapsed);
        // The rate after the first token: `prefill_ns` is the prefill's
        // SUBMISSION time, and a one-shot prefill's execution lands in the
        // first step's wait, so `wall - prefill_ns` charged it to decode and
        // read a 512-token tile 6 % slower than a 16-token one on the same
        // rounds (2026-10-19, #859). Tokens after the first over the wall
        // after the first token is what a client measures too.
        let after_first = decoded.tokens.len().saturating_sub(1) as f64;
        let decode_ms = (wall - first_at.unwrap_or(wall)).max(1e-9);
        eprintln!(
            "[turn {turn}: prompt_tokens={} completion_tokens={} stop={:?} ttft={:.0} ms total={:.0} ms {:.1} tok/s (after the first token), readback={} reuse={:?} reused={} prefilled={} commits={} overrun={} tokens_sha256={} logits_sha256={} prompt_ids_sha256={} eos_gate={} host_fill={} max_in_flight={} step_markers={} step_waits={} cut={} pipelined_cut={} ring_publishes={} poll_fallbacks={} publish_parks={} eos_index={}/{} prefill_units={} param-style={} sampling={} reasoning_effort={effort} think={} think_by={} thinking_budget={} thought={} budget_fired={} prompt_budget={} prompt_dropped={} gen_prompt={} renders={} history={}/{} spliced={}/{} reuse_policy={} {} checkpoint={}]",
            ids.len(),
            decoded.tokens.len(),
            decoded.stop,
            first_at.unwrap_or(wall),
            wall,
            after_first / (decode_ms / 1e3),
            decoded.readback.name(),
            decoded.reuse,
            decoded.reused,
            decoded.prefilled,
            st.commits,
            st.overrun,
            tokens_sha256(&decoded.tokens),
            decoded.logits_sha256.as_deref().unwrap_or("-"),
            tokens_sha256(&ids),
            eos_gate_name(st.eos_gate),
            host_fill_name(st.host_feed.as_ref()),
            st.max_in_flight,
            st.step_markers,
            st.step_waits,
            report::cut_policy_arm(engine.device()),
            report::pipelined_cut_arm(engine.device()),
            st.readback.ring_publishes,
            st.readback.poll_fallbacks,
            st.readback.publish_parks,
            st.readback.eos_index_hits,
            st.readback.eos_index_late,
            st.prefill_units,
            report::param_style_arm(),
            params.sampling().name(),
            think.map_or("none", Thinking::name),
            think.map_or("none", |t| t.by.name()),
            opt(sampling.thinking_budget),
            opt(decoded.thinking.map(|t| t.thought)),
            opt(decoded.thinking.map(|t| t.fired)),
            opt(sampling.prompt_budget),
            dropped.tokens,
            params.turn_tail,
            engine.template_renders() - renders_before,
            built.as_ref().map_or(0, |p| p.spliced.matched),
            built.as_ref().map_or(0, |p| p.spliced.replies),
            built.as_ref().map_or(0, |p| p.spliced.spliced_tokens),
            built.as_ref().map_or(0, |p| p.spliced.reply_tokens),
            crate::turn::policy_name(engine.reuse_prefix()),
            decoded.turn.with_policy(engine.reuse_prefix()).fields(),
            decoded.checkpoint_name(),
        );
        if st.readback.poll_fallbacks > 0 {
            eprintln!(
                "[{}: the ring's words were not CPU-mapped; the EOS read took the blocking \
                 fallback ({} of {} steps)]",
                decoded.readback.name(),
                st.readback.poll_fallbacks,
                st.steps
            );
        }
        if let Some((ramp, bulk)) = options.decode.prefill.schedule() {
            let tags: Vec<String> = ramp
                .iter()
                .map(|&m| format!("{m}:{}", report::tile_tag(m)))
                .collect();
            let bulk = if bulk > 0 {
                format!("{bulk}:{}", report::tile_tag(bulk))
            } else {
                "seq_len=1".to_string()
            };
            eprintln!(
                "[spark: ramp=[{}] bulk={bulk} (remainder), then <bulk seq_len=1 tail; \
                 prefill_units={}{}]",
                tags.join(", "),
                st.prefill_units,
                if options.decode.prefill.in_flight > 0 {
                    format!("; prefill_in_flight={}", options.decode.prefill.in_flight)
                } else {
                    String::new()
                },
            );
        }
        if verbose {
            eprintln!(
                "[first-token: submit {:.2} ms (host encode, no GPU wait) | first fragment {} ms \
                 (received and detokenized, handed to the printer thread); tail after the loop {:.2} ms]",
                st.ttft_compute_ns as f64 / 1e6,
                first_at
                    .map(|t| format!("{t:.2}"))
                    .unwrap_or_else(|| "n/a".into()),
                st.tail_ns as f64 / 1e6,
            );
            report::print_visible_cadence(&stamps, decoded.tokens.len());
            if let Some(bytes) = first_device_allocated {
                eprintln!(
                    "[prefill: device_allocated {:.3} GB at the first token (after the prefill and the first step)]",
                    bytes as f64 / 1e9
                );
            }
            for line in engine.diagnostics_checked()? {
                eprintln!("{line}");
            }
        }
        probe.finish(
            engine.device(),
            st,
            report::RunFacts {
                tokens: decoded.tokens.len(),
                prompt_tokens: ids.len(),
                wall: elapsed,
                k: match decoded.readback {
                    crate::ReadbackArm::Speculative { k } => Some(k),
                    _ => None,
                },
            },
            engine.installed(),
            report::Wanted {
                verbose,
                phase_sync: options.decode.phase_sync,
                run_ahead: options.decode.readback.run_ahead,
            },
        );
        let report = engine.vision_report_checked()?;
        if !report.is_null() {
            eprintln!("[vision: {report}]");
        }
        let _ = NEVER_EOS;
    }
    engine.shutdown();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::tokens_sha256;

    /// Pinned to vectors computed outside the crate (`hashlib.sha256` over
    /// `struct.pack('<3I', …)`), not to a second call of the same function.
    #[test]
    fn the_digest_is_the_harness_definition() {
        assert_eq!(
            tokens_sha256(&[]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            tokens_sha256(&[1, 2, 0xdead_beef]),
            "901863406394c0791303fda1bae0e78733375a7c7dbe7db1ea450c6145a0f8d7"
        );
    }
}
