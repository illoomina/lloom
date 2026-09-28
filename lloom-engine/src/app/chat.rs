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

use super::common::{eos_gate_name, host_fill_name, ms, MemSnapshot, NEVER_EOS};
use super::report;
use super::vision::{expand_image_tokens, DecodedImage};
use super::{load_engine, ModelArgs, Shape};
use anyhow::{Context, Result};
use clap::Parser;
use serde_json::{json, Value};
use std::io::{BufRead, Write};
use std::time::Instant;

#[derive(Parser)]
#[command(
    name = "lloom chat",
    about = "Chat with a checkpoint on the lloom engine"
)]
pub struct ChatArgs {
    #[command(flatten)]
    pub model: ModelArgs,

    /// The option layers: `--config`, `--decode-config`, `--set`, and the
    /// generated flags (`--readback`, `--lookahead`, `--commit-budget`, …).
    /// `--readback auto` is `lloom serve`'s on both families: the flow on
    /// LFM2 (the adaptive commit, the reader, the EOS gate — the forwards
    /// past a turn's EOS write nothing, so the next turn extends the cache)
    /// and the blocking arm on Qwen4-exp, where `--readback ring-fixed`
    /// feeds the PLE from the host instead (faster while the streamed expert
    /// pool holds the working set, slower while it misses).
    /// `--readback ring-la0` is the ring without the lookahead, whose cache
    /// also ends at the EOS.
    #[command(flatten)]
    pub common: super::CommonArgs,

    /// A system message for the conversation.
    #[arg(long)]
    pub system: Option<String>,

    /// Let the model think (`enable_thinking` on the template); off by
    /// default so the reply is the answer.
    #[arg(long)]
    pub think: bool,

    /// Re-prefill the whole conversation every turn instead of extending
    /// the cache (a measurement arm).
    #[arg(long)]
    pub no_reuse: bool,

    /// One turn from the command line instead of the REPL, then exit.
    #[arg(long, short = 'p')]
    pub prompt: Option<String>,

    /// The prompt is already rendered: no chat template, the text encoded as
    /// given with the checkpoint's BOS — what a benchmark runner feeds, so a
    /// reply can be compared token for token.
    #[arg(long)]
    pub raw: bool,

    /// An image for the first message (PNG or JPEG), repeatable; in the REPL,
    /// `/image PATH` attaches one to the next message. A family or
    /// checkpoint without a vision tower refuses it by name.
    #[arg(long, value_name = "PATH")]
    pub image: Vec<std::path::PathBuf>,

    /// Encode the prompt without the checkpoint's BOS (the template-off A/B:
    /// LFM2's `generation_config.json` starts a sequence with it).
    #[arg(long)]
    pub no_bos: bool,

    /// Sample the memory gauges every 10 ms over each turn and print their
    /// peaks (`MEMPEAK`): the device allocation, the pool's live / free /
    /// pending bytes, the footprint. Off the hot thread.
    #[arg(long)]
    pub mem_sample: bool,

    /// Print each turn's diagnostics: the first token's split, the visible
    /// cadence, the hot thread and page faults, the churn, the lock
    /// contention, the pacer's depth, the KV slab, the speculative rounds.
    #[arg(long, short = 'v')]
    pub verbose: bool,
}

/// The reply's token digest in `lfm2-determinism`'s definition: SHA-256 over
/// the generated ids as little-endian `u32`, EOS excluded — so a turn is
/// comparable to the harness's `tokens_sha256` on the same prompt and arm.
fn tokens_sha256(ids: &[u32]) -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    for id in ids {
        h.update(id.to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

pub fn run(args: ChatArgs) -> Result<()> {
    let options = super::resolve_options(&args.common)?;
    // After the tuning install: the trace switch is a tuning knob, and a knob
    // read before the install cannot be installed.
    report::arm_trace();
    // The chat's own default is a long turn; the group's 512 is the server's.
    let mut sampling = options.sampling.clone();
    if args.common.options.sampling.max_tokens.is_none()
        && !args.common.set.iter().any(|s| s.starts_with("sampling.max_tokens"))
        && sampling.max_tokens == 512
    {
        sampling.max_tokens = 1024;
    }
    let t = Instant::now();
    let (engine, path) = load_engine(
        &args.model,
        Shape {
            decode: options.decode.clone(),
            model: options.model.clone(),
            speculative: options.speculative.clone(),
            executor: options.executor.clone(),
            memory: options.memory.clone(),
            reuse_prefix: !args.no_reuse,
            seed_text: "The loom threads warp and weft into cloth. ".to_string(),
            batch: 1,
            mem_sample: false,
            rows: "auto".to_string(),
            share_prefix: false,
        },
    )?;
    eprintln!(
        "[loaded {} from {} in {:.1}s on {:?}; readback={} reuse_prefix={} {}]",
        engine.model_name(),
        path.display(),
        t.elapsed().as_secs_f64(),
        engine.device(),
        engine.readback().name(),
        !args.no_reuse,
        engine.describe()
    );
    eprintln!("[capabilities: {}]", engine.capabilities().line());
    sampling.apply_checkpoint(engine.generation_config().as_ref());
    sampling
        .check()
        .map_err(|e| anyhow::anyhow!("the checkpoint's generation_config.json: {e}"))?;
    let kwargs = json!({ "enable_thinking": args.think });
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
    loop {
        let user = match &args.prompt {
            Some(p) if turn == 0 => p.clone(),
            Some(_) => break,
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
        if pending.is_empty() {
            messages.push(json!({ "role": "user", "content": user }));
        } else {
            let mut parts: Vec<Value> =
                pending.iter().map(|_| json!({ "type": "image" })).collect();
            parts.push(json!({ "type": "text", "text": user }));
            messages.push(json!({ "role": "user", "content": parts }));
            images.append(&mut pending);
        }
        let rendered = if args.raw {
            user.clone()
        } else {
            engine.render_chat(&json!(messages), Some(&kwargs))?
        };
        let rendered = if images.is_empty() {
            rendered
        } else {
            let expansions = images
                .iter()
                .map(|img| engine.image_placeholder(img))
                .collect::<Result<Vec<_>>>()?;
            expand_image_tokens(&rendered, &expansions)?
        };
        let ids = engine.encode(&rendered, !args.no_bos)?;
        let params = sampling.params(engine.eos_ids(), engine.readback(), engine.prefill_tile());
        let probe = report::RunProbe::start(engine.device(), args.mem_sample);
        let started = Instant::now();
        let mut first_at: Option<f64> = None;
        let mut first_device_allocated: Option<u64> = None;
        let mut stamps: Vec<f64> = Vec::new();
        let verbose = args.verbose;
        let device = engine.device().clone();
        let mut on_fragment = |frag: &str| {
            if first_at.is_none() {
                first_at = Some(ms(started.elapsed()));
                if verbose {
                    first_device_allocated = Some(MemSnapshot::read(&device).device_allocated);
                }
            }
            print!("{frag}");
            let _ = std::io::stdout().flush();
            if verbose {
                stamps.push(ms(started.elapsed()));
            }
            true
        };
        let decoded = if images.is_empty() {
            engine.complete_until(&ids, &params, &mut on_fragment)
        } else {
            engine.complete_images_until(&ids, &images, &params, &mut on_fragment)
        }
        .context("decoding the turn")?;
        let elapsed = started.elapsed();
        println!();
        messages.push(json!({ "role": "assistant", "content": decoded.text }));
        let st = &decoded.stats;
        let wall = ms(elapsed);
        let decode_ms = (wall - st.prefill_ns as f64 / 1e6).max(1e-9);
        eprintln!(
            "[turn {turn}: prompt_tokens={} completion_tokens={} stop={:?} ttft={:.0} ms total={:.0} ms {:.1} tok/s (wall minus submit), readback={} reuse={:?} reused={} prefilled={} commits={} overrun={} tokens_sha256={} prompt_ids_sha256={} eos_gate={} host_fill={} max_in_flight={} step_markers={} step_waits={} cut={} pipelined_cut={} ring_publishes={} poll_fallbacks={} eos_index={}/{} prefill_units={} param-style={}]",
            ids.len(),
            decoded.tokens.len(),
            decoded.stop,
            first_at.unwrap_or(wall),
            wall,
            decoded.tokens.len() as f64 / (decode_ms / 1e3),
            decoded.readback.name(),
            decoded.reuse,
            decoded.reused,
            decoded.prefilled,
            st.commits,
            st.overrun,
            tokens_sha256(&decoded.tokens),
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
            st.readback.eos_index_hits,
            st.readback.eos_index_late,
            st.prefill_units,
            report::param_style_arm(),
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
        if args.verbose {
            eprintln!(
                "[first-token: submit {:.2} ms (host encode, no GPU wait) | first fragment {} ms \
                 (received and detokenized, to stdout); tail after the loop {:.2} ms]",
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
            for line in engine.diagnostics() {
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
                verbose: args.verbose,
                phase_sync: options.decode.phase_sync,
                run_ahead: options.decode.readback.run_ahead,
            },
        );
        let report = engine.vision_report();
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
