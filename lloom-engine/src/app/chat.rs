//! `lloom chat` — a streaming REPL on the engine: the conversation is
//! rendered through the checkpoint's own chat template each turn and the
//! session reuses the prefix its cache already holds (`Session::prepare`),
//! so turn N pays for its own tokens. One line on stderr per turn says what
//! ran: tokens, the steady rate, the readback arm, what was reused.

use super::common::{ms, NEVER_EOS};
use super::{detect_family, load_engine, Family, ModelArgs, Shape};
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
    /// `--readback auto` is the ring WITHOUT the lookahead on LFM2 (the
    /// cache ends at the EOS, so the next turn extends it rather than
    /// re-prefilling the conversation) and the blocking arm on Qwen4-exp.
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
    /// given with the checkpoint's BOS — what `lfm2-chat` and a benchmark
    /// runner feed, so a reply can be compared token for token.
    #[arg(long)]
    pub raw: bool,
}

pub fn run(args: ChatArgs) -> Result<()> {
    let family = detect_family(args.model.model.as_deref())?;
    let mut options = super::resolve_options(&args.common)?;
    if options.decode.readback.kind == crate::ReadbackKind::Auto {
        if let Family::Lfm2(_) = &family {
            // The chat's `auto` on LFM2: the ring without the lookahead, so
            // the cache ends at the EOS and the next turn extends it — as
            // the ring `auto` resolves to, so a host-side sampler (top-p,
            // top-k) still falls back to the blocking arm per request.
            options.decode.readback.lookahead = 0;
            options.decode.readback.adaptive_commit = false;
        }
    }
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
    let kwargs = json!({ "enable_thinking": args.think });
    let mut messages: Vec<Value> = Vec::new();
    if let Some(sys) = &args.system {
        messages.push(json!({ "role": "system", "content": sys }));
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
        turn += 1;
        messages.push(json!({ "role": "user", "content": user }));
        let rendered = if args.raw {
            user.clone()
        } else {
            engine.render_chat(&json!(messages), Some(&kwargs))?
        };
        let ids = engine.encode(&rendered, true)?;
        let params = sampling.params(engine.eos_ids(), engine.readback(), engine.prefill_tile());
        let started = Instant::now();
        let mut first_at: Option<f64> = None;
        let decoded = engine
            .complete_until(&ids, &params, &mut |frag| {
                if first_at.is_none() {
                    first_at = Some(ms(started.elapsed()));
                }
                print!("{frag}");
                let _ = std::io::stdout().flush();
                true
            })
            .context("decoding the turn")?;
        println!();
        messages.push(json!({ "role": "assistant", "content": decoded.text }));
        let st = &decoded.stats;
        let wall = ms(started.elapsed());
        let decode_ms = (wall - st.prefill_ns as f64 / 1e6).max(1e-9);
        eprintln!(
            "[turn {turn}: prompt_tokens={} completion_tokens={} stop={:?} ttft={:.0} ms total={:.0} ms {:.1} tok/s (wall minus submit), readback={} reuse={:?} reused={} prefilled={} commits={} overrun={}]",
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
        );
        let _ = NEVER_EOS;
    }
    engine.shutdown();
    Ok(())
}
