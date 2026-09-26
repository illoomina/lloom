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

use super::common::{ms, NEVER_EOS};
use super::vision::{expand_image_tokens, DecodedImage};
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

    /// An image for the first message (PNG or JPEG), repeatable; in the REPL,
    /// `/image PATH` attaches one to the next message. A family or
    /// checkpoint without a vision tower refuses it by name.
    #[arg(long, value_name = "PATH")]
    pub image: Vec<std::path::PathBuf>,
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
        let ids = engine.encode(&rendered, true)?;
        let params = sampling.params(engine.eos_ids(), engine.readback(), engine.prefill_tile());
        let started = Instant::now();
        let mut first_at: Option<f64> = None;
        let mut on_fragment = |frag: &str| {
            if first_at.is_none() {
                first_at = Some(ms(started.elapsed()));
            }
            print!("{frag}");
            let _ = std::io::stdout().flush();
            true
        };
        let decoded = if images.is_empty() {
            engine.complete_until(&ids, &params, &mut on_fragment)
        } else {
            engine.complete_images_until(&ids, &images, &params, &mut on_fragment)
        }
        .context("decoding the turn")?;
        println!();
        messages.push(json!({ "role": "assistant", "content": decoded.text }));
        let st = &decoded.stats;
        let wall = ms(started.elapsed());
        let decode_ms = (wall - st.prefill_ns as f64 / 1e6).max(1e-9);
        eprintln!(
            "[turn {turn}: prompt_tokens={} completion_tokens={} stop={:?} ttft={:.0} ms total={:.0} ms {:.1} tok/s (wall minus submit), readback={} reuse={:?} reused={} prefilled={} commits={} overrun={} tokens_sha256={} prompt_ids_sha256={}]",
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
