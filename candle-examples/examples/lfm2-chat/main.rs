//! LFM2 chat, the smallest useful shape.
//!
//! Reads a prompt from **stdin**, prints the reply to **stdout**, and prints a
//! rough decode throughput (tok/s) to **stderr**. Single turn, greedy by
//! default. No flags you have to learn — point it at a checkpoint and pipe a
//! prompt in.
//!
//!   echo "Explain how a hash map works." \
//!     | cargo run --release --features metal --example lfm2-chat
//!
//! The model directory defaults to the LFM2.5-2.6B snapshot in the HF cache;
//! override with `--model-dir <path>` or `LFM2_MODEL_DIR=<path>`.
//!
//! This is the "pytorchy" path: load config + weights, build the model and a KV
//! cache, then one `Session::decode` call. The loop itself lives in
//! `lloom::nn::decode` (`design/docs/decode-driver-design.md`, stage 1); what
//! this file selects is the readback strategy — `--readback blocking` (the
//! default, the host reads every token back) or `ring` / `ring-nonblocking`
//! (the token stays on the device) — and, on the ring arms, `--lookahead 1`,
//! which lets the host encode the next step before it has learned the previous
//! one's EOS. The perf harness is `lfm2-release`; this one is for reading.

use anyhow::{Context, Result};
use candle::{DType, Device};
use clap::Parser;
use lloom::generation::poll::PollMechanism;
use lloom::generation::{
    Blocking, CommitBudget, EosRead, LogitsProcessor, ReadbackStrategy, Ring, Sampling,
};
use lloom::models::lfm2::{Cache, Lfm2Config, Model};
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, Session, VarBuilder};
use std::io::Read;
use std::path::PathBuf;
use std::str::FromStr;
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    /// Checkpoint directory (config.json, tokenizer.json, *.safetensors).
    /// Defaults to LFM2_MODEL_DIR, then the LFM2.5-2.6B snapshot in the HF cache.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Safety cap on new tokens; generation runs until the model emits EOS
    /// (`<|im_end|>` / `<|endoftext|>`) and this is only the ceiling. Default is
    /// generous so a normal reply — including LFM2.5's `<think>` block — finishes
    /// on EOS rather than being truncated.
    #[arg(long, short = 'n', default_value_t = 2048)]
    max_tokens: usize,

    /// Sampling temperature. 0 (the default) is greedy / argmax.
    #[arg(long, default_value_t = 0.0)]
    temperature: f64,

    /// Sampling seed (only matters when temperature > 0).
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Force CPU even if a GPU is available.
    #[arg(long)]
    cpu: bool,

    /// How the sampled token feeds the next step and how EOS is learned:
    /// `blocking` (the token is read back every step), `ring` (device-resident
    /// token, one blocking flag read per step) or `ring-nonblocking`
    /// (device-resident token, the flag is polled). The ring arms need argmax
    /// or plain temperature; top-p is refused on them. **Defaults to
    /// `ring-nonblocking`**: this example is a speed demo and it is the fastest
    /// arm — the token stays on the device AND the EOS flag is polled rather than
    /// read back, so nothing crosses the host boundary per step (§11.5g). It is
    /// the arm that reaches ~60 tok/s steady on M1 Max at B=1.
    #[arg(long, default_value = "ring-nonblocking")]
    readback: String,

    /// Forward passes the host may enqueue before it has learned the previous
    /// step's EOS (ring arms only): 0 or 1. With 1 the GPU always has the next
    /// step queued, and the cache ends one position past an EOS. **Defaults to
    /// 1**: it is the +3.5 % that collects the non-GPU gap (§11.2b), and this
    /// example is single-turn so the one-past-EOS cache ends with the run — the
    /// exact condition `lfm2-release` refuses `--turns > 1` under (§10.2a).
    #[arg(long, default_value_t = 1)]
    lookahead: usize,

    /// How often the decode loop commits and prunes the in-flight queue:
    /// `per-token` or `steps:<N>`. **Defaults to `steps:8`**: the queue bound is
    /// flat at 13–15 across a 7.8× span of n (#520), so committing every 8 steps
    /// never overflows and takes the commit off the per-token path.
    #[arg(long, default_value = "steps:8")]
    commit_budget: String,
}

/// The LFM2.5-2.6B snapshot dir in the HF cache, if it is there.
fn default_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LFM2_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B/snapshots");
    // One snapshot hash under there; take the first that has a config.json.
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
}

/// The safetensors shards for a checkpoint: the sharded index if present, else
/// the single `model.safetensors`.
fn weight_files(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if !index.is_file() {
        return Ok(vec![dir.join("model.safetensors")]);
    }
    let json: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&index)?).context("reading safetensors index")?;
    let weight_map = json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("safetensors index has no weight_map")?;
    let mut names: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    names.sort();
    names.dedup();
    Ok(names.into_iter().map(|n| dir.join(n)).collect())
}

fn main() -> Result<()> {
    let args = Args::parse();

    let model_dir = args
        .model_dir
        .clone()
        .or_else(default_model_dir)
        .context("no --model-dir given and the LFM2.5-2.6B snapshot was not in the HF cache")?;

    // Device: Metal / CUDA / CPU, chosen by what is available (see
    // candle_examples::device). F16 on GPU matches the inference path; CPU has
    // no f16 kernels for everything here, so it runs in f32.
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    };

    // ---- load: tokenizer, config, weights --------------------------------
    let load_start = std::time::Instant::now();
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let config: Lfm2Config = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)
        .context("parsing config.json")?;
    let mut config = config.into_config(false);
    // The fastest B=1 decode config lloom has, baked in — this example exists to
    // show what the engine does at its best, not the conservative defaults every
    // axis ships at (§7.1a keeps those off in the library; a demo turns them on).
    // The crown attention/KV/conv arms (Sdpa × Cat × Shuffle, §13.5) plus the two
    // GEMV epilogue fusions (§12.2d/§12.2h) plus the conv fold (§12.2 row 1). Every
    // one is token-identical to the generic path — only the logits reorder — so
    // this is the same answer, faster. The readback strategy (ring + lookahead)
    // is selected below, on the loop side.
    config.attn_impl = lloom::models::lfm2::AttnImpl::Sdpa;
    config.kv_append = lloom::models::lfm2::KvAppend::Cat;
    config.conv_state = lloom::models::lfm2::ConvState::Shuffle;
    config.gemv_silu = true;
    config.gemv_residual = true;
    config.conv_fused = true;

    let files = weight_files(&model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    // `Model::new` is the bulk of the load (~950 ms): it reads the 266 weight
    // tensors through the mmap. Measured config-independent — the crown arms
    // above cost it nothing — so it is the construction touching the mapped
    // bytes, not anything this example selects. A warm page cache hides it (the
    // "sub-second" cold-vs-warm difference); the number the demo shows is the
    // honest steady one.
    let model = Model::new(&config, vb).context("constructing LFM2")?;
    let cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;

    // EOS ids: the config's, plus the chat-end / end-of-text tokens.
    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }

    // Load time, printed before we block on stdin so it is attributable. This is
    // mmap + model construction + cache allocation — the safetensors are memory
    // mapped, so the weight *bytes* fault in lazily on the first forward (they
    // land in the prefill / TTFT below), not here.
    eprintln!(
        "[loaded in \x1b[1;96m{:.2}s\x1b[0m (mmap + construct)]",
        load_start.elapsed().as_secs_f64()
    );

    let sampling = if args.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        Sampling::All {
            temperature: args.temperature,
        }
    };
    let mut logits_processor = LogitsProcessor::from_sampling(args.seed, sampling);

    // ---- prompt: read stdin, wrap in the LFM2 chat template ---------------
    let mut prompt = String::new();
    std::io::stdin()
        .read_to_string(&mut prompt)
        .context("reading prompt from stdin")?;
    let prompt = prompt.trim();
    let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let prompt_ids = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
        .get_ids()
        .to_vec();

    // ---- decode: one Session::decode call --------------------------------
    //
    // The strategy is selected here and the loop never names it. Each arm is
    // one value of the `ReadbackStrategy` seam; the loop is generic over it and
    // monomorphizes, so selecting an arm costs nothing per token.
    let budget = CommitBudget::from_str(&args.commit_budget).map_err(anyhow::Error::msg)?;
    let opts = DecodeOptions {
        max_tokens: args.max_tokens,
        budget,
        lookahead: args.lookahead,
        sample_split: false,
    };
    let mut session = Session::new(&model, cache);
    let mut spec = IdentityProposer;

    // ---- decode: stream tokens to stdout as they are produced -------------
    //
    // The callback fires with each committed batch (ring) or each token
    // (blocking) and prints incrementally, so a reply appears as it is written
    // rather than after EOS. `TokenOutputStream` handles the BPE boundary — a
    // token that lands mid-multibyte yields nothing until the next completes it,
    // so the printed text is never broken UTF-8. That the tok/s below holds under
    // this is the point: the drain reads finished work off the ring, it does not
    // wait, so streaming does not re-serialize the non-blocking path.
    use std::io::Write;
    let mut tos = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);
    let decode_start = std::time::Instant::now();
    let out = {
        // Returns `candle::Result`, which is the type `Session::decode`'s callback
        // wants: `candle` is the workspace alias for `lloom-tensor` in this tree,
        // so `candle::Result` and `lloom_tensor::Result` are one type. `next_token`
        // already returns it, so no map is needed.
        let mut on_tokens = |batch: &[u32]| -> candle::Result<()> {
            for &id in batch {
                if let Some(text) = tos.next_token(id)? {
                    print!("{text}");
                    std::io::stdout().flush().ok();
                }
            }
            Ok(())
        };
        decode_with(
            &args.readback,
            &mut session,
            &mut spec,
            &mut logits_processor,
            &prompt_ids,
            &opts,
            &eos_ids,
            &device,
            &mut on_tokens,
        )?
    };
    let decode_elapsed = decode_start.elapsed();
    // The tail the streaming decoder is still holding (a partial multibyte run),
    // then a newline so the reply is its own line before the perf report.
    if let Some(rest) = tos.decode_rest()? {
        print!("{rest}");
    }

    let out_ids = out.ids;
    let n = out_ids.len();
    let end_to_end = n as f64 / decode_elapsed.as_secs_f64();

    println!();
    std::io::stdout().flush().ok();

    let st = &out.stats;

    // Two figures, because they answer different questions (§2.5.3):
    //  * end-to-end includes the prompt's prefill, which is a different shape of
    //    work and drifts with prompt length — what a user of a one-shot waits.
    //  * steady is the median decode step, prefill excluded — what the engine
    //    sustains token-to-token, and the number to compare against the §320
    //    floor. The median, not the mean, because one scheduling hiccup (or a
    //    loaded machine) drags the mean and says nothing about the steady path.
    let mut steps: Vec<u64> = st.step_ns.clone();
    let steady = if steps.is_empty() {
        f64::NAN
    } else {
        // Median is taken on a sorted copy; the unsorted `step_ns[0]` above is the
        // first step, so sort AFTER reading the TTFT term.
        steps.sort_unstable();
        let med_ns = steps[steps.len() / 2] as f64;
        1e9 / med_ns
    };
    // TTFT = prompt submitted → first token available = the prefill forward plus
    // the first decode step (the first token is the prefill's own argmax, fed and
    // sampled in step 0). This is where the mmap'd weights fault in, so a cold
    // run's TTFT carries the load the `[loaded …]` line above deferred.
    let ttft_ms = (st.prefill_ns + st.step_ns.first().copied().unwrap_or(0)) as f64 / 1e6;
    // Bold bright-cyan on the figures worth eyeballing (the reset closes it), so
    // they stand out when scanning a terminal. `\x1b[1;96m` = bold + bright cyan,
    // `\x1b[0m` = reset. Only the numbers are wrapped; the labels stay plain.
    let b = |s: String| format!("\x1b[1;96m{s}\x1b[0m");
    eprintln!(
        "[{n} tokens: {} tok/s end-to-end (prefill included), \
         {} tok/s steady (median decode step), TTFT {} ms, \
         {} on {:?}; \
         readback={} lookahead={} commit_budget={} commits={} max_in_flight={} \
         ring_publishes={} poll_fallbacks={} overrun={} eos={}]",
        b(format!("{end_to_end:.1}")),
        b(format!("{steady:.1}")),
        b(format!("{ttft_ms:.1}")),
        if dtype == DType::F16 { "f16" } else { "f32" },
        device,
        args.readback,
        args.lookahead,
        budget.arm(),
        st.commits,
        st.max_in_flight,
        st.readback.ring_publishes,
        st.readback.poll_fallbacks,
        st.overrun,
        out.eos,
    );
    Ok(())
}

/// Build the selected strategy and run the loop over it.
///
/// Three arms, one call site each: the loop is generic and this is where the
/// type is chosen. An unknown name is an error rather than a fallback (a run
/// that reports one arm and executes another is the failure this refuses).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn decode_with(
    readback: &str,
    session: &mut Session<'_, Model>,
    spec: &mut IdentityProposer,
    sampler: &mut LogitsProcessor,
    prompt_ids: &[u32],
    opts: &DecodeOptions,
    eos_ids: &[u32],
    device: &Device,
    on_tokens: &mut dyn FnMut(&[u32]) -> candle::Result<()>,
) -> Result<DecodeOut> {
    // The device-side id history, sized for this generation; `Ring::new` floors
    // it at a few pages (`RING_HISTORY_FLOOR`), so a short run still gets a
    // generous ring. It refuses to overflow.
    let capacity = opts.max_tokens + 1;
    let out = match readback {
        "blocking" => {
            let mut rb = Blocking::new(eos_ids, device)?;
            session.decode(spec, &mut rb, sampler, prompt_ids, opts, on_tokens)?
        }
        "ring" => {
            let mut rb = Ring::new(eos_ids, device, capacity, EosRead::Blocking)?;
            session.decode(spec, &mut rb, sampler, prompt_ids, opts, on_tokens)?
        }
        "ring-nonblocking" => {
            let mut rb = Ring::new(
                eos_ids,
                device,
                capacity,
                EosRead::Poll(PollMechanism::default()),
            )?;
            let out = session.decode(spec, &mut rb, sampler, prompt_ids, opts, on_tokens)?;
            // The arm that ran is the arm reported: a ring whose poll could not
            // read the mapped words fell back to the blocking read on every
            // step, and says so rather than passing as non-blocking.
            if !rb.poll_is_mapped() {
                eprintln!(
                    "[ring-nonblocking: the ring's words were not CPU-mapped, every EOS read \
                     took the blocking fallback ({} of {} steps)]",
                    out.stats.readback.poll_fallbacks, out.stats.steps
                );
            }
            debug_assert!(rb.arms().nonblocking_eos);
            out
        }
        other => anyhow::bail!(
            "--readback must be `blocking`, `ring` or `ring-nonblocking`, got `{other}`"
        ),
    };
    Ok(out)
}
