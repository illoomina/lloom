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

    /// Max new tokens to generate.
    #[arg(long, short = 'n', default_value_t = 256)]
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
    /// or plain temperature; top-p is refused on them.
    #[arg(long, default_value = "blocking")]
    readback: String,

    /// Forward passes the host may enqueue before it has learned the previous
    /// step's EOS (ring arms only): 0 or 1. With 1 the GPU always has the next
    /// step queued, and the cache ends one position past an EOS.
    #[arg(long, default_value_t = 0)]
    lookahead: usize,

    /// How often the decode loop commits and prunes the in-flight queue:
    /// `per-token` or `steps:<N>`.
    #[arg(long, default_value = "per-token")]
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
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let config: Lfm2Config = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)
        .context("parsing config.json")?;
    let config = config.into_config(false);

    let files = weight_files(&model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let model = Model::new(&config, vb).context("constructing LFM2")?;
    let cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;

    // EOS ids: the config's, plus the chat-end / end-of-text tokens.
    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }

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
    let decode_start = std::time::Instant::now();
    let out = decode_with(
        &args.readback,
        &mut session,
        &mut spec,
        &mut logits_processor,
        &prompt_ids,
        &opts,
        &eos_ids,
        &device,
    )?;
    let decode_elapsed = decode_start.elapsed();
    let out_ids = out.ids;

    // ---- output: reply to stdout, throughput to stderr --------------------
    let reply = tokenizer
        .decode(&out_ids, true)
        .map_err(|e| anyhow::anyhow!("decoding reply: {e}"))?;
    println!("{reply}");

    let n = out_ids.len();
    let tok_s = n as f64 / decode_elapsed.as_secs_f64();
    let st = &out.stats;
    eprintln!(
        "[{n} tokens in {:.2}s = {tok_s:.1} tok/s decode (prefill included), {} on {:?}; \
         readback={} lookahead={} commit_budget={} commits={} max_in_flight={} \
         ring_publishes={} poll_fallbacks={} overrun={} eos={}]",
        decode_elapsed.as_secs_f64(),
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
fn decode_with(
    readback: &str,
    session: &mut Session<'_, Model>,
    spec: &mut IdentityProposer,
    sampler: &mut LogitsProcessor,
    prompt_ids: &[u32],
    opts: &DecodeOptions,
    eos_ids: &[u32],
    device: &Device,
) -> Result<DecodeOut> {
    // Sized for this one generation; the ring refuses to overflow it.
    let capacity = opts.max_tokens + 1;
    let out = match readback {
        "blocking" => {
            let mut rb = Blocking::new(eos_ids, device)?;
            session.decode(spec, &mut rb, sampler, prompt_ids, opts)?
        }
        "ring" => {
            let mut rb = Ring::new(eos_ids, device, capacity, EosRead::Blocking)?;
            session.decode(spec, &mut rb, sampler, prompt_ids, opts)?
        }
        "ring-nonblocking" => {
            let mut rb = Ring::new(
                eos_ids,
                device,
                capacity,
                EosRead::Poll(PollMechanism::default()),
            )?;
            let out = session.decode(spec, &mut rb, sampler, prompt_ids, opts)?;
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
