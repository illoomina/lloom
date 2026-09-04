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
//! cache, then a sample-then-forward loop. The perf harness is `lfm2-release`;
//! this one is for reading.

use anyhow::{Context, Result};
use candle::{DType, Tensor};
use clap::Parser;
use lloom::generation::{LogitsProcessor, Sampling};
use lloom::models::lfm2::{Cache, Lfm2Config, Model};
use lloom::nn::VarBuilder;
use std::io::Read;
use std::path::PathBuf;
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
}

/// The LFM2.5-2.6B snapshot dir in the HF cache, if it is there.
fn default_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LFM2_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B/snapshots");
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
    let dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };

    // ---- load: tokenizer, config, weights --------------------------------
    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;

    let config: Lfm2Config = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)
        .context("parsing config.json")?;
    let config = config.into_config(false);

    let files = weight_files(&model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let model = Model::new(&config, vb).context("constructing LFM2")?;
    let mut cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;

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

    // ---- prefill ----------------------------------------------------------
    let input = Tensor::new(prompt_ids.as_slice(), &device)?.unsqueeze(0)?;
    let mut logits = model
        .forward(&input, 0, &mut cache)
        .context("prefill")?
        .squeeze(0)?;
    let mut kv_len = prompt_ids.len();

    // ---- decode: sample-then-forward -------------------------------------
    // The next iteration's sample() reads the logits back to the host, which is
    // itself the GPU sync point; there is no explicit synchronize.
    let mut out_ids: Vec<u32> = Vec::new();
    let decode_start = std::time::Instant::now();
    for _ in 0..args.max_tokens {
        let next = logits_processor.sample(&logits).context("sampling")?;
        if eos_ids.contains(&next) {
            break;
        }
        out_ids.push(next);

        let input = Tensor::new(&[next], &device)?.unsqueeze(0)?;
        logits = model.forward(&input, kv_len, &mut cache).context("decode")?.squeeze(0)?;
        kv_len += 1;
    }
    let decode_elapsed = decode_start.elapsed();

    // ---- output: reply to stdout, throughput to stderr --------------------
    let reply = tokenizer
        .decode(&out_ids, true)
        .map_err(|e| anyhow::anyhow!("decoding reply: {e}"))?;
    println!("{reply}");

    let n = out_ids.len();
    let tok_s = n as f64 / decode_elapsed.as_secs_f64();
    eprintln!(
        "[{n} tokens in {:.2}s = {tok_s:.1} tok/s decode, {} on {:?}]",
        decode_elapsed.as_secs_f64(),
        if dtype == DType::F16 { "f16" } else { "f32" },
        device,
    );
    Ok(())
}
