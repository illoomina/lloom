//! Qwen3.8-Flash-Next light gate (issue #747): load the real GGUF, decode a
//! prompt greedily, print the text, and hash the token stream — the digest +
//! text pair on the real model (CONTRIBUTING §3.1a), run N times from a fresh
//! cache so the digest also answers "is the build run-to-run identical".
//!
//! No timing beyond the wall clock of load and decode (phase-0 is
//! correctness-only); no perf harness. Exits non-zero when `--expect` is not a
//! substring of the output, or when the runs' digests disagree.
//!
//! ```bash
//! cargo run --release --features metal --example qwen4exp-chat -- --runs 3
//! ```

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use clap::Parser;
use lloom::models::qwen4exp::{Cache, Qwen4Exp, Resident};
use lloom::nn::model::CausalLM;
use lloom::nn::observer::{ForwardObserver, HookPoint};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[path = "../lfm2-determinism/sha256.rs"]
mod sha256;

#[derive(Parser, Debug)]
#[command(about = "Qwen3.8-Flash-Next real-model light gate: digest + text, greedy")]
struct Args {
    /// The first GGUF shard (the whole text model). Default: the HF cache's
    /// ISTA-DASLab IQ3_XXS snapshot.
    #[arg(long)]
    gguf: Option<PathBuf>,

    /// tokenizer.json. Default: the HF cache's Qwen/Qwen3.8-Flash-Next snapshot.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    #[arg(
        long,
        default_value = "What is the capital of France? Answer in one short sentence."
    )]
    prompt: String,

    /// Send the prompt raw (no chat template, no thinking-off prefix).
    #[arg(long)]
    raw: bool,

    /// Substring the output must contain (case-insensitive); exits 1 otherwise.
    #[arg(long)]
    expect: Option<String>,

    #[arg(long, default_value_t = 32)]
    max_tokens: usize,

    /// Fresh-cache repetitions; all must produce one token digest.
    #[arg(long, default_value_t = 1)]
    runs: usize,

    #[arg(long)]
    cpu: bool,

    /// Print per-layer activation statistics (finite count, max |x|, mean |x|)
    /// of the HC carry for the FIRST prompt token, through the observer seam,
    /// plus the final logits' statistics. A bisection aid, not a gate.
    #[arg(long)]
    probe: bool,
}

/// Per-layer activation statistics through the `ForwardObserver` seam.
struct Probe;

fn stats(t: &Tensor) -> Result<(usize, usize, f32, f32)> {
    let v = t.flatten_all()?.to_dtype(candle::DType::F32)?.to_vec1::<f32>()?;
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    let maxabs = v.iter().filter(|x| x.is_finite()).fold(0f32, |m, x| m.max(x.abs()));
    let meanabs = v.iter().filter(|x| x.is_finite()).map(|x| x.abs()).sum::<f32>() / v.len().max(1) as f32;
    Ok((v.len(), nonfinite, maxabs, meanabs))
}

impl ForwardObserver for Probe {
    fn at<P: HookPoint>(&mut self, layer: usize, value: &Tensor) -> candle::Result<()> {
        let (n, nonfinite, maxabs, meanabs) = stats(value).map_err(|e| candle::Error::Msg(e.to_string()))?;
        println!(
            "PROBE layer={layer} point={:?} n={n} nonfinite={nonfinite} maxabs={maxabs:.4e} meanabs={meanabs:.4e}",
            P::KIND
        );
        Ok(())
    }
    fn wants<P: HookPoint>(&self, _layer: usize) -> bool {
        true
    }
}

fn hf_snapshot(repo_dir: &str) -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo_dir)
        .join("snapshots");
    std::fs::read_dir(base)
        .ok()?
        .flatten()
        .next()
        .map(|e| e.path())
}

fn main() -> Result<()> {
    let args = Args::parse();
    let gguf = match args.gguf {
        Some(p) => p,
        None => hf_snapshot("models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
            .context("no HF snapshot for the GGUF; pass --gguf")?
            .join("IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
    };
    let tokenizer = match args.tokenizer {
        Some(p) => p,
        None => hf_snapshot("models--Qwen--Qwen3.8-Flash-Next")
            .context("no HF snapshot for the tokenizer; pass --tokenizer")?
            .join("tokenizer.json"),
    };
    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_metal(0)?
    };

    let tokenizer = Tokenizer::from_file(&tokenizer).map_err(anyhow::Error::msg)?;
    let mut eos_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect();
    eos_ids.dedup();

    let t0 = std::time::Instant::now();
    let mut file =
        std::fs::File::open(&gguf).with_context(|| format!("open {}", gguf.display()))?;
    let content = candle::quantized::gguf_file::Content::read(&mut file)?;
    let n_tensors = content.tensor_infos.len();
    let model: Qwen4Exp<Resident> = Qwen4Exp::from_gguf(content, &mut file, &device)?;
    let cfg = model.config().clone();
    let load_ms = t0.elapsed().as_millis();
    println!(
        "LOADED gguf={} tensors={n_tensors} layers={} hidden={} vocab={} experts={}x{} device={:?} load_ms={load_ms}",
        gguf.display(),
        cfg.n_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.n_experts,
        cfg.n_experts_per_tok,
        device
    );

    let text = if args.raw {
        args.prompt.clone()
    } else {
        // The checkpoint's chat template with thinking disabled
        // (`enable_thinking=false` → an empty think block after the assistant tag).
        format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            args.prompt
        )
    };
    let prompt_ids: Vec<u32> = tokenizer
        .encode(text.as_str(), false)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let mut digests = Vec::with_capacity(args.runs);
    let mut last_text = String::new();
    for run in 0..args.runs {
        let mut cache: Cache = model.new_cache(&cfg, &device)?;
        if args.probe && run == 0 {
            let first = Tensor::from_vec(vec![prompt_ids[0]], (1, 1), &device)?;
            let h = model.embed(&first)?;
            let (n, nonfinite, maxabs, meanabs) = stats(&h)?;
            println!("PROBE embed token={} n={n} nonfinite={nonfinite} maxabs={maxabs:.4e} meanabs={meanabs:.4e}", prompt_ids[0]);
            let mut probe_cache: Cache = model.new_cache(&cfg, &device)?;
            let logits = model.decode_embed(&h, &mut probe_cache, &mut Probe)?;
            let (n, nonfinite, maxabs, meanabs) = stats(&logits)?;
            println!("PROBE logits n={n} nonfinite={nonfinite} maxabs={maxabs:.4e} meanabs={meanabs:.4e} argmax={}", argmax(&logits)?);
        }
        let t1 = std::time::Instant::now();
        let input = Tensor::from_vec(prompt_ids.clone(), (1, prompt_ids.len()), &device)?;
        let mut logits = model.forward(&input, 0, &mut cache)?; // [1, vocab]
        let prefill_ms = t1.elapsed().as_millis();
        let mut generated: Vec<u32> = Vec::with_capacity(args.max_tokens);
        let mut hit_eos = false;
        for _ in 0..args.max_tokens {
            let next = argmax(&logits)?;
            if eos_ids.contains(&next) {
                hit_eos = true;
                break;
            }
            generated.push(next);
            let input = Tensor::from_vec(vec![next], (1, 1), &device)?;
            logits = model.forward(&input, cache.pos, &mut cache)?;
        }
        let decode_ms = t1.elapsed().as_millis() - prefill_ms;
        let mut h = sha256::Sha256::new();
        for t in &generated {
            h.update(&t.to_le_bytes());
        }
        let digest = sha256::hex(&h.finalize());
        last_text = tokenizer
            .decode(&generated, true)
            .map_err(anyhow::Error::msg)?;
        println!("--- run {run} ---");
        println!("prompt: {}", args.prompt);
        println!("output: {last_text}");
        println!(
            "TOKENS {}",
            generated
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
        );
        println!(
            "RESULT run={run} tokens_sha256={digest} prompt_tokens={} generated={} hit_eos={hit_eos} prefill_ms={prefill_ms} decode_ms={decode_ms} max_tokens={}",
            prompt_ids.len(),
            generated.len(),
            args.max_tokens
        );
        digests.push(digest);
    }
    let identical = digests.windows(2).all(|w| w[0] == w[1]);
    println!(
        "DETERMINISM runs={} identical={identical} digest={}",
        args.runs,
        digests.first().cloned().unwrap_or_default()
    );
    let mut ok = identical;
    if let Some(expect) = &args.expect {
        let found = last_text.to_lowercase().contains(&expect.to_lowercase());
        println!("TEXT expect={expect:?} found={found}");
        ok &= found;
    }
    if !ok {
        anyhow::bail!("gate FAILED (identical={identical})");
    }
    Ok(())
}

fn argmax(logits: &Tensor) -> Result<u32> {
    let v = logits.flatten_all()?.to_vec1::<f32>()?;
    let mut best = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    Ok(best as u32)
}
