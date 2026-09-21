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
use lloom::models::qwen4exp::loader::LoadOptions;
use lloom::models::qwen4exp::model::PrefillMode;
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

    /// The GGUF shard holding the N-gram PLE table (`per_layer_token_embd`).
    /// Default: the `-00002-of-00002` sibling of `--gguf` when it exists.
    #[arg(long)]
    ple_shard: Option<PathBuf>,

    /// Load without the N-gram PLE (the #747 baseline the digests were
    /// recorded on).
    #[arg(long)]
    no_ple: bool,

    /// How the prompt is run: `chunk` (Kernel B + chunk attention, the
    /// reference prefill graph) or `replay` (the decode recurrence one position
    /// at a time, #747's path).
    #[arg(long, default_value = "chunk")]
    prefill: String,

    /// The routed experts' activation dtype: `f32` (the byte-identical routed
    /// arm) or `f16` (the `(f16, f32, f16)` arm, issue #748).
    #[arg(long, default_value = "f32")]
    moe_act: String,

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

    /// After the prompt and after generation, print a checksum per state
    /// channel and layer (`STATE`), and a per-step logits checksum (`STEP`),
    /// so two runs can be diffed for where they first diverge.
    #[arg(long)]
    state_digest: bool,

    /// Feed the prompt in segments ending at these positions (comma list, e.g.
    /// `512,2048`) and, with `--state-digest`, print a `STATE ... phase=at<N>`
    /// checksum after each — a bisection aid locating WHERE in the prompt two
    /// runs first diverge. The model chunks each segment exactly as it chunks
    /// the whole prompt when the splits fall on chunk boundaries.
    #[arg(long, value_delimiter = ',')]
    prompt_split: Vec<usize>,

    /// Print per-layer activation statistics (finite count, max |x|, mean |x|)
    /// of the HC carry for the FIRST prompt token, through the observer seam,
    /// plus the final logits' statistics. A bisection aid, not a gate.
    #[arg(long)]
    probe: bool,
}

/// Activation statistics through the `ForwardObserver` seam: verbose prints
/// every point; otherwise it folds a step's points into (first non-finite
/// layer, max |x| over the HC carry) for a per-step summary line.
struct Probe {
    verbose: bool,
    first_nonfinite: Option<(usize, lloom::nn::observer::HookKind)>,
    max_abs: f32,
}

impl Probe {
    fn new(verbose: bool) -> Self {
        Self {
            verbose,
            first_nonfinite: None,
            max_abs: 0.0,
        }
    }
    fn reset(&mut self) {
        self.first_nonfinite = None;
        self.max_abs = 0.0;
    }
}

fn stats(t: &Tensor) -> Result<(usize, usize, f32, f32)> {
    let v = t
        .flatten_all()?
        .to_dtype(candle::DType::F32)?
        .to_vec1::<f32>()?;
    let nonfinite = v.iter().filter(|x| !x.is_finite()).count();
    let maxabs = v
        .iter()
        .filter(|x| x.is_finite())
        .fold(0f32, |m, x| m.max(x.abs()));
    let meanabs = v
        .iter()
        .filter(|x| x.is_finite())
        .map(|x| x.abs())
        .sum::<f32>()
        / v.len().max(1) as f32;
    Ok((v.len(), nonfinite, maxabs, meanabs))
}

impl ForwardObserver for Probe {
    fn at<P: HookPoint>(&mut self, layer: usize, value: &Tensor) -> candle::Result<()> {
        let (n, nonfinite, maxabs, meanabs) =
            stats(value).map_err(|e| candle::Error::Msg(e.to_string()))?;
        if nonfinite > 0 && self.first_nonfinite.is_none() {
            self.first_nonfinite = Some((layer, P::KIND));
        }
        self.max_abs = self.max_abs.max(maxabs);
        if self.verbose {
            println!(
                "PROBE layer={layer} point={:?} n={n} nonfinite={nonfinite} maxabs={maxabs:.4e} meanabs={meanabs:.4e}",
                P::KIND
            );
        }
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
    // `QWEN4EXP_POOL_POISON=<hex byte>`: the #706 GPU-side free-buffer poison
    // (a `fuzzing`-feature build; inert otherwise). A bisection aid for the
    // long-context divergence: a still-reading command buffer sees the
    // sentinel instead of a plausible stale value.
    if let (Ok(hex), Device::Metal(m)) = (std::env::var("QWEN4EXP_POOL_POISON"), &device) {
        let byte = u8::from_str_radix(hex.trim_start_matches("0x"), 16)
            .map_err(|e| anyhow::anyhow!("QWEN4EXP_POOL_POISON={hex}: {e}"))?;
        m.set_free_poison(Some(byte));
        eprintln!("pool free-poison armed: 0x{byte:02x}");
    }

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
    let ple_shard = if args.no_ple {
        None
    } else {
        match args.ple_shard {
            Some(p) => Some(p),
            None => {
                let name = gguf
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default();
                let sibling = gguf.with_file_name(name.replace("00001-of-00002", "00002-of-00002"));
                (sibling != gguf && sibling.exists()).then_some(sibling)
            }
        }
    };
    let moe_act = match args.moe_act.as_str() {
        "f32" => None,
        "f16" => Some(candle::DType::F16),
        other => anyhow::bail!("--moe-act {other}: f32 | f16"),
    };
    let opts = LoadOptions { ple_shard, moe_act };
    let mut model: Qwen4Exp<Resident> =
        Qwen4Exp::from_gguf_with(content, &mut file, &device, &opts)?;
    let prefill_mode = match args.prefill.as_str() {
        "chunk" => PrefillMode::Chunk,
        "replay" => PrefillMode::Replay,
        other => anyhow::bail!("--prefill {other}: chunk | replay"),
    };
    model.set_prefill_mode(prefill_mode);
    let cfg = model.config().clone();
    let load_ms = t0.elapsed().as_millis();
    println!(
        "LOADED gguf={} tensors={n_tensors} layers={} hidden={} vocab={} experts={}x{} device={:?} ple={} prefill={:?} moe_act={} load_ms={load_ms}",
        gguf.display(),
        cfg.n_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.n_experts,
        cfg.n_experts_per_tok,
        device,
        if model.has_ple() { "on" } else { "off" },
        prefill_mode,
        args.moe_act
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
            // Every prompt position, one at a time, with a per-step summary
            // (the first step verbose): where does the state-threaded forward
            // first go non-finite, and how does the carry's magnitude move.
            let mut probe_cache: Cache = model.new_cache(&cfg, &device)?;
            let mut probe = Probe::new(true);
            for (t, &tok) in prompt_ids.iter().enumerate() {
                probe.reset();
                probe.verbose = t == 0;
                let input = Tensor::from_vec(vec![tok], (1, 1), &device)?;
                let h = model.embed(&input)?;
                let (_, e_nonfinite, e_maxabs, _) = stats(&h)?;
                probe_cache.set_pending_tokens(&[tok]);
                let logits = model.decode_embed(&h, &mut probe_cache, &mut probe)?;
                let (n, nonfinite, maxabs, meanabs) = stats(&logits)?;
                println!(
                    "PROBE step={t} token={tok} embed_nonfinite={e_nonfinite} embed_maxabs={e_maxabs:.3e} first_nonfinite={:?} carry_maxabs={:.3e} logits: n={n} nonfinite={nonfinite} maxabs={maxabs:.4e} meanabs={meanabs:.4e} argmax={}",
                    probe.first_nonfinite,
                    probe.max_abs,
                    argmax(&logits)?
                );
            }
        }
        let t1 = std::time::Instant::now();
        let mut logits = None;
        let mut fed = 0usize;
        let mut splits: Vec<usize> = args
            .prompt_split
            .iter()
            .copied()
            .filter(|&n| n > 0 && n < prompt_ids.len())
            .collect();
        splits.sort_unstable();
        splits.dedup();
        splits.push(prompt_ids.len());
        for end in splits {
            let seg = &prompt_ids[fed..end];
            let input = Tensor::from_vec(seg.to_vec(), (1, seg.len()), &device)?;
            logits = Some(model.forward(&input, cache.pos, &mut cache)?); // [1, vocab]
            fed = end;
            if args.state_digest && end < prompt_ids.len() {
                for (name, sum, nonfinite) in cache.checksums()? {
                    println!(
                        "STATE run={run} phase=at{end} {name}={sum:016x} nonfinite={nonfinite}"
                    );
                }
            }
        }
        let mut logits = logits.expect("a non-empty prompt");
        let prefill_ms = t1.elapsed().as_millis();
        if args.state_digest {
            for (name, sum, nonfinite) in cache.checksums()? {
                println!("STATE run={run} phase=prompt {name}={sum:016x} nonfinite={nonfinite}");
            }
        }
        let mut step_no = 0usize;
        let mut generated: Vec<u32> = Vec::with_capacity(args.max_tokens);
        let mut hit_eos = false;
        // The logits digest covers every position's full logit row (prefill's
        // last position + each generated step), so a computation that moves
        // by rounding while the argmax holds is still visible.
        let mut lh = sha256::Sha256::new();
        let mut logits_max = f32::MIN;
        let mut digest_logits = |t: &Tensor, lh: &mut sha256::Sha256| -> Result<()> {
            let v = t.flatten_all()?.to_vec1::<f32>()?;
            for x in &v {
                lh.update(&x.to_le_bytes());
                logits_max = logits_max.max(*x);
            }
            Ok(())
        };
        digest_logits(&logits, &mut lh)?;
        if args.state_digest {
            let v = logits.flatten_all()?.to_vec1::<f32>()?;
            let mut h = 0xcbf29ce484222325u64;
            for x in &v {
                h ^= x.to_bits() as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            println!("STEP run={run} step={step_no} logits={h:016x}");
        }
        for _ in 0..args.max_tokens {
            let next = argmax(&logits)?;
            if eos_ids.contains(&next) {
                hit_eos = true;
                break;
            }
            generated.push(next);
            let input = Tensor::from_vec(vec![next], (1, 1), &device)?;
            logits = model.forward(&input, cache.pos, &mut cache)?;
            digest_logits(&logits, &mut lh)?;
            step_no += 1;
            if args.state_digest {
                let v = logits.flatten_all()?.to_vec1::<f32>()?;
                let mut h = 0xcbf29ce484222325u64;
                for x in &v {
                    h ^= x.to_bits() as u64;
                    h = h.wrapping_mul(0x100000001b3);
                }
                println!("STEP run={run} step={step_no} logits={h:016x}");
            }
        }
        if args.state_digest {
            for (name, sum, nonfinite) in cache.checksums()? {
                println!("STATE run={run} phase=end {name}={sum:016x} nonfinite={nonfinite}");
            }
        }
        let decode_ms = t1.elapsed().as_millis() - prefill_ms;
        let logits_digest = sha256::hex(&lh.finalize());
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
            "RESULT run={run} tokens_sha256={digest} logits_sha256={logits_digest} logits_max={logits_max:.4} prompt_tokens={} generated={} hit_eos={hit_eos} prefill_ms={prefill_ms} decode_ms={decode_ms} max_tokens={}",
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
