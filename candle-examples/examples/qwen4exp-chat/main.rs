//! Qwen3.8-Flash-Next light gate (issue #747): load the real GGUF, decode a
//! prompt greedily, print the text, and hash the token stream — the digest +
//! text pair on the real model (CONTRIBUTING §3.1a), run N times from a fresh
//! cache so the digest also answers "is the build run-to-run identical".
//!
//! `--hotlist <file>` (with `--experts stream`) warms every layer's pool
//! from the expert frequency profile of earlier runs before the first token
//! (§20.3 R2) and writes the profile back at exit with this run merged in.
//! `--experts stream` runs the routed experts through the streamed tier
//! (issue #749): a per-layer slot pool of `--slots` experts filled from the
//! shards by `--loader pread|io`; the gate is that its digests equal the
//! resident arm's. `--ple-cache-rows` puts a hot-row cache in front of the
//! N-gram table. A sharded checkpoint is found from its first shard.
//!
//! No timing beyond the wall clock of load and decode (phase-0 is
//! correctness-only); no perf harness. Exits non-zero when `--expect` is not a
//! substring of the output, or when the runs' digests disagree.
//!
//! ```bash
//! cargo run --release --features metal --example qwen4exp-chat -- --runs 3
//! cargo run --release --features metal --example qwen4exp-chat -- --experts stream --slots 64 --runs 2
//! ```

use anyhow::{Context, Result};
use candle::{Device, Tensor};
use clap::Parser;
use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::loader::{LoadOptions, SlotBudget, StreamLoader, StreamOptions};
use lloom::models::qwen4exp::model::PrefillMode;
use lloom::models::qwen4exp::{Cache, ExpertResidency, MtpDrafter, Qwen4Exp, Resident, Streamed};
use lloom::nn::expert_stream::HasStreamStats;
use lloom::nn::model::CausalLM;
use lloom::nn::observer::{ForwardObserver, HookPoint};

/// `CANDLE_METAL_PROFILE=1`: one line per phase from the backend's profiler
/// (§13.1a) — dispatches, command buffers, GPU-busy union against the wall,
/// and the dispatch count per kernel label — then a reset for the next phase.
/// Inert otherwise.
fn profile_line(run: usize, phase: &str, tokens: usize, wall_ms: f64) {
    #[cfg(feature = "metal")]
    {
        use candle::kernels::metal::profile;
        if !profile::enabled() {
            return;
        }
        let s = profile::snapshot();
        let busy = 1e3 * s.gpu_busy_union_s;
        println!(
            "PROFILE run={run} phase={phase} tokens={tokens} wall_ms={wall_ms:.1} gpu_busy_ms={busy:.1} gpu_span_ms={:.1} busy_over_wall={:.3} dispatches={} per_token={:.0} encoders={} command_buffers={} commits={} commit_ms={:.1}",
            1e3 * s.gpu_span_s,
            busy / wall_ms.max(1e-9),
            s.dispatches,
            s.dispatches as f64 / tokens.max(1) as f64,
            s.encoders,
            s.command_buffers,
            s.commits,
            s.commit_ns as f64 / 1e6,
        );
        let mut by = s.by_label.clone();
        by.sort_by(|a, b| b.1.cmp(&a.1));
        let top: Vec<String> = by
            .iter()
            .take(24)
            .map(|(l, n)| format!("{l}={:.1}", *n as f64 / tokens.max(1) as f64))
            .collect();
        println!(
            "PROFILE run={run} phase={phase} labels_per_token {}",
            top.join(" ")
        );
        profile::reset();
    }
    #[cfg(not(feature = "metal"))]
    let _ = (run, phase, tokens, wall_ms);
}
use lloom::nn::speculator::{Anchor, Draft, Probes};
use lloom::nn::{DecodeOptions, Session};
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[path = "../lfm2-determinism/sha256.rs"]
mod sha256;

#[derive(Parser, Debug)]
#[command(about = "Qwen3.8-Flash-Next real-model light gate: digest + text, greedy")]
struct Args {
    /// The first GGUF shard (`-00001-of-0000N`); the siblings are found next
    /// to it. Default: the HF cache's ISTA-DASLab IQ3_XXS snapshot.
    #[arg(long)]
    gguf: Option<PathBuf>,

    /// tokenizer.json. Default: the HF cache's Qwen/Qwen3.8-Flash-Next snapshot.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// The GGUF shard holding the N-gram PLE table (`per_layer_token_embd`).
    /// Default: the shard among the discovered siblings that holds it.
    #[arg(long)]
    ple_shard: Option<PathBuf>,

    /// Load without the N-gram PLE (the #747 baseline the digests were
    /// recorded on).
    #[arg(long)]
    no_ple: bool,

    /// Rows of the PLE hot-row cache (0 = read the memory map every gather).
    #[arg(long, default_value_t = 0)]
    ple_cache_rows: usize,

    /// The GGUF carrying the MTP head (`blk.{n_layers}.*` + `nextn.*`, ds4's
    /// naming — `qwen4exp-mtp-quantize` writes one). After the greedy runs the
    /// prompt is decoded once more through `Session::decode_speculative` with
    /// the `MtpDrafter`; its stream must equal the greedy stream, and the
    /// draft acceptance rate is printed (`MTP …`).
    #[arg(long)]
    mtp: Option<PathBuf>,

    /// Rollback probe: after the prompt, run N verify rounds whose draft is
    /// always wrong (each round rewinds and replays its anchor), then compare
    /// every state channel's checksum against a fresh cache that decoded the
    /// same tokens greedily. Prints the channels that differ. 0 = off.
    #[arg(long, default_value_t = 0)]
    rollback_probe: usize,

    /// How the routed experts are held: `resident` (every expert in RAM) or
    /// `stream` (a per-layer slot pool filled from the shards).
    #[arg(long, default_value = "resident")]
    experts: String,

    /// Streamed experts per layer per projection: a count, or `auto` (sized
    /// from `recommendedMaxWorkingSetSize × --memory-fraction` minus the
    /// resident tier; the `SLOTPLAN` line says what it resolved to).
    #[arg(long, default_value = "64")]
    slots: String,

    /// The fraction of the device's recommended working set `--slots auto`
    /// may plan for.
    #[arg(long, default_value_t = 0.65)]
    memory_fraction: f64,

    /// The streaming loader primitive: `pread` (read + compute-encoder copy)
    /// or `io` (`MTLIOCommandQueue`).
    #[arg(long, default_value = "pread")]
    loader: String,

    /// Decode `N` sequences of the prompt as one batch (`Session::
    /// decode_batched` over `BatchedCausalLM::forward_rows`: one weight
    /// sweep per step for all rows, the state per row) and report the
    /// aggregate rate. 1 = the single-sequence loop.
    #[arg(long, default_value_t = 1)]
    rows: usize,

    /// Run even `--rows 1` through the batched driver (the batched arm's
    /// own baseline).
    #[arg(long)]
    batched_driver: bool,

    /// `--rows N`: the rows' recurrent state in one slot-indexed pool (the
    /// DeltaNet step as one launch per kernel over the rows); off, every
    /// row keeps its own slabs and the step launches per row.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    rows_pool: bool,

    /// The expert hotlist file (`--experts stream`): read at load to warm
    /// every layer's pool before the first token (§20.3 R2; a missing file
    /// warms nothing), written at exit with this run's usage merged in.
    #[arg(long)]
    hotlist: Option<PathBuf>,

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

    /// Print, per generated step, the argmax, the runner-up and their logit
    /// margin (`MARGIN` lines): the discriminator for a token digest that
    /// moved between two arms — a flip at a near-tie is rounding, a flip at
    /// a wide margin is a defect (§2.3.5a).
    #[arg(long)]
    margins: bool,

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

/// `first` plus its `-0000k-of-0000N` siblings, in shard order.
fn discover_shards(first: &PathBuf) -> Vec<PathBuf> {
    let name = first
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default()
        .to_string();
    let Some(pos) = name.find("-of-") else {
        return vec![first.clone()];
    };
    // "...-00001-of-00003.gguf": the count follows "-of-".
    let tail = &name[pos + 4..];
    let count: usize = tail.trim_end_matches(".gguf").parse().unwrap_or(1);
    let head = &name[..pos.saturating_sub(5)];
    let mut out = Vec::with_capacity(count);
    for k in 1..=count {
        let sibling = first.with_file_name(format!("{head}{k:05}-of-{count:05}.gguf"));
        if sibling.exists() {
            out.push(sibling);
        }
    }
    if out.is_empty() {
        vec![first.clone()]
    } else {
        out
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let gguf = match &args.gguf {
        Some(p) => p.clone(),
        None => hf_snapshot("models--ISTA-DASLab--Qwen3.8-Flash-Next-GSQ-RCO-GGUF")
            .context("no HF snapshot for the GGUF; pass --gguf")?
            .join("IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf"),
    };
    let tokenizer = match &args.tokenizer {
        Some(p) => p.clone(),
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
    #[cfg(feature = "metal")]
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
    let mut shards = discover_shards(&gguf);
    if let Some(p) = &args.ple_shard {
        if !shards.contains(p) {
            shards.push(p.clone());
        }
    }
    if let Some(p) = &args.mtp {
        shards.push(p.clone());
    }
    let moe_act = match args.moe_act.as_str() {
        "f32" => None,
        "f16" => Some(candle::DType::F16),
        other => anyhow::bail!("--moe-act {other}: f32 | f16"),
    };
    let opts = LoadOptions {
        ple_shard: None,
        ple: !args.no_ple,
        moe_act,
        ple_cache_rows: (args.ple_cache_rows > 0).then_some(args.ple_cache_rows),
        mtp: args.mtp.is_some(),
    };
    let prefill_mode = match args.prefill.as_str() {
        "chunk" => PrefillMode::Chunk,
        "replay" => PrefillMode::Replay,
        other => anyhow::bail!("--prefill {other}: chunk | replay"),
    };
    let loader = match args.loader.as_str() {
        "pread" => StreamLoader::Pread,
        "io" => StreamLoader::Io,
        other => anyhow::bail!("--loader {other}: pread | io"),
    };
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

    match args.experts.as_str() {
        "resident" => {
            let mut model: Qwen4Exp<Resident> =
                Qwen4Exp::from_gguf_shards(&shards, &device, &opts)?;
            model.set_prefill_mode(prefill_mode);
            if args.rows > 1 || args.batched_driver {
                rows_gate(&model, &args, &device, &prompt_ids, &eos_ids)
            } else {
                gate(
                    &model,
                    &args,
                    &shards,
                    &device,
                    &tokenizer,
                    &prompt_ids,
                    &eos_ids,
                    t0,
                )
            }
        }
        "stream" => {
            let slots = match args.slots.as_str() {
                "auto" => SlotBudget::Auto {
                    fraction: args.memory_fraction,
                },
                n => SlotBudget::Fixed(n.parse().context("--slots: a count or `auto`")?),
            };
            let stream = StreamOptions { slots, loader };
            let mut model: Qwen4Exp<Streamed> =
                Qwen4Exp::from_gguf_streamed(&shards, &device, &opts, &stream)?;
            model.set_prefill_mode(prefill_mode);
            if let Some(path) = &args.hotlist {
                if path.exists() {
                    let tw = std::time::Instant::now();
                    let hot = lloom::models::qwen4exp::hotlist::Hotlist::read(path)?;
                    let loaded = model.warm_from_hotlist(&hot)?;
                    let s = model.stream_stats();
                    println!(
                        "WARM hotlist={} layers={} loaded={loaded} bytes={} ms={}",
                        path.display(),
                        hot.layers.len(),
                        s.bytes_loaded,
                        tw.elapsed().as_millis()
                    );
                } else {
                    println!("WARM hotlist={} absent: a cold pool", path.display());
                }
            }
            let out = if args.rows > 1 || args.batched_driver {
                rows_gate(&model, &args, &device, &prompt_ids, &eos_ids)
            } else {
                gate(
                    &model,
                    &args,
                    &shards,
                    &device,
                    &tokenizer,
                    &prompt_ids,
                    &eos_ids,
                    t0,
                )
            };
            if let Some(path) = &args.hotlist {
                use lloom::models::qwen4exp::hotlist::Hotlist;
                let mut hot = if path.exists() {
                    Hotlist::read(path)?
                } else {
                    Hotlist::default()
                };
                hot.merge(&Hotlist::from_usage(&model.stream_usage()));
                hot.write(path)?;
                println!(
                    "HOTLIST written={} layers={} entries={}",
                    path.display(),
                    hot.layers.len(),
                    hot.layers.iter().map(|l| l.len()).sum::<usize>()
                );
            }
            out
        }
        other => anyhow::bail!("--experts {other}: resident | stream"),
    }
}

#[allow(clippy::too_many_arguments)]
/// `--rows N`: the prompt decoded `N` times at once through the batched
/// driver; `ROWS` lines with the per-run aggregate rate.
#[allow(clippy::too_many_arguments)]
fn rows_gate<R: ExpertResidency + HasStreamStats>(
    model: &Qwen4Exp<R>,
    args: &Args,
    device: &Device,
    prompt_ids: &[u32],
    eos_ids: &[u32],
) -> Result<()> {
    use lloom::generation::{LogitsProcessor, Sampling};
    use lloom::nn::{DecodeOptions, Session};
    let cfg = model.config();
    for run in 0..args.runs {
        let pools = args.rows_pool.then(|| model.state_pools(args.rows, device));
        let mut sessions: Vec<Session<'_, Qwen4Exp<R>>> = (0..args.rows)
            .map(|_| {
                let cache = match &pools {
                    Some(p) => lloom::models::qwen4exp::Cache::in_pools(cfg, p)?,
                    None => model.new_cache(cfg, device)?,
                };
                Ok(Session::new(model, cache))
            })
            .collect::<Result<_>>()?;
        let prompts: Vec<&[u32]> = (0..args.rows).map(|_| prompt_ids).collect();
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(args.max_tokens);
        #[cfg(feature = "metal")]
        candle::kernels::metal::profile::reset();
        let t = std::time::Instant::now();
        let outs = Session::decode_batched(
            &mut sessions,
            &mut sampler,
            &prompts,
            &opts,
            eos_ids,
            device,
            &mut |_row, _toks| Ok(()),
        )?;
        let wall = t.elapsed();
        let generated: usize = outs.iter().map(|o| o.ids.len()).sum();
        // The whole run: the rows' prefills and the batched steps (per
        // token = per generated token over all rows).
        profile_line(run, "rows", generated, wall.as_secs_f64() * 1e3);
        let digests: Vec<String> = outs
            .iter()
            .map(|o| {
                let mut h = sha256::Sha256::new();
                for id in &o.ids {
                    h.update(&id.to_le_bytes());
                }
                sha256::hex(&h.finalize())[..8].to_string()
            })
            .collect();
        println!(
            "ROWS run={run} rows={} pool={} prompt_tokens={} generated={generated} wall_ms={} tok_s={:.2} per_row_tok_s={:.2} digests={:?}",
            args.rows,
            args.rows_pool,
            prompt_ids.len(),
            wall.as_millis(),
            generated as f64 / wall.as_secs_f64(),
            generated as f64 / wall.as_secs_f64() / args.rows as f64,
            digests
        );
    }
    Ok(())
}

fn gate<R: ExpertResidency + HasStreamStats>(
    model: &Qwen4Exp<R>,
    args: &Args,
    shards: &[PathBuf],
    device: &Device,
    tokenizer: &Tokenizer,
    prompt_ids: &[u32],
    eos_ids: &[u32],
    t0: std::time::Instant,
) -> Result<()> {
    let cfg = model.config().clone();
    let load_ms = t0.elapsed().as_millis();
    println!(
        "LOADED gguf={} shards={} layers={} hidden={} vocab={} experts={}x{} device={:?} ple={} ple_cache_rows={} prefill={:?} moe_act={} experts_mode={} slots={} loader={} load_ms={load_ms}",
        shards[0].display(),
        shards.len(),
        cfg.n_layers,
        cfg.hidden_size,
        cfg.vocab_size,
        cfg.n_experts,
        cfg.n_experts_per_tok,
        device,
        if model.has_ple() { "on" } else { "off" },
        args.ple_cache_rows,
        model.prefill_mode(),
        args.moe_act,
        args.experts,
        args.slots,
        args.loader,
    );

    if args.rollback_probe > 0 {
        rollback_probe(
            model,
            &cfg,
            device,
            prompt_ids,
            eos_ids,
            args.rollback_probe,
        )?;
    }
    let mut digests = Vec::with_capacity(args.runs);
    let mut last_text = String::new();
    for run in 0..args.runs {
        let mut cache: Cache = model.new_cache(&cfg, device)?;
        if args.probe && run == 0 {
            let first = Tensor::from_vec(vec![prompt_ids[0]], (1, 1), device)?;
            let h = model.embed(&first)?;
            let (n, nonfinite, maxabs, meanabs) = stats(&h)?;
            println!("PROBE embed token={} n={n} nonfinite={nonfinite} maxabs={maxabs:.4e} meanabs={meanabs:.4e}", prompt_ids[0]);
            // Every prompt position, one at a time, with a per-step summary
            // (the first step verbose): where does the state-threaded forward
            // first go non-finite, and how does the carry's magnitude move.
            let mut probe_cache: Cache = model.new_cache(&cfg, device)?;
            let mut probe = Probe::new(true);
            for (t, &tok) in prompt_ids.iter().enumerate() {
                probe.reset();
                probe.verbose = t == 0;
                let input = Tensor::from_vec(vec![tok], (1, 1), device)?;
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
            let input = Tensor::from_vec(seg.to_vec(), (1, seg.len()), device)?;
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
        profile_line(run, "prefill", prompt_ids.len(), prefill_ms as f64);
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
            println!(
                "STEP run={run} step={step_no} logits={:016x}",
                fnv(&logits)?
            );
        }
        for _ in 0..args.max_tokens {
            let next = argmax(&logits)?;
            if args.margins {
                let v = logits.flatten_all()?.to_vec1::<f32>()?;
                let (mut b1, mut i1, mut b2, mut i2) = (f32::MIN, 0usize, f32::MIN, 0usize);
                for (i, &x) in v.iter().enumerate() {
                    if x > b1 {
                        b2 = b1;
                        i2 = i1;
                        b1 = x;
                        i1 = i;
                    } else if x > b2 {
                        b2 = x;
                        i2 = i;
                    }
                }
                println!(
                    "MARGIN run={run} step={step_no} argmax={i1} second={i2} margin={:.4} top={b1:.4}",
                    b1 - b2
                );
            }
            if eos_ids.contains(&next) {
                hit_eos = true;
                break;
            }
            generated.push(next);
            let input = Tensor::from_vec(vec![next], (1, 1), device)?;
            logits = model.forward(&input, cache.pos, &mut cache)?;
            digest_logits(&logits, &mut lh)?;
            step_no += 1;
            if args.state_digest {
                println!(
                    "STEP run={run} step={step_no} logits={:016x}",
                    fnv(&logits)?
                );
            }
        }
        if args.state_digest {
            for (name, sum, nonfinite) in cache.checksums()? {
                println!("STATE run={run} phase=end {name}={sum:016x} nonfinite={nonfinite}");
            }
        }
        let decode_ms = t1.elapsed().as_millis() - prefill_ms;
        profile_line(run, "decode", generated.len(), decode_ms as f64);
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
        let s = model.stream_stats();
        if args.experts == "stream" {
            let st = model.staging_stats();
            println!(
                "STAGING run={run} buffers={} bytes={} allocations={} reuses={} overflows={}",
                st.buffers, st.bytes, st.allocations, st.reuses, st.overflows
            );
            println!(
                "STREAM run={run} loader={} slots={} plans={} groups={} hits={} misses={} evictions={} bytes_loaded={} hit_rate={:.4} device_plans={} settled_misses={} redos={} host_plan_passes={}",
                args.loader,
                model.slots_per_layer(),
                s.plans,
                s.groups,
                s.hits,
                s.misses,
                s.evictions,
                s.bytes_loaded,
                s.hits as f64 / (s.hits + s.misses).max(1) as f64,
                s.device_plans,
                s.settled_misses,
                cache.redos,
                cache.host_plan_passes
            );
        }
        if let Some((hits, misses)) = model.ple_cache_stats() {
            println!(
                "PLECACHE run={run} rows={} hits={hits} misses={misses} hit_rate={:.4}",
                args.ple_cache_rows,
                hits as f64 / (hits + misses).max(1) as f64
            );
        }
        digests.push(digest);
    }
    let identical = digests.windows(2).all(|w| w[0] == w[1]);
    println!(
        "DETERMINISM runs={} identical={identical} digest={}",
        args.runs,
        digests.first().cloned().unwrap_or_default()
    );
    let mut ok = identical;
    if let Some(head) = model.mtp_head() {
        // One speculative decode of the same prompt: the MTP drafter on the
        // seam, verified by the target; must equal the greedy stream.
        let mut cache: Cache = model.new_cache(&cfg, device)?;
        cache.want_nextn = true;
        let mut session = Session::new(model, cache);
        let mut drafter =
            MtpDrafter::new(head, model.embed_tokens(), model.lm_head(), &cfg, device);
        let mut rb = Blocking::new(eos_ids, device)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let opts = DecodeOptions::new(args.max_tokens);
        let t2 = std::time::Instant::now();
        let out = session.decode_speculative(
            &mut drafter,
            &mut rb,
            &mut sampler,
            prompt_ids,
            &opts,
            eos_ids,
            &mut |_| Ok(()),
        )?;
        let mtp_ms = t2.elapsed().as_millis();
        let mut h = sha256::Sha256::new();
        for t in &out.ids {
            h.update(&t.to_le_bytes());
        }
        let mtp_digest = sha256::hex(&h.finalize());
        let rounds = out.stats.rounds.len();
        let accepted_drafts = out.stats.rounds.iter().filter(|r| r.accepted > 1).count();
        let tokens: usize = out.stats.rounds.iter().map(|r| r.accepted).sum();
        let same = digests.first().map(|d| d == &mtp_digest).unwrap_or(false);
        println!(
            "output(mtp): {}",
            tokenizer
                .decode(&out.ids, true)
                .map_err(anyhow::Error::msg)?
        );
        println!(
            "MTP tokens_sha256={mtp_digest} identical_to_greedy={same} generated={} rounds={rounds} accepted_drafts={accepted_drafts} draft_accept_rate={:.4} tokens_per_round={:.3} drafter_rows={} wall_ms={mtp_ms}",
            out.ids.len(),
            accepted_drafts as f64 / rounds.max(1) as f64,
            tokens as f64 / rounds.max(1) as f64,
            drafter.rows_seen()
        );
        ok &= same;
    }
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

/// The rollback probe (see `--rollback-probe`).
fn rollback_probe<R: ExpertResidency>(
    model: &Qwen4Exp<R>,
    cfg: &lloom::models::qwen4exp::Config,
    device: &Device,
    prompt_ids: &[u32],
    eos_ids: &[u32],
    rounds: usize,
) -> Result<()> {
    let wrong = cfg.vocab_size as u32 - 2;
    let mode = std::env::var("QWEN4EXP_ROLLBACK_MODE").unwrap_or_else(|_| "rounds".into());
    // subject: verify rounds with a wrong draft, each a partial accept
    let mut sub: Cache = model.new_cache(cfg, device)?;
    sub.want_nextn = true;
    let input = Tensor::from_vec(prompt_ids.to_vec(), (1, prompt_ids.len()), device)?;
    let logits = model.forward(&input, 0, &mut sub)?;
    let _ = model.take_probe(&mut sub)?;
    let mut anchor = argmax(&logits)?;
    let mut consumed: Vec<u32> = Vec::new();
    if mode == "roundtrip" {
        // open + rewind with no forward in between: the snapshot/restore alone
        let before = sub.checksums()?;
        for _ in 0..rounds {
            sub.open_window()?;
            sub.rewind_window()?;
        }
        let after = sub.checksums()?;
        let diff = before
            .iter()
            .zip(&after)
            .filter(|(a, b)| a.1 != b.1)
            .count();
        println!(
            "ROLLBACK roundtrip x{rounds}: channels={} differing={diff}",
            before.len()
        );
        // and open + one forward + rewind: the restore after a write
        let before = sub.checksums()?;
        sub.open_window()?;
        let inp = Tensor::from_vec(vec![anchor], (1, 1), device)?;
        model.forward(&inp, sub.pos, &mut sub)?;
        sub.rewind_window()?;
        let after = sub.checksums()?;
        let diff = before
            .iter()
            .zip(&after)
            .filter(|(a, b)| a.1 != b.1)
            .count();
        println!(
            "ROLLBACK open+forward+rewind: channels={} differing={diff} pos={}",
            before.len(),
            sub.pos
        );
        return Ok(());
    }
    for r in 0..rounds {
        let (acc, _probe) = model.verify_and_probe(
            &[Draft::host(vec![wrong])],
            &[Anchor {
                last: anchor,
                pos: sub.pos,
            }],
            &mut sub,
            eos_ids,
        )?;
        let acc = &acc[0];
        println!(
            "ROLLBACK round={r} anchor={anchor} accepted={acc:?} pos_after={}",
            sub.pos
        );
        consumed.push(anchor);
        anchor = acc[0];
        if acc.len() != 1 {
            anyhow::bail!("the wrong draft was accepted");
        }
    }
    // reference: the same tokens decoded greedily, one per forward
    let mut refc: Cache = model.new_cache(cfg, device)?;
    model.forward(&input, 0, &mut refc)?;
    let mut ref_next = Vec::new();
    for &t in &consumed {
        let inp = Tensor::from_vec(vec![t], (1, 1), device)?;
        let lg = model.forward(&inp, refc.pos, &mut refc)?;
        ref_next.push(argmax(&lg)?);
    }
    println!("ROLLBACK consumed={consumed:?} reference_next={ref_next:?} subject_next={anchor} pos sub={} ref={}", sub.pos, refc.pos);
    let a = sub.checksums()?;
    let b = refc.checksums()?;
    let mut diff = 0;
    for ((n1, h1, _), (n2, h2, _)) in a.iter().zip(&b) {
        if n1 != n2 || h1 != h2 {
            diff += 1;
            if diff <= 12 {
                println!("ROLLBACK DIFF {n1}: subject={h1:016x} reference={h2:016x}");
            }
        }
    }
    println!("ROLLBACK channels={} differing={diff}", a.len());
    // magnitudes on a few channels
    for l in [0usize, 1, 2] {
        if let (Some(x), Some(y)) = (sub.s_state.get(l), refc.s_state.get(l)) {
            let (m, n) = maxdiff(x, y)?;
            println!("ROLLBACK S[{l}] max|Δ|={m:.3e} max|ref|={n:.3e}");
        }
        if let (Some(x), Some(y)) = (sub.conv_state.get(l), refc.conv_state.get(l)) {
            let (m, n) = maxdiff(x, y)?;
            println!("ROLLBACK conv[{l}] max|Δ|={m:.3e} max|ref|={n:.3e}");
        }
    }
    Ok(())
}

fn maxdiff(a: &Tensor, b: &Tensor) -> Result<(f32, f32)> {
    let a = a
        .flatten_all()?
        .to_dtype(candle::DType::F32)?
        .to_vec1::<f32>()?;
    let b = b
        .flatten_all()?
        .to_dtype(candle::DType::F32)?
        .to_vec1::<f32>()?;
    let mut m = 0f32;
    let mut n = 0f32;
    for (x, y) in a.iter().zip(&b) {
        m = m.max((x - y).abs());
        n = n.max(y.abs());
    }
    Ok((m, n))
}

/// FNV over the f32 bit patterns of a logits row (the `STEP` line).
fn fnv(logits: &Tensor) -> Result<u64> {
    let v = logits.flatten_all()?.to_vec1::<f32>()?;
    let mut h = 0xcbf29ce484222325u64;
    for x in &v {
        h ^= x.to_bits() as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    Ok(h)
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
