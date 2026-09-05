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
use lloom::models::lfm2::{Cache, ConvState, KvAppend, Lfm2Config, Model};
use lloom::models::lfm2_dspark::{DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, Session, VarBuilder};
use std::io::Read;
use std::path::{Path, PathBuf};
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

    /// The most command buffers the host may leave in flight before it waits for
    /// the GPU — explicit backpressure. `0` (the default) is off. Only meaningful
    /// on the `ring-hostdrain` arm, which has no per-token EOS wait to pace it and
    /// otherwise runs the host arbitrarily far ahead of the GPU (measured: 63 vs
    /// the poll arm's ~9). A value near the commit budget keeps in-flight bounded
    /// like the poll arm while letting the GPU run ahead up to the cap.
    #[arg(long, default_value_t = 0)]
    max_in_flight: usize,

    /// Drive decode with the **DSpark** speculative drafter instead of the K=1
    /// identity proposer. The drafter proposes a block of up to `--draft` tokens
    /// per round and the target verifies them in one `forward_all`, accepting the
    /// longest argmax-matching prefix plus the target's own bonus token — so the
    /// output is **byte-identical** to the non-speculative path (the target
    /// corrects every mismatch; `design/docs/speculation.md` §10.2n). Off by
    /// default (the current behavior is unchanged). See `--draft-dir` for where
    /// the drafter weights are found.
    ///
    /// At B=1 on an M1 Max this is measured FASTER than the crown non-spec path
    /// (`design/docs/speculation.md` §10.2u: ~80 against ~62 tok/s steady at
    /// 3.5 accepted tokens per round) — once the verify pass took the 16-row
    /// GEMM tile and the ring conv fused; before those it was 0.72×. The round
    /// is still verify-dominated, so the figure moves with the prompt's
    /// acceptance rate; `--phase-sync` shows where a round's time goes.
    #[arg(long)]
    dspark: bool,

    /// The draft block size K when `--dspark` is set — how many tokens the drafter
    /// proposes per round. The DSpark drafter's block is FIXED at
    /// `block_size − logits_start` (9 for the 2.6B DSpark checkpoint) and has no
    /// runtime cap, so the only accepted value is that ceiling; omit this to use
    /// it. Passing a different K is refused with the reason rather than silently
    /// clamped (a smaller ring would overflow the drafter's fixed block). The flag
    /// is kept so the block size is visible and so a future drafter with a runtime
    /// cap can honor it.
    #[arg(long)]
    draft: Option<usize>,

    /// The DSpark drafter checkpoint directory (config.json + model.safetensors).
    /// Only used with `--dspark`. Defaults to the newest LFM2.5-2.6B-DSpark
    /// snapshot in the HF cache.
    #[arg(long)]
    draft_dir: Option<PathBuf>,

    /// Which cache configuration the NON-speculative path runs on: `crown`
    /// (`Sdpa × Cat × Shuffle`, the fastest B=1 config) or `rewindable` — the
    /// one `--dspark` is forced onto (`Sdpa × InPlace × RotatingRing{K+1}` with
    /// the F32 taps on), run here WITHOUT a drafter, K=1. The pair isolates the
    /// cost of leaving the crown from the cost of the verify loop (#675).
    /// `rewindable-notap` is the same cache with the taps off, so the taps'
    /// own cost separates from the ring's. `--dspark` ignores this and always
    /// runs `rewindable`.
    #[arg(long, default_value = "crown")]
    cache: String,

    /// **Diagnostic, perturbs what it measures**: with `--dspark`, synchronize
    /// the GPU at the end of each phase of a round (draft / verify / observe) so
    /// each phase's wall carries its own GPU tail, and — with
    /// `CANDLE_METAL_PROFILE=1` — take each phase's GPU-busy, dispatch count and
    /// one round's per-kernel census. Off for any throughput figure.
    #[arg(long)]
    phase_sync: bool,
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

/// The DSpark drafter setup resolved before the target config: where its weights
/// live, its constants, and the resolved draft block size `K`.
struct DsparkSetup {
    draft_dir: PathBuf,
    dcfg: DSparkConfig,
    /// The draft block size, capped at the drafter's `max_draft()`.
    k: usize,
}

/// The newest snapshot of a HF repo in the cache with a `config.json`, if any.
/// The same discovery the `lfm2-dspark-verify` example uses for the drafter.
fn newest_snapshot(repo: &str) -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub")
        .join(repo)
        .join("snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.join("config.json").exists())
        .collect();
    entries.sort();
    entries.pop()
}

/// Read the DSpark drafter's constants from its own `config.json`.
///
/// Never defaulted for the two fields a wrong default silently corrupts —
/// `rope_theta` and `rope_is_neox_style` alter no shape and no dispatch count,
/// and on a drafter there is not even a text gate to catch them because the
/// target verifies every token (`design/docs/speculation.md` §10.2n). Adapted
/// from the `lfm2-dspark-verify` example's `read_drafter_config`.
fn read_drafter_config(dir: &Path) -> Result<DSparkConfig> {
    let raw = std::fs::read_to_string(dir.join("config.json"))
        .with_context(|| format!("reading {}/config.json", dir.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let dfc = v.get("dflash_config").cloned().unwrap_or_else(|| v.clone());

    let usize_at = |k: &str| -> Result<usize> {
        v.get(k)
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .with_context(|| format!("drafter config has no {k}"))
    };

    let target_layer_ids: Vec<usize> = dfc
        .get("target_layer_ids")
        .and_then(|x| x.as_array())
        .context("drafter config has no dflash_config.target_layer_ids")?
        .iter()
        .filter_map(|x| x.as_u64().map(|n| n as usize))
        .collect();

    Ok(DSparkConfig {
        hidden_size: usize_at("hidden_size")?,
        num_hidden_layers: usize_at("num_hidden_layers")?,
        num_attention_heads: usize_at("num_attention_heads")?,
        num_key_value_heads: usize_at("num_key_value_heads")?,
        head_dim: usize_at("head_dim")?,
        intermediate_size: usize_at("intermediate_size")?,
        vocab_size: usize_at("vocab_size")?,
        rms_norm_eps: v
            .get("rms_norm_eps")
            .and_then(|x| x.as_f64())
            .unwrap_or(1e-5),
        rope_theta: v
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .context("drafter config has no rope_theta")?,
        rope_is_neox_style: v
            .get("rope_is_neox_style")
            .and_then(|x| x.as_bool())
            .context("drafter config has no rope_is_neox_style")?,
        block_size: usize_at("block_size")?,
        mask_token_id: dfc
            .get("mask_token_id")
            .and_then(|x| x.as_u64())
            .context("drafter config has no dflash_config.mask_token_id")? as u32,
        num_target_layers: dfc
            .get("num_target_layers")
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .unwrap_or(30),
        target_layer_ids,
        markov_rank: v
            .get("markov_rank")
            .and_then(|x| x.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0),
        enable_confidence_head: v
            .get("enable_confidence_head")
            .and_then(|x| x.as_bool())
            .unwrap_or(false),
        // NOT a key in config.json — a property of the packaging (§10.2p).
        logits_start: 0,
    })
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

    // The DSpark drafter's constants, resolved BEFORE the target config because
    // the verify window's width (`K + 1`) sizes the conv ring, and taps must be
    // requested on the target's own layers. Only when `--dspark` is set; the
    // default path never reads a drafter.
    let dspark_setup: Option<DsparkSetup> = if args.dspark {
        let draft_dir = args
            .draft_dir
            .clone()
            .or_else(|| newest_snapshot("models--LiquidAI--LFM2.5-2.6B-DSpark"))
            .context(
                "no LFM2.5-2.6B-DSpark snapshot in the HF cache and no --draft-dir. The drafter is \
                 trained against THIS target (design/docs/speculation.md §10.2l).",
            )?;
        let dcfg = read_drafter_config(&draft_dir)?;
        if dcfg.hidden_size != config.hidden_size {
            anyhow::bail!(
                "drafter hidden {} != target hidden {} — the drafter is not this target's \
                 (design/docs/speculation.md §10.2l)",
                dcfg.hidden_size,
                config.hidden_size
            );
        }
        // The DSpark drafter proposes a FIXED block every round —
        // `block_size − logits_start` tokens (`DSparkSpeculator::draft`, which reads
        // `max_draft()` and does not take a runtime cap). So the verify window is
        // always `max_draft() + 1` positions and the conv ring must reserve that
        // many; a smaller `--draft` cannot make the drafter propose fewer, and
        // sizing the ring below what the drafter emits is the `advance()` overflow
        // this refuses rather than misreports (§7.1a). `--draft` therefore selects
        // the block only when it EQUALS the drafter's own ceiling; anything else is
        // an error with the reason, not a silent clamp.
        let ceiling = dcfg.max_draft();
        let k = match args.draft {
            None => ceiling,
            Some(requested) if requested == ceiling => ceiling,
            Some(requested) => anyhow::bail!(
                "--draft {requested}: the DSpark drafter proposes a fixed block of {ceiling} \
                 tokens (block_size {} − logits_start {}) and has no runtime cap, so only \
                 --draft {ceiling} is supported on this checkpoint. Omit --draft to use the \
                 ceiling.",
                dcfg.block_size,
                dcfg.logits_start,
            ),
        };
        Some(DsparkSetup { draft_dir, dcfg, k })
    } else {
        None
    };

    // The cache arm that runs. `--dspark` needs the REWINDABLE cache, which the
    // crown arms cannot provide: `Cache::advance` requires
    // `ConvState::RotatingRing{ >= K+1 }` (the shuffle is destructive) and
    // `Cache::resolve` requires `KvAppend::InPlace` (the `Cat` arm has no length
    // to rewind) — both refused otherwise (`lfm2.rs`). The verify window is
    // `K + 1` positions (the K proposals plus the target's bonus), so the ring
    // reserves that many history slots. Taps are ON — the drafter's projection
    // consumes the target's layer-output residuals — at F32 (`§10.2m`:
    // deliberate, buys acceptance rate). `Sdpa` attention still applies
    // (token-identical); the GEMV fusions stay on where they compose, and the
    // whole-block conv fusion does not (it is the `Shuffle` arm's, §12.2 row 1).
    // `--cache rewindable` runs the SAME cache without a drafter, so the cost of
    // the cache is measured apart from the cost of the verify loop.
    let cache_arm: &str = if dspark_setup.is_some() {
        "rewindable"
    } else {
        args.cache.as_str()
    };
    if cache_arm == "rewindable" || cache_arm == "rewindable-notap" {
        let k = match &dspark_setup {
            Some(setup) => setup.k,
            None => args.draft.unwrap_or(9),
        };
        config.attn_impl = lloom::models::lfm2::AttnImpl::Sdpa;
        config.kv_append = KvAppend::InPlace;
        config.conv_state = ConvState::RotatingRing { k: k + 1 };
        config.gemv_silu = true;
        config.gemv_residual = true;
        config.conv_fused = true;
        config.tap_dtype = DType::F32;
        config.tap_layers = match (&dspark_setup, cache_arm) {
            (Some(setup), _) => setup.dcfg.tap_layers(),
            (None, "rewindable") => {
                // The 2.6B drafter's taps, `target_layer_ids + 1`, without
                // loading the drafter: the same five layers the `--dspark` run
                // taps, so the arm carries the same per-token capture cost.
                DSparkConfig::lfm2_2_6b().tap_layers()
            }
            _ => Vec::new(),
        };
    } else if cache_arm == "crown" {
        // The fastest B=1 decode config lloom has, baked in — this example exists to
        // show what the engine does at its best, not the conservative defaults every
        // axis ships at (§7.1a keeps those off in the library; a demo turns them on).
        // The crown attention/conv arms (Sdpa × Shuffle, §13.5) plus the two GEMV
        // epilogue fusions (§12.2d/§12.2h) plus the conv fold (§12.2 row 1). Every one
        // is token-identical to the generic path — only the logits reorder — so this
        // is the same answer, faster. The readback strategy (ring + lookahead) is
        // selected below, on the loop side.
        config.attn_impl = lloom::models::lfm2::AttnImpl::Sdpa;
        // KV append: `Cat` is the demo default because it is the fastest arm
        // *today* (measured 2026-09-05: Cat ~60.5 vs InPlace ~58.3 tok/s steady,
        // both byte-identical). InPlace writes in place at a moving offset (stable
        // buffer identity, one `copy2d`/token) and stabilises the attention
        // positions for replay (memory.md §9.2i: 77.9% → 81.7% coverage) — BUT that
        // coverage is latent: no ICB executor is installed on this decode path
        // (`ExecutorSlot::Classical`), so InPlace pays its per-token cost with the
        // replay benefit switched off and loses. InPlace's win is a TWO-part change
        // (InPlace + engage the ICB replay executor); until the second half lands,
        // Cat is faster for this single-turn speed demo. A real serving engine wants
        // InPlace regardless — `Cat` reallocates a wider KV every token and grows
        // unbounded, which does not fly for many concurrent long generations — but
        // that is the serving path's decision, not this demo's. `LLOOM_CROWN_KV=in-place`
        // selects it for the A/B and for the replay work.
        // Flipped: InPlace is now the crown default (was Cat). `LLOOM_CROWN_KV=cat`
        // restores the old arm for the A/B. Being measured: InPlace's per-token
        // cost (one `copy2d`/token/layer for K and V) vs Cat's O(N) realloc+copy
        // — hunting the drain that made InPlace lose (58.3 vs 60.5) despite the
        // better algorithm and the latent replay coverage (§9.2i).
        config.kv_append = match std::env::var("LLOOM_CROWN_KV").as_deref() {
            Ok("cat") => lloom::models::lfm2::KvAppend::Cat,
            _ => lloom::models::lfm2::KvAppend::InPlace,
        };
        config.conv_state = lloom::models::lfm2::ConvState::Shuffle;
        config.gemv_silu = true;
        config.gemv_residual = true;
        config.conv_fused = true;
    } else {
        anyhow::bail!(
            "--cache must be `crown`, `rewindable` or `rewindable-notap`, got `{cache_arm}`"
        );
    }

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

    // Warm the Metal pipelines before the clock the user reads starts. A real
    // forward compiles the prefill GEMM-tile / F32-attention / batched-conv
    // pipelines that the decode step never touches (it routes to GEMV first), so
    // without this they compile *inside* the first prompt's TTFT — ~90 ms of
    // one-time cost `llama.cpp` pays up front and we otherwise pay on the first
    // token. `warmup` runs one prefill + one decode step and resets, so the
    // session is untouched; the cost folds into the load figure below.
    let mut session = Session::new(&model, cache);
    // `LLOOM_SKIP_WARMUP=1` measures the counterfactual: TTFT with the pipelines
    // left cold, so the warmup's effect on the first prompt is attributable
    // rather than asserted. Default is to warm.
    let (libs, pipes) = if std::env::var("LLOOM_SKIP_WARMUP").is_ok() {
        (0, 0)
    } else {
        session.warmup(&device).context("warming Metal pipelines")?
    };

    // Load time, printed before we block on stdin so it is attributable. This is
    // mmap + model construction + cache allocation + pipeline warmup — the
    // safetensors are memory mapped, so the weight *bytes* fault in lazily on the
    // first forward, but the pipelines are now compiled, so the first prompt's
    // TTFT is real prefill work rather than compilation.
    eprintln!(
        "[loaded in \x1b[1;96m{:.2}s\x1b[0m (mmap + construct + warmup: {libs} libs, {pipes} pipelines)]",
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
    if args.phase_sync && !args.dspark {
        anyhow::bail!("--phase-sync brackets the speculative round's phases; it needs --dspark");
    }
    let opts = DecodeOptions {
        max_tokens: args.max_tokens,
        budget,
        lookahead: args.lookahead,
        max_in_flight: args.max_in_flight,
        // Under the speculative body this is the per-phase drain (decode.rs).
        sample_split: args.phase_sync,
    };

    // The DSpark proposer, built only when `--dspark` is set. It borrows the
    // target's token embedding and `lm_head` (the drafter ships neither — 62
    // tensors, `lfm2_dspark.rs`) as cheap `Arc`-backed clones. The RoPE table is
    // sized for the whole generation, context positions included.
    let mut dspark_spec: Option<DSparkSpeculator> = match &dspark_setup {
        None => None,
        Some(setup) => {
            let draft_file = setup.draft_dir.join("model.safetensors");
            if !draft_file.exists() {
                anyhow::bail!("{} does not exist", draft_file.display());
            }
            // Cover the largest absolute position the draft will attend: the prompt,
            // every accepted token (≤ max_tokens), plus the block width and slack.
            let max_pos = prompt_ids.len() + args.max_tokens + setup.dcfg.block_size + 64;
            // F16 to match the target the block is embedded with — the drafter's
            // stored BF16 has no CPU matmul and would need a cast at every meeting
            // of the two dtypes (`lfm2_dspark.rs` `DSparkDrafter::new` docs).
            let dvb = unsafe {
                VarBuilder::from_mmaped_safetensors(&[draft_file], DType::F16, &device)?
            };
            let drafter = DSparkDrafter::new(setup.dcfg.clone(), max_pos, dvb)
                .context("constructing DSpark drafter")?;
            eprintln!(
                "[dspark: drafter loaded, {} layers, draft block K={} (verify window {} positions, \
                 conv ring reserves {} history slots)]",
                setup.dcfg.num_hidden_layers,
                setup.k,
                setup.k + 1,
                setup.k + 1
            );
            Some(DSparkSpeculator::new(
                drafter,
                model.draft_embedding().clone(),
                model.lm_head_linear().clone(),
                1,
            ))
        }
    };

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
        match dspark_spec.as_mut() {
            Some(dspark) => {
                // The speculative body. It does NOT ride the device-resident ring —
                // the accepted tokens are host-side after each verify pass's argmax
                // compare (`design/docs/decode-driver-design.md`: the speculative
                // path's one readback per round is inherent) — so it streams the
                // accepted block straight to `on_tokens` and learns EOS host-side.
                // A `Blocking` strategy supplies the device handle and commit
                // cadence; its own feed/EOS paths are unused here.
                let mut rb = Blocking::new(&eos_ids, &device)?;
                session.decode_speculative(
                    dspark,
                    &mut rb,
                    &mut logits_processor,
                    &prompt_ids,
                    &opts,
                    &eos_ids,
                    &mut on_tokens,
                )?
            }
            None => decode_with(
                &args.readback,
                &mut session,
                &mut spec,
                &mut logits_processor,
                &prompt_ids,
                &opts,
                &eos_ids,
                &device,
                &mut on_tokens,
            )?,
        }
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

    // Under `--dspark` a "step" is a speculative ROUND, not a token: a round
    // publishes 1..K+1 tokens (the accepted prefix plus the target's bonus). So
    // `steady` above is rounds/s, and the token throughput is that times the mean
    // tokens per round. Both are reported honestly rather than relabeling one as
    // the other. `n` counts the streamed anchor plus every round's accepted
    // tokens; `st.steps` counts the rounds after the anchor.
    if args.dspark {
        let rounds = st.steps.max(1);
        // Tokens the ROUNDS produced (the whole stream minus the prefill anchor
        // that is emitted before round 0).
        let round_tokens = n.saturating_sub(if n > 0 { 1 } else { 0 });
        let tokens_per_round = round_tokens as f64 / rounds as f64;
        let steady_tok_s = steady * tokens_per_round; // rounds/s × tokens/round
        let setup = dspark_setup.as_ref().expect("--dspark implies a setup");
        eprintln!(
            "[dspark: {} accepted-tokens/round (mean over {} rounds), \
             {} tok/s steady (rounds/s × tokens/round), K={}]",
            b(format!("{tokens_per_round:.2}")),
            rounds,
            b(format!("{steady_tok_s:.1}")),
            setup.k,
        );
        report_rounds(st, args.phase_sync);
    }

    eprintln!(
        "[{n} tokens: {} tok/s end-to-end (prefill included), \
         {} {} , TTFT {} ms, \
         {} on {:?}; cache={cache_arm} \
         readback={} lookahead={} commit_budget={} commits={} max_in_flight={} \
         ring_publishes={} poll_fallbacks={} overrun={} eos={}]",
        b(format!("{end_to_end:.1}")),
        b(format!("{steady:.1}")),
        if args.dspark {
            "rounds/s steady (median round); see the dspark line above for tok/s"
        } else {
            "tok/s steady (median decode step)"
        },
        b(format!("{ttft_ms:.1}")),
        if dtype == DType::F16 { "f16" } else { "f32" },
        device,
        if args.dspark { "dspark" } else { &args.readback },
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

/// The per-round breakdown of a `--dspark` run: where a round's wall goes, per
/// round and per emitted token, from `DecodeStats::rounds`.
///
/// The first `WARMUP_ROUNDS` are dropped from every figure — the first pass
/// through the drafter's kernels pays pipeline construction and a cold pool —
/// and the dropped count is printed. Wall figures are medians per round and
/// ratios of sums per token (§10.2j's denominator rule); GPU-busy and dispatch
/// figures appear only under `--phase-sync` with `CANDLE_METAL_PROFILE=1`, and
/// the census is one round's, the one the loop kept.
fn report_rounds(st: &lloom::nn::DecodeStats, phase_sync: bool) {
    const WARMUP_ROUNDS: usize = 4;
    let rounds = &st.rounds;
    if rounds.len() <= WARMUP_ROUNDS {
        eprintln!(
            "[dspark rounds: only {} rounds, fewer than the {WARMUP_ROUNDS} dropped as warm-up; \
             no breakdown]",
            rounds.len()
        );
        return;
    }
    let timed = &rounds[WARMUP_ROUNDS..];
    let n = timed.len() as f64;
    let ms = |ns: u64| ns as f64 / 1e6;
    let median = |mut v: Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            0.5 * (v[n / 2 - 1] + v[n / 2])
        }
    };
    let accepted: usize = timed.iter().map(|r| r.accepted).sum();
    let tokens_per_round = accepted as f64 / n;
    let sum_draft: f64 = timed.iter().map(|r| ms(r.timing.draft_ns)).sum();
    let sum_verify: f64 = timed.iter().map(|r| ms(r.timing.verify_ns)).sum();
    let sum_observe: f64 = timed.iter().map(|r| ms(r.timing.observe_ns)).sum();
    let sum_round = sum_draft + sum_verify + sum_observe;
    let med_round = median(timed.iter().map(|r| ms(r.timing.wall_ns())).collect());
    let med_draft = median(timed.iter().map(|r| ms(r.timing.draft_ns)).collect());
    let med_verify = median(timed.iter().map(|r| ms(r.timing.verify_ns)).collect());
    let med_observe = median(timed.iter().map(|r| ms(r.timing.observe_ns)).collect());
    let e = accepted as f64;
    eprintln!(
        "[dspark rounds: {} timed ({WARMUP_ROUNDS} warm-up dropped), {accepted} tokens accepted, \
         {tokens_per_round:.3} tokens/round; phase_sync={phase_sync}]",
        timed.len()
    );
    eprintln!("[dspark per round, ms (median | mean):  draft {med_draft:.3} | {:.3}   verify {med_verify:.3} | {:.3}   observe {med_observe:.3} | {:.3}   round {med_round:.3} | {:.3}]",
        sum_draft / n, sum_verify / n, sum_observe / n, sum_round / n);
    eprintln!("[dspark per accepted token, ms (ratio of sums):  draft {:.3}   verify {:.3}   observe {:.3}   total {:.3}]",
        sum_draft / e, sum_verify / e, sum_observe / e, sum_round / e);
    // The accepted-length distribution: the round cost is paid whatever the
    // acceptance, so this is the other factor in the per-token figure.
    let mut hist = vec![0usize; timed.iter().map(|r| r.accepted).max().unwrap_or(0) + 1];
    for r in timed {
        hist[r.accepted] += 1;
    }
    eprintln!(
        "[dspark accepted-per-round histogram (1..): {}]",
        hist.iter()
            .enumerate()
            .skip(1)
            .map(|(k, c)| format!("{k}:{c}"))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let gpu_draft: f64 = timed.iter().map(|r| r.timing.draft_gpu_s * 1e3).sum();
    let gpu_verify: f64 = timed.iter().map(|r| r.timing.verify_gpu_s * 1e3).sum();
    let gpu_observe: f64 = timed.iter().map(|r| r.timing.observe_gpu_s * 1e3).sum();
    let disp_draft: u64 = timed.iter().map(|r| r.timing.draft_disp).sum();
    let disp_verify: u64 = timed.iter().map(|r| r.timing.verify_disp).sum();
    let disp_observe: u64 = timed.iter().map(|r| r.timing.observe_disp).sum();
    let profiled = disp_draft + disp_verify + disp_observe > 0;
    if profiled {
        eprintln!("[dspark GPU-busy per round, ms (mean of union):  draft {:.3}   verify {:.3}   observe {:.3}   round {:.3}  — idle (wall−busy) {:.3}]",
            gpu_draft / n, gpu_verify / n, gpu_observe / n, (gpu_draft + gpu_verify + gpu_observe) / n,
            (sum_round - gpu_draft - gpu_verify - gpu_observe) / n);
        eprintln!(
            "[dspark dispatches per round (mean):  draft {:.1}   verify {:.1}   observe {:.1}]",
            disp_draft as f64 / n,
            disp_verify as f64 / n,
            disp_observe as f64 / n
        );
        for (name, census) in [
            (
                "draft",
                timed.iter().find_map(|r| r.timing.draft_census.as_ref()),
            ),
            (
                "verify",
                timed.iter().find_map(|r| r.timing.verify_census.as_ref()),
            ),
            (
                "observe",
                timed.iter().find_map(|r| r.timing.observe_census.as_ref()),
            ),
        ] {
            if let Some(c) = census {
                let total: u64 = c.iter().map(|(_, n)| n).sum();
                eprintln!(
                    "[dspark census {name} (one round, {total} dispatches, {} labels):",
                    c.len()
                );
                for (label, count) in c {
                    eprintln!("    {count:>6}  {label}");
                }
                eprintln!("]");
            }
        }
    }
    eprintln!(
        "DSPARK_ROUNDS rounds={} tokens={accepted} tokens_per_round={tokens_per_round:.4} \
         round_ms_median={med_round:.4} round_ms_mean={:.4} \
         draft_ms_median={med_draft:.4} verify_ms_median={med_verify:.4} observe_ms_median={med_observe:.4} \
         draft_ms_per_token={:.4} verify_ms_per_token={:.4} observe_ms_per_token={:.4} total_ms_per_token={:.4} \
         draft_gpu_ms={:.4} verify_gpu_ms={:.4} observe_gpu_ms={:.4} \
         draft_disp={:.1} verify_disp={:.1} observe_disp={:.1} phase_sync={phase_sync} profiled={profiled}",
        timed.len(),
        sum_round / n,
        sum_draft / e,
        sum_verify / e,
        sum_observe / e,
        sum_round / e,
        gpu_draft / n,
        gpu_verify / n,
        gpu_observe / n,
        disp_draft as f64 / n,
        disp_verify as f64 / n,
        disp_observe as f64 / n,
    );
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
        "ring-hostdrain" => {
            // The device never computes the EOS flag (~5 dispatches/token gone);
            // the host learns EOS from the drained ids on the commit cadence, so
            // up to a commit-worth of forwards may run past the EOS and their
            // tokens are truncated from the output. Byte-identical to the poll
            // arm — the truncation lands at the same position the poll would
            // break at.
            let mut rb = Ring::new(eos_ids, device, capacity, EosRead::HostOnDrain)?;
            let out = session.decode(spec, &mut rb, sampler, prompt_ids, opts, on_tokens)?;
            if !rb.poll_is_mapped() {
                eprintln!(
                    "[ring-hostdrain: the ring's words were not CPU-mapped; the drain took the \
                     synchronising snapshot each commit]"
                );
            }
            debug_assert!(rb.arms().nonblocking_eos);
            out
        }
        other => anyhow::bail!(
            "--readback must be `blocking`, `ring`, `ring-nonblocking` or `ring-hostdrain`, \
             got `{other}`"
        ),
    };
    Ok(out)
}
