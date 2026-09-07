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
use lloom::models::lfm2::{Cache, ConvState, KvAppend, Lfm2Config, Model, DEFAULT_KV_CAPACITY};
use lloom::models::lfm2_dspark::{DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, Session, VarBuilder};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use tokenizers::Tokenizer;

mod concurrent;

#[derive(Parser)]
struct Args {
    /// Checkpoint directory (config.json, tokenizer.json, *.safetensors).
    /// Defaults to LFM2_MODEL_DIR, then the LFM2.5-2.6B snapshot in the HF cache.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// New-token ceiling. **`0` (the default) means UNBOUNDED** — generation runs
    /// until the model emits EOS (`<|im_end|>`), with no cap, streaming at any
    /// length (the token-ring history wraps, so a long reply no longer bails). A
    /// positive value is a hard cap that stops the run early even before EOS —
    /// use it to bound a measurement or a runaway. (`--ignore-eos` with the
    /// default is refused: nothing would ever stop it.)
    #[arg(long, short = 'n', default_value_t = 0)]
    max_tokens: usize,

    /// Sampling temperature. 0 (the default) is greedy / argmax.
    #[arg(long, short = 't', default_value_t = 0.0)]
    temperature: f64,

    /// Sampling seed (only matters when temperature > 0). If unset, a RANDOM
    /// seed is drawn each run, so repeated `-t > 0` runs explore different
    /// samples; pass an explicit value to reproduce a specific run.
    #[arg(long)]
    seed: Option<u64>,

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

    /// Never stop on EOS: the run produces exactly `-n` tokens. The EOS id set
    /// is replaced by an id no tokenizer emits, so every arm's own EOS
    /// mechanism (device flag, host compare, drain scan) still runs and simply
    /// never fires. For long-run gates; the text degenerates past the model's
    /// own stop and is not what a user would read.
    #[arg(long)]
    ignore_eos: bool,

    /// Install the ICB replay executor on the decode path (issue #146/#169). The
    /// stable, packed subset of a decode step is recorded into an
    /// `MTLIndirectCommandBuffer` and replayed each step instead of re-encoded on
    /// the CPU. Forces `--param-style packed` (an ICB command cannot carry an
    /// inline constant). The executor's own overhead was measured +2.28% in the
    /// trace harness (§11.3s), BUT that number was single-turn/single-regime and
    /// never run on this full non-blocking decode — this flag is how we A/B it
    /// here. Off by default. Pair with `--replay-barriers skip-replayed` for the
    /// separable barrier-suppression win (§11.3r, −0.83% GPU-side).
    #[arg(long)]
    replay: bool,

    /// Install the activation arena (`DESIGN.md` §9.2) without the replay
    /// executor: the arm that separates what the arena costs or buys on this
    /// loop from what replay adds on top of it. `--replay` implies it. Recorded
    /// from the first decode steps, as under `--replay`. Not under `--dspark`.
    #[arg(long)]
    arena: bool,

    /// With `--replay`: whether candle's own `auto_barrier` still fires at a
    /// position the ICB already replays. `always` (the shipped behavior — a
    /// redundant barrier at 393 of 505 positions, §11.3p) or `skip-replayed`
    /// (issue #144: suppress candle's barrier at a non-head replayed member,
    /// keeping the ICB's own ordering). `skip-replayed` is the measured −0.83%
    /// GPU-side win (§11.3r) and the reason to run replay at all at B=1.
    #[arg(long, default_value = "skip-replayed")]
    replay_barriers: String,

    /// With `--replay`: how far back an ICB command's barrier scan looks —
    /// `run-start` (#115's shipped behavior: barrier wherever a command conflicts
    /// with anything earlier in the run) or `since-barrier` (scan only to the
    /// previous barrier, matching candle's `auto_barrier`). §11.3l open question 3.
    #[arg(long, default_value = "run-start")]
    replay_barrier_scope: String,

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

    /// Drop the leading BOS (`<|startoftext|>`, 124894). It is prepended by
    /// DEFAULT: `generation_config.json` and llama.cpp both start LFM2 with it,
    /// and `tokenizers::encode(…, false)` does not, so omitting it ran the model
    /// off its own template. With BOS on (and the EOS set corrected to
    /// `<|im_end|>` only — the two are coupled), DSpark acceptance and throughput
    /// rise on every prompt and replies terminate cleanly: measured
    /// relativity 75.7 → 93.7, structured-extract 139 → 178 tok/s, short-factual
    /// completes on EOS at 90.8. This flag restores the old off-template
    /// tokenization for the A/B only; it lowers acceptance and, without the EOS
    /// fix, drives a non-terminating ramble. Turning BOS on moves the canonical
    /// digests — to the correct, template-conformant output.
    #[arg(long)]
    no_bos: bool,

    /// Override the GEMV+SiLU epilogue fusion (§12.2d) regardless of the cache
    /// arm's baked value. Unset = the arm's default (crown/rewindable bake it
    /// ON). `--gemv-silu false` forces it OFF for the #509 A/B on this harness;
    /// `--gemv-silu true` forces it on. Takes effect after the arm is selected.
    #[arg(long)]
    gemv_silu: Option<bool>,

    /// Override the GEMV residual-add epilogue fusion (§12.2h) regardless of the
    /// cache arm's baked value. Unset = the arm's default (crown/rewindable bake
    /// it ON). `--gemv-residual false` forces it OFF for the #509 A/B.
    #[arg(long)]
    gemv_residual: Option<bool>,

    // ---- multi-sequence modes (absorbed from the former lfm2-sequence) -------
    /// `single` (default — read one prompt from stdin), `reuse` (a two-turn
    /// conversation decoded with prefix reuse vs a fresh prefill), or
    /// `concurrent` (`--sequences` predefined prompts interleaved on one model,
    /// or pipelined under `--dspark`). The non-single modes print no token text —
    /// they repaint a live raw-stats block and a RESULT line.
    #[arg(long, default_value = "single")]
    mode: String,

    /// Sequences to run under `--mode concurrent`. The prompts are a varied
    /// predefined set of eight (varied so the run exercises prefill), cycled
    /// past eight.
    #[arg(long, default_value_t = 2)]
    sequences: usize,

    /// Under `--mode concurrent`: drive the sequences through the batched-forward
    /// driver (`Session::decode_batched`, #694) — one `m = B` forward per step
    /// for every live sequence, the weight sweep paid once per step — against
    /// each prompt run solo on the selected readback arm. The batched arm takes
    /// `--lookahead` (0: read the ids back before the next forward; 1: feed the
    /// device ids and read back while the next step runs).
    #[arg(long)]
    batched: bool,

    /// Under `--batched`: skip the neighbour-independence arm (the rows re-run
    /// as two separately batched halves, each row asserted equal to its
    /// full-batch stream). On by default; off for a timing-only rotation.
    #[arg(long)]
    no_independence_check: bool,

    /// Under `--batched`: skip the solo arm — a tile A/B at a large `B` wants
    /// only the batched step; the identity columns read `false`.
    #[arg(long)]
    skip_solo: bool,

    /// Tokens per visit under `--mode concurrent` (non-speculative interleave).
    #[arg(long, default_value_t = 16)]
    chunk: usize,

    /// Prompt tokens per prefill micro-batch — the `n_ubatch` of the streamed
    /// prefill. **`0` (default) is the incumbent one-shot** (the whole prompt in
    /// one forward). A positive value steps the state machine over the prompt in
    /// chunks of this width through the SAME in-flight pacer the decode loop uses,
    /// so the pipeline spans prefill→decode with no drain and the host encodes the
    /// next chunk while the GPU runs the current one. It is the TTFT-vs-throughput
    /// dial: larger keeps the matmul units fuller (throughput; it reaches one-shot
    /// as it reaches the prompt length), smaller overlaps more and frees the
    /// pipeline for the first generated token sooner. Streamed multi-chunk turns
    /// on the conv's left-context carry so it stays byte-identical to one-shot.
    #[arg(long, default_value_t = 0)]
    prefill_ubatch: usize,

    /// The rotating ring's history depth under `--cache rewindable` (non-dspark):
    /// the rewind horizon. `--dspark` sizes it from the drafter's block.
    #[arg(long, default_value_t = 16)]
    ring_k: usize,

    /// Carry the conv state across a turn-boundary prefill (`ConvPrefillCtx`,
    /// `DESIGN.md` §13.2f), for `--mode reuse`. Off by default (§7.1a).
    #[arg(long)]
    conv_prefill_ctx: bool,

    /// Under `--mode concurrent --dspark`: which arm runs first, `solo` or
    /// `pipelined`, so a rotation can alternate the order.
    #[arg(long, default_value = "solo")]
    first: String,

    /// Under `--mode concurrent --dspark`: write every round of both arms as CSV.
    #[arg(long)]
    timeline: Option<PathBuf>,
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
            .context("drafter config has no dflash_config.mask_token_id")?
            as u32,
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

    // #509 A/B override: the cache arms above bake the GEMV epilogues ON (crown
    // 506-507, rewindable 458-459). These flags force either fusion regardless,
    // so the crown-on vs crown-off delta is measurable on this harness (the
    // library `Config::default` still ships them OFF; this measures the win
    // before flipping that default). Unset leaves the arm's baked value.
    if let Some(v) = args.gemv_silu {
        config.gemv_silu = v;
    }
    if let Some(v) = args.gemv_residual {
        config.gemv_residual = v;
    }
    // `--mode reuse` carries the conv state across the turn-boundary prefill
    // (§13.2f). Set on the config the model is built from; a no-op for `single`.
    //
    // Streamed prefill (`--prefill-ubatch > 0`) REQUIRES it: each chunk after the
    // first is a `seq_len>1` pass onto a warm cache, and without the left-context
    // carry the conv zero-pads the chunk boundary — a different convolution than
    // one pass over the whole prompt, so the output diverges (measured: garbage
    // from char 1). With it on, a chunked prefill computes exactly what one-shot
    // does. A single chunk (or one-shot) is identical either way (cold cache, no
    // left context to carry), so forcing it on when streaming is always safe.
    config.conv_prefill_ctx = args.conv_prefill_ctx || args.prefill_ubatch > 0;

    let files = weight_files(&model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    // `Model::new` is the bulk of the load (~950 ms): it reads the 266 weight
    // tensors through the mmap. Measured config-independent — the crown arms
    // above cost it nothing — so it is the construction touching the mapped
    // bytes, not anything this example selects. A warm page cache hides it (the
    // "sub-second" cold-vs-warm difference); the number the demo shows is the
    // honest steady one.
    // `--replay` needs packed params BEFORE any recorded forward: an ICB command
    // binds one `device const Params*`, not the inline scalars the split arm
    // emits, so the step the executor records must already be packed. Set it here
    // (before warmup) so the recorded shape matches the replayed one.
    // The other two switches replay needs, both process-wide and both before the
    // first pipeline is built: `supportIndirectCommandBuffers` is a pipeline
    // property (§3.7d -- encoding one without it segfaults), and the constants
    // pool is what gives dispatch position N the same params buffer every step
    // (without it every packed position varies and coverage is zero, §11.3d).
    #[cfg(feature = "metal")]
    if args.replay {
        use candle::kernels::{
            set_constants_pool_enabled, set_default_param_style, set_pipelines_support_icb,
            ParamStyle,
        };
        set_default_param_style(ParamStyle::Packed);
        set_pipelines_support_icb(true)
            .map_err(|e| anyhow::anyhow!("enabling supportIndirectCommandBuffers: {e}"))?;
        set_constants_pool_enabled(true);
    }

    let model = Model::new(&config, vb).context("constructing LFM2")?;
    let cache = Cache::new(true, dtype, &config, &device).context("allocating KV cache")?;

    // EOS ids: the config's, plus the chat-end / end-of-text tokens.
    // Stop on `<|im_end|>` (124900) only — the model's declared EOS. `config.json`,
    // `generation_config.json` (`eos_token_id: [124900]`) and the GGUF llama.cpp
    // uses all agree on that single id. `<|endoftext|>` (124895) was ALSO added
    // here, but nothing declares it an EOS — it is the base model's document
    // separator, which the model can emit mid-reply; stopping on it cuts replies
    // early and, in the speculative loop, lets the drafter propose a spurious
    // boundary the target never treats as one. Removed to match the checkpoint's
    // (and llama.cpp's) actual stop set.
    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
        if !eos_ids.contains(&id) {
            eos_ids.push(id);
        }
    }
    if args.ignore_eos {
        // Contradiction: no EOS AND no token ceiling would never stop. Refuse
        // rather than spin forever (§2.4 — a bad setting bails, not hangs).
        if args.max_tokens == 0 {
            anyhow::bail!(
                "--ignore-eos with -n 0 (unbounded) would never terminate — pass a positive \
                 -n to bound the run when EOS is disabled"
            );
        }
        eos_ids = vec![u32::MAX];
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

    // The multi-sequence modes (absorbed from lfm2-sequence) branch off here,
    // after the shared load/warmup and before the single path's stdin read and
    // the ICB replay install (which is a single-turn arm). They reuse `model`,
    // `config`, `device` and `eos_ids` — for `--dspark` the config was already
    // forced rewindable+taps above, so the built model is correct for the mode.
    if args.mode != "single" {
        if args.batched && args.dspark {
            anyhow::bail!(
                "--batched drives the K=1 batched-forward body; a batched verify is not built \
                 (design/docs/speculative-pipeline-scheduler-design.md §4b). Pick one of \
                 --batched / --dspark"
            );
        }
        if args.replay || args.arena {
            anyhow::bail!(
                "--replay/--arena are single-turn arms; they do not compose with --mode {} \
                 (the executor records one sequence's step and the concurrent path drives many)",
                args.mode
            );
        }
        let encode = |text: &str| -> Result<Vec<u32>> {
            Ok(tokenizer
                .encode(text, false)
                .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
                .get_ids()
                .to_vec())
        };
        let ctx = concurrent::Ctx {
            model: &model,
            config: &config,
            device: &device,
            dtype,
            eos_ids: &eos_ids,
            mode: &args.mode,
            max_tokens: args.max_tokens,
            readback: &args.readback,
            lookahead: args.lookahead,
            commit_budget: &args.commit_budget,
            sequences: args.sequences,
            chunk: args.chunk,
            dspark: args.dspark,
            batched: args.batched,
            check_independence: !args.no_independence_check,
            skip_solo: args.skip_solo,
            dspark_setup: dspark_setup
                .as_ref()
                .map(|s| (s.draft_dir.clone(), s.dcfg.clone())),
            first: &args.first,
            timeline: args.timeline.clone(),
            encode: &encode,
        };
        return concurrent::run(ctx);
    }

    // Install the ICB replay executor AFTER warmup — so the warmup's forwards do
    // not pollute the recorded step — and before the timed decode. The executor
    // records `ICB_RECORD_STEPS` real decode steps, then replays the stable
    // packed subset each step. `--replay-barriers skip-replayed` is the separable
    // barrier-suppression win (§11.3r); `--replay-barrier-scope` is §11.3l's
    // open question 3. Metal-only; a no-op wherever `--replay` is off.
    // Held so the run can report what the executor actually did: a flag being
    // passed does not establish that anything was replayed (§2.4), and the
    // A/B below is only about the executor if `covered > 0` and `replaying`.
    #[cfg(feature = "metal")]
    let mut icb_exec: Option<std::sync::Arc<candle::kernels::IcbExecutor>> = None;
    // The arena's recording parameters, shared by `--arena` and `--replay`: one
    // step skipped (the first token after the prefill allocates a different
    // set), two recorded, installed at the end of the second.
    #[cfg(feature = "metal")]
    const ARENA_SKIP: usize = 1;
    #[cfg(feature = "metal")]
    const ARENA_RECORD_STEPS: usize = 2;
    #[cfg(feature = "metal")]
    let arena = (args.arena || args.replay) && !args.dspark;
    #[cfg(feature = "metal")]
    if arena {
        if let Device::Metal(dev) = &device {
            dev.arm_arena_recording(
                candle::metal_backend::ArenaLayout::Packed,
                ARENA_SKIP,
                ARENA_RECORD_STEPS,
            );
        }
        if !args.replay {
            eprintln!("[arena: packed, recorded from the first decode steps; no executor]");
        }
    }
    #[cfg(feature = "metal")]
    if args.replay {
        use candle::kernels::metal::ExecutorSlot;
        use candle::kernels::{BarrierScope, IcbExecutor, ReplayBarriers};
        const ICB_RECORD_STEPS: usize = 3;
        let scope = match args.replay_barrier_scope.as_str() {
            "run-start" => BarrierScope::RunStart,
            "since-barrier" => BarrierScope::SinceBarrier,
            other => anyhow::bail!(
                "--replay-barrier-scope must be `run-start` or `since-barrier`, got `{other}`"
            ),
        };
        let barriers = match args.replay_barriers.as_str() {
            "always" => ReplayBarriers::Always,
            "skip-replayed" => ReplayBarriers::SkipReplayed,
            other => anyhow::bail!(
                "--replay-barriers must be `always` or `skip-replayed`, got `{other}`"
            ),
        };
        // `configured` already returns `Arc<IcbExecutor>` — do not re-wrap it
        // (the `Arc<Arc<_>>` is what does not coerce to `Arc<dyn Executor>`).
        let exec = IcbExecutor::configured(ICB_RECORD_STEPS, scope, barriers);
        // The third thing a replay needs after packed params and the pool: stable
        // buffer identity. Under the buffer pool every activation is a different
        // buffer on every step (§9.2c), so every position reads as `varies` and
        // coverage is zero. The activation arena (§9.2) is recorded from the
        // decode loop's own steps -- one skipped, two recorded, installed at the
        // end of the second -- through the brackets `Session::decode` calls. The
        // executor lets the prefill's window, those three steps and one settling
        // step pass before it records; §11.3 says why a recording taken earlier
        // excludes everything for a reason that is not the executor's.
        //
        // Not under `--dspark`: a round's allocation order moves with the
        // accepted length, and the arena serves by ordinal.
        if let Device::Metal(dev) = &device {
            if arena {
                exec.set_warm_in(1 + ARENA_SKIP + ARENA_RECORD_STEPS + 1);
                // After a stale step the arena re-records over the next
                // `ARENA_RECORD_STEPS` steps and installs at the end of the
                // last; the executor records after that and one settling step.
                exec.set_rerecord_warm_in(ARENA_RECORD_STEPS + 1);
            }
            dev.set_executor(std::sync::Arc::new(ExecutorSlot::Custom(exec.clone())));
        }
        icb_exec = Some(exec);
        eprintln!(
            "[replay: ICB executor installed — barriers={barriers:?} scope={scope:?} \
             param-style=packed arena={}]",
            if arena {
                "packed (recorded from the first decode steps)"
            } else {
                "off"
            }
        );
    }

    let sampling = if args.temperature <= 0.0 {
        Sampling::ArgMax
    } else {
        Sampling::All {
            temperature: args.temperature,
        }
    };
    // Random seed when unspecified, so repeated `-t > 0` runs differ; a passed
    // `--seed` reproduces. At temp 0 (greedy/argmax) the seed is irrelevant —
    // the path is deterministic and does not sample.
    let seed = args.seed.unwrap_or_else(rand::random::<u64>);
    let mut logits_processor = LogitsProcessor::from_sampling(seed, sampling);

    // ---- prompt: read stdin, wrap in the LFM2 chat template ---------------
    let mut prompt = String::new();
    std::io::stdin()
        .read_to_string(&mut prompt)
        .context("reading prompt from stdin")?;
    let prompt = prompt.trim();
    let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n");
    let mut prompt_ids = tokenizer
        .encode(text.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenizing prompt: {e}"))?
        .get_ids()
        .to_vec();
    // BOS by default — the model's template start (see `--no-bos`). Coupled with
    // the `<|im_end|>`-only EOS set above: together they lift acceptance and make
    // replies terminate; BOS without the EOS fix drove a non-terminating ramble.
    if !args.no_bos {
        let bos = tokenizer
            .token_to_id("<|startoftext|>")
            .context("tokenizer has no <|startoftext|> — LFM2's BOS")?;
        prompt_ids.insert(0, bos);
    }

    // ---- decode: one Session::decode call --------------------------------
    //
    // The strategy is selected here and the loop never names it. Each arm is
    // one value of the `ReadbackStrategy` seam; the loop is generic over it and
    // monomorphizes, so selecting an arm costs nothing per token.
    let budget = CommitBudget::from_str(&args.commit_budget).map_err(anyhow::Error::msg)?;
    if args.phase_sync && !args.dspark {
        anyhow::bail!("--phase-sync brackets the speculative round's phases; it needs --dspark");
    }
    // A speculative round is not a steady-state step: its dispatch structure moves
    // with the accepted length, so every run goes stale within a few rounds. The
    // executor validates a run before issuing it and re-records after a stale
    // step, so the stream is correct either way -- but a plan that is dropped
    // every few rounds covers nothing, and the arena is not armed here at all.
    // Refused so the arm's name is not worn by a classical run (§7.1a-i).
    if args.replay && args.dspark {
        anyhow::bail!(
            "--replay cannot be combined with --dspark: the speculative round's dispatch \
             structure is not stable across rounds, so the executor would re-record every \
             few rounds and replay nothing (DESIGN.md §11.3x)"
        );
    }
    // The in-flight depth: the host/GPU balance warmup measured (`compute_in_flight_depth`),
    // so the pacer is not a static cap — 2 here (GPU step >> host encode), deeper
    // on a faster GPU / smaller model. `LLOOM_IN_FLIGHT` overrides for an A/B.
    let in_flight_depth = match std::env::var("LLOOM_IN_FLIGHT") {
        Ok(v) => v
            .parse()
            .unwrap_or_else(|_| panic!("LLOOM_IN_FLIGHT must be an integer, got {v:?}")),
        Err(_) => session.measured_in_flight_depth(),
    };
    let opts = DecodeOptions {
        max_tokens: args.max_tokens,
        budget,
        lookahead: args.lookahead,
        // Under the speculative body this is the per-phase drain (decode.rs).
        sample_split: args.phase_sync,
        in_flight_depth,
        // 0 = one-shot prefill (incumbent); >0 streams the prompt through the pacer.
        prefill_ubatch: args.prefill_ubatch,
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
            // every accepted token, plus the block width and slack. `max_tokens`
            // is the token ceiling — EXCEPT `0` means UNBOUNDED (run until EOS),
            // where it is NOT the bound: the run is bounded in practice by the KV
            // cache (`DEFAULT_KV_CAPACITY`), which is the furthest the target — and
            // so the drafter, which attends the same positions — can reach. Sizing
            // the drafter's RoPE table from `max_tokens=0` gave it just
            // `prompt+block+64` rows (89 for a 16-token prompt), so an unbounded
            // `--dspark` run crashed the moment it drafted past ~position 82
            // (`DSparkRotary::apply`'s `cos.narrow`). Use the KV ceiling instead.
            let generation_ceiling = if args.max_tokens == 0 {
                DEFAULT_KV_CAPACITY
            } else {
                args.max_tokens
            };
            let max_pos = prompt_ids.len() + generation_ceiling + setup.dcfg.block_size + 64;
            // F16 to match the target the block is embedded with — the drafter's
            // stored BF16 has no CPU matmul and would need a cast at every meeting
            // of the two dtypes (`lfm2_dspark.rs` `DSparkDrafter::new` docs).
            let dvb =
                unsafe { VarBuilder::from_mmaped_safetensors(&[draft_file], DType::F16, &device)? };
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
    // Wall-clock to the first token the consumer is HANDED (the batch arrives),
    // and to the first VISIBLE token (a byte reaches stdout). The reported
    // `ttft_ms` is compute-availability (prefill + step0); this pair is what the
    // user actually waits, and the gap between them is the commit-cadence /
    // consumer coupling we are chasing. Measurement seam only.
    let mut first_batch_at: Option<std::time::Duration> = None;
    let mut first_visible_at: Option<std::time::Duration> = None;
    let out = {
        // Returns `candle::Result`, which is the type `Session::decode`'s callback
        // wants: `candle` is the workspace alias for `lloom-tensor` in this tree,
        // so `candle::Result` and `lloom_tensor::Result` are one type. `next_token`
        // already returns it, so no map is needed.
        let mut on_tokens = |batch: &[u32]| -> candle::Result<()> {
            if first_batch_at.is_none() && !batch.is_empty() {
                first_batch_at = Some(decode_start.elapsed());
            }
            for &id in batch {
                if let Some(text) = tos.next_token(id)? {
                    if first_visible_at.is_none() {
                        first_visible_at = Some(decode_start.elapsed());
                    }
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
    // The measured user-facing latency: when the first batch was HANDED to the
    // consumer vs the compute-availability `ttft_ms`. A gap ≫ 0 means the first
    // token waited on the commit cadence (period), not on compute — the coupling
    // the consumer decouple targets. Measurement seam.
    eprintln!(
        "[first-token wall-clock: batch-handed {} ms, visible {} ms | ttft(compute) {:.2} ms]",
        first_batch_at
            .map(|d| format!("{:.2}", d.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "n/a".into()),
        first_visible_at
            .map(|d| format!("{:.2}", d.as_secs_f64() * 1e3))
            .unwrap_or_else(|| "n/a".into()),
        ttft_ms,
    );
    // The two TTFT terms apart, and the first steps by phase: `prefill_ns` is
    // the prompt's encode, and under `lookahead = 1` step 0's EOS wait is where
    // the prefill's GPU tail lands, so the split says which side a first-token
    // cost sits on.
    let ms = |ns: u64| ns as f64 / 1e6;
    eprintln!(
        "[ttft split: prefill {:.2} ms + step0 {:.2} ms; step_ns[1..={}] = {}; prompt_tokens={}]",
        ms(st.prefill_ns),
        ms(st.step_ns.first().copied().unwrap_or(0)),
        st.step_ns.len().min(5).saturating_sub(1),
        st.step_ns
            .iter()
            .skip(1)
            .take(4)
            .map(|&n| format!("{:.2}", ms(n)))
            .collect::<Vec<_>>()
            .join(" "),
        prompt_ids.len(),
    );
    for (name, census) in [
        ("prefill", &st.prefill_census),
        ("step 0", &st.step0_census),
    ] {
        if census.is_empty() {
            continue;
        }
        let total: u64 = census.iter().map(|(_, n)| n).sum();
        eprintln!(
            "[{name} census: {total} dispatches, {} kernels]",
            census.len()
        );
        for (label, n) in census {
            eprintln!("    {n:>6}  {label}");
        }
    }
    for (i, p) in st.first_steps.iter().enumerate() {
        eprintln!(
            "[step {i} phases: sample {:.2} ms, forward {:.2} ms, backpressure {:.2} ms, eos_wait {:.2} ms]",
            ms(p.sample_ns),
            ms(p.forward_ns),
            ms(p.backpressure_ns),
            ms(p.eos_ns)
        );
    }
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

    // The KV-append arm within the cache config, so `cache=crown` is not
    // ambiguous between the `Cat` and `InPlace` arms (they print the same
    // `cache_arm` string otherwise; the crown default is `InPlace` since the
    // A/B inverted, `LLOOM_CROWN_KV=cat` restores `Cat`).
    let kv_arm = match config.kv_append {
        KvAppend::InPlace => "inplace",
        KvAppend::Cat => "cat",
        KvAppend::Paged => "paged",
    };
    eprintln!(
        "[{n} tokens: {} tok/s end-to-end (prefill included), \
         {} {} , TTFT {} ms, \
         {} on {:?}; cache={cache_arm} kv={kv_arm} bos={} \
         readback={} lookahead={} commit_budget={} commits={} max_in_flight={} \
         step_markers={} step_waits={} cut={} \
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
        !args.no_bos,
        if args.dspark {
            "dspark"
        } else {
            &args.readback
        },
        args.lookahead,
        budget.arm(),
        st.commits,
        st.max_in_flight,
        st.step_markers,
        st.step_waits,
        cut_policy_arm(&device),
        st.readback.ring_publishes,
        st.readback.poll_fallbacks,
        st.overrun,
        out.eos,
    );
    #[cfg(feature = "metal")]
    if let Some(exec) = icb_exec.as_ref() {
        report_replay(exec, st.steps);
    }
    // GPU busy per step, when the census is on: the split between an arm
    // costing the host and an arm costing the GPU (§11.3s landed GPU-side).
    // `drain_all` above synchronised, so every completion handler has run.
    #[cfg(feature = "metal")]
    if candle::kernels::metal::profile::enabled() {
        let s = candle::kernels::metal::profile::snapshot();
        let steps = st.steps.max(1) as f64;
        eprintln!(
            "[profile: gpu_busy {:.3} ms/step (union over {} command buffers, {} timed), \
             {:.1} dispatches/step, {} steps]",
            1e3 * s.gpu_busy_union_s / steps,
            s.command_buffers,
            s.timed_command_buffers,
            s.dispatches as f64 / steps,
            st.steps,
        );
    }
    #[cfg(feature = "metal")]
    if arena {
        if let Device::Metal(dev) = &device {
            match dev.arena_auto_report() {
                Some(r) => eprintln!(
                    "[arena: {:?} installed after decode step {}: {} of {} ordinals served; \
                     rerecordings={} counters={:?}]",
                    r.layout,
                    r.installed_at_step,
                    r.covered,
                    r.total,
                    dev.arena_rerecordings(),
                    dev.arena_counters(),
                ),
                None => eprintln!("[arena: not installed]"),
            }
        }
    }
    Ok(())
}

/// The command-buffer cut policy in force, read back from the device rather
/// than from the environment that selected it (§7.1a-i), with the counter's
/// period beside it.
fn cut_policy_arm(device: &Device) -> String {
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = device {
        return candle::metal_backend::CutPolicy::render(
            dev.cut_policy(),
            dev.compute_per_buffer(),
        );
    }
    let _ = device;
    "n/a".to_string()
}

/// What the ICB executor did on this run: engagement first, then the residency
/// term (`DESIGN.md` §11.3w). `replaying=false` or `covered=0` means the A/B
/// above compared the classical path against itself wearing this arm's name.
#[cfg(feature = "metal")]
fn report_replay(exec: &candle::kernels::IcbExecutor, steps: u64) {
    use candle::kernels::metal::trace;
    let cov = exec.coverage();
    eprintln!(
        "[replay: replaying={} covered={}/{} runs={} barriers_encoded={} stale={} \
         stale_runs={} prefix_flushes={} rerecordings={} discarded_windows={} \
         excluded_varies={} excluded_inline_consts={} excluded_no_pipeline={}]",
        exec.is_replaying(),
        cov.covered,
        cov.positions,
        exec.runs(),
        exec.encoded_barriers(),
        exec.stale_positions(),
        exec.stale_runs(),
        exec.prefix_flushes(),
        exec.rerecordings(),
        exec.discarded_windows(),
        cov.varies,
        cov.inline_constants,
        cov.no_pipeline,
    );
    eprintln!("[replay: warm_in_skipped={}]", exec.warm_in_skipped());
    let ranges = trace::execute_range_count();
    let use_resource = trace::use_resource_count();
    let steps = steps.max(1) as f64;
    // The residency arm is named from the plan, not from the variable that
    // selected it (§7.1a-i): per-run declares `residency_per_step`, plan-wide
    // declares `runs x plan_wide`.
    let arm = if exec.runs() > 0 && exec.residency_per_step() == exec.runs() * exec.plan_resident()
    {
        "plan-wide"
    } else {
        "per-run"
    };
    eprintln!(
        "[replay: execute_ranges={ranges} ({:.1}/step) useResource={use_resource} ({:.1}/step) \
         residency={arm} per_step={} plan_wide={} barriers_suppressed={}]",
        ranges as f64 / steps,
        use_resource as f64 / steps,
        exec.residency_per_step(),
        exec.plan_resident(),
        trace::barriers_suppressed(),
    );
    // The pool's four arms (`ConstantsPool`): `detached > 0` is the check that
    // keeps a lookahead loop from rewriting a params block under an in-flight
    // reader having fired; `rewritten` past the first steps means a position's
    // block changes between steps and cannot be replayed from its slot.
    let pool = candle::kernels::constants_pool()
        .map(|p| (p.len(), p.counters()))
        .unwrap_or_default();
    eprintln!(
        "[replay: constants_pool slots={} allocated={} unchanged={} rewritten={} detached={}]",
        pool.0, pool.1.allocated, pool.1.unchanged, pool.1.rewritten, pool.1.detached,
    );
    for ((kernel, reason), n) in &cov.excluded_by_kernel {
        eprintln!("[replay excluded {n:4} {kernel:<44} {reason}]");
    }
    // What went stale, attributed: which axis of the recording moved, on which
    // kernels, and the first occurrences dated by step -- a count says a run
    // stopped matching, this says what moved and when.
    let stale = exec.stale_report();
    if !stale.first.is_empty() {
        let by_axis: Vec<String> = stale
            .by_axis
            .iter()
            .map(|(axis, n)| format!("{axis}={n}"))
            .collect();
        eprintln!(
            "[replay stale: steps_with_stale={} by_axis: {}]",
            stale.steps,
            by_axis.join(" ")
        );
        for (kernel, n) in &stale.by_kernel {
            eprintln!("[replay stale {n:4} {kernel}]");
        }
        for (step, pos, kernel, axis, at_head) in &stale.first {
            eprintln!(
                "[replay stale first: step={step} pos={pos} {kernel} {axis} {}]",
                if *at_head { "head" } else { "member" }
            );
        }
    }
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
    // Host readbacks per phase: each is a wait for the GPU, and the round owes
    // exactly one (the verify's argmax). Counted, not inferred — a draft that
    // drained its ids to the host shows here and nowhere in the digests.
    let rb_draft: u64 = timed.iter().map(|r| r.timing.draft_readbacks).sum();
    let rb_verify: u64 = timed.iter().map(|r| r.timing.verify_readbacks).sum();
    let rb_observe: u64 = timed.iter().map(|r| r.timing.observe_readbacks).sum();
    eprintln!(
        "[dspark host readbacks per round (mean):  draft {:.2}   verify {:.2}   observe {:.2}   round {:.2}]",
        rb_draft as f64 / n,
        rb_verify as f64 / n,
        rb_observe as f64 / n,
        (rb_draft + rb_verify + rb_observe) as f64 / n
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
         draft_disp={:.1} verify_disp={:.1} observe_disp={:.1} \
         draft_readbacks={:.3} verify_readbacks={:.3} observe_readbacks={:.3} \
         phase_sync={phase_sync} profiled={profiled}",
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
        rb_draft as f64 / n,
        rb_verify as f64 / n,
        rb_observe as f64 / n,
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
    // The device-side id history, sized EXACTLY from the per-step geometry — a
    // wrapping ring whose live window is `period × quantum`, NOT the whole
    // generation, so an unbounded (`-n 0`) run streams at any length by reusing
    // the ring. B=1 greedy here (IdentityProposer): quantum = 1, so the ring is
    // tens of slots, always CPU-mapped, zero-copy. (`Geometry::cap` is the tight,
    // no-margin size; a lap is a loud bail, not silent corruption.)
    let geometry = lloom::generation::Geometry {
        batch: 1,
        max_draft: 0,
        period: opts.budget.period(),
        lookahead: opts.lookahead,
        // Widen the ring's input buffer to the streamed-prefill chunk so a chunk
        // stages into one reused buffer; 0 leaves it at `quantum` (the one-shot
        // path allocates its whole-prompt input tensor via the trait default, so
        // the input buffer width is not consulted there).
        max_input_width: opts.prefill_ubatch(),
    };
    let out = match readback {
        "blocking" => {
            let mut rb = Blocking::new(eos_ids, device)?;
            session.decode(spec, &mut rb, sampler, prompt_ids, opts, on_tokens)?
        }
        "ring" => {
            let mut rb = Ring::with_geometry(eos_ids, device, geometry, EosRead::Blocking)?;
            session.decode(spec, &mut rb, sampler, prompt_ids, opts, on_tokens)?
        }
        "ring-nonblocking" => {
            let mut rb = Ring::with_geometry(
                eos_ids,
                device,
                geometry,
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
            let mut rb = Ring::with_geometry(eos_ids, device, geometry, EosRead::HostOnDrain)?;
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
