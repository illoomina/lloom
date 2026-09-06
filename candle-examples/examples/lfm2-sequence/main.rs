//! The sequence layer on the real model (issue #677,
//! `design/docs/sequence-reuse-design.md`): prefix reuse across a turn
//! boundary, the EOS overrun rewound at the boundary, and several sequences
//! interleaved on one model.
//!
//! Two modes, each printing one `RESULT` line per arm:
//!
//! * `--mode reuse` — a two-turn conversation decoded two ways. The REUSE arm
//!   decodes turn 1, then hands the whole conversation (turn 1's prompt, its
//!   reply, turn 2's prompt) to `Session::prepare`, which reuses the cached
//!   prefix and prefills only turn 2; the FRESH arm prefills the whole
//!   conversation into an empty cache. Both decode turn 2 greedily and the
//!   line reports whether the streams agree, how many tokens each arm
//!   prefilled, and turn 2's TTFT on each. With `--conv-prefill-ctx` off the
//!   two disagree (`DESIGN.md` §13.2f defect 1); on, they must agree.
//! * `--mode concurrent` — `--sequences` independent conversations, each
//!   decoded solo and then all of them interleaved on one model in
//!   `--chunk`-token slices (each visit resumes its sequence through
//!   `Session::prepare`). The line reports whether every interleaved stream
//!   equals its solo stream, and the aggregate token rate of the interleaved
//!   run — the honest "N sequences, X tok/s aggregate" the issue asks for,
//!   which is time-sliced rather than batched.
//!
//! * `--mode concurrent --dspark` — the same `--sequences` prompts through the
//!   DSpark speculative loop: each decoded solo with `decode_speculative`, then
//!   all of them through `Session::decode_speculative_pipelined`, which keeps
//!   one verify per sequence in flight and overlaps one sequence's draft and
//!   observe with the others' verifies
//!   (`design/docs/speculative-pipeline-scheduler-design.md` §10). Reports per-sequence
//!   identity against the solo stream, aggregate tok/s against the solo mean,
//!   and the per-round phase brackets; `--timeline` dumps them per round.
//!
//! Loading and the ChatML wrapping are `lfm2-chat`'s. The model directory
//! defaults to the LFM2.5-2.6B snapshot in the HF cache.

use anyhow::{Context, Result};
use candle::{DType, Device};
use clap::Parser;
use lloom::generation::poll::PollMechanism;
use lloom::generation::{
    Blocking, CommitBudget, EosRead, LogitsProcessor, ReadbackStrategy, Ring, Sampling,
};
use lloom::models::lfm2::{AttnImpl, Cache, ConvState, KvAppend, Lfm2Config, Model};
use lloom::models::lfm2_dspark::{DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, ReuseAction, Session, VarBuilder};
use std::path::PathBuf;
use std::str::FromStr;
use tokenizers::Tokenizer;

/// `design/workloads/prompts/`, the #549 corpus, embedded so the binary needs
/// no path: a multi-turn pair, and four single prompts of different shapes.
const TURN_1: &str = "I am designing a cache for a single-threaded interpreter. Entries are immutable once written and the working set is about eight times the cache. What eviction policy should I start with, and why not LRU?";
const TURN_2: &str = "Suppose I measure and find the hit rate is 40% lower than you predicted. What are the two most likely causes, and which measurement separates them?";
const PROMPTS: [&str; 4] = [
    "Name the three branches of the United States federal government.",
    "Count from one to forty, writing each number as a word on its own line, with no other text.",
    "Here is a Rust function that is missing its body:\n\nfn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32> {\n    // Merge two already-sorted slices into one sorted Vec.\n    // Do not allocate more than once.\n}\n\nWrite the body. Explain your choice of capacity for the allocation, and say what happens when one slice is empty.",
    "Extract every date, monetary amount, and party name from the paragraph below and return them as a JSON object with the keys \"dates\", \"amounts\", and \"parties\". Return only the JSON.\n\nOn 14 March 2019, Halloran Freight Ltd. agreed to pay Baswick Terminals the sum of 48,250 GBP in settlement of demurrage accrued between 2 January 2019 and 27 February 2019. A further 6,900 GBP was withheld pending inspection, to be released on or before 30 April 2019 unless Baswick Terminals lodged an objection.",
];

#[derive(Parser)]
struct Args {
    /// Checkpoint directory. Defaults to LFM2_MODEL_DIR, then the LFM2.5-2.6B
    /// snapshot in the HF cache.
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// `reuse` or `concurrent`.
    #[arg(long, default_value = "reuse")]
    mode: String,

    /// Tokens per turn (`reuse`) or per sequence (`concurrent`); EOS may end
    /// a run earlier.
    #[arg(long, short = 'n', default_value_t = 64)]
    max_tokens: usize,

    /// `crown` (`Sdpa × Cat × Shuffle` — cannot rewind) or `rewindable`
    /// (`Sdpa × InPlace × RotatingRing{K}` — can), as `lfm2-chat --cache`.
    #[arg(long, default_value = "rewindable")]
    cache: String,

    /// The rotating ring's history depth under `--cache rewindable`: the
    /// rewind horizon. Must cover the overrun the readback arm can produce
    /// (1 under `--lookahead 1`, the commit period under `ring-hostdrain`).
    #[arg(long, default_value_t = 16)]
    ring_k: usize,

    /// Carry the conv state across the turn-boundary prefill
    /// (`ConvPrefillCtx`, `DESIGN.md` §13.2f). Off by default (§7.1a).
    #[arg(long)]
    conv_prefill_ctx: bool,

    /// `blocking`, `ring`, `ring-nonblocking` or `ring-hostdrain`, as
    /// `lfm2-chat --readback`.
    #[arg(long, default_value = "ring-nonblocking")]
    readback: String,

    /// Forward passes the host may enqueue before learning the previous
    /// step's EOS (ring arms): 0 or 1.
    #[arg(long, default_value_t = 1)]
    lookahead: usize,

    /// `per-token` or `steps:<N>`.
    #[arg(long, default_value = "steps:8")]
    commit_budget: String,

    /// Sequences to interleave under `--mode concurrent` (at most 4).
    #[arg(long, default_value_t = 2)]
    sequences: usize,

    /// Tokens per visit under `--mode concurrent`.
    #[arg(long, default_value_t = 16)]
    chunk: usize,

    /// Force CPU.
    #[arg(long)]
    cpu: bool,

    /// Under `--mode concurrent`: drive every sequence with the DSpark
    /// speculative drafter — solo through `decode_speculative`, then all of
    /// them through the pipelined driver. Forces the rewindable cache with the
    /// drafter's taps, as `lfm2-chat --dspark` does.
    #[arg(long)]
    dspark: bool,

    /// The DSpark drafter checkpoint directory (config.json + model.safetensors).
    /// Defaults to the newest LFM2.5-2.6B-DSpark snapshot in the HF cache.
    #[arg(long)]
    draft_dir: Option<PathBuf>,

    /// Under `--dspark`: which arm runs first, `solo` or `pipelined`, so a
    /// rotation can alternate the order and the null control has both.
    #[arg(long, default_value = "solo")]
    first: String,

    /// Under `--dspark`: write every round of both arms as CSV
    /// (`arm,seq,round,start_ms,draft_ms,enqueue_ms,wait_ms,observe_ms,accepted`).
    #[arg(long)]
    timeline: Option<PathBuf>,
}

fn default_model_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("LFM2_MODEL_DIR") {
        return Some(PathBuf::from(dir));
    }
    let home = std::env::var("HOME").ok()?;
    let snapshots =
        PathBuf::from(home).join(".cache/huggingface/hub/models--LiquidAI--LFM2.5-2.6B/snapshots");
    std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
}

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

/// The newest snapshot of a HF repo in the cache with a `config.json`, if any
/// (`lfm2-chat`'s discovery).
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

/// The DSpark drafter's constants from its own `config.json` — `lfm2-chat`'s
/// reader, verbatim: the two fields a wrong default silently corrupts
/// (`rope_theta`, `rope_is_neox_style`) are never defaulted.
fn read_drafter_config(dir: &std::path::Path) -> Result<DSparkConfig> {
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
        logits_start: 0,
    })
}

/// Host readback counts since process start: `(to_cpu fast, to_cpu blit,
/// staged)`; zeros off Metal. The first two wait on the queue's tail, the
/// third on its own producer.
fn readback_counts() -> (u64, u64, u64) {
    #[cfg(feature = "metal")]
    {
        let (f, b) = candle::metal_backend::to_cpu_counts();
        (f, b, candle::metal_backend::to_cpu_staged_count())
    }
    #[cfg(not(feature = "metal"))]
    {
        (0, 0, 0)
    }
}

/// `(gpu_busy_union_s, gpu_busy_sum_s, dispatches, timed command buffers)`
/// from the Metal profiler since its last reset, resetting it; zeros when the
/// profiler is off or off Metal. Sum over union is how much of the GPU time
/// two command buffers spent executing at once.
fn profile_take(device: &Device) -> Result<(f64, f64, u64, u64)> {
    #[cfg(feature = "metal")]
    {
        use candle::metal_backend::profile;
        if profile::enabled() {
            device.synchronize()?;
            let snap = profile::snapshot();
            profile::reset();
            return Ok((
                snap.gpu_busy_union_s,
                snap.gpu_busy_sum_s,
                snap.dispatches,
                snap.timed_command_buffers,
            ));
        }
    }
    let _ = device;
    Ok((0.0, 0.0, 0, 0))
}

/// Median of a sample, `NaN` when empty.
fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// One arm of the `--dspark` concurrent run, for the RESULT line and the timeline.
struct SpecArm {
    name: &'static str,
    streams: Vec<Vec<u32>>,
    outs: Vec<DecodeOut>,
    wall_s: f64,
    gpu_busy_s: f64,
    gpu_busy_sum_s: f64,
    dispatches: u64,
    command_buffers: u64,
    /// Readbacks the arm performed: `(to_cpu fast, to_cpu blit, staged)`.
    readbacks: (u64, u64, u64),
}

impl SpecArm {
    fn tokens(&self) -> usize {
        self.streams.iter().map(|s| s.len()).sum()
    }
    fn rounds(&self) -> usize {
        self.outs.iter().map(|o| o.stats.rounds.len()).sum()
    }
    fn per_round_ms(&self, f: impl Fn(&lloom::nn::RoundStats) -> u64) -> f64 {
        median(
            self.outs
                .iter()
                .flat_map(|o| o.stats.rounds.iter().map(|r| f(r) as f64 / 1e6))
                .collect(),
        )
    }
}

/// `lfm2-release`'s digest of a token stream, so the two harnesses' digests
/// compare directly. Its self-test and one-shot helper are that harness's.
#[allow(dead_code)]
#[path = "../lfm2-release/sha256.rs"]
mod sha256;

fn sha256_hex(ids: &[u32]) -> String {
    let mut h = sha256::Sha256::new();
    for t in ids {
        h.update(&t.to_le_bytes());
    }
    sha256::hex(&h.finalize())
}

/// One readback strategy per sequence, chosen by name.
enum Arm {
    Blocking(Blocking),
    Ring(Ring),
}

impl Arm {
    fn new(name: &str, eos_ids: &[u32], device: &Device, capacity: usize) -> Result<Self> {
        Ok(match name {
            "blocking" => Arm::Blocking(Blocking::new(eos_ids, device)?),
            "ring" => Arm::Ring(Ring::new(eos_ids, device, capacity, EosRead::Blocking)?),
            "ring-nonblocking" => Arm::Ring(Ring::new(
                eos_ids,
                device,
                capacity,
                EosRead::Poll(PollMechanism::default()),
            )?),
            "ring-hostdrain" => {
                Arm::Ring(Ring::new(eos_ids, device, capacity, EosRead::HostOnDrain)?)
            }
            other => anyhow::bail!(
                "--readback must be `blocking`, `ring`, `ring-nonblocking` or `ring-hostdrain`, \
                 got `{other}`"
            ),
        })
    }

    fn eos_read(&self) -> &'static str {
        match self {
            Arm::Blocking(b) => b.eos_read_arm(),
            Arm::Ring(r) => r.eos_read_arm(),
        }
    }
}

/// A timed decode call: the outcome, wall, and TTFT (prefill submit + step 0).
struct Timed {
    out: DecodeOut,
    wall: std::time::Duration,
    ttft_ms: f64,
}

fn decode(
    session: &mut Session<'_, Model>,
    arm: &mut Arm,
    prompt: &[u32],
    opts: &DecodeOptions,
) -> Result<Timed> {
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let mut spec = IdentityProposer;
    let t = std::time::Instant::now();
    let out = match arm {
        Arm::Blocking(rb) => {
            session.decode(&mut spec, rb, &mut sampler, prompt, opts, &mut |_| Ok(()))?
        }
        Arm::Ring(rb) => {
            session.decode(&mut spec, rb, &mut sampler, prompt, opts, &mut |_| Ok(()))?
        }
    };
    let wall = t.elapsed();
    let ttft_ms =
        (out.stats.prefill_ns + out.stats.step_ns.first().copied().unwrap_or(0)) as f64 / 1e6;
    Ok(Timed { out, wall, ttft_ms })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let model_dir = args
        .model_dir
        .clone()
        .or_else(default_model_dir)
        .context("no --model-dir given and the LFM2.5-2.6B snapshot was not in the HF cache")?;
    let device = candle_examples::device(args.cpu)?;
    let dtype = if device.is_cpu() {
        DType::F32
    } else {
        DType::F16
    };

    let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let config: Lfm2Config = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)
        .context("parsing config.json")?;
    let mut config = config.into_config(false);

    // The two caches `lfm2-chat` names, with the crown's fusions on both so
    // the only difference is whether the cache can rewind.
    config.attn_impl = AttnImpl::Sdpa;
    config.gemv_silu = true;
    config.gemv_residual = true;
    config.conv_fused = true;
    // The drafter's constants are resolved before the target config: `--dspark`
    // needs the rewindable cache (`advance`/`resolve`), the ring sized for the
    // `K + 1` verify window, and the drafter's five taps at F32 — the same
    // forcing `lfm2-chat --dspark` applies.
    let dspark_setup: Option<(PathBuf, DSparkConfig)> = if args.dspark {
        if args.mode != "concurrent" {
            anyhow::bail!("--dspark is a `--mode concurrent` arm");
        }
        let draft_dir = args
            .draft_dir
            .clone()
            .or_else(|| newest_snapshot("models--LiquidAI--LFM2.5-2.6B-DSpark"))
            .context("no LFM2.5-2.6B-DSpark snapshot in the HF cache and no --draft-dir")?;
        let dcfg = read_drafter_config(&draft_dir)?;
        Some((draft_dir, dcfg))
    } else {
        None
    };
    let cache_name = if dspark_setup.is_some() {
        "rewindable"
    } else {
        args.cache.as_str()
    };
    match cache_name {
        "crown" => {
            config.kv_append = KvAppend::Cat;
            config.conv_state = ConvState::Shuffle;
        }
        "rewindable" => {
            config.kv_append = KvAppend::InPlace;
            let k = match &dspark_setup {
                Some((_, dcfg)) => dcfg.max_draft() + 1,
                None => args.ring_k,
            };
            config.conv_state = ConvState::RotatingRing { k };
        }
        other => anyhow::bail!("--cache must be `crown` or `rewindable`, got `{other}`"),
    }
    if let Some((_, dcfg)) = &dspark_setup {
        config.tap_dtype = DType::F32;
        config.tap_layers = dcfg.tap_layers();
    }
    config.conv_prefill_ctx = args.conv_prefill_ctx;

    let files = weight_files(&model_dir)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device)? };
    let model = Model::new(&config, vb).context("constructing LFM2")?;

    let mut eos_ids: Vec<u32> = config.eos_token_id.into_iter().collect();
    for tok in ["<|im_end|>", "<|endoftext|>"] {
        if let Some(id) = tokenizer.token_to_id(tok) {
            eos_ids.push(id);
        }
    }
    eos_ids.sort_unstable();
    eos_ids.dedup();

    let budget = CommitBudget::from_str(&args.commit_budget).map_err(anyhow::Error::msg)?;
    let lookahead = if args.readback == "blocking" {
        0
    } else {
        args.lookahead
    };
    let opts = DecodeOptions {
        max_tokens: args.max_tokens,
        budget,
        lookahead,
        sample_split: false,
    };
    // Enough ring slots for every turn or visit plus the overrun each may
    // publish; `Ring::new` floors it at a few pages anyway.
    let capacity = 8 * args.max_tokens + 8 * (budget.period() + 2);

    let encode = |text: &str| -> Result<Vec<u32>> {
        Ok(tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizing: {e}"))?
            .get_ids()
            .to_vec())
    };
    let user_turn = |p: &str| format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n");

    // Warm the pipelines once, off every clock below.
    {
        let mut s = Session::new(&model, Cache::new(true, dtype, &config, &device)?);
        s.warmup(&device)?;
    }

    let common = format!(
        "cache={} ring_k={} conv_prefill_ctx={} readback={} lookahead={} commit_budget={} \
         in_flight_steps={} max_tokens={} dtype={:?}",
        args.cache,
        args.ring_k,
        config.conv_prefill_ctx,
        args.readback,
        lookahead,
        budget.arm(),
        lloom::nn::decode::IN_FLIGHT_STEPS,
        args.max_tokens,
        dtype
    );

    match args.mode.as_str() {
        "reuse" => {
            // ---- REUSE: turn 1, then turn 2 through `prepare` ------------
            let turn1 = encode(&user_turn(TURN_1))?;
            let mut session = Session::new(&model, Cache::new(true, dtype, &config, &device)?);
            let mut arm = Arm::new(&args.readback, &eos_ids, &device, capacity)?;
            let t1 = decode(&mut session, &mut arm, &turn1, &opts)?;
            let reply1 = t1.out.ids.clone();
            eprintln!(
                "[turn 1: {} tokens, eos={}, overrun={}, {:.1} tok/s]",
                reply1.len(),
                t1.out.eos,
                t1.out.stats.overrun,
                reply1.len() as f64 / t1.wall.as_secs_f64()
            );
            // The whole conversation at the id level: what the cache holds is
            // ids, so the request is built from ids rather than re-tokenized
            // text (a re-tokenized reply need not reproduce the sampled ids).
            let mut full2 = turn1.clone();
            full2.extend_from_slice(&reply1);
            let turn2_text = if t1.out.eos {
                format!("<|im_end|>\n{}", user_turn(TURN_2))
            } else {
                user_turn(TURN_2)
            };
            full2.extend(encode(&turn2_text)?);
            let overrun_before = session.overrun();
            let prepared = session.prepare(&full2)?;
            let t2 = decode(&mut session, &mut arm, &prepared.suffix, &opts)?;
            let reply2_reuse = t2.out.ids.clone();

            // ---- FRESH: the whole conversation into an empty cache --------
            let mut fresh = Session::new(&model, Cache::new(true, dtype, &config, &device)?);
            let mut arm_f = Arm::new(&args.readback, &eos_ids, &device, capacity)?;
            let tf = decode(&mut fresh, &mut arm_f, &full2, &opts)?;
            let reply2_fresh = tf.out.ids.clone();

            let same = reply2_reuse == reply2_fresh;
            let common_len = reply2_reuse
                .iter()
                .zip(reply2_fresh.iter())
                .take_while(|(a, b)| a == b)
                .count();
            eprintln!(
                "[turn 2 reuse: action={:?} reused={} prefilled={} of {} full, overrun_before={} \
                 eos={} {} tokens, TTFT {:.1} ms]",
                prepared.action,
                prepared.reused,
                prepared.suffix.len(),
                full2.len(),
                overrun_before,
                t2.out.eos,
                reply2_reuse.len(),
                t2.ttft_ms
            );
            eprintln!(
                "[turn 2 fresh: prefilled={} eos={} {} tokens, TTFT {:.1} ms]",
                full2.len(),
                tf.out.eos,
                reply2_fresh.len(),
                tf.ttft_ms
            );
            eprintln!(
                "[turn 2 streams {}: common prefix {} of {} / {}]",
                if same { "IDENTICAL" } else { "DIFFER" },
                common_len,
                reply2_reuse.len(),
                reply2_fresh.len()
            );
            println!(
                "RESULT mode=reuse {common} eos_read={} turn1_tokens={} turn1_eos={} \
                 turn1_overrun={} reuse_action={:?} reused={} prefilled_reuse={} \
                 prefilled_fresh={} full={} turn2_tokens_reuse={} turn2_tokens_fresh={} \
                 identical={} common_prefix={} ttft_reuse_ms={:.3} ttft_fresh_ms={:.3} \
                 prefill_submit_reuse_ms={:.3} prefill_submit_fresh_ms={:.3} \
                 turn2_reuse_wall_s={:.4} turn2_fresh_wall_s={:.4} \
                 digest_reuse={} digest_fresh={}",
                arm.eos_read(),
                reply1.len(),
                t1.out.eos,
                t1.out.stats.overrun,
                prepared.action,
                prepared.reused,
                prepared.suffix.len(),
                full2.len(),
                full2.len(),
                reply2_reuse.len(),
                reply2_fresh.len(),
                same,
                common_len,
                t2.ttft_ms,
                tf.ttft_ms,
                t2.out.stats.prefill_ns as f64 / 1e6,
                tf.out.stats.prefill_ns as f64 / 1e6,
                t2.wall.as_secs_f64(),
                tf.wall.as_secs_f64(),
                sha256_hex(&reply2_reuse),
                sha256_hex(&reply2_fresh),
            );
        }
        "concurrent" if dspark_setup.is_some() => {
            let (draft_dir, dcfg) = dspark_setup.as_ref().expect("checked");
            let n = args.sequences.clamp(1, PROMPTS.len());
            let prompts: Vec<Vec<u32>> = PROMPTS[..n]
                .iter()
                .map(|p| encode(&user_turn(p)))
                .collect::<Result<_>>()?;
            run_dspark_concurrent(
                &args, &model, &config, &device, dtype, &eos_ids, &prompts, draft_dir, dcfg,
                &common,
            )?;
        }
        "concurrent" => {
            let n = args.sequences.clamp(1, PROMPTS.len());
            let prompts: Vec<Vec<u32>> = PROMPTS[..n]
                .iter()
                .map(|p| encode(&user_turn(p)))
                .collect::<Result<_>>()?;

            // ---- SOLO: each sequence alone -------------------------------
            let mut solo: Vec<Vec<u32>> = Vec::with_capacity(n);
            let mut solo_wall = 0f64;
            let mut solo_rates = Vec::with_capacity(n);
            for p in &prompts {
                let mut s = Session::new(&model, Cache::new(true, dtype, &config, &device)?);
                let mut arm = Arm::new(&args.readback, &eos_ids, &device, capacity)?;
                let t = decode(&mut s, &mut arm, p, &opts)?;
                solo_wall += t.wall.as_secs_f64();
                solo_rates.push(t.out.ids.len() as f64 / t.wall.as_secs_f64());
                solo.push(t.out.ids);
            }

            // ---- INTERLEAVED: round-robin visits of `chunk` tokens ---------
            let mut sessions: Vec<Session<'_, Model>> = (0..n)
                .map(|_| {
                    Ok(Session::new(
                        &model,
                        Cache::new(true, dtype, &config, &device)?,
                    ))
                })
                .collect::<Result<_>>()?;
            let mut arms: Vec<Arm> = (0..n)
                .map(|_| Arm::new(&args.readback, &eos_ids, &device, capacity))
                .collect::<Result<_>>()?;
            let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
            let mut done = vec![false; n];
            let mut visits = 0usize;
            let mut resumes = 0usize;
            let t = std::time::Instant::now();
            while done.iter().any(|d| !d) {
                for i in 0..n {
                    if done[i] {
                        continue;
                    }
                    let full: Vec<u32> = [prompts[i].clone(), streams[i].clone()].concat();
                    let prepared = sessions[i].prepare(&full)?;
                    if prepared.action == ReuseAction::Resume {
                        resumes += 1;
                    }
                    let want = args.chunk.min(args.max_tokens - streams[i].len());
                    let visit_opts = DecodeOptions {
                        max_tokens: want,
                        ..opts
                    };
                    let out = decode(
                        &mut sessions[i],
                        &mut arms[i],
                        &prepared.suffix,
                        &visit_opts,
                    )?
                    .out;
                    streams[i].extend_from_slice(&out.ids);
                    visits += 1;
                    if out.eos || streams[i].len() >= args.max_tokens {
                        done[i] = true;
                    }
                }
            }
            let wall = t.elapsed().as_secs_f64();
            let total: usize = streams.iter().map(|s| s.len()).sum();
            let identical: Vec<bool> = streams
                .iter()
                .zip(solo.iter())
                .map(|(a, b)| a == b)
                .collect();
            for i in 0..n {
                eprintln!(
                    "[seq {i}: solo {} tokens at {:.1} tok/s; interleaved {} tokens; {}]",
                    solo[i].len(),
                    solo_rates[i],
                    streams[i].len(),
                    if identical[i] { "IDENTICAL" } else { "DIFFERS" }
                );
            }
            eprintln!(
                "[{n} sequences interleaved in {chunk}-token visits: {total} tokens in {wall:.3} s \
                 = {:.1} tok/s aggregate; solo runs {:.1} tok/s mean, {:.3} s summed]",
                total as f64 / wall,
                solo_rates.iter().sum::<f64>() / n as f64,
                solo_wall,
                chunk = args.chunk
            );
            println!(
                "RESULT mode=concurrent {common} eos_read={} sequences={n} chunk={} visits={} \
                 resumes={} total_tokens={} interleaved_wall_s={:.4} aggregate_tok_s={:.4} \
                 solo_wall_s={:.4} solo_mean_tok_s={:.4} identical={} per_seq_identical={:?} \
                 per_seq_tokens={:?} per_seq_solo_tok_s={:?}",
                arms[0].eos_read(),
                args.chunk,
                visits,
                resumes,
                total,
                wall,
                total as f64 / wall,
                solo_wall,
                solo_rates.iter().sum::<f64>() / n as f64,
                identical.iter().all(|&b| b),
                identical,
                streams.iter().map(|s| s.len()).collect::<Vec<_>>(),
                solo_rates
                    .iter()
                    .map(|r| format!("{r:.1}"))
                    .collect::<Vec<_>>(),
            );
        }
        other => anyhow::bail!("--mode must be `reuse` or `concurrent`, got `{other}`"),
    }
    Ok(())
}

/// The `--mode concurrent --dspark` arms: every prompt solo through the serial
/// speculative loop, and all of them through the pipelined driver; each
/// pipelined stream is compared byte for byte with its solo stream.
#[allow(clippy::too_many_arguments)]
fn run_dspark_concurrent(
    args: &Args,
    model: &Model,
    config: &lloom::models::lfm2::Config,
    device: &Device,
    dtype: DType,
    eos_ids: &[u32],
    prompts: &[Vec<u32>],
    draft_dir: &std::path::Path,
    dcfg: &DSparkConfig,
    common: &str,
) -> Result<()> {
    let n = prompts.len();
    let k = dcfg.max_draft();
    let draft_file = draft_dir.join("model.safetensors");
    if !draft_file.exists() {
        anyhow::bail!("{} does not exist", draft_file.display());
    }
    let longest = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
    let max_pos = longest + args.max_tokens + dcfg.block_size + 64;
    let dvb = unsafe { VarBuilder::from_mmaped_safetensors(&[draft_file], DType::F16, device)? };
    let drafter =
        DSparkDrafter::new(dcfg.clone(), max_pos, dvb).context("constructing DSpark drafter")?;
    // One drafter with `n` slots serves both arms: `begin` resets a slot, so
    // the solo arm uses slot 0 for every prompt and the pipelined arm slot `i`
    // for prompt `i`.
    let mut spec = DSparkSpeculator::new(
        drafter,
        model.draft_embedding().clone(),
        model.lm_head_linear().clone(),
        n,
    );
    eprintln!(
        "[dspark: drafter loaded, {} layers, K={k} (verify window {} positions), {n} slots]",
        dcfg.num_hidden_layers,
        k + 1
    );
    let opts = DecodeOptions {
        max_tokens: args.max_tokens,
        ..DecodeOptions::new(args.max_tokens)
    };
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);

    // Warm the drafter's pipelines off every clock below: one short solo run.
    {
        let mut s = Session::new(model, Cache::new(true, dtype, config, device)?);
        let mut rb = Blocking::new(eos_ids, device)?;
        let warm = DecodeOptions::new(16);
        let _ = s.decode_speculative(
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompts[0],
            &warm,
            eos_ids,
            &mut |_| Ok(()),
        )?;
        device.synchronize()?;
    }

    let run_solo = |spec: &mut DSparkSpeculator| -> Result<SpecArm> {
        let rb0 = readback_counts();
        let _ = profile_take(device)?;
        let t = std::time::Instant::now();
        let mut outs = Vec::with_capacity(n);
        for p in prompts {
            let mut s = Session::new(model, Cache::new(true, dtype, config, device)?);
            let mut rb = Blocking::new(eos_ids, device)?;
            let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
            outs.push(s.decode_speculative(
                spec,
                &mut rb,
                &mut sampler,
                p,
                &opts,
                eos_ids,
                &mut |_| Ok(()),
            )?);
        }
        device.synchronize()?;
        let wall_s = t.elapsed().as_secs_f64();
        let (gpu_busy_s, gpu_busy_sum_s, dispatches, command_buffers) = profile_take(device)?;
        let rb1 = readback_counts();
        Ok(SpecArm {
            name: "solo",
            streams: outs.iter().map(|o| o.ids.clone()).collect(),
            outs,
            wall_s,
            gpu_busy_s,
            gpu_busy_sum_s,
            dispatches,
            command_buffers,
            readbacks: (rb1.0 - rb0.0, rb1.1 - rb0.1, rb1.2 - rb0.2),
        })
    };
    let run_pipelined = |spec: &mut DSparkSpeculator| -> Result<SpecArm> {
        let mut sessions: Vec<Session<'_, Model>> = (0..n)
            .map(|_| {
                Ok(Session::new(
                    model,
                    Cache::new(true, dtype, config, device)?,
                ))
            })
            .collect::<Result<_>>()?;
        let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
        let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
        let rb0 = readback_counts();
        let _ = profile_take(device)?;
        let t = std::time::Instant::now();
        let outs = Session::decode_speculative_pipelined(
            &mut sessions,
            spec,
            &prompt_refs,
            &opts,
            eos_ids,
            device,
            &mut |i, ids| {
                streams[i].extend_from_slice(ids);
                Ok(())
            },
        )?;
        device.synchronize()?;
        let wall_s = t.elapsed().as_secs_f64();
        let (gpu_busy_s, gpu_busy_sum_s, dispatches, command_buffers) = profile_take(device)?;
        let rb1 = readback_counts();
        Ok(SpecArm {
            name: "pipelined",
            streams,
            outs,
            wall_s,
            gpu_busy_s,
            gpu_busy_sum_s,
            dispatches,
            command_buffers,
            readbacks: (rb1.0 - rb0.0, rb1.1 - rb0.1, rb1.2 - rb0.2),
        })
    };

    let (solo, pipe) = match args.first.as_str() {
        "solo" => {
            let s = run_solo(&mut spec)?;
            let p = run_pipelined(&mut spec)?;
            (s, p)
        }
        "pipelined" => {
            let p = run_pipelined(&mut spec)?;
            let s = run_solo(&mut spec)?;
            (s, p)
        }
        other => anyhow::bail!("--first must be `solo` or `pipelined`, got `{other}`"),
    };
    for (i, o) in pipe.outs.iter().enumerate() {
        if o.ids != pipe.streams[i] {
            anyhow::bail!("sequence {i}: the pipelined callback stream differs from its ids");
        }
    }

    let identical: Vec<bool> = (0..n).map(|i| pipe.streams[i] == solo.streams[i]).collect();
    let solo_rates: Vec<f64> = solo
        .outs
        .iter()
        .map(|o| {
            // Per-sequence solo rate from its own rounds' wall: prefill excluded,
            // as `lfm2-chat`'s steady figure is.
            let round_s: f64 = o
                .stats
                .rounds
                .iter()
                .map(|r| r.timing.wall_ns() as f64 / 1e9)
                .sum();
            o.ids.len().saturating_sub(1) as f64 / round_s
        })
        .collect();
    for i in 0..n {
        eprintln!(
            "[seq {i}: solo {} tokens in {} rounds ({:.1} tok/s over its rounds); pipelined {} tokens in \
             {} rounds; {}]",
            solo.streams[i].len(),
            solo.outs[i].stats.rounds.len(),
            solo_rates[i],
            pipe.streams[i].len(),
            pipe.outs[i].stats.rounds.len(),
            if identical[i] { "IDENTICAL" } else { "DIFFERS" }
        );
    }
    for arm in [&solo, &pipe] {
        eprintln!(
            "[{}: {} tokens, {} rounds, {:.3} s wall = {:.1} tok/s aggregate; GPU busy union {:.3} s \
             (idle {:.3} s = {:.1} %), busy sum {:.3} s over {} command buffers; per round ms (median): \
             draft {:.3} enqueue {:.3} wait {:.3} verify {:.3} observe {:.3} round {:.3}; \
             readbacks tail-wait fast/blit {}/{} staged {}]",
            arm.name,
            arm.tokens(),
            arm.rounds(),
            arm.wall_s,
            arm.tokens() as f64 / arm.wall_s,
            arm.gpu_busy_s,
            arm.wall_s - arm.gpu_busy_s,
            if arm.gpu_busy_s > 0.0 {
                100.0 * (arm.wall_s - arm.gpu_busy_s) / arm.wall_s
            } else {
                f64::NAN
            },
            arm.gpu_busy_sum_s,
            arm.command_buffers,
            arm.per_round_ms(|r| r.timing.draft_ns),
            arm.per_round_ms(|r| r.timing.verify_enqueue_ns),
            arm.per_round_ms(|r| r.timing.verify_wait_ns),
            arm.per_round_ms(|r| r.timing.verify_ns),
            arm.per_round_ms(|r| r.timing.observe_ns),
            arm.per_round_ms(|r| r.timing.wall_ns()),
            arm.readbacks.0,
            arm.readbacks.1,
            arm.readbacks.2,
        );
    }
    if let Some(path) = &args.timeline {
        let mut w = String::from(
            "arm,seq,round,start_ms,draft_ms,enqueue_ms,wait_ms,observe_ms,accepted\n",
        );
        for arm in [&solo, &pipe] {
            for (i, o) in arm.outs.iter().enumerate() {
                for (r, round) in o.stats.rounds.iter().enumerate() {
                    w.push_str(&format!(
                        "{},{i},{r},{:.4},{:.4},{:.4},{:.4},{:.4},{}\n",
                        arm.name,
                        round.start_ns as f64 / 1e6,
                        round.timing.draft_ns as f64 / 1e6,
                        round.timing.verify_enqueue_ns as f64 / 1e6,
                        round.timing.verify_wait_ns as f64 / 1e6,
                        round.timing.observe_ns as f64 / 1e6,
                        round.accepted
                    ));
                }
            }
        }
        std::fs::write(path, w).with_context(|| format!("writing {}", path.display()))?;
    }
    println!(
        "RESULT mode=concurrent dspark=true first={} {common} k={k} sequences={n} \
         solo_tokens={} solo_rounds={} solo_wall_s={:.4} solo_aggregate_tok_s={:.4} \
         solo_gpu_busy_s={:.4} solo_gpu_busy_sum_s={:.4} solo_round_ms_median={:.4} \
         solo_wait_ms_median={:.4} \
         pipelined_tokens={} pipelined_rounds={} pipelined_wall_s={:.4} \
         pipelined_aggregate_tok_s={:.4} pipelined_gpu_busy_s={:.4} pipelined_gpu_busy_sum_s={:.4} \
         pipelined_round_ms_median={:.4} pipelined_wait_ms_median={:.4} \
         pipelined_enqueue_ms_median={:.4} pipelined_draft_ms_median={:.4} \
         pipelined_observe_ms_median={:.4} \
         speedup={:.4} identical={} per_seq_identical={:?} per_seq_tokens={:?} \
         per_seq_solo_tok_s={:?} solo_readbacks={:?} pipelined_readbacks={:?} \
         solo_dispatches={} pipelined_dispatches={}",
        args.first,
        solo.tokens(),
        solo.rounds(),
        solo.wall_s,
        solo.tokens() as f64 / solo.wall_s,
        solo.gpu_busy_s,
        solo.gpu_busy_sum_s,
        solo.per_round_ms(|r| r.timing.wall_ns()),
        solo.per_round_ms(|r| r.timing.verify_ns),
        pipe.tokens(),
        pipe.rounds(),
        pipe.wall_s,
        pipe.tokens() as f64 / pipe.wall_s,
        pipe.gpu_busy_s,
        pipe.gpu_busy_sum_s,
        pipe.per_round_ms(|r| r.timing.wall_ns()),
        pipe.per_round_ms(|r| r.timing.verify_wait_ns),
        pipe.per_round_ms(|r| r.timing.verify_enqueue_ns),
        pipe.per_round_ms(|r| r.timing.draft_ns),
        pipe.per_round_ms(|r| r.timing.observe_ns),
        (pipe.tokens() as f64 / pipe.wall_s) / (solo.tokens() as f64 / solo.wall_s),
        identical.iter().all(|&b| b),
        identical,
        pipe.streams.iter().map(|s| s.len()).collect::<Vec<_>>(),
        solo_rates
            .iter()
            .map(|r| format!("{r:.1}"))
            .collect::<Vec<_>>(),
        solo.readbacks,
        pipe.readbacks,
        solo.dispatches,
        pipe.dispatches,
    );
    Ok(())
}
