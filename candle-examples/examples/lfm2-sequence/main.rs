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

    /// In-flight cap for the host-on-drain arm; 0 is off.
    #[arg(long, default_value_t = 0)]
    max_in_flight: usize,

    /// Sequences to interleave under `--mode concurrent` (at most 4).
    #[arg(long, default_value_t = 2)]
    sequences: usize,

    /// Tokens per visit under `--mode concurrent`.
    #[arg(long, default_value_t = 16)]
    chunk: usize,

    /// Force CPU.
    #[arg(long)]
    cpu: bool,
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
    match args.cache.as_str() {
        "crown" => {
            config.kv_append = KvAppend::Cat;
            config.conv_state = ConvState::Shuffle;
        }
        "rewindable" => {
            config.kv_append = KvAppend::InPlace;
            config.conv_state = ConvState::RotatingRing { k: args.ring_k };
        }
        other => anyhow::bail!("--cache must be `crown` or `rewindable`, got `{other}`"),
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
        max_in_flight: args.max_in_flight,
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
         max_in_flight={} max_tokens={} dtype={:?}",
        args.cache,
        args.ring_k,
        config.conv_prefill_ctx,
        args.readback,
        lookahead,
        budget.arm(),
        args.max_in_flight,
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
