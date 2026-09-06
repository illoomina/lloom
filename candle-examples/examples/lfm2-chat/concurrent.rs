//! The multi-sequence modes of `lfm2-chat` (absorbed from the former
//! `lfm2-sequence`, issue #677 + #685 + #148):
//!
//! * `--mode reuse` — a two-turn conversation decoded two ways (prefix reuse
//!   through `Session::prepare` vs a fresh prefill), reporting whether the
//!   streams agree and turn 2's TTFT on each. `DESIGN.md` §13.2f.
//! * `--mode concurrent` — `--sequences` independent conversations interleaved
//!   on one model in `--chunk`-token visits (non-speculative, time-sliced), or
//!   — with `--dspark` — all of them through `Session::decode_speculative_pipelined`
//!   (#685), which overlaps one sequence's draft/observe with the others'
//!   verifies. Reports per-sequence identity against the solo stream and the
//!   aggregate token rate.
//!
//! The concurrent modes print **no token text** (they are a throughput/identity
//! surface, not a chat) and instead repaint a **live raw-stats block** on stderr
//! as they run — aggregate tokens + tok/s and per-sequence progress — so a long
//! run shows it is alive and previews the numbers before the RESULT line.
//!
//! The predefined prompt set is DELIBERATELY VARIED (four different shapes): a
//! concurrent run over identical prompts would share a prefix and skip the
//! prefill work the harness exists to exercise.

use anyhow::{Context, Result};
use candle::{DType, Device};
use lloom::generation::poll::PollMechanism;
use lloom::generation::{
    Blocking, CommitBudget, EosRead, LogitsProcessor, ReadbackStrategy, Ring, Sampling,
};
use lloom::models::lfm2::{Cache, Config, Model};
use lloom::models::lfm2_dspark::{DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, ReuseAction, Session};
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

/// `design/workloads/prompts/`, the #549 corpus: a multi-turn pair and four
/// single prompts of DIFFERENT shapes — varied so the concurrent run exercises
/// prefill rather than sharing one prefix.
const TURN_1: &str = "I am designing a cache for a single-threaded interpreter. Entries are immutable once written and the working set is about eight times the cache. What eviction policy should I start with, and why not LRU?";
const TURN_2: &str = "Suppose I measure and find the hit rate is 40% lower than you predicted. What are the two most likely causes, and which measurement separates them?";
const PROMPTS: [&str; 4] = [
    "Name the three branches of the United States federal government.",
    "Count from one to forty, writing each number as a word on its own line, with no other text.",
    "Here is a Rust function that is missing its body:\n\nfn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32> {\n    // Merge two already-sorted slices into one sorted Vec.\n    // Do not allocate more than once.\n}\n\nWrite the body. Explain your choice of capacity for the allocation, and say what happens when one slice is empty.",
    "Extract every date, monetary amount, and party name from the paragraph below and return them as a JSON object with the keys \"dates\", \"amounts\", and \"parties\". Return only the JSON.\n\nOn 14 March 2019, Halloran Freight Ltd. agreed to pay Baswick Terminals the sum of 48,250 GBP in settlement of demurrage accrued between 2 January 2019 and 27 February 2019. A further 6,900 GBP was withheld pending inspection, to be released on or before 30 April 2019 unless Baswick Terminals lodged an objection.",
];

/// What `run` needs from `main` after it has built the model, config and eos set.
pub struct Ctx<'a> {
    pub model: &'a Model,
    pub config: &'a Config,
    pub device: &'a Device,
    pub dtype: DType,
    pub eos_ids: &'a [u32],
    /// `mode` ("reuse" | "concurrent"), and the knobs the modes read.
    pub mode: &'a str,
    pub max_tokens: usize,
    pub readback: &'a str,
    pub lookahead: usize,
    pub commit_budget: &'a str,
    pub sequences: usize,
    pub chunk: usize,
    /// `--dspark` — drive the concurrent sequences through the pipelined
    /// speculative driver instead of the time-sliced non-spec interleave.
    pub dspark: bool,
    /// The drafter dir + config, resolved by `main` when `--dspark` is set.
    pub dspark_setup: Option<(PathBuf, DSparkConfig)>,
    pub first: &'a str,
    pub timeline: Option<PathBuf>,
    /// Raw tokenizer encode (NO chat wrapping), passed as a closure so `main`
    /// owns the tokenizer. The modes wrap with `user_turn_text` themselves, and
    /// the reuse mode also encodes pre-wrapped text raw.
    pub encode: &'a dyn Fn(&str) -> Result<Vec<u32>>,
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
    let t = Instant::now();
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

/// A live-updating raw-stats block on stderr, repainted in place. It moves the
/// cursor up over its own previous height and rewrites, so a long concurrent run
/// shows aggregate + per-sequence progress ticking rather than a wall of text.
/// No token text is printed in the concurrent modes; this IS the "it's alive,
/// here's a preview of the numbers" surface.
struct LiveStats {
    n: usize,
    targets: Vec<usize>,
    tokens: Vec<usize>,
    /// Per-sequence accepted-per-round mean, `dspark` only (else `f64::NAN`).
    accept: Vec<f64>,
    rounds: Vec<usize>,
    done: Vec<bool>,
    start: Instant,
    /// Lines the last paint wrote, so the next paint knows how far up to move.
    painted: usize,
    dspark: bool,
    label: &'static str,
    enabled: bool,
}

impl LiveStats {
    fn new(label: &'static str, targets: Vec<usize>, dspark: bool) -> Self {
        let n = targets.len();
        // Only paint when stderr is a terminal; under a pipe (a measurement
        // capture) the repaint escapes are noise, so fall back to nothing —
        // the RESULT line is the artifact there.
        let enabled = std::io::stderr().is_terminal();
        LiveStats {
            n,
            targets,
            tokens: vec![0; n],
            accept: vec![f64::NAN; n],
            rounds: vec![0; n],
            done: vec![false; n],
            start: Instant::now(),
            painted: 0,
            dspark,
            label,
            enabled,
        }
    }

    fn set(&mut self, i: usize, tokens: usize, rounds: usize, done: bool) {
        self.tokens[i] = tokens;
        self.rounds[i] = rounds;
        if rounds > 0 {
            self.accept[i] = tokens as f64 / rounds as f64;
        }
        self.done[i] = done;
    }

    fn paint(&mut self) {
        if !self.enabled {
            return;
        }
        let mut err = std::io::stderr().lock();
        // Move up over the previous block and clear each line as we go.
        if self.painted > 0 {
            let _ = write!(err, "\x1b[{}A", self.painted);
        }
        let elapsed = self.start.elapsed().as_secs_f64().max(1e-9);
        let total: usize = self.tokens.iter().sum();
        let agg = total as f64 / elapsed;
        let ndone = self.done.iter().filter(|&&d| d).count();
        let mut lines = 0usize;
        let _ = write!(err, "\x1b[2K");
        let _ = writeln!(
            err,
            "\x1b[1;96m[{} live]\x1b[0m {}/{} seqs done | {} tok in {:.1}s = \x1b[1;96m{:.1}\x1b[0m tok/s aggregate",
            self.label, ndone, self.n, total, elapsed, agg
        );
        lines += 1;
        for i in 0..self.n {
            let _ = write!(err, "\x1b[2K");
            let bar = if self.done[i] { "done" } else { "····" };
            if self.dspark {
                let _ = writeln!(
                    err,
                    "  seq {i}: {:>4}/{:<4} tok  {} rounds  accept {:.2}/round  [{bar}]",
                    self.tokens[i],
                    self.targets[i],
                    self.rounds[i],
                    self.accept[i],
                );
            } else {
                let _ = writeln!(
                    err,
                    "  seq {i}: {:>4}/{:<4} tok  [{bar}]",
                    self.tokens[i], self.targets[i],
                );
            }
            lines += 1;
        }
        let _ = err.flush();
        self.painted = lines;
    }

    /// Leave the final block on screen (a plain newline so the RESULT line that
    /// follows does not overwrite it).
    fn finish(&mut self) {
        if self.enabled {
            let mut err = std::io::stderr().lock();
            let _ = writeln!(err);
            let _ = err.flush();
        }
    }
}

/// Entry point from `main` once `mode != "single"`.
pub fn run(ctx: Ctx<'_>) -> Result<()> {
    let budget = CommitBudget::from_str(ctx.commit_budget).map_err(anyhow::Error::msg)?;
    let lookahead = if ctx.readback == "blocking" {
        0
    } else {
        ctx.lookahead
    };
    let opts = DecodeOptions {
        max_tokens: ctx.max_tokens,
        budget,
        lookahead,
        sample_split: false,
    };
    let capacity = 8 * ctx.max_tokens + 8 * (budget.period() + 2);

    let common = format!(
        "readback={} lookahead={} commit_budget={} in_flight_steps={} max_tokens={} dtype={:?}",
        ctx.readback,
        lookahead,
        budget.arm(),
        lloom::nn::decode::IN_FLIGHT_STEPS,
        ctx.max_tokens,
        ctx.dtype
    );

    match ctx.mode {
        "reuse" => run_reuse(&ctx, &opts, capacity, &common),
        "concurrent" if ctx.dspark => run_dspark_concurrent(&ctx, &common),
        "concurrent" => run_concurrent(&ctx, &opts, capacity, &common),
        other => anyhow::bail!("--mode must be `single`, `reuse` or `concurrent`, got `{other}`"),
    }
}

/// `--mode reuse`: turn 1, then turn 2 through `prepare` (reuse) vs a fresh
/// prefill, and whether the two turn-2 streams agree.
fn run_reuse(ctx: &Ctx<'_>, opts: &DecodeOptions, capacity: usize, common: &str) -> Result<()> {
    let (model, config, device, dtype) = (ctx.model, ctx.config, ctx.device, ctx.dtype);
    let turn1 = (ctx.encode)(&user_turn_text(TURN_1))?;
    let mut session = Session::new(model, Cache::new(true, dtype, config, device)?);
    let mut arm = Arm::new(ctx.readback, ctx.eos_ids, device, capacity)?;
    let t1 = decode(&mut session, &mut arm, &turn1, opts)?;
    let reply1 = t1.out.ids.clone();
    eprintln!(
        "[turn 1: {} tokens, eos={}, overrun={}, {:.1} tok/s]",
        reply1.len(),
        t1.out.eos,
        t1.out.stats.overrun,
        reply1.len() as f64 / t1.wall.as_secs_f64()
    );
    // The whole conversation at the id level: the cache holds ids, so build the
    // turn-2 request from ids (a re-tokenized reply need not reproduce the
    // sampled ids). The turn-2 text is pre-wrapped and encoded raw.
    let mut full2 = turn1.clone();
    full2.extend_from_slice(&reply1);
    let turn2_text = if t1.out.eos {
        format!("<|im_end|>\n{}", user_turn_text(TURN_2))
    } else {
        user_turn_text(TURN_2)
    };
    full2.extend((ctx.encode)(&turn2_text)?);
    let overrun_before = session.overrun();
    let prepared = session.prepare(&full2)?;
    let t2 = decode(&mut session, &mut arm, &prepared.suffix, opts)?;
    let reply2_reuse = t2.out.ids.clone();

    let mut fresh = Session::new(model, Cache::new(true, dtype, config, device)?);
    let mut arm_f = Arm::new(ctx.readback, ctx.eos_ids, device, capacity)?;
    let tf = decode(&mut fresh, &mut arm_f, &full2, opts)?;
    let reply2_fresh = tf.out.ids.clone();

    let same = reply2_reuse == reply2_fresh;
    let common_len = reply2_reuse
        .iter()
        .zip(reply2_fresh.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "[turn 2 reuse: action={:?} reused={} prefilled={} of {} full, overrun_before={} eos={} {} tokens, TTFT {:.1} ms]",
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
        "RESULT mode=reuse {common} eos_read={} turn1_tokens={} reuse_action={:?} reused={} \
         prefilled_reuse={} prefilled_fresh={} turn2_tokens_reuse={} turn2_tokens_fresh={} \
         identical={} common_prefix={} ttft_reuse_ms={:.3} ttft_fresh_ms={:.3}",
        arm.eos_read(),
        reply1.len(),
        prepared.action,
        prepared.reused,
        prepared.suffix.len(),
        full2.len(),
        reply2_reuse.len(),
        reply2_fresh.len(),
        same,
        common_len,
        t2.ttft_ms,
        tf.ttft_ms,
    );
    Ok(())
}

/// `--mode concurrent` (non-speculative): N sequences interleaved in `chunk`-token
/// visits, live stats repainting per visit, then the RESULT line.
fn run_concurrent(ctx: &Ctx<'_>, opts: &DecodeOptions, capacity: usize, common: &str) -> Result<()> {
    let (model, config, device, dtype) = (ctx.model, ctx.config, ctx.device, ctx.dtype);
    let n = ctx.sequences.clamp(1, PROMPTS.len());
    let prompts: Vec<Vec<u32>> = PROMPTS[..n]
        .iter()
        .map(|p| (ctx.encode)(&user_turn_text(p)))
        .collect::<Result<_>>()?;

    // SOLO: each sequence alone (the identity + rate baseline).
    let mut solo: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut solo_wall = 0f64;
    let mut solo_rates = Vec::with_capacity(n);
    for p in &prompts {
        let mut s = Session::new(model, Cache::new(true, dtype, config, device)?);
        let mut arm = Arm::new(ctx.readback, ctx.eos_ids, device, capacity)?;
        let t = decode(&mut s, &mut arm, p, opts)?;
        solo_wall += t.wall.as_secs_f64();
        solo_rates.push(t.out.ids.len() as f64 / t.wall.as_secs_f64());
        solo.push(t.out.ids);
    }

    // INTERLEAVED: round-robin visits, live-repainting the block per visit.
    let mut sessions: Vec<Session<'_, Model>> = (0..n)
        .map(|_| Ok(Session::new(model, Cache::new(true, dtype, config, device)?)))
        .collect::<Result<_>>()?;
    let mut arms: Vec<Arm> = (0..n)
        .map(|_| Arm::new(ctx.readback, ctx.eos_ids, device, capacity))
        .collect::<Result<_>>()?;
    let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut done = vec![false; n];
    let mut visits = 0usize;
    let mut resumes = 0usize;
    let mut live = LiveStats::new("concurrent", vec![ctx.max_tokens; n], false);
    live.paint();
    let t = Instant::now();
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
            let want = ctx.chunk.min(ctx.max_tokens - streams[i].len());
            let visit_opts = DecodeOptions {
                max_tokens: want,
                ..*opts
            };
            let out = decode(&mut sessions[i], &mut arms[i], &prepared.suffix, &visit_opts)?.out;
            streams[i].extend_from_slice(&out.ids);
            visits += 1;
            if out.eos || streams[i].len() >= ctx.max_tokens {
                done[i] = true;
            }
            live.set(i, streams[i].len(), 0, done[i]);
            live.paint();
        }
    }
    live.finish();
    let wall = t.elapsed().as_secs_f64();
    let total: usize = streams.iter().map(|s| s.len()).sum();
    let identical: Vec<bool> = streams.iter().zip(solo.iter()).map(|(a, b)| a == b).collect();
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
        "[{n} sequences interleaved in {}-token visits: {total} tokens in {wall:.3} s = {:.1} tok/s aggregate; solo {:.1} tok/s mean]",
        ctx.chunk,
        total as f64 / wall,
        solo_rates.iter().sum::<f64>() / n as f64,
    );
    println!(
        "RESULT mode=concurrent {common} eos_read={} sequences={n} chunk={} visits={} resumes={} \
         total_tokens={} interleaved_wall_s={:.4} aggregate_tok_s={:.4} solo_wall_s={:.4} \
         solo_mean_tok_s={:.4} identical={} per_seq_identical={:?} per_seq_tokens={:?}",
        arms[0].eos_read(),
        ctx.chunk,
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
    );
    Ok(())
}

/// One arm of the `--dspark` concurrent run.
struct SpecArm {
    name: &'static str,
    streams: Vec<Vec<u32>>,
    outs: Vec<DecodeOut>,
    wall_s: f64,
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

/// `--mode concurrent --dspark`: every prompt solo through the serial
/// speculative loop, and all through the pipelined driver (#685), with the live
/// stats repainting on the pipelined arm's per-sequence callback.
fn run_dspark_concurrent(ctx: &Ctx<'_>, common: &str) -> Result<()> {
    let (model, config, device, dtype) = (ctx.model, ctx.config, ctx.device, ctx.dtype);
    let (draft_dir, dcfg) = ctx
        .dspark_setup
        .as_ref()
        .context("--dspark set but no drafter resolved")?;
    let n = ctx.sequences.clamp(1, PROMPTS.len());
    let prompts: Vec<Vec<u32>> = PROMPTS[..n]
        .iter()
        .map(|p| (ctx.encode)(&user_turn_text(p)))
        .collect::<Result<_>>()?;

    let k = dcfg.max_draft();
    let draft_file = draft_dir.join("model.safetensors");
    if !draft_file.exists() {
        anyhow::bail!("{} does not exist", draft_file.display());
    }
    let longest = prompts.iter().map(|p| p.len()).max().unwrap_or(0);
    let max_pos = longest + ctx.max_tokens + dcfg.block_size + 64;
    let dvb =
        unsafe { lloom::nn::VarBuilder::from_mmaped_safetensors(&[draft_file], DType::F16, device)? };
    let drafter =
        DSparkDrafter::new(dcfg.clone(), max_pos, dvb).context("constructing DSpark drafter")?;
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
        max_tokens: ctx.max_tokens,
        ..DecodeOptions::new(ctx.max_tokens)
    };
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);

    // Warm the drafter's pipelines off every clock below.
    {
        let mut s = Session::new(model, Cache::new(true, dtype, config, device)?);
        let mut rb = Blocking::new(ctx.eos_ids, device)?;
        let warm = DecodeOptions::new(16);
        let _ = s.decode_speculative(
            &mut spec,
            &mut rb,
            &mut sampler,
            &prompts[0],
            &warm,
            ctx.eos_ids,
            &mut |_| Ok(()),
        )?;
        device.synchronize()?;
    }

    let run_solo = |spec: &mut DSparkSpeculator| -> Result<SpecArm> {
        let t = Instant::now();
        let mut outs = Vec::with_capacity(n);
        let mut live = LiveStats::new("solo", vec![ctx.max_tokens; n], true);
        live.paint();
        for (i, p) in prompts.iter().enumerate() {
            let mut s = Session::new(model, Cache::new(true, dtype, config, device)?);
            let mut rb = Blocking::new(ctx.eos_ids, device)?;
            let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
            let out = s.decode_speculative(
                spec,
                &mut rb,
                &mut sampler,
                p,
                &opts,
                ctx.eos_ids,
                &mut |_| Ok(()),
            )?;
            live.set(i, out.ids.len(), out.stats.rounds.len(), true);
            live.paint();
            outs.push(out);
        }
        live.finish();
        device.synchronize()?;
        let wall_s = t.elapsed().as_secs_f64();
        Ok(SpecArm {
            name: "solo",
            streams: outs.iter().map(|o| o.ids.clone()).collect(),
            outs,
            wall_s,
        })
    };
    let run_pipelined = |spec: &mut DSparkSpeculator| -> Result<SpecArm> {
        let mut sessions: Vec<Session<'_, Model>> = (0..n)
            .map(|_| Ok(Session::new(model, Cache::new(true, dtype, config, device)?)))
            .collect::<Result<_>>()?;
        let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
        let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut live = LiveStats::new("pipelined", vec![ctx.max_tokens; n], true);
        live.paint();
        let t = Instant::now();
        let outs = Session::decode_speculative_pipelined(
            &mut sessions,
            spec,
            &prompt_refs,
            &opts,
            ctx.eos_ids,
            device,
            &mut |i, ids| {
                streams[i].extend_from_slice(ids);
                // rounds unknown mid-flight here; show tokens ticking (accept
                // updates from the final outs below). The paint is the liveness
                // signal — aggregate tok/s climbs as the callbacks arrive.
                live.set(i, streams[i].len(), live.rounds[i], false);
                live.paint();
                Ok(())
            },
        )?;
        for (i, o) in outs.iter().enumerate() {
            live.set(i, streams[i].len(), o.stats.rounds.len(), true);
        }
        live.paint();
        live.finish();
        device.synchronize()?;
        let wall_s = t.elapsed().as_secs_f64();
        Ok(SpecArm {
            name: "pipelined",
            streams,
            outs,
            wall_s,
        })
    };

    let (solo, pipe) = match ctx.first {
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
    for i in 0..n {
        eprintln!(
            "[seq {i}: solo {} tokens in {} rounds; pipelined {} tokens in {} rounds; {}]",
            solo.streams[i].len(),
            solo.outs[i].stats.rounds.len(),
            pipe.streams[i].len(),
            pipe.outs[i].stats.rounds.len(),
            if identical[i] { "IDENTICAL" } else { "DIFFERS" }
        );
    }
    for arm in [&solo, &pipe] {
        eprintln!(
            "[{}: {} tokens, {} rounds, {:.3} s wall = {:.1} tok/s aggregate; per round ms (median): \
             draft {:.3} verify {:.3} observe {:.3} round {:.3}]",
            arm.name,
            arm.tokens(),
            arm.rounds(),
            arm.wall_s,
            arm.tokens() as f64 / arm.wall_s,
            arm.per_round_ms(|r| r.timing.draft_ns),
            arm.per_round_ms(|r| r.timing.verify_ns),
            arm.per_round_ms(|r| r.timing.observe_ns),
            arm.per_round_ms(|r| r.timing.wall_ns()),
        );
    }
    if let Some(path) = &ctx.timeline {
        let mut w =
            String::from("arm,seq,round,start_ms,draft_ms,enqueue_ms,wait_ms,observe_ms,accepted\n");
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
    let speedup = (pipe.tokens() as f64 / pipe.wall_s) / (solo.tokens() as f64 / solo.wall_s);
    println!(
        "RESULT mode=concurrent dspark=true first={} {common} k={k} sequences={n} \
         solo_tokens={} solo_rounds={} solo_wall_s={:.4} solo_aggregate_tok_s={:.4} \
         pipelined_tokens={} pipelined_rounds={} pipelined_wall_s={:.4} pipelined_aggregate_tok_s={:.4} \
         speedup={:.4} identical={} per_seq_identical={:?} per_seq_tokens={:?}",
        ctx.first,
        solo.tokens(),
        solo.rounds(),
        solo.wall_s,
        solo.tokens() as f64 / solo.wall_s,
        pipe.tokens(),
        pipe.rounds(),
        pipe.wall_s,
        pipe.tokens() as f64 / pipe.wall_s,
        speedup,
        identical.iter().all(|&b| b),
        identical,
        pipe.streams.iter().map(|s| s.len()).collect::<Vec<_>>(),
    );
    Ok(())
}

/// The ChatML user-turn wrapping, matching `lfm2-chat`'s single path.
fn user_turn_text(p: &str) -> String {
    format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
}
