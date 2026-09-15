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
//!   verifies, or — with `--batched` — all of them through
//!   `Session::decode_batched` (#694), one `m = B` forward per step for every
//!   live sequence. Reports per-sequence identity against the solo stream and
//!   the aggregate token rate.
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
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, PrefillOut, ReuseAction, Session};
use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

/// `design/workloads/prompts/`, the #549 corpus: a multi-turn pair and four
/// single prompts of DIFFERENT shapes — varied so the concurrent run exercises
/// prefill rather than sharing one prefix.
const TURN_1: &str = "I am designing a cache for a single-threaded interpreter. Entries are immutable once written and the working set is about eight times the cache. What eviction policy should I start with, and why not LRU?";
const TURN_2: &str = "Suppose I measure and find the hit rate is 40% lower than you predicted. What are the two most likely causes, and which measurement separates them?";
const PROMPTS: [&str; 8] = [
    "Name the three branches of the United States federal government.",
    "Count from one to forty, writing each number as a word on its own line, with no other text.",
    "Here is a Rust function that is missing its body:\n\nfn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32> {\n    // Merge two already-sorted slices into one sorted Vec.\n    // Do not allocate more than once.\n}\n\nWrite the body. Explain your choice of capacity for the allocation, and say what happens when one slice is empty.",
    "Extract every date, monetary amount, and party name from the paragraph below and return them as a JSON object with the keys \"dates\", \"amounts\", and \"parties\". Return only the JSON.\n\nOn 14 March 2019, Halloran Freight Ltd. agreed to pay Baswick Terminals the sum of 48,250 GBP in settlement of demurrage accrued between 2 January 2019 and 27 February 2019. A further 6,900 GBP was withheld pending inspection, to be released on or before 30 April 2019 unless Baswick Terminals lodged an objection.",
    // The `design/workloads/prompts/` corpus (#549), embedded so a batched run
    // at `--sequences` above four still has distinct prompts of distinct
    // shapes. The four above ARE the corpus's `short-factual`,
    // `low-entropy-list`, `code-completion` and `structured-extract`, so the
    // four added here are the corpus members the built-ins do not already
    // carry. Beyond eight the set cycles.
    include_str!("../../../../design/workloads/prompts/long-summarize.txt"),
    include_str!("../../../../design/workloads/prompts/multiturn-t1.txt"),
    include_str!("../../../../design/workloads/prompts/multiturn-t2.txt"),
    include_str!("../../../../design/workloads/prompts/multiturn-t3.txt"),
];

/// Render a knob's value only when the run read it. `--dspark` drives the
/// speculative body over a fixed `Blocking` handle whose commit, lookahead and
/// pacer paths it never enters (`Session::decode_speculative` commits only
/// under a ring-owning strategy and holds no pipelined period), so
/// `--readback`, `--lookahead`, `--commit-budget` and `--wake-cadence` are
/// parsed there and unread. A value printed beside a run that did not read it
/// compares equal to every run that did — §7.1a-i's wrong-arm shape — so it
/// renders as unread instead.
pub fn read_or_na(read: bool, value: impl std::fmt::Display) -> String {
    if read {
        value.to_string()
    } else {
        "n/a(dspark)".to_string()
    }
}

/// The first `n` prompts of the run: the set above, cycled past its length so
/// a row count the set does not cover still runs (two rows on one prompt
/// decode identical streams, which is a free neighbour-independence check).
fn prompt_set(n: usize) -> Vec<&'static str> {
    (0..n)
        .map(|i| PROMPTS[i % PROMPTS.len()].trim_end())
        .collect()
}

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
    /// `--prefill-ubatch`: the prefill's spark under `--dspark` (both its
    /// arms seed the drafter unit by unit, `Session::prefill_streamed`). The
    /// K=1 interleave and the batched arm prefill one-shot and do not read it.
    pub prefill_ubatch: usize,
    /// `--prefill-tile`: the most prompt tokens one prefill forward takes, on
    /// the K=1 interleave and the batched arm (#728).
    pub prefill_tile: usize,
    /// `--prefill-chunk`: the most prompt tokens fed of one sequence per
    /// visit (interleave) or per step (batched), between the others' decode.
    /// `0` prefills a prompt whole at its arrival.
    pub prefill_chunk: usize,
    /// `--pass-widths`: how a prompt piece's width is chosen (`lru` |
    /// `ladder`), on the batched arm's pieces and this driver's visits.
    pub pass_widths: lloom::nn::PassWidths,
    /// `--stagger`: sequence `i` arrives at round/step `i × stagger`.
    pub stagger: usize,
    /// `--long-prompt-tokens`: the last sequence's synthetic prompt length in
    /// body tokens; `0` keeps the corpus prompt.
    pub long_prompt_tokens: usize,
    /// `--dspark` — drive the concurrent sequences through the pipelined
    /// speculative driver instead of the time-sliced non-spec interleave.
    pub dspark: bool,
    /// `--batched` — drive the concurrent sequences through the batched-forward
    /// driver (`Session::decode_batched`, #694): one `m = B` forward per step.
    pub batched: bool,
    /// Under `--batched`: also run the rows in two separately batched halves
    /// and assert each row's stream equals its full-batch stream — the
    /// neighbour-independence check no digest sees.
    pub check_independence: bool,
    /// Under `--batched`: skip the solo arm (a tile A/B at a large `B` wants
    /// only the batched arm's step; the identity columns then read `false`).
    pub skip_solo: bool,
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

/// One prefill visit: feed `prompt` onto the session and produce nothing
/// (`Session::prefill`, the CHUNK seam).
fn prefill(
    session: &mut Session<'_, Model>,
    arm: &mut Arm,
    prompt: &[u32],
    opts: &DecodeOptions,
) -> Result<PrefillOut> {
    Ok(match arm {
        Arm::Blocking(rb) => session.prefill(rb, prompt, opts)?,
        Arm::Ring(rb) => session.prefill(rb, prompt, opts)?,
    })
}

/// The `n` prompts of a concurrent run as ids: the corpus set, chat-wrapped —
/// and, under `--long-prompt-tokens P`, the LAST sequence replaced by a
/// synthetic prompt of exactly `P` body tokens (the corpus's long-summarize
/// text repeated, cut at `P`, inside the same chat template), so a long prompt
/// arrives beside the short ones.
fn prompt_ids(ctx: &Ctx<'_>, n: usize) -> Result<Vec<Vec<u32>>> {
    let mut prompts: Vec<Vec<u32>> = prompt_set(n)
        .iter()
        .map(|p| (ctx.encode)(&user_turn_text(p)))
        .collect::<Result<_>>()?;
    if ctx.long_prompt_tokens > 0 {
        let target = ctx.long_prompt_tokens;
        let body_text = PROMPTS[4].trim_end();
        let mut body: Vec<u32> = Vec::new();
        while body.len() < target {
            body.extend((ctx.encode)(&format!("{body_text}\n\n"))?);
        }
        body.truncate(target);
        let mut ids = (ctx.encode)("<|im_start|>user\n")?;
        ids.extend(body);
        ids.extend((ctx.encode)("<|im_end|>\n<|im_start|>assistant\n")?);
        let last = n - 1;
        prompts[last] = ids;
    }
    Ok(prompts)
}

/// Gaps between consecutive stamps, in ms: `(p50, max)`, `NaN` when fewer
/// than two stamps.
fn gap_stats_ms(stamps: &[f64]) -> (f64, f64) {
    if stamps.len() < 2 {
        return (f64::NAN, f64::NAN);
    }
    let gaps: Vec<f64> = stamps.windows(2).map(|w| (w[1] - w[0]) * 1e3).collect();
    let max = gaps.iter().cloned().fold(f64::MIN, f64::max);
    (median(gaps), max)
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
                    self.tokens[i], self.targets[i], self.rounds[i], self.accept[i],
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
        // Auto (floor) here — the concurrent non-spec interleave paces per
        // visit; the computed depth is the single-decode path's concern.
        in_flight_depth: 0,
        // One-shot prefill: the concurrent interleave is not the streamed-prefill
        // path (that is the single-decode CLI's dial).
        prefill_ubatch: 0,
        // `--spark` is refused on the concurrent path (main.rs); one-shot here.
        prefill_schedule: None,
        prefill_bulk: 0,
        prefill_pacer_depth: 0,
        // Concurrent collects per-sequence ids for identity checks, not a live
        // per-token stream; the RingReader path is the single-turn streaming arm.
        stream_via_reader: false,
        adaptive_commit: false,
        run_ahead: 0,
        stop: None,
        // The tile bounds every prefill forward, a visit's chunk included.
        prefill_tile: ctx.prefill_tile,
        // The time-sliced interleave chunks at the visit level (`run_concurrent`)
        // and the batched driver reads its own copy (`run_batched_concurrent`).
        prefill_chunk: 0,
        // A visit feeds one whole chunk; this driver applies the ladder to
        // the visit's width itself (`run_concurrent`).
        pass_widths: lloom::nn::PassWidths::Lru,
    };
    let capacity = 8 * ctx.max_tokens + 8 * (budget.period() + 2);

    // Under `--dspark` the speculative drivers read none of the readback knobs
    // (`read_or_na`); the line names the arm that ran and marks the rest unread.
    let common = format!(
        "readback={} lookahead={} commit_budget={} in_flight_steps={} max_tokens={} dtype={:?}",
        if ctx.dspark { "dspark" } else { ctx.readback },
        read_or_na(!ctx.dspark, lookahead),
        read_or_na(!ctx.dspark, budget.arm()),
        read_or_na(!ctx.dspark, lloom::nn::decode::IN_FLIGHT_STEPS),
        ctx.max_tokens,
        ctx.dtype
    );

    match ctx.mode {
        "reuse" => run_reuse(&ctx, &opts, capacity, &common),
        "concurrent" if ctx.dspark => run_dspark_concurrent(&ctx, &common),
        "concurrent" if ctx.batched => run_batched_concurrent(&ctx, &opts, capacity, &common),
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
fn run_concurrent(
    ctx: &Ctx<'_>,
    opts: &DecodeOptions,
    capacity: usize,
    common: &str,
) -> Result<()> {
    let (model, config, device, dtype) = (ctx.model, ctx.config, ctx.device, ctx.dtype);
    let n = ctx.sequences.max(1);
    let prompts = prompt_ids(ctx, n)?;
    let prompt_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();

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
    // A sequence arrives at round `i × stagger`; while its prompt is not
    // fully fed a visit is a PREFILL visit of at most `prefill_chunk` prompt
    // tokens (the whole remainder when the chunk is 0) through
    // `Session::prefill`, then its visits decode `chunk` tokens each, resuming
    // through `prepare`. Per-sequence stamps: the arrival, the first token,
    // and every visit that streamed a token (the inter-visit gap of a decoding
    // sequence is what another sequence's prefill stretches).
    let mut sessions: Vec<Session<'_, Model>> = (0..n)
        .map(|_| {
            Ok(Session::new(
                model,
                Cache::new(true, dtype, config, device)?,
            ))
        })
        .collect::<Result<_>>()?;
    let mut arms: Vec<Arm> = (0..n)
        .map(|_| Arm::new(ctx.readback, ctx.eos_ids, device, capacity))
        .collect::<Result<_>>()?;
    let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut done = vec![false; n];
    let arrivals: Vec<usize> = (0..n).map(|i| i * ctx.stagger).collect();
    let mut prefilled = vec![0usize; n];
    let mut arrival_at: Vec<Option<f64>> = vec![None; n];
    let mut first_token_at: Vec<Option<f64>> = vec![None; n];
    let mut visit_stamps: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut prefill_visits = vec![0usize; n];
    let mut prefill_units = vec![0usize; n];
    let mut visits = 0usize;
    let mut resumes = 0usize;
    let mut round = 0usize;
    let mut live = LiveStats::new("concurrent", vec![ctx.max_tokens; n], false);
    live.paint();
    let t = Instant::now();
    while done.iter().any(|d| !d) {
        for i in 0..n {
            if done[i] || arrivals[i] > round {
                continue;
            }
            if arrival_at[i].is_none() {
                arrival_at[i] = Some(t.elapsed().as_secs_f64());
            }
            if prefilled[i] < prompts[i].len() {
                let remaining = prompts[i].len() - prefilled[i];
                let take = if ctx.prefill_chunk == 0 {
                    remaining
                } else if ctx.pass_widths == lloom::nn::PassWidths::Ladder {
                    // The visit's width from the ladder's rungs (§9.2aa I4):
                    // the tile bounded by the chunk is the top rung.
                    let top = if ctx.prefill_tile > 0 {
                        ctx.prefill_tile.min(ctx.prefill_chunk)
                    } else {
                        ctx.prefill_chunk
                    };
                    lloom::nn::ladder_width(remaining.min(ctx.prefill_chunk), top)
                } else {
                    remaining.min(ctx.prefill_chunk)
                };
                let prepared = sessions[i].prepare(&prompts[i][..prefilled[i] + take])?;
                if prepared.suffix.len() != take {
                    anyhow::bail!(
                        "seq {i}: prepare handed {} tokens to prefill where the chunk is {take} \
                         (action {:?})",
                        prepared.suffix.len(),
                        prepared.action
                    );
                }
                let p = prefill(&mut sessions[i], &mut arms[i], &prepared.suffix, opts)?;
                prefilled[i] += take;
                prefill_visits[i] += 1;
                prefill_units[i] += p.units;
                visits += 1;
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
                ..opts.clone()
            };
            let out = decode(
                &mut sessions[i],
                &mut arms[i],
                &prepared.suffix,
                &visit_opts,
            )?
            .out;
            let now = t.elapsed().as_secs_f64();
            if !out.ids.is_empty() {
                if first_token_at[i].is_none() {
                    first_token_at[i] = Some(now);
                }
                visit_stamps[i].push(now);
            }
            streams[i].extend_from_slice(&out.ids);
            visits += 1;
            if out.eos || streams[i].len() >= ctx.max_tokens {
                done[i] = true;
            }
            live.set(i, streams[i].len(), 0, done[i]);
            live.paint();
        }
        round += 1;
    }
    live.finish();
    let wall = t.elapsed().as_secs_f64();
    let total: usize = streams.iter().map(|s| s.len()).sum();
    let identical: Vec<bool> = streams
        .iter()
        .zip(solo.iter())
        .map(|(a, b)| a == b)
        .collect();
    let ttft_ms: Vec<f64> = (0..n)
        .map(|i| match (arrival_at[i], first_token_at[i]) {
            (Some(a), Some(f)) => (f - a) * 1e3,
            _ => f64::NAN,
        })
        .collect();
    let gaps: Vec<(f64, f64)> = visit_stamps.iter().map(|v| gap_stats_ms(v)).collect();
    for i in 0..n {
        eprintln!(
            "[seq {i}: prompt {} tok, arrival round {} at {:.1} ms; TTFT {:.1} ms; prefill visits {} \
             ({} forwards); visit gap p50 {:.1} max {:.1} ms; solo {} tokens at {:.1} tok/s; \
             interleaved {} tokens; {}]",
            prompt_lens[i],
            arrivals[i],
            arrival_at[i].map(|a| a * 1e3).unwrap_or(f64::NAN),
            ttft_ms[i],
            prefill_visits[i],
            prefill_units[i],
            gaps[i].0,
            gaps[i].1,
            solo[i].len(),
            solo_rates[i],
            streams[i].len(),
            if identical[i] { "IDENTICAL" } else { "DIFFERS" }
        );
    }
    eprintln!(
        "[{n} sequences interleaved in {}-token visits (prefill chunk {}, tile {}, stagger {}): \
         {total} tokens in {wall:.3} s = {:.1} tok/s aggregate; solo {:.1} tok/s mean]",
        ctx.chunk,
        ctx.prefill_chunk,
        ctx.prefill_tile,
        ctx.stagger,
        total as f64 / wall,
        solo_rates.iter().sum::<f64>() / n as f64,
    );
    println!(
        "RESULT mode=concurrent {common} eos_read={} sequences={n} chunk={} prefill_chunk={} \
         prefill_tile={} stagger={} prompt_tokens={:?} visits={} prefill_visits={:?} \
         prefill_units={:?} resumes={} total_tokens={} interleaved_wall_s={:.4} \
         aggregate_tok_s={:.4} solo_wall_s={:.4} solo_mean_tok_s={:.4} identical={} \
         per_seq_identical={:?} per_seq_tokens={:?} per_seq_ttft_ms={:?} \
         per_seq_gap_p50_ms={:?} per_seq_gap_max_ms={:?}",
        arms[0].eos_read(),
        ctx.chunk,
        ctx.prefill_chunk,
        ctx.prefill_tile,
        ctx.stagger,
        prompt_lens,
        visits,
        prefill_visits,
        prefill_units,
        resumes,
        total,
        wall,
        total as f64 / wall,
        solo_wall,
        solo_rates.iter().sum::<f64>() / n as f64,
        identical.iter().all(|&b| b),
        identical,
        streams.iter().map(|s| s.len()).collect::<Vec<_>>(),
        ttft_ms
            .iter()
            .map(|v| format!("{v:.1}"))
            .collect::<Vec<_>>(),
        gaps.iter()
            .map(|g| format!("{:.1}", g.0))
            .collect::<Vec<_>>(),
        gaps.iter()
            .map(|g| format!("{:.1}", g.1))
            .collect::<Vec<_>>(),
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
    let n = ctx.sequences.max(1);
    let prompts: Vec<Vec<u32>> = prompt_set(n)
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
    let dvb = unsafe {
        lloom::nn::VarBuilder::from_mmaped_safetensors(&[draft_file], DType::F16, device)?
    };
    let drafter =
        DSparkDrafter::new(dcfg.clone(), max_pos, dvb).context("constructing DSpark drafter")?;
    let mut spec = DSparkSpeculator::new(
        drafter,
        model.draft_embedding().clone(),
        model.lm_head_linear().clone(),
        n,
    )
    .context("constructing DSpark speculator")?;
    eprintln!(
        "[dspark: drafter loaded, {} layers, K={k} (verify window {} positions), {n} slots, \
         context kv {}]",
        dcfg.num_hidden_layers,
        k + 1,
        spec.ctx_kv().name()
    );
    let opts = DecodeOptions {
        max_tokens: ctx.max_tokens,
        prefill_ubatch: ctx.prefill_ubatch,
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
            .map(|_| {
                Ok(Session::new(
                    model,
                    Cache::new(true, dtype, config, device)?,
                ))
            })
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
    let speedup = (pipe.tokens() as f64 / pipe.wall_s) / (solo.tokens() as f64 / solo.wall_s);
    println!(
        "RESULT mode=concurrent dspark=true first={} {common} k={k} sequences={n} \
         prefill_ubatch={} solo_prefill_units={:?} pipelined_prefill_units={:?} \
         solo_tokens={} solo_rounds={} solo_wall_s={:.4} solo_aggregate_tok_s={:.4} \
         pipelined_tokens={} pipelined_rounds={} pipelined_wall_s={:.4} pipelined_aggregate_tok_s={:.4} \
         speedup={:.4} identical={} per_seq_identical={:?} per_seq_tokens={:?}",
        ctx.first,
        ctx.prefill_ubatch,
        solo.outs.iter().map(|o| o.stats.prefill_units).collect::<Vec<_>>(),
        pipe.outs.iter().map(|o| o.stats.prefill_units).collect::<Vec<_>>(),
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

/// The Metal profiler's `(gpu_busy_union_s, dispatches, command_buffers)`
/// since the last reset, then reset; zeros when the profiler is off or off
/// Metal. Taken per arm so an arm's GPU-busy is its own.
fn profile_take() -> (f64, u64, u64) {
    #[cfg(feature = "metal")]
    {
        use candle::kernels::metal::profile;
        if profile::enabled() {
            let s = profile::snapshot();
            profile::reset();
            return (s.gpu_busy_union_s, s.dispatches, s.command_buffers);
        }
    }
    (0.0, 0, 0)
}

/// The per-kernel dispatch census since the last reset, under
/// `CANDLE_METAL_PROFILE=1`; empty otherwise. Read BEFORE `profile_take`,
/// which resets it.
fn profile_census() -> HashMap<String, u64> {
    #[cfg(feature = "metal")]
    {
        use candle::kernels::metal::profile;
        if profile::enabled() {
            return profile::snapshot().by_label.into_iter().collect();
        }
    }
    HashMap::new()
}

/// One arm of the `--batched` concurrent run.
struct BatchArm {
    name: &'static str,
    streams: Vec<Vec<u32>>,
    outs: Vec<DecodeOut>,
    wall_s: f64,
    gpu_busy_s: f64,
    dispatches: u64,
    command_buffers: u64,
    /// The weight-GEMM/GEMV kernels dispatched, from the census (engagement:
    /// the batched arm must show `gemm_*` where solo shows `gemv_*`).
    census: HashMap<String, u64>,
    /// Per row, the wall stamp (s since the arm's start) of every token the
    /// callback handed over — the batched arm's TTFT and inter-token gaps.
    stamps: Vec<Vec<f64>>,
}

impl BatchArm {
    fn tokens(&self) -> usize {
        self.streams.iter().map(|s| s.len()).sum()
    }
    fn steps(&self) -> u64 {
        self.outs.iter().map(|o| o.stats.steps).max().unwrap_or(0)
    }
}

/// `--mode concurrent --batched`: every prompt solo through the selected
/// readback arm (the crown loop), and all of them through
/// `Session::decode_batched` — one `m = B` forward per step. Per-sequence
/// identity against solo, the aggregate rates, GPU busy per arm from the
/// profiler, and — under `--check-independence` — the rows re-run in two
/// separately batched halves, each row asserted equal to its full-batch
/// stream.
fn run_batched_concurrent(
    ctx: &Ctx<'_>,
    opts: &DecodeOptions,
    capacity: usize,
    common: &str,
) -> Result<()> {
    let (model, config, device, dtype) = (ctx.model, ctx.config, ctx.device, ctx.dtype);
    let n = ctx.sequences.max(1);
    let prompts = prompt_ids(ctx, n)?;
    let prompt_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    // The batched driver's own options: the lookahead is the arm's, the commit
    // budget and sample split have no meaning under it; the tile bounds each
    // prefill forward and the chunk is the per-step prompt budget (#728).
    let batched_opts = DecodeOptions {
        max_tokens: ctx.max_tokens,
        lookahead: ctx.lookahead.min(1),
        prefill_tile: ctx.prefill_tile,
        prefill_chunk: ctx.prefill_chunk,
        pass_widths: ctx.pass_widths,
        ..DecodeOptions::new(ctx.max_tokens)
    };
    // Row `i` arrives at step `i × stagger`.
    let arrivals: Vec<u64> = (0..n).map(|i| (i * ctx.stagger) as u64).collect();

    // Warm the batched GEMM pipelines at this `B` off the clock: the row-tile
    // pipelines `Session::warmup` compiled are the prefill's; a batched step at
    // `m = B` keys its own `align_m` variant (§13.1i).
    {
        let mut sessions: Vec<Session<'_, Model>> = (0..n)
            .map(|_| {
                Ok(Session::new(
                    model,
                    Cache::new(true, dtype, config, device)?,
                ))
            })
            .collect::<Result<_>>()?;
        let refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
        let warm = DecodeOptions {
            max_tokens: 4,
            ..batched_opts.clone()
        };
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let _ = Session::decode_batched(
            &mut sessions,
            &mut sampler,
            &refs,
            &warm,
            ctx.eos_ids,
            device,
            &mut |_, _| Ok(()),
        )?;
        device.synchronize()?;
        let _ = profile_take();
    }

    let run_solo = || -> Result<BatchArm> {
        let mut live = LiveStats::new("solo", vec![ctx.max_tokens; n], false);
        live.paint();
        let t = Instant::now();
        let mut outs = Vec::with_capacity(n);
        for (i, p) in prompts.iter().enumerate() {
            let mut s = Session::new(model, Cache::new(true, dtype, config, device)?);
            let mut arm = Arm::new(ctx.readback, ctx.eos_ids, device, capacity)?;
            let out = decode(&mut s, &mut arm, p, opts)?.out;
            live.set(i, out.ids.len(), 0, true);
            live.paint();
            outs.push(out);
        }
        device.synchronize()?;
        let wall_s = t.elapsed().as_secs_f64();
        live.finish();
        let census = profile_census();
        let (gpu_busy_s, dispatches, command_buffers) = profile_take();
        Ok(BatchArm {
            name: "solo",
            streams: outs.iter().map(|o| o.ids.clone()).collect(),
            outs,
            wall_s,
            gpu_busy_s,
            dispatches,
            command_buffers,
            census,
            stamps: vec![Vec::new(); n],
        })
    };
    let run_batched = |label: &'static str, rows: &[usize]| -> Result<BatchArm> {
        let mut sessions: Vec<Session<'_, Model>> = rows
            .iter()
            .map(|_| {
                Ok(Session::new(
                    model,
                    Cache::new(true, dtype, config, device)?,
                ))
            })
            .collect::<Result<_>>()?;
        let refs: Vec<&[u32]> = rows.iter().map(|&i| prompts[i].as_slice()).collect();
        let row_arrivals: Vec<u64> = rows.iter().map(|&i| arrivals[i]).collect();
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let mut streams: Vec<Vec<u32>> = vec![Vec::new(); rows.len()];
        let mut stamps: Vec<Vec<f64>> = vec![Vec::new(); rows.len()];
        let mut live = LiveStats::new(label, vec![ctx.max_tokens; rows.len()], false);
        live.paint();
        let t = Instant::now();
        let outs = Session::decode_batched_admitted(
            &mut sessions,
            &mut sampler,
            &refs,
            &row_arrivals,
            &batched_opts,
            ctx.eos_ids,
            device,
            &mut |i, ids| {
                let now = t.elapsed().as_secs_f64();
                streams[i].extend_from_slice(ids);
                stamps[i].extend(ids.iter().map(|_| now));
                live.set(i, streams[i].len(), 0, false);
                live.paint();
                Ok(())
            },
        )?;
        device.synchronize()?;
        let wall_s = t.elapsed().as_secs_f64();
        for (i, o) in outs.iter().enumerate() {
            live.set(i, streams[i].len(), 0, true);
            if o.ids != streams[i] {
                anyhow::bail!("row {i}: the batched callback stream differs from its ids");
            }
        }
        live.paint();
        live.finish();
        let census = profile_census();
        let (gpu_busy_s, dispatches, command_buffers) = profile_take();
        Ok(BatchArm {
            name: label,
            streams,
            outs,
            wall_s,
            gpu_busy_s,
            dispatches,
            command_buffers,
            census,
            stamps,
        })
    };

    let all: Vec<usize> = (0..n).collect();
    let empty_solo = || BatchArm {
        name: "solo (skipped)",
        streams: vec![Vec::new(); n],
        outs: Vec::new(),
        wall_s: f64::NAN,
        gpu_busy_s: 0.0,
        dispatches: 0,
        command_buffers: 0,
        census: HashMap::new(),
        stamps: vec![Vec::new(); n],
    };
    let (solo, batched) = match (ctx.first, ctx.skip_solo) {
        (_, true) => (empty_solo(), run_batched("batched", &all)?),
        ("solo", _) => {
            let s = run_solo()?;
            let b = run_batched("batched", &all)?;
            (s, b)
        }
        ("batched", _) => {
            let b = run_batched("batched", &all)?;
            let s = run_solo()?;
            (s, b)
        }
        (other, _) => anyhow::bail!("--first must be `solo` or `batched`, got `{other}`"),
    };

    // Identity against solo: tier 2 across the GEMV/GEMM seam (§13.5b), so
    // the first divergence index is reported beside the verdict.
    let mut identical = Vec::with_capacity(n);
    let mut common_prefix = Vec::with_capacity(n);
    for (i, &prompt_len) in prompt_lens.iter().enumerate() {
        let same = batched.streams[i] == solo.streams[i];
        let lcp = batched.streams[i]
            .iter()
            .zip(solo.streams[i].iter())
            .take_while(|(a, b)| a == b)
            .count();
        identical.push(same);
        common_prefix.push(lcp);
        let (gap_p50, gap_max) = gap_stats_ms(&batched.stamps[i]);
        eprintln!(
            "[seq {i}: prompt {prompt_len} tok, arrival step {}; batched first token at {:.1} ms \
             (prefill {} forwards), gap p50 {:.1} max {:.1} ms; solo {} tokens eos={}; batched {} tokens eos={} overrun={}; {}{}]",
            arrivals[i],
            batched.stamps[i].first().map(|s| s * 1e3).unwrap_or(f64::NAN),
            batched.outs[i].stats.prefill_units,
            gap_p50,
            gap_max,
            solo.streams[i].len(),
            solo.outs.get(i).map(|o| o.eos).unwrap_or(false),
            batched.streams[i].len(),
            batched.outs[i].eos,
            batched.outs[i].stats.overrun,
            if same { "IDENTICAL" } else { "DIFFERS" },
            if same {
                String::new()
            } else {
                format!(" at {lcp} of {} / {}", batched.streams[i].len(), solo.streams[i].len())
            }
        );
    }

    // Neighbour independence: the same rows batched in two halves must give
    // each row the full batch's stream — both arms take the `m > 1` GEMM,
    // so this is exact where the solo comparison is tier 2.
    let independent = if ctx.check_independence && n >= 2 {
        let half = n / 2;
        let lo: Vec<usize> = (0..half).collect();
        let hi: Vec<usize> = (half..n).collect();
        let a = run_batched("batched-lo", &lo)?;
        let b = run_batched("batched-hi", &hi)?;
        let mut ok = true;
        for (k, &i) in lo.iter().enumerate() {
            if a.streams[k] != batched.streams[i] {
                ok = false;
                eprintln!("[independence: row {i} differs between B={n} and B={half}]");
            }
        }
        for (k, &i) in hi.iter().enumerate() {
            if b.streams[k] != batched.streams[i] {
                ok = false;
                eprintln!(
                    "[independence: row {i} differs between B={n} and B={}]",
                    n - half
                );
            }
        }
        eprintln!(
            "[independence: rows re-batched as {half} + {}: {}]",
            n - half,
            if ok { "IDENTICAL" } else { "DIFFERS" }
        );
        Some(ok)
    } else {
        None
    };

    // The batched arm's step shape: median step, mean live rows per step.
    let mut step_ms: Vec<f64> = batched
        .outs
        .iter()
        .flat_map(|o| o.stats.step_ns.iter().map(|&ns| ns as f64 / 1e6))
        .collect();
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let batched_steps = batched.steps();
    let row_steps: u64 = batched.outs.iter().map(|o| o.stats.steps).sum();
    let mean_rows = if batched_steps > 0 {
        row_steps as f64 / batched_steps as f64
    } else {
        0.0
    };
    let solo_steps: u64 = solo.outs.iter().map(|o| o.stats.steps).sum();
    let gemm_disp = |c: &HashMap<String, u64>| -> u64 {
        c.iter()
            .filter(|(k, _)| k.starts_with("gemm_"))
            .map(|(_, v)| v)
            .sum()
    };
    let gemv_disp = |c: &HashMap<String, u64>| -> u64 {
        c.iter()
            .filter(|(k, _)| k.starts_with("gemv"))
            .map(|(_, v)| v)
            .sum()
    };
    for arm in [&solo, &batched] {
        eprintln!(
            "[{}: {} tokens in {:.3} s = {:.1} tok/s aggregate; gpu_busy {:.3} s ({:.1} %), \
             {} dispatches, {} command buffers; gemm {} gemv {}]",
            arm.name,
            arm.tokens(),
            arm.wall_s,
            arm.tokens() as f64 / arm.wall_s,
            arm.gpu_busy_s,
            100.0 * arm.gpu_busy_s / arm.wall_s.max(1e-9),
            arm.dispatches,
            arm.command_buffers,
            gemm_disp(&arm.census),
            gemv_disp(&arm.census),
        );
        // The weight-matmul kernels by name, so the tile an arm took is read
        // from the census rather than inferred from a flag (§2.4).
        let mut mm: Vec<(&String, &u64)> = arm
            .census
            .iter()
            .filter(|(k, _)| k.starts_with("gemm_") || k.starts_with("gemv"))
            .collect();
        mm.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
        for (label, n) in mm.iter().take(6) {
            eprintln!("    {n:>7}  {label}");
        }
    }
    eprintln!(
        "[batched: {batched_steps} steps, {mean_rows:.2} live rows/step mean, step median {:.3} ms, \
         gpu_busy/step {:.3} ms, dispatches/step {:.1}; solo: {solo_steps} steps, \
         gpu_busy/step {:.3} ms, dispatches/step {:.1}]",
        if step_ms.is_empty() { f64::NAN } else { step_ms[step_ms.len() / 2] },
        1e3 * batched.gpu_busy_s / batched_steps.max(1) as f64,
        batched.dispatches as f64 / batched_steps.max(1) as f64,
        1e3 * solo.gpu_busy_s / solo_steps.max(1) as f64,
        solo.dispatches as f64 / solo_steps.max(1) as f64,
    );
    let speedup = (batched.tokens() as f64 / batched.wall_s) / (solo.tokens() as f64 / solo.wall_s);
    let batched_gaps: Vec<(f64, f64)> = batched.stamps.iter().map(|v| gap_stats_ms(v)).collect();
    println!(
        "RESULT mode=concurrent batched=true first={} {common} batched_lookahead={} sequences={n} \
         prefill_chunk={} prefill_tile={} stagger={} arrivals={:?} prefill_units={:?} \
         first_token_ms={:?} gap_p50_ms={:?} gap_max_ms={:?} \
         prompt_tokens={:?} solo_tokens={} solo_steps={} solo_wall_s={:.4} solo_aggregate_tok_s={:.4} \
         solo_gpu_busy_s={:.4} solo_dispatches={} solo_gemm={} solo_gemv={} \
         batched_tokens={} batched_steps={} batched_rows_per_step={:.4} batched_wall_s={:.4} \
         batched_aggregate_tok_s={:.4} batched_gpu_busy_s={:.4} batched_dispatches={} \
         batched_gemm={} batched_gemv={} batched_step_ms_median={:.4} \
         speedup={:.4} identical={} per_seq_identical={:?} common_prefix={:?} independent={:?} \
         per_seq_tokens={:?} per_seq_overrun={:?}",
        ctx.first,
        batched_opts.lookahead,
        ctx.prefill_chunk,
        ctx.prefill_tile,
        ctx.stagger,
        arrivals,
        batched.outs.iter().map(|o| o.stats.prefill_units).collect::<Vec<_>>(),
        batched
            .stamps
            .iter()
            .map(|v| format!("{:.1}", v.first().map(|s| s * 1e3).unwrap_or(f64::NAN)))
            .collect::<Vec<_>>(),
        batched_gaps.iter().map(|g| format!("{:.1}", g.0)).collect::<Vec<_>>(),
        batched_gaps.iter().map(|g| format!("{:.1}", g.1)).collect::<Vec<_>>(),
        prompt_lens,
        solo.tokens(),
        solo_steps,
        solo.wall_s,
        solo.tokens() as f64 / solo.wall_s,
        solo.gpu_busy_s,
        solo.dispatches,
        gemm_disp(&solo.census),
        gemv_disp(&solo.census),
        batched.tokens(),
        batched_steps,
        mean_rows,
        batched.wall_s,
        batched.tokens() as f64 / batched.wall_s,
        batched.gpu_busy_s,
        batched.dispatches,
        gemm_disp(&batched.census),
        gemv_disp(&batched.census),
        if step_ms.is_empty() { f64::NAN } else { step_ms[step_ms.len() / 2] },
        speedup,
        identical.iter().all(|&b| b),
        identical,
        common_prefix,
        independent,
        batched.streams.iter().map(|s| s.len()).collect::<Vec<_>>(),
        batched.outs.iter().map(|o| o.stats.overrun).collect::<Vec<_>>(),
    );
    Ok(())
}

/// The ChatML user-turn wrapping, matching `lfm2-chat`'s single path.
fn user_turn_text(p: &str) -> String {
    format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
}

#[cfg(test)]
mod tests {
    use super::read_or_na;

    /// A `--dspark` line must not report a readback knob the speculative body
    /// never read; a K=1 line reports it verbatim.
    #[test]
    fn a_dspark_line_renders_unread_knobs_as_unread() {
        assert_eq!(read_or_na(false, 1usize), "n/a(dspark)");
        assert_eq!(read_or_na(false, "steps:8"), "n/a(dspark)");
        assert_eq!(read_or_na(true, 1usize), "1");
        assert_eq!(read_or_na(true, "steps:8"), "steps:8");
    }
}
