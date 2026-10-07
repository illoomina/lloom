//! `lloom bench --mode reuse | concurrent`: the multi-sequence measurement
//! modes, on LFM2 (`Served::run_mode`):
//!
//! * `--mode reuse` — a two-turn conversation decoded two ways (prefix reuse
//!   through `Session::prepare` vs a fresh prefill), reporting whether the
//!   streams agree and turn 2's TTFT on each. `DESIGN.md` §13.2f.
//! * `--mode concurrent` — `--sequences` independent conversations interleaved
//!   on one model in `--chunk`-token visits (non-speculative, time-sliced), or
//!   — with `--dspark` — all of them through
//!   `Session::decode_speculative_pipelined`, which overlaps one sequence's
//!   draft/observe with the others' verifies, or — with `--batched` — all of
//!   them through `Session::decode_batched_admitted`, one `m = B` forward per
//!   step for every live sequence. Reports per-sequence identity against the
//!   solo stream and the aggregate token rate.
//!
//! These are A/B instruments over the library's drivers, not serving shapes:
//! each carries its own solo reference arm and identity check, which a server
//! has no use for. The serving shape of the batched driver is `lloom serve
//! --batch N` (the scheduler, `batch.rs`).
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
use lloom::generation::poll::PollMechanism;
use lloom::generation::{
    Blocking, CommitBudget, EosRead, LogitsProcessor, ReadbackStrategy, Ring, Sampling,
};
use lloom::models::lfm2::{BatchPool, Config, GateFusion, Model, Shape as Lfm2Shape, State};
use lloom::models::lfm2_dspark::{DSparkConfig, DSparkDrafter, DSparkSpeculator};
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, PrefillOut, ReuseAction, Session};
use lloom::profile::Profile;
use lloom_tensor::{DType, Device};
use std::collections::HashMap;
use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Which mode `lloom bench --mode` runs besides the cells.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Reuse,
    Concurrent,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Reuse => "reuse",
            Mode::Concurrent => "concurrent",
        }
    }
}

/// A mode's knobs, resolved from the option groups and `lloom bench`'s own
/// flags before the engine loads.
#[derive(Debug, Clone)]
pub struct ModeSpec {
    pub mode: Mode,
    pub max_tokens: usize,
    /// The readback arm every sequence takes, by its `--readback` spelling.
    pub readback: String,
    pub lookahead: usize,
    pub commit_budget: CommitBudget,
    pub sequences: usize,
    /// Tokens per visit on the time-sliced interleave.
    pub chunk: usize,
    /// The prefill's spark under `--dspark` (both its arms seed the drafter
    /// unit by unit, `Session::prefill_streamed`); the K=1 interleave and the
    /// batched arm prefill one-shot and do not read it.
    pub prefill_ubatch: usize,
    /// The most prompt tokens one prefill forward takes, on the K=1
    /// interleave and the batched arm.
    pub prefill_tile: usize,
    /// The most prompt tokens fed of one sequence per visit (interleave) or
    /// per step (batched), between the others' decode; `0` prefills a prompt
    /// whole at its arrival.
    pub prefill_chunk: usize,
    /// How a prompt piece's width is chosen (`lru` | `ladder`), on the batched
    /// arm's pieces and the interleave's visits.
    pub pass_widths: lloom::nn::PassWidths,
    /// Sequence `i` arrives at round/step `i × stagger`.
    pub stagger: usize,
    /// The last sequence's synthetic prompt length in body tokens; `0` keeps
    /// the corpus prompt.
    pub long_prompt_tokens: usize,
    /// Drive the sequences through the pipelined speculative driver instead
    /// of the time-sliced interleave.
    pub dspark: bool,
    /// Drive the sequences through the batched-forward driver.
    pub batched: bool,
    /// How the sequences hold their state: `private` (a cache each), or a slot
    /// of one pool (`pool`, `pool-per-row`).
    pub rows: super::lfm2::RowsArm,
    /// Under `batched`: re-run the rows as two separately batched halves and
    /// assert each row's stream equals its full-batch stream — the
    /// neighbour-independence check no digest sees.
    pub check_independence: bool,
    /// Under `batched`: skip the solo arm (a tile A/B at a large `B` wants
    /// only the batched arm's step; the identity columns then read `false`).
    pub skip_solo: bool,
    /// Under `batched`: exit non-zero when the rows gate reads `fail`
    /// (`bench.fail_on_gate`). Off, the verdicts are reported and the run
    /// exits 0: a gate run executes every arm and reads each verdict off its
    /// `gate=` field (CONTRIBUTING §3.1a).
    pub fail_on_gate: bool,
    /// Which arm runs first (`solo`, or `pipelined` / `batched`), so a
    /// rotation can alternate the order.
    pub first: String,
    /// Under `dspark`: every round of both arms as CSV.
    pub timeline: Option<PathBuf>,
}

/// The rows gate's exit: the failures are reported either way, and they fail
/// the process only when `fail_on_gate` asks for it.
fn gate_exit(failures: &[String], fail_on_gate: bool) -> Result<()> {
    if failures.is_empty() {
        return Ok(());
    }
    let why = format!(
        "--mode concurrent --batched: {} row(s) broke what they are held to:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    if fail_on_gate {
        anyhow::bail!(why);
    }
    eprintln!("[gate fail, exit 0 (--fail-on-gate exits non-zero): {why}]");
    Ok(())
}

/// `design/workloads/prompts/`, the workloads corpus: a multi-turn pair and
/// four single prompts of DIFFERENT shapes — varied so the concurrent run
/// exercises prefill rather than sharing one prefix.
const TURN_1: &str = "I am designing a cache for a single-threaded interpreter. Entries are immutable once written and the working set is about eight times the cache. What eviction policy should I start with, and why not LRU?";
const TURN_2: &str = "Suppose I measure and find the hit rate is 40% lower than you predicted. What are the two most likely causes, and which measurement separates them?";
const PROMPTS: [&str; 8] = [
    "Name the three branches of the United States federal government.",
    "Count from one to forty, writing each number as a word on its own line, with no other text.",
    "Here is a Rust function that is missing its body:\n\nfn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32> {\n    // Merge two already-sorted slices into one sorted Vec.\n    // Do not allocate more than once.\n}\n\nWrite the body. Explain your choice of capacity for the allocation, and say what happens when one slice is empty.",
    "Extract every date, monetary amount, and party name from the paragraph below and return them as a JSON object with the keys \"dates\", \"amounts\", and \"parties\". Return only the JSON.\n\nOn 14 March 2019, Halloran Freight Ltd. agreed to pay Baswick Terminals the sum of 48,250 GBP in settlement of demurrage accrued between 2 January 2019 and 27 February 2019. A further 6,900 GBP was withheld pending inspection, to be released on or before 30 April 2019 unless Baswick Terminals lodged an objection.",
    // The corpus's other four, embedded so a batched run at `--sequences`
    // above four still has distinct prompts of distinct shapes (the four
    // above ARE its `short-factual`, `low-entropy-list`, `code-completion`
    // and `structured-extract`). Beyond eight the set cycles.
    include_str!("../../../../design/workloads/prompts/long-summarize.txt"),
    include_str!("../../../../design/workloads/prompts/multiturn-t1.txt"),
    include_str!("../../../../design/workloads/prompts/multiturn-t2.txt"),
    include_str!("../../../../design/workloads/prompts/multiturn-t3.txt"),
];

/// Render a knob's value only when the run read it. `--dspark` drives the
/// speculative body over a fixed `Blocking` handle whose commit, lookahead and
/// pacer paths it never enters, so `--readback`, `--lookahead` and
/// `--commit-budget` are parsed there and unread. A value printed beside a run
/// that did not read it compares equal to every run that did — §7.1a-i's
/// wrong-arm shape — so it renders as unread instead.
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

/// What `run` needs from the engine: the model it loaded, the config it was
/// built from, the EOS set, the knobs, and — under `--rows pool*` — the pool
/// every sequence's cache is a slot of.
pub struct Ctx<'a, P: Profile> {
    pub model: &'a Model<P>,
    pub config: &'a Config,
    pub device: &'a Device,
    pub dtype: DType,
    pub eos_ids: &'a [u32],
    pub spec: &'a ModeSpec,
    pub pool: Option<&'a BatchPool>,
    /// The drafter dir + config, under `--dspark`.
    pub dspark_setup: Option<(PathBuf, DSparkConfig)>,
    /// Raw tokenizer encode (no chat wrapping): the modes wrap with
    /// `user_turn_text` themselves, and the reuse mode also encodes
    /// pre-wrapped text raw.
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

/// One readback strategy per sequence, chosen by name. One per sequence,
/// held for the run: the ring's size is not worth a box.
#[allow(clippy::large_enum_variant)]
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

fn decode<P: Lfm2Shape>(
    session: &mut Session<'_, Model<P>>,
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
fn prefill<P: Lfm2Shape>(
    session: &mut Session<'_, Model<P>>,
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
fn prompt_ids<P: Lfm2Shape>(ctx: &Ctx<'_, P>, n: usize) -> Result<Vec<Vec<u32>>> {
    let mut prompts: Vec<Vec<u32>> = prompt_set(n)
        .iter()
        .map(|p| (ctx.encode)(&user_turn_text(p)))
        .collect::<Result<_>>()?;
    if ctx.spec.long_prompt_tokens > 0 {
        let target = ctx.spec.long_prompt_tokens;
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

/// The evidence the speculative drivers take: the `Ring` shapes give it,
/// the `No` shapes refuse by name (`profiles-and-memory-design.md` §7.3).
fn rewindable<P: Profile>() -> Result<lloom::profile::RewindProof<P>> {
    lloom::profile::RewindProof::<P>::ask().with_context(|| {
        format!(
            "the speculative mode on the {} shape: it needs `Ring`",
            P::KEY
        )
    })
}

/// A sequence's cache: a slot of the pool under `--rows pool*`, else its own.
fn new_cache<P: Lfm2Shape>(
    pool: Option<&BatchPool>,
    dtype: DType,
    config: &Config,
    device: &Device,
) -> Result<State<P>> {
    Ok(P::state(
        config,
        dtype,
        device,
        config.max_position_embeddings,
        pool,
    )?)
}

pub fn run<P: Lfm2Shape>(ctx: Ctx<'_, P>) -> Result<()> {
    let budget = ctx.spec.commit_budget;
    let lookahead = if ctx.spec.readback == "blocking" {
        0
    } else {
        ctx.spec.lookahead
    };
    let opts = DecodeOptions {
        max_tokens: ctx.spec.max_tokens,
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
        host_feed_wait: None,
        eos_park: None,
        adaptive_commit: false,
        run_ahead: 0,
        stop: None,
        eos_gate: false,
        fused_publish: true,
        // The tile bounds every prefill forward, a visit's chunk included.
        prefill_tile: ctx.spec.prefill_tile,
        // The time-sliced interleave chunks at the visit level (`run_concurrent`)
        // and the batched driver reads its own copy (`run_batched_concurrent`).
        prefill_chunk: 0,
        // A visit feeds one whole chunk; this driver applies the ladder to
        // the visit's width itself (`run_concurrent`).
        pass_widths: lloom::nn::PassWidths::Lru,
    };
    let capacity = 8 * ctx.spec.max_tokens + 8 * (budget.period() + 2);

    // Under `--dspark` the speculative drivers read none of the readback knobs
    // (`read_or_na`); the line names the arm that ran and marks the rest unread.
    let common = format!(
        "readback={} lookahead={} commit_budget={} in_flight_steps={} max_tokens={} dtype={:?}",
        if ctx.spec.dspark {
            "dspark"
        } else {
            ctx.spec.readback.as_str()
        },
        read_or_na(!ctx.spec.dspark, lookahead),
        read_or_na(!ctx.spec.dspark, budget.arm()),
        read_or_na(!ctx.spec.dspark, lloom::nn::decode::IN_FLIGHT_STEPS),
        ctx.spec.max_tokens,
        ctx.dtype
    );

    match ctx.spec.mode {
        Mode::Reuse => run_reuse(&ctx, &opts, capacity, &common),
        Mode::Concurrent if ctx.spec.dspark => run_dspark_concurrent(&ctx, &common),
        Mode::Concurrent if ctx.spec.batched => {
            run_batched_concurrent(&ctx, &opts, capacity, &common)
        }
        Mode::Concurrent => run_concurrent(&ctx, &opts, capacity, &common),
    }
}

/// `--mode reuse`: turn 1, then turn 2 through `prepare` (reuse) vs a fresh
/// prefill, and whether the two turn-2 streams agree.
fn run_reuse<P: Lfm2Shape>(
    ctx: &Ctx<'_, P>,
    opts: &DecodeOptions,
    capacity: usize,
    common: &str,
) -> Result<()> {
    let (model, config, device, dtype) = (ctx.model, ctx.config, ctx.device, ctx.dtype);
    let turn1 = (ctx.encode)(&user_turn_text(TURN_1))?;
    let mut session = Session::new(model, new_cache(ctx.pool, dtype, config, device)?);
    let mut arm = Arm::new(&ctx.spec.readback, ctx.eos_ids, device, capacity)?;
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

    let mut fresh = Session::new(model, new_cache(ctx.pool, dtype, config, device)?);
    let mut arm_f = Arm::new(&ctx.spec.readback, ctx.eos_ids, device, capacity)?;
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
fn run_concurrent<P: Lfm2Shape>(
    ctx: &Ctx<'_, P>,
    opts: &DecodeOptions,
    capacity: usize,
    common: &str,
) -> Result<()> {
    let (model, config, device, dtype) = (ctx.model, ctx.config, ctx.device, ctx.dtype);
    let n = ctx.spec.sequences.max(1);
    let prompts = prompt_ids(ctx, n)?;
    let prompt_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();

    // SOLO: each sequence alone (the identity + rate baseline).
    let mut solo: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut solo_wall = 0f64;
    let mut solo_rates = Vec::with_capacity(n);
    for p in &prompts {
        let mut s = Session::new(model, new_cache(ctx.pool, dtype, config, device)?);
        let mut arm = Arm::new(&ctx.spec.readback, ctx.eos_ids, device, capacity)?;
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
    let mut sessions: Vec<Session<'_, Model<P>>> = (0..n)
        .map(|_| {
            Ok(Session::new(
                model,
                new_cache(ctx.pool, dtype, config, device)?,
            ))
        })
        .collect::<Result<_>>()?;
    let mut arms: Vec<Arm> = (0..n)
        .map(|_| Arm::new(&ctx.spec.readback, ctx.eos_ids, device, capacity))
        .collect::<Result<_>>()?;
    let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut done = vec![false; n];
    let arrivals: Vec<usize> = (0..n).map(|i| i * ctx.spec.stagger).collect();
    let mut prefilled = vec![0usize; n];
    let mut arrival_at: Vec<Option<f64>> = vec![None; n];
    let mut first_token_at: Vec<Option<f64>> = vec![None; n];
    let mut visit_stamps: Vec<Vec<f64>> = vec![Vec::new(); n];
    let mut prefill_visits = vec![0usize; n];
    let mut prefill_units = vec![0usize; n];
    let mut visits = 0usize;
    let mut resumes = 0usize;
    let mut round = 0usize;
    let mut live = LiveStats::new("concurrent", vec![ctx.spec.max_tokens; n], false);
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
                let take = if ctx.spec.prefill_chunk == 0 {
                    remaining
                } else if ctx.spec.pass_widths == lloom::nn::PassWidths::Ladder {
                    // The visit's width from the ladder's rungs (§9.2aa I4):
                    // the tile bounded by the chunk is the top rung.
                    let top = if ctx.spec.prefill_tile > 0 {
                        ctx.spec.prefill_tile.min(ctx.spec.prefill_chunk)
                    } else {
                        ctx.spec.prefill_chunk
                    };
                    lloom::nn::ladder_width(remaining.min(ctx.spec.prefill_chunk), top)
                } else {
                    remaining.min(ctx.spec.prefill_chunk)
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
            let want = ctx.spec.chunk.min(ctx.spec.max_tokens - streams[i].len());
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
            if out.eos || streams[i].len() >= ctx.spec.max_tokens {
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
        ctx.spec.chunk,
        ctx.spec.prefill_chunk,
        ctx.spec.prefill_tile,
        ctx.spec.stagger,
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
        ctx.spec.chunk,
        ctx.spec.prefill_chunk,
        ctx.spec.prefill_tile,
        ctx.spec.stagger,
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
/// speculative loop, and all through the pipelined driver, with the live stats
/// repainting on the pipelined arm's per-sequence callback.
fn run_dspark_concurrent<P: Lfm2Shape>(ctx: &Ctx<'_, P>, common: &str) -> Result<()> {
    let (model, config, device, dtype) = (ctx.model, ctx.config, ctx.device, ctx.dtype);
    let (draft_dir, dcfg) = ctx
        .dspark_setup
        .as_ref()
        .context("--dspark set but no drafter resolved")?;
    let n = ctx.spec.sequences.max(1);
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
    let max_pos = longest + ctx.spec.max_tokens + dcfg.block_size + 64;
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
        max_tokens: ctx.spec.max_tokens,
        prefill_ubatch: ctx.spec.prefill_ubatch,
        ..DecodeOptions::new(ctx.spec.max_tokens)
    };
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);

    // Warm the drafter's pipelines off every clock below.
    {
        let mut s = Session::new(model, new_cache(ctx.pool, dtype, config, device)?);
        let mut rb = Blocking::new(ctx.eos_ids, device)?;
        let warm = DecodeOptions::new(16);
        let _ = s.decode_speculative(
            rewindable::<P>()?,
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
        let mut live = LiveStats::new("solo", vec![ctx.spec.max_tokens; n], true);
        live.paint();
        for (i, p) in prompts.iter().enumerate() {
            let mut s = Session::new(model, new_cache(ctx.pool, dtype, config, device)?);
            let mut rb = Blocking::new(ctx.eos_ids, device)?;
            let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
            let out = s.decode_speculative(
                rewindable::<P>()?,
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
        let mut sessions: Vec<Session<'_, Model<P>>> = (0..n)
            .map(|_| {
                Ok(Session::new(
                    model,
                    new_cache(ctx.pool, dtype, config, device)?,
                ))
            })
            .collect::<Result<_>>()?;
        let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
        let mut streams: Vec<Vec<u32>> = vec![Vec::new(); n];
        let mut live = LiveStats::new("pipelined", vec![ctx.spec.max_tokens; n], true);
        live.paint();
        let t = Instant::now();
        let outs = Session::decode_speculative_pipelined(
            &mut sessions,
            rewindable::<P>()?,
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

    let (solo, pipe) = match ctx.spec.first.as_str() {
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
    for (i, same) in identical.iter().enumerate() {
        eprintln!(
            "[seq {i}: solo {} tokens in {} rounds; pipelined {} tokens in {} rounds; {}]",
            solo.streams[i].len(),
            solo.outs[i].stats.rounds.len(),
            pipe.streams[i].len(),
            pipe.outs[i].stats.rounds.len(),
            if *same { "IDENTICAL" } else { "DIFFERS" }
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
    if let Some(path) = &ctx.spec.timeline {
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
        ctx.spec.first,
        ctx.spec.prefill_ubatch,
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
        use lloom_tensor::kernels::metal::profile;
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
        use lloom_tensor::kernels::metal::profile;
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

/// Which kernels the solo arm's `m = 1` matmuls take: the shipping GEMV with
/// the model's GEMV-only epilogues, or the batch's own GEMM. The batch is held
/// to the solo stream bit for bit on the second and to tier 2 on the first,
/// where a near-tie argmax may part across the seam
/// (`batched-decode-design.md` §10.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SoloKernels {
    Gemv,
    Gemm,
}

impl SoloKernels {
    /// `--force-gemm m` (or `both`) with every GEMV-only epilogue the model
    /// runs off — the model's, since `Fusion=Derived` decides them rather
    /// than the config.
    fn of<P: Profile>(model: &Model<P>, config: &Config) -> Self {
        let forced = matches!(
            lloom_tensor::tuning::tuning().gemm.force.as_deref(),
            Some("m" | "both")
        );
        let (silu, residual, gate) = match model.fusion_report() {
            Some(r) => (r.gemv_silu, r.gemv_residual, r.gemv_gate),
            None => (config.gemv_silu, config.gemv_residual, config.gemv_gate),
        };
        if forced && !silu && !residual && gate == GateFusion::Off {
            SoloKernels::Gemm
        } else {
            SoloKernels::Gemv
        }
    }

    fn name(self) -> &'static str {
        match self {
            SoloKernels::Gemv => "gemv",
            SoloKernels::Gemm => "gemm",
        }
    }
}

/// The forward that produced one of a row's tokens: `forward` (the solo arm)
/// or a batched step over `m` rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pass {
    Solo,
    Rows(usize),
}

impl Pass {
    /// The `m = 1` route: the GEMV, unless `--force-gemm m`.
    fn single(self) -> bool {
        matches!(self, Pass::Solo | Pass::Rows(1))
    }
}

/// The pass behind each of a batched row's tokens `1..tokens` (entry `t - 1`;
/// token 0 is the prefill's), from the live-row count of every step the row
/// took part in. Under the lookahead the forward of the step that streams
/// token `t - 1` produces token `t` over that step's rows; without it the
/// forward runs after the readback over the rows kept, which are the next
/// step's when no row joins. `None` when the count is short, or when prompt
/// pieces ride the steps (`--prefill-chunk`): a ragged pass is not `m` rows.
fn batched_passes(
    batch_rows: &[usize],
    lookahead: usize,
    tokens: usize,
    prefill_chunk: usize,
) -> Option<Vec<Pass>> {
    if prefill_chunk > 0 {
        return None;
    }
    let shift = if lookahead > 0 { 1 } else { 0 };
    (1..tokens)
        .map(|t| batch_rows.get(t - shift).map(|&m| Pass::Rows(m)))
        .collect()
}

/// The first token whose forward took the `m = 1` route on one arm and not
/// on the other: where the two arms' GEMV/GEMM seam is first crossed.
fn seam_crossed(a: &[Pass], b: &[Pass], through: usize) -> Option<usize> {
    (1..=through).find(|&t| match (a.get(t - 1), b.get(t - 1)) {
        (Some(x), Some(y)) => x.single() != y.single(),
        _ => false,
    })
}

/// A row's parting between two arms, re-derived: the prompt and the shared
/// prefix replayed through each arm's own passes (`forward`, or `forward_rows`
/// over `m` copies of the row — the batch's `m`, so its pipeline) in lockstep,
/// both logit vectors read at every token up to the parting.
#[derive(Debug, Clone, PartialEq)]
struct Parting {
    at: usize,
    a_token: u32,
    b_token: u32,
    /// The first token whose argmax left that arm's stream, when the replay
    /// did not reproduce it through `at`.
    a_left: Option<usize>,
    b_left: Option<usize>,
    /// The copies of a batched row disagreed at some token.
    copies_differ: bool,
    /// Each path's logits for `(a_token, b_token)` at `at`.
    a_logits: (f32, f32),
    b_logits: (f32, f32),
    /// How many tokens besides the path's own beat the other arm's token at
    /// `at` on that path: 0 when it is the runner-up or tied with the leader.
    a_others_above: usize,
    b_others_above: usize,
    /// max |Δlogit| between the paths at `at`, and over every token before it.
    delta_at: f32,
    delta_before: f32,
    /// The widest step's max |Δlogit| through `at`, in ulps of the logits'
    /// dtype at that step's logit scale (`max |logit|`).
    ulps: f32,
    /// The same, token by token `0..=at`: flat is a reordering, growing is
    /// error accumulating (§2.3.5a).
    trace: Vec<f32>,
}

/// The most the GEMV/GEMM seam may move a logit vector, in ulps of the dtype
/// at the vector's scale: past 4× the widest reordering §2.3.5a measured on
/// LFM2's logits (3.50 ulp) and two orders of magnitude below its measured
/// defect (1840 ulp).
const SEAM_ULPS: f32 = 16.0;

/// The gap above `|x|` at the precision `dtype` carries (§2.3.5a's
/// `logit_delta.py`, the same definition).
fn ulp_at(x: f32, dtype: DType) -> f32 {
    let bits = match dtype {
        DType::F16 => 10,
        DType::BF16 => 7,
        _ => 23,
    };
    let x = x.abs();
    if x == 0.0 {
        return (2f32).powi(-24 - bits);
    }
    (2f32).powi(x.log2().floor() as i32 - bits)
}

/// max |a - b| in ulps of `dtype` at `a`'s scale.
fn ulps_apart(a: &[f32], b: &[f32], dtype: DType) -> f32 {
    let scale = a.iter().fold(0f32, |m, x| m.max(x.abs()));
    max_abs_diff(a, b) / ulp_at(scale, dtype)
}

impl Parting {
    /// A near-tie: both replays reproduce their arm's stream, the paths stay
    /// within the seam's reach of each other (`SEAM_ULPS`), and at the parting
    /// nothing but each path's own token stands above the other arm's —
    /// runner-up, or tied with the leader and losing on the id.
    fn is_near_tie(&self) -> bool {
        self.a_left.is_none()
            && self.b_left.is_none()
            && !self.copies_differ
            && self.ulps <= SEAM_ULPS
            && self.a_others_above == 0
            && self.b_others_above == 0
    }

    /// The widest step in the first and in the last `w` tokens of the trace,
    /// `w` up to 50 (§2.3.5a's growth reading, `logit_delta.py`'s window).
    fn growth(&self) -> (usize, f32, f32) {
        let n = self.trace.len();
        let w = 50.min(n / 2).max(1).min(n);
        let widest = |s: &[f32]| s.iter().fold(0f32, |m, &x| m.max(x));
        (w, widest(&self.trace[..w]), widest(&self.trace[n - w..]))
    }

    /// Each path's logits at the parting, `a`'s path first.
    fn logits(&self, a: &str, b: &str) -> String {
        format!(
            "{a} path {} = {:.4}, {} = {:.4}; {b} path {} = {:.4}, {} = {:.4}",
            self.a_token,
            self.a_logits.0,
            self.b_token,
            self.a_logits.1,
            self.a_token,
            self.b_logits.0,
            self.b_token,
            self.b_logits.1
        )
    }
}

/// One row's comparison between two arms.
#[derive(Debug, Clone, PartialEq)]
enum RowVerdict {
    Identical,
    /// Tier 2: the arms' `m = 1` routes differ from token `seam` on, and the
    /// row parts at a near-tie.
    SeamTie {
        seam: usize,
        parting: Parting,
    },
    /// Not re-derivable from this run (`--prefill-chunk`, or a replay that
    /// could not run); neither a pass nor a failure.
    Unverified(String),
    Differs(String),
}

impl RowVerdict {
    fn name(&self) -> &'static str {
        match self {
            RowVerdict::Identical => "identical",
            RowVerdict::SeamTie { .. } => "seam-tie",
            RowVerdict::Unverified(_) => "unverified",
            RowVerdict::Differs(_) => "differs",
        }
    }
}

/// What two arms of one row are held to. Bit for bit when the solo arm runs
/// the batch's GEMM, or when both arms took the `m = 1` route at the same
/// tokens up to the parting; otherwise tier 2, where the parting must
/// re-derive as a near-tie (`replay`, called with the parting index).
fn row_verdict(
    kernels: SoloKernels,
    a: &[u32],
    b: &[u32],
    passes: Option<(&[Pass], &[Pass])>,
    replay: impl FnOnce(usize) -> Result<Parting>,
) -> RowVerdict {
    if a == b {
        return RowVerdict::Identical;
    }
    let at = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    if at == a.len().min(b.len()) {
        return RowVerdict::Differs(format!(
            "one stream ends where the other goes on ({} against {} tokens)",
            a.len(),
            b.len()
        ));
    }
    if kernels == SoloKernels::Gemm {
        return RowVerdict::Differs(
            "every forward runs the GEMM here, where the row is owed its stream bit for bit".into(),
        );
    }
    let Some((pa, pb)) = passes else {
        return RowVerdict::Unverified(
            "the row's step record does not name its forwards (prompt pieces rode the steps, or \
             the record is short)"
                .into(),
        );
    };
    let Some(seam) = seam_crossed(pa, pb, at) else {
        return RowVerdict::Differs(format!(
            "both arms took the same route at every token through {at}, where the row is owed \
             its stream bit for bit"
        ));
    };
    match replay(at) {
        Ok(p) if p.is_near_tie() => RowVerdict::SeamTie { seam, parting: p },
        Ok(p) => RowVerdict::Differs(format!(
            "not a near-tie: the replay left the arms at {:?} / {:?}, copies differ {}, {} / {} \
             other tokens above the other arm's, the paths {:.2} ulp apart (the seam's reach is \
             {SEAM_ULPS}); {}",
            p.a_left,
            p.b_left,
            p.copies_differ,
            p.a_others_above,
            p.b_others_above,
            p.ulps,
            p.logits("one arm's", "the other's")
        )),
        Err(e) => RowVerdict::Unverified(format!("the replay could not run: {e}")),
    }
}

impl RowVerdict {
    /// The verdict's word on the per-row line.
    fn label(&self) -> &'static str {
        match self {
            RowVerdict::Identical => "IDENTICAL",
            RowVerdict::SeamTie { .. } => "PARTS (tier 2)",
            RowVerdict::Unverified(_) => "PARTS (not re-derived)",
            RowVerdict::Differs(_) => "DIFFERS",
        }
    }

    /// The verdict as a line prints it: `a` and `b` name the two arms, whose
    /// streams run `a_len` and `b_len` tokens.
    fn describe(&self, (a, a_len): (&str, usize), (b, b_len): (&str, usize), at: usize) -> String {
        let why = match self {
            RowVerdict::Identical => return self.label().into(),
            RowVerdict::SeamTie { seam, parting: p } => format!(
                "a near-tie across the GEMV/GEMM seam, crossed from token {seam}: {} (margins \
                 {:.4} / {:.4}); max |Δlogit| between the paths {:.4} there, {:.4} before, at \
                 most {:.2} ulp at the logit scale ({:.2} / {:.2} over the first / last {} tokens)",
                p.logits(a, b),
                p.a_logits.0 - p.a_logits.1,
                p.b_logits.1 - p.b_logits.0,
                p.delta_at,
                p.delta_before,
                p.ulps,
                p.growth().1,
                p.growth().2,
                p.growth().0
            ),
            RowVerdict::Unverified(why) | RowVerdict::Differs(why) => why.clone(),
        };
        format!("{} at {at} of {b_len} / {a_len}: {why}", self.label())
    }
}

/// `v`'s first maximum.
fn argmax(v: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best as u32
}

/// How many logits other than `own`'s beat `token`'s.
fn others_above(v: &[f32], token: u32, own: u32) -> usize {
    let l = v[token as usize];
    v.iter()
        .enumerate()
        .filter(|&(i, &x)| i != own as usize && x > l)
        .count()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// One arm of a replay: its passes, and as many copies of the row's cache as
/// its widest step, prefilled as every arm prefills (`forward` per
/// `prefill_tile` of the prompt).
struct ReplayArm<'p, P: Profile> {
    passes: &'p [Pass],
    caches: Vec<State<P>>,
    /// The copies still in step with the row: a step of `m` rows advances the
    /// first `m`.
    live: usize,
}

impl<'p, P: Lfm2Shape> ReplayArm<'p, P> {
    fn new(
        ctx: &Ctx<'_, P>,
        prompt: &[u32],
        passes: &'p [Pass],
        upto: usize,
    ) -> Result<(Self, Vec<f32>)> {
        use lloom::nn::CausalLM;
        let width = passes[..upto]
            .iter()
            .map(|p| match p {
                Pass::Solo => 1,
                Pass::Rows(m) => *m,
            })
            .max()
            .unwrap_or(1)
            .max(1);
        let tile = match ctx.spec.prefill_tile {
            0 => prompt.len().max(1),
            t => t,
        };
        let mut caches = Vec::with_capacity(width);
        let mut first = None;
        for _ in 0..width {
            let mut cache = new_cache(ctx.pool, ctx.dtype, ctx.config, ctx.device)?;
            let (mut pos, mut last) = (0, None);
            for unit in prompt.chunks(tile) {
                CausalLM::prepare_pass(ctx.model, &mut cache, unit.len())?;
                let input = lloom_tensor::Tensor::new(unit, ctx.device)?.unsqueeze(0)?;
                last = Some(CausalLM::forward(ctx.model, &input, pos, &mut cache)?);
                pos += unit.len();
            }
            let logits = host_logits(ctx, &last.context("an empty prompt")?)?;
            first.get_or_insert(logits);
            caches.push(cache);
        }
        Ok((
            ReplayArm {
                passes,
                caches,
                live: width,
            },
            first.expect("at least one copy"),
        ))
    }

    /// Feed `token` as the row's token `t - 1` at `pos`; the logits for token
    /// `t` and whether the copies agreed.
    fn step(
        &mut self,
        ctx: &Ctx<'_, P>,
        t: usize,
        token: u32,
        pos: usize,
    ) -> Result<(Vec<f32>, bool)> {
        use lloom::nn::{BatchedCausalLM, CausalLM, Row};
        use lloom_tensor::Tensor;
        match self.passes[t - 1] {
            Pass::Solo => {
                let cache = &mut self.caches[0];
                CausalLM::prepare_pass(ctx.model, cache, 1)?;
                let input = Tensor::new(&[[token]], ctx.device)?;
                let out = CausalLM::forward(ctx.model, &input, pos, cache)?;
                self.live = 1;
                Ok((host_logits(ctx, &out)?, true))
            }
            Pass::Rows(m) => {
                anyhow::ensure!(
                    m >= 1 && m <= self.live,
                    "token {t}: a step of {m} rows after one of {} (a row joined mid-run)",
                    self.live
                );
                for cache in &mut self.caches[..m] {
                    CausalLM::prepare_pass(ctx.model, cache, 1)?;
                }
                let input = Tensor::new(vec![token; m].as_slice(), ctx.device)?.reshape((m, 1))?;
                let mut rows: Vec<Row<'_, State<P>>> = self.caches[..m]
                    .iter_mut()
                    .map(|cache| Row { pos, cache })
                    .collect();
                let out = BatchedCausalLM::forward_rows(ctx.model, &input, &mut rows)?;
                self.live = m;
                let first = host_logits(ctx, &out.get(0)?)?;
                let mut agree = true;
                for k in 1..m {
                    agree &= host_logits(ctx, &out.get(k)?)? == first;
                }
                Ok((first, agree))
            }
        }
    }
}

/// The last position's logits, on the host, as f32.
fn host_logits<P: Lfm2Shape>(ctx: &Ctx<'_, P>, t: &lloom_tensor::Tensor) -> Result<Vec<f32>> {
    use lloom::nn::CausalLM;
    let vocab = CausalLM::vocab_size(ctx.model);
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    anyhow::ensure!(
        v.len() >= vocab,
        "{} logits for a vocabulary of {vocab}",
        v.len()
    );
    Ok(v[v.len() - vocab..].to_vec())
}

/// Replay one row through two arms to their parting at token `at` (`Parting`).
fn replay_parting<P: Lfm2Shape>(
    ctx: &Ctx<'_, P>,
    prompt: &[u32],
    (a, pa): (&[u32], &[Pass]),
    (b, pb): (&[u32], &[Pass]),
    at: usize,
) -> Result<Parting> {
    anyhow::ensure!(
        pa.len() >= at && pb.len() >= at,
        "passes recorded for {} / {} tokens, the parting is at {at}",
        pa.len() + 1,
        pb.len() + 1
    );
    let (mut ra, mut la) = ReplayArm::new(ctx, prompt, pa, at)?;
    let (mut rb, mut lb) = ReplayArm::new(ctx, prompt, pb, at)?;
    let (mut a_left, mut b_left) = (None, None);
    let mut copies_differ = false;
    let mut delta_before = 0f32;
    let mut trace = Vec::with_capacity(at + 1);
    for t in 0..at {
        delta_before = delta_before.max(max_abs_diff(&la, &lb));
        trace.push(ulps_apart(&la, &lb, ctx.dtype));
        if a_left.is_none() && argmax(&la) != a[t] {
            a_left = Some(t);
        }
        if b_left.is_none() && argmax(&lb) != b[t] {
            b_left = Some(t);
        }
        let pos = prompt.len() + t;
        let (next, agree) = ra.step(ctx, t + 1, a[t], pos)?;
        la = next;
        copies_differ |= !agree;
        let (next, agree) = rb.step(ctx, t + 1, b[t], pos)?;
        lb = next;
        copies_differ |= !agree;
    }
    let (at_a, at_b) = (a[at], b[at]);
    if a_left.is_none() && argmax(&la) != at_a {
        a_left = Some(at);
    }
    if b_left.is_none() && argmax(&lb) != at_b {
        b_left = Some(at);
    }
    trace.push(ulps_apart(&la, &lb, ctx.dtype));
    Ok(Parting {
        at,
        a_token: at_a,
        b_token: at_b,
        a_left,
        b_left,
        copies_differ,
        a_logits: (la[at_a as usize], la[at_b as usize]),
        b_logits: (lb[at_a as usize], lb[at_b as usize]),
        a_others_above: others_above(&la, at_b, at_a),
        b_others_above: others_above(&lb, at_a, at_b),
        delta_at: max_abs_diff(&la, &lb),
        delta_before,
        ulps: trace.iter().fold(0f32, |m, &x| m.max(x)),
        trace,
    })
}

/// `--mode concurrent --batched`: every prompt solo through the selected
/// readback arm (the crown loop), and all of them through
/// `Session::decode_batched` — one `m = B` forward per step. Per-sequence
/// identity against solo, the aggregate rates, GPU busy per arm from the
/// profiler, and — unless `--no-independence-check` — the rows re-run in two
/// separately batched halves, each row held to its full-batch stream.
///
/// A row is held to the other arm's stream bit for bit wherever both ran its
/// forwards on the same route, and to tier 2 where one took the `m = 1` GEMV
/// and the other the batch's GEMM (`row_verdict`); a parting there must
/// replay as a near-tie. Anything else fails the run.
fn run_batched_concurrent<P: Lfm2Shape>(
    ctx: &Ctx<'_, P>,
    opts: &DecodeOptions,
    capacity: usize,
    common: &str,
) -> Result<()> {
    let (model, config, device, dtype) = (ctx.model, ctx.config, ctx.device, ctx.dtype);
    let n = ctx.spec.sequences.max(1);
    let prompts = prompt_ids(ctx, n)?;
    let prompt_lens: Vec<usize> = prompts.iter().map(|p| p.len()).collect();
    // The batched driver's own options: the lookahead is the arm's, the commit
    // budget and sample split have no meaning under it; the tile bounds each
    // prefill forward and the chunk is the per-step prompt budget.
    let batched_opts = DecodeOptions {
        max_tokens: ctx.spec.max_tokens,
        lookahead: ctx.spec.lookahead.min(1),
        prefill_tile: ctx.spec.prefill_tile,
        prefill_chunk: ctx.spec.prefill_chunk,
        pass_widths: ctx.spec.pass_widths,
        ..DecodeOptions::new(ctx.spec.max_tokens)
    };
    // Row `i` arrives at step `i × stagger`.
    let arrivals: Vec<u64> = (0..n).map(|i| (i * ctx.spec.stagger) as u64).collect();

    // Warm the batched GEMM pipelines at this `B` off the clock: the row-tile
    // pipelines `Session::warmup` compiled are the prefill's; a batched step at
    // `m = B` keys its own `align_m` variant (§13.1i).
    {
        let mut sessions: Vec<Session<'_, Model<P>>> = (0..n)
            .map(|_| {
                Ok(Session::new(
                    model,
                    new_cache(ctx.pool, dtype, config, device)?,
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
        let mut live = LiveStats::new("solo", vec![ctx.spec.max_tokens; n], false);
        live.paint();
        let t = Instant::now();
        let mut outs = Vec::with_capacity(n);
        for (i, p) in prompts.iter().enumerate() {
            let mut s = Session::new(model, new_cache(ctx.pool, dtype, config, device)?);
            let mut arm = Arm::new(&ctx.spec.readback, ctx.eos_ids, device, capacity)?;
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
        let mut sessions: Vec<Session<'_, Model<P>>> = rows
            .iter()
            .map(|_| {
                Ok(Session::new(
                    model,
                    new_cache(ctx.pool, dtype, config, device)?,
                ))
            })
            .collect::<Result<_>>()?;
        let refs: Vec<&[u32]> = rows.iter().map(|&i| prompts[i].as_slice()).collect();
        let row_arrivals: Vec<u64> = rows.iter().map(|&i| arrivals[i]).collect();
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let mut streams: Vec<Vec<u32>> = vec![Vec::new(); rows.len()];
        let mut stamps: Vec<Vec<f64>> = vec![Vec::new(); rows.len()];
        let mut live = LiveStats::new(label, vec![ctx.spec.max_tokens; rows.len()], false);
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
    let (solo, batched) = match (ctx.spec.first.as_str(), ctx.spec.skip_solo) {
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

    // What the rows are held to (`batched-decode-design.md` §10.4): bit for
    // bit wherever both arms run a token's forward on the same route, tier 2
    // where one took the `m = 1` GEMV and the other the batch's GEMM — there a
    // row may part only at a near-tie, which the replay re-derives.
    let kernels = SoloKernels::of(model, config);
    let lookahead = batched_opts.lookahead;
    let chunk = ctx.spec.prefill_chunk;
    let passes_of = |arm: &BatchArm, k: usize| {
        batched_passes(
            &arm.outs[k].stats.batch_rows,
            lookahead,
            arm.streams[k].len(),
            chunk,
        )
    };
    let judge =
        |i: usize, (a, pa): (&[u32], Option<Vec<Pass>>), (b, pb): (&[u32], Option<Vec<Pass>>)| {
            let passes = pa.as_deref().zip(pb.as_deref());
            row_verdict(kernels, a, b, passes, |at| {
                let (pa, pb) = passes.expect("a replay is asked for only with both arms' passes");
                replay_parting(ctx, &prompts[i], (a, pa), (b, pb), at)
            })
        };
    if !ctx.spec.skip_solo {
        eprintln!(
            "[identity: the solo arm's m = 1 matmuls take the {} ({}); a batched row is held to \
             its solo stream bit for bit wherever both run a token's forward on the same route, \
             and to tier 2 where the routes differ — a parting there must re-derive as a \
             near-tie{}]",
            kernels.name(),
            match kernels {
                SoloKernels::Gemv => "the shipping decode",
                SoloKernels::Gemm => "--force-gemm m, the GEMV-only epilogues off",
            },
            match kernels {
                SoloKernels::Gemv => {
                    "; the arm exact at every token is --force-gemm m --gemv-silu=false \
                     --gemv-residual=false"
                }
                SoloKernels::Gemm => "",
            }
        );
    }
    let mut identical = Vec::with_capacity(n);
    let mut common_prefix = Vec::with_capacity(n);
    let mut verdicts: Vec<RowVerdict> = Vec::with_capacity(n);
    for (i, &prompt_len) in prompt_lens.iter().enumerate() {
        let same = batched.streams[i] == solo.streams[i];
        let lcp = batched.streams[i]
            .iter()
            .zip(solo.streams[i].iter())
            .take_while(|(a, b)| a == b)
            .count();
        identical.push(same);
        common_prefix.push(lcp);
        let verdict = if ctx.spec.skip_solo {
            RowVerdict::Unverified("the solo arm was skipped".into())
        } else {
            judge(
                i,
                (
                    &solo.streams[i],
                    Some(vec![Pass::Solo; solo.streams[i].len()]),
                ),
                (&batched.streams[i], passes_of(&batched, i)),
            )
        };
        let (gap_p50, gap_max) = gap_stats_ms(&batched.stamps[i]);
        eprintln!(
            "[seq {i}: prompt {prompt_len} tok, arrival step {}; batched first token at {:.1} ms \
             (prefill {} forwards), gap p50 {:.1} max {:.1} ms; solo {} tokens eos={}; batched {} tokens eos={} overrun={}; {}]",
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
            verdict.describe(
                ("solo", solo.streams[i].len()),
                ("batched", batched.streams[i].len()),
                lcp
            ),
        );
        if let RowVerdict::SeamTie { parting, .. } = &verdict {
            eprintln!(
                "[seq {i}: the replay's max |Δlogit| per token 0..={}, ulp at the logit scale: {}]",
                parting.at,
                parting
                    .trace
                    .iter()
                    .map(|u| format!("{u:.2}"))
                    .collect::<Vec<_>>()
                    .join(" ")
            );
        }
        verdicts.push(verdict);
    }

    // Neighbour independence: the same rows batched in two halves, each row
    // held to its full-batch stream on the same terms — bit for bit while
    // both batches run it on the same route; a row left alone in one of them
    // takes the `m = 1` route there, and from that token on the pair is
    // tier 2.
    let independence: Option<Vec<(usize, RowVerdict, String)>> =
        if ctx.spec.check_independence && n >= 2 {
            let half = n / 2;
            let lo: Vec<usize> = (0..half).collect();
            let hi: Vec<usize> = (half..n).collect();
            let a = run_batched("batched-lo", &lo)?;
            let b = run_batched("batched-hi", &hi)?;
            let mut out = Vec::with_capacity(n);
            for (arm, rows) in [(&a, &lo), (&b, &hi)] {
                for (k, &i) in rows.iter().enumerate() {
                    let (full, part) = (&batched.streams[i], &arm.streams[k]);
                    let v = judge(i, (full, passes_of(&batched, i)), (part, passes_of(arm, k)));
                    let at = full.iter().zip(part).take_while(|(x, y)| x == y).count();
                    let line = v.describe(
                        (&format!("B={n}"), full.len()),
                        (&format!("B={}", rows.len()), part.len()),
                        at,
                    );
                    if v != RowVerdict::Identical {
                        eprintln!("[independence: row {i}: {line}]");
                    }
                    out.push((i, v, line));
                }
            }
            let differs = out
                .iter()
                .filter(|(_, v, _)| matches!(v, RowVerdict::Differs(_)))
                .count();
            eprintln!(
                "[independence: rows re-batched as {half} + {}: {}]",
                n - half,
                if out.iter().all(|(_, v, _)| *v == RowVerdict::Identical) {
                    "IDENTICAL".to_string()
                } else if differs == 0 {
                    "tier 2 only (the partings above re-derive as near-ties)".to_string()
                } else {
                    format!("DIFFERS on {differs} row(s)")
                }
            );
            out.sort_by_key(|(i, _, _)| *i);
            Some(out)
        } else {
            None
        };
    let independent = independence
        .as_ref()
        .map(|v| v.iter().all(|(_, v, _)| *v == RowVerdict::Identical));
    let failures: Vec<String> = verdicts
        .iter()
        .enumerate()
        .filter(|(_, v)| matches!(v, RowVerdict::Differs(_)))
        .map(|(i, v)| {
            format!(
                "seq {i} against solo: {}",
                v.describe(
                    ("solo", solo.streams[i].len()),
                    ("batched", batched.streams[i].len()),
                    common_prefix[i]
                )
            )
        })
        .chain(
            independence
                .iter()
                .flatten()
                .filter(|(_, v, _)| matches!(v, RowVerdict::Differs(_)))
                .map(|(i, _, line)| format!("seq {i} across the re-batch: {line}")),
        )
        .collect();
    // A bit gate under the `fast` contract certifies nothing: its arms may
    // move a row's bits by design (architecture.md §2.3.5b), so the rows
    // gate reads `unverified` there and passes only under `--profile
    // deterministic`. A failure stays a failure under either.
    let fast = lloom_tensor::tuning::contract().is_fast();
    let gate = if !failures.is_empty() {
        "fail"
    } else if fast
        || verdicts
            .iter()
            .chain(independence.iter().flatten().map(|(_, v, _)| v))
            .any(|v| matches!(v, RowVerdict::Unverified(_)))
    {
        "unverified"
    } else {
        "pass"
    };
    if fast && failures.is_empty() {
        eprintln!(
            "[rows gate: unverified under the fast contract; --profile deterministic to certify]"
        );
    }

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
         per_seq_tokens={:?} per_seq_overrun={:?} solo_kernels={} per_seq_verdict={:?} \
         independence_verdict={:?} gate={gate}",
        ctx.spec.first,
        batched_opts.lookahead,
        ctx.spec.prefill_chunk,
        ctx.spec.prefill_tile,
        ctx.spec.stagger,
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
        kernels.name(),
        verdicts.iter().map(RowVerdict::name).collect::<Vec<_>>(),
        independence
            .as_ref()
            .map(|v| v.iter().map(|(_, v, _)| v.name()).collect::<Vec<_>>()),
    );
    gate_exit(&failures, ctx.spec.fail_on_gate)
}

/// The ChatML user-turn wrapping.
fn user_turn_text(p: &str) -> String {
    format!("<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n")
}

#[cfg(test)]
mod tests {
    use super::{
        argmax, batched_passes, gate_exit, others_above, read_or_na, row_verdict, seam_crossed,
        ulp_at, ulps_apart, Parting, Pass, RowVerdict, SoloKernels,
    };
    use anyhow::Result;

    /// A replayed parting at `at` where `above` tokens besides each path's own
    /// beat the other arm's, both streams reproduced.
    fn parting(at: usize, above: (usize, usize)) -> Parting {
        Parting {
            at,
            a_token: 7,
            b_token: 9,
            a_left: None,
            b_left: None,
            copies_differ: false,
            a_logits: (12.5, 12.4375),
            b_logits: (12.4375, 12.5),
            a_others_above: above.0,
            b_others_above: above.1,
            delta_at: 0.0625,
            delta_before: 0.0625,
            ulps: 4.0,
            trace: vec![4.0; at + 1],
        }
    }

    const SOLO: [u32; 5] = [1, 2, 3, 7, 4];
    const BATCHED: [u32; 5] = [1, 2, 3, 9, 5];

    fn never(_: usize) -> Result<Parting> {
        panic!("the replay must not run for this verdict")
    }

    /// Token `t`'s forward is the step that streamed token `t - 1` under the
    /// lookahead, and the next step's kept rows without it; a ragged step or
    /// a short count is not re-derivable.
    #[test]
    fn a_batched_rows_passes_are_read_off_its_live_counts() {
        let rows = [4, 4, 3, 1];
        assert_eq!(
            batched_passes(&rows, 1, 4, 0),
            Some(vec![Pass::Rows(4), Pass::Rows(4), Pass::Rows(3)])
        );
        assert_eq!(
            batched_passes(&rows, 0, 4, 0),
            Some(vec![Pass::Rows(4), Pass::Rows(3), Pass::Rows(1)])
        );
        assert_eq!(batched_passes(&rows, 0, 5, 0), None);
        assert_eq!(batched_passes(&rows, 1, 4, 64), None);
    }

    /// The seam is crossed at the first token one arm computed on the `m = 1`
    /// route and the other did not; `Solo` and a one-row step are one route.
    #[test]
    fn the_seam_is_the_first_token_the_routes_differ() {
        let solo = [Pass::Solo; 4];
        let alone = [Pass::Rows(2), Pass::Rows(1), Pass::Rows(1), Pass::Rows(1)];
        assert_eq!(seam_crossed(&solo, &[Pass::Rows(4); 4], 4), Some(1));
        assert_eq!(seam_crossed(&solo, &alone, 4), Some(1));
        assert_eq!(seam_crossed(&[Pass::Rows(4); 4], &alone, 4), Some(2));
        assert_eq!(
            seam_crossed(&[Pass::Rows(4); 4], &[Pass::Rows(2); 4], 4),
            None
        );
        assert_eq!(seam_crossed(&solo, &[Pass::Rows(1); 4], 4), None);
    }

    /// Streams that agree are identical and replay nothing.
    #[test]
    fn identical_streams_replay_nothing() {
        let v = row_verdict(SoloKernels::Gemv, &SOLO, &SOLO, None, never);
        assert_eq!(v, RowVerdict::Identical);
    }

    /// On the GEMM kernel set every token is owed bit for bit: a parting fails
    /// without a replay, near-tie or not.
    #[test]
    fn on_the_gemm_set_any_parting_fails() {
        let passes = ([Pass::Solo; 4], [Pass::Rows(4); 4]);
        let v = row_verdict(
            SoloKernels::Gemm,
            &SOLO,
            &BATCHED,
            Some((&passes.0, &passes.1)),
            never,
        );
        assert!(matches!(v, RowVerdict::Differs(_)), "{v:?}");
    }

    /// Two arms that ran every forward before the parting on the same route
    /// owe each other the stream: a parting there is a failure whatever its
    /// logits look like — the re-batch whose halves both keep `m > 1`.
    #[test]
    fn a_parting_on_one_route_fails_without_a_replay() {
        let passes = ([Pass::Rows(4); 4], [Pass::Rows(2); 4]);
        let v = row_verdict(
            SoloKernels::Gemv,
            &SOLO,
            &BATCHED,
            Some((&passes.0, &passes.1)),
            never,
        );
        match v {
            RowVerdict::Differs(why) => assert!(why.contains("same route"), "{why}"),
            v => panic!("{v:?}"),
        }
    }

    /// Across the seam a parting passes as tier 2 exactly when it replays as a
    /// near-tie: the replay is asked at the parting, and on each arm's path
    /// nothing but its own token may stand above the other arm's.
    #[test]
    fn across_the_seam_a_parting_passes_only_as_a_near_tie() {
        let passes = ([Pass::Solo; 4], [Pass::Rows(4); 4]);
        let judge = |above: (usize, usize)| {
            row_verdict(
                SoloKernels::Gemv,
                &SOLO,
                &BATCHED,
                Some((&passes.0, &passes.1)),
                |at| {
                    assert_eq!(at, 3, "the replay is asked for the first differing token");
                    Ok(parting(at, above))
                },
            )
        };
        assert_eq!(
            judge((0, 0)),
            RowVerdict::SeamTie {
                seam: 1,
                parting: parting(3, (0, 0))
            }
        );
        assert!(matches!(judge((1, 0)), RowVerdict::Differs(_)));
        assert!(matches!(judge((0, 3)), RowVerdict::Differs(_)));
    }

    /// A replay that did not reproduce an arm's stream, whose copies of a
    /// batched row disagreed, or whose paths drifted past the seam's reach,
    /// proves no tie.
    #[test]
    fn a_replay_that_left_its_stream_is_not_a_tie() {
        let passes = ([Pass::Solo; 4], [Pass::Rows(4); 4]);
        for p in [
            Parting {
                a_left: Some(2),
                ..parting(3, (0, 0))
            },
            Parting {
                b_left: Some(3),
                ..parting(3, (0, 0))
            },
            Parting {
                copies_differ: true,
                ..parting(3, (0, 0))
            },
            Parting {
                ulps: 17.0,
                ..parting(3, (0, 0))
            },
        ] {
            let v = row_verdict(
                SoloKernels::Gemv,
                &SOLO,
                &BATCHED,
                Some((&passes.0, &passes.1)),
                |_| Ok(p.clone()),
            );
            assert!(matches!(v, RowVerdict::Differs(_)), "{v:?}");
        }
    }

    /// What cannot be re-derived is reported as such, never as a pass or a
    /// failure; a stream that stops where the other goes on is a failure.
    #[test]
    fn the_unverifiable_and_the_truncated() {
        let passes = ([Pass::Solo; 4], [Pass::Rows(4); 4]);
        let v = row_verdict(SoloKernels::Gemv, &SOLO, &BATCHED, None, never);
        assert!(matches!(v, RowVerdict::Unverified(_)), "{v:?}");
        let v = row_verdict(
            SoloKernels::Gemv,
            &SOLO,
            &BATCHED,
            Some((&passes.0, &passes.1)),
            |_| anyhow::bail!("no slot"),
        );
        assert!(matches!(v, RowVerdict::Unverified(_)), "{v:?}");
        let v = row_verdict(SoloKernels::Gemv, &SOLO, &SOLO[..3], None, never);
        assert!(matches!(v, RowVerdict::Differs(_)), "{v:?}");
    }

    /// The ulp is the dtype's at the vector's scale: one f16 ulp in [16, 32)
    /// is 1/64, so 22.4531 against 22.4375 is one ulp apart.
    #[test]
    fn ulps_are_counted_at_the_logit_scale() {
        use lloom_tensor::DType;
        assert_eq!(ulp_at(22.4531, DType::F16), 1.0 / 64.0);
        assert_eq!(ulp_at(0.75, DType::F16), 1.0 / 2048.0);
        assert_eq!(ulp_at(3.0, DType::BF16), 1.0 / 64.0);
        let a = [22.453125, -3.0, 1.0];
        let b = [22.4375, -3.0, 1.0];
        assert_eq!(ulps_apart(&a, &b, DType::F16), 1.0);
    }

    /// The first maximum wins the argmax, so a token tied with the leader
    /// loses on the id and has nothing but the leader above it — the exact
    /// tie is a near-tie; a third token above it is not.
    #[test]
    fn a_tie_with_the_leader_leaves_nothing_else_above() {
        let v = [0.5, 2.0, 2.0, -1.0, 1.5];
        assert_eq!(argmax(&v), 1);
        assert_eq!(others_above(&v, 2, 1), 0);
        assert_eq!(others_above(&v, 4, 1), 1);
        assert_eq!(others_above(&v, 3, 1), 3);
        assert_eq!(others_above(&v, 1, 2), 0);
    }

    /// A failing gate exits 0 by default and carries its failures to the
    /// report; only the opt-in turns them into an error, and a passing gate
    /// exits 0 under it too.
    #[test]
    fn a_failing_gate_exits_zero_unless_asked_to_fail() {
        let failures = vec!["seq 0 against solo: DIFFERS at 181".to_string()];
        assert!(
            gate_exit(&failures, false).is_ok(),
            "the default reports a failing gate and exits 0"
        );
        let e = gate_exit(&failures, true).expect_err("the opt-in exits non-zero");
        let e = e.to_string();
        assert!(e.contains("1 row(s) broke"), "{e}");
        assert!(e.contains("seq 0 against solo: DIFFERS at 181"), "{e}");
        assert!(
            gate_exit(&[], true).is_ok(),
            "a passing gate exits 0 under the opt-in"
        );
        assert!(gate_exit(&[], false).is_ok());
    }

    /// The opt-in reaches the key from its flag, its legacy env name and its
    /// canonical one, the flag over the env.
    #[test]
    fn the_opt_in_is_a_flag_and_an_env_alias() {
        use clap::Parser;
        #[derive(Parser)]
        struct Cli {
            #[command(flatten)]
            bench: crate::options::BenchConfigArgs,
        }
        const ENV: &str = "LLOOM_FAIL_ON_GATE";
        let resolve =
            |argv: &[&str], var: &str, value: Option<&str>| -> std::result::Result<bool, String> {
                let mut full = vec!["lloom bench"];
                full.extend_from_slice(argv);
                let mut b = crate::options::BenchConfig::default();
                b.from_env_with(&|n| (n == var).then(|| value.map(str::to_string)).flatten())?;
                Cli::try_parse_from(full).unwrap().bench.apply(&mut b)?;
                Ok(b.fail_on_gate)
            };
        assert!(!resolve(&[], ENV, None).unwrap(), "off by default");
        assert!(resolve(&["--fail-on-gate"], ENV, None).unwrap());
        assert!(!resolve(&["--fail-on-gate=false"], ENV, None).unwrap());
        assert!(
            resolve(&[], ENV, Some("1")).unwrap(),
            "the env alias turns it on"
        );
        assert!(
            resolve(&[], "LLOOM_BENCH__FAIL_ON_GATE", Some("1")).unwrap(),
            "and the key's"
        );
        assert!(
            !resolve(&["--fail-on-gate=false"], ENV, Some("1")).unwrap(),
            "the flag wins over the env"
        );
        assert!(!resolve(&[], ENV, Some("0")).unwrap());
        let e = resolve(&[], ENV, Some("maybe")).expect_err("a malformed alias is refused");
        assert!(e.contains(ENV), "{e}");
    }

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
