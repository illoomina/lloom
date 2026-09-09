//! The decode loop — `Session::decode`, the baseline (K=1) body of
//! `design/docs/decode-driver-design.md`, stage 1.
//!
//! The loop names two seams and neither concrete arm: a [`Speculator`] for
//! *which* proposer (stage 1 admits only [`IdentityProposer`], the K=1 identity;
//! the verify body is stage 2 and is refused rather than approximated) and a
//! [`ReadbackStrategy`] for how a step feeds the next forward and learns EOS.
//! The harness selects both and never writes the body.
//!
//! # The invariant, and what it costs to hold
//!
//! The design's hot path *does not commit, does not block, does not read
//! back*: sample on the device, feed the ring, enqueue the forward, bump
//! counters. The one thing the loop must eventually learn per token is whether
//! it was EOS, and where that wait sits decides everything:
//!
//! - **`lookahead = 0`** waits for step `s`'s publish *before* encoding forward
//!   `s`. The cache never advances past an EOS, so the arm is safe for a
//!   multi-turn conversation on any cache — and the GPU idles from the publish
//!   to the first commit of the next forward, which is the non-GPU gap
//!   (§11.2b's 0.74 ms). The publish must be committed before the host can wait
//!   on it, so this arm commits per token whatever the budget says; the
//!   `commits` counter reports it.
//! - **`lookahead = 1`** encodes forward `s` first and waits for step `s`'s
//!   publish afterwards, while that forward runs. The GPU always has the next
//!   step queued, so the gap closes — and when step `s` turns out to be EOS,
//!   forward `s` has already been enqueued: **the cache is one position past
//!   the EOS**, reported as [`DecodeStats::overrun`]. A caller that continues
//!   the conversation from that cache needs a rewind the crown configuration
//!   (`KvAppend=Cat`, `ConvState=Shuffle`) cannot express, so this arm is for a
//!   run whose cache ends with it, or for a cache that can rewind (stage 2's
//!   `advance`/`resolve` window is the mechanism).
//!
//! The ordering across steps is candle's own fence chain and is unchanged by
//! the lookahead: forward `s+1`'s first encoder waits on the ring slot's last
//! writer (publish `s+1`), which waited on the argmax, which waited on forward
//! `s`'s logits — so the GPU runs the steps serially and only the *host* has
//! stopped standing between them. What the lookahead adds is that step `s+1`
//! is committed before step `s` retires.

use crate::generation::{
    CommitBudget, HotThread, HotThreadReport, LogitsProcessor, ReadbackCounters, ReadbackStrategy,
    Sampling,
};
use crate::nn::model::{BatchedCausalLM, CacheReset, CausalLM, Row};
use crate::nn::speculator::{
    readbacks, Anchor, Draft, NoProbe, Probes, RoundProbe, RoundTiming, SeqId, Session, Speculator,
    TargetProbe,
};
use lloom_tensor::{Device, Result, Tensor};

/// The K=1 identity proposer: drafts nothing, so the loop takes the baseline
/// body. Stage 1's only admitted family.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityProposer;

impl Speculator for IdentityProposer {
    type Probe = NoProbe;

    fn n_seq(&self) -> usize {
        1
    }

    fn max_draft(&self) -> usize {
        0
    }

    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        Ok(())
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        Ok(active.iter().map(|_| Draft::host(Vec::new())).collect())
    }

    fn observe(
        &mut self,
        _seq: SeqId,
        _probe: &<Self::Probe as TargetProbe>::Output,
        _accepted: &[u32],
    ) -> Result<()> {
        Ok(())
    }
}

/// The knobs of one decode run.
// Not `Copy`: `prefill_schedule` is an owned `Vec`. `DecodeOptions` is passed
// by reference everywhere (`opts: &DecodeOptions`), so this costs the hot path
// nothing; the two struct-update sites (`..opts.clone()`) clone explicitly.
#[derive(Debug, Clone)]
pub struct DecodeOptions {
    /// Tokens to produce at most; EOS may stop the run earlier. **`0` means
    /// UNBOUNDED** — run until EOS with no ceiling (the token-ring history wraps,
    /// so an unbounded run streams at any length). A positive value is a hard
    /// cap. Use [`Self::reached`]/[`Self::unbounded`] rather than comparing
    /// directly, so the `0`-is-unbounded convention is honoured everywhere.
    pub max_tokens: usize,
    /// How often the loop commits and prunes the in-flight queue.
    pub budget: CommitBudget,
    /// Forward passes the host may enqueue before it has learned the previous
    /// step's EOS: `0` or `1`. See the module docs for what `1` costs.
    pub lookahead: usize,
    /// **Diagnostic, and it perturbs the thing it measures**: drain the GPU
    /// before each sample so the sample term can be timed apart from the
    /// forward pass (§11.2a). Off for every published number. Under
    /// [`Session::decode_speculative`] it drains the GPU at the end of each
    /// phase of a round instead — draft, verify, observe — so
    /// [`DecodeStats::rounds`] charges each phase its own GPU tail, and takes
    /// the profiler's per-phase counters when it is on.
    pub sample_split: bool,
    /// How many forwards the host may run ahead of the GPU (the pacer's bound).
    /// **`0` means AUTO** — computed from the host/GPU balance measured at
    /// warmup ([`compute_in_flight_depth`]), floored at [`IN_FLIGHT_FLOOR`]. A
    /// positive value overrides it (the tunable knob, e.g. an env A/B). Set from
    /// [`Session::warmup`]'s return, or left 0 to fall back to the floor when no
    /// warmup ran (CPU, or warmup skipped).
    pub in_flight_depth: usize,
    /// The spark: how many prompt tokens the FIRST prefill forward takes.
    /// **`0` means the whole prompt in one pass** — the incumbent one-shot. A
    /// positive value splits the prompt into TWO forwards, the spark and then
    /// the whole remainder (not `prompt_len / ubatch` chunks: the weight sweep
    /// is per forward, so the bulk is fed once), the first through the SAME
    /// in-flight pacer the decode loop uses so the first token streams off the
    /// spark while the bulk runs. It is the TTFT-vs-throughput dial: a wider
    /// spark keeps the matmul units fuller and approaches one-shot as it
    /// reaches the prompt length. Honoured by [`Session::decode`] and by both
    /// speculative drivers ([`Session::decode_speculative`],
    /// [`Session::decode_speculative_pipelined`], which seed the drafter from
    /// each unit's probe as it lands); [`Session::decode_batched`] prefills
    /// one-shot. Correctness across the unit boundary needs the conv's
    /// left-context carry (`Config::conv_prefill_ctx`), which the streamed
    /// path turns on; a single unit is identical either way (a cold cache has
    /// no left context to carry). [`DecodeStats::prefill_units`] reports the
    /// forwards a run actually took.
    pub prefill_ubatch: usize,
    /// An explicit prefill schedule: the exact widths of the forwards the
    /// prompt is fed through, in order. **`None` (the default) keeps the
    /// incumbent `prefill_ubatch` ramp** (one-shot, or spark-then-remainder) —
    /// byte-identical to before this field existed. `Some(units)` overrides it:
    /// the prompt is fed as forwards of `units[0], units[1], …` tokens, each
    /// clamped to the remaining prompt, and any prompt left after the schedule
    /// is exhausted is covered by `seq_len == 1` forwards (the tail).
    ///
    /// The point is to keep every prefill forward on an *aligned* GEMM tile
    /// variant. `Model::forward` selects a tile by the row count `m` (= the
    /// forward's `seq_len`) and compiles an M-boundary-guarded variant unless
    /// `m % bm == 0` (`mlx_gemm::select_tile_config`, `align_m`); a one-shot
    /// forward at an arbitrary `m = prompt.len()` lands on whatever variant that
    /// `m` picks, frequently the slower guarded one on the largest forward of
    /// the request. A schedule of aligned buckets ({16, 32} on the 16-row tile,
    /// 64 on the 64-row tile) keeps each forward on the no-guard variant, and a
    /// `1` unit is the `seq_len == 1` decode path. The CLI's `--spark` builds
    /// this; see that flag.
    ///
    /// **Read once, in the prefill prologue** — the steady decode loop never
    /// consults it, so a schedule adds no per-token cost on the hot path.
    pub prefill_schedule: Option<Vec<usize>>,
    /// The **bulk tile** width the prefill remainder is chopped into, after the
    /// [`Self::prefill_schedule`] ramp is exhausted. `0` (the default) means the
    /// remainder is a `seq_len == 1` tail (one forward per leftover token — the
    /// pathological many-forwards case on a long prompt); a positive aligned
    /// width (16 / 32 / 64) chops the remainder into that-wide forwards, each on
    /// the no-guard GEMM variant, collapsing the forward count to
    /// `remainder / bulk`. Only meaningful with `prefill_schedule = Some`.
    pub prefill_bulk: usize,
    /// The pacer's in-flight bound **during prefill** — how many prefill
    /// forwards the host may run ahead of the GPU. `0` (the default) uses the
    /// same depth as decode ([`Self::pacer_depth`]). A positive value runs
    /// prefill deeper: a chunked / many-small-forward prefill is compute-bound
    /// and serializes at the decode depth of 2 (each forward waits on the one
    /// two back), so a deeper bound lets the forwards overlap. Restored to the
    /// decode depth before the step loop, so it has no effect on steady decode.
    pub prefill_pacer_depth: usize,
    /// Stream tokens through a caller-owned [`RingReader`] instead of the hot
    /// loop's commit-cadence drain. `false` (default) is the incumbent: on a
    /// ring arm the loop drains the device ring at each commit and hands the
    /// batch to `on_tokens` — which batches visibility into commit-sized bursts.
    /// `true` (ring arms only) means the loop does **not** drain or call
    /// `on_tokens` per commit at all: the ids already live on the device ring,
    /// the caller reads them per-token off a
    /// [`RingReader`](crate::generation::readback::RingReader) (no sync, no
    /// wait), and the loop materialises the host id vec **once** from the ring
    /// at end-of-run (for `DecodeOut::ids` and the sequence's token log). This
    /// removes the visibility serialization (tokens stream per-token, not
    /// per-commit-burst) AND the redundant per-commit host mirror — the ring is
    /// the record.
    ///
    /// No effect on the blocking arm (its ids are host-side per step; it streams
    /// per token already) or on [`Self::decode_until`]'s stop return (a caller
    /// that needs stop-strings under a reader thread signals the loop itself).
    pub stream_via_reader: bool,
    /// The TILE: the most prompt tokens one prefill forward may take
    /// (llama.cpp's `n_ubatch`). **`0` means unbounded** — every forward is
    /// as wide as the ramp or the schedule makes it, the incumbent. A
    /// positive value splits any wider forward into sequential forwards of
    /// at most this many tokens (the last one narrower), so the prompt's
    /// activation working set — §9.5n's `a·C² + b·C`, the class admission
    /// sums — is bounded by the tile rather than by the prompt, and the
    /// buffers one tile's forward released are what the next tile's forward
    /// reuses. Composes with the spark and the schedule: it bounds whatever
    /// forwards they name. Read by every prefill this crate runs
    /// ([`Session::decode`], [`Session::prefill`], [`Session::decode_batched`])
    /// and by neither speculative driver, whose seed is `prefill_streamed`'s
    /// two forwards. A tiled prefill onto a warm cache needs the conv's
    /// left-context carry (`Config::conv_prefill_ctx`), as any multi-forward
    /// prefill does.
    pub prefill_tile: usize,
    /// The CHUNK under [`Session::decode_batched`]: the most prompt tokens one
    /// pending row is prefilled by per batched step, interleaved with the
    /// live rows' decode (vLLM's chunked prefill, its token budget). **`0`
    /// means every prompt is prefilled whole before the loop** — the
    /// incumbent, no interleave. A positive value admits the rows' prompts a
    /// chunk per step, one row at a time in order, each row joining the batch
    /// at the step after its last chunk; a row's prompt KV is built
    /// incrementally and its stream is what a whole prefill gives it. The
    /// time-sliced driver's chunk is the caller's — it feeds [`Session::prefill`]
    /// per visit — and the single-sequence loop has no other sequence to
    /// interleave with, so neither reads this.
    pub prefill_chunk: usize,
}

/// One speculative round of [`Session::decode_speculative`], for the per-round
/// breakdown a B=1 verdict rests on.
#[derive(Debug, Clone, Default)]
pub struct RoundStats {
    /// Tokens the verify returned, the bonus included — before the EOS and
    /// `max_tokens` cuts, so the round's work is counted whole.
    pub accepted: usize,
    /// The three phases' brackets; see [`RoundTiming`].
    pub timing: RoundTiming,
    /// Host time from the run's first prefill to this round's start, so rounds
    /// of different sequences can be laid on one timeline.
    pub start_ns: u64,
}

impl DecodeOptions {
    /// Whether the run has no token ceiling (`max_tokens == 0`) — it stops only
    /// on EOS. See [`Self::max_tokens`].
    pub fn unbounded(&self) -> bool {
        self.max_tokens == 0
    }

    /// Whether `produced` tokens have reached the cap. Always `false` when
    /// [`Self::unbounded`] (the `0`-is-unbounded convention), so the loops run
    /// until EOS. Use this instead of `produced >= max_tokens`.
    pub fn reached(&self, produced: usize) -> bool {
        self.max_tokens != 0 && produced >= self.max_tokens
    }

    /// The remaining token budget from `produced`, or `usize::MAX` when
    /// unbounded (for `min`-style clamps that want "no limit").
    pub fn remaining(&self, produced: usize) -> usize {
        if self.max_tokens == 0 {
            usize::MAX
        } else {
            self.max_tokens.saturating_sub(produced)
        }
    }

    /// The pacer depth: the explicit override if set, else the floor. (The
    /// AUTO path sets `in_flight_depth` from warmup; this reads it back with the
    /// `0 = auto -> floor` convention applied.)
    pub fn pacer_depth(&self) -> usize {
        if self.in_flight_depth == 0 {
            IN_FLIGHT_FLOOR
        } else {
            self.in_flight_depth
        }
    }

    /// Greedy defaults: `max_tokens`, per-token commit, no lookahead, no split,
    /// auto in-flight depth.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            budget: CommitBudget::per_token(),
            lookahead: 0,
            sample_split: false,
            in_flight_depth: 0,
            prefill_ubatch: 0,
            prefill_schedule: None,
            prefill_bulk: 0,
            prefill_pacer_depth: 0,
            stream_via_reader: false,
            prefill_tile: 0,
            prefill_chunk: 0,
        }
    }

    /// The spark width, or the whole prompt in one pass when `0` (the incumbent
    /// one-shot). See the `prefill_ubatch` field.
    pub fn prefill_ubatch(&self) -> usize {
        self.prefill_ubatch
    }

    /// The widest input one prefill forward can stage under these options:
    /// what a ring's input tile must hold for every forward to stage
    /// zero-copy. The tile when set (it bounds every forward), else the spark
    /// (one-shot's whole-prompt forward takes the allocating fallback either
    /// way), else the widest of the schedule's units and the bulk.
    pub fn max_prefill_width(&self) -> usize {
        if self.prefill_tile > 0 {
            return self.prefill_tile;
        }
        let widest_unit = self
            .prefill_schedule
            .as_ref()
            .and_then(|u| u.iter().copied().max())
            .unwrap_or(0);
        self.prefill_ubatch.max(widest_unit).max(self.prefill_bulk)
    }
}

/// What one decode run counted, for the `RESULT` line.
#[derive(Debug, Clone, Default)]
pub struct DecodeStats {
    /// Tokens returned, EOS excluded.
    pub produced: usize,
    /// Loop iterations, EOS step included.
    pub steps: u64,
    /// Submission time of the prompt's forward pass — encode, not execution.
    pub prefill_ns: u64,
    /// Forward passes the prompt took: 1 one-shot, 2 streamed (the spark, then
    /// the remainder — `prefill_ubatch` names the spark's width, not a chunk
    /// count), 0 when the run resumed from held logits. Read it off the run
    /// rather than deriving it from the option: a unit count that is only ever
    /// computed from `prompt_len / ubatch` is not a measurement.
    pub prefill_units: usize,
    /// Host time from the loop's exit to the run's return: the final drain
    /// (`ReadbackStrategy::drain_all`, which waits for the GPU) and the
    /// bookkeeping after it. Inside a caller's wall clock and outside the loop.
    pub tail_ns: u64,
    /// Compute time to the first token: prefill + the first decode step, stamped
    /// ONCE (at step 0) so it survives release without any per-step clock read —
    /// the fair, llama-bench-comparable TTFT (their synchronous `llama_decode` of
    /// the prompt → first logits), measured here WITHOUT a forced sync (the async
    /// pipeline is the design). Distinct from the CLI's `first_batch_at` (when the
    /// host DRAINS the first token, which adds the commit cadence) and
    /// `first_visible_at` (+ the consumer/print). 0 if the run produced nothing.
    pub ttft_compute_ns: u64,
    /// Per-step host time, one entry per returned token. Under `lookahead = 1`
    /// the wait sits after the forward's encode, so the entry is still one
    /// token's cadence and not a submit time.
    pub step_ns: Vec<u64>,
    /// Per-step sample time, only under `DecodeOptions::sample_split`.
    pub sample_ns: Vec<u64>,
    /// `flush_and_prune` calls made by the loop.
    pub commits: u64,
    /// The largest in-flight count `flush_and_prune` returned.
    pub max_in_flight: usize,
    /// Step markers taken: one per forward the loop committed at its end
    /// ([`IN_FLIGHT_STEPS`]).
    pub step_markers: u64,
    /// Markers the loop had to wait on before encoding the next step -- the
    /// host had run [`IN_FLIGHT_STEPS`] forwards ahead of the GPU. Zero on an
    /// arm the per-token EOS wait already paces.
    pub step_waits: u64,
    /// Forward passes enqueued past the last token this run returned — the
    /// positions the cache holds beyond the sequence's token log. One under
    /// `lookahead = 1` when EOS ended the run, up to a commit period under
    /// `EosRead::HostOnDrain`, and whatever a speculative round accepted past a
    /// mid-block EOS. Computed from the cache position, so a loop cannot
    /// under-report it. The session carries it as [`Session::overrun`] until
    /// [`Session::reconcile`] discards it.
    pub overrun: usize,
    /// The strategy's own counters.
    pub readback: ReadbackCounters,
    /// One entry per speculative round, empty on the baseline body.
    pub rounds: Vec<RoundStats>,
    /// The first [`FIRST_STEPS_TRACED`] steps' host time, by phase, on the
    /// `lookahead = 1` arm; empty on `lookahead = 0` and the speculative bodies.
    pub first_steps: Vec<StepPhases>,
    /// Every step's host time by phase, summed, on the `lookahead = 1` arm in a
    /// debug or `profile` build (zero in release, where the clock reads are
    /// compiled out): the steady-state split of the hot thread between encoding
    /// and waiting for the GPU. `phase_steps` is the number of steps summed.
    pub phase_totals: StepPhases,
    pub phase_steps: u64,
    /// The decode thread's scheduling class for this run: what it was, what
    /// the loop claimed (`generation::HotThread`), and whether that engaged.
    pub hot_thread: HotThreadReport,
    /// The prefill's dispatch census, `(kernel, count)`, under
    /// `CANDLE_METAL_PROFILE=1`; empty otherwise and off Metal.
    pub prefill_census: Vec<(String, u64)>,
    /// Step 0's dispatch census, on the same terms.
    pub step0_census: Vec<(String, u64)>,
    /// Under [`Session::decode_batched`], the number of live rows in each
    /// step this sequence took part in — one entry per `step_ns` entry. Empty
    /// on the single-sequence bodies.
    pub batch_rows: Vec<usize>,
}

/// The per-kernel dispatch counts so far, under `CANDLE_METAL_PROFILE=1`.
/// `None` when the census is off, which is every timed run.
fn census_snapshot() -> Option<std::collections::HashMap<String, u64>> {
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::metal::profile;
        if profile::enabled() {
            return Some(profile::snapshot().by_label.into_iter().collect());
        }
    }
    None
}

/// The dispatches between two census snapshots, sorted by count.
fn census_delta(
    before: &Option<std::collections::HashMap<String, u64>>,
    after: Option<std::collections::HashMap<String, u64>>,
) -> Vec<(String, u64)> {
    let (Some(before), Some(after)) = (before, after) else {
        return Vec::new();
    };
    let mut out: Vec<(String, u64)> = after
        .into_iter()
        .filter_map(|(k, n)| {
            let d = n - before.get(&k).copied().unwrap_or(0);
            (d > 0).then_some((k, d))
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// Where one decode step's host time went, for the first few steps of a run.
///
/// TTFT is `prefill_ns + step_ns[0]`, and the first step is not a steady-state
/// step: under `lookahead = 1` its EOS wait is where the prefill's GPU tail
/// lands, and its forward is the first at the decode shape. The four brackets
/// say which term carries a first-token cost; the sum is `step_ns[i]` minus
/// the loop's own bookkeeping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepPhases {
    /// `sample_and_feed`: the argmax and the ring publish, encode only.
    pub sample_ns: u64,
    /// The forward pass, encode only.
    pub forward_ns: u64,
    /// `apply_backpressure`: zero unless the host outran the GPU.
    pub backpressure_ns: u64,
    /// `eos_of_last`: the wait for this step's token to be published.
    pub eos_ns: u64,
}

impl StepPhases {
    /// Accumulate `other` into this, saturating.
    pub fn add(&mut self, other: &StepPhases) {
        self.sample_ns = self.sample_ns.saturating_add(other.sample_ns);
        self.forward_ns = self.forward_ns.saturating_add(other.forward_ns);
        self.backpressure_ns = self.backpressure_ns.saturating_add(other.backpressure_ns);
        self.eos_ns = self.eos_ns.saturating_add(other.eos_ns);
    }
}

/// How many leading steps [`DecodeStats::first_steps`] brackets.
pub const FIRST_STEPS_TRACED: u64 = 4;

/// What one [`Session::prefill`] took: the counters a scheduler reports per
/// visit. Zero on every field for an empty prompt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillOut {
    /// Forward passes the prompt took ([`DecodeStats::prefill_units`]).
    pub units: usize,
    /// Submission time of the forwards — encode, not execution.
    pub prefill_ns: u64,
    /// Step markers the pacer took, one per forward.
    pub step_markers: u64,
    /// Markers the pacer had to wait on before encoding a forward.
    pub step_waits: u64,
}

/// The outcome of one decode run.
#[derive(Debug, Clone)]
pub struct DecodeOut {
    /// The produced ids, EOS excluded.
    pub ids: Vec<u32>,
    /// Whether the run stopped on an EOS rather than on `max_tokens`.
    pub eos: bool,
    pub stats: DecodeStats,
}

/// One sequence's cursor through [`Session::decode_speculative_pipelined`].
struct Lane<P> {
    anchor_tok: u32,
    produced: usize,
    eos: bool,
    all_ids: Vec<u32>,
    pos_before: usize,
    prompt_len: usize,
    stats: DecodeStats,
    /// The verify in flight, if one is.
    pending: Option<P>,
    /// The in-flight round's draft and enqueue brackets, completed when its
    /// verify is read back.
    in_flight: RoundTiming,
    round_start_ns: u64,
    done: bool,
}

/// One sequence's cursor through [`Session::decode_batched`].
struct BatchLane {
    produced: usize,
    eos: bool,
    done: bool,
    all_ids: Vec<u32>,
    pos_before: usize,
    prompt_len: usize,
    /// The logits after the last fed token, when the row ended with every
    /// returned id fed (the lookahead arm at `max_tokens`).
    last_logits: Option<Tensor>,
    stats: DecodeStats,
    /// The step this row's prompt may start being fed (its arrival); `0` is
    /// the run's start.
    arrival: u64,
    /// Prompt tokens fed so far; the row joins the batch once this reaches
    /// `prompt_len` (or at once, on a resume with held logits).
    prompt_off: usize,
    /// Whether the row has joined the batch — its logits are in `logits`.
    joined: bool,
    /// The wall clock when the row's first prompt chunk was fed, for
    /// `stats.prefill_ns` (the submit time from the first chunk to the last).
    prefill_start: Option<std::time::Instant>,
}

/// Commit without waiting and prune the in-flight queue, on the backends that
/// have one. `None` where there is no queue to prune — a CPU run reports no
/// count, which is honest rather than zero (§14.1).
/// Close the installed dispatch executor's step, on the backend that has one.
///
/// The executor cannot see a token boundary from inside the encoder, so the
/// loop that owns the step declares it -- after the prefill, and after each
/// step's or round's last dispatch, so the window an ICB executor records is
/// one whole step and the same shape every time (`icb::IcbExecutor::end_step`).
/// A no-op with no executor installed, and off Metal.
fn end_executor_step(device: &Device) -> Result<()> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        m.end_executor_step()?;
    }
    let _ = device;
    Ok(())
}

/// Bracket one decode step's forward pass for the activation arena
/// (`DESIGN.md` §9.2): the ordinal that ties an allocation to a slot restarts
/// here, and an armed recording (`MetalDevice::arm_arena_recording`) observes
/// the step and installs the arena when it has seen enough. No-ops with no
/// arena installed or armed, and off Metal. The sampling before the forward and
/// the drain after it stay outside, as the harnesses that record by hand keep
/// them.
fn begin_decode_step(device: &Device) {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        m.begin_decode_step();
    }
    let _ = device;
}

fn end_decode_step(device: &Device) -> Result<()> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        m.finish_decode_step()?;
    }
    let _ = device;
    Ok(())
}

fn flush_and_prune(device: &Device) -> Result<Option<usize>> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return Ok(Some(m.flush_and_prune()?));
    }
    let _ = device;
    Ok(None)
}

/// The FLOOR on how far the host may run ahead of the GPU -- the minimum the
/// lookahead-1 dependency already implies (host waits for step `s`'s publish
/// after encoding `s + 1`, so `s - 1` retires before `s + 2`). The ACTUAL depth
/// is computed from the host/GPU balance at warmup ([`compute_in_flight_depth`])
/// and is `>= this`; a constant `2` was only ever right for a config where the
/// GPU step dwarfs the host encode (this model on this machine). See
/// [`compute_in_flight_depth`] for why a static value caps other models/machines.
pub const IN_FLIGHT_FLOOR: usize = 2;

/// Back-compat alias for [`IN_FLIGHT_FLOOR`]: callers that sized a buffer to the
/// old constant get the floor, which is a safe lower bound (the real depth is
/// `>=` it). Prefer `Session::measured_in_flight_depth` / `DecodeOptions::
/// in_flight_depth` for the actual computed depth.
pub const IN_FLIGHT_STEPS: usize = IN_FLIGHT_FLOOR;

/// The in-flight depth that keeps the GPU fed without the host running
/// pointlessly far ahead -- a *quantitative* bound, not an arbitrary constant.
///
/// The host must stay enqueued far enough that the GPU never starves waiting for
/// the next step: `depth = ceil(host_step / gpu_step) x safety`, floored at
/// [`IN_FLIGHT_FLOOR`] and capped by `mem_cap` (a memory-derived ceiling, so a
/// pathological ratio cannot pile up unbounded in-flight buffers). Both times
/// are MEASURED at warmup (a real decode step already runs there), so this
/// self-calibrates to the actual (model, machine, thermal) balance:
///
/// * **big model / slow GPU** (2.6B on M1 Max: gpu ~ 15 ms, host ~ 1.5 ms ->
///   ratio 0.1) -> `depth = 2` (the floor) -- identical to the old constant, so
///   no regression where the constant happened to be right.
/// * **small model** (350M: gpu ~ 2 ms, host ~ 1.5 ms -> ratio 0.75) -> `depth
///   ~ 3-4` -- the host runs deeper to feed the fast-draining GPU, which a
///   static `2` capped badly on the very same machine.
/// * **faster GPU** (M4 Max: higher bandwidth -> smaller gpu_step) -> deeper.
///
/// The old rationale ("a deeper queue buys the GPU nothing, the steps run
/// serially") holds ONLY when `gpu_step >> host_step`; when the GPU can outrun
/// the host, depth is exactly what stops it starving.
pub fn compute_in_flight_depth(host_step_ns: u64, gpu_step_ns: u64, mem_cap: usize) -> usize {
    if gpu_step_ns == 0 {
        return IN_FLIGHT_FLOOR;
    }
    // ceil(host/gpu) with a x2 safety, so a host step that occasionally spikes
    // toward the GPU step still has a queued forward to hide behind.
    let ratio_x2 = (2 * host_step_ns).div_ceil(gpu_step_ns) as usize;
    ratio_x2
        .max(IN_FLIGHT_FLOOR)
        .min(mem_cap.max(IN_FLIGHT_FLOOR))
}

/// The memory ceiling on in-flight depth: the most forwards' worth of retained
/// command buffers + their buffers the host may hold ahead of the GPU, so a
/// pathological host/GPU ratio (a tiny model on a huge GPU) cannot pile up
/// unbounded in-flight state (the 63-deep drain-arm tail this pacer replaced).
///
/// A generous constant for now — 64 steps is far above any measured healthy
/// depth (~2-4) yet bounds a runaway; a config where the computed ratio wants
/// more than this is one whose memory should be checked first. A future refinement
/// derives it from `recommendedMaxWorkingSetSize` / per-step in-flight bytes
/// (the §9.5 admission machinery), so it scales with the machine — left as the
/// seam, not built, since the ratio floors at 2 on the shipping config anyway.
pub fn in_flight_mem_cap(_device: &Device) -> usize {
    64
}

/// The host's bound on how far it runs ahead of the GPU, as a wait on the
/// command buffer that carried a step rather than on anything the GPU writes.
///
/// A marker is a committed command buffer; waiting on it is Metal's own
/// completion, which retires or errors and cannot wait for a value that is
/// never published. That is the whole difference from a cap spun on the ring's
/// generation counter: a counter a stale replay stops advancing is a livelock,
/// a command buffer that never retires is a reported error. When the GPU is
/// within the bound the marker is already complete and the wait is a status
/// read.
struct StepPacer {
    #[cfg(feature = "metal")]
    markers: std::collections::VecDeque<lloom_tensor::metal_backend::CommandMarker>,
    /// How many forwards the host may have in flight — the computed depth
    /// ([`compute_in_flight_depth`]), not a constant. `>= IN_FLIGHT_FLOOR`.
    depth: usize,
    taken: u64,
    waited: u64,
}

impl StepPacer {
    fn new(depth: usize) -> Self {
        let depth = depth.max(IN_FLIGHT_FLOOR);
        Self {
            #[cfg(feature = "metal")]
            markers: std::collections::VecDeque::with_capacity(depth),
            depth,
            taken: 0,
            waited: 0,
        }
    }

    /// Set the in-flight bound. Used to run the **prefill** at a different
    /// (deeper) depth than steady decode: prefill is compute-bound and many
    /// small forwards want to overlap, not serialize at the decode depth of 2.
    /// Restored to the decode depth before the step loop. Does not retire
    /// markers — only the bound the next `before_forward` enforces changes; a
    /// LOWER depth is honoured on the next `before_forward` (it retires down to
    /// the new bound then).
    fn set_depth(&mut self, depth: usize) {
        self.depth = depth.max(IN_FLIGHT_FLOOR);
    }

    /// Before encoding a forward: retire the oldest marker if the bound would
    /// otherwise be exceeded.
    fn before_forward(&mut self, device: &Device) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            while self.markers.len() >= self.depth {
                let oldest = self.markers.pop_front().expect("len >= depth");
                if !m.marker_complete(&oldest) {
                    self.waited += 1;
                }
                m.wait_marker(&oldest)?;
            }
        }
        let _ = device;
        Ok(())
    }

    /// After encoding a forward: commit what the step encoded and keep the
    /// marker. The cut this makes lands at the step boundary -- the one
    /// dependency edge every step has, the publish the next forward reads.
    fn after_forward(&mut self, device: &Device) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            self.markers.push_back(m.flush_marker()?);
            self.taken += 1;
        }
        let _ = device;
        Ok(())
    }
}

impl<'m, M: CausalLM> Session<'m, M> {
    /// Decode from `prompt` against this session's cache, advancing `pos`.
    ///
    /// The prompt's forward pass is one blocking-free submission; the loop
    /// then runs `sample → feed → forward` per token with the wait placed by
    /// `opts.lookahead` (module docs). Returns the ids, whether EOS ended the
    /// run, and the counters the harness asserts.
    ///
    /// Refused rather than approximated: a proposer with `max_draft() > 0`
    /// (the verify body is stage 2), a lookahead the strategy cannot use, a
    /// lookahead above 1, and a sampler the strategy cannot serve without a
    /// host round trip it would not report.
    /// `on_tokens` is called with the ids produced since the last call, at each
    /// commit and once at the end — the STREAMING hook, and it is streaming
    /// *without re-serializing*. On the ring arms the ids live device-side and a
    /// commit is where the GPU has already retired up to `fed` (the host is
    /// `lookahead` steps ahead, [`IN_FLIGHT_STEPS`] deep), so draining
    /// `ReadbackStrategy::take_ids` there reads work that is already done — one
    /// batched readback per commit, not the per-token round trip §11.5a deleted.
    /// A caller that does not stream passes a no-op; the tail drain at the end is
    /// unchanged either way. Streaming appears in commit-sized chunks, which is
    /// the price of not blocking: a token cannot be shown before the GPU makes
    /// it, and the batch is what keeps that observation off the per-token path.
    pub fn decode<S, R>(
        &mut self,
        spec: &mut S,
        readback: &mut R,
        sampler: &mut LogitsProcessor,
        prompt: &[u32],
        opts: &DecodeOptions,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<()>,
    ) -> Result<DecodeOut>
    where
        S: Speculator,
        R: ReadbackStrategy,
    {
        self.decode_until(spec, readback, sampler, prompt, opts, &mut |batch| {
            on_tokens(batch)?;
            Ok(true)
        })
    }

    /// [`Self::decode`] whose streaming hook can end the run: a `false` from
    /// `on_tokens` stops after the batch it was handed. The serving seam's stop
    /// strings are decided on detokenized text the loop never sees, so the
    /// consumer decides. A run stopped this way leaves no overrun — every
    /// returned id was fed before its batch was streamed — and holds the last
    /// logits, so it resumes like a run that reached `max_tokens`.
    pub fn decode_until<S, R>(
        &mut self,
        spec: &mut S,
        readback: &mut R,
        sampler: &mut LogitsProcessor,
        prompt: &[u32],
        opts: &DecodeOptions,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<DecodeOut>
    where
        S: Speculator,
        R: ReadbackStrategy,
    {
        if spec.max_draft() > 0 {
            lloom_tensor::bail!(
                "Session::decode: a proposer with max_draft {} needs the speculative body \
                 (`Session::decode_speculative`), whose bounds are `M: Speculative + \
                 Probes<S::Probe>` — stronger than this method's `M: CausalLM`. Call \
                 `decode_speculative` directly; the K=1 identity path stays here.",
                spec.max_draft()
            );
        }
        if opts.lookahead > 1 {
            lloom_tensor::bail!(
                "Session::decode: lookahead {} — the ring holds one input slot and the \
                 overrun past an EOS is bounded at one forward only for lookahead <= 1",
                opts.lookahead
            );
        }
        if opts.lookahead > 0 && !readback.admits_lookahead() {
            lloom_tensor::bail!(
                "Session::decode: lookahead {} on a strategy that learns EOS at feed time; \
                 there is nothing to overlap and the extra forward would only pollute the cache",
                opts.lookahead
            );
        }
        if self.seq.overrun > 0 {
            lloom_tensor::bail!(
                "Session::decode: the cache is {} positions past the sequence's last token \
                 (an earlier run's overrun); `Session::reconcile` or `Session::reset` first",
                self.seq.overrun
            );
        }
        // An empty prompt is a resume: continue from the logits the previous
        // run left, which exist iff it ended with the cache one prediction
        // ahead of the token log (`Session::prepare`'s `Resume`).
        let resume = prompt.is_empty();
        if resume && self.seq.last_logits.is_none() {
            lloom_tensor::bail!("Session::decode: empty prompt and no held logits to resume from");
        }
        readback.check_sampler(sampler.sampling())?;
        spec.begin(SeqId(0), prompt)?;

        let device = readback.device().clone();
        // The hot thread's scheduling class, held for the run and restored on
        // return (`generation::hot_thread`): two syscalls a generation, none a
        // step.
        let hot_thread = HotThread::claim();
        let mut stats = DecodeStats {
            step_ns: Vec::with_capacity(opts.max_tokens),
            sample_ns: Vec::with_capacity(if opts.sample_split {
                opts.max_tokens
            } else {
                0
            }),
            hot_thread: hot_thread.report(),
            ..DecodeStats::default()
        };

        // The in-flight bound, created BEFORE the prefill so the prompt-steps and
        // the generate-steps share ONE pacer — the pipeline spans prefill→decode
        // with no drain at the boundary, and the pacer's depth lets the host encode
        // the next unit while the GPU runs the current one (host↔GPU overlap). The
        // blocking arm's readback drains the queue every step, so its stream stays
        // the incumbent's; depth is the computed host/GPU balance
        // (`opts.in_flight_depth`, from warmup) or the floor — not a constant.
        let mut pacer = StepPacer::new(opts.pacer_depth());
        let pace = readback.loop_commits();
        // The caller streams off its own `RingReader`: the ring arm skips the
        // hot-loop drain and materialises the host id vec once at end-of-run.
        // `pace == loop_commits`, so this is the ring-arm condition. Only the ring
        // arms carry a device-resident ring to read; the blocking arm is
        // host-side per step and this is a no-op for it.
        let reader_owned = opts.stream_via_reader && pace;
        // Run prefill deeper than decode when asked: prefill is compute-bound and
        // a chunked / many-small-forward prefill serializes at the decode depth of
        // 2 (each forward waits on the one two back). A deeper bound lets the
        // prefill forwards overlap. Restored to the decode depth after the prefill
        // block, so steady decode is unaffected. `0` = same as decode.
        if opts.prefill_pacer_depth > 0 {
            pacer.set_depth(opts.prefill_pacer_depth);
        }

        // Prefill: the state machine STEPS OVER THE PROMPT before it steps over
        // its own generated tokens — same stepping, different input source. Rather
        // than one monolithic `forward(whole_prompt)` the host blocks on, the
        // prompt flows in as `prefill_ubatch`-wide units through the shared pacer:
        // each unit is staged into the ring's input buffer (zero-copy) and read as
        // a narrow, exactly as a decode step feeds its one token. A unit does NOT
        // sample (the tokens are given) and produces no output (nothing generated
        // yet); only the LAST unit's last-position logits seed the first generate
        // step. `prefill_ubatch == 0` means one unit over the whole prompt — the
        // incumbent one-shot, so the default path is byte-identical.
        //
        // Correctness across unit boundaries: a `seq_len>1` conv writer zero-pads
        // its left, exact only for a cold cache; a unit onto a WARM cache must
        // carry the previous window (`Config::conv_prefill_ctx`). The caller turns
        // that on when it streams (`prefill_ubatch>0`); a single unit is identical
        // either way (a cold cache has no left context to carry).
        let pos_before = self.seq.pos;
        let census_before = census_snapshot();
        let prefill_start = std::time::Instant::now();
        let mut prefill_units = 0usize;
        let mut logits = match self.seq.last_logits.take() {
            Some(held) if resume => held,
            _ => {
                let (last, units) =
                    self.prefill_forwards(readback, prompt, opts, &mut pacer, pace)?;
                prefill_units = units;
                last.expect("a non-empty prompt steps at least once")
            }
        };
        end_executor_step(&device)?;
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;
        stats.prefill_units = prefill_units;
        let census_prefill = census_snapshot();
        stats.prefill_census = census_delta(&census_before, census_prefill.clone());
        // Restore the decode in-flight bound: prefill's deeper bound (if any) is
        // done. A lower depth here is honoured on the next `before_forward`,
        // which retires markers down to it before the first decode step encodes.
        if opts.prefill_pacer_depth > 0 {
            pacer.set_depth(opts.pacer_depth());
        }

        let mut produced = 0usize;
        let mut since_commit = 0usize;
        let mut eos = false;
        // Every id the loop drains, streamed or not, accumulates here so the
        // returned `ids` is complete and the reconciliation below counts the
        // whole run. Draining for the stream advances the strategy's cursor, so
        // the final `take_ids` only sees the tail; the total is what matches
        // `produced`, not either part alone.
        let mut all_ids: Vec<u32> = Vec::with_capacity(opts.max_tokens);

        // With a lookahead the host encodes step `s+1` while the GPU runs step
        // `s`, so a command-buffer cut inside a step buys no overlap and costs
        // a seam: the step loop runs at the pipelined cut period (512, one
        // interior cut a step at most) and the default (50) is restored when
        // the loop ends. Without a lookahead the default is the overlap and
        // stays (`Device::pipelined_cut_period`; lloom #715).
        let _pipelined_period = (opts.lookahead >= 1).then(|| device.pipelined_cut_period());

        loop {
            if opts.reached(produced) {
                break;
            }
            // One autorelease pool per step, dropped last: the step's encoders
            // and command buffers are released here instead of never (the
            // release loop leaked ~200 autoreleased objects a token with no
            // pool on the thread -- `AutoreleaseScope` docs).
            let _pool = lloom_tensor::AutoreleaseScope::new();
            stats.steps += 1;
            // Under `CANDLE_METAL_PROFILE=1` only: tell the profile recorder
            // which step the command buffers committed from here on belong to,
            // so an idle gap between two buffers can be laid at a step (issue
            // #702). A cached-bool check and nothing else on every other run.
            #[cfg(feature = "metal")]
            {
                use lloom_tensor::kernels::metal::profile;
                if profile::enabled() {
                    profile::set_step(stats.steps);
                }
            }
            // Only the first `FIRST_STEPS_TRACED` steps record per-phase timings —
            // and only in a debug/profile build. In release `traced` is a constant
            // `false`, so every `traced.then(Instant::now())` below is `None` and NO
            // phase clock read (`mach_absolute_time()`) reaches even the first steps:
            // the release decode path is free of per-step instrumentation entirely.
            // `timed`: every step's phases are bracketed and summed into
            // `phase_totals` (the steady-state split); `traced`: the first few
            // are also kept individually. Both are constant `false` in release.
            #[cfg(any(debug_assertions, feature = "profile"))]
            let timed = true;
            #[cfg(not(any(debug_assertions, feature = "profile")))]
            let timed = false;
            let traced = timed && stats.steps <= FIRST_STEPS_TRACED;
            // The per-step median clock read — DEBUG/PROFILE ONLY. In a plain
            // release build this `mach_absolute_time()` (and the `step_ns.push`es it
            // feeds) is compiled OUT, so the steady decode path takes no per-step
            // clock read at all; release reports coarse tokens ÷ wall-clock instead
            // (measured host-side, off the loop). `any(debug_assertions, profile)`.
            #[cfg(any(debug_assertions, feature = "profile"))]
            let step_start = std::time::Instant::now();
            if opts.sample_split {
                device.synchronize()?;
            }
            // `sample_start` feeds only the phase trace (`t_sample`) and the
            // sample-split split; skip the clock read on the steady path.
            let sample_start = (timed || opts.sample_split).then(std::time::Instant::now);

            let feed = readback.sample_and_feed(sampler, &logits)?;
            produced += 1;
            // A step whose commit is due below closes a drain window: let the
            // strategy enqueue what that drain will read BEFORE the lookahead
            // forward, so it completes with the window rather than a step later
            // (`ReadbackStrategy::close_window`; a no-op on every arm but the
            // non-blocking batched-EOS one). The condition mirrors the commit
            // block's: every step under `lookahead = 0`, else the TTFT ramp or
            // the budget's period.
            if readback.loop_commits()
                && (opts.lookahead == 0
                    || produced <= opts.budget.period()
                    || opts.budget.due(since_commit + 1))
            {
                readback.close_window(produced <= opts.budget.period())?;
            }

            // The fair, llama-bench-comparable TTFT, stamped ONCE (step 0) — the
            // first output token is the argmax of the prefill's last-position
            // logits, sampled here, so `prefill_start.elapsed()` now is the host
            // time to submit prefill + sample the first token. No forced sync (the
            // async pipeline is the design — this is the honest difference from
            // llama-bench, which blocks on `llama_decode`), no drain cadence, no
            // consumer/print. ONE clock read per generation, so release-safe.
            if stats.steps == 1 {
                stats.ttft_compute_ns = prefill_start.elapsed().as_nanos() as u64;
            }

            // Whether this step's commit is the loop's to make at all: the
            // blocking arm's own readback commits and waits every step, and a
            // loop-side commit beside it would be a second command buffer per
            // token the incumbent never cut.
            let loop_commits = readback.loop_commits();
            let mut committed_this_step = false;

            if opts.lookahead == 0 {
                // The publish must reach the GPU before the host waits on it.
                // Structural on this arm, not a budget choice, and counted.
                if loop_commits {
                    // The completion marker for the window closing at this commit,
                    // encoded BEFORE the commit so its `slice_set` rides the
                    // committing buffer AFTER this step's input slot — the ordering
                    // the drain's clamp rests on (§3.5). A no-op on the arms that
                    // stamp per step or read back per step. `force_live` during the
                    // TTFT ramp so token 0 never waits `live_every` commits.
                    readback.publish_marker(produced <= opts.budget.period())?;
                    self.commit(&device, &mut stats)?;
                    since_commit = 0;
                    committed_this_step = true;
                }
                let is_eos = readback.eos_of_last()?;
                if let Some(s) = sample_start {
                    if opts.sample_split {
                        stats.sample_ns.push(s.elapsed().as_nanos() as u64);
                    }
                }
                if is_eos {
                    produced -= 1;
                    eos = true;
                    break;
                }
                if pace {
                    pacer.before_forward(&device)?;
                }
                begin_decode_step(&device);
                logits = self
                    .model
                    .forward(&feed, self.seq.pos, &mut self.seq.cache)?
                    .squeeze(0)?;
                end_decode_step(&device)?;
                self.seq.pos += 1;
                if pace {
                    pacer.after_forward(&device)?;
                }
            } else {
                if let Some(s) = sample_start {
                    if opts.sample_split {
                        stats.sample_ns.push(s.elapsed().as_nanos() as u64);
                    }
                }
                // ENQUEUE, then learn. The GPU has this forward queued before
                // it retires the previous one. The pacer bounds the enqueue:
                // an arm with no per-token wait would otherwise run ahead
                // without limit (#690's marker pacer, replacing the removed
                // `apply_backpressure` spin).
                //
                // The per-PHASE timings (sample / forward / backpressure / eos) are
                // recorded ONLY for the first `FIRST_STEPS_TRACED` steps (`traced`,
                // set at the loop top). Every clock read is a `mach_absolute_time()`
                // on the hot thread — a call the OS scheduler sees — so reading them
                // on all 2382 steps and discarding them past step 4 was pure
                // hot-path noise. Gated behind `traced`, so the steady path takes
                // NONE; the one clock read it still needs is `step_ns` (the median).
                let t_sample = sample_start.map_or(0, |s| s.elapsed().as_nanos() as u64);
                if pace {
                    pacer.before_forward(&device)?;
                }
                let t = timed.then(std::time::Instant::now);
                begin_decode_step(&device);
                logits = self
                    .model
                    .forward(&feed, self.seq.pos, &mut self.seq.cache)?
                    .squeeze(0)?;
                end_decode_step(&device)?;
                self.seq.pos += 1;
                let t_forward = t.map_or(0, |t| t.elapsed().as_nanos() as u64);
                // The in-flight bound AFTER the enqueue: retire the oldest
                // marker if the host would exceed `IN_FLIGHT_STEPS` ahead. This
                // is #690's pacer (Metal completion), which replaced the
                // unbounded counter-spin `apply_backpressure` livelocked on.
                // `backpressure_ns` reports what that bound cost this step.
                let t = timed.then(std::time::Instant::now);
                if pace {
                    pacer.after_forward(&device)?;
                }
                let t_backpressure = t.map_or(0, |t| t.elapsed().as_nanos() as u64);
                let t = timed.then(std::time::Instant::now);
                let is_eos = readback.eos_of_last()?;
                if timed {
                    let phases = StepPhases {
                        sample_ns: t_sample,
                        forward_ns: t_forward,
                        backpressure_ns: t_backpressure,
                        eos_ns: t.map_or(0, |t| t.elapsed().as_nanos() as u64),
                    };
                    stats.phase_totals.add(&phases);
                    stats.phase_steps += 1;
                    if traced {
                        stats.first_steps.push(phases);
                    }
                }
                if is_eos {
                    produced -= 1;
                    eos = true;
                    break;
                }
            }

            // The periodic machinery: the only place the lookahead arm
            // commits. A step that already committed before its wait is not
            // committed twice.
            //
            // EARLY-COMMIT RAMP (TTFT). The steady commit cadence is `period`
            // (e.g. 8), which batches the drain — good for throughput. But the
            // drain is the ONLY place a token streams to the consumer, so at
            // `period = 8` the FIRST token the user sees waits for step 8: measured
            // ~161 ms of user-visible first-token latency while the token was
            // computed at ~48 ms. That is the reported TTFT made moot. So for the
            // first `period` tokens, commit EVERY step — each reads finished work
            // (the drain never waits, §11.5a), so it costs a commit, not a stall —
            // and the first token streams as soon as it is computed. After the
            // ramp the steady period takes over and throughput is unchanged. The
            // ramp only SHORTENS early commits, so it cannot delay token 0; it is
            // TTFT-safe by construction. `period == 1` (per-token) already commits
            // every step, so the ramp is a no-op there.
            let in_ttft_ramp = produced <= opts.budget.period();
            let mut committed_now = committed_this_step;
            if loop_commits && !committed_this_step {
                since_commit += 1;
                if in_ttft_ramp || opts.budget.due(since_commit) {
                    // The completion marker for the window closing here, encoded
                    // before the commit so it rides the committing buffer after the
                    // window's slots (§3.5). No-op on the per-step arms.
                    // `force_live` during the TTFT ramp so early tokens stream at
                    // once regardless of the wake cadence.
                    readback.publish_marker(in_ttft_ramp)?;
                    self.commit(&device, &mut stats)?;
                    since_commit = 0;
                    committed_now = true;
                }
            }

            // Stream what is confirmed. This step's token has passed its EOS
            // check (an EOS `break`s above and is never streamed), so a drain
            // here carries only good ids.
            //
            // WHEN to drain is the whole of keeping this non-blocking. On the
            // ring arm (`loop_commits`) the ids live device-side and a readback
            // is a sync, so the drain is gated to commit steps: that is where the
            // GPU has retired up to `fed`, so the read is of finished work rather
            // than a wait, and it is one batched readback per commit instead of
            // the per-token round trip §11.5a deleted. On the blocking arm
            // (`!loop_commits`) the id is already host-side after this step's own
            // readback, so there is nothing to gate — it streams per token, which
            // is what that arm is.
            // When the caller streams through its own `RingReader`, the ring arm
            // does NOT drain here: the ids live on the device ring (read
            // per-token off the reader, no sync) and the host id vec is
            // materialised once at end-of-run. `produced` and EOS are per-step
            // signals (`produced += 1`, `eos_of_last`), independent of the drain,
            // so the loop is correct without it. The blocking arm ignores this
            // (its ids are host-side per step; nothing to defer). `reader_owned`
            // is hoisted above the loop.
            let stream_now = !reader_owned && if loop_commits { committed_now } else { true };
            if stream_now {
                let batch = readback.take_ids()?;
                if !batch.is_empty() {
                    // The host-side EOS check (`EosRead::HostOnDrain`): this arm
                    // does not learn EOS per token, so a drained batch may
                    // contain the EOS token and the overrun the loop enqueued
                    // after it. Scan here, BEFORE streaming, so `on_tokens` still
                    // only ever sees good ids (the invariant the per-token arms
                    // hold by `break`ing before they stream). On other arms
                    // `eos_in_drained` is `None` and the batch streams whole.
                    if let Some(k) = readback.eos_in_drained(&batch) {
                        // `batch[..k]` are good; `batch[k]` is EOS (excluded, as
                        // the per-token retraction excludes it); `batch[k+1..]`
                        // are overrun (discarded), and so is everything the
                        // strategy still holds undrained — told to it, so a later
                        // call on the same strategy does not drain them as new
                        // tokens. Reconcile `produced` down to the good count so
                        // it matches `all_ids`.
                        let good = &batch[..k];
                        if !good.is_empty() {
                            // The run is over either way; the hook's answer
                            // changes nothing past an EOS.
                            let _ = on_tokens(good)?;
                            all_ids.extend_from_slice(good);
                        }
                        readback.note_eos_truncation(batch.len() - k - 1);
                        produced = all_ids.len();
                        eos = true;
                        #[cfg(any(debug_assertions, feature = "profile"))]
                        stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                        break;
                    }
                    let go_on = on_tokens(&batch)?;
                    all_ids.extend_from_slice(&batch);
                    if !go_on {
                        #[cfg(any(debug_assertions, feature = "profile"))]
                        stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                        break;
                    }
                }
            }
            end_executor_step(&device)?;
            // The per-step median + the step-0 census: DEBUG/PROFILE ONLY. Release
            // takes neither clock read nor census on the steady path.
            #[cfg(any(debug_assertions, feature = "profile"))]
            {
                stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                if stats.steps == 1 {
                    stats.step0_census = census_delta(&census_prefill, census_snapshot());
                }
            }
        }

        // After the clock: EVERY id not yet drained — `drain_all`, not the
        // incremental `take_ids`, because the poll arm may still have the last
        // forwards in flight and its incremental drain refuses to read past what
        // the GPU has published. `drain_all` synchronises up to the final slot,
        // so the tail is complete; streamed too, so a caller sees the last
        // tokens, then folded into the full `ids`. A host-on-drain arm that
        // truncated at EOS above has already discarded its undrained slots, so
        // its tail is empty and this costs it nothing.
        //
        // The tail is scanned for EOS too, for `EosRead::HostOnDrain`: if the run
        // ended on `max_tokens` (not an EOS `break` above) the EOS may sit in the
        // never-committed tail, and streaming it whole would emit tokens past it.
        // On the per-token arms `eos_in_drained` is `None`, so the tail streams
        // whole as before.
        let tail_start = std::time::Instant::now();
        if reader_owned {
            // The reader streamed every token off the device ring already, and
            // the loop drained nothing, so `all_ids` is empty. Materialise the
            // host id vec ONCE here: `drain_all` synchronises and reads the whole
            // published range — `produced` good ids plus, on the poll arm, the
            // trailing EOS token (published before the loop broke) and, on a
            // deferred arm, any overrun. `produced` is the per-step count with
            // EOS already excluded, so the first `produced` ids are exactly the
            // good sequence, in order — trim to it. No `on_tokens` (the reader is
            // the stream). The count/EOS are authoritative from the loop.
            all_ids = readback.drain_all()?;
            all_ids.truncate(produced);
        } else {
            // After the clock: EVERY id not yet drained — `drain_all`, not the
            // incremental `take_ids`, because the poll arm may still have the last
            // forwards in flight and its incremental drain refuses to read past what
            // the GPU has published. `drain_all` synchronises up to the final slot,
            // so the tail is complete; streamed too, so a caller sees the last
            // tokens, then folded into the full `ids`. A host-on-drain arm that
            // truncated at EOS above has already discarded its undrained slots, so
            // its tail is empty and this costs it nothing.
            //
            // The tail is scanned for EOS too, for `EosRead::HostOnDrain`: if the run
            // ended on `max_tokens` (not an EOS `break` above) the EOS may sit in the
            // never-committed tail, and streaming it whole would emit tokens past it.
            // On the per-token arms `eos_in_drained` is `None`, so the tail streams
            // whole as before.
            let tail = readback.drain_all()?;
            if !tail.is_empty() {
                if let Some(k) = readback.eos_in_drained(&tail) {
                    let good = &tail[..k];
                    if !good.is_empty() {
                        let _ = on_tokens(good)?;
                        all_ids.extend_from_slice(good);
                    }
                    readback.note_eos_truncation(tail.len() - k - 1);
                    produced = all_ids.len();
                    eos = true;
                } else {
                    let _ = on_tokens(&tail)?;
                    all_ids.extend_from_slice(&tail);
                }
            }
        }
        if all_ids.len() != produced {
            lloom_tensor::bail!(
                "Session::decode: the strategy returned {} ids for {} produced tokens — the \
                 count and the history disagree",
                all_ids.len(),
                produced
            );
        }

        // The sequence's bookkeeping, from the cache position rather than from
        // the arm's own accounting: every returned id was fed before its
        // successor's forward, so the cache holds the prompt, the returned ids
        // and — on an arm that learns EOS late — the positions past them.
        let fed = self.seq.pos - pos_before - prompt.len();
        if fed < produced {
            lloom_tensor::bail!(
                "Session::decode: {fed} forwards ran for {produced} returned tokens — a \
                 returned token whose forward never ran"
            );
        }
        let overrun = fed - produced;
        self.seq.tokens.extend_from_slice(&all_ids);
        self.seq.overrun = overrun;
        // The last forward's logits predict the token after the last returned
        // one iff nothing overran; they are what a continuation of exactly this
        // sequence resumes from. Past an EOS they predict nothing useful, but
        // they are still the right logits for that position.
        self.seq.last_logits = if overrun == 0 { Some(logits) } else { None };
        stats.overrun = overrun;
        stats.produced = produced;
        stats.tail_ns = tail_start.elapsed().as_nanos() as u64;
        stats.step_markers = pacer.taken;
        stats.step_waits = pacer.waited;
        stats.readback = readback.counters();
        Ok(DecodeOut {
            ids: all_ids,
            eos,
            stats,
        })
    }

    fn commit(&self, device: &Device, stats: &mut DecodeStats) -> Result<()> {
        stats.commits += 1;
        if let Some(in_flight) = flush_and_prune(device)? {
            stats.max_in_flight = stats.max_in_flight.max(in_flight);
        }
        Ok(())
    }

    /// Feed `prompt` through the model as the forwards [`prefill_widths`]
    /// names, each paced by `pacer` exactly as a decode step is, and return
    /// the last forward's last-position logits and the number of forwards.
    ///
    /// The state machine steps over the prompt before it steps over its own
    /// generated tokens — same stepping, different input source. A unit is
    /// staged into the strategy's input (zero-copy on the ring arm when it
    /// fits the input tile, a fresh tensor otherwise), does not sample (the
    /// tokens are given) and produces no output; only the last unit's logits
    /// seed the first generated token. One unit over the whole prompt is the
    /// incumbent one-shot pass, so the default is byte-identical.
    ///
    /// Correctness across unit boundaries is the conv writer's left-context
    /// carry (`Config::conv_prefill_ctx`): a `seq_len > 1` pass onto a warm
    /// cache zero-pads its left without it. The caller turns it on whenever
    /// more than one forward can run; a single unit is identical either way.
    fn prefill_forwards<R>(
        &mut self,
        readback: &mut R,
        prompt: &[u32],
        opts: &DecodeOptions,
        pacer: &mut StepPacer,
        pace: bool,
    ) -> Result<(Option<Tensor>, usize)>
    where
        R: ReadbackStrategy,
    {
        let device = readback.device().clone();
        // The widths of the forwards this prompt is fed through, in order:
        // the incumbent ramp (one-shot, or spark then remainder), the explicit
        // schedule with its tail, and the tile bound over either. Computed
        // once, in the prologue; the steady loop never reads it.
        let widths = tile_widths(
            prefill_widths(
                prompt.len(),
                opts.prefill_ubatch(),
                &opts.prefill_schedule,
                opts.prefill_bulk,
            ),
            opts.prefill_tile,
        );
        let mut last_logits = None;
        let mut units = 0usize;
        let mut off = 0usize;
        for w in widths {
            // Drains this unit's autoreleased encoders and command buffers
            // when the iteration ends (`AutoreleaseScope` docs).
            let _pool = lloom_tensor::AutoreleaseScope::new();
            let chunk = &prompt[off..off + w];
            let input = readback.stage_prompt_chunk(chunk)?;
            if pace {
                pacer.before_forward(&device)?;
            }
            let out = self
                .model
                .forward(&input, self.seq.pos, &mut self.seq.cache)?
                .squeeze(0)?;
            if pace {
                pacer.after_forward(&device)?;
            }
            self.seq.pos += w;
            self.seq.tokens.extend_from_slice(chunk);
            last_logits = Some(out);
            off += w;
            units += 1;
        }
        Ok((last_logits, units))
    }

    /// Prefill `prompt` onto this sequence and produce nothing: the logical
    /// CHUNK seam. The cache and the token log advance by the prompt and the
    /// last position's logits are held, so a later [`Self::decode`] with an
    /// empty prompt resumes from them (`Session::prepare` reports `Resume`),
    /// and a later call with more of the sequence extends it. A scheduler
    /// that interleaves one request's prompt with other requests' decode
    /// feeds each request `≤ budget` prompt tokens per visit through this and
    /// visits the others between — the prompt's KV is built incrementally and
    /// the final state is what one pass over the prompt leaves, given the
    /// conv writer's left-context carry.
    ///
    /// The forwards taken are [`prefill_widths`]'s for `opts` (the tile bound
    /// applies here too), paced by their own [`StepPacer`]; nothing waits for
    /// the GPU at the end, so the next visit's work queues behind this one.
    /// Refused on an unreconciled overrun, as `decode` is. An empty prompt is
    /// a no-op that takes no forward.
    pub fn prefill<R>(
        &mut self,
        readback: &mut R,
        prompt: &[u32],
        opts: &DecodeOptions,
    ) -> Result<PrefillOut>
    where
        R: ReadbackStrategy,
    {
        if self.seq.overrun > 0 {
            lloom_tensor::bail!(
                "Session::prefill: the cache is {} positions past the sequence's last token \
                 (an earlier run's overrun); `Session::reconcile` or `Session::reset` first",
                self.seq.overrun
            );
        }
        if prompt.is_empty() {
            return Ok(PrefillOut::default());
        }
        let device = readback.device().clone();
        // The prefill's own in-flight bound when one is set, as `decode` runs
        // its prefill block at (`prefill_pacer_depth`); else the decode depth.
        let depth = if opts.prefill_pacer_depth > 0 {
            opts.prefill_pacer_depth
        } else {
            opts.pacer_depth()
        };
        let mut pacer = StepPacer::new(depth);
        let pace = readback.loop_commits();
        let start = std::time::Instant::now();
        let (last, units) = self.prefill_forwards(readback, prompt, opts, &mut pacer, pace)?;
        end_executor_step(&device)?;
        // The held logits are the resumable state: a decode with an empty
        // prompt continues from them, and a further prefill discards them.
        self.seq.last_logits = last;
        Ok(PrefillOut {
            units,
            prefill_ns: start.elapsed().as_nanos() as u64,
            step_markers: pacer.taken,
            step_waits: pacer.waited,
        })
    }

    /// Prefill `prompt` in the units `ubatch` names and seed `spec`'s sequence
    /// `seq` from every one of them: `begin`, then per unit a forward, its
    /// probe taken and `observe`d with an empty accepted tail — the drafter
    /// keys the unit's positions into its own context at the offset it
    /// carries — so the drafter's context covers the whole prompt when the
    /// last unit's logits come back. Returns those logits and the LAST unit's
    /// probe, still unobserved: the caller reads the argmax back (the one host
    /// wait of a seed, kept out of here so a driver can prefill N sequences
    /// before it waits on any) and observes with it as the accepted tail.
    ///
    /// The units are `decode`'s — `ubatch` first, then the remainder — and
    /// `0` is the incumbent one-shot. A target that overwrites its probe per
    /// forward would seed the drafter from the last unit alone if a unit were
    /// skipped; `probe_drops` is checked across the seed so that is a refusal
    /// rather than a lower accept rate nothing else sees (§10.2w). The third
    /// element is the number of forwards taken (`DecodeStats::prefill_units`).
    fn prefill_streamed<S>(
        &mut self,
        spec: &mut S,
        seq: SeqId,
        prompt: &[u32],
        ubatch: usize,
        device: &Device,
    ) -> Result<(Tensor, <S::Probe as TargetProbe>::Output, usize)>
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
        spec.begin(seq, prompt)?;
        // Whatever an earlier forward left in the probe slots — a warmup's, a
        // previous request's — predates this seed and is nobody's positions:
        // drained here, so the count below can move only between two probes
        // of THIS seed, which is the defect the refusal is for.
        let _stale = self.model.take_probe(&mut self.seq.cache)?;
        let drops_before = self.model.probe_drops(&self.seq.cache);
        let mut off = 0usize;
        let mut units = 0usize;
        let mut step = if ubatch == 0 { prompt.len() } else { ubatch };
        loop {
            let _pool = lloom_tensor::AutoreleaseScope::new();
            let w = step.min(prompt.len() - off);
            let chunk = &prompt[off..off + w];
            let input = Tensor::new(chunk, device)?.unsqueeze(0)?;
            let logits = self
                .model
                .forward(&input, self.seq.pos, &mut self.seq.cache)?;
            self.seq.pos += w;
            self.seq.tokens.extend_from_slice(chunk);
            off += w;
            units += 1;
            step = prompt.len();
            let probe = self.model.take_probe(&mut self.seq.cache)?;
            let drops = self.model.probe_drops(&self.seq.cache);
            if drops != drops_before {
                lloom_tensor::bail!(
                    "Session: the target dropped {} probe set(s) during the prefill — a forward \
                     ran between two probes, so the drafter would be seeded from part of the \
                     prompt (design/docs/speculation.md §10.2w)",
                    drops - drops_before
                );
            }
            if off < prompt.len() {
                spec.observe(seq, &probe, &[])?;
            } else {
                return Ok((logits, probe, units));
            }
        }
    }

    /// Decode from `prompt` with a real speculative proposer — stage 2 of
    /// `design/docs/decode-driver-design.md` ("the speculative body").
    ///
    /// This is the arm [`Self::decode`] refuses: a proposer with
    /// `max_draft() > 0` (DSpark). The bound is stronger than `decode`'s
    /// `M: CausalLM` — `M: Speculative` for the `forward_all`/`advance`/`resolve`
    /// rollback window and `M: Probes<S::Probe>` for the pairing contract
    /// (`speculator.rs`: an unsatisfiable target+drafter pair is a **compile
    /// error**, not a runtime bail). It is a separate method rather than a branch
    /// inside `decode` for exactly that reason: the K=1 identity path stays
    /// callable with only `CausalLM`, so widening the bound there would refuse a
    /// caller that has no drafter at all (`design/docs/decode-driver-design.md`,
    /// "two loop bodies, not one").
    ///
    /// # The round, and why it is output-identical to greedy
    ///
    /// Per round: draft up to `K` tokens, run the target's `forward_all` over
    /// `[anchor, draft…]`, accept the longest prefix whose argmax matches the
    /// draft **plus the target's own next token** (the bonus), roll the cache
    /// back to the accepted length, re-key the drafter from the accepted rows.
    /// The whole round is [`Self::speculative_step`], which is `draft →
    /// verify_and_probe → observe` — the verify accepts a draft token iff it
    /// equals `argmax(target logits)`, so the emitted stream is **byte-identical
    /// to non-speculative greedy decoding by construction** (`lfm2.rs`
    /// `verify_step`'s "greedy needs no epsilon"; DESIGN §10.2n). A drafter that
    /// proposes garbage costs acceptance rate and nothing else; it cannot change
    /// the output. That is the property the correctness gate asserts.
    ///
    /// # What this owns that `speculative_step` does not
    ///
    /// `speculative_step` runs one round across a batch and returns the accepted
    /// ids; it threads no decode state. The loop owns the decode cursor —
    /// [`Anchor`] (`last` = the last committed token, `pos` = the target position
    /// the window starts at), advancing `pos` by the accepted length and `last`
    /// to the accepted tail each round — plus EOS detection, the `max_tokens`
    /// cap (which may fall MID-block, so a round publishes 1..K+1 tokens), and
    /// the streaming hand-off.
    ///
    /// # Streaming, EOS and the cache
    ///
    /// The accepted tokens are already host-side — `verify_step` reads the argmax
    /// back to compare it (`design/docs/decode-driver-design.md`: the
    /// speculative body's one readback per round is inherent, not the ring's
    /// device-resident path). So this arm does NOT feed through the
    /// [`ReadbackStrategy`]'s device seam; it streams the accepted ids straight
    /// to `on_tokens` and tests EOS host-side against `eos_ids`. The strategy is
    /// still taken so the signature matches [`Self::decode`] and a future arm can
    /// publish the accepted block through the ring for a B>1 driver (#535's "the
    /// ring's real consumer is the speculation loop"); today it is used only for
    /// its device handle and its commit cadence.
    ///
    /// An EOS inside a block ends the run at that token: the tokens before it are
    /// streamed, the EOS is not. The accepted prefix itself ends at the EOS
    /// (`accept_prefix` takes `eos_ids`), so the cache keeps exactly the
    /// streamed tokens' positions, the drafter is keyed with no position past
    /// the EOS, and the run leaves no overrun — the conservative K=1 arm's
    /// shape, not the lookahead's one-past-EOS.
    ///
    /// # The drafter context is seeded from the prompt before round 0
    ///
    /// The prefill's probe (`Probes::take_probe`, every prompt position) is
    /// handed to `observe` before the first draft, with the prefill's argmax as
    /// the accepted tail — the `lfm2-dspark-verify` example's `--prefill-ctx
    /// all`, which is what llama.cpp's `speculative.cpp::process` does. An
    /// earlier form seeded lazily (round 0 drafted with an empty context and
    /// the verify taps keyed it) on the claim that this cost only round-0
    /// acceptance; measured, the drafter's context then never contained the
    /// prompt at all, and acceptance on prompts a draft copies from fell by up
    /// to half (`design/docs/speculation.md` §10.2w). Output-identical either
    /// way: the target verifies every token.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_speculative<S, R>(
        &mut self,
        spec: &mut S,
        readback: &mut R,
        _sampler: &mut LogitsProcessor,
        prompt: &[u32],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<()>,
    ) -> Result<DecodeOut>
    where
        S: Speculator,
        M: Probes<S::Probe>,
        R: ReadbackStrategy,
    {
        if spec.max_draft() == 0 {
            lloom_tensor::bail!(
                "Session::decode_speculative: the proposer has max_draft 0 (the K=1 identity); \
                 it belongs on `Session::decode`, which has the device-resident ring path this \
                 method deliberately does not."
            );
        }
        if eos_ids.is_empty() {
            lloom_tensor::bail!(
                "Session::decode_speculative: eos_ids is empty, so the loop would never terminate"
            );
        }
        if prompt.is_empty() {
            lloom_tensor::bail!("Session::decode_speculative: empty prompt");
        }
        if self.seq.overrun > 0 {
            lloom_tensor::bail!(
                "Session::decode_speculative: the cache is {} positions past the sequence's \
                 last token (an earlier run's overrun); `Session::reconcile` or \
                 `Session::reset` first",
                self.seq.overrun
            );
        }
        // A speculative run never leaves resumable logits (the verify reads
        // its argmax back and keeps no per-position logits), so a held one from
        // an earlier K=1 run is stale here.
        self.seq.last_logits = None;

        let device = readback.device().clone();
        let seq = SeqId(0);
        let hot_thread = HotThread::claim();
        let mut stats = DecodeStats {
            step_ns: Vec::with_capacity(opts.max_tokens),
            hot_thread: hot_thread.report(),
            ..DecodeStats::default()
        };

        // Prefill, in `opts.prefill_ubatch()`'s units, each unit's probe keyed
        // into the drafter as it lands (`prefill_streamed`); the last unit's
        // argmax is the first anchor — the token whose logits produced the
        // first draft — exactly as the `lfm2-dspark-verify` example takes
        // `logits.argmax` when it does not hold the last prompt token back.
        let pos_before = self.seq.pos;
        let prefill_start = std::time::Instant::now();
        let (logits, prefill_probe, prefill_units) =
            self.prefill_streamed(spec, seq, prompt, opts.prefill_ubatch(), &device)?;
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;
        stats.prefill_units = prefill_units;

        let mut anchor_tok = logits
            .squeeze(0)?
            .argmax(lloom_tensor::D::Minus1)?
            .to_scalar::<u32>()?;

        // The last unit's probe, with the prefill's own argmax as the accepted
        // tail (the block anchor) — the same state the first round would see
        // after a fully-accepted prompt.
        spec.observe(seq, &prefill_probe, std::slice::from_ref(&anchor_tok))?;
        end_executor_step(&device)?;

        // The prefill's own argmax is the first generated token — it is what
        // non-speculative greedy would emit as token 0 (the prefill logits' argmax),
        // and the verify loop's anchor. Stream it and count it, unless it is EOS.
        let mut produced = 0usize;
        let mut eos = false;
        let mut all_ids: Vec<u32> = Vec::with_capacity(opts.max_tokens);

        if eos_ids.contains(&anchor_tok) {
            // The model wanted to stop immediately (an empty reply). Nothing is
            // produced and the run ends before any round.
            eos = true;
        } else if !opts.reached(produced) {
            on_tokens(std::slice::from_ref(&anchor_tok))?;
            all_ids.push(anchor_tok);
            produced += 1;
        }

        // The verify window is `[anchor, draft…]` at this position; `pos` advances
        // by the accepted length each round (`lfm2-dspark-verify`'s `index_pos`).
        let mut since_commit = 0usize;

        while !eos && !opts.reached(produced) {
            let _pool = lloom_tensor::AutoreleaseScope::new(); // per window, see the decode loop
            stats.steps += 1;
            let step_start = std::time::Instant::now();

            // One speculative round: draft → verify (accept longest argmax-match
            // prefix + bonus, roll back) → observe (re-key drafter from the
            // accepted rows). `p_min = 0.0` disables the confidence early-exit
            // (`speculative.cpp:796`; the driver owns the knob). `speculative_step`
            // returns the accepted tokens, INCLUDING the bonus, and has already
            // advanced the drafter's own anchor via `observe`.
            let anchor = Anchor {
                last: anchor_tok,
                pos: self.seq.pos,
            };
            // The census is one round's worth of allocation; take it once,
            // past the warm-up rounds, so the breakdown carries which kernels
            // each phase dispatches and not only how many.
            let probe = RoundProbe {
                sync: if opts.sample_split {
                    Some(&device)
                } else {
                    None
                },
                census: opts.sample_split && stats.steps == CENSUS_ROUND,
            };
            let (accepted, timing) =
                self.speculative_round(spec, &[seq], &[anchor], 0.0, eos_ids, probe)?;
            let accepted = accepted.into_iter().next().ok_or_else(|| {
                lloom_tensor::Error::msg("decode_speculative: verify returned no sequence")
            })?;
            stats.rounds.push(RoundStats {
                accepted: accepted.len(),
                timing,
                start_ns: (step_start - prefill_start).as_nanos() as u64,
            });
            if accepted.is_empty() {
                lloom_tensor::bail!(
                    "decode_speculative: a verify round accepted zero tokens; the target's own \
                     prediction at position 0 is always correct, so this is a broken verify"
                );
            }

            // The cache advanced by the accepted length; the next window starts
            // there. The loop's anchor follows the accepted tail — which is what
            // `observe` set the drafter's own anchor to, kept in step here so the
            // next `Anchor` matches.
            self.seq.pos += accepted.len();
            anchor_tok = *accepted.last().unwrap();

            // Publish the accepted tokens, stopping at the first EOS or at
            // `max_tokens`, whichever comes first — a round may straddle either.
            // An EOS is not streamed and retracts from the count, exactly as the
            // baseline loop drops an EOS token.
            let mut batch: Vec<u32> = Vec::with_capacity(accepted.len());
            for &id in &accepted {
                if eos_ids.contains(&id) {
                    eos = true;
                    break;
                }
                if opts.reached(produced + batch.len()) {
                    break;
                }
                batch.push(id);
            }
            if !batch.is_empty() {
                on_tokens(&batch)?;
                all_ids.extend_from_slice(&batch);
                produced += batch.len();
            }

            // Periodic commit/prune. The verify pass already read the argmax back
            // (a host sync), so the round is naturally drained; the commit bounds
            // the in-flight queue on the ring-owning strategies and is a no-op
            // elsewhere. Gated on the budget like the baseline loop.
            if readback.loop_commits() {
                since_commit += 1;
                if opts.budget.due(since_commit) {
                    self.commit(&device, &mut stats)?;
                    since_commit = 0;
                }
            }

            end_executor_step(&device)?;
            stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
        }

        // The sequence's bookkeeping. The cache holds each round's window —
        // the anchor and the proposals the target agreed with — so it runs one
        // token BEHIND the stream while the run is going (the final anchor is
        // streamed and not yet fed) and AHEAD of it when an EOS fell mid-block
        // (the accepted tokens after it were fed before the host saw it). Both
        // are read off the position: the fed ids are the stream's prefix of
        // that length, and anything fed past the stream is overrun.
        let fed = self.seq.pos - pos_before - prompt.len();
        let in_cache = fed.min(produced);
        self.seq.tokens.extend_from_slice(&all_ids[..in_cache]);
        let overrun = fed - in_cache;
        self.seq.overrun = overrun;
        stats.overrun = overrun;

        stats.produced = produced;
        stats.readback = readback.counters();
        Ok(DecodeOut {
            ids: all_ids,
            eos,
            stats,
        })
    }

    /// [`Self::decode_speculative`] over several sequences at once, their
    /// rounds software-pipelined so one sequence's verify runs on the GPU while
    /// another's draft and observe are encoded
    /// (`design/docs/speculative-pipeline-scheduler-design.md` §4a, §10).
    ///
    /// # What overlaps, and what cannot
    ///
    /// Within one sequence the round is dependency-locked: the next draft reads
    /// the context `observe` rebuilds from this verify's taps. Across sequences
    /// there is no such edge — each has its own cache and its own drafter slot
    /// — so the loop keeps one verify per live sequence enqueued and services
    /// them in turn: complete `i`'s verify, observe, publish, draft and enqueue
    /// `i`'s next verify, then move to `i + 1`, whose verify has meanwhile been
    /// running. The GPU is fed for as long as the host's turnaround on one
    /// sequence is shorter than the others' queued work.
    ///
    /// The readback that ends a verify waits on that verify's own command
    /// buffer ([`Probes::verify_complete`]), not on the queue's tail; a
    /// tail wait would drain the other sequences' verifies and serialise the
    /// whole thing back to the solo round.
    ///
    /// # Correctness
    ///
    /// Each sequence's stream is byte-identical to `decode_speculative` run
    /// solo — the drafter's `PerSeq` slots and the target caches are disjoint,
    /// and the accept rule is the same `accept_prefix`. The per-sequence
    /// bookkeeping (`pos`, `tokens`, `overrun`, EOS mid-block, the `max_tokens`
    /// cut) is the serial loop's. `sessions[i]` is driven with drafter slot
    /// `SeqId(i)` and prompt `prompts[i]`; `on_tokens` receives `(i, ids)`.
    ///
    /// Refused: fewer drafter slots than sessions, an empty prompt, a session
    /// with an unreconciled overrun, a proposer with `max_draft() == 0`.
    /// `opts.sample_split` has no per-phase meaning here and is ignored.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_speculative_pipelined<S>(
        sessions: &mut [Self],
        spec: &mut S,
        prompts: &[&[u32]],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        device: &Device,
        on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
    ) -> Result<Vec<DecodeOut>>
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
        let n = sessions.len();
        if n == 0 {
            lloom_tensor::bail!("decode_speculative_pipelined: no sessions");
        }
        if prompts.len() != n {
            lloom_tensor::bail!(
                "decode_speculative_pipelined: {} sessions and {} prompts",
                n,
                prompts.len()
            );
        }
        if spec.max_draft() == 0 {
            lloom_tensor::bail!(
                "decode_speculative_pipelined: the proposer has max_draft 0 (the K=1 identity); \
                 it belongs on `Session::decode`"
            );
        }
        if spec.n_seq() < n {
            lloom_tensor::bail!(
                "decode_speculative_pipelined: the proposer has {} sequence slots for {} sessions",
                spec.n_seq(),
                n
            );
        }
        if eos_ids.is_empty() {
            lloom_tensor::bail!(
                "decode_speculative_pipelined: eos_ids is empty, so the loop would never terminate"
            );
        }
        for (i, s) in sessions.iter().enumerate() {
            if prompts[i].is_empty() {
                lloom_tensor::bail!(
                    "decode_speculative_pipelined: sequence {i} has an empty prompt"
                );
            }
            if s.seq.overrun > 0 {
                lloom_tensor::bail!(
                    "decode_speculative_pipelined: sequence {i}'s cache is {} positions past its \
                     last token (an earlier run's overrun); `Session::reconcile` or \
                     `Session::reset` first",
                    s.seq.overrun
                );
            }
        }
        spec.want_confidence(false);
        let run_start = std::time::Instant::now();
        let hot_thread = HotThread::claim();

        let mut lanes: Vec<Lane<<M as Probes<S::Probe>>::Pending>> = Vec::with_capacity(n);

        // Prefill every sequence before reading any back, so the GPU has all
        // the prompts queued while the host waits on the first. Each prefill
        // runs in `opts.prefill_ubatch()`'s units and seeds its drafter slot
        // unit by unit (`prefill_streamed`); only the last unit's probe waits
        // here for its anchor.
        let mut logits = Vec::with_capacity(n);
        for (i, s) in sessions.iter_mut().enumerate() {
            s.seq.last_logits = None;
            let pos_before = s.seq.pos;
            let prefill_start = std::time::Instant::now();
            let (l, probe, prefill_units) =
                s.prefill_streamed(spec, SeqId(i), prompts[i], opts.prefill_ubatch(), device)?;
            logits.push((l, probe));
            lanes.push(Lane {
                anchor_tok: 0,
                produced: 0,
                eos: false,
                all_ids: Vec::with_capacity(opts.max_tokens),
                pos_before,
                prompt_len: prompts[i].len(),
                stats: DecodeStats {
                    step_ns: Vec::with_capacity(opts.max_tokens),
                    prefill_ns: prefill_start.elapsed().as_nanos() as u64,
                    prefill_units,
                    hot_thread: hot_thread.report(),
                    ..DecodeStats::default()
                },
                pending: None,
                in_flight: RoundTiming::default(),
                round_start_ns: 0,
                done: false,
            });
        }

        // Anchors and the last unit's seed, as the serial loop: the prefill's
        // argmax is token 0 and the block anchor; the earlier units were keyed
        // as they landed (`design/docs/speculation.md` §10.2w).
        for (i, (l, prefill_probe)) in logits.into_iter().enumerate() {
            let lane = &mut lanes[i];
            let anchor_tok = l
                .squeeze(0)?
                .argmax(lloom_tensor::D::Minus1)?
                .to_scalar::<u32>()?;
            spec.observe(SeqId(i), &prefill_probe, std::slice::from_ref(&anchor_tok))?;
            lane.anchor_tok = anchor_tok;
            if eos_ids.contains(&anchor_tok) {
                lane.eos = true;
            } else if !opts.reached(lane.produced) {
                on_tokens(i, std::slice::from_ref(&anchor_tok))?;
                lane.all_ids.push(anchor_tok);
                lane.produced += 1;
            }
            lane.done = lane.eos || opts.reached(lane.produced);
        }

        // Prime: one verify in flight per live sequence.
        for i in 0..n {
            if !lanes[i].done {
                Self::pipeline_enqueue(&mut sessions[i], &mut lanes[i], spec, i, run_start)?;
            }
        }

        // Steady state: service the in-flight verifies round-robin. Completing
        // lane `i` waits on its own buffer only; its next verify is enqueued
        // behind the others' before the loop moves on.
        loop {
            let _pool = lloom_tensor::AutoreleaseScope::new(); // per round, see the decode loop
            let mut progressed = false;
            for i in 0..n {
                let Some(pending) = lanes[i].pending.take() else {
                    continue;
                };
                progressed = true;
                let lane = &mut lanes[i];
                let s = &mut sessions[i];
                let mut timing = std::mem::take(&mut lane.in_flight);

                let t = std::time::Instant::now();
                let rb = readbacks();
                let (accepted, probe_out) =
                    s.model
                        .verify_complete(pending, &mut s.seq.cache, eos_ids)?;
                timing.verify_wait_ns = t.elapsed().as_nanos() as u64;
                timing.verify_ns = timing.verify_enqueue_ns + timing.verify_wait_ns;
                timing.verify_readbacks += readbacks() - rb;

                let t = std::time::Instant::now();
                let rb = readbacks();
                spec.observe(SeqId(i), &probe_out, &accepted)?;
                timing.observe_ns = t.elapsed().as_nanos() as u64;
                timing.observe_readbacks = readbacks() - rb;

                if accepted.is_empty() {
                    lloom_tensor::bail!(
                        "decode_speculative_pipelined: a verify round accepted zero tokens; the \
                         target's own prediction at position 0 is always correct, so this is a \
                         broken verify"
                    );
                }
                lane.stats.steps += 1;
                lane.stats.rounds.push(RoundStats {
                    accepted: accepted.len(),
                    timing,
                    start_ns: lane.round_start_ns,
                });
                s.seq.pos += accepted.len();
                lane.anchor_tok = *accepted.last().unwrap();

                let mut batch: Vec<u32> = Vec::with_capacity(accepted.len());
                for &id in &accepted {
                    if eos_ids.contains(&id) {
                        lane.eos = true;
                        break;
                    }
                    if opts.reached(lane.produced + batch.len()) {
                        break;
                    }
                    batch.push(id);
                }
                if !batch.is_empty() {
                    on_tokens(i, &batch)?;
                    lane.all_ids.extend_from_slice(&batch);
                    lane.produced += batch.len();
                }
                lane.stats.step_ns.push(
                    (run_start.elapsed().as_nanos() as u64).saturating_sub(lane.round_start_ns),
                );
                lane.done = lane.eos || opts.reached(lane.produced);
                if !lane.done {
                    Self::pipeline_enqueue(s, lane, spec, i, run_start)?;
                }
            }
            if !progressed {
                break;
            }
        }

        // Per-sequence bookkeeping, as the serial loop: the fed ids are the
        // stream's prefix of the cache's length, anything past it is overrun.
        let mut outs = Vec::with_capacity(n);
        for (i, lane) in lanes.into_iter().enumerate() {
            let s = &mut sessions[i];
            let fed = s.seq.pos - lane.pos_before - lane.prompt_len;
            let in_cache = fed.min(lane.produced);
            s.seq.tokens.extend_from_slice(&lane.all_ids[..in_cache]);
            let overrun = fed - in_cache;
            s.seq.overrun = overrun;
            let mut stats = lane.stats;
            stats.overrun = overrun;
            stats.produced = lane.produced;
            outs.push(DecodeOut {
                ids: lane.all_ids,
                eos: lane.eos,
                stats,
            });
        }
        Ok(outs)
    }

    /// Draft sequence `i` and enqueue its verify, leaving the receipt in the
    /// lane. The draft is one drafter forward; nothing here waits for the GPU.
    fn pipeline_enqueue<S>(
        s: &mut Self,
        lane: &mut Lane<<M as Probes<S::Probe>>::Pending>,
        spec: &mut S,
        i: usize,
        run_start: std::time::Instant,
    ) -> Result<()>
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
        let anchor = Anchor {
            last: lane.anchor_tok,
            pos: s.seq.pos,
        };
        lane.round_start_ns = run_start.elapsed().as_nanos() as u64;
        let mut timing = RoundTiming::default();
        let t = std::time::Instant::now();
        let rb = readbacks();
        let drafts = spec.draft(&[SeqId(i)])?;
        let draft = drafts.into_iter().next().ok_or_else(|| {
            lloom_tensor::Error::msg("decode_speculative_pipelined: the drafter returned no block")
        })?;
        timing.draft_ns = t.elapsed().as_nanos() as u64;
        timing.draft_readbacks = readbacks() - rb;

        let t = std::time::Instant::now();
        let rb = readbacks();
        let pending = s.model.verify_enqueue(&draft, &anchor, &mut s.seq.cache)?;
        timing.verify_enqueue_ns = t.elapsed().as_nanos() as u64;
        timing.verify_readbacks = readbacks() - rb;
        lane.in_flight = timing;
        lane.pending = Some(pending);
        Ok(())
    }

    /// Decode `sessions.len()` independent sequences together, one token for
    /// every live sequence per forward pass — the batched-forward body
    /// (issue #694, `design/docs/batched-decode-design.md` §10).
    ///
    /// # What a step is
    ///
    /// Every live row's next token is sampled on the device from the `[B,
    /// vocab]` logits in one argmax, and the `[B, 1]` ids are fed to
    /// [`BatchedCausalLM::forward_rows`], which applies each weight matrix to
    /// all `B` rows as one `m = B` GEMM and everything per sequence — RoPE,
    /// the KV append and read, the conv window — per row against that row's
    /// own cache. So the 5.394 GB weight sweep is paid once per step for `B`
    /// tokens (§13.5), and a row computes what it would compute alone: its
    /// stream does not depend on its batch neighbours (§2.3.3 #7).
    ///
    /// The rows are independent sequences: any prompts, any lengths, each
    /// prefilled through its own `forward` before the loop. A row leaves the
    /// batch at its EOS or at `opts.max_tokens`; the step shrinks to the rows
    /// still live and the run ends when none is.
    ///
    /// # Where the host waits — `opts.lookahead`
    ///
    /// The one thing the host must learn per step is which rows ended.
    /// `lookahead = 0` reads the `B` ids back before it encodes the next
    /// forward (the GPU idles for the round trip, no row runs past its end).
    /// `lookahead = 1` feeds the device-resident ids straight into the next
    /// forward, encodes it, and only then reads the ids back through a staged
    /// read that waits on the argmax's own command buffer (#685's
    /// `Tensor::stage_read`) — so the next step is on the GPU while the host
    /// learns this one's EOS, at the price [`Session::decode`]'s own lookahead
    /// pays: a row that ended has had one more forward run for it, reported as
    /// its [`DecodeStats::overrun`]. A row that ends on `max_tokens` under
    /// `lookahead = 1` keeps that forward's logits as its resumable
    /// [`Sequence`](crate::nn::speculator::Sequence) state.
    ///
    /// Argmax only: the ids are sampled on the device for the whole batch, and
    /// a per-row temperature sample would need a per-row noise stream the
    /// host `StdRng` does not give (§2.3.3 #7's counter-based RNG is not
    /// built). Refused rather than approximated, as the ring refuses top-p.
    ///
    /// `on_tokens` receives `(row, ids)` as each row's token is confirmed.
    /// `opts.budget` and `opts.sample_split` have no meaning here: the
    /// lookahead arm commits once per step at its staged read, the blocking
    /// arm's readback drains the queue every step.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_batched(
        sessions: &mut [Self],
        sampler: &mut LogitsProcessor,
        prompts: &[&[u32]],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        device: &Device,
        on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
    ) -> Result<Vec<DecodeOut>>
    where
        M: BatchedCausalLM,
    {
        let arrivals = vec![0u64; sessions.len()];
        Self::decode_batched_admitted(
            sessions, sampler, prompts, &arrivals, opts, eos_ids, device, on_tokens,
        )
    }

    /// [`Self::decode_batched`] with each row ARRIVING at a step of the run:
    /// `arrivals[i]` is the first step (counting from 0) at which row `i`'s
    /// prompt may be fed. A row whose arrival is `0` is admitted before the
    /// first step, which is `decode_batched`'s shape; a later arrival joins a
    /// batch that is already decoding — the serving shape, where a request
    /// lands while others are mid-reply.
    ///
    /// # How a prompt is fed — `opts.prefill_chunk`, the CHUNK
    ///
    /// `prefill_chunk == 0` feeds an arrived row's whole prompt at its
    /// arrival step, in one forward (or in `opts.prefill_tile` forwards when
    /// the tile is set), before that step's batched decode is encoded: every
    /// live row's next token waits for the whole prompt. `prefill_chunk > 0`
    /// feeds at most that many prompt tokens per step, of the earliest
    /// arrived row whose prompt is not yet complete, AFTER the step's batched
    /// forward is enqueued — so a long prompt is spread over many steps, each
    /// live row's inter-token interval grows by one chunk's forward rather
    /// than by the prompt, and the row joins the batch at the step after its
    /// last chunk. The prompt's KV is built incrementally, which is exact
    /// given the conv writer's left-context carry (`Config::conv_prefill_ctx`).
    ///
    /// A step with no live row and no arrived prompt to feed (every pending
    /// row's arrival is in the future) is an idle step: nothing is encoded,
    /// and the step count advances so the arrival can come due. The run ends
    /// when no row is live and none is pending.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_batched_admitted(
        sessions: &mut [Self],
        sampler: &mut LogitsProcessor,
        prompts: &[&[u32]],
        arrivals: &[u64],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        device: &Device,
        on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
    ) -> Result<Vec<DecodeOut>>
    where
        M: BatchedCausalLM,
    {
        let n = sessions.len();
        if n == 0 {
            lloom_tensor::bail!("decode_batched: no sessions");
        }
        if prompts.len() != n {
            lloom_tensor::bail!(
                "decode_batched: {} sessions and {} prompts",
                n,
                prompts.len()
            );
        }
        if arrivals.len() != n {
            lloom_tensor::bail!(
                "decode_batched: {} sessions and {} arrivals",
                n,
                arrivals.len()
            );
        }
        if eos_ids.is_empty() {
            lloom_tensor::bail!(
                "decode_batched: eos_ids is empty, so the loop would never terminate"
            );
        }
        if opts.lookahead > 1 {
            lloom_tensor::bail!(
                "decode_batched: lookahead {} — a row runs at most one forward past its end",
                opts.lookahead
            );
        }
        if sampler.sampling() != &Sampling::ArgMax {
            lloom_tensor::bail!(
                "decode_batched: the batch is sampled on the device in one argmax; {:?} needs a \
                 per-row host sample this body does not take",
                sampler.sampling()
            );
        }
        for (i, s) in sessions.iter().enumerate() {
            if s.seq.overrun > 0 {
                lloom_tensor::bail!(
                    "decode_batched: sequence {i}'s cache is {} positions past its last token (an \
                     earlier run's overrun); `Session::reconcile` or `Session::reset` first",
                    s.seq.overrun
                );
            }
            if prompts[i].is_empty() && s.seq.last_logits.is_none() {
                lloom_tensor::bail!(
                    "decode_batched: sequence {i} has an empty prompt and no held logits to resume from"
                );
            }
        }
        let run_start = std::time::Instant::now();
        let hot_thread = HotThread::claim();

        let mut lanes: Vec<BatchLane> = (0..n)
            .map(|i| BatchLane {
                produced: 0,
                eos: false,
                done: opts.max_tokens == 0,
                all_ids: Vec::with_capacity(opts.max_tokens),
                pos_before: sessions[i].seq.pos,
                prompt_len: prompts[i].len(),
                last_logits: None,
                stats: DecodeStats {
                    step_ns: Vec::with_capacity(opts.max_tokens),
                    hot_thread: hot_thread.report(),
                    ..DecodeStats::default()
                },
                arrival: arrivals[i],
                prompt_off: 0,
                joined: false,
                prefill_start: None,
            })
            .collect();

        // The live rows, in the order the logits' rows are in; `logits` is
        // `[live.len(), vocab]` (unset while nothing is live).
        let mut live: Vec<usize> = Vec::with_capacity(n);
        let mut logits: Option<Tensor> = None;
        let mut step: u64 = 0;

        // Feed prompt tokens for this step: with no chunk budget, every
        // arrived row's whole remaining prompt; with one, `chunk` tokens of
        // the earliest arrived row still pending. Returns the rows that
        // completed their prompt, with their logits, in row order.
        let admit = |sessions: &mut [Self],
                     lanes: &mut [BatchLane],
                     step: u64|
         -> Result<Vec<(usize, Tensor)>> {
            let mut joined = Vec::new();
            let mut budget_left = if opts.prefill_chunk == 0 {
                usize::MAX
            } else {
                opts.prefill_chunk
            };
            for i in 0..n {
                if budget_left == 0 {
                    break;
                }
                let lane = &mut lanes[i];
                if lane.joined || lane.arrival > step {
                    continue;
                }
                let s = &mut sessions[i];
                let prompt = prompts[i];
                if prompt.is_empty() {
                    // A resume: the held logits are the row's state.
                    let held = s.seq.last_logits.take().expect("checked at entry");
                    lane.joined = true;
                    joined.push((i, held.unsqueeze(0)?));
                    continue;
                }
                let start = lane
                    .prefill_start
                    .get_or_insert_with(std::time::Instant::now);
                let remaining = prompt.len() - lane.prompt_off;
                let take = if opts.prefill_chunk == 0 {
                    remaining
                } else {
                    remaining.min(budget_left)
                };
                let piece = &prompt[lane.prompt_off..lane.prompt_off + take];
                // The tile bounds each forward of the piece; the chunk is what
                // this step feeds of it.
                let mut last = None;
                let mut off = 0usize;
                for w in tile_widths(vec![take], opts.prefill_tile) {
                    let _pool = lloom_tensor::AutoreleaseScope::new();
                    let unit = &piece[off..off + w];
                    let input = Tensor::new(unit, device)?.unsqueeze(0)?;
                    last = Some(s.model.forward(&input, s.seq.pos, &mut s.seq.cache)?);
                    s.seq.pos += w;
                    s.seq.tokens.extend_from_slice(unit);
                    off += w;
                    lane.stats.prefill_units += 1;
                }
                lane.prompt_off += take;
                if opts.prefill_chunk > 0 {
                    budget_left -= take;
                }
                lane.stats.prefill_ns = start.elapsed().as_nanos() as u64;
                if lane.prompt_off == prompt.len() {
                    lane.joined = true;
                    joined.push((i, last.expect("a non-empty piece steps at least once")));
                }
            }
            Ok(joined)
        };

        loop {
            let _pool = lloom_tensor::AutoreleaseScope::new(); // per batched step, see the decode loop
            let pending = (0..n).any(|i| !lanes[i].joined);
            if live.is_empty() && !pending {
                break;
            }
            let step_start = std::time::Instant::now();
            let b = live.len();
            // Without a chunk budget an arrived prompt is fed whole before
            // this step's decode — every live row waits on it, the
            // head-of-line shape. With one, the chunk rides behind the
            // decode below; when nothing is live there is nothing to ride
            // behind, so it is fed here. `admit` runs once per step.
            let mut joined = if opts.prefill_chunk == 0 || b == 0 {
                admit(sessions, &mut lanes, step)?
            } else {
                Vec::new()
            };
            for &i in &live {
                lanes[i].stats.steps += 1;
                lanes[i].stats.batch_rows.push(b);
            }
            // One argmax over `[B, vocab]`: the ids stay on the device.
            let ids = match &logits {
                Some(l) if b > 0 => Some(sampler.sample_argmax_on_device(l)?),
                _ => None,
            };

            let (host_ids, next_logits) = match (&ids, opts.lookahead) {
                (None, _) => (Vec::new(), None),
                (Some(ids), 0) => {
                    // Learn first, then feed: the readback waits on the queue's
                    // tail, which is exactly this step's argmax.
                    (ids.to_vec1::<u32>()?, None)
                }
                (Some(ids), _) => {
                    // Stage the readback BEFORE the next forward is encoded, so
                    // the wait is on the argmax's command buffer and not on the
                    // forward queued behind it; then feed the device ids.
                    let staged = ids.stage_read()?;
                    let input = ids.reshape((b, 1))?;
                    let next = Self::forward_live(sessions, &live, &input)?;
                    for &i in &live {
                        lanes[i].stats.commits += 1;
                    }
                    // The chunk queues behind the batched forward and runs
                    // while the host waits for this step's ids.
                    if opts.prefill_chunk > 0 {
                        joined = admit(sessions, &mut lanes, step)?;
                    }
                    (staged.to_vec1::<u32>()?, Some(next))
                }
            };
            if host_ids.len() != b {
                lloom_tensor::bail!(
                    "decode_batched: {} ids read back for {b} live rows",
                    host_ids.len()
                );
            }

            // Stream and retire. An EOS is not streamed and ends its row; a
            // row reaching `max_tokens` ends after its token is streamed.
            for (k, &i) in live.iter().enumerate() {
                let id = host_ids[k];
                let lane = &mut lanes[i];
                if eos_ids.contains(&id) {
                    lane.eos = true;
                    lane.done = true;
                    continue;
                }
                on_tokens(i, std::slice::from_ref(&id))?;
                lane.all_ids.push(id);
                lane.produced += 1;
                if lane.produced >= opts.max_tokens {
                    lane.done = true;
                    // Under the lookahead this row's last token was fed, so
                    // the logits after it exist and are the state a
                    // continuation resumes from.
                    if let Some(next) = &next_logits {
                        lane.last_logits = Some(next.narrow(0, k, 1)?.squeeze(0)?);
                    }
                }
            }
            let step_ns = step_start.elapsed().as_nanos() as u64;
            for &i in &live {
                lanes[i].stats.step_ns.push(step_ns);
            }

            let kept: Vec<usize> = live.iter().copied().filter(|&i| !lanes[i].done).collect();
            let kept_logits = match next_logits {
                // The forward already ran for every row that was live; keep
                // the rows still live, in their order.
                Some(next) if kept.len() == live.len() => Some(next),
                Some(_) if kept.is_empty() => None,
                Some(next) => {
                    let keep: Vec<u32> = live
                        .iter()
                        .enumerate()
                        .filter(|(_, i)| !lanes[**i].done)
                        .map(|(k, _)| k as u32)
                        .collect();
                    Some(next.index_select(&Tensor::new(keep.as_slice(), device)?, 0)?)
                }
                None if kept.is_empty() => None,
                None => {
                    // Blocking arm: feed the ids the host just read, live rows
                    // only; under a chunk budget the chunk rides behind it.
                    let next_ids: Vec<u32> = live
                        .iter()
                        .enumerate()
                        .filter(|(_, i)| !lanes[**i].done)
                        .map(|(k, _)| host_ids[k])
                        .collect();
                    let input =
                        Tensor::new(next_ids.as_slice(), device)?.reshape((kept.len(), 1))?;
                    Some(Self::forward_live(sessions, &kept, &input)?)
                }
            };
            if opts.lookahead == 0 && opts.prefill_chunk > 0 && b > 0 {
                joined = admit(sessions, &mut lanes, step)?;
            }
            end_executor_step(device)?;

            // Next step's rows: the kept rows, then the rows that joined this
            // step, each with its own logits row. A row joining with `done`
            // already set (`max_tokens == 0`) is retired at once.
            let mut next_live = kept;
            let mut rows: Vec<Tensor> = Vec::with_capacity(next_live.len() + joined.len());
            if let Some(l) = kept_logits {
                rows.push(l);
            }
            for (i, l) in joined {
                if lanes[i].done {
                    continue;
                }
                next_live.push(i);
                rows.push(l);
            }
            logits = match rows.len() {
                0 => None,
                1 => Some(rows.remove(0)),
                _ => Some(Tensor::cat(&rows, 0)?),
            };
            live = next_live;
            step += 1;
        }

        // Per-sequence bookkeeping, as the other bodies: the fed ids are the
        // stream's prefix of the cache's length, anything fed past the
        // stream is overrun.
        let mut outs = Vec::with_capacity(n);
        for (i, lane) in lanes.into_iter().enumerate() {
            let s = &mut sessions[i];
            let fed = s.seq.pos - lane.pos_before - lane.prompt_len;
            let in_cache = fed.min(lane.produced);
            s.seq.tokens.extend_from_slice(&lane.all_ids[..in_cache]);
            let overrun = fed - in_cache;
            s.seq.overrun = overrun;
            s.seq.last_logits = if overrun == 0 { lane.last_logits } else { None };
            let mut stats = lane.stats;
            stats.overrun = overrun;
            stats.produced = lane.produced;
            outs.push(DecodeOut {
                ids: lane.all_ids,
                eos: lane.eos,
                stats,
            });
        }
        let _ = run_start;
        Ok(outs)
    }

    /// One batched forward over the sessions named by `live`, in that order,
    /// advancing each one's position.
    fn forward_live(sessions: &mut [Self], live: &[usize], input: &Tensor) -> Result<Tensor>
    where
        M: BatchedCausalLM,
    {
        let model = sessions[0].model;
        // Disjoint `&mut` borrows of the live sessions, in `live`'s order: the
        // indices are distinct by construction (each row is live at most once).
        let mut slots: Vec<Option<&mut Self>> = sessions.iter_mut().map(Some).collect();
        let mut picked: Vec<&mut Self> = Vec::with_capacity(live.len());
        for &i in live {
            match slots[i].take() {
                Some(s) => picked.push(s),
                None => {
                    lloom_tensor::bail!("decode_batched: row {i} appears twice in the live set")
                }
            }
        }
        let mut rows: Vec<Row<'_, M::Cache>> = picked
            .iter_mut()
            .map(|s| Row {
                pos: s.seq.pos,
                cache: &mut s.seq.cache,
            })
            .collect();
        let logits = model.forward_rows(input, &mut rows)?;
        drop(rows);
        for s in picked {
            s.seq.pos += 1;
        }
        Ok(logits)
    }

    /// Compile the Metal pipelines the first real forward would otherwise
    /// compile *inside its own timed window*, then reset so the session is
    /// untouched.
    ///
    /// # What this moves, and why it is not a compute change
    ///
    /// `DESIGN.md` §13.1i measures pipeline creation at **~2 ms cold, ~0.01 ms
    /// warm** and library compilation at seconds, both lazily cached in the
    /// device's `Kernels`. The decode step (`m == 1`) routes to GEMV **before**
    /// `select_tile_config` (`mlx_gemm.rs`), so the prefill GEMM-tile pipelines
    /// — and the F32 generic-attention and batched-`Conv1d` pipelines the
    /// `seq_len > 1` path takes — are **prefill-exclusive**: none is touched by
    /// any decode step, so all are cold on the first real prefill. That is the
    /// fixed cost a single-shot TTFT carries and `llama.cpp` does not (it
    /// compiles every kernel up front). A warmup pays it here, folded into model
    /// load, where no clock the user reads is running.
    ///
    /// # Why a real forward rather than an enumerated pipeline list
    ///
    /// The GEMM pipeline key varies with `M` through both the tile name and
    /// `align_m = m % bm`, so a hand-written list would be incomplete by
    /// construction and a near-miss on the constants keys a *different* cache
    /// entry — the real forward would still compile cold. Running the production
    /// `forward` cannot miss a pipeline that same `forward` will later use for
    /// the same shape bucket. The buckets are the row tiles `select_tile_config`
    /// picks by `m` (§13.5o: 16 rows to `m` = 32, 64 rows above; `m == 1` is
    /// the GEMV), each in an aligned (`m % bm == 0`) and an unaligned variant —
    /// four pipelines a prefill can land on, none of which a decode step
    /// touches. [`WARMUP_PREFILL_LENS`] is one prefill per variant, then the
    /// decode step is the GEMV: the set every real run actually takes.
    ///
    /// Token id `0` is always a valid vocabulary index. The forward's output is
    /// discarded — only its compilation side effect is wanted — and
    /// `M::Cache: CacheReset` returns the cache to its pre-prefill state
    /// afterwards, so the next real prompt starts exactly as it would have.
    /// `synchronize` forces compilation to *complete* here rather than leak into
    /// the first real forward.
    ///
    /// Returns the device's `(libraries, pipelines)` cache sizes after warming,
    /// so a caller can report engagement — a nonzero pipeline count is what
    /// distinguishes a warmed cache from a probe that compiled nothing
    /// (§14.5d).
    pub fn warmup(&mut self, device: &Device) -> Result<(usize, usize)>
    where
        M::Cache: CacheReset,
    {
        // One prefill per GEMM pipeline variant a prompt can select (each row
        // tile, aligned and unaligned), then one decode step for the GEMV. A
        // variant left cold here is compiled inside the first prompt's own timed
        // prefill (~2 ms, §13.1i), a fresh cost every process since the
        // pipeline cache does not outlive one.
        let mut pos = 0;
        for &len in &WARMUP_PREFILL_LENS {
            let tokens = vec![0u32; len];
            let prefill = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
            let _ = self.model.forward(&prefill, pos, &mut self.seq.cache)?;
            pos += len;
        }

        // Measure the host/GPU balance from a real decode step, to compute the
        // in-flight depth (`compute_in_flight_depth`) instead of a static cap.
        // `host_ns` = the CPU time to encode+submit the step; `gpu_ns` = the
        // step's total time including its GPU tail (drained by the sync). Their
        // ratio is what decides how far the host must run ahead to keep the GPU
        // fed — 2 here (gpu >> host), deeper on a faster GPU / smaller model.
        let decode = Tensor::new(&[0u32], device)?.unsqueeze(0)?;
        let t_host = std::time::Instant::now();
        let _ = self.model.forward(&decode, pos, &mut self.seq.cache)?;
        // Host submit is done here (the forward returns before the GPU finishes).
        let host_ns = t_host.elapsed().as_nanos() as u64;

        // Force the driver to finish compiling before the clock the caller is
        // about to stop; a cold pipeline creation left in flight would surface
        // on the first real forward instead. This sync also completes the decode
        // step's GPU tail, so the elapsed-from-submit is the GPU step time.
        let t_gpu = std::time::Instant::now();
        device.synchronize()?;
        // `gpu_ns` = host submit + GPU tail: an upper bound on the step's GPU
        // time (the sync waited out the tail). Good enough for the ratio, which
        // only needs to know whether the GPU dwarfs the host or not.
        let gpu_ns = host_ns + t_gpu.elapsed().as_nanos() as u64;
        self.warm_in_flight_depth =
            compute_in_flight_depth(host_ns, gpu_ns, in_flight_mem_cap(device));

        // Leave nothing behind: the warmup wrote KV, conv state and the position
        // counter, and the next call is the user's real prefill from position 0.
        self.reset();

        Ok(cache_sizes(device))
    }

    /// The in-flight depth warmup measured (`0` if warmup has not run). A caller
    /// threads this into `DecodeOptions::in_flight_depth` so the pacer uses the
    /// computed host/GPU balance rather than the floor.
    pub fn measured_in_flight_depth(&self) -> usize {
        self.warm_in_flight_depth
    }
}

/// The speculative round whose per-phase dispatch census is kept under
/// `DecodeOptions::sample_split`: past the first rounds' pipeline warm-up,
/// early enough that a short run still reaches it.
const CENSUS_ROUND: u64 = 6;

/// The prefill lengths [`Session::warmup`] runs, one per GEMM pipeline variant
/// a prompt can select: the 16-row tile unaligned and aligned (`m` = 8, 16)
/// and the 64-row tile unaligned and aligned (40, 64) — `select_tile_config`
/// picks the row tile by `m` (§13.5o) and `align_m = m % bm` keys a second
/// pipeline per tile. A length here that stopped matching a variant would leave
/// that variant cold on the first prompt that selects it, the ~2 ms §13.1i
/// prices a pipeline creation at (measured 40 ms of `prefill_ns` at `m` = 64
/// before this list carried the aligned 64-row variant, §13.5o).
pub const WARMUP_PREFILL_LENS: [usize; 4] = [8, 16, 40, 64];

/// The forward widths a prefill of `prompt_len` tokens is fed through, in order.
///
/// This is the one place the prefill's forward shape is decided, so the
/// incumbent ramp and the explicit `--spark` schedule cannot drift.
///
/// * `schedule = None` — the **incumbent** `prefill_ubatch` ramp, reproduced
///   byte-for-byte: `ubatch == 0` is one-shot (`[prompt_len]`); a positive
///   `ubatch` is the spark then the whole remainder (`[ubatch, prompt_len −
///   ubatch]`), clamped so a spark wider than the prompt is a single forward.
/// * `schedule = Some(units)` — feed forwards of `units[0], units[1], …` (the
///   **ramp**), each clamped to what is left; then cover the remainder with
///   **`bulk`-wide forwards** (the bulk tile), and only the final `< bulk`
///   tokens as a `seq_len == 1` tail. `bulk == 0` (or `bulk == 1`) means the
///   whole remainder is the `seq_len == 1` tail — the pre-bulk behaviour.
///
///   The bulk tile is the fix for the many-tiny-forwards pathology: a ramp with
///   a `1`-token tail over a long prompt runs one full weight sweep per leftover
///   token (a 400-token prompt = ~400 forwards). Chopping the remainder at an
///   *aligned* bulk width (16 / 32 / 64) instead keeps every bulk forward on the
///   no-guard GEMM tile and collapses the forward count to `remainder / bulk`.
///
/// The TILE bound ([`tile_widths`]) is applied over the result by the callers.
///
/// Total is always exactly `prompt_len` (each width ≤ the remainder, and the
/// bulk + tail close any gap), so the caller's `off` walks the prompt exactly
/// once. An empty prompt yields no forwards.
fn prefill_widths(
    prompt_len: usize,
    ubatch: usize,
    schedule: &Option<Vec<usize>>,
    bulk: usize,
) -> Vec<usize> {
    if prompt_len == 0 {
        return Vec::new();
    }
    match schedule {
        None => {
            // The incumbent ramp: one-shot, or spark + remainder.
            if ubatch == 0 || ubatch >= prompt_len {
                vec![prompt_len]
            } else {
                vec![ubatch, prompt_len - ubatch]
            }
        }
        Some(units) => {
            let mut widths = Vec::new();
            let mut left = prompt_len;
            for &u in units {
                if left == 0 {
                    break;
                }
                let w = u.min(left);
                if w == 0 {
                    // A `0` in the schedule is a no-op unit, not a stall.
                    continue;
                }
                widths.push(w);
                left -= w;
            }
            // The remainder rides the bulk tile: `bulk`-wide forwards (each on
            // an aligned GEMM variant), then the final `< bulk` tokens one at a
            // time. `bulk <= 1` degenerates to the all-`seq_len==1` tail.
            let bulk = bulk.max(1);
            while left >= bulk && bulk > 1 {
                widths.push(bulk);
                left -= bulk;
            }
            // Whatever is left (`< bulk`, or all of it when `bulk <= 1`) as a
            // `seq_len == 1` tail — no arbitrary-width forward sneaks in.
            for _ in 0..left {
                widths.push(1);
            }
            widths
        }
    }
}

/// The TILE bound over [`prefill_widths`]'s result: every width above `tile`
/// is split into sequential forwards of `tile` tokens and a narrower last
/// one, so no forward is wider than the tile. `0` leaves the widths as they
/// are. The total is unchanged.
fn tile_widths(widths: Vec<usize>, tile: usize) -> Vec<usize> {
    if tile == 0 {
        return widths;
    }
    let mut tiled = Vec::with_capacity(widths.len());
    for w in widths {
        let mut left = w;
        while left > 0 {
            let t = left.min(tile);
            tiled.push(t);
            left -= t;
        }
    }
    tiled
}

#[cfg(test)]
mod prefill_widths_tests {
    use super::{prefill_widths, tile_widths};

    // The `None` schedule must reproduce the incumbent ramp exactly — this is
    // the byte-identity contract the crown rests on. `bulk` is irrelevant on the
    // `None` path (no ramp), so pass 0.
    #[test]
    fn none_one_shot_is_the_whole_prompt() {
        assert_eq!(prefill_widths(400, 0, &None, 0), vec![400]);
        assert_eq!(prefill_widths(1, 0, &None, 0), vec![1]);
    }

    #[test]
    fn none_spark_is_spark_then_remainder() {
        assert_eq!(prefill_widths(400, 16, &None, 0), vec![16, 384]);
        assert_eq!(prefill_widths(400, 64, &None, 0), vec![64, 336]);
    }

    #[test]
    fn none_spark_wider_than_prompt_is_one_shot() {
        // A spark ≥ the prompt is a single forward, as the old `step.min` did.
        assert_eq!(prefill_widths(10, 16, &None, 0), vec![10]);
        assert_eq!(prefill_widths(16, 16, &None, 0), vec![16]);
    }

    #[test]
    fn none_empty_prompt_is_no_forwards() {
        assert_eq!(prefill_widths(0, 0, &None, 0), Vec::<usize>::new());
        assert_eq!(prefill_widths(0, 16, &None, 0), Vec::<usize>::new());
    }

    // With `bulk == 0` an explicit schedule feeds its units, then a `1`-token
    // tail — the pre-bulk behaviour.
    #[test]
    fn schedule_fixed_bucket_covers_with_tail() {
        // `--spark=16` on a 40-token prompt, no bulk: [16], then 24×1.
        let want: Vec<usize> = std::iter::once(16)
            .chain(std::iter::repeat(1).take(24))
            .collect();
        assert_eq!(prefill_widths(40, 0, &Some(vec![16]), 0), want);
    }

    #[test]
    fn schedule_progressive_ramps_then_tail() {
        // progressive:1,64 → units [1,16,32,64], prompt 200, no bulk: 113
        // consumed, 87 left as 1-token forwards.
        let widths = prefill_widths(200, 0, &Some(vec![1, 16, 32, 64]), 0);
        assert_eq!(&widths[..4], &[1, 16, 32, 64]);
        assert_eq!(widths.iter().sum::<usize>(), 200);
        assert_eq!(widths.len(), 4 + 87);
    }

    #[test]
    fn schedule_exhausted_exactly_has_no_tail() {
        // A schedule whose units sum to the prompt leaves no tail.
        assert_eq!(prefill_widths(64, 0, &Some(vec![64]), 0), vec![64]);
        assert_eq!(prefill_widths(48, 0, &Some(vec![16, 32]), 0), vec![16, 32]);
    }

    #[test]
    fn schedule_unit_wider_than_remainder_is_clamped() {
        // A `64` unit on a 40-token prompt clamps to 40, no negative left.
        assert_eq!(prefill_widths(40, 0, &Some(vec![64]), 0), vec![40]);
    }

    // The bulk tile: after the ramp, the remainder rides `bulk`-wide forwards,
    // then a `< bulk` tail. This is what collapses the many-tiny-forwards count.
    #[test]
    fn bulk_chops_remainder_into_bucket_forwards() {
        // ramp [1,16] on a 200-token prompt, bulk 32: 1+16=17 consumed, 183
        // left → 5×32=160, then 23 left as 1-token tail.
        let widths = prefill_widths(200, 0, &Some(vec![1, 16]), 32);
        assert_eq!(&widths[..2], &[1, 16]);
        // five 32-wide bulk forwards
        assert_eq!(widths.iter().filter(|&&w| w == 32).count(), 5);
        // 23-token tail of 1s
        assert_eq!(widths.iter().filter(|&&w| w == 1).count(), 1 + 23);
        assert_eq!(widths.iter().sum::<usize>(), 200);
        // FAR fewer forwards than the seq_len=1 tail would give (183 vs 2+5+23=30)
        assert_eq!(widths.len(), 2 + 5 + 23);
    }

    #[test]
    fn bulk_evenly_divides_has_no_tail() {
        // ramp [16] on a 400-token prompt, bulk 64: 16 consumed, 384 left =
        // 6×64 exactly, no tail.
        let widths = prefill_widths(400, 0, &Some(vec![16]), 64);
        assert_eq!(widths[0], 16);
        assert_eq!(widths.iter().filter(|&&w| w == 64).count(), 6);
        assert_eq!(widths.iter().filter(|&&w| w == 1).count(), 0);
        assert_eq!(widths.iter().sum::<usize>(), 400);
    }

    #[test]
    fn bulk_one_degenerates_to_seq_len_1_tail() {
        // bulk 1 (or 0) is the pre-bulk behaviour: all-1 tail.
        assert_eq!(
            prefill_widths(20, 0, &Some(vec![16]), 1),
            prefill_widths(20, 0, &Some(vec![16]), 0),
        );
    }

    #[test]
    fn bulk_larger_than_remainder_is_all_tail() {
        // ramp [16] on 40, bulk 64: 24 left < 64 → no bulk forward, 24×1 tail.
        let widths = prefill_widths(40, 0, &Some(vec![16]), 64);
        assert_eq!(widths[0], 16);
        assert_eq!(widths.iter().filter(|&&w| w == 1).count(), 24);
        assert_eq!(widths.iter().sum::<usize>(), 40);
    }

    #[test]
    fn schedule_total_is_always_prompt_len() {
        for &plen in &[1usize, 7, 16, 33, 64, 100, 347, 512] {
            for units in [
                vec![1],
                vec![16],
                vec![64],
                vec![1, 16, 32, 64],
                vec![32, 32],
            ] {
                for bulk in [0usize, 1, 16, 32, 64] {
                    let widths = prefill_widths(plen, 0, &Some(units.clone()), bulk);
                    assert_eq!(
                        widths.iter().sum::<usize>(),
                        plen,
                        "schedule {units:?} bulk {bulk} on prompt {plen} must total the prompt",
                    );
                }
            }
        }
    }

    // The TILE bounds every forward the ramp, the schedule or the bulk names,
    // and changes nothing when it is 0 or wider than every forward.
    #[test]
    fn tile_splits_one_shot_into_sequential_tiles_with_a_narrower_last() {
        let tiled = |plen, tile| tile_widths(prefill_widths(plen, 0, &None, 0), tile);
        assert_eq!(tiled(2048, 512), vec![512, 512, 512, 512]);
        assert_eq!(tiled(1000, 512), vec![512, 488]);
        assert_eq!(tiled(10, 3), vec![3, 3, 3, 1]);
        // A tile at or above the prompt is one-shot, byte for byte.
        assert_eq!(tiled(400, 400), vec![400]);
        assert_eq!(tiled(400, 4096), vec![400]);
    }

    #[test]
    fn tile_composes_with_the_spark_the_schedule_and_the_bulk() {
        // The spark stays a spark; the remainder is tiled.
        assert_eq!(
            tile_widths(prefill_widths(400, 16, &None, 0), 128),
            vec![16, 128, 128, 128]
        );
        // A schedule unit wider than the tile is split; the 1-token tail is not.
        let widths = tile_widths(prefill_widths(100, 0, &Some(vec![64]), 0), 32);
        assert_eq!(&widths[..2], &[32, 32]);
        assert_eq!(widths.len(), 2 + 36);
        assert!(widths[2..].iter().all(|&w| w == 1));
        // A bulk wider than the tile is split too; a bulk at the tile is not.
        let widths = tile_widths(prefill_widths(200, 0, &Some(vec![16]), 64), 32);
        assert_eq!(widths.iter().sum::<usize>(), 200);
        assert!(widths.iter().all(|&w| w <= 32));
        assert_eq!(
            tile_widths(prefill_widths(200, 0, &Some(vec![16]), 64), 64),
            prefill_widths(200, 0, &Some(vec![16]), 64)
        );
    }

    #[test]
    fn tile_zero_is_the_incumbent_and_a_tile_never_widens_a_forward() {
        for &plen in &[1usize, 7, 16, 33, 64, 100, 347, 512, 2049] {
            for &ubatch in &[0usize, 16, 64] {
                let base = prefill_widths(plen, ubatch, &None, 0);
                assert_eq!(tile_widths(base.clone(), 0), base);
                for &tile in &[1usize, 3, 16, 100, 512, 4096] {
                    let tiled = tile_widths(base.clone(), tile);
                    assert_eq!(tiled.iter().sum::<usize>(), plen, "tile {tile} on {plen}");
                    assert!(
                        tiled.iter().all(|&w| w <= tile && w > 0),
                        "tile {tile} on {plen}: {tiled:?}"
                    );
                    // A tile wider than every base forward leaves the widths alone.
                    if base.iter().all(|&w| w <= tile) {
                        assert_eq!(tiled, base);
                    }
                }
            }
        }
    }

    #[test]
    fn max_prefill_width_is_the_tile_else_the_spark_else_the_widest_unit() {
        let mut o = super::DecodeOptions::new(1);
        assert_eq!(o.max_prefill_width(), 0);
        o.prefill_ubatch = 16;
        assert_eq!(o.max_prefill_width(), 16);
        o.prefill_schedule = Some(vec![1, 16, 32, 64]);
        assert_eq!(o.max_prefill_width(), 64);
        o.prefill_bulk = 128;
        assert_eq!(o.max_prefill_width(), 128);
        o.prefill_tile = 24;
        assert_eq!(o.max_prefill_width(), 24);
    }
}

/// The device's `(libraries, pipelines)` cache sizes, or `(0, 0)` off Metal.
fn cache_sizes(device: &Device) -> (usize, usize) {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return m.kernels().cache_sizes();
    }
    let _ = device;
    (0, 0)
}
