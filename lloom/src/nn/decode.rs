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
#[derive(Debug, Clone, Copy)]
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
        }
    }

    /// The spark width, or the whole prompt in one pass when `0` (the incumbent
    /// one-shot). See the `prefill_ubatch` field.
    pub fn prefill_ubatch(&self) -> usize {
        self.prefill_ubatch
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
                // THE ADAPTIVE SPARK→SEGMENT RAMP. The prefill weight-sweep is
                // amortized over a forward's tokens, so MANY small chunks are slow
                // (each re-reads the 5.4 GB of weights) — but the FIRST token cannot
                // stream until its chunk lands, so a small FIRST chunk gets the
                // machinery flowing early (low TTFT). The resolution: ramp the chunk
                // width. Chunk 1 = `spark` (the tunable TTFT knob), then DOUBLE each
                // chunk up to the whole remaining prompt. For a normal prompt this is
                // "spark, then one-shot the rest" — the first token sparks early and
                // the bulk is ONE big amortized forward, so TTFT drops without the
                // per-chunk weight-sweep tax. `spark == 0` means one-shot (the whole
                // prompt in one chunk): the incumbent, byte-identical, no ramp.
                let spark = opts.prefill_ubatch();
                let mut last_logits = None;
                let mut off = 0usize; // the machine's position over the prompt
                let mut step = if spark == 0 { prompt.len() } else { spark };
                while off < prompt.len() {
                    // Drains this chunk's autoreleased encoders and command
                    // buffers when the iteration ends (`AutoreleaseScope` docs).
                    let _pool = lloom_tensor::AutoreleaseScope::new();
                    let w = step.min(prompt.len() - off);
                    let chunk = &prompt[off..off + w];
                    // Stage this unit's tokens as the forward's input — zero-copy on
                    // the ring arm for a spark-sized chunk (≤ the input tile width),
                    // a one-shot `Tensor::new` for the bigger amortizing chunks (few,
                    // so the alloc is off the hot path). Then step: bound the
                    // enqueue, forward, keep the marker — the same `before/
                    // after_forward` a generate step takes, so prompt units and
                    // generated tokens ride one pipeline.
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
                    prefill_units += 1;
                    // After the spark, jump straight to the WHOLE remaining prompt
                    // (one big amortized forward) rather than doubling: the weight
                    // sweep is per-forward, so the only way to pay it ~once on the
                    // bulk is to feed the bulk in ONE chunk. Doubling would pay it
                    // log(N) times; "spark + rest" pays it twice total (the spark's
                    // small sweep + one bulk sweep). The GPU is already producing
                    // after the spark, so the big chunk does not hurt TTFT — the
                    // first token streamed off the spark. (A prompt larger than the
                    // input machinery can hold in one pass would cap here; for the
                    // chat path the whole rest fits.)
                    step = prompt.len();
                }
                last_logits.expect("a non-empty prompt steps at least once")
            }
        };
        end_executor_step(&device)?;
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;
        stats.prefill_units = prefill_units;
        let census_prefill = census_snapshot();
        stats.prefill_census = census_delta(&census_before, census_prefill.clone());

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
            let stream_now = if loop_commits { committed_now } else { true };
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

        // Prefill every sequence before reading anything back: each prompt is
        // its own `forward` at its own length, queued behind the previous.
        let mut lanes: Vec<BatchLane> = Vec::with_capacity(n);
        let mut logits_rows: Vec<Tensor> = Vec::with_capacity(n);
        for (i, s) in sessions.iter_mut().enumerate() {
            let pos_before = s.seq.pos;
            let prefill_start = std::time::Instant::now();
            let (logits, prefill_units) = match s.seq.last_logits.take() {
                Some(held) if prompts[i].is_empty() => (held.unsqueeze(0)?, 0),
                _ => {
                    let input = Tensor::new(prompts[i], device)?.unsqueeze(0)?;
                    let logits = s.model.forward(&input, s.seq.pos, &mut s.seq.cache)?;
                    s.seq.pos += prompts[i].len();
                    s.seq.tokens.extend_from_slice(prompts[i]);
                    (logits, 1)
                }
            };
            logits_rows.push(logits);
            lanes.push(BatchLane {
                produced: 0,
                eos: false,
                done: opts.max_tokens == 0,
                all_ids: Vec::with_capacity(opts.max_tokens),
                pos_before,
                prompt_len: prompts[i].len(),
                last_logits: None,
                stats: DecodeStats {
                    step_ns: Vec::with_capacity(opts.max_tokens),
                    prefill_ns: prefill_start.elapsed().as_nanos() as u64,
                    prefill_units,
                    hot_thread: hot_thread.report(),
                    ..DecodeStats::default()
                },
            });
        }
        end_executor_step(device)?;

        // The live rows, in the order the logits' rows are in.
        let mut live: Vec<usize> = (0..n).filter(|&i| !lanes[i].done).collect();
        let mut logits = Tensor::cat(
            &live.iter().map(|&i| &logits_rows[i]).collect::<Vec<_>>(),
            0,
        )?;
        drop(logits_rows);

        while !live.is_empty() {
            let _pool = lloom_tensor::AutoreleaseScope::new(); // per batched step, see the decode loop
            let step_start = std::time::Instant::now();
            let b = live.len();
            for &i in &live {
                lanes[i].stats.steps += 1;
                lanes[i].stats.batch_rows.push(b);
            }
            // One argmax over `[B, vocab]`: the ids stay on the device.
            let ids = sampler.sample_argmax_on_device(&logits)?;

            let (host_ids, next_logits) = if opts.lookahead == 0 {
                // Learn first, then feed: the readback waits on the queue's
                // tail, which is exactly this step's argmax.
                (ids.to_vec1::<u32>()?, None)
            } else {
                // Stage the readback BEFORE the next forward is encoded, so
                // the wait is on the argmax's command buffer and not on the
                // forward queued behind it; then feed the device ids.
                let staged = ids.stage_read()?;
                let input = ids.reshape((b, 1))?;
                let next = Self::forward_live(sessions, &live, &input)?;
                for &i in &live {
                    lanes[i].stats.commits += 1;
                }
                (staged.to_vec1::<u32>()?, Some(next))
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
            end_executor_step(device)?;

            let next_live: Vec<usize> = live.iter().copied().filter(|&i| !lanes[i].done).collect();
            if next_live.is_empty() {
                break;
            }
            logits = match next_logits {
                // The forward already ran for every row that was live; keep
                // the rows still live, in their order.
                Some(next) if next_live.len() == live.len() => next,
                Some(next) => {
                    let keep: Vec<u32> = live
                        .iter()
                        .enumerate()
                        .filter(|(_, i)| !lanes[**i].done)
                        .map(|(k, _)| k as u32)
                        .collect();
                    next.index_select(&Tensor::new(keep.as_slice(), device)?, 0)?
                }
                None => {
                    // Blocking arm: feed the ids the host just read, live rows only.
                    let next_ids: Vec<u32> = live
                        .iter()
                        .enumerate()
                        .filter(|(_, i)| !lanes[**i].done)
                        .map(|(k, _)| host_ids[k])
                        .collect();
                    let input =
                        Tensor::new(next_ids.as_slice(), device)?.reshape((next_live.len(), 1))?;
                    Self::forward_live(sessions, &next_live, &input)?
                }
            };
            live = next_live;
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

/// The device's `(libraries, pipelines)` cache sizes, or `(0, 0)` off Metal.
fn cache_sizes(device: &Device) -> (usize, usize) {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return m.kernels().cache_sizes();
    }
    let _ = device;
    (0, 0)
}
