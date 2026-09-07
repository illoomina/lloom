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

use crate::generation::{CommitBudget, LogitsProcessor, ReadbackCounters, ReadbackStrategy};
use crate::nn::model::{CacheReset, CausalLM};
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

    /// Greedy defaults: `max_tokens`, per-token commit, no lookahead, no split.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            budget: CommitBudget::per_token(),
            lookahead: 0,
            sample_split: false,
        }
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
    /// The prefill's dispatch census, `(kernel, count)`, under
    /// `CANDLE_METAL_PROFILE=1`; empty otherwise and off Metal.
    pub prefill_census: Vec<(String, u64)>,
    /// Step 0's dispatch census, on the same terms.
    pub step0_census: Vec<(String, u64)>,
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

/// Forward passes the host may have in flight -- committed and not yet
/// retired -- once it has encoded the next one.
///
/// Two is the depth the paced arms already run at: under `lookahead = 1` the
/// host waits for step `s`'s publish after encoding forward `s + 1`, and the
/// publish depends on forward `s - 1`'s logits, so forward `s - 1` has retired
/// before forward `s + 2` is encoded. The pacer makes the same bound hold on
/// an arm with no per-token wait, and it is a constant rather than a knob
/// because a deeper queue buys the GPU nothing -- the steps run serially on
/// their own dependency chain -- and costs the host a retained command buffer
/// and its completion handler per step it runs ahead (measured at 63 deep on
/// the drain arm before this).
pub const IN_FLIGHT_STEPS: usize = 2;

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
    taken: u64,
    waited: u64,
}

impl StepPacer {
    fn new() -> Self {
        Self {
            #[cfg(feature = "metal")]
            markers: std::collections::VecDeque::with_capacity(IN_FLIGHT_STEPS),
            taken: 0,
            waited: 0,
        }
    }

    /// Before encoding a forward: retire the oldest marker if the bound would
    /// otherwise be exceeded.
    fn before_forward(&mut self, device: &Device) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            while self.markers.len() >= IN_FLIGHT_STEPS {
                let oldest = self.markers.pop_front().expect("len >= IN_FLIGHT_STEPS");
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
        let mut stats = DecodeStats {
            step_ns: Vec::with_capacity(opts.max_tokens),
            sample_ns: Vec::with_capacity(if opts.sample_split {
                opts.max_tokens
            } else {
                0
            }),
            ..DecodeStats::default()
        };

        // Prefill: the whole prompt in one pass. Its GPU tail lands in the
        // first decode steps; what is timed here is the submission.
        let pos_before = self.seq.pos;
        let census_before = census_snapshot();
        let prefill_start = std::time::Instant::now();
        let mut logits = match self.seq.last_logits.take() {
            Some(held) if resume => held,
            _ => {
                let input = Tensor::new(prompt, &device)?.unsqueeze(0)?;
                let logits = self
                    .model
                    .forward(&input, self.seq.pos, &mut self.seq.cache)?
                    .squeeze(0)?;
                self.seq.pos += prompt.len();
                self.seq.tokens.extend_from_slice(prompt);
                logits
            }
        };
        end_executor_step(&device)?;
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;
        let census_prefill = census_snapshot();
        stats.prefill_census = census_delta(&census_before, census_prefill.clone());

        let mut produced = 0usize;
        let mut since_commit = 0usize;
        let mut eos = false;
        // The in-flight bound, on the arms whose commits the loop owns. The
        // blocking arm's readback drains the queue every step, so its stream
        // stays the incumbent's.
        let mut pacer = StepPacer::new();
        let pace = readback.loop_commits();
        // Every id the loop drains, streamed or not, accumulates here so the
        // returned `ids` is complete and the reconciliation below counts the
        // whole run. Draining for the stream advances the strategy's cursor, so
        // the final `take_ids` only sees the tail; the total is what matches
        // `produced`, not either part alone.
        let mut all_ids: Vec<u32> = Vec::with_capacity(opts.max_tokens);

        loop {
            if opts.reached(produced) {
                break;
            }
            stats.steps += 1;
            let step_start = std::time::Instant::now();
            if opts.sample_split {
                device.synchronize()?;
            }
            let sample_start = std::time::Instant::now();

            let feed = readback.sample_and_feed(sampler, &logits)?;
            produced += 1;

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
                    self.commit(&device, &mut stats)?;
                    since_commit = 0;
                    committed_this_step = true;
                }
                let is_eos = readback.eos_of_last()?;
                if opts.sample_split {
                    stats
                        .sample_ns
                        .push(sample_start.elapsed().as_nanos() as u64);
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
                if opts.sample_split {
                    stats
                        .sample_ns
                        .push(sample_start.elapsed().as_nanos() as u64);
                }
                // ENQUEUE, then learn. The GPU has this forward queued before
                // it retires the previous one. The pacer bounds the enqueue:
                // an arm with no per-token wait would otherwise run ahead
                // without limit (#690's marker pacer, replacing the removed
                // `apply_backpressure` spin).
                let t_sample = sample_start.elapsed().as_nanos() as u64;
                if pace {
                    pacer.before_forward(&device)?;
                }
                let t = std::time::Instant::now();
                begin_decode_step(&device);
                logits = self
                    .model
                    .forward(&feed, self.seq.pos, &mut self.seq.cache)?
                    .squeeze(0)?;
                end_decode_step(&device)?;
                self.seq.pos += 1;
                let t_forward = t.elapsed().as_nanos() as u64;
                // The in-flight bound AFTER the enqueue: retire the oldest
                // marker if the host would exceed `IN_FLIGHT_STEPS` ahead. This
                // is #690's pacer (Metal completion), which replaced the
                // unbounded counter-spin `apply_backpressure` livelocked on.
                // `backpressure_ns` reports what that bound cost this step.
                let t = std::time::Instant::now();
                if pace {
                    pacer.after_forward(&device)?;
                }
                let t_backpressure = t.elapsed().as_nanos() as u64;
                let t = std::time::Instant::now();
                let is_eos = readback.eos_of_last()?;
                if stats.steps <= FIRST_STEPS_TRACED {
                    stats.first_steps.push(StepPhases {
                        sample_ns: t_sample,
                        forward_ns: t_forward,
                        backpressure_ns: t_backpressure,
                        eos_ns: t.elapsed().as_nanos() as u64,
                    });
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
            let mut committed_now = committed_this_step;
            if loop_commits && !committed_this_step {
                since_commit += 1;
                if opts.budget.due(since_commit) {
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
                        stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                        break;
                    }
                    let go_on = on_tokens(&batch)?;
                    all_ids.extend_from_slice(&batch);
                    if !go_on {
                        stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                        break;
                    }
                }
            }
            end_executor_step(&device)?;
            stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
            if stats.steps == 1 {
                stats.step0_census = census_delta(&census_prefill, census_snapshot());
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
    /// streamed, the EOS and anything after it are not. The cache has advanced
    /// past the accepted prefix (`resolve` kept `accepted.len()` positions), so
    /// like `decode`'s `lookahead = 1` arm it can end one region past the EOS —
    /// which is safe for a single-turn run whose cache ends with it
    /// (`lfm2-chat`'s shape) and is why speculation needs the rewindable cache
    /// (`RotatingRing` + `InPlace`) `advance`/`resolve` require.
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
        let mut stats = DecodeStats {
            step_ns: Vec::with_capacity(opts.max_tokens),
            ..DecodeStats::default()
        };

        // Prefill: the whole prompt in one pass. Its argmax is the first anchor —
        // the token whose logits produced the first draft — exactly as the
        // `lfm2-dspark-verify` example takes `logits.argmax` when it does not hold
        // the last prompt token back. `forward` taps (the tap hooks are woven into
        // both entry points, `lfm2.rs`), so the cache carries the prefill's taps
        // for `take_probe` below.
        let pos_before = self.seq.pos;
        let prefill_start = std::time::Instant::now();
        let input = Tensor::new(prompt, &device)?.unsqueeze(0)?;
        let logits = self
            .model
            .forward(&input, self.seq.pos, &mut self.seq.cache)?;
        self.seq.pos += prompt.len();
        self.seq.tokens.extend_from_slice(prompt);
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;

        let mut anchor_tok = logits
            .squeeze(0)?
            .argmax(lloom_tensor::D::Minus1)?
            .to_scalar::<u32>()?;

        // Seed the drafter's per-seq state: `begin` clears the context, then the
        // prefill's probe keys it from every prompt position and the prefill's
        // own argmax becomes the block anchor (the accepted tail) — the same
        // state the first round would see after a fully-accepted prompt.
        spec.begin(seq, prompt)?;
        let prefill_probe = self.model.take_probe(&mut self.seq.cache)?;
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
            let (accepted, timing) = self.speculative_round(spec, &[seq], &[anchor], 0.0, probe)?;
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

        let mut lanes: Vec<Lane<<M as Probes<S::Probe>>::Pending>> = Vec::with_capacity(n);

        // Prefill every sequence before reading any back, so the GPU has all
        // the prompts queued while the host waits on the first.
        let mut logits = Vec::with_capacity(n);
        for (i, s) in sessions.iter_mut().enumerate() {
            s.seq.last_logits = None;
            let pos_before = s.seq.pos;
            let prefill_start = std::time::Instant::now();
            let input = Tensor::new(prompts[i], device)?.unsqueeze(0)?;
            let l = s.model.forward(&input, s.seq.pos, &mut s.seq.cache)?;
            s.seq.pos += prompts[i].len();
            s.seq.tokens.extend_from_slice(prompts[i]);
            logits.push(l);
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
                    ..DecodeStats::default()
                },
                pending: None,
                in_flight: RoundTiming::default(),
                round_start_ns: 0,
                done: false,
            });
        }

        // Anchors and drafter seeds, as the serial loop: the prefill's argmax is
        // token 0 and the block anchor; the drafter is keyed from every prompt
        // position (`design/docs/speculation.md` §10.2w).
        for i in 0..n {
            let lane = &mut lanes[i];
            let s = &mut sessions[i];
            let anchor_tok = logits[i]
                .squeeze(0)?
                .argmax(lloom_tensor::D::Minus1)?
                .to_scalar::<u32>()?;
            spec.begin(SeqId(i), prompts[i])?;
            let prefill_probe = s.model.take_probe(&mut s.seq.cache)?;
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
        drop(logits);

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
                let (accepted, probe_out) = s.model.verify_complete(pending, &mut s.seq.cache)?;
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

        let decode = Tensor::new(&[0u32], device)?.unsqueeze(0)?;
        let _ = self.model.forward(&decode, pos, &mut self.seq.cache)?;

        // Force the driver to finish compiling before the clock the caller is
        // about to stop; a cold pipeline creation left in flight would surface
        // on the first real forward instead.
        device.synchronize()?;

        // Leave nothing behind: the warmup wrote KV, conv state and the position
        // counter, and the next call is the user's real prefill from position 0.
        self.reset();

        Ok(cache_sizes(device))
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
