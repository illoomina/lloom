//! The readback strategy: how a decode step feeds the next forward pass and how
//! the host learns EOS (`design/docs/decode-driver-design.md`, seam A).
//!
//! Three arms on one axis, and they differ only in where the round trip is:
//!
//! | arm | token comes to the host | EOS is learned by | per-token wait |
//! |---|---|---|---|
//! | [`Blocking`] | every step (`to_scalar`) | a host compare | `waitUntilCompleted`, inside the readback |
//! | [`Ring`] with [`EosRead::Blocking`] | never — device-resident slot | one 1-byte readback per step | `waitUntilCompleted`, inside the flag read |
//! | [`Ring`] with [`EosRead::Poll`] | never | polling the kernel-written counter | a poll, no Mach trap (§11.5n) |
//!
//! # What a [`RingReader`] streams
//!
//! The ring's stamp, and each arm stamps on a stated grain: the per-step reads
//! (the flag and the poll) stamp every token; the deferred reads stamp once
//! per commit window on the fixed cadence, every token under the adaptive
//! commit (the stamp after the fused kernel, or the loop's `publish_marker`
//! after an unfused step), and their last window at `drain_all`. Every stamp
//! is a dispatch of its own bound to the slots it certifies: the host sees a
//! store of one dispatch and a store of another in the order the command
//! stream's barrier between them gives, and two stores of one dispatch in
//! either order. A reader on a
//! deferred read on the fixed cadence therefore streams in windows, and sees
//! the whole run once the run has returned.
//!
//! The loop ([`crate::nn::Session::decode`]) names the trait and never a
//! concrete arm, so the harness's `RESULT` line renders the strategy that ran
//! from [`ReadbackStrategy::arms`] rather than echoing a flag (§7.1a-i).
//!
//! # What the ring arms hold that the blocking arm does not
//!
//! The device-side history. A loop that never learns its tokens cannot hash
//! them, and reading each token back is the round trip the ring removes, so the
//! ids accumulate into one `[capacity]` `U32` slot by the same `slice_set` the
//! ring uses for its input and are read back **once**, by [`ReadbackStrategy::take_ids`],
//! after the loop (§11.5j). An EOS is written to the slot too — the write is
//! encoded before the host knows — and is retracted by [`ReadbackStrategy::eos_of_last`]
//! rolling the write cursor back one, so the next token overwrites it.

use crate::generation::budget::{DeviceForcing, HostForcing, ThinkingBudget};
use crate::generation::poll::PollMechanism;
use crate::generation::token_ring::TokenRing;
use crate::generation::{LogitsProcessor, Sampling};
use lloom_tensor::{DType, Device, Result, Tensor};

/// How long the EOS wait parks on the publish event before it takes the
/// blocking read: a fail-safe two orders of magnitude above a token, never a
/// latency budget.
pub const EOS_PARK_TIMEOUT_MS: u64 = 2_000;

/// Which readback arms a strategy is, for the `RESULT` line.
///
/// Rendered from the strategy that ran rather than from the flags that selected
/// it, which is the whole of §7.1a-i: a wrong arm compares equal to every run
/// of that arm, where an absent one compares equal to nothing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadbackArms {
    /// The token stays on the device (`TokenRing=on`).
    pub token_ring: bool,
    /// The EOS flag is polled rather than read back (`NonblockingEos=on`).
    pub nonblocking_eos: bool,
}

/// The engagement counters a strategy reports, for the `RESULT` line.
///
/// A mechanism whose engagement nobody counts is one whose null reads as a
/// result (§2.4, #500): `ring_publishes` must reconcile against the ids the
/// loop returns plus the EOS tokens it retracted, or one of the two is wrong.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadbackCounters {
    /// Tokens published into the ring's input slot.
    pub ring_publishes: u64,
    /// Of those, the ones the fused argmax·eos kernel published
    /// (`Device::argmax_publish`, lloom #735 step 5) and a `ring_stamp`
    /// stamped: two dispatches in place of the cast, the argmax, three blits,
    /// the gate kernel and the stamp.
    pub fused_publishes: u64,
    /// Publishes that turned out to be EOS and were retracted from the history.
    pub ring_eos_publishes: u64,
    /// Publishes taken back because the step their draw came from was redone
    /// (a provisional step settled behind the next forward,
    /// [`ReadbackStrategy::retract_draw`]) and drawn again.
    pub ring_retracted: u64,
    /// Iterations of the EOS wait, across every step.
    pub poll_iters: u64,
    /// Steps on which the non-blocking poll gave up and took the blocking read.
    pub poll_fallbacks: u64,
    /// Steps whose EOS wait parked on the ring's publish event instead of
    /// polling the stamp (`EosRead::Poll` with the event armed).
    pub publish_parks: u64,
    /// Publishes discarded past an EOS the host learned late
    /// ([`EosRead::HostOnDrain`]): the forwards the loop enqueued after the EOS
    /// before the drain showed it. `ring_publishes == ids + ring_eos_publishes
    /// + ring_discarded + ring_retracted` is the reconciliation.
    pub ring_discarded: u64,
    /// [`EosRead::GpuBatchedNonblocking`]: drains answered by a staged
    /// `eos_index` whose command buffer had already completed (no wait).
    pub eos_index_hits: u64,
    /// [`EosRead::GpuBatchedNonblocking`]: drains at which no staged index
    /// covering the batch had completed, answered by the host scan instead
    /// (no wait either; the GPU's answer arrived too late to be used).
    pub eos_index_late: u64,
}

/// How a decode step feeds the next forward pass and learns EOS.
///
/// The trait is the seam; the loop is generic over it and monomorphizes
/// (`CONTRIBUTING.md` §3.6: no `dyn` on the decode path).
///
/// # Protocol
///
/// One [`Self::sample_and_feed`] per step, then exactly one
/// [`Self::eos_of_last`] for that step before the next `sample_and_feed`. The
/// loop may encode the next forward pass between the two — that is the
/// lookahead the design's hot-path invariant asks for — iff
/// [`Self::admits_lookahead`] says the answer is not already known at feed
/// time.
pub trait ReadbackStrategy {
    /// The device the strategy's tensors live on; the loop's uploads and its
    /// periodic commit go through it.
    fn device(&self) -> &Device;

    /// Which arms this strategy renders as.
    fn arms(&self) -> ReadbackArms;

    /// Whether the loop may encode the next forward before [`Self::eos_of_last`]
    /// has answered. `false` when the answer is known at feed time (the
    /// blocking arm), where a lookahead would only add a wasted pass.
    fn admits_lookahead(&self) -> bool;

    /// Whether the loop owns the commit: `true` for a strategy whose wait
    /// needs the publish committed first and whose in-flight queue nothing
    /// else prunes (the ring arms), `false` for one whose own readback commits
    /// and waits every step (the blocking arm), where a loop-side commit would
    /// be a second command buffer per token the incumbent never cut.
    fn loop_commits(&self) -> bool;

    /// Refuse a sampling arm this strategy cannot serve without a host round
    /// trip it would not report (§7.1a-i's wrong-arm shape).
    fn check_sampler(&self, sampling: &Sampling) -> Result<()>;

    /// Sample this step's token from `logits` and return the next forward's
    /// `[1, 1]` input. The blocking arm reads the id to the host here; the ring
    /// arms leave it on the device and return the ring's own slot.
    fn sample_and_feed(&mut self, sampler: &mut LogitsProcessor, logits: &Tensor)
        -> Result<Tensor>;

    /// Arm a thinking budget from the next [`Self::sample_and_feed`] on: each
    /// step feeds the budget's token ([`crate::generation::budget`]), which is
    /// the draw except while its forced tail runs. `None` disarms. A strategy
    /// arms it for the next run only, so its caller disarms after the run.
    /// The default refuses a budget rather than feed the draws under it.
    fn set_thinking_budget(&mut self, budget: Option<&ThinkingBudget>) -> Result<()> {
        if budget.is_some() {
            lloom_tensor::bail!(
                "the `{}` readback strategy cannot feed a thinking budget's forced tokens",
                self.eos_read_arm()
            );
        }
        Ok(())
    }

    /// Stage a **prompt chunk** as the next forward's input and return the tensor
    /// to feed it — the streamed-prefill counterpart of [`Self::sample_and_feed`],
    /// where the tokens are GIVEN (the prompt) rather than sampled. The state
    /// machine steps over the prompt in these chunks before it steps over its own
    /// generated tokens; feeding a chunk here is one such step's input.
    ///
    /// **Default**: allocate a fresh `[1, n]` input tensor (the incumbent
    /// `Tensor::new(prompt)` path) — correct for any arm, one alloc per chunk. The
    /// ring arm overrides this to stage the tokens into its reused input buffer
    /// (zero-copy [`TokenRing::stage_prompt`]) and return an `input_for` narrow,
    /// so a streamed prefill pays no per-chunk allocation.
    fn stage_prompt_chunk(&mut self, tokens: &[u32]) -> Result<Tensor> {
        Tensor::new(tokens, self.device())?.unsqueeze(0)
    }

    /// Publish the completion marker for the steps fed since the last call — the
    /// per-commit-window write the drain clamps to and the host WFE-waits on.
    ///
    /// The loop calls this at each commit; the strategy decides whether THIS
    /// commit is a LIVE marker write (advancing the device word the host waits on)
    /// or a skipped one (the wake-cadence throttle). `force_live` is the loop's
    /// signal that the marker must be written regardless of cadence — the TTFT
    /// ramp passes it so token 0 never waits `live_every` commits.
    ///
    /// **The default is a no-op**: the blocking arm has its own per-step readback,
    /// and the per-step device-EOS arm already stamped inside
    /// [`Self::sample_and_feed`] (its marker and EOS flag coincide). Only the arm
    /// that defers its stamp to the commit window overrides this.
    fn publish_marker(&mut self, force_live: bool) -> Result<()> {
        let _ = force_live;
        Ok(())
    }

    /// The window closing at this step's commit is about to be drained: enqueue
    /// whatever device work the drain will want to read, NOW — before the loop
    /// encodes its lookahead forward — so that work rides a command buffer that
    /// holds the window and nothing after it. The loop calls this right after
    /// [`Self::sample_and_feed`] on a step whose commit is due; a strategy that
    /// has nothing to enqueue ahead of time ignores it (the default). Only the
    /// non-blocking batched-EOS arm overrides it: its `eos_index` staged here
    /// completes with the window's own buffer, where one enqueued at
    /// [`Self::publish_marker`] sits behind the lookahead forward and completes
    /// a step later. `in_ttft_ramp` is the loop's early-commit ramp (a commit
    /// every step for the first period): windows there are a token or two wide
    /// and the host scan answers them for free, so nothing is staged and the
    /// first token's latency carries no extra cut.
    fn close_window(&mut self, in_ttft_ramp: bool) -> Result<()> {
        let _ = in_ttft_ramp;
        Ok(())
    }

    /// The ids the last [`Self::sample_and_feed`] fed, if this arm holds them
    /// on the host — the blocking arm's sampled id. `None` on the ring arms,
    /// whose token never comes to the host. The loop hands them to
    /// [`crate::nn::CausalLM::note_input_ids`] so a model that needs its
    /// input ids host-side does not read them back from the device.
    fn fed_host_ids(&self) -> Option<&[u32]> {
        None
    }

    /// Whether [`Self::retract_draw`] can take the last draw back (a
    /// provisional step's redraw, `CausalLM::settle_step`). The default
    /// cannot.
    fn retracts_draw(&self) -> bool {
        false
    }

    /// Take back the last [`Self::sample_and_feed`], before
    /// [`Self::eos_of_last`] or a drain has seen it: its id leaves the ids,
    /// and the strategy is as it was before the call. Refused by default.
    fn retract_draw(&mut self) -> Result<()> {
        lloom_tensor::bail!(
            "{}: this readback arm cannot take a draw back",
            self.eos_read_arm()
        )
    }

    /// Was the token fed by the last [`Self::sample_and_feed`] an EOS? Waits
    /// by this arm's own mechanism until it knows; a `true` retracts that
    /// token from the ids [`Self::take_ids`] will return.
    fn eos_of_last(&mut self) -> Result<bool>;

    /// The ids published SO FAR and not yet drained, EOS excluded — the
    /// incremental, non-blocking drain for streaming. On the ring's poll arm it
    /// returns only what the GPU has published (it never waits and never drains
    /// past the visibility signal), so a caller mid-loop may get fewer than were
    /// fed; the rest arrive on a later call or from [`Self::drain_all`]. On the
    /// blocking arm every fed id is already visible, so it returns them all.
    fn take_ids(&mut self) -> Result<Vec<u32>>;

    /// [`Self::take_ids`], appended to `out` — for a loop keeping the ids a
    /// sequence owns while a reader streams them, at no allocation a step. The
    /// default goes through `take_ids`; the ring reads through its mapped
    /// window into a buffer it keeps.
    fn take_ids_into(&mut self, out: &mut Vec<u32>) -> Result<()> {
        out.extend(self.take_ids()?);
        Ok(())
    }

    /// EVERY id not yet drained, waiting for the GPU if it must — the final drain
    /// after the loop, where the contract is completeness rather than
    /// non-blocking. On the ring arms this synchronises up to the last fed slot
    /// (the poll arm cannot otherwise know the tail landed) and returns the whole
    /// remainder; on the blocking arm it is the same as [`Self::take_ids`], which
    /// already had everything. A caller streams this too, so nothing is lost
    /// between the last incremental drain and the end.
    fn drain_all(&mut self) -> Result<Vec<u32>>;

    /// The index of the first EOS id in a drained `batch`, for the arm that
    /// learns EOS host-side ([`EosRead::HostOnDrain`]) rather than from a
    /// per-token device flag. The loop calls this on each streamed batch and, on
    /// `Some(i)`, streams `batch[..i]` and stops, and once at the end of a run
    /// on the tail or on the run's whole ids. A strategy that answers from a
    /// device index answers only for the batch its last [`Self::take_ids`]
    /// returned, whose frame the index is in, and scans any other.
    ///
    /// **The default is `None`** — every arm that learns EOS per token
    /// ([`Self::eos_of_last`]) has already excluded the EOS token before it
    /// reaches a batch, so their batches carry no EOS to find. Only the
    /// host-on-drain ring overrides this.
    fn eos_in_drained(&self, batch: &[u32]) -> Option<usize> {
        let _ = batch;
        None
    }

    /// The loop found an EOS at [`Self::eos_in_drained`]'s index in a batch and
    /// stopped: `dropped_in_batch` ids followed the EOS in that batch, and
    /// everything the strategy still holds undrained is overrun too. The
    /// strategy accounts for the EOS and the discards, and forgets the
    /// undrained slots so a later run on the same strategy does not return
    /// them as new tokens. Only the host-on-drain ring has anything to do;
    /// the default is a no-op because the per-token arms never reach a batch
    /// with an EOS in it.
    fn note_eos_truncation(&mut self, dropped_in_batch: usize) {
        let _ = dropped_in_batch;
    }

    /// Which EOS-learning arm this strategy is, for the `RESULT` line: `host`
    /// (the id is read back and compared at feed time), `flag` (the ring's
    /// device flag, one blocking readback per step), `poll` (the flag polled
    /// through the generation counter) or `drain` (host-side, from the drained
    /// ids). Rendered from the strategy rather than from a flag (§7.1a-i).
    fn eos_read_arm(&self) -> &'static str;

    /// The stop set as a U32 device tensor, for the EOS gate
    /// (`DecodeOptions::eos_gate`) to compare the sampled token against on
    /// the GPU. `None` on a strategy that holds no device copy.
    fn eos_ids_tensor(&self) -> Option<&Tensor> {
        None
    }

    /// A reader over the ring's mapped words for another thread
    /// (`Ring::reader`); `None` on a strategy with no mapped ring. On the
    /// trait so a loop generic over the strategy (`lfm2-release`) can take one.
    fn reader(&self) -> Option<RingReader> {
        None
    }

    /// Whether every token is stamped as it is published, so a
    /// [`RingReader`] sees token `k` before the command stream runs anything
    /// encoded after it — what a GPU wait on a host-filled slot needs
    /// (`CausalLM::host_feed`). A deferred read on the fixed cadence stamps
    /// once per window, after the window's forwards (§11.5z-xvi), and a
    /// forward waiting on one of its tokens would never run. `false` (the
    /// default) on a strategy with no reader.
    fn stamps_each_token(&self, adaptive_commit: bool) -> bool {
        let _ = adaptive_commit;
        false
    }

    /// Arm the ring's publish event for this run where a host thread will park
    /// on the stamp: a [`RingReader`] that parks (`for_reader`), or the per-step
    /// EOS wait on the fixed cadence when it parks (`eos_park`). Disarmed
    /// otherwise, so a run with no such waiter (the adaptive commit's, and the
    /// default fixed cadence's) encodes nothing more per stamp. A no-op on a
    /// strategy with no ring.
    fn arm_publish_event(
        &mut self,
        for_reader: bool,
        eos_park: bool,
        adaptive_commit: bool,
    ) -> Result<()> {
        let _ = (for_reader, eos_park, adaptive_commit);
        Ok(())
    }

    /// Under the adaptive commit: publish each step with the fused
    /// argmax·eos·stamp kernel (`Device::argmax_publish`) where the strategy
    /// can — the ring with an argmax sampler on Metal. After a step for which
    /// [`Self::publishes_fused`] holds, the loop skips its own stamp and gate
    /// step: the strategy stamped, and the kernel raised the gate's flag.
    fn set_fused_publish(&mut self, on: bool) {
        let _ = on;
    }

    /// Whether the last [`Self::sample_and_feed`] published through the fused
    /// kernel. Per step, not per run: a sampler the kernel does not serve
    /// takes the unfused chain, whose stamp and gate are the loop's.
    fn publishes_fused(&self) -> bool {
        false
    }

    /// The engagement counters so far.
    fn counters(&self) -> ReadbackCounters;
}

/// Today's arm: the sampled id comes to the host every step, the EOS test is a
/// host compare, and the 441 µs seam (§11.5a) lives here, honestly.
pub struct Blocking {
    device: Device,
    eos_ids: Vec<u32>,
    ids: Vec<u32>,
    last: Option<u32>,
    forcing: Option<HostForcing>,
    /// The budget's state before the last draw ([`ReadbackStrategy::retract_draw`]).
    forcing_before: Option<crate::generation::budget::ForcingMark>,
}

impl Blocking {
    pub fn new(eos_ids: &[u32], device: &Device) -> Result<Self> {
        if eos_ids.is_empty() {
            lloom_tensor::bail!("Blocking: eos_ids is empty, so the loop would never terminate");
        }
        Ok(Self {
            device: device.clone(),
            eos_ids: eos_ids.to_vec(),
            ids: Vec::new(),
            last: None,
            forcing: None,
            forcing_before: None,
        })
    }
}

impl ReadbackStrategy for Blocking {
    fn device(&self) -> &Device {
        &self.device
    }

    fn arms(&self) -> ReadbackArms {
        ReadbackArms {
            token_ring: false,
            nonblocking_eos: false,
        }
    }

    fn admits_lookahead(&self) -> bool {
        false
    }

    fn loop_commits(&self) -> bool {
        false
    }

    fn check_sampler(&self, _sampling: &Sampling) -> Result<()> {
        // Every arm round-trips here, so every arm is honest.
        Ok(())
    }

    fn sample_and_feed(
        &mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        // The argmax arm draws ON THE DEVICE and feeds that tensor: the id
        // comes to the host from a slot for the stream and the EOS test, and
        // the next forward's input is the draw itself, never re-uploaded
        // (#814 part 5 item A — the ring's shape, on the blocking arm). The
        // other arms are host draws by construction (a sort, a multinomial),
        // so they read the id and upload it as they always did.
        self.forcing_before = self.forcing.as_ref().map(HostForcing::mark);
        // A digesting sampler reads the logits back anyway: the host draw.
        if self.forcing.is_none() && *sampler.sampling() == Sampling::ArgMax && !sampler.digests() {
            if let Some((draw, id)) = sampler.draw_and_read(logits)? {
                self.ids.push(id);
                self.last = Some(id);
                return draw.reshape((1, 1));
            }
        }
        let id = match &mut self.forcing {
            Some(f) => sampler.sample_fed(logits, |d| f.feed(d))?,
            None => sampler.sample(logits)?,
        };
        self.ids.push(id);
        self.last = Some(id);
        Tensor::new(&[id], &self.device)?.unsqueeze(0)
    }

    fn set_thinking_budget(&mut self, budget: Option<&ThinkingBudget>) -> Result<()> {
        self.forcing = budget.cloned().map(HostForcing::new);
        Ok(())
    }

    fn fed_host_ids(&self) -> Option<&[u32]> {
        self.last.as_ref().map(std::slice::from_ref)
    }

    fn retracts_draw(&self) -> bool {
        true
    }

    fn retract_draw(&mut self) -> Result<()> {
        if self.last.take().is_none() {
            lloom_tensor::bail!("Blocking::retract_draw: no draw to take back");
        }
        self.ids.pop();
        if let (Some(f), Some(m)) = (self.forcing.as_mut(), self.forcing_before.take()) {
            f.rewind(m);
        }
        Ok(())
    }

    fn eos_of_last(&mut self) -> Result<bool> {
        let Some(id) = self.last.take() else {
            lloom_tensor::bail!("Blocking::eos_of_last called before sample_and_feed");
        };
        if self.eos_ids.contains(&id) {
            self.ids.pop();
            return Ok(true);
        }
        Ok(false)
    }

    fn take_ids(&mut self) -> Result<Vec<u32>> {
        Ok(std::mem::take(&mut self.ids))
    }

    fn drain_all(&mut self) -> Result<Vec<u32>> {
        // The blocking arm's ids are already host-side and always complete, so
        // the final drain is the incremental one.
        Ok(std::mem::take(&mut self.ids))
    }

    fn eos_read_arm(&self) -> &'static str {
        "host"
    }

    fn counters(&self) -> ReadbackCounters {
        ReadbackCounters::default()
    }
}

/// How a [`Ring`] learns EOS.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EosRead {
    /// Read the 1-byte flag through `to_vec1` — one blocking readback per step,
    /// narrowed from the token to a flag (§11.5j's arm).
    Blocking,
    /// Poll the kernel-written generation counter with the given mechanism,
    /// falling back to the blocking read after a bounded number of iterations
    /// (§11.5n's arm).
    Poll(PollMechanism),
    /// Learn EOS **host-side, off the hot path**: the device never computes the
    /// EOS flag (the `broadcast_eq → max_all → slice_set(eos)` dispatches are
    /// skipped, ~5/token), and the host compares the drained ids against the EOS
    /// set in plain Rust at the drain point instead. `eos_of_last` therefore
    /// always answers `false` — the loop learns EOS from
    /// [`Ring::eos_in_drained`] on the batch it streams, not per token.
    ///
    /// The trade is the overrun: EOS is discovered only when the ids are drained
    /// (on the commit cadence), so up to a cadence-worth of forwards may run past
    /// it and their tokens are truncated from the output. At B=1 with a ring
    /// wide enough to hold the tail that is a rare, once-per-generation cost, not
    /// a per-token one — and the bound is a knob, not a committed shape.
    ///
    /// Like [`Self::Poll`], this rides the device-resident token and the stamped
    /// generation counter (the drain's visibility signal), so it needs the ring
    /// mapped for the zero-copy drain; the drain's own fallback covers the
    /// unmapped case.
    HostOnDrain,
    /// Learn EOS with a **batched GPU check, once per commit window** — the middle
    /// ground between [`Self::Poll`] (a ~5-dispatch flag EVERY token) and
    /// [`Self::HostOnDrain`] (a host scan, no GPU work). The per-step path writes
    /// only the input slot (like `HostOnDrain`); at each commit the window's tokens
    /// go through ONE `eos_index` dispatch (a full-threadgroup min-reduction) that
    /// returns the first stop-token index in the window, or `n`. The host reads
    /// that one index per commit and never scans the tokens itself — this holds at
    /// B>1 too, where the GPU hands back the position per sequence.
    ///
    /// Same overrun trade as `HostOnDrain` (EOS discovered on the commit cadence,
    /// ≤ a window past it), and it rides the same stamped counter for drain
    /// visibility; the batched check is the added device work, paid once per window
    /// rather than per token. A window that straddles a ring segment boundary or a
    /// quantum>1 (waste-gap) layout falls back to the host scan for that drain.
    GpuBatched,
    /// [`Self::GpuBatched`] without its one wait. `GpuBatched` reads the window's
    /// `eos_index` result with `to_vec1`, which waits on the queue's TAIL — under
    /// `lookahead = 1` that is the next step's forward, so every commit window
    /// drains the pipeline the loop just filled. Here the index is staged
    /// (`Tensor::stage_read`, fenced by its own command buffer) and, through
    /// [`ReadbackStrategy::close_window`], enqueued BEFORE the lookahead forward
    /// so that buffer holds the window and nothing after it; at the drain the
    /// index is read only if that buffer has already completed
    /// (`StagedRead::is_complete`, no wait) and the host scans the batch
    /// otherwise — so the drain never waits and is always right, and the
    /// counters say which path answered (`eos_index_hits` / `eos_index_late`).
    /// Two windows' indices are kept, because the drain lags the marker by up to
    /// a window on the deferred arms and the older index is the one that has
    /// completed.
    GpuBatchedNonblocking,
}

/// A window's staged `eos_index`, with the absolute token range it scanned
/// (`[start, start + len)`, in the history's monotonic token count), so a drain
/// starting elsewhere can still map the index onto its batch.
struct StagedWindow {
    staged: lloom_tensor::StagedRead,
    start: usize,
    len: usize,
}

/// The per-step token production geometry, known BEFORE the run (unlike a KV
/// length): what sizes a lock-free history whose window never laps the drain.
///
/// `quantum = batch × (max_draft + 1)` is the most tokens ONE step/round can
/// publish — 1 for B=1 greedy, `K+1` for DSpark (`accepted ≤ K` + the target's
/// bonus, proven at `decode.rs`'s accept loop). The live window the host can be
/// behind is `period × quantum + batch × lookahead` (drain fires every `period`
/// steps; lookahead enqueues that many extra forwards). See [`Geometry::cap`].
#[derive(Clone, Copy, Debug, Default)]
pub struct Geometry {
    /// Sequences in the batch (`B`). 1 for the single-sequence chat path.
    pub batch: usize,
    /// The speculator's draft-block ceiling (`K`); 0 for greedy.
    pub max_draft: usize,
    /// The commit/drain period in steps (`steps:N` → N, `per-token` → 1).
    pub period: usize,
    /// Forwards the host may enqueue before learning EOS (0 or 1).
    pub lookahead: usize,
    /// The streamed-prefill micro-batch width (`prefill_ubatch`) — how many prompt
    /// tokens one prefill step stages into the input buffer. `0` when the caller
    /// does not stream the prefill (the input buffer is then sized to `quantum`,
    /// the decode/draft width). Sizes the ring's input buffer to
    /// `max(quantum, max_input_width)`; it does NOT affect `cap()` (the history
    /// window) or the commit cadence.
    pub max_input_width: usize,
    /// The host/GPU in-flight depth (the pacer's bound). The input side needs
    /// `depth + 1` buffers so a host-staged chunk does not overwrite a buffer an
    /// in-flight forward is reading (see `TokenRing::inputs`). `0` reduces to a
    /// single input buffer (the decode path, whose device-side writes are already
    /// queue-ordered). Only matters when streaming a host token source
    /// (`max_input_width > quantum`).
    pub in_flight_depth: usize,
}

impl Geometry {
    /// The most tokens one step/round publishes: `batch × (max_draft + 1)`.
    pub fn quantum(&self) -> usize {
        self.batch.max(1) * (self.max_draft + 1)
    }

    /// The tight history capacity — no safety margin, derived exactly so a wrong
    /// bound surfaces as the hard lap bail (caught by ASan/fuzz) rather than
    /// hidden. `window = period × quantum + batch × lookahead`; `cap` rounds that
    /// up to a whole `quantum` and adds ONE quantum of headroom for the
    /// write-reset's dead tail (a step that would straddle the wrap resets to the
    /// start, leaving ≤ `quantum − 1` slots unused). Provable: the loop drains
    /// every `period` steps and each step adds ≤ `quantum`, so `fed − drained ≤
    /// window ≤ cap − quantum` always.
    pub fn cap(&self) -> usize {
        let q = self.quantum();
        let window = self.period.max(1) * q + self.batch.max(1) * self.lookahead;
        // round window up to a multiple of q, then + q for the reset tail.
        let rounded = window.div_ceil(q) * q;
        rounded + q
    }
}

/// A zero-copy, lock-free view of a [`Ring`]'s published history for a thread
/// that is not the decode loop — a consumer that wants to read the tokens
/// straight off the device-resident ring instead of through a host queue.
///
/// It owns nothing the loop uses: clones of the segment tensors and of the
/// ring's `step` word (so the buffers outlive it), the mapped pointers into
/// them resolved ONCE at construction, and its own cursor. The loop is never
/// touched, locked or slowed by it: `published()` is one plain load of the
/// word the GPU stamps after each slot write (the same visibility signal the
/// loop's own drain clamps to — the slot is written BEFORE the stamp, so
/// `[0, published)` is readable), and `read_new` copies the slots between the
/// cursor and that value through the mapped pointers. No lock on either path:
/// the tensor's storage lock is taken when the pointers are resolved and
/// never again. The loads are `read_volatile` only so the compiler cannot
/// fold them — the writer is the GPU, which it does not see — which on arm64
/// is the same `ldr` a plain load is; nothing atomic is involved.
///
/// Two contracts the reader carries itself, because the loop does not know it
/// exists:
///
/// * **EOS**: the loop retracts an EOS token from its history and reuses the
///   slot, but the stamp counted it; the reader stops at the first stop id it
///   sees and reports `ended()` — after that it reads nothing more. Every such
///   retraction leaves the stamp one further ahead of the history's slots for
///   the rest of the ring's life, so the reader maps a stamp to its slot
///   through the shift the ring had when the reader was taken (`slot_shift`),
///   which holds for every token of the run it was taken for.
/// * **Laps**: the loop's lap guard protects the loop's own drain, not this
///   reader. If the reader falls more than `capacity` tokens behind the writer,
///   `read_new` returns [`RingLap`] rather than stale slots; how far behind a
///   consumer may fall is the ring's `capacity` (the consumer slack,
///   `LLOOM_RING_SLACK_BYTES`), which the caller sizes.
///
/// Only the dense (`quantum == 1`, K=1) history has a slot-per-token layout a
/// reader can index; [`Ring::reader`] returns `None` otherwise, and off Metal
/// (no mapped words).
#[derive(Debug)]
pub struct RingReader {
    /// Held for liveness only: the pointers below are into these buffers.
    _segments: Vec<Tensor>,
    _step: Tensor,
    #[cfg(feature = "metal")]
    seg_bases: Vec<*const u32>,
    #[cfg(feature = "metal")]
    step_ptr: *const u32,
    tiles_per_seg: usize,
    total_tiles: usize,
    eos_ids: Vec<u32>,
    /// The last stamp this reader has consumed.
    cursor: usize,
    /// The ring's generation when the reader was taken: the stamp before the
    /// first token of the run the reader was taken for.
    head: usize,
    /// Added to a stamp ordinal, modulo `total_tiles`, to give its slot: the
    /// history slots run behind the stamps by one per EOS retracted so far.
    slot_shift: usize,
    ended: bool,
    /// The ring's publish event when the run armed it: a reader with nothing
    /// new is woken from it ([`Self::notify_on_publish`]).
    #[cfg(feature = "metal")]
    publish: Option<std::sync::Arc<lloom_tensor::metal_backend::PublishEvent>>,
}

// SAFETY: the pointers are into Metal buffers the held tensors keep alive; the
// reader only ever loads through them, and a consumer thread is the point.
unsafe impl Send for RingReader {}

/// The writer overtook a [`RingReader`]: more than `capacity` tokens were
/// published since the reader's `cursor`, so the slots between are no longer
/// the tokens they were.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RingLap {
    pub cursor: usize,
    pub published: usize,
    pub capacity: usize,
}

impl std::fmt::Display for RingLap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ring reader lapped: cursor {} published {} capacity {}",
            self.cursor, self.published, self.capacity
        )
    }
}

impl RingReader {
    /// The generation the GPU has published so far — one load, no lock, no
    /// wait. `None` off Metal (no mapped word).
    pub fn published(&self) -> Option<u32> {
        #[cfg(feature = "metal")]
        {
            // SAFETY: a mapped pointer into a buffer `_step` keeps alive.
            Some(unsafe { self.step_ptr.read_volatile() })
        }
        #[cfg(not(feature = "metal"))]
        None
    }

    /// The last stamp consumed, EOS included — the reader's own position.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Whether a stop id ended the stream for this reader.
    pub fn ended(&self) -> bool {
        self.ended
    }

    /// The loop took back the last token this reader read
    /// ([`ReadbackStrategy::retract_draw`]): its history slot is written
    /// again under the NEXT stamp, as an EOS retraction leaves it, so the
    /// stamps run one further ahead of the slots from here on. The cursor
    /// stays — the stamp it consumed is spent; the next one is the token
    /// drawn again.
    pub fn retract_one(&mut self) {
        self.slot_shift = (self.slot_shift + self.total_tiles - 1) % self.total_tiles;
    }

    /// Skip everything the ring had published when this reader was taken: a
    /// reader for a LATER run on the same ring (a follow-up turn, the next
    /// request on a pooled ring) starts at that run's first token instead of
    /// at zero, where it would re-read the earlier runs' tokens — and their
    /// EOS, which would end it at once. The position is the ring's own count
    /// of the stamps it issued ([`Ring::reader`]), not a load of the device
    /// word, so it is exact whether or not the earlier run's tail has landed.
    pub fn resume_at_published(&mut self) {
        self.cursor = self.head;
    }

    /// The tokens published since the last call, appended to `out`; stops at
    /// the first stop id (not appended) and marks the reader ended. Returns
    /// how many were appended, or [`RingLap`] if the writer overtook the
    /// cursor. Never waits: an unmapped word or segment reads as nothing new.
    pub fn read_new(&mut self, out: &mut Vec<u32>) -> std::result::Result<usize, RingLap> {
        self.read_published(out, true)
    }

    /// [`Self::read_new`] that does not stop at a stop id: every token the
    /// ring stamped, the EOS and the forwards queued past it included — what
    /// a consumer that must answer every forward the loop encoded reads
    /// (the host fill's producer, `CausalLM::host_feed`).
    pub fn read_all_new(&mut self, out: &mut Vec<u32>) -> std::result::Result<usize, RingLap> {
        self.read_published(out, false)
    }

    /// Wait, by `mechanism`, until the ring may have stamped past this
    /// reader's cursor — a bounded wait (`PollMechanism::wait_on`), so the
    /// caller re-reads and re-tests after it returns.
    pub fn wait_for_publish(&self, mechanism: PollMechanism) {
        // SAFETY: a mapped, aligned pointer into the step word `_step` keeps
        // alive for this reader's life.
        #[cfg(feature = "metal")]
        unsafe {
            mechanism.wait_on(self.step_ptr, self.cursor as u32);
        }
        #[cfg(not(feature = "metal"))]
        mechanism.wait();
    }

    /// Run `f` once the ring stamps `value` (on a dispatch-queue thread), on
    /// the publish event the run armed; `false` when there is none.
    pub fn notify_on_publish<F: Fn() + Send + Sync + 'static>(&self, value: u64, f: F) -> bool {
        #[cfg(feature = "metal")]
        if let Some(ev) = &self.publish {
            ev.notify_at(value, f);
            return true;
        }
        let _ = (value, f);
        false
    }

    /// Whether the ring has stamped past this reader's cursor.
    pub fn has_new(&self) -> bool {
        self.published().is_some_and(|p| p as usize > self.cursor)
    }

    /// Whether the run armed a publish event this reader can park on.
    pub fn parks(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            self.publish.is_some()
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    fn read_published(
        &mut self,
        out: &mut Vec<u32>,
        stop_at_eos: bool,
    ) -> std::result::Result<usize, RingLap> {
        if self.ended {
            return Ok(0);
        }
        let Some(published) = self.published() else {
            return Ok(0);
        };
        let published = published as usize;
        if published <= self.cursor {
            return Ok(0);
        }
        if published - self.cursor > self.total_tiles {
            return Err(RingLap {
                cursor: self.cursor,
                published,
                capacity: self.total_tiles,
            });
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (published, stop_at_eos, out);
            return Ok(0);
        }
        #[cfg(feature = "metal")]
        {
            let mut n = 0usize;
            while self.cursor < published {
                let t = (self.cursor + self.slot_shift) % self.total_tiles;
                let seg = t / self.tiles_per_seg;
                let off = t % self.tiles_per_seg;
                // SAFETY: a mapped pointer into a buffer `_segments` keeps alive,
                // `off < tiles_per_seg`; the slot was written before the stamp
                // that made `published` cover it.
                let id = unsafe { self.seg_bases[seg].add(off).read_volatile() };
                self.cursor += 1;
                if stop_at_eos && self.eos_ids.contains(&id) {
                    self.ended = true;
                    break;
                }
                out.push(id);
                n += 1;
            }
            Ok(n)
        }
    }
}

/// A poison sentinel written into a tile's unused (waste) tail so a read of it
/// — always a bug, since the drain only takes each tile's real prefix — is loud
/// rather than a plausible stale token. Only paid in debug/ASan builds and under
/// the `fuzzing` feature (the release write skips it; the geometry proves the
/// waste is never read, and the fuzz harnesses run release codegen and need the
/// sentinel to prove it). Chosen as an id no tokenizer emits and one that stands
/// out in a dump.
#[cfg(any(debug_assertions, feature = "fuzzing"))]
const TILE_POISON: u32 = 0xDEAD_B171; // "DEAD BITs"

/// Slots per segment ceiling: a `StorageModePrivate` buffer is CPU-mapped only
/// below 16384 B (§15.2 #12a), i.e. < 4096 `U32` slots. Each ring segment stays
/// under this so the strided drain reads it zero-copy — the property a single
/// big buffer loses past 4092. 4092 (16368 B) is just inside the line. The ring
/// grows by adding SEGMENTS, each of this size, rather than one buffer past it.
const SEG_MAX_SLOTS: usize = 4092;

/// The segmented-buffer mechanism, shared by the OUTPUT history ring and (later)
/// the INPUT staging ring — extracted so both use the SAME battle-tested zero-copy
/// path rather than reinventing it (readback.rs's own N_in-vs-N_out note).
///
/// It owns ONLY the buffers and their addressing: `segments` (each ≤ [`SEG_MAX_SLOTS`]
/// so it stays CPU-mapped under the 16384 B `StorageModePrivate` window §15.2 #12a),
/// the tile geometry (`quantum` slots per tile, `tiles_per_seg`, `total_tiles`), the
/// [`Self::tile_home`] arithmetic (a tile index → `(segment, slot-offset)`, tiles
/// flowing across segments as ONE logical ring), and the memoized mapped base
/// pointers ([`Self::seg_bases`], resolved once — segments are never reallocated).
/// It does NOT own any read/write CURSOR or count — those are the caller's (the
/// history keeps `fed`/`drained`/`tile_counts`; the input keeps its own). So this is
/// pure placement + mapping, no policy, and introduces no lock, no allocation on any
/// hot path, no blit, no volatile — exactly the existing mechanism, named.
pub(crate) struct SegmentedRing {
    /// N segment buffers, each `tiles_per_seg · quantum` `U32` slots, Private-mapped
    /// (small → under the window → zero-copy). N = 1 is the single-buffer case.
    segments: Vec<Tensor>,
    /// Slots per tile — the write/read stride.
    quantum: usize,
    /// Tiles per segment (each segment is `tiles_per_seg · quantum` slots).
    tiles_per_seg: usize,
    /// `segments.len() · tiles_per_seg` — the ring's total tile count (the wrap).
    total_tiles: usize,
    /// The segments' CPU-mapped base pointers, resolved ONCE and memoized — stable
    /// for the ring's life (segments never reallocated). `None` until first resolve;
    /// `Some` once every segment mapped; stays `None` if any is unmapped (the caller
    /// takes the copy fallback). `*const` is not `Send`/`Sync`, but a `Ring` is
    /// single-consumer and never shared across threads (§11.5g).
    #[cfg(feature = "metal")]
    seg_bases: Option<Vec<*const u32>>,
}

impl SegmentedRing {
    /// `cap` slots total, split into segments that each stay under the mapping
    /// window; `quantum` slots per tile. N segments emerge from how many
    /// whole-tile-multiples fit under [`SEG_MAX_SLOTS`]. `total_tiles` is rounded up
    /// so every segment is full (uniform geometry). At least 2 tiles so one is free
    /// for the writer while the other drains.
    pub(crate) fn new(cap: usize, quantum: usize, device: &Device) -> Result<Self> {
        // One autorelease pool for the construction: every `Tensor::zeros` below
        // is a blit fill in its own command buffer, and at the default slack
        // that is ~2 050 encoders + buffers autoreleased on a thread with no
        // pool (`AutoreleaseScope` docs). Drained when this returns.
        let _pool = lloom_tensor::AutoreleaseScope::new();
        let q = quantum.max(1);
        let total_tiles = (cap / q).max(2);
        let tiles_per_seg = (SEG_MAX_SLOTS / q).max(1);
        let n_segments = total_tiles.div_ceil(tiles_per_seg).max(1);
        let total_tiles = n_segments * tiles_per_seg;
        let segments = (0..n_segments)
            .map(|_| Tensor::zeros(tiles_per_seg * q, DType::U32, device))
            .collect::<Result<Vec<_>>>()?;
        // The zero fills are GPU work, queued; a host write into a segment
        // (`stage_at_tile`) before they execute is overwritten by them. One
        // wait at construction — off every hot path — makes every later host
        // write ordered after the fill.
        device.synchronize()?;
        Ok(Self {
            segments,
            quantum: q,
            tiles_per_seg,
            total_tiles,
            #[cfg(feature = "metal")]
            seg_bases: None,
        })
    }

    /// Physical home of a ring-tile index: which segment, and the slot offset of its
    /// first token within that segment's buffer. Pure arithmetic, no scan.
    fn tile_home(&self, tile: usize) -> (usize, usize) {
        let t = tile % self.total_tiles;
        let seg = t / self.tiles_per_seg;
        let offset = (t % self.tiles_per_seg) * self.quantum;
        (seg, offset)
    }

    /// Resolve (once) and return the memoized mapped base pointers, or `None` if any
    /// segment is unmapped (the caller then takes the copy fallback). All segments
    /// are the same size in the same pool, so they map together or not at all.
    #[cfg(feature = "metal")]
    fn resolve_bases(&mut self) -> Option<&[*const u32]> {
        if self.seg_bases.is_none() {
            let bases: Option<Vec<*const u32>> = self
                .segments
                .iter()
                .map(|s| unsafe { s.peek_scalar_ptr::<u32>(0) })
                .collect();
            self.seg_bases = bases;
        }
        self.seg_bases.as_deref()
    }

    /// The ring's tile count (the wrap) — for a caller sizing its cursor to the
    /// geometry.
    pub(crate) fn total_tiles(&self) -> usize {
        self.total_tiles
    }
    /// The first segment tensor — a stable handle for a caller that only needs the
    /// device or a representative buffer (not a specific tile).
    pub(crate) fn segment0(&self) -> &Tensor {
        &self.segments[0]
    }

    /// Write a single DEVICE token (the sampled argmax) into `tile`'s first slot —
    /// the decode-side input write. Device-side `slice_set`, so it is ordered on the
    /// GPU queue AFTER the prior forward's read (unlike the host `stage_at_tile`,
    /// which the caller must order via the pacer). `token` is a `[1]` U32 tensor.
    pub(crate) fn write_device_at_tile(&self, tile: usize, token: &Tensor) -> Result<()> {
        let (seg, offset) = self.tile_home(tile);
        // The segment is a flat rank-1 `[len]` U32 buffer; set the one slot at
        // `offset`. `token` is `[1]` (a single element), so this writes one slot.
        self.segments[seg].slice_set(token, 0, offset)?;
        Ok(())
    }

    /// Stage a host token slice into `tile`, front-packed at the tile's first slot
    /// — the INPUT-side write. Zero-copy through the segment's CPU-mapped pointer
    /// ([`Tensor::write_host_slice`]); an unmapped segment (discrete GPU / a tile
    /// wider than the window) falls back to one `slice_set` upload. `tokens.len()`
    /// must be `≤ quantum` (a tile holds one chunk of up to `quantum` tokens).
    ///
    /// SAFETY of the fast path: the caller guarantees no GPU work reads this tile
    /// while the stage runs — the input ring cycles `depth + 1` tiles and the pacer
    /// waited on a tile's forward before it laps back, exactly as the history's
    /// write-before-read discipline holds (§10.3e).
    pub(crate) fn stage_at_tile(&self, tile: usize, tokens: &[u32]) -> Result<()> {
        debug_assert!(tokens.len() <= self.quantum, "chunk wider than a tile");
        let (seg, offset) = self.tile_home(tile);
        let buf = &self.segments[seg];
        // Mapped write at the tile's slot offset — no intermediate tensor, no blit.
        // SAFETY: `offset + tokens.len() ≤ (seg+1)·quantum ≤ segment len` (a tile
        // fits its segment); the tile is not being read (caller's pacer ordering).
        #[cfg(feature = "metal")]
        {
            if unsafe { buf.write_host_slice::<u32>(offset, tokens) } {
                return Ok(());
            }
        }
        // Unmapped fallback: upload + slice_set into the tile's slots. One alloc,
        // taken only where the mapped write is unavailable. The segment is rank-1
        // `[len]`, so write the `[n]` run at slot `offset`.
        let src = Tensor::from_vec(tokens.to_vec(), tokens.len(), buf.device())?;
        buf.slice_set(&src, 0, offset)?;
        Ok(())
    }

    /// A `[1, width]` view of `tile`'s first `width` slots — the INPUT the forward
    /// reads. A contiguous prefix of the tile's run within one mapped segment, so no
    /// copy. `width ≤ quantum`.
    pub(crate) fn tile_view(&self, tile: usize, width: usize) -> Result<Tensor> {
        debug_assert!(width <= self.quantum, "view wider than a tile");
        let (seg, offset) = self.tile_home(tile);
        // The segment is a flat `[tiles_per_seg·quantum]` U32 buffer; the tile's
        // window is `[offset, offset+width)`. Reshape to `[1, width]` for the model.
        self.segments[seg]
            .narrow(0, offset, width)?
            .reshape((1, width))
    }

    /// Read `tile`'s first `width` slots back through the segment's CPU-MAPPED
    /// pointer — the SAME zero-copy access the GPU forward makes of this input
    /// (`embed_tokens` reads the device buffer, which for a Private-mapped segment
    /// is the memory this pointer sees). For fuzzing the input ring on the real
    /// path rather than through a `to_cpu` copy. `None` if the segment is unmapped
    /// (the caller then has no zero-copy read to check). `width ≤ quantum`.
    pub(crate) fn peek_tile(&mut self, tile: usize, width: usize) -> Option<Vec<u32>> {
        debug_assert!(width <= self.quantum);
        let (seg, offset) = self.tile_home(tile);
        #[cfg(feature = "metal")]
        {
            let base = *self.resolve_bases()?.get(seg)?;
            // SAFETY: `base` is the mapped base of a never-reallocated segment;
            // `[offset, offset+width) ≤ segment len` (a tile fits its segment). A plain
            // read of finished host-visible memory (no GPU writing it in the fuzz).
            let src = unsafe { std::slice::from_raw_parts(base.add(offset), width) };
            Some(src.to_vec())
        }
        // Without Metal there is no mapped pointer and no other read path: the
        // segment IS host memory, so the copy is the same read the forward makes.
        #[cfg(not(feature = "metal"))]
        {
            self.segments
                .get(seg)?
                .narrow(0, offset, width)
                .ok()?
                .to_vec1::<u32>()
                .ok()
        }
    }

    /// Poison EVERY slot of every segment with a sentinel — for a fuzz that then
    /// asserts a read of a NEVER-STAGED slot (a wrong tile, or past a chunk's
    /// width) returns the sentinel rather than a plausible-stale token. Debug/test
    /// only. Uses the same mapped write the staging does.
    pub(crate) fn poison_all(&self, sentinel: u32) -> Result<()> {
        for seg in &self.segments {
            let len = seg.elem_count();
            let poison = vec![sentinel; len];
            // SAFETY: writing the whole (mapped) segment; nothing reads it here.
            #[cfg(feature = "metal")]
            if unsafe { seg.write_host_slice::<u32>(0, &poison) } {
                continue;
            }
            // Unmapped, or off Metal: a filled tensor.
            let t = Tensor::from_vec(poison, len, seg.device())?;
            seg.slice_set(&t, 0, 0)?;
        }
        Ok(())
    }
}

/// The device-side id history: a **ring of segments** ("lord of the rings").
/// Owns the tile placement and the lap invariant; the [`Ring`] keeps the publish
/// protocol (slot-before-stamp ordering, the published clamp).
///
/// # Tiles, and why the read needs no scan
///
/// Each step writes into a `quantum`-slot **tile**, its real tokens front-packed,
/// the tail wasted (and poisoned in debug). `quantum = batch × (max_draft+1)` is
/// the most a step can produce, so a tile always fits and never straddles a
/// segment boundary (segments are whole-tile multiples). The per-tile real count
/// is recorded when the tile closes (`tile_counts`), so the drain is a **strided
/// read with NO scan**: walk live tiles, take each tile's recorded real prefix
/// through its segment's mapped pointer. At `quantum = 1` (B=1 greedy) every tile
/// is one slot and this is a dense contiguous read; at `quantum > 1` (batched /
/// speculative-in-one-forward) it is tile-strided. One path.
///
/// # Ring of segments, and why
///
/// `segments` is N separate `U32` buffers, each `tiles_per_seg · quantum` slots.
/// The write walks tiles across the whole ring (`total_tiles = N ·
/// tiles_per_seg`) and hops to the next segment at a segment boundary — a buffer
/// switch, not index-wrap arithmetic. **Each segment is small enough to stay
/// CPU-mapped under the 16384 B `StorageModePrivate` window (§15.2 #12a), so the
/// read is zero-copy even when the whole ring exceeds that window** — the single
/// buffer could not (a big-B window > 4092 slots falls off the map). N = 1 is the
/// degenerate single-buffer case (the B=1 chat path); N > 1 is what lets big-B /
/// long windows stay mapped. Growth is "add a segment", never a realloc-copy.
///
/// # The lock-free invariant
///
/// `tiles − drained_tiles ≤ total_tiles − 1`, held by construction: the commit
/// budget drains every `period` steps and each step is one tile, so the writer
/// never laps the un-drained reader (`Geometry::cap`). A violation (a future
/// model whose per-step production the geometry does not bound) is a hard bail,
/// never a silent overwrite (§3.5).
struct History {
    /// The segmented-buffer mechanism (segments, tile geometry, `tile_home`, mapped
    /// pointers). The history layers its OUTPUT state — the cursors and counts
    /// below — over this shared placement primitive.
    ring: SegmentedRing,
    /// Total real tokens ever written (monotonic, EOS-retracted).
    fed: usize,
    /// Total real tokens ever drained (monotonic). Live window = `[drained, fed)`.
    drained: usize,
    /// Total tiles ever CLOSED (monotonic). The open tile (if any) is `tiles %
    /// total_tiles`; `fed`/`drained` count tokens, `tiles` counts closed tiles.
    tiles: usize,
    /// Tiles fully consumed by the drain (monotonic); the drain advances whole
    /// tiles at a time.
    drained_tiles: usize,
    /// Real tokens in each tile (the OUT-side count, ≤ `quantum`), indexed by
    /// `tile % total_tiles`, recorded at close. What lets the strided read skip
    /// each tile's waste with no scan. `u32` not `u16`: `quantum = batch ×
    /// (max_draft+1)` can exceed 65535 for a large batch, and a `u16` would
    /// silently truncate the count — a wrong-output cap, not a loud one. This is
    /// the N_out cap; N_in (a burst fed into the input slot) is a SEPARATE
    /// structure and never sizes this ring.
    tile_counts: Vec<u32>,
    /// Real tokens written into the currently open tile so far, `< quantum`; 0
    /// when no tile is open (between close and the next `begin_step`).
    in_tile: usize,
    /// Whether a tile is currently open (written-to but not yet closed).
    tile_open: bool,
}

impl History {
    /// `cap` slots total across the ring, split into segments that each stay
    /// under the mapping window. `quantum` slots per tile. The buffer geometry is
    /// [`SegmentedRing`]; the history adds the output cursors + `tile_counts`.
    fn new(cap: usize, quantum: usize, device: &Device) -> Result<Self> {
        let ring = SegmentedRing::new(cap, quantum, device)?;
        let total_tiles = ring.total_tiles;
        Ok(Self {
            ring,
            fed: 0,
            drained: 0,
            tiles: 0,
            drained_tiles: 0,
            tile_counts: vec![0u32; total_tiles],
            in_tile: 0,
            tile_open: false,
        })
    }

    /// The ring's total tile count — the wrap. Delegated to the buffer geometry.
    #[inline]
    fn total_tiles(&self) -> usize {
        self.ring.total_tiles
    }

    /// Physical home of a ring-tile index: `(segment, slot-offset)`. Delegated.
    #[inline]
    fn tile_home(&self, tile: usize) -> (usize, usize) {
        self.ring.tile_home(tile)
    }

    /// Begin this step's tile and return `(segment, offset)` for its FIRST token.
    /// Lazily closes the previous step's tile (so `eos_of_last`'s retract could
    /// still reach it), then checks the lap: the tile about to be reused must be
    /// drained. Bails rather than overwriting — provably unreachable under the
    /// known geometry, loud if a future per-step production exceeds `quantum`.
    fn begin_step(&mut self) -> Result<(usize, usize)> {
        self.close_open_tile()?;
        // At most `total_tiles - 1` tiles may be undrained, leaving one to open.
        let live_tiles = self.tiles - self.drained_tiles;
        if live_tiles >= self.total_tiles() {
            lloom_tensor::bail!(
                "Ring history lap: {live_tiles} live tiles ≥ {} — the drain fell behind the \
                 geometry bound (period). This cannot happen under the known per-step geometry \
                 (quantum {}); a per-step production larger than one tile needs the \
                 unbounded-geometry fallback, not this ring.",
                self.total_tiles(),
                self.ring.quantum
            );
        }
        self.in_tile = 0;
        self.tile_open = true;
        Ok(self.tile_home(self.tiles))
    }

    /// Record one real token just written into the open tile: advance `fed` and
    /// the in-tile count.
    fn wrote_one(&mut self) {
        debug_assert!(self.tile_open && self.in_tile < self.ring.quantum);
        self.fed += 1;
        self.in_tile += 1;
    }

    /// The `(segment, offset)` for the NEXT token within the open tile (a
    /// multi-token, quantum>1 step: call `wrote_one` after each write). Used by
    /// the quantum>1 fuzz test today; the batched (#694) writer's seam in prod.
    #[cfg_attr(not(test), allow(dead_code))]
    fn next_slot(&self) -> (usize, usize) {
        let (seg, base) = self.tile_home(self.tiles);
        (seg, base + self.in_tile)
    }

    /// Retract the last real token from the open tile (an EOS the loop excludes).
    /// Valid between `wrote_one` and the next `begin_step`, where the retracting
    /// arms call it (`eos_of_last`, right after the write).
    fn retract_one(&mut self) {
        if self.tile_open && self.in_tile > 0 {
            self.in_tile -= 1;
            self.fed -= 1;
        }
    }

    /// Close the open tile if any: record its real count (`tile_counts`), poison
    /// the waste tail in debug so a stray read is loud, and advance `tiles`.
    /// Called lazily by the next `begin_step` and eagerly by the final drain.
    fn close_open_tile(&mut self) -> Result<()> {
        if !self.tile_open {
            return Ok(());
        }
        self.tile_open = false;
        if self.in_tile == 0 {
            // The step's only token was an EOS, retracted — nothing to count, the
            // tile stays available for reuse.
            return Ok(());
        }
        let slot = self.tiles % self.total_tiles();
        self.tile_counts[slot] = self.in_tile as u32;
        #[cfg(any(debug_assertions, feature = "fuzzing"))]
        if self.in_tile < self.ring.quantum {
            let (seg, base) = self.tile_home(self.tiles);
            let waste = self.ring.quantum - self.in_tile;
            let poison = Tensor::full(TILE_POISON, waste, self.ring.segments[seg].device())?;
            self.ring.segments[seg].slice_set(&poison, 0, base + self.in_tile)?;
        }
        self.tiles += 1;
        Ok(())
    }

    /// The real tokens written, EOS-retracted: `fed`. The device stamp counts
    /// the retracted EOS too, and runs ahead of this by [`Ring::stamps_ahead`].
    fn generation(&self) -> usize {
        self.fed
    }

    /// Undrained real tokens: the live window `[drained, fed)`.
    fn undrained(&self) -> usize {
        self.fed - self.drained
    }

    /// A `[n]` `U32` view of the undrained window `[drained, fed)` for the batched
    /// EOS check — the tokens the GpuBatched arm scans in one dispatch. Returns the
    /// view AND its start offset within the window (always 0 here — the window's
    /// first token is index 0 of the returned view, so a found EOS index maps
    /// directly to a window position). `None` (caller falls back to the host scan)
    /// when the window is empty, the quantum is not 1 (waste-gap layout — the dense
    /// window view is exact only at quantum 1, as `drain_mapped`), or the window
    /// would straddle a segment boundary (a rare split the single-view kernel does
    /// not cover; the host-scan fallback is correct there).
    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    fn window_view(&self) -> Option<Tensor> {
        let n = self.fed - self.drained;
        if n == 0 || self.ring.quantum != 1 {
            return None;
        }
        // quantum == 1: token index == slot index == tile index within a segment.
        let (seg, off) = self.ring.tile_home(self.drained);
        let slots_per_seg = self.ring.tiles_per_seg; // × quantum, quantum == 1
        if off + n > slots_per_seg {
            return None; // spans a segment boundary — fall back to the host scan.
        }
        self.ring.segments[seg].narrow(0, off, n).ok()
    }

    /// Forget the undrained window without reading it (the HostOnDrain arm's
    /// overrun past an EOS): advance the drain cursors to `fed`, closing the open
    /// tile, so a later run's first drain starts fresh. Returns the count
    /// forgotten. No device read — the tokens are simply abandoned.
    fn forget_undrained(&mut self) -> usize {
        let n = self.fed - self.drained;
        if self.tile_open {
            if self.in_tile > 0 {
                let slot = self.tiles % self.total_tiles();
                self.tile_counts[slot] = self.in_tile as u32;
                self.tiles += 1;
            }
            self.tile_open = false;
            self.in_tile = 0;
        }
        self.drained = self.fed;
        self.drained_tiles = self.tiles;
        n
    }

    /// Drain real tokens `[drained, end)` (end already clamped to the published
    /// generation and to `fed`), returning them in order and advancing the drain
    /// cursors. `read(seg, slot) -> u32` reads one physical slot of one segment
    /// (the caller supplies the mapped-pointer reader or the copy fallback).
    ///
    /// **The strided read, no scan**: walk whole tiles from `drained_tiles`,
    /// reading each tile's recorded real count (`tile_counts`) from its segment,
    /// skipping the waste tail. The layout is computed (`tile_home`), never
    /// searched. At `quantum == 1` each tile is one slot and this is a dense
    /// contiguous read; at `quantum > 1` it is the tile-strided read across
    /// segments. One path — the quantum unifies them.
    ///
    /// `end` is a real-token index (typically `fed` clamped to the published
    /// generation). The drain reads whole CLOSED tiles only — the open tile's
    /// tokens are in `fed` but not yet in `tile_counts`, so the read stops at the
    /// last closed tile and never past it. It also stops as soon as the next
    /// whole tile would exceed `end` (the published clamp lands on a tile
    /// boundary here because a tile == a step and the stamp is per step), so a
    /// partially-visible tile is left for the next drain.
    /// Fills `out` (a reused scratch buffer, cleared first — no per-call alloc)
    /// with the drained tokens. `read(seg, slot)` resolves ONE physical slot; the
    /// base is resolved once per tile (outside the inner per-token loop), so the
    /// per-token cost is a raw read, no bounds-checked index. Lock-free: the
    /// mapped read path (`peek_scalar_ptr`) does not lock; this adds no lock and
    /// no allocation.
    fn drain_into<F: FnMut(usize, usize) -> u32>(
        &mut self,
        end: usize,
        out: &mut Vec<u32>,
        mut read: F,
    ) {
        out.clear();
        if end <= self.drained {
            return;
        }
        while self.drained_tiles < self.tiles {
            let slot = self.drained_tiles % self.total_tiles();
            let count = self.tile_counts[slot] as usize;
            if self.drained + count > end {
                break; // this tile is not fully visible yet; leave it.
            }
            let (seg, base) = self.tile_home(self.drained_tiles);
            for j in 0..count {
                out.push(read(seg, base + j));
            }
            self.drained += count;
            self.drained_tiles += 1;
        }
    }

    /// The mapped drain — the hot path, **lock-free, alloc-free, and scan-free**.
    ///
    /// **The window drain, recovered.** The live window `[drained, end)` is read as
    /// a small number of CONTIGUOUS runs — one per segment the window spans —
    /// whose bounds are COMPUTED by arithmetic (`tile_home` + `slots_per_seg`),
    /// never by walking tiles. This is the single-ring window read (`peek` a base,
    /// `copy_from_slice` a run) generalised to a ring of segments: the segment
    /// boundary is the ONLY place a new base is taken. At `quantum == 1` (B=1) the
    /// whole window inside one segment is ONE run — the dense page the consumer
    /// wanted. The segment-hopping per-tile walk this replaced fragmented that into
    /// a patch per tile; the window needs no such scan.
    ///
    /// Contiguity: a segment lays its tiles out consecutively
    /// (`offset = (tile % tiles_per_seg) × quantum`), and at `quantum == 1` every
    /// token is one full tile, so slots are dense and `token == slot` within a
    /// segment. The run is `[home(drained) .. min(end, next-segment-boundary))`,
    /// read in one shot. `end` is the published clamp, so every slot read is
    /// finished, stable memory (§9.3 acquire: once `published ≥ N`, `[0..N)` are
    /// written and never rewritten) — a plain `copy_from_slice`, no per-element
    /// volatile (that cost ~1 tok/s at the ceiling).
    ///
    /// Requires `quantum == 1` (the greedy/crown path — dense, no waste tails). A
    /// `quantum > 1` writer with partial tiles (DSpark/batched with variable
    /// accepts) has waste gaps between tiles and takes `drain_copy`'s tile-aware
    /// path; this hot window read is the dense case. Returns `false` if any segment
    /// is not CPU-mapped (caller takes the copy fallback).
    #[cfg(feature = "metal")]
    fn drain_mapped(&mut self, end: usize, out: &mut Vec<u32>) -> bool {
        // The window read is exact only for the dense (`quantum == 1`) layout,
        // where `token == slot` within a segment. Defer the waste-tail case to the
        // tile-aware copy path rather than scan tiles on the hot path.
        if self.ring.quantum != 1 {
            return false;
        }
        // Resolve (once, memoized) the segments' mapped base pointers; unmapped →
        // the copy fallback. The `&mut` borrow ends before the read loop below.
        if self.ring.resolve_bases().is_none() {
            return false;
        }
        out.clear();
        let end = end.min(self.fed);
        if end <= self.drained {
            return true;
        }
        // `quantum == 1`: slots == tokens, `slots_per_seg == tiles_per_seg`. Walk
        // the window one segment-run at a time, bounds COMPUTED, no tile scan.
        // `× quantum`, and `quantum == 1` here.
        let slots_per_seg = self.ring.tiles_per_seg;
        // Immutable borrow of the resolved bases + geometry for the loop (both from
        // `self.ring`), so `tile_home` (also immutable) composes without conflict.
        let bases = self
            .ring
            .seg_bases
            .as_ref()
            .expect("resolve_bases returned Some just above");
        while self.drained < end {
            // `home` of the next token: which segment, which offset within it.
            // `drained` is a token index == a tile index at quantum 1.
            let (seg, off) = self.ring.tile_home(self.drained);
            // This run ends at the earlier of the window end and this segment's
            // last slot — one contiguous span within a single mapped buffer.
            let seg_remaining = slots_per_seg - off;
            let run = (end - self.drained).min(seg_remaining);
            // SAFETY: `bases[seg]` is the mapped base of a buffer never
            // reallocated for this ring's life; `[off, off+run) ≤ slots_per_seg`
            // (the run is clamped to the segment); `drained + run ≤ end ≤
            // published`, so the GPU is not writing it — no aliasing.
            let ptr = bases[seg];
            let src = unsafe { std::slice::from_raw_parts(ptr.add(off), run) };
            out.extend_from_slice(src);
            self.drained += run;
        }
        // Advance the tile cursor in lockstep (one tile per token at quantum 1) so
        // the ring's live/lap accounting (`begin_step`) stays consistent.
        self.drained_tiles = self.drained;
        true
    }

    /// The copy fallback (unmapped segments, or the `quantum > 1` waste-tail
    /// layout): a `to_vec1` per segment synchronises and is correct if a copy —
    /// the honest degradation, off the hot path. This path IS tile-aware
    /// (`drain_into` walks tiles to skip waste), which the dense window read does
    /// not need. Fills `out` (caller-owned, pre-sized).
    fn drain_copy(&mut self, end: usize, out: &mut Vec<u32>) -> Result<()> {
        out.clear();
        let end = end.min(self.fed);
        if end <= self.drained {
            return Ok(());
        }
        // Only the segments the undrained window spans are read back: each
        // `to_vec1` is a blocking readback, and the ring holds many more
        // segments (the consumer slack) than a window ever touches. At
        // `quantum == 1` the window `[drained, end)` is dense (token == slot)
        // and is read whole, the open tile's token included, exactly as the
        // mapped read does; the tile walk is for the waste-tail layouts.
        let mut snapshots: Vec<Option<Vec<u32>>> = vec![None; self.ring.segments.len()];
        if self.ring.quantum == 1 {
            let slots_per_seg = self.ring.tiles_per_seg;
            while self.drained < end {
                let (seg, off) = self.ring.tile_home(self.drained);
                if snapshots[seg].is_none() {
                    snapshots[seg] = Some(self.ring.segments[seg].to_vec1::<u32>()?);
                }
                let run = (end - self.drained).min(slots_per_seg - off);
                if let Some(s) = snapshots[seg].as_ref() {
                    out.extend_from_slice(&s[off..off + run]);
                }
                self.drained += run;
            }
            self.drained_tiles = self.drained;
            return Ok(());
        }
        for tile in self.drained_tiles..self.tiles {
            let (seg, _) = self.tile_home(tile);
            if snapshots[seg].is_none() {
                snapshots[seg] = Some(self.ring.segments[seg].to_vec1::<u32>()?);
            }
        }
        self.drain_into(end, out, |seg, slot| {
            snapshots[seg].as_ref().map_or(0, |s| s[slot])
        });
        Ok(())
    }

    /// The final drain: everything to `fed`, after one wait for the GPU.
    ///
    /// One `synchronize` retires every in-flight write, after which the mapped
    /// window read is exact; the copy path is taken only where a segment is
    /// not mapped or the layout has waste tails (`quantum > 1`).
    fn drain_all_into(&mut self, device: &Device, out: &mut Vec<u32>) -> Result<()> {
        self.close_open_tile()?;
        let fed = self.generation();
        device.synchronize()?;
        #[cfg(feature = "metal")]
        if self.drain_mapped(fed, out) {
            return Ok(());
        }
        self.drain_copy(fed, out)
    }
}

/// The device-resident arms: the token never comes to the host, the next
/// forward reads the ring's own slot, and the ids accumulate device-side.
pub struct Ring {
    ring: TokenRing,
    eos: EosRead,
    /// Poll iterations before the non-blocking read gives up on a step.
    poll_budget: u64,
    /// Commits between LIVE marker writes — the wake-cadence knob. `1` writes the
    /// device marker every commit (the incumbent per-commit stamp); `N` writes it
    /// every Nth commit so the host WFE-waiter wakes N-commits' worth of tokens at
    /// a time and the core has a longer quiet window to go dormant between wakes
    /// (`DESIGN.md` poll.rs: WFE is a bounded spin, not a cheap park — a longer
    /// wait is what lets the core actually power down). The TTFT ramp overrides it
    /// (`force_live`), so token 0 never waits N commits. Skipping a marker write
    /// only makes the drain's clamp more conservative — it can never read an
    /// unwritten slot (§3.5). Default [`WAKE_LIVE_EVERY_DEFAULT`]; single-thread
    /// (the decode loop owns it), so a plain counter, not an atomic.
    live_every: usize,
    /// Commits since the last LIVE marker write; when it reaches `live_every` the
    /// next commit is live and this resets.
    commits_since_live: std::cell::Cell<usize>,
    /// The device-side id history (the wrapping ring behind the swap seam).
    history: History,
    /// The ordinal stamped on the last publish; 0 before any.
    generation: u32,
    /// The history segments written since the deferred arms' last stamp, each
    /// with the generation its last slot carries, oldest first. The window's
    /// stamp certifies each in turn ([`TokenRing::publish_marker`]), so a
    /// window that crosses a segment boundary is stamped once per segment,
    /// every stamp ordered after its own segment's writes.
    unstamped: Vec<(usize, u32)>,
    /// `set_fused_publish`: the loop asked for the fused publish; taken on a
    /// mapped ring when the sampler is argmax (`sample_and_feed`).
    fused_publish: bool,
    /// Whether the last `sample_and_feed` published through the fused kernel,
    /// which stamped (and gated) that step. The run's `fused_publish` is a
    /// request; a sampler other than argmax takes the unfused chain, whose
    /// stamp is the loop's to ask for (`publish_marker`).
    fused_step: bool,
    /// The EOS id set, host-side, for the [`EosRead::HostOnDrain`] (and the
    /// `GpuBatched` fallback) compare. Read back from the device ring once at
    /// construction (a loop invariant), so the per-token path never touches it.
    /// Empty on the pure-device arms.
    eos_ids_host: Vec<u32>,
    /// [`EosRead::GpuBatched`]: the `[1]` `U32` result of the window's `eos_index`
    /// dispatch, enqueued at `publish_marker` and read back once at the drain
    /// (`eos_in_drained`) — a per-commit sync, off the per-token path. Holds the
    /// window length the position is relative to, so a host that gets index `< len`
    /// knows an EOS fell at that window offset. `None` between commits and on every
    /// other arm.
    pending_eos_index: std::cell::RefCell<Option<(Tensor, usize)>>,
    /// [`EosRead::GpuBatchedNonblocking`]: the staged indices of the last two
    /// closed windows, newest last.
    staged_eos: std::cell::RefCell<Vec<StagedWindow>>,
    /// The history's drained count before the last `take_ids`: the absolute
    /// start of the batch it returned.
    drain_start: usize,
    /// The length of that batch, the one frame a device EOS index answers
    /// for; `None` after `drain_all`, whose tail and a reader-owned run's
    /// whole ids are scanned on the host.
    taken: Option<usize>,
    eos_index_hits: std::cell::Cell<u64>,
    eos_index_late: std::cell::Cell<u64>,
    counters: ReadbackCounters,
    /// [`ReadbackStrategy::take_ids_into`]'s window, reused across calls.
    scratch: Vec<u32>,
    /// The armed thinking budget's device state; while it can still change
    /// a fed token, steps take the unfused chain with its select.
    forcing: Option<DeviceForcing>,
}

/// Iterations before the non-blocking poll takes the blocking read instead.
///
/// `poll_eos` returns `None` both for *not published yet* and for *this
/// buffer is not CPU-mapped*, and the second is an allocator property (§15.2
/// #12a) that #454 measures as unstable run to run — so an unbounded loop
/// would hang silently. Generous against a ~19 ms step: it converts a hang
/// into a counted fallback and times nothing.
pub const POLL_BUDGET: u64 = 50_000_000;

/// Commits between LIVE marker writes (the wake-cadence default, [`Ring::live_every`]).
///
/// `4` is a starting point, not a tuned optimum: it wakes the host WFE-waiter
/// every ~4 commits' worth of tokens instead of every commit, trading a bounded
/// overrun-in-visibility (the drain lags ≤ the un-written windows, caught up on
/// the next live commit or the final `drain_all`) for a longer quiet window in
/// which the core can go dormant (poll.rs: WFE is a bounded spin, so a longer
/// wait is the only thing that lets it power down). `1` restores the per-commit
/// stamp. Adaptive tuning later, as with the pacer's warmup-derived depth.
pub const WAKE_LIVE_EVERY_DEFAULT: usize = 4;

/// The id history slot count, chosen to stay CPU-mapped under **Private**
/// storage so the streaming drain reads it through a mapped pointer with no copy
/// and no blit (§15.2 #12: no blits on the decode path). Metal maps a
/// `StorageModePrivate` buffer iff it is **< 16384 bytes** (§15.2 #12a); at 4
/// bytes per `U32` that is **< 4096 slots**, so 4092 slots (16368 B) stays just
/// under the line while giving the host a wide window — it drains every commit,
/// far fewer than this behind. This is deliberately NOT Shared: Shared maps at
/// any size but its GPU-write coherency cost on the per-token `slice_set` is
/// unmeasured, and staying inside the Private mapping window keeps the write on
/// the same footing as every other decode buffer.
///
/// The raw ctor floors a caller's requested slot count here so a small explicit
/// value still gets a generous window. The history is a **ring of segments** now
/// (each ≤ [`SEG_MAX_SLOTS`]); a run longer than the ring reuses it (the drain
/// recycles whole tiles), and a big window that exceeds one segment grows by
/// adding segments — so this is a floor on the window, not a ceiling on the
/// generation. [`Self::with_geometry`] sizes it exactly instead.
pub const RING_HISTORY_SLOTS: usize = 4092;

/// Default consumer-slack budget for the token-ring history: 4 MB of preallocated
/// `U32` slots (≈ 1M tokens at quantum 1, ≈ 256 Private segments). Sized so a
/// blocked consumer (a `print!` to a slow terminal, a stalled SSE flush) can lag
/// by seconds of generation before the lap guard fires — the wall between the hot
/// producer and a jittery consumer. Preallocated: costs memory, not per-step work.
/// Overridable via `LLOOM_RING_SLACK_BYTES`.
pub const RING_SLACK_BYTES_DEFAULT: usize = 32 * 1024 * 1024;

/// The slot count for the consumer-slack window: `LLOOM_RING_SLACK_BYTES`
/// (default [`RING_SLACK_BYTES_DEFAULT`]) worth of `U32` slots, rounded to a whole
/// number of `quantum`-slot tiles so the geometry stays uniform. `0` disables the
/// slack (fall back to the tight `cap()`), for the A/B baseline.
fn history_slack_slots(quantum: usize) -> usize {
    let bytes = crate::tuning::tuning().ring.slack_bytes;
    let slots = bytes / std::mem::size_of::<u32>();
    // Round down to a whole number of tiles (uniform geometry); 0 => no slack.
    let q = quantum.max(1);
    (slots / q) * q
}

impl Ring {
    /// `capacity` bounds the ids one ring will accumulate across every decode
    /// call it serves; exceeding it is an error rather than a silent drop. The
    /// allocation is floored at [`RING_HISTORY_FLOOR`], so a caller that asks for
    /// less still gets a generous buffer.
    /// Construct from an explicit slot count (the raw primitive). The history is
    /// a wrapping ring of `capacity` slots; the write-reset keeps each step
    /// contiguous and the drain recycles slots, so `capacity` need only cover the
    /// live window (`period × quantum`), NOT the whole generation — an unbounded
    /// run reuses the ring. Floored at [`RING_HISTORY_SLOTS`] so a small explicit
    /// value still gives a wide mapped window; `quantum` defaults to 1 (the
    /// greedy step), overridden by [`Self::with_geometry`]. Prefer
    /// [`Self::with_geometry`], which sizes the ring exactly from the run's own
    /// per-step geometry.
    pub fn new(eos_ids: &[u32], device: &Device, capacity: usize, eos: EosRead) -> Result<Self> {
        Self::with_cap_and_quantum(
            eos_ids,
            device,
            capacity.max(RING_HISTORY_SLOTS),
            1,
            1,
            1,
            eos,
        )
    }

    /// Construct sized from the per-step geometry PLUS a consumer-slack window.
    ///
    /// `Geometry::cap()` is the tight correctness bound — the minimum that cannot
    /// lap while the drain keeps its commit cadence. But the drain runs the
    /// consumer INLINE (`on_tokens` at each commit), so a consumer that blocks (a
    /// `print!` to a slow terminal, an SSE flush) stalls the producer once the ring
    /// fills. The cure is slack: allocate MANY more segments than `cap()` needs, so
    /// the writer can run far ahead of a lagging consumer before the lap guard
    /// fires. It is pure headroom — preallocated, so it costs memory, never
    /// per-step work (the drain reads only the live window; empty segments are
    /// never touched). And it is orthogonal to TTFT: the commit cadence (`period`)
    /// is unchanged, so the first token still streams when it always did.
    ///
    /// The slack is a BYTE budget (`LLOOM_RING_SLACK_BYTES`, default
    /// [`RING_SLACK_BYTES_DEFAULT`]); the ring is sized to the larger of the tight
    /// `cap()` and that budget's slot count. The lap guard then tolerates
    /// `total_tiles − 1` undrained tiles — the slack, in tiles — bailing loudly
    /// only if the consumer falls that far behind (a real stall to surface, not
    /// silent corruption). `cap()` itself is untouched, so the fuzz's tight-bound
    /// checks and the geometry's own proof still hold.
    pub fn with_geometry(
        eos_ids: &[u32],
        device: &Device,
        geometry: Geometry,
        eos: EosRead,
    ) -> Result<Self> {
        let cap = geometry.cap().max(history_slack_slots(geometry.quantum()));
        // The input buffer holds at least one step's `quantum` tokens; streamed
        // prefill widens it to `prefill_ubatch` when that is larger, so a chunk
        // stages into one reused buffer. `quantum` covers decode (1) and a drafted
        // block (`K+1`); `max_input_width` on the geometry is the streamed-prefill
        // chunk width, 0 when the caller does not stream (then it is just
        // `quantum`).
        let max_input_width = geometry.max_input_width.max(geometry.quantum());
        // The input ring needs `depth + 1` buffers ONLY when a host source stages
        // concurrently (streamed prefill, `max_input_width > quantum`); the decode
        // path's device writes are queue-ordered and need one. `depth + 1` covers
        // the pacer's in-flight window; the buffers are tiny (`[1, ubatch]`), so
        // this is negligible memory.
        let input_slots = if max_input_width > geometry.quantum() {
            geometry.in_flight_depth + 1
        } else {
            1
        };
        Self::with_cap_and_quantum(
            eos_ids,
            device,
            cap,
            geometry.quantum(),
            max_input_width,
            input_slots,
            eos,
        )
    }

    fn with_cap_and_quantum(
        eos_ids: &[u32],
        device: &Device,
        cap: usize,
        quantum: usize,
        max_input_width: usize,
        input_slots: usize,
        eos: EosRead,
    ) -> Result<Self> {
        let ring = TokenRing::with_input_ring(0, eos_ids, device, max_input_width, input_slots)?;
        let history = History::new(cap, quantum, device)?;
        // Cache the EOS ids host-side for the arms that need them there: the pure
        // host compare (`HostOnDrain`) and the `GpuBatched` fallback + its `n_eos`,
        // and the poll arm — which learns EOS per step and never reaches a batch
        // with one in it on its own loop, but under the adaptive commit (lloom
        // #735) the loop skips its per-step wait and finds the EOS in the drained
        // ids at the end, through `eos_in_drained`'s host scan. The blocking arm
        // never touches this.
        let eos_ids_host = if matches!(
            eos,
            EosRead::Poll(_)
                | EosRead::HostOnDrain
                | EosRead::GpuBatched
                | EosRead::GpuBatchedNonblocking
        ) {
            ring.eos_ids()?
        } else {
            Vec::new()
        };
        Ok(Self {
            ring,
            eos,
            history,
            generation: 0,
            unstamped: Vec::with_capacity(2),
            fused_publish: false,
            fused_step: false,
            eos_ids_host,
            pending_eos_index: std::cell::RefCell::new(None),
            staged_eos: std::cell::RefCell::new(Vec::with_capacity(2)),
            drain_start: 0,
            taken: None,
            eos_index_hits: std::cell::Cell::new(0),
            eos_index_late: std::cell::Cell::new(0),
            poll_budget: POLL_BUDGET,
            live_every: WAKE_LIVE_EVERY_DEFAULT,
            commits_since_live: std::cell::Cell::new(0),
            counters: ReadbackCounters::default(),
            scratch: Vec::new(),
            forcing: None,
        })
    }

    /// Bound the non-blocking poll at `iterations` before it falls back to the
    /// blocking read. The default is [`POLL_BUDGET`]; a test with a model too
    /// small to reach the command-buffer cut wants a short one.
    pub fn with_poll_budget(mut self, iterations: u64) -> Self {
        self.poll_budget = iterations.max(1);
        self
    }

    /// Set the wake cadence: write the LIVE device marker every `commits` commits
    /// ([`Ring::live_every`]). `1` is the per-commit stamp; `N` wakes the host
    /// every N-commits' worth of tokens so the core can go dormant between waits.
    /// Floored at 1. The TTFT ramp still forces the early commits live.
    pub fn with_wake_cadence(mut self, commits: usize) -> Self {
        self.set_wake_cadence(commits);
        self
    }

    /// [`Self::with_wake_cadence`] on a ring a pool already holds.
    pub fn set_wake_cadence(&mut self, commits: usize) {
        self.live_every = commits.max(1);
    }

    /// The ring itself, for a caller that wants to observe its counters.
    pub fn token_ring(&self) -> &TokenRing {
        &self.ring
    }

    /// Whether the non-blocking poll reads through resolved addresses (#578).
    pub fn poll_is_mapped(&self) -> bool {
        self.ring.poll_is_mapped()
    }

    /// The generation ordinal of the last publish.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// A [`RingReader`] over this ring's history for another thread: `None`
    /// off Metal, when the ring's words are not CPU-mapped, or on a
    /// `quantum > 1` layout (no slot-per-token indexing). Take it before the
    /// run; it costs the loop nothing afterwards.
    ///
    /// The reader starts at stamp zero; on a ring an earlier run used,
    /// [`RingReader::resume_at_published`] starts it at this run's first
    /// token. Its stamp-to-slot mapping is exact from there on: every
    /// publish stamps the next generation and writes the next history slot,
    /// and the two drift apart only when `eos_of_last` retracts an EOS, which
    /// ends the run.
    pub fn reader(&self) -> Option<RingReader> {
        if self.history.ring.quantum != 1 || !self.ring.poll_is_mapped() {
            return None;
        }
        #[cfg(not(feature = "metal"))]
        {
            None
        }
        #[cfg(feature = "metal")]
        {
            let step = self.ring.step_counter().clone();
            // Resolved once, here, under the tensors' locks; never again.
            // SAFETY: the pointers are read through only while the clones live.
            let step_ptr = unsafe { step.peek_scalar_ptr::<u32>(0)? };
            let segments = self.history.ring.segments.clone();
            let seg_bases = segments
                .iter()
                .map(|s| unsafe { s.peek_scalar_ptr::<u32>(0) })
                .collect::<Option<Vec<_>>>()?;
            // The host copy where the arm keeps one: a read of the device's
            // would wait on the queue's tail, a prefill the loop just encoded.
            let eos_ids = if self.eos_ids_host.is_empty() {
                self.ring.eos_ids().ok()?
            } else {
                self.eos_ids_host.clone()
            };
            let total_tiles = self.history.ring.total_tiles;
            // At quantum 1 the next publish stamps `generation + 1` and
            // writes slot `fed`.
            let behind = self.stamps_ahead() % total_tiles;
            Some(RingReader {
                _segments: segments,
                _step: step,
                seg_bases,
                step_ptr,
                tiles_per_seg: self.history.ring.tiles_per_seg,
                total_tiles,
                eos_ids,
                cursor: 0,
                head: self.generation as usize,
                slot_shift: (total_tiles - behind) % total_tiles,
                ended: false,
                publish: self.ring.publish_event(),
            })
        }
    }
}

impl ReadbackStrategy for Ring {
    fn device(&self) -> &Device {
        self.ring.input_device().device()
    }

    fn arms(&self) -> ReadbackArms {
        ReadbackArms {
            token_ring: true,
            // Both the poll and the host-on-drain arms learn EOS without a
            // per-step blocking readback of the device flag.
            nonblocking_eos: matches!(
                self.eos,
                EosRead::Poll(_)
                    | EosRead::HostOnDrain
                    | EosRead::GpuBatched
                    | EosRead::GpuBatchedNonblocking
            ),
        }
    }

    /// Zero-copy prompt staging: write the chunk into the ring's reused input
    /// buffer ([`TokenRing::stage_prompt`], mapped write with an upload fallback)
    /// and return the `input_for` narrow the forward consumes — no per-chunk
    /// `Tensor` allocation, unlike the trait default.
    ///
    /// Falls back to the default whole-tensor path when the chunk is WIDER than
    /// the input buffer (`Geometry::max_input_width`) — the one-shot case, where
    /// the buffer is sized to `quantum` (1 for greedy) but the "chunk" is the
    /// whole prompt. So one-shot allocates its prompt tensor exactly as before,
    /// and a streamed run (chunk ≤ the widened buffer) takes the zero-copy path.
    fn stage_prompt_chunk(&mut self, tokens: &[u32]) -> Result<Tensor> {
        if tokens.len() > self.ring.input_width() {
            return Tensor::new(tokens, self.device())?.unsqueeze(0);
        }
        self.ring.stage_prompt(tokens)?;
        self.ring.input_for(tokens.len())
    }

    fn admits_lookahead(&self) -> bool {
        true
    }

    fn loop_commits(&self) -> bool {
        true
    }

    fn check_sampler(&self, sampling: &Sampling) -> Result<()> {
        // The ring is built on `sample_argmax_on_device` leaving the token on
        // the device: the body's table (`nn::decode::refuse`) says which
        // samplers that holds for.
        crate::nn::decode::refuse(
            crate::nn::decode::Body::Step { ring: true },
            &crate::nn::decode::Asks {
                sampling,
                processors: false,
                budget: false,
                digest: false,
            },
        )
    }

    fn sample_and_feed(
        &mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        let forcing = self.forcing.as_mut().filter(|f| f.live());
        if self.fused_publish && forcing.is_none() && matches!(sampler.sampling(), Sampling::ArgMax)
        {
            // The fused publish (lloom #735 step 5): the argmax, the history
            // slot, the input slot and the EOS test in one dispatch, then the
            // stamp as its own dispatch certifying the history segment. The
            // kernel's in-dispatch fence orders its stores for the GPU only;
            // a host can see its stamp before its slot (§11.5z-xvi), and the
            // command stream's barrier before a dispatch that binds the
            // segment is what a reader of the stamp relies on.
            let logits = if logits.rank() == 1 {
                logits.clone()
            } else {
                logits.flatten_all()?
            };
            self.fused_step = true;
            self.counters.ring_publishes += 1;
            self.counters.fused_publishes += 1;
            // A pooled ring whose previous run was unfused may still owe a window
            // stamp; it goes first, so the counter never steps back.
            self.stamp_unstamped()?;
            let (seg, offset) = self.history.begin_step()?;
            self.generation += 1;
            let history_slot = self.history.ring.segments[seg].narrow(0, offset, 1)?;
            let token_slot = self.ring.next_input_slot_view()?;
            self.device().argmax_publish(
                &logits,
                &token_slot,
                &history_slot,
                self.ring.eos_ids_tensor(),
                self.ring.step_counter(),
                0,
            )?;
            self.ring
                .publish_marker(self.generation, &[&self.history.ring.segments[seg]])?;
            self.history.wrote_one();
            return self.ring.input_for(1);
        }
        self.fused_step = false;
        // The fed token: the draw, or the budget's select over it — a device
        // tensor either way, published below exactly as a draw is.
        let sampled = match forcing {
            Some(f) => sampler.sample_argmax_on_device_fed(logits, |t| f.feed(&t))?,
            None => sampler.sample_argmax_on_device(logits)?,
        };
        self.counters.ring_publishes += 1;
        // Begin this step's tile (one token per step on this arm → quantum=1 →
        // each tile is one slot, so this is a dense write; the tile machinery is
        // identity at quantum=1). `begin_step` lazily closes the previous tile —
        // after `eos_of_last` had its chance to retract — and bails rather than
        // overwriting un-drained data (provably unreachable under the geometry).
        let (seg, offset) = self.history.begin_step()?;

        // Order matters, and it is the streaming protocol. The history slot is
        // written FIRST, then the stamp, and the stamp CERTIFIES the slot's
        // segment — so a host that observes the generation counter at `>= N`
        // has the slot written behind it, and the zero-copy `take_ids` reads the
        // tile through the mapped pointer without a `to_vec1` sync. Encoding
        // the stamp after the slot is not what orders it: on a concurrent
        // encoder two `copy2d` dispatches on different buffers are unordered,
        // and the stamp waits only on what it binds (`TokenRing::publish_marker`).
        self.generation += 1;
        // Into this step's tile (segment `seg`, `offset`), before the host knows
        // whether it is EOS; `eos_of_last` retracts by stepping the tile cursor
        // back. `begin_step` guaranteed the tile begins a contiguous run within
        // one segment, so no step ever straddles a segment boundary.
        let segment = &self.history.ring.segments[seg];
        segment.slice_set(&sampled.reshape(1)?, 0, offset)?;
        match self.eos {
            // Both per-step reads stamp every token: the stamp is what a
            // `RingReader` streams. The flag read waits on its flag anyway;
            // the stamp is one dispatch beside that round trip.
            EosRead::Poll(_) | EosRead::Blocking => {
                self.ring
                    .publish_step_certifying(&sampled, Some(self.generation), Some(segment))?
            }
            // Both deferred-EOS arms write only the input slot per step (the ~5 EOS
            // dispatches AND the per-token stamp both skipped); they learn EOS at
            // the commit window (host scan for `HostOnDrain`, the batched
            // `eos_index` dispatch for `GpuBatched`), and the marker the drain
            // clamps to is written once per window by `publish_marker`, which
            // certifies every segment noted here.
            EosRead::HostOnDrain | EosRead::GpuBatched | EosRead::GpuBatchedNonblocking => {
                self.ring.publish_input(&sampled)?;
                match self.unstamped.last_mut() {
                    Some((s, g)) if *s == seg => *g = self.generation,
                    _ => self.unstamped.push((seg, self.generation)),
                }
            }
        }
        self.history.wrote_one();
        // The tile is left OPEN; `eos_of_last` may retract this token, and the
        // next `begin_step` (or the final drain) closes it. A batched writer
        // (quantum>1) would loop `next_slot`+`wrote_one` per accepted token
        // within the one tile, then let the next step close it.
        //
        // Return the WIDTH-1 narrow, not the whole buffer: `publish` wrote one
        // sampled token at slot 0, and a decode step consumes exactly that one. The
        // input buffer may be WIDER than 1 (streamed prefill sized it to
        // `max_input_width`), so feeding `input()` whole would hand the forward
        // stale slots past the token. `input_for(1)` is a contiguous prefix view,
        // no copy — the same allocation, just the one live slot.
        self.ring.input_for(1)
    }

    fn set_thinking_budget(&mut self, budget: Option<&ThinkingBudget>) -> Result<()> {
        self.forcing = match budget {
            Some(b) => Some(DeviceForcing::new(b.clone(), self.device())?),
            None => None,
        };
        Ok(())
    }

    fn publish_marker(&mut self, force_live: bool) -> Result<()> {
        // The fused publish stamped inside `sample_and_feed`.
        if self.fused_step {
            return Ok(());
        }
        // Only the deferred-EOS arms write their marker at the commit window; the
        // per-step reads (poll, flag) stamped inside `sample_and_feed`.
        if !matches!(
            self.eos,
            EosRead::HostOnDrain | EosRead::GpuBatched | EosRead::GpuBatchedNonblocking
        ) || self.generation == 0
        {
            return Ok(());
        }
        // The non-blocking batched arm stages its index at `close_window`; a loop
        // that did not call it (or a window that grew since) is staged here, a
        // step later than ideal but still fenced by its own buffer, not the tail.
        // Not during the TTFT ramp (`force_live`), for `close_window`'s reason.
        if matches!(self.eos, EosRead::GpuBatchedNonblocking) && !force_live {
            let covered = self.staged_eos.borrow().last().is_some_and(|w| {
                w.start == self.history.drained && w.start + w.len == self.history.fed
            });
            if !covered {
                self.stage_window()?;
            }
        }
        // GpuBatched: enqueue the batched EOS check over this window's tokens
        // BEFORE the marker/commit, so the `eos_index` dispatch rides the committing
        // buffer AFTER the window's slots (queue order) and its `[1]` result is
        // ready to read back at the drain. The host reads that index once per
        // commit (`eos_in_drained`), never scanning the tokens itself. A window that
        // does not present a dense single-segment view (quantum>1, or a
        // segment-boundary straddle) yields `None` here and the drain falls back to
        // the host scan for this commit.
        if matches!(self.eos, EosRead::GpuBatched) {
            // `eos_index` is a Metal kernel; off Metal the window has no dense
            // device check and every drain takes the host scan.
            #[cfg(feature = "metal")]
            let pending = match self.history.window_view().filter(|t| t.device().is_metal()) {
                Some(tokens) => {
                    #[cfg(feature = "metal")]
                    {
                        let n = tokens.elem_count();
                        let idx = tokens.eos_index(n, self.ring.eos_ids_tensor())?;
                        Some((idx, n))
                    }
                    // `eos_index` is a Metal dispatch; without it there is nothing
                    // to enqueue and the drain's host scan covers this commit.
                    #[cfg(not(feature = "metal"))]
                    {
                        let _ = tokens;
                        None
                    }
                }
                None => None,
            };
            #[cfg(not(feature = "metal"))]
            let pending = None;
            *self.pending_eos_index.borrow_mut() = pending;
        }
        // The wake-cadence throttle. A commit is LIVE — writes the device word the
        // host WFE-waits on — when the ramp forces it, or when `live_every`
        // commits have passed since the last live one. A non-live commit writes
        // NOTHING: the device word stays at the last live value, so the drain
        // clamps more conservatively and simply catches up on the next live commit
        // (or the final `drain_all`, which force-synchronises). Skipping the write
        // can never read an unwritten slot (§3.5) — it only defers visibility.
        let since = self.commits_since_live.get() + 1;
        let live = force_live || since >= self.live_every;
        if live {
            // The last entry carries `self.generation`, `fed` through the last fed
            // step, so the marker publishes exactly the window the drain is about
            // to read.
            self.stamp_unstamped()?;
            self.commits_since_live.set(0);
        } else {
            self.commits_since_live.set(since);
        }
        Ok(())
    }

    /// The per-step EOS reads, unfused, with no thinking budget live: the
    /// draw is one history slot and one stamp, which a take-back leaves as
    /// an EOS retraction does. The deferred reads stamp per commit window, a
    /// fused publish draws inside its kernel, a live budget keeps device
    /// state a draw advances: none is taken back.
    fn retracts_draw(&self) -> bool {
        matches!(self.eos, EosRead::Poll(_) | EosRead::Blocking)
            && !self.fused_step
            && !self.fused_publish
            && self.forcing.as_ref().is_none_or(|f| !f.live())
    }

    /// Take back the last draw before `eos_of_last` has seen it: its history
    /// slot is retracted (the next publish writes it again), its stamp is
    /// spent — the token drawn again carries the next one, so a reader that
    /// read this one sees the new one as new ([`RingReader::retract_one`]),
    /// and `eos_of_last` waits for the new stamp, never reading the old
    /// flag. The input slot it wrote is not read again: the forward
    /// encoded again reads the next publish's.
    fn retract_draw(&mut self) -> Result<()> {
        if !self.retracts_draw() {
            lloom_tensor::bail!("Ring::retract_draw: this arm does not take a draw back");
        }
        if !(self.history.tile_open && self.history.in_tile > 0) {
            lloom_tensor::bail!("Ring::retract_draw: no draw in the open tile to take back");
        }
        self.history.retract_one();
        self.counters.ring_retracted += 1;
        Ok(())
    }

    fn eos_of_last(&mut self) -> Result<bool> {
        let is_eos = match self.eos {
            // EOS is not learned per token on the deferred arms — the loop learns
            // it from `eos_in_drained` on the batch it streams (a host scan for
            // `HostOnDrain`, the batched `eos_index` result for `GpuBatched`), so
            // this always answers `false` and never retracts here (the truncation
            // happens host-side, at the drain).
            EosRead::HostOnDrain | EosRead::GpuBatched | EosRead::GpuBatchedNonblocking => false,
            EosRead::Blocking => {
                self.counters.poll_iters += 1;
                self.ring.eos_flag().to_vec1::<u8>()?[0] != 0
            }
            // A ring whose words are not CPU-mapped cannot be polled at all —
            // `poll_eos` declines every time — so the wait is the blocking read
            // from the start, and counted as the fallback it is.
            EosRead::Poll(_) if !self.ring.poll_is_mapped() => {
                self.counters.poll_iters += 1;
                self.counters.poll_fallbacks += 1;
                self.ring.eos_flag().to_vec1::<u8>()?[0] != 0
            }
            #[cfg(feature = "metal")]
            EosRead::Poll(_) if self.ring.publish_event().is_some() => {
                self.counters.poll_iters += 1;
                let generation = self.generation;
                match self.ring.poll_eos(generation) {
                    Some(eos) => eos,
                    None => {
                        let ev = self.ring.publish_event().expect("armed above");
                        // The stamp must be on the GPU before the host parks on
                        // it: commit the buffer it sits in if nothing has yet.
                        if !ev.committed() {
                            if let Device::Metal(m) = self.ring.input_device().device() {
                                m.flush_and_prune()?;
                            }
                        }
                        self.counters.publish_parks += 1;
                        let landed = ev.wait(generation as u64, EOS_PARK_TIMEOUT_MS);
                        match self.ring.poll_eos(generation) {
                            Some(eos) if landed => eos,
                            // A timeout, or a stamp not yet visible behind its
                            // signal: the blocking read, counted.
                            _ => {
                                self.counters.poll_fallbacks += 1;
                                self.ring.eos_flag().to_vec1::<u8>()?[0] != 0
                            }
                        }
                    }
                }
            }
            EosRead::Poll(poll) => {
                let generation = self.generation;
                let poll_budget = self.poll_budget;
                let mut spins: u64 = 0;
                // The watched word, resolved once; `Wfe` arms the exclusive
                // monitor on it and every other arm ignores it.
                //
                // SAFETY: the pointer is valid while `self.ring` is alive, and
                // the ring outlives this loop.
                let watched = unsafe { self.ring.step_addr() };
                loop {
                    self.counters.poll_iters += 1;
                    match self.ring.poll_eos(generation) {
                        Some(eos) => break eos,
                        None => {
                            spins += 1;
                            if spins > poll_budget {
                                // Counted, not silent: a fallback nobody
                                // counts is a null that reads as a result.
                                self.counters.poll_fallbacks += 1;
                                break self.ring.eos_flag().to_vec1::<u8>()?[0] != 0;
                            }
                            match watched {
                                // Wait for the counter to CHANGE from what the
                                // poll just saw; the compare skips the wait
                                // when the word has already moved.
                                Some(p) => {
                                    // SAFETY: as above; the word may be
                                    // concurrently written, which is the point.
                                    let seen = unsafe { p.read_volatile() };
                                    unsafe { poll.wait_on(p, seen) }
                                }
                                None => poll.wait(),
                            }
                        }
                    }
                }
            }
        };
        if is_eos {
            self.counters.ring_eos_publishes += 1;
            // Retract the EOS token from the still-open tile (the loop never
            // streams it). The tile closes on the next `begin_step`.
            self.history.retract_one();
        }
        Ok(is_eos)
    }

    fn take_ids(&mut self) -> Result<Vec<u32>> {
        let mut out = Vec::with_capacity(self.history.undrained());
        self.drain_window(&mut out)?;
        Ok(out)
    }

    fn take_ids_into(&mut self, out: &mut Vec<u32>) -> Result<()> {
        let mut window = std::mem::take(&mut self.scratch);
        self.drain_window(&mut window)?;
        out.extend_from_slice(&window);
        self.scratch = window;
        Ok(())
    }

    fn drain_all(&mut self) -> Result<Vec<u32>> {
        // The deferred reads' last window may not be stamped yet (the wake
        // cadence skips commits, and the run can end between commits): stamp
        // it, so a reader's last read after the run sees every token. Its
        // tokens are undrained, so the wait below retires the stamp too.
        self.stamp_unstamped()?;
        self.taken = None;
        if self.history.undrained() == 0 {
            return Ok(Vec::new());
        }
        // The final drain: completeness over non-blocking. One wait for the
        // GPU, then the window is read — what the poll arm needs here, since
        // its incremental `take_ids` refuses to read past the published
        // counter. Drains everything to `fed`.
        let device = self.ring.input_device().device().clone();
        let mut out = Vec::with_capacity(self.history.undrained());
        self.history.drain_all_into(&device, &mut out)?;
        Ok(out)
    }

    /// The EOS position in a drained `batch` for the deferred arms.
    ///
    /// - [`EosRead::HostOnDrain`]: a plain Rust `position` scan against the id set
    ///   read back once at construction (`eos_ids_host`) — no device work.
    /// - [`EosRead::GpuBatched`]: the index the window's `eos_index` dispatch
    ///   already computed, read back ONCE here (a per-commit sync, off the
    ///   per-token path). The window the GPU scanned is exactly this drained batch
    ///   (`[drained, fed)` at the commit), so its index maps directly to a batch
    ///   position; a `>= n` result means no EOS in the window. When the drain
    ///   clamped SHORT of the matched index (the EOS token is fed but not yet
    ///   drained this commit), the match is left for a later drain — returning it
    ///   now would point past `batch`. If the window offered no dense view
    ///   (`pending` is `None` — quantum>1 or a boundary straddle), fall back to the
    ///   host scan.
    ///
    /// `None` (and empty `eos_ids_host`) on the per-token arms, which excluded EOS
    /// already.
    fn eos_in_drained(&self, batch: &[u32]) -> Option<usize> {
        if self.eos_ids_host.is_empty() {
            return None;
        }
        // A device index is in the frame of the window it scanned, which is
        // the batch the last `take_ids` returned; any other batch (the tail
        // `drain_all` returned, a reader-owned run's whole ids) is scanned
        // here.
        if self.taken != Some(batch.len()) {
            if matches!(self.eos, EosRead::GpuBatchedNonblocking) {
                self.eos_index_late.set(self.eos_index_late.get() + 1);
            }
            return batch.iter().position(|id| self.eos_ids_host.contains(id));
        }
        if matches!(self.eos, EosRead::GpuBatchedNonblocking) {
            // Newest window first: it is the widest. A window answers only if it
            // covers the whole batch AND its buffer has completed — then the read
            // is free. Otherwise the host scan below answers, and the count says so.
            let start = self.drain_start;
            let end = start + batch.len();
            for w in self.staged_eos.borrow().iter().rev() {
                if w.start > start || w.start + w.len < end || !w.staged.is_complete() {
                    continue;
                }
                let Ok(v) = w.staged.to_vec1::<u32>() else {
                    break;
                };
                let Some(&idx) = v.first() else { break };
                self.eos_index_hits.set(self.eos_index_hits.get() + 1);
                let idx = idx as usize;
                if idx >= w.len {
                    return None; // the window holds no EOS, so neither does the batch.
                }
                let abs = w.start + idx;
                return (abs >= start && abs < end).then(|| abs - start);
            }
            self.eos_index_late.set(self.eos_index_late.get() + 1);
            return batch.iter().position(|id| self.eos_ids_host.contains(id));
        }
        if matches!(self.eos, EosRead::GpuBatched) {
            if let Some((idx_t, n)) = self.pending_eos_index.borrow().as_ref() {
                // Read the one index the dispatch wrote (per-commit sync).
                let idx = idx_t.to_vec1::<u32>().ok()?.first().copied()? as usize;
                if idx < *n && idx < batch.len() {
                    return Some(idx);
                }
                // idx == n (no match) or the match is past what this drain took.
                return None;
            }
            // No dense window view this commit — fall back to the host scan below.
        }
        batch.iter().position(|id| self.eos_ids_host.contains(id))
    }

    fn note_eos_truncation(&mut self, dropped_in_batch: usize) {
        // Only the host-on-drain arm can hand the loop a batch with an EOS in
        // it; the others retracted theirs per step and never get here.
        self.counters.ring_eos_publishes += 1;
        // The undrained slots are overrun the loop enqueued before it could
        // stop: forget them, so the next run's first drain starts at the next
        // publish rather than returning them as its tokens.
        let undrained = self.history.forget_undrained();
        self.counters.ring_discarded += (dropped_in_batch + undrained) as u64;
    }

    fn eos_ids_tensor(&self) -> Option<&Tensor> {
        Some(self.ring.eos_ids_tensor())
    }

    fn reader(&self) -> Option<RingReader> {
        Ring::reader(self)
    }

    fn arm_publish_event(
        &mut self,
        for_reader: bool,
        eos_park: bool,
        adaptive_commit: bool,
    ) -> Result<()> {
        let eos_waits = eos_park && !adaptive_commit && matches!(self.eos, EosRead::Poll(_));
        let on = self.ring.poll_is_mapped() && (for_reader || eos_waits);
        self.ring.arm_publish_event(on)
    }

    fn stamps_each_token(&self, adaptive_commit: bool) -> bool {
        match self.eos {
            EosRead::Blocking | EosRead::Poll(_) => true,
            EosRead::HostOnDrain | EosRead::GpuBatched | EosRead::GpuBatchedNonblocking => {
                adaptive_commit
            }
        }
    }

    fn set_fused_publish(&mut self, on: bool) {
        self.fused_publish = on && self.ring.poll_is_mapped();
        self.fused_step = false;
    }

    fn publishes_fused(&self) -> bool {
        self.fused_step
    }

    fn eos_read_arm(&self) -> &'static str {
        match self.eos {
            EosRead::Blocking => "flag",
            EosRead::Poll(_) => "poll",
            EosRead::HostOnDrain => "drain",
            EosRead::GpuBatched => "gpubatch",
            EosRead::GpuBatchedNonblocking => "gpubatch-nonblocking",
        }
    }

    fn counters(&self) -> ReadbackCounters {
        ReadbackCounters {
            eos_index_hits: self.eos_index_hits.get(),
            eos_index_late: self.eos_index_late.get(),
            ..self.counters
        }
    }

    fn close_window(&mut self, in_ttft_ramp: bool) -> Result<()> {
        if matches!(self.eos, EosRead::GpuBatchedNonblocking)
            && self.generation > 0
            && !in_ttft_ramp
        {
            self.stage_window()?;
        }
        Ok(())
    }
}

impl Ring {
    /// Stamps issued beyond the history's real tokens: one for every EOS a
    /// per-step arm stamped and then retracted (`eos_of_last`), whose slot the
    /// next write reuses. Changes only at such a retraction, which ends its
    /// run, so it is fixed for the whole of any later run: stamp `g` of that
    /// run certifies the history's first `g − stamps_ahead` tokens.
    fn stamps_ahead(&self) -> usize {
        let issued = self.generation as usize;
        debug_assert!(
            issued >= self.history.fed,
            "a history token without a stamp"
        );
        issued.saturating_sub(self.history.fed)
    }

    /// Stamp every history segment written since the last stamp, oldest first,
    /// each with the generation of its own last slot: every stamp certifies
    /// only slots behind it, and the last carries `fed`.
    fn stamp_unstamped(&mut self) -> Result<()> {
        for &(seg, g) in &self.unstamped {
            self.ring
                .publish_marker(g, &[&self.history.ring.segments[seg]])?;
        }
        self.unstamped.clear();
        Ok(())
    }

    /// The published window `[drained, min(fed, published))` into `out`
    /// (cleared first), advancing the drain — the non-blocking incremental
    /// drain behind [`ReadbackStrategy::take_ids`]. On the stamped arms the
    /// loop may be ahead of the GPU, so the window is clamped to the tokens
    /// the GPU's published stamp certifies ([`Self::stamps_ahead`]): the slot
    /// is written before the stamp, and a read past it would read a slot not
    /// yet written (§3.5). With no mapped counter to clamp to it waits for the
    /// GPU instead. The mapped window read takes no lock and makes no copy;
    /// the copy fallback synchronises.
    fn drain_window(&mut self, out: &mut Vec<u32>) -> Result<()> {
        out.clear();
        self.drain_start = self.history.drained;
        self.taken = Some(0);
        if self.history.undrained() == 0 {
            return Ok(());
        }
        let fed = self.history.generation();
        let end = match self.eos {
            EosRead::Poll(_)
            | EosRead::HostOnDrain
            | EosRead::GpuBatched
            | EosRead::GpuBatchedNonblocking => match self.ring.published_generation() {
                Some(g) => fed.min((g as usize).saturating_sub(self.stamps_ahead())),
                None => {
                    self.ring.input_device().device().synchronize()?;
                    fed
                }
            },
            EosRead::Blocking => fed,
        };
        #[cfg(feature = "metal")]
        let mapped = self.history.drain_mapped(end, out);
        #[cfg(not(feature = "metal"))]
        let mapped = {
            let _ = end;
            false
        };
        if !mapped {
            self.history.drain_copy(fed, out)?;
        }
        self.taken = Some(out.len());
        Ok(())
    }

    /// [`EosRead::GpuBatchedNonblocking`]: enqueue the `eos_index` over the
    /// undrained window and stage its `[1]` result behind the window's own
    /// command buffer; keep it beside the previous window's. Nothing to stage
    /// when the window has no dense view (quantum > 1, a segment straddle) or
    /// off Metal, where the drain's host scan answers.
    fn stage_window(&mut self) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Some(tokens) = self.history.window_view().filter(|t| t.device().is_metal()) {
            let n = tokens.elem_count();
            let idx = tokens.eos_index(n, self.ring.eos_ids_tensor())?;
            let staged = idx.stage_read()?;
            let mut windows = self.staged_eos.borrow_mut();
            if windows.len() == 2 {
                windows.remove(0);
            }
            windows.push(StagedWindow {
                staged,
                start: self.history.drained,
                len: n,
            });
        }
        Ok(())
    }
}

/// How often the loop commits and prunes the in-flight queue
/// (`design/docs/decode-driver-design.md`, "where the CommitBudget's cadence
/// comes from").
///
/// `steps(1)` is the incumbent ring arm's per-token commit, rendered
/// `per-token`; `steps(N)` commits and prunes every N steps. Under a loop that
/// waits before encoding the next step the cadence is moot — the publish must
/// be committed before the host can wait on it, and the loop commits there
/// regardless — so the budget only becomes real with lookahead on. The
/// `commits` counter says which happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommitBudget {
    steps: usize,
}

impl std::fmt::Display for CommitBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.arm())
    }
}

lloom_tensor::serde_via_str!(CommitBudget);

impl CommitBudget {
    /// Commit and prune every step — the incumbent ring arm.
    pub fn per_token() -> Self {
        Self { steps: 1 }
    }

    /// Commit and prune every `n` steps. `n = 0` is refused: a budget that
    /// never prunes lets `Commands::in_flight` grow by one command buffer per
    /// commit for the length of the generation (§11.5h).
    pub fn steps(n: usize) -> Result<Self> {
        if n == 0 {
            lloom_tensor::bail!("CommitBudget::steps(0): the queue would never be pruned");
        }
        Ok(Self { steps: n })
    }

    /// Whether `since` steps since the last commit are enough to commit again.
    pub fn due(&self, since: usize) -> bool {
        since >= self.steps
    }

    /// Steps per commit.
    pub fn period(&self) -> usize {
        self.steps
    }

    /// The arm's name for a `RESULT` line: `per-token` or `steps:<N>`, the
    /// period carried verbatim (§13.5m: an arm whose period is dropped pools
    /// with every other period of itself).
    pub fn arm(&self) -> String {
        if self.steps == 1 {
            "per-token".to_string()
        } else {
            format!("steps:{}", self.steps)
        }
    }
}

impl std::str::FromStr for CommitBudget {
    type Err = String;

    /// Parses `per-token` and `steps:<N>`. `steps:1` is refused in favour of
    /// its own name, so one arm has one spelling.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "per-token" => Ok(Self::per_token()),
            other => match other.strip_prefix("steps:") {
                Some(n) => {
                    let n: usize = n
                        .parse()
                        .map_err(|e| format!("commit budget {other:?}: bad step count: {e}"))?;
                    if n == 1 {
                        return Err(format!(
                            "commit budget {other:?}: a one-step budget is spelled `per-token`"
                        ));
                    }
                    Self::steps(n).map_err(|e| e.to_string())
                }
                None => Err(format!(
                    "unknown commit budget {other:?} (expected per-token or steps:<N>)"
                )),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::str::FromStr;

    /// A draw taken back leaves the blocking arm as it was before it: the
    /// ids, and a thinking budget's position — a draw retracted inside the
    /// forced tail and made again feeds the tail's same token, so the run's
    /// fed ids are a run's without the retraction.
    #[test]
    fn a_retracted_draw_leaves_the_ids_and_the_budget_as_before() {
        let dev = Device::Cpu;
        // Logits (as the loop hands them, one row squeezed) whose argmax is
        // 1, every step.
        let logits =
            Tensor::new(&[0f32, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &dev).unwrap();
        let budget = ThinkingBudget::new(2, vec![9], vec![3, 4, 9]).unwrap();
        let fed = |rb: &mut Blocking, s: &mut LogitsProcessor| -> u32 {
            rb.sample_and_feed(s, &logits)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<u32>()
                .unwrap()[0]
        };
        let run = |retract_at: &[usize]| -> (Vec<u32>, Vec<u32>) {
            let mut rb = Blocking::new(&[100], &dev).unwrap();
            rb.set_thinking_budget(Some(&budget)).unwrap();
            let mut s = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
            let mut out = Vec::new();
            for step in 0..6 {
                if retract_at.contains(&step) {
                    fed(&mut rb, &mut s);
                    rb.retract_draw().unwrap();
                }
                out.push(fed(&mut rb, &mut s));
                assert!(!rb.eos_of_last().unwrap());
            }
            (out, rb.take_ids().unwrap())
        };
        let want = run(&[]);
        assert_eq!(want.0, vec![1, 1, 3, 4, 9, 1], "the budget's tail is fed");
        assert_eq!(run(&[0, 2, 3, 5]), want);
    }

    #[test]
    fn budget_parses_and_renders_its_period() {
        assert_eq!(
            CommitBudget::from_str("per-token").unwrap().arm(),
            "per-token"
        );
        assert_eq!(CommitBudget::from_str("steps:8").unwrap().arm(), "steps:8");
        assert_eq!(CommitBudget::from_str("steps:8").unwrap().period(), 8);
        assert!(CommitBudget::from_str("steps:0").is_err());
        assert!(CommitBudget::from_str("steps:1").is_err());
        assert!(CommitBudget::from_str("steps:").is_err());
        assert!(CommitBudget::from_str("every-token").is_err());
        assert!(CommitBudget::from_str("").is_err());
    }

    #[test]
    fn budget_is_due_at_its_period() {
        let b = CommitBudget::from_str("steps:4").unwrap();
        assert!(!b.due(3));
        assert!(b.due(4));
        assert!(b.due(5));
        assert!(CommitBudget::per_token().due(1));
        assert!(!CommitBudget::per_token().due(0));
    }

    // ---- History: the ring-of-rings, driven directly (no Metal) -------------
    //
    // These exercise the tile placement, the segment hop, the write-reset
    // (implicit: tiles never straddle), the strided read, and the lap invariant
    // at quantum>1 — the path the GPU pipeline cannot reach until a batched
    // (#694) writer exists. A CPU `Device` backs the buffers, so this runs
    // anywhere, fast, under ASan/Miri. The oracle: whatever real tokens were
    // written-and-kept must come back from the drain in order, with no waste and
    // no overwrite.

    /// A tiny deterministic PRNG (SplitMix64), so the fuzz is reproducible and
    /// needs no `rand` dep. `Math.random`/`Date` are irrelevant here — a fixed
    /// seed sweeps the space and any failure reproduces exactly.
    struct SmallRng(u64);
    impl SmallRng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn upto(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    /// Write `n` real tokens into the open tile of a `quantum>1` step, using the
    /// `next_slot`/`wrote_one` seam a batched writer would — the test's stand-in
    /// for that writer.
    fn write_tile(h: &mut History, ids: &[u32]) {
        for &id in ids {
            let (seg, slot) = h.next_slot();
            let one = Tensor::new(&[id], h.ring.segments[seg].device()).unwrap();
            h.ring.segments[seg].slice_set(&one, 0, slot).unwrap();
            h.wrote_one();
        }
    }

    /// Drain via the copy path (the deviceless equivalent of the mapped read).
    /// Closes the open tile first, so the test sees ALL kept tokens (the prod
    /// incremental `take_ids` deliberately leaves the open tile for the next
    /// drain; the final `drain_all` closes it — this mirrors `drain_all`).
    fn drain(h: &mut History) -> Vec<u32> {
        h.close_open_tile().unwrap();
        let fed = h.generation();
        let snapshots: Vec<Vec<u32>> = h
            .ring
            .segments
            .iter()
            .map(|s| s.to_vec1::<u32>().unwrap())
            .collect();
        let mut out = Vec::new();
        h.drain_into(fed, &mut out, |seg, slot| snapshots[seg][slot]);
        out
    }

    #[test]
    fn history_quantum1_is_a_dense_ring() {
        // The B=1 degenerate case: quantum=1, each step one token; the drain must
        // return exactly what was fed, and the ring must reuse slots (a run far
        // longer than the ring does not lap because we drain each step).
        let dev = Device::Cpu;
        let mut h = History::new(8, 1, &dev).unwrap();
        let mut expect = Vec::new();
        for i in 0..100u32 {
            h.begin_step().unwrap();
            write_tile(&mut h, &[i]);
            expect.push(i);
            let got = drain(&mut h);
            assert_eq!(got, &expect[expect.len() - got.len()..]);
        }
        // Everything drained, in order.
        assert_eq!(h.undrained(), 0);
    }

    #[test]
    fn history_quantum_gt1_strided_read_skips_waste() {
        // A quantum=4 ring: each step writes 1..4 real tokens (the rest of the
        // tile is waste). The drain must return ONLY the real tokens, in order,
        // never a waste slot — the strided read's whole job. Drains every few
        // steps so the ring wraps across its segments repeatedly.
        let dev = Device::Cpu;
        let quantum = 4;
        // Small ring so it wraps often; SEG_MAX_SLOTS makes each segment large,
        // so force several segments by asking for many tiles.
        let mut h = History::new(quantum * 6, quantum, &dev).unwrap();
        let mut rng = SmallRng(0xC0FF_EE00_1234_5678);
        let mut expect = Vec::new();
        let mut got_all = Vec::new();
        let mut next_id = 1u32;
        for step in 0..2000 {
            h.begin_step().unwrap();
            let n = 1 + rng.upto(quantum); // 1..=quantum real tokens
            let ids: Vec<u32> = (0..n)
                .map(|_| {
                    let x = next_id;
                    next_id += 1;
                    x
                })
                .collect();
            write_tile(&mut h, &ids);
            expect.extend_from_slice(&ids);
            // Drain on a varying cadence, but always before the window could lap
            // (leave room: drain whenever we have >= a couple tiles live).
            if step % 3 == 2 {
                got_all.extend(drain(&mut h));
            }
        }
        got_all.extend(drain(&mut h));
        // Every real token, in order, no waste (waste is never 0-valued here:
        // ids start at 1, and poison in debug would show as 0xDEADB171).
        assert_eq!(
            got_all, expect,
            "strided read must return exactly the real tokens"
        );
        assert!(
            !got_all.contains(&TILE_POISON_TEST),
            "a poison/ waste slot leaked into the drain"
        );
    }

    #[test]
    fn history_lap_bails_not_corrupts() {
        // If the writer opens more tiles than the ring holds without draining,
        // the lap guard must BAIL (loud), never overwrite live data. `begin_step`
        // lazily closes the previous tile, so the lap fires once enough tiles are
        // CLOSED to leave no free tile; keep opening (never draining) and assert
        // it bails before it could ever overwrite — and within a bounded number
        // of steps (never silently accepts an unbounded backlog).
        let dev = Device::Cpu;
        let quantum = 2;
        let mut h = History::new(quantum * 3, quantum, &dev).unwrap();
        let n_tiles = h.total_tiles();
        let mut bailed = false;
        for i in 0..n_tiles + 2 {
            match h.begin_step() {
                Ok(_) => write_tile(&mut h, &[i as u32 + 1]),
                Err(_) => {
                    bailed = true;
                    break;
                }
            }
        }
        assert!(
            bailed,
            "opening past the live-tile bound must bail within the ring's size, not overwrite"
        );
    }

    #[test]
    fn history_retract_drops_the_eos_token() {
        // A tile whose last (only) token is retracted (an EOS) must not appear in
        // the drain, and its tile must be reusable (not counted).
        let dev = Device::Cpu;
        let mut h = History::new(8, 1, &dev).unwrap();
        h.begin_step().unwrap();
        write_tile(&mut h, &[42]);
        h.retract_one(); // 42 was an EOS
                         // Next step's token is the real one.
        h.begin_step().unwrap();
        write_tile(&mut h, &[7]);
        let got = drain(&mut h);
        assert_eq!(got, vec![7], "the retracted EOS must not be drained");
    }

    // The poison constant is debug-only; give the test a stable name for it.
    #[cfg(any(debug_assertions, feature = "fuzzing"))]
    const TILE_POISON_TEST: u32 = TILE_POISON;
    #[cfg(not(any(debug_assertions, feature = "fuzzing")))]
    const TILE_POISON_TEST: u32 = 0xDEAD_B171;

    /// The WINDOW drain (`drain_mapped`) on a REAL Metal device — the scan-free,
    /// mapped-pointer path the deviceless `drain_into` fuzz above cannot reach.
    /// This is the read the crown actually runs (`poll_fallbacks=0`), and its bug
    /// class is the one this change fixes: a window that spans a SEGMENT BOUNDARY
    /// must come back as one in-order run, not a fragmented patchwork.
    ///
    /// Forces ≥2 segments (a ring larger than one `SEG_MAX_SLOTS` segment) and
    /// feeds a run long enough to WRAP across the boundary many times, draining on
    /// a varying cadence so the window `[drained, end)` lands both inside a segment
    /// and straddling two. Asserts the drained ids equal the fed ids exactly.
    #[test]
    fn history_window_drain_spans_segments_on_device() {
        #[cfg(feature = "metal")]
        {
            let Ok(dev) = Device::new_metal(0) else {
                eprintln!("no Metal device; skipping window-drain-on-device fuzz");
                return;
            };
            // total_tiles > tiles_per_seg (== SEG_MAX_SLOTS at quantum 1) forces
            // ≥2 segments. Ask for two segments' worth of slots.
            let cap = SEG_MAX_SLOTS * 2 + 5;
            let mut h = History::new(cap, 1, &dev).unwrap();
            assert!(
                h.ring.segments.len() >= 2,
                "test needs ≥2 segments, got {}",
                h.ring.segments.len()
            );

            let mut rng = SmallRng(0x5E6D_E175_0BAD_F00Du64);
            let mut next_id = 1u32;
            let mut fed_since_drain: Vec<u32> = Vec::new();
            // Run well past 2× the ring so the window wraps the segment boundary
            // repeatedly; ids stay non-zero so a stray init/poison slot is visible.
            for step in 0..(SEG_MAX_SLOTS * 5) {
                h.begin_step().unwrap();
                let id = next_id;
                // Keep ids non-zero so a stray init slot (0) or poison is visible.
                next_id = next_id.wrapping_add(1);
                if next_id == 0 {
                    next_id = 1;
                }
                let one = Tensor::new(&[id], &dev).unwrap();
                let (seg, off) = h.tile_home(h.tiles);
                h.ring.segments[seg].slice_set(&one, 0, off).unwrap();
                h.wrote_one();
                fed_since_drain.push(id);

                // Vary the drain cadence (1..=7 steps) so the window sometimes
                // sits inside one segment and sometimes straddles the boundary,
                // but always before the ring could lap.
                if step % (1 + rng.upto(7)) == 0 {
                    // Close the open tile so the whole fed run is visible, then
                    // drain the window by mapped pointer. Sync first: `slice_set`
                    // enqueues an async copy, and this test reads the RAW mapped
                    // buffer (`peek_scalar_ptr`) without the GPU-counter clamp the
                    // production path uses (`published_generation`), so the writes
                    // must be retired before the read. The property under test is
                    // the window ARITHMETIC (which slots are read), not the
                    // visibility ordering the counter handles in prod.
                    dev.synchronize().unwrap();
                    h.close_open_tile().unwrap();
                    let end = h.generation();
                    let mut out = Vec::new();
                    let mapped = h.drain_mapped(end, &mut out);
                    assert!(
                        mapped,
                        "drain_mapped must take the mapped path at quantum 1"
                    );
                    assert_eq!(
                        out, fed_since_drain,
                        "window drain diverged at step {step} (seg-spanning run wrong)"
                    );
                    assert!(
                        !out.contains(&TILE_POISON_TEST),
                        "a poison/waste slot leaked into the window drain at step {step}"
                    );
                    fed_since_drain.clear();
                }
            }
            // Final sweep: whatever remains must drain in order.
            dev.synchronize().unwrap();
            h.close_open_tile().unwrap();
            let end = h.generation();
            let mut out = Vec::new();
            assert!(h.drain_mapped(end, &mut out));
            assert_eq!(out, fed_since_drain, "final window drain diverged");
            assert_eq!(h.undrained(), 0, "everything must be drained");
        }
    }
}
