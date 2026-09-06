//! Record one decode step, encode the replayable subset into an
//! `MTLIndirectCommandBuffer`, and execute it by range (`DESIGN.md` §11.3).
//!
//! This is `DESIGN.md` §17 Phase 2 item 10, and it is deliberately **partial**.
//! §9.2i measures 123 of the 554 decode dispatch positions as varying across
//! steady-state steps under `Sdpa × arena`, and 48 of those need
//! `dispatchThreadgroupsWithIndirectBuffer` whatever the allocator does. Those
//! are not covered here and are not worked around: a second mechanism in the
//! same change would make a bisect impossible, which is the rule #64 split #82
//! out under and #82 declined again.
//!
//! # What "replayable" means here, and the axis that is easy to miss
//!
//! §9.2h and §9.2i classify the stream by what *varies* across steps -- buffer
//! identity, grid, binding offset. That is necessary and it is not sufficient.
//! An ICB command has no `setBytes` in any form (§3.7c), so a position whose
//! kernel binds its constants inline is unencodable **however stable it is**.
//! Coverage is therefore the intersection of two independent axes:
//!
//! * stable across steady-state steps (identity, grid, offset), and
//! * dispatched through a `_packed` entry point, so the constants live in a
//!   buffer a `setKernelBuffer` can bind.
//!
//! The second axis is what `set_default_param_style` exists for. Before it,
//! every decode dispatch bound its scalars inline and the covered set was
//! **empty** -- the packed variants from #38 through #81 were compiled and never
//! dispatched. Reporting only the first axis is what makes "123 positions stand
//! between here and replay" read as though closing that gap were enough.
//!
//! # Why record-then-replay, and what happens when the recording goes stale
//!
//! §11.1a gives the two options: record-and-validate, or have the model declare
//! a stable region. Candle is eager and declares nothing, so this records. The
//! validation is not a checksum over the recording -- it is that every replayed
//! position was *observed* to be invariant over the steps that were recorded,
//! and that the live step's own dispatches still match. A position that stops
//! matching falls back to the classical path for that step rather than replaying
//! a stale command, because under `HazardTrackingModeUntracked` a wrong buffer
//! is silent corruption rather than an error (§3.5).
use crate::kernels::metal::{Buffer, ComputePipeline};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLComputePipelineState, MTLDevice, MTLIndirectCommandBuffer,
    MTLIndirectCommandBufferDescriptor, MTLIndirectCommandType, MTLIndirectComputeCommand,
    MTLResourceOptions, MTLSize,
};
use parking_lot::Mutex;
use std::sync::Arc;

use super::executor::{
    DispatchAction, DispatchRecord, Disposition, Executor, Grid, IcbRange, Size3,
};

/// One buffer binding, as an ICB command needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BindingRecord {
    pub index: usize,
    /// Kept as the pointer rather than the `Buffer`, so comparing two steps'
    /// recordings compares identity rather than a handle that may have been
    /// reissued. The `Buffer` itself is held separately, in `retained`.
    pub ptr: usize,
    pub offset: usize,
    /// How long the handle addresses from `offset`, so two bindings can be
    /// tested for overlap rather than only for sharing an allocation.
    ///
    /// Under an arena every activation shares one pointer (§9.2e), so a
    /// pointer-only test would call every pair inside the arena a conflict and
    /// emit a barrier before nearly every command.
    pub len: usize,
    /// Whether the dispatch writes this binding.
    ///
    /// Needed to decide which replayed commands require a barrier: an ICB's
    /// commands are `ConcurrentDispatch` and the GPU does not drain between
    /// them (§3.5), so the read-after-write edges candle's `auto_barrier` emits
    /// classically have to be re-expressed on the commands themselves.
    pub is_output: bool,
}

/// One dispatch, with everything an ICB command must carry.
///
/// `DispatchRecord` deliberately does not hold bindings -- an executor that only
/// counts should not pay for them -- so this is the replay-facing record that
/// pairs a dispatch with the binds that preceded it.
#[derive(Clone, Debug)]
pub struct StepDispatch {
    pub kernel: Option<Arc<str>>,
    pub grid: Grid,
    pub threads_per_threadgroup: Size3,
    pub bindings: Vec<BindingRecord>,
    /// The pipeline this position ran, retained so the ICB command can be
    /// encoded against it later. Metal offers no way to read a bound pipeline
    /// back off an encoder, which is why `ComputePipeline` carries its name and
    /// why this has to be captured as it happens.
    pub pipeline: Option<ComputePipeline>,
    /// Every buffer bound at this position, retained for residency.
    ///
    /// `useResource` is mandatory for anything an ICB command touches and
    /// omitting it is silent corruption rather than an error (§3.7). Unified
    /// memory can mask the omission, which makes it exactly the kind of thing to
    /// do correctly rather than discover.
    pub retained: Vec<Buffer>,
}

impl BindingRecord {
    /// Whether two bindings can touch the same byte.
    ///
    /// Mirrors `encoder::BoundRange::overlaps`, including its rule that a zero
    /// length covers the whole allocation: an unknown extent must fail toward
    /// ordering, because a spurious barrier costs throughput and a missing one
    /// is silent corruption (§3.5).
    fn overlaps(&self, other: &BindingRecord) -> bool {
        if self.ptr != other.ptr {
            return false;
        }
        if self.len == 0 || other.len == 0 {
            return true;
        }
        self.offset < other.offset + other.len && other.offset < self.offset + self.len
    }
}

impl StepDispatch {
    /// Whether this dispatch must be ordered after `earlier`.
    ///
    /// Read-after-write, write-after-write and write-after-read against the same
    /// bytes -- the three candle's `auto_barrier` detects, applied to the same
    /// binding records rather than to a second notion of a hazard.
    fn conflicts_with(&self, earlier: &StepDispatch) -> bool {
        self.bindings.iter().any(|b| {
            earlier
                .bindings
                .iter()
                .any(|e| (b.is_output || e.is_output) && b.overlaps(e))
        })
    }

    /// Whether two recordings of the same position agree in everything an ICB
    /// command freezes.
    ///
    /// Deliberately compares the *encoded* quantities and not, say, a step
    /// index: an ICB command holds a pipeline, a grid, a threadgroup size and a
    /// set of `(index, buffer, offset)` triples, and two steps that agree on all
    /// of those are two steps one command can serve.
    fn replay_compatible(&self, other: &StepDispatch) -> bool {
        self.kernel == other.kernel
            && self.grid == other.grid
            && self.threads_per_threadgroup == other.threads_per_threadgroup
            && self.bindings == other.bindings
    }
}

/// How far back a command's barrier scan looks, i.e. what "ordered already"
/// means when deciding whether this command needs a `setBarrier`.
///
/// This is the axis `DESIGN.md` §11.3l's open question 3 asks about. An ICB's
/// `setBarrier` orders its command after **every** command before it in the
/// buffer, where candle's `auto_barrier` is per hazard -- so the ICB primitive
/// is strictly stronger and can only over-order. What was not priced is *how
/// much*, and the answer turns on how far back the conflict scan runs rather
/// than on the primitive's coarseness.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BarrierScope {
    /// Scan to the head of the run. Conservative, and the shipped default.
    #[default]
    RunStart,
    /// Scan back only to the previous barrier in this run.
    ///
    /// **Measured wrong, and kept as the negative control that shows why**
    /// (issue #115 follow-up). It halves the barrier count — 401 → 204 — and
    /// it corrupts, which is the outcome that answers open question 3.
    ///
    /// The reasoning it was built on: `auto_barrier` emits one barrier and then
    /// *replaces* `prev_outputs` with what followed it, so an edge an earlier
    /// barrier already covers is not re-ordered. That is sound for candle and
    /// does **not** transfer, because the two primitives are different objects:
    ///
    /// * `memoryBarrierWithScope` is a **fence in the stream**. It sits between
    ///   two dispatches, so everything encoded before it is ordered against
    ///   everything encoded after it. Clearing the pending set is therefore
    ///   correct — the edge really is discharged for every later dispatch.
    /// * `setBarrier` is a **property of one command**. It orders *that command*
    ///   after the commands before it and says nothing about its successors. So
    ///   a barrier on command `k` does not discharge an edge from `k-2` to
    ///   `k+3`: commands `k+1..` are still concurrent with `k-2`.
    ///
    /// Observed on LFM2, three runs: one produced a wrong-but-valid digest
    /// (`590dd18e…`, 8 tokens against the canonical 82), and two failed
    /// sampling outright with *"A weight is negative, too large or not a valid
    /// number"* — the same signature §11.3l records for replaying with no
    /// barriers at all. **Nondeterministic across runs**, which is §3.5's
    /// missing-edge class rather than a different arithmetic.
    ///
    /// Kept compiled, as #70 keeps its concurrent bump allocator, so the check
    /// has something to reject: a test asserting the shipped scan is necessary
    /// over a one-variant type asserts very little.
    SinceBarrier,
}

/// Whether candle's own `auto_barrier` still fires at a position the ICB is
/// replaying (`DESIGN.md` §11.3p, issue #144).
///
/// # What this axis is
///
/// `auto_barrier` is the **first statement** of `dispatch_thread_groups` and
/// `offer_to_executor` is consulted **after** it (`encoder.rs:405` and `:416`),
/// so a replayed position emits candle's barrier *and* the ICB's. §11.3n
/// measures the consequence: the 401 encoded `setBarrier` commands are
/// **additive** rather than substitutive, for 906 orderings against the
/// classical arm's 505. §11.3p then attributes the 505 and finds **393 of them
/// fire at covered non-head positions** -- where the ICB has already encoded
/// ordering covering the same edges.
///
/// This axis is the mechanism for asking whether those 393 can go. It is
/// deliberately **not** a new arm of [`BarrierScope`], which governs how far
/// back the ICB's own scan looks and whose `SinceBarrier` arm is wrong on
/// purpose -- issue #144's "Not in scope" is explicit that `BarrierScope` stays
/// as it is.
///
/// # Why it is safe to land dark
///
/// `Always` is the default and is byte-for-byte what ships. And `Classical` is
/// unaffected **by construction**: `ExecutorSlot::Classical` answers
/// `Disposition::Unknown` without a virtual call, so with no executor installed
/// there is no second ordering source and the axis has nothing to act on. That
/// is the property `HazardKey` holds (§9.2e, §9.2f) and the one that makes this
/// reviewable.
///
/// # The predicate, and why it is narrower than "the position is covered"
///
/// Argued in full in `measurements/issue-144-predicate.md`, committed before
/// this code was written. §11.3p gives three obstacles and each is discharged
/// differently:
///
/// * **Run heads are excluded by construction.** A head's scan slice is empty
///   (`covered[run_first_cmd..cmd_index]`), so candle's fence is the only thing
///   ordering a gap dispatch into it -- 30 of the 505, every one required.
///   Suppression is decided **once per run at the head** and applied only to the
///   run's interior, which makes "head" structural rather than a predicate
///   someone has to remember to exclude. 32 decisions per step, not 433.
/// * **The primitives are different objects** (§11.3m). Candle's barrier at a
///   covered position also orders the 123 uncovered positions interleaved
///   between runs, which `setBarrier` does not. Measured as edges rather than
///   counts: removing the ICB's own barriers breaks **1960** edges, so its
///   ordering is load-bearing, and the specific gap-writer edges this obstacle
///   warns about are **11**, all out of `sdpa_vector` and `rope_f16`.
/// * **7 of 31 covered runs cross an encoder-session seam**, at multiples of
///   `CANDLE_METAL_COMPUTE_PER_BUFFER`. Those positions carry no classical
///   barrier at all (the new session's hazard state is empty), so nothing is
///   suppressed at them -- and it is the seam's own ordering, via
///   `prev_ce_outputs` and `wait_for_buffer` on the shared pointer, that saves
///   all 11 of obstacle 2's edges. **No barrier count shows this**, which is why
///   the argument had to be made in edges.
///
/// # The gate
///
/// This **removes ordering edges**, and under `HazardTrackingModeUntracked` a
/// wrongly-removed edge is silent corruption rather than an error (§3.5) -- the
/// same standing as `HazardKey::Range`. So it is gated on §15.1 #7 and #8:
/// `--turns 3` digests for both canonical pairs over three runs, plus
/// `lfm2-smoke`. **A moved barrier count is engagement proof and never the
/// correctness argument** (§2.4): a wrongly-dropped edge leaves the count lower
/// and entirely plausible, which is the failure mode that rule exists for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ReplayBarriers {
    /// `auto_barrier` fires unconditionally. **The default, and byte-for-byte
    /// what ships.**
    #[default]
    Always,
    /// Suppress candle's barrier where the executor will replay the position
    /// **and** the ICB has encoded ordering covering the same edges.
    ///
    /// Narrower than "the position is covered": only a **non-head member of a
    /// run already in flight** qualifies. See this type's own documentation for
    /// how each of §11.3p's three obstacles is discharged.
    SkipReplayed,
}

/// Why a position is not replayed, kept so the executor can report coverage
/// with the reasons attached rather than as a bare count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Excluded {
    /// Bindings, grid or threadgroup size differed between recorded steps.
    Varies,
    /// The pipeline was not captured, so there is nothing to encode.
    NoPipeline,
    /// The kernel binds its constants inline, so no ICB command can hold it
    /// (§3.7c). Observed as a name without the `_packed` suffix.
    InlineConstants,
}

/// What one recording pass learned about the decode step.
#[derive(Default)]
struct Recording {
    /// Dispatches per recorded step, in submission order.
    steps: Vec<Vec<StepDispatch>>,
    /// The step being accumulated.
    current: Vec<StepDispatch>,
    /// Bindings seen since the last dispatch, i.e. those belonging to the next.
    pending: Vec<BindingRecord>,
    pending_retained: Vec<Buffer>,
    pending_pipeline: Option<ComputePipeline>,
    /// Dispatch count the recorded windows agree on, once two have.
    expected_len: Option<usize>,
    /// Windows dropped for disagreeing with what followed, reported rather than
    /// absorbed: a nonzero count means the caller's step boundary and the
    /// executor's did not line up at first, which is worth knowing even when the
    /// recording recovers.
    discarded: usize,
}

/// The executor's phase. Recording, then replaying, and never back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    /// Forwarding every dispatch and recording what it was.
    Recording,
    /// Replaying the covered positions, forwarding the rest.
    Replaying,
}

/// Coverage, as observed rather than as intended.
#[derive(Clone, Debug, Default)]
pub struct Coverage {
    pub positions: usize,
    pub covered: usize,
    pub varies: usize,
    pub no_pipeline: usize,
    pub inline_constants: usize,
    /// Covered positions per kernel name, so a report can say *what* is replayed
    /// rather than how much.
    pub covered_by_kernel: std::collections::BTreeMap<String, usize>,
    pub excluded_by_kernel: std::collections::BTreeMap<(String, &'static str), usize>,
}

/// Replay a decode step's stable, packed-parameter dispatches from an ICB.
pub struct IcbExecutor {
    inner: Mutex<Inner>,
}

struct Inner {
    phase: Phase,
    /// How many steps to record before deciding what is replayable.
    ///
    /// Two is the minimum that can distinguish "stable" from "seen once", and
    /// more is better: a position that happens to agree between two consecutive
    /// steps but drifts on the third would be wrongly admitted. The default is
    /// higher than two for that reason.
    record_steps: usize,
    recording: Recording,
    plan: Option<Plan>,
    coverage: Coverage,
    /// Positions whose live dispatch stopped matching the recording.
    ///
    /// Zero is the expected value on a steady-state decode, and a nonzero value
    /// means coverage is lower than the plan claims -- reporting the plan's
    /// figure while this is nonzero would overstate what ran.
    stale_positions: usize,
    /// Runs that went stale and are never replayed again.
    poisoned: std::collections::HashSet<usize>,
    /// Windows still to be dropped before recording begins. See
    /// [`IcbExecutor::set_warm_in`].
    warm_in: usize,
    /// Windows dropped by the warm-in, reported beside `discarded`.
    warm_in_skipped: usize,
    /// Positions before this index are inside a run whose range already ran.
    suppress_until: usize,
    /// The run whose range was most recently requested, so a stale member can
    /// name the run to poison.
    run_in_flight: usize,
    /// Dispatch index within the step being executed.
    position: usize,
    /// How far back a barrier scan looks. See [`BarrierScope`].
    barrier_scope: BarrierScope,
    /// Whether candle's own barrier is suppressed at replayed positions.
    /// See [`ReplayBarriers`].
    replay_barriers: ReplayBarriers,
}

/// One maximal run of consecutive covered positions.
///
/// # Why runs, and not one range over the whole step
///
/// An ICB executes its commands as a block wherever the encoder calls
/// `executeCommandsInBuffer`. The covered set is **not** contiguous -- measured
/// on `Sdpa × arena`, 431 covered positions form **31 runs** (12/14/17 long, one
/// group per layer, no singletons; `measurements/issue-115-raw/covered-runs.txt`).
/// So a single execute call would run all 431 at one point in the step, moving
/// each of them across the uncovered dispatches that sat between them.
///
/// That is not a reordering the classical path would tolerate. Dispatches within
/// an encoder overlap and the GPU does not drain between them (§3.5), so the
/// only thing that orders a write against a later read is a barrier candle
/// emitted from the *original* interleaving. Hoisting a dispatch across one is a
/// silent wrong answer rather than an error, which is §3.5's whole point.
///
/// One execute call per run keeps every covered dispatch between the same two
/// uncovered neighbours it had classically, so the barrier structure still
/// applies to it. This is `DESIGN.md` §11.3's range trick used for the reason
/// §11.3 gives -- a computed jump into a pre-encoded list -- rather than for a
/// variable chunk count.
///
/// # Each run carries its own residency set
///
/// `useResource` has to be declared on the encoder that executes the range, so
/// it is paid at every execute call. Declaring the whole plan's buffers at each
/// of them is `runs x |plan|` calls per step -- 32 x 245 = 7840 on LFM2, against
/// a sum over runs of 275 (`DESIGN.md` §11.3w) -- when a run's commands touch
/// only 8 to 12 buffers. The set here is built from the same bindings the run's
/// commands were encoded from, so it covers them by construction; a missed
/// declaration is silent corruption rather than an error (§3.7).
struct Run {
    /// First dispatch position in the run.
    start: usize,
    /// First ICB command index for the run.
    command: usize,
    len: usize,
    /// The ICB and every buffer a command in this run binds.
    range: Arc<IcbRange>,
}

/// The encoded ICB (held by each run's range) and the map from dispatch
/// position to command index.
struct Plan {
    /// `command_of[pos]` is the ICB command index for that dispatch position, or
    /// `None` when the position is not covered.
    command_of: Vec<Option<usize>>,
    /// The covered positions grouped into maximal consecutive runs, in order.
    runs: Vec<Run>,
    /// `run_at[pos]` is the run starting at that position, if one does.
    run_at: Vec<Option<usize>>,
    /// The recording each covered position was encoded from, kept so a live
    /// dispatch can be checked against it before its command is trusted.
    reference: Vec<StepDispatch>,
    /// Distinct buffers bound across the whole plan. Reported, not declared: the
    /// residency a step declares is the sum over runs of each run's own set.
    plan_resident: usize,
    /// Barriers encoded on commands, so the ordering re-expressed inside the ICB
    /// is a reported quantity rather than an assumed one.
    barriers: usize,
}

// SAFETY: `Plan` holds Metal objects, which are internally thread-safe for the
// operations used here (encoding under the executor's mutex, execution on the
// command buffer's thread). The executor's `Mutex` is what actually serialises
// access; these markers only assert that moving the guard between threads is
// sound, which it is because every use goes through that lock.
unsafe impl Send for Plan {}
unsafe impl Sync for Plan {}

impl IcbExecutor {
    /// Record `record_steps` decode steps, then replay what was stable.
    pub fn new(record_steps: usize) -> Arc<IcbExecutor> {
        Self::with_barrier_scope(record_steps, BarrierScope::default())
    }

    /// As [`IcbExecutor::new`], choosing how far back a barrier scan looks.
    ///
    /// The axis exists so the cost of the conservative scan can be *measured*
    /// rather than argued: both arms are built, both are gated on the digest,
    /// and the barrier count moves between them, which is the quantity that
    /// shows the switch engaged (§2.4, §9.2f).
    pub fn with_barrier_scope(
        record_steps: usize,
        barrier_scope: BarrierScope,
    ) -> Arc<IcbExecutor> {
        Self::configured(record_steps, barrier_scope, ReplayBarriers::default())
    }

    /// As [`IcbExecutor::with_barrier_scope`], also choosing whether candle's
    /// own barrier is suppressed at replayed positions ([`ReplayBarriers`],
    /// issue #144).
    ///
    /// The two axes are independent and are deliberately kept so.
    /// [`BarrierScope`] governs the ordering the ICB **encodes**;
    /// [`ReplayBarriers`] governs the ordering candle **still emits** beside it.
    /// §11.3n is what separates them: the two counts are additive rather than
    /// substitutive, so they are two quantities and not one.
    pub fn configured(
        record_steps: usize,
        barrier_scope: BarrierScope,
        replay_barriers: ReplayBarriers,
    ) -> Arc<IcbExecutor> {
        assert!(
            record_steps >= 2,
            "recording fewer than two steps cannot distinguish a stable position from one \
             seen once"
        );
        Arc::new(IcbExecutor {
            inner: Mutex::new(Inner {
                phase: Phase::Recording,
                record_steps,
                recording: Recording::default(),
                plan: None,
                coverage: Coverage::default(),
                stale_positions: 0,
                poisoned: std::collections::HashSet::new(),
                warm_in: 0,
                warm_in_skipped: 0,
                suppress_until: 0,
                run_in_flight: 0,
                position: 0,
                barrier_scope,
                replay_barriers,
            }),
        })
    }

    /// Drop the first `windows` closed steps instead of recording them.
    ///
    /// A recording compares its windows position by position, so it has to
    /// start once the operands are stable -- after the prefill's window, and
    /// after any decode steps the allocator is still learning from: an arena
    /// recorded over the first steps serves the ones after it, and the pooled
    /// params blocks settle a step later still. The harnesses that drive their
    /// own loop install the executor at that point instead; a loop that installs
    /// it before the run begins says how many windows to let pass. Only
    /// meaningful before the recording begins.
    pub fn set_warm_in(&self, windows: usize) {
        self.inner.lock().warm_in = windows;
    }

    /// Windows the warm-in dropped.
    pub fn warm_in_skipped(&self) -> usize {
        self.inner.lock().warm_in_skipped
    }

    /// Close the current step.
    ///
    /// Called by the harness at the token boundary, because nothing inside the
    /// encoder knows where a decode step ends -- candle packs dispatches into
    /// command buffers on its own cadence (`CANDLE_METAL_COMPUTE_PER_BUFFER`,
    /// 14 sessions per token), so an encoder session boundary is not a step
    /// boundary. §9.2f records the same distinction costing two simulations
    /// their credibility.
    pub fn end_step(
        &self,
        device: &crate::kernels::metal::Device,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        // The constants pool hands position N the same buffer every step, which
        // is what makes a packed dispatch's identity stable enough to encode
        // (see `ConstantsPool`). Its cursor is per step, so it resets here --
        // the same boundary the executor's own position counter resets at, and
        // for the same reason.
        if let Some(pool) = crate::kernels::kernels::params::constants_pool() {
            pool.reset();
        }
        let mut inner = self.inner.lock();
        inner.position = 0;
        // Per-step, not cumulative: a run's suppression window belongs to the
        // step that opened it. `poisoned` is deliberately *not* reset -- a run
        // that went stale once is not re-armed, see `dispatch_action`.
        inner.suppress_until = 0;
        inner.run_in_flight = 0;
        match inner.phase {
            Phase::Recording => {
                let step = std::mem::take(&mut inner.recording.current);
                inner.recording.pending.clear();
                inner.recording.pending_retained.clear();
                inner.recording.pending_pipeline = None;

                if inner.warm_in > 0 {
                    inner.warm_in -= 1;
                    inner.warm_in_skipped += 1;
                    return Ok(());
                }

                // The first window an executor sees is partial by construction:
                // it opens wherever the caller installed the executor, which is
                // mid-step unless the caller happened to install exactly at a
                // boundary. Measured on LFM2, installing after the arena's
                // recording steps gives a first window of 554 dispatches and a
                // second of 556 -- the two extra being the `softmax_f32` and
                // `affine_f32` that sampling contributes, which the first window
                // opened too late to see.
                //
                // Discarding a window whose length differs from its predecessor,
                // rather than requiring the caller to install at the right
                // instant, keeps that timing out of the harness. `record_steps`
                // counts *usable* windows, so this cannot silently shorten the
                // recording: a discarded window does not count toward it.
                //
                // A window is only discarded while there is nothing to compare
                // against or it disagrees with what came before. Two consecutive
                // agreeing windows start the recording proper, and a later
                // disagreement is a genuine non-steady-state step -- which
                // `build_plan` refuses rather than aligning, because positions
                // are compared by index and a shifted window pairs unrelated
                // dispatches.
                let len = step.len();
                match inner.recording.expected_len {
                    None => {
                        // Nothing to compare against yet: remember the shape and
                        // keep the window as a candidate.
                        inner.recording.expected_len = Some(len);
                        inner.recording.steps.push(step);
                    }
                    Some(expected) if expected == len => {
                        inner.recording.steps.push(step);
                    }
                    Some(expected) => {
                        // Disagreement. The earlier windows are the suspect ones
                        // -- a partial first window is the common case and a
                        // model that changes shape mid-run is not -- so restart
                        // from this one rather than discarding it.
                        inner.recording.discarded += inner.recording.steps.len();
                        inner.recording.steps.clear();
                        inner.recording.expected_len = Some(len);
                        inner.recording.steps.push(step);
                        let _ = expected;
                    }
                }

                if inner.recording.steps.len() >= inner.record_steps {
                    inner.build_plan(device)?;
                    inner.phase = Phase::Replaying;
                }
            }
            Phase::Replaying => {}
        }
        Ok(())
    }

    /// Coverage as observed, once a plan exists.
    pub fn coverage(&self) -> Coverage {
        self.inner.lock().coverage.clone()
    }

    /// Whether a plan has been built and is being replayed.
    pub fn is_replaying(&self) -> bool {
        self.inner.lock().phase == Phase::Replaying
    }

    /// Positions that fell back to the classical path because the live dispatch
    /// no longer matched what was recorded.
    ///
    /// Zero is the expected value on a steady-state decode. A nonzero count is
    /// not a correctness problem -- the fallback is what keeps it correct -- but
    /// it means coverage is lower than the plan claims, and reporting the plan's
    /// figure while this is nonzero would overstate what ran.
    pub fn stale_positions(&self) -> usize {
        self.inner.lock().stale_positions
    }

    /// How many contiguous runs the covered positions form.
    ///
    /// One `executeCommandsInBuffer` is issued per run, so this is the number of
    /// execute calls a replayed step makes -- and the quantity that says whether
    /// replay is a single computed jump or many. Measured at 31 on
    /// `Sdpa × arena`; see `Run`.
    pub fn runs(&self) -> usize {
        self.inner
            .lock()
            .plan
            .as_ref()
            .map_or(0, |p| p.runs.len())
    }

    /// Runs that went stale and are no longer replayed.
    pub fn poisoned_runs(&self) -> usize {
        self.inner.lock().poisoned.len()
    }

    /// `useResource` declarations one replayed step makes: the sum over runs of
    /// each run's own residency set (`DESIGN.md` §11.3w's `runs x |resident|`,
    /// with `|resident|` per run rather than plan-wide).
    pub fn residency_per_step(&self) -> usize {
        self.inner
            .lock()
            .plan
            .as_ref()
            .map_or(0, |p| p.runs.iter().map(|r| r.range.resident.len()).sum())
    }

    /// Distinct buffers bound across the whole plan -- what every run declared
    /// before residency was per run, and the `|resident|` in §11.3w's 7840.
    pub fn plan_resident(&self) -> usize {
        self.inner
            .lock()
            .plan
            .as_ref()
            .map_or(0, |p| p.plan_resident)
    }

    /// Each run's residency set, as buffer addresses, in run order.
    pub fn run_residency(&self) -> Vec<Vec<usize>> {
        self.inner
            .lock()
            .plan
            .as_ref()
            .map_or(Vec::new(), |p| {
                p.runs
                    .iter()
                    .map(|r| r.range.resident.iter().map(Buffer::raw_addr).collect())
                    .collect()
            })
    }

    /// Barriers encoded onto ICB commands.
    ///
    /// The ordering candle's `auto_barrier` would have emitted between the
    /// replayed dispatches, re-expressed inside the buffer -- see `encode_plan`
    /// for why executing a run without these is silent corruption. Reported
    /// rather than assumed, because it is the quantity whose absence produced
    /// the only wrong answer this work has seen.
    pub fn encoded_barriers(&self) -> usize {
        self.inner
            .lock()
            .plan
            .as_ref()
            .map_or(0, |p| p.barriers)
    }

    /// Recording windows discarded for disagreeing in dispatch count with what
    /// followed.
    ///
    /// One is the expected value: the window an executor opens at install time
    /// is partial unless the caller installed exactly at a step boundary. More
    /// than one, or a nonzero count on a caller that *does* install at a
    /// boundary, means the boundary is in the wrong place.
    pub fn discarded_windows(&self) -> usize {
        self.inner.lock().recording.discarded
    }
}

impl Inner {
    /// Decide what is replayable and encode it.
    fn build_plan(
        &mut self,
        device: &crate::kernels::metal::Device,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        let steps = &self.recording.steps;
        let n = steps[0].len();
        // A step whose dispatch count differs is not a steady-state step, and
        // comparing position-by-position across two different lengths would
        // align unrelated dispatches. Refuse rather than truncate.
        for (i, s) in steps.iter().enumerate() {
            if s.len() != n {
                return Err(crate::kernels::MetalKernelError::FailedToCreateResource(
                    format!(
                        "recorded step {i} has {} dispatches against step 0's {n}; the recording \
                     window includes a non-steady-state step",
                        s.len()
                    ),
                ));
            }
        }

        let mut coverage = Coverage {
            positions: n,
            ..Default::default()
        };
        let mut covered_positions = Vec::new();
        for pos in 0..n {
            let base = &steps[0][pos];
            let name = base.kernel.as_deref().unwrap_or("<unnamed>").to_string();
            let mut exclude = None;

            if base.pipeline.is_none() {
                exclude = Some(Excluded::NoPipeline);
            } else if !is_packed(&name) {
                // §3.7c: no ICB command can carry a `setBytes` scalar, so a
                // classical entry point is unencodable whatever else is true of
                // it. Checked by name because that is what the pipeline records
                // and what `ParamStyle::kernel_name` appends.
                exclude = Some(Excluded::InlineConstants);
            } else if !steps[1..].iter().all(|s| base.replay_compatible(&s[pos])) {
                exclude = Some(Excluded::Varies);
            }

            match exclude {
                None => {
                    coverage.covered += 1;
                    *coverage.covered_by_kernel.entry(name).or_default() += 1;
                    covered_positions.push(pos);
                }
                Some(reason) => {
                    let tag = match reason {
                        Excluded::Varies => {
                            coverage.varies += 1;
                            "varies"
                        }
                        Excluded::NoPipeline => {
                            coverage.no_pipeline += 1;
                            "no-pipeline"
                        }
                        Excluded::InlineConstants => {
                            coverage.inline_constants += 1;
                            "inline-constants"
                        }
                    };
                    *coverage.excluded_by_kernel.entry((name, tag)).or_default() += 1;
                }
            }
        }

        let plan = if covered_positions.is_empty() {
            None
        } else {
            Some(encode_plan(
                device,
                &steps[0],
                &covered_positions,
                n,
                self.barrier_scope,
            )?)
        };
        self.coverage = coverage;
        self.plan = plan;
        Ok(())
    }
}

/// Whether a kernel name is a packed entry point, i.e. whether its constants
/// arrive in a buffer.
fn is_packed(name: &str) -> bool {
    name.ends_with(crate::kernels::kernels::params::PACKED_SUFFIX)
}

/// Build the ICB and encode one command per covered position.
fn encode_plan(
    device: &crate::kernels::metal::Device,
    step: &[StepDispatch],
    covered: &[usize],
    positions: usize,
    barrier_scope: BarrierScope,
) -> Result<Plan, crate::kernels::MetalKernelError> {
    // Trap 1 (§3.7d). Checked here rather than trusted, because the failure is a
    // segfault inside `setComputePipelineState:` a few lines below and there is
    // no error to catch. `set_pipelines_support_icb` has to have been called
    // before any pipeline was built, and by this point they all have been.
    if !crate::kernels::metal::device::pipelines_support_icb() {
        return Err(crate::kernels::MetalKernelError::FailedToCreatePipeline(
            "pipelines were not built with supportIndirectCommandBuffers; encoding one into an \
             ICB segfaults at encode time (DESIGN.md §3.7d). Call \
             set_pipelines_support_icb(true) before the first dispatch"
                .to_string(),
        ));
    }

    let max_bind = covered
        .iter()
        .flat_map(|&p| step[p].bindings.iter().map(|b| b.index + 1))
        .max()
        .unwrap_or(1);

    let desc = MTLIndirectCommandBufferDescriptor::new();
    // Dispatch cannot be mixed with any other command type in one descriptor
    // (`MTLIndirectCommandBuffer.h`), so this is the only type set.
    desc.setCommandTypes(MTLIndirectCommandType::ConcurrentDispatch);
    // Every command carries its own pipeline and its own bindings.
    // `inheritBuffers` would make them all inherit the parent encoder's
    // identically, which cannot express per-dispatch bindings -- the only reason
    // to want them (§3.7c).
    desc.setInheritBuffers(false);
    desc.setInheritPipelineState(false);
    desc.setMaxKernelBufferBindCount(max_bind);

    // SAFETY: the descriptor is fully initialised above and the count is the
    // number of commands about to be encoded.
    let icb = unsafe {
        device
            .as_ref()
            .newIndirectCommandBufferWithDescriptor_maxCommandCount_options(
                &desc,
                covered.len(),
                MTLResourceOptions::empty(),
            )
    }
    .ok_or_else(|| {
        crate::kernels::MetalKernelError::FailedToCreateResource(format!(
            "MTLIndirectCommandBuffer of {} commands",
            covered.len()
        ))
    })?;

    let mut command_of = vec![None; positions];
    let mut run_at = vec![None; positions];
    let mut runs: Vec<RunBuild> = Vec::new();
    let mut reference = Vec::with_capacity(covered.len());
    let mut plan_resident = std::collections::HashSet::new();
    // Barriers encoded on commands, reported so the count is observed rather
    // than inferred -- §9.2f's rule, applied to the quantity this file adds.
    let mut barriers = 0usize;
    // First command index of the run being encoded, so a barrier scan stays
    // inside it. Positions are ascending, so a run continues while the position
    // is one past the previous.
    let mut run_first_cmd = 0usize;
    // The last command in this run that carried a barrier, for
    // `BarrierScope::SinceBarrier`. Per run, not per buffer: a run is executed
    // by its own `executeCommandsInBuffer`, so a barrier in an earlier run says
    // nothing about ordering inside this one.
    let mut last_barrier_cmd: Option<usize> = None;

    for (cmd_index, &pos) in covered.iter().enumerate() {
        // `covered` is ascending and `cmd_index` walks it, so a run's commands
        // are consecutive in the ICB exactly when its positions are consecutive
        // in the step -- which is what lets one `executeCommandsInBuffer` serve
        // a whole run.
        if cmd_index == 0 || covered[cmd_index - 1] + 1 != pos {
            run_first_cmd = cmd_index;
            last_barrier_cmd = None;
            run_at[pos] = Some(runs.len());
            runs.push(RunBuild::open(pos, cmd_index));
        }
        let run = runs
            .last_mut()
            .expect("a run is open: the branch above opens one at cmd_index 0");
        let d = &step[pos];
        // SAFETY: `cmd_index < covered.len()`, the count the ICB was created
        // with -- the loop is over exactly that slice.
        let cmd = unsafe { icb.indirectComputeCommandAtIndex(cmd_index) };
        let pipeline = d
            .pipeline
            .as_ref()
            .expect("covered positions have a pipeline; checked in build_plan");
        cmd.setComputePipelineState(pipeline_raw(pipeline));

        // Re-express the ordering candle's `auto_barrier` emits classically.
        //
        // **This is the correctness centre of the whole file.** An ICB's
        // commands are `MTLIndirectCommandType::ConcurrentDispatch` and the GPU
        // does not drain between them (§3.5), so executing a run as one range
        // runs its members concurrently -- while classically each was separated
        // from its neighbour by a barrier wherever they touched the same bytes.
        // Measured on LFM2: 384 of the 431 covered positions have an earlier
        // write inside their own run that a later member reads, the very first
        // being `rmsnorm_f16` writing what the next `gemv` reads. Replaying
        // without this loses every one of those edges, and §3.5 says a missing
        // barrier is silent corruption rather than an error -- observed, as
        // logits so corrupt that sampling refused them.
        //
        // `setBarrier` is one of the eleven methods §3.7c enumerates on
        // `MTLIndirectComputeCommand`, and it orders this command after every
        // command before it in the buffer.
        //
        // **It is coarser than candle's barrier per command and weaker than it
        // per stream, and the second half is what was not obvious.** #115
        // recorded it as "strictly stronger, so it can only over-order"; that
        // reads the coarseness and misses that the ordering attaches to one
        // command rather than to a point in the stream. Measured: the arm that
        // exploited the assumed transitivity corrupts (`BarrierScope`).
        //
        // Emitted only where a conflict actually exists, using the same
        // read/write/overlap test candle uses, so a run of independent
        // dispatches still runs concurrently.
        //
        // Scanned back to the start of *this run* rather than of the buffer.
        // Commands in an earlier run were executed by an earlier
        // `executeCommandsInBuffer` on the same encoder, and the classical
        // dispatches between the two runs carry candle's own barriers, so the
        // cross-run edge is already ordered by the encoder. Scanning the whole
        // buffer would additionally be wrong in the other direction: command
        // indices are contiguous across runs, so it would order a command
        // against one that does not precede it in this execution at all.
        //
        // O(run length squared) at plan time, once -- not on the per-token path,
        // so §15.2 #10 is not engaged. Runs are 12 to 17 commands.
        // **How far back to scan answers #115's open question 3, and the answer
        // is that the conservative scan is not conservatism.** Both arms emit a
        // barrier only where a conflict exists; they differ in what "already
        // ordered" means, and measuring the difference is what showed the
        // reduced arm to be wrong rather than cheap:
        //
        // * `RunStart` scans to the head of the run. Shipped, and correct.
        // * `SinceBarrier` scans back only to the previous barrier, on the
        //   reasoning that `auto_barrier` clears `prev_outputs` when it fires.
        //   **401 -> 204 barriers, and it corrupts** -- see `BarrierScope`.
        //
        // The two primitives are not the same object, which is the whole of it.
        // `memoryBarrierWithScope` is a *fence in the stream*: it separates
        // everything encoded before it from everything after, so discharging
        // the pending set is sound. `setBarrier` is a *property of one command*:
        // it orders that command against its predecessors and says nothing
        // about its successors, so a barrier on `k` leaves `k+1..` concurrent
        // with everything `k` was ordered against. An edge from `k-2` to `k+3`
        // is therefore **not** discharged by a barrier at `k`.
        //
        // So the 401 are not 401 redundant orderings with a cheaper equivalent
        // available. Scanning the whole run is what the primitive requires.
        //
        // O(run length squared) at plan time, once -- not on the per-token path,
        // so §15.2 #10 is not engaged. Runs are 12 to 17 commands.
        let scan_from = match barrier_scope {
            BarrierScope::RunStart => run_first_cmd,
            // Everything at or before the last barrier is already ordered
            // against this command by that barrier, exactly as candle's
            // `prev_outputs` replacement expresses.
            BarrierScope::SinceBarrier => last_barrier_cmd.map_or(run_first_cmd, |b| b + 1),
        };
        let needs_barrier = covered[scan_from..cmd_index]
            .iter()
            .any(|&earlier| d.conflicts_with(&step[earlier]));
        if needs_barrier {
            cmd.setBarrier();
            barriers += 1;
            last_barrier_cmd = Some(cmd_index);
        }

        for (b, buf) in d.bindings.iter().zip(d.retained.iter()) {
            // SAFETY: the buffer outlives the ICB -- it is retained in the run's
            // residency set below and the plan owns both -- and the index is
            // under `maxKernelBufferBindCount`, which was computed as the
            // maximum over exactly these bindings.
            unsafe { cmd.setKernelBuffer_offset_atIndex(buf.as_ref(), b.offset, b.index) };
            // The run's residency set is derived from the very binding just
            // encoded, so it cannot fall behind the command.
            run.bind(buf);
            plan_resident.insert(buf.raw_addr());
        }
        let tg = MTLSize {
            width: d.threads_per_threadgroup.width,
            height: d.threads_per_threadgroup.height,
            depth: d.threads_per_threadgroup.depth,
        };
        // The two forms are kept distinct rather than normalised. §11.1c: the
        // conversion is lossy, because `dispatchThreads` lets Metal size the
        // final partial threadgroup and a rounded replay would run more threads
        // and report a different `threads_per_threadgroup` inside the kernel.
        // Any threadgroup-wide reduction would then compute a different answer.
        match d.grid {
            Grid::Threadgroups(g) => cmd.concurrentDispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: g.width,
                    height: g.height,
                    depth: g.depth,
                },
                tg,
            ),
            Grid::Threads(g) => cmd.concurrentDispatchThreads_threadsPerThreadgroup(
                MTLSize {
                    width: g.width,
                    height: g.height,
                    depth: g.depth,
                },
                tg,
            ),
        }
        command_of[pos] = Some(cmd_index);
        reference.push(d.clone());
        run.len += 1;
    }

    // Trap 2 (§3.7a): commands the CPU never initialised are undefined
    // behaviour. Here the ICB is created with exactly `covered.len()` commands
    // and all of them are encoded, so there is no tail -- but the reset is
    // written anyway, over the empty range, so that changing the sizing later
    // cannot silently reintroduce the hazard. `icb_unused_tail_must_be_reset`
    // is the test that pins the obligation.
    for i in covered.len()..icb.size() {
        // SAFETY: `i < icb.size()`, the count the ICB was created with.
        unsafe { icb.indirectComputeCommandAtIndex(i) }.reset();
    }

    // The measurement arm: every run declares the whole plan, which is what
    // shipped before residency was per run and the `runs x |plan|` term
    // §11.3w priced. Selected by environment so the two arms are one binary,
    // and read back from the plan (`residency_per_step` against
    // `runs x plan_resident`) rather than from the variable.
    let plan_wide = residency_arm() == ResidencyArm::PlanWide;
    let everything: Vec<Buffer> = if plan_wide {
        let mut seen = std::collections::HashSet::new();
        runs.iter()
            .flat_map(|r| r.resident.iter())
            .filter(|b| seen.insert(b.raw_addr()))
            .cloned()
            .collect()
    } else {
        Vec::new()
    };
    let runs = runs
        .into_iter()
        .map(|r| Run {
            start: r.start,
            command: r.command,
            len: r.len,
            range: Arc::new(IcbRange {
                icb: icb.clone(),
                resident: if plan_wide {
                    everything.clone()
                } else {
                    r.resident
                },
            }),
        })
        .collect();

    Ok(Plan {
        command_of,
        runs,
        run_at,
        reference,
        plan_resident: plan_resident.len(),
        barriers,
    })
}

/// Which residency set each run declares at its `executeCommandsInBuffer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidencyArm {
    /// The buffers the run's own commands bind. The default.
    PerRun,
    /// Every buffer the plan binds, at every run -- the pre-#688 declaration,
    /// kept as the A/B's other arm. `LLOOM_688_RESIDENCY=plan-wide`.
    PlanWide,
}

/// The arm in force, from `LLOOM_688_RESIDENCY`. Read at plan time, once.
pub fn residency_arm() -> ResidencyArm {
    match std::env::var("LLOOM_688_RESIDENCY").as_deref() {
        Ok("plan-wide") => ResidencyArm::PlanWide,
        _ => ResidencyArm::PerRun,
    }
}

/// A run while its commands are being encoded.
struct RunBuild {
    start: usize,
    command: usize,
    len: usize,
    resident: Vec<Buffer>,
    seen: std::collections::HashSet<usize>,
}

impl RunBuild {
    fn open(start: usize, command: usize) -> RunBuild {
        RunBuild {
            start,
            command,
            len: 0,
            resident: Vec::new(),
            seen: std::collections::HashSet::new(),
        }
    }

    /// A command in this run binds `buf`; declare it once per run.
    fn bind(&mut self, buf: &Buffer) {
        if self.seen.insert(buf.raw_addr()) {
            self.resident.push(buf.clone());
        }
    }
}

fn pipeline_raw(p: &ComputePipeline) -> &ProtocolObject<dyn MTLComputePipelineState> {
    p.as_ref()
}

impl Executor for IcbExecutor {
    fn dispatch_action(&self, record: &DispatchRecord) -> DispatchAction {
        let mut inner = self.inner.lock();
        let pos = inner.position;
        inner.position += 1;

        let bindings = std::mem::take(&mut inner.recording.pending);
        let retained = std::mem::take(&mut inner.recording.pending_retained);
        let pipeline = inner.recording.pending_pipeline.take();
        let live = StepDispatch {
            kernel: record.kernel.clone(),
            grid: record.grid,
            threads_per_threadgroup: record.threads_per_threadgroup,
            bindings,
            pipeline,
            retained,
        };

        match inner.phase {
            Phase::Recording => {
                inner.recording.current.push(live);
                // Recording forwards everything: the recorded step has to be a
                // real decode step, or the model's state diverges from the one
                // the replay was derived from.
                DispatchAction::Encode
            }
            Phase::Replaying => {
                let Some(plan) = inner.plan.as_ref() else {
                    return DispatchAction::Encode;
                };
                let Some(Some(cmd)) = plan.command_of.get(pos).copied() else {
                    // Not covered: the classical path runs it, which is what
                    // makes this executor partial rather than wrong.
                    return DispatchAction::Encode;
                };
                // The live dispatch must still be what the command was encoded
                // from. This is the validation half of §11.1a's
                // record-and-validate: replaying a command whose buffers have
                // moved is silent corruption under
                // `HazardTrackingModeUntracked` (§3.5), so a mismatch falls back
                // rather than trusting the plan.
                //
                // **The unit of that decision is the run, not the position.** A
                // run is executed by one `executeCommandsInBuffer` at its head,
                // so by the time a later member of the run is offered, its
                // command has already run. Falling back *there* would encode the
                // dispatch a second time and the position would execute twice --
                // for `badd_f16` that is a residual added twice, which is a
                // plausible wrong answer rather than a crash. So a run is
                // validated as a whole, at its head, and either replays entirely
                // or not at all.
                //
                // Positions inside a run are therefore not re-checked here: the
                // head checked them, and its verdict is recorded in
                // `suppress_until`.
                // Inside a run whose head already executed the range.
                if pos < inner.suppress_until {
                    // Still checked, and the verdict is *reported* rather than
                    // acted on: the command has already run, so encoding it now
                    // would execute this position twice -- for `badd_f16` a
                    // residual added twice, which is a plausible wrong answer
                    // rather than a crash. What the check buys is that a run
                    // going stale mid-flight becomes a nonzero `stale_positions`
                    // the harness must report, instead of silence.
                    //
                    // It is not a race that can be lost quietly: a stale member
                    // disables its run from the *next* step onward (`poisoned`),
                    // and the digest gate is what says whether the step it was
                    // first seen on was already wrong.
                    if !plan.reference[cmd].replay_compatible(&live) {
                        inner.stale_positions += 1;
                        let run_index = inner.run_in_flight;
                        inner.poisoned.insert(run_index);
                    }
                    return DispatchAction::Suppress;
                }
                let Some(run_index) = plan.run_at.get(pos).copied().flatten() else {
                    // Covered, but neither a run head nor inside a live run --
                    // so its run fell back at the head. Encode it classically.
                    let _ = cmd;
                    return DispatchAction::Encode;
                };
                // A run that ever went stale is never replayed again. One
                // position's operands moving means the recording no longer
                // describes this step, and re-arming it per step would rediscover
                // that after executing the range rather than before.
                if inner.poisoned.contains(&run_index) {
                    return DispatchAction::Encode;
                }
                if !plan.reference[cmd].replay_compatible(&live) {
                    inner.stale_positions += 1;
                    inner.poisoned.insert(run_index);
                    return DispatchAction::Encode;
                }
                let run = &plan.runs[run_index];
                let (start, command, len) = (run.start, run.command, run.len);
                let range = Arc::clone(&run.range);
                // Suppress the rest of this run: their commands run as part of
                // the range being requested now.
                inner.suppress_until = start + len;
                inner.run_in_flight = run_index;
                DispatchAction::ExecuteIcb {
                    icb: range,
                    location: command,
                    length: len,
                }
            }
        }
    }

    /// The decision half of [`Self::dispatch_action`], split out so it can be
    /// asked at `auto_barrier` time (issue #144).
    ///
    /// # Why this is a separate method and not a second call to `dispatch_action`
    ///
    /// `dispatch_action` **drains**: three `std::mem::take`s empty `pending`,
    /// `pending_retained` and `pending_pipeline`, which
    /// [`Executor::will_bind_buffer`] spent the whole preceding argument walk
    /// filling. A speculative call would therefore record the *next* dispatch
    /// with no bindings, which is a corrupted recording rather than a
    /// double-advanced cursor. The issue body named the cursor as the blocker;
    /// the drain is the real one, and splitting the decision from it is the
    /// seam.
    ///
    /// This reads `phase`, `suppress_until` and `poisoned` and **touches none of
    /// the pending state**. `position` is likewise only read: the increment
    /// stays in `dispatch_action`, so calling this any number of times per
    /// dispatch -- including zero, which is what `ReplayBarriers::Always` does
    /// -- leaves the executor's cursor exactly where it was.
    ///
    /// # Why the answer is `ReplayedInFlightRun` and not "covered"
    ///
    /// `pos < suppress_until` is precisely *"a non-head member of a run whose
    /// head has already validated and executed its range"*, which is the
    /// predicate `measurements/issue-144-predicate.md` argues and no wider:
    ///
    /// * a **run head** has `pos == suppress_until`'s lower bound at the moment
    ///   it is offered and is therefore never inside the window -- so obstacle 1
    ///   is discharged by construction rather than by an exclusion someone has
    ///   to remember;
    /// * a **gap** position is not in any run's window;
    /// * a **poisoned or fallen-back** run never sets the window, so its members
    ///   are encoded classically and keep their barriers.
    fn disposition(&self) -> Disposition {
        let inner = self.inner.lock();
        if inner.replay_barriers != ReplayBarriers::SkipReplayed {
            return Disposition::Unknown;
        }
        if inner.phase != Phase::Replaying {
            return Disposition::Unknown;
        }
        // Inside a run whose head already executed the range. The head itself is
        // excluded: when a head is offered, `suppress_until` still holds the
        // *previous* run's bound, which never covers it.
        if inner.position < inner.suppress_until {
            return Disposition::ReplayedInFlightRun;
        }
        Disposition::Unknown
    }

    fn will_bind_buffer(&self, index: usize, buffer: &Buffer, offset: usize, is_output: bool) {
        let mut inner = self.inner.lock();
        inner.recording.pending.push(BindingRecord {
            index,
            ptr: buffer.raw_addr(),
            offset,
            len: buffer.length(),
            is_output,
        });
        inner.recording.pending_retained.push(buffer.clone());
    }

    fn will_set_pipeline(&self, pipeline: &ComputePipeline) {
        let mut inner = self.inner.lock();
        inner.recording.pending_pipeline = Some(pipeline.clone());
    }

    fn end_step(
        &self,
        device: &crate::kernels::metal::Device,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        IcbExecutor::end_step(self, device)
    }
}
