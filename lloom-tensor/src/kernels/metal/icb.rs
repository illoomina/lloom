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
use crate::kernels::metal::{Buffer, Commands, ComputePipeline};
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
    /// The kernel is on the quarantine list ([`is_quarantined`]): replayed
    /// from an ICB on the model its output diverges from the classical
    /// dispatch, and the cause is open (lloom #732, `measurements/
    /// issue-732-replay-on-the-loop.md` §5). It runs classically until then.
    Quarantined,
}

/// Kernels the plan refuses to replay because a replayed instance was
/// measured computing something other than its classical dispatch on the
/// real model — while the same kernel replays correctly in isolation
/// (`icb_executor_tests::replayed_fused_gemv_epilogues_match_the_classical_ones`
/// and `a_classical_consumer_of_a_replayed_long_command`). lloom #732 found
/// this with the two-weight SwiGLU GEMV (`_swiglu_packed`: the crown's reply
/// diverges from token ~12 with the swiglu positions replayed and is
/// byte-identical with them excluded), reproduced it with a plain `gemv_mul`
/// made three times slower, and could not reproduce it outside the model.
/// Until the ordering hole is found, the positions run classically and the
/// exclusion table says so.
pub fn is_quarantined(name: &str) -> bool {
    name.contains("_swiglu_")
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
    /// Positions excluded by the quarantine list (`is_quarantined`).
    pub quarantined: usize,
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
    /// What went stale, attributed: the first axis of `replay_compatible` on
    /// which each stale position differed, per kernel, and the first few
    /// occurrences with the step they were seen on. A bare count says a run
    /// stopped matching; this says which operand moved and when.
    stale: StaleReport,
    /// Steps closed so far, the clock `stale.first` is dated on.
    step: u64,
    /// Stale positions seen in the step being executed. Nonzero at the step's
    /// close means the recording no longer describes the stream and the plan
    /// is dropped for a new one (`IcbExecutor::end_step`).
    stale_this_step: usize,
    /// Distinct runs that went stale at least once under the current plan.
    stale_runs: std::collections::HashSet<usize>,
    /// Runs issued as a validated prefix because a later member stopped
    /// matching: the members before it ran from the ICB, it and the rest ran
    /// classically, none twice and none from a recording the live dispatch
    /// contradicted.
    prefix_flushes: usize,
    /// Plans dropped and re-recorded after a step went stale.
    rerecordings: usize,
    /// Windows to let pass before a re-recording begins. See
    /// [`IcbExecutor::set_rerecord_warm_in`].
    rerecord_warm_in: usize,
    /// Windows still to be dropped before recording begins. See
    /// [`IcbExecutor::set_warm_in`].
    warm_in: usize,
    /// Windows dropped by the warm-in, reported beside `discarded`.
    warm_in_skipped: usize,
    /// The run whose head has validated and whose members are being validated
    /// as they are offered. Its range is issued at the tail, once every member
    /// has matched, or as a validated prefix at the first member that does not.
    pending: Option<PendingRun>,
    /// Dispatch index within the step being executed.
    position: usize,
    /// How far back a barrier scan looks. See [`BarrierScope`].
    barrier_scope: BarrierScope,
    /// Whether candle's own barrier is suppressed at replayed positions.
    /// See [`ReplayBarriers`].
    replay_barriers: ReplayBarriers,
}

/// Stale positions, attributed rather than counted.
#[derive(Clone, Debug, Default)]
pub struct StaleReport {
    /// Stale positions per axis of `StepDispatch::replay_compatible`.
    pub by_axis: std::collections::BTreeMap<&'static str, usize>,
    /// Stale positions per kernel name.
    pub by_kernel: std::collections::BTreeMap<String, usize>,
    /// The first occurrences, in order: `(step, position, kernel, axis, at_head)`.
    pub first: Vec<(u64, usize, String, &'static str, bool)>,
    /// Steps on which at least one position went stale.
    pub steps: usize,
    last_step: Option<u64>,
}

/// How many occurrences `StaleReport::first` keeps.
const STALE_FIRST_KEPT: usize = 64;

impl StaleReport {
    fn note(&mut self, step: u64, pos: usize, kernel: String, axis: &'static str, at_head: bool) {
        *self.by_axis.entry(axis).or_default() += 1;
        *self.by_kernel.entry(kernel.clone()).or_default() += 1;
        if self.last_step != Some(step) {
            self.last_step = Some(step);
            self.steps += 1;
        }
        if self.first.len() < STALE_FIRST_KEPT {
            self.first.push((step, pos, kernel, axis, at_head));
        }
    }
}

/// The first axis on which `live` stops matching `reference`, in the order
/// `replay_compatible` compares them.
fn stale_axis(reference: &StepDispatch, live: &StepDispatch) -> &'static str {
    if reference.kernel != live.kernel {
        return "kernel";
    }
    if reference.grid != live.grid {
        return "grid";
    }
    if reference.threads_per_threadgroup != live.threads_per_threadgroup {
        return "threadgroup";
    }
    if reference.bindings.len() != live.bindings.len() {
        return "binding-count";
    }
    for (r, l) in reference.bindings.iter().zip(live.bindings.iter()) {
        if r.index != l.index {
            return "binding-index";
        }
        if r.ptr != l.ptr {
            return "binding-identity";
        }
        if r.offset != l.offset {
            return "binding-offset";
        }
        if r.len != l.len {
            return "binding-length";
        }
        if r.is_output != l.is_output {
            return "binding-direction";
        }
    }
    "none"
}

/// A run whose head matched its recording and whose range has not been
/// issued yet.
///
/// # Why a run executes at its tail and not at its head
///
/// The unit of validation is the run, but the unit of *knowledge* is the
/// position: a member's live bindings are known only when it is offered, and
/// an ICB command that has already executed cannot be un-executed. Issuing the
/// range at the head therefore trusted every member on the head's word, and a
/// member whose operand had moved -- a regime change inserting a dispatch, a
/// pooled buffer reissued -- ran from the recording anyway, as §11.3x's
/// *"a plausible wrong answer rather than a crash"*. Issuing at the tail means
/// every command in the range was checked against the dispatch it replaces
/// before it runs. The point in the stream is the same: nothing classical is
/// encoded between a run's head and its tail, so the range lands between the
/// same two uncovered neighbours either way.
struct PendingRun {
    run: usize,
    start: usize,
    command: usize,
    len: usize,
    /// Members matched so far, the head included.
    validated: usize,
    range: Arc<IcbRange>,
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
                stale: StaleReport::default(),
                step: 0,
                stale_this_step: 0,
                stale_runs: std::collections::HashSet::new(),
                prefix_flushes: 0,
                rerecordings: 0,
                rerecord_warm_in: 0,
                warm_in: 0,
                warm_in_skipped: 0,
                pending: None,
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

    /// Windows to let pass before a plan dropped for going stale is recorded
    /// again.
    ///
    /// A step that goes stale is one whose stream stopped matching the
    /// recording -- on LFM2, attention routing to its two-pass kernel at
    /// `kv_len` 1024 inserts dispatches and allocations at every attention
    /// layer -- so the plan is dropped at that step's close and a new one
    /// recorded. An allocator that re-records over the same event (the arena,
    /// `MetalDevice::finish_decode_step`) needs its own steps first, or the new
    /// recording sees pool-served operands and covers nothing; a loop that
    /// runs no such allocator passes 0.
    pub fn set_rerecord_warm_in(&self, windows: usize) {
        self.inner.lock().rerecord_warm_in = windows;
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
        commands: &Commands,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        // A run still pending at the boundary is one the stream ended inside:
        // its validated members were suppressed on the promise of a range that
        // never got issued, and the only correct place for that range is here,
        // after the step's last dispatch and before the next step's first.
        // Taken out from under the lock, because issuing it reaches the queue.
        let unissued = {
            let mut inner = self.inner.lock();
            inner.pending.take()
        };
        if let Some(p) = unissued {
            let encoder = commands.command_encoder()?;
            encoder
                .as_ref()
                .execute_icb_range(&p.range, p.command, p.validated);
            self.inner.lock().prefix_flushes += 1;
        }
        // The constants pool hands position N the same buffer every step, which
        // is what makes a packed dispatch's identity stable enough to encode
        // (see `ConstantsPool`). Its cursor is per step, so it resets here --
        // the same boundary the executor's own position counter resets at, and
        // for the same reason.
        if let Some(pool) = crate::kernels::kernels::params::constants_pool() {
            pool.reset();
        }
        // The recorded steps and the scope, taken out from under the lock when
        // the window is complete, so the device work in `build_plan`
        // (`encode_plan`) runs with `inner` released. `None` unless this step
        // completes the recording.
        let to_build: Option<(Vec<Vec<StepDispatch>>, BarrierScope)> = {
            let mut inner = self.inner.lock();
            inner.position = 0;
            inner.step += 1;
            let stale_this_step = std::mem::take(&mut inner.stale_this_step);
            // A stale step is a stream the recording no longer describes: the
            // plan is dropped and recorded again rather than replayed as a
            // shrinking set of runs that still match. Every position of the
            // stale step itself ran validated -- at a head that fell back, or
            // as a prefix -- so nothing is owed for it; what is owed is
            // coverage on the steps after it.
            if inner.phase == Phase::Replaying && stale_this_step > 0 {
                inner.plan = None;
                inner.phase = Phase::Recording;
                inner.recording = Recording::default();
                inner.stale_runs.clear();
                inner.warm_in = inner.rerecord_warm_in;
                inner.rerecordings += 1;
            }
            match inner.phase {
                Phase::Recording => {
                    let step = std::mem::take(&mut inner.recording.current);
                    inner.recording.pending.clear();
                    inner.recording.pending_retained.clear();
                    inner.recording.pending_pipeline = None;

                    if inner.warm_in > 0 {
                        inner.warm_in -= 1;
                        inner.warm_in_skipped += 1;
                        None
                    } else {
                        Self::accumulate_recording(&mut inner, step)
                    }
                }
                Phase::Replaying => None,
            }
        };

        // Device work with `inner` released; re-lock only to store the result.
        if let Some((steps, scope)) = to_build {
            let (coverage, plan) = Inner::build_plan(device, &steps, scope)?;
            let mut inner = self.inner.lock();
            inner.coverage = coverage;
            inner.plan = plan;
            inner.phase = Phase::Replaying;
        }
        Ok(())
    }

    /// Push `step` into the recording and, if the window is now complete, return
    /// the steps to build a plan from (and the scope), taken out so the caller
    /// can build with the lock released. Runs under the `inner` lock.
    fn accumulate_recording(
        inner: &mut Inner,
        step: Vec<StepDispatch>,
    ) -> Option<(Vec<Vec<StepDispatch>>, BarrierScope)> {
        {
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
                // Window complete: hand the steps and scope back so the caller
                // builds the plan with the lock released, then re-locks to store
                // it and flip to `Replaying`.
                Some((
                    std::mem::take(&mut inner.recording.steps),
                    inner.barrier_scope,
                ))
            } else {
                None
            }
        }
    }

    /// Coverage as observed, once a plan exists.
    pub fn coverage(&self) -> Coverage {
        self.inner.lock().coverage.clone()
    }

    /// Whether a plan has been built and is being replayed.
    pub fn is_replaying(&self) -> bool {
        self.inner.lock().phase == Phase::Replaying
    }

    /// Positions whose live dispatch no longer matched what was recorded.
    ///
    /// Zero is the expected value on a steady-state decode. A nonzero count is
    /// not a correctness problem -- a stale head falls back, a stale member
    /// closes its run at the validated prefix, and no command executes from a
    /// recording its live dispatch contradicts -- but it means the plan no
    /// longer describes the stream, which is why the step that shows one is
    /// the step the plan is dropped and recorded again ([`Self::rerecordings`]).
    pub fn stale_positions(&self) -> usize {
        self.inner.lock().stale_positions
    }

    /// The stale positions, attributed by axis and kernel, with the first
    /// occurrences dated by step.
    pub fn stale_report(&self) -> StaleReport {
        self.inner.lock().stale.clone()
    }

    /// How many contiguous runs the covered positions form.
    ///
    /// One `executeCommandsInBuffer` is issued per run, so this is the number of
    /// execute calls a replayed step makes -- and the quantity that says whether
    /// replay is a single computed jump or many. Measured at 31 on
    /// `Sdpa × arena`; see `Run`.
    pub fn runs(&self) -> usize {
        self.inner.lock().plan.as_ref().map_or(0, |p| p.runs.len())
    }

    /// Distinct runs that went stale at least once under the current plan.
    pub fn stale_runs(&self) -> usize {
        self.inner.lock().stale_runs.len()
    }

    /// Runs issued as a validated prefix because a later member stopped
    /// matching, or because the step ended before the tail.
    pub fn prefix_flushes(&self) -> usize {
        self.inner.lock().prefix_flushes
    }

    /// Plans dropped and recorded again after a step went stale.
    pub fn rerecordings(&self) -> usize {
        self.inner.lock().rerecordings
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
        self.inner.lock().plan.as_ref().map_or(Vec::new(), |p| {
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
        self.inner.lock().plan.as_ref().map_or(0, |p| p.barriers)
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

impl IcbExecutor {
    /// What to do with position `pos` of a replaying step, whose live dispatch
    /// is `live`.
    ///
    /// The validation half of §11.1a's record-and-validate, with the run as the
    /// unit of *issue* and the position as the unit of *check*: a run's head
    /// opens a pending run, each member is checked against its recording as it
    /// is offered, and the range is issued only once its tail has matched.
    /// Replaying a command whose buffers have moved is silent corruption under
    /// `HazardTrackingModeUntracked` (§3.5), so a member that does not match
    /// closes the run at that point -- the validated prefix is issued as a
    /// range, the member itself and everything after it in the run encode
    /// classically -- and no position executes twice, because a suppressed
    /// member's command is exactly the one in the prefix.
    fn replay_action(inner: &mut Inner, pos: usize, live: &StepDispatch) -> DispatchAction {
        let step = inner.step;
        let Some(plan) = inner.plan.as_ref() else {
            return DispatchAction::Encode;
        };

        // Inside a pending run: this position's command is in the range the
        // tail will issue, provided it still matches.
        if let Some(p) = inner.pending.as_mut() {
            let cmd = p.command + (pos - p.start);
            if plan.reference[cmd].replay_compatible(live) {
                p.validated += 1;
                if pos + 1 == p.start + p.len {
                    let p = inner.pending.take().expect("pending was Some");
                    return DispatchAction::ExecuteIcb {
                        icb: p.range,
                        location: p.command,
                        length: p.len,
                    };
                }
                return DispatchAction::Suppress;
            }
            // The member moved. Issue what was validated, fence, and let this
            // dispatch -- whose bindings are on the encoder right now -- run as
            // itself. The rest of the run is offered position by position and
            // encodes classically: no pending run, no `run_at` entry.
            let axis = stale_axis(&plan.reference[cmd], live);
            let kernel = live.kernel.as_deref().unwrap_or("<unnamed>").to_string();
            let p = inner.pending.take().expect("pending was Some");
            inner.stale.note(step, pos, kernel, axis, false);
            inner.stale_positions += 1;
            inner.stale_this_step += 1;
            inner.stale_runs.insert(p.run);
            inner.prefix_flushes += 1;
            return DispatchAction::ExecuteIcbThenEncode {
                icb: p.range,
                location: p.command,
                length: p.validated,
            };
        }

        let Some(cmd) = plan.command_of.get(pos).copied().flatten() else {
            // Not covered: the classical path runs it, which is what makes
            // this executor partial rather than wrong.
            return DispatchAction::Encode;
        };
        let Some(run_index) = plan.run_at.get(pos).copied().flatten() else {
            // Covered, but neither a run head nor inside a pending run -- so
            // its run closed early at a member that moved. Encode it.
            return DispatchAction::Encode;
        };
        if !plan.reference[cmd].replay_compatible(live) {
            let axis = stale_axis(&plan.reference[cmd], live);
            let kernel = live.kernel.as_deref().unwrap_or("<unnamed>").to_string();
            inner.stale.note(step, pos, kernel, axis, true);
            inner.stale_positions += 1;
            inner.stale_this_step += 1;
            inner.stale_runs.insert(run_index);
            return DispatchAction::Encode;
        }
        let run = &plan.runs[run_index];
        let (start, command, len) = (run.start, run.command, run.len);
        let range = Arc::clone(&run.range);
        if len == 1 {
            // A run of one is validated in full at its head.
            return DispatchAction::ExecuteIcb {
                icb: range,
                location: command,
                length: 1,
            };
        }
        inner.pending = Some(PendingRun {
            run: run_index,
            start,
            command,
            len,
            validated: 1,
            range,
        });
        DispatchAction::Suppress
    }
}

impl Inner {
    /// Decide what is replayable and encode it.
    /// Build the replay plan from the recorded steps — the device work
    /// (`encode_plan`, which allocates and encodes an `MTLIndirectCommandBuffer`)
    /// is here, and it is deliberately **not** a `&mut self` method: `end_step`
    /// takes the steps out from under the `inner` lock, drops the lock, calls
    /// this, then re-locks briefly to store the result. Holding `inner` across
    /// `encode_plan` would be a lock held across device work that commits and can
    /// fire a driver-thread completion handler — a latent cross-thread deadlock
    /// (the reason this was hoisted out).
    fn build_plan(
        device: &crate::kernels::metal::Device,
        steps: &[Vec<StepDispatch>],
        barrier_scope: BarrierScope,
    ) -> Result<(Coverage, Option<Plan>), crate::kernels::MetalKernelError> {
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
            } else if is_quarantined(&name) {
                exclude = Some(Excluded::Quarantined);
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
                        Excluded::Quarantined => {
                            coverage.quarantined += 1;
                            "quarantined"
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
                barrier_scope,
            )?)
        };
        Ok((coverage, plan))
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
            run.bind(buf, b.is_output);
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
    let everything_written: Vec<usize> = if plan_wide {
        let mut seen = std::collections::HashSet::new();
        runs.iter()
            .flat_map(|r| r.written.iter().copied())
            .filter(|p| seen.insert(*p))
            .collect()
    } else {
        Vec::new()
    };
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
        .map(|r| {
            let resident = if plan_wide {
                everything.clone()
            } else {
                r.resident
            };
            // Cache the `MTLResource` pointer array ONCE (from the same `resident`
            // the range owns), so `execute_icb_range` declares residency with a
            // single `useResources:count:usage:` instead of a per-buffer loop each
            // replay — and without allocating a fresh array per execute.
            let resident_ptrs = resident
                .iter()
                .map(|b| {
                    let r: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLResource> =
                        objc2::runtime::ProtocolObject::from_ref(b.as_ref());
                    core::ptr::NonNull::from(r)
                })
                .collect();
            Run {
                start: r.start,
                command: r.command,
                len: r.len,
                range: Arc::new(IcbRange {
                    icb: icb.clone(),
                    resident,
                    resident_ptrs,
                    written: if plan_wide {
                        everything_written.clone()
                    } else {
                        r.written
                    },
                }),
            }
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
    /// Buffers a command in this run writes, once each.
    written: Vec<usize>,
    written_seen: std::collections::HashSet<usize>,
}

impl RunBuild {
    fn open(start: usize, command: usize) -> RunBuild {
        RunBuild {
            start,
            command,
            len: 0,
            resident: Vec::new(),
            seen: std::collections::HashSet::new(),
            written: Vec::new(),
            written_seen: std::collections::HashSet::new(),
        }
    }

    /// A command in this run binds `buf`; declare it once per run, and once
    /// among the run's writes if the command writes it.
    fn bind(&mut self, buf: &Buffer, is_output: bool) {
        if self.seen.insert(buf.raw_addr()) {
            self.resident.push(buf.clone());
        }
        if is_output && self.written_seen.insert(buf.raw_addr()) {
            self.written.push(buf.raw_addr());
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
            Phase::Replaying => Self::replay_action(&mut inner, pos, &live),
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
    /// This reads `phase` and the pending run and **touches none of the
    /// pending state**. `position` is likewise only read: the increment stays
    /// in `dispatch_action`, so calling this any number of times per dispatch
    /// -- including zero, which is what `ReplayBarriers::Always` does -- leaves
    /// the executor's cursor exactly where it was.
    ///
    /// # Why the answer is `ReplayedInFlightRun` and not "covered"
    ///
    /// A pending run is one whose head has matched and whose range will be
    /// issued at its tail, so a position inside it is precisely *"a non-head
    /// member of a run being replayed"*, which is the predicate
    /// `measurements/issue-144-predicate.md` argues and no wider:
    ///
    /// * a **run head** is offered while no run is pending -- the previous
    ///   run's window closed at its tail -- so obstacle 1 is discharged by
    ///   construction rather than by an exclusion someone has to remember;
    /// * a **gap** position is not in any run's window;
    /// * a **fallen-back** run never opens a window, so its members are
    ///   encoded classically and keep their barriers.
    ///
    /// The barrier a member's `auto_barrier` would have emitted is deferred
    /// rather than dropped (`encoder.rs`), so when the range is issued at the
    /// tail the first classical dispatch after it still fences against the
    /// run's writes; and a member that fails validation is fenced by the
    /// encoder itself, after the validated prefix and before it
    /// (`DispatchAction::ExecuteIcbThenEncode`).
    fn disposition(&self) -> Disposition {
        let inner = self.inner.lock();
        if inner.replay_barriers != ReplayBarriers::SkipReplayed {
            return Disposition::Unknown;
        }
        if inner.phase != Phase::Replaying {
            return Disposition::Unknown;
        }
        match inner.pending.as_ref() {
            Some(p) if inner.position < p.start + p.len => Disposition::ReplayedInFlightRun,
            _ => Disposition::Unknown,
        }
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
        commands: &Commands,
    ) -> Result<(), crate::kernels::MetalKernelError> {
        IcbExecutor::end_step(self, device, commands)
    }
}
