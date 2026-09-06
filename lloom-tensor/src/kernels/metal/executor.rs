//! How a dispatch reaches the GPU.
//!
//! `DESIGN.md` §11.1 asks for the classical command-buffer path and an ICB path
//! behind one interface, so the two can be A/B'd on the same workload. This is
//! that interface. The classical path is the default and the correctness bar;
//! it is also what already worked, so the trait is interposed in front of
//! working code rather than duplicating it.
//!
//! # Where the seam is, and why here
//!
//! Every dispatch in the crate funnels through
//! [`ComputeCommandEncoder::dispatch_thread_groups`] or `dispatch_threads` --
//! verified, no kernel reaches `MTLComputeCommandEncoder` directly. So a trait
//! at the point of *submission* sees every dispatch without touching any of the
//! 51 `call_*` entry points or the 57 sites that set pipeline state. That
//! matters more than elegance: a seam that required editing every kernel would
//! have to be re-reviewed against every kernel, and `DESIGN.md` §11.1's "must
//! not regress" is much harder to argue when the diff is that wide.
//!
//! # What an executor may and may not assume
//!
//! An executor sees dispatches in submission order and may record, forward, or
//! defer them. It may **not** change what a dispatch computes: bindings are
//! already on the encoder by the time [`Executor::dispatch`] is called, because
//! `set_params!` binds as it walks its argument list. An executor that wants to
//! replay must therefore capture bindings as they happen, which is what
//! [`Executor::will_bind_buffer`] exists for -- see `IcbFeasibility` below for
//! why no such executor is included here.
use crate::kernels::metal::{Buffer, Commands, ComputePipeline, Device};
use crate::kernels::MetalKernelError;
use objc2_metal::MTLSize;

/// One dispatch, as the executor sees it.
///
/// Deliberately not a borrow of the encoder: an executor that records for
/// replay needs these values to outlive the call, and one that forwards
/// immediately does not care either way.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DispatchRecord {
    /// Kernel function name, when the pipeline carried one (it does whenever
    /// the pipeline came through `Kernels::load_pipeline`).
    pub kernel: Option<std::sync::Arc<str>>,
    /// Threadgroups per grid, or threads per grid for a `dispatch_threads`.
    pub grid: Grid,
    pub threads_per_threadgroup: Size3,
}

/// Which of Metal's two dispatch forms a record came from.
///
/// Kept distinct rather than normalised to threadgroups because the conversion
/// is lossy: `dispatchThreads` lets Metal size the final partial threadgroup,
/// and a replay that rounded it to threadgroups would compute a different
/// number of threads for any grid that is not a multiple of the threadgroup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Grid {
    Threadgroups(Size3),
    Threads(Size3),
}

/// `MTLSize` without the Metal dependency, so a record is comparable and
/// hashable. `MTLSize` is three `NSUInteger`s and derives neither.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Size3 {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}

impl From<MTLSize> for Size3 {
    fn from(s: MTLSize) -> Self {
        Size3 {
            width: s.width,
            height: s.height,
            depth: s.depth,
        }
    }
}

/// How a dispatch is submitted to the GPU.
///
/// Implementors are consulted on the hot path -- 675 times per LFM2 decode
/// token (`DESIGN.md` §11.2) -- so every method has a default that does
/// nothing. An executor that only wants to count dispatches pays for nothing
/// else, and adding a method later does not break implementors.
///
/// `DESIGN.md` §15.2 #9 forbids `dyn Trait` on the per-token path. This trait
/// is deliberately shaped so the default path never dispatches dynamically:
/// [`Classical`] is a zero-sized type whose methods are empty, and
/// [`ExecutorSlot`] resolves to it by a branch on an enum rather than a vtable.
pub trait Executor: Send + Sync {
    /// Called before a dispatch is forwarded to the encoder.
    ///
    /// Returning `false` suppresses the underlying dispatch, which is what a
    /// record-only executor wants. Returning `true` (the default) leaves the
    /// classical behaviour exactly as it was.
    fn dispatch(&self, _record: &DispatchRecord) -> bool {
        true
    }

    /// As [`Self::dispatch`], but able to ask for a pre-encoded range to run
    /// here instead (`DESIGN.md` §11.3).
    ///
    /// The `bool` above can say "do not encode this dispatch" and nothing more,
    /// which is enough for a recorder and not for a replayer: an ICB executes
    /// its commands wherever `executeCommandsInBuffer` is called, so a replaying
    /// executor has to name a *point in the stream* as well as a range. The
    /// covered positions are not one contiguous block -- 431 of them form 31
    /// runs on `Sdpa × arena` (issue #115) -- so the range has to be requested at
    /// the position where the run starts, or the replayed dispatches move across
    /// the uncovered ones between them, and §3.5 makes that a silent wrong
    /// answer rather than an error.
    ///
    /// Defaulted in terms of [`Self::dispatch`] so no existing implementor
    /// changes and the classical path is untouched.
    fn dispatch_action(&self, record: &DispatchRecord) -> DispatchAction {
        if self.dispatch(record) {
            DispatchAction::Encode
        } else {
            DispatchAction::Suppress
        }
    }

    /// What this position's disposition will be, **without advancing anything**.
    ///
    /// Consulted by `auto_barrier` *before* the dispatch is offered, which is
    /// the ordering §11.3p identifies as the obstacle: `auto_barrier` runs at
    /// `encoder.rs:405` and `offer_to_executor` at `:416`, so at fence time the
    /// encoder does not yet know whether the position will be replayed.
    ///
    /// # Why this is not `dispatch_action`
    ///
    /// **`dispatch_action` is not pure.** It advances `position`, `suppress_until`
    /// and `run_in_flight`, and -- the part the issue body understates -- it
    /// **drains** the binding state accumulated by [`Self::will_bind_buffer`]
    /// with three `std::mem::take`s, so a second call sees empty bindings and
    /// would record a dispatch with no operands. Calling it speculatively to
    /// learn the disposition would therefore corrupt the recording, not merely
    /// double-advance a cursor.
    ///
    /// So the seam is to **split the decision from the drain**. The decision
    /// half is a phase check and a `command_of` lookup, neither of which touches
    /// `pending` -- which is why this can be defaulted to
    /// [`Disposition::Unknown`] and cost every existing implementor nothing.
    ///
    /// Defaulted so no existing executor changes behaviour: an implementor that
    /// does not override this is treated as though every position were
    /// classically encoded, which is what they all are.
    fn disposition(&self) -> Disposition {
        Disposition::Unknown
    }

    /// Called when a buffer is bound, before the bind reaches Metal.
    ///
    /// A replaying executor needs this because bindings are applied as
    /// `set_params!` walks its arguments, so by `dispatch` time they are on the
    /// encoder and no longer enumerable.
    fn will_bind_buffer(&self, _index: usize, _buffer: &Buffer, _offset: usize, _is_output: bool) {}

    /// Called when pipeline state is set, before it reaches Metal.
    fn will_set_pipeline(&self, _pipeline: &ComputePipeline) {}

    /// The decode step just encoded is complete; the next dispatch opens a new one.
    ///
    /// Nothing inside the encoder knows where a step ends -- command buffers are
    /// cut on a dispatch count, not on a token -- so the loop that owns the step
    /// says so. A replaying executor keys its recording on this boundary; every
    /// other executor ignores it, and the classical slot never reaches it.
    ///
    /// `commands` is the queue the step was encoded on, for an executor that
    /// still owes the stream work at the boundary: a replaying executor that
    /// validates a run before executing it may reach the step's end with a
    /// validated prefix it has not yet issued, and the only correct place for
    /// that prefix is here, after the step's last dispatch and before the next
    /// step's first.
    fn end_step(&self, _device: &Device, _commands: &Commands) -> Result<(), MetalKernelError> {
        Ok(())
    }
}

/// What an executor will do with the position about to be dispatched, as far as
/// can be known *before* the dispatch is offered.
///
/// This is the decision half of `dispatch_action`, split out so it can be asked
/// at `auto_barrier` time without consuming the pending bindings (see
/// [`Executor::disposition`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Disposition {
    /// Nothing is known, or the position will be encoded classically. The
    /// default, and what every executor that does not override
    /// [`Executor::disposition`] reports.
    #[default]
    Unknown,
    /// This position is a **non-head member of a run being replayed** -- its
    /// ICB command is in the range the run's tail issues once every member has
    /// matched its recording, and the ordering it needs was encoded onto that
    /// command as a `setBarrier`.
    ///
    /// This is the *only* disposition that licenses suppressing candle's
    /// barrier, and the narrowness is the whole of the argument
    /// (`measurements/issue-144-predicate.md`). A run head is deliberately not
    /// included: its scan slice is empty by construction, so candle's fence is
    /// the only thing ordering a gap dispatch into it, and §11.3p records 30 of
    /// the 505 firing at heads with every one required.
    ReplayedInFlightRun,
}

/// What the encoder should do at a dispatch position.
///
/// `Encode` and `Suppress` are the two states [`Executor::dispatch`]'s `bool`
/// could express. `ExecuteIcb` is the third one replay needs: suppress this
/// dispatch *and* run a pre-encoded range at this point, because the commands
/// that replace it and the ones after it must land where the originals were.
pub enum DispatchAction {
    /// Encode the dispatch normally. The classical behaviour.
    Encode,
    /// Do not encode it, and run nothing in its place.
    Suppress,
    /// Do not encode it; execute `length` ICB commands from `location` here.
    ///
    /// The resources every command touches must already be resident on this
    /// encoder: `useResource` is mandatory for an ICB and omitting it is silent
    /// corruption rather than an error (`DESIGN.md` §3.7). Unified memory can
    /// hide the omission, which is why the executor hands the residency set over
    /// rather than leaving it to the caller to remember.
    ExecuteIcb {
        icb: std::sync::Arc<IcbRange>,
        location: usize,
        length: usize,
    },
    /// Execute `length` ICB commands from `location` here, order this dispatch
    /// after them with a barrier, **then encode it classically**.
    ///
    /// The shape a replaying executor needs when a run it is validating stops
    /// matching part-way: the members it has already validated are issued as a
    /// range and the one that did not match runs as the classical dispatch it
    /// is, so no position executes from a recording its live dispatch
    /// contradicts and none executes twice. The barrier is the encoder's, not
    /// the executor's, because the dispatch's own `auto_barrier` ran before the
    /// executor was consulted and cannot order it after commands issued later.
    ExecuteIcbThenEncode {
        icb: std::sync::Arc<IcbRange>,
        location: usize,
        length: usize,
    },
}

/// An ICB and the resources the commands of one run reference.
///
/// Bundled so a residency set cannot be forgotten at the call site that executes
/// a range, and one per run so that site declares the run's own buffers rather
/// than the whole plan's (`DESIGN.md` §11.3w).
pub struct IcbRange {
    pub icb: objc2::rc::Retained<
        objc2::runtime::ProtocolObject<dyn objc2_metal::MTLIndirectCommandBuffer>,
    >,
    pub resident: Vec<Buffer>,
    /// The buffers a command in this run **writes**, by raw address, so the
    /// encoder that executes the range can register them as its outputs. A
    /// later encoder session orders its reads against a buffer's last writer
    /// through that registration (`Commands::prev_ce_outputs`); a range whose
    /// writes it did not carry would leave the reader unfenced under
    /// `HazardTrackingModeUntracked` (§3.5).
    pub written: Vec<usize>,
}

// SAFETY: the Metal objects held here are used under the owning executor's
// mutex, and residency plus execution happen on the encoding thread. These
// markers assert only that the handle may be moved between threads, which is
// sound because every use is serialised by that lock.
unsafe impl Send for IcbRange {}
unsafe impl Sync for IcbRange {}

/// The default: forward every dispatch to the command encoder unchanged.
///
/// Zero-sized, and every method is the trait default, so selecting this costs
/// one branch on an enum discriminant and nothing else. This is what makes
/// "the classical path must not regress" (`DESIGN.md` §11.1) a structural
/// property rather than a measured hope -- there is no other code on it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Classical;

impl Executor for Classical {}

/// Which executor a device is using.
///
/// An enum rather than `Box<dyn Executor>` because of §15.2 #9: the decode path
/// must not pay a virtual call per dispatch. The cost of an executor being
/// installed is a discriminant test that predicts perfectly, and `Classical`
/// falls through to the same code that ran before this module existed.
///
/// Not `#[non_exhaustive]`: an ICB variant will be added here when one is
/// buildable (see [`IcbFeasibility`]), and making that addition visible to
/// downstream matches is the point.
#[derive(Default)]
pub enum ExecutorSlot {
    /// Classical command buffers. The default, and the A/B baseline.
    #[default]
    Classical,
    /// An executor supplied by a caller, for recording or experiment.
    ///
    /// This one *is* a dynamic dispatch and is intended for measurement
    /// harnesses, not for the decode path. It is behind the same enum so that
    /// installing one cannot accidentally become the default.
    Custom(std::sync::Arc<dyn Executor>),
}

impl ExecutorSlot {
    /// True when no executor is installed, so callers can skip building a
    /// [`DispatchRecord`] at all.
    ///
    /// Building a record costs an `Arc<str>` clone and three struct copies.
    /// That is small, but it is per-dispatch on a path where `DESIGN.md` §6.4a
    /// measured the whole per-bind fence probe at 5.1 % of non-GPU time, so
    /// "small and per-dispatch" is exactly the shape worth not paying by
    /// default.
    #[inline(always)]
    pub fn is_classical(&self) -> bool {
        matches!(self, ExecutorSlot::Classical)
    }

    #[inline(always)]
    pub fn dispatch(&self, record: &DispatchRecord) -> bool {
        match self {
            ExecutorSlot::Classical => true,
            ExecutorSlot::Custom(e) => e.dispatch(record),
        }
    }

    #[inline(always)]
    pub fn dispatch_action(&self, record: &DispatchRecord) -> DispatchAction {
        match self {
            ExecutorSlot::Classical => DispatchAction::Encode,
            ExecutorSlot::Custom(e) => e.dispatch_action(record),
        }
    }

    /// What the installed executor will do with this position, without
    /// advancing it (see [`Executor::disposition`]).
    ///
    /// `Classical` answers [`Disposition::Unknown`] **without a virtual call**,
    /// which is what makes issue #144's axis a no-op on the default path by
    /// construction rather than by measurement: with no executor installed there
    /// is no second ordering source, so there is nothing a suppression decision
    /// could be made against.
    #[inline(always)]
    pub fn disposition(&self) -> Disposition {
        match self {
            ExecutorSlot::Classical => Disposition::Unknown,
            ExecutorSlot::Custom(e) => e.disposition(),
        }
    }

    #[inline(always)]
    pub fn will_bind_buffer(&self, index: usize, buffer: &Buffer, offset: usize, is_output: bool) {
        match self {
            ExecutorSlot::Classical => {}
            ExecutorSlot::Custom(e) => e.will_bind_buffer(index, buffer, offset, is_output),
        }
    }

    #[inline(always)]
    pub fn will_set_pipeline(&self, pipeline: &ComputePipeline) {
        match self {
            ExecutorSlot::Classical => {}
            ExecutorSlot::Custom(e) => e.will_set_pipeline(pipeline),
        }
    }

    /// Close the installed executor's step (see [`Executor::end_step`]). Once per
    /// token, not per dispatch, so the branch here is not on §3.3's path.
    #[inline]
    pub fn end_step(&self, device: &Device, commands: &Commands) -> Result<(), MetalKernelError> {
        match self {
            ExecutorSlot::Classical => Ok(()),
            ExecutorSlot::Custom(e) => e.end_step(device, commands),
        }
    }
}

impl std::fmt::Debug for ExecutorSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutorSlot::Classical => f.write_str("Classical"),
            ExecutorSlot::Custom(_) => f.write_str("Custom"),
        }
    }
}

/// Why the ICB path is not implemented here, recorded at the seam it would
/// occupy so it is not rediscovered.
///
/// `DESIGN.md` §11.1a and issue #32 establish that GPU-side ICB encoding works
/// on this machine, including a GPU-written execution range, and treat the
/// remaining obstacles as buffer identity (fixed by issues #21/#23) and
/// `kv_len`-dependent grids. There is a third, and it is not mentioned in
/// either: **an ICB command cannot carry an inline constant.**
///
/// `MTLIndirectComputeCommand` is 11 methods, enumerated from the Objective-C
/// runtime rather than read from a header:
///
/// ```text
/// setComputePipelineState:                        setBarrier
/// setKernelBuffer:offset:atIndex:                 clearBarrier
/// setKernelBuffer:offset:attributeStride:atIndex: setImageblockWidth:height:
/// concurrentDispatchThreadgroups:...              setStageInRegion:
/// concurrentDispatchThreads:...                   setThreadgroupMemoryLength:atIndex:
/// reset
/// ```
///
/// There is no `setBytes` in any form; `respondsToSelector:` is `NO` for both
/// `setKernelBytes:length:atIndex:` and `setBytes:length:atIndex:`. The same
/// holds for the render-side protocol, so this is Metal's design rather than an
/// omission.
///
/// Candle binds inline constants at essentially every dispatch -- 56
/// `set_params!` sites, and `EncoderParam` lowers every primitive to
/// `setBytes`. `call_rms_norm` is the clearest case: 77 of the 675 dispatches
/// in a decode token, passing `length`, `elements_to_sum` and `eps` inline.
/// None of the three is expressible in an ICB command.
///
/// `inheritBuffers` does not rescue it: it makes commands inherit the parent
/// encoder's bindings *identically*, so it cannot express constants that differ
/// per dispatch, which is the only reason to want them.
///
/// So an ICB path requires promoting every inline constant into a device
/// buffer. That is a prerequisite for the ICB path rather than part of it, and
/// it is recorded here and in `DESIGN.md` §11.3a as the next thing that must
/// land before this enum grows an `Icb` variant.
///
/// **Two corrections to that framing, from doing it for one file** (issue #38,
/// `DESIGN.md` §11.3c):
///
/// It is not "changing every kernel signature". Nothing is changed: a `_packed`
/// entry point is *added* beside each existing one, generated from the same
/// body, so the binding style is a compile-tier variant axis in §7.1's sense --
/// the same kind of thing as dtype -- and the classical path is never in
/// question. `reduce.metal`'s 90 variants became 180 that way.
///
/// And "the invasive refactor upstream would decline" was speculation that had
/// been repeated as fact. It originates in §8.1b, about a *build script*
/// emitting generated `.metal` into `OUT_DIR` -- a different change, with a
/// different cost. **No candle maintainer has been asked about this one.**
///
/// # Built 2026-08-26 (issue #115) -- see [`crate::kernels::metal::icb::IcbExecutor`]
///
/// The constants prerequisite above was discharged per family across #38 to #81,
/// and this note is kept because **it was not sufficient, for a reason it does
/// not state**. Every family gained a `_packed` entry point and *nothing
/// selected one*: `candle-core` calls only the classical `call_*`, each passing
/// `ParamStyle::default()`, which derived to `Split`. So an executor installed at
/// this seam would have found every position unencodable, with the whole
/// prerequisite recorded as done.
///
/// Expressible and selected are different properties. `set_default_param_style`
/// is what closes the second, and the executor covers **433 of 556 decode
/// positions** once it is set.
///
/// The other thing this note does not say, and which turned out to matter more:
/// the eleven-method list above establishes what is *absent*, and it equally
/// establishes what is *present*. `setBarrier` is in it, and an ICB whose
/// commands are `ConcurrentDispatch` carries none of candle's ordering (§3.5) --
/// so a replayed run needs those edges re-expressed on the commands themselves
/// or it computes a plausible wrong answer. Read the list in both directions.
#[derive(Debug)]
pub struct IcbFeasibility;
