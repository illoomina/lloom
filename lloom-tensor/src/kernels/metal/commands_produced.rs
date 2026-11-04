//! Work behind an input a host thread or the IO queue produces is committed by
//! its producer, never waited for on the GPU (`docs/invariants.md` §15.1q).
//!
//! A GPU wait is for a signal the GPU produces. A queue wait times out — a
//! Metal 4 queue then releases the work behind it, a Metal 3 command buffer
//! fails — and nothing bounds a host fill or a disk read by that timeout. So
//! the commands behind such an input are recorded as before, and their commit
//! waits: the current recording carries a [`Gate`] of the inputs it reads; a
//! cut while anything is gated moves the recording into the held queue,
//! closed, with the epoch an eager commit would have taken; the producer, once
//! its input exists, submits the released prefix through the owner's own
//! submission transaction (`Commands::produced`). Commit order stays execution
//! order and the epoch timeline is the eager one; only the native entry moves.
//! No host thread waits on the GPU and the GPU waits on no host thread.

use super::{
    CommandBuffer, Commands, CommandsCore, EntryGuard, EntryState, OwnerSubmission,
    PublishedRemovals,
};
use crate::io_terminal::{NativeDeferredIo, NativeIoDependency};
use crate::kernels::MetalKernelError;
use crate::retire::Epoch;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{
    MTLCommandBuffer, MTLCommandQueue, MTLDevice, MTLIOCommandBuffer, MTLSharedEvent,
};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{OnceLock, Weak};

/// Inputs one recording may wait for; more cut an empty recording between.
pub(super) const GATE_INPUTS: usize = 8;
/// Recordings a queue holds back at most. A cut past it parks the encoding
/// thread until producers release some: backpressure at a depth the decode
/// loop's run-ahead never reaches, below Metal 3's 64 live command buffers.
pub(super) const HELD_CAP: usize = 16;

/// How the command stream orders work behind a host- or IO-produced input
/// (`backend.metal.command_buffer.host_input`, read once).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HostInputArm {
    /// The producer commits the dependent recording once the input exists.
    Commit,
    /// The dependent recording is committed at once behind a GPU wait on the
    /// producer's event: the A/B control. A timed-out Metal 4 queue wait
    /// releases the work behind it.
    GpuWait,
}

/// A test's arm for the queues it builds next, over the process's setting:
/// 0 none, 1 commit, 2 gpu-wait.
#[cfg(test)]
pub(super) static TEST_ARM: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

impl HostInputArm {
    pub(super) fn selected() -> Result<Self, MetalKernelError> {
        #[cfg(test)]
        match TEST_ARM.load(Ordering::Acquire) {
            1 => return Ok(HostInputArm::Commit),
            2 => return Ok(HostInputArm::GpuWait),
            _ => {}
        }
        static ARM: OnceLock<HostInputArm> = OnceLock::new();
        if let Some(arm) = ARM.get() {
            return Ok(*arm);
        }
        let tuning = crate::tuning::try_tuning()
            .map_err(|error| MetalKernelError::InvalidInput(error.to_string()))?;
        let arm = match tuning.metal.host_input.as_str() {
            "commit" => HostInputArm::Commit,
            "gpu-wait" => HostInputArm::GpuWait,
            other => {
                return Err(MetalKernelError::InvalidInput(format!(
                    "backend.metal.command_buffer.host_input: `{other}` is not commit or gpu-wait"
                )))
            }
        };
        Ok(*ARM.get_or_init(|| arm))
    }

    pub fn as_str(self) -> &'static str {
        match self {
            HostInputArm::Commit => "commit",
            HostInputArm::GpuWait => "gpu-wait",
        }
    }
}

/// One input a recording reads: `event` raised to `value` by its producer.
pub(super) struct Input {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    value: u64,
    /// When it was gated, in the queue's gating order (0: not a gate's); a
    /// repeated input keeps the earliest.
    seq: u64,
    // None preserves the ordinary event/cancellation contract.
    io: Option<NativeIoDependency>,
}

/// Cold enrolled compute authority. One event certifies ONE Commands queue;
/// cloning/moving a Commands wrapper cannot change the retained core identity.
pub(crate) struct IoComputeAuthority {
    owner: Weak<CommandsCore>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    next: AtomicU64,
}
unsafe impl Send for IoComputeAuthority {}
unsafe impl Sync for IoComputeAuthority {}
impl IoComputeAuthority {
    pub(crate) fn event(&self) -> &ProtocolObject<dyn MTLSharedEvent> {
        &self.event
    }
}

/// How [`Gate::add`] took an input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Added {
    /// A slot of its own, the only one on its event.
    New,
    /// A slot of its own, beside another value of its event.
    BesideItsEvent,
    /// The same value on the same event was gated already.
    Repeat,
}

impl Input {
    fn new(
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<Self, MetalKernelError> {
        // SAFETY: the borrowed event is live; the retain keeps it for the gate.
        let event = unsafe { Retained::retain(std::ptr::from_ref(event).cast_mut()) }
            .ok_or(MetalKernelError::QueueUnavailable("event retain refused"))?;
        Ok(Self {
            event,
            value,
            seq: 0,
            io: None,
        })
    }

    fn from_io(io: NativeIoDependency) -> Result<Self, MetalKernelError> {
        let mut input = Self::new(io.notification(), io.value())?;
        input.io = Some(io);
        Ok(input)
    }

    fn failed_io_cancel(&self) -> Option<Cancel> {
        let io = self.io.as_ref()?;
        if !io.failed() {
            return None;
        }
        // Taken while applying under owner exclusion, NOT an asynchronously
        // captured gating snapshot. Ordinary Cancel remains unchanged.
        Some(Cancel {
            event: self.event.clone(),
            produced: io.successful(),
            upto: u64::MAX,
            gated_through: u64::MAX,
            reason: "validated IO generation failed before dependent submission",
        })
    }

    fn produced(&self) -> bool {
        self.io.as_ref().map_or_else(
            || self.event.signaledValue() >= self.value,
            NativeIoDependency::ready,
        )
    }

    fn is_on(&self, event: &ProtocolObject<dyn MTLSharedEvent>) -> bool {
        std::ptr::eq(&*self.event, event)
    }
}

/// The inputs a recording reads that were not produced when it was recorded.
#[derive(Default)]
pub(super) struct Gate {
    inputs: [Option<Input>; GATE_INPUTS],
}

impl Gate {
    pub(super) fn is_empty(&self) -> bool {
        self.inputs.iter().all(Option::is_none)
    }

    /// Every input produced: the recording may be committed.
    pub(super) fn is_open(&self) -> bool {
        self.inputs.iter().flatten().all(Input::produced)
    }

    fn is_full(&self) -> bool {
        self.inputs.iter().all(Option::is_some)
    }

    /// Add `input`. A repeat of a gated `(event, value)` keeps the earlier
    /// gating; another value of a gated event takes a slot of its own, so a
    /// cancellation through any one of them finds it (`covered_by`).
    fn add(&mut self, input: Input) -> Result<Added, Input> {
        let mut beside = false;
        for held in self.inputs.iter().flatten() {
            if held.is_on(&input.event) {
                if held.value == input.value {
                    return Ok(Added::Repeat);
                }
                beside = true;
            }
        }
        match self.inputs.iter_mut().find(|slot| slot.is_none()) {
            Some(slot) => {
                *slot = Some(input);
                Ok(if beside {
                    Added::BesideItsEvent
                } else {
                    Added::New
                })
            }
            None => Err(input),
        }
    }

    /// An input the cancellation covers: gated before it was asked for, on
    /// its event, above the value it found produced and at or below its
    /// bound. A cancellation applied late takes nothing gated since.
    fn covered_by(&self, cancel: &Cancel) -> bool {
        self.inputs.iter().flatten().any(|input| {
            input.seq != 0
                && input.seq <= cancel.gated_through
                && input.is_on(&cancel.event)
                && input.value > cancel.produced
                && input.value <= cancel.upto
        })
    }

    /// An input is an IO load whose bytes are not validated yet.
    fn waits_for_unvalidated_io(&self) -> bool {
        self.inputs
            .iter()
            .flatten()
            .any(|input| input.io.as_ref().is_some_and(|io| !io.ready()))
    }

    fn highest_on(&self, event: &ProtocolObject<dyn MTLSharedEvent>) -> u64 {
        self.inputs
            .iter()
            .flatten()
            .filter(|input| input.is_on(event))
            .map(|input| input.value)
            .max()
            .unwrap_or(0)
    }
}

/// What a held recording's commit leaves to run after it.
pub(super) enum AfterCommit {
    /// Already rooted Prepared custody; cancellation drops/refuses, never rehomes.
    ValidatedIo(NativeDeferredIo),
    /// An IO command buffer that waits on `signal`, encoded in the recording:
    /// committed once that signal is on the GPU, so the IO queue never waits
    /// on a signal nothing will run. `rehomed` once the recording carrying
    /// the signal was cancelled: the signal is then issued on the queue
    /// itself, after everything committed before it, ahead of the IO buffer.
    Io {
        io: Retained<ProtocolObject<dyn MTLIOCommandBuffer>>,
        signal: Input,
        rehomed: bool,
    },
}

/// Fill `spare` with an emptied after-commit list for every recording the
/// held queue can hold and the current one, each with room for a
/// point-of-use batch's IO buffers: built once, with the queue.
pub(super) fn spare_after(spare: &mut Vec<Vec<AfterCommit>>) -> Result<(), MetalKernelError> {
    let refused = |_| MetalKernelError::QueueUnavailable("held IO storage refused");
    spare.try_reserve_exact(HELD_CAP + 1).map_err(refused)?;
    let lanes = crate::kernels::metal::io_queue::io_lanes() + 1;
    for _ in 0..HELD_CAP + 1 {
        let mut list = Vec::new();
        list.try_reserve_exact(lanes).map_err(refused)?;
        spare.push(list);
    }
    Ok(())
}

/// Give an emptied after-commit list back for the next held recording,
/// keeping its storage; one beyond what was built is dropped.
fn recycle_after(spare: &mut Vec<Vec<AfterCommit>>, mut list: Vec<AfterCommit>) {
    list.clear();
    if list.capacity() > 0 && spare.len() < spare.capacity() {
        spare.push(list);
    }
}

/// A recording closed at a cut while an input it reads, or one a recording
/// ahead of it reads, was not yet produced.
pub(super) struct Held {
    pub(super) cb: CommandBuffer,
    pub(super) gate: Gate,
    /// The epoch its cut took: the one an eager commit would have.
    pub(super) epoch: Epoch,
    pub(super) after: Vec<AfterCommit>,
    /// When the cut held it (the held time, `ProducedStats::held_us`).
    pub(super) at: std::time::Instant,
}

/// Epochs held recordings took at their cut and gave up to a cancellation:
/// never committed, so a later commit's completion completes them. The
/// completion hook's n-th run completes epoch n (`completion_hook.rs`), so
/// how many runs a commit carries decides which epochs its completion marks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Owed {
    count: u32,
    /// The epoch the next hook run completes: the runs every owner commit
    /// so far carried. Every epoch below it is committed or paid.
    flagged: u64,
}

impl Owed {
    /// The most epochs owed at once: a commit's hook runs, `1 + count`, are
    /// a `u32`. Past it a cancellation is refused and the queue fails.
    pub(super) const MAX_OWED: u32 = u32::MAX - 1;

    /// A held recording that took `epoch` was cancelled before its commit.
    pub(super) fn cancel(&mut self, epoch: Epoch) -> Result<(), MetalKernelError> {
        debug_assert!(epoch.0 >= self.flagged, "a flagged epoch was cancelled");
        if self.count >= Self::MAX_OWED {
            return Err(MetalKernelError::QueueUnavailable(
                "owed epoch bound reached",
            ));
        }
        self.count += 1;
        Ok(())
    }

    /// The hook runs a commit carries. With no other recording live and
    /// uncommitted (`others_live == 0`: the last held one, or the current one
    /// with nothing held) it carries every owed epoch. A held front with live
    /// work behind it carries the holes below its reserved `epoch` and its
    /// own: every epoch from the frontier up to it is cancelled, and none
    /// above it is marked. Either way its completion marks only epochs that
    /// are complete or cancelled. `epoch` is `None` for a commit that takes
    /// its epoch at the commit.
    pub(super) fn runs_for(&self, epoch: Option<Epoch>, others_live: usize) -> u32 {
        match (epoch, others_live) {
            (_, 0) => 1 + self.count,
            (Some(e), _) => {
                let holes = e.0.saturating_sub(self.flagged);
                debug_assert!(holes <= u64::from(self.count), "a hole that is not owed");
                1 + holes.min(u64::from(self.count)) as u32
            }
            // The current recording commits only with nothing held.
            (None, _) => 1,
        }
    }

    /// A commit carrying `runs` was accepted. Precondition: `runs` came from
    /// [`Self::runs_for`] on this state, for that commit. The reserved epochs
    /// are then the frontier plus the owed and the held ones (`S = F + C + H`),
    /// so `flagged + runs` stays at or below them and the settle is exact. The
    /// saturating arithmetic only keeps a broken precondition from wrapping;
    /// it cannot correct hook runs already installed on the commit.
    pub(super) fn settle(&mut self, runs: u32) {
        let paid = runs.saturating_sub(1);
        debug_assert!(runs >= 1 && paid <= self.count);
        self.count = self.count.saturating_sub(paid);
        self.flagged = self.flagged.saturating_add(u64::from(runs));
    }

    /// The final owner abandoned its current recording, so no commit will
    /// carry what is owed: the epochs `[flagged, flagged + count)` complete
    /// by accounting instead, published by `publish`
    /// (`CompletionHook::abandon`). Nothing is held then, and the abandoned
    /// recording took no epoch, so the range ends at the reserved epochs.
    /// Settled only when `publish` accepts it, as a commit's runs are.
    pub(super) fn abandon(
        &mut self,
        publish: impl FnOnce(u64, u64) -> Result<(), MetalKernelError>,
    ) -> Result<(), MetalKernelError> {
        let from = self.flagged;
        let through =
            from.checked_add(u64::from(self.count))
                .ok_or(MetalKernelError::QueueUnavailable(
                    "owed epochs past the epoch range",
                ))?;
        publish(from, through)?;
        self.count = 0;
        self.flagged = through;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn count(&self) -> u32 {
        self.count
    }

    /// A state no short test reaches by cancelling: near the bound.
    #[cfg(test)]
    pub(super) fn for_test(count: u32, flagged: u64) -> Self {
        Self { count, flagged }
    }

    #[cfg(test)]
    pub(super) fn flagged(&self) -> u64 {
        self.flagged
    }
}

/// Counts of what the held queue did, since the queue was built.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProducedStats {
    /// Recordings held back right now.
    pub held_now: usize,
    /// Recordings ever held back at a cut.
    pub held: u64,
    /// Held recordings their producers released and committed.
    pub released: u64,
    /// Recordings cancelled before their commit, the current one included.
    pub cancelled: u64,
    /// Inputs produced after the work waiting for them was cancelled.
    pub late_after_cancel: u64,
    /// Times a held cut parked the encoding thread (a full held queue, or no
    /// free Metal 4 slot behind held work) until producers released some.
    pub parks: u64,
    /// Of `released`, the ones a producer's own call committed (a host
    /// producer's `produced`, or an IO event's notification); the rest the
    /// owner committed at a cut, a park or a wait.
    pub released_by_producer: u64,
    /// Summed time from a recording's hold to its commit, µs.
    pub held_us: u64,
    /// IO command buffers committed after a held recording instead of at
    /// their issue (the recording carrying their compute signal was held).
    pub io_deferred: u64,
    /// Inputs gated beside another value of their event in one recording:
    /// each holds a gate slot of its own.
    pub beside_its_event: u64,
    /// Summed time producers spent at the state lock — a try, and the
    /// release when it was free — µs. No producer waits for it.
    pub producer_lock_us: u64,
    /// Times a producer found the state lock held and left its release to
    /// the holder: producers never wait for the lock.
    pub delegated: u64,
    /// Times a cancellation found the state lock held and left itself to the
    /// holder: a cancellation never waits for the lock either.
    pub cancels_delegated: u64,
}

/// Part of every state-lock acquisition's custody, dropped after the lock:
/// runs the release or cancellation that a producer or a canceller, finding
/// the lock held, left to this holder.
pub(super) struct AfterUnlock<'a>(pub(super) &'a CommandsCore);

impl Drop for AfterUnlock<'_> {
    fn drop(&mut self) {
        let core = self.0;
        // `live` changes only under the lock: zero means nothing is held or
        // gated, so nothing left pending can be waiting or covered.
        if core.produced.live.load(Ordering::Relaxed) == 0 || std::thread::panicking() {
            return;
        }
        if let Err(error) = core.run_pending(Pending::Holder) {
            core.failure
                .record(format!("held work its input released was refused: {error}"));
        }
    }
}

/// A cancellation's first refusal. Dropped with one still noted, on any
/// exit, an early `?` included, it fails the queue: a recording whose epoch
/// could not be owed, or whose cancellation could not finish, must never
/// leave a queue that reads as running.
struct Refusal<'a> {
    failure: &'a super::CommandBufferFailure,
    message: &'static str,
    first: Option<MetalKernelError>,
}

impl<'a> Refusal<'a> {
    fn new(failure: &'a super::CommandBufferFailure, message: &'static str) -> Self {
        Self {
            failure,
            message,
            first: None,
        }
    }

    fn note(&mut self, error: MetalKernelError) {
        self.first.get_or_insert(error);
    }

    /// The normal exit: the first refusal, latched, or none.
    fn into_result(mut self) -> Result<(), MetalKernelError> {
        match self.first.take() {
            Some(error) => {
                self.failure.record_static(self.message);
                Err(error)
            }
            None => Ok(()),
        }
    }
}

impl Drop for Refusal<'_> {
    fn drop(&mut self) {
        if self.first.is_some() {
            self.failure.record_static(self.message);
        }
    }
}

/// Who runs what is left pending, for the delegation counts.
#[derive(Clone, Copy)]
enum Pending {
    Producer,
    Canceller,
    Holder,
}

/// A cancellation asked for and not yet applied: every held recording with
/// an input gated by then (`gated_through`) on `event`, above `produced`
/// (the event's value when it was asked for) and at or below `upto`,
/// everything behind it, and the current recording.
pub(super) struct Cancel {
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    produced: u64,
    upto: u64,
    gated_through: u64,
    reason: &'static str,
}

/// What applying one cancellation did: every recording it covers abandoned,
/// or its close of the current encoder refused only for now and the
/// cancellation is to be kept and applied again (`cancel_locked`).
enum Applied {
    Done,
    Kept(MetalKernelError),
}

/// The block an event's notification runs: the event and the value reached.
type NotifyBlock = block2::RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLSharedEvent>>, u64)>;

struct Listener {
    listener: Retained<objc2_metal::MTLSharedEventListener>,
    block: NotifyBlock,
}

/// The producer-side half of the held queue, beside the state lock.
pub(super) struct ProducedState {
    me: Weak<CommandsCore>,
    /// Held recordings plus a gated current one: a producer with nothing
    /// held returns without the state lock.
    live: AtomicUsize,
    /// A waiter on a held recording parks here; producers and cancellation
    /// wake it.
    pub(super) wake: crate::completion_wake::Wake,
    held: AtomicU64,
    released: AtomicU64,
    cancelled: AtomicU64,
    late_after_cancel: AtomicU64,
    parks: AtomicU64,
    released_by_producer: AtomicU64,
    held_us: AtomicU64,
    io_deferred: AtomicU64,
    producer_lock_us: AtomicU64,
    delegated: AtomicU64,
    /// A producer found the state lock held: the holder runs the release
    /// once it unlocks (`AfterUnlock`).
    release_pending: AtomicBool,
    // Exceptional terminal notification, not a success-path full gate scan.
    io_failure_pending: AtomicBool,
    /// A cancellation found the state lock held: whichever holder looks next
    /// applies it, before any release (`Commands::cancel_produced`).
    cancel_pending: AtomicBool,
    cancels: parking_lot::Mutex<Vec<Cancel>>,
    cancels_delegated: AtomicU64,
    beside_its_event: AtomicU64,
    /// Inputs gated so far: an input's `seq`, a cancellation's bound.
    gated: AtomicU64,
    late_logged: AtomicBool,
    /// The event and highest value of the last cancellation.
    cancelled_event: AtomicUsize,
    cancelled_upto: AtomicU64,
    /// Why the last cancellation happened: a waiter on a cancelled
    /// recording reads it in its error.
    cancel_reason: parking_lot::Mutex<&'static str>,
    /// Wakes the queue for an input the GPU or the IO queue produces.
    listener: OnceLock<Listener>,
    // IO-only callback: ordinary host/event notifications keep their old path.
    io_notify: OnceLock<NotifyBlock>,
}

impl ProducedState {
    pub(super) fn new(me: Weak<CommandsCore>) -> Self {
        Self {
            me,
            live: AtomicUsize::new(0),
            wake: crate::completion_wake::Wake::default(),
            held: AtomicU64::new(0),
            released: AtomicU64::new(0),
            cancelled: AtomicU64::new(0),
            late_after_cancel: AtomicU64::new(0),
            parks: AtomicU64::new(0),
            released_by_producer: AtomicU64::new(0),
            held_us: AtomicU64::new(0),
            io_deferred: AtomicU64::new(0),
            producer_lock_us: AtomicU64::new(0),
            delegated: AtomicU64::new(0),
            release_pending: AtomicBool::new(false),
            io_failure_pending: AtomicBool::new(false),
            cancel_pending: AtomicBool::new(false),
            cancels: parking_lot::Mutex::new(Vec::new()),
            cancels_delegated: AtomicU64::new(0),
            beside_its_event: AtomicU64::new(0),
            gated: AtomicU64::new(0),
            late_logged: AtomicBool::new(false),
            cancelled_event: AtomicUsize::new(0),
            cancelled_upto: AtomicU64::new(0),
            cancel_reason: parking_lot::Mutex::new("cancelled before its input was produced"),
            listener: OnceLock::new(),
            io_notify: OnceLock::new(),
        }
    }

    /// The error a wait on a cancelled recording returns.
    pub(super) fn cancelled_error(&self) -> MetalKernelError {
        MetalKernelError::QueueUnavailable(*self.cancel_reason.lock())
    }

    /// A recording gated a second value of one event: counted, since each
    /// value holds a slot a full gate cuts for. Called under the state lock,
    /// so it only counts; the count is printed at the run's report.
    fn note_beside_its_event(&self) {
        self.beside_its_event.fetch_add(1, Ordering::Relaxed);
    }

    #[cfg(test)]
    pub(super) fn late_logged_for_test(&self) -> bool {
        self.late_logged.load(Ordering::Relaxed)
    }

    #[cfg(test)]
    pub(super) fn me_for_test(&self) -> Weak<CommandsCore> {
        self.me.clone()
    }

    pub(super) fn io_notification(&self) -> &NotifyBlock {
        self.io_notify.get_or_init(|| {
            let me = self.me.clone();
            block2::RcBlock::new(
                move |event: NonNull<ProtocolObject<dyn MTLSharedEvent>>, _threshold: u64| {
                    if let Some(core) = me.upgrade() {
                        // SAFETY: Metal supplies the retained registered event.
                        let event = unsafe { event.as_ref() };
                        // A callback argument need not report a later terminal
                        // signal. Read THIS IO notice, never infer data validity
                        // from it: produced_on checks the retained typed gates.
                        if let Err(error) = core.produced_on(event, event.signaledValue()) {
                            core.failure.record(format!(
                                "an IO notification could not release its held work: {error}"
                            ));
                        }
                    }
                },
            )
        })
    }

    fn listener(&self) -> &Listener {
        self.listener.get_or_init(|| {
            let me = self.me.clone();
            let block = block2::RcBlock::new(
                move |event: NonNull<ProtocolObject<dyn MTLSharedEvent>>, value: u64| {
                    if let Some(core) = me.upgrade() {
                        // SAFETY: Metal passes the live event this block was
                        // registered on.
                        let event = unsafe { event.as_ref() };
                        if let Err(error) = core.produced_on(event, value) {
                            core.failure.record(format!(
                                "a produced input could not release its held work: {error}"
                            ));
                        }
                    }
                },
            );
            Listener {
                listener: objc2_metal::MTLSharedEventListener::new(),
                block,
            }
        })
    }
}

impl CommandsCore {
    /// A cancellation is left pending for the state lock's holder.
    pub(super) fn produced_cancel_pending(&self) -> bool {
        self.produced.cancel_pending.load(Ordering::SeqCst)
    }

    /// Publish how much is held, then order the publication before the
    /// caller's next look at the inputs: a producer that raised its event
    /// before the look either sees this count or is seen by the look.
    pub(super) fn held_live_update(&self, state: &EntryState) {
        self.produced.live.store(
            state.held.len() + usize::from(!state.gate.is_empty()),
            Ordering::SeqCst,
        );
        std::sync::atomic::fence(Ordering::SeqCst);
    }

    /// Commit, in order, every held recording whose inputs all exist, and
    /// open the current recording's gate once nothing is held ahead of it.
    /// A cancellation left pending is applied first, and looked for again
    /// after every gate found open, so one asked for before its event moved
    /// is applied before the move can release what it covers. Each commit is
    /// the owner's submission transaction, with the epoch the recording's cut
    /// took; residency is published once per acquisition. A caller that goes
    /// on to use the current recording checks `current_abandoned` after.
    pub(super) fn release_produced_locked(
        &self,
        state: &mut EntryGuard<'_>,
    ) -> Result<usize, MetalKernelError> {
        let mut released = 0usize;
        let mut opened = false;
        let result = 'release: loop {
            if self.produced.cancel_pending.load(Ordering::SeqCst) {
                if let Err(error) = self.apply_cancels_locked(state) {
                    break Err(error);
                }
            }
            if self
                .produced
                .io_failure_pending
                .swap(false, Ordering::SeqCst)
            {
                // Bounded CURRENT/HELD gates only, not retained IO/history. Each
                // failed generation cancels its first dependent plus suffix.
                loop {
                    let failure = state
                        .held
                        .iter()
                        .flat_map(|h| h.gate.inputs.iter())
                        .chain(state.gate.inputs.iter())
                        .flatten()
                        .find_map(Input::failed_io_cancel);
                    let Some(cancel) = failure else {
                        break;
                    };
                    match self.cancel_locked(state, &cancel) {
                        Ok(Applied::Done) => {}
                        // The validated gate never opens on a failed
                        // generation, so the dependent work stays held.
                        Ok(Applied::Kept(error)) | Err(error) => break 'release Err(error),
                    }
                }
            }
            let head = state.held.front().is_some_and(|held| held.gate.is_open());
            let current =
                !head && state.held.is_empty() && !state.gate.is_empty() && state.gate.is_open();
            if !head && !current {
                break Ok(());
            }
            // The gate's look, then the flag: a cancellation asked for before
            // its event was raised is seen here. The producer wrote its input
            // before it raised the event.
            std::sync::atomic::fence(Ordering::SeqCst);
            if self.produced.cancel_pending.load(Ordering::SeqCst) {
                continue;
            }
            if current {
                state.gate = Gate::default();
                opened = true;
                break Ok(());
            }
            let (entry, custody) = state.parts_mut();
            let publish = !custody.residency_published;
            let removals: Option<&mut Option<PublishedRemovals>> = if publish {
                match custody.removals.reserve() {
                    Some(slot) => Some(slot),
                    None => {
                        break Err(MetalKernelError::QueueUnavailable(
                            "published residency custody bound (2) exhausted",
                        ))
                    }
                }
            } else {
                None
            };
            let held = entry.held.front().expect("checked above");
            let authority = OwnerSubmission {
                buffer: &held.cb,
                clock: &self.clock,
            };
            let epoch = held.epoch;
            let runs = entry
                .owed_epochs
                .runs_for(Some(epoch), entry.held.len() - 1);
            match self.submit_owned_recording(
                authority,
                removals,
                false,
                None,
                None,
                Some(epoch),
                runs,
            ) {
                Ok(_) => {
                    entry.owed_epochs.settle(runs);
                    custody.residency_published |= publish;
                    let held = entry.held.pop_front().expect("checked above");
                    self.produced.held_us.fetch_add(
                        held.at.elapsed().as_micros().min(u64::MAX as u128) as u64,
                        Ordering::Relaxed,
                    );
                    entry.in_flight.push(held.cb);
                    released += 1;
                    let mut after = held.after;
                    let ran = self.run_after_commit(&mut after);
                    recycle_after(&mut entry.spare_after, after);
                    if let Err(error) = ran {
                        break Err(error);
                    }
                }
                Err(error) => break Err(error),
            }
        };
        self.held_live_update(state);
        if released > 0 {
            self.produced
                .released
                .fetch_add(released as u64, Ordering::Relaxed);
        }
        if released > 0 || opened {
            self.produced.wake.notify();
        }
        result.map(|()| released)
    }

    /// Run, in order, what a commit left to run after it, emptying `after`.
    fn run_after_commit(&self, after: &mut Vec<AfterCommit>) -> Result<(), MetalKernelError> {
        for action in after.drain(..) {
            match action {
                AfterCommit::ValidatedIo(io) => {
                    io.commit().map_err(|_| {
                        MetalKernelError::QueueUnavailable("validated IO handoff refused")
                    })?;
                }
                AfterCommit::Io {
                    io,
                    signal,
                    rehomed,
                } => {
                    if rehomed {
                        self.signal_after_committed(&signal.event, signal.value)?;
                    }
                    io.commit();
                }
            }
        }
        Ok(())
    }

    /// Signal `event` to `value` after every command buffer committed so far.
    fn signal_after_committed(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        if let Some(queue) = &self.modern_queue {
            return queue.signal_after_committed(event, value);
        }
        // A plain retaining buffer outside the engine's tracking: the queue
        // runs it after every buffer committed before it.
        let raw = self
            .command_queue
            .commandBuffer()
            .ok_or_else(|| MetalKernelError::FailedToCreateResource("CommandBuffer".into()))?;
        raw.encodeSignalEvent_value(ProtocolObject::from_ref(event), value);
        raw.commit();
        Ok(())
    }

    /// A producer raised `event` to `value`: commit what that released.
    /// Never waits for the state lock. An owner parked holding it (a Metal 4
    /// slot, a full held queue) is woken first and submits the released work
    /// itself; any other holder runs the release once it unlocks. A producer
    /// waiting for the lock would hold back every input its thread or its
    /// event listener delivers next, the one a parked owner waits for among
    /// them (`docs/invariants.md` §15.1q).
    pub(super) fn produced_on(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<usize, MetalKernelError> {
        if value == u64::MAX {
            self.produced
                .io_failure_pending
                .store(true, Ordering::SeqCst);
        }
        std::sync::atomic::fence(Ordering::SeqCst);
        if self.produced.live.load(Ordering::SeqCst) == 0 {
            self.note_late(event, value);
            return Ok(0);
        }
        self.produced.wake.notify();
        if let Some(queue) = &self.modern_queue {
            queue.notify_ready();
        }
        let asked = std::time::Instant::now();
        self.produced.release_pending.store(true, Ordering::SeqCst);
        let released = self.run_pending(Pending::Producer);
        self.produced.producer_lock_us.fetch_add(
            asked.elapsed().as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        if let Ok(n) = released {
            self.produced
                .released_by_producer
                .fetch_add(n as u64, Ordering::Relaxed);
        }
        if let Err(error) = &released {
            // Held work no thread will release again: a visible failure, not
            // a stall.
            self.failure
                .record(format!("held work its input released was refused: {error}"));
        }
        released
    }

    /// Run the release and the cancellations left pending, while the state
    /// lock is free; returns what this call released. When the lock is held,
    /// its holder runs this once it unlocks: every unlock is followed by a
    /// look at the flags, and every flag raised is followed by an attempt at
    /// the lock, so one of the two sees the other.
    fn run_pending(&self, by: Pending) -> Result<usize, MetalKernelError> {
        let mut released = 0;
        loop {
            std::sync::atomic::fence(Ordering::SeqCst);
            if !self.produced.release_pending.load(Ordering::SeqCst)
                && !self.produced.cancel_pending.load(Ordering::SeqCst)
            {
                return Ok(released);
            }
            let Some(mut state) = self.try_lock_entry_unhooked() else {
                match by {
                    Pending::Producer => &self.produced.delegated,
                    Pending::Canceller => &self.produced.cancels_delegated,
                    Pending::Holder => return Ok(released),
                }
                .fetch_add(1, Ordering::Relaxed);
                return Ok(released);
            };
            self.produced.release_pending.store(false, Ordering::SeqCst);
            let ran = self.release_produced_locked(&mut state);
            drop(state);
            released += ran?;
        }
    }

    /// Apply every cancellation left pending, in the order asked for. One
    /// whose close of the current encoder was only busy is kept, ahead of any
    /// asked for meanwhile, and left pending again: whoever holds the lock
    /// next applies it before any release (`cancel_locked`).
    fn apply_cancels_locked(&self, state: &mut EntryGuard<'_>) -> Result<(), MetalKernelError> {
        self.produced.cancel_pending.store(false, Ordering::SeqCst);
        let pending = std::mem::take(&mut *self.produced.cancels.lock());
        let mut refused = None;
        let mut busy = None;
        let mut kept = Vec::new();
        for cancel in pending {
            match self.cancel_locked(state, &cancel) {
                Ok(Applied::Done) => {}
                Ok(Applied::Kept(error)) => {
                    busy.get_or_insert(error);
                    kept.push(cancel);
                }
                Err(error) => {
                    refused.get_or_insert(error);
                }
            }
        }
        if !kept.is_empty() {
            let mut cancels = self.produced.cancels.lock();
            kept.append(&mut cancels);
            *cancels = kept;
            self.produced.cancel_pending.store(true, Ordering::SeqCst);
        }
        refused.or(busy).map_or(Ok(()), Err)
    }

    /// One cancellation, under the lock. The current recording is left
    /// abandoned, never replaced here: a replacement may park for a slot, and
    /// a cancellation runs wherever the lock is found — a producer's thread,
    /// an owner parked mid-cut. The next recording operation or cut replaces
    /// it.
    fn cancel_locked(
        &self,
        state: &mut EntryGuard<'_>,
        cancel: &Cancel,
    ) -> Result<Applied, MetalKernelError> {
        let first = state
            .held
            .iter()
            .position(|held| held.gate.covered_by(cancel));
        if first.is_none() && !state.gate.covered_by(cancel) {
            return Ok(Applied::Done);
        }
        let from = first.unwrap_or(state.held.len());
        let event: &ProtocolObject<dyn MTLSharedEvent> = &cancel.event;
        let mut upto = state.gate.highest_on(event);
        let mut rehome: Vec<AfterCommit> = Vec::new();
        let mut cancelled = 0usize;
        let mut refused = Refusal::new(
            &self.failure,
            "a held recording could not be cancelled; queue failed",
        );
        {
            let (entry, custody) = state.parts_mut();
            let entry: &mut EntryState = entry;
            // Cancellation is not on the token's path: the drained
            // recordings are collected so their after-lists can go back.
            let drained: Vec<Held> = entry.held.drain(from..).collect();
            for held in drained {
                upto = upto.max(held.gate.highest_on(event));
                if let Err(error) = held.cb.cancel_unsubmitted() {
                    refused.note(error);
                }
                if let Err(error) = entry.owed_epochs.cancel(held.epoch) {
                    refused.note(error);
                }
                let mut after = held.after;
                for action in after.drain(..) {
                    match action {
                        AfterCommit::Io { io, signal, .. } => rehome.push(AfterCommit::Io {
                            io,
                            signal,
                            rehomed: true,
                        }),
                        AfterCommit::ValidatedIo(io) => {
                            if io.refuse().is_err() {
                                refused.note(MetalKernelError::QueueUnavailable(
                                    "cancelled IO refusal failed",
                                ));
                            }
                        }
                    }
                }
                recycle_after(&mut entry.spare_after, after);
                custody.cancelled.push(held.cb);
                cancelled += 1;
            }
        }
        // Loads ordered behind a cancelled signal keep that order. Placed
        // before the closes below, which may refuse: the current recording
        // is never committed, so nothing it could order moves.
        match state.held.back_mut() {
            Some(held) => held.after.extend(rehome),
            None => {
                if let Err(error) = self.run_after_commit(&mut rehome) {
                    refused.note(error);
                }
            }
        }
        // The current recording is behind every held one, or waits itself.
        // A close that refuses must not lose the cancellation: the current
        // recording stays gated on the event its producer gave up on, and the
        // producer may raise that event as soon as this returns (#1099). A
        // refusal that holds only for now (`Busy`, a recording edit in
        // progress) keeps the cancellation, for whoever holds the lock next
        // to apply before any release; any other fails the queue, so nothing
        // it covers is released. The bookkeeping below runs either way.
        let mut kept = None;
        match self
            .close_pending_blit(state)
            .and_then(|()| self.close_compute(state))
        {
            Ok(()) => {
                if !state.current_abandoned {
                    if let Some(current) = state.current.recording().cloned() {
                        if let Err(error) = current.cancel_unsubmitted() {
                            refused.note(error);
                        }
                        cancelled += 1;
                        state.gate = Gate::default();
                        state.signals_since_cut = false;
                        self.compute_count.store(0, Ordering::Release);
                        state.current_abandoned = true;
                    }
                }
            }
            Err(
                error @ MetalKernelError::RecordingAdmission(
                    crate::command_lifecycle::LifecycleError::Busy,
                ),
            ) => kept = Some(error),
            Err(error) => refused.note(error),
        }
        let p = &self.produced;
        *p.cancel_reason.lock() = cancel.reason;
        p.cancelled.fetch_add(cancelled as u64, Ordering::Relaxed);
        p.cancelled_upto.store(upto, Ordering::Release);
        p.cancelled_event.store(
            event as *const ProtocolObject<dyn MTLSharedEvent> as *const () as usize,
            Ordering::Release,
        );
        self.held_live_update(state);
        p.wake.notify();
        refused.into_result()?;
        Ok(match kept {
            Some(error) => Applied::Kept(error),
            None => Applied::Done,
        })
    }

    /// An input produced after the work waiting for it was cancelled finds
    /// nothing to commit: counted, and reported once.
    fn note_late(&self, event: &ProtocolObject<dyn MTLSharedEvent>, value: u64) {
        if self.produced.cancelled_event.load(Ordering::Acquire)
            != event as *const ProtocolObject<dyn MTLSharedEvent> as *const () as usize
            || value > self.produced.cancelled_upto.load(Ordering::Acquire)
        {
            return;
        }
        self.produced
            .late_after_cancel
            .fetch_add(1, Ordering::Relaxed);
        if !self.produced.late_logged.swap(true, Ordering::Relaxed) {
            eprintln!(
                "[metal] host input {value} produced after the work waiting for it was cancelled: nothing to commit"
            );
        }
    }

    /// Final teardown: no producer is left to raise the inputs, and nothing
    /// is left to observe the work, so the held recordings are committed as
    /// recorded, in order — up to the first that waits for an IO load not
    /// validated yet, which would read bytes that never landed. That one,
    /// every recording behind it and the current one are cancelled instead
    /// (`cancel_for_teardown`). `false` when a lock was busy; retried later.
    pub(super) fn release_held_for_teardown(
        &self,
        state: &mut EntryState,
        captures: &mut super::lifecycle::FinalCaptures,
    ) -> Result<bool, MetalKernelError> {
        let mut committed = 0usize;
        while let Some(held) = state.held.front() {
            if held.gate.waits_for_unvalidated_io() {
                break;
            }
            let runs = state
                .owed_epochs
                .runs_for(Some(held.epoch), state.held.len() - 1);
            let authority = OwnerSubmission {
                buffer: &held.cb,
                clock: &self.clock,
            };
            let staged = if captures.removals.is_none() {
                Some(&mut captures.removals)
            } else {
                None
            };
            match self.submit_owned_recording(
                authority,
                staged,
                true,
                None,
                None,
                Some(held.epoch),
                runs,
            ) {
                Ok(_) => {
                    state.owed_epochs.settle(runs);
                    let held = state.held.pop_front().expect("checked above");
                    state.in_flight.push(held.cb);
                    committed += 1;
                    let mut after = held.after;
                    let ran = self.run_after_commit(&mut after);
                    recycle_after(&mut state.spare_after, after);
                    ran?;
                }
                Err(MetalKernelError::RecordingAdmission(
                    crate::command_lifecycle::LifecycleError::Busy,
                )) => return Ok(false),
                Err(error) => return Err(error),
            }
        }
        let gated = !state.gate.is_empty();
        let cancelled = self.cancel_for_teardown(state, captures)?;
        if committed > 0 || (gated && cancelled == 0) {
            eprintln!(
                "[metal] teardown committed {committed} held recording(s) whose input was never produced"
            );
        }
        if cancelled > 0 {
            eprintln!(
                "[metal] teardown cancelled {cancelled} recording(s) behind IO not yet validated"
            );
        }
        state.gate = Gate::default();
        Ok(true)
    }

    /// The teardown half of `cancel_locked`: every held recording left (the
    /// first waits for unvalidated IO) and the current one, when it is behind
    /// them or waits itself, are abandoned before any native entry. Their
    /// epochs are owed to the final submission, validated IO deferred behind
    /// them is refused and ordinary IO keeps its order, as a cancellation's.
    /// The abandoned recordings leave with the final captures.
    fn cancel_for_teardown(
        &self,
        state: &mut EntryState,
        captures: &mut super::lifecycle::FinalCaptures,
    ) -> Result<usize, MetalKernelError> {
        if state.held.is_empty() && !state.gate.waits_for_unvalidated_io() {
            return Ok(0);
        }
        captures
            .cancelled
            .try_reserve(state.held.len())
            .map_err(|_| {
                MetalKernelError::QueueUnavailable("final cancelled-recording custody refused")
            })?;
        let mut rehome: Vec<AfterCommit> = Vec::new();
        let mut cancelled = 0usize;
        let mut refused = None;
        for held in state.held.drain(..) {
            if let Err(error) = held.cb.cancel_unsubmitted() {
                refused.get_or_insert(error);
            }
            if let Err(error) = state.owed_epochs.cancel(held.epoch) {
                refused.get_or_insert(error);
            }
            let mut after = held.after;
            for action in after.drain(..) {
                match action {
                    AfterCommit::Io { io, signal, .. } => rehome.push(AfterCommit::Io {
                        io,
                        signal,
                        rehomed: true,
                    }),
                    AfterCommit::ValidatedIo(io) => {
                        if io.refuse().is_err() {
                            refused.get_or_insert(MetalKernelError::QueueUnavailable(
                                "cancelled IO refusal failed",
                            ));
                        }
                    }
                }
            }
            recycle_after(&mut state.spare_after, after);
            captures.cancelled.push(held.cb);
            cancelled += 1;
        }
        if !state.current_abandoned {
            if let Some(current) = state.current.recording() {
                if let Err(error) = current.cancel_unsubmitted() {
                    refused.get_or_insert(error);
                }
                cancelled += 1;
                state.signals_since_cut = false;
                self.compute_count.store(0, Ordering::Release);
                state.current_abandoned = true;
            }
        }
        state.gate = Gate::default();
        if let Err(error) = self.run_after_commit(&mut rehome) {
            refused.get_or_insert(error);
        }
        self.held_live_update(state);
        self.produced
            .cancelled
            .fetch_add(cancelled as u64, Ordering::Relaxed);
        if let Some(error) = refused {
            self.failure
                .record_static("a recording could not be cancelled at teardown; queue failed");
            return Err(error);
        }
        Ok(cancelled)
    }
}

impl Commands {
    /// The arm ordering work behind host- and IO-produced inputs.
    pub fn host_input_arm(&self) -> HostInputArm {
        self.core.host_input
    }

    /// Counts of the held queue.
    pub fn produced_stats(&self) -> ProducedStats {
        let p = &self.core.produced;
        ProducedStats {
            held_now: self.core.produced.live.load(Ordering::Acquire),
            held: p.held.load(Ordering::Relaxed),
            released: p.released.load(Ordering::Relaxed),
            cancelled: p.cancelled.load(Ordering::Relaxed),
            late_after_cancel: p.late_after_cancel.load(Ordering::Relaxed),
            parks: p.parks.load(Ordering::Relaxed),
            released_by_producer: p.released_by_producer.load(Ordering::Relaxed),
            held_us: p.held_us.load(Ordering::Relaxed),
            io_deferred: p.io_deferred.load(Ordering::Relaxed),
            producer_lock_us: p.producer_lock_us.load(Ordering::Relaxed),
            delegated: p.delegated.load(Ordering::Relaxed),
            cancels_delegated: p.cancels_delegated.load(Ordering::Relaxed),
            beside_its_event: p.beside_its_event.load(Ordering::Relaxed),
        }
    }

    /// Commit every command encoded after this one only once `event` has
    /// reached `value`, raised by a host thread or the IO queue that then
    /// calls [`Self::produced`]. Nothing waits: the encoding thread records on,
    /// and the producer commits the released recordings. An input already
    /// produced costs nothing. Ends the open compute encoder when it has to
    /// cut.
    pub fn commit_when_produced(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        self.commit_when_produced_unless(event, value, || Ok(()))
    }

    /// As [`Self::commit_when_produced`], refused by `admit` under the state
    /// lock — the lock [`Self::cancel_produced`] takes — so an input cancelled
    /// concurrently is never gated after its cancellation.
    pub fn commit_when_produced_unless(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
        admit: impl FnOnce() -> Result<(), MetalKernelError>,
    ) -> Result<(), MetalKernelError> {
        if self.core.host_input == HostInputArm::GpuWait {
            admit()?;
            return self.encode_wait_for_event(event, value);
        }
        let mut state = self.lock_recording_state()?;
        self.ensure_recording_locked(&mut state)?;
        admit()?;
        if event.signaledValue() >= value {
            std::sync::atomic::fence(Ordering::Acquire);
            return Ok(());
        }
        let pristine = self.core.compute_count.load(Ordering::Acquire) == 0
            && !state.signals_since_cut
            && state.current_encoder.is_none()
            && state.pending_blit.is_none();
        if !pristine || state.gate.is_full() {
            self.cut_locked(&mut state, 0)?;
        }
        let mut input = Input::new(event, value)?;
        input.seq = self.core.produced.gated.fetch_add(1, Ordering::SeqCst) + 1;
        match state.gate.add(input) {
            Ok(Added::BesideItsEvent) => self.core.produced.note_beside_its_event(),
            Ok(Added::New | Added::Repeat) => {}
            Err(_) => {
                return Err(MetalKernelError::QueueUnavailable(
                    "a fresh recording's gate refused an input",
                ))
            }
        }
        self.core.held_live_update(&state);
        // A producer that raised the event since the look above found the
        // count without this gate: the release below sees the event.
        self.core.release_produced_locked(&mut state)?;
        if state.current_abandoned {
            // A cancellation applied here took the gated recording.
            return Err(self.core.produced.cancelled_error());
        }
        Ok(())
    }

    /// A producer raised `event` to `value` (the write it guards done
    /// first): commit the held recordings that released, in order. Returns
    /// how many. Called by every producer after its signal; with nothing
    /// held it returns without the state lock.
    pub fn produced(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<usize, MetalKernelError> {
        self.core.produced_on(event, value)
    }

    /// For an input the GPU side produces (an IO queue load): have the
    /// event's own notification commit what it releases when it reaches
    /// `value`. Nothing when it already has, or on the GPU-wait arm.
    pub fn notify_produced_at(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<(), MetalKernelError> {
        if self.core.host_input == HostInputArm::GpuWait || event.signaledValue() >= value {
            return Ok(());
        }
        let listener = self.core.produced.listener();
        // SAFETY: the block is valid for the call and Metal copies (retains)
        // it; it owns only a weak handle to this queue's core.
        unsafe {
            event.notifyListener_atValue_block(
                &listener.listener,
                value,
                block2::RcBlock::as_ptr(&listener.block),
            );
        }
        Ok(())
    }

    /// Cold enrollment; the caller accounts this metadata/native event before
    /// publishing an experimental producer. No event is created per issue.
    pub(crate) fn new_io_compute_authority(&self) -> Result<IoComputeAuthority, MetalKernelError> {
        let event =
            self.core.device.as_ref().newSharedEvent().ok_or_else(|| {
                MetalKernelError::FailedToCreateResource("IO compute event".into())
            })?;
        Ok(IoComputeAuthority {
            owner: self.core.produced.me.clone(),
            event,
            next: AtomicU64::new(0),
        })
    }

    pub(crate) fn signal_enrolled_io(
        &self,
        authority: &IoComputeAuthority,
    ) -> Result<u64, MetalKernelError> {
        if authority.owner.as_ptr() != &*self.core as *const CommandsCore {
            return Err(MetalKernelError::QueueUnavailable(
                "IO compute authority belongs to another Commands queue",
            ));
        }
        let mut state = self.lock_recording_state()?;
        self.ensure_recording_locked(&mut state)?;
        let value = authority
            .next
            .load(Ordering::Relaxed)
            .checked_add(1)
            .ok_or(MetalKernelError::QueueUnavailable(
                "IO compute sequence exhausted",
            ))?;
        // Reserve even when encoding later fails: a partially encoded signal
        // must never satisfy a retried IO with a reused numeric identity.
        authority.next.store(value, Ordering::Release);
        self.encode_signal_event_locked(&mut state, authority.event(), value)?;
        Ok(value)
    }

    /// Experimental A only. Recording remains overlapped; this is a status
    /// dependency in the EXISTING held-recording queue, not a CPU wait.
    pub(crate) fn commit_when_io_validated(
        &self,
        io: NativeIoDependency,
    ) -> Result<(), MetalKernelError> {
        if !self.gate_when_io_validated(&io)? {
            return Ok(());
        }
        // Race closure: terminal before or during listener installation must
        // run status release. The retained status gate, not event MAX, decides
        // which accepted values are valid or cancelled.
        let notified = io.notification().signaledValue();
        if notified < io.value() {
            let listener = self.core.produced.listener();
            let block = self.core.produced.io_notification();
            // SAFETY: both cached blocks remain owned by ProducedState; Metal
            // copies the supplied block, which retains only a weak queue root.
            unsafe {
                io.notification().notifyListener_atValue_block(
                    &listener.listener,
                    io.value(),
                    block2::RcBlock::as_ptr(block),
                );
            }
        }
        let notified = io.notification().signaledValue();
        if notified >= io.value() {
            self.core.produced_on(io.notification(), notified)?;
        }
        Ok(())
    }

    // The same admission transition is exercised without a platform listener
    // in deterministic callback-argument controls. No alternate gate algorithm.
    pub(super) fn gate_when_io_validated(
        &self,
        io: &NativeIoDependency,
    ) -> Result<bool, MetalKernelError> {
        if self.core.host_input != HostInputArm::Commit {
            return Err(MetalKernelError::QueueUnavailable(
                "validated-submission arm requires explicit commit selection",
            ));
        }
        let mut state = self.lock_recording_state()?;
        self.ensure_recording_locked(&mut state)?;
        if io.generation_failed() {
            return Err(MetalKernelError::QueueUnavailable(
                "IO generation refuses new dependent admission",
            ));
        }
        if io.ready() {
            return Ok(false);
        }
        let pristine = self.core.compute_count.load(Ordering::Acquire) == 0
            && !state.signals_since_cut
            && state.current_encoder.is_none()
            && state.pending_blit.is_none();
        if !pristine || state.gate.is_full() {
            self.cut_locked(&mut state, 0)?;
        }
        let mut input = Input::from_io(io.clone())?;
        input.seq = self.core.produced.gated.fetch_add(1, Ordering::SeqCst) + 1;
        state
            .gate
            .add(input)
            .map_err(|_| MetalKernelError::QueueUnavailable("fresh IO gate refused input"))?;
        self.core.held_live_update(&state);
        if io.generation_failed() {
            self.core
                .produced
                .io_failure_pending
                .store(true, Ordering::SeqCst);
        }
        self.core.release_produced_locked(&mut state)?;
        let abandoned = state.current_abandoned;
        drop(state);
        if abandoned {
            return Err(self.core.produced.cancelled_error());
        }
        Ok(true)
    }

    /// Private typed counterpart; moving into the after-list leaves Prepared.
    pub(crate) fn commit_validated_io_after(
        &self,
        io: NativeDeferredIo,
    ) -> Result<(), MetalKernelError> {
        if self.core.host_input != HostInputArm::Commit {
            return Err(MetalKernelError::QueueUnavailable(
                "validated IO arm requires explicit commit selection",
            ));
        }
        let mut state = self.lock_recording_state()?;
        self.ensure_recording_locked(&mut state)?;
        let pristine = self.core.compute_count.load(Ordering::Acquire) == 0
            && !state.signals_since_cut
            && state.current_encoder.is_none()
            && state.pending_blit.is_none();
        if !pristine {
            self.cut_locked(&mut state, 0)?;
        }
        if let Some(held) = state.held.back_mut() {
            held.after
                .try_reserve(1)
                .map_err(|_| MetalKernelError::QueueUnavailable("held IO storage refused"))?;
            held.after.push(AfterCommit::ValidatedIo(io));
            self.core
                .produced
                .io_deferred
                .fetch_add(1, Ordering::Relaxed);
        } else {
            io.commit()
                .map_err(|_| MetalKernelError::QueueUnavailable("validated IO handoff refused"))?;
        }
        Ok(())
    }

    /// Commit `io`, an IO command buffer waiting for `signal_value` on
    /// `signal_event` — a signal already encoded on this stream — once the
    /// recording carrying that signal is on the GPU: at once when it is,
    /// after its producer otherwise. An IO wait on a signal nothing has
    /// committed could time out under the loads.
    pub fn commit_io_after(
        &self,
        io: &ProtocolObject<dyn MTLIOCommandBuffer>,
        signal_event: &ProtocolObject<dyn MTLSharedEvent>,
        signal_value: u64,
    ) -> Result<(), MetalKernelError> {
        if self.core.host_input == HostInputArm::GpuWait {
            self.flush_marker()?;
            io.commit();
            return Ok(());
        }
        let mut state = self.lock_recording_state()?;
        self.ensure_recording_locked(&mut state)?;
        let pristine = self.core.compute_count.load(Ordering::Acquire) == 0
            && !state.signals_since_cut
            && state.current_encoder.is_none()
            && state.pending_blit.is_none();
        if !pristine {
            self.cut_locked(&mut state, 0)?;
        }
        match state.held.back_mut() {
            Some(held) => {
                held.after
                    .try_reserve(1)
                    .map_err(|_| MetalKernelError::QueueUnavailable("held IO storage refused"))?;
                // SAFETY: the borrowed IO command buffer is live; the retain
                // keeps it until its commit.
                let io = unsafe { Retained::retain(std::ptr::from_ref(io).cast_mut()) }.ok_or(
                    MetalKernelError::QueueUnavailable("IO buffer retain refused"),
                )?;
                held.after.push(AfterCommit::Io {
                    io,
                    signal: Input::new(signal_event, signal_value)?,
                    rehomed: false,
                });
                self.core
                    .produced
                    .io_deferred
                    .fetch_add(1, Ordering::Relaxed);
            }
            None => io.commit(),
        }
        Ok(())
    }

    /// The inputs on `event` will never be produced (its producer failed, or
    /// its run ended early): cancel the held recordings waiting for one,
    /// every recording behind them and the current one. Nothing they encoded
    /// runs; their native work is abandoned through host retirement, their
    /// epochs complete with the next commit, and an IO load ordered behind
    /// them keeps its order through a signal re-issued on the queue. A wait on
    /// a cancelled recording, or a cut of the recording a cancellation takes,
    /// fails with `reason`. Never waits for the state lock: when it is held,
    /// the holder applies the cancellation as it unlocks, an owner parked
    /// behind the work applies it in its park, and every release applies it
    /// first, so raising `event` after this call cannot release what it
    /// covers. Returns the recordings cancelled while this call held the
    /// lock: 0 when nothing waited on `event` or a holder applies it.
    pub fn cancel_produced(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        reason: &'static str,
    ) -> Result<usize, MetalKernelError> {
        self.cancel_produced_through(event, u64::MAX, reason)
    }

    /// As [`Self::cancel_produced`], for the values on `event` up to `upto`:
    /// a recording gated on any value in that range is covered, whatever
    /// other values of `event` it also waits for; work behind only later
    /// values, a next run's, is not, whenever the cancellation is applied.
    pub fn cancel_produced_through(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        upto: u64,
        reason: &'static str,
    ) -> Result<usize, MetalKernelError> {
        let core = &self.core;
        std::sync::atomic::fence(Ordering::SeqCst);
        if core.produced.live.load(Ordering::SeqCst) == 0 {
            return Ok(0);
        }
        // SAFETY: the borrowed event is live; the retain keeps it until the
        // cancellation is applied.
        let retained = unsafe { Retained::retain(std::ptr::from_ref(event).cast_mut()) }
            .ok_or(MetalKernelError::QueueUnavailable("event retain refused"))?;
        core.produced.cancels.lock().push(Cancel {
            event: retained,
            produced: event.signaledValue(),
            upto,
            gated_through: core.produced.gated.load(Ordering::SeqCst),
            reason,
        });
        core.produced.cancel_pending.store(true, Ordering::SeqCst);
        // An owner parked behind the work applies it in its park.
        core.produced.wake.notify();
        if let Some(queue) = &core.modern_queue {
            queue.notify_ready();
        }
        let before = core.produced.cancelled.load(Ordering::Relaxed);
        core.run_pending(Pending::Canceller)?;
        Ok(core
            .produced
            .cancelled
            .load(Ordering::Relaxed)
            .saturating_sub(before) as usize)
    }

    /// Swap a fresh recording in for an abandoned current one.
    pub(super) fn replace_abandoned_locked<'a>(
        &'a self,
        state: &mut EntryGuard<'a>,
    ) -> Result<(), MetalKernelError> {
        if !state.current_abandoned {
            return Ok(());
        }
        let credit = state
            .binding_credits
            .iter_mut()
            .find(|slot| slot.is_some())
            .and_then(Option::take);
        let core = &self.core;
        let next = self.create_replacement_parking(credit, &mut |ready, seen| {
            let cancelled = core.produced.cancelled.load(Ordering::Relaxed);
            if core.release_produced_locked(state)? == 0
                && core.produced.cancelled.load(Ordering::Relaxed) == cancelled
            {
                ready.wait(seen);
            }
            Ok(())
        })?;
        let (entry, custody) = state.parts_mut();
        match entry.current.replace_recording(next) {
            Ok(old) => custody.cancelled.push(old),
            Err(unused) => custody.cancelled.push(unused),
        }
        entry.current_abandoned = false;
        entry.namespace.reset();
        self.core.compute_count.store(0, Ordering::Release);
        self.core
            .replacement_pending
            .store(false, Ordering::Release);
        Ok(())
    }

    /// The cut every commit path takes: an ordinary commit, or — while an
    /// input this recording or one held ahead of it reads is not produced —
    /// a held cut. A current recording a cancellation abandoned before the
    /// cut is replaced first; one a cancellation takes during the cut fails
    /// it with the cancellation's reason.
    pub(super) fn cut_locked<'a>(
        &'a self,
        state: &mut EntryGuard<'a>,
        reset_to: usize,
    ) -> Result<Option<u64>, MetalKernelError> {
        if state.current_abandoned {
            self.replace_abandoned_locked(state)?;
        }
        while state.held.is_empty() && state.gate.is_open() {
            // The gate's look, then the flag, as `release_produced_locked`.
            std::sync::atomic::fence(Ordering::SeqCst);
            if !self.core.produced.cancel_pending.load(Ordering::SeqCst) {
                if !state.gate.is_empty() {
                    state.gate = Gate::default();
                    self.core.held_live_update(state);
                }
                return self.commit_swap_locked(state, reset_to);
            }
            self.core.release_produced_locked(state)?;
            if state.current_abandoned {
                return Err(self.cut_cancelled(state));
            }
        }
        self.hold_swap_locked(state, reset_to)?;
        if state.current_abandoned {
            return Err(self.cut_cancelled(state));
        }
        Ok(None)
    }

    /// A cancellation took the recording being cut: replace it, and fail the
    /// cut with the cancellation's reason.
    fn cut_cancelled<'a>(&'a self, state: &mut EntryGuard<'a>) -> MetalKernelError {
        match self.replace_abandoned_locked(state) {
            Ok(()) => self.core.produced.cancelled_error(),
            Err(error) => error,
        }
    }

    /// The buffer the last cut closed: held, or committed.
    pub(super) fn last_cut(state: &EntryState) -> Option<&CommandBuffer> {
        state
            .held
            .back()
            .map(|held| &held.cb)
            .or_else(|| state.in_flight.last())
    }

    /// Close the current recording and hold it back: its epoch taken now,
    /// as an eager commit would have, its commit left to its producer.
    fn hold_swap_locked<'a>(
        &'a self,
        state: &mut EntryGuard<'a>,
        reset_to: usize,
    ) -> Result<(), MetalKernelError> {
        self.core.hook.seal()?;
        self.core.close_pending_blit(state)?;
        self.core.close_compute(state)?;
        if state.current.recording().is_none() {
            return Err(MetalKernelError::QueueUnavailable(
                "a held cut needs a recording",
            ));
        }
        while state.held.len() >= HELD_CAP {
            let seen = self.core.produced.wake.snapshot();
            let cancelled = self.core.produced.cancelled.load(Ordering::Relaxed);
            if self.core.release_produced_locked(state)? > 0 {
                continue;
            }
            if state.current_abandoned {
                return Err(self.cut_cancelled(state));
            }
            if self.core.produced.cancelled.load(Ordering::Relaxed) != cancelled {
                continue;
            }
            self.core.failure.check()?;
            self.core.refuse_park_in_callback()?;
            // Parked holding the state lock: a producer or a canceller wakes
            // this and never waits for the lock, and the release above, a
            // cancellation first, then runs here.
            self.core.produced.parks.fetch_add(1, Ordering::Relaxed);
            self.core.produced.wake.wait(seen);
        }
        let credit = state
            .binding_credits
            .iter_mut()
            .find(|slot| slot.is_some())
            .and_then(Option::take);
        let core = &self.core;
        let next = self.create_replacement_parking(credit, &mut |ready, seen| {
            // No slot comes back before the held recordings run: submit what
            // producers released, apply what was cancelled, then wait.
            let cancelled = core.produced.cancelled.load(Ordering::Relaxed);
            if core.release_produced_locked(state)? == 0
                && core.produced.cancelled.load(Ordering::Relaxed) == cancelled
            {
                core.produced.parks.fetch_add(1, Ordering::Relaxed);
                ready.wait(seen);
            }
            Ok(())
        })?;
        if state.current_abandoned {
            // A cancellation in the park took the recording being cut: the
            // slot acquired for its successor replaces it.
            let (entry, custody) = state.parts_mut();
            match entry.current.replace_recording(next) {
                Ok(old) => custody.cancelled.push(old),
                Err(unused) => custody.cancelled.push(unused),
            }
            entry.current_abandoned = false;
            entry.namespace.reset();
            self.core.compute_count.store(0, Ordering::Release);
            self.core
                .replacement_pending
                .store(false, Ordering::Release);
            return Err(self.core.produced.cancelled_error());
        }
        let epoch = self.core.clock.commit_epoch();
        let gate = std::mem::take(&mut state.gate);
        let old = match state.current.replace_recording(next) {
            Ok(old) => old,
            Err(_) => unreachable!("checked recording above"),
        };
        let after = state.spare_after.pop().unwrap_or_default();
        state.held.push_back(Held {
            cb: old,
            gate,
            epoch,
            after,
            at: std::time::Instant::now(),
        });
        self.core.compute_count.store(reset_to, Ordering::Release);
        self.core
            .replacement_pending
            .store(false, Ordering::Release);
        state.namespace.reset();
        state.signals_since_cut = false;
        self.core.produced.held.fetch_add(1, Ordering::Relaxed);
        self.core.held_live_update(state);
        // A producer that raised its event since the gate's last look found
        // the old count: the release below sees the event.
        self.core.release_produced_locked(state)?;
        Ok(())
    }
}

#[cfg(test)]
mod scope_tests {
    use super::*;
    use objc2_metal::MTLDevice;

    type Event = Retained<ProtocolObject<dyn MTLSharedEvent>>;

    fn new_event() -> Event {
        let device = super::super::Device::system_default().unwrap();
        device.as_ref().newSharedEvent().unwrap()
    }

    fn input(event: &Event, value: u64, seq: u64) -> Input {
        let mut input = Input::new(event, value).unwrap();
        input.seq = seq;
        input
    }

    /// A gate of `(value, seq)` inputs on `event`, in the order given.
    fn gate(event: &Event, inputs: &[(u64, u64)]) -> Gate {
        let mut gate = Gate::default();
        for &(value, seq) in inputs {
            assert!(gate.add(input(event, value, seq)).is_ok());
        }
        gate
    }

    fn cancel(event: &Event, produced: u64, upto: u64, gated_through: u64) -> Cancel {
        Cancel {
            event: event.clone(),
            produced,
            upto,
            gated_through,
            reason: "test",
        }
    }

    /// A cancellation's scope is fixed when it is asked for: an input gated
    /// after it is not covered, whenever it is applied, and a repeated input
    /// keeps the earlier gating.
    #[test]
    fn a_cancellation_covers_only_inputs_gated_before_it() {
        let event = new_event();
        let all = cancel(&event, 0, u64::MAX, 2);
        assert!(
            gate(&event, &[(1, 1)]).covered_by(&all),
            "an input gated before the request"
        );
        assert!(
            !gate(&event, &[(1, 3)]).covered_by(&all),
            "an input gated after the request"
        );
        assert!(
            gate(&event, &[(1, 1), (5, 3)]).covered_by(&all),
            "a second value lost the earlier gating"
        );
        assert!(
            gate(&event, &[(3, 1), (3, 3)]).covered_by(&all),
            "a repeat lost the earlier gating"
        );
    }

    /// Every value a recording gates on one event is its own prerequisite: a
    /// cancellation through any of them covers the recording, whichever
    /// order they were gated in and whatever was produced since; one through
    /// none of them does not.
    #[test]
    fn a_cancellation_through_a_value_finds_it_beside_other_values_of_its_event() {
        eprintln!("SCOPE_BUILD debug_assertions={}", cfg!(debug_assertions));
        let event = new_event();
        assert!(
            gate(&event, &[(1, 1), (5, 2)]).covered_by(&cancel(&event, 0, 1, 2)),
            "two values: the lower one"
        );
        assert!(
            gate(&event, &[(5, 1), (1, 2)]).covered_by(&cancel(&event, 0, 1, 2)),
            "descending values: the lower one, gated second"
        );
        assert!(
            gate(&event, &[(1, 1), (3, 2), (5, 3)]).covered_by(&cancel(&event, 2, 3, 3)),
            "an interior value after partial production"
        );
        assert!(
            !gate(&event, &[(1, 1), (3, 2), (5, 3)]).covered_by(&cancel(&event, 3, 4, 3)),
            "no value in (produced, upto]"
        );
        assert!(
            !gate(&event, &[(5, 1)]).covered_by(&cancel(&event, 0, 1, 1)),
            "a later value alone"
        );
        assert!(
            !gate(&event, &[(5, 1), (1, 3)]).covered_by(&cancel(&event, 0, 1, 2)),
            "the covered value was gated after the request"
        );
    }

    /// Readiness is every prerequisite produced; a repeat takes no slot, a
    /// distinct value does, and a full gate refuses the next one.
    #[test]
    fn a_gate_opens_once_every_value_is_produced_and_refuses_past_its_slots() {
        let event = new_event();
        let mut three = gate(&event, &[(5, 1), (1, 2), (3, 3)]);
        event.setSignaledValue(3);
        assert!(!three.is_open(), "open before its highest value");
        event.setSignaledValue(5);
        assert!(three.is_open(), "closed with every value produced");
        assert_eq!(three.add(input(&event, 3, 4)).ok(), Some(Added::Repeat));
        assert_eq!(
            three.add(input(&event, 7, 5)).ok(),
            Some(Added::BesideItsEvent)
        );
        let other = new_event();
        assert_eq!(three.add(input(&other, 1, 6)).ok(), Some(Added::New));

        let mut full = Gate::default();
        for value in 1..=GATE_INPUTS as u64 {
            assert!(full.add(input(&event, 10 + value, value)).is_ok());
        }
        assert!(full.is_full());
        assert_eq!(full.add(input(&event, 11, 99)).ok(), Some(Added::Repeat));
        assert!(
            full.add(input(&event, 99, 99)).is_err(),
            "a full gate took a ninth value"
        );
    }
}

#[cfg(test)]
mod owed_tests {
    //! The owed-epoch transfer over a model of the held queue's metadata:
    //! epochs taken at holds, cancellations of a suffix, releases from the
    //! front, the ordinary commit. The decision is production's
    //! ([`Owed::runs_for`], [`Owed::settle`]); the hook's n-th run completes
    //! epoch n, as `completion_hook.rs`. No device.
    use super::Owed;
    use crate::retire::Epoch;
    use std::collections::{BTreeSet, VecDeque};

    #[derive(Default)]
    struct Queue {
        owed: Owed,
        next: u64,
        held: VecDeque<u64>,
        committed: BTreeSet<u64>,
        cancelled: BTreeSet<u64>,
        /// Hook runs flagged so far: the epoch the next run completes.
        runs: u64,
        /// Per completed commit, in order: (its epoch, the epochs its runs
        /// marked).
        completions: Vec<(u64, Vec<u64>)>,
    }

    impl Queue {
        fn hold(&mut self) -> u64 {
            self.next += 1;
            self.held.push_back(self.next - 1);
            self.next - 1
        }

        fn cancel_from(&mut self, at: usize) {
            for epoch in self.held.drain(at..) {
                self.owed.cancel(Epoch(epoch)).unwrap();
                self.cancelled.insert(epoch);
            }
        }

        fn complete(&mut self, epoch: u64, runs: u32) {
            self.owed.settle(runs);
            let marked: Vec<u64> = (self.runs..self.runs + u64::from(runs)).collect();
            self.runs += u64::from(runs);
            self.committed.insert(epoch);
            // Every epoch a completion marks is committed by then (in-order
            // queue) or cancelled; never one still held or not taken.
            for m in &marked {
                assert!(
                    self.committed.contains(m) || self.cancelled.contains(m),
                    "the completion of epoch {epoch} marked epoch {m}, still held={:?}",
                    self.held
                );
            }
            // Exact: a commit's completion completes its own epoch.
            assert!(
                marked.contains(&epoch),
                "the completion of epoch {epoch} left it open: marked {marked:?}"
            );
            assert_eq!(self.owed.flagged(), self.runs, "the frontier drifted");
            self.completions.push((epoch, marked));
        }

        /// Release the front held recording; it completes at once.
        fn release(&mut self) -> u64 {
            let epoch = self.held[0];
            let runs = self.owed.runs_for(Some(Epoch(epoch)), self.held.len() - 1);
            self.held.pop_front();
            self.complete(epoch, runs);
            epoch
        }

        /// An ordinary commit of the current recording (nothing held).
        fn commit_current(&mut self) -> u64 {
            assert!(self.held.is_empty());
            let runs = self.owed.runs_for(None, self.held.len());
            self.next += 1;
            self.complete(self.next - 1, runs);
            self.next - 1
        }

        fn marked_by(&self, epoch: u64) -> Vec<u64> {
            self.completions
                .iter()
                .find(|(e, _)| *e == epoch)
                .map(|(_, m)| m.clone())
                .expect("completed")
        }

        fn all_complete(&self) {
            assert_eq!(self.runs, self.next, "an epoch taken never completed");
            assert_eq!(self.owed.count(), 0, "an owed epoch was never paid");
            assert_eq!(
                self.owed.flagged(),
                self.next,
                "the frontier is not the epoch count"
            );
        }
    }

    #[test]
    fn the_counterexample_h0_completes_its_own_epoch_and_h1_completes_h2() {
        let mut q = Queue::default();
        let [e0, e1, e2] = [q.hold(), q.hold(), q.hold()];
        q.cancel_from(2);
        q.release();
        assert_eq!(q.marked_by(e0), vec![e0]);
        assert_eq!(q.owed.count(), 1);
        q.release();
        assert_eq!(q.marked_by(e1), vec![e1, e2]);
        q.all_complete();
    }

    #[test]
    fn work_held_after_a_cancellation_carries_the_owed_epochs_from_above_them() {
        let mut q = Queue::default();
        let [e0, e1] = [q.hold(), q.hold()];
        q.cancel_from(1);
        let [e2, e3] = [q.hold(), q.hold()];
        q.release();
        q.release();
        q.release();
        assert_eq!(q.marked_by(e0), vec![e0]);
        assert_eq!(q.marked_by(e2), vec![e1, e2], "H2 left its own epoch open");
        assert_eq!(q.marked_by(e3), vec![e3]);
        q.all_complete();
    }

    #[test]
    fn repeated_cancellations_ride_the_last_live_recording() {
        let mut q = Queue::default();
        let [e0, e1, e2, e3] = [q.hold(), q.hold(), q.hold(), q.hold()];
        q.cancel_from(3);
        q.cancel_from(2);
        q.release();
        q.release();
        assert_eq!(q.marked_by(e0), vec![e0]);
        assert_eq!(q.marked_by(e1), vec![e1, e2, e3]);
        q.all_complete();
    }

    #[test]
    fn interleaved_holes_each_ride_the_first_live_commit_above_them() {
        // H0 | cancelled e1 | H2 | cancelled e3 | H4: H2 completes e1 and its
        // own epoch with H4 still held; H4, the last, completes e3 and its own.
        let mut q = Queue::default();
        let [e0, e1] = [q.hold(), q.hold()];
        q.cancel_from(1);
        let [e2, e3] = [q.hold(), q.hold()];
        q.cancel_from(2);
        let e4 = q.hold();
        q.release();
        q.release();
        assert_eq!(q.marked_by(e0), vec![e0]);
        assert_eq!(q.marked_by(e2), vec![e1, e2]);
        assert_eq!(q.owed.count(), 1);
        q.release();
        assert_eq!(q.marked_by(e4), vec![e3, e4]);
        q.all_complete();
    }

    #[test]
    fn a_cancelled_tail_rides_the_last_live_commit_below_it() {
        // H0 | H1 | cancelled e2, e3: H1 is the last live recording and
        // completes both, with nothing behind it to wait for.
        let mut q = Queue::default();
        let [e0, e1, e2, e3] = [q.hold(), q.hold(), q.hold(), q.hold()];
        q.cancel_from(2);
        q.release();
        q.release();
        assert_eq!(q.marked_by(e0), vec![e0]);
        assert_eq!(q.marked_by(e1), vec![e1, e2, e3]);
        q.all_complete();
    }

    #[test]
    fn the_ordinary_commit_after_an_emptied_queue_carries_what_is_owed() {
        let mut q = Queue::default();
        let e0 = q.hold();
        q.cancel_from(0);
        let e1 = q.commit_current();
        assert_eq!(q.marked_by(e1), vec![e0, e1]);
        q.all_complete();
    }

    #[test]
    fn a_refused_commit_keeps_what_is_owed() {
        let mut q = Queue::default();
        let [e0, _] = [q.hold(), q.hold()];
        q.cancel_from(1);
        let before = q.owed;
        // Decided, refused before submission: nothing settles.
        let runs = q.owed.runs_for(Some(Epoch(e0)), 0);
        assert_eq!(runs, 2);
        assert_eq!(q.owed, before);
        q.release();
        assert_eq!(q.marked_by(e0), vec![0, 1]);
        q.all_complete();
    }

    #[test]
    fn the_owed_count_refuses_past_a_u32_of_hook_runs() {
        let mut owed = Owed {
            count: Owed::MAX_OWED - 1,
            flagged: 7,
        };
        owed.cancel(Epoch(7)).unwrap();
        assert_eq!(owed.count(), Owed::MAX_OWED);
        assert!(owed.cancel(Epoch(8)).is_err(), "the bound did not refuse");
        assert_eq!(
            owed.count(),
            Owed::MAX_OWED,
            "a refused cancellation counted"
        );
        // The largest carry is representable: every owed epoch, or a front
        // with every owed epoch below it.
        assert_eq!(owed.runs_for(None, 0), u32::MAX);
        assert_eq!(
            owed.runs_for(Some(Epoch(7 + u64::from(Owed::MAX_OWED))), 1),
            u32::MAX
        );
        owed.settle(u32::MAX);
        assert_eq!((owed.count(), owed.flagged()), (0, 7 + u64::from(u32::MAX)));
    }

    /// Defensive arithmetic only: a frontier this close to `u64::MAX` with
    /// epochs owed is unreachable on a valid schedule (reserved epochs are
    /// the frontier plus the owed and held ones).
    #[test]
    fn the_frontier_saturates_and_then_carries_no_holes() {
        let mut owed = Owed {
            count: 3,
            flagged: u64::MAX - 1,
        };
        owed.settle(1);
        owed.settle(1);
        assert_eq!((owed.count(), owed.flagged()), (3, u64::MAX));
        // At its ceiling the frontier is above any front: no hole is carried,
        // so nothing is marked ahead; the tail still carries what is owed.
        assert_eq!(owed.runs_for(Some(Epoch(9)), 1), 1);
        assert_eq!(owed.runs_for(Some(Epoch(9)), 0), 4);
    }

    /// Lane C's invariant (#944 6556673766 §1) over the production decisions,
    /// from states no short run reaches. `S` is the reserved epochs, `F` the
    /// frontier (`Owed::flagged`), `C` the owed count and `H` the held
    /// recordings. After every transition `S = F + C + H`, and every commit
    /// carries no more runs than the epochs reserved (`F + runs <= S`), so a
    /// settle never saturates. Completions are delivered late, in commit
    /// order, and never cover a held epoch. A refused submission decides its
    /// runs and settles nothing: no completion is published for it and the
    /// held epoch stays uncovered. No native callback runs.
    struct Invariant {
        owed: Owed,
        reserved: u64,
        held: VecDeque<u64>,
        /// Accepted commits not completed yet: (first run, runs).
        pending: VecDeque<(u64, u32)>,
        completed: u64,
        failed: bool,
    }

    impl Invariant {
        fn seeded(count: u32, flagged: u64) -> Self {
            let reserved = flagged + u64::from(count);
            let me = Self {
                owed: Owed::for_test(count, flagged),
                reserved,
                held: VecDeque::new(),
                pending: VecDeque::new(),
                completed: flagged,
                failed: false,
            };
            me.check();
            me
        }

        fn check(&self) {
            assert_eq!(
                self.reserved,
                self.owed.flagged() + u64::from(self.owed.count()) + self.held.len() as u64,
                "S != F + C + H"
            );
            if let Some(&front) = self.held.front() {
                assert!(
                    self.completed <= front,
                    "held epoch {front} covered: completed {}",
                    self.completed
                );
            }
        }

        fn hold(&mut self) {
            if self.reserved == u64::MAX {
                return;
            }
            self.held.push_back(self.reserved);
            self.reserved += 1;
            self.check();
        }

        fn cancel_from(&mut self, at: usize) {
            let at = at.min(self.held.len());
            while self.held.len() > at {
                let epoch = self.held.pop_back().expect("checked");
                let before = self.owed;
                if self.owed.cancel(Epoch(epoch)).is_err() {
                    // Refused at the bound: the queue fails, nothing counted.
                    assert_eq!(self.owed, before, "a refused cancellation counted");
                    assert_eq!(self.owed.count(), Owed::MAX_OWED);
                    self.held.push_back(epoch);
                    self.failed = true;
                    self.check();
                    return;
                }
                self.check();
            }
        }

        fn accept(&mut self, runs: u32) {
            let frontier = self.owed.flagged();
            assert!(
                frontier
                    .checked_add(u64::from(runs))
                    .is_some_and(|end| end <= self.reserved),
                "a commit carries runs past the reserved epochs"
            );
            self.owed.settle(runs);
            assert_eq!(self.owed.flagged(), frontier + u64::from(runs), "saturated");
            self.pending.push_back((frontier, runs));
        }

        fn release(&mut self, refused: bool) {
            let Some(&epoch) = self.held.front() else {
                return;
            };
            let runs = self.owed.runs_for(Some(Epoch(epoch)), self.held.len() - 1);
            if refused {
                let before = (self.owed, self.completed, self.pending.len());
                self.check();
                assert_eq!(before, (self.owed, self.completed, self.pending.len()));
                return;
            }
            self.held.pop_front();
            self.accept(runs);
            self.check();
        }

        fn commit_current(&mut self, refused: bool) {
            if !self.held.is_empty() || self.reserved == u64::MAX {
                return;
            }
            let runs = self.owed.runs_for(None, 0);
            if refused {
                let before = (self.owed, self.reserved, self.pending.len());
                self.check();
                assert_eq!(before, (self.owed, self.reserved, self.pending.len()));
                return;
            }
            self.reserved += 1;
            self.accept(runs);
            self.check();
        }

        fn deliver(&mut self) {
            let Some((first, runs)) = self.pending.pop_front() else {
                return;
            };
            assert!(first >= self.completed, "completions out of order");
            let end = first + u64::from(runs);
            assert!(end <= self.reserved, "a run past the reserved epochs");
            for &epoch in &self.held {
                assert!(
                    !(first..end).contains(&epoch),
                    "a run marked held epoch {epoch}"
                );
            }
            self.completed = end;
            self.check();
        }

        /// The abandoned final: nothing held, every pending completion
        /// delivered first (the settle waits for the prefix), then the owed
        /// range is exactly what is left of the reserved epochs.
        fn abandon(&mut self) {
            assert!(self.held.is_empty());
            while !self.pending.is_empty() {
                self.deliver();
            }
            let (frontier, reserved) = (self.owed.flagged(), self.reserved);
            let mut published = None;
            self.owed
                .abandon(|from, through| {
                    published = Some((from, through));
                    Ok(())
                })
                .unwrap();
            assert_eq!(
                published,
                Some((frontier, reserved)),
                "the range is not [F, S)"
            );
            assert_eq!((self.owed.count(), self.owed.flagged()), (0, reserved));
            self.completed = reserved;
            self.check();
        }

        /// A refused publication settles nothing.
        fn refused_abandon(&mut self) {
            let before = self.owed;
            assert!(self
                .owed
                .abandon(
                    |_, _| Err(crate::kernels::MetalKernelError::QueueUnavailable(
                        "test: refused"
                    ))
                )
                .is_err());
            assert_eq!(self.owed, before);
            self.check();
        }

        /// A deterministic mix of every transition.
        fn run(&mut self, steps: u32, mut seed: u64) {
            for _ in 0..steps {
                if self.failed {
                    return;
                }
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                match (seed >> 33) % 8 {
                    0 | 1 => self.hold(),
                    2 => self.cancel_from(((seed >> 40) % 4) as usize),
                    3 => self.release(false),
                    4 => self.release(true),
                    5 => self.commit_current((seed >> 45) & 1 == 1),
                    _ => self.deliver(),
                }
            }
        }

        /// Drain to the final: held work released or cancelled, then the
        /// final either submits the current recording or abandons it.
        fn finish(&mut self, abandon: bool) {
            if self.failed {
                return;
            }
            while self.held.len() > 1 {
                self.release(false);
            }
            if abandon {
                self.cancel_from(0);
                if self.failed {
                    return;
                }
                self.refused_abandon();
                self.abandon();
            } else {
                // The last live recording, or the final current one with
                // nothing held, carries every owed epoch.
                if self.held.is_empty() {
                    self.commit_current(false);
                } else {
                    self.release(false);
                }
                while !self.pending.is_empty() {
                    self.deliver();
                }
                assert_eq!(self.completed, self.reserved, "an epoch never completed");
            }
            assert_eq!(self.owed.count(), 0, "an owed epoch was never paid");
        }
    }

    #[test]
    fn the_reserved_epochs_are_the_frontier_owed_and_held_on_every_transition() {
        // Small state, every schedule kind, both finals.
        for seed in 0..64 {
            for abandon in [false, true] {
                let mut model = Invariant::seeded(0, 0);
                model.run(400, seed);
                model.finish(abandon);
            }
        }
    }

    #[test]
    fn the_invariant_holds_at_the_owed_bound() {
        // Owed one short of its bound: the next cancellation counts, the one
        // after is refused and fails the queue with nothing counted.
        let mut model = Invariant::seeded(Owed::MAX_OWED - 1, 1000);
        model.hold();
        model.hold();
        model.cancel_from(0);
        assert!(model.failed, "the bound did not refuse");
        assert_eq!(model.owed.count(), Owed::MAX_OWED);
        // The refused recording is still held. At the bound it is the last
        // live one and carries every owed epoch: u32::MAX hook runs, which
        // end exactly at the reserved epochs.
        assert_eq!(model.held.len(), 1);
        model.release(false);
        assert_eq!(model.pending.back(), Some(&(1000, u32::MAX)));
        model.deliver();
        assert_eq!(model.completed, model.reserved);
        assert_eq!(model.owed.count(), 0);
        let mut abandoned = Invariant::seeded(Owed::MAX_OWED, 7);
        abandoned.finish(true);
        assert_eq!(abandoned.owed.flagged(), 7 + u64::from(Owed::MAX_OWED));
    }

    #[test]
    fn the_invariant_holds_up_to_the_last_epoch() {
        // Reserved epochs reach u64::MAX exactly; no counter saturates and
        // the abandoned final's range ends there.
        for seed in 0..32 {
            for abandon in [false, true] {
                let flagged = u64::MAX - 1000;
                let mut model = Invariant::seeded(3, flagged);
                model.run(200, seed);
                assert!(model.reserved < u64::MAX);
                while model.reserved < u64::MAX {
                    model.hold();
                }
                assert_eq!(model.reserved, u64::MAX);
                model.cancel_from(1);
                model.finish(abandon);
                if abandon && !model.failed {
                    assert_eq!(model.owed.flagged(), u64::MAX);
                }
            }
        }
    }

    #[test]
    fn no_cancellation_carries_one_run_each() {
        let mut q = Queue::default();
        let e = [q.hold(), q.hold(), q.hold()];
        for _ in e {
            q.release();
        }
        let c = q.commit_current();
        for epoch in e.into_iter().chain([c]) {
            assert_eq!(q.marked_by(epoch), vec![epoch]);
        }
        q.all_complete();
    }
}
