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
use crate::kernels::MetalKernelError;
use crate::retire::Epoch;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue, MTLIOCommandBuffer, MTLSharedEvent};
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
}

impl Input {
    fn new(
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<Self, MetalKernelError> {
        // SAFETY: the borrowed event is live; the retain keeps it for the gate.
        let event = unsafe { Retained::retain(std::ptr::from_ref(event).cast_mut()) }
            .ok_or(MetalKernelError::QueueUnavailable("event retain refused"))?;
        Ok(Self { event, value })
    }

    fn produced(&self) -> bool {
        self.event.signaledValue() >= self.value
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

    /// Add `input`; an input on the same event keeps the higher value (an
    /// event's values are monotone, so the higher one covers both).
    fn add(&mut self, input: Input) -> Result<(), Input> {
        if let Some(same) = self
            .inputs
            .iter_mut()
            .flatten()
            .find(|held| held.is_on(&input.event))
        {
            same.value = same.value.max(input.value);
            return Ok(());
        }
        match self.inputs.iter_mut().find(|slot| slot.is_none()) {
            Some(slot) => {
                *slot = Some(input);
                Ok(())
            }
            None => Err(input),
        }
    }

    /// An input on `event` this recording still waits for.
    fn waits_on(&self, event: &ProtocolObject<dyn MTLSharedEvent>) -> bool {
        self.inputs
            .iter()
            .flatten()
            .any(|input| input.is_on(event) && !input.produced())
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
    /// Summed time producers waited for the state lock, µs.
    pub producer_lock_us: u64,
}

struct Listener {
    listener: Retained<objc2_metal::MTLSharedEventListener>,
    block: block2::RcBlock<dyn Fn(NonNull<ProtocolObject<dyn MTLSharedEvent>>, u64)>,
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
    late_logged: AtomicBool,
    /// The event and highest value of the last cancellation.
    cancelled_event: AtomicUsize,
    cancelled_upto: AtomicU64,
    /// Why the last cancellation happened: a waiter on a cancelled
    /// recording reads it in its error.
    cancel_reason: parking_lot::Mutex<&'static str>,
    /// Wakes the queue for an input the GPU or the IO queue produces.
    listener: OnceLock<Listener>,
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
            late_logged: AtomicBool::new(false),
            cancelled_event: AtomicUsize::new(0),
            cancelled_upto: AtomicU64::new(0),
            cancel_reason: parking_lot::Mutex::new("cancelled before its input was produced"),
            listener: OnceLock::new(),
        }
    }

    /// The error a wait on a cancelled recording returns.
    pub(super) fn cancelled_error(&self) -> MetalKernelError {
        MetalKernelError::QueueUnavailable(*self.cancel_reason.lock())
    }

    #[cfg(test)]
    pub(super) fn late_logged_for_test(&self) -> bool {
        self.late_logged.load(Ordering::Relaxed)
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
    /// Each commit is the owner's submission transaction, with the epoch the
    /// recording's cut took; residency is published once per acquisition.
    pub(super) fn release_produced_locked(
        &self,
        state: &mut EntryGuard<'_>,
    ) -> Result<usize, MetalKernelError> {
        let mut released = 0usize;
        let result = loop {
            if !state.held.front().is_some_and(|held| held.gate.is_open()) {
                break Ok(());
            }
            // The producer wrote its input before it raised the event.
            std::sync::atomic::fence(Ordering::Acquire);
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
            let owed = std::mem::take(&mut entry.owed_epochs);
            let held = entry.held.front().expect("checked above");
            let authority = OwnerSubmission {
                buffer: &held.cb,
                clock: &self.clock,
            };
            let epoch = held.epoch;
            match self.submit_owned_recording(
                authority,
                removals,
                false,
                None,
                None,
                Some(epoch),
                1 + owed,
            ) {
                Ok(_) => {
                    custody.residency_published |= publish;
                    let held = entry.held.pop_front().expect("checked above");
                    self.produced.held_us.fetch_add(
                        held.at.elapsed().as_micros().min(u64::MAX as u128) as u64,
                        Ordering::Relaxed,
                    );
                    entry.in_flight.push(held.cb);
                    released += 1;
                    if let Err(error) = self.run_after_commit(held.after) {
                        break Err(error);
                    }
                }
                Err(error) => {
                    entry.owed_epochs += owed;
                    break Err(error);
                }
            }
        };
        let opened = state.held.is_empty() && !state.gate.is_empty() && state.gate.is_open();
        if opened {
            state.gate = Gate::default();
        }
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

    fn run_after_commit(&self, after: Vec<AfterCommit>) -> Result<(), MetalKernelError> {
        for action in after {
            match action {
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
    /// Wakes an owner parked with the state lock (a Metal 4 slot, a full
    /// held queue) before taking the lock, so it submits the released work
    /// itself rather than wait on this thread.
    pub(super) fn produced_on(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        value: u64,
    ) -> Result<usize, MetalKernelError> {
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
        let released = self.lock_entry().and_then(|mut state| {
            self.produced.producer_lock_us.fetch_add(
                asked.elapsed().as_micros().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            self.release_produced_locked(&mut state)
        });
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
    /// recorded, in order. `false` when a lock was busy; retried later.
    pub(super) fn release_held_for_teardown(
        &self,
        state: &mut EntryState,
        removals: &mut Option<PublishedRemovals>,
    ) -> Result<bool, MetalKernelError> {
        let had = state.held.len();
        while let Some(held) = state.held.front() {
            let owed = std::mem::take(&mut state.owed_epochs);
            let authority = OwnerSubmission {
                buffer: &held.cb,
                clock: &self.clock,
            };
            let staged = if removals.is_none() {
                Some(&mut *removals)
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
                1 + owed,
            ) {
                Ok(_) => {
                    let held = state.held.pop_front().expect("checked above");
                    state.in_flight.push(held.cb);
                    self.run_after_commit(held.after)?;
                }
                Err(MetalKernelError::RecordingAdmission(
                    crate::command_lifecycle::LifecycleError::Busy,
                )) => {
                    state.owed_epochs += owed;
                    return Ok(false);
                }
                Err(error) => {
                    state.owed_epochs += owed;
                    return Err(error);
                }
            }
        }
        if had > 0 || !state.gate.is_empty() {
            eprintln!(
                "[metal] teardown committed {had} held recording(s) whose input was never produced"
            );
        }
        state.gate = Gate::default();
        Ok(true)
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
        let input = Input::new(event, value)?;
        if state.gate.add(input).is_err() {
            return Err(MetalKernelError::QueueUnavailable(
                "a fresh recording's gate refused an input",
            ));
        }
        self.core.held_live_update(&state);
        // A producer that raised the event since the look above found the
        // count without this gate: the release below sees the event.
        self.core.release_produced_locked(&mut state)?;
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

    /// The input on `event` will never be produced (its producer failed, or
    /// its run ended early): cancel the held recordings waiting for it, every
    /// recording behind them and the current one. Nothing they encoded runs;
    /// their native work is abandoned through host retirement, their epochs
    /// complete with the next commit, and an IO load ordered behind them
    /// keeps its order through a signal re-issued on the queue. A wait on a
    /// cancelled recording fails with `reason`. Returns the recordings
    /// cancelled; 0 when nothing waits on `event`.
    pub fn cancel_produced(
        &self,
        event: &ProtocolObject<dyn MTLSharedEvent>,
        reason: &'static str,
    ) -> Result<usize, MetalKernelError> {
        let mut state = self.lock_recording_state()?;
        let first = state.held.iter().position(|held| held.gate.waits_on(event));
        if first.is_none() && !state.gate.waits_on(event) {
            return Ok(0);
        }
        let from = first.unwrap_or(state.held.len());
        let mut upto = state.gate.highest_on(event);
        let mut rehome: Vec<AfterCommit> = Vec::new();
        let mut cancelled = 0usize;
        let mut refused = None;
        {
            let (entry, custody) = state.parts_mut();
            let entry: &mut EntryState = entry;
            for held in entry.held.drain(from..) {
                upto = upto.max(held.gate.highest_on(event));
                if let Err(error) = held.cb.cancel_unsubmitted() {
                    refused.get_or_insert(error);
                }
                entry.owed_epochs += 1;
                for action in held.after {
                    let AfterCommit::Io { io, signal, .. } = action;
                    rehome.push(AfterCommit::Io {
                        io,
                        signal,
                        rehomed: true,
                    });
                }
                custody.cancelled.push(held.cb);
                cancelled += 1;
            }
        }
        // The current recording is behind every held one, or waits itself.
        self.core.close_pending_blit(&mut state)?;
        self.core.close_compute(&mut state)?;
        if let Some(current) = state.current.recording().cloned() {
            if let Err(error) = current.cancel_unsubmitted() {
                refused.get_or_insert(error);
            }
            cancelled += 1;
            state.gate = Gate::default();
            state.signals_since_cut = false;
            self.core.compute_count.store(0, Ordering::Release);
            state.current_abandoned = true;
            // Replaced now when a slot is free; else by the next recording.
            let replaced = self.replace_abandoned_locked(&mut state);
            if let Err(error) = replaced {
                refused.get_or_insert(error);
            }
        }
        // Loads ordered behind a cancelled signal keep that order.
        match state.held.back_mut() {
            Some(held) => held.after.extend(rehome),
            None => {
                if let Err(error) = self.core.run_after_commit(rehome) {
                    refused.get_or_insert(error);
                }
            }
        }
        let p = &self.core.produced;
        *p.cancel_reason.lock() = reason;
        p.cancelled.fetch_add(cancelled as u64, Ordering::Relaxed);
        p.cancelled_upto.store(upto, Ordering::Release);
        p.cancelled_event.store(
            event as *const ProtocolObject<dyn MTLSharedEvent> as *const () as usize,
            Ordering::Release,
        );
        self.core.held_live_update(&state);
        p.wake.notify();
        drop(state);
        if let Some(error) = refused {
            self.core
                .failure
                .record_static("a held recording could not be cancelled; queue failed");
            return Err(error);
        }
        Ok(cancelled)
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
            if core.release_produced_locked(state)? == 0 {
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
    /// a held cut.
    pub(super) fn cut_locked<'a>(
        &'a self,
        state: &mut EntryGuard<'a>,
        reset_to: usize,
    ) -> Result<Option<u64>, MetalKernelError> {
        if state.held.is_empty() && state.gate.is_open() {
            if !state.gate.is_empty() {
                state.gate = Gate::default();
                self.core.held_live_update(state);
            }
            return self.commit_swap_locked(state, reset_to);
        }
        self.hold_swap_locked(state, reset_to)?;
        Ok(None)
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
            if self.core.release_produced_locked(state)? > 0 {
                continue;
            }
            self.core.failure.check()?;
            self.core.refuse_park_in_callback()?;
            // Parked holding the state lock: a producer wakes this before it
            // takes the lock, and the release above then runs here.
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
            // No slot comes back before the held recordings run: submit
            // what producers released, then wait for a slot.
            if core.release_produced_locked(state)? == 0 {
                core.produced.parks.fetch_add(1, Ordering::Relaxed);
                ready.wait(seen);
            }
            Ok(())
        })?;
        let epoch = self.core.clock.commit_epoch();
        let gate = std::mem::take(&mut state.gate);
        let old = match state.current.replace_recording(next) {
            Ok(old) => old,
            Err(_) => unreachable!("checked recording above"),
        };
        state.held.push_back(Held {
            cb: old,
            gate,
            epoch,
            after: Vec::new(),
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
