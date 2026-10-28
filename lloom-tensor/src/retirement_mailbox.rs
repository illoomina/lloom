//! Atomic owner transfer for host registration, native feedback and host cleanup.
//!
//! This cell does not own a native queue or create a cleanup worker. A native
//! adapter must root its containing slab independently until every owner leaves
//! it, and must invoke callbacks only after the shared prefix collector elects
//! this record. Native block tokens own metadata, never a payload.

use crate::completion_wake::Wake;
use crate::retirement_owners::{Failure, FailureSignal};
use std::{
    any::Any,
    cell::UnsafeCell,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

const PHASE_MASK: u64 = 15;
const RETURNED: u64 = 16;
const UNKNOWN: u64 = 32;
const SLOT_RETURNED: u64 = 64;
const PINNING: u64 = 128;
const GENERATION_SHIFT: u32 = 8;
const MAX_GENERATION: u64 = u64::MAX >> GENERATION_SHIFT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u64)]
enum Phase {
    Empty = 0,
    Preparing = 1,
    Prepared = 2,
    Pending = 3,
    WritingFeedback = 4,
    Ready = 5,
    CallbackActive = 6,
    CallbacksFinished = 7,
    CleanupReady = 8,
    Taking = 9,
    Detached = 10,
    CallbackFailed = 11,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ticket {
    mailbox: u64,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MailboxError {
    Busy,
    Stale,
    WrongPhase,
    GenerationExhausted,
    IdentityExhausted,
    CallbackPanicked,
    Failed(Failure),
}

#[derive(Debug)]
pub(crate) struct Rejected<T> {
    pub error: MailboxError,
    pub value: T,
}

/// Rejection preserves caller ownership. Observer failure occurs only after
/// feedback moved into rooted custody; that value is deliberately not returned.
#[derive(Debug)]
pub(crate) enum FeedbackPublicationError<F> {
    Rejected(Rejected<F>),
    ObserverPanicked,
}

/// Before phase acquisition, return the caller's unregistered input unchanged.
/// After an internal editor panic, existing registered owners/panic stay rooted.
pub(crate) enum PreparedEditError<V> {
    Rejected(Rejected<V>),
    Panicked { input: Option<V> },
}

/// Generic protocol retains its terminal-feedback cleanup capability. Native
/// adapters that also require host call-return choose the stricter policy.
#[derive(Clone, Copy)]
pub(crate) enum CleanupPolicy {
    TerminalFeedback,
    ReturnedHandoff,
}

pub(crate) struct Detached<T, F> {
    pub owner: T,
    pub feedback: Option<F>,
}

/// The generation is part of every phase CAS: a stale native token cannot
/// acquire a newer occupant between a separate generation check and phase claim.
/// Reuse also waits for metadata readers, which never borrow resource payloads.
pub(crate) struct OwnerMailbox<T, F> {
    identity: u64,
    state: AtomicU64,
    readers: AtomicUsize,
    owner: UnsafeCell<Option<T>>,
    feedback: UnsafeCell<Option<F>>,
    panic: UnsafeCell<Option<Box<dyn Any + Send>>>,
    failure: Arc<FailureSignal>,
    wake: Arc<Wake>,
}

// SAFETY: phase ownership serializes writes and the only shared payload borrow
// is within WritingFeedback (shared) or CallbackActive (mutable), each with
// exactly one phase owner. Reclamation
// requires Taking and no callback borrow.
// T/F cross threads and may be shared by a callback, hence Send + Sync.
unsafe impl<T: Send + Sync, F: Send + Sync> Send for OwnerMailbox<T, F> {}
unsafe impl<T: Send + Sync, F: Send + Sync> Sync for OwnerMailbox<T, F> {}

impl<T, F> OwnerMailbox<T, F> {
    pub fn new(failure: Arc<FailureSignal>, wake: Arc<Wake>) -> Result<Self, MailboxError> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(0);
        let identity = NEXT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| MailboxError::IdentityExhausted)?;
        Ok(Self {
            identity,
            state: AtomicU64::new(Phase::Empty as u64),
            readers: AtomicUsize::new(0),
            owner: UnsafeCell::new(None),
            feedback: UnsafeCell::new(None),
            panic: UnsafeCell::new(None),
            failure,
            wake,
        })
    }

    pub fn prepare(&self, owner: T) -> Result<Ticket, Rejected<T>> {
        let reject = |error, value| Err(Rejected { error, value });
        if let Some(failure) = self.failure.get() {
            return reject(MailboxError::Failed(failure), owner);
        }
        let state = self.state.load(Ordering::Acquire);
        if !matches!(state & PHASE_MASK, x if x == Phase::Empty as u64 || x == Phase::Detached as u64)
            || state & PINNING != 0
        {
            return reject(MailboxError::Busy, owner);
        }
        let generation = state >> GENERATION_SHIFT;
        if generation == MAX_GENERATION {
            return reject(MailboxError::GenerationExhausted, owner);
        }
        // Initial pins cannot start on Empty/Detached. A pin that began earlier
        // publishes its count before clearing PINNING; existing pin clones
        // cannot resurrect zero. Do not change generation while readers remain.
        if self.readers.load(Ordering::Acquire) != 0 {
            return reject(MailboxError::Busy, owner);
        }
        let preparing = ((generation + 1) << GENERATION_SHIFT) | Phase::Preparing as u64;
        if self
            .state
            .compare_exchange(state, preparing, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return reject(MailboxError::Busy, owner);
        }
        // SAFETY: Preparing is exclusively ours. Empty/Detached have no payload;
        // metadata readers ended and native tickets cannot claim Preparing.
        unsafe {
            *self.owner.get() = Some(owner);
        }
        let ticket = Ticket {
            mailbox: self.identity,
            generation: generation + 1,
        };
        self.state.store(
            (ticket.generation << GENERATION_SHIFT) | Phase::Prepared as u64,
            Ordering::Release,
        );
        Ok(ticket)
    }

    /// Install a payload into capacity reserved before submission side effects.
    /// No arbitrary callback or destructor runs while the phase is exclusive;
    /// the displaced value returns to caller custody.
    pub(crate) fn replace_prepared(&self, ticket: Ticket, owner: T) -> Result<T, Rejected<T>> {
        if let Err(error) = self.transition(ticket, Phase::Prepared, Phase::Preparing, 0) {
            return Err(Rejected {
                error,
                value: owner,
            });
        }
        // SAFETY: Preparing excludes callbacks, cancellation and handoff. Pins
        // borrow metadata only. A prepared cell always contains one owner.
        let previous = unsafe { (*self.owner.get()).replace(owner).expect("prepared owner") };
        self.transition(ticket, Phase::Preparing, Phase::Prepared, 0)
            .expect("exclusive prepared replacement retains its phase");
        Ok(previous)
    }

    /// Host-only trusted recording edit. Preparing excludes native handoff,
    /// cancellation and every other payload borrow; this never runs a user's
    /// callback. Input stays outside catch_unwind until trusted editor takes it
    /// after fallible preparation. The editor must not panic after taking input
    /// and before installing it in owner. Every unconsumed input is returned,
    /// including panic, for caller disposal outside recording/commit locks.
    pub(crate) fn edit_prepared<V, R>(
        &self,
        ticket: Ticket,
        input: V,
        edit: impl FnOnce(&mut T, &mut Option<V>) -> R,
        wake_all: impl FnOnce(),
    ) -> Result<(R, Option<V>), PreparedEditError<V>> {
        if let Err(error) = self.transition(ticket, Phase::Prepared, Phase::Preparing, 0) {
            return Err(PreparedEditError::Rejected(Rejected {
                error,
                value: input,
            }));
        }
        let mut input = Some(input);
        let result = catch_unwind(AssertUnwindSafe(|| {
            // SAFETY: this edit exclusively owns Preparing and its reference
            // cannot escape the closure. Input has not entered native custody.
            edit(
                unsafe { (*self.owner.get()).as_mut().expect("prepared owner") },
                &mut input,
            )
        }));
        match result {
            Ok(value) => {
                self.transition(ticket, Phase::Preparing, Phase::Prepared, 0)
                    .expect("exclusive recording edit retains phase");
                Ok((value, input))
            }
            Err(payload) => {
                // SAFETY: editor unwound, exclusive phase retains owner.
                unsafe {
                    *self.panic.get() = Some(payload);
                }
                self.transition(ticket, Phase::Preparing, Phase::CallbackFailed, 0)
                    .expect("failed recording edit retains phase");
                self.failure.publish(Failure::CallbackPanic, || {
                    self.wake.notify();
                    wake_all();
                });
                Err(PreparedEditError::Panicked { input })
            }
        }
    }

    fn state_for(&self, ticket: Ticket) -> Result<u64, MailboxError> {
        let state = self.state.load(Ordering::Acquire);
        if ticket.mailbox != self.identity || state >> GENERATION_SHIFT != ticket.generation {
            return Err(MailboxError::Stale);
        }
        Ok(state)
    }

    fn transition(
        &self,
        ticket: Ticket,
        from: Phase,
        to: Phase,
        required: u64,
    ) -> Result<(), MailboxError> {
        loop {
            let state = self.state_for(ticket)?;
            if state & PHASE_MASK != from as u64 || state & required != required {
                return Err(MailboxError::WrongPhase);
            }
            let next = (state & !PHASE_MASK) | to as u64;
            if self
                .state
                .compare_exchange_weak(state, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    pub fn begin_handoff(&self, ticket: Ticket) -> Result<(), MailboxError> {
        if let Some(failure) = self.failure.get() {
            return Err(MailboxError::Failed(failure));
        }
        self.transition(ticket, Phase::Prepared, Phase::Pending, 0)
    }

    fn handoff_exited(&self, ticket: Ticket, flag: u64) -> Result<(), MailboxError> {
        loop {
            let state = self.state_for(ticket)?;
            let phase = state & PHASE_MASK;
            if phase < Phase::Pending as u64
                || phase == Phase::Taking as u64
                || phase == Phase::Detached as u64
                || state & (RETURNED | UNKNOWN) != 0
            {
                return Err(MailboxError::WrongPhase);
            }
            if self
                .state
                .compare_exchange_weak(state, state | flag, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    pub fn handoff_returned(&self, ticket: Ticket) -> Result<(), MailboxError> {
        self.handoff_exited(ticket, RETURNED)?;
        self.wake.notify();
        Ok(())
    }

    pub fn handoff_unknown(
        &self,
        ticket: Ticket,
        wake_all: impl FnOnce(),
    ) -> Result<(), MailboxError> {
        self.handoff_exited(ticket, UNKNOWN)?;
        self.failure.publish(Failure::UnknownHandoff, || {
            self.wake.notify();
            wake_all();
        });
        Ok(())
    }

    #[cfg(test)]
    pub fn publish_feedback(&self, ticket: Ticket, feedback: F) -> Result<(), Rejected<F>> {
        match self.publish_feedback_observed(ticket, feedback, |_, _| {}, || {}) {
            Ok(()) => Ok(()),
            Err(FeedbackPublicationError::Rejected(rejected)) => Err(rejected),
            Err(FeedbackPublicationError::ObserverPanicked) => {
                unreachable!("the default feedback observer does not execute code")
            }
        }
    }

    /// Internal native data/error publication, never a user callback. The
    /// WritingFeedback phase exclusively borrows initialized owner and feedback;
    /// logical callbacks and cleanup cannot enter until this observer returns.
    pub(crate) fn publish_feedback_observed(
        &self,
        ticket: Ticket,
        feedback: F,
        observe: impl FnOnce(&T, &F),
        wake_all: impl FnOnce(),
    ) -> Result<(), FeedbackPublicationError<F>> {
        if let Err(error) = self.transition(ticket, Phase::Pending, Phase::WritingFeedback, 0) {
            return Err(FeedbackPublicationError::Rejected(Rejected {
                error,
                value: feedback,
            }));
        }
        // SAFETY: WritingFeedback is exclusively ours; feedback was empty.
        // Handoff-return changes only atomic flags, never this value.
        unsafe {
            *self.feedback.get() = Some(feedback);
        }
        // SAFETY: WritingFeedback owns both fields, excludes replacement,
        // callbacks, detach and reuse; handoff exit changes atomic flags only.
        // The shared references cannot outlive observe. Catch before leaving
        // this phase so even observer panic retains the moved feedback/owner.
        let observed = catch_unwind(AssertUnwindSafe(|| unsafe {
            observe(
                (*self.owner.get()).as_ref().expect("registered owner"),
                (*self.feedback.get()).as_ref().expect("published feedback"),
            );
        }));
        if let Err(payload) = observed {
            // SAFETY: exclusive writer; no previous panic in this phase.
            unsafe {
                *self.panic.get() = Some(payload);
            }
            self.transition(ticket, Phase::WritingFeedback, Phase::CallbackFailed, 0)
                .expect("failed feedback observer retains exclusive phase");
            self.failure.publish(Failure::CallbackPanic, || {
                self.wake.notify();
                wake_all();
            });
            return Err(FeedbackPublicationError::ObserverPanicked);
        }
        self.transition(ticket, Phase::WritingFeedback, Phase::Ready, 0)
            .expect("exclusive feedback writer retains its phase");
        Ok(())
    }

    /// Only the ordered drainer calls this. Rust panic is retained in the cell,
    /// with its owner/captures, for a later reviewed host-side quarantine policy.
    /// This does not catch or recover foreign exceptions.
    pub fn with_callbacks(
        &self,
        ticket: Ticket,
        run: impl FnOnce(&mut T, &F),
        wake_all: impl FnOnce(),
    ) -> Result<(), MailboxError> {
        self.transition(ticket, Phase::Ready, Phase::CallbackActive, 0)?;
        // SAFETY: Ready publication makes both initialized values visible.
        // CallbackActive excludes every other owner access, detach and reuse; references do not escape
        // the closure call. Native handoff-return touches only atomic flags.
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            run(
                (*self.owner.get()).as_mut().expect("registered owner"),
                (*self.feedback.get()).as_ref().expect("published feedback"),
            );
        }));
        match result {
            Ok(()) => self.transition(ticket, Phase::CallbackActive, Phase::CallbacksFinished, 0),
            Err(payload) => {
                // SAFETY: the callback borrow ended during unwinding, and this
                // exclusive callback owner is the sole writer of panic payload.
                unsafe {
                    *self.panic.get() = Some(payload);
                }
                self.transition(ticket, Phase::CallbackActive, Phase::CallbackFailed, 0)
                    .expect("exclusive callback retains its phase");
                self.failure.publish(Failure::CallbackPanic, || {
                    self.wake.notify();
                    wake_all();
                });
                Err(MailboxError::CallbackPanicked)
            }
        }
    }

    /// The native boundary retains an internal Rust panic without replacing a
    /// panic already caught during callbacks. A rejected box stays caller-owned.
    /// No claim of foreign exception recovery is made.
    pub(crate) fn retain_boundary_panic(
        &self,
        ticket: Ticket,
        payload: Box<dyn Any + Send>,
    ) -> Result<(), Box<dyn Any + Send>> {
        loop {
            let Ok(state) = self.state_for(ticket) else {
                return Err(payload);
            };
            let phase = state & PHASE_MASK;
            if !matches!(phase, x if x == Phase::Pending as u64
                || x == Phase::Ready as u64 || x == Phase::CallbacksFinished as u64
                || x == Phase::CleanupReady as u64)
            {
                return Err(payload);
            }
            let next = (state & !PHASE_MASK) | Phase::CallbackFailed as u64;
            if self
                .state
                .compare_exchange_weak(state, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // SAFETY: claimed failure phase excludes callbacks and cleanup.
                // These source phases never contain an earlier panic payload.
                unsafe {
                    *self.panic.get() = Some(payload);
                }
                return Ok(());
            }
        }
    }

    pub fn slot_returned(&self, ticket: Ticket) -> Result<(), MailboxError> {
        loop {
            let state = self.state_for(ticket)?;
            if state & PHASE_MASK != Phase::CallbacksFinished as u64 || state & SLOT_RETURNED != 0 {
                return Err(MailboxError::WrongPhase);
            }
            if self
                .state
                .compare_exchange_weak(
                    state,
                    state | SLOT_RETURNED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    pub fn publish_cleanup(&self, ticket: Ticket) -> Result<(), MailboxError> {
        self.transition(
            ticket,
            Phase::CallbacksFinished,
            Phase::CleanupReady,
            SLOT_RETURNED,
        )?;
        self.wake.notify();
        Ok(())
    }

    #[cfg(test)]
    pub fn try_detach_cleanup(&self, ticket: Ticket) -> Result<Detached<T, F>, MailboxError> {
        self.try_detach_cleanup_with_policy(ticket, CleanupPolicy::TerminalFeedback)
    }

    pub(crate) fn try_detach_cleanup_with_policy(
        &self,
        ticket: Ticket,
        policy: CleanupPolicy,
    ) -> Result<Detached<T, F>, MailboxError> {
        let state = self.state_for(ticket)?;
        if matches!(policy, CleanupPolicy::ReturnedHandoff)
            && (state & RETURNED == 0 || state & UNKNOWN != 0)
        {
            return Err(MailboxError::WrongPhase);
        }
        // RETURNED and UNKNOWN are mutually exclusive final handoff facts.
        // Eligibility cannot change from returned to unknown before Taking.
        if state & (RETURNED | UNKNOWN) == 0 {
            return Err(MailboxError::WrongPhase);
        }
        self.transition(ticket, Phase::CleanupReady, Phase::Taking, SLOT_RETURNED)?;
        self.take(ticket)
    }

    /// Host cleanup observes only an already published cleanup phase. The
    /// generation remains checked again by try_detach_cleanup.
    pub(crate) fn cleanup_ticket(&self) -> Option<Ticket> {
        let state = self.state.load(Ordering::Acquire);
        (state & PHASE_MASK == Phase::CleanupReady as u64).then_some(Ticket {
            mailbox: self.identity,
            generation: state >> GENERATION_SHIFT,
        })
    }

    /// A registration that never began native handoff has no GPU dependency.
    /// Publish it for host destruction rather than dropping arbitrary captures
    /// while the producer still owns its recording mutex.
    pub(crate) fn publish_cancelled(&self, ticket: Ticket) -> Result<(), MailboxError> {
        loop {
            let state = self.state_for(ticket)?;
            if state & PHASE_MASK != Phase::Prepared as u64 {
                return Err(MailboxError::WrongPhase);
            }
            let next =
                (state & !PHASE_MASK) | Phase::CleanupReady as u64 | RETURNED | SLOT_RETURNED;
            if self
                .state
                .compare_exchange_weak(state, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                self.wake.notify();
                return Ok(());
            }
        }
    }

    #[cfg(test)]
    pub fn cancel_prepared(&self, ticket: Ticket) -> Result<Detached<T, F>, MailboxError> {
        self.transition(ticket, Phase::Prepared, Phase::Taking, 0)?;
        self.take(ticket)
    }

    fn take(&self, ticket: Ticket) -> Result<Detached<T, F>, MailboxError> {
        // SAFETY: Taking is exclusively ours and neither callback nor native
        // writer can access payloads. Returned values leave the cell before
        // publication of Detached permits metadata reuse.
        let detached = unsafe {
            Detached {
                owner: (*self.owner.get()).take().expect("registered owner"),
                feedback: (*self.feedback.get()).take(),
            }
        };
        self.transition(ticket, Phase::Taking, Phase::Detached, 0)
            .expect("exclusive cleanup owner retains its phase");
        self.wake.notify();
        Ok(detached)
    }

    /// Initial acquisition is phase-claimed; cloning can start only from an
    /// existing live reader. A native block must use Ticket, not keep this pin.
    pub(crate) fn acquire_pin(&self, ticket: Ticket) -> Result<PinPermit, MailboxError> {
        loop {
            let state = self.state_for(ticket)?;
            let phase = state & PHASE_MASK;
            if matches!(phase, x if x == Phase::Empty as u64 || x == Phase::Preparing as u64 || x == Phase::Taking as u64 || x == Phase::Detached as u64)
            {
                return Err(MailboxError::WrongPhase);
            }
            if state & PINNING != 0 {
                std::hint::spin_loop();
                continue;
            }
            if self
                .state
                .compare_exchange_weak(state, state | PINNING, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            let incremented = self
                .readers
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1));
            self.state.fetch_and(!PINNING, Ordering::Release);
            if incremented.is_err() {
                return Err(MailboxError::Busy);
            }
            return Ok(PinPermit { ticket });
        }
    }

    pub(crate) fn metadata_reusable(&self) -> bool {
        let state = self.state.load(Ordering::Acquire);
        matches!(state & PHASE_MASK, x if x == Phase::Empty as u64 || x == Phase::Detached as u64)
            && state & PINNING == 0
            && self.readers.load(Ordering::Acquire) == 0
    }

    pub(crate) fn clone_pin(&self, permit: &PinPermit) -> Result<PinPermit, MailboxError> {
        self.state_for(permit.ticket)?;
        self.readers
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                if n == 0 {
                    None
                } else {
                    n.checked_add(1)
                }
            })
            .map_err(|_| MailboxError::Busy)?;
        Ok(PinPermit {
            ticket: permit.ticket,
        })
    }

    pub(crate) fn release_pin(&self, permit: PinPermit) {
        self.state_for(permit.ticket)
            .expect("a live permit prevents generation reuse");
        let previous = self.readers.fetch_sub(1, Ordering::Release);
        assert!(previous > 0, "one release per opaque pin permit");
        if previous == 1 {
            self.wake.notify();
        }
    }

    #[cfg(test)]
    pub fn pin(&self, ticket: Ticket) -> Result<MetadataPin<'_, T, F>, MailboxError> {
        Ok(MetadataPin {
            mailbox: self,
            permit: Some(self.acquire_pin(ticket)?),
        })
    }
}

/// Counted metadata claim, never a resource borrow. The fields are private and
/// the token cannot be copied. Dropping without release leaks reuse credit,
/// never releases another reader's claim or permits premature reuse.
pub(crate) struct PinPermit {
    ticket: Ticket,
}

#[cfg(test)]
pub(crate) struct MetadataPin<'a, T, F> {
    mailbox: &'a OwnerMailbox<T, F>,
    permit: Option<PinPermit>,
}
#[cfg(test)]
impl<T, F> MetadataPin<'_, T, F> {
    pub fn ticket(&self) -> Ticket {
        self.permit.as_ref().expect("live pin").ticket
    }
    pub fn try_clone(&self) -> Result<Self, MailboxError> {
        Ok(Self {
            mailbox: self.mailbox,
            permit: Some(
                self.mailbox
                    .clone_pin(self.permit.as_ref().expect("live pin"))?,
            ),
        })
    }
}
#[cfg(test)]
impl<T, F> Drop for MetadataPin<'_, T, F> {
    fn drop(&mut self) {
        self.mailbox
            .release_pin(self.permit.take().expect("one pin release"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::mpsc::sync_channel,
        time::{Duration, Instant},
    };

    fn cell<T, F>() -> OwnerMailbox<T, F> {
        OwnerMailbox::new(
            Arc::new(FailureSignal::default()),
            Arc::new(Wake::default()),
        )
        .unwrap()
    }
    fn finish<T, F>(cell: &OwnerMailbox<T, F>, ticket: Ticket, feedback: F) {
        cell.publish_feedback(ticket, feedback)
            .unwrap_or_else(|_| panic!("feedback rejected"));
        cell.with_callbacks(ticket, |_, _| {}, || {}).unwrap();
        cell.slot_returned(ticket).unwrap();
        cell.publish_cleanup(ticket).unwrap();
    }
    fn not_timeout(result: parking_lot_core::ParkResult, message: &str) {
        assert_ne!(result, parking_lot_core::ParkResult::TimedOut, "{message}");
    }

    #[test]
    fn prepared_editor_panic_returns_unconsumed_input_without_drop() {
        let owner_drops = Arc::new(AtomicUsize::new(0));
        let input_drops = Arc::new(AtomicUsize::new(0));
        let panic_drops = Arc::new(AtomicUsize::new(0));
        let cell = cell::<Counted, ()>();
        let ticket = cell.prepare(Counted(owner_drops.clone())).unwrap();
        let result = cell.edit_prepared(
            ticket,
            Counted(input_drops.clone()),
            |_, input| -> () {
                assert!(input.is_some());
                std::panic::panic_any(Counted(panic_drops.clone()));
            },
            || {},
        );
        let Err(PreparedEditError::Panicked { input: Some(input) }) = result else {
            panic!("editor panic failed to return unconsumed input");
        };
        assert_eq!(
            input_drops.load(Ordering::SeqCst),
            0,
            "input destructed under editor/caller lock"
        );
        assert_eq!(owner_drops.load(Ordering::SeqCst), 0);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 0);
        assert_eq!(cell.failure.get(), Some(Failure::CallbackPanic));
        assert!(matches!(
            cell.cancel_prepared(ticket),
            Err(MailboxError::WrongPhase)
        ));
        drop(input);
        assert_eq!(input_drops.load(Ordering::SeqCst), 1);
        assert_eq!(owner_drops.load(Ordering::SeqCst), 0);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 0);
        // Fake fixture lifetime ends explicitly; not a native quarantine release.
        drop(cell);
        assert_eq!(owner_drops.load(Ordering::SeqCst), 1);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn prepared_editor_transfers_after_reserve_and_returns_phase_refusal() {
        let accepted_drops = Arc::new(AtomicUsize::new(0));
        let refused_drops = Arc::new(AtomicUsize::new(0));
        let cell = cell::<Vec<Counted>, ()>();
        let ticket = cell.prepare(Vec::new()).unwrap();
        let result = cell.edit_prepared(
            ticket,
            Counted(accepted_drops.clone()),
            |owner, input| {
                owner.try_reserve(1).unwrap();
                owner.push(input.take().unwrap());
                owner.len()
            },
            || {},
        );
        assert!(matches!(result, Ok((1, None))));
        cell.begin_handoff(ticket).unwrap();
        let result = cell.edit_prepared(
            ticket,
            Counted(refused_drops.clone()),
            |_, _| panic!("refused editor ran"),
            || {},
        );
        let Err(PreparedEditError::Rejected(Rejected {
            error: MailboxError::WrongPhase,
            value,
        })) = result
        else {
            panic!("phase refusal lost caller input");
        };
        assert_eq!(accepted_drops.load(Ordering::SeqCst), 0);
        assert_eq!(refused_drops.load(Ordering::SeqCst), 0);
        drop(value);
        assert_eq!(refused_drops.load(Ordering::SeqCst), 1);
        cell.handoff_returned(ticket).unwrap();
        finish(&cell, ticket, ());
        let detached = cell.try_detach_cleanup(ticket).unwrap();
        assert_eq!(detached.owner.len(), 1);
        assert_eq!(accepted_drops.load(Ordering::SeqCst), 0);
        drop(detached);
        assert_eq!(accepted_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn prepared_replacement_returns_displaced_owner_and_rejects_after_handoff() {
        let drops = Arc::new(AtomicUsize::new(0));
        let cell = cell::<Option<Counted>, ()>();
        let ticket = cell.prepare(None).unwrap();
        assert!(cell
            .replace_prepared(ticket, Some(Counted(drops.clone())))
            .unwrap()
            .is_none());
        let previous = cell.replace_prepared(ticket, None).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(cell.replace_prepared(ticket, previous).unwrap().is_none());
        cell.begin_handoff(ticket).unwrap();
        let rejected = cell
            .replace_prepared(ticket, Some(Counted(drops.clone())))
            .unwrap_err();
        assert_eq!(rejected.error, MailboxError::WrongPhase);
        assert_eq!(
            drops.load(Ordering::SeqCst),
            0,
            "rejection released either owner"
        );
        drop(rejected.value);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        cell.handoff_returned(ticket).unwrap();
        cell.publish_feedback(ticket, ()).unwrap();
        let mut removed = None;
        cell.with_callbacks(
            ticket,
            |owner, _| {
                removed = owner.take();
                assert!(removed.is_some());
                assert!(matches!(
                    cell.try_detach_cleanup(ticket),
                    Err(MailboxError::WrongPhase)
                ));
            },
            || {},
        )
        .unwrap();
        cell.slot_returned(ticket).unwrap();
        cell.publish_cleanup(ticket).unwrap();
        let detached = cell.try_detach_cleanup(ticket).unwrap();
        assert!(
            detached.owner.is_none(),
            "mutable callback update was not retained"
        );
        drop(detached);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(removed);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn late_handoff_return_wakes_cleanup_after_early_feedback() {
        let cell = cell::<u64, u64>();
        let ticket = cell.prepare(17).unwrap();
        cell.begin_handoff(ticket).unwrap();
        finish(&cell, ticket, 23);
        std::thread::scope(|scope| {
            let (tx, rx) = sync_channel(0);
            let c = &cell;
            let service = scope.spawn(move || {
                let cell = c;
                let seen = cell.wake.snapshot();
                assert!(matches!(
                    cell.try_detach_cleanup(ticket),
                    Err(MailboxError::WrongPhase)
                ));
                tx.send(()).unwrap();
                not_timeout(
                    cell.wake
                        .park(seen, Some(Instant::now() + Duration::from_secs(2))),
                    "handoff return did not wake cleanup",
                );
                let detached = cell.try_detach_cleanup(ticket).unwrap();
                assert_eq!((detached.owner, detached.feedback), (17, Some(23)));
            });
            rx.recv_timeout(Duration::from_secs(2)).unwrap();
            cell.handoff_returned(ticket).unwrap();
            service.join().unwrap();
        });
    }

    #[derive(Debug)]
    struct Counted(Arc<AtomicUsize>);
    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn active_callback_borrow_excludes_cleanup_and_resource_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let cell = cell::<Counted, Counted>();
        let ticket = cell.prepare(Counted(drops.clone())).unwrap();
        cell.begin_handoff(ticket).unwrap();
        cell.handoff_returned(ticket).unwrap();
        cell.publish_feedback(ticket, Counted(drops.clone()))
            .unwrap();
        std::thread::scope(|scope| {
            let (entered_tx, entered_rx) = sync_channel(0);
            let (leave_tx, leave_rx) = sync_channel(0);
            let c = &cell;
            let callback = scope.spawn(move || {
                c.with_callbacks(
                    ticket,
                    |owner, feedback| {
                        entered_tx.send(()).unwrap();
                        leave_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                        assert!(Arc::ptr_eq(&owner.0, &feedback.0));
                    },
                    || {},
                )
                .unwrap()
            });
            entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
            assert!(matches!(
                cell.try_detach_cleanup(ticket),
                Err(MailboxError::WrongPhase)
            ));
            assert_eq!(cell.slot_returned(ticket), Err(MailboxError::WrongPhase));
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            leave_tx.send(()).unwrap();
            callback.join().unwrap();
        });
        cell.slot_returned(ticket).unwrap();
        cell.publish_cleanup(ticket).unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(cell.try_detach_cleanup(ticket).unwrap());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn final_metadata_pin_drop_wakes_reuse_and_stale_ticket_is_rejected() {
        let cell = cell::<u64, u64>();
        let old = cell.prepare(1).unwrap();
        let first = cell.pin(old).unwrap();
        let last = first.try_clone().unwrap();
        assert_eq!(last.ticket(), old);
        drop(first);
        {
            let detached = cell.cancel_prepared(old).unwrap();
            assert_eq!(detached.owner, 1);
            assert_eq!(detached.feedback, None);
        }
        std::thread::scope(|scope| {
            let (tx, rx) = sync_channel(0);
            let c = &cell;
            let service = scope.spawn(move || {
                let cell = c;
                let seen = cell.wake.snapshot();
                let rejected = cell.prepare(2).unwrap_err();
                assert_eq!((rejected.error, rejected.value), (MailboxError::Busy, 2));
                tx.send(()).unwrap();
                not_timeout(
                    cell.wake
                        .park(seen, Some(Instant::now() + Duration::from_secs(2))),
                    "last metadata pin did not wake reuse",
                );
                let next = cell.prepare(rejected.value).unwrap();
                assert_eq!(next.generation, old.generation + 1);
                assert_eq!(cell.begin_handoff(old), Err(MailboxError::Stale));
                // A delayed native block owns this copyable metadata token only.
                let delayed_empty_block = Box::new(old);
                drop(delayed_empty_block);
                assert_eq!(cell.cancel_prepared(next).unwrap().owner, 2);
            });
            rx.recv_timeout(Duration::from_secs(2)).unwrap();
            drop(last);
            service.join().unwrap();
        });
    }

    #[test]
    fn foreign_ticket_duplicate_feedback_and_busy_prepare_return_ownership() {
        let drops = Arc::new(AtomicUsize::new(0));
        let a = cell::<Counted, Counted>();
        let b = cell::<Counted, Counted>();
        let ta = a.prepare(Counted(drops.clone())).unwrap();
        let tb = b.prepare(Counted(drops.clone())).unwrap();
        assert_eq!(ta.generation, tb.generation);
        b.begin_handoff(tb).unwrap();
        let rejected = b.publish_feedback(ta, Counted(drops.clone())).unwrap_err();
        assert_eq!(rejected.error, MailboxError::Stale);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        b.publish_feedback(tb, rejected.value).unwrap();
        let duplicate = b.publish_feedback(tb, Counted(drops.clone())).unwrap_err();
        assert_eq!(duplicate.error, MailboxError::WrongPhase);
        let busy = a.prepare(Counted(drops.clone())).unwrap_err();
        assert_eq!(busy.error, MailboxError::Busy);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(duplicate.value);
        drop(busy.value);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        drop(a.cancel_prepared(ta).unwrap());
        b.handoff_returned(tb).unwrap();
        b.with_callbacks(tb, |_, _| {}, || {}).unwrap();
        b.slot_returned(tb).unwrap();
        b.publish_cleanup(tb).unwrap();
        drop(b.try_detach_cleanup(tb).unwrap());
        assert_eq!(drops.load(Ordering::SeqCst), 5);
    }

    #[test]
    fn unknown_handoff_needs_actual_feedback_and_refuses_future_registration() {
        let cell = cell::<u64, u64>();
        let ticket = cell.prepare(1).unwrap();
        cell.begin_handoff(ticket).unwrap();
        let seen = cell.wake.snapshot();
        cell.handoff_unknown(ticket, || {}).unwrap();
        assert_ne!(seen, cell.wake.snapshot());
        assert_eq!(cell.failure.get(), Some(Failure::UnknownHandoff));
        assert!(matches!(
            cell.cancel_prepared(ticket),
            Err(MailboxError::WrongPhase)
        ));
        assert!(matches!(
            cell.try_detach_cleanup(ticket),
            Err(MailboxError::WrongPhase)
        ));
        assert_eq!(
            cell.prepare(2).unwrap_err().error,
            MailboxError::Failed(Failure::UnknownHandoff)
        );
        finish(&cell, ticket, 3);
        assert_eq!(cell.try_detach_cleanup(ticket).unwrap().feedback, Some(3));
        assert_eq!(cell.failure.get(), Some(Failure::UnknownHandoff));
    }

    #[test]
    fn callback_panic_retains_owner_and_panic_until_host_custody_ends() {
        let owner_drops = Arc::new(AtomicUsize::new(0));
        let panic_drops = Arc::new(AtomicUsize::new(0));
        let cell = cell::<Counted, u64>();
        let ticket = cell.prepare(Counted(owner_drops.clone())).unwrap();
        cell.begin_handoff(ticket).unwrap();
        cell.handoff_returned(ticket).unwrap();
        cell.publish_feedback(ticket, 7).unwrap();
        let seen = cell.wake.snapshot();
        assert_eq!(
            cell.with_callbacks(
                ticket,
                |_, _| std::panic::panic_any(Counted(panic_drops.clone())),
                || {}
            ),
            Err(MailboxError::CallbackPanicked)
        );
        assert_ne!(seen, cell.wake.snapshot());
        assert_eq!(cell.failure.get(), Some(Failure::CallbackPanic));
        assert!(matches!(
            cell.try_detach_cleanup(ticket),
            Err(MailboxError::WrongPhase)
        ));
        assert_eq!(cell.slot_returned(ticket), Err(MailboxError::WrongPhase));
        assert_eq!(owner_drops.load(Ordering::SeqCst), 0);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 0);
        // This CPU fixture is the sole host custody owner, not a native adapter.
        drop(cell);
        assert_eq!(owner_drops.load(Ordering::SeqCst), 1);
        assert_eq!(panic_drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn boundary_panic_claims_only_nonborrowed_phases_and_keeps_rejections_owned() {
        for phase in 0..4 {
            let owners = Arc::new(AtomicUsize::new(0));
            let panics = Arc::new(AtomicUsize::new(0));
            let cell = cell::<Counted, u64>();
            let ticket = cell.prepare(Counted(owners.clone())).unwrap();
            cell.begin_handoff(ticket).unwrap();
            cell.handoff_returned(ticket).unwrap();
            if phase >= 1 {
                cell.publish_feedback(ticket, 9).unwrap();
            }
            if phase >= 2 {
                cell.with_callbacks(ticket, |_, _| {}, || {}).unwrap();
            }
            if phase >= 3 {
                cell.slot_returned(ticket).unwrap();
                cell.publish_cleanup(ticket).unwrap();
            }
            assert!(cell
                .retain_boundary_panic(ticket, Box::new(Counted(panics.clone())))
                .is_ok());
            let rejected = cell
                .retain_boundary_panic(ticket, Box::new(Counted(panics.clone())))
                .unwrap_err();
            assert_eq!(panics.load(Ordering::SeqCst), 0);
            assert!(matches!(
                cell.try_detach_cleanup(ticket),
                Err(MailboxError::WrongPhase)
            ));
            drop(rejected);
            assert_eq!(panics.load(Ordering::SeqCst), 1);
            assert_eq!(owners.load(Ordering::SeqCst), 0);
            // CPU fixture custody only; production failed cells remain rooted.
            drop(cell);
            assert_eq!(owners.load(Ordering::SeqCst), 1);
            assert_eq!(panics.load(Ordering::SeqCst), 2);
        }
    }

    #[test]
    fn boundary_panic_cannot_mutate_active_callback_owner() {
        let cell = Arc::new(cell::<u64, u64>());
        let ticket = cell.prepare(10).unwrap();
        cell.begin_handoff(ticket).unwrap();
        cell.handoff_returned(ticket).unwrap();
        cell.publish_feedback(ticket, 7).unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let worker_cell = cell.clone();
        let worker = std::thread::spawn(move || {
            worker_cell.with_callbacks(
                ticket,
                |owner, feedback| {
                    started_tx.send(()).unwrap();
                    resume_rx
                        .recv_timeout(std::time::Duration::from_secs(2))
                        .unwrap();
                    *owner += *feedback;
                },
                || {},
            )
        });
        started_rx
            .recv_timeout(std::time::Duration::from_secs(2))
            .unwrap();
        let payload = cell
            .retain_boundary_panic(ticket, Box::new(99u64))
            .unwrap_err();
        assert_eq!(*payload.downcast::<u64>().unwrap(), 99);
        resume_tx.send(()).unwrap();
        worker.join().unwrap().unwrap();
        cell.slot_returned(ticket).unwrap();
        cell.publish_cleanup(ticket).unwrap();
        assert_eq!(cell.try_detach_cleanup(ticket).unwrap().owner, 17);
    }

    #[test]
    fn generation_exhaustion_returns_owner_without_wrapping() {
        let cell = cell::<u64, u64>();
        cell.state.store(
            (MAX_GENERATION << GENERATION_SHIFT) | Phase::Detached as u64,
            Ordering::Release,
        );
        let rejected = cell.prepare(19).unwrap_err();
        assert_eq!(
            (rejected.error, rejected.value),
            (MailboxError::GenerationExhausted, 19)
        );
        assert_eq!(
            cell.state.load(Ordering::Acquire) >> GENERATION_SHIFT,
            MAX_GENERATION
        );
    }

    struct Reentrant {
        action: Option<Box<dyn FnOnce() + Send + Sync>>,
    }
    impl Drop for Reentrant {
        fn drop(&mut self) {
            if let Some(action) = self.action.take() {
                action();
            }
        }
    }
    #[test]
    fn detached_credit_allows_destructor_to_queue_successor_without_recursive_drop() {
        let cell = Arc::new(cell::<Reentrant, ()>());
        let weak = Arc::downgrade(&cell);
        let (tx, rx) = sync_channel(1);
        let active = Arc::new(AtomicUsize::new(0));
        let successor_done = Arc::new(AtomicUsize::new(0));
        let active_outer = active.clone();
        let done_outer = successor_done.clone();
        let ticket = cell
            .prepare(Reentrant {
                action: Some(Box::new(move || {
                    assert_eq!(active_outer.fetch_add(1, Ordering::SeqCst), 0);
                    let cell = weak.upgrade().unwrap();
                    let active_next = active_outer.clone();
                    let next = cell
                        .prepare(Reentrant {
                            action: Some(Box::new(move || {
                                assert_eq!(
                                    active_next.load(Ordering::SeqCst),
                                    0,
                                    "recursive active destructor"
                                );
                                done_outer.fetch_add(1, Ordering::SeqCst);
                            })),
                        })
                        .unwrap_or_else(|_| panic!("detached credit unavailable to destructor"));
                    cell.begin_handoff(next).unwrap();
                    cell.handoff_returned(next).unwrap();
                    finish(&cell, next, ());
                    tx.send(next).unwrap();
                    assert_eq!(active_outer.fetch_sub(1, Ordering::SeqCst), 1);
                })),
            })
            .unwrap_or_else(|_| panic!("first prepare rejected"));
        cell.begin_handoff(ticket).unwrap();
        cell.handoff_returned(ticket).unwrap();
        finish(&cell, ticket, ());
        drop(cell.try_detach_cleanup(ticket).unwrap());
        let next = rx.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(successor_done.load(Ordering::SeqCst), 0);
        drop(cell.try_detach_cleanup(next).unwrap());
        assert_eq!(successor_done.load(Ordering::SeqCst), 1);
        assert_eq!(active.load(Ordering::SeqCst), 0);
    }
}
