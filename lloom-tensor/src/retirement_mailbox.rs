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
// is within CallbackActive. Reclamation requires Taking and no callback borrow.
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

    pub fn publish_feedback(&self, ticket: Ticket, feedback: F) -> Result<(), Rejected<F>> {
        if let Err(error) = self.transition(ticket, Phase::Pending, Phase::WritingFeedback, 0) {
            return Err(Rejected {
                error,
                value: feedback,
            });
        }
        // SAFETY: WritingFeedback is exclusively ours; feedback was empty.
        // Handoff-return changes only atomic flags, never this value.
        unsafe {
            *self.feedback.get() = Some(feedback);
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
        run: impl FnOnce(&T, &F),
        wake_all: impl FnOnce(),
    ) -> Result<(), MailboxError> {
        self.transition(ticket, Phase::Ready, Phase::CallbackActive, 0)?;
        // SAFETY: Ready publication makes both initialized values visible.
        // CallbackActive excludes detach/reuse; these references do not escape
        // the closure call. Native handoff-return touches only atomic flags.
        let result = catch_unwind(AssertUnwindSafe(|| unsafe {
            run(
                (*self.owner.get()).as_ref().expect("registered owner"),
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

    pub fn try_detach_cleanup(&self, ticket: Ticket) -> Result<Detached<T, F>, MailboxError> {
        let state = self.state_for(ticket)?;
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
    pub fn pin(&self, ticket: Ticket) -> Result<MetadataPin<'_, T, F>, MailboxError> {
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
            return Ok(MetadataPin {
                mailbox: self,
                ticket,
            });
        }
    }
}

pub(crate) struct MetadataPin<'a, T, F> {
    mailbox: &'a OwnerMailbox<T, F>,
    ticket: Ticket,
}
impl<T, F> MetadataPin<'_, T, F> {
    pub fn ticket(&self) -> Ticket {
        self.ticket
    }
    pub fn try_clone(&self) -> Result<Self, MailboxError> {
        self.mailbox
            .readers
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
            .map_err(|_| MailboxError::Busy)?;
        Ok(Self {
            mailbox: self.mailbox,
            ticket: self.ticket,
        })
    }
}
impl<T, F> Drop for MetadataPin<'_, T, F> {
    fn drop(&mut self) {
        if self.mailbox.readers.fetch_sub(1, Ordering::Release) == 1 {
            self.mailbox.wake.notify();
        }
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
        drop(cell.cancel_prepared(old).unwrap());
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
