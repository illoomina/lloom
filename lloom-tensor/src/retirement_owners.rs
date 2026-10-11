//! Bounded host-owned registration for native completion payloads.
//!
//! Private protocol core; not yet wired to Metal. The registry owner serializes
//! transitions and outlives native tokens. Tokens never own payloads. This is
//! distinct from CompletionFrontier (ordering) and does not implement a worker,
//! lock-free publication, native terminality detection, or a teardown policy.

use std::sync::{
    atomic::{AtomicU64, AtomicU8, Ordering},
    Arc,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Ticket {
    registry: u64,
    index: usize,
    generation: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Failure {
    UnknownHandoff = 1,
    CallbackPanic = 2,
    WorkerFailed = 3,
    Closing = 4,
}

/// Independently shareable notification predicate. A native adapter supplies
/// wake-all; publication precedes notification and never clears the first error.
#[derive(Default)]
pub(crate) struct FailureSignal(AtomicU8);
impl FailureSignal {
    pub fn get(&self) -> Option<Failure> {
        match self.0.load(Ordering::Acquire) {
            0 => None,
            1 => Some(Failure::UnknownHandoff),
            2 => Some(Failure::CallbackPanic),
            3 => Some(Failure::WorkerFailed),
            4 => Some(Failure::Closing),
            _ => unreachable!("only Failure discriminants are published"),
        }
    }

    pub fn publish(&self, failure: Failure, wake_all: impl FnOnce()) {
        let _ = self
            .0
            .compare_exchange(0, failure as u8, Ordering::AcqRel, Ordering::Acquire);
        wake_all();
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OwnerError {
    ZeroCapacity,
    Full,
    GenerationExhausted,
    RegistryExhausted,
    StaleTicket,
    WrongPhase,
    Failed(Failure),
}

#[derive(Debug)]
pub(crate) struct Rejected<T> {
    pub error: OwnerError,
    pub owner: T,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Handoff {
    Prepared,
    Started,
    Returned,
    Unknown,
}

struct Record<T> {
    owner: T,
    handoff: Handoff,
    native_terminal: bool,
    callback_active: bool,
    callbacks_finished: bool,
    slot_returned: bool,
    cleanup_ready: bool,
}

struct Cell<T> {
    generation: u64,
    occupied: bool,
    record: Option<Record<T>>,
}

pub(crate) struct OwnerCells<T, const N: usize> {
    registry: u64,
    cells: [Cell<T>; N],
    failure: Arc<FailureSignal>,
}

impl<T, const N: usize> OwnerCells<T, N> {
    pub fn new(failure: Arc<FailureSignal>) -> Result<Self, OwnerError> {
        if N == 0 {
            return Err(OwnerError::ZeroCapacity);
        }
        static NEXT_REGISTRY: AtomicU64 = AtomicU64::new(0);
        let registry = NEXT_REGISTRY
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| OwnerError::RegistryExhausted)?;
        Ok(Self {
            registry,
            cells: std::array::from_fn(|_| Cell {
                generation: 0,
                occupied: false,
                record: None,
            }),
            failure,
        })
    }

    /// Every rejection returns ownership. Reserving alone consumes no native
    /// submission sequence; the native adapter registers before its handoff.
    pub fn reserve(&mut self, owner: T) -> Result<Ticket, Rejected<T>> {
        if let Some(failure) = self.failure.get() {
            return Err(Rejected {
                error: OwnerError::Failed(failure),
                owner,
            });
        }
        let Some(index) = self
            .cells
            .iter()
            .position(|cell| !cell.occupied && cell.generation.checked_add(1).is_some())
        else {
            let error = if self.cells.iter().any(|cell| !cell.occupied) {
                OwnerError::GenerationExhausted
            } else {
                OwnerError::Full
            };
            return Err(Rejected { error, owner });
        };
        let cell = &mut self.cells[index];
        cell.generation += 1;
        cell.occupied = true;
        cell.record = Some(Record {
            owner,
            handoff: Handoff::Prepared,
            native_terminal: false,
            callback_active: false,
            callbacks_finished: false,
            slot_returned: false,
            cleanup_ready: false,
        });
        Ok(Ticket {
            registry: self.registry,
            index,
            generation: cell.generation,
        })
    }

    fn record(&mut self, ticket: Ticket) -> Result<&mut Record<T>, OwnerError> {
        if ticket.registry != self.registry {
            return Err(OwnerError::StaleTicket);
        }
        let cell = self
            .cells
            .get_mut(ticket.index)
            .ok_or(OwnerError::StaleTicket)?;
        if cell.generation != ticket.generation {
            return Err(OwnerError::StaleTicket);
        }
        cell.record.as_mut().ok_or(OwnerError::StaleTicket)
    }

    pub fn begin_handoff(&mut self, ticket: Ticket) -> Result<(), OwnerError> {
        if let Some(failure) = self.failure.get() {
            return Err(OwnerError::Failed(failure));
        }
        let record = self.record(ticket)?;
        if record.handoff != Handoff::Prepared {
            return Err(OwnerError::WrongPhase);
        }
        record.handoff = Handoff::Started;
        Ok(())
    }

    pub fn handoff_returned(&mut self, ticket: Ticket) -> Result<(), OwnerError> {
        let record = self.record(ticket)?;
        if record.handoff != Handoff::Started {
            return Err(OwnerError::WrongPhase);
        }
        // Do not overwrite native_terminal: feedback can arrive before return.
        record.handoff = Handoff::Returned;
        Ok(())
    }

    pub fn handoff_unknown(&mut self, ticket: Ticket) -> Result<(), OwnerError> {
        let record = self.record(ticket)?;
        if record.handoff != Handoff::Started {
            return Err(OwnerError::WrongPhase);
        }
        record.handoff = Handoff::Unknown;
        // Caller publishes failure/wakes outside its registry serialization.
        // No native completion or resource release is manufactured here.
        Ok(())
    }

    pub fn native_feedback(&mut self, ticket: Ticket) -> Result<(), OwnerError> {
        let record = self.record(ticket)?;
        if record.handoff == Handoff::Prepared || record.native_terminal {
            return Err(OwnerError::WrongPhase);
        }
        record.native_terminal = true;
        Ok(())
    }

    /// The ordered frontier calls this only for its elected next record. The
    /// borrow ends before cleanup publication; no owned payload enters a block.
    pub fn begin_callbacks(&mut self, ticket: Ticket) -> Result<&T, OwnerError> {
        let record = self.record(ticket)?;
        if !record.native_terminal || record.callback_active || record.callbacks_finished {
            return Err(OwnerError::WrongPhase);
        }
        record.callback_active = true;
        Ok(&record.owner)
    }

    pub fn callbacks_finished(&mut self, ticket: Ticket) -> Result<(), OwnerError> {
        let record = self.record(ticket)?;
        if !record.callback_active || record.callbacks_finished {
            return Err(OwnerError::WrongPhase);
        }
        record.callback_active = false;
        record.callbacks_finished = true;
        Ok(())
    }

    pub fn slot_returned(&mut self, ticket: Ticket) -> Result<(), OwnerError> {
        let record = self.record(ticket)?;
        if !record.callbacks_finished || record.slot_returned {
            return Err(OwnerError::WrongPhase);
        }
        record.slot_returned = true;
        Ok(())
    }

    pub fn publish_cleanup(&mut self, ticket: Ticket) -> Result<(), OwnerError> {
        let record = self.record(ticket)?;
        if !record.native_terminal
            || !record.callbacks_finished
            || !record.slot_returned
            || record.cleanup_ready
        {
            return Err(OwnerError::WrongPhase);
        }
        record.cleanup_ready = true;
        Ok(())
    }

    /// Detaching frees credit BEFORE the caller runs arbitrary payload Drop.
    /// The caller releases registry serialization before dropping the value.
    /// Thus storage is N pending records plus one active detached worker value.
    pub fn detach_cleanup(&mut self, ticket: Ticket) -> Result<T, OwnerError> {
        let record = self.record(ticket)?;
        if !record.cleanup_ready || record.handoff == Handoff::Started {
            return Err(OwnerError::WrongPhase);
        }
        let cell = &mut self.cells[ticket.index];
        let record = cell.record.take().expect("validated record");
        cell.occupied = false; // release credit before returning arbitrary Drop
        Ok(record.owner)
    }

    /// Only a positively known never-started registration can be canceled.
    pub fn cancel_prepared(&mut self, ticket: Ticket) -> Result<T, OwnerError> {
        if self.record(ticket)?.handoff != Handoff::Prepared {
            return Err(OwnerError::WrongPhase);
        }
        let cell = &mut self.cells[ticket.index];
        let owner = cell.record.take().expect("validated record").owner;
        cell.occupied = false;
        Ok(owner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        cell::{Cell as Count, RefCell},
        rc::{Rc, Weak},
    };

    fn ready<T, const N: usize>(cells: &mut OwnerCells<T, N>, ticket: Ticket) {
        cells.begin_handoff(ticket).unwrap();
        cells.handoff_returned(ticket).unwrap();
        cells.native_feedback(ticket).unwrap();
        cells.begin_callbacks(ticket).unwrap();
        cells.callbacks_finished(ticket).unwrap();
        cells.slot_returned(ticket).unwrap();
        cells.publish_cleanup(ticket).unwrap();
    }

    #[derive(Debug)]
    struct Owner(Rc<Count<usize>>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    fn native_return_does_not_erase_early_feedback_and_cleanup_requires_each_phase() {
        let mut cells = OwnerCells::<u64, 1>::new(Arc::default()).unwrap();
        let ticket = cells.reserve(17).unwrap();
        assert_eq!(cells.native_feedback(ticket), Err(OwnerError::WrongPhase));
        cells.begin_handoff(ticket).unwrap();
        cells.native_feedback(ticket).unwrap();
        cells.handoff_returned(ticket).unwrap();
        assert_eq!(*cells.begin_callbacks(ticket).unwrap(), 17);
        assert_eq!(cells.detach_cleanup(ticket), Err(OwnerError::WrongPhase));
        assert_eq!(cells.publish_cleanup(ticket), Err(OwnerError::WrongPhase));
        cells.callbacks_finished(ticket).unwrap();
        assert_eq!(cells.publish_cleanup(ticket), Err(OwnerError::WrongPhase));
        cells.slot_returned(ticket).unwrap();
        cells.publish_cleanup(ticket).unwrap();
        assert_eq!(cells.detach_cleanup(ticket), Ok(17));
    }

    #[test]
    fn early_complete_payload_stays_registered_until_native_handoff_exits() {
        let mut cells = OwnerCells::<u64, 1>::new(Arc::default()).unwrap();
        let ticket = cells.reserve(23).unwrap();
        cells.begin_handoff(ticket).unwrap();
        cells.native_feedback(ticket).unwrap();
        cells.begin_callbacks(ticket).unwrap();
        cells.callbacks_finished(ticket).unwrap();
        cells.slot_returned(ticket).unwrap();
        cells.publish_cleanup(ticket).unwrap();
        assert_eq!(cells.detach_cleanup(ticket), Err(OwnerError::WrongPhase));
        cells.handoff_returned(ticket).unwrap();
        assert_eq!(cells.detach_cleanup(ticket), Ok(23));
    }

    #[test]
    fn disposing_tokens_never_owns_payload_and_late_generations_cannot_touch_reuse() {
        // This Drop is the CPU stand-in for block disposal: its production
        // token contains only index/generation, not an owner or Queue Arc.
        struct BlockToken(Ticket);
        impl Drop for BlockToken {
            fn drop(&mut self) {
                let _ = self.0;
            }
        }
        let drops = Rc::new(Count::new(0));
        let mut cells = OwnerCells::<Owner, 1>::new(Arc::default()).unwrap();
        let old = cells.reserve(Owner(drops.clone())).unwrap();
        let early_block = BlockToken(old);
        let late_block = BlockToken(old);
        drop(early_block);
        assert_eq!(drops.get(), 0, "registry, not block, owns pending payload");
        ready(&mut cells, old);
        let owner = cells.detach_cleanup(old).unwrap();
        let new = cells.reserve(Owner(drops.clone())).unwrap();
        assert_ne!(new.generation, old.generation);
        drop(late_block);
        assert_eq!(drops.get(), 0);
        for reject in [
            cells.begin_handoff(old),
            cells.native_feedback(old),
            cells.publish_cleanup(old),
        ] {
            assert_eq!(reject, Err(OwnerError::StaleTicket));
        }
        assert_eq!(drops.get(), 0);
        ready(&mut cells, new);
        drop(cells.detach_cleanup(new).unwrap());
        drop(owner);
        assert_eq!(drops.get(), 2);
    }

    #[test]
    fn one_cell_detaches_credit_before_destructor_reserves_a_successor() {
        struct Reentrant {
            cells: Weak<RefCell<OwnerCells<Reentrant, 1>>>,
            recurse: bool,
            published: Rc<Count<Option<Ticket>>>,
            drops: Rc<Count<usize>>,
        }
        impl Drop for Reentrant {
            fn drop(&mut self) {
                if self.recurse {
                    let shared = self.cells.upgrade().unwrap();
                    let successor = Reentrant {
                        cells: self.cells.clone(),
                        recurse: false,
                        published: self.published.clone(),
                        drops: self.drops.clone(),
                    };
                    let ticket = match shared.borrow_mut().reserve(successor) {
                        Ok(ticket) => ticket,
                        Err(rejected) => panic!(
                            "destruction could not reserve released credit: {:?}",
                            rejected.error
                        ),
                    };
                    ready(&mut shared.borrow_mut(), ticket);
                    self.published.set(Some(ticket));
                    // Native logical retirement completes independently. Leave
                    // cleanup queued until the outer worker Drop returns.
                }
                self.drops.set(self.drops.get() + 1);
            }
        }
        let shared = Rc::new(RefCell::new(OwnerCells::new(Arc::default()).unwrap()));
        let drops = Rc::new(Count::new(0));
        let published = Rc::new(Count::new(None));
        let ticket = shared
            .borrow_mut()
            .reserve(Reentrant {
                cells: Rc::downgrade(&shared),
                recurse: true,
                published: published.clone(),
                drops: drops.clone(),
            })
            .ok()
            .unwrap();
        ready(&mut shared.borrow_mut(), ticket);
        let active = shared.borrow_mut().detach_cleanup(ticket).unwrap();
        assert_eq!(drops.get(), 0);
        drop(active);
        assert_eq!(
            drops.get(),
            1,
            "successor remains pending, not nested destruction"
        );
        let successor = shared
            .borrow_mut()
            .detach_cleanup(published.get().unwrap())
            .unwrap();
        drop(successor);
        assert_eq!(drops.get(), 2);
    }

    #[test]
    fn unknown_handoff_keeps_owner_and_never_becomes_canceled_or_synthetic_completion() {
        let failure = Arc::new(FailureSignal::default());
        let mut cells = OwnerCells::<Owner, 1>::new(failure.clone()).unwrap();
        let drops = Rc::new(Count::new(0));
        let ticket = cells.reserve(Owner(drops.clone())).unwrap();
        cells.begin_handoff(ticket).unwrap();
        cells.handoff_unknown(ticket).unwrap();
        failure.publish(Failure::UnknownHandoff, || {});
        assert_eq!(
            cells.cancel_prepared(ticket).err(),
            Some(OwnerError::WrongPhase)
        );
        assert_eq!(
            cells.detach_cleanup(ticket).err(),
            Some(OwnerError::WrongPhase)
        );
        assert_eq!(
            cells.begin_callbacks(ticket).err(),
            Some(OwnerError::WrongPhase)
        );
        let rejected = cells.reserve(Owner(drops.clone())).unwrap_err();
        assert_eq!(rejected.error, OwnerError::Failed(Failure::UnknownHandoff));
        assert_eq!(drops.get(), 0);
        drop(rejected.owner);
        assert_eq!(drops.get(), 1);
        // Later actual feedback is an independent fact, not invented by poison.
        cells.native_feedback(ticket).unwrap();
        cells.begin_callbacks(ticket).unwrap();
        cells.callbacks_finished(ticket).unwrap();
        cells.slot_returned(ticket).unwrap();
        cells.publish_cleanup(ticket).unwrap();
        drop(cells.detach_cleanup(ticket).unwrap());
        assert_eq!(drops.get(), 2);
        assert_eq!(failure.get(), Some(Failure::UnknownHandoff));
    }

    #[test]
    fn failure_publishes_before_wake_and_preserves_first_reason_for_parked_waiter() {
        let failure = Arc::new(FailureSignal::default());
        let signal = failure.clone();
        let (notify, notified) = std::sync::mpsc::sync_channel(1);
        let waiter = std::thread::spawn(move || {
            notified
                .recv_timeout(std::time::Duration::from_secs(2))
                .expect("failure did not notify waiter");
            signal
                .get()
                .expect("notification preceded failure publication")
        });
        failure.publish(Failure::WorkerFailed, || {
            assert_eq!(failure.get(), Some(Failure::WorkerFailed));
            notify.try_send(()).unwrap();
        });
        assert_eq!(waiter.join().unwrap(), Failure::WorkerFailed);
        failure.publish(Failure::Closing, || {
            assert_eq!(failure.get(), Some(Failure::WorkerFailed))
        });
        failure.publish(Failure::CallbackPanic, || {
            assert_eq!(failure.get(), Some(Failure::WorkerFailed))
        });
        let mut cells = OwnerCells::<u64, 1>::new(failure).unwrap();
        let rejected = cells.reserve(5).unwrap_err();
        assert_eq!(
            (rejected.error, rejected.owner),
            (OwnerError::Failed(Failure::WorkerFailed), 5)
        );
    }

    #[test]
    fn failure_after_preparation_refuses_native_handoff_without_losing_owner() {
        let failure = Arc::new(FailureSignal::default());
        let mut cells = OwnerCells::<u64, 1>::new(failure.clone()).unwrap();
        let ticket = cells.reserve(31).unwrap();
        failure.publish(Failure::WorkerFailed, || {});
        assert_eq!(
            cells.begin_handoff(ticket),
            Err(OwnerError::Failed(Failure::WorkerFailed))
        );
        assert_eq!(cells.cancel_prepared(ticket), Ok(31));
    }

    #[test]
    fn equal_slot_generations_from_distinct_registries_are_not_interchangeable() {
        let mut first = OwnerCells::<u64, 1>::new(Arc::default()).unwrap();
        let mut second = OwnerCells::<u64, 1>::new(Arc::default()).unwrap();
        let a = first.reserve(5).unwrap();
        let b = second.reserve(7).unwrap();
        assert_eq!((a.index, a.generation), (b.index, b.generation));
        assert_ne!(a.registry, b.registry);
        assert_eq!(first.begin_handoff(b), Err(OwnerError::StaleTicket));
        assert_eq!(second.cancel_prepared(a), Err(OwnerError::StaleTicket));
        assert_eq!(first.cancel_prepared(a), Ok(5));
        assert_eq!(second.cancel_prepared(b), Ok(7));
    }

    #[test]
    fn capacity_exhaustion_and_prepared_cancel_return_exact_owners() {
        assert!(matches!(
            OwnerCells::<u64, 0>::new(Arc::default()),
            Err(OwnerError::ZeroCapacity)
        ));
        let mut cells = OwnerCells::<u64, 1>::new(Arc::default()).unwrap();
        let first = cells.reserve(7).unwrap();
        let rejected = cells.reserve(9).unwrap_err();
        assert_eq!((rejected.error, rejected.owner), (OwnerError::Full, 9));
        assert_eq!(cells.cancel_prepared(first), Ok(7));
        cells.cells[0].generation = u64::MAX;
        let rejected = cells.reserve(11).unwrap_err();
        assert_eq!(
            (rejected.error, rejected.owner),
            (OwnerError::GenerationExhausted, 11)
        );
    }
}
