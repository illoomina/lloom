//! Reusable host-grown owner slabs. Native callbacks never take the producer
//! registry lock; cleanup returns metadata credit before arbitrary destruction.
//!
//! This is the registration/cleanup side of the native adapter. Ordered callback
//! selection remains the queue's responsibility, not native delivery order.

use crate::{
    completion_wake::Wake,
    retirement_mailbox::{Detached, MailboxError, OwnerMailbox, Ticket},
    retirement_owners::{Failure, FailureSignal},
    retirement_service::{ActiveRetirement, CleanupInbox, HostRetirementService, ServiceError},
};
use parking_lot::Mutex;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc,
};

const CELLS: usize = 32;
const ALL_FREE: u64 = (1u64 << CELLS) - 1;
type Owner<C, A> = (C, A);
type Active<C, A, F> = ActiveRetirement<(C, Option<F>), A>;

#[derive(Debug)]
pub(crate) enum RegistryError {
    Mailbox(MailboxError),
    Service(ServiceError),
    MetadataAllocation,
    GrowthRefused,
}
pub(crate) struct Rejected<C, A> {
    pub error: RegistryError,
    pub captures: C,
    pub anchors: A,
}

/// One producer-owned pool per logical queue. All submission records, including
/// reentrant submissions from capture destructors, may grow metadata rather than
/// wait for this same host worker to return cleanup credit. The native-slot
/// bound is a separate queue property.
pub(crate) struct OwnerRegistry<C, A, F> {
    service: Arc<HostRetirementService>,
    failure: Arc<FailureSignal>,
    wake: Arc<Wake>,
    producer: Mutex<Producer<C, A, F>>,
}
struct Producer<C, A, F> {
    slabs: Vec<Arc<Slab<C, A, F>>>,
    current: usize,
}

struct Slab<C, A, F> {
    cells: Vec<OwnerMailbox<Owner<C, A>, F>>,
    free: AtomicU64,
    occupied: AtomicUsize,
    producer_alive: AtomicBool,
    failure: Arc<FailureSignal>,
    wake: Arc<Wake>,
    active: Active<C, A, F>,
}

/// A native block retains this metadata handle, not an independently owned
/// resource payload. The service roots its slab before a handle is returned.
/// Old handles cannot access a reused cell because every transition checks the
/// mailbox identity and generation. Logical marker data lives outside this
/// handle (M4 Submission); M3 marker pins must be added before its integration.
pub(crate) struct Registration<C, A, F> {
    slab: Arc<Slab<C, A, F>>,
    index: usize,
    ticket: Ticket,
}
impl<C, A, F> Clone for Registration<C, A, F> {
    fn clone(&self) -> Self {
        Self {
            slab: self.slab.clone(),
            index: self.index,
            ticket: self.ticket,
        }
    }
}

impl<C: Send + Sync + 'static, A: Send + Sync + 'static, F: Send + Sync + 'static>
    OwnerRegistry<C, A, F>
{
    pub(crate) fn new(failure: Arc<FailureSignal>) -> Result<Self, RegistryError> {
        let service = HostRetirementService::shared()
            .map_err(RegistryError::Service)?
            .clone();
        Ok(Self::from_service(failure, service))
    }

    fn from_service(failure: Arc<FailureSignal>, service: Arc<HostRetirementService>) -> Self {
        Self {
            wake: service.wake(),
            service,
            failure,
            producer: Mutex::new(Producer {
                slabs: Vec::new(),
                current: 0,
            }),
        }
    }

    pub(crate) fn prepare(
        &self,
        captures: C,
        anchors: A,
        admit_growth: impl FnOnce(usize) -> Result<(), RegistryError>,
    ) -> Result<Registration<C, A, F>, Rejected<C, A>> {
        let reject = |error, (captures, anchors)| Rejected {
            error,
            captures,
            anchors,
        };
        let owner = (captures, anchors);
        if let Some(reason) = self.failure.get() {
            return Err(reject(
                RegistryError::Mailbox(MailboxError::Failed(reason)),
                owner,
            ));
        }
        let mut producer = self.producer.lock();
        let current = producer.current;
        let found = producer
            .slabs
            .get(current)
            .and_then(|slab| slab.reserve().map(|cell| (current, cell)))
            .or_else(|| {
                producer.slabs.iter().enumerate().find_map(|(index, slab)| {
                    if index == current {
                        None
                    } else {
                        slab.reserve().map(|cell| (index, cell))
                    }
                })
            });
        let (slab_index, cell_index) = match found {
            Some(found) => found,
            None => {
                // Cold growth is explicit and fallible; never park the cleanup
                // worker waiting for metadata it would have to release itself.
                if let Err(error) = admit_growth(CELLS) {
                    return Err(reject(error, owner));
                }
                if producer.slabs.try_reserve(1).is_err() {
                    return Err(reject(RegistryError::MetadataAllocation, owner));
                }
                let slab = match Slab::new(self.failure.clone(), self.wake.clone()) {
                    Ok(slab) => Arc::new(slab),
                    Err(error) => return Err(reject(error, owner)),
                };
                let inbox: Arc<dyn CleanupInbox> = slab.clone();
                if let Err(error) = self.service.register(&inbox) {
                    return Err(reject(RegistryError::Service(error), owner));
                }
                let index = producer.slabs.len();
                producer.slabs.push(slab);
                let cell = producer.slabs[index]
                    .reserve()
                    .expect("new slab has free cells");
                (index, cell)
            }
        };
        producer.current = slab_index;
        let slab = &producer.slabs[slab_index];
        slab.occupied.fetch_add(1, Ordering::AcqRel);
        match slab.cells[cell_index].prepare(owner) {
            Ok(ticket) => Ok(Registration {
                slab: slab.clone(),
                index: cell_index,
                ticket,
            }),
            Err(rejected) => {
                slab.occupied.fetch_sub(1, Ordering::AcqRel);
                // An exhausted generation stays unavailable. No wrap/reuse.
                if rejected.error != MailboxError::GenerationExhausted {
                    slab.free.fetch_or(1u64 << cell_index, Ordering::Release);
                }
                Err(reject(
                    RegistryError::Mailbox(rejected.error),
                    rejected.value,
                ))
            }
        }
    }
}
impl<C, A, F> Drop for OwnerRegistry<C, A, F> {
    fn drop(&mut self) {
        for slab in &self.producer.get_mut().slabs {
            slab.producer_alive.store(false, Ordering::Release);
        }
        self.wake.notify();
    }
}

impl<C, A, F> Slab<C, A, F> {
    fn new(failure: Arc<FailureSignal>, wake: Arc<Wake>) -> Result<Self, RegistryError> {
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(CELLS)
            .map_err(|_| RegistryError::MetadataAllocation)?;
        for _ in 0..CELLS {
            cells.push(
                OwnerMailbox::new(failure.clone(), wake.clone()).map_err(RegistryError::Mailbox)?,
            );
        }
        Ok(Self {
            cells,
            free: AtomicU64::new(ALL_FREE),
            occupied: AtomicUsize::new(0),
            producer_alive: AtomicBool::new(true),
            failure,
            wake,
            active: ActiveRetirement::new(),
        })
    }
    fn reserve(&self) -> Option<usize> {
        self.free
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bits| {
                (bits != 0).then(|| bits & (bits - 1))
            })
            .ok()
            .map(|bits| bits.trailing_zeros() as usize)
    }
}

impl<C, A, F> Registration<C, A, F> {
    pub(crate) fn cancel_before_handoff(&self) -> Result<(), MailboxError> {
        self.slab.cells[self.index].publish_cancelled(self.ticket)
    }
    pub(crate) fn begin_handoff(&self) -> Result<(), MailboxError> {
        self.slab.cells[self.index].begin_handoff(self.ticket)
    }
    pub(crate) fn handoff_returned(&self) -> Result<(), MailboxError> {
        self.slab.cells[self.index].handoff_returned(self.ticket)
    }
    pub(crate) fn handoff_unknown(&self, wake_waiters: impl FnOnce()) -> Result<(), MailboxError> {
        self.slab.cells[self.index].handoff_unknown(self.ticket, wake_waiters)
    }
    pub(crate) fn publish_feedback(
        &self,
        feedback: F,
    ) -> Result<(), crate::retirement_mailbox::Rejected<F>> {
        self.slab.cells[self.index].publish_feedback(self.ticket, feedback)
    }
    /// The queue's elected prefix drainer alone calls this. Slot return and
    /// prefix publication belong inside run, after all logical callbacks.
    pub(crate) fn run_callbacks(
        &self,
        run: impl FnOnce(&C, &A, &F),
        wake_waiters: impl FnOnce(),
    ) -> Result<(), MailboxError> {
        self.slab.cells[self.index].with_callbacks(
            self.ticket,
            |owner, feedback| run(&owner.0, &owner.1, feedback),
            wake_waiters,
        )?;
        self.slab.cells[self.index].slot_returned(self.ticket)?;
        self.slab.cells[self.index].publish_cleanup(self.ticket)
    }
}

// SAFETY: only the service owns the active cell. Callback references are ended
// by OwnerMailbox before detachment; handoff-return and terminal feedback are
// separate mailbox facts. Free credit is published only after active staging.
// A failed callback/unknown handoff without real feedback never becomes ready.
unsafe impl<C: Send + Sync + 'static, A: Send + Sync + 'static, F: Send + Sync + 'static>
    CleanupInbox for Slab<C, A, F>
{
    fn begin_retire(&self) -> bool {
        if !self.active.is_empty() {
            return false;
        }
        for (index, cell) in self.cells.iter().enumerate() {
            let Some(ticket) = cell.cleanup_ticket() else {
                continue;
            };
            let Ok(Detached {
                owner: (captures, anchors),
                feedback,
            }) = cell.try_detach_cleanup(ticket)
            else {
                continue;
            };
            if let Err((_error, captures, anchors)) =
                self.active.stage((captures, feedback), anchors)
            {
                // Single service access makes this unreachable absent a broken
                // adapter. Preserve owners even then; never unwind/drop them.
                std::mem::forget((captures, anchors));
                self.fail(Failure::WorkerFailed);
                return false;
            }
            self.occupied.fetch_sub(1, Ordering::AcqRel);
            self.free.fetch_or(1u64 << index, Ordering::Release);
            return true;
        }
        false
    }
    fn drop_captures(&self) {
        self.active
            .drop_captures()
            .expect("service staged active capture");
    }
    fn finish_retire(&self) {
        drop(self.active.finish().expect("capture destruction returned"));
    }
    fn can_unregister(&self) -> bool {
        !self.producer_alive.load(Ordering::Acquire)
            && self.occupied.load(Ordering::Acquire) == 0
            && self.active.is_empty()
    }
    fn fail(&self, reason: Failure) {
        self.failure.publish(reason, || self.wake.notify());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct OnDrop(Option<Box<dyn FnOnce() + Send + Sync>>);
    impl Drop for OnDrop {
        fn drop(&mut self) {
            if let Some(f) = self.0.take() {
                f();
            }
        }
    }
    struct Anchor(Arc<AtomicUsize>);
    impl Drop for Anchor {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    type Registry = OwnerRegistry<OnDrop, Anchor, ()>;
    type Record = Registration<OnDrop, Anchor, ()>;

    fn setup() -> (Arc<HostRetirementService>, Arc<Registry>) {
        let service = HostRetirementService::manual();
        let registry = Arc::new(Registry::from_service(
            Arc::new(FailureSignal::default()),
            service.clone(),
        ));
        (service, registry)
    }
    fn prepare(registry: &Registry, captures: OnDrop, drops: &Arc<AtomicUsize>) -> Record {
        registry
            .prepare(captures, Anchor(drops.clone()), |_| Ok(()))
            .unwrap_or_else(|_| panic!("prepare refused"))
    }
    fn drain(service: &HostRetirementService, drops: &AtomicUsize, expected: usize) {
        for _ in 0..(CELLS * 3) {
            if drops.load(Ordering::SeqCst) == expected {
                return;
            }
            service.manual_pass();
        }
        assert_eq!(
            drops.load(Ordering::SeqCst),
            expected,
            "manual service did not drain bounded records"
        );
    }

    #[test]
    fn two_reentrant_successors_grow_instead_of_waiting_for_cleanup_credit() {
        let (service, registry) = setup();
        let drops = Arc::new(AtomicUsize::new(0));
        let successors = Arc::new(Mutex::new(Vec::<Record>::new()));
        let growths = Arc::new(AtomicUsize::new(0));
        let weak = Arc::downgrade(&registry);
        let out = successors.clone();
        let grown = growths.clone();
        let count = drops.clone();
        let first = prepare(
            &registry,
            OnDrop(Some(Box::new(move || {
                let registry = weak.upgrade().unwrap();
                for _ in 0..2 {
                    let next = registry
                        .prepare(OnDrop(None), Anchor(count.clone()), |cells| {
                            assert_eq!(cells, CELLS);
                            grown.fetch_add(1, Ordering::SeqCst);
                            Ok(())
                        })
                        .unwrap_or_else(|_| panic!("reentrant successor was refused"));
                    out.lock().push(next);
                }
                assert_eq!(
                    count.load(Ordering::SeqCst),
                    0,
                    "active anchor dropped during captures"
                );
            }))),
            &drops,
        );
        let fillers: Vec<_> = (1..CELLS)
            .map(|_| prepare(&registry, OnDrop(None), &drops))
            .collect();
        assert_eq!(registry.producer.lock().slabs.len(), 1);
        first.cancel_before_handoff().unwrap();
        assert!(service.manual_pass());
        assert_eq!(successors.lock().len(), 2);
        {
            let successors = successors.lock();
            assert!(
                Arc::ptr_eq(&first.slab, &successors[0].slab) && first.index == successors[0].index,
                "first successor did not reuse detached metadata credit"
            );
        }
        assert_eq!(
            growths.load(Ordering::SeqCst),
            1,
            "second successor must grow the full slab"
        );
        assert_eq!(registry.producer.lock().slabs.len(), 2);
        for record in fillers.iter().chain(successors.lock().iter()) {
            record.cancel_before_handoff().unwrap();
        }
        drain(&service, &drops, CELLS + 2);
        // Old block metadata survives reuse; it cannot cancel a new occupant.
        assert_eq!(first.cancel_before_handoff(), Err(MailboxError::Stale));
        drop(registry);
        service.manual_pass();
        assert_eq!(drops.load(Ordering::SeqCst), CELLS + 2);
    }

    #[test]
    fn reentrant_growth_refusal_returns_owners_without_blocking_or_fallback() {
        let (service, registry) = setup();
        let drops = Arc::new(AtomicUsize::new(0));
        let successor = Arc::new(Mutex::new(None::<Record>));
        let rejected = Arc::new(Mutex::new(None::<Rejected<OnDrop, Anchor>>));
        let refusals = Arc::new(AtomicUsize::new(0));
        let weak = Arc::downgrade(&registry);
        let out = successor.clone();
        let rejected_out = rejected.clone();
        let refused = refusals.clone();
        let count = drops.clone();
        let first = prepare(
            &registry,
            OnDrop(Some(Box::new(move || {
                let registry = weak.upgrade().unwrap();
                *out.lock() = Some(prepare(&registry, OnDrop(None), &count));
                let error = match registry.prepare(OnDrop(None), Anchor(count.clone()), |_| {
                    refused.fetch_add(1, Ordering::SeqCst);
                    Err(RegistryError::GrowthRefused)
                }) {
                    Ok(_) => panic!("refused growth silently admitted"),
                    Err(error) => error,
                };
                assert!(matches!(error.error, RegistryError::GrowthRefused));
                assert_eq!(
                    count.load(Ordering::SeqCst),
                    0,
                    "refusal dropped caller-owned anchors"
                );
                *rejected_out.lock() = Some(error);
            }))),
            &drops,
        );
        let fillers: Vec<_> = (1..CELLS)
            .map(|_| prepare(&registry, OnDrop(None), &drops))
            .collect();
        first.cancel_before_handoff().unwrap();
        assert!(service.manual_pass());
        assert_eq!(refusals.load(Ordering::SeqCst), 1);
        assert_eq!(
            registry.producer.lock().slabs.len(),
            1,
            "refusal must not allocate a fallback slab"
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(rejected.lock().take().unwrap());
        assert_eq!(
            drops.load(Ordering::SeqCst),
            2,
            "returned owner drops only in caller custody"
        );
        for record in &fillers {
            record.cancel_before_handoff().unwrap();
        }
        successor
            .lock()
            .as_ref()
            .unwrap()
            .cancel_before_handoff()
            .unwrap();
        drain(&service, &drops, CELLS + 2);
        drop(registry);
        service.manual_pass();
    }

    #[test]
    fn deferred_cancel_rejects_started_handoff_and_retains_terminal_proof_requirement() {
        let (service, registry) = setup();
        let drops = Arc::new(AtomicUsize::new(0));
        let record = prepare(&registry, OnDrop(None), &drops);
        record.begin_handoff().unwrap();
        assert_eq!(
            record.cancel_before_handoff(),
            Err(MailboxError::WrongPhase)
        );
        record.handoff_returned().unwrap();
        assert!(!service.manual_pass());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        record
            .publish_feedback(())
            .unwrap_or_else(|_| panic!("feedback refused"));
        record.run_callbacks(|_, _, _| {}, || {}).unwrap();
        assert!(service.manual_pass());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        drop(registry);
        service.manual_pass();
    }
}
