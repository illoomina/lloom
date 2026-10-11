//! Reusable host-grown owner slabs. Native callbacks never take the producer
//! registry lock; cleanup returns metadata credit before arbitrary destruction.
//!
//! This is the registration/cleanup side of the native adapter. Ordered callback
//! selection remains the queue's responsibility, not native delivery order.

use crate::{
    completion_wake::Wake,
    retirement_mailbox::{CleanupPolicy, Detached, MailboxError, OwnerMailbox, Ticket},
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
    waiters: Arc<Wake>,
    cleanup_policy: CleanupPolicy,
    producer: Mutex<Producer<C, A, F>>,
}
struct Producer<C, A, F> {
    slabs: Vec<Arc<Slab<C, A, F>>>,
    current: usize,
}

struct Slab<C, A, F> {
    cells: Vec<OwnerMailbox<Owner<C, A>, F>>,
    ordered: Vec<crate::ordered_metadata::OrderedMetadata<PinnedRegistration<C, A, F>>>,
    free: AtomicU64,
    occupied: AtomicUsize,
    producer_alive: AtomicBool,
    failure: Arc<FailureSignal>,
    wake: Arc<Wake>,
    waiters: Arc<Wake>,
    cleanup_policy: CleanupPolicy,
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

    #[cfg(test)]
    pub(crate) fn manual(failure: Arc<FailureSignal>, service: Arc<HostRetirementService>) -> Self {
        Self::from_service(failure, service)
    }

    #[cfg(test)]
    pub(crate) fn manual_with_waiters(
        failure: Arc<FailureSignal>,
        service: Arc<HostRetirementService>,
        waiters: Arc<Wake>,
        cleanup_policy: CleanupPolicy,
    ) -> Self {
        let mut registry = Self::from_service(failure, service);
        registry.waiters = waiters;
        registry.cleanup_policy = cleanup_policy;
        registry
    }

    pub(crate) fn new_with_waiters(
        failure: Arc<FailureSignal>,
        waiters: Arc<Wake>,
        cleanup_policy: CleanupPolicy,
    ) -> Result<Self, RegistryError> {
        let mut registry = Self::new(failure)?;
        registry.waiters = waiters;
        registry.cleanup_policy = cleanup_policy;
        Ok(registry)
    }

    fn from_service(failure: Arc<FailureSignal>, service: Arc<HostRetirementService>) -> Self {
        Self {
            wake: service.wake(),
            waiters: service.wake(),
            cleanup_policy: CleanupPolicy::TerminalFeedback,
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
                let slab = match Slab::new(
                    self.failure.clone(),
                    self.wake.clone(),
                    self.waiters.clone(),
                    self.cleanup_policy,
                ) {
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
            Ok(ticket) => {
                // SAFETY: prepare advanced generation only after previous
                // readers/PINNING ended. No new Registration has escaped yet.
                unsafe { slab.ordered[cell_index].reset_generation() };
                Ok(Registration {
                    slab: slab.clone(),
                    index: cell_index,
                    ticket,
                })
            }
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
    fn new(
        failure: Arc<FailureSignal>,
        wake: Arc<Wake>,
        waiters: Arc<Wake>,
        cleanup_policy: CleanupPolicy,
    ) -> Result<Self, RegistryError> {
        let mut cells = Vec::new();
        cells
            .try_reserve_exact(CELLS)
            .map_err(|_| RegistryError::MetadataAllocation)?;
        for _ in 0..CELLS {
            cells.push(
                OwnerMailbox::new(failure.clone(), wake.clone()).map_err(RegistryError::Mailbox)?,
            );
        }
        let mut ordered = Vec::new();
        ordered
            .try_reserve_exact(CELLS)
            .map_err(|_| RegistryError::MetadataAllocation)?;
        ordered.resize_with(CELLS, crate::ordered_metadata::OrderedMetadata::new);
        Ok(Self {
            cells,
            ordered,
            free: AtomicU64::new(ALL_FREE),
            occupied: AtomicUsize::new(0),
            producer_alive: AtomicBool::new(true),
            failure,
            wake,
            waiters,
            cleanup_policy,
            active: ActiveRetirement::new(),
        })
    }
    fn reserve(&self) -> Option<usize> {
        // Payload credit can precede the final old marker. Leave pinned free
        // bits intact and choose another cell or grow instead of false Busy.
        let mut selected = None;
        self.free
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |bits| {
                selected = None;
                let mut candidates = bits;
                while candidates != 0 {
                    let index = candidates.trailing_zeros() as usize;
                    candidates &= candidates - 1;
                    if self.cells[index].metadata_reusable() {
                        selected = Some(index);
                        return Some(bits & !(1u64 << index));
                    }
                }
                None
            })
            .ok()?;
        selected
    }
}

/// Owned metadata-only marker. Slab custody keeps the mailbox address valid;
/// the opaque permit prevents generation reuse even after payload cleanup.
/// Neither a self-referential borrow nor a new per-marker allocation is needed.
pub(crate) struct PinnedRegistration<C, A, F> {
    record: Registration<C, A, F>,
    permit: Option<crate::retirement_mailbox::PinPermit>,
}
impl<C, A, F> PinnedRegistration<C, A, F> {
    pub(crate) fn add_wrapper_handle(&self) -> Result<(), ()> {
        self.ordered().add_wrapper_handle()
    }
    pub(crate) fn release_wrapper_handle(&self) -> bool {
        self.ordered().release_wrapper_handle()
    }
    pub(crate) fn record(&self) -> &Registration<C, A, F> {
        &self.record
    }
    fn ordered(&self) -> &crate::ordered_metadata::OrderedMetadata<Self> {
        &self.record.slab.ordered[self.record.index]
    }
    pub(crate) fn append_successor(&self, next: Self) -> Result<(), Self> {
        self.ordered().append(next)
    }
    pub(crate) fn take_successor(&self) -> Option<Self> {
        self.ordered().take_successor()
    }
    pub(crate) fn publish_native_ready(&self) {
        self.ordered().publish_native_ready();
    }
    pub(crate) fn native_ready(&self) -> bool {
        self.ordered().native_ready()
    }
    pub(crate) fn publish_ordered_ready(&self) {
        self.ordered().publish_ordered_ready();
    }
    pub(crate) fn ordered_ready(&self) -> bool {
        self.ordered().ordered_ready()
    }
    pub(crate) fn publish_callbacks_finished(&self) {
        self.ordered().publish_callbacks_finished();
    }
    pub(crate) fn callbacks_finished(&self) -> bool {
        self.ordered().callbacks_finished()
    }
    pub(crate) fn try_clone(&self) -> Result<Self, MailboxError> {
        let cell = &self.record.slab.cells[self.record.index];
        let permit = cell.clone_pin(self.permit.as_ref().expect("live metadata pin"))?;
        Ok(Self {
            record: self.record.clone(),
            permit: Some(permit),
        })
    }
}
impl<C, A, F> Drop for PinnedRegistration<C, A, F> {
    fn drop(&mut self) {
        self.record.slab.cells[self.record.index]
            .release_pin(self.permit.take().expect("one metadata release"));
    }
}

impl<C, A, F> Registration<C, A, F> {
    pub(crate) fn pin(&self) -> Result<PinnedRegistration<C, A, F>, MailboxError> {
        let permit = self.slab.cells[self.index].acquire_pin(self.ticket)?;
        Ok(PinnedRegistration {
            record: self.clone(),
            permit: Some(permit),
        })
    }

    pub(crate) fn edit_prepared<V, R>(
        &self,
        input: V,
        edit: impl FnOnce(&mut C, &mut A, &mut Option<V>) -> R,
    ) -> Result<(R, Option<V>), crate::retirement_mailbox::PreparedEditError<V>> {
        self.slab.cells[self.index].edit_prepared(
            self.ticket,
            input,
            |owner, input| edit(&mut owner.0, &mut owner.1, input),
            || self.slab.waiters.notify(),
        )
    }

    pub(crate) fn replace_prepared(
        &self,
        captures: C,
        anchors: A,
    ) -> Result<(C, A), crate::retirement_mailbox::Rejected<(C, A)>> {
        self.slab.cells[self.index].replace_prepared(self.ticket, (captures, anchors))
    }

    pub(crate) fn report_failure(&self, reason: Failure) {
        self.slab.failure.publish(reason, || {
            self.slab.wake.notify();
            self.slab.waiters.notify();
        });
    }

    pub(crate) fn cancel_before_handoff(&self) -> Result<(), MailboxError> {
        self.slab.cells[self.index].publish_cancelled(self.ticket)
    }
    pub(crate) fn retain_boundary_panic(
        &self,
        payload: Box<dyn std::any::Any + Send>,
    ) -> Result<(), Box<dyn std::any::Any + Send>> {
        self.slab.cells[self.index].retain_boundary_panic(self.ticket, payload)
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
    /// Native data/error observer sees anchors only, not arbitrary captures.
    /// It runs before Ready publication and before ordered logical callbacks.
    pub(crate) fn publish_feedback_observed(
        &self,
        feedback: F,
        observe: impl FnOnce(&A, &F),
    ) -> Result<(), crate::retirement_mailbox::FeedbackPublicationError<F>> {
        self.slab.cells[self.index].publish_feedback_observed(
            self.ticket,
            feedback,
            |owner, feedback| observe(&owner.1, feedback),
            || self.slab.waiters.notify(),
        )
    }

    /// The queue's elected prefix drainer alone calls this. Slot return and
    /// prefix publication belong inside run, after all logical callbacks.
    pub(crate) fn run_callbacks(
        &self,
        run: impl FnOnce(&mut C, &mut A, &F),
        wake_waiters: impl FnOnce(),
    ) -> Result<(), MailboxError> {
        self.slab.cells[self.index].with_callbacks(
            self.ticket,
            |owner, feedback| run(&mut owner.0, &mut owner.1, feedback),
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
            }) = cell.try_detach_cleanup_with_policy(ticket, self.cleanup_policy)
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
        self.failure.publish(reason, || {
            self.wake.notify();
            self.waiters.notify();
        });
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
    #[test]
    fn returned_policy_roots_native_owner_after_unknown_external_handles_drop() {
        use crate::native_reuse::{NativeReuse, ReuseSide};
        for returned in [false, true] {
            let service = HostRetirementService::manual();
            let failure = Arc::new(FailureSignal::default());
            let mut registry = OwnerRegistry::<(), NativeReuse<Anchor>, ()>::from_service(
                failure,
                service.clone(),
            );
            registry.cleanup_policy = CleanupPolicy::ReturnedHandoff;
            let drops = Arc::new(AtomicUsize::new(0));
            let native = NativeReuse::new();
            assert!(native.install(Anchor(drops.clone())).is_ok());
            let record = registry
                .prepare((), native, |_| Ok(()))
                .unwrap_or_else(|_| panic!("prepare"));
            record.begin_handoff().unwrap();
            if returned {
                record.handoff_returned().unwrap();
            } else {
                record.handoff_unknown(|| {}).unwrap();
            }
            record.publish_feedback(()).unwrap();
            record
                .run_callbacks(
                    |_, native, _| {
                        assert!(native.publish(ReuseSide::CallbacksDone).unwrap().is_none());
                    },
                    || {},
                )
                .unwrap();
            drop(record);
            drop(registry);
            assert_eq!(drops.load(Ordering::SeqCst), 0);
            assert_eq!(
                service.manual_pass(),
                returned,
                "cleanup policy accepted unknown or refused returned handoff"
            );
            assert_eq!(drops.load(Ordering::SeqCst), usize::from(returned));
            assert!(!service.manual_pass());
            assert_eq!(drops.load(Ordering::SeqCst), usize::from(returned));
            // No actual native object/work exists. Dropping manual fixture
            // custody is not production unknown-handoff recovery permission.
            drop(service);
            assert_eq!(drops.load(Ordering::SeqCst), 1);
        }
    }
    #[test]
    fn internal_feedback_observer_precedes_ordered_earlier_callback() {
        use crate::completion_frontier::CompletionFrontier;
        let service = HostRetirementService::manual();
        let registry = OwnerRegistry::<usize, (Arc<AtomicUsize>, Anchor), usize>::from_service(
            Arc::new(FailureSignal::default()),
            service.clone(),
        );
        let ready = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let records: Vec<_> = (0..2)
            .map(|id| {
                let record = registry
                    .prepare(id, (ready.clone(), Anchor(drops.clone())), |_| Ok(()))
                    .unwrap_or_else(|_| panic!("prepare"));
                record.begin_handoff().unwrap();
                record.handoff_returned().unwrap();
                record
            })
            .collect();
        let mut frontier = CompletionFrontier::<_, 2>::new().unwrap();
        // Later native completion arrives first, with no earlier callback yet.
        records[1]
            .publish_feedback_observed(1, |anchor, value| {
                anchor.0.fetch_or(1 << value, Ordering::Release);
            })
            .unwrap_or_else(|_| panic!("later feedback"));
        assert_eq!(
            ready.load(Ordering::Acquire),
            2,
            "later data readiness was held behind earlier logical callback"
        );
        assert!(!frontier
            .publish(1, records[1].clone())
            .unwrap_or_else(|_| panic!("later publish")));
        assert_eq!(frontier.retired_prefix(), 0);
        assert!(!service.manual_pass());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        records[0]
            .publish_feedback_observed(0, |anchor, value| {
                anchor.0.fetch_or(1 << value, Ordering::Release);
            })
            .unwrap_or_else(|_| panic!("first feedback"));
        assert!(frontier
            .publish(0, records[0].clone())
            .unwrap_or_else(|_| panic!("first publish")));
        let mut callbacks = Vec::new();
        while let Some((sequence, record)) = frontier.pop_ready_or_release_drainer().unwrap() {
            record
                .run_callbacks(
                    |capture, _, feedback| {
                        assert_eq!(ready.load(Ordering::Acquire), 3);
                        assert_eq!(*capture, *feedback);
                        callbacks.push(*capture);
                        frontier.finish(sequence).unwrap();
                    },
                    || {},
                )
                .unwrap();
        }
        assert_eq!(callbacks, [0, 1]);
        drain(&service, &drops, 2);
    }

    #[test]
    fn internal_feedback_observer_panic_keeps_consumed_owners_rooted() {
        use crate::retirement_mailbox::FeedbackPublicationError;
        let service = HostRetirementService::manual();
        let failure = Arc::new(FailureSignal::default());
        let registry =
            OwnerRegistry::<OnDrop, Anchor, OnDrop>::from_service(failure.clone(), service.clone());
        let captures = Arc::new(AtomicUsize::new(0));
        let anchors = Arc::new(AtomicUsize::new(0));
        let feedbacks = Arc::new(AtomicUsize::new(0));
        let panics = Arc::new(AtomicUsize::new(0));
        let counted = |count: &Arc<AtomicUsize>| {
            let count = count.clone();
            OnDrop(Some(Box::new(move || {
                count.fetch_add(1, Ordering::SeqCst);
            })))
        };
        let record = registry
            .prepare(counted(&captures), Anchor(anchors.clone()), |_| Ok(()))
            .unwrap_or_else(|_| panic!("prepare"));
        record.begin_handoff().unwrap();
        record.handoff_returned().unwrap();
        let payload = counted(&panics);
        let outcome = record.publish_feedback_observed(counted(&feedbacks), |_, _| {
            std::panic::panic_any(payload);
        });
        assert!(matches!(
            outcome,
            Err(FeedbackPublicationError::ObserverPanicked)
        ));
        assert_eq!(failure.get(), Some(Failure::CallbackPanic));
        assert_eq!(
            record.run_callbacks(|_, _, _| panic!("failed observer became Ready"), || {}),
            Err(MailboxError::WrongPhase)
        );
        // Rejected feedback remains caller-owned; consumed feedback is not
        // fabricated as a return value after the observer panics.
        let rejected_count = Arc::new(AtomicUsize::new(0));
        let rejected = record.publish_feedback_observed(counted(&rejected_count), |_, _| {});
        let Err(FeedbackPublicationError::Rejected(rejected)) = rejected else {
            panic!("duplicate publication did not return caller feedback");
        };
        assert_eq!(rejected_count.load(Ordering::SeqCst), 0);
        drop(rejected.value);
        assert_eq!(rejected_count.load(Ordering::SeqCst), 1);
        drop(record);
        drop(registry);
        assert!(!service.manual_pass());
        for count in [&captures, &anchors, &feedbacks, &panics] {
            assert_eq!(
                count.load(Ordering::SeqCst),
                0,
                "observer failure released rooted custody"
            );
        }
        // Fake-only fixture teardown, not native-quarantine recovery permission.
        drop(service);
        for count in [&captures, &anchors, &feedbacks, &panics] {
            assert_eq!(count.load(Ordering::SeqCst), 1);
        }
    }
    #[test]
    fn old_owned_marker_keeps_generation_while_cleanup_credit_can_grow() {
        let (service, registry) = setup();
        let drops = Arc::new(AtomicUsize::new(0));
        let first = prepare(&registry, OnDrop(None), &drops);
        let pin = first.pin().unwrap();
        let old_ticket = first.ticket;
        let fillers: Vec<_> = (1..CELLS)
            .map(|_| prepare(&registry, OnDrop(None), &drops))
            .collect();
        first.cancel_before_handoff().unwrap();
        assert!(service.manual_pass());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        // The record payload is gone, but existing old markers can still clone.
        let cloned_pin = pin.try_clone().unwrap();
        assert!(!first.slab.cells[first.index].metadata_reusable());
        let growths = AtomicUsize::new(0);
        let successor = registry
            .prepare(OnDrop(None), Anchor(drops.clone()), |_| {
                growths.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
            .unwrap_or_else(|_| panic!("pinned free cell caused false Busy instead of growth"));
        assert_eq!(growths.load(Ordering::SeqCst), 1);
        assert!(!Arc::ptr_eq(&first.slab, &successor.slab));
        assert_eq!(first.ticket, old_ticket);
        drop(pin);
        assert!(!first.slab.cells[first.index].metadata_reusable());
        drop(cloned_pin);
        assert!(first.slab.cells[first.index].metadata_reusable());
        // Force the producer to inspect the original slab; its only free cell
        // must now be reusable with a distinct generation, never the old ticket.
        registry.producer.lock().current = 0;
        let reused = prepare(&registry, OnDrop(None), &drops);
        assert!(Arc::ptr_eq(&first.slab, &reused.slab));
        assert_eq!(first.index, reused.index);
        assert_ne!(first.ticket, reused.ticket);
        assert_eq!(first.cancel_before_handoff(), Err(MailboxError::Stale));
        for record in fillers.iter().chain([&successor, &reused]) {
            record.cancel_before_handoff().unwrap();
        }
        drain(&service, &drops, CELLS + 2);
    }

    #[test]
    fn pinned_metadata_growth_refusal_keeps_caller_owner_and_old_marker() {
        let (service, registry) = setup();
        let drops = Arc::new(AtomicUsize::new(0));
        let first = prepare(&registry, OnDrop(None), &drops);
        let pin = first.pin().unwrap();
        let fillers: Vec<_> = (1..CELLS)
            .map(|_| prepare(&registry, OnDrop(None), &drops))
            .collect();
        first.cancel_before_handoff().unwrap();
        assert!(service.manual_pass());
        let rejected = match registry.prepare(OnDrop(None), Anchor(drops.clone()), |_| {
            Err(RegistryError::GrowthRefused)
        }) {
            Ok(_) => panic!("growth refusal admitted a pinned generation"),
            Err(rejected) => rejected,
        };
        assert!(
            matches!(rejected.error, RegistryError::GrowthRefused),
            "pinned free cell hid the explicit growth refusal"
        );
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(registry.producer.lock().slabs.len(), 1);
        drop(rejected);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        drop(pin);
        for record in &fillers {
            record.cancel_before_handoff().unwrap();
        }
        drain(&service, &drops, CELLS + 1);
    }
}
