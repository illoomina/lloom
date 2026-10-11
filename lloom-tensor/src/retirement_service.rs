//! Independent host custody for completion-owned resources (B1).
//!
//! Logical callbacks run on native feedback. This worker only destroys payloads
//! after an inbox has detached them and returned its cleanup credit. The static
//! registry, not a native block or the worker stack, is the custody root.
//! Engine queue-associated adapters register payloads here. Their physical/IO
//! custody integration and native runtime validation remain separate obligations.

use crate::{
    completion_wake::Wake,
    memory_admission::PhysicalCharge,
    retirement_owners::{Failure, FailureSignal},
};
use parking_lot::Mutex;
use std::{
    any::Any,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Arc, OnceLock},
};

/// Native adapter contract, in addition to Rust thread safety.
///
/// # Safety
/// `begin_retire` may detach ONLY independently native-terminal payloads whose
/// callback borrow and native handoff have ended. It releases registration
/// credit and every registry/producer lock BEFORE arbitrary captured-value Drop.
/// It installs an inbox-owned active record before returning; both its raw engine
/// anchors and physical charges remain in that independently rooted record.
/// `drop_captures` may remove/drop only arbitrary callback captures. On unwind,
/// active engine anchors remain in the inbox, not on the unwinding stack.
/// `finish_retire` releases the known-terminal anchors after capture Drop returns.
/// No phase recursively services another record. Logical retirement cannot
/// depend on this service. Native adapters must use a concrete rooted active cell,
/// not only promise that a stack local will survive unwinding.
///
/// `can_unregister` is true ONLY when no producer capable of installing a new
/// payload exists and every resource/panic payload has left the inbox. Delayed
/// native block copies then retain empty metadata only. Quarantined/unknown
/// handoffs keep this false; queue poisoning is not native terminal proof.
///
/// `fail` publishes admission/wait failure and wakes waiters without dropping
/// pending owners, acquiring a producer lock, or inventing completed status.
pub(crate) unsafe trait CleanupInbox: Send + Sync {
    fn begin_retire(&self) -> bool;
    fn drop_captures(&self);
    fn finish_retire(&self);
    fn can_unregister(&self) -> bool;
    /// Opt-in final metadata destruction on the service thread. Safety requires
    /// no Weak/raw-Arc acquisition path: every external clone must start from
    /// an already counted Arc. Service snapshots acquire under registry lock.
    /// Every foreign final release must drop its Arc BEFORE notifying Wake.
    /// Non-opt-in lifecycle adapters retain their existing unregister policy.
    fn worker_final_metadata_drop(&self) -> bool {
        false
    }
    /// Called only after successful worker-final opt-in root removal under the
    /// shared census/snapshot lock. No Weak/raw acquisition and count2 then
    /// leave only removed root + pass, both exclusively service-controlled.
    /// Extracts metadata charge without payload destruction or panicking.
    /// Service custody holds it across actual final Arc deallocation, including
    /// unwind. Default None preserves non-opt-in and lifecycle policy.
    fn take_final_metadata_charge(&self) -> Option<PhysicalCharge> {
        None
    }
    fn fail(&self, reason: Failure);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ServiceError {
    StartFailed,
    Failed(Failure),
}

// Release snapshot references before waking the worker, including on unwind.
// Otherwise the worker can observe the old count then park without a new wake.
struct InboxSnapshot {
    entries: Vec<Arc<dyn CleanupInbox>>,
    wake: Arc<Wake>,
}
impl Drop for InboxSnapshot {
    fn drop(&mut self) {
        self.entries.clear();
        self.wake.notify();
    }
}

pub(crate) struct HostRetirementService {
    // Only host registration and this host worker access the registry lock.
    // Native feedback publishes into its rooted inbox and calls wake.notify.
    registry: Mutex<Vec<Arc<dyn CleanupInbox>>>,
    wake: Arc<Wake>,
    failure: FailureSignal,
    // An arbitrary destructor panic can itself carry owned values. Keep those
    // under static host custody too; do not drop/retry them on worker unwind.
    panic_payloads: Mutex<Vec<Box<dyn Any + Send>>>,
    // One single-worker final metadata Drop at a time. Rooted in this static
    // service, not the unwinding stack; never cleared if metadata Drop panics.
    active_metadata_charge: Mutex<Option<PhysicalCharge>>,
}

impl HostRetirementService {
    /// Setup-only. Start failure is returned before any inbox/owner can be
    /// registered. The immutable static retains successful service custody even
    /// if its worker exits unexpectedly. No mandatory worker until first use.
    pub(crate) fn shared() -> Result<&'static Arc<Self>, ServiceError> {
        static SERVICE: OnceLock<Result<Arc<HostRetirementService>, ServiceError>> =
            OnceLock::new();
        match SERVICE.get_or_init(|| {
            let service = Arc::new(Self {
                registry: Mutex::new(Vec::new()),
                wake: Arc::new(Wake::default()),
                failure: FailureSignal::default(),
                panic_payloads: Mutex::new(Vec::new()),
                active_metadata_charge: Mutex::new(None),
            });
            let worker_service = service.clone();
            std::thread::Builder::new()
                .name("lloom-metal-retire".into())
                .spawn(move || worker_service.worker())
                .map_err(|_| ServiceError::StartFailed)?;
            Ok(service)
        }) {
            Ok(service) => Ok(service),
            Err(error) => Err(*error),
        }
    }

    #[cfg(test)]
    pub(crate) fn manual() -> Arc<Self> {
        Arc::new(Self {
            registry: Mutex::new(Vec::new()),
            wake: Arc::new(Wake::default()),
            failure: FailureSignal::default(),
            panic_payloads: Mutex::new(Vec::new()),
            active_metadata_charge: Mutex::new(None),
        })
    }

    #[cfg(test)]
    pub(crate) fn manual_pass(&self) -> bool {
        self.pass()
    }

    #[cfg(test)]
    pub(crate) fn manual_with_snapshot(&self, run: impl FnOnce()) {
        let _snapshot = InboxSnapshot {
            entries: self.registry.lock().clone(),
            wake: self.wake.clone(),
        };
        run();
    }

    pub(crate) fn wake(&self) -> Arc<Wake> {
        self.wake.clone()
    }

    /// Queue/slab setup, not per-CB allocation. Caller retains its own handle on
    /// rejection. Registry publication completes before any native handoff.
    pub(crate) fn register(&self, inbox: &Arc<dyn CleanupInbox>) -> Result<(), ServiceError> {
        let mut registry = self.registry.lock();
        if let Some(reason) = self.failure.get() {
            return Err(ServiceError::Failed(reason));
        }
        if !registry.iter().any(|entry| Arc::ptr_eq(entry, inbox)) {
            registry.push(inbox.clone());
        }
        drop(registry);
        self.wake.notify();
        Ok(())
    }

    /// Explicit conservative shutdown: close admission/wake existing queues,
    /// but continue servicing genuinely completed payloads. Unknown ownership
    /// remains registered. There is no "timeout then free" or join that claims
    /// native quiescence. Process-static custody survives a stopped worker.
    pub(crate) fn close_admission(&self) {
        self.failure
            .publish(Failure::Closing, || self.wake.notify());
        self.fail_inboxes(Failure::Closing);
    }

    fn fail_inboxes(&self, reason: Failure) {
        // Rare failure/shutdown path: snapshot under the host registry lock so
        // concurrent worker removal cannot skip an entry by shifting indices.
        // Admission was closed before this call, so no new entry can be missed.
        let snapshot = InboxSnapshot {
            entries: self.registry.lock().clone(),
            wake: self.wake.clone(),
        };
        for entry in &snapshot.entries {
            // An arbitrary adapter failure must not unwind past the worker and
            // lose the remaining wakeups. A bad adapter stays rooted, too.
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| entry.fail(reason))) {
                self.panic_payloads.lock().push(payload);
            }
        }
    }

    fn pass(&self) -> bool {
        // Production has one worker and exits on unwind. A manual/erroneous
        // later pass must not clear a charge retained across prior failed Drop.
        if self.active_metadata_charge.lock().is_some() {
            self.failure
                .publish(Failure::WorkerFailed, || self.wake.notify());
            return false;
        }
        let mut index = 0;
        let mut progressed = false;
        loop {
            let entry = self.registry.lock().get(index).cloned();
            let Some(entry) = entry else {
                return progressed;
            };
            // Crucially outside the registry lock. Detaching one credit before
            // Drop permits that destructor to register/submit another record.
            if entry.begin_retire() {
                entry.drop_captures();
                entry.finish_retire();
                progressed = true;
            }
            let removed = if entry.can_unregister() {
                let mut registry = self.registry.lock();
                // The census and every service snapshot acquisition share this
                // mutex. count2 is the registry root plus this pass handle;
                // any foreign clone needs an already counted owner. Opt-in
                // forbids Weak/raw acquisition after the last foreign release.
                if registry.get(index).is_some_and(|current| {
                    Arc::ptr_eq(current, &entry)
                        && (!entry.worker_final_metadata_drop() || Arc::strong_count(current) == 2)
                }) {
                    Some(registry.swap_remove(index))
                } else {
                    None
                }
            } else {
                None
            };
            if removed.is_none() {
                // Empty payload does not imply foreign metadata quiescence.
                // Move on rather than spin while can_unregister stays true.
                index += 1;
            }
            if let Some(removed) = removed {
                // Only an opted-in metadata inbox participates. Non-opt-in
                // lifecycle root removal still has its original semantics.
                let charge = if entry.worker_final_metadata_drop() {
                    entry.take_final_metadata_charge()
                } else {
                    None
                };
                let staged_charge = charge.is_some();
                if let Some(charge) = charge {
                    let mut active = self.active_metadata_charge.lock();
                    if active.is_some() {
                        // A prior unwind must not be retried or overwritten.
                        // Retain both unexplained inbox custody and charge.
                        drop(active);
                        std::mem::forget((charge, removed, entry));
                        self.failure
                            .publish(Failure::WorkerFailed, || self.wake.notify());
                        return progressed;
                    }
                    *active = Some(charge);
                }
                // No registry/producer/active-charge lock spans metadata Drop.
                // On unwind the service-rooted charge remains populated.
                drop(removed);
                drop(entry);
                if staged_charge {
                    let charge = self.active_metadata_charge.lock().take();
                    drop(charge); // All managed inline and Vec storage is gone.
                }
            } else {
                drop(entry);
            }
        }
    }

    fn worker(self: Arc<Self>) {
        let result = catch_unwind(AssertUnwindSafe(|| loop {
            // Snapshot before examining inboxes, including RETURNED and final
            // reader notifications; the shared Wake closes check -> park.
            let seen = self.wake.snapshot();
            #[cfg(feature = "metal")]
            let progressed = objc2::rc::autoreleasepool(|_| self.pass());
            #[cfg(not(feature = "metal"))]
            let progressed = self.pass();
            if !progressed {
                self.wake.wait(seen);
            }
        }));
        if let Err(payload) = result {
            self.panic_payloads.lock().push(payload);
            // Never retry an arbitrary partially executed destructor. Static
            // registry roots remain, all admissions/waiters observe failure.
            self.failure
                .publish(Failure::WorkerFailed, || self.wake.notify());
            self.fail_inboxes(Failure::WorkerFailed);
        }
    }
}

/// The one detached record held by an inbox while the shared worker destroys
/// callbacks. Stored INSIDE the registered inbox, never on the worker stack.
/// Native feedback never accesses this mutex. Its only production accessor is
/// the single cleanup worker, so reentrant submission queues another record
/// rather than recursively beginning another active cleanup.
pub(crate) struct ActiveRetirement<C, A> {
    record: Mutex<Option<ActiveRecord<C, A>>>,
}
struct ActiveRecord<C, A> {
    captures: Option<C>,
    anchors: A,
    state: ActivePhase,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ActivePhase {
    Staged,
    Dropping,
    CapturesDropped,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ActiveError {
    Busy,
    Empty,
    WrongPhase,
}

impl<C, A> ActiveRetirement<C, A> {
    pub(crate) fn new() -> Self {
        Self {
            record: Mutex::new(None),
        }
    }

    /// Called after mailbox detachment returned cleanup credit, BEFORE any C
    /// destructor. Rejection returns both owners; a failed adapter must preserve
    /// them under host custody rather than treating rejection as cancellation.
    pub(crate) fn stage(&self, captures: C, anchors: A) -> Result<(), (ActiveError, C, A)> {
        let mut record = self.record.lock();
        if record.is_some() {
            return Err((ActiveError::Busy, captures, anchors));
        }
        *record = Some(ActiveRecord {
            captures: Some(captures),
            anchors,
            state: ActivePhase::Staged,
        });
        Ok(())
    }

    pub(crate) fn drop_captures(&self) -> Result<(), ActiveError> {
        let captures = {
            let mut slot = self.record.lock();
            let record = slot.as_mut().ok_or(ActiveError::Empty)?;
            if record.state != ActivePhase::Staged {
                return Err(ActiveError::WrongPhase);
            }
            record.state = ActivePhase::Dropping;
            record.captures.take().expect("staged captures")
        };
        // An unwind leaves Dropping plus anchors under registry-rooted custody.
        // No lock spans this arbitrary Drop, so it may submit/reenter host work.
        drop(captures);
        let mut slot = self.record.lock();
        let record = slot.as_mut().expect("exclusive worker owns active record");
        record.state = ActivePhase::CapturesDropped;
        Ok(())
    }

    /// Only normal callback-capture destruction permits anchor release. The
    /// adapter must supply already native-terminal engine anchors, not hidden
    /// arbitrary callback boxes in A. This helper does not promise rollback of
    /// arbitrary A::drop; it specifically preserves A across C::drop unwind.
    pub(crate) fn finish(&self) -> Result<A, ActiveError> {
        let mut slot = self.record.lock();
        let record = slot.as_ref().ok_or(ActiveError::Empty)?;
        if record.state != ActivePhase::CapturesDropped {
            return Err(ActiveError::WrongPhase);
        }
        let record = slot.take().expect("checked active record");
        drop(slot);
        Ok(record.anchors)
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.record.lock().is_none()
    }
}

/// One setup-time anchor shared by submit-capable engine wrappers. It is NOT
/// captured by native blocks or retained by an idle Inbox; only rooted payloads
/// clone its owner across native handoff. Standalone raw wrappers may omit it.
#[derive(Default)]
pub(crate) struct PhysicalAnchor {
    owner: OnceLock<Arc<dyn Send + Sync>>,
}
impl PhysicalAnchor {
    pub(crate) fn install(&self, owner: Arc<dyn Send + Sync>) -> Result<(), Arc<dyn Send + Sync>> {
        self.owner.set(owner)
    }
    pub(crate) fn owner(&self) -> Option<Arc<dyn Send + Sync>> {
        self.owner.get().cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retirement_mailbox::OwnerMailbox;
    use std::sync::atomic::{AtomicUsize, Ordering};

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

    #[test]
    fn reentrant_capture_drop_queues_successor_before_active_anchor_release() {
        let mailbox = Arc::new(
            OwnerMailbox::<(OnDrop, Anchor), ()>::new(
                Arc::new(FailureSignal::default()),
                Arc::new(Wake::default()),
            )
            .unwrap(),
        );
        let active = Arc::new(ActiveRetirement::new());
        let drops = Arc::new(AtomicUsize::new(0));
        let next_ticket = Arc::new(Mutex::new(None));
        let weak_mailbox = Arc::downgrade(&mailbox);
        let weak_active = Arc::downgrade(&active);
        let out_ticket = next_ticket.clone();
        let drop_count = drops.clone();
        let captures = OnDrop(Some(Box::new(move || {
            let mailbox = weak_mailbox.upgrade().unwrap();
            let active = weak_active.upgrade().unwrap();
            assert!(
                !active.is_empty(),
                "active anchor disappeared during capture Drop"
            );
            assert_eq!(drop_count.load(Ordering::SeqCst), 0);
            let successor = mailbox
                .prepare((OnDrop(None), Anchor(drop_count)))
                .unwrap_or_else(|_| panic!("cleanup credit not released before capture Drop"));
            *out_ticket.lock() = Some(successor);
        })));
        let first = mailbox
            .prepare((captures, Anchor(drops.clone())))
            .unwrap_or_else(|_| panic!("first prepare"));
        // Prepared cancellation is the bounded CPU control for detachment;
        // native adapter reaches the same take only after terminal publication.
        let detached = mailbox.cancel_prepared(first).unwrap();
        active
            .stage(detached.owner.0, detached.owner.1)
            .unwrap_or_else(|_| panic!("active busy"));
        active.drop_captures().unwrap();
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(next_ticket.lock().is_some());
        drop(active.finish().unwrap());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let next = mailbox
            .cancel_prepared(next_ticket.lock().take().unwrap())
            .unwrap();
        active
            .stage(next.owner.0, next.owner.1)
            .unwrap_or_else(|_| panic!("successor active busy"));
        active.drop_captures().unwrap();
        drop(active.finish().unwrap());
        assert_eq!(drops.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn capture_panic_retains_anchor_and_refuses_finish_or_retry() {
        let drops = Arc::new(AtomicUsize::new(0));
        let active = ActiveRetirement::new();
        active
            .stage(
                OnDrop(Some(Box::new(|| panic!("capture Drop fault")))),
                Anchor(drops.clone()),
            )
            .unwrap_or_else(|_| panic!("active busy"));
        let panic = catch_unwind(AssertUnwindSafe(|| active.drop_captures())).unwrap_err();
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"capture Drop fault"));
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(matches!(active.finish(), Err(ActiveError::WrongPhase)));
        assert_eq!(active.drop_captures(), Err(ActiveError::WrongPhase));
        assert!(!active.is_empty());
        // CPU fixture ends sole host custody explicitly. Production failure
        // keeps this cell in the static registry instead of dropping it.
        drop(active);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn normal_capture_destruction_precedes_anchor_release() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let capture_events = events.clone();
        let anchor_events = events.clone();
        let active = ActiveRetirement::new();
        active
            .stage(
                OnDrop(Some(Box::new(move || {
                    capture_events.lock().push("capture")
                }))),
                OnDrop(Some(Box::new(move || anchor_events.lock().push("anchor")))),
            )
            .unwrap_or_else(|_| panic!("active busy"));
        assert!(matches!(active.finish(), Err(ActiveError::WrongPhase)));
        active.drop_captures().unwrap();
        assert_eq!(&*events.lock(), &["capture"]);
        drop(active.finish().unwrap());
        assert_eq!(&*events.lock(), &["capture", "anchor"]);
        assert!(active.is_empty());
    }
}

#[cfg(test)]
mod metadata_charge_tests {
    use super::*;
    use crate::{
        memory_admission::ModelDomain,
        retirement_registry::{HostMetadataHandle, MetadataWake},
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct MetadataInbox {
        service: Arc<HostRetirementService>,
        domain: ModelDomain,
        wake: Arc<Wake>,
        drops: Arc<AtomicUsize>,
        panic: bool,
        charge: Mutex<Option<PhysicalCharge>>,
    }
    impl MetadataWake for MetadataInbox {
        fn metadata_wake(&self) -> &Arc<Wake> {
            &self.wake
        }
    }
    // SAFETY: no native/arbitrary callback/payload exists. This fixture opts in
    // with counted handles only, no Weak/raw acquisition. Charge extraction is
    // an engine Mutex::take; final metadata Drop assertions/panic are deliberate
    // source-controlled unwind observations, not a hidden physical owner.
    unsafe impl CleanupInbox for MetadataInbox {
        fn begin_retire(&self) -> bool {
            false
        }
        fn drop_captures(&self) {}
        fn finish_retire(&self) {}
        fn can_unregister(&self) -> bool {
            true
        }
        fn worker_final_metadata_drop(&self) -> bool {
            true
        }
        fn take_final_metadata_charge(&self) -> Option<PhysicalCharge> {
            self.charge.lock().take()
        }
        fn fail(&self, _: Failure) {
            self.wake.notify();
        }
    }
    impl Drop for MetadataInbox {
        fn drop(&mut self) {
            assert!(
                self.service.registry.try_lock().is_some(),
                "final metadata Drop under registry lock"
            );
            assert_eq!(
                self.domain.snapshot().committed,
                8,
                "charge released during final metadata Drop"
            );
            assert!(
                self.service.active_metadata_charge.lock().is_some(),
                "post-drop charge was not independently rooted"
            );
            self.drops.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panic, "injected final metadata Drop panic");
        }
    }
    fn setup(panic: bool) -> (Arc<HostRetirementService>, ModelDomain, Arc<AtomicUsize>) {
        let service = HostRetirementService::manual();
        let domain = ModelDomain::isolated_numeric_fixture(32);
        let pass = domain.reserve(8).unwrap();
        let charge = pass.context().take(8).unwrap().commit(8).unwrap();
        drop(pass);
        let drops = Arc::new(AtomicUsize::new(0));
        let handle = HostMetadataHandle::new(Arc::new(MetadataInbox {
            service: service.clone(),
            domain: domain.clone(),
            wake: service.wake(),
            drops: drops.clone(),
            panic,
            charge: Mutex::new(Some(charge)),
        }));
        let inbox: Arc<dyn CleanupInbox> = (*handle).clone();
        service.register(&inbox).unwrap();
        drop(inbox);
        drop(handle); // Counted release before wake, just like actual adapters.
        (service, domain, drops)
    }
    #[test]
    fn metadata_charge_remains_rooted_through_final_drop_and_returns_afterward() {
        let (service, domain, drops) = setup(false);
        service.manual_pass();
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert!(service.active_metadata_charge.lock().is_none());
        assert_eq!(domain.snapshot().committed, 0);
        assert_eq!(domain.snapshot().pending, 0);
    }
    #[test]
    fn metadata_charge_unwind_keeps_independent_custody_and_later_pass_refuses() {
        let (service, domain, drops) = setup(true);
        let panic = catch_unwind(AssertUnwindSafe(|| service.manual_pass()));
        assert!(panic.is_err(), "injected final metadata panic was missed");
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(domain.snapshot().committed, 8);
        assert!(service.active_metadata_charge.lock().is_some());
        assert!(
            !service.manual_pass(),
            "later pass resumed failed metadata Drop"
        );
        assert_eq!(service.failure.get(), Some(Failure::WorkerFailed));
        assert_eq!(domain.snapshot().committed, 8);
        drop(panic);
        drop(service); // Fake-only teardown; production static never does this.
        assert_eq!(domain.snapshot().committed, 0);
    }
}
