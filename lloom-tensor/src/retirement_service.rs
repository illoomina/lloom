//! Independent host custody for completion-owned resources (B1).
//!
//! Logical callbacks run on native feedback. This worker only destroys payloads
//! after an inbox has detached them and returned its cleanup credit. The static
//! registry, not a native block or the worker stack, is the custody root.
//! Private draft: no native adapter calls `shared`/`register` yet.

use crate::{
    completion_wake::Wake,
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
    fn fail(&self, reason: Failure);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ServiceError {
    StartFailed,
    Failed(Failure),
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
        let entries = self.registry.lock().clone();
        for entry in entries {
            // An arbitrary adapter failure must not unwind past the worker and
            // lose the remaining wakeups. A bad adapter stays rooted, too.
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| entry.fail(reason))) {
                self.panic_payloads.lock().push(payload);
            }
        }
    }

    fn pass(&self) -> bool {
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
            if entry.can_unregister() {
                let mut registry = self.registry.lock();
                // This worker is the only remover; registration only appends.
                if registry
                    .get(index)
                    .is_some_and(|current| Arc::ptr_eq(current, &entry))
                {
                    registry.swap_remove(index);
                }
                drop(registry);
                // entry is the last local handle only AFTER unlocking. Its
                // contract guarantees no physical/panic owner remains inside.
            } else {
                index += 1;
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
