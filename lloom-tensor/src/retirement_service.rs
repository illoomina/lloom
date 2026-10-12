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
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, OnceLock,
    },
    time::{Duration, Instant},
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
/// Without worker-final metadata opt-in, `can_unregister` is true ONLY when
/// no producer capable of installing a payload exists and every resource/panic
/// payload has left the inbox. For immutable worker-final opt-in it may instead
/// report local resource/panic quiescence: removal additionally requires the
/// service's exclusive count2 census under its snapshot/acquisition guard, then
/// revalidation of this predicate while that guard still excludes acquisition.
/// No Weak/raw-Arc path is permitted; count2 proves no foreign producer remains.
/// Opt-in revalidation must be trusted nonpanicking host metadata observation,
/// without registry re-entry or an inverse lock edge to that service guard.
/// Delayed native block copies keep counted roots, and quarantined/unknown
/// payloads keep this predicate false; poisoning is not native terminal proof.
///
/// `fail` publishes admission/wait failure and wakes waiters without dropping
/// pending owners, acquiring a producer lock, or inventing completed status.
pub(crate) unsafe trait CleanupInbox: Send + Sync {
    fn begin_retire(&self) -> bool;
    fn drop_captures(&self);
    fn finish_retire(&self);
    fn can_unregister(&self) -> bool;
    /// Immutable opt-in for inboxes whose warm publishers use atomics only.
    /// Such inboxes never call Wake from last-owner release. Their registered
    /// root must remain eligible for polling until every possible delayed
    /// publication has been consumed: dropping application producers alone is
    /// not enough to permit can_unregister. This engine-only immutable query
    /// must not panic or acquire a lock. Defaults preserve existing adapters.
    fn requires_polling(&self) -> bool {
        false
    }
    /// Opt-in final metadata destruction on the service thread. Safety requires
    /// no Weak/raw-Arc acquisition path: every external clone must start from
    /// an already counted Arc. Service snapshots acquire under registry lock.
    /// Every foreign final release must drop its Arc BEFORE notifying Wake.
    /// An immutable requires_polling opt-in may omit that notification; the
    /// registered polling root then survives every foreign counted reference.
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
    /// Trusted deferred engine notification, outside active/registry guards.
    /// Concurrent worker/shutdown consumers are permitted; each caller retains
    /// this counted inbox until its notification obligation completes.
    fn service_notify(&self) {}
    fn fail(&self, reason: Failure);
}

/// Nonblocking progress for a last-owner teardown, distinct from terminal
/// payload destruction. Registration precedes any owner becoming externally
/// visible; final-owner Drop only publishes a request and notifies `Wake`.
///
/// # Safety
/// `poll` never waits for GPU work, native slots, another inbox, or a producer
/// lock. It returns after bounded progress; a pending dependency must notify the
/// service Wake when its predicate can change. Native commit may run only after
/// exclusive recording ownership is established, without acquiring a replacement.
/// Every owner remains inside this registered inbox across native entry, error,
/// and unwind. Unknown handoff or failed callbacks cannot become terminal by
/// poisoning. An error is retained and observable, not a successful drain.
///
/// Arbitrary capture destruction must use separately rooted active anchors as in
/// `CleanupInbox`; `poll` cannot move the last physical custody onto its stack.
/// `can_unregister` requires all submit-capable owners and all pending/active/
/// quarantined payloads to be gone. `fail` only publishes refusal and wakes.
/// Logical callback progress never depends on this service running `poll`.
pub(crate) unsafe trait LifecycleInbox: Send + Sync {
    fn poll(&self) -> bool;
    fn can_unregister(&self) -> bool;
    fn fail(&self, reason: Failure);
}

#[derive(Clone)]
enum Inbox {
    Cleanup(Arc<dyn CleanupInbox>),
    Lifecycle(Arc<dyn LifecycleInbox>),
}
impl Inbox {
    fn same(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Cleanup(a), Self::Cleanup(b)) => Arc::ptr_eq(a, b),
            (Self::Lifecycle(a), Self::Lifecycle(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
    fn progress(&self) -> bool {
        match self {
            Self::Cleanup(entry) => {
                let ready = entry.begin_retire();
                entry.service_notify();
                if !ready {
                    return false;
                }
                entry.drop_captures();
                entry.finish_retire();
                true
            }
            Self::Lifecycle(entry) => entry.poll(),
        }
    }
    fn can_unregister(&self) -> bool {
        match self {
            Self::Cleanup(entry) => entry.can_unregister(),
            Self::Lifecycle(entry) => entry.can_unregister(),
        }
    }
    /// Called only under the registry snapshot/census lock. The pass holds
    /// one matching inner Arc and the registry holds the other. No extra clone
    /// is made to inspect the opt-in contract. Revalidate local quiescence
    /// only AFTER count2: an earlier offguard observation can precede a final
    /// producer publication and Arc release; the guard now excludes snapshots.
    fn metadata_removable(&self) -> bool {
        match self {
            Self::Cleanup(entry) => {
                if !entry.worker_final_metadata_drop() {
                    return true;
                }
                if Arc::strong_count(entry) != 2 {
                    return false;
                }
                // Arc::strong_count is Relaxed. The pass's own Arc clone RMW
                // precedes this load, so atomic coherence excludes older counts:
                // clone read1 -> wrote2, following any prior foreign Release drop;
                // clone read>=2 -> wrote>=3, so count2 follows a foreign decrement.
                // All subsequent count changes are RMWs in that release sequence.
                // Acquire the foreign writes BEFORE rechecking local quiescence.
                // No Weak/raw resurrection; this guard excludes service snapshots.
                // Proof/source versions: tagged-service-revalidation report.
                std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
                entry.can_unregister()
            }
            Self::Lifecycle(_) => true,
        }
    }
    fn requires_polling(&self) -> bool {
        match self {
            Self::Cleanup(entry) => entry.requires_polling(),
            Self::Lifecycle(_) => false,
        }
    }
    fn worker_final_metadata_drop(&self) -> bool {
        match self {
            Self::Cleanup(entry) => entry.worker_final_metadata_drop(),
            Self::Lifecycle(_) => false,
        }
    }
    fn take_final_metadata_charge(&self) -> Option<PhysicalCharge> {
        match self {
            Self::Cleanup(entry) => entry.take_final_metadata_charge(),
            Self::Lifecycle(_) => None,
        }
    }
    fn fail(&self, reason: Failure) {
        match self {
            Self::Cleanup(entry) => {
                entry.fail(reason);
                entry.service_notify();
            }
            Self::Lifecycle(entry) => entry.fail(reason),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ServiceError {
    StartFailed,
    AllocationFailed,
    Failed(Failure),
}

// Release snapshot references before waking the worker, including on unwind.
// Otherwise the worker can observe the old count then park without a new wake.
struct InboxSnapshot {
    entries: Vec<Inbox>,
    wake: Arc<Wake>,
}
impl Drop for InboxSnapshot {
    fn drop(&mut self) {
        self.entries.clear();
        self.wake.notify();
    }
}

/// Existing legacy registry remains explicit. Priced header roots use one
/// exact Box node instead of unpriced Vec capacity growth. Both routes share
/// this SAME registry guard, service worker, census and notification policy.
#[derive(Default)]
struct InboxRegistry {
    legacy: Vec<Inbox>,
    priced: Option<Box<RegistrationNode>>,
    priced_len: usize,
    polling_len: usize,
}
struct RegistrationNode {
    entry: Option<Inbox>,
    next: Option<Box<RegistrationNode>>,
    previous: Option<std::ptr::NonNull<RegistrationNode>>,
    polling: bool,
}
// SAFETY: previous is an address-only link into this registry's stable Boxes.
// Only the existing registry mutex permits link access/mutation; no link or
// pointee reference escapes except the exclusive host-pass cursor below.
unsafe impl Send for RegistrationNode {}
struct RemovedRegistration {
    _block: Option<Box<RegistrationNode>>,
    _entry: Inbox,
}
#[derive(Clone, Copy)]
enum RegistryCursor {
    Legacy(usize),
    Priced(std::ptr::NonNull<RegistrationNode>),
}
struct PassLease<'a>(&'a AtomicBool);
impl Drop for PassLease<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}
impl InboxRegistry {
    fn len(&self) -> usize {
        self.legacy.len() + self.priced_len
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn iter(&self) -> impl Iterator<Item = &Inbox> {
        self.legacy.iter().chain(
            std::iter::successors(self.priced.as_deref(), |node| node.next.as_deref())
                .map(|node| node.entry.as_ref().expect("queued root")),
        )
    }
    fn push(&mut self, entry: Inbox) {
        self.polling_len += usize::from(entry.requires_polling());
        self.legacy.push(entry);
    }
    fn push_priced(&mut self, mut node: Box<RegistrationNode>) {
        let address = std::ptr::NonNull::from(&mut *node);
        node.next = self.priced.take();
        if let Some(next) = node.next.as_mut() {
            next.previous = Some(address);
        }
        self.polling_len += usize::from(node.polling);
        self.priced = Some(node);
        self.priced_len += 1;
    }
    fn priced_head(&self) -> Option<RegistryCursor> {
        self.priced
            .as_deref()
            .map(|node| RegistryCursor::Priced(std::ptr::NonNull::from(node)))
    }
    /// # Safety
    /// Priced cursor belongs to this registry's elected PassLease. Only that
    /// pass removes nodes; cold registration only prepends stable Boxes. Caller
    /// holds the registry mutex for this borrow and never borrows across unlock.
    unsafe fn entry(&self, cursor: RegistryCursor) -> Option<&Inbox> {
        match cursor {
            RegistryCursor::Legacy(index) => self.legacy.get(index),
            RegistryCursor::Priced(address) => {
                // SAFETY: elected pass cursor remains reachable and stable.
                unsafe { address.as_ref() }.entry.as_ref()
            }
        }
    }
    /// Same elected-pass/mutex contract as entry; next is captured before
    /// offguard progress and is not freed by cold prepend or other consumers.
    unsafe fn next_priced(&self, cursor: RegistryCursor) -> Option<RegistryCursor> {
        match cursor {
            RegistryCursor::Legacy(_) => None,
            RegistryCursor::Priced(address) => {
                // SAFETY: only this pass can remove the current/next nodes.
                unsafe { address.as_ref() }
                    .next
                    .as_deref()
                    .map(|node| RegistryCursor::Priced(std::ptr::NonNull::from(node)))
            }
        }
    }
    fn snapshot(&self) -> Vec<Inbox> {
        self.iter().cloned().collect()
    }
    /// # Safety
    /// Same registry-owned cursor and exclusive PassLease as entry. Pointer
    /// links are dereferenced only while the existing registry guard is held.
    unsafe fn remove(&mut self, cursor: RegistryCursor) -> RemovedRegistration {
        match cursor {
            RegistryCursor::Legacy(index) => {
                let polling = self.legacy[index].requires_polling();
                let entry = self.legacy.swap_remove(index);
                self.polling_len -= usize::from(polling);
                RemovedRegistration {
                    _block: None,
                    _entry: entry,
                }
            }
            RegistryCursor::Priced(address) => {
                // SAFETY: address names the current live node under this guard.
                let previous = unsafe { address.as_ref() }.previous;
                let link = match previous {
                    None => &mut self.priced,
                    Some(mut previous) => {
                        // SAFETY: predecessor is live/reachable; cold prepend
                        // updates this link under the same guard before removal.
                        &mut unsafe { previous.as_mut() }.next
                    }
                };
                assert_eq!(
                    link.as_deref().map(std::ptr::NonNull::from),
                    Some(address),
                    "current cursor remains linked"
                );
                let mut block = link.take().expect("current live node");
                *link = block.next.take();
                if let Some(next) = link.as_mut() {
                    next.previous = previous;
                }
                block.previous = None;
                let entry = block.entry.take().expect("queued root");
                self.priced_len -= 1;
                self.polling_len -= usize::from(block.polling);
                RemovedRegistration {
                    _block: Some(block),
                    _entry: entry,
                }
            }
        }
    }
}
impl Drop for InboxRegistry {
    fn drop(&mut self) {
        while let Some(mut node) = self.priced.take() {
            self.priced = node.next.take();
            let entry = node.entry.take();
            drop(node);
            drop(entry);
        }
    }
}

pub(crate) struct HostRetirementService {
    // Only host registration and this host worker access the registry lock.
    // Native feedback publishes into its rooted inbox and calls wake.notify.
    registry: Mutex<InboxRegistry>,
    // Service-only cursor election; no producer path acquires this lease.
    passing: AtomicBool,
    wake: Arc<Wake>,
    failure: FailureSignal,
    // Cold registry edits publish this summary under the existing registry
    // guard; warm inbox publishers never touch service locks or Wake. The
    // service owns the policy, not allocation callers: 10 ms is an initial,
    // unmeasured latency/idle-cost tradeoff, not an optimum or deadline promise.
    polling: AtomicBool,
    polling_interval: Duration,
    #[cfg(test)]
    test_stop: AtomicBool,
    #[cfg(test)]
    test_idle_parks: std::sync::atomic::AtomicUsize,
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
                registry: Mutex::new(InboxRegistry::default()),
                passing: AtomicBool::new(false),
                wake: Arc::new(Wake::default()),
                failure: FailureSignal::default(),
                polling: AtomicBool::new(false),
                polling_interval: Duration::from_millis(10),
                #[cfg(test)]
                test_stop: AtomicBool::new(false),
                #[cfg(test)]
                test_idle_parks: std::sync::atomic::AtomicUsize::new(0),
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
            registry: Mutex::new(InboxRegistry::default()),
            passing: AtomicBool::new(false),
            wake: Arc::new(Wake::default()),
            failure: FailureSignal::default(),
            polling: AtomicBool::new(false),
            polling_interval: Duration::from_millis(10),
            #[cfg(test)]
            test_stop: AtomicBool::new(false),
            #[cfg(test)]
            test_idle_parks: std::sync::atomic::AtomicUsize::new(0),
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
            entries: self.registry.lock().snapshot(),
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
        self.register_inbox(Inbox::Cleanup(inbox.clone()))
    }

    /// Requested Box storage for one priced registration. Root's same ticket
    /// holds this term through offguard node free and final root Arc destruction.
    pub(crate) fn requested_registration_bytes() -> usize {
        std::alloc::Layout::new::<RegistrationNode>().size()
    }
    /// Cold only; caller's live root already owns an admitted registration term
    /// BEFORE this Box allocation. Root remains privately held through refusal.
    /// New nodes share the original registry/census, not a second inbox/worker.
    pub(crate) fn register_owned(&self, inbox: &Arc<dyn CleanupInbox>) -> Result<(), ServiceError> {
        let node = crate::cold_box::try_new(RegistrationNode {
            entry: Some(Inbox::Cleanup(inbox.clone())),
            next: None,
            previous: None,
            polling: inbox.requires_polling(),
        })
        .map_err(|unallocated| {
            // No registry guard is held; caller still owns the root. No Box
            // exists, so release the input Arc before caller refunds its term.
            drop(unallocated);
            ServiceError::AllocationFailed
        })?;
        let mut candidate = Some(node);
        let mut registry = self.registry.lock();
        if let Some(reason) = self.failure.get() {
            drop(registry);
            drop(candidate); // caller still owns the root, now storage is free
            return Err(ServiceError::Failed(reason));
        }
        if !registry.iter().any(|entry| {
            entry.same(
                candidate
                    .as_ref()
                    .expect("candidate")
                    .entry
                    .as_ref()
                    .expect("candidate root"),
            )
        }) {
            if registry.priced_len == usize::MAX - registry.legacy.len() {
                drop(registry);
                drop(candidate);
                return Err(ServiceError::StartFailed);
            }
            let polling = inbox.requires_polling();
            registry.push_priced(candidate.take().expect("candidate"));
            if polling {
                self.polling.store(true, Ordering::Release);
            }
        }
        drop(registry);
        drop(candidate); // duplicate cold node free is offguard, before caller root
        self.wake.notify();
        Ok(())
    }

    pub(crate) fn register_lifecycle(
        &self,
        inbox: &Arc<dyn LifecycleInbox>,
    ) -> Result<(), ServiceError> {
        self.register_inbox(Inbox::Lifecycle(inbox.clone()))
    }

    fn register_inbox(&self, inbox: Inbox) -> Result<(), ServiceError> {
        let mut registry = self.registry.lock();
        if let Some(reason) = self.failure.get() {
            return Err(ServiceError::Failed(reason));
        }
        if !registry.iter().any(|entry| entry.same(&inbox)) {
            if registry.len() == usize::MAX {
                drop(registry);
                return Err(ServiceError::StartFailed);
            }
            let requires_polling = inbox.requires_polling();
            registry.push(inbox);
            if requires_polling {
                self.polling.store(true, Ordering::Release);
            }
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
            entries: self.registry.lock().snapshot(),
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
        if self
            .passing
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return false;
        }
        let _pass = PassLease(&self.passing);
        // A later pass cannot clear a charge retained across prior failed Drop.
        if self.active_metadata_charge.lock().is_some() {
            self.failure
                .publish(Failure::WorkerFailed, || self.wake.notify());
            return false;
        }
        let mut cursor = Some(RegistryCursor::Legacy(0));
        let mut progressed = false;
        loop {
            let Some(current_cursor) = cursor else {
                return progressed;
            };
            let (entry, next_priced) = {
                let registry = self.registry.lock();
                // SAFETY: this pass owns the sole removal lease and retains no
                // borrowed node reference across registry unlock.
                let entry = unsafe { registry.entry(current_cursor) }.cloned();
                if entry.is_none() {
                    cursor = registry.priced_head();
                    continue;
                }
                // SAFETY: next node remains registered until this same pass
                // reaches it; cold registration cannot remove/move it.
                (entry.expect("located entry"), unsafe {
                    registry.next_priced(current_cursor)
                })
            };
            // Crucially outside the registry lock. Detaching one credit before
            // Drop permits that destructor to register/submit another record.
            progressed |= entry.progress();
            let removed = if entry.can_unregister() {
                let mut registry = self.registry.lock();
                // Snapshot cloning and final metadata census share this lock.
                // Count the actual inner Cleanup Arc, not an enum wrapper.
                // Lifecycle inboxes retain their independent phase contract.
                // SAFETY: exclusive pass + registry guard retain current node.
                if unsafe { registry.entry(current_cursor) }
                    .is_some_and(|current| current.same(&entry) && current.metadata_removable())
                {
                    // Immutable cold policy is counted, not rescanned per
                    // removal; legacy's query completes before root unlink.
                    // SAFETY: same elected cursor; node storage is returned
                    // intact and deallocated only after this guard is released.
                    let removed = unsafe { registry.remove(current_cursor) };
                    self.polling
                        .store(registry.polling_len != 0, Ordering::Release);
                    Some(removed)
                } else {
                    None
                }
            } else {
                None
            };
            cursor = match current_cursor {
                RegistryCursor::Legacy(index) => Some(RegistryCursor::Legacy(
                    index + usize::from(removed.is_none()),
                )),
                RegistryCursor::Priced(_) => next_priced,
            };
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
            // Test stop uses the same snapshot -> check -> park handshake.
            #[cfg(test)]
            if self.test_stop.load(Ordering::Acquire) {
                return;
            }
            #[cfg(feature = "metal")]
            let progressed = objc2::rc::autoreleasepool(|_| self.pass());
            #[cfg(not(feature = "metal"))]
            let progressed = self.pass();
            if !progressed && self.polling.load(Ordering::Acquire) {
                // Existing notifications remain responsive: unlike sleep,
                // timed park returns on a legacy Wake as well as timeout.
                // Snapshot-before-pass + cold register's notify closes a
                // registration racing this check, including false -> park.
                // Productive passes keep the original immediate-drain policy.
                // With no notifications, each retained polled root causes
                // about 100 passes/s; every pass scans unrelated roots too.
                // Cost and eventual latency require measurement separately.
                let Some(deadline) = Instant::now().checked_add(self.polling_interval) else {
                    // Never turn an unrepresentable deadline into indefinite
                    // park or a retry spin. Close admission, preserve static
                    // inbox custody, and stop this worker conservatively.
                    self.failure
                        .publish(Failure::WorkerFailed, || self.wake.notify());
                    self.fail_inboxes(Failure::WorkerFailed);
                    return;
                };
                #[cfg(test)]
                self.test_idle_parks.fetch_add(1, Ordering::Release);
                self.wake.park(seen, Some(deadline));
            } else if !progressed {
                #[cfg(test)]
                self.test_idle_parks.fetch_add(1, Ordering::Release);
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

    /// Trusted custody transfer into an already rooted, staged C. The editor
    /// borrows C: it must move source owners directly into C and must not run
    /// arbitrary user code or return an owned capture tuple. If it unwinds,
    /// every completed transfer remains in this registered active record with A.
    pub(crate) fn edit_staged<R>(&self, edit: impl FnOnce(&mut C) -> R) -> Result<R, ActiveError> {
        let mut slot = self.record.lock();
        let record = slot.as_mut().ok_or(ActiveError::Empty)?;
        if record.state != ActivePhase::Staged {
            return Err(ActiveError::WrongPhase);
        }
        Ok(edit(record.captures.as_mut().expect("staged captures")))
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

/// One typed setup-time anchor shared by submit-capable engine wrappers. It is NOT
/// captured by native blocks or retained by an idle Inbox; only rooted payloads
/// clone its owner across native handoff. Standalone raw wrappers may omit it.
pub(crate) struct PhysicalAnchor<T> {
    owner: OnceLock<T>,
}
impl<T> Default for PhysicalAnchor<T> {
    fn default() -> Self {
        Self {
            owner: OnceLock::new(),
        }
    }
}
impl<T> PhysicalAnchor<T> {
    pub(crate) fn install(&self, owner: T) -> Result<(), T> {
        self.owner.set(owner)
    }
    pub(crate) fn try_owner<E>(
        &self,
        clone: impl FnOnce(&T) -> Result<T, E>,
    ) -> Result<Option<T>, E> {
        self.owner.get().map(clone).transpose()
    }
}
impl<T: Clone> PhysicalAnchor<T> {
    pub(crate) fn owner(&self) -> Option<T> {
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
    struct LifecycleControl {
        requested: std::sync::atomic::AtomicBool,
        terminal: std::sync::atomic::AtomicBool,
        phase: AtomicUsize,
        anchor: Mutex<Option<Anchor>>,
        failure: FailureSignal,
        wake: Arc<Wake>,
    }
    impl LifecycleControl {
        fn new(service: &Arc<HostRetirementService>, drops: Arc<AtomicUsize>) -> Arc<Self> {
            let control = Arc::new(Self {
                requested: std::sync::atomic::AtomicBool::new(false),
                terminal: std::sync::atomic::AtomicBool::new(false),
                phase: AtomicUsize::new(0),
                anchor: Mutex::new(Some(Anchor(drops))),
                failure: FailureSignal::default(),
                wake: service.wake(),
            });
            service
                .register_lifecycle(&(control.clone() as Arc<dyn LifecycleInbox>))
                .unwrap();
            control
        }
        fn request(&self) {
            self.requested.store(true, Ordering::Release);
            self.wake.notify();
        }
        fn complete(&self) {
            self.terminal.store(true, Ordering::Release);
            self.wake.notify();
        }
    }
    // SAFETY: fake native completion is one atomic predicate. No native object,
    // blocking wait, arbitrary capture, or owner transfer onto a callback stack.
    // Anchor::drop only increments its independent test counter.
    unsafe impl LifecycleInbox for LifecycleControl {
        fn poll(&self) -> bool {
            if self.failure.get().is_some() {
                return false;
            }
            match self.phase.load(Ordering::Acquire) {
                0 if self.requested.load(Ordering::Acquire) => {
                    self.phase.store(1, Ordering::Release);
                    true
                }
                1 if self.terminal.load(Ordering::Acquire) => {
                    let anchor = self.anchor.lock().take();
                    drop(anchor);
                    self.phase.store(2, Ordering::Release);
                    true
                }
                _ => false,
            }
        }
        fn can_unregister(&self) -> bool {
            self.phase.load(Ordering::Acquire) == 2
        }
        fn fail(&self, reason: Failure) {
            self.failure.publish(reason, || self.wake.notify());
        }
    }

    #[test]
    fn pending_lifecycle_preserves_owner_without_blocking_terminal_cleanup() {
        use crate::host_anchor::HostAnchorRegistry;
        let service = HostRetirementService::manual();
        let lifecycle_drops = Arc::new(AtomicUsize::new(0));
        let cleanup_drops = Arc::new(AtomicUsize::new(0));
        let control = LifecycleControl::new(&service, lifecycle_drops.clone());
        let cleanup = HostAnchorRegistry::manual(
            Arc::new(FailureSignal::default()),
            service.clone(),
            Arc::new(Wake::default()),
        );
        let permit = cleanup
            .reserve((), Anchor(cleanup_drops.clone()), |_| Ok(()))
            .unwrap_or_else(|_| panic!("fixture cleanup registration"));
        assert!(!service.manual_pass());
        control.request();
        assert!(service.manual_pass());
        assert_eq!(control.phase.load(Ordering::Acquire), 1);
        drop(permit);
        assert!(service.manual_pass());
        assert_eq!(cleanup_drops.load(Ordering::SeqCst), 1);
        assert_eq!(lifecycle_drops.load(Ordering::SeqCst), 0);
        assert!(!service.manual_pass());
        control.complete();
        assert!(service.manual_pass());
        assert_eq!(lifecycle_drops.load(Ordering::SeqCst), 1);
        assert_eq!(control.phase.load(Ordering::Acquire), 2);
    }

    #[test]
    fn closing_lifecycle_retains_unknown_ownership_and_refuses_new_registration() {
        let service = HostRetirementService::manual();
        let drops = Arc::new(AtomicUsize::new(0));
        let control = LifecycleControl::new(&service, drops.clone());
        control.request();
        assert!(service.manual_pass());
        service.close_admission();
        assert_eq!(control.failure.get(), Some(Failure::Closing));
        control.complete();
        assert!(!service.manual_pass());
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert!(!control.can_unregister());
        assert_eq!(
            service.register_lifecycle(&(control.clone() as Arc<dyn LifecycleInbox>)),
            Err(ServiceError::Failed(Failure::Closing))
        );
        drop(control);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        // Fake fixture explicitly ends all host custody. Production's static
        // service does not disappear on Closing or infer terminality from it.
        drop(service);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }
}

#[cfg(test)]
mod metadata_charge_tests {
    use super::*;
    use crate::{
        memory_admission::ModelDomain,
        retirement_registry::{HostMetadataHandle, MetadataWake},
    };
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

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
    struct StaleQuiescenceInbox {
        service: Arc<HostRetirementService>,
        domain: ModelDomain,
        charge: Mutex<Option<PhysicalCharge>>,
        producer: Mutex<Option<Arc<dyn CleanupInbox>>>,
        first_observation: AtomicBool,
        pending: AtomicBool,
        finished: AtomicUsize,
        drops: Arc<AtomicUsize>,
    }
    // SAFETY: fixed fake payload, no native/use/callback obligations. Its first
    // offguard empty observation deterministically performs a final publication
    // then producer Arc release. Revalidation observes atoms only, no registry
    // re-entry or inverse lock. Actual Drop witnesses retained final charge.
    unsafe impl CleanupInbox for StaleQuiescenceInbox {
        fn begin_retire(&self) -> bool {
            self.pending.swap(false, Ordering::AcqRel)
        }
        fn drop_captures(&self) {}
        fn finish_retire(&self) {
            self.finished.fetch_add(1, Ordering::Release);
        }
        fn can_unregister(&self) -> bool {
            let empty = !self.pending.load(Ordering::Acquire);
            if self.first_observation.swap(false, Ordering::AcqRel) {
                assert!(
                    self.service.registry.try_lock().is_some(),
                    "first local observation is offguard"
                );
                // Actual sequence at the production predicate/census seam:
                // observe empty -> final publish -> last producer root release.
                self.pending.store(true, Ordering::Release);
                let producer = self.producer.lock().take().expect("one fake producer");
                drop(producer);
                return empty;
            }
            empty && self.finished.load(Ordering::Acquire) == 1
        }
        fn worker_final_metadata_drop(&self) -> bool {
            true
        }
        fn take_final_metadata_charge(&self) -> Option<PhysicalCharge> {
            self.charge.lock().take()
        }
        fn fail(&self, _: Failure) {}
    }
    impl Drop for StaleQuiescenceInbox {
        fn drop(&mut self) {
            assert_eq!(
                self.finished.load(Ordering::Acquire),
                1,
                "stale quiescence must not discard final publication"
            );
            assert!(
                self.service.registry.try_lock().is_some(),
                "metadata Drop offguard"
            );
            assert_eq!(
                self.domain.snapshot().committed,
                8,
                "charge spans actual metadata Drop"
            );
            assert!(
                self.service.active_metadata_charge.lock().is_some(),
                "service charge custody"
            );
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }
    #[test]
    fn worker_final_census_rechecks_quiescence_after_last_publication() {
        let service = HostRetirementService::manual();
        let domain = ModelDomain::isolated_numeric_fixture(16);
        let charge = domain
            .reserve(8)
            .unwrap()
            .context()
            .take(8)
            .unwrap()
            .commit(8)
            .unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let inbox = Arc::new(StaleQuiescenceInbox {
            service: service.clone(),
            domain: domain.clone(),
            charge: Mutex::new(Some(charge)),
            producer: Mutex::new(None),
            first_observation: AtomicBool::new(true),
            pending: AtomicBool::new(false),
            finished: AtomicUsize::new(0),
            drops: drops.clone(),
        });
        let erased: Arc<dyn CleanupInbox> = inbox.clone();
        *inbox.producer.lock() = Some(erased.clone());
        service.register(&erased).unwrap();
        drop(erased);
        drop(inbox); // Now only service + fake delayed producer roots exist.
        assert!(!service.manual_pass());
        assert_eq!(
            service.registry.lock().len(),
            1,
            "pending publication retains service root"
        );
        assert_eq!(drops.load(Ordering::Relaxed), 0);
        assert_eq!(domain.snapshot().committed, 8);
        assert!(service.manual_pass());
        assert!(service.registry.lock().is_empty());
        assert_eq!(drops.load(Ordering::Relaxed), 1);
        assert_eq!(domain.snapshot().committed, 0);
    }
}

#[cfg(test)]
mod polling_tests {
    use super::*;
    use crate::memory_admission::ModelDomain;
    use std::sync::atomic::AtomicUsize;
    use std::thread::JoinHandle;

    // Test-only teardown wakes an indefinite park too, including the mutation
    // that removes polling eligibility. worker() is the production decision.
    struct RunningWorker {
        service: Arc<HostRetirementService>,
        thread: Option<JoinHandle<()>>,
    }
    impl RunningWorker {
        fn start(service: Arc<HostRetirementService>) -> Self {
            let worker = service.clone();
            let thread = std::thread::Builder::new()
                .name("957-tagged-polling-control".into())
                .spawn(move || worker.worker())
                .unwrap();
            Self {
                service,
                thread: Some(thread),
            }
        }
    }
    impl Drop for RunningWorker {
        fn drop(&mut self) {
            self.service.test_stop.store(true, Ordering::Release);
            self.service.wake.notify();
            self.thread.take().unwrap().join().unwrap();
        }
    }
    fn eventually(predicate: impl Fn() -> bool) {
        let deadline = Instant::now().checked_add(Duration::from_secs(2)).unwrap();
        while !predicate() {
            assert!(
                Instant::now() < deadline,
                "eventual progress watchdog expired"
            );
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    struct LegacyInbox;
    // SAFETY: no producer, native work, callback, physical charge or payload.
    unsafe impl CleanupInbox for LegacyInbox {
        fn begin_retire(&self) -> bool {
            false
        }
        fn drop_captures(&self) {}
        fn finish_retire(&self) {}
        fn can_unregister(&self) -> bool {
            true
        }
        fn fail(&self, _: Failure) {}
    }
    #[test]
    fn old_adapter_default_does_not_enroll_idle_polling() {
        let service = HostRetirementService::manual();
        let inbox: Arc<dyn CleanupInbox> = Arc::new(LegacyInbox);
        assert!(!inbox.requires_polling());
        service.register(&inbox).unwrap();
        assert!(!service.polling.load(Ordering::Acquire));
        assert!(!service.manual_pass());
        assert!(service.registry.lock().is_empty());
        assert!(!service.polling.load(Ordering::Acquire));
    }

    struct PollingInbox {
        service: Arc<HostRetirementService>,
        domain: ModelDomain,
        charge: Mutex<Option<PhysicalCharge>>,
        producer_alive: AtomicBool,
        pending: AtomicUsize,
        passes: Arc<AtomicUsize>,
        finished: Arc<AtomicUsize>,
        first_idle: AtomicUsize,
        interrupted_drain: Arc<AtomicBool>,
        drops: Arc<AtomicUsize>,
    }
    // SAFETY: this bounded host fixture owns no native work or arbitrary
    // callbacks. Three release-published scalar records retire once each.
    // Unregistration requires producer gone and all three consumed; counted
    // foreign Arcs forbid final metadata destruction until publisher Drop ends.
    // No Weak/raw Arc acquisition exists. The immutable opt-in uses no lock.
    // Only the one service worker takes the charge and destroys metadata.
    unsafe impl CleanupInbox for PollingInbox {
        fn begin_retire(&self) -> bool {
            self.passes.fetch_add(1, Ordering::Release);
            self.pending
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |pending| {
                    pending.checked_sub(1)
                })
                .is_ok()
        }
        fn drop_captures(&self) {}
        fn finish_retire(&self) {
            let idle = self.service.test_idle_parks.load(Ordering::Acquire);
            let first = self
                .first_idle
                .compare_exchange(usize::MAX, idle, Ordering::AcqRel, Ordering::Acquire)
                .unwrap_or_else(|first| first);
            if first != usize::MAX && first != idle {
                self.interrupted_drain.store(true, Ordering::Release);
            }
            self.finished.fetch_add(1, Ordering::Release);
        }
        fn can_unregister(&self) -> bool {
            !self.producer_alive.load(Ordering::Acquire)
                && self.finished.load(Ordering::Acquire) == 3
        }
        fn requires_polling(&self) -> bool {
            true
        }
        fn worker_final_metadata_drop(&self) -> bool {
            true
        }
        fn take_final_metadata_charge(&self) -> Option<PhysicalCharge> {
            self.charge.lock().take()
        }
        fn fail(&self, _: Failure) {}
    }
    impl Drop for PollingInbox {
        fn drop(&mut self) {
            assert!(
                self.service.registry.try_lock().is_some(),
                "metadata destructor ran under registry guard"
            );
            assert_eq!(self.finished.load(Ordering::Acquire), 3);
            assert_eq!(
                self.domain.snapshot().committed,
                8,
                "metadata charge returned before actual destructor"
            );
            assert!(
                self.service.active_metadata_charge.lock().is_some(),
                "metadata charge lacked independent custody during destructor"
            );
            self.drops.fetch_add(1, Ordering::Release);
        }
    }
    struct AtomicPublisher {
        inbox: Arc<PollingInbox>,
    }
    impl Drop for AtomicPublisher {
        fn drop(&mut self) {
            // This publication and following counted Arc decrement use atomics;
            // no wake, allocation, native Drop or final physical charge Drop.
            self.inbox.pending.store(3, Ordering::Release);
            self.inbox.producer_alive.store(false, Ordering::Release);
        }
    }
    #[test]
    fn atomics_only_final_publication_progresses_after_idle_without_owner() {
        let service = HostRetirementService::manual();
        let running = RunningWorker::start(service.clone());
        // Start from the legacy indefinite park; cold registration must wake it.
        eventually(|| service.wake.test_is_parked());
        let domain = ModelDomain::isolated_numeric_fixture(32);
        let pass = domain.reserve(8).unwrap();
        let charge = pass.context().take(8).unwrap().commit(8).unwrap();
        drop(pass);
        let passes = Arc::new(AtomicUsize::new(0));
        let finished = Arc::new(AtomicUsize::new(0));
        let drops = Arc::new(AtomicUsize::new(0));
        let interrupted_drain = Arc::new(AtomicBool::new(false));
        let publisher = AtomicPublisher {
            inbox: Arc::new(PollingInbox {
                service: service.clone(),
                domain: domain.clone(),
                charge: Mutex::new(Some(charge)),
                producer_alive: AtomicBool::new(true),
                pending: AtomicUsize::new(0),
                passes: passes.clone(),
                finished: finished.clone(),
                first_idle: AtomicUsize::new(usize::MAX),
                interrupted_drain: interrupted_drain.clone(),
                drops: drops.clone(),
            }),
        };
        let inbox: Arc<dyn CleanupInbox> = publisher.inbox.clone();
        service.register(&inbox).unwrap();
        service.register(&inbox).unwrap();
        assert_eq!(service.registry.lock().len(), 1, "duplicate cold root");
        drop(inbox);
        // Both cold notifications have returned. Queue observation (Skip,
        // never unpark) proves validation accepted the final registration epoch,
        // not merely waiter enrollment with an old seen value that returns
        // Invalid. Otherwise the no-polling mutant can drain by that stale wake.
        eventually(|| passes.load(Ordering::Acquire) > 0 && service.wake.test_is_parked());
        let wake_before = service.wake.snapshot();
        drop(publisher);
        assert_eq!(
            service.wake.snapshot(),
            wake_before,
            "warm publication notified Wake"
        );
        eventually(|| drops.load(Ordering::Acquire) == 1 && domain.snapshot().committed == 0);
        assert_eq!(finished.load(Ordering::Acquire), 3);
        assert!(
            !interrupted_drain.load(Ordering::Acquire),
            "productive drain parked between records"
        );
        assert!(service.registry.lock().is_empty());
        assert!(!service.polling.load(Ordering::Acquire));
        assert_eq!(service.failure.get(), None);
        drop(running); // Stop+notify+join even after removal restores indefinite wait.
    }
}

#[cfg(test)]
mod priced_cursor_controls {
    use super::*;
    use std::sync::atomic::AtomicUsize;
    struct CursorInbox {
        id: usize,
        polled: bool,
        removable: AtomicBool,
        visits: Arc<Mutex<Vec<usize>>>,
        action: Mutex<Option<Box<dyn FnOnce() + Send + Sync>>>,
    }
    // SAFETY: manual scalar fake has no payload/native handoff or delayed
    // publisher. Local removable=true means no resource exists to detach.
    // Actions exercise offguard cold registration; nested pass is refused by
    // the real production election, never recursive payload destruction.
    unsafe impl CleanupInbox for CursorInbox {
        fn begin_retire(&self) -> bool {
            self.visits.lock().push(self.id);
            let action = self.action.lock().take();
            if let Some(action) = action {
                action();
            }
            false
        }
        fn drop_captures(&self) {
            panic!("fake has no captures")
        }
        fn finish_retire(&self) {
            panic!("fake has no anchors")
        }
        fn can_unregister(&self) -> bool {
            self.removable.load(Ordering::Acquire)
        }
        fn requires_polling(&self) -> bool {
            self.polled
        }
        fn fail(&self, _: Failure) {}
    }
    fn inbox(id: usize, polled: bool, visits: &Arc<Mutex<Vec<usize>>>) -> Arc<CursorInbox> {
        Arc::new(CursorInbox {
            id,
            polled,
            removable: AtomicBool::new(false),
            visits: visits.clone(),
            action: Mutex::new(None),
        })
    }
    fn enroll(service: &HostRetirementService, inbox: &Arc<CursorInbox>) {
        // Structural fixture only: no claim to price the fake's own Arc/control.
        let root: Arc<dyn CleanupInbox> = inbox.clone();
        service.register_owned(&root).unwrap();
    }
    fn assert_registry(
        service: &HostRetirementService,
        expected: &[&Arc<CursorInbox>],
        polling: usize,
    ) {
        let registry = service.registry.lock();
        assert_eq!(registry.len(), expected.len());
        assert_eq!(registry.polling_len, polling);
        for (actual, expected) in registry.iter().zip(expected) {
            let reference: Arc<dyn CleanupInbox> = (*expected).clone();
            assert!(actual.same(&Inbox::Cleanup(reference)));
        }
        assert_eq!(service.polling.load(Ordering::Acquire), polling != 0);
    }
    #[test]
    fn priced_cursor_prepend_offguard_and_reentrant_pass_preserve_original_walk() {
        let service = HostRetirementService::manual();
        let visits = Arc::new(Mutex::new(Vec::new()));
        let tail = inbox(1, false, &visits);
        let head = inbox(2, true, &visits);
        let added = inbox(3, true, &visits);
        let nested_calls = Arc::new(AtomicUsize::new(0));
        enroll(&service, &tail);
        enroll(&service, &head);
        let weak_service = Arc::downgrade(&service);
        let added_for_action = added.clone();
        let nested_for_action = nested_calls.clone();
        *head.action.lock() = Some(Box::new(move || {
            let service = weak_service.upgrade().unwrap();
            assert!(
                service.registry.try_lock().is_some(),
                "progress ran under registry guard"
            );
            enroll(&service, &added_for_action);
            assert!(
                !service.manual_pass(),
                "nested pass acquired existing cursor"
            );
            nested_for_action.fetch_add(1, Ordering::SeqCst);
        }));
        assert!(!service.manual_pass());
        assert_eq!(
            *visits.lock(),
            [2, 1],
            "cold prepend changed in-flight walk"
        );
        assert_eq!(nested_calls.load(Ordering::SeqCst), 1);
        assert_registry(&service, &[&added, &head, &tail], 2);
        visits.lock().clear();
        assert!(!service.manual_pass());
        assert_eq!(
            *visits.lock(),
            [3, 2, 1],
            "new head was not visited next pass"
        );
        for root in [&tail, &head, &added] {
            root.removable.store(true, Ordering::Release);
        }
        assert!(!service.manual_pass());
        assert_registry(&service, &[], 0);
        assert!(!service.passing.load(Ordering::Acquire));
    }
    #[test]
    fn priced_cursor_middle_head_tail_unlink_preserves_polling_summary() {
        let service = HostRetirementService::manual();
        let visits = Arc::new(Mutex::new(Vec::new()));
        let tail = inbox(1, false, &visits);
        let middle = inbox(2, true, &visits);
        let head = inbox(3, true, &visits);
        for root in [&tail, &middle, &head] {
            enroll(&service, root);
        }
        assert_registry(&service, &[&head, &middle, &tail], 2);
        middle.removable.store(true, Ordering::Release);
        service.manual_pass();
        assert_registry(&service, &[&head, &tail], 1);
        head.removable.store(true, Ordering::Release);
        service.manual_pass();
        assert_registry(&service, &[&tail], 0);
        tail.removable.store(true, Ordering::Release);
        service.manual_pass();
        assert_registry(&service, &[], 0);
        assert!(!service.passing.load(Ordering::Acquire));
    }
}
