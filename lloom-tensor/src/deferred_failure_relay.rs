//! Engine-only atomic refusal publication with existing service-owned wake.
//! Cold constructor enrollment and exact root custody are required separately.
use crate::{
    completion_wake::Wake,
    retirement_owners::{Failure, FailureSignal},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

/// Cold preheld identities only; no Queue, Commands, namespace, native payload,
/// or HeaderPool backedge. HeaderRoot retains this relay through every producer
/// publication and the following service pass. Application code receives no
/// relay Arc/acquisition API. A private payload may retain the same relay for
/// service-only readiness, but cannot export it as a producer capability.
///
/// Any operation producer publishes through its genuine HeaderStrong helper,
/// not an independently surviving relay Arc. The root's immutable polling and
/// pending-aware final census are part of this notification progress contract.
/// This is not a native terminality or physical-release authorization.
pub(crate) struct DeferredFailureRelay {
    failure: Arc<FailureSignal>,
    wake: Arc<Wake>,
    pending: AtomicBool,
}
impl DeferredFailureRelay {
    /// Cold only. Caller supplies the SAME actual queue/completion failure
    /// signal and Wake observed by blocked waiters, not an arbitrary pair.
    /// Concrete caller identity/enrollment remains separately unproved here.
    /// Constructor price includes relay/Arc metadata separately from current
    /// HeaderStorage Layout quote; no total allocation cap is inferred.
    pub(crate) fn cold(failure: Arc<FailureSignal>, wake: Arc<Wake>) -> Self {
        Self {
            failure,
            wake,
            pending: AtomicBool::new(false),
        }
    }
    /// Private engine producer/readiness code; root custody spans this borrow.
    /// Every access is atomic. Existing first-error status is never cleared.
    pub(crate) fn publish(&self, reason: Failure) {
        self.failure.publish(reason, || {});
        self.pending.store(true, Ordering::Release);
    }
    /// Trusted existing worker pass OR host shutdown/failure snapshot, after
    /// adapter active and registry guards have been released. These consumers
    /// may run concurrently: each AcqRel swap that observes true claims one
    /// coalesced notification, and each such claimant must finish Wake::notify
    /// before releasing its counted root/snapshot. A false observation does not
    /// cancel another claimant. Publications after a swap remain pending for
    /// the next immutable-polling pass. Stable root/relay Wake custody spans
    /// notify; no exclusive-consumer assumption is required.
    pub(crate) fn notify_on_service(&self) {
        if self.pending.swap(false, Ordering::AcqRel) {
            self.wake.notify();
        }
    }
    /// Root can_unregister includes this after its existing local predicates;
    /// service repeats it AFTER count2+Acquire fence before final root removal.
    pub(crate) fn quiet(&self) -> bool {
        !self.pending.load(Ordering::Acquire)
    }
}
