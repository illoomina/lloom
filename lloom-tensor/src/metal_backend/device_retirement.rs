//! Independently registered namespace custody. No Commands/logical Queue backedge.
//! This enrollment does not close raw-derived alias, IO, or admission coverage.
use super::allocation_namespace::AllocationNamespace;
use crate::{
    allocation_owner::AllocationLease,
    completion_wake::Wake,
    host_anchor::{HostAnchorPermit, HostAnchorRegistry},
    kernels::{
        metal::{Backings, Buffer, Device, InFlightHold, ResidencySet},
        Kernels, MetalKernelError,
    },
    retirement_owners::FailureSignal,
};
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_metal::MTLBuffer;
use parking_lot::Mutex;
use std::sync::{Arc, Weak};

// C contains namespace cache/recorder containers and arbitrary destruction. A remains
// independently rooted if any C destructor unwinds; never put these containers
// in trusted A merely because their outer type is private.
struct DeviceCaptures {
    _namespace: Arc<AllocationNamespace>,
    _seed: Arc<Mutex<Buffer>>,
    _kernels: Arc<Kernels>,
}
// Constructor census: namespace Backings' two trait objects are exclusively
// engine HeapChunks or DeviceBuffer; resident/mapped fields are DeviceBuffer.
// HeapChunks contains native heap/owner/membership/range/reservation metadata;
// its sole production AllocationOwner anchor is HeapAnchor (no user Drop).
// InFlightHold contains only native Fence or concrete Buffer leases/membership.
// Ledger contains atomics and weak tenant/machine metadata, not tenant payloads.
// All arbitrary cache/recorder containers stay in C above. Native retains follow
// the same trusted engine release contract as existing NativeOwners/Membership.
struct DeviceAnchors {
    _backings: Arc<Backings>,
    _hold: Option<Arc<InFlightHold>>,
    _seed: AllocationLease<Retained<ProtocolObject<dyn MTLBuffer>>>,
    _device: Device,
    _residency: Arc<ResidencySet>,
    _ledger: Arc<crate::kernels::metal::ledger::Ledger>,
}
// SAFETY: same retained Metal buffer cross-thread release contract as Buffer;
// all other fields are engine metadata/native owners from the constructor census.
unsafe impl Send for DeviceAnchors {}
unsafe impl Sync for DeviceAnchors {}
type Permit = HostAnchorPermit<DeviceCaptures, DeviceAnchors>;

/// Only counted host-record metadata; final Drop cannot run the C/A payload.
/// Keep this concrete type in native owner fields, rather than an arbitrary
/// Arc destructor smuggled into trusted A.
#[derive(Clone)]
pub(crate) struct NamespaceOwner(Arc<Permit>);

/// Weak identity memo only. It cannot keep a namespace payload alive and cannot
/// confuse reused addresses: its Weak keeps the old Arc allocation distinct.
/// EntryState resets it only after successful replacement, not receipt or Err.
pub(crate) struct GenerationMemo<T>(Option<Weak<T>>);
impl<T> Default for GenerationMemo<T> {
    fn default() -> Self {
        Self(None)
    }
}
impl<T> GenerationMemo<T> {
    fn contains(&self, owner: &Arc<T>) -> bool {
        self.0
            .as_ref()
            .is_some_and(|old| old.as_ptr() == Arc::as_ptr(owner))
    }
    fn remember(&mut self, owner: &Arc<T>) {
        self.0 = Some(Arc::downgrade(owner));
    }
    pub(crate) fn reset(&mut self) {
        self.0 = None;
    }
}
#[derive(Default)]
pub(crate) struct NamespaceMemo(GenerationMemo<Permit>);
impl NamespaceMemo {
    pub(crate) fn reset(&mut self) {
        self.0.reset();
    }
}
impl NamespaceOwner {
    pub(crate) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub(crate) fn already_recorded(&self, memo: &NamespaceMemo) -> bool {
        memo.0.contains(&self.0)
    }
    pub(crate) fn remember(&self, memo: &mut NamespaceMemo) {
        memo.0.remember(&self.0);
    }
}

// Production and scalar controls share the actual C/A registration boundary.
fn register_root<C: Send + Sync + 'static, A: Send + Sync + 'static>(
    registry: &HostAnchorRegistry<C, A>,
    captures: C,
    anchors: A,
) -> Result<Arc<HostAnchorPermit<C, A>>, crate::host_anchor::AnchorRejected<C, A>> {
    registry
        .reserve(captures, anchors, |_| Ok(()))
        .map(Arc::new)
}

pub(crate) struct DeviceTeardown {
    owner: NamespaceOwner,
}
impl DeviceTeardown {
    pub(crate) fn new(
        namespace: Arc<AllocationNamespace>,
        seed: Arc<Mutex<Buffer>>,
        kernels: Arc<Kernels>,
        hold: Option<Arc<InFlightHold>>,
        device: Device,
        residency: Arc<ResidencySet>,
        ledger: Arc<crate::kernels::metal::ledger::Ledger>,
    ) -> Result<Arc<Self>, MetalKernelError> {
        // Cold constructor: seed borrow is released before registration/growth.
        let seed_lease = seed.lock().allocation_lease();
        let anchors = DeviceAnchors {
            _backings: namespace.backings.clone(),
            _hold: hold,
            _seed: seed_lease,
            _device: device,
            _residency: residency,
            _ledger: ledger,
        };
        let captures = DeviceCaptures {
            _namespace: namespace,
            _seed: seed,
            _kernels: kernels,
        };
        let registry = HostAnchorRegistry::new(
            Arc::new(FailureSignal::default()),
            Arc::new(Wake::default()),
        )
        .map_err(|_| MetalKernelError::QueueUnavailable("namespace host root unavailable"))?;
        let permit = register_root(&registry, captures, anchors).map_err(|rejected| {
            crate::kernels::err::CompletionRegistrationError::new(
                MetalKernelError::QueueUnavailable("namespace host root registration refused"),
                rejected.owner,
            )
        })?;
        Ok(Arc::new(Self {
            owner: NamespaceOwner(permit),
        }))
    }
    pub(crate) fn namespace_owner(&self) -> NamespaceOwner {
        self.owner.clone()
    }
    pub(crate) fn physical_owner(&self) -> NamespaceOwner {
        self.owner.clone()
    }
    pub(crate) fn handles(self: &Arc<Self>) -> usize {
        Arc::strong_count(self)
    }
}
// No wait/remove_all. Core's terminal cleanup releases its default pin, while
// unknown/error/unwind retains Core. Native raw owners and M4 cached slots retain
// generation-specific namespace pins until actual native owners are released.
// Targeted Membership cleanup, not global completion Result, removes members.

#[cfg(test)]
mod namespace_controls {
    use super::*;
    use crate::{
        retirement_mailbox::CleanupPolicy, retirement_registry::OwnerRegistry,
        retirement_service::HostRetirementService,
    };
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Capture {
        drops: Arc<AtomicUsize>,
        panic: bool,
    }
    impl Drop for Capture {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::SeqCst);
            assert!(!self.panic, "injected namespace capture panic");
        }
    }
    struct Anchor(Arc<AtomicUsize>);
    impl Drop for Anchor {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }
    type FakePermit = HostAnchorPermit<Capture, Anchor>;
    fn root(
        service: &Arc<HostRetirementService>,
        captures: &Arc<AtomicUsize>,
        anchors: &Arc<AtomicUsize>,
        panic: bool,
    ) -> Arc<FakePermit> {
        let registry = HostAnchorRegistry::manual(
            Arc::new(FailureSignal::default()),
            service.clone(),
            service.wake(),
        );
        register_root(
            &registry,
            Capture {
                drops: captures.clone(),
                panic,
            },
            Anchor(anchors.clone()),
        )
        .unwrap_or_else(|_| panic!("fake namespace root registration"))
    }
    fn drain(service: &HostRetirementService) {
        for _ in 0..128 {
            service.manual_pass();
        }
    }
    #[test]
    fn namespace_roots_terminal_join_releases_two_distinct_scopes() {
        let service = HostRetirementService::manual();
        let captures = Arc::new(AtomicUsize::new(0));
        let anchors = Arc::new(AtomicUsize::new(0));
        let first = root(&service, &captures, &anchors, false);
        let second = root(&service, &captures, &anchors, false);
        let native = OwnerRegistry::<(), Vec<Arc<FakePermit>>, ()>::manual_with_waiters(
            Arc::new(FailureSignal::default()),
            service.clone(),
            service.wake(),
            CleanupPolicy::ReturnedHandoff,
        );
        let record = native
            .prepare((), vec![first.clone(), second.clone()], |_| Ok(()))
            .unwrap_or_else(|_| panic!("fake native owner registration"));
        drop(first);
        drop(second);
        record.begin_handoff().unwrap();
        record
            .publish_feedback(())
            .unwrap_or_else(|_| panic!("fake feedback"));
        record.run_callbacks(|_, _, _| {}, || {}).unwrap();
        drain(&service);
        assert_eq!(
            anchors.load(Ordering::SeqCst),
            0,
            "feedback alone released namespace roots"
        );
        record.handoff_returned().unwrap();
        drain(&service);
        assert_eq!(
            captures.load(Ordering::SeqCst),
            2,
            "scoped captures not independently retired"
        );
        assert_eq!(
            anchors.load(Ordering::SeqCst),
            2,
            "terminal join lost scoped physical release"
        );
    }
    #[test]
    fn namespace_roots_unknown_join_retains_actual_registered_payload() {
        let service = HostRetirementService::manual();
        let captures = Arc::new(AtomicUsize::new(0));
        let anchors = Arc::new(AtomicUsize::new(0));
        let owner = root(&service, &captures, &anchors, false);
        let native = OwnerRegistry::<(), Arc<FakePermit>, ()>::manual_with_waiters(
            Arc::new(FailureSignal::default()),
            service.clone(),
            service.wake(),
            CleanupPolicy::ReturnedHandoff,
        );
        let record = native
            .prepare((), owner.clone(), |_| Ok(()))
            .unwrap_or_else(|_| panic!("fake native owner registration"));
        record.begin_handoff().unwrap();
        record.handoff_unknown(|| {}).unwrap();
        record
            .publish_feedback(())
            .unwrap_or_else(|_| panic!("fake feedback"));
        record.run_callbacks(|_, _, _| {}, || {}).unwrap();
        drop(owner);
        drop(record);
        drop(native);
        drain(&service);
        assert_eq!(
            captures.load(Ordering::SeqCst),
            0,
            "unknown join dropped namespace captures"
        );
        assert_eq!(
            anchors.load(Ordering::SeqCst),
            0,
            "unknown join released physical root"
        );
        // Fake service destruction only; production quarantine is never released.
        drop(service);
        assert_eq!(anchors.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn namespace_capture_unwind_keeps_separate_physical_anchor() {
        let service = HostRetirementService::manual();
        let captures = Arc::new(AtomicUsize::new(0));
        let anchors = Arc::new(AtomicUsize::new(0));
        drop(root(&service, &captures, &anchors, true));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| service.manual_pass()))
                .is_err()
        );
        assert_eq!(captures.load(Ordering::SeqCst), 1);
        assert_eq!(
            anchors.load(Ordering::SeqCst),
            0,
            "namespace C unwind released trusted A"
        );
        // Fake-only disposal after caught single panic; Capture already moved out.
        drop(service);
        assert_eq!(anchors.load(Ordering::SeqCst), 1);
    }
    #[test]
    fn namespace_memo_is_generation_local_and_cannot_pin_payload() {
        let service = HostRetirementService::manual();
        let captures = Arc::new(AtomicUsize::new(0));
        let anchors = Arc::new(AtomicUsize::new(0));
        let owner = root(&service, &captures, &anchors, false);
        let mut memo = GenerationMemo::default();
        assert!(!memo.contains(&owner));
        memo.remember(&owner);
        for _ in 0..64 {
            assert!(memo.contains(&owner));
        }
        memo.reset();
        assert!(
            !memo.contains(&owner),
            "new generation skipped namespace enrollment"
        );
        memo.remember(&owner);
        drop(owner);
        drain(&service);
        assert_eq!(
            anchors.load(Ordering::SeqCst),
            1,
            "weak memo kept namespace payload alive"
        );
        drop(memo);
    }
}
