//! Independently registered namespace custody. No Commands/logical Queue backedge.
//! This enrollment does not close raw-derived alias, IO, or admission coverage.
use super::allocation_namespace::AllocationNamespace;
use crate::{
    allocation_header_root::{HeaderPool, HeaderStrong, HeaderWeak},
    allocation_owner::AllocationLease,
    completion_wake::Wake,
    host_anchor::{HostAnchorPermit, HostAnchorRegistry},
    kernels::{
        metal::{Backings, Buffer, Device, InFlightHold, ResidencySet},
        Kernels, MetalKernelError,
    },
    memory_admission::AllocationContext,
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
// This outer cell owns ONLY final permit metadata. The independently registered
// HostAnchor still holds the actual native C/A and existing terminal joins.
// Never expose this Arc through captures(), clone it, or construct a Weak/raw
// acquisition path: all default/view/native holders use the counted cell below.
type MetadataPool = HeaderPool<Arc<Permit>, (), 1>;
type MetadataStrong = HeaderStrong<Arc<Permit>, (), 1>;
type MetadataWeak = HeaderWeak<Arc<Permit>, (), 1>;

/// Only counted host-record metadata; final Drop cannot run the C/A payload.
/// Keep this concrete type in native owner fields, rather than an arbitrary
/// Arc destructor smuggled into trusted A.
pub(crate) struct NamespaceOwner(MetadataStrong);

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
/// Generation identity only: opaque rooted HeaderWeak plus ticket/generation
/// equality prevents reuse confusion. No std Arc allocation/address semantics
/// or namespace-payload lifetime is inferred from this memo.
#[derive(Default)]
pub(crate) struct NamespaceMemo(Option<MetadataWeak>);
impl NamespaceMemo {
    pub(crate) fn reset(&mut self) {
        self.0 = None; // Last weak only publishes atoms to the rooted service.
    }
}
impl NamespaceOwner {
    pub(crate) fn try_clone(&self) -> Result<Self, MetalKernelError> {
        self.0
            .try_clone()
            .map(Self)
            .map_err(|_| MetalKernelError::QueueUnavailable("namespace counted owner exhausted"))
    }
    pub(crate) fn same(&self, other: &Self) -> bool {
        MetadataStrong::same(&self.0, &other.0)
    }
    pub(crate) fn already_recorded(&self, memo: &NamespaceMemo) -> bool {
        memo.0
            .as_ref()
            .is_some_and(|weak| weak.same_strong(&self.0))
    }
    pub(crate) fn remember(&self, memo: &mut NamespaceMemo) -> Result<(), MetalKernelError> {
        // Acquire before replacing the old memo. Refusal leaves it unchanged;
        // successful native attachment remains rooted and a retry deduplicates.
        let weak = self
            .0
            .downgrade()
            .map_err(|_| MetalKernelError::QueueUnavailable("namespace counted memo exhausted"))?;
        memo.0 = Some(weak);
        Ok(())
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
        metadata: &AllocationContext,
    ) -> Result<Arc<Self>, MetalKernelError> {
        // SAFETY: outer C is metadata-only Permit Arc, never exposed/cloned.
        // Its actual native C/A remains independently rooted by HostAnchor.
        // Strong0 authorizes final permit metadata release ONLY, not raw alias
        // terminality, final physical charge or namespace remove_all.
        let pool = unsafe { MetadataPool::cold_in(metadata) }.map_err(|_| {
            MetalKernelError::QueueUnavailable("namespace metadata admission refused")
        })?;
        let empty = pool.reserve_header().map_err(|_| {
            MetalKernelError::QueueUnavailable("namespace metadata capacity refused")
        })?;
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
        let permit = registry
            .reserve_in(metadata, captures, anchors)
            .map(Arc::new)
            .map_err(|rejected| {
                crate::kernels::err::CompletionRegistrationError::new(
                    MetalKernelError::QueueUnavailable("namespace host root registration refused"),
                    rejected.owner,
                )
            })?;
        let prepared = empty.initialize(permit, ()).map_err(|rejected| {
            crate::kernels::err::CompletionRegistrationError::new(
                MetalKernelError::QueueUnavailable("namespace metadata initialization refused"),
                rejected.owners,
            )
        })?;
        let owner = prepared.publish().map_err(|_| {
            MetalKernelError::QueueUnavailable("namespace metadata publication refused")
        })?;
        Ok(Arc::new(Self {
            owner: NamespaceOwner(owner),
        }))
    }
    pub(crate) fn namespace_owner(&self) -> Result<NamespaceOwner, MetalKernelError> {
        self.owner.try_clone()
    }
    pub(crate) fn physical_owner(&self) -> Result<NamespaceOwner, MetalKernelError> {
        self.owner.try_clone()
    }
    /// Outer header plus managed first native-owner slab Layouts only.
    /// Arc/control/registry/service/namespace/generation storage remain open.
    pub(crate) fn requested_metadata_bytes() -> Result<usize, MetalKernelError> {
        let outer = MetadataPool::requested_bytes().map_err(|_| {
            MetalKernelError::QueueUnavailable("namespace metadata layout overflow")
        })?;
        let native = HostAnchorRegistry::<DeviceCaptures, DeviceAnchors>::requested_slab_bytes()
            .map_err(|_| {
                MetalKernelError::QueueUnavailable("namespace native slab layout overflow")
            })?;
        outer
            .checked_add(native)
            .ok_or(MetalKernelError::QueueUnavailable(
                "namespace combined metadata layout overflow",
            ))
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
