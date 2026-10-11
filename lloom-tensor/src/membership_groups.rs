//! Immutable per-origin retention groups for standalone backing compatibility.
//!
//! Each published node has an independent service root. A link releases only a
//! counted handle, never the final Arc of its successor: chains retire through
//! repeated bounded service passes, not recursive final-link destruction.
//! Accounted nodes reserve their inline managed layout before birth. Arc headers,
//! allocator rounding, producer/service vectors and token payload allocations
//! are not counted here; this is not an allocator/RSS ceiling.
use crate::{
    completion_wake::Wake,
    host_anchor::{AnchorError, HostAnchorPermit, HostAnchorRegistry},
    memory_admission::{
        AdmissionError, AllocationOrigin, AllocationTicket, DomainKey, ModelDomain, PhysicalCharge,
    },
    retirement_owners::{Failure, FailureSignal},
    retirement_registry::{HostMetadataHandle, MetadataWake, RegistryError},
    retirement_service::{CleanupInbox, HostRetirementService, ServiceError},
};
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock};

/// A standalone backing chooses from its committed allocation owner. A fixed
/// namespace never changes payer and never consults ambient ExecutionScope.
pub(crate) enum MembershipSource {
    Standalone,
    Namespace(Option<ModelDomain>),
}
impl MembershipSource {
    pub(crate) fn resolve(
        &self,
        owner: Option<DomainKey>,
    ) -> Result<AllocationOrigin, AdmissionError> {
        let actual = owner
            .map(AllocationOrigin::Domain)
            .unwrap_or(AllocationOrigin::Legacy);
        if let Self::Namespace(domain) = self {
            let expected = domain
                .as_ref()
                .map(|d| AllocationOrigin::Domain(d.key()))
                .unwrap_or(AllocationOrigin::Legacy);
            expected.validate_reuse(&actual)?;
        }
        Ok(actual)
    }
}
#[derive(Debug)]
pub(crate) enum SelectError<E> {
    Create(E),
    Registry(RegistryError),
    Anchor(AnchorError),
    ChargeEdit,
}
type ChargeOwners = HostAnchorRegistry<(), Option<PhysicalCharge>>;
type ChargePermit = HostAnchorPermit<(), Option<PhysicalCharge>>;

/// Trusted engine metadata, not an arbitrary Clone/Drop callback surface.
///
/// # Safety
/// Clone is nonallocating, nonpanicking counted metadata retention; it runs
/// under the head lock. Drop is nonpanicking metadata release/publication only:
/// resource owners remain in independently rooted host records. No recursive
/// retirement-service invocation, native payload destructor or user closure.
/// Implementation requires a concrete caller/field census.
pub(crate) unsafe trait GroupMetadata: Clone + Send + Sync + 'static {}

pub(crate) struct Groups<T: GroupMetadata> {
    head: Mutex<Option<HostMetadataHandle<GroupNode<T>>>>,
    owners: Mutex<Option<Arc<ChargeOwners>>>,
    service: OnceLock<Result<Arc<HostRetirementService>, ServiceError>>,
    failure: Arc<FailureSignal>,
}
struct GroupNode<T: GroupMetadata> {
    origin: AllocationOrigin,
    value: T,
    next: OnceLock<HostMetadataHandle<GroupNode<T>>>,
    wake: Arc<Wake>,
    failure: Arc<FailureSignal>,
    // Last: final node destruction publishes this independent charge record.
    // On the single service worker, the record cannot be processed until Arc
    // deallocation returns. Unpublished nodes are try_unwrapped first instead.
    charge_permit: Option<ChargePermit>,
}
impl<T: GroupMetadata> MetadataWake for GroupNode<T> {
    fn metadata_wake(&self) -> &Arc<Wake> {
        &self.wake
    }
}
// SAFETY: trusted metadata-only T (production Membership, scalar CPU fixtures).
// No native handoff/payload or callback borrow exists in these immutable nodes.
// All nodes are independently registered before linking; no Weak/raw acquisition
// exists, and external head/link handles release their Arc before waking. T's
// own host-record permit retains native/charge ownership independently.
unsafe impl<T: GroupMetadata> CleanupInbox for GroupNode<T> {
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
    fn fail(&self, reason: Failure) {
        self.failure.publish(reason, || self.wake.notify());
    }
}
impl<T: GroupMetadata> Groups<T> {
    pub(crate) fn new() -> Self {
        Self {
            head: Mutex::new(None),
            owners: Mutex::new(None),
            service: OnceLock::new(),
            failure: Arc::new(FailureSignal::default()),
        }
    }
    #[cfg(test)]
    fn manual(service: Arc<HostRetirementService>) -> Self {
        let groups = Self::new();
        let owners = ChargeOwners::manual(groups.failure.clone(), service.clone(), service.wake());
        *groups.owners.lock() = Some(Arc::new(owners));
        let _ = groups.service.set(Ok(service));
        groups
    }
    fn service<E>(&self) -> Result<Arc<HostRetirementService>, SelectError<E>> {
        self.service
            .get_or_init(|| HostRetirementService::shared().cloned())
            .clone()
            .map_err(|e| SelectError::Registry(RegistryError::Service(e)))
    }
    fn owners<E>(&self) -> Result<Arc<ChargeOwners>, SelectError<E>> {
        if let Some(owners) = self.owners.lock().as_ref() {
            return Ok(owners.clone());
        }
        let candidate = Arc::new(
            ChargeOwners::new(self.failure.clone(), self.service::<E>()?.wake())
                .map_err(SelectError::Registry)?,
        );
        let selected = {
            let mut slot = self.owners.lock();
            slot.get_or_insert_with(|| candidate.clone()).clone()
        };
        drop(candidate); // Losing empty registry finalization is unlocked.
        Ok(selected)
    }
    fn find(
        head: &Option<HostMetadataHandle<GroupNode<T>>>,
        origin: &AllocationOrigin,
    ) -> Option<T> {
        let mut cursor = head.as_ref();
        while let Some(node) = cursor {
            if &node.origin == origin {
                return Some(node.value.clone());
            }
            cursor = node.next.get();
        }
        None
    }
    fn failed<E>(&self) -> Result<(), SelectError<E>> {
        match self.failure.get() {
            Some(reason) => Err(SelectError::Registry(RegistryError::Service(
                ServiceError::Failed(reason),
            ))),
            None => Ok(()),
        }
    }
    /// The caller supplies metadata-only values. Admission, construction,
    /// registration, losing-candidate destruction and payer-key finalization
    /// happen outside head/owners producer locks.
    pub(crate) fn select<E>(
        &self,
        origin: AllocationOrigin,
        create: impl FnOnce() -> Result<T, E>,
    ) -> Result<T, SelectError<E>> {
        {
            let head = self.head.lock();
            self.failed()?;
            if let Some(value) = Self::find(&head, &origin) {
                return Ok(value);
            }
        }
        let candidate = self.candidate(origin.clone(), create);
        let selected = {
            let mut head = self.head.lock();
            self.failed().and_then(|()| {
                if let Some(value) = Self::find(&head, &origin) {
                    return Ok(Some(value));
                }
                let Ok(node) = candidate.as_ref() else {
                    return Ok(None);
                };
                // Fresh OnceLock, one publisher. Move the old root into a
                // counted link; no final Arc/key/permit Drop under this lock.
                if let Some(old) = head.take() {
                    match node.next.set(old) {
                        Ok(()) => {}
                        Err(old) => {
                            *head = Some(old);
                            return Err(SelectError::ChargeEdit);
                        }
                    }
                }
                let value = node.value.clone();
                *head = Some(node.clone());
                Ok(Some(value))
            })
        };
        match (selected, candidate) {
            (Ok(Some(value)), candidate) => {
                drop(candidate);
                Ok(value)
            }
            (Err(error), candidate) => {
                drop(candidate);
                Err(error)
            }
            (Ok(None), Err(cold)) => Err(cold),
            (Ok(None), Ok(candidate)) => {
                drop(candidate);
                Err(SelectError::ChargeEdit)
            }
        }
    }
    fn candidate<E>(
        &self,
        origin: AllocationOrigin,
        create: impl FnOnce() -> Result<T, E>,
    ) -> Result<HostMetadataHandle<GroupNode<T>>, SelectError<E>> {
        let service = self.service()?;
        self.failed()?;
        let bytes = std::mem::size_of::<GroupNode<T>>();
        let (mut ticket, permit) = match &origin {
            AllocationOrigin::Legacy => (None, None),
            AllocationOrigin::Domain(key) => {
                let domain = key.model_domain();
                let ticket = Self::ticket(&domain, bytes).map_err(SelectError::Registry)?;
                origin
                    .validate_reuse(&AllocationOrigin::Domain(ticket.domain_key()))
                    .map_err(|e| SelectError::Registry(RegistryError::Admission(e)))?;
                let owners = self.owners()?;
                let permit = owners
                    .reserve_admitted(key.clone(), (), None, |n| Self::ticket(&domain, n))
                    .map_err(|rejected| {
                        drop(rejected.owner);
                        SelectError::Anchor(rejected.error)
                    })?;
                (Some(ticket), Some(permit))
            }
        };
        let value = create().map_err(SelectError::Create)?;
        let node = Arc::new(GroupNode {
            origin,
            value,
            next: OnceLock::new(),
            wake: service.wake(),
            failure: self.failure.clone(),
            charge_permit: permit,
        });
        if let Some(ticket) = ticket.as_mut() {
            let charge = match ticket.try_commit(bytes) {
                Ok(charge) => charge,
                Err(error) => {
                    Self::drop_unpublished(node);
                    return Err(SelectError::Registry(RegistryError::Admission(error)));
                }
            };
            match node
                .charge_permit
                .as_ref()
                .expect("accounted node permit")
                .edit(charge, |_, owner, input| {
                    *owner = input.take();
                }) {
                Ok(((), None)) => {}
                Ok(((), Some(charge))) => {
                    Self::drop_unpublished(node);
                    drop(charge);
                    return Err(SelectError::ChargeEdit);
                }
                Err(error) => {
                    Self::drop_unpublished(node);
                    drop(error);
                    return Err(SelectError::ChargeEdit);
                }
            }
        }
        let inbox: Arc<dyn CleanupInbox> = node.clone();
        let registered = service.register(&inbox);
        drop(inbox); // Temporary registration Arc gone before handle release/wake.
        if let Err(error) = registered {
            Self::drop_unpublished(node);
            return Err(SelectError::Registry(RegistryError::Service(error)));
        }
        Ok(HostMetadataHandle::new(node))
    }
    fn ticket(domain: &ModelDomain, bytes: usize) -> Result<AllocationTicket, RegistryError> {
        let pass = domain.reserve(bytes).map_err(RegistryError::Admission)?;
        pass.context().take(bytes).map_err(RegistryError::Admission)
    }
    fn drop_unpublished(node: Arc<GroupNode<T>>) {
        match Arc::try_unwrap(node) {
            Ok(node) => drop(node), // Arc allocation gone before permit publishes.
            Err(node) => {
                node.failure
                    .publish(Failure::AdapterFailed, || node.wake.notify());
                std::mem::forget(node); // Unexpected custody: retain, never invent quiescence.
            }
        }
    }
}
impl<T: GroupMetadata> Drop for Groups<T> {
    fn drop(&mut self) {
        let head = self.head.get_mut().take();
        drop(head); // Roots and links remain independently service-owned.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    // SAFETY: usize copy/drop has no allocation, callback or owned payload.
    unsafe impl GroupMetadata for usize {}
    const CAP: usize = 1 << 20;
    fn drain(service: &HostRetirementService) {
        // Metadata-only root removal may return no payload progress while a
        // released link wakes the next pass. Deliberately do not stop on false.
        for _ in 0..128 {
            service.manual_pass();
        }
    }
    fn origin(domain: &ModelDomain) -> AllocationOrigin {
        AllocationOrigin::Domain(domain.key())
    }
    #[test]
    fn metadata_groups_standalone_keeps_legacy_and_two_committed_payers_separate() {
        let service = HostRetirementService::manual();
        let groups = Groups::<usize>::manual(service.clone());
        let domains = ModelDomain::isolated_numeric_domains(CAP, 2);
        let source = MembershipSource::Standalone;
        for (payer, value) in [
            (None, 1),
            (Some(domains[0].key()), 2),
            (Some(domains[1].key()), 3),
        ] {
            assert_eq!(
                groups
                    .select(source.resolve(payer).unwrap(), || Ok::<_, ()>(value))
                    .unwrap(),
                value
            );
        }
        for (payer, value) in [
            (None, 1),
            (Some(domains[0].key()), 2),
            (Some(domains[1].key()), 3),
        ] {
            assert_eq!(
                groups
                    .select(source.resolve(payer).unwrap(), || -> Result<usize, ()> {
                        panic!("warm group rebuilt")
                    })
                    .unwrap(),
                value
            );
        }
        assert!(domains.iter().all(|d| d.snapshot().committed > 0));
        drop(groups);
        drain(&service);
        assert!(domains
            .iter()
            .all(|d| d.snapshot().committed == 0 && d.snapshot().pending == 0));
    }
    #[test]
    fn metadata_groups_namespace_rejects_foreign_and_missing_owner_without_construction() {
        let domains = ModelDomain::isolated_numeric_domains(CAP, 2);
        let fixed = MembershipSource::Namespace(Some(domains[0].clone()));
        assert_eq!(
            fixed.resolve(Some(domains[0].key())).unwrap(),
            origin(&domains[0])
        );
        assert!(fixed.resolve(Some(domains[1].key())).is_err());
        assert!(fixed.resolve(None).is_err());
        let legacy = MembershipSource::Namespace(None);
        assert_eq!(legacy.resolve(None).unwrap(), AllocationOrigin::Legacy);
        assert!(legacy.resolve(Some(domains[0].key())).is_err());
        assert!(domains
            .iter()
            .all(|d| d.snapshot().committed == 0 && d.snapshot().pending == 0));
    }
    #[test]
    fn metadata_groups_same_payer_warm_reuse_does_not_add_charge() {
        let service = HostRetirementService::manual();
        let groups = Groups::<usize>::manual(service.clone());
        let domain = ModelDomain::isolated_numeric_fixture(CAP);
        groups.select(origin(&domain), || Ok::<_, ()>(7)).unwrap();
        let charged = domain.snapshot().committed;
        assert!(charged >= std::mem::size_of::<GroupNode<usize>>());
        assert_eq!(
            groups
                .select(origin(&domain), || -> Result<usize, ()> {
                    panic!("warm node birth")
                })
                .unwrap(),
            7
        );
        assert_eq!(domain.snapshot().committed, charged);
        assert_eq!(domain.snapshot().pending, 0);
        drop(groups);
        drain(&service);
        assert_eq!(domain.snapshot().committed, 0);
    }
    #[test]
    fn metadata_groups_refused_birth_preserves_legacy_group_and_skips_builder() {
        let service = HostRetirementService::manual();
        let groups = Groups::<usize>::manual(service.clone());
        let domain = ModelDomain::isolated_numeric_fixture(1);
        groups
            .select(AllocationOrigin::Legacy, || Ok::<_, ()>(4))
            .unwrap();
        let builds = AtomicUsize::new(0);
        assert!(groups
            .select(origin(&domain), || {
                builds.fetch_add(1, Ordering::SeqCst);
                Ok::<_, ()>(5)
            })
            .is_err());
        assert_eq!(builds.load(Ordering::SeqCst), 0);
        assert_eq!(
            groups
                .select(AllocationOrigin::Legacy, || -> Result<usize, ()> {
                    panic!("legacy root lost")
                })
                .unwrap(),
            4
        );
        assert_eq!(domain.snapshot().pending, 0);
        assert_eq!(domain.snapshot().committed, 0);
        drop(groups);
        drain(&service);
    }
    #[test]
    fn metadata_groups_losing_cold_candidate_reuses_winner_and_retires_unlocked() {
        let service = HostRetirementService::manual();
        let groups = Groups::<usize>::manual(service.clone());
        let domain = ModelDomain::isolated_numeric_fixture(CAP);
        let selected = groups
            .select(origin(&domain), || {
                assert!(groups.head.try_lock().is_some(), "builder under head lock");
                groups.select(origin(&domain), || Ok::<_, ()>(9)).unwrap();
                Ok::<_, ()>(8)
            })
            .unwrap();
        assert_eq!(selected, 9);
        let two_nodes = domain.snapshot().committed;
        drain(&service);
        assert_eq!(
            two_nodes - domain.snapshot().committed,
            std::mem::size_of::<GroupNode<usize>>()
        );
        assert_eq!(domain.snapshot().pending, 0);
        drop(groups);
        drain(&service);
        assert_eq!(domain.snapshot().committed, 0);
    }
    #[test]
    fn metadata_groups_failed_cold_builder_rechecks_published_same_payer_winner() {
        let service = HostRetirementService::manual();
        let groups = Groups::<usize>::manual(service.clone());
        let domain = ModelDomain::isolated_numeric_fixture(CAP);
        let value = groups
            .select(origin(&domain), || {
                assert!(groups.head.try_lock().is_some(), "builder under head lock");
                groups.select(origin(&domain), || Ok::<_, ()>(12)).unwrap();
                Err::<usize, ()>(())
            })
            .unwrap();
        assert_eq!(value, 12);
        assert_eq!(domain.snapshot().pending, 0);
        drain(&service);
        drop(groups);
        drain(&service);
        assert_eq!(domain.snapshot().committed, 0);
    }
    #[test]
    fn metadata_groups_closing_during_cold_birth_rechecks_failure_and_cancels_charge() {
        let service = HostRetirementService::manual();
        let groups = Groups::<usize>::manual(service.clone());
        let domain = ModelDomain::isolated_numeric_fixture(CAP);
        assert!(groups
            .select(origin(&domain), || {
                service.close_admission();
                Ok::<_, ()>(8)
            })
            .is_err());
        assert!(groups.head.lock().is_none(), "failed node published");
        assert_eq!(domain.snapshot().pending, 0);
        drain(&service);
        drop(groups);
        drain(&service);
        assert_eq!(domain.snapshot().committed, 0);
    }
    struct DropOracle {
        groups: Arc<Groups<Arc<DropOracle>>>,
        domain: ModelDomain,
        in_host_pass: Arc<AtomicBool>,
        drops: Arc<AtomicUsize>,
    }
    // SAFETY: Arc clone only increments a count. Under this scalar fixture's
    // valid protocol, final Drop is an unlocked assertion oracle over numeric
    // metadata; it neither owns native payloads nor recursively services work.
    unsafe impl GroupMetadata for Arc<DropOracle> {}
    impl Drop for DropOracle {
        fn drop(&mut self) {
            assert!(
                self.groups.head.try_lock().is_some(),
                "node value destroyed under head lock"
            );
            assert!(
                self.in_host_pass.load(Ordering::SeqCst),
                "published node final Drop was foreign"
            );
            assert!(
                self.domain.snapshot().committed
                    >= std::mem::size_of::<GroupNode<Arc<DropOracle>>>(),
                "charge returned before node payload storage Drop"
            );
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    #[test]
    fn metadata_groups_linked_nodes_retire_iteratively_with_charge_after_value_drop() {
        let service = HostRetirementService::manual();
        let groups = Arc::new(Groups::<Arc<DropOracle>>::manual(service.clone()));
        let domains = ModelDomain::isolated_numeric_domains(CAP, 16);
        let in_host_pass = Arc::new(AtomicBool::new(false));
        let drops = Arc::new(AtomicUsize::new(0));
        for domain in &domains {
            let value = groups
                .select(origin(domain), || {
                    Ok::<_, ()>(Arc::new(DropOracle {
                        groups: groups.clone(),
                        domain: domain.clone(),
                        in_host_pass: in_host_pass.clone(),
                        drops: drops.clone(),
                    }))
                })
                .unwrap();
            drop(value);
        }
        // Test-only explicit head detachment avoids an oracle backedge holding
        // the Groups producer forever. Production Membership has no backedge.
        let head = groups.head.lock().take();
        drop(head);
        in_host_pass.store(true, Ordering::SeqCst);
        drain(&service);
        in_host_pass.store(false, Ordering::SeqCst);
        assert_eq!(drops.load(Ordering::SeqCst), 16);
        assert!(domains.iter().all(|d| d.snapshot().pending == 0));
        drop(groups);
        drain(&service);
        assert!(domains.iter().all(|d| d.snapshot().committed == 0));
    }
}
