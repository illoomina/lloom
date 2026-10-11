//! Host-owned native leases, not native completion or preallocation enforcement.
//! Both retaining and unretained queues must retain a lease through native
//! terminality. Legacy unadmitted raw handles remain supported explicitly.

use crate::memory_admission::{DomainKey, PhysicalCharge};

/// Clones retain ONE charge. Heap suballocations must share the chunk's owner.
#[derive(Clone)]
pub(crate) struct AllocationOwner {
    charge: PhysicalCharge,
}
impl AllocationOwner {
    pub(crate) fn same_allocation(&self, other: &Self) -> bool {
        self.charge.same_allocation(&other.charge)
    }
    pub(crate) fn new(charge: PhysicalCharge) -> Self {
        Self { charge }
    }
    pub(crate) fn domain_key(&self) -> DomainKey {
        self.charge.domain_key()
    }
    pub(crate) fn bytes(&self) -> usize {
        self.charge.bytes()
    }
}

/// Pair a retained raw allocation with its accounting owner. Field order is
/// load-bearing: release this native retain BEFORE releasing its charge retain.
/// Other native retains must have their own lease (residency, submissions, IO).
/// Final Drop belongs to host custody; this type itself supplies no retirement.
#[derive(Clone)]
pub(crate) struct AllocationLease<R> {
    raw: R,
    owner: Option<AllocationOwner>,
}
impl<R> AllocationLease<R> {
    pub(crate) fn new(raw: R, owner: Option<AllocationOwner>) -> Self {
        Self { raw, owner }
    }
    pub(crate) fn raw(&self) -> &R {
        &self.raw
    }
    pub(crate) fn owner(&self) -> Option<&AllocationOwner> {
        self.owner.as_ref()
    }
}
// Buffer's existing Hash/PartialEq identity must not change with metadata.
impl<R: std::hash::Hash> std::hash::Hash for AllocationLease<R> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.raw.hash(state);
    }
}
impl<R: PartialEq> PartialEq for AllocationLease<R> {
    fn eq(&self, other: &Self) -> bool {
        self.raw == other.raw
    }
}
impl<R: std::fmt::Debug> std::fmt::Debug for AllocationLease<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.raw.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory_admission::{process_account, AccountSnapshot, ModelDomain};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    struct RawOracle {
        domain: ModelDomain,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for RawOracle {
        fn drop(&mut self) {
            assert_eq!(
                self.domain.snapshot().committed,
                8,
                "raw release precedes final charge release"
            );
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn numeric_lease_clones_keep_one_charge_and_release_raw_first() {
        let domain = process_account().unadmitted_domain(Some(8)).unwrap();
        let pass = domain.reserve(8).unwrap();
        let charge = pass.context().take(8).unwrap().commit(8).unwrap();
        let drops = Arc::new(AtomicUsize::new(0));
        let raw = Arc::new(RawOracle {
            domain: domain.clone(),
            drops: drops.clone(),
        });
        let lease = AllocationLease::new(raw, Some(AllocationOwner::new(charge)));
        assert_eq!(lease.owner().unwrap().domain_key(), domain.key());
        assert_eq!(lease.owner().unwrap().bytes(), 8);
        let view_lease = lease.clone(); // Buffer::view uses this exact Clone.
        let native_submission = view_lease.clone();
        drop(pass);
        drop(lease);
        drop(view_lease);
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 8
            }
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(native_submission);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(domain.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn numeric_lease_identity_ignores_owner_and_preserves_legacy() {
        use std::hash::{Hash, Hasher};
        let domain = process_account().unadmitted_domain(Some(8)).unwrap();
        let pass = domain.reserve(8).unwrap();
        let charge = pass.context().take(8).unwrap().commit(8).unwrap();
        let owned = AllocationLease::new(17usize, Some(AllocationOwner::new(charge)));
        let legacy = AllocationLease::new(17usize, None);
        let other = AllocationLease::new(18usize, None);
        assert_eq!(owned, legacy);
        assert_ne!(owned, other);
        let hash = |x: &AllocationLease<usize>| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            x.hash(&mut h);
            h.finish()
        };
        assert_eq!(hash(&owned), hash(&legacy));
        assert!(legacy.owner().is_none());
        assert_eq!(*owned.raw(), 17);
    }
}
