//! Accounting custody beside a native residency membership record. Native calls
//! remain the caller's job; `published_removals` requires a completed native
//! residency commit (NOT command-buffer completion). Return values are destroyed
//! after the caller releases its membership mutex.

use std::collections::{hash_map::Entry, HashMap};

struct Record<L> {
    primary: Option<L>,
    aliases: Vec<L>,
}
impl<L> Record<L> {
    fn append_to(self, out: &mut Vec<L>) {
        out.extend(self.primary);
        out.extend(self.aliases);
    }
}

pub(crate) struct ResidencyOwners<L> {
    active: HashMap<usize, Record<L>>,
    pending_remove: Vec<L>,
    retired: HashMap<usize, Record<L>>,
}
impl<L> Default for ResidencyOwners<L> {
    fn default() -> Self {
        Self {
            active: HashMap::new(),
            pending_remove: Vec::new(),
            retired: HashMap::new(),
        }
    }
}
impl<L> ResidencyOwners<L> {
    /// Legacy insertion has no accounting lease. A later owned alias upgrades
    /// custody even though membership already exists. Only an identical charge
    /// may be deduplicated; same raw address/domain/size alone does not suffice.
    /// Redundant leases are returned, never dropped under the caller's mutex.
    pub(crate) fn insert(
        &mut self,
        key: usize,
        lease: Option<L>,
        same: impl Fn(&L, &L) -> bool,
    ) -> (bool, Option<L>) {
        // A retired allocation is STILL in the native set. Restore its keyed
        // custody before merging the new alias; do not issue a second native
        // add, and do not leave stale retired custody after a later remove.
        if let Some(retired) = self.retired.remove(&key) {
            debug_assert!(!self.active.contains_key(&key));
            self.active.insert(key, retired);
        }
        match self.active.entry(key) {
            Entry::Vacant(entry) => {
                entry.insert(Record {
                    primary: lease,
                    aliases: Vec::new(),
                });
                (true, None)
            }
            Entry::Occupied(mut entry) => {
                let record = entry.get_mut();
                if let Some(lease) = lease {
                    if record
                        .primary
                        .iter()
                        .chain(&record.aliases)
                        .any(|old| same(old, &lease))
                    {
                        return (false, Some(lease));
                    }
                    if record.primary.is_none() {
                        record.primary = Some(lease);
                    } else {
                        record.aliases.push(lease);
                    }
                }
                (false, None)
            }
        }
    }

    pub(crate) fn remove(&mut self, key: usize) -> bool {
        if let Some(record) = self.active.remove(&key) {
            record.append_to(&mut self.pending_remove);
            true
        } else {
            false
        }
    }

    pub(crate) fn retire(&mut self, key: usize) -> bool {
        if let Some(record) = self.active.remove(&key) {
            let previous = self.retired.insert(key, record);
            debug_assert!(previous.is_none(), "active and retired keys are disjoint");
            true
        } else {
            false
        }
    }

    /// Allocation-local finalization, unlike ordinary eviction, also removes
    /// intentionally retained membership. The caller holds the raw allocation
    /// independently and issues ONE native remove when this returns true.
    pub(crate) fn remove_including_retired(&mut self, key: usize) -> bool {
        if self.remove(key) {
            true
        } else if let Some(record) = self.retired.remove(&key) {
            record.append_to(&mut self.pending_remove);
            true
        } else {
            false
        }
    }

    pub(crate) fn published_removals(&mut self) -> Vec<L> {
        std::mem::take(&mut self.pending_remove)
    }

    /// After removeAllAllocations + commit, including retired records even when
    /// active membership is empty. The caller drops this vector outside locks.
    pub(crate) fn cleared(&mut self) -> Vec<L> {
        let mut released = self.published_removals();
        for (_, record) in self.retired.drain() {
            record.append_to(&mut released);
        }
        for (_, record) in self.active.drain() {
            record.append_to(&mut released);
        }
        released
    }

    pub(crate) fn len(&self) -> usize {
        self.active.len()
    }
    pub(crate) fn contains(&self, key: usize) -> bool {
        self.active.contains_key(&key)
    }
    /// Whether the native set still names this allocation: an active member,
    /// or one retired from the record but not removed from Metal.
    pub(crate) fn lists(&self, key: usize) -> bool {
        self.active.contains_key(&key) || self.retired.contains_key(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::allocation_owner::{AllocationLease, AllocationOwner};
    use crate::memory_admission::{process_account, ModelDomain, PassReservation};
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };

    type Lease = AllocationLease<Arc<RawOracle>>;
    struct RawOracle {
        domain: ModelDomain,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for RawOracle {
        fn drop(&mut self) {
            assert!(
                self.domain.snapshot().committed >= 8,
                "raw released while still charged"
            );
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }
    fn fixture() -> (ModelDomain, PassReservation, Arc<AtomicUsize>) {
        let domain = process_account().unadmitted_domain(Some(32)).unwrap();
        let pass = domain.reserve(32).unwrap();
        (domain, pass, Arc::new(AtomicUsize::new(0)))
    }
    fn lease(domain: &ModelDomain, pass: &PassReservation, drops: &Arc<AtomicUsize>) -> Lease {
        let charge = pass.context().take(8).unwrap().commit(8).unwrap();
        AllocationLease::new(
            Arc::new(RawOracle {
                domain: domain.clone(),
                drops: drops.clone(),
            }),
            Some(AllocationOwner::new(charge)),
        )
    }
    fn same(a: &Lease, b: &Lease) -> bool {
        a.owner().unwrap().same_custody(b.owner().unwrap())
    }

    #[test]
    fn numeric_legacy_upgrade_duplicate_and_distinct_charge_aliases() {
        let (domain, pass, drops) = fixture();
        let mut state = ResidencyOwners::default();
        assert!(state.insert(7, None, same).0);
        assert!(!state.insert(7, Some(lease(&domain, &pass, &drops)), same).0);
        // Equal domain and byte count do NOT mean equal physical charge.
        let distinct = lease(&domain, &pass, &drops);
        let duplicate = distinct.clone();
        assert!(!state.insert(7, Some(distinct), same).0);
        let (added, redundant) = state.insert(7, Some(duplicate), same);
        assert!(!added);
        assert!(redundant.is_some());
        assert_eq!(domain.snapshot().committed, 16);
        drop(redundant);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(state.len(), 1);
        assert!(state.contains(7));
        assert!(state.remove(7));
        assert!(!state.remove(7));
        assert_eq!(domain.snapshot().committed, 16);
        let released_after_native_commit = state.published_removals();
        assert_eq!(released_after_native_commit.len(), 2);
        assert_eq!(domain.snapshot().committed, 16);
        drop(released_after_native_commit);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert_eq!(domain.snapshot().committed, 0);
    }

    #[test]
    fn numeric_same_charge_distinct_anchors_are_not_discarded() {
        let (domain, pass, drops) = fixture();
        let charge = pass.context().take(8).unwrap().commit(8).unwrap();
        let base = AllocationOwner::new(charge);
        let raw = || {
            Arc::new(RawOracle {
                domain: domain.clone(),
                drops: drops.clone(),
            })
        };
        let a = AllocationLease::new(raw(), Some(base.clone().with_anchor(raw())));
        let b = AllocationLease::new(raw(), Some(base.with_anchor(raw())));
        assert!(!same(&a, &b));
        let mut state = ResidencyOwners::default();
        assert!(state.insert(7, Some(a), same).0);
        let (added, redundant) = state.insert(7, Some(b), same);
        assert!(!added);
        assert!(
            redundant.is_none(),
            "distinct native anchor must remain in custody"
        );
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        assert_eq!(domain.snapshot().committed, 8);
        drop(state.cleared());
        assert_eq!(drops.load(Ordering::SeqCst), 4);
        assert_eq!(domain.snapshot().committed, 0);
    }

    #[test]
    fn numeric_remove_readd_preserves_active_lease_across_publication() {
        let (domain, pass, drops) = fixture();
        let mut state = ResidencyOwners::default();
        let original = lease(&domain, &pass, &drops);
        let readd = original.clone();
        assert!(state.insert(7, Some(original), same).0);
        assert!(state.remove(7));
        assert!(state.insert(7, Some(readd), same).0);
        drop(state.published_removals());
        assert!(state.contains(7));
        assert_eq!(domain.snapshot().committed, 8);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(state.cleared());
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        assert_eq!(domain.snapshot().committed, 0);
    }

    #[test]
    fn numeric_retired_custody_survives_unrelated_commit_until_clear() {
        let (domain, pass, drops) = fixture();
        let mut state = ResidencyOwners::default();
        state.insert(7, Some(lease(&domain, &pass, &drops)), same);
        state.insert(9, Some(lease(&domain, &pass, &drops)), same);
        assert!(state.retire(7));
        assert!(!state.retire(7));
        assert!(!state.remove(7));
        assert!(state.remove(9));
        assert_eq!(state.len(), 0);
        drop(state.published_removals());
        assert_eq!(domain.snapshot().committed, 8);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
        let retired_after_native_clear = state.cleared();
        assert_eq!(domain.snapshot().committed, 8);
        drop(retired_after_native_clear);
        assert_eq!(domain.snapshot().committed, 0);
        assert_eq!(drops.load(Ordering::SeqCst), 2);
        assert!(state.cleared().is_empty());
    }

    #[test]
    fn numeric_retired_readd_and_targeted_finalization_keep_one_native_key() {
        let (domain, pass, drops) = fixture();
        let mut state = ResidencyOwners::default();
        let allocation = lease(&domain, &pass, &drops);
        assert!(state.insert(7, Some(allocation.clone()), same).0);
        assert!(state.retire(7));
        // Native membership survived retirement: reinsertion must not ask the
        // native set to add the same key a second time.
        let (added, redundant) = state.insert(7, Some(allocation), same);
        assert!(!added);
        drop(redundant);
        assert!(state.remove(7));
        assert!(!state.remove_including_retired(7));
        assert_eq!(domain.snapshot().committed, 8);
        drop(state.published_removals());
        assert_eq!(domain.snapshot().committed, 0);
        assert_eq!(drops.load(Ordering::SeqCst), 1);

        state.insert(9, Some(lease(&domain, &pass, &drops)), same);
        state.insert(11, Some(lease(&domain, &pass, &drops)), same);
        assert!(state.retire(9));
        assert!(state.retire(11));
        assert!(state.remove_including_retired(9));
        assert!(!state.remove_including_retired(9));
        drop(state.published_removals());
        assert_eq!(
            domain.snapshot().committed,
            8,
            "another namespace stays resident"
        );
        assert!(state.remove_including_retired(11));
        drop(state.published_removals());
        assert_eq!(domain.snapshot().committed, 0);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn numeric_clear_extracts_active_pending_and_retired_without_dropping() {
        let (domain, pass, drops) = fixture();
        let mut state = ResidencyOwners::default();
        for key in 0..3 {
            state.insert(key, Some(lease(&domain, &pass, &drops)), same);
        }
        assert!(state.remove(0));
        assert!(state.retire(1));
        let released = state.cleared();
        assert_eq!(released.len(), 3);
        assert_eq!(state.len(), 0);
        assert_eq!(domain.snapshot().committed, 24);
        assert_eq!(drops.load(Ordering::SeqCst), 0);
        drop(released);
        assert_eq!(domain.snapshot().committed, 0);
        assert_eq!(drops.load(Ordering::SeqCst), 3);
    }
}
