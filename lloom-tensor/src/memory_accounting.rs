//! Atomic numeric ownership for pass promises and physical allocations.
//!
//! profiles-and-memory-design.md §4.2 requires a pass reservation, not a
//! check/use snapshot. Policy domains remain independent; process accounting
//! is shared explicitly. This is an internal production primitive awaiting
//! backing/retirement wiring, NOT yet an enforcement path for Metal allocations.
//! Reclaim and machine admission belong outside this lock. Native owners must
//! retain PhysicalCharge until actual retirement, not merely Rust view drop.

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Snapshot {
    pub pending: usize,
    pub committed: usize,
}

impl Snapshot {
    fn total(self) -> Result<usize, Refusal> {
        self.pending
            .checked_add(self.committed)
            .ok_or(Refusal::Overflow)
    }

    fn reserve(self, bytes: usize, cap: Option<usize>, scope: Scope) -> Result<Self, Refusal> {
        let total = self.total()?.checked_add(bytes).ok_or(Refusal::Overflow)?;
        if let Some(limit) = cap {
            if total > limit {
                return Err(Refusal::Limit {
                    scope,
                    limit,
                    requested_total: total,
                });
            }
        }
        Ok(Self {
            pending: self.pending.checked_add(bytes).ok_or(Refusal::Overflow)?,
            ..self
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Scope {
    Process,
    Domain,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    Overflow,
    Limit {
        scope: Scope,
        limit: usize,
        requested_total: usize,
    },
    PromiseTooSmall {
        remaining: usize,
        requested: usize,
    },
}

#[derive(Default)]
struct State {
    total: Snapshot,
    domains: HashMap<u64, Snapshot>,
    next_id: u64,
}

pub(crate) struct ProcessAccount {
    // None is accounting without an aggregate cap, never a machine ceiling.
    cap: Option<usize>,
    state: Mutex<State>,
}

impl ProcessAccount {
    pub fn new(cap: Option<usize>) -> Arc<Self> {
        Arc::new(Self {
            cap,
            state: Mutex::new(State::default()),
        })
    }

    pub fn domain(self: &Arc<Self>, quota: Option<usize>) -> Result<Domain, Refusal> {
        let mut state = self.state.lock();
        let id = state.next_id;
        let next_id = id.checked_add(1).ok_or(Refusal::Overflow)?;
        state.domains.insert(id, Snapshot::default());
        state.next_id = next_id;
        Ok(Domain(Arc::new(DomainInner {
            account: self.clone(),
            id,
            quota,
        })))
    }

    pub fn snapshot(&self) -> Snapshot {
        self.state.lock().total
    }
}

#[derive(Clone)]
pub(crate) struct Domain(Arc<DomainInner>);

struct DomainInner {
    account: Arc<ProcessAccount>,
    id: u64,
    quota: Option<usize>,
}

impl Drop for DomainInner {
    fn drop(&mut self) {
        let removed = self.account.state.lock().domains.remove(&self.id);
        // Promises and physical charges keep this owner alive until empty.
        assert_eq!(removed, Some(Snapshot::default()));
    }
}

impl Domain {
    pub fn snapshot(&self) -> Snapshot {
        self.0.account.state.lock().domains[&self.0.id]
    }

    pub fn try_reserve(&self, bytes: usize) -> Result<Promise, Refusal> {
        let mut state = self.0.account.state.lock();
        let own = state.domains[&self.0.id].reserve(bytes, self.0.quota, Scope::Domain)?;
        let total = state
            .total
            .reserve(bytes, self.0.account.cap, Scope::Process)?;
        // No update until both scope checks succeed.
        state.domains.insert(self.0.id, own);
        state.total = total;
        Ok(Promise {
            domain: self.clone(),
            remaining: bytes,
        })
    }
}

#[must_use = "unused promises cancel on drop"]
pub(crate) struct Promise {
    domain: Domain,
    remaining: usize,
}

impl Promise {
    /// Transfer already promised bytes, without a second quota charge.
    /// Caller must reserve a verified upper bound BEFORE native allocation.
    pub fn commit(&mut self, bytes: usize) -> Result<PhysicalCharge, Refusal> {
        if bytes > self.remaining {
            return Err(Refusal::PromiseTooSmall {
                remaining: self.remaining,
                requested: bytes,
            });
        }
        let mut state = self.domain.0.account.state.lock();
        let own = state.domains[&self.domain.0.id];
        let committed = own.committed.checked_add(bytes).ok_or(Refusal::Overflow)?;
        let total_committed = state
            .total
            .committed
            .checked_add(bytes)
            .ok_or(Refusal::Overflow)?;
        state.domains.insert(
            self.domain.0.id,
            Snapshot {
                pending: own.pending - bytes,
                committed,
            },
        );
        state.total.pending -= bytes;
        state.total.committed = total_committed;
        self.remaining -= bytes;
        Ok(PhysicalCharge(Arc::new(ChargeInner {
            domain: self.domain.clone(),
            bytes,
        })))
    }
}

impl Drop for Promise {
    fn drop(&mut self) {
        let mut state = self.domain.0.account.state.lock();
        state.domains.get_mut(&self.domain.0.id).unwrap().pending -= self.remaining;
        state.total.pending -= self.remaining;
    }
}

/// Clones are owners of ONE physical allocation. Heap views must share the
/// chunk's charge; returning a range is not physical release. Native command
/// and residency owners must explicitly retain this token until retirement.
#[derive(Clone)]
pub(crate) struct PhysicalCharge(Arc<ChargeInner>);

impl PhysicalCharge {
    pub fn bytes(&self) -> usize {
        self.0.bytes
    }
}

struct ChargeInner {
    domain: Domain,
    bytes: usize,
}

impl Drop for ChargeInner {
    fn drop(&mut self) {
        let mut state = self.domain.0.account.state.lock();
        state.domains.get_mut(&self.domain.0.id).unwrap().committed -= self.bytes;
        state.total.committed -= self.bytes;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    #[test]
    fn independent_quotas_and_shared_cap() {
        let account = ProcessAccount::new(Some(12));
        let a = account.domain(Some(4)).unwrap();
        let b = account.domain(Some(10)).unwrap();
        let pa = a.try_reserve(4).unwrap();
        assert!(matches!(
            a.try_reserve(1),
            Err(Refusal::Limit {
                scope: Scope::Domain,
                ..
            })
        ));
        let pb = b.try_reserve(8).unwrap();
        let before = b.snapshot();
        assert!(matches!(
            b.try_reserve(1),
            Err(Refusal::Limit {
                scope: Scope::Process,
                ..
            })
        ));
        assert_eq!(b.snapshot(), before);
        assert_eq!(account.snapshot().pending, 12);
        drop(pa);
        drop(pb);
        let full_b = b.try_reserve(10).unwrap();
        assert_eq!(a.snapshot(), Snapshot::default());
        drop(full_b);
        assert_eq!(account.snapshot(), Snapshot::default());
    }

    #[test]
    fn pending_and_committed_compete_without_double_charge() {
        let account = ProcessAccount::new(Some(10));
        let domain = account.domain(Some(10)).unwrap();
        let mut promise = domain.try_reserve(10).unwrap();
        let charge = promise.commit(6).unwrap();
        assert_eq!(charge.bytes(), 6);
        assert_eq!(
            domain.snapshot(),
            Snapshot {
                pending: 4,
                committed: 6
            }
        );
        assert!(domain.try_reserve(1).is_err());
        assert!(matches!(
            promise.commit(5),
            Err(Refusal::PromiseTooSmall {
                remaining: 4,
                requested: 5
            })
        ));
        assert_eq!(
            account.snapshot(),
            Snapshot {
                pending: 4,
                committed: 6
            }
        );
        drop(promise); // failed allocation/cancel releases unused bytes only
        let replacement = domain.try_reserve(4).unwrap();
        assert_eq!(
            account.snapshot(),
            Snapshot {
                pending: 4,
                committed: 6
            }
        );
        drop(replacement);
        drop(charge);
        assert_eq!(account.snapshot(), Snapshot::default());
    }

    #[test]
    fn final_native_owner_token_survives_model_handle() {
        let account = ProcessAccount::new(None);
        let domain = account.domain(None).unwrap();
        let mut promise = domain.try_reserve(8).unwrap();
        let charge = promise.commit(5).unwrap();
        let retained = charge.clone();
        drop(domain);
        drop(promise);
        drop(charge);
        assert_eq!(
            account.snapshot(),
            Snapshot {
                pending: 0,
                committed: 5
            }
        );
        assert_eq!(account.state.lock().domains.len(), 1);
        drop(retained);
        assert_eq!(account.snapshot(), Snapshot::default());
        assert!(account.state.lock().domains.is_empty());
    }

    #[test]
    fn arithmetic_boundaries_are_transactional() {
        let account = ProcessAccount::new(None);
        let a = account.domain(None).unwrap();
        let b = account.domain(None).unwrap();
        let mut promise = a.try_reserve(usize::MAX).unwrap();
        let charge = promise.commit(1).unwrap();
        let before = account.snapshot();
        assert!(matches!(a.try_reserve(1), Err(Refusal::Overflow)));
        assert!(matches!(b.try_reserve(1), Err(Refusal::Overflow)));
        assert_eq!(account.snapshot(), before);
        assert_eq!(b.snapshot(), Snapshot::default());
        drop(promise);
        drop(charge);
        assert_eq!(account.snapshot(), Snapshot::default());
        account.state.lock().next_id = u64::MAX;
        assert!(matches!(account.domain(None), Err(Refusal::Overflow)));
        assert_eq!(account.state.lock().domains.len(), 2);
    }

    #[test]
    fn racing_domains_cannot_promise_same_capacity() {
        let account = ProcessAccount::new(Some(10));
        let ready = Arc::new(Barrier::new(3));
        let held = Arc::new(Barrier::new(3));
        let mut workers = Vec::new();
        for _ in 0..2 {
            let domain = account.domain(Some(10)).unwrap();
            let ready = ready.clone();
            let held = held.clone();
            workers.push(std::thread::spawn(move || {
                ready.wait();
                let result = domain.try_reserve(8);
                held.wait(); // retain the winning promise until both attempts finish
                result
            }));
        }
        ready.wait();
        held.wait();
        let results: Vec<_> = workers.into_iter().map(|w| w.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        assert_eq!(
            account.snapshot(),
            Snapshot {
                pending: 8,
                committed: 0
            }
        );
        drop(results);
        assert_eq!(account.snapshot(), Snapshot::default());
    }

    #[test]
    fn zero_promise_and_zero_charge_preserve_ownership_without_bytes() {
        let account = ProcessAccount::new(Some(0));
        let domain = account.domain(Some(0)).unwrap();
        let mut promise = domain.try_reserve(0).unwrap();
        let charge = promise.commit(0).unwrap();
        assert!(domain.try_reserve(1).is_err());
        drop(domain);
        drop(promise);
        assert_eq!(account.snapshot(), Snapshot::default());
        drop(charge);
        assert!(account.state.lock().domains.is_empty());
    }
}
