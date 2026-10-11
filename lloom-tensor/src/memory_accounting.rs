//! Atomic numeric ownership for pass promises and physical allocations.
//!
//! profiles-and-memory-design.md §4.2 requires a pass reservation, not a
//! check/use snapshot. Policy domains remain independent; process accounting
//! is shared explicitly. This is an internal production primitive awaiting
//! backing/retirement wiring, NOT yet an enforcement path for Metal allocations.
//! Reclaim and machine admission belong outside this lock. Native owners must
//! retain PhysicalCharge until actual retirement, not merely Rust view drop.
//!
//! Lock scope: domain creation may grow the HashMap and allocate an Arc;
//! commit allocates an Arc. The allocator and contended mutex may enter the
//! OS. This is NOT an allocation-free or syscall-free critical section.
//! No explicit native allocation, reclaim callback, or registry operation is
//! performed while numeric state is locked. Machine growth runs after unlock;
//! native allocation and reclaim remain outer-protocol operations.
//! Promise, DomainInner and final PhysicalCharge destruction acquire this
//! shared mutex. They must run on host retirement paths, NOT native completion
//! callbacks. A callback may publish progress; it must not become the last
//! owner of one of these tokens. Native resource/residency release must precede
//! final charge retirement, even if GPU completion has already been observed.

use crate::memory_accounting_machine::{MachineAdmission, MachineOwner};
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Refusal {
    Overflow,
    DifferentDomain,
    Machine(String),
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
        self.make_domain(quota, None)
    }

    /// Internal bridge: consume one machine owner into exactly one domain.
    /// Existing public loader/ledger attachment paths have not migrated here.
    pub fn domain_with_machine(
        self: &Arc<Self>,
        quota: Option<usize>,
        machine: MachineOwner,
    ) -> Result<Domain, Refusal> {
        self.make_domain(quota, Some(machine))
    }

    fn make_domain(
        self: &Arc<Self>,
        quota: Option<usize>,
        machine: Option<MachineOwner>,
    ) -> Result<Domain, Refusal> {
        let mut state = self.state.lock();
        let id = state.next_id;
        let next_id = id.checked_add(1).ok_or(Refusal::Overflow)?;
        state.domains.insert(id, Snapshot::default());
        state.next_id = next_id;
        Ok(Domain(Arc::new(DomainInner {
            account: self.clone(),
            id,
            quota,
            machine,
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
    machine: Option<MachineOwner>,
}

impl Drop for DomainInner {
    fn drop(&mut self) {
        let removed = self.account.state.lock().domains.remove(&self.id);
        // Promises and physical charges keep this owner alive until empty.
        assert_eq!(removed, Some(Snapshot::default()));
    }
}

impl Domain {
    /// Identity is stable while a Domain clone is retained; never dereferenced.
    pub fn identity(&self) -> usize {
        Arc::as_ptr(&self.0) as usize
    }

    pub fn machine_admission(&self) -> Option<MachineAdmission> {
        self.0.machine.as_ref().map(MachineOwner::status)
    }

    pub fn snapshot(&self) -> Snapshot {
        self.0.account.state.lock().domains[&self.0.id]
    }

    pub fn try_reserve(&self, bytes: usize) -> Result<Promise, Refusal> {
        self.reserve_with(bytes, |_| {})
    }

    // The hook permits deterministic scheduling AFTER publishing the intent
    // and unlocking, before machine growth. Production's hook is empty.
    fn reserve_with(
        &self,
        bytes: usize,
        before_machine: impl FnOnce(u64),
    ) -> Result<Promise, Refusal> {
        let mut state = self.0.account.state.lock();
        let own = state.domains[&self.0.id].reserve(bytes, self.0.quota, Scope::Domain)?;
        let total = state
            .total
            .reserve(bytes, self.0.account.cap, Scope::Process)?;
        let target = u64::try_from(own.total()?).map_err(|_| Refusal::Overflow)?;
        // No update until both scope checks and target conversion succeed.
        state.domains.insert(self.0.id, own);
        state.total = total;
        let mut promise = Promise {
            domain: self.clone(),
            remaining: bytes,
            machine: None,
        };
        drop(state);
        before_machine(target);
        // Never hold the numeric mutex across registry/flock/host observation.
        // On failure, the unpublished promise drops and cancels its intent.
        if let Some(owner) = &self.0.machine {
            promise.machine = Some(
                owner
                    .ensure_reserved(target)
                    .map_err(|e| Refusal::Machine(e.to_string()))?,
            );
        }
        Ok(promise)
    }
}

#[must_use = "unused promises cancel on drop"]
pub(crate) struct Promise {
    domain: Domain,
    remaining: usize,
    machine: Option<MachineAdmission>,
}

impl Promise {
    /// None means this domain has no machine bridge, not machine coverage.
    pub fn machine_admission(&self) -> Option<&MachineAdmission> {
        self.machine.as_ref()
    }

    /// Partition already-admitted demand. No counter or machine target changes.
    /// The child owns its domain even if the originating pass is canceled.
    /// Cloning machine diagnostics can allocate Verdict vectors/strings; this
    /// is host-side ownership work, not an allocation-free hot-path claim.
    pub fn split(&mut self, bytes: usize) -> Result<Self, Refusal> {
        if bytes > self.remaining {
            return Err(Refusal::PromiseTooSmall {
                remaining: self.remaining,
                requested: bytes,
            });
        }
        let child = Self {
            domain: self.domain.clone(),
            remaining: bytes,
            machine: self.machine.clone(),
        };
        self.remaining -= bytes;
        Ok(child)
    }

    /// Join unused admitted demand from ANY promise in this same domain, not
    /// only a split child. This explicitly transfers the caller's remaining
    /// pass entitlement. Failure changes neither owner. Charges already made
    /// from `child` are NOT returned, only its unused remainder.
    pub fn rejoin(&mut self, child: &mut Self) -> Result<(), Refusal> {
        if !Arc::ptr_eq(&self.domain.0, &child.domain.0) {
            return Err(Refusal::DifferentDomain);
        }
        let remaining = self
            .remaining
            .checked_add(child.remaining)
            .ok_or(Refusal::Overflow)?;
        self.remaining = remaining;
        child.remaining = 0;
        Ok(())
    }

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
    pub fn domain(&self) -> &Domain {
        &self.0.domain
    }

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

#[cfg(test)]
mod machine_bridge_tests {
    use super::*;
    use crate::machine::{Gate, GateConfig, Peak};
    use std::path::PathBuf;
    use std::sync::mpsc::channel;
    use std::time::Duration;

    struct RegistryDir(PathBuf);
    impl RegistryDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "lloom-domain-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn gate(&self, available: u64) -> Gate {
            Gate::new(GateConfig {
                registry: Some(self.0.clone()),
                available: Some(available),
                total: Some(available),
                reserve: Some(0),
                ..Default::default()
            })
        }
        fn owner(&self, label: &str, initial: u64) -> (MachineOwner, PathBuf) {
            let admitted = self
                .gate(10_000)
                .admit(label, &Peak::new().with("numeric", initial))
                .unwrap();
            let path = admitted.holding.as_ref().unwrap().path().to_path_buf();
            (MachineOwner::new(admitted).unwrap(), path)
        }
    }
    impl Drop for RegistryDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn partition_return_preserves_admission_and_refuses_cross_domain_transfer() {
        let account = ProcessAccount::new(Some(101));
        let domain = account.domain(Some(100)).unwrap();
        let other = account.domain(Some(1)).unwrap();
        let mut parent = domain.try_reserve(100).unwrap();
        let mut child = parent.split(40).unwrap();
        let charge = child.commit(15).unwrap();
        parent.rejoin(&mut child).unwrap();
        assert_eq!(parent.remaining, 85);
        assert_eq!(child.remaining, 0);
        assert_eq!(
            account.snapshot(),
            Snapshot {
                pending: 85,
                committed: 15
            }
        );
        assert!(parent.split(86).is_err());
        assert_eq!(parent.remaining, 85);
        let mut foreign = other.try_reserve(1).unwrap();
        assert_eq!(parent.rejoin(&mut foreign), Err(Refusal::DifferentDomain));
        assert_eq!((parent.remaining, foreign.remaining), (85, 1));
        drop(foreign);
        drop(child);
        drop(parent);
        assert_eq!(
            account.snapshot(),
            Snapshot {
                pending: 0,
                committed: 15
            }
        );
        drop(charge);
        assert_eq!(account.snapshot(), Snapshot::default());
    }

    #[test]
    fn canceled_pass_worker_and_physical_charge_keep_machine_owner_alive() {
        let dir = RegistryDir::new();
        let (owner, path) = dir.owner("cancel", 0);
        let account = ProcessAccount::new(None);
        let domain = account.domain_with_machine(None, owner).unwrap();
        let mut pass = domain.try_reserve(100).unwrap();
        let mut worker = pass.split(40).unwrap();
        let (resume_tx, resume_rx) = channel();
        let join = std::thread::spawn(move || {
            resume_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let physical = worker.commit(20).unwrap();
            drop(worker); // unused 20 canceled, committed 20 stays owned
            physical
        });
        drop(pass);
        drop(domain);
        assert_eq!(
            account.snapshot(),
            Snapshot {
                pending: 40,
                committed: 0
            }
        );
        assert!(path.exists());
        resume_tx.send(()).unwrap();
        let charge = join.join().unwrap();
        assert!(path.exists(), "worker result still owns physical bytes");
        assert_eq!(
            account.snapshot(),
            Snapshot {
                pending: 0,
                committed: 20
            }
        );
        drop(charge);
        assert!(!path.exists(), "final charge releases holding on host");
        assert_eq!(account.snapshot(), Snapshot::default());
    }

    #[test]
    fn older_intent_never_shrinks_target_and_other_domain_pays_its_own_growth() {
        let dir = RegistryDir::new();
        let account = ProcessAccount::new(Some(200));
        let (owner_a, path_a) = dir.owner("a", 0);
        let (owner_b, path_b) = dir.owner("b", 0);
        let a = account.domain_with_machine(Some(100), owner_a).unwrap();
        let b = account.domain_with_machine(Some(150), owner_b).unwrap();
        let (paused_tx, paused_rx) = channel();
        let (resume_tx, resume_rx) = channel();
        let cloned = a.clone();
        let worker = std::thread::spawn(move || {
            cloned
                .reserve_with(30, |target| {
                    assert_eq!(target, 30);
                    // Reading snapshot here proves the account mutex was released.
                    assert_eq!(cloned.snapshot().pending, 30);
                    paused_tx.send(()).unwrap();
                    resume_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                })
                .unwrap()
        });
        paused_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let newer = a.try_reserve(70).unwrap();
        resume_tx.send(()).unwrap();
        let older = worker.join().unwrap();
        for promise in [&older, &newer] {
            assert!(matches!(
                promise.machine_admission(),
                Some(MachineAdmission::Registered { reserved: 100, .. })
            ));
        }
        let separate = b.try_reserve(80).unwrap();
        assert!(matches!(
            separate.machine_admission(),
            Some(MachineAdmission::Registered { reserved: 80, .. })
        ));
        assert!(matches!(
            b.try_reserve(21),
            Err(Refusal::Limit {
                scope: Scope::Process,
                ..
            })
        )); // shared cap; unchanged owner
        assert_eq!(account.snapshot().pending, 180);
        drop(a);
        drop(older);
        drop(newer);
        assert!(!path_a.exists());
        assert!(path_b.exists(), "a's teardown cannot erase b's reservation");
        drop(b);
        drop(separate);
        assert!(!path_b.exists());
    }

    #[test]
    fn failed_growth_cancels_only_unpublished_intent_and_keeps_durable_target() {
        let dir = RegistryDir::new();
        let (owner, path) = dir.owner("rollback", 10);
        let account = ProcessAccount::new(None);
        let domain = account.domain_with_machine(None, owner).unwrap();
        let before = std::fs::read(&path).unwrap();
        assert!(matches!(
            domain.try_reserve(20_000),
            Err(Refusal::Machine(_))
        ));
        assert_eq!(std::fs::read(&path).unwrap(), before);
        assert_eq!(domain.snapshot(), Snapshot::default());
        assert_eq!(account.snapshot(), Snapshot::default());
        let good = domain.try_reserve(100).unwrap();
        assert!(matches!(
            good.machine_admission(),
            Some(MachineAdmission::Registered {
                reserved: 100,
                growth: Some(_),
                ..
            })
        ));
        assert_eq!(domain.snapshot().pending, 100);
        drop(good);
        drop(domain);
        assert!(!path.exists());
    }

    #[test]
    fn reconstructed_admitted_cannot_bind_one_holding_to_two_domains() {
        let dir = RegistryDir::new();
        let admitted = dir.gate(10_000).admit("alias", &Peak::new()).unwrap();
        let shared = admitted.holding.as_ref().unwrap().clone();
        let verdict = admitted.verdict.clone();
        let alias = || crate::machine::Admitted {
            holding: Some(shared.clone()),
            verdict: verdict.clone(),
        };
        let account = ProcessAccount::new(None);
        let domain = account
            .domain_with_machine(None, MachineOwner::new(admitted).unwrap())
            .unwrap();
        let mut promise = domain.try_reserve(80).unwrap();
        let err = match MachineOwner::new(alias()) {
            Ok(_) => panic!("aliased holding was bound to a second domain"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("reuse that Domain"), "{err}");
        // Recovery 1: share the original domain, hence the same pending total.
        let same = domain.clone();
        let extra = same.try_reserve(20).unwrap();
        assert_eq!(shared.reserved(), 100);
        // Recovery 2: distinct admission/holding, independently counted target.
        let (fresh, _) = dir.owner("distinct", 0);
        let independent = account.domain_with_machine(None, fresh).unwrap();
        let independent_promise = independent.try_reserve(80).unwrap();
        assert!(matches!(
            independent_promise.machine_admission(),
            Some(MachineAdmission::Registered { reserved: 80, .. })
        ));
        let charge = promise.commit(30).unwrap();
        drop(promise);
        drop(extra);
        drop(same);
        drop(domain);
        assert!(
            MachineOwner::new(alias()).is_err(),
            "charge retains exclusive binding"
        );
        drop(charge);
        // An external Arc may keep the old high-water ticket alive, but once
        // the domain/charges are gone, a new domain can bind it conservatively.
        let rebound = MachineOwner::new(alias()).unwrap();
        assert!(matches!(
            rebound.ensure_reserved(20).unwrap(),
            MachineAdmission::Registered { reserved: 100, .. }
        ));
    }

    #[test]
    fn racing_reconstructed_admitted_values_get_one_exclusive_binding() {
        let dir = RegistryDir::new();
        let admitted = dir.gate(10_000).admit("bind race", &Peak::new()).unwrap();
        let ready = Arc::new(std::sync::Barrier::new(3));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let alias = crate::machine::Admitted {
                holding: admitted.holding.clone(),
                verdict: admitted.verdict.clone(),
            };
            let ready = ready.clone();
            threads.push(std::thread::spawn(move || {
                ready.wait();
                MachineOwner::new(alias)
            }));
        }
        ready.wait();
        // Retain successful results while all other attempts finish.
        let results: Vec<_> = threads.into_iter().map(|t| t.join().unwrap()).collect();
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1);
        drop(results);
        assert!(
            MachineOwner::new(admitted).is_ok(),
            "binding released after winner drops"
        );
    }

    #[test]
    fn force_and_off_are_explicit_not_machine_coverage_or_overflow_permission() {
        let dir = RegistryDir::new();
        let account = ProcessAccount::new(None);
        let off = Gate::new(GateConfig {
            off: true,
            ..Default::default()
        })
        .admit("off", &Peak::new())
        .unwrap();
        let domain = account
            .domain_with_machine(None, MachineOwner::new(off).unwrap())
            .unwrap();
        let p = domain.try_reserve(usize::MAX).unwrap();
        assert!(matches!(p.machine_admission(), Some(MachineAdmission::Off)));
        assert!(matches!(domain.try_reserve(1), Err(Refusal::Overflow)));
        drop(p);
        drop(domain);
        let mut config = dir.gate(0).config().clone();
        config.force = true;
        let admitted = Gate::new(config.clone())
            .admit("forced", &Peak::new().with("numeric", 1))
            .unwrap();
        let domain = account
            .domain_with_machine(None, MachineOwner::new(admitted).unwrap())
            .unwrap();
        let p = domain.try_reserve(2).unwrap();
        assert!(matches!(
            p.machine_admission(),
            Some(MachineAdmission::Registered {
                forced: true,
                reserved: 2,
                ..
            })
        ));
        drop(p);
        drop(domain);
        // Deliberate fixture: a file cannot be opened as a registry directory.
        let blocked = dir.0.join("not-a-directory");
        std::fs::write(&blocked, b"fixture").unwrap();
        config.registry = Some(blocked);
        let unregistered = Gate::new(config)
            .admit("unregistered", &Peak::new().with("numeric", 1))
            .unwrap();
        let domain = account
            .domain_with_machine(None, MachineOwner::new(unregistered).unwrap())
            .unwrap();
        let p = domain.try_reserve(2).unwrap();
        assert!(matches!(p.machine_admission(),
            Some(MachineAdmission::ForcedUnregistered { initial }) if initial.forced));
    }
}
