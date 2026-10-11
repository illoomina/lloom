//! Backend-neutral, host-side admission ownership over the shared accounting core.
//!
//! Configure the process cap before the first load. The default is `None`:
//! accounting without an aggregate cap, NOT machine coverage. A model domain has
//! its own immutable quota even when models share a device. Legacy unadmitted
//! domains remain explicit. These tokens do not yet wire native allocations.
//!
//! Pass cancellation closes future issuance, not already-issued tickets. Native
//! owners must retain charges until actual host-side retirement. Mutexes, Arc
//! allocation and final machine Holding destruction make this a host protocol,
//! not a lock-free, allocation-free or callback-safe interface.

use crate::machine::{Admitted, Verdict};
use crate::memory_accounting as core;
use crate::memory_accounting_machine::{MachineAdmission, MachineOwner};
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock, Weak};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AccountSnapshot {
    pub pending: usize,
    pub committed: usize,
}

impl From<core::Snapshot> for AccountSnapshot {
    fn from(s: core::Snapshot) -> Self {
        Self {
            pending: s.pending,
            committed: s.committed,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitScope {
    Process,
    Domain,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdmissionError {
    Overflow,
    DifferentDomain,
    Machine(String),
    Limit {
        scope: LimitScope,
        limit: usize,
        requested_total: usize,
    },
    PromiseTooSmall {
        remaining: usize,
        requested: usize,
    },
    Cancelled,
    ConfigurationConflict {
        active: Option<usize>,
        requested: Option<usize>,
    },
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow => write!(f, "memory admission arithmetic exceeds usize"),
            Self::DifferentDomain => write!(f, "cannot return a promise to a different model domain"),
            Self::Machine(why) => write!(f, "machine admission: {why}"),
            Self::Limit { scope, limit, requested_total } => write!(f, "{scope:?} memory demand {requested_total} exceeds limit {limit}"),
            Self::PromiseTooSmall { remaining, requested } => write!(f, "allocation requests {requested} bytes from a {remaining}-byte promise"),
            Self::Cancelled => write!(f, "allocation context was cancelled"),
            Self::ConfigurationConflict { active, requested } => write!(f, "process memory cap already initialized to {active:?}, requested {requested:?}; configure before the first load"),
        }
    }
}
impl std::error::Error for AdmissionError {}
impl From<core::Refusal> for AdmissionError {
    fn from(e: core::Refusal) -> Self {
        match e {
            core::Refusal::Overflow => Self::Overflow,
            core::Refusal::DifferentDomain => Self::DifferentDomain,
            core::Refusal::Machine(s) => Self::Machine(s),
            core::Refusal::Limit {
                scope,
                limit,
                requested_total,
            } => Self::Limit {
                scope: match scope {
                    core::Scope::Process => LimitScope::Process,
                    core::Scope::Domain => LimitScope::Domain,
                },
                limit,
                requested_total,
            },
            core::Refusal::PromiseTooSmall {
                remaining,
                requested,
            } => Self::PromiseTooSmall {
                remaining,
                requested,
            },
        }
    }
}
pub type Result<T> = std::result::Result<T, AdmissionError>;

/// Machine-policy status is separate from numerical accounting. Registered
/// includes the forced flag: even a registered reservation can bypass refusal.
#[derive(Clone, Debug)]
pub enum MachineStatus {
    Unadmitted,
    Off,
    ForcedUnregistered {
        initial: Verdict,
    },
    Registered {
        reserved: u64,
        forced: bool,
        growth: Option<Verdict>,
    },
}
impl From<Option<MachineAdmission>> for MachineStatus {
    fn from(s: Option<MachineAdmission>) -> Self {
        match s {
            None => Self::Unadmitted,
            Some(MachineAdmission::Off) => Self::Off,
            Some(MachineAdmission::ForcedUnregistered { initial }) => {
                Self::ForcedUnregistered { initial }
            }
            Some(MachineAdmission::Registered {
                reserved,
                forced,
                growth,
            }) => Self::Registered {
                reserved,
                forced,
                growth,
            },
        }
    }
}

/// Only the singleton is public; callers cannot create disjoint process caps.
pub struct ProcessAccount {
    cap: Option<usize>,
    core: Arc<core::ProcessAccount>,
}

#[derive(Default)]
struct ProcessCell(OnceLock<ProcessAccount>);
impl ProcessCell {
    fn get(&self) -> &ProcessAccount {
        self.0.get_or_init(|| ProcessAccount {
            cap: None,
            core: core::ProcessAccount::new(None),
        })
    }
    fn configure(&self, cap: Option<usize>) -> Result<()> {
        let active = self.0.get_or_init(|| ProcessAccount {
            cap,
            core: core::ProcessAccount::new(cap),
        });
        if active.cap != cap {
            return Err(AdmissionError::ConfigurationConflict {
                active: active.cap,
                requested: cap,
            });
        }
        Ok(())
    }
}
static PROCESS: ProcessCell = ProcessCell(OnceLock::new());

/// Identical repeated configuration is idempotent; conflicting configuration is
/// an error, including after default initialization by `process_account`.
pub fn configure_process_account(cap: Option<usize>) -> Result<()> {
    PROCESS.configure(cap)
}
/// Initializes accounting without an aggregate cap unless explicitly configured.
pub fn process_account() -> &'static ProcessAccount {
    PROCESS.get()
}

impl ProcessAccount {
    pub fn snapshot(&self) -> AccountSnapshot {
        self.core.snapshot().into()
    }
    pub fn cap(&self) -> Option<usize> {
        self.cap
    }
    pub fn domain(&self, quota: Option<usize>, admitted: Admitted) -> Result<ModelDomain> {
        let owner =
            MachineOwner::new(admitted).map_err(|e| AdmissionError::Machine(e.to_string()))?;
        Ok(ModelDomain(self.core.domain_with_machine(quota, owner)?))
    }
    /// Compatibility path: numerical accounting, explicitly no machine owner.
    pub fn unadmitted_domain(&self, quota: Option<usize>) -> Result<ModelDomain> {
        Ok(ModelDomain(self.core.domain(quota)?))
    }
}

/// Opaque quota identity, independent of native device/buffer identity. Keeping
/// a key retains its domain (including machine holding), but reserves no bytes.
/// The retained Arc prevents address reuse for as long as this key exists.
#[derive(Clone)]
pub struct DomainKey(core::Domain);
impl PartialEq for DomainKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.identity() == other.0.identity()
    }
}
impl Eq for DomainKey {}
impl std::hash::Hash for DomainKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.0.identity(), state);
    }
}
impl std::fmt::Debug for DomainKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DomainKey").finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub struct ModelDomain(core::Domain);
impl ModelDomain {
    pub fn key(&self) -> DomainKey {
        DomainKey(self.0.clone())
    }
    pub fn snapshot(&self) -> AccountSnapshot {
        self.0.snapshot().into()
    }
    pub fn machine_status(&self) -> MachineStatus {
        self.0.machine_admission().into()
    }
    pub fn reserve(&self, bytes: usize) -> Result<PassReservation> {
        let promise = self.0.try_reserve(bytes)?;
        Ok(PassReservation(Arc::new(PassState {
            promise: Mutex::new(Some(promise)),
            domain: self.0.clone(),
        })))
    }
}
struct PassState {
    promise: Mutex<Option<core::Promise>>,
    // Remains identifiable after cancellation; it cannot reopen issuance.
    domain: core::Domain,
}

#[must_use = "dropping a pass cancels unused demand and closes its contexts"]
pub struct PassReservation(Arc<PassState>);
impl PassReservation {
    pub fn context(&self) -> AllocationContext {
        AllocationContext(self.0.clone())
    }
}
impl Drop for PassReservation {
    fn drop(&mut self) {
        let unused = self.0.promise.lock().take();
        drop(unused); // Account/machine finalizers run after the context unlock.
    }
}

#[derive(Clone)]
pub struct AllocationContext(Arc<PassState>);
impl AllocationContext {
    pub fn domain_key(&self) -> DomainKey {
        DomainKey(self.0.domain.clone())
    }
    pub fn take(&self, upper_bound: usize) -> Result<AllocationTicket> {
        let promise = self
            .0
            .promise
            .lock()
            .as_mut()
            .ok_or(AdmissionError::Cancelled)?
            .split(upper_bound)?;
        Ok(AllocationTicket {
            promise: Some(promise),
            parent: Arc::downgrade(&self.0),
        })
    }
}

#[must_use = "unused tickets return demand to an open parent or cancel it"]
pub struct AllocationTicket {
    promise: Option<core::Promise>,
    parent: Weak<PassState>,
}
impl AllocationTicket {
    /// The machine decision for this issued promise, not proof of native wiring.
    pub fn machine_status(&self) -> MachineStatus {
        self.promise
            .as_ref()
            .and_then(|p| p.machine_admission().cloned())
            .into()
    }
    /// Commit only after allocation with a validated upper bound. Any unused
    /// remainder returns on drop; an over-bound commit fails without charging.
    pub fn commit(mut self, actual: usize) -> Result<PhysicalCharge> {
        Ok(PhysicalCharge(
            self.promise.as_mut().expect("live ticket").commit(actual)?,
        ))
    }
}
impl Drop for AllocationTicket {
    fn drop(&mut self) {
        let Some(mut unused) = self.promise.take() else {
            return;
        };
        if let Some(parent) = self.parent.upgrade() {
            let mut state = parent.promise.lock();
            if let Some(open) = state.as_mut() {
                // Same-domain by construction. A refused rejoin leaves both
                // promises unchanged; cancel this remainder after unlocking.
                let _ = open.rejoin(&mut unused);
            }
        }
        drop(unused);
    }
}

/// Clones retain one physical allocation's charge, not another allocation.
/// Final drop belongs after native resource retirement on a host thread.
#[derive(Clone)]
pub struct PhysicalCharge(core::PhysicalCharge);
impl PhysicalCharge {
    pub(crate) fn same_allocation(&self, other: &Self) -> bool {
        self.0.same_allocation(&other.0)
    }
    pub fn domain_key(&self) -> DomainKey {
        DomainKey(self.0.domain().clone())
    }
    pub fn bytes(&self) -> usize {
        self.0.bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn numeric_domain_keys_distinguish_accounts_and_survive_cancellation() {
        let account = || ProcessAccount {
            core: core::ProcessAccount::new(Some(16)),
            cap: Some(16),
        };
        let a = account().unadmitted_domain(Some(8)).unwrap();
        let b = account().unadmitted_domain(Some(8)).unwrap();
        let key = a.key();
        assert_ne!(
            key,
            b.key(),
            "both isolated core accounts begin at numeric id0"
        );
        let pass = a.reserve(8).unwrap();
        let context = pass.context();
        let ticket = context.take(4).unwrap();
        let mut keys = std::collections::HashSet::new();
        keys.insert(key.clone());
        keys.insert(context.domain_key());
        assert_eq!(keys.len(), 1);
        drop(pass);
        assert_eq!(context.domain_key(), key);
        assert!(matches!(context.take(1), Err(AdmissionError::Cancelled)));
        let charge = ticket.commit(4).unwrap();
        assert_eq!(charge.domain_key(), key);
        drop(a);
        assert!(keys.contains(&charge.domain_key()));
        drop(charge);
        drop(context);
        drop(keys);
        drop(key);
    }

    #[test]
    fn configuration_is_once_and_uses_the_production_cell() {
        let default = ProcessCell::default();
        assert_eq!(default.get().cap(), None);
        default.configure(None).unwrap();
        assert_eq!(
            default.configure(Some(8)),
            Err(AdmissionError::ConfigurationConflict {
                active: None,
                requested: Some(8)
            })
        );
        let cell = Arc::new(ProcessCell::default());
        let (tx, rx) = std::sync::mpsc::channel();
        let jobs: Vec<_> = [8, 16]
            .into_iter()
            .map(|cap| {
                let cell = cell.clone();
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let result = cell.configure(Some(cap));
                    tx.send(result.is_ok()).unwrap();
                })
            })
            .collect();
        let a = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        let b = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert_ne!(a, b);
        for job in jobs {
            job.join().unwrap();
        }
        cell.configure(cell.get().cap()).unwrap();
    }

    #[test]
    fn independent_domains_partition_return_and_charge_lifetimes() {
        let cell = ProcessCell::default();
        cell.configure(Some(16)).unwrap();
        let account = cell.get();
        let a = account.unadmitted_domain(Some(8)).unwrap();
        let b = account.unadmitted_domain(Some(12)).unwrap();
        assert!(matches!(a.machine_status(), MachineStatus::Unadmitted));
        let pa = a.reserve(8).unwrap();
        let pb = b.reserve(8).unwrap();
        assert!(matches!(
            a.reserve(1),
            Err(AdmissionError::Limit {
                scope: LimitScope::Domain,
                ..
            })
        ));
        assert!(matches!(
            b.reserve(1),
            Err(AdmissionError::Limit {
                scope: LimitScope::Process,
                ..
            })
        ));
        let context = pa.context();
        let ticket = context.take(6).unwrap();
        let charge = ticket.commit(4).unwrap(); // two unused bytes rejoin the open pass
        assert_eq!(
            a.snapshot(),
            AccountSnapshot {
                pending: 4,
                committed: 4
            }
        );
        let again = context.take(4).unwrap();
        drop(again);
        let alias = charge.clone();
        drop(pa);
        drop(pb);
        drop(a);
        drop(b);
        drop(charge);
        assert_eq!(
            account.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 4
            }
        );
        assert_eq!(alias.bytes(), 4);
        drop(alias);
        assert_eq!(account.snapshot(), AccountSnapshot::default());
        assert!(matches!(context.take(1), Err(AdmissionError::Cancelled)));
    }

    #[test]
    fn cancellation_closes_context_but_not_issued_ticket() {
        let cell = ProcessCell::default();
        let account = cell.get();
        let domain = account.unadmitted_domain(Some(16)).unwrap();
        let pass = domain.reserve(12).unwrap();
        let context = pass.context();
        let ticket = context.take(8).unwrap();
        let (go, start) = std::sync::mpsc::channel();
        let (done, result) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            start.recv_timeout(Duration::from_secs(5)).unwrap();
            done.send(matches!(context.take(1), Err(AdmissionError::Cancelled)))
                .unwrap();
            ticket.commit(3).unwrap()
        });
        drop(pass);
        drop(domain);
        go.send(()).unwrap();
        assert!(result.recv_timeout(Duration::from_secs(5)).unwrap());
        let charge = thread.join().unwrap();
        assert_eq!(
            account.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 3
            }
        );
        drop(charge);
        assert_eq!(account.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn foreign_rejoin_and_over_bound_commit_preserve_accounting() {
        let cell = ProcessCell::default();
        let account = cell.get();
        let a = account.unadmitted_domain(Some(8)).unwrap();
        let b = account.unadmitted_domain(Some(8)).unwrap();
        let pa = a.reserve(4).unwrap();
        let pb = b.reserve(4).unwrap();
        // Actual core method used by ticket return; public tickets cannot be
        // reassigned to another parent. No duplicate test implementation.
        let mut left = pa.0.promise.lock();
        let mut right = pb.0.promise.lock();
        assert_eq!(
            left.as_mut()
                .unwrap()
                .rejoin(right.as_mut().unwrap())
                .map_err(AdmissionError::from),
            Err(AdmissionError::DifferentDomain)
        );
        drop(left);
        drop(right);
        assert!(matches!(
            pa.context().take(4).unwrap().commit(5),
            Err(AdmissionError::PromiseTooSmall { .. })
        ));
        assert_eq!(
            account.snapshot(),
            AccountSnapshot {
                pending: 8,
                committed: 0
            }
        );
        drop(pa);
        drop(pb);
        assert_eq!(account.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn explicit_off_is_not_unadmitted_or_registered() {
        let cell = ProcessCell::default();
        let domain = cell
            .get()
            .domain(
                Some(8),
                Admitted {
                    holding: None,
                    verdict: None,
                },
            )
            .unwrap();
        assert!(matches!(domain.machine_status(), MachineStatus::Off));
        let pass = domain.reserve(4).unwrap();
        assert!(matches!(
            pass.context().take(2).unwrap().machine_status(),
            MachineStatus::Off
        ));
    }
}
