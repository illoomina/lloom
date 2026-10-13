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
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, OnceLock, Weak,
};

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

/// Which account refused an allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefusalDomain {
    /// The process cap every model domain shares.
    Process,
    /// One model domain's own quota.
    Model,
    /// The issued promise of a pass (`AllocationContext`).
    Promise,
    /// The machine gate's room beside every other lloom process (`machine`).
    Machine,
    /// The per-process budget's residual (`memory.md` §9.5k).
    Budget,
}

/// Why the machine could not admit a reservation's growth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MachineFailure {
    /// The registry or the host's memory could not be read.
    Unavailable,
    /// The holding is absent without an explicit override, or bound to
    /// another domain.
    Binding,
}

impl MachineFailure {
    fn describe(self) -> &'static str {
        match self {
            Self::Unavailable => {
                "the registry or the host's memory could not be read; admission cannot establish room"
            }
            Self::Binding => {
                "the machine holding is absent without an explicit force/off override, or already \
                 belongs to another admission domain; reuse that domain or obtain a distinct \
                 holding with Gate::admit"
            }
        }
    }
}

/// Every refusal an allocation entry point can return, on every backend.
///
/// `Copy` and allocation-free by construction: a refusal on a warm path is
/// built without `format!`, `Box` or a backtrace, and widens into
/// [`crate::Error::Allocation`] with `?`. Detail that needs text (the machine
/// gate's arithmetic) stays with its owner (`machine::Holding::last_refusal`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AllocationError {
    #[error("{domain:?} memory refused {requested} bytes: {available} bytes available")]
    Refused {
        requested: usize,
        available: usize,
        domain: RefusalDomain,
    },
    #[error("allocation reference count exhausted")]
    CountExhausted,
    #[error("allocation context was cancelled")]
    Cancelled,
    #[error("the device refused a {requested}-byte allocation")]
    DeviceOom { requested: usize },
    #[error("cold host allocation returned null")]
    HostAllocation,
    #[error("memory admission arithmetic exceeds usize")]
    Overflow,
    #[error("allocation geometry: {0}")]
    Geometry(&'static str),
    #[error(
        "allocation prediction exceeded: {actual} actual bytes > {predicted} predicted ({basis:?}); \
         raw allocation released; transient peak was not prevented"
    )]
    PredictionExceeded {
        predicted: usize,
        actual: usize,
        basis: PredictionBasis,
    },
    #[error("admitted allocation unsupported: {0}")]
    Unsupported(&'static str),
    #[error("allocation retirement registration: {0}")]
    Retirement(&'static str),
    #[error("allocation custody invariant violated")]
    InvalidCustody,
    #[error(
        "owned Arc quote requires an audited std toolchain/source; legacy allocation APIs remain available"
    )]
    UnsupportedLayout,
    #[error("cannot return a promise to a different model domain")]
    DifferentDomain,
    #[error("reusable allocation container and operation target different native devices")]
    DifferentAllocationDevice,
    #[error("clock-only allocation container has no native device binding")]
    UnboundAllocationDevice,
    #[error(
        "reusable allocation domain mismatch: container={expected}, active payer={actual}; \
         accounted domains require identical origin"
    )]
    ReuseDomainMismatch {
        expected: &'static str,
        actual: &'static str,
    },
    #[error("machine admission: {}", .0.describe())]
    Machine(MachineFailure),
    #[error(
        "process memory cap already initialized to {active:?}, requested {requested:?}; \
         configure before the first load"
    )]
    ConfigurationConflict {
        active: Option<usize>,
        requested: Option<usize>,
    },
}

/// How an allocation's byte prediction was derived.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionBasis {
    /// Heap size/alignment queried for the same options, at least requested
    /// bytes, rounded with checked arithmetic up to the larger of that
    /// alignment and the VM page. An estimator bounding a standalone buffer's
    /// `allocatedSize` (checked natively), NOT Apple's guarantee.
    HeapSizeAndAlignEstimate,
    /// Checked placement chunk request; actual heap allocation may differ.
    HeapDescriptorSizeEstimate,
}

impl From<core::Refusal> for AllocationError {
    fn from(e: core::Refusal) -> Self {
        match e {
            core::Refusal::Overflow => Self::Overflow,
            core::Refusal::UnsupportedLayout => Self::UnsupportedLayout,
            core::Refusal::DifferentDomain => Self::DifferentDomain,
            core::Refusal::ConfigurationConflict { active, requested } => {
                Self::ConfigurationConflict { active, requested }
            }
            core::Refusal::Machine(m) => m.into(),
            core::Refusal::Limit {
                scope,
                limit,
                requested,
                requested_total,
            } => Self::Refused {
                requested,
                available: limit.saturating_sub(requested_total.saturating_sub(requested)),
                domain: match scope {
                    core::Scope::Process => RefusalDomain::Process,
                    core::Scope::Domain => RefusalDomain::Model,
                },
            },
            core::Refusal::PromiseTooSmall {
                remaining,
                requested,
            } => Self::Refused {
                requested,
                available: remaining,
                domain: RefusalDomain::Promise,
            },
        }
    }
}

impl From<crate::memory_accounting_machine::MachineRefusal> for AllocationError {
    fn from(m: crate::memory_accounting_machine::MachineRefusal) -> Self {
        use crate::memory_accounting_machine::MachineRefusal;
        match m {
            MachineRefusal::Room { need, room } => Self::Refused {
                requested: usize::try_from(need).unwrap_or(usize::MAX),
                available: usize::try_from(room.max(0)).unwrap_or(usize::MAX),
                domain: RefusalDomain::Machine,
            },
            MachineRefusal::Unavailable => Self::Machine(MachineFailure::Unavailable),
            MachineRefusal::Binding => Self::Machine(MachineFailure::Binding),
        }
    }
}
pub type Result<T> = std::result::Result<T, AllocationError>;

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
    core: Arc<core::ProcessAccount>,
}

#[derive(Default)]
struct ProcessCell(OnceLock<ProcessAccount>);
impl ProcessCell {
    fn get(&self) -> &ProcessAccount {
        self.0.get_or_init(|| ProcessAccount {
            core: core::ProcessAccount::bootstrap(),
        })
    }
    fn configure(&self, cap: Option<usize>) -> Result<()> {
        self.get()
            .core
            .configure_cap(cap)
            .map_err(AllocationError::from)
    }
}
static PROCESS: ProcessCell = ProcessCell(OnceLock::new());

/// Cold configuration may follow device/bootstrap metadata setup, but must
/// precede the first genuine model load. Existing pending/committed demand is
/// retained and checked. Explicit None/Some policy is immutable/idempotent.
pub fn configure_process_account(cap: Option<usize>) -> Result<()> {
    PROCESS.configure(cap)
}
/// Initializes bootstrap accounting without freezing unspecified cap policy.
pub fn process_account() -> &'static ProcessAccount {
    PROCESS.get()
}

/// Genuine cold model-load entry only. Factories must call this before model
/// preflight/materialization; metadata/domain/device creation must not. Native
/// or legacy factory enrollment is not inferred from this additive hook.
pub fn begin_model_load() {
    process_account().core.begin_model_load();
}

impl ProcessAccount {
    /// Private device/bootstrap metadata-only numerical payer. Does not freeze
    /// unspecified process policy or claim machine/native allocation admission.
    /// Domain/Arc/control overhead still needs its separate price enrollment.
    pub(crate) fn metadata_domain(&self) -> Result<ModelDomain> {
        Ok(ModelDomain(self.core.domain(None)?))
    }
    pub fn snapshot(&self) -> AccountSnapshot {
        self.core.snapshot().into()
    }
    pub fn cap(&self) -> Option<usize> {
        self.core.cap()
    }
    pub fn domain(&self, quota: Option<usize>, admitted: Admitted) -> Result<ModelDomain> {
        let owner = MachineOwner::new(admitted)?;
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
impl DomainKey {
    /// Exact immutable quota/machine owner, not an ambient execution binding.
    pub(crate) fn model_domain(&self) -> ModelDomain {
        ModelDomain(self.0.clone())
    }
}
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

/// Immutable allocation namespace of a reusable container. Legacy is explicit;
/// a missing context is never permission to consume an admitted container.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocationOrigin {
    Legacy,
    Domain(DomainKey),
}
impl AllocationOrigin {
    pub fn category(&self) -> &'static str {
        match self {
            Self::Legacy => "legacy/unadmitted",
            Self::Domain(_) => "accounted domain",
        }
    }
    pub fn validate_reuse(&self, active: &Self) -> Result<()> {
        if self == active {
            Ok(())
        } else {
            Err(AllocationError::ReuseDomainMismatch {
                expected: self.category(),
                actual: active.category(),
            })
        }
    }
}

/// Immutable backing/clock affinity plus quota origin. Generic identity lets
/// CPU controls exercise the exact policy without constructing a native device.
#[derive(Clone, Debug)]
pub struct AllocationAffinity<D> {
    origin: AllocationOrigin,
    device: Option<D>,
}
impl<D: PartialEq> AllocationAffinity<D> {
    pub fn bound(device: D, origin: AllocationOrigin) -> Self {
        Self {
            origin,
            device: Some(device),
        }
    }
    /// Numeric clock-only fixtures cannot acquire native buffers or bind later.
    pub fn clock_only() -> Self {
        Self {
            origin: AllocationOrigin::Legacy,
            device: None,
        }
    }
    pub fn validate_reuse(&self, device: &D, payer: &AllocationOrigin) -> Result<()> {
        match &self.device {
            None => Err(AllocationError::UnboundAllocationDevice),
            Some(bound) if bound != device => Err(AllocationError::DifferentAllocationDevice),
            Some(_) => self.origin.validate_reuse(payer),
        }
    }
}

#[derive(Clone)]
pub struct ModelDomain(core::Domain);
impl ModelDomain {
    #[cfg(test)]
    pub(crate) fn isolated_numeric_fixture(cap: usize) -> Self {
        Self(
            core::ProcessAccount::new(Some(cap))
                .domain(Some(cap))
                .expect("numeric fixture domain"),
        )
    }
    #[cfg(test)]
    pub(crate) fn isolated_numeric_domains(cap: usize, count: usize) -> Vec<Self> {
        assert!(count <= 16, "bounded scalar fixture domains");
        let account = core::ProcessAccount::new(Some(cap));
        (0..count)
            .map(|_| Self(account.domain(Some(cap)).expect("numeric fixture domain")))
            .collect()
    }
    pub fn key(&self) -> DomainKey {
        DomainKey(self.0.clone())
    }
    pub fn snapshot(&self) -> AccountSnapshot {
        self.0.snapshot().into()
    }
    #[cfg(test)]
    pub(crate) fn test_account_snapshot_nonblocking(&self) -> Option<AccountSnapshot> {
        self.0.test_account_snapshot_nonblocking().map(Into::into)
    }
    pub fn machine_status(&self) -> MachineStatus {
        self.0.machine_admission().into()
    }
    pub fn reserve(&self, bytes: usize) -> Result<PassReservation> {
        let promise = self.0.try_reserve(bytes)?;
        Ok(PassReservation(Arc::new(PassState {
            promise: Mutex::new(Some(promise)),
            closed: AtomicBool::new(false),
            domain: self.0.clone(),
        })))
    }
}
struct PassState {
    // Monotonic cancellation observation for warm paths; issuance still uses
    // the authoritative promise mutex. No context can reopen this pass.
    closed: AtomicBool,
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
        let unused = {
            let mut promise = self.0.promise.lock();
            // Cancellation linearizes under the same mutex as take(): tickets
            // issued before this point remain valid; later issuance refuses.
            self.0.closed.store(true, Ordering::Release);
            promise.take()
        };
        drop(unused); // Account/machine finalizers run after the context unlock.
    }
}

#[derive(Clone)]
pub struct AllocationContext(Arc<PassState>);
impl AllocationContext {
    /// Borrow warm permission only; no promise issuance, allocation or lock.
    pub(crate) fn operation_context(&self) -> OperationContext<'_> {
        OperationContext(OperationPermission::Supplied(self))
    }
    pub fn domain_key(&self) -> DomainKey {
        DomainKey(self.0.domain.clone())
    }
    /// Observe cancellation without taking the promise mutex on warm reuse.
    /// This is not an execution lease: cancellation may follow the observation.
    /// `take` remains the authoritative serialized issuance boundary on misses.
    pub(crate) fn check_open(&self) -> Result<()> {
        if self.0.closed.load(Ordering::Acquire) {
            Err(AllocationError::Cancelled)
        } else {
            Ok(())
        }
    }
    pub fn take(&self, upper_bound: usize) -> Result<AllocationTicket> {
        let promise = self
            .0
            .promise
            .lock()
            .as_mut()
            .ok_or(AllocationError::Cancelled)?
            .split(upper_bound)?;
        Ok(AllocationTicket {
            promise: Some(promise),
            parent: Arc::downgrade(&self.0),
        })
    }
}

/// Borrowed warm-reuse permission. This is NOT a numerical promise or an
/// execution lease: supplied cancellation may follow a successful observation.
/// No take/reserve/owner clone exists. Actual cold misses still issue from the
/// authoritative same-account AllocationContext outside native/engine guards.
/// Never store this borrow in catalog/native cleanup captures.
pub(crate) struct OperationContext<'a>(OperationPermission<'a>);
enum OperationPermission<'a> {
    Supplied(&'a AllocationContext),
    DefaultNamespace(&'a ModelDomain),
}
impl OperationContext<'_> {
    /// # Safety
    /// `domain` is borrowed from the actual DEFAULT AllocationNamespace whose
    /// lifetime/root custody spans this operation. Its model-domain field is
    /// None. Never use this route for a supplied/admitted namespace or to bypass
    /// an active supplied context's cancellation/origin checks. This permission
    /// authorizes no allocation, uninitialized read, raw escape or native use.
    pub(crate) unsafe fn default_namespace(domain: &ModelDomain) -> OperationContext<'_> {
        OperationContext(OperationPermission::DefaultNamespace(domain))
    }
    pub(crate) fn domain_key(&self) -> DomainKey {
        match self.0 {
            OperationPermission::Supplied(context) => context.domain_key(),
            OperationPermission::DefaultNamespace(domain) => domain.key(),
        }
    }
    pub(crate) fn check_open(&self) -> Result<()> {
        match self.0 {
            OperationPermission::Supplied(context) => context.check_open(),
            // Default has no pass to cancel. Namespace borrow/root and the
            // actual capsule/Commands failure checks still gate every attempt.
            OperationPermission::DefaultNamespace(_) => Ok(()),
        }
    }
}

#[must_use = "unused tickets return demand to an open parent or cancel it"]
pub struct AllocationTicket {
    promise: Option<core::Promise>,
    parent: Weak<PassState>,
}
impl AllocationTicket {
    /// Issued promise identity survives parent-pass cancellation. Cold producers
    /// verify this before allocating; returned keys must die outside their locks.
    pub(crate) fn domain_key(&self) -> DomainKey {
        DomainKey(self.promise.as_ref().expect("live ticket").domain().clone())
    }
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
        self.try_commit(actual)
    }
    /// Cold requested-layout path. The prediction includes actual payload plus
    /// requested_charge_control_bytes BEFORE allocation. A refusal keeps this
    /// ticket live; destroy unchargeable storage before returning its promise.
    pub(crate) fn try_commit_with_control(&mut self, actual: usize) -> Result<PhysicalCharge> {
        Ok(PhysicalCharge(
            self.promise
                .as_mut()
                .expect("live ticket")
                .commit_with_control(actual)?,
        ))
    }
    pub(crate) fn requested_charge_control_bytes() -> Result<usize> {
        core::PhysicalCharge::requested_control_bytes().map_err(AllocationError::from)
    }
    /// Transfer demand without consuming the ticket on refusal. Native callers
    /// must destroy an unchargeable raw allocation BEFORE dropping this ticket.
    /// Success can leave unused demand; dropping the ticket returns it normally.
    pub fn try_commit(&mut self, actual: usize) -> Result<PhysicalCharge> {
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
    fn reusable_origin_checks_actual_domain_identity_and_explicit_legacy() {
        let account = ProcessAccount {
            core: core::ProcessAccount::new(Some(16)),
        };
        let a = account.unadmitted_domain(Some(8)).unwrap();
        let b = account.unadmitted_domain(Some(8)).unwrap();
        let origin = AllocationOrigin::Domain(a.key());
        let same = AllocationOrigin::Domain(a.key());
        let foreign = AllocationOrigin::Domain(b.key());
        let legacy = AllocationOrigin::Legacy;
        assert!(origin.validate_reuse(&same).is_ok());
        assert!(legacy.validate_reuse(&legacy).is_ok());
        for (container, payer) in [(&origin, &foreign), (&origin, &legacy), (&legacy, &origin)] {
            assert_eq!(
                container.validate_reuse(payer),
                Err(AllocationError::ReuseDomainMismatch {
                    expected: container.category(),
                    actual: payer.category(),
                })
            );
        }
        assert_eq!(account.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn reusable_affinity_checks_device_before_domain_and_rejects_unbound() {
        let account = ProcessAccount {
            core: core::ProcessAccount::new(Some(16)),
        };
        let domain = account.unadmitted_domain(Some(8)).unwrap();
        let payer = AllocationOrigin::Domain(domain.key());
        let ring = AllocationAffinity::bound(7u32, payer.clone());
        assert!(ring.validate_reuse(&7, &payer).is_ok());
        assert_eq!(
            ring.validate_reuse(&8, &payer),
            Err(AllocationError::DifferentAllocationDevice)
        );
        assert!(matches!(
            ring.validate_reuse(&7, &AllocationOrigin::Legacy),
            Err(AllocationError::ReuseDomainMismatch { .. })
        ));
        let legacy = AllocationAffinity::bound(7u32, AllocationOrigin::Legacy);
        assert!(legacy.validate_reuse(&7, &AllocationOrigin::Legacy).is_ok());
        assert_eq!(
            AllocationAffinity::<u32>::clock_only().validate_reuse(&7, &AllocationOrigin::Legacy),
            Err(AllocationError::UnboundAllocationDevice)
        );
        assert_eq!(account.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn cancellation_flag_and_racing_issuance_preserve_owned_ticket() {
        let account = ProcessAccount {
            core: core::ProcessAccount::new(Some(16)),
        };
        let domain = account.unadmitted_domain(Some(16)).unwrap();
        let pass = domain.reserve(8).unwrap();
        let context = pass.context();
        assert!(context.check_open().is_ok());
        // An issued ticket is independent of cancellation; the racing request
        // may issue before Drop or refuse after it. Both conserve the same sum.
        let issued = context.take(4).unwrap();
        let (start_tx, start_rx) = std::sync::mpsc::channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            let context = context.clone();
            let worker = scope.spawn(move || {
                start_rx.recv_timeout(Duration::from_secs(5)).unwrap();
                result_tx.send(context.take(4)).unwrap();
            });
            start_tx.send(()).unwrap();
            drop(pass);
            let raced = result_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            match raced {
                Ok(ticket) => {
                    assert_eq!(domain.snapshot().pending, 8);
                    drop(ticket);
                }
                Err(e) => assert_eq!(e, AllocationError::Cancelled),
            }
            worker.join().unwrap();
        });
        assert_eq!(context.check_open(), Err(AllocationError::Cancelled));
        assert!(matches!(context.take(0), Err(AllocationError::Cancelled)));
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 4,
                committed: 0
            }
        );
        let charge = issued.commit(4).unwrap();
        assert_eq!(
            domain.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 4
            }
        );
        drop(charge);
        assert_eq!(account.snapshot(), AccountSnapshot::default());
    }

    #[test]
    fn numeric_domain_keys_distinguish_accounts_and_survive_cancellation() {
        let account = || ProcessAccount {
            core: core::ProcessAccount::new(Some(16)),
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
        assert!(matches!(context.take(1), Err(AllocationError::Cancelled)));
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
            Err(AllocationError::ConfigurationConflict {
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
        assert_eq!(
            a.reserve(1).err(),
            Some(AllocationError::Refused {
                requested: 1,
                available: 0,
                domain: RefusalDomain::Model,
            })
        );
        assert_eq!(
            b.reserve(1).err(),
            Some(AllocationError::Refused {
                requested: 1,
                available: 0,
                domain: RefusalDomain::Process,
            })
        );
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
        assert!(matches!(context.take(1), Err(AllocationError::Cancelled)));
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
            done.send(matches!(context.take(1), Err(AllocationError::Cancelled)))
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
                .map_err(AllocationError::from),
            Err(AllocationError::DifferentDomain)
        );
        drop(left);
        drop(right);
        assert_eq!(
            pa.context().take(4).unwrap().commit(5).err(),
            Some(AllocationError::Refused {
                requested: 5,
                available: 4,
                domain: RefusalDomain::Promise,
            })
        );
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
    #[test]
    fn cap_adoption_metadata_facade_remains_configurable_without_reset() {
        let cell = ProcessCell::default();
        let account = cell.get();
        let metadata = account.metadata_domain().unwrap();
        let pass = metadata.reserve(4).unwrap();
        let charge = pass.context().take(4).unwrap().commit(4).unwrap();
        drop(pass);
        assert_eq!(
            account.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 4
            }
        );
        assert!(matches!(
            cell.configure(Some(3)),
            Err(AdmissionError::Limit {
                scope: LimitScope::Process,
                limit: 3,
                requested_total: 4
            })
        ));
        assert_eq!(account.cap(), None);
        cell.configure(Some(4)).unwrap();
        assert_eq!(account.cap(), Some(4));
        assert_eq!(
            account.snapshot(),
            AccountSnapshot {
                pending: 0,
                committed: 4
            }
        );
        account.core.begin_model_load(); // Same core method used by public hook.
        assert_eq!(
            cell.configure(Some(8)),
            Err(AdmissionError::ConfigurationConflict {
                active: Some(4),
                requested: Some(8)
            })
        );
        assert!(matches!(
            metadata.reserve(1),
            Err(AdmissionError::Limit {
                limit: 4,
                requested_total: 5,
                ..
            })
        ));
        drop(charge);
        assert_eq!(account.snapshot(), AccountSnapshot::default());
        assert_eq!(account.cap(), Some(4));
    }
}
