//! L1 of the memory system (`profiles-and-memory-design.md` §4.1, §4.2): the
//! ledger. Integers and atomics; `currentAllocatedSize` is a gauge read beside
//! it, never the thing a decision is made on (§6.3d: accounting is cheap,
//! Metal calls are not).
//!
//! Three things live here. A [`Reservation`] is bytes an allocation holds for
//! its lifetime — every byte through a backing (`backing.rs`), the slot
//! pools, the arena and pass-plan bases, the heap chunks — released when it
//! drops, so the committed figure cannot drift from what exists. A [`Tenant`]
//! is a holder that can say what it needs and wants (a [`Claim`]) and what it
//! holds now; [`arbitrate`] fits the claims into the quota and
//! [`Ledger::yield_for`] asks the caches to give back. And the process's
//! reservations on the machine (`machine.rs`): a pass admitted against the
//! quota whose bytes would pass what the process reserved at load widens that
//! reservation first ([`Ledger::admit_machine`]).
//!
//! `reserve` never refuses: admission is the pass's
//! (`MetalDevice::admit_bytes`), never the allocation's.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

use crate::machine::Holding;

pub use crate::memory_arbitration::{
    arbitrate, Arbitration, BulkYield, Claim, Class, Grant, LedgerSnapshot, TenantId, TenantRow,
    YieldReport,
};

/// A holder of memory the ledger can ask about. Reads must be cheap: the
/// snapshot runs on a report, never on a step, but a tenant's `claim` and
/// `held` must not take a lock a step holds.
pub trait Tenant: Send + Sync {
    fn name(&self) -> String;
    fn claim(&self) -> Claim;
    /// Bytes held now.
    fn held(&self) -> usize;
    /// Opt in only when one atomic operation can report an exact pledged delta.
    /// Default retains the legacy callback-per-unit behavior.
    fn yield_bulk(&self, _bytes: usize) -> BulkYield {
        BulkYield::Legacy
    }
    /// Cold path: grow by refilling, shrink by evict + retire
    /// (`profiles-and-memory-design.md` §4.2). Returns what the tenant
    /// holds after. A fixed tenant (the default) holds what it holds.
    fn resize(&self, _grant: usize) -> usize {
        self.held()
    }
}

/// A tenant whose claim and holding its owner sets: the plain shape for a
/// cache or a pool that knows its own numbers. The owner keeps the `Arc`;
/// the ledger keeps a `Weak`, so a dropped owner drops its row.
#[derive(Debug)]
pub struct Gauge {
    name: String,
    claim: parking_lot::Mutex<Claim>,
    held: AtomicUsize,
}

impl Gauge {
    pub fn new(name: impl Into<String>, claim: Claim, held: usize) -> Arc<Self> {
        Arc::new(Self {
            name: name.into(),
            claim: parking_lot::Mutex::new(claim),
            held: AtomicUsize::new(held),
        })
    }

    pub fn set_held(&self, bytes: usize) {
        self.held.store(bytes, Ordering::Relaxed);
    }

    pub fn set_claim(&self, claim: Claim) {
        *self.claim.lock() = claim;
    }
}

impl Tenant for Gauge {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn claim(&self) -> Claim {
        *self.claim.lock()
    }
    fn held(&self) -> usize {
        self.held.load(Ordering::Relaxed)
    }
}

/// Bytes committed for the life of the holder: released on drop, so the
/// ledger's committed figure is a sum of what exists.
#[must_use = "a reservation released at once accounts for nothing"]
#[derive(Debug)]
pub struct Reservation {
    ledger: Arc<Ledger>,
    bytes: usize,
}

impl Reservation {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        self.ledger
            .committed
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// One per device: the quota, the committed bytes and the tenant registry.
#[derive(Debug)]
pub struct Ledger {
    quota: AtomicUsize,
    committed: AtomicUsize,
    /// Pass-level admissions that gave the pools' free lists back before
    /// deciding, and the bytes they freed (`MetalDevice::admit_bytes`): the
    /// release counter on the path a refusal used to take at once.
    reclaims: AtomicU32,
    reclaimed_bytes: AtomicUsize,
    drains: AtomicU32,
    next_id: AtomicU32,
    tenants: parking_lot::Mutex<Vec<(TenantId, Weak<dyn Tenant>)>>,
    /// The process's reservations on the machine, held by the loads that
    /// made them (`machine::Gate::admit`); a dropped load drops its row.
    machine: parking_lot::Mutex<Vec<Weak<Holding>>>,
}

impl Default for Ledger {
    fn default() -> Self {
        Self {
            quota: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
            reclaims: AtomicU32::new(0),
            reclaimed_bytes: AtomicUsize::new(0),
            drains: AtomicU32::new(0),
            next_id: AtomicU32::new(1),
            tenants: parking_lot::Mutex::new(Vec::new()),
            machine: parking_lot::Mutex::new(Vec::new()),
        }
    }
}

impl Ledger {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    /// The bytes this process may hold on the device, from the binary's
    /// fraction of the working set. 0 = unset (a harness without a budget).
    pub fn set_quota(&self, bytes: usize) {
        self.quota.store(bytes, Ordering::Relaxed);
    }

    pub fn quota(&self) -> usize {
        self.quota.load(Ordering::Relaxed)
    }

    /// Bytes under live reservations.
    pub fn committed(&self) -> usize {
        self.committed.load(Ordering::Acquire)
    }

    fn live_tenants(&self) -> Vec<(TenantId, Arc<dyn Tenant>)> {
        let t = self.tenants.lock();
        t.iter()
            .filter_map(|(id, w)| w.upgrade().map(|t| (*id, t)))
            .collect()
    }

    /// The arbiter (§4.2) over the registered tenants' claims: refuse on
    /// the floors, grant every want that fits, else fill upward by class;
    /// then resize the tenants whose grant differs from what they hold.
    /// Returns the arbitration and what each resized tenant holds after.
    pub fn arbitrate(&self) -> (Arbitration, Vec<(TenantId, usize)>) {
        let snap = self.snapshot();
        let a = arbitrate(snap.quota, &snap.tenants);
        let mut resized = Vec::new();
        if let Arbitration::Granted(grants) = &a {
            let tenants = self.live_tenants();
            for g in grants {
                if let Some((_, t)) = tenants.iter().find(|(id, _)| *id == g.id) {
                    if t.held() != g.bytes {
                        let after = t.resize(g.bytes);
                        resized.push((g.id, after));
                    }
                }
            }
        }
        (a, resized)
    }

    /// Compatibility scalar: caps an unrepresentable pledge at usize::MAX and
    /// omits contention. New admission callers must use `yield_for_report`.
    pub fn yield_for(&self, bytes: usize) -> usize {
        self.yield_for_report(bytes).pledged.min(usize::MAX as u128) as usize
    }

    /// Ask Scratch, then Cache (never Capability) to yield. Only opt-in tenants
    /// use bulk; arbitrary third-party resize callbacks still run once per unit.
    /// Pledge is not credit: committed changes only when physical owners release.
    pub fn yield_for_report(&self, bytes: usize) -> YieldReport {
        let mut report = YieldReport::default();
        let tenants = self.live_tenants();
        for class in [Class::Scratch, Class::Cache] {
            for (_, t) in tenants.iter().filter(|(_, t)| t.claim().class == class) {
                if report.pledged >= bytes as u128 {
                    return report;
                }
                let remaining = (bytes as u128 - report.pledged) as usize;
                match t.yield_bulk(remaining) {
                    BulkYield::Pledged { bytes } => {
                        report.pledged += bytes as u128;
                        continue;
                    }
                    BulkYield::Contended => {
                        report.contended += 1;
                        continue;
                    }
                    BulkYield::Legacy => {}
                }
                loop {
                    if report.pledged >= bytes as u128 {
                        return report;
                    }
                    let claim = t.claim();
                    let held = t.held();
                    let unit = claim.unit.max(1);
                    if held <= claim.floor || held < unit {
                        break;
                    }
                    let target = (held - unit).max(claim.floor);
                    let after = t.resize(target);
                    if after >= held {
                        break;
                    }
                    report.pledged += (held - after) as u128;
                }
            }
        }
        report
    }

    /// A pass-level admission gave the free lists back: count it.
    /// A refusal waited the queue out for the pending lists (the second
    /// rung); counted so the arm's cost is a number.
    pub fn note_drain(&self) {
        self.drains.fetch_add(1, Ordering::Relaxed);
    }

    pub fn drains(&self) -> u32 {
        self.drains.load(Ordering::Relaxed)
    }

    pub fn note_reclaim(&self, bytes: usize) {
        self.reclaims.fetch_add(1, Ordering::Relaxed);
        self.reclaimed_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    /// `(reclaims, bytes freed by them)` so far.
    pub fn reclaims(&self) -> (u32, usize) {
        (
            self.reclaims.load(Ordering::Relaxed),
            self.reclaimed_bytes.load(Ordering::Relaxed),
        )
    }

    /// Commit `bytes` for as long as the reservation lives. Never refuses
    /// here: admission against the quota is the pass's, not the
    /// allocation's.
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Reservation {
        self.committed.fetch_add(bytes, Ordering::AcqRel);
        Reservation {
            ledger: Arc::clone(self),
            bytes,
        }
    }

    /// Count `holding` as this process's reservation on the machine for as
    /// long as its owner keeps it.
    pub fn attach_machine(&self, holding: &Arc<Holding>) {
        let mut m = self.machine.lock();
        m.retain(|w| w.strong_count() > 0);
        m.push(Arc::downgrade(holding));
    }

    /// Bytes the process holds reserved on the machine; `None` when no load
    /// reserved any (a harness, a test, the gate off).
    pub fn machine_reserved(&self) -> Option<u64> {
        let m = self.machine.lock();
        let live: Vec<Arc<Holding>> = m.iter().filter_map(Weak::upgrade).collect();
        (!live.is_empty()).then(|| live.iter().map(|h| h.reserved()).sum())
    }

    /// Admit `bytes` on the machine: inside what the process reserved,
    /// nothing is read; past it, the first reservation is widened by the
    /// excess through the machine gate's arithmetic (`Holding::grow_to`), or
    /// the refusal's text is returned. No reservation, nothing to check.
    pub fn admit_machine(&self, bytes: u64) -> std::result::Result<(), String> {
        let live: Vec<Arc<Holding>> = {
            let m = self.machine.lock();
            m.iter().filter_map(Weak::upgrade).collect()
        };
        let Some(first) = live.first() else {
            return Ok(());
        };
        let reserved: u64 = live.iter().map(|h| h.reserved()).sum();
        if bytes <= reserved {
            return Ok(());
        }
        first
            .grow_to(first.reserved() + (bytes - reserved))
            .map(|_| ())
            .map_err(|e| e.to_string())
    }

    /// List `tenant` until its owner drops it.
    pub fn register(&self, tenant: &Arc<dyn Tenant>) -> TenantId {
        let id = TenantId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let mut t = self.tenants.lock();
        t.retain(|(_, w)| w.strong_count() > 0);
        t.push((id, Arc::downgrade(tenant)));
        id
    }

    /// Drop `id`'s row before its owner goes away.
    pub fn unregister(&self, id: TenantId) {
        self.tenants.lock().retain(|(i, _)| *i != id);
    }

    /// Every live tenant's claim and holding, with the quota and the
    /// committed bytes. A report-time read: each tenant's `claim` and
    /// `held`, and the registry's lock.
    pub fn snapshot(&self) -> LedgerSnapshot {
        let mut tenants = Vec::new();
        {
            let mut t = self.tenants.lock();
            t.retain(|(_, w)| w.strong_count() > 0);
            for (id, w) in t.iter() {
                if let Some(tenant) = w.upgrade() {
                    tenants.push(TenantRow {
                        id: *id,
                        name: tenant.name(),
                        claim: tenant.claim(),
                        held: tenant.held(),
                    });
                }
            }
        }
        LedgerSnapshot {
            quota: self.quota(),
            committed: self.committed(),
            reclaims: self.reclaims(),
            drains: self.drains(),
            tenants,
        }
    }

    /// Live tenants.
    pub fn tenants(&self) -> usize {
        let mut t = self.tenants.lock();
        t.retain(|(_, w)| w.strong_count() > 0);
        t.len()
    }
}

/// One tenant's grant from an arbitration.
#[cfg(test)]
mod tests {
    use super::*;

    struct BulkOnly {
        held: AtomicUsize,
        contended: bool,
        class: Class,
    }
    impl Tenant for BulkOnly {
        fn name(&self) -> String {
            "bulk-test".into()
        }
        fn claim(&self) -> Claim {
            Claim {
                floor: 0,
                want: self.held(),
                unit: 1,
                class: self.class,
            }
        }
        fn held(&self) -> usize {
            self.held.load(Ordering::Relaxed)
        }
        fn yield_bulk(&self, _: usize) -> BulkYield {
            if self.contended {
                BulkYield::Contended
            } else {
                BulkYield::Pledged {
                    bytes: self.held.swap(0, Ordering::Relaxed),
                }
            }
        }
        fn resize(&self, _: usize) -> usize {
            panic!("bulk tenant must not use legacy resize")
        }
    }

    #[test]
    fn numeric_bulk_yield_reports_exact_overshoot_and_contention() {
        let ledger = Ledger::new();
        let tenants: Vec<Arc<dyn Tenant>> = vec![
            Arc::new(BulkOnly {
                held: AtomicUsize::new(1),
                contended: true,
                class: Class::Scratch,
            }),
            Arc::new(BulkOnly {
                held: AtomicUsize::new(usize::MAX - 1),
                contended: false,
                class: Class::Cache,
            }),
            Arc::new(BulkOnly {
                held: AtomicUsize::new(usize::MAX),
                contended: false,
                class: Class::Cache,
            }),
            Arc::new(BulkOnly {
                held: AtomicUsize::new(1),
                contended: true,
                class: Class::Capability,
            }),
        ];
        for t in &tenants {
            ledger.register(t);
        }
        assert_eq!(
            ledger.yield_for_report(usize::MAX),
            YieldReport {
                pledged: 2 * usize::MAX as u128 - 1,
                contended: 1
            }
        );
        assert_eq!(ledger.committed(), 0, "a pledge is not physical credit");
        assert_eq!(tenants[3].held(), 1, "capability not visited");
        assert_eq!(ledger.yield_for_report(0), YieldReport::default());
    }

    #[test]
    fn numeric_legacy_yield_keeps_each_callback() {
        struct Stateful {
            held: AtomicUsize,
            calls: AtomicUsize,
        }
        impl Tenant for Stateful {
            fn name(&self) -> String {
                "legacy-stateful".into()
            }
            fn claim(&self) -> Claim {
                Claim {
                    floor: 2,
                    want: 8,
                    unit: 2,
                    class: Class::Cache,
                }
            }
            fn held(&self) -> usize {
                self.held.load(Ordering::Relaxed)
            }
            fn resize(&self, grant: usize) -> usize {
                assert_eq!(grant, self.held() - 2, "one-unit callback contract");
                self.calls.fetch_add(1, Ordering::Relaxed);
                self.held.store(grant, Ordering::Relaxed);
                grant
            }
        }
        let ledger = Ledger::new();
        let owner = Arc::new(Stateful {
            held: AtomicUsize::new(8),
            calls: AtomicUsize::new(0),
        });
        let tenant: Arc<dyn Tenant> = owner.clone();
        ledger.register(&tenant);
        assert_eq!(
            ledger.yield_for_report(5),
            YieldReport {
                pledged: 6,
                contended: 0
            }
        );
        assert_eq!(owner.calls.load(Ordering::Relaxed), 3);
        assert_eq!(ledger.yield_for(1), 0);
    }

    /// A tenant that yields: its `resize` drops a unit's reservation at
    /// once and reports the pledge; a fixed tenant (a `Gauge`) holds what
    /// it holds.
    struct Elastic {
        ledger: Arc<Ledger>,
        unit: usize,
        floor: usize,
        class: Class,
        held: AtomicUsize,
        units: parking_lot::Mutex<Vec<Reservation>>,
    }

    impl Tenant for Elastic {
        fn name(&self) -> String {
            "elastic".into()
        }
        fn claim(&self) -> Claim {
            Claim {
                floor: self.floor,
                want: 10 * self.unit,
                unit: self.unit,
                class: self.class,
            }
        }
        fn held(&self) -> usize {
            self.held.load(Ordering::Relaxed)
        }
        fn resize(&self, grant: usize) -> usize {
            let before = self.held();
            let after = grant.max(self.floor);
            let mut units = self.units.lock();
            while units.len() * self.unit > after {
                units.pop();
            }
            while units.len() * self.unit < after {
                units.push(self.ledger.reserve(self.unit));
            }
            let _ = before;
            self.held.store(units.len() * self.unit, Ordering::Relaxed);
            self.held()
        }
    }

    #[test]
    fn a_yield_asks_the_caches_by_units_down_to_their_floor_and_never_a_capability() {
        let ledger = Ledger::new();
        let mib = 1usize << 20;
        let _fixed_bytes = ledger.reserve(60 * mib);
        let elastic = Arc::new(Elastic {
            ledger: Arc::clone(&ledger),
            unit: 4 * mib,
            floor: 8 * mib,
            class: Class::Cache,
            held: AtomicUsize::new(40 * mib),
            units: parking_lot::Mutex::new((0..10).map(|_| ledger.reserve(4 * mib)).collect()),
        });
        // A capability that COULD shrink if asked: the yield must never ask.
        let able = Arc::new(Elastic {
            ledger: Arc::clone(&ledger),
            unit: 4 * mib,
            floor: 8 * mib,
            class: Class::Capability,
            held: AtomicUsize::new(40 * mib),
            units: parking_lot::Mutex::new((0..10).map(|_| ledger.reserve(4 * mib)).collect()),
        });
        let c: Arc<dyn Tenant> = able.clone();
        ledger.register(&c);
        let fixed = Gauge::new(
            "kv",
            Claim {
                floor: 50 * mib,
                want: 50 * mib,
                unit: mib,
                class: Class::Capability,
            },
            50 * mib,
        );
        let e: Arc<dyn Tenant> = elastic.clone();
        let f: Arc<dyn Tenant> = fixed.clone();
        ledger.register(&e);
        ledger.register(&f);
        assert_eq!(ledger.committed(), 140 * mib);
        // Short by 10 MiB: three units of 4 MiB are pledged, no more.
        assert_eq!(ledger.yield_for(10 * mib), 12 * mib);
        assert_eq!(elastic.held(), 28 * mib);
        assert_eq!(
            ledger.committed(),
            128 * mib,
            "the units' reservations dropped"
        );
        // Short by more than it can give: down to the floor; the fixed
        // capability untouched, and the capability that could shrink never
        // asked.
        assert_eq!(ledger.yield_for(100 * mib), 20 * mib);
        assert_eq!(elastic.held(), 8 * mib);
        assert_eq!(fixed.held(), 50 * mib);
        assert_eq!(able.held(), 40 * mib, "a capability was asked to yield");
        assert_eq!(ledger.committed(), 108 * mib);
        // Growing back: the arbiter grants every want that fits, and the
        // tenant whose grant moved is resized.
        ledger.set_quota(200 * mib);
        let (a, resized) = ledger.arbitrate();
        assert!(matches!(a, Arbitration::Granted(_)));
        assert_eq!(resized.len(), 1, "{resized:?}");
        assert_eq!(elastic.held(), 40 * mib, "the cache's want");
        assert_eq!(able.held(), 40 * mib);
        assert_eq!(ledger.committed(), 140 * mib);
    }

    #[test]
    fn a_reservation_is_committed_for_its_life_and_released_on_drop() {
        let ledger = Ledger::new();
        assert_eq!(ledger.committed(), 0);
        let a = ledger.reserve(1 << 20);
        let b = ledger.reserve(3 << 20);
        assert_eq!(ledger.committed(), 4 << 20, "two live reservations sum");
        drop(a);
        assert_eq!(
            ledger.committed(),
            3 << 20,
            "a dropped reservation is released"
        );
        drop(b);
        assert_eq!(ledger.committed(), 0);
    }

    #[test]
    fn a_tenant_is_listed_until_its_owner_drops_it() {
        let ledger = Ledger::new();
        let claim = Claim {
            floor: 64 << 20,
            want: 136 << 20,
            unit: 64 << 20,
            class: Class::Capability,
        };
        let kv = Gauge::new("kv", claim, 64 << 20);
        let kv_dyn: Arc<dyn Tenant> = kv.clone();
        let id = ledger.register(&kv_dyn);
        let snap = ledger.snapshot();
        assert_eq!(snap.tenants.len(), 1);
        assert_eq!(snap.tenants[0].id, id);
        assert_eq!(snap.tenants[0].claim, claim);
        assert_eq!(snap.floors(), 64 << 20);
        assert_eq!(snap.wants(), 136 << 20);
        kv.set_held(128 << 20);
        assert_eq!(ledger.snapshot().tenants[0].held, 128 << 20);
        drop(kv_dyn);
        drop(kv);
        assert_eq!(
            ledger.snapshot().tenants.len(),
            0,
            "a dropped owner's row leaves the snapshot"
        );
        assert_eq!(ledger.tenants(), 0);
    }

    #[test]
    fn the_line_names_the_quota_and_every_tenant() {
        let ledger = Ledger::new();
        ledger.set_quota(100 << 20);
        let g = Gauge::new(
            "experts",
            Claim {
                floor: 8 << 20,
                want: 80 << 20,
                unit: 1 << 20,
                class: Class::Cache,
            },
            40 << 20,
        );
        let g_dyn: Arc<dyn Tenant> = g.clone();
        let _id = ledger.register(&g_dyn);
        let _r = ledger.reserve(40 << 20);
        let line = ledger.snapshot().line();
        assert!(
            line.starts_with("quota=100.0 MiB committed=40.0 MiB"),
            "{line}"
        );
        assert!(line.contains("reclaims=0/0.0 MiB"), "{line}");
        assert!(
            line.contains("experts:cache floor=8.0 want=80.0 held=40.0"),
            "{line}"
        );
    }
}

#[cfg(all(test, feature = "metal"))]
mod device_tests {
    //! The device's persistent paths commit their bytes for the handle's
    //! life (`MetalDevice::new_persistent_*`, `new_unpooled_shared`): the
    //! committed figure rises with the allocation and falls with its drop.
    use crate::Device;

    #[test]
    fn a_persistent_buffer_is_committed_until_its_handle_drops() {
        let Ok(Device::Metal(m)) = Device::new_metal(0) else {
            return;
        };
        let ledger = m.ledger().clone();
        let before = ledger.committed();
        let a = m.new_persistent_private(1 << 20).unwrap();
        let b = m.new_persistent_shared(2 << 20).unwrap();
        let c = m.new_unpooled_shared(4 << 20).unwrap();
        assert_eq!(
            ledger.committed() - before,
            7 << 20,
            "three persistent allocations commit their bytes"
        );
        drop(b);
        assert_eq!(
            ledger.committed() - before,
            5 << 20,
            "a dropped handle releases"
        );
        drop(a);
        drop(c);
        assert_eq!(ledger.committed(), before, "all released");
    }
}

#[cfg(all(test, feature = "metal"))]
mod admit_tests {
    //! A pass-level admission that would refuse gives the pools' free lists
    //! back first and decides on the figure after (`MetalDevice::admit_bytes`).
    use crate::Device;

    #[test]
    fn a_refusal_reclaims_the_free_lists_before_it_stands() {
        let Ok(Device::Metal(m)) = Device::new_metal(0) else {
            return;
        };
        let mib = 1usize << 20;
        // 32 MiB through the Shared pool, released: parked, then free once
        // the epoch they were released in completes.
        let held: Vec<_> = (0..32).map(|_| m.allocate_buffer(mib).unwrap()).collect();
        drop(held);
        let clock = m.gpu_clock();
        let e = clock.commit_epoch();
        clock.mark_completed(e);
        // The ledger's figure: a pool buffer's reservation rides its node,
        // live or free, so the 32 free MiB are in it.
        let committed = m.ledger().committed();
        // A budget 8 MiB short for `need` on the first figure, and 24 MiB
        // over it once the 32 MiB of free lists are gone.
        let need = 4 * mib;
        let budget = committed + need - 8 * mib;
        let before = m.ledger().reclaims();
        let a = m.admit_bytes(need, budget);
        assert!(
            a.reclaimed >= 32 * mib,
            "the free lists held 32 MiB; reclaimed {}",
            a.reclaimed
        );
        assert!(
            a.committed + need <= budget,
            "after the reclaim the pass fits: committed {} + need {} vs budget {}",
            a.committed,
            need,
            budget
        );
        assert!(a.fits, "{a:?}");
        assert!(
            a.committed + 32 * mib <= committed,
            "the trim released the free buffers' reservations: {} after, {committed} before",
            a.committed
        );
        let after = m.ledger().reclaims();
        assert_eq!(after.0, before.0 + 1, "one reclaim counted");
        assert!(after.1 >= before.1 + 32 * mib);
        // A pass that fits on the first figure reclaims nothing.
        let a = m.admit_bytes(need, m.ledger().committed() + need);
        assert_eq!(a.reclaimed, 0);
        assert!(a.fits);
        assert_eq!(m.ledger().reclaims().0, after.0);
    }
    /// The second rung: buffers released while their command buffer is still
    /// open sit in the pending lists, out of the first reclaim's reach; the
    /// admission waits the queue out once and gives them back too.
    /// The admission's second rung is a process-wide knob
    /// (`set_admission_drain`): the two tests that read it hold this lock
    /// (libtest runs tests on threads; one flipping the knob under the
    /// other was the first red).
    static DRAIN_KNOB: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

    #[test]
    fn a_refusal_drains_the_pending_lists_after_one_wait() {
        let _knob = DRAIN_KNOB.lock();
        crate::metal_backend::set_admission_drain(true);
        let Ok(dev @ Device::Metal(_)) = Device::new_metal(0) else {
            return;
        };
        let Device::Metal(m) = &dev else {
            unreachable!()
        };
        let mib = 1usize << 20;
        // 32 MiB of fills encoded into the open command buffer, then dropped:
        // pending against an epoch nothing has committed.
        let held: Vec<_> = (0..32)
            .map(|_| crate::Tensor::zeros(mib / 4, crate::DType::F32, &dev).unwrap())
            .collect();
        drop(held);
        // The ledger's figure: a pool buffer's reservation rides its node,
        // live or free, so the 32 free MiB are in it.
        let committed = m.ledger().committed();
        let need = 4 * mib;
        let budget = committed + need - 8 * mib;
        let a = m.admit_bytes(need, budget);
        assert!(
            a.reclaimed >= 32 * mib,
            "the pending lists held 32 MiB behind an open epoch; reclaimed {}",
            a.reclaimed
        );
        assert!(a.fits && a.drained, "{a:?}");
    }

    /// A pool buffer is committed for as long as it exists, live or free:
    /// through the node going back to the free list (the common release)
    /// and through a `PooledBuffer` released from outside its node (a weak
    /// handle outstanding) alike; the trim that destroys it releases it.
    /// The arm without the second rung (`--pool-no-admission-drain`): the
    /// same 32 MiB behind an open epoch is NOT given back, the refusal
    /// stands, and the ledger counted no drain; the rung back on, the twin
    /// above holds and the count moves. The default is the rung on.
    #[test]
    fn the_second_rung_off_leaves_the_pending_lists_and_refuses() {
        let _knob = DRAIN_KNOB.lock();
        let Ok(dev @ Device::Metal(_)) = Device::new_metal(0) else {
            return;
        };
        let Device::Metal(m) = &dev else {
            unreachable!()
        };
        let mib = 1usize << 20;
        let drains = m.ledger().drains();
        crate::metal_backend::set_admission_drain(false);
        let held: Vec<_> = (0..32)
            .map(|_| crate::Tensor::zeros(mib / 4, crate::DType::F32, &dev).unwrap())
            .collect();
        drop(held);
        let committed = m.ledger().committed();
        let need = 4 * mib;
        let budget = committed + need - 8 * mib;
        let a = m.admit_bytes(need, budget);
        crate::metal_backend::set_admission_drain(true);
        assert!(!a.drained, "{a:?}");
        assert_eq!(m.ledger().drains(), drains, "no drain counted");
        assert!(
            a.reclaimed < 32 * mib && !a.fits,
            "the rung off: the pending lists stay behind their epoch and the refusal stands; {a:?}"
        );
        // The rung on again: the same shape is given back and counted.
        let held: Vec<_> = (0..32)
            .map(|_| crate::Tensor::zeros(mib / 4, crate::DType::F32, &dev).unwrap())
            .collect();
        drop(held);
        let committed = m.ledger().committed();
        let budget = committed + need - 8 * mib;
        let a = m.admit_bytes(need, budget);
        assert!(a.drained && a.fits, "{a:?}");
        assert_eq!(m.ledger().drains(), drains + 1);
    }

    #[test]
    fn a_free_buffer_stays_committed_until_the_pool_destroys_it() {
        let Ok(Device::Metal(m)) = Device::new_metal(0) else {
            return;
        };
        let mib = 1usize << 20;
        let clock = m.gpu_clock();
        let before = m.ledger().committed();
        let a = m.allocate_buffer(mib).unwrap();
        let b = m.allocate_buffer(mib).unwrap();
        let held = m.ledger().committed();
        assert!(held >= before + 2 * mib, "{held} < {before} + 2 MiB");
        // `a`'s node goes back whole; `b`'s value is released from outside
        // its node, a weak handle keeping the node.
        let weak = crate::kernels::metal::BufferHandle::downgrade(&b);
        drop(a);
        drop(b);
        let e = clock.commit_epoch();
        clock.mark_completed(e);
        assert_eq!(
            m.ledger().committed(),
            held,
            "a free buffer left the ledger before the pool destroyed it"
        );
        assert!(
            weak.upgrade().is_none(),
            "the released buffer is not a handle"
        );
        drop(weak);
        let freed = m.trim_unused_buffers();
        assert!(freed >= 2 * mib, "the trim freed {freed}");
        assert!(
            m.ledger().committed() + 2 * mib <= held,
            "the trim did not release the free buffers' reservations: {} vs {held}",
            m.ledger().committed()
        );
    }

    /// The verdict is arithmetic on the ledger, not a read of the device:
    /// bytes Metal holds that no backing handed out (here a raw buffer made
    /// beside the backings) do not enter it, where `currentAllocatedSize`
    /// counts them — so two processes with the same allocation history give
    /// the same verdict (`profiles-and-memory-design.md` §11).
    #[test]
    fn the_verdict_is_the_ledgers_not_the_devices() {
        let Ok(Device::Metal(m)) = Device::new_metal(0) else {
            return;
        };
        let mib = 1usize << 20;
        let need = 4 * mib;
        // A persistent buffer is on the ledger: it moves the verdict.
        let committed = m.ledger().committed();
        let held = m.new_persistent_private(64 * mib).unwrap();
        assert_eq!(m.ledger().committed(), committed + 64 * mib);
        let budget = m.ledger().committed() + need + mib;
        assert!(m.admit_bytes(need, budget).fits);
        // A raw Metal buffer beside the backings is not: the device's figure
        // grows by 256 MiB, the ledger's by nothing, and the verdict stands.
        let raw = m
            .device()
            .new_buffer(256 * mib, crate::kernels::RESOURCE_OPTIONS)
            .unwrap();
        assert!(
            m.device().current_allocated_size() >= committed + 64 * mib + 256 * mib,
            "the device counts the raw buffer"
        );
        let a = m.admit_bytes(need, budget);
        assert_eq!(a.committed, committed + 64 * mib);
        assert_eq!(
            a.reclaimed, 0,
            "nothing to reclaim: the verdict needed no rung"
        );
        assert!(a.fits, "{a:?}");
        drop(raw);
        drop(held);
        assert_eq!(m.ledger().committed(), committed);
    }

    /// Past what the load reserved on the machine, a pass that fits the
    /// quota widens the reservation through the machine gate, or is refused
    /// with the gate's arithmetic; inside it nothing is asked. The
    /// non-vacuity arm: with no reservation attached, the same pass fits.
    #[test]
    fn a_pass_past_the_machine_reservation_widens_it_or_is_refused() {
        use crate::machine::{Gate, GateConfig, Peak};
        let Ok(Device::Metal(m)) = Device::new_metal(0) else {
            return;
        };
        let gb = 1_000_000_000usize;
        let budget = usize::MAX;
        assert!(
            m.admit_bytes(20 * gb, budget).fits,
            "no reservation attached: the machine is not asked"
        );
        let dir = std::env::temp_dir().join(format!(
            "lloom-ledger-machine-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // 10 GB available, nothing kept: the room is 10 GB less what this
        // process has reserved and does not hold yet (its footprint is far
        // under a gigabyte, so the arms below hold whatever it is).
        let gate = Gate::new(GateConfig {
            reserve: Some(0),
            registry: Some(dir.clone()),
            available: Some(10 * gb as u64),
            total: Some(64 * gb as u64),
            ..Default::default()
        });
        let held = gate
            .admit("ledger test", &Peak::new().with("weights", gb as u64))
            .unwrap()
            .holding
            .unwrap();
        m.ledger().attach_machine(&held);
        assert_eq!(m.ledger().machine_reserved(), Some(gb as u64));
        let committed = m.ledger().committed();
        // Inside the reservation: admitted, the reservation unmoved.
        let a = m.admit_bytes(gb / 2 - committed.min(gb / 2), budget);
        assert!(a.fits && a.machine.is_none(), "{a:?}");
        assert_eq!(held.reserved(), gb as u64);
        // Past it, with room: admitted, the reservation widened to the pass.
        let a = m.admit_bytes(2 * gb, budget);
        assert!(a.fits && a.machine.is_none(), "{a:?}");
        assert_eq!(held.reserved(), (committed + 2 * gb) as u64);
        // Past the room: refused with the gate's arithmetic, unmoved.
        let a = m.admit_bytes(20 * gb, budget);
        assert!(!a.fits, "{a:?}");
        let why = a.machine.expect("the machine's refusal");
        assert!(
            why.contains("the machine cannot hold this now") && why.contains("of growth"),
            "{why}"
        );
        assert_eq!(held.reserved(), (committed + 2 * gb) as u64);
        drop(held);
        assert_eq!(m.ledger().machine_reserved(), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A pass past the reservation on a machine that cannot widen it gives
    /// back what the process holds and does not need — buffers its earlier
    /// passes released — and is admitted inside the reservation, unwidened,
    /// rather than refused on bytes it would have reused.
    #[test]
    fn a_pass_the_machine_refuses_is_asked_again_after_the_reclaim() {
        use crate::machine::{Gate, GateConfig, Peak};
        let _knob = DRAIN_KNOB.lock();
        crate::metal_backend::set_admission_drain(true);
        let Ok(Device::Metal(m)) = Device::new_metal(0) else {
            return;
        };
        let mib = 1usize << 20;
        let dir = std::env::temp_dir().join(format!(
            "lloom-ledger-machine-again-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        // Reserved: what the process holds now, the pass's 32 MiB and 8 MiB
        // of slack, on a machine with exactly that available and nothing kept. An injected
        // reading does not fall as the process allocates, so the room to
        // widen it stays the reservation itself: smaller than the earlier
        // pass's buffers below, which the first figure counts.
        let need = 32 * mib;
        let committed = m.ledger().committed();
        let reserved = committed + need + 8 * mib;
        let released = (reserved + 32 * mib).next_multiple_of(4 * mib);
        assert!(
            released <= 256 * mib,
            "a fresh device commits {committed} bytes: the released buffers would pass the free list's cap"
        );
        let gate = Gate::new(GateConfig {
            reserve: Some(0),
            registry: Some(dir.clone()),
            available: Some(reserved as u64),
            total: Some(reserved as u64),
            ..Default::default()
        });
        let held = gate
            .admit("ledger test", &Peak::new().with("weights", reserved as u64))
            .unwrap()
            .holding
            .unwrap();
        m.ledger().attach_machine(&held);
        // An earlier pass's buffers, released: free once its epoch completes.
        let bufs: Vec<_> = (0..released / (4 * mib))
            .map(|_| m.allocate_buffer(4 * mib).unwrap())
            .collect();
        drop(bufs);
        let clock = m.gpu_clock();
        let e = clock.commit_epoch();
        clock.mark_completed(e);
        assert!(m.ledger().committed() >= committed + released);
        let a = m.admit_bytes(need, usize::MAX);
        assert!(a.fits && a.machine.is_none(), "{a:?}");
        assert!(
            a.reclaimed >= released,
            "the released buffers went back: {a:?}"
        );
        assert!(a.committed + need <= reserved, "{a:?}");
        assert_eq!(
            held.reserved(),
            reserved as u64,
            "admitted inside it, unwidened"
        );
        // Past the reservation with nothing to give back: refused.
        let a = m.admit_bytes(2 * reserved, usize::MAX);
        assert!(!a.fits, "{a:?}");
        assert!(
            a.machine
                .as_deref()
                .is_some_and(|w| w.contains("the machine cannot hold this now")),
            "{a:?}"
        );
        assert_eq!(held.reserved(), reserved as u64);
        drop(held);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
