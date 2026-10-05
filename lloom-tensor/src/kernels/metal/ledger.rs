//! L1 of the memory system (`profiles-and-memory-design.md` §4.1, §4.2): the
//! ledger. Integers and atomics; `currentAllocatedSize` is a gauge read beside
//! it, never the thing a decision is made on (§6.3d: accounting is cheap,
//! Metal calls are not).
//!
//! Two things live here. A [`Reservation`] is bytes a persistent allocation
//! holds for its lifetime — the slot pools, the arena base, the pass-plan
//! base, the heap chunks, every unpooled buffer — released when it drops, so
//! the committed figure cannot drift from what exists. A [`Tenant`] is a
//! holder that can say what it needs and wants (a [`Claim`]) and what it
//! holds now; the registry lists them so a report, and later the arbiter
//! (A5), sees every claim in one place.
//!
//! **Accounting only, for now.** `reserve` never refuses and no claim is
//! granted or shrunk here; the per-pass admission still reads the device
//! (`Cache::admit_grow`, `prepare_pass_as`). Reservations as the admission
//! mechanism are A4; the arbiter over the claims is A5. What this commit
//! buys is the figure beside `currentAllocatedSize` that tells committed from
//! elastic bytes (§1.1 #3) and the claims written down.

use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Weak};

/// What a tenant's memory is to the process (§2 principle 5).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Class {
    /// Chosen by the caller (a context, a row count, a tile): refused when
    /// it does not fit, never silently reduced.
    Capability,
    /// Shrinks under pressure and refills later (expert slots, free lists,
    /// quarantines, staging spares).
    Cache,
    /// Per-pass transients.
    Scratch,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Capability => "capability",
            Class::Cache => "cache",
            Class::Scratch => "scratch",
        }
    }
}

/// A tenant's claim on the quota, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Claim {
    /// Below this the tenant cannot work; a capability's floor is a refusal.
    pub floor: usize,
    /// What it would use if memory were free.
    pub want: usize,
    /// It grows and shrinks in these steps.
    pub unit: usize,
    pub class: Class,
}

/// A holder of memory the ledger can ask about. Reads must be cheap: the
/// snapshot runs on a report, never on a step, but a tenant's `claim` and
/// `held` must not take a lock a step holds.
pub trait Tenant: Send + Sync {
    fn name(&self) -> String;
    fn claim(&self) -> Claim;
    /// Bytes held now.
    fn held(&self) -> usize;
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

/// A registered tenant's handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TenantId(pub u32);

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

/// One tenant's row in a snapshot.
#[derive(Clone, Debug)]
pub struct TenantRow {
    pub id: TenantId,
    pub name: String,
    pub claim: Claim,
    pub held: usize,
}

/// The ledger at one instant.
#[derive(Clone, Debug, Default)]
pub struct LedgerSnapshot {
    /// `fraction × recommendedMaxWorkingSetSize`; 0 until the binary sets it.
    pub quota: usize,
    /// Bytes under live reservations.
    pub committed: usize,
    /// Pass-level admissions that reclaimed the free lists, and their bytes.
    pub reclaims: (u32, usize),
    pub tenants: Vec<TenantRow>,
}

impl LedgerSnapshot {
    /// The sum of the tenants' floors and wants.
    pub fn floors(&self) -> usize {
        self.tenants.iter().map(|t| t.claim.floor).sum()
    }

    pub fn wants(&self) -> usize {
        self.tenants.iter().map(|t| t.claim.want).sum()
    }

    /// One line for a report: the quota, the committed bytes, then each
    /// tenant as `name:class floor/want/held` in MiB.
    pub fn line(&self) -> String {
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        let mut s = format!(
            "quota={:.1} MiB committed={:.1} MiB floors={:.1} MiB wants={:.1} MiB reclaims={}/{:.1} MiB",
            mib(self.quota),
            mib(self.committed),
            mib(self.floors()),
            mib(self.wants()),
            self.reclaims.0,
            mib(self.reclaims.1)
        );
        for t in &self.tenants {
            s.push_str(&format!(
                " {}:{} floor={:.1} want={:.1} held={:.1}",
                t.name,
                t.claim.class.name(),
                mib(t.claim.floor),
                mib(t.claim.want),
                mib(t.held)
            ));
        }
        s
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
    next_id: AtomicU32,
    tenants: parking_lot::Mutex<Vec<(TenantId, Weak<dyn Tenant>)>>,
}

impl Default for Ledger {
    fn default() -> Self {
        Self {
            quota: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
            reclaims: AtomicU32::new(0),
            reclaimed_bytes: AtomicUsize::new(0),
            next_id: AtomicU32::new(1),
            tenants: parking_lot::Mutex::new(Vec::new()),
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

    /// Ask the tenants that yield — `Scratch` first, then `Cache`, never a
    /// `Capability` — to shrink by units until `bytes` are pledged or none
    /// can give more. A pledge is what a tenant reports holding less after
    /// `resize`; a tenant that frees at once (drops reservations) moves
    /// `committed` here, one whose owner applies the grant later (the
    /// expert tier reallocating its pools) moves it then. Returns the
    /// bytes pledged.
    pub fn yield_for(&self, bytes: usize) -> usize {
        let mut pledged = 0usize;
        let tenants = self.live_tenants();
        for class in [Class::Scratch, Class::Cache] {
            for (_, t) in tenants.iter().filter(|(_, t)| t.claim().class == class) {
                loop {
                    if pledged >= bytes {
                        return pledged;
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
                    pledged += held - after;
                }
            }
        }
        pledged
    }

    /// A pass-level admission gave the free lists back: count it.
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
    /// here: admission against the quota is the pass's (A4), not the
    /// allocation's.
    pub fn reserve(self: &Arc<Self>, bytes: usize) -> Reservation {
        self.committed.fetch_add(bytes, Ordering::AcqRel);
        Reservation {
            ledger: Arc::clone(self),
            bytes,
        }
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grant {
    pub id: TenantId,
    pub bytes: usize,
}

/// What the arbiter decided (`profiles-and-memory-design.md` §4.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arbitration {
    /// Every tenant's grant: at least its floor, at most its want.
    Granted(Vec<Grant>),
    /// The floors alone exceed the quota: the arithmetic, for the refusal's
    /// message (§9.5g's shape). Only a `Capability` floor can cause this.
    Refused { floors: usize, quota: usize },
}

/// The arbiter's arithmetic, over a snapshot of the tenants' claims:
/// 1. floors first — if Σ floors > quota, refuse with the arithmetic;
/// 2. if Σ wants ≤ quota, grant every want (nothing shrinks);
/// 3. else fill upward from the floors, a unit at a time, to the tenant
///    whose next unit is worth most — by class order (`Capability`, then
///    `Cache`, then `Scratch`), the tenants of a class in registration
///    order; a tenant whose next unit would pass its want, or the quota,
///    is skipped. (`marginal_value` is §14 Q2's later hook.)
///
/// Pure: no tenant is resized here; the ledger's [`Ledger::arbitrate`]
/// applies the grants that moved. Deterministic in the snapshot's order.
pub fn arbitrate(quota: usize, tenants: &[TenantRow]) -> Arbitration {
    let floors: usize = tenants.iter().map(|t| t.claim.floor).sum();
    if floors > quota {
        return Arbitration::Refused { floors, quota };
    }
    let wants: usize = tenants
        .iter()
        .map(|t| t.claim.want.max(t.claim.floor))
        .sum();
    if wants <= quota {
        return Arbitration::Granted(
            tenants
                .iter()
                .map(|t| Grant {
                    id: t.id,
                    bytes: t.claim.want.max(t.claim.floor),
                })
                .collect(),
        );
    }
    let mut grants: Vec<Grant> = tenants
        .iter()
        .map(|t| Grant {
            id: t.id,
            bytes: t.claim.floor,
        })
        .collect();
    let mut left = quota - floors;
    // Class order, registration order within a class.
    let mut order: Vec<usize> = (0..tenants.len()).collect();
    order.sort_by_key(|&i| match tenants[i].claim.class {
        Class::Capability => 0u8,
        Class::Cache => 1,
        Class::Scratch => 2,
    });
    // Each unit to the first tenant in class order with room for it: the
    // capability fills to its want before the cache takes a unit, the
    // cache before the scratch.
    loop {
        let next = order.iter().copied().find(|&i| {
            let t = &tenants[i];
            let unit = t.claim.unit.max(1);
            let want = t.claim.want.max(t.claim.floor);
            grants[i].bytes + unit <= want && unit <= left
        });
        let Some(i) = next else {
            break;
        };
        let unit = tenants[i].claim.unit.max(1);
        grants[i].bytes += unit;
        left -= unit;
    }
    Arbitration::Granted(grants)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(id: u32, class: Class, floor: usize, want: usize, unit: usize) -> TenantRow {
        TenantRow {
            id: TenantId(id),
            name: format!("t{id}"),
            claim: Claim {
                floor,
                want,
                unit,
                class,
            },
            held: 0,
        }
    }

    #[test]
    fn the_arbiter_refuses_on_the_floors_grants_every_want_that_fits_and_fills_by_class() {
        let mib = 1usize << 20;
        let kv = row(1, Class::Capability, 64 * mib, 136 * mib, 4 * mib);
        let experts = row(2, Class::Cache, 80 * mib, 400 * mib, 8 * mib);
        let scratch = row(3, Class::Scratch, 0, 32 * mib, 1 * mib);
        // 1. The floors alone exceed the quota: refused with the arithmetic.
        assert_eq!(
            arbitrate(100 * mib, &[kv.clone(), experts.clone()]),
            Arbitration::Refused {
                floors: 144 * mib,
                quota: 100 * mib
            }
        );
        // 2. Every want fits: every want granted, nothing shrinks.
        assert_eq!(
            arbitrate(1000 * mib, &[kv.clone(), experts.clone(), scratch.clone()]),
            Arbitration::Granted(vec![
                Grant {
                    id: TenantId(1),
                    bytes: 136 * mib
                },
                Grant {
                    id: TenantId(2),
                    bytes: 400 * mib
                },
                Grant {
                    id: TenantId(3),
                    bytes: 32 * mib
                },
            ])
        );
        // 3. Between: the floors, then units upward by class — the
        // capability fills first, the cache next, the scratch last; a
        // tenant at its want is skipped; the sum never passes the quota.
        let quota = 144 * mib + 100 * mib;
        let Arbitration::Granted(g) =
            arbitrate(quota, &[scratch.clone(), experts.clone(), kv.clone()])
        else {
            panic!("fits above the floors")
        };
        let by = |id: u32| g.iter().find(|x| x.id == TenantId(id)).unwrap().bytes;
        assert_eq!(by(1), 136 * mib, "the capability reached its want");
        assert!(
            by(2) >= 80 * mib && by(2) < 400 * mib,
            "the cache got the rest: {}",
            by(2)
        );
        assert!(by(3) < 32 * mib, "the scratch got what was left: {}", by(3));
        let total: usize = g.iter().map(|x| x.bytes).sum();
        assert!(
            total <= quota && total > quota - 8 * mib,
            "{total} vs {quota}"
        );
        assert_eq!(by(1) + by(2) + by(3), total);
        // Registration order within a class decides between equals.
        let a = row(4, Class::Cache, 0, 10 * mib, 10 * mib);
        let b = row(5, Class::Cache, 0, 10 * mib, 10 * mib);
        assert_eq!(
            arbitrate(10 * mib, &[a, b]),
            Arbitration::Granted(vec![
                Grant {
                    id: TenantId(4),
                    bytes: 10 * mib
                },
                Grant {
                    id: TenantId(5),
                    bytes: 0
                },
            ])
        );
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

#[cfg(test)]
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

#[cfg(test)]
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
    #[test]
    fn a_refusal_drains_the_pending_lists_after_one_wait() {
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
        assert!(a.fits, "{a:?}");
    }

    /// A pool buffer is committed for as long as it exists, live or free:
    /// through the node going back to the free list (the common release)
    /// and through a `PooledBuffer` released from outside its node (a weak
    /// handle outstanding) alike; the trim that destroys it releases it.
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
}
