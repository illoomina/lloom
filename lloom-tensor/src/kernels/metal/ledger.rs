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
        self.ledger.committed.fetch_sub(self.bytes, Ordering::AcqRel);
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
            "quota={:.1} MiB committed={:.1} MiB floors={:.1} MiB wants={:.1} MiB",
            mib(self.quota),
            mib(self.committed),
            mib(self.floors()),
            mib(self.wants())
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
    next_id: AtomicU32,
    tenants: parking_lot::Mutex<Vec<(TenantId, Weak<dyn Tenant>)>>,
}

impl Default for Ledger {
    fn default() -> Self {
        Self {
            quota: AtomicUsize::new(0),
            committed: AtomicUsize::new(0),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reservation_is_committed_for_its_life_and_released_on_drop() {
        let ledger = Ledger::new();
        assert_eq!(ledger.committed(), 0);
        let a = ledger.reserve(1 << 20);
        let b = ledger.reserve(3 << 20);
        assert_eq!(ledger.committed(), 4 << 20, "two live reservations sum");
        drop(a);
        assert_eq!(ledger.committed(), 3 << 20, "a dropped reservation is released");
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
        assert!(line.starts_with("quota=100.0 MiB committed=40.0 MiB"), "{line}");
        assert!(line.contains("experts:cache floor=8.0 want=80.0 held=40.0"), "{line}");
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
        assert_eq!(ledger.committed() - before, 5 << 20, "a dropped handle releases");
        drop(a);
        drop(c);
        assert_eq!(ledger.committed(), before, "all released");
    }
}
