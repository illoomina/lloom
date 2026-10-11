//! One tenant for every weight-group tier (`profiles-and-memory-design.md`
//! §4.8 "As a tenant", #892 A6b): the routed experts' pools and the dense
//! layers' slots claim the ledger the same way — a `Cache` whose floor is a
//! ring of units, whose want is every group resident, whose unit is one
//! slot's bytes — and yield the same way: the arbiter's grant is recorded
//! here, and the tier's owner applies it at a boundary of its own (a pass,
//! a turn), never inside one. The expert tier's `ExpertTenant` and the
//! dense tier's gauge were two copies of this; this is the one.
//!
//! `held` is the grant as recorded, not what the owner runs yet: the
//! arbiter's yield (`Ledger::yield_for`) asks a tenant down one unit at a
//! time and reads `held` back after each `resize`, so a `held` that waited
//! for the owner to apply would be asked for the same unit again and again
//! — a pledge of many units the owner then meets with one, and the pass
//! refused on the figure after (the Qwen long prompt at `--slots auto`,
//! `gates-a6b/long-auto-branch.txt`: the first form of this did that).

use lloom_tensor::Result;
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub struct GroupTenant {
    name: String,
    /// One slot's bytes: the claim's `unit`.
    unit: usize,
    /// The ring: below this the tier cannot work.
    floor_units: usize,
    /// Every group resident: the want.
    want_units: usize,
    /// The most units the owner can run without reallocating: the expert
    /// tier's pool (its slots), the dense tier's want (it reallocates).
    max_units: usize,
    /// The arbiter's last grant, in units — `held`; the owner applies it.
    wanted: AtomicUsize,
}

impl GroupTenant {
    /// `units` in service now, `floor_units` the ring, `want_units` the
    /// groups, `max_units` the most the owner can run without reallocating
    /// (a grant never exceeds it).
    pub fn new(
        name: impl Into<String>,
        unit: usize,
        units: usize,
        floor_units: usize,
        want_units: usize,
        max_units: usize,
    ) -> Result<Self> {
        let floor_units = floor_units.min(units.max(1));
        let max_units = max_units.max(units.max(1));
        // These bounds cover both the initial holding and every later grant.
        // Keep zero-byte units valid: their byte products are all zero.
        for (kind, count) in [
            ("floor", floor_units),
            ("want", want_units),
            ("maximum", max_units),
            ("initial", units),
        ] {
            if unit.checked_mul(count).is_none() {
                lloom_tensor::bail!(
                    "group tenant: {kind} {count} units of {unit} bytes exceeds usize"
                )
            }
        }
        Ok(Self {
            name: name.into(),
            unit,
            floor_units,
            want_units,
            max_units,
            wanted: AtomicUsize::new(units),
        })
    }

    /// Exact atomic pledged delta with bounded retries. Contention leaves a
    /// competing grant intact and is reported separately from no available yield.
    pub fn yield_bulk(&self, bytes: usize) -> lloom_tensor::memory_arbitration::BulkYield {
        self.yield_bulk_with(bytes, || {})
    }

    fn yield_bulk_with(
        &self,
        bytes: usize,
        mut before_cas: impl FnMut(),
    ) -> lloom_tensor::memory_arbitration::BulkYield {
        use lloom_tensor::memory_arbitration::BulkYield;
        if self.unit == 0 || bytes == 0 {
            return BulkYield::Pledged { bytes: 0 };
        }
        let requested = bytes.div_ceil(self.unit);
        let mut held = self.wanted.load(Ordering::Relaxed);
        for _ in 0..4 {
            let units = requested.min(held.saturating_sub(self.floor_units));
            if units == 0 {
                return BulkYield::Pledged { bytes: 0 };
            }
            let top = self.max_units.min(self.want_units).max(self.floor_units);
            let target = (held - units).min(top);
            before_cas();
            match self
                .wanted
                .compare_exchange(held, target, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => {
                    return BulkYield::Pledged {
                        bytes: (held - target) * self.unit,
                    }
                }
                Err(now) => held = now,
            }
        }
        BulkYield::Contended
    }

    pub fn unit(&self) -> usize {
        self.unit
    }

    pub fn floor_units(&self) -> usize {
        self.floor_units
    }

    pub fn max_units(&self) -> usize {
        self.max_units
    }

    /// The arbiter's grant, in units: what the owner should be running.
    pub fn wanted_units(&self) -> usize {
        self.wanted.load(Ordering::Relaxed)
    }
}

#[cfg(feature = "metal")]
impl lloom_tensor::kernels::metal::ledger::Tenant for GroupTenant {
    fn name(&self) -> String {
        self.name.clone()
    }
    fn claim(&self) -> lloom_tensor::kernels::metal::ledger::Claim {
        use lloom_tensor::kernels::metal::ledger::{Claim, Class};
        Claim {
            floor: self.floor_units * self.unit,
            want: self.want_units * self.unit,
            unit: self.unit,
            class: Class::Cache,
        }
    }
    fn held(&self) -> usize {
        self.wanted_units() * self.unit
    }
    fn yield_bulk(&self, bytes: usize) -> lloom_tensor::memory_arbitration::BulkYield {
        GroupTenant::yield_bulk(self, bytes)
    }
    /// Record the grant as units, between the floor and the most the owner
    /// can run; the bytes those units are, back to the arbiter. Nothing
    /// moves here.
    fn resize(&self, grant: usize) -> usize {
        let top = self.max_units.min(self.want_units).max(self.floor_units);
        let units = (grant / self.unit.max(1)).clamp(self.floor_units, top);
        self.wanted.store(units, Ordering::Relaxed);
        units * self.unit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_bulk_yield_is_bounded_and_preserves_competing_grants() {
        use lloom_tensor::memory_arbitration::BulkYield;
        let tenant = GroupTenant::new("bulk", 100, 8, 3, 30, 10).unwrap();
        assert_eq!(tenant.yield_bulk(150), BulkYield::Pledged { bytes: 200 });
        assert_eq!(tenant.wanted_units(), 6);
        assert_eq!(
            tenant.yield_bulk(usize::MAX),
            BulkYield::Pledged { bytes: 300 }
        );
        assert_eq!(tenant.yield_bulk(1), BulkYield::Pledged { bytes: 0 });
        let zero = GroupTenant::new("zero", 0, usize::MAX, 0, usize::MAX, usize::MAX).unwrap();
        assert_eq!(zero.yield_bulk(usize::MAX), BulkYield::Pledged { bytes: 0 });
        let huge = GroupTenant::new("huge", 1, usize::MAX, 0, usize::MAX, usize::MAX).unwrap();
        assert_eq!(
            huge.yield_bulk(usize::MAX),
            BulkYield::Pledged { bytes: usize::MAX }
        );
        let low_want = GroupTenant::new("low-want", 100, 8, 1, 3, 10).unwrap();
        assert_eq!(low_want.yield_bulk(100), BulkYield::Pledged { bytes: 500 });
        assert_eq!(low_want.wanted_units(), 3, "preserve resize's want clamp");
        let tenant = GroupTenant::new("race", 100, 8, 3, 30, 10).unwrap();
        let mut attempts = 0;
        let result = tenant.yield_bulk_with(100, || {
            attempts += 1;
            tenant
                .wanted
                .store(if attempts % 2 == 1 { 9 } else { 8 }, Ordering::Relaxed);
        });
        assert_eq!(result, BulkYield::Contended);
        assert_eq!(attempts, 4);
        assert_eq!(
            tenant.wanted_units(),
            8,
            "no failed CAS overwrote the competing grant"
        );
        let mut first = true;
        let result = tenant.yield_bulk_with(100, || {
            if first {
                tenant.wanted.store(10, Ordering::Relaxed);
                first = false;
            }
        });
        assert_eq!(result, BulkYield::Pledged { bytes: 100 });
        assert_eq!(
            tenant.wanted_units(),
            9,
            "successful CAS used the fresh grant, not stale eight"
        );
    }

    #[test]
    fn numeric_bounds_are_checked_after_normalization() {
        assert!(GroupTenant::new("floor", usize::MAX, 2, 2, 0, 0).is_err());
        assert!(GroupTenant::new("want", usize::MAX, 1, 0, 2, 1).is_err());
        assert!(GroupTenant::new("maximum", usize::MAX, 1, 0, 0, 2).is_err());
        // The normalized maximum also covers the initial holding.
        assert!(GroupTenant::new("initial", usize::MAX, 2, 0, 0, 0).is_err());
        let t = GroupTenant::new("normalized", usize::MAX, 0, usize::MAX, 1, 0).unwrap();
        assert_eq!(
            (t.floor_units(), t.max_units(), t.wanted_units()),
            (1, 1, 0)
        );
        let t = GroupTenant::new("zero", 0, usize::MAX, usize::MAX, usize::MAX, 0).unwrap();
        assert_eq!(
            (t.unit(), t.floor_units(), t.max_units(), t.wanted_units()),
            (0, usize::MAX, usize::MAX, usize::MAX)
        );
    }

    #[test]
    fn a_grant_is_recorded_in_units_between_the_floor_and_the_pool() {
        let t = GroupTenant::new("t", 100, 8, 3, 30, 10).unwrap();
        assert_eq!(
            (t.wanted_units(), t.floor_units(), t.max_units()),
            (8, 3, 10)
        );
        #[cfg(feature = "metal")]
        {
            use lloom_tensor::kernels::metal::ledger::Tenant;
            assert_eq!(t.claim().floor, 300);
            assert_eq!(t.claim().want, 3000);
            assert_eq!(t.held(), 800);
            assert_eq!(t.resize(550), 500, "rounded down to whole units");
            assert_eq!(t.wanted_units(), 5);
            assert_eq!(
                t.held(),
                500,
                "held follows the grant: the yield reads it back"
            );
            assert_eq!(t.resize(0), 300, "never below the floor");
            assert_eq!(
                t.resize(1 << 40),
                1000,
                "never above the pool, whatever the want"
            );
            let d = GroupTenant::new("dense", 100, 8, 3, 30, 30).unwrap();
            assert_eq!(
                d.resize(1 << 40),
                3000,
                "a tier that reallocates may grow to its want"
            );
        }
    }

    /// The yield's loop as `Ledger::yield_for` runs it: one unit at a time,
    /// `held` read back after each `resize`. A `held` that did not follow
    /// the grant would be asked for the same unit forever.
    #[test]
    #[cfg(feature = "metal")]
    fn the_yield_walks_down_one_unit_at_a_time_and_stops_at_the_floor() {
        use lloom_tensor::kernels::metal::ledger::Tenant;
        let t = GroupTenant::new("t", 100, 8, 3, 30, 8).unwrap();
        let mut pledged = 0;
        let mut rounds = 0;
        loop {
            let held = t.held();
            if held <= t.claim().floor {
                break;
            }
            let after = t.resize(held - 100);
            assert!(after < held, "round {rounds}: held {held}, after {after}");
            pledged += held - after;
            rounds += 1;
        }
        assert_eq!((rounds, pledged, t.wanted_units()), (5, 500, 3));
    }
}
