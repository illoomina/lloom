//! Backend-neutral production claim arbitration and snapshot reporting.
//!
//! Metal ledger reexports these types/functions; CPU tests execute the SAME
//! arbiter used by the loader and Ledger::arbitrate. No native allocations.
//! API: refusal floors and snapshot totals are u128, not saturating usize.

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

/// Optional bulk-yield protocol. Legacy preserves arbitrary per-unit resize
/// callbacks; Contended is a retryable observation, never a zero-byte pledge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BulkYield {
    Legacy,
    Pledged { bytes: usize },
    Contended,
}

/// Exact pledge diagnostics, not physical memory credit. Owners apply their
/// grants and admission re-reads committed bytes before deciding whether to fit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct YieldReport {
    pub pledged: u128,
    pub contended: usize,
}

impl YieldReport {
    pub fn needs_retry(self, fits: bool) -> bool {
        !fits && (self.pledged > 0 || self.contended > 0)
    }

    /// Preserve both attempts' diagnostics without hiding impossible totals.
    pub fn checked_merge(self, previous: Self) -> Option<Self> {
        Some(Self {
            pledged: self.pledged.checked_add(previous.pledged)?,
            contended: self.contended.checked_add(previous.contended)?,
        })
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

/// A registered tenant's handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TenantId(pub u32);

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
    /// Refusals that waited the queue out for the pending lists (the
    /// admission's second rung, `MetalDevice::admit_bytes`).
    pub drains: u32,
    pub tenants: Vec<TenantRow>,
}

impl LedgerSnapshot {
    /// The sum of the tenants' floors and wants.
    pub fn floors(&self) -> u128 {
        sum_bytes(&self.tenants, |c| c.floor)
    }

    pub fn wants(&self) -> u128 {
        sum_bytes(&self.tenants, |c| c.want)
    }

    /// One line for a report: the quota, the committed bytes, then each
    /// tenant as `name:class floor/want/held` in MiB.
    pub fn line(&self) -> String {
        let mib = |b: usize| b as f64 / (1024.0 * 1024.0);
        let mut s = format!(
            "quota={:.1} MiB committed={:.1} MiB floors={:.1} MiB wants={:.1} MiB reclaims={}/{:.1} MiB drains={}",
            mib(self.quota),
            mib(self.committed),
            self.floors() as f64 / (1024.0 * 1024.0),
            self.wants() as f64 / (1024.0 * 1024.0),
            self.reclaims.0,
            mib(self.reclaims.1),
            self.drains
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

// An addressable slice has at most usize::MAX elements, each byte value
// at most usize::MAX. On supported 32/64-bit targets their product fits u128.
// Keep checked addition so a future wider target cannot silently wrap.
fn sum_bytes(rows: &[TenantRow], value: impl Fn(&Claim) -> usize) -> u128 {
    rows.iter().fold(0u128, |sum, row| {
        sum.checked_add(value(&row.claim) as u128)
            .expect("addressable 32/64-bit claim slice fits u128")
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Grant {
    pub id: TenantId,
    pub bytes: usize,
}

/// What the arbiter decided (`profiles-and-memory-design.md` §4.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Arbitration {
    /// Every tenant's grant: between floor and max(want, floor).
    Granted(Vec<Grant>),
    /// The floors alone exceed the quota: the arithmetic, for the refusal's
    /// message (§9.5g's shape), without narrowing an oversized sum.
    Refused { floors: u128, quota: usize },
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
/// Pure: no tenant is resized here; the Metal ledger's `Ledger::arbitrate`
/// applies the grants that moved. Deterministic in the snapshot's order.
/// Bulk filling takes one step per sorted tenant, not per granted byte/unit.
pub fn arbitrate(quota: usize, tenants: &[TenantRow]) -> Arbitration {
    let floors = sum_bytes(tenants, |c| c.floor);
    if floors > quota as u128 {
        return Arbitration::Refused { floors, quota };
    }
    let wants = sum_bytes(tenants, |c| c.want.max(c.floor));
    if wants <= quota as u128 {
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
    // floors <= quota was established before this narrowing.
    let mut left = quota - floors as usize;
    // Class order, registration order within a class.
    let mut order: Vec<usize> = (0..tenants.len()).collect();
    order.sort_by_key(|&i| match tenants[i].claim.class {
        Class::Capability => 0u8,
        Class::Cache => 1,
        Class::Scratch => 2,
    });
    // The iterative rule always picks the FIRST eligible tenant. Exhaust its
    // fitting whole units in one step. Later tenants can only reduce `left`,
    // so this tenant can never become eligible again. Stable sort preserves
    // registration priority, and grants remain in original input order.
    for i in order {
        let claim = tenants[i].claim;
        let room = claim.want.max(claim.floor) - grants[i].bytes;
        let unit = claim.unit.max(1);
        let available = room.min(left);
        // Subtractive rounding avoids both grant+unit and count*unit overflow.
        let extra = available - available % unit;
        grants[i].bytes = grants[i]
            .bytes
            .checked_add(extra)
            .expect("extra is bounded by want minus grant");
        left -= extra;
    }
    Arbitration::Granted(grants)
}

/// Admission never treats an overflowing sum as equality at usize::MAX.
pub(crate) fn fits_budget(committed: usize, need: usize, budget: usize) -> bool {
    committed
        .checked_add(need)
        .is_some_and(|total| total <= budget)
}

/// A request to yield at most usize::MAX bytes, NOT an exact usage diagnostic.
/// Compute the shortfall wide first, so overflow cannot turn a refusal into a
/// zero-byte reclaim. The caller rechecks actual committed bytes afterward.
pub(crate) fn reclaim_target(committed: usize, need: usize, budget: usize) -> usize {
    let shortfall = (committed as u128 + need as u128).saturating_sub(budget as u128);
    shortfall.min(usize::MAX as u128) as usize
}

/// What the process would hold on the machine with `need` admitted: an
/// overflow refuses rather than wraps.
pub(crate) fn machine_target(
    committed: usize,
    need: usize,
) -> Result<u64, crate::memory_admission::AllocationError> {
    use crate::memory_admission::AllocationError;
    let total = committed
        .checked_add(need)
        .ok_or(AllocationError::Overflow)?;
    u64::try_from(total).map_err(|_| AllocationError::Overflow)
}

#[cfg(test)]
mod tests {
    #[test]
    fn numeric_yield_retry_and_merge_use_both_attempts() {
        let empty = super::YieldReport::default();
        let pledge = super::YieldReport {
            pledged: 3,
            contended: 0,
        };
        let contention = super::YieldReport {
            pledged: 0,
            contended: 2,
        };
        for report in [empty, pledge, contention] {
            assert!(!report.needs_retry(true));
        }
        assert!(!empty.needs_retry(false));
        assert!(pledge.needs_retry(false));
        assert!(contention.needs_retry(false));
        assert_eq!(
            pledge.checked_merge(contention),
            Some(super::YieldReport {
                pledged: 3,
                contended: 2
            })
        );
        assert!(super::YieldReport {
            pledged: u128::MAX,
            contended: 0
        }
        .checked_merge(pledge)
        .is_none());
        assert!(super::YieldReport {
            pledged: 0,
            contended: usize::MAX
        }
        .checked_merge(contention)
        .is_none());
    }

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
        let scratch = row(3, Class::Scratch, 0, 32 * mib, mib);
        // 1. The floors alone exceed the quota: refused with the arithmetic.
        assert_eq!(
            arbitrate(100 * mib, &[kv.clone(), experts.clone()]),
            Arbitration::Refused {
                floors: (144 * mib) as u128,
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

    fn iterative_oracle(quota: usize, tenants: &[TenantRow]) -> Arbitration {
        assert!(quota <= 16 && tenants.len() <= 3);
        assert!(tenants
            .iter()
            .all(|t| t.claim.floor <= 4 && t.claim.want <= 4 && t.claim.unit <= 4));
        let floors: usize = tenants.iter().map(|t| t.claim.floor).sum();
        if floors > quota {
            return Arbitration::Refused {
                floors: floors as u128,
                quota,
            };
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

    #[test]
    fn exhaustive_two_tenant_equivalence_to_bounded_incumbent() {
        let mut claims = Vec::new();
        for class in [Class::Capability, Class::Cache, Class::Scratch] {
            for floor in 0..=3 {
                for want in 0..=3 {
                    for unit in 0..=3 {
                        claims.push(Claim {
                            floor,
                            want,
                            unit,
                            class,
                        });
                    }
                }
            }
        }
        let mut checked = 0;
        for a in &claims {
            for b in &claims {
                let rows = [
                    TenantRow {
                        id: TenantId(7),
                        name: String::new(),
                        claim: *a,
                        held: 0,
                    },
                    TenantRow {
                        id: TenantId(2),
                        name: String::new(),
                        claim: *b,
                        held: 0,
                    },
                ];
                for quota in 0..=9 {
                    assert_eq!(
                        arbitrate(quota, &rows),
                        iterative_oracle(quota, &rows),
                        "quota={quota} rows={rows:?}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 368_640);
    }

    #[test]
    fn three_tenant_priorities_and_skippable_remainders() {
        // All class assignments, all registration permutations, all units
        // including zero. Numeric demand is bounded before calling the oracle.
        let classes = [Class::Capability, Class::Cache, Class::Scratch];
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        let mut checked = 0;
        for a in classes {
            for b in classes {
                for c in classes {
                    for ua in 0..=3 {
                        for ub in 0..=3 {
                            for uc in 0..=3 {
                                let rows = [
                                    row(1, a, 0, 4, ua),
                                    row(2, b, 1, 4, ub),
                                    row(3, c, 0, 3, uc),
                                ];
                                for p in permutations {
                                    let ordered = p.map(|i| rows[i].clone());
                                    for quota in 0..=12 {
                                        assert_eq!(
                                            arbitrate(quota, &ordered),
                                            iterative_oracle(quota, &ordered),
                                            "quota={quota} rows={ordered:?}"
                                        );
                                        checked += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 134_784);
    }

    #[test]
    fn extreme_totals_are_exact_and_grants_do_not_iterate_per_byte() {
        let max = usize::MAX;
        let rows = [
            row(1, Class::Capability, max, max, 1),
            row(2, Class::Cache, max, max, 1),
        ];
        assert_eq!(
            arbitrate(max, &rows),
            Arbitration::Refused {
                floors: 2 * max as u128,
                quota: max,
            }
        );
        let snapshot = LedgerSnapshot {
            tenants: rows.to_vec(),
            ..Default::default()
        };
        assert_eq!(snapshot.floors(), 2 * max as u128);
        assert_eq!(snapshot.wants(), 2 * max as u128);
        assert!(snapshot.line().contains("floors="));
        // Never pass these to the iterative oracle: they are repaired-path tests.
        assert_eq!(
            arbitrate(max - 1, &[row(9, Class::Cache, 0, max, 1)]),
            Arbitration::Granted(vec![Grant {
                id: TenantId(9),
                bytes: max - 1
            }])
        );
        assert_eq!(
            arbitrate(
                max,
                &[
                    row(1, Class::Cache, max - 1, max, 2),
                    row(2, Class::Scratch, 0, max, 1),
                ]
            ),
            Arbitration::Granted(vec![
                Grant {
                    id: TenantId(1),
                    bytes: max - 1
                },
                Grant {
                    id: TenantId(2),
                    bytes: 1
                },
            ])
        );
        assert_eq!(
            arbitrate(
                max,
                &[
                    row(1, Class::Scratch, 0, max, 0),
                    row(2, Class::Capability, 0, max, max),
                ]
            ),
            Arbitration::Granted(vec![
                Grant {
                    id: TenantId(1),
                    bytes: 0
                },
                Grant {
                    id: TenantId(2),
                    bytes: max
                },
            ])
        );
        assert_eq!(
            arbitrate(max, &[row(1, Class::Cache, 0, max, max - 1)]),
            Arbitration::Granted(vec![Grant {
                id: TenantId(1),
                bytes: max
            }])
        );
    }

    #[test]
    fn empty_and_raw_snapshot_wants_preserve_existing_semantics() {
        assert_eq!(arbitrate(0, &[]), Arbitration::Granted(vec![]));
        let snapshot = LedgerSnapshot {
            tenants: vec![row(1, Class::Capability, 3, 1, 0)],
            ..Default::default()
        };
        // Reporting uses raw wants; arbitration normalizes want up to floor.
        assert_eq!(snapshot.wants(), 1);
        assert_eq!(
            arbitrate(3, &snapshot.tenants),
            Arbitration::Granted(vec![Grant {
                id: TenantId(1),
                bytes: 3
            }])
        );
    }
    #[test]
    fn admission_overflow_refuses_but_reclaim_can_restore_representability() {
        let max = usize::MAX;
        assert!(fits_budget(max - 1, 1, max));
        assert_eq!(machine_target(max - 1, 1).unwrap(), max as u64);
        assert!(!fits_budget(max, 1, max));
        assert_eq!(
            machine_target(max, 1),
            Err(crate::memory_admission::AllocationError::Overflow)
        );
        assert_eq!(reclaim_target(max, 1, max), 1);
        // After actual release, not a pledge, the same request can fit.
        assert!(fits_budget(max - 1, 1, max));
        assert_eq!(reclaim_target(max, max, 0), max);
        assert!(!fits_budget(max, max, 0));
        assert!(machine_target(max, max).is_err());
        assert_eq!(reclaim_target(2, 3, 8), 0);
        for committed in 0..=8 {
            for need in 0..=8 {
                for budget in 0..=16 {
                    assert_eq!(
                        fits_budget(committed, need, budget),
                        committed + need <= budget
                    );
                    assert_eq!(
                        reclaim_target(committed, need, budget),
                        (committed + need).saturating_sub(budget)
                    );
                }
            }
        }
    }
}
