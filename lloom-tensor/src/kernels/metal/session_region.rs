//! The session region: one owned allocation serving the class the arena declines.
//!
//! `DESIGN.md` §9.2, issue #47. The activation arena (`arena.rs`) is already an
//! owned slab — one `MTLBuffer`, slots at fixed offsets, handles held so the free
//! list never sees them. What it does **not** cover is the population that
//! carries 84.1 % of the residual overhang (§9.2u), and the reason is structural
//! rather than a gap: `Arena::acquire` serves an ordinal only when the request
//! size equals what that ordinal was planned for, and this population's size
//! **grows every token**.
//!
//! # What this class is, measured rather than named
//!
//! Under `KvAppend::Cat`, `lfm2.rs` builds each layer's cache with
//! `Tensor::cat(&[k_cache, &k], 2)?.contiguous()?` and stores the result in
//! `cache.kvs[block_idx]`. So each of the 8 KV tensors is reallocated one token
//! wider every step, and the previous one is discarded into the pool. Measured
//! on this tree at `kv_len` 44 → 234, that is **12 084 acquisitions in the
//! 4 KB–256 KB band**, served at **15.82×** what they asked for.
//!
//! The arena declines every one of them, and it declines them **by plan**:
//! `lfm2-dispatch-trace --arena` reports `declined(size) 0` and
//! `declined(past plan) 0`, with every decline `declined(session)`. That is the
//! correct verdict — a slot at a fixed offset cannot hold an occupant that grows,
//! and §9.2c measured what happens when session state is admitted anyway (969
//! bindings fail the 674-position stability check).
//!
//! # Why a region rather than a wider plan
//!
//! The size test is not the obstacle to remove; it is the invariant that keeps
//! the arena correct. What this module adds is a **second** owned allocation with
//! a different placement rule, for the class whose defining property the arena's
//! rule cannot express:
//!
//! | | the arena (`arena.rs`) | this module |
//! |---|---|---|
//! | population | activations, fixed size per ordinal | session state, monotonically growing |
//! | placement | packed slots, reused across layers | one span per tenant, never overlapping |
//! | sized by | the packed peak live set (68 KB) | the worst case at `capacity` tokens |
//! | reset | ordinal restarts every step | never — the tenant reads its own bytes next step |
//!
//! That last row is the whole difference and is why this is not the arena with a
//! relaxed test. An arena slot is safe to reuse because its occupant is dead by
//! the end of the step; a session tenant is read *by the next step*, which is the
//! property `ArenaRecorder::outlives_step` detects and the reason these ordinals
//! are excluded. **A per-step bump reset over this population would corrupt it**,
//! and under `HazardTrackingModeUntracked` (§9.3) nothing would report it.
//!
//! # ggml's `reserve`, which is the borrow (§9.2t)
//!
//! §9.2t records what we take from `ggml-alloc`: *plan once behind a validity
//! check, where the check does the work.* `ggml_gallocr_reserve` runs the whole
//! allocation against a worst-case graph and caches the plan. This region is that
//! shape for a class whose worst case is computable in closed form: a tenant that
//! grows by a fixed stride per token, to a known capacity, needs
//! `stride × capacity` bytes and never more.
//!
//! So the region is sized **once**, at install, and every later request is served
//! by returning a view — no `newBufferWithLength`, no residency insert, no
//! best-fit range query, and nothing entering the free list to be evicted.
//!
//! # What the win actually is, and what it is not
//!
//! §9.2u is emphatic and it is inherited here rather than restated loosely: the
//! overhang is a **flow**, not a stock. It accumulates 11.3 GB while
//! `device_allocated` moves 0.164 GB, so this region does **not** reclaim 11 GB.
//! What it removes is the per-token *allocation event* — and with it the pool
//! traffic that makes band 1's ratio what it is, because the tenants that were
//! being discarded into the free list every token are the same buffers a 76 KB
//! activation request was then being served out of (§9.2u's *one class served out
//! of another class's garbage*).
//!
//! # The cost this removes is the residency insert, not `newBuffer`
//!
//! Stated because #47's design comment gets it the other way round and a gate
//! built on the wrong term measures the wrong thing. `newBufferWithLength:` is
//! **not** reliably a driver VM round-trip: Metal itself sub-allocates, which
//! this tree witnessed in `__bzero` under
//! `-[IOGPUMetalDevice allocBufferSubDataWithLength:]` (`buffer_pool.rs`, #475).
//! The Metal call that is unconditionally on the miss path is
//! `residency_set.insert`, one per fresh allocation, which `device.rs` documents
//! as *"already on the per-token path at the allocation rate"* — and measured on
//! this tree `residency_commits` (+2 635) exceeds `allocations` (+2 016) over the
//! same window.
//!
//! # The hazard
//!
//! §9.3: `HazardTrackingModeUntracked` gives no safety net, so correctness rests
//! on this module's offset arithmetic exactly as it does on the arena's. Two
//! tenants must never receive overlapping spans, and a tenant must never be
//! handed a span shorter than it asked for. Both are asserted on every request
//! rather than argued ([`SessionRegion::acquire`]), and
//! `two_tenants_never_receive_overlapping_spans` pins the first against a
//! mutation.

use super::{Buffer, BufferHandle, PooledBuffer};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Offsets are 128 B aligned, for §9.2's reason: it is safe for every Metal
/// dtype (our fork's PR #8) and is also `hw.cachelinesize` on this machine, so
/// lowering it would introduce false sharing between adjacent spans on any
/// CPU-side access. Deliberately the same constant the arena uses.
pub const REGION_ALIGNMENT: usize = 128;

fn align_up(n: usize, to: usize) -> usize {
    n.div_ceil(to) * to
}

/// A tenant's identity within the region.
///
/// Session state is per-sequence and per-layer, so a tenant is named by the
/// ordinal at which it is allocated within a step. That is the same key the
/// arena uses and for the same reason (§9.2c): the kernel sequence is
/// byte-identical across tokens (§11.1a.1), so the Nth allocation of one step is
/// the Nth of the next. Keying on the *buffer* would merge unrelated values, and
/// keying on size would merge the K and V of one layer.
pub type TenantId = usize;

/// One tenant's reserved span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Span {
    /// Byte offset within the region's base. 128 B aligned.
    pub offset: usize,
    /// Bytes reserved for this tenant at full capacity.
    pub reserved: usize,
}

impl Span {
    fn end(&self) -> usize {
        self.offset + self.reserved
    }
}

/// The plan: which tenant owns which span, computed once.
///
/// # Reserve the worst case, which is what makes this zero-allocation
///
/// §9.2t's borrow from ggml. A tenant growing by `stride` bytes per token to a
/// capacity of `capacity` tokens occupies at most `stride * capacity` bytes, and
/// that bound is exact rather than estimated because the growth is
/// `Tensor::cat`'s and its stride is `b * kv_heads * head_dim * dtype`.
#[derive(Clone, Debug, Default)]
pub struct RegionPlan {
    spans: HashMap<TenantId, Span>,
    bytes: usize,
}

impl RegionPlan {
    /// Reserve `reserved[i]` bytes for tenant `tenants[i]`.
    ///
    /// Spans are laid out in tenant order with each start aligned up, so the
    /// layout is a pure function of the input — the same reservation always
    /// produces the same offsets, which §2.3.2 requires (a layout that varied run
    /// to run would be indistinguishable from a missing fence).
    pub fn reserve(reservations: &[(TenantId, usize)]) -> Self {
        let mut spans = HashMap::with_capacity(reservations.len());
        let mut cursor = 0usize;
        // Sorted so the layout does not depend on the caller's iteration order.
        let mut ordered: Vec<(TenantId, usize)> = reservations.to_vec();
        ordered.sort_unstable_by_key(|&(t, _)| t);
        for (tenant, reserved) in ordered {
            let offset = align_up(cursor, REGION_ALIGNMENT);
            spans.insert(tenant, Span { offset, reserved });
            cursor = offset + reserved;
        }
        Self {
            spans,
            bytes: align_up(cursor, REGION_ALIGNMENT),
        }
    }

    /// Fallible native-allocation route. Preserve tenant ordering and duplicate
    /// replacement behavior while rejecting unrepresentable offsets/padding.
    pub fn checked_reserve(reservations: &[(TenantId, usize)]) -> Result<Self, String> {
        let aligned = |n: usize| {
            n.checked_add(REGION_ALIGNMENT - 1)
                .map(|n| n / REGION_ALIGNMENT * REGION_ALIGNMENT)
                .ok_or_else(|| "session region alignment overflow".to_string())
        };
        let mut spans = HashMap::with_capacity(reservations.len());
        let mut ordered = reservations.to_vec();
        ordered.sort_unstable_by_key(|&(tenant, _)| tenant);
        let mut cursor = 0usize;
        for (tenant, reserved) in ordered {
            let offset = aligned(cursor)?;
            cursor = offset
                .checked_add(reserved)
                .ok_or_else(|| "session region size overflow".to_string())?;
            spans.insert(tenant, Span { offset, reserved });
        }
        Ok(Self {
            spans,
            bytes: aligned(cursor)?,
        })
    }

    /// Build a plan from explicit `(tenant, offset, reserved)` triples.
    ///
    /// **For tests only, and it exists so `check_disjoint` has an inverted arm.**
    /// `reserve` cannot produce an overlap by construction, so a test that only
    /// ever built plans through it would assert that the guard passes on inputs
    /// that cannot fail it — §3.1a rule 4's all-green case, and the shape #512's
    /// vacuous harness had. This lets a test construct the state the guard
    /// exists to refuse.
    #[cfg(test)]
    pub fn overlapping_for_test(triples: &[(TenantId, usize, usize)]) -> Self {
        let mut spans = HashMap::with_capacity(triples.len());
        let mut end = 0usize;
        for &(tenant, offset, reserved) in triples {
            spans.insert(tenant, Span { offset, reserved });
            end = end.max(offset + reserved);
        }
        Self {
            spans,
            bytes: align_up(end, REGION_ALIGNMENT),
        }
    }

    /// Total bytes the region's base must hold.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// How many tenants are planned.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn span(&self, tenant: TenantId) -> Option<Span> {
        self.spans.get(&tenant).copied()
    }

    /// Every planned span, ascending by offset.
    pub fn spans(&self) -> Vec<(TenantId, Span)> {
        let mut v: Vec<(TenantId, Span)> = self.spans.iter().map(|(&t, &s)| (t, s)).collect();
        v.sort_unstable_by_key(|&(_, s)| s.offset);
        v
    }

    /// Refuse a plan whose spans overlap.
    ///
    /// §9.3 gives no safety net, so this is checked rather than argued. Two
    /// tenants sharing bytes would alias silently, and each tenant here is read
    /// a step after it is written — so the corruption would appear one token
    /// later than the write that caused it.
    pub fn check_disjoint(&self) -> Result<(), String> {
        let ordered = self.spans();
        for w in ordered.windows(2) {
            let (ta, a) = w[0];
            let (tb, b) = w[1];
            if a.end() > b.offset {
                return Err(format!(
                    "tenants {ta} [{}, {}) and {tb} [{}, {}) overlap",
                    a.offset,
                    a.end(),
                    b.offset,
                    b.end()
                ));
            }
        }
        for (t, s) in &ordered {
            if !s.offset.is_multiple_of(REGION_ALIGNMENT) {
                return Err(format!(
                    "tenant {t} at offset {} is not {REGION_ALIGNMENT} B aligned",
                    s.offset
                ));
            }
        }
        Ok(())
    }
}

/// Counters, one relaxed atomic each — the shape `arena.rs` uses (#562).
#[derive(Default)]
struct LiveCounters {
    offers: AtomicU64,
    hits: AtomicU64,
    declined_unplanned: AtomicU64,
    declined_capacity: AtomicU64,
}

/// A snapshot of the region's counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RegionCounters {
    /// Requests offered to the region.
    pub offers: u64,
    /// Requests served from the region.
    pub hits: u64,
    /// Declined: the tenant has no reservation.
    ///
    /// Nonzero means the plan does not cover the population, which is a coverage
    /// fact and not an error — the request falls through to the pool exactly as
    /// it did before.
    pub declined_unplanned: u64,
    /// Declined: the request exceeds what was reserved for that tenant.
    ///
    /// **This is the counter that says the `reserve` was too small.** It is
    /// separate from `declined_unplanned` because the two want opposite
    /// responses: an unplanned tenant means widen the plan, an over-capacity
    /// request means raise the capacity.
    pub declined_capacity: u64,
}

struct RegionInner {
    /// The one allocation. Owned here; freed by Metal when the last clone drops.
    base: Buffer,
    plan: RegionPlan,
    /// One handle per tenant, made once and held for the region's lifetime, so
    /// the free list never sees them (§9.2a layer 3, `PooledBuffer::unpooled`).
    ///
    /// A handle covers the tenant's **whole reserved span**, and a request is
    /// served a view of its prefix. Holding the full-span handle rather than
    /// making one per request is what keeps the per-token path free of any
    /// allocation at all.
    handles: Mutex<HashMap<TenantId, BufferHandle>>,
    counters: LiveCounters,
}

/// The session region.
///
/// Cheap to clone; every clone refers to the same region.
#[derive(Clone)]
pub struct SessionRegion {
    inner: Arc<RegionInner>,
}

impl SessionRegion {
    /// Build a region over `base` for `plan`.
    ///
    /// Takes no pool, for `Arena::new`'s reason (#489): the region cannot hand
    /// the pool a view of its base because it is not given a pool to hand it to.
    /// A view that reached the free list would be handed out as an independent
    /// allocation — `acquire` keys on size alone — and two session tenants would
    /// alias.
    pub fn new(base: Buffer, plan: RegionPlan) -> Result<Self, String> {
        plan.check_disjoint()?;
        if base.length() < plan.bytes() {
            return Err(format!(
                "session region buffer is {} B, plan needs {} B",
                base.length(),
                plan.bytes()
            ));
        }
        Ok(Self {
            inner: Arc::new(RegionInner {
                base,
                plan,
                handles: Mutex::new(HashMap::new()),
                counters: LiveCounters::default(),
            }),
        })
    }

    /// Serve `size` bytes for `tenant`, or decline.
    ///
    /// # Why a view of a held handle, and not a fresh allocation
    ///
    /// The tenant's full span is allocated once — as part of the region's base —
    /// and every request returns a view of its prefix. So the per-token cost is a
    /// hash lookup and a `Buffer::view`, with **no `newBufferWithLength`, no
    /// `residency_set.insert`, and no free-list traffic**. That is the owned-slab
    /// thesis's actual mechanism (§9.2w: the gate *"requires removing the choice
    /// rather than reordering it"*).
    ///
    /// # The two declines are different facts and are counted separately
    ///
    /// An unplanned tenant is a **coverage** statement; an over-capacity request
    /// is a **sizing** statement. Both fall through to the pool, which is the
    /// safe direction — it costs coverage, never correctness — and both are
    /// counted so a run can say which happened rather than reporting one number
    /// that could mean either (§3.1c: a zero is not evidence unless something
    /// nonzero came from the same command).
    pub fn acquire(&self, tenant: TenantId, size: usize) -> Option<BufferHandle> {
        let c = &self.inner.counters;
        c.offers.fetch_add(1, Ordering::Relaxed);

        let Some(span) = self.inner.plan.span(tenant) else {
            c.declined_unplanned.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        // A request past the reservation would run off the end of the span and
        // into the next tenant's bytes. §9.3: nothing would report it, so this is
        // checked rather than assumed.
        if size > span.reserved {
            c.declined_capacity.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // `parking_lot` never poisons, so there is no lock error to decline on.
        let mut handles = self.inner.handles.lock();
        let full = handles.entry(tenant).or_insert_with(|| {
            BufferHandle::new(PooledBuffer::unpooled(
                self.inner.base.view(span.offset, span.reserved),
                span.reserved,
            ))
        });
        // The view is of the tenant's own span, so two tenants can never be
        // handed overlapping bytes as long as the plan is disjoint — which
        // `RegionPlan::check_disjoint` established at construction.
        let handle = BufferHandle::new(PooledBuffer::unpooled(full.view(0, size), size));
        c.hits.fetch_add(1, Ordering::Relaxed);
        Some(handle)
    }

    /// The region's base, for residency registration.
    pub fn base(&self) -> &Buffer {
        &self.inner.base
    }

    /// Bytes the region holds.
    pub fn bytes(&self) -> usize {
        self.inner.plan.bytes()
    }

    /// How many tenants the plan covers.
    pub fn tenants(&self) -> usize {
        self.inner.plan.len()
    }

    pub fn counters(&self) -> RegionCounters {
        let c = &self.inner.counters;
        RegionCounters {
            offers: c.offers.load(Ordering::Relaxed),
            hits: c.hits.load(Ordering::Relaxed),
            declined_unplanned: c.declined_unplanned.load(Ordering::Relaxed),
            declined_capacity: c.declined_capacity.load(Ordering::Relaxed),
        }
    }

    pub fn plan(&self) -> &RegionPlan {
        &self.inner.plan
    }
}

impl std::fmt::Debug for SessionRegion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionRegion")
            .field("tenants", &self.inner.plan.len())
            .field("bytes", &self.inner.plan.bytes())
            .finish()
    }
}

#[cfg(test)]
mod checked_planner_tests {
    use super::*;
    #[test]
    fn checked_planner_region_geometry() {
        assert_eq!(RegionPlan::checked_reserve(&[]).unwrap().bytes(), 0);
        let plan = RegionPlan::checked_reserve(&[(2, 1), (0, 0), (1, 128)]).unwrap();
        assert_eq!(
            plan.spans[&0],
            Span {
                offset: 0,
                reserved: 0
            }
        );
        assert_eq!(
            plan.spans[&1],
            Span {
                offset: 0,
                reserved: 128
            }
        );
        assert_eq!(
            plan.spans[&2],
            Span {
                offset: 128,
                reserved: 1
            }
        );
        assert_eq!(plan.bytes(), 256);
        let duplicate = RegionPlan::checked_reserve(&[(1, 1), (1, 1)]).unwrap();
        assert_eq!(
            duplicate.spans[&1],
            Span {
                offset: 128,
                reserved: 1
            }
        );
        assert_eq!(duplicate.bytes(), 256);
        assert_eq!(
            RegionPlan::checked_reserve(&[(0, usize::MAX - 127)])
                .unwrap()
                .bytes(),
            usize::MAX - 127
        );
        assert!(RegionPlan::checked_reserve(&[(0, usize::MAX)]).is_err());
        assert!(RegionPlan::checked_reserve(&[(0, usize::MAX - 127), (1, 128)]).is_err());
        assert!(RegionPlan::checked_reserve(&[(0, usize::MAX - 126), (1, 0)]).is_err());
    }
}
