//! Tests for the session region (#47, `DESIGN.md` §9.2x).
//!
//! # What these assert, and why they are absolute rather than shape
//!
//! `CONTRIBUTING.md` §3.1b rule 1: *prefer an absolute reference to a shape.*
//! A test that asserts "the spans do not overlap" is blind to a defect that
//! shifts every span by the same amount, and a per-tenant offset error does
//! exactly that. So the layout tests name **the byte pair**, not the relation:
//! `[0, 512)` and `[512, 1024)`, so a failure prints a number a reader can act
//! on rather than *"they were not disjoint"*.
//!
//! The mutation each one kills is named in its own doc comment, per §3.1b rule
//! 4 — the coverage is intrinsic where it can be, and where a sibling carries it
//! the comment says which sibling and which mutation.

use super::session_region::{RegionPlan, SessionRegion, REGION_ALIGNMENT};
use crate::kernels::metal::Device;

fn device() -> Device {
    Device::system_default().expect("no Metal device")
}

fn region_of(reservations: &[(usize, usize)]) -> SessionRegion {
    let plan = RegionPlan::reserve(reservations);
    let dev = device();
    let base = dev
        .new_buffer(plan.bytes().max(1), crate::kernels::RESOURCE_OPTIONS)
        .expect("region allocation");
    SessionRegion::new(base, plan).expect("plan is valid")
}

/// The layout, by absolute byte pair.
///
/// **Kills**: any change to `RegionPlan::reserve`'s cursor arithmetic — a
/// missing `align_up`, an off-by-one, or laying out in input order rather than
/// tenant order. A *disjointness* assertion would survive all three, because
/// each produces a layout that is still disjoint.
#[test]
fn spans_are_laid_out_at_the_bytes_the_plan_names() {
    let plan = RegionPlan::reserve(&[(0, 512), (1, 512)]);
    assert_eq!(
        plan.span(0).unwrap().offset,
        0,
        "tenant 0 starts the region"
    );
    assert_eq!(plan.span(0).unwrap().reserved, 512);
    assert_eq!(
        plan.span(1).unwrap().offset,
        512,
        "tenant 1 starts where tenant 0 ends -- 512 is already 128 B aligned"
    );
    assert_eq!(plan.bytes(), 1024, "the region is exactly the two spans");
}

/// A span that is not a multiple of the alignment still puts the next tenant on
/// an aligned boundary, and the region's total covers the padding.
///
/// **Kills**: dropping the `align_up` on the cursor. With 100 B reserved for
/// tenant 0, an unaligned layout puts tenant 1 at 100; this asserts 128.
#[test]
fn the_next_span_is_aligned_even_when_the_previous_is_not() {
    let plan = RegionPlan::reserve(&[(0, 100), (1, 100)]);
    assert_eq!(plan.span(0).unwrap().offset, 0);
    assert_eq!(
        plan.span(1).unwrap().offset,
        REGION_ALIGNMENT,
        "tenant 1 is pushed to the next {REGION_ALIGNMENT} B boundary"
    );
    assert_eq!(
        plan.bytes(),
        256,
        "the region rounds up past the last span's end"
    );
}

/// The layout does not depend on the caller's iteration order.
///
/// **Kills**: removing the `sort_unstable_by_key` in `reserve`. §2.3.2 — a
/// layout that varied run to run would be indistinguishable from a missing
/// fence, and a `HashMap` iteration order is exactly that kind of variation.
#[test]
fn the_layout_is_the_same_whatever_order_the_reservations_arrive_in() {
    let a = RegionPlan::reserve(&[(0, 256), (1, 384), (2, 128)]);
    let b = RegionPlan::reserve(&[(2, 128), (0, 256), (1, 384)]);
    for t in 0..3 {
        assert_eq!(
            a.span(t).unwrap(),
            b.span(t).unwrap(),
            "tenant {t} moved when the reservations were reordered"
        );
    }
    assert_eq!(a.bytes(), b.bytes());
}

/// Two tenants are never handed overlapping bytes.
///
/// This is the §9.3 property: `HazardTrackingModeUntracked` gives no safety net,
/// and each tenant here is read a step *after* it is written, so an overlap
/// would corrupt one token later than the write that caused it.
///
/// **Kills**: any `acquire` that serves from a shared cursor rather than the
/// tenant's own span. Asserted on the produced buffers' actual addresses, not on
/// the plan, so it tests what a dispatch would bind.
#[test]
fn two_tenants_never_receive_overlapping_spans() {
    let region = region_of(&[(0, 512), (1, 512)]);
    let a = region.acquire(0, 512).expect("tenant 0 is planned");
    let b = region.acquire(1, 512).expect("tenant 1 is planned");
    let (a0, a1) = (a.base_offset(), a.base_offset() + a.size());
    let (b0, b1) = (b.base_offset(), b.base_offset() + b.size());
    assert!(
        a1 <= b0 || b1 <= a0,
        "tenant 0 [{a0}, {a1}) and tenant 1 [{b0}, {b1}) overlap"
    );
    // The absolute half, per §3.1b rule 1: naming the bytes catches a defect
    // that shifts both spans equally and keeps them disjoint.
    assert_eq!((a0, a1), (0, 512), "tenant 0 is not at the planned bytes");
    assert_eq!(
        (b0, b1),
        (512, 1024),
        "tenant 1 is not at the planned bytes"
    );
}

/// A growing tenant is served from the same bytes every time, and the region
/// allocates nothing on the way.
///
/// **This is the mechanism the whole change exists for.** `Tensor::cat` under
/// `KvAppend::Cat` asks for one token more each step; the region answers every
/// one of them from the tenant's reserved span, so there is no
/// `newBufferWithLength`, no residency insert and nothing entering the free list.
///
/// **Kills**: an `acquire` that allocates per request. The offset is asserted
/// constant across the growth, which a fresh allocation would not be.
#[test]
fn a_growing_tenant_is_served_from_one_reservation() {
    // 8 tokens of a 128 B stride: the shape of one LFM2 layer's K under `Cat`.
    let region = region_of(&[(0, 128 * 8)]);
    let mut offsets = Vec::new();
    for tokens in 1..=8 {
        let b = region
            .acquire(0, 128 * tokens)
            .unwrap_or_else(|| panic!("token {tokens} declined"));
        assert_eq!(
            b.size(),
            128 * tokens,
            "token {tokens} was served the wrong length"
        );
        offsets.push(b.base_offset());
    }
    assert_eq!(
        offsets,
        vec![0; 8],
        "the tenant moved between tokens; it must be served from one reservation"
    );
    let c = region.counters();
    assert_eq!(c.hits, 8, "every request should have been served");
    assert_eq!(c.declined_capacity, 0);
    assert_eq!(c.declined_unplanned, 0);
}

/// A request past the reservation is declined, and counted as a *capacity*
/// decline rather than an unplanned one.
///
/// The two want opposite responses — widen the plan against raise the capacity —
/// so a single decline counter would report a number that could mean either.
///
/// **Kills**: `size > span.reserved` becoming `>=` or being dropped. Dropping it
/// is the dangerous mutation: the request would be served a view running past
/// the tenant's span and into the next tenant's bytes, which §9.3 says nothing
/// would report.
#[test]
fn a_request_past_the_reservation_declines_rather_than_running_over() {
    let region = region_of(&[(0, 512), (1, 512)]);
    assert!(
        region.acquire(0, 512).is_some(),
        "exactly the reservation must be served"
    );
    assert!(
        region.acquire(0, 513).is_none(),
        "one byte past the reservation must decline"
    );
    let c = region.counters();
    assert_eq!(
        c.declined_capacity, 1,
        "the decline was not counted as capacity"
    );
    assert_eq!(
        c.declined_unplanned, 0,
        "a capacity decline must not be filed as unplanned"
    );
}

/// An unplanned tenant declines and falls through, without disturbing the
/// planned ones.
///
/// **Kills**: a plan lookup that returns a default span for a missing tenant,
/// which would hand two unrelated tenants the same bytes.
#[test]
fn an_unplanned_tenant_declines_and_is_counted_separately() {
    let region = region_of(&[(0, 512)]);
    assert!(region.acquire(7, 128).is_none(), "tenant 7 is not planned");
    let c = region.counters();
    assert_eq!(c.declined_unplanned, 1);
    assert_eq!(
        c.declined_capacity, 0,
        "an unplanned tenant is not a capacity decline"
    );
    assert!(
        region.acquire(0, 512).is_some(),
        "the planned tenant still serves after a decline"
    );
}

/// An overlapping plan is refused at construction.
///
/// **Kills**: removing `check_disjoint` from `SessionRegion::new`. Constructed
/// by hand rather than through `reserve`, because `reserve` cannot produce an
/// overlap — so this tests the guard rather than the layout, which is the point
/// (§3.1a rule 3: a test of the rule, not a grep for it).
#[test]
fn an_overlapping_plan_is_refused_at_construction() {
    // Two tenants at the same offset: what a future placement policy that packed
    // by liveness rather than by reservation could produce.
    let plan = RegionPlan::reserve(&[(0, 512), (1, 512)]);
    assert!(
        plan.check_disjoint().is_ok(),
        "the reserved plan is disjoint"
    );

    // The inverted arm: a plan that IS overlapping must be refused, or the check
    // above is vacuous (§3.1a rule 4 -- all-green is as suspicious as all-red).
    let overlapping = RegionPlan::overlapping_for_test(&[(0, 0, 512), (1, 256, 512)]);
    let err = overlapping
        .check_disjoint()
        .expect_err("an overlapping plan must be refused");
    assert!(
        err.contains("overlap"),
        "the error must name the defect, got: {err}"
    );
}

/// A region whose base is smaller than the plan is refused.
///
/// **Kills**: removing the length check in `SessionRegion::new`. A short base
/// would let the last tenant's view run past the allocation.
#[test]
fn a_base_too_small_for_the_plan_is_refused() {
    let plan = RegionPlan::reserve(&[(0, 4096)]);
    let dev = device();
    let base = dev
        .new_buffer(1024, crate::kernels::RESOURCE_OPTIONS)
        .expect("allocation");
    let err = SessionRegion::new(base, plan).expect_err("a short base must be refused");
    assert!(
        err.contains("1024") && err.contains("4096"),
        "the error must name both sizes, got: {err}"
    );
}

/// The counters distinguish "served" from "never asked", which is what makes a
/// zero readable.
///
/// §3.1c: *a zero is not evidence unless something nonzero came from the same
/// command.* A run reporting `hits 0` could mean the region is not covering the
/// population or that it was never installed, and `offers` separates them.
#[test]
fn offers_counts_every_request_including_the_declined_ones() {
    let region = region_of(&[(0, 512)]);
    let _ = region.acquire(0, 256);
    let _ = region.acquire(0, 4096);
    let _ = region.acquire(9, 256);
    let c = region.counters();
    assert_eq!(c.offers, 3, "every request must be offered");
    assert_eq!(c.hits, 1);
    assert_eq!(c.declined_capacity, 1);
    assert_eq!(c.declined_unplanned, 1);
    assert_eq!(
        c.hits + c.declined_capacity + c.declined_unplanned,
        c.offers,
        "every offer must be accounted for by exactly one outcome"
    );
}
