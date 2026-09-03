//! Arithmetic tests for the scratch class (`DESIGN.md` §9.1, issue #71).
//!
//! Pure functions, no device: [`plan_scratch`] allocates nothing, so the sizing
//! arithmetic is testable without a GPU and the fixtures can carry shapes LFM2
//! never produces. The GPU-side tests -- disjointness, the fence, the merge
//! order as executed -- are in `scratch_gpu_tests.rs`.
//!
//! # The rule every fixture here obeys, and why it is stated at the top
//!
//! **Any fixture for arena arithmetic must contain a size that is not a multiple
//! of [`ARENA_ALIGNMENT`], or it is testing the identity function.**
//!
//! That is #70's warning, now §9.2c. Every LFM2 decode activation is a
//! 128-multiple, so `align_up` is a no-op on every shape our own model produces:
//! #70 **deleted `align_up` and its acceptance test still passed**, until a
//! deliberately unaligned size was added to the fixture.
//!
//! The scratch class is the first consumer where this bites -- but **only at one
//! of two levels**, which is finer than #70 stated it and is the thing that
//! decides whether a fixture can fail at all. See
//! [`the_capacity_comparison_is_blind_under_both_layouts_and_the_record_is_not`].

use super::scratch::*;
use super::ARENA_ALIGNMENT;

/// LFM2's geometry, from §5.2's verified config.
fn lfm2() -> PartialsGeometry {
    PartialsGeometry::default()
}

/// LFM2's attention layer count: 8, from §5.3's `layer_types` array.
const LFM2_ATTENTION_LAYERS: usize = 8;

/// 128k, §5.2's `max_position_embeddings`.
const MAX_CONTEXT: usize = 131_072;

// ---------------------------------------------------------------------------
// The size table §9.1 carries
// ---------------------------------------------------------------------------

/// **§9.1's table, reproduced from the geometry rather than quoted.**
///
/// The four rows are the reason this class was worth naming: at 128k the
/// partials are ~62x the 68 KB activation arena (§9.2c), and this is the only
/// class whose size depends on `kv_len`.
///
/// Computing them here rather than asserting the MB figures means the table in
/// `DESIGN.md` is checkable against the code that implements it -- §11.3h's
/// lesson, three times learned: *a number written once beside an artifact that
/// later changes is a number nobody re-checks.*
#[test]
fn the_size_table_in_9_1_reproduces_from_the_geometry() {
    let g = lfm2();
    // (kv_len, chunks, bytes) -- the MB column of §9.1's table.
    let rows = [
        (2_720usize, 11usize, 92_928usize),
        (32_768, 128, 1_081_344),
        (131_072, 512, 4_325_376),
        (262_144, 1024, 8_650_752),
    ];
    for (kv, want_chunks, want_bytes) in rows {
        assert_eq!(g.chunks(kv), want_chunks, "chunk count at kv_len {kv}");
        assert_eq!(
            g.partials_bytes(want_chunks),
            want_bytes,
            "partial bytes at kv_len {kv}"
        );
    }

    // The headline ratio against #68's measured 68.00 KB activation arena.
    let at_128k = g.partials_bytes(g.chunks(131_072));
    let activation_arena = 69_632usize;
    let ratio = at_128k as f64 / activation_arena as f64;
    assert!(
        (62.0..63.0).contains(&ratio),
        "scratch/activation ratio at 128k is {ratio:.1}x, not the ~62x 9.1 states"
    );
}

/// The class cannot be shrunk to f16, and the constant that says so is F32.
///
/// §8.1 principle 4 (accumulate in F32) and §10.4 (the combine merges these, so
/// the merge inherits the precision). Pinned because "just store them as half"
/// is the first thing anyone will propose about a 4 MB buffer, and the reason it
/// is wrong is two sections apart from the size.
#[test]
fn partials_accumulate_in_f32() {
    assert_eq!(PARTIAL_ELEM_BYTES, 4, "partials must accumulate in F32");
    assert_eq!(PARTIAL_STATS, 2, "online softmax carries m and l");
}

// ---------------------------------------------------------------------------
// Alignment -- the fixture rule
// ---------------------------------------------------------------------------

/// **The unaligned fixture #70's warning demands, and the finding it produces.**
///
/// LFM2's interleaved record is `(64 + 2) * 4` = **264 B**, which is not a
/// multiple of 128. That makes it the first shape in this project on which
/// `align_up` is load-bearing *for our own model* -- every decode activation
/// (§5.9) and every planes-layout region is already a 128-multiple.
///
/// The assertion is on the padded stride rather than on `align_up` directly,
/// because that is the number a wrong rounding would move.
#[test]
fn the_interleaved_record_is_not_a_multiple_of_the_alignment() {
    let g = lfm2();
    let record = g.interleaved_record_bytes();
    assert_eq!(record, 264, "LFM2's interleaved record");
    assert_ne!(
        record % ARENA_ALIGNMENT,
        0,
        "the fixture's record is 128-aligned, so it tests the identity function"
    );

    // Padded to 384: three cache lines, not 264 B of straddling.
    let plan = plan_scratch(
        &g,
        2_720,
        LFM2_ATTENTION_LAYERS,
        Sizing::Grow,
        ScratchLayout::Interleaved,
        MAX_CONTEXT,
    )
    .expect("plan");
    let chunks = g.chunks(2_720);
    let expected = g.batch * g.n_heads * chunks * 384;
    assert_eq!(
        plan.regions()[0].size,
        expected,
        "the interleaved region is not padded to the alignment"
    );
}

/// **The blindness has two levels, and only one of them can fail.**
///
/// #70's warning says a fixture built from LFM2's own shapes cannot expose an
/// alignment defect. Running it here sharpens that by one level, and the
/// distinction is easy to conflate:
///
/// - **region level** -- `bump_capacity` against `arena_bytes`. Blind under
///   **both** layouts on LFM2's shapes, and the interleaved case is the
///   surprising half: its region is `32 x 11 x 384`, a 128-multiple, *because
///   the padding is already inside the region size*. A capacity comparison
///   therefore cannot see a bad `align_up` under either layout.
/// - **record level** -- the (head, chunk) stride *within* a region: 264 B
///   unpadded against 384 padded. This is the level where `align_up` decides
///   anything at all on shapes our own model produces.
///
/// So the useful statement is not "pick the interleaved layout and your fixture
/// can see it" -- it is "pick the right *level*". This test pins both halves so
/// the weaker claim cannot creep back in, and the mutation record bears it out:
/// deleting `align_up` is killed by the record-stride fixtures and by no
/// capacity comparison anywhere in the suite.
#[test]
fn the_capacity_comparison_is_blind_under_both_layouts_and_the_record_is_not() {
    let g = lfm2();
    for layout in [ScratchLayout::Planes, ScratchLayout::Interleaved] {
        for kv in [2_720usize, 32_768, 131_072] {
            let plan = plan_scratch(
                &g,
                kv,
                LFM2_ATTENTION_LAYERS,
                Sizing::Grow,
                layout,
                MAX_CONTEXT,
            )
            .expect("plan");
            assert_eq!(
                plan.arena_bytes(),
                plan.bump_capacity(),
                "{layout:?} at kv_len {kv} is not capacity-blind after all -- if this \
                 fails the geometry changed and the warning above needs rewriting"
            );
        }
    }

    // The level that is not blind, and the only one on LFM2's shapes that is not.
    assert_ne!(
        g.interleaved_record_bytes() % ARENA_ALIGNMENT,
        0,
        "the interleaved record became a 128-multiple, so nothing in this class \
         exercises align_up on the model's own shapes any more"
    );
    assert_eq!(
        g.partials_bytes(1) % ARENA_ALIGNMENT,
        0,
        "planes plane size"
    );
}

/// **`bump_capacity` is not `arena_bytes`** (#70), on a shape where it shows.
///
/// `arena_bytes` reports where the last *value* ends; a bump cursor rounds every
/// request including the last, so it ends where the last *slot* ends. Handing a
/// bump allocator the smaller figure makes it decline a region that fits the
/// plan perfectly -- a **quiet loss of coverage, not a corruption**, so nothing
/// goes red.
///
/// #70 records the shape as `[100, 300, 5000]` giving 5512 against 5632. Here
/// the same distinction appears on the real class, under the layout whose sizes
/// are not 128-multiples.
#[test]
fn bump_capacity_exceeds_arena_bytes_when_the_last_region_is_unaligned() {
    // A geometry whose interleaved record is deliberately awkward: head_dim 5
    // gives (5 + 2) * 4 = 28 B, padded to 128.
    let g = PartialsGeometry {
        n_heads: 3,
        head_dim: 5,
        page_size: 256,
        batch: 1,
    };
    // One region, so the difference is the tail rather than accumulated padding.
    let plan = plan_scratch(
        &g,
        256,
        1,
        Sizing::Grow,
        ScratchLayout::Interleaved,
        MAX_CONTEXT,
    )
    .expect("plan");
    assert_eq!(plan.regions().len(), 1);

    // 3 heads x 1 chunk x 128 B padded record = 384, which *is* aligned --
    // so this layout alone does not show it. Use the planes layout with an
    // awkward head_dim instead, where the plane sizes are the raw figures.
    let planes =
        plan_scratch(&g, 256, 1, Sizing::Grow, ScratchLayout::Planes, MAX_CONTEXT).expect("plan");
    let raw = g.partials_bytes(1); // 3 * 1 * 5 * 4 + 3 * 1 * 2 * 4 = 84
    assert_eq!(raw, 84);
    assert_ne!(raw % ARENA_ALIGNMENT, 0, "the fixture size is aligned");
    assert_eq!(planes.arena_bytes(), 84, "value end");
    assert_eq!(planes.bump_capacity(), 128, "slot end");
    assert!(
        planes.bump_capacity() > planes.arena_bytes(),
        "bump_capacity {} did not exceed arena_bytes {} -- align_up is a no-op here",
        planes.bump_capacity(),
        planes.arena_bytes()
    );
}

/// A multi-region plan on unaligned sizes: every region starts on the alignment
/// and none overlaps.
///
/// The overlap half is the one that matters. Two regions sharing bytes would
/// alias two layers' partials and §9.3 says nothing in the driver would catch
/// it -- under `HazardTrackingModeUntracked` a wrong offset is a silent wrong
/// answer.
#[test]
fn regions_are_aligned_and_disjoint_on_unaligned_sizes() {
    let g = PartialsGeometry {
        n_heads: 3,
        head_dim: 5,
        page_size: 256,
        batch: 1,
    };
    let plan =
        plan_scratch(&g, 700, 4, Sizing::Grow, ScratchLayout::Planes, MAX_CONTEXT).expect("plan");
    assert_eq!(plan.regions().len(), 4);
    plan.check_disjoint().expect("regions overlap");
    for (i, r) in plan.regions().iter().enumerate() {
        assert_eq!(
            r.offset % ARENA_ALIGNMENT,
            0,
            "region {i} at {} is not aligned",
            r.offset
        );
        assert_ne!(
            r.size % ARENA_ALIGNMENT,
            0,
            "region {i}'s size is aligned -- the fixture stopped exercising align_up"
        );
    }
}

/// The disjointness check must be able to fail (`CONTRIBUTING.md` §3.1).
#[test]
fn the_overlap_check_catches_overlapping_regions() {
    let g = lfm2();
    let mut plan = plan_scratch(
        &g,
        2_720,
        2,
        Sizing::Grow,
        ScratchLayout::Planes,
        MAX_CONTEXT,
    )
    .expect("plan");
    plan.check_disjoint().expect("a valid plan was rejected");

    // Mutation: pull the second region back inside the first.
    let overlapping = ScratchPlanForTest::overlap(&plan);
    let err = overlapping
        .check_disjoint()
        .expect_err("overlapping regions were accepted");
    assert!(err.contains("overlap"), "unexpected error: {err}");

    // Mutation: a region on a 64 B boundary rather than 128.
    let misaligned = ScratchPlanForTest::misalign(&plan);
    let err = misaligned
        .check_disjoint()
        .expect_err("a 64 B aligned region was accepted");
    assert!(err.contains("aligned"), "unexpected error: {err}");

    // And the untouched plan is still good, so the mutations are the cause.
    plan.check_disjoint().expect("the original plan changed");
    let _ = &mut plan;
}

/// Helpers that build deliberately-broken plans, so the checks have something
/// to reject. Kept beside the test that uses them rather than exported.
struct ScratchPlanForTest;

impl ScratchPlanForTest {
    fn overlap(plan: &ScratchPlan) -> ScratchPlan {
        let mut regions: Vec<ScratchRegion> = plan.regions().to_vec();
        regions[1].offset = regions[0].offset + ARENA_ALIGNMENT;
        rebuild(plan, regions)
    }

    fn misalign(plan: &ScratchPlan) -> ScratchPlan {
        let mut regions: Vec<ScratchRegion> = plan.regions().to_vec();
        regions[0].offset = 64;
        rebuild(plan, regions)
    }
}

fn rebuild(plan: &ScratchPlan, regions: Vec<ScratchRegion>) -> ScratchPlan {
    ScratchPlan::from_parts(
        regions,
        plan.sizing(),
        plan.layout(),
        plan.combine_order(),
        plan.kv_len(),
        plan.live_chunks(),
    )
}

// ---------------------------------------------------------------------------
// The three policies
// ---------------------------------------------------------------------------

/// All three policies compile, are enumerable, and none is chosen.
///
/// Written as an exhaustive `match` so it cannot fall behind the enum -- #58's
/// mechanism (§11.3i): a policy added without an arm is `error[E0004]`, not a
/// silently unchecked variant.
#[test]
fn every_sizing_policy_is_in_all() {
    for s in Sizing::ALL {
        // Exhaustive: adding a variant without extending this fails to compile.
        let suffix = match s {
            Sizing::Reserve => "reserve",
            Sizing::Grow => "grow",
            Sizing::Bucket => "bucket",
        };
        assert_eq!(s.suffix(), suffix);
    }
    assert_eq!(Sizing::ALL.len(), 3);
    // Distinct suffixes, or two policies would resolve to one kernel.
    let mut seen: Vec<&str> = Sizing::ALL.iter().map(|s| s.suffix()).collect();
    seen.sort_unstable();
    seen.dedup();
    assert_eq!(seen.len(), 3, "two policies share a name suffix");
}

/// The three policies size the same `kv_len` differently, which is the whole
/// point of carrying three.
///
/// Reserve holds the max-context figure regardless of `kv_len`; Bucket holds the
/// rung; Grow holds exactly what the step needs. At a short `kv_len` that is a
/// 46x spread, and it is the quantity an A/B at long context would resolve.
#[test]
fn the_policies_differ_in_what_they_reserve() {
    let g = lfm2();
    let kv = 2_720; // the largest ever measured (§13.2)
    let mut sizes = Vec::new();
    for s in Sizing::ALL {
        let plan = plan_scratch(
            &g,
            kv,
            LFM2_ATTENTION_LAYERS,
            s,
            ScratchLayout::Planes,
            MAX_CONTEXT,
        )
        .expect("plan");
        sizes.push((s, plan.arena_bytes(), plan.reserved_waste(&g)));
    }

    let grow = sizes.iter().find(|(s, _, _)| *s == Sizing::Grow).unwrap();
    let bucket = sizes.iter().find(|(s, _, _)| *s == Sizing::Bucket).unwrap();
    let reserve = sizes
        .iter()
        .find(|(s, _, _)| *s == Sizing::Reserve)
        .unwrap();

    assert_eq!(grow.2, 0, "Grow reserved bytes it does not need");
    assert!(
        grow.1 < bucket.1 && bucket.1 < reserve.1,
        "expected Grow < Bucket < Reserve at kv_len {kv}, got {sizes:?}"
    );
    // Reserve at 128k against Grow at 2720: the spread the issue predicts.
    assert!(
        reserve.1 / grow.1 >= 40,
        "Reserve/Grow is {}x at kv_len {kv}; expected the ~46x the chunk ratio gives",
        reserve.1 / grow.1
    );
}

/// **`Grow` rebinds and the other two do not**, reported as a value.
///
/// A realloc changes buffer identity, which is the thing #69 exists to prevent
/// (§9.2c: "674 varying identities -> 0"). A policy whose cost is invisible in
/// the type is a policy that gets chosen by accident, so the cost is a method
/// rather than a paragraph.
#[test]
fn only_grow_rebinds_on_growth() {
    assert!(Sizing::Grow.rebinds_on_growth());
    assert!(!Sizing::Reserve.rebinds_on_growth());
    assert!(!Sizing::Bucket.rebinds_on_growth());
}

/// **`sized_chunks` is the policy, and it is reachable without a plan** (#234).
///
/// The arithmetic was four lines inside `plan_scratch` until #234, which is why
/// the axis had three compiled arms and no consumer: a caller allocating **one
/// region per call** — FlashDecoding's, §9.1a's first real consumer — needs the
/// chunk count and none of the offsets a plan carries. Splitting the two is
/// what makes the policy reachable from an op; `plan_scratch` routes through
/// the same function, so the two cannot disagree about what an arm means.
///
/// Asserted per arm rather than by a total, so a policy that stopped
/// discriminating is named rather than absorbed.
#[test]
fn sized_chunks_is_the_policy_each_arm_names() {
    // Rungs in CHUNKS, which is what a caller holding a chunk size converts
    // `BUCKET_LADDER` to. 8k/32k/128k at page size 256.
    let rungs = [32usize, 128, 512];

    // `Grow` is exactly the step, at every input: the arm with no reservation
    // and therefore the one whose region moves.
    for live in [0usize, 1, 7, 11, 512] {
        assert_eq!(Sizing::Grow.sized_chunks(live, 512, &rungs), Ok(live));
    }

    // `Reserve` is the maximum whatever the step needs, which is the property
    // it trades bytes for: the region is one size for the process.
    for live in [0usize, 1, 11, 512] {
        assert_eq!(Sizing::Reserve.sized_chunks(live, 512, &rungs), Ok(512));
    }

    // `Bucket` is the smallest rung that covers the step. The boundary cases
    // are the ones a ladder gets wrong: exactly a rung takes that rung, one
    // past it takes the next.
    assert_eq!(Sizing::Bucket.sized_chunks(1, 512, &rungs), Ok(32));
    assert_eq!(Sizing::Bucket.sized_chunks(32, 512, &rungs), Ok(32));
    assert_eq!(Sizing::Bucket.sized_chunks(33, 512, &rungs), Ok(128));
    assert_eq!(Sizing::Bucket.sized_chunks(512, 512, &rungs), Ok(512));

    // No arm may reserve LESS than the step computes, at any input. That is the
    // invariant the partial pass rests on: a region sized for fewer chunks than
    // are dispatched is an overrun, and §3.5 says nothing reports it.
    for arm in Sizing::ALL {
        for live in [0usize, 1, 7, 32, 33, 128, 512] {
            if let Ok(sized) = arm.sized_chunks(live, 512, &rungs) {
                assert!(
                    sized >= live,
                    "{arm:?} sized {sized} for a step computing {live}"
                );
            }
        }
    }
}

/// **Both reserving arms refuse rather than rounding down** (#234).
///
/// A region sized for fewer chunks than the step writes is an overrun, which
/// §3.5 makes silent corruption rather than an error — so the failure is a
/// returned `Err` at a known site. `Grow` has nothing to refuse, and asserting
/// that is what makes this a discrimination rather than a check that everything
/// fails (§8.1g's own lesson: a suite testing only the refusal passes under a
/// mutation that refuses everything).
#[test]
fn a_step_past_the_reservation_is_refused_and_grow_has_nothing_to_refuse() {
    let rungs = [32usize, 128, 512];

    assert!(
        Sizing::Reserve.sized_chunks(513, 512, &rungs).is_err(),
        "Reserve must refuse a step past its configured maximum"
    );
    assert!(
        Sizing::Bucket.sized_chunks(513, 512, &rungs).is_err(),
        "Bucket must refuse a step past its last rung"
    );

    // The other bound, and it is the load-bearing half: an arm that refused
    // everything would pass the two assertions above.
    assert!(Sizing::Reserve.sized_chunks(512, 512, &rungs).is_ok());
    assert!(Sizing::Bucket.sized_chunks(512, 512, &rungs).is_ok());
    // `Grow` reserves what the step needs, so no input is past its reservation
    // — including one that both others refuse.
    assert_eq!(Sizing::Grow.sized_chunks(513, 512, &rungs), Ok(513));
}

/// Under `Reserve` the offsets do not move as `kv_len` grows. That is the
/// property it trades bytes for.
#[test]
fn reserve_holds_its_offsets_as_kv_len_grows() {
    let g = lfm2();
    let mut first: Option<Vec<usize>> = None;
    for kv in [256usize, 2_720, 32_768, 131_072] {
        let plan = plan_scratch(
            &g,
            kv,
            LFM2_ATTENTION_LAYERS,
            Sizing::Reserve,
            ScratchLayout::Planes,
            MAX_CONTEXT,
        )
        .expect("plan");
        let offsets: Vec<usize> = plan.regions().iter().map(|r| r.offset).collect();
        match &first {
            None => first = Some(offsets),
            Some(f) => assert_eq!(
                &offsets, f,
                "Reserve moved its offsets between kv_len values"
            ),
        }
        // And the live chunk count still tracks kv_len, so the *dispatch* knows
        // how much of the region is real.
        assert_eq!(plan.live_chunks(), g.chunks(kv));
    }
}

/// Under `Grow` the offsets do move -- the counterpart, so the test above is not
/// passing because every policy is static.
#[test]
fn grow_moves_its_offsets_as_kv_len_grows() {
    let g = lfm2();
    let short = plan_scratch(
        &g,
        256,
        LFM2_ATTENTION_LAYERS,
        Sizing::Grow,
        ScratchLayout::Planes,
        MAX_CONTEXT,
    )
    .expect("plan");
    let long = plan_scratch(
        &g,
        32_768,
        LFM2_ATTENTION_LAYERS,
        Sizing::Grow,
        ScratchLayout::Planes,
        MAX_CONTEXT,
    )
    .expect("plan");
    let a: Vec<usize> = short.regions().iter().map(|r| r.offset).collect();
    let b: Vec<usize> = long.regions().iter().map(|r| r.offset).collect();
    assert_ne!(a, b, "Grow did not move its offsets, so it is not Grow");
}

/// `Bucket` holds its offsets *within* a rung and moves between rungs.
#[test]
fn bucket_is_stable_within_a_rung_and_moves_between_them() {
    let g = lfm2();
    let plan_at = |kv| {
        plan_scratch(
            &g,
            kv,
            LFM2_ATTENTION_LAYERS,
            Sizing::Bucket,
            ScratchLayout::Planes,
            MAX_CONTEXT,
        )
        .expect("plan")
    };
    // Both inside the 8k rung.
    assert_eq!(plan_at(1_000).arena_bytes(), plan_at(8_192).arena_bytes());
    // Across into the 32k rung.
    assert!(plan_at(8_193).arena_bytes() > plan_at(8_192).arena_bytes());
    assert_eq!(plan_at(8_193).arena_bytes(), plan_at(32_768).arena_bytes());
}

/// A `kv_len` past the last rung is **refused, not rounded down**.
///
/// Rounding down would size a region for fewer chunks than the step writes --
/// an overrun, and §3.5 says nothing catches it. Refusing costs a caller an
/// error; rounding costs silent corruption.
#[test]
fn a_kv_len_past_the_last_bucket_is_refused() {
    let g = lfm2();
    let err = plan_scratch(
        &g,
        BUCKET_LADDER[BUCKET_LADDER.len() - 1] + 1,
        LFM2_ATTENTION_LAYERS,
        Sizing::Bucket,
        ScratchLayout::Planes,
        MAX_CONTEXT,
    )
    .expect_err("a kv_len past the ladder was accepted");
    assert!(err.contains("bucket"), "unexpected error: {err}");

    // And Reserve refuses past its configured max, for the same reason.
    let err = plan_scratch(
        &g,
        MAX_CONTEXT + 1,
        LFM2_ATTENTION_LAYERS,
        Sizing::Reserve,
        ScratchLayout::Planes,
        MAX_CONTEXT,
    )
    .expect_err("a kv_len past max context was accepted");
    assert!(err.contains("max context"), "unexpected error: {err}");
}

// ---------------------------------------------------------------------------
// The merge order
// ---------------------------------------------------------------------------

/// **The combine merges in index order, asserted rather than assumed** (§10.4,
/// §2.3.3 #1).
///
/// Online softmax is associative in R and not in floating point, so a
/// completion-ordered merge gives different bits run to run -- and §2.3.2 says
/// that symptom is indistinguishable from a missing fence. The order is a value
/// so it can be checked; `CombineOrder::Completion` exists so the check has
/// something to reject.
#[test]
fn the_combine_order_is_index_and_completion_is_refused() {
    let g = lfm2();
    let plan = plan_scratch(
        &g,
        2_720,
        LFM2_ATTENTION_LAYERS,
        Sizing::Reserve,
        ScratchLayout::Planes,
        MAX_CONTEXT,
    )
    .expect("plan");

    assert_eq!(plan.combine_order(), CombineOrder::Index);
    plan.check_index_ordered()
        .expect("an index-ordered plan was rejected");

    // The merge sequence is ascending and covers every live chunk exactly once.
    let seq = plan.merge_sequence();
    assert_eq!(seq.len(), plan.live_chunks());
    assert!(
        seq.windows(2).all(|w| w[0] < w[1]),
        "the merge sequence is not ascending: {seq:?}"
    );
    // The absolute half. `len` and `windows(2)` are both SHAPES -- they
    // constrain the relationship between members and are blind to a defect
    // shifting the whole sequence (§3.1b). This line is the one that is not:
    // under `merge_sequence` mutated to `(1..=live_chunks)` it reads
    // `Some(1)` against `Some(0)` and kills it, where the two above stay
    // green. Recorded because #524 measured it doing so.
    assert_eq!(seq.first().copied(), Some(0));

    // Mutation: the check must reject the other order. Without this the
    // assertion above is a tautology over a one-variant type.
    let completion = ScratchPlan::from_parts(
        plan.regions().to_vec(),
        plan.sizing(),
        plan.layout(),
        CombineOrder::Completion,
        plan.kv_len(),
        plan.live_chunks(),
    );
    let err = completion
        .check_index_ordered()
        .expect_err("a completion-ordered plan was accepted");
    assert!(err.contains("nondeterministic"), "unexpected error: {err}");

    // The SECOND wrong order, and it is not completion order (§10.3h).
    let table_order = ScratchPlan::from_parts(
        plan.regions().to_vec(),
        plan.sizing(),
        plan.layout(),
        CombineOrder::TableOrder,
        plan.kv_len(),
        plan.live_chunks(),
    );
    let err = table_order
        .check_index_ordered()
        .expect_err("a table-ordered plan was accepted");
    assert!(
        err.contains("INDEXED") && err.contains("WALKED"),
        "unexpected error: {err}"
    );
}

/// `TableOrder` is identical to `Index` at B=1 and diverges on a recycled pool.
///
/// # Why this test is the point of building the arm at all
///
/// §10.3h specifies `CombineOrder::TableOrder` as a second negative control and
/// #181 recorded it as **the only one of six deliberately-wrong arms this
/// document names that was never built**, deferred on the ground that *"the arm
/// it controls for is indistinguishable from `Index` at B=1"*. That ground is
/// real, and it is exactly what makes a refusal alone insufficient: a control
/// that is only ever rejected leaves *"it would diverge elsewhere"* as an
/// assertion nobody has evaluated.
///
/// So this evaluates it, on both sides:
///
/// - on the **identity** table a fresh pool produces, the two orders agree
///   exactly — which is the hazard, stated as a measurement rather than as a
///   warning;
/// - on the **non-identity** table an ordinary allocate/release/reallocate
///   history produces, they disagree — which is what a B=1 gate with a fresh
///   pool structurally cannot show.
///
/// The tables come from [`KvPagePool`] rather than from literals, because a
/// hand-written permutation would demonstrate that *a* permuted table diverges
/// and not that the allocator **produces** one (§10.3i's sequencing argument).
#[test]
fn table_order_coincides_with_index_at_the_identity_and_diverges_after_recycling() {
    use super::kv_pages::KvPagePool;

    let g = lfm2();
    let plan = plan_scratch(
        &g,
        2_720,
        LFM2_ATTENTION_LAYERS,
        Sizing::Reserve,
        ScratchLayout::Planes,
        MAX_CONTEXT,
    )
    .expect("plan");
    // Two chunks, which is the smallest fixture in which an order can differ at
    // all -- with one chunk every order is the same order.
    let plan = ScratchPlan::from_parts(
        plan.regions().to_vec(),
        plan.sizing(),
        plan.layout(),
        CombineOrder::Index,
        plan.kv_len(),
        2,
    );

    // A fresh pool: the table is the identity and the two orders COINCIDE.
    let mut fresh = KvPagePool::new(4, 8, 4, 64);
    fresh.append(0, 8).unwrap();
    let identity = fresh.table_for(0, 1);
    assert_eq!(identity, vec![0, 1], "a fresh pool must hand out 0, 1");
    assert_eq!(
        plan.merge_sequence_under(CombineOrder::TableOrder, &identity),
        plan.merge_sequence_under(CombineOrder::Index, &identity),
        "at page_table[c] == c the two orders MUST agree -- if they did not, this test would \
         be measuring something other than the hazard DESIGN.md 10.3h describes"
    );

    // A recycled pool: the table is not the identity and they DIVERGE.
    let mut recycled = KvPagePool::new(4, 8, 4, 64);
    recycled.append(0, 8).unwrap();
    recycled.append(1, 8).unwrap();
    let freed = recycled.release(0);
    recycled.recycle(&freed).unwrap();
    recycled.append(2, 8).unwrap();
    let permuted = recycled.table_for(2, 1);
    assert_ne!(
        permuted, identity,
        "the fixture is degenerate: recycling reproduced the identity table, so the \
         divergence below could not be observed"
    );

    let indexed = plan
        .merge_sequence_under(CombineOrder::Index, &permuted)
        .expect("Index always has a sequence");
    let walked = plan
        .merge_sequence_under(CombineOrder::TableOrder, &permuted)
        .expect("TableOrder has a sequence for a given history");
    assert_eq!(
        indexed,
        vec![0, 1],
        "the indexed order is a property of the LOOP BOUND and must not move with the table"
    );
    assert_eq!(
        walked,
        vec![1, 0],
        "walking the table must follow its contents -- this is the order that makes a \
         sequence's logits depend on what its batch neighbours allocated (DESIGN.md 2.3.3 #7)"
    );
    assert_ne!(
        indexed, walked,
        "the two orders must differ on a recycled pool, or TableOrder controls for nothing"
    );

    // Completion has no deterministic sequence by construction, and is reported
    // as absent rather than given an invented one (§15.1a).
    assert_eq!(
        plan.merge_sequence_under(CombineOrder::Completion, &permuted),
        None
    );
}

/// **`kv_len` determines the merge tree, so the same `kv_len` always merges the
/// same way** (§10.4's corollary).
///
/// The chunk count must not depend on load, occupancy, or anything else
/// dynamic. Nothing in [`plan_scratch`] consults such a thing; this states it as
/// a property so a future change that did would be caught.
///
/// **The absolute assertion below is what makes that last sentence true, and it
/// was added after the sentence was measured false (#524).** Comparing the eight
/// plans to *each other* takes iteration 0 of this very loop as the reference,
/// which is `CONTRIBUTING.md` §3.1b's third place for a reference to be wrong --
/// *the same run*, which looks like a comparison and is not. Demonstrated rather
/// than argued: mutating `merge_sequence` from `(0..live_chunks)` to
/// `(1..=live_chunks)` shifts every element while preserving ascendingness,
/// length and run-to-run identity, and this test **passed on `[1,2,..,11]` while
/// claiming to pin `[0,1,..,10]`** -- under a real defect, the combine skipping
/// chunk 0 and reading one past the end, which is §10.4's silent-wrong-answer
/// class.
///
/// The class was caught anyway, by
/// [`the_combine_order_is_index_and_completion_is_refused`] and
/// [`one_chunk_and_zero_chunks_are_degenerate_cases_not_special_ones`] -- two
/// tests with **no connection to this one**, in the same file by accident of
/// subject. That is coverage by *incident*: delete either sibling in a refactor
/// and the class goes live with nothing saying so, and a refactor removing a
/// redundant-looking degenerate-case test is exactly the kind nobody re-runs a
/// mutation harness for. The line below makes the coverage **intrinsic**, so
/// this test now kills that mutation on its own.
#[test]
fn the_same_kv_len_always_gives_the_same_merge_sequence() {
    let g = lfm2();
    for s in Sizing::ALL {
        let mut first: Option<Vec<usize>> = None;
        for _ in 0..8 {
            let plan = plan_scratch(
                &g,
                2_720,
                LFM2_ATTENTION_LAYERS,
                s,
                ScratchLayout::Planes,
                MAX_CONTEXT,
            )
            .expect("plan");
            let seq = plan.merge_sequence();
            // The reference is ABSOLUTE -- a value this loop did not produce --
            // rather than iteration 0 of the loop. §3.1b rule 1: prefer an
            // absolute reference to a shape, because it names *why*.
            assert_eq!(
                seq,
                (0..plan.live_chunks()).collect::<Vec<_>>(),
                "{s:?} merged {seq:?} rather than 0..{}",
                plan.live_chunks()
            );
            match &first {
                None => first = Some(seq),
                Some(f) => assert_eq!(&seq, f, "{s:?} produced a different merge sequence"),
            }
        }
    }
}

/// The merge sequence follows the *live* chunk count, not the reserved one.
///
/// Under `Reserve` a region is sized for 512 chunks at 128k while a 2720-token
/// step has 11 live. Merging over the reserved count would fold in 501 chunks of
/// uninitialised memory -- a silent wrong answer, and one that no size check
/// would catch.
#[test]
fn the_merge_covers_live_chunks_not_reserved_ones() {
    let g = lfm2();
    let plan = plan_scratch(
        &g,
        2_720,
        LFM2_ATTENTION_LAYERS,
        Sizing::Reserve,
        ScratchLayout::Planes,
        MAX_CONTEXT,
    )
    .expect("plan");
    assert_eq!(plan.live_chunks(), 11);
    assert_eq!(plan.regions()[0].chunks, 512, "reserved for max context");
    assert_eq!(
        plan.merge_sequence().len(),
        11,
        "the merge walked reserved chunks rather than live ones"
    );
}

// ---------------------------------------------------------------------------
// Degenerate cases
// ---------------------------------------------------------------------------

/// §10.6: contiguous is paged with one page, single-pass is FlashDecoding with
/// one chunk. Both fall out rather than needing their own path -- §15.2 #14: if
/// a "simple" mode needs its own kernel the abstraction is at the wrong level.
#[test]
fn one_chunk_and_zero_chunks_are_degenerate_cases_not_special_ones() {
    let g = lfm2();
    assert_eq!(g.chunks(1), 1, "a single token is one chunk");
    assert_eq!(g.chunks(256), 1, "a full page is one chunk");
    assert_eq!(g.chunks(257), 2);
    assert_eq!(g.chunks(0), 0, "an empty cache reserves nothing");

    let plan =
        plan_scratch(&g, 1, 1, Sizing::Grow, ScratchLayout::Planes, MAX_CONTEXT).expect("plan");
    assert_eq!(plan.live_chunks(), 1);
    // `vec![0]` is an ABSOLUTE reference and the degenerate case is where it is
    // sharpest: a merge sequence off by one has nowhere to hide at length 1.
    // Measured (#524): under `merge_sequence` mutated to `(1..=live_chunks)`
    // this reads `[1]` against `[0]` and kills it. Not the point of this test --
    // §10.6's degenerate-case claim is -- and worth not losing in a refactor
    // that reads it as redundant with the ascending check above.
    assert_eq!(plan.merge_sequence(), vec![0]);
}

/// The request sizes and expected offsets are shaped for #70's GPU bump
/// allocator, so the scratch arena can be served by the allocator that already
/// exists and is already verified rather than by a second one.
#[test]
fn the_plan_exports_what_the_gpu_bump_allocator_consumes() {
    let g = lfm2();
    let plan = plan_scratch(
        &g,
        2_720,
        LFM2_ATTENTION_LAYERS,
        Sizing::Grow,
        ScratchLayout::Planes,
        MAX_CONTEXT,
    )
    .expect("plan");

    let sizes = plan.request_sizes();
    let offsets = plan.expected_offsets();
    assert_eq!(sizes.len(), LFM2_ATTENTION_LAYERS);
    assert_eq!(offsets.len(), LFM2_ATTENTION_LAYERS);

    // A forward-only cursor reproduces them: strictly increasing offsets, each
    // the previous rounded up. That is the property #70's
    // `is_bump_reproducible` checks for the activation arena, here by
    // construction because regions are laid out in order and never packed.
    let mut cursor = 0usize;
    for (i, (&size, off)) in sizes.iter().zip(offsets.iter()).enumerate() {
        assert_eq!(
            off,
            &Some(cursor),
            "region {i} is not where a cursor puts it"
        );
        cursor = super::arena::align_up(cursor + size as usize, ARENA_ALIGNMENT);
    }
    assert_eq!(cursor, plan.bump_capacity());
}
