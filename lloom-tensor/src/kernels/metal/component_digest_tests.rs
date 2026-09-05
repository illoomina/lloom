//! Component digests: one stable fingerprint per mechanism per variant, in the
//! normal test suite.
//!
//! **This is the cheap layer of the two #105 specifies, and it is the one that
//! makes the expensive layer small.** A probe on a `bench/*` branch rots the
//! moment the tip moves -- 29 of 36 such branches are more than 20 commits
//! behind -- where a digest in the suite runs on every commit. If each mechanism
//! is pinned individually, an end-to-end run exists only to catch
//! **interactions**, which is the only thing it can tell you that a component
//! test cannot.
//!
//! # What a component digest is, and what it is not
//!
//! It is a hash over the *output of one mechanism on a fixed input*, recorded
//! per variant of that mechanism's axis. It answers "did this component change
//! bits?" -- today a multi-turn model run costing minutes and an exclusive GPU
//! lease, here a few microseconds of pure arithmetic.
//!
//! It is **not** a correctness test. A digest can be stable and wrong; §2.3.8c
//! makes the point at the model scale, and it holds here. The property tests
//! beside these say the plan is *right*; these say it has not *moved*. Both are
//! needed and neither substitutes for the other, which is why these are added
//! beside the existing arena tests rather than replacing any of them.
//!
//! # Every digest asserts non-vacuity beside it
//!
//! #53 established the rule for parity arms and §3.7a says why it is not
//! hypothetical: *"status=completed, error=nil, output all zeros"* is the ICB
//! path's characteristic failure, and **two variants that both compute nothing
//! agree perfectly**. A digest over an empty or constant output is a number that
//! cannot fail.
//!
//! The predicate is `distinct >= 2`, not `!= 0`, for #53's reason: zero is a
//! legitimate output for an index-returning kernel, and a guard that fails a
//! correct test gets weakened. Counting distinct values catches an all-zero
//! output *and* a one-constant-everywhere output at the same cost.
//!
//! Here the guard is stronger than a value count alone, because a plan has
//! structure a hash flattens. `assert_digest` requires the digested *record
//! stream* to carry at least two distinct records, so a planner that returned
//! one slot repeated, or nothing at all, fails the guard rather than producing a
//! stable fingerprint of emptiness.
//!
//! # Why the expected digests are written down
//!
//! A digest nobody compares against is a log line. These are pinned as literals
//! so a change fails **where the change was made**, rather than three merges
//! later in an end-to-end digest -- which is the whole argument for the layer.
//! When a change legitimately moves one, §2.3.5a applies: predict which digest
//! should move and why *before* running, then update the literal in the same
//! commit as the change that moved it.

use super::arena::{plan_from_sizes, ArenaLayout, StepPlan, ARENA_ALIGNMENT};

/// A digest over a stream of records, with the record count that produced it.
///
/// The count travels with the hash so the non-vacuity guard has something to
/// check: a hash alone cannot distinguish "hashed nothing" from "hashed
/// something that happened to be empty".
struct Digest {
    hex: String,
    /// Distinct records seen. The non-vacuity quantity.
    distinct: usize,
}

/// FNV-1a over the record stream.
///
/// Deliberately **not** SHA-256. A component digest is compared against a
/// literal in this file by a human reading a diff; the cryptographic properties
/// buy nothing here, and pulling a hash dependency into `candle-metal-kernels`
/// for a test would be a real cost for no gain. What is needed is that a
/// one-byte change to the record stream changes the output, which FNV-1a gives.
///
/// Fixed-width big-endian encoding per field, so the digest cannot be changed by
/// a platform's word size or endianness -- a digest that differs between
/// machines is worse than none, because the first person to see it assumes a
/// regression.
fn digest_records(records: &[Vec<u64>]) -> Digest {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for rec in records {
        for field in rec {
            for b in field.to_be_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x1000_0000_01b3);
            }
        }
        // A record separator, so `[[1], [2, 3]]` and `[[1, 2], [3]]` do not
        // collide. Without it the digest is a hash of the concatenation and the
        // structure it exists to pin is invisible.
        h ^= 0xff;
        h = h.wrapping_mul(0x1000_0000_01b3);
    }
    let distinct = records
        .iter()
        .collect::<std::collections::HashSet<_>>()
        .len();
    Digest {
        hex: format!("{h:016x}"),
        distinct,
    }
}

/// Assert a component digest, and that the comparison was not vacuous.
///
/// **The only way a component digest is checked.** Both assertions or neither:
/// there is no argument that disables the guard, for the same reason
/// `assert_packed_matches_split` offers none (#53).
///
/// `what` names the mechanism and its variant, so a failure says which
/// component moved without the reader mapping a test name onto an axis.
#[track_caller]
fn assert_digest(what: &str, records: &[Vec<u64>], expected: &str) {
    let d = digest_records(records);

    // The guard, first: a vacuous stream makes the equality below meaningless,
    // so reporting emptiness is more useful than reporting a hash mismatch.
    assert!(
        d.distinct >= 2,
        "{what} digested {} distinct record(s) over {} -- the digest is vacuous. \
         Two variants that both compute nothing agree perfectly, and §3.7a records \
         all-zero output as the ICB path's characteristic failure. Fix the \
         *input* so a correct mechanism produces a varied output; do not relax \
         this check.",
        d.distinct,
        records.len(),
    );

    assert_eq!(
        d.hex, expected,
        "{what} digest moved: {} -> {}. This is a component digest, so the change \
         is in this mechanism rather than downstream of it. If the move is \
         intended, §2.3.5a applies: state which digest should move and why before \
         running, prove the new behaviour against a reference, and update the \
         literal in the same commit as the change that moved it.",
        expected, d.hex,
    );
}

/// The fixed input every arena digest is taken over.
///
/// **Deliberately not LFM2's shapes**, and that is the load-bearing choice.
/// §9.2c records that every LFM2 decode activation is a multiple of 128 B, so
/// alignment is a no-op on every shape our own model produces -- deleting
/// `align_up` left every LFM2-shaped offset unchanged and #70's acceptance test
/// **passed under that mutation**. A fixture built only from the model's own
/// shapes is blind to a whole defect class.
///
/// So the sizes below include values that are *not* 128-multiples (100, 300,
/// 5000, 21503), at the level where the rounding happens (§9.1a narrows #70's
/// warning to exactly this: the fixture must be unaligned where `align_up` is
/// applied, since an aggregate built from padded records inherits their
/// padding).
///
/// The liveness pattern mixes disjoint and overlapping intervals so the packed
/// layout has something to pack and the reference has something to refuse to
/// pack -- if they ever tie, the reference has stopped being a reference.
const FIXTURE_SIZES: [usize; 8] = [100, 4096, 300, 21503, 5000, 4096, 21504, 128];
const FIXTURE_LAST_USE: [usize; 8] = [1, 3, 4, 5, 6, 7, 7, 7];

/// A plan as a record stream: one record per slot, then one per ordinal.
///
/// Both halves are digested because they can move independently. Two plans can
/// agree on total bytes while assigning different ordinals to different slots --
/// §9.2c's start-major-vs-size-major result is exactly that shape, 68 KB against
/// 85 KB from assignment order alone -- so a digest over the byte total would
/// miss the thing most worth pinning.
fn plan_records(plan: &StepPlan) -> Vec<Vec<u64>> {
    let mut records: Vec<Vec<u64>> = Vec::new();
    for (i, s) in plan.slots().iter().enumerate() {
        // Tagged `0` so a slot record and an ordinal record cannot collide even
        // if their numbers coincide.
        records.push(vec![0, i as u64, s.offset as u64, s.size as u64]);
    }
    for ord in 0..plan.allocations() {
        // `None` -- session state, kept in the sequence so excluding it does not
        // renumber later ordinals -- is encoded distinctly from slot 0 rather
        // than skipped, because *which* ordinals are excluded is part of what
        // the plan decides (§9.2c: two detectors, neither subsuming the other).
        let slot = plan.slot_of(ord).map(|s| s as u64 + 1).unwrap_or(0);
        records.push(vec![1, ord as u64, slot]);
    }
    records
}

/// `ArenaLayout::Packed` -- the layout #69 built and ships behind `--arena`.
///
/// The digest pins slot assignment, not just size: §9.2c's size-major result is
/// a 1.25x difference in peak that comes *entirely* from assignment order, so a
/// planner that silently reverted to start order would keep every property test
/// green and move this digest.
#[test]
fn arena_plan_packed_digest() {
    let plan = plan_from_sizes(&FIXTURE_SIZES, &FIXTURE_LAST_USE, ArenaLayout::Packed);
    assert_digest(
        "ArenaLayout::Packed plan",
        &plan_records(&plan),
        PACKED_PLAN_DIGEST,
    );
}

/// `ArenaLayout::NonAliasing` -- §9.3's reference layout, the oracle the packed
/// one is validated against.
///
/// Digested separately rather than derived from the packed one, because the two
/// are different mechanisms: the reference exists to be *independent* of the
/// packing logic, and a digest computed from the packed plan would inherit
/// exactly the fault it is meant to detect.
#[test]
fn arena_plan_non_aliasing_digest() {
    let plan = plan_from_sizes(&FIXTURE_SIZES, &FIXTURE_LAST_USE, ArenaLayout::NonAliasing);
    assert_digest(
        "ArenaLayout::NonAliasing plan",
        &plan_records(&plan),
        NON_ALIASING_PLAN_DIGEST,
    );
}

/// The two layouts must not produce the same digest on this fixture.
///
/// A component digest per variant is only informative if the variants differ
/// *on the fixture the digest is taken over*. If packing and the reference
/// agreed here, both digests would be pinning one behaviour under two names and
/// a packing regression would move neither.
///
/// This is the same argument `non_aliasing_reference_never_shares_bytes` makes
/// about bytes, applied to the digest: it checks that the **fixture
/// discriminates**, which is a property of the test data rather than of the
/// planner.
#[test]
fn the_two_layouts_digest_differently_on_this_fixture() {
    let packed = digest_records(&plan_records(&plan_from_sizes(
        &FIXTURE_SIZES,
        &FIXTURE_LAST_USE,
        ArenaLayout::Packed,
    )));
    let reference = digest_records(&plan_records(&plan_from_sizes(
        &FIXTURE_SIZES,
        &FIXTURE_LAST_USE,
        ArenaLayout::NonAliasing,
    )));
    assert_ne!(
        packed.hex, reference.hex,
        "the packed and reference layouts digest identically on this fixture, so \
         neither digest discriminates between them and a packing regression would \
         move neither. Fix the fixture, not the assertion."
    );
}

/// The fixture reaches the alignment path.
///
/// **The fixture's own guard**, and it exists because of a measured near-miss:
/// deleting `align_up` left every LFM2-shaped offset unchanged and #70's
/// acceptance test passed under that mutation (§9.2c, §9.2g). A fixture whose
/// sizes are all 128-multiples digests the identity function and reports
/// nothing.
///
/// Asserted rather than commented so the fixture cannot drift into alignment
/// later: someone tidying `FIXTURE_SIZES` into round numbers turns this red
/// instead of silently disarming both digests above.
#[test]
fn the_fixture_is_unaligned_where_the_rounding_happens() {
    let unaligned = FIXTURE_SIZES
        .iter()
        .filter(|s| *s % ARENA_ALIGNMENT != 0)
        .count();
    assert!(
        unaligned >= 2,
        "only {unaligned} of the fixture's sizes are not {ARENA_ALIGNMENT} B \
         multiples, so the digests barely exercise `align_up`. §9.2c: a parity \
         test built only from the model's own shapes is blind to a whole defect \
         class, because every LFM2 decode activation is already aligned."
    );

    // And the alignment is actually applied: every slot offset is aligned even
    // though the sizes that produced them are not. This is what makes the count
    // above meaningful rather than decorative.
    let plan = plan_from_sizes(&FIXTURE_SIZES, &FIXTURE_LAST_USE, ArenaLayout::Packed);
    for (i, s) in plan.slots().iter().enumerate() {
        assert_eq!(
            s.offset % ARENA_ALIGNMENT,
            0,
            "slot {i} at offset {} is not {ARENA_ALIGNMENT} B aligned",
            s.offset,
        );
    }
}

/// A mutation that changes the plan must change the digest.
///
/// **The digest's own mutation test.** A digest that cannot fail is not a
/// digest, and the failure mode is specific here: `digest_records` hashes a
/// record stream, and a hash that ignored the record structure would be stable
/// under a reordering that changes which ordinal lands in which slot. §3.1's
/// mutation requirement, applied to the instrument rather than to a kernel.
///
/// Two mutations, because they break different things. Perturbing a *size*
/// changes what is packed; perturbing a *liveness interval* changes what may
/// share. A digest sensitive to one and blind to the other would pass a
/// single-mutation test and still miss half the mechanism.
#[test]
fn a_changed_plan_changes_the_digest() {
    let base = digest_records(&plan_records(&plan_from_sizes(
        &FIXTURE_SIZES,
        &FIXTURE_LAST_USE,
        ArenaLayout::Packed,
    )));

    // The mutated ordinal must be one that *determines* a slot's size, and
    // finding that out is itself a result worth recording.
    //
    // A slot's size is the `max` over its occupants, so growing a **dominated**
    // ordinal changes nothing about the plan -- the first draft of this test
    // perturbed `FIXTURE_SIZES[0]` (100 B, sharing a slot with 4096 B) and
    // failed, correctly: the digest had not gone blind, the plan genuinely does
    // not move. That is a property of arena packing, not a defect, and a test
    // asserting otherwise would have been a test asserting a falsehood.
    //
    // Ordinal 6 is the fixture's largest (21504 B) and so is its slot's `max`
    // by construction. Growing it past a 128 B boundary moves the slot, and
    // moves every offset after it.
    let largest = FIXTURE_SIZES
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| **s)
        .map(|(i, _)| i)
        .expect("the fixture is not empty");
    let mut sizes = FIXTURE_SIZES;
    sizes[largest] += ARENA_ALIGNMENT;
    let size_mutant = digest_records(&plan_records(&plan_from_sizes(
        &sizes,
        &FIXTURE_LAST_USE,
        ArenaLayout::Packed,
    )));
    assert_ne!(
        base.hex, size_mutant.hex,
        "growing the slot-determining request size left the plan digest \
         unchanged, so the digest is not pinning what it claims to"
    );

    // Extending ordinal 0's life past ordinal 2's start makes them overlap, so
    // they can no longer share a slot. The sizes are untouched.
    let mut last_use = FIXTURE_LAST_USE;
    last_use[0] = FIXTURE_LAST_USE.len();
    let liveness_mutant = digest_records(&plan_records(&plan_from_sizes(
        &FIXTURE_SIZES,
        &last_use,
        ArenaLayout::Packed,
    )));
    assert_ne!(
        base.hex, liveness_mutant.hex,
        "extending a value's liveness so it can no longer share a slot left the \
         plan digest unchanged, so the digest sees sizes but not the packing"
    );
}

/// The guard fires on a vacuous record stream.
///
/// The control #53's argument requires: without it, "every digest asserts
/// non-vacuity" is a claim about code nobody has run in the failing direction.
/// A stream of identical records is exactly the shape two do-nothing variants
/// produce, and it must be rejected before the hash is compared.
#[test]
#[should_panic(expected = "the digest is vacuous")]
fn the_non_vacuity_guard_rejects_a_constant_record_stream() {
    let constant = vec![vec![0u64, 0, 0]; 16];
    // The expected digest is irrelevant: the guard runs first, deliberately, so
    // that emptiness is reported as emptiness rather than as a hash mismatch.
    assert_digest("a mechanism that computes nothing", &constant, "irrelevant");
}

// ---- the pinned digests ------------------------------------------------
//
// Recorded from the mechanisms as they stand at the commit that added this
// file. They are literals rather than golden files because there are two of
// them: a file would add I/O and a path to a test that is otherwise pure
// arithmetic, and `DESIGN.md` §11.3h's recurring lesson is that a number written
// once beside an artifact that later changes is a number nobody re-checks --
// which argues for putting the number where the diff shows it.

/// `ArenaLayout::Packed` over `FIXTURE_SIZES`/`FIXTURE_LAST_USE`.
const PACKED_PLAN_DIGEST: &str = "ae0f21712c4c6d23";

/// `ArenaLayout::NonAliasing` over the same fixture.
const NON_ALIASING_PLAN_DIGEST: &str = "e57b78609936ed07";
