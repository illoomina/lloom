//! Does the strict mode discriminate? (issue #185)
//!
//! **A mode that cannot report the bad case has not been shown to
//! discriminate.** #144 established why that sentence is the acceptance
//! criterion rather than a nicety: its run-heads mutation left *both* canonical
//! digest pairs byte-identical over three runs, so the project's strongest
//! outcome gate is provably blind to this class. An audit validated only against
//! a correct configuration would be a display, not a test.
//!
//! So every test below comes in pairs: the known-good predicate must report
//! **zero** uncovered edges, and a mutation of the model must report **more than
//! zero**. The mutations are the four `measurements/issue-144-raw/edge-cover.py`
//! runs, reproduced against this implementation rather than against a script:
//!
//! ```text
//!   0  PROPOSAL: suppress all 393 covered non-head   <- the question
//! 208  MUTATION: also suppress the 30 run heads      <- the bug
//! ```
//!
//! These run on a **synthetic** stream rather than a decode trace, deliberately.
//! The full-model validation is the artifact in `measurements/`, which is what
//! checks this implementation against #144's own numbers over 85595 edges; these
//! check the *mechanism* -- that each of the three primitives is modelled, that
//! each one alone is load-bearing, and that removing any of them is visible.
//! A unit test on a 554-position trace would be neither.

#![cfg(feature = "hazard-audit")]

use crate::kernels::metal::encoder::{HazardKind, HazardKinds};
use crate::kernels::metal::hazard_audit::{cover, AuditBinding, AuditDispatch};

/// A slot every dispatch in these fixtures shares, so hazards are easy to arrange.
const SLOT: usize = 0x1000;

fn bind(is_output: bool) -> AuditBinding {
    AuditBinding {
        ptr: SLOT,
        offset: 0,
        len: 128,
        is_output,
    }
}

/// A dispatch at `seq`, writing or reading the shared slot.
fn d(seq: u64, is_output: bool) -> AuditDispatch {
    AuditDispatch {
        seq,
        kernel: format!("k{seq}"),
        bindings: vec![bind(is_output)],
        barrier: false,
        kinds: HazardKinds::NONE,
        barrier_suppressed: false,
        encoder: 0,
        run: None,
        is_run_head: false,
        icb_barrier: false,
    }
}

/// The three directions are detected, and read-after-read is not one of them.
///
/// The last assertion is the load-bearing one: it is why 5.394 GB of weights,
/// bound on every dispatch and read every time, contributes zero edges.
#[test]
fn the_three_directions_are_detected_and_rar_is_not_a_hazard() {
    // write -> read is RAW.
    let raw = cover("raw", &[d(0, true), d(1, false)]);
    assert_eq!(raw.edges, 1);
    assert_eq!(raw.edges_raw, 1, "a write-then-read was not RAW");

    // write -> write is WAW.
    let waw = cover("waw", &[d(0, true), d(1, true)]);
    assert_eq!(waw.edges_waw, 1, "a write-then-write was not WAW");

    // read -> write is WAR.
    let war = cover("war", &[d(0, false), d(1, true)]);
    assert_eq!(war.edges_war, 1, "a read-then-write was not WAR");

    // read -> read is nothing at all.
    let rar = cover("rar", &[d(0, false), d(1, false)]);
    assert_eq!(
        rar.edges, 0,
        "two reads produced an edge; the weights would dominate the graph"
    );
}

/// Disjoint bytes in one allocation are not an edge, and an unknown extent is.
///
/// The asymmetry is deliberate: a spurious edge costs a false positive a human
/// reads, a missed one costs the silence this module exists to break (§3.5).
#[test]
fn overlap_decides_an_edge_and_an_unknown_extent_fails_toward_reporting() {
    let mut lo = d(0, true);
    lo.bindings = vec![AuditBinding {
        ptr: SLOT,
        offset: 0,
        len: 128,
        is_output: true,
    }];
    let mut hi = d(1, true);
    hi.bindings = vec![AuditBinding {
        ptr: SLOT,
        offset: 128,
        len: 128,
        is_output: true,
    }];
    assert_eq!(
        cover("adjacent", &[lo.clone(), hi.clone()]).edges,
        0,
        "adjacent slots in one allocation were called an edge"
    );

    // A one-byte overlap is an edge.
    let mut overlapping = hi.clone();
    overlapping.bindings[0].offset = 127;
    assert_eq!(
        cover("overlapping", &[lo.clone(), overlapping]).edges,
        1,
        "a one-byte overlap was missed"
    );

    // A zero length covers the allocation, so it orders against everything in it.
    let mut unknown = hi.clone();
    unknown.bindings[0].len = 0;
    assert_eq!(
        cover("unknown-extent", &[lo, unknown]).edges,
        1,
        "an unknown extent did not fail toward reporting"
    );

    // ... but never across allocations. Offsets are only comparable within one.
    let mut other_buffer = d(1, true);
    other_buffer.bindings = vec![AuditBinding {
        ptr: SLOT + 0x10000,
        offset: 0,
        len: 0,
        is_output: true,
    }];
    assert_eq!(
        cover("other-buffer", &[d(0, true), other_buffer]).edges,
        0,
        "a zero length crossed allocations"
    );
}

/// **Primitive 1**: a surviving classical fence orders an edge, and removing it
/// is visible.
#[test]
fn a_surviving_fence_covers_an_edge_and_its_absence_is_reported() {
    let w = d(0, true);
    let mut r = d(1, false);

    // Known-good: the reader carries candle's barrier.
    r.barrier = true;
    r.kinds = {
        let mut k = HazardKinds::NONE;
        k.insert(HazardKind::Raw);
        k
    };
    let good = cover("fence", &[w.clone(), r.clone()]);
    assert_eq!(good.by_barrier, 1);
    assert!(good.is_clean(), "a fenced edge was reported uncovered");
    assert_eq!(
        good.barriers_raw, 1,
        "the barrier was not attributed to RAW"
    );

    // Known-bad: the same stream with the fence gone. This is the shape a
    // wrongly-dropped edge has, and §3.5 makes it silent corruption.
    r.barrier = false;
    let bad = cover("no-fence", &[w, r]);
    assert_eq!(
        bad.uncovered.len(),
        1,
        "removing the only fence reported nothing -- the model is vacuous"
    );
    assert_eq!(bad.uncovered[0].kind, HazardKind::Raw);
}

/// A fence orders **across** the positions interleaved between ICB runs, which
/// `setBarrier` does not (§11.3m). The audit must model that asymmetry, since
/// it is the reason candle's barrier at a covered position is not simply
/// redundant with the ICB's.
#[test]
fn a_fence_orders_across_an_interleaved_gap_position() {
    // The gap position must touch a *different* allocation, or it forms its own
    // edge with the writer and the fixture stops testing what it names. Found
    // by the test failing: an earlier version had it read the shared slot, and
    // the reported uncovered edge was `#0 -> #1`, which is a genuine finding
    // about that stream rather than a defect in the cover test.
    let mut gap = d(1, false);
    gap.bindings = vec![AuditBinding {
        ptr: SLOT + 0x10000,
        offset: 0,
        len: 128,
        is_output: false,
    }];

    let w = d(0, true);
    let mut r = d(2, false);
    r.barrier = true;

    let rep = cover("interleaved", &[w, gap, r]);
    assert!(
        rep.is_clean(),
        "a fence failed to order an edge spanning a gap position: {:?}",
        rep.uncovered
    );
    assert_eq!(rep.by_barrier, 1, "the edge was covered by something else");
}

/// **Primitive 2**: an encoder-session seam orders an edge that no barrier
/// could, and modelling it is what stops 11 false positives.
///
/// This is #144's obstacle 3 defusing its obstacle 2. `edge-cover.py`'s
/// `MUTATION: 393, but seams do not order` reports **11**, every one out of
/// `sdpa_vector` at #249; the mutation here is the same one at unit scale.
#[test]
fn a_session_seam_covers_an_edge_and_removing_it_is_visible() {
    let w = d(0, true);
    let mut r = d(1, false);
    r.encoder = 1; // a new session: hazard state is empty, so no barrier fires
    assert!(!r.barrier, "the fixture must not also carry a fence");

    let good = cover("seam", &[w.clone(), r.clone()]);
    assert_eq!(good.by_seam, 1, "the seam did not cover the edge");
    assert!(good.is_clean());

    // The mutation: same stream, one session. Now nothing orders it.
    let mut r_same = r;
    r_same.encoder = 0;
    let bad = cover("seam-removed", &[w, r_same]);
    assert_eq!(
        bad.uncovered.len(),
        1,
        "an audit modelling only barriers would have called this correct"
    );
}

/// **Primitive 3**: the ICB's `setBarrier` covers an edge within one run, and
/// each of its four conditions is load-bearing.
///
/// `setBarrier` is a property of *one command*: it orders that command against
/// its predecessors **in its own run**, and successors inherit nothing. So an
/// audit that modelled it as a fence would accept edges Metal does not order.
#[test]
fn the_icb_set_barrier_covers_within_a_run_and_every_condition_is_load_bearing() {
    let mut w = d(0, true);
    w.run = Some(0);
    w.is_run_head = true;

    let mut r = d(1, false);
    r.run = Some(0);
    r.icb_barrier = true;
    // Candle's own barrier was suppressed here -- §11.3r's arm. That is what
    // makes the ICB's the only thing left, which is the case under test.
    r.barrier_suppressed = true;

    let good = cover("icb", &[w.clone(), r.clone()]);
    assert_eq!(good.by_icb, 1, "setBarrier did not cover an in-run edge");
    assert!(good.is_clean());
    assert_eq!(good.barriers_suppressed, 1);

    // Mutation A: the ICB emits no `setBarrier`. `edge-cover.py`'s
    // "393, but ICB emits no setBarrier" reports 1960.
    let mut no_barrier = r.clone();
    no_barrier.icb_barrier = false;
    assert_eq!(
        cover("icb-off", &[w.clone(), no_barrier]).uncovered.len(),
        1,
        "removing the ICB barrier reported nothing"
    );

    // Mutation B: the reader is a run head. Its scan slice
    // `covered[run_first_cmd..cmd_index]` is empty by construction, so its
    // command orders nothing within the run -- which is why §11.3p's 30 head
    // barriers must every one survive, and why #144's run-heads mutation is
    // the known-bad case the digest gate could not see.
    let mut head = r.clone();
    head.is_run_head = true;
    assert_eq!(
        cover("icb-head", &[w.clone(), head]).uncovered.len(),
        1,
        "a run head was credited with ordering its own run -- this is exactly \
         the predicate #144 measured the digest gate cannot discriminate"
    );

    // Mutation C: the two ends are in different runs. `setBarrier` says nothing
    // across a run boundary.
    let mut other_run = r;
    other_run.run = Some(1);
    assert_eq!(
        cover("icb-cross-run", &[w, other_run]).uncovered.len(),
        1,
        "setBarrier was credited across a run boundary"
    );
}

/// The known-good and known-bad predicates, end to end, on one stream.
///
/// This is the acceptance criterion in miniature: the same positions, audited
/// twice, differing only in whether the run heads keep their barriers. The good
/// arm must be clean and the bad arm must not, **and #144 measured that the
/// digest gate reports the same answer for both**.
#[test]
fn the_run_heads_mutation_is_reported_where_the_digest_gate_is_blind() {
    // A run of three replayed positions in one session. The head writes; the
    // two interior positions read it back and rewrite it.
    let mk = |seq: u64, out: bool, head: bool| {
        let mut x = d(seq, out);
        x.run = Some(0);
        x.is_run_head = head;
        x.icb_barrier = !head; // a head carries no in-run ordering
        x
    };

    // GOOD: candle's barrier survives at the head, suppressed in the interior.
    let mut head = mk(0, true, true);
    head.barrier = true;
    head.kinds = {
        let mut k = HazardKinds::NONE;
        k.insert(HazardKind::Waw);
        k
    };
    let mut a = mk(1, false, false);
    a.barrier_suppressed = true;
    let mut b = mk(2, true, false);
    b.barrier_suppressed = true;

    let good = cover("proposal", &[head.clone(), a.clone(), b.clone()]);
    assert!(
        good.is_clean(),
        "the known-good predicate reported uncovered edges: {}",
        good.render()
    );
    assert_eq!(good.barriers_suppressed, 2);
    assert_eq!(good.barriers_waw, 1, "the head barrier was not attributed");

    // BAD: also suppress the head's barrier. Now a preceding gap writer has
    // nothing ordering it into the run -- which is #144's 208 broken edges.
    let mut gap_writer = d(0, true);
    gap_writer.seq = 0;
    let mut head_suppressed = head;
    head_suppressed.seq = 1;
    head_suppressed.barrier = false;
    head_suppressed.barrier_suppressed = true;
    let mut a2 = a;
    a2.seq = 2;
    let mut b2 = b;
    b2.seq = 3;

    let bad = cover("run-heads-mutation", &[gap_writer, head_suppressed, a2, b2]);
    assert!(
        !bad.is_clean(),
        "the run-heads mutation was reported clean -- the mode cannot \
         discriminate the case the digest gate is already blind to, and is \
         therefore a display rather than a test"
    );
}

/// Attribution is reported per direction, which is the artifact §11.3p could
/// not produce.
///
/// *"These N are WAR, and WAR is the one a different layout would remove"* is
/// the sentence this makes available (issue #185).
#[test]
fn barriers_are_attributed_by_direction() {
    let mut raw_barrier = d(1, false);
    raw_barrier.barrier = true;
    raw_barrier.kinds = {
        let mut k = HazardKinds::NONE;
        k.insert(HazardKind::Raw);
        k
    };

    let mut both = d(3, true);
    both.barrier = true;
    both.kinds = {
        let mut k = HazardKinds::NONE;
        k.insert(HazardKind::Waw);
        k.insert(HazardKind::War);
        k
    };

    let rep = cover("attribution", &[d(0, true), raw_barrier, d(2, false), both]);
    assert_eq!(rep.barriers, 2);
    assert_eq!(rep.barriers_raw, 1);
    assert_eq!(rep.barriers_waw, 1);
    assert_eq!(
        rep.barriers_war, 1,
        "a barrier owed to two directions was attributed to only one -- the \
         set, not the last writer, is what makes this honest"
    );
    // The per-direction counts deliberately do NOT partition `barriers`: one
    // barrier owed to both WAW and WAR is counted in each, so the sum exceeds
    // the number of barriers. Stated as an equation rather than an inequality,
    // because the inequality is what an off-by-one would still satisfy -- 2
    // barriers here, 3 direction-hits across them.
    assert_eq!(
        rep.barriers_raw + rep.barriers_waw + rep.barriers_war,
        3,
        "the direction counts do not add up to the hits across both barriers"
    );
    assert_eq!(
        rep.barriers, 2,
        "a barrier owed to two directions was double-counted"
    );
}
