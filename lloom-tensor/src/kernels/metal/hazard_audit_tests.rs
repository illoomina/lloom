//! Does the cover test discriminate?
//!
//! A test of an ordering audit owes both arms: the known-good stream reports
//! zero uncovered edges, and the same stream with one synchronization removed
//! or misplaced reports the edge that synchronization carried. An audit shown
//! only a correct stream is a display, not a test (`docs/dispatch.md` §11.3t).
//!
//! Synthetic streams, so each primitive and each placement rule is exercised
//! alone. `hazard_audit_gpu_tests` checks that the recorder sees the same
//! primitives in a stream the engine actually encodes.

#![cfg(feature = "hazard-audit")]

use crate::kernels::metal::encoder::{HazardKind, HazardKinds};
use crate::kernels::metal::hazard_audit::{
    cover, AuditBinding, AuditDispatch, AuditStream, EventScope, SeamPolicy, StageScope, Stages,
};

/// A slot every dispatch in the single-buffer fixtures shares.
const SLOT: usize = 0x1000;
const X: usize = 0x1000;
const Y: usize = 0x2000;
const Z: usize = 0x3000;

fn bind(is_output: bool) -> AuditBinding {
    AuditBinding {
        ptr: SLOT,
        offset: 0,
        len: 128,
        is_output,
    }
}

/// A dispatch at `seq` in pass 0, writing or reading the shared slot.
fn d(seq: u64, is_output: bool) -> AuditDispatch {
    let mut x = AuditDispatch::new(seq, 0, Stages::DISPATCH, format!("k{seq}"));
    x.bindings = vec![bind(is_output)];
    x
}

fn read(ptr: usize) -> AuditBinding {
    AuditBinding {
        ptr,
        offset: 0,
        len: 128,
        is_output: false,
    }
}

fn write(ptr: usize) -> AuditBinding {
    AuditBinding {
        ptr,
        offset: 0,
        len: 128,
        is_output: true,
    }
}

/// Append a command in `pass` and `stage`; returns its position.
fn cmd(s: &mut AuditStream, pass: u64, stage: Stages, binds: &[AuditBinding]) -> usize {
    let seq = s.dispatches.len();
    let mut x = AuditDispatch::new(seq as u64, pass, stage, format!("k{seq}"));
    x.bindings = binds.to_vec();
    s.push_dispatch(x);
    seq
}

fn dispatch(s: &mut AuditStream, pass: u64, binds: &[AuditBinding]) -> usize {
    cmd(s, pass, Stages::DISPATCH, binds)
}

/// Append a dispatch preceded by a full pass barrier.
fn after_barrier(s: &mut AuditStream, pass: u64, binds: &[AuditBinding]) -> usize {
    let i = dispatch(s, pass, binds);
    s.dispatches[i].barrier = true;
    i
}

/// The one uncovered edge as `(writer position, reader position, kind)`.
fn only_uncovered(s: &AuditStream) -> Vec<(u64, u64, HazardKind)> {
    s.cover("t", SeamPolicy::Observed)
        .uncovered
        .iter()
        .map(|e| (e.writer_seq, e.reader_seq, e.kind))
        .collect()
}

/// The three directions are detected, and read-after-read is not one of them;
/// the last is why weights bound on every dispatch contribute no edges.
#[test]
fn the_three_directions_are_detected_and_rar_is_not_a_hazard() {
    let raw = cover("raw", &[d(0, true), d(1, false)]);
    assert_eq!(raw.edges, 1);
    assert_eq!(raw.edges_raw, 1, "a write-then-read was not RAW");

    let waw = cover("waw", &[d(0, true), d(1, true)]);
    assert_eq!(waw.edges_waw, 1, "a write-then-write was not WAW");

    let war = cover("war", &[d(0, false), d(1, true)]);
    assert_eq!(war.edges_war, 1, "a read-then-write was not WAR");

    let rar = cover("rar", &[d(0, false), d(1, false)]);
    assert_eq!(
        rar.edges, 0,
        "two reads produced an edge; the weights would dominate the graph"
    );
}

/// Disjoint bytes in one allocation are not an edge, and an unknown extent is.
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

    let mut overlapping = hi.clone();
    overlapping.bindings[0].offset = 127;
    assert_eq!(
        cover("overlapping", &[lo.clone(), overlapping]).edges,
        1,
        "a one-byte overlap was missed"
    );

    let mut unknown = hi.clone();
    unknown.bindings[0].len = 0;
    assert_eq!(
        cover("unknown-extent", &[lo, unknown]).edges,
        1,
        "an unknown extent did not fail toward reporting"
    );

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

/// A pass barrier orders an edge in its pass, and removing it is visible.
#[test]
fn a_surviving_fence_covers_an_edge_and_its_absence_is_reported() {
    let w = d(0, true);
    let mut r = d(1, false);
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

    r.barrier = false;
    let bad = cover("no-fence", &[w, r]);
    assert_eq!(
        bad.uncovered.len(),
        1,
        "removing the only fence reported nothing -- the model is vacuous"
    );
    assert_eq!(bad.uncovered[0].kind, HazardKind::Raw);
}

/// A pass barrier orders every earlier command of its pass against every
/// later one, including commands interleaved between ICB runs, which
/// `setBarrier` does not.
#[test]
fn a_fence_orders_across_an_interleaved_gap_position() {
    // The gap position touches a different allocation, or it would form its
    // own edge with the writer and the fixture would test something else.
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

/// A pass boundary orders nothing by itself; a fence the reader's pass waits
/// on, updated after the writer, does.
///
/// The older model credited every cross-pass edge to the boundary, which is
/// the assumption the cross-encoder counterexample breaks
/// (`docs/dependency-ordered-reuse-design.md` §2). That verdict survives only
/// as the labelled [`SeamPolicy::AssumedOrdered`].
#[test]
fn a_session_seam_covers_an_edge_only_through_an_observed_fence() {
    let mut s = AuditStream::default();
    let w = dispatch(&mut s, 0, &[write(X)]);
    dispatch(&mut s, 1, &[read(X)]);

    let bare = s.cover("seam-only", SeamPolicy::Observed);
    assert_eq!(
        bare.uncovered.len(),
        1,
        "a pass boundary with no fence was credited as ordering"
    );
    assert_eq!(bare.by_seam, 0);

    let assumed = s.cover("seam-assumed", SeamPolicy::AssumedOrdered);
    assert!(assumed.is_clean());
    assert_eq!(
        assumed.by_seam, 1,
        "the legacy policy no longer reproduces the older verdict"
    );

    let mut fenced = AuditStream::default();
    dispatch(&mut fenced, 0, &[write(X)]);
    fenced.fence_update(0, 0xF0, Stages::DISPATCH);
    fenced.fence_wait(1, 0xF0, Stages::DISPATCH);
    dispatch(&mut fenced, 1, &[read(X)]);
    let good = fenced.cover("seam-fenced", SeamPolicy::Observed);
    assert!(good.is_clean(), "{}", good.render());
    assert_eq!(good.by_fence, 1);
    assert_eq!((good.fence_updates, good.fence_waits), (1, 1));

    // The fenced stream collapsed into one pass: the fence pair is now an
    // update followed by a wait in one pass, which orders nothing.
    let mut one_pass = fenced.clone();
    for x in one_pass.dispatches.iter_mut() {
        x.encoder = 0;
    }
    for x in one_pass.syncs.iter_mut() {
        if let crate::kernels::metal::hazard_audit::SyncKind::FenceWait { encoder, .. } =
            &mut x.kind
        {
            *encoder = 0;
        }
    }
    let collapsed = one_pass.cover("one-pass", SeamPolicy::Observed);
    assert_eq!(collapsed.uncovered.len(), 1);
    assert_eq!(collapsed.same_pass_fence_waits, 1);
    let _ = w;
}

/// The ICB's `setBarrier` covers an edge within one run, and each of its four
/// conditions is load-bearing: both ends replayed, the same run, the reader
/// carrying a barrier, and the reader not a run head.
#[test]
fn the_icb_set_barrier_covers_within_a_run_and_every_condition_is_load_bearing() {
    let mut w = d(0, true);
    w.run = Some(0);
    w.is_run_head = true;

    let mut r = d(1, false);
    r.run = Some(0);
    r.icb_barrier = true;
    r.barrier_suppressed = true;

    let good = cover("icb", &[w.clone(), r.clone()]);
    assert_eq!(good.by_icb, 1, "setBarrier did not cover an in-run edge");
    assert!(good.is_clean());
    assert_eq!(good.barriers_suppressed, 1);

    let mut no_barrier = r.clone();
    no_barrier.icb_barrier = false;
    assert_eq!(
        cover("icb-off", &[w.clone(), no_barrier]).uncovered.len(),
        1,
        "removing the ICB barrier reported nothing"
    );

    let mut head = r.clone();
    head.is_run_head = true;
    assert_eq!(
        cover("icb-head", &[w.clone(), head]).uncovered.len(),
        1,
        "a run head was credited with ordering its own run"
    );

    let mut other_run = r;
    other_run.run = Some(1);
    assert_eq!(
        cover("icb-cross-run", &[w, other_run]).uncovered.len(),
        1,
        "setBarrier was credited across a run boundary"
    );
}

/// The known-good and known-bad suppression predicates on one stream: the
/// good arm keeps the run head's barrier, the bad arm also suppresses it, and
/// only the audit tells them apart (`docs/dispatch.md` §11.3p).
#[test]
fn the_run_heads_mutation_is_reported_where_the_digest_gate_is_blind() {
    let mk = |seq: u64, out: bool, head: bool| {
        let mut x = d(seq, out);
        x.run = Some(0);
        x.is_run_head = head;
        x.icb_barrier = !head;
        x
    };

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

    let gap_writer = d(0, true);
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
         discriminate the case the digest gate is already blind to"
    );
}

/// Barriers are attributed per direction; one barrier owed to two directions
/// counts in each, so the direction counts sum past the barrier count.
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
    assert_eq!(rep.barriers_war, 1);
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

/// The characterization probe's first blind spot, rejected: a read in one pass
/// and a write in the next with nothing between them
/// (`measurements/issue-985-design/audit_characterization.py`,
/// `cross_encoder_missing_WAR`).
#[test]
fn characterized_cross_encoder_missing_war_is_rejected() {
    let op = |seq: u64, encoder: u64, write: bool, ptr: usize| {
        let mut x = AuditDispatch::new(seq, encoder, Stages::DISPATCH, format!("op{seq}"));
        x.bindings = vec![AuditBinding {
            ptr,
            offset: 0,
            len: 64,
            is_output: write,
        }];
        x
    };
    let rep = cover(
        "cross_encoder_missing_WAR",
        &[op(0, 0, false, 1), op(1, 1, true, 1)],
    );
    assert_eq!(rep.edges_war, 1);
    assert_eq!(rep.by_seam, 0, "the pass boundary was credited");
    assert_eq!(
        rep.uncovered.len(),
        1,
        "a cross-pass WAR with no dependency was reported clean"
    );
    assert_eq!(rep.uncovered[0].kind, HazardKind::War);
}

/// The characterization probe's second blind spot, rejected: a barrier in an
/// unrelated pass, between the two ends, orders nothing for them
/// (`unrelated_pass_barrier`). Also within one pass interleaved with another
/// pass's barrier.
#[test]
fn characterized_unrelated_pass_barrier_is_rejected() {
    let op = |seq: u64, encoder: u64, write: bool, barrier: bool, ptr: usize| {
        let mut x = AuditDispatch::new(seq, encoder, Stages::DISPATCH, format!("op{seq}"));
        x.bindings = vec![AuditBinding {
            ptr,
            offset: 0,
            len: 64,
            is_output: write,
        }];
        x.barrier = barrier;
        x
    };
    let rep = cover(
        "unrelated_pass_barrier",
        &[
            op(0, 0, false, false, 1),
            op(1, 1, true, true, 2),
            op(2, 1, true, false, 1),
        ],
    );
    assert_eq!(rep.edges_war, 1);
    assert_eq!(
        rep.by_barrier, 0,
        "a barrier in an unrelated pass was credited"
    );
    assert_eq!(rep.uncovered.len(), 1);

    let same_pass = cover(
        "foreign_barrier_between_one_pass",
        &[
            op(0, 0, true, false, 1),
            op(1, 1, false, true, 2),
            op(2, 0, false, false, 1),
        ],
    );
    assert_eq!(same_pass.edges_raw, 1);
    assert_eq!(
        same_pass.by_barrier, 0,
        "another pass's barrier ordered this pass's commands"
    );
    assert_eq!(same_pass.uncovered.len(), 1);
}

/// The issue's counterexample: pass A writes X, pass B reads it, pass C
/// overwrites it. C waiting on A's fence orders A before C, never B before C.
/// A frontier holding only the last reader misses an earlier one.
#[test]
fn a_reader_in_an_earlier_pass_needs_an_edge_the_overwrite_waits_on() {
    let build = |c_waits_on: &[usize]| {
        let mut s = AuditStream::default();
        dispatch(&mut s, 0, &[write(X)]);
        s.fence_update(0, 0xA, Stages::DISPATCH);
        s.fence_wait(1, 0xA, Stages::DISPATCH);
        dispatch(&mut s, 1, &[read(X), write(Y)]);
        s.fence_update(1, 0xB, Stages::DISPATCH);
        s.fence_wait(2, 0xA, Stages::DISPATCH);
        dispatch(&mut s, 2, &[read(X), write(Z)]);
        s.fence_update(2, 0xC, Stages::DISPATCH);
        for &f in c_waits_on {
            s.fence_wait(3, f, Stages::DISPATCH);
        }
        dispatch(&mut s, 3, &[write(X)]);
        s
    };

    // The writer-only map: the overwrite waits on the last writer alone.
    assert_eq!(
        only_uncovered(&build(&[0xA])),
        vec![(1, 3, HazardKind::War), (2, 3, HazardKind::War)],
        "readers in earlier passes were taken as ordered before the overwrite"
    );
    // The single-reader mutant: only the last reader joined.
    assert_eq!(
        only_uncovered(&build(&[0xA, 0xC])),
        vec![(1, 3, HazardKind::War)],
        "a frontier of the last reader alone was reported complete"
    );
    // The complete frontier.
    let full = build(&[0xA, 0xB, 0xC]).cover("frontier", SeamPolicy::Observed);
    assert!(full.is_clean(), "{}", full.render());
    assert_eq!(full.by_fence, 5);
}

/// An update encoded before the writer, or no update at all, covers nothing.
#[test]
fn a_missing_or_early_fence_update_covers_nothing() {
    let mut early = AuditStream::default();
    early.fence_update(0, 0xA, Stages::DISPATCH);
    dispatch(&mut early, 0, &[write(X)]);
    early.fence_wait(1, 0xA, Stages::DISPATCH);
    dispatch(&mut early, 1, &[read(X)]);
    assert_eq!(
        only_uncovered(&early),
        vec![(0, 1, HazardKind::Raw)],
        "an update encoded before the writer was credited"
    );

    let mut missing = AuditStream::default();
    dispatch(&mut missing, 0, &[write(X)]);
    missing.fence_wait(1, 0xA, Stages::DISPATCH);
    dispatch(&mut missing, 1, &[read(X)]);
    let rep = missing.cover("missing-update", SeamPolicy::Observed);
    assert_eq!(
        rep.uncovered.len(),
        1,
        "a wait on a never-updated fence ordered"
    );
    assert_eq!(rep.unmatched_fence_waits, 1);
}

/// A wait encoded after the reader, or no wait at all, covers nothing.
#[test]
fn a_missing_or_late_fence_wait_covers_nothing() {
    let mut late = AuditStream::default();
    dispatch(&mut late, 0, &[write(X)]);
    late.fence_update(0, 0xA, Stages::DISPATCH);
    dispatch(&mut late, 1, &[read(X)]);
    late.fence_wait(1, 0xA, Stages::DISPATCH);
    assert_eq!(
        only_uncovered(&late),
        vec![(0, 1, HazardKind::Raw)],
        "a wait encoded after the reader was credited"
    );

    let mut missing = AuditStream::default();
    dispatch(&mut missing, 0, &[write(X)]);
    missing.fence_update(0, 0xA, Stages::DISPATCH);
    dispatch(&mut missing, 1, &[read(X)]);
    assert_eq!(
        only_uncovered(&missing),
        vec![(0, 1, HazardKind::Raw)],
        "an update nobody waits on was credited"
    );
}

/// A buffer first read in a pass and later written in it: the write needs
/// the earlier passes' readers, which the read never asked for. The wait must
/// sit between the read and the write; the read's own wait on the writer and
/// the pass barrier before the write do not supply it.
#[test]
fn a_read_to_write_upgrade_needs_the_earlier_readers() {
    let build = |reader_wait: Option<bool>| {
        let mut s = AuditStream::default();
        dispatch(&mut s, 0, &[write(X)]);
        s.fence_update(0, 0xA, Stages::DISPATCH);
        s.fence_wait(1, 0xA, Stages::DISPATCH);
        dispatch(&mut s, 1, &[read(X), write(Y)]);
        s.fence_update(1, 0xB, Stages::DISPATCH);
        s.fence_wait(2, 0xA, Stages::DISPATCH);
        dispatch(&mut s, 2, &[read(X), write(Z)]);
        if reader_wait == Some(true) {
            s.fence_wait(2, 0xB, Stages::DISPATCH);
        }
        after_barrier(&mut s, 2, &[write(X)]);
        if reader_wait == Some(false) {
            s.fence_wait(2, 0xB, Stages::DISPATCH);
        }
        s
    };

    assert_eq!(
        only_uncovered(&build(None)),
        vec![(1, 3, HazardKind::War)],
        "the upgrade was covered by the read's wait on the writer"
    );
    assert_eq!(
        only_uncovered(&build(Some(false))),
        vec![(1, 3, HazardKind::War)],
        "a reader wait encoded after the write was credited"
    );
    let good = build(Some(true)).cover("upgrade", SeamPolicy::Observed);
    assert!(good.is_clean(), "{}", good.render());
    assert_eq!(good.by_barrier, 1, "the in-pass WAR was not the barrier's");
}

/// A wait observes the latest update of its fence; an earlier generation's
/// work is not credited to it.
#[test]
fn a_wait_credits_only_the_latest_update_of_its_fence() {
    let build = |reupdate_before_wait: bool| {
        let mut s = AuditStream::default();
        dispatch(&mut s, 0, &[write(X)]);
        s.fence_update(0, 0xA, Stages::DISPATCH);
        if reupdate_before_wait {
            dispatch(&mut s, 1, &[write(Y)]);
            s.fence_update(1, 0xA, Stages::DISPATCH);
        }
        s.fence_wait(2, 0xA, Stages::DISPATCH);
        dispatch(&mut s, 2, &[read(X)]);
        s
    };
    assert!(build(false).cover("gen1", SeamPolicy::Observed).is_clean());
    assert_eq!(
        only_uncovered(&build(true)),
        vec![(0, 2, HazardKind::Raw)],
        "a wait was credited with an earlier generation of its fence"
    );
}

/// An update followed by a wait on the same fence in one pass orders nothing,
/// and is counted.
#[test]
fn an_update_then_wait_in_one_pass_orders_nothing() {
    let mut s = AuditStream::default();
    dispatch(&mut s, 0, &[write(X)]);
    s.fence_update(0, 0xA, Stages::DISPATCH);
    s.fence_wait(0, 0xA, Stages::DISPATCH);
    dispatch(&mut s, 0, &[read(X)]);
    let rep = s.cover("same-pass", SeamPolicy::Observed);
    assert_eq!(rep.uncovered.len(), 1);
    assert_eq!(rep.same_pass_fence_waits, 1);
}

/// A synchronization orders only the stages it names: an update after
/// dispatch does not cover a blit, a wait before blit does not block a
/// dispatch, and a dispatch-only pass barrier does not order a blit.
#[test]
fn stages_a_synchronization_does_not_name_are_not_ordered() {
    let fenced = |writer: Stages, after: Stages, before: Stages, reader: Stages| {
        let mut s = AuditStream::default();
        cmd(&mut s, 0, writer, &[write(X)]);
        s.fence_update(0, 0xA, after);
        s.fence_wait(1, 0xA, before);
        cmd(&mut s, 1, reader, &[read(X)]);
        s.cover("stages", SeamPolicy::Observed).uncovered.len()
    };
    assert_eq!(
        fenced(Stages::BLIT, Stages::BLIT, Stages::ALL, Stages::DISPATCH),
        0
    );
    assert_eq!(
        fenced(
            Stages::BLIT,
            Stages::DISPATCH,
            Stages::ALL,
            Stages::DISPATCH
        ),
        1,
        "an update after dispatch was credited with a blit"
    );
    assert_eq!(
        fenced(
            Stages::DISPATCH,
            Stages::ALL,
            Stages::BLIT,
            Stages::DISPATCH
        ),
        1,
        "a wait before blit was credited with blocking a dispatch"
    );

    let barriered = |scope: StageScope| {
        let mut s = AuditStream::default();
        cmd(&mut s, 0, Stages::BLIT, &[write(X)]);
        let r = cmd(&mut s, 0, Stages::DISPATCH, &[read(X)]);
        s.dispatches[r].barrier = true;
        s.dispatches[r].barrier_scope = scope;
        s.cover("barrier-stages", SeamPolicy::Observed)
            .uncovered
            .len()
    };
    assert_eq!(barriered(StageScope::ALL), 0);
    assert_eq!(
        barriered(StageScope {
            after: Stages::DISPATCH,
            before: Stages::DISPATCH,
        }),
        1,
        "a dispatch-only barrier ordered a blit"
    );
}

/// A queue barrier orders every earlier pass's work before the barrier pass's
/// later work, but not its own pass's earlier work, not a pass opened after
/// it, and not a command encoded before it.
#[test]
fn a_queue_barrier_orders_earlier_passes_only() {
    let mut good = AuditStream::default();
    dispatch(&mut good, 0, &[write(X)]);
    good.queue_barrier(1, Stages::ALL, Stages::ALL);
    dispatch(&mut good, 1, &[read(X)]);
    let rep = good.cover("queue-barrier", SeamPolicy::Observed);
    assert!(rep.is_clean(), "{}", rep.render());
    assert_eq!(rep.by_queue_barrier, 1);

    let mut late = AuditStream::default();
    dispatch(&mut late, 0, &[write(X)]);
    dispatch(&mut late, 1, &[read(X)]);
    late.queue_barrier(1, Stages::ALL, Stages::ALL);
    assert_eq!(
        only_uncovered(&late),
        vec![(0, 1, HazardKind::Raw)],
        "a queue barrier encoded after the reader was credited"
    );

    let mut own_pass = AuditStream::default();
    dispatch(&mut own_pass, 0, &[write(X)]);
    own_pass.queue_barrier(0, Stages::ALL, Stages::ALL);
    dispatch(&mut own_pass, 0, &[read(X)]);
    assert_eq!(
        only_uncovered(&own_pass),
        vec![(0, 1, HazardKind::Raw)],
        "a queue barrier ordered its own pass's earlier work"
    );

    let mut later_pass = AuditStream::default();
    later_pass.open_pass(1, None);
    dispatch(&mut later_pass, 0, &[write(X)]);
    later_pass.queue_barrier(1, Stages::ALL, Stages::ALL);
    dispatch(&mut later_pass, 1, &[read(X)]);
    assert_eq!(
        only_uncovered(&later_pass),
        vec![(0, 1, HazardKind::Raw)],
        "a queue barrier ordered a pass opened after its own"
    );
}

/// A shared-event wait is satisfied by any signal at or above its value, so a
/// higher signal from unrelated work is not a completed-prefix certificate:
/// the wait is credited only with what every satisfying signal follows
/// (`docs/dependency-ordered-reuse-design.md` §7).
#[test]
fn an_event_wait_is_not_a_prefix_certificate() {
    let build = |unrelated_higher: bool, later_signal: bool| {
        let mut s = AuditStream::default();
        s.open_pass(0, Some(10));
        dispatch(&mut s, 0, &[write(X)]);
        s.event_signal(EventScope::CommandBuffer(10), 0xE, 1);
        if unrelated_higher {
            s.open_pass(1, Some(11));
            dispatch(&mut s, 1, &[write(Y)]);
            s.event_signal(EventScope::CommandBuffer(11), 0xE, 2);
        }
        s.open_pass(2, Some(12));
        s.event_wait(EventScope::CommandBuffer(12), 0xE, 1);
        dispatch(&mut s, 2, &[read(X)]);
        if later_signal {
            s.event_signal(EventScope::Queue, 0xE, 3);
        }
        s
    };
    let good = build(false, false).cover("event", SeamPolicy::Observed);
    assert!(good.is_clean(), "{}", good.render());
    assert_eq!(good.by_event, 1);

    let scalar = build(true, false).cover("event-scalar", SeamPolicy::Observed);
    assert_eq!(
        scalar.uncovered.len(),
        1,
        "an unrelated higher signal certified the earlier producer"
    );

    let later = build(false, true).cover("event-later", SeamPolicy::Observed);
    assert_eq!(
        later.uncovered.len(),
        1,
        "a wait a later signal could satisfy ordered"
    );
    assert_eq!(later.unordered_event_waits, 1);
}

/// Happens-before is transitive through fences, but only through a command:
/// a pass whose wait and update have nothing between them carries nothing.
#[test]
fn a_chain_of_fences_orders_only_through_a_command() {
    let build = |relay_command: bool| {
        let mut s = AuditStream::default();
        dispatch(&mut s, 0, &[write(X)]);
        s.fence_update(0, 0xA, Stages::DISPATCH);
        s.fence_wait(1, 0xA, Stages::DISPATCH);
        if relay_command {
            dispatch(&mut s, 1, &[read(Z)]);
        }
        s.fence_update(1, 0xB, Stages::DISPATCH);
        s.fence_wait(2, 0xB, Stages::DISPATCH);
        dispatch(&mut s, 2, &[read(X)]);
        s
    };
    let chained = build(true).cover("chain", SeamPolicy::Observed);
    assert!(chained.is_clean(), "{}", chained.render());
    assert_eq!(chained.by_fence, 1);
    assert_eq!(
        build(false)
            .cover("chain-empty", SeamPolicy::Observed)
            .uncovered
            .len(),
        1,
        "an update with no command after the wait carried the wait's ordering"
    );
}

/// What a covered edge is attributed to is the primitive that carried it.
#[test]
fn an_edge_names_the_primitive_that_covered_it() {
    let mut s = AuditStream::default();
    dispatch(&mut s, 0, &[write(X)]);
    s.fence_update(0, 0xA, Stages::DISPATCH);
    s.fence_wait(1, 0xA, Stages::DISPATCH);
    dispatch(&mut s, 1, &[read(X), write(Y)]);
    after_barrier(&mut s, 1, &[read(Y)]);
    let rep = s.cover("attribution", SeamPolicy::Observed);
    assert!(rep.is_clean(), "{}", rep.render());
    assert_eq!((rep.by_fence, rep.by_barrier), (1, 1));
    assert_eq!((rep.by_seam, rep.by_transitive), (0, 0));
    assert_eq!(rep.edges, 2);
}
