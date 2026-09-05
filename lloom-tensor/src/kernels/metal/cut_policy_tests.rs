//! Where the command buffer is cut (`DESIGN.md` §13.5m, issue #420 stage 3).
//!
//! Pure arithmetic, no device: [`CutPolicy::cuts_at_node`] and
//! [`CutPolicy::sessions_for`] allocate nothing and touch no `MTLCommandQueue`,
//! so the partition is testable without a GPU. That is `scratch_tests.rs`'s
//! split between the arithmetic and what executes it, and it matters here for
//! the same reason: the property this arm rests on — **the session count is
//! `n_cb` by construction** — is an arithmetic claim, and an arithmetic claim
//! checked only end-to-end is checked by a digest that §2.3.8d records as
//! stable under a model emitting `picture picture picture`.
//!
//! # What these tests are for, stated once
//!
//! §13.5j measured the cut's cost at **~7.2–7.4 µs per encoder session and not
//! a function of the barrier count** — but barriers and sessions are collinear
//! along `COMPUTE_PER_BUFFER` (R² 0.983 and 0.985), so **no setting of that
//! knob separates them**. A node-index partition is the arm that can, and it
//! separates them *only if the count is genuinely held*. So the count is not a
//! side effect to be observed after the fact; it is the property the arm is
//! built on, and it is asserted here.
//!
//! # The geometry these fixtures use is LFM2's, and one is deliberately not
//!
//! §5.3: 30 layers, attention at 2, 5, 9, 13, 17, 21, 24, 27. Every fixture
//! that divides 30 evenly would leave `sessions_for`'s `div_ceil` untested —
//! `30 / 3` and `30.div_ceil(3)` are the same number — which is #70's
//! align_up blindness in a second quantity: a fixture built only from the
//! model's own shapes cannot see a whole defect class. So a fixture with a
//! remainder is carried deliberately.

use super::commands::CutPolicy;

/// §5.3, verified from `layer_types` in the checkpoint config.
const LFM2_LAYERS: usize = 30;

/// The shipping arm's decode step: §13.5j's own engagement table, and #421's
/// `sdpa/in-place` row.
const LFM2_DISPATCHES: usize = 538;

#[test]
fn the_dispatch_counter_never_cuts_on_a_node() {
    // The whole of §13.5i in one assertion: a dispatch counter has no notion of
    // graph position, so it cannot answer this question at any index. If this
    // ever returns true the default path has acquired a second cut condition
    // and its session count is no longer the counter's alone.
    let policy = CutPolicy::DispatchCounter;
    for index in 0..LFM2_LAYERS {
        assert!(
            !policy.cuts_at_node(index),
            "the counter cut at node {index}; the default path must be unchanged (§7.1a)"
        );
    }
}

#[test]
fn node_zero_never_cuts_under_either_policy() {
    // `n_cb` runs have `n_cb - 1` interior boundaries. A cut at the first node
    // would add a seam the partition does not name and make the session count
    // `n_cb + 1`, which is exactly the held-count property this arm rests on.
    for nodes_per_cb in 1..=LFM2_LAYERS {
        let policy = CutPolicy::NodeIndex { nodes_per_cb };
        assert!(
            !policy.cuts_at_node(0),
            "node 0 cut at nodes_per_cb={nodes_per_cb}; that is one session too many"
        );
    }
    assert!(!CutPolicy::DispatchCounter.cuts_at_node(0));
}

#[test]
fn the_partition_cuts_exactly_at_the_run_boundaries() {
    // nodes_per_cb = 3 over 30 layers: runs are [0,3) [3,6) ... [27,30), so the
    // interior boundaries are 3, 6, 9, ... 27 -- nine of them, and the measured
    // arm puts its nine seams at exactly those layers (§13.5m's table).
    let policy = CutPolicy::NodeIndex { nodes_per_cb: 3 };
    let cuts: Vec<usize> = (0..LFM2_LAYERS)
        .filter(|&i| policy.cuts_at_node(i))
        .collect();
    assert_eq!(
        cuts,
        vec![3, 6, 9, 12, 15, 18, 21, 24, 27],
        "the partition's interior boundaries are not where the runs divide"
    );
}

#[test]
fn the_session_count_is_the_partition_count_by_construction() {
    // The held-count property, as arithmetic. One session per run, and the
    // number of runs is a function of the LAYER count alone -- so it does not
    // move when the dispatch count does, which is what lets this arm hold the
    // count while `COMPUTE_PER_BUFFER` cannot (§13.5j).
    for nodes_per_cb in 1..=LFM2_LAYERS {
        let policy = CutPolicy::NodeIndex { nodes_per_cb };
        let interior = (0..LFM2_LAYERS).filter(|&i| policy.cuts_at_node(i)).count();
        let expected = LFM2_LAYERS.div_ceil(nodes_per_cb);
        assert_eq!(
            interior + 1,
            expected,
            "nodes_per_cb={nodes_per_cb}: {interior} interior cuts is not {expected} runs"
        );
        assert_eq!(
            policy.sessions_for(LFM2_LAYERS, 50, LFM2_DISPATCHES),
            expected,
            "sessions_for disagrees with the cut sites it is supposed to describe"
        );
    }
}

#[test]
fn the_node_arms_count_is_independent_of_the_dispatch_total() {
    // Why the arm can hold the count at all: the partition is over layers, so
    // a change that adds or removes dispatches -- a fusion, a different
    // attention arm, a different KV append -- moves the COUNTER's session count
    // and leaves this one alone. §13.5i measured the other side of this: the
    // same counter setting puts 6 seams in attention on one arm and 4 on
    // another, purely because the graph shifted under a fixed stride.
    let node = CutPolicy::NodeIndex { nodes_per_cb: 3 };
    let counter = CutPolicy::DispatchCounter;
    for dispatches in [538, 554, 562, 674] {
        assert_eq!(
            node.sessions_for(LFM2_LAYERS, 50, dispatches),
            10,
            "the node arm's count moved with the dispatch total"
        );
    }
    // And the counter's does move, which is the contrast that makes the line
    // above a finding rather than a tautology. 674 is §13.5i's generic arm and
    // 538 the shipping one.
    assert_ne!(
        counter.sessions_for(LFM2_LAYERS, 50, 538),
        counter.sessions_for(LFM2_LAYERS, 50, 674),
        "the counter's session count must depend on the dispatch total; if it \
         does not, this comparison is vacuous"
    );
}

#[test]
fn the_run_length_that_matches_the_shipping_session_count_is_the_measured_one() {
    // The arm's own design constraint, pinned. The shipping decode step is 538
    // dispatches at `COMPUTE_PER_BUFFER=50`, which is 11 sessions; no integer
    // run length gives 11 over 30 layers, because `ceil(30/k)` takes the values
    // 30, 15, 10, 8, 6, 5, 4, 3, 2, 1 and skips 11.
    //
    // **So the control is a counter arm re-tuned to the node arm's count, not
    // the shipping default.** That is stated here as a test rather than only in
    // the artifact, because it is the reason the measurement compares against
    // `Counter(55)` and a reader who assumed `Counter(50)` would think the
    // comparison confounded.
    let counter = CutPolicy::DispatchCounter;
    assert_eq!(counter.sessions_for(LFM2_LAYERS, 50, LFM2_DISPATCHES), 11);

    let reachable: Vec<usize> = (1..=LFM2_LAYERS)
        .map(|k| {
            CutPolicy::NodeIndex { nodes_per_cb: k }.sessions_for(LFM2_LAYERS, 50, LFM2_DISPATCHES)
        })
        .collect();
    assert!(
        !reachable.contains(&11),
        "11 sessions is reachable by a layer partition after all: {reachable:?} -- \
         if this fires, the matched control should be Counter(50) and not Counter(55)"
    );

    // 10 is reachable, at k = 3, and that is the arm that was run.
    assert_eq!(
        CutPolicy::NodeIndex { nodes_per_cb: 3 }.sessions_for(LFM2_LAYERS, 50, LFM2_DISPATCHES),
        10
    );
    // And the counter reaches 10 at 55, which is the matched control's period.
    assert_eq!(counter.sessions_for(LFM2_LAYERS, 55, LFM2_DISPATCHES), 10);
}

#[test]
fn a_run_length_that_does_not_divide_the_layer_count_still_partitions() {
    // The fixture with a remainder, per the module header. `30 / 4` is 7 and
    // `30.div_ceil(4)` is 8: the last run is short. A `div` where this wants a
    // `div_ceil` loses the tail run entirely, and every LFM2-shaped fixture
    // (k = 1, 2, 3, 5, 6, 10, 15, 30) is blind to that -- #70's `align_up`
    // blindness in a second quantity.
    let policy = CutPolicy::NodeIndex { nodes_per_cb: 4 };
    let cuts: Vec<usize> = (0..LFM2_LAYERS)
        .filter(|&i| policy.cuts_at_node(i))
        .collect();
    assert_eq!(cuts, vec![4, 8, 12, 16, 20, 24, 28]);
    assert_eq!(cuts.len() + 1, 8, "the short tail run is not counted");
    assert_eq!(policy.sessions_for(LFM2_LAYERS, 50, LFM2_DISPATCHES), 8);
    assert_ne!(
        LFM2_LAYERS / 4,
        LFM2_LAYERS.div_ceil(4),
        "this fixture is supposed to have a remainder; without one it tests the \
         identity function"
    );
}

#[test]
fn a_run_length_of_one_cuts_at_every_layer() {
    // The degenerate end, and the check that the partition is not off by one at
    // it: 30 layers, 29 interior boundaries, 30 sessions.
    let policy = CutPolicy::NodeIndex { nodes_per_cb: 1 };
    let cuts: Vec<usize> = (0..LFM2_LAYERS)
        .filter(|&i| policy.cuts_at_node(i))
        .collect();
    assert_eq!(cuts.len(), LFM2_LAYERS - 1);
    assert_eq!(cuts.first(), Some(&1));
    assert_eq!(cuts.last(), Some(&29));
    assert_eq!(policy.sessions_for(LFM2_LAYERS, 50, LFM2_DISPATCHES), 30);
}

#[test]
fn a_run_length_at_or_past_the_layer_count_cuts_nowhere() {
    // The other degenerate end: one run, no interior boundary, one session --
    // which is llama.cpp's shipping `n_cb = 1` (#419), so this is not a
    // hypothetical corner.
    for nodes_per_cb in [LFM2_LAYERS, LFM2_LAYERS + 1, 1024] {
        let policy = CutPolicy::NodeIndex { nodes_per_cb };
        assert_eq!(
            (0..LFM2_LAYERS).filter(|&i| policy.cuts_at_node(i)).count(),
            0
        );
        assert_eq!(policy.sessions_for(LFM2_LAYERS, 50, LFM2_DISPATCHES), 1);
    }
}
