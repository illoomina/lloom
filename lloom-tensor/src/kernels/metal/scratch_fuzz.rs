//! A stateful fuzzer over the scratch class's **long-context** state machine
//! (lloom issue #575).
//!
//! # Why this target, and why `kv_len` is the axis
//!
//! `DESIGN.md` §15.1h and §15.1i fuzzed the pool, the submission queue and the
//! token ring. Every one of those state machines is **context-length
//! independent**: the pool's alphabet is sizes, the queue's is commits, the
//! ring's is turns, and none of them has a state that only exists past some
//! `kv_len`. #575's premise is that the mechanisms whose behaviour *is*
//! context-length dependent were the unfuzzed ones, and that is right --
//! grepped over the four existing harnesses, `kv_len` appears in **none**.
//!
//! This file takes the half of that surface which is **pure arithmetic**:
//! [`plan_scratch`], [`Sizing::sized_chunks`], [`PartialsGeometry::chunks`] and
//! the plan's own checks. They allocate nothing and touch no device
//! (`scratch.rs:394`), so a campaign reaching `kv_len` 131 072 costs
//! milliseconds and runs in `cargo test` on any backend -- which is the same
//! argument §15.1h makes for the pool at 1.7 µs/op, one step further because
//! there is not even an `MTLBuffer`.
//!
//! # What a "transition" is here, and why it is not line coverage
//!
//! #575 asks for **transitions taken, not lines executed**, and names #546's
//! `208 prunes, zero of which removed anything` as the reason. The transitions
//! this state machine has are the ones a `kv_len` walk can cross:
//!
//! * a **chunk boundary** -- `live_chunks` increments, so every region resizes
//!   under `Grow` and none do under `Reserve`;
//! * a **bucket rung** -- `Bucket` jumps to the next ladder entry (8k/32k/128k);
//! * a **refusal** -- `Reserve` past `max_context`, `Bucket` past the last rung;
//! * the **degenerate** `kv_len == 0`, which must give 0 chunks and not 1.
//!
//! [`Transitions`] counts each, and the engagement test below asserts
//! every one is reached and **names the ones that are not**. Without it a green
//! run is #504's `evicted = 0` and #546's zero-effect prunes: a campaign that
//! measured nothing, reported comfortably.
//!
//! # The fixture is built so it can fail
//!
//! Two blindnesses are already recorded against this exact code and both are
//! designed around here rather than rediscovered:
//!
//! * **§9.2c / #70**: every LFM2 activation is a 128-multiple, so `align_up` is
//!   a no-op on our own shapes and #70 deleted it with its test still passing.
//!   So the generator draws [`ScratchLayout::Interleaved`] -- whose 264 B record
//!   is *not* a 128-multiple -- as often as `Planes`.
//! * **§15.1h / #574's mutant 2**: a fixture where the quantity under test is
//!   constant across the reduction cannot see a defect in it. `page_size` is
//!   therefore drawn from a set that is **not** all 256, and `layers` is not
//!   always 8, so a defect proportional to either is not multiplied by a
//!   constant the model shares.

#![cfg(any(test, feature = "fuzzing"))]

use super::scratch::*;
use super::{align_up, ARENA_ALIGNMENT};

/// LFM2's attention layer count (§5.3: attention at 2, 5, 9, 13, 17, 21, 24,
/// 27), used where a run wants the shipping shape rather than a drawn one.
pub const LFM2_ATTENTION_LAYERS: usize = 8;

/// §5.2's `max_position_embeddings`.
const MAX_CONTEXT: usize = 131_072;

// ---------------------------------------------------------------------------
// The alphabet.
// ---------------------------------------------------------------------------

/// One symbol of the decode loop's `kv_len` walk.
///
/// Deliberately the *loop's* operations rather than the module's functions: the
/// class under test is what happens as a sequence grows, so a symbol is
/// something a decode step does between one plan and the next. This is
/// `token_ring_fuzz`'s framing (its `Op` is the loop's alphabet, not
/// `TokenRing`'s methods) applied to context length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    /// Decode `n` tokens: `kv_len += n`. The overwhelmingly common step, and
    /// `n > 1` is a prefill or a speculative window landing at once.
    Append { n: usize },
    /// Jump `kv_len` straight to a value near a chunk or rung boundary.
    ///
    /// A uniform walk essentially never lands *on* a boundary, and the
    /// boundaries are where this state machine changes behaviour. Same
    /// reasoning as `buffer_pool_fuzz`'s repeated sizes: without them the
    /// exact-match path never fires.
    SeekNear { boundary: usize, delta: i8 },
    /// A turn boundary: the cache is cleared and the walk restarts from 0.
    ///
    /// #575 asks for sequences that cross a turn boundary, and §10.9 records
    /// that neither multi-turn harness calls `Cache::clear()` -- so 0 is a
    /// state a real run reaches only here.
    Reset,
    /// Re-plan at the current `kv_len` under a different sizing policy.
    ///
    /// The mechanism *combination* #575 asks for: a policy change at a live
    /// `kv_len` is what a caller switching arms mid-run does.
    Repolicy { sizing: Sizing },
    /// Re-plan under a different layout, for the same reason.
    Relayout { layout: ScratchLayout },
}

/// What the plan *should* be, re-derived from the documented contract.
///
/// A model that mirrors the implementation cannot find a bug in it
/// (§15.1h). This one is written from the doc comments on `sized_chunks`,
/// `plan_scratch`, `chunks`, `bump_capacity` and `reserved_waste` -- not from
/// reading their bodies -- and it keeps only numbers, never a `ScratchRegion`.
#[derive(Debug, Clone)]
struct Model {
    kv_len: usize,
    geometry: PartialsGeometry,
    layers: usize,
    sizing: Sizing,
    layout: ScratchLayout,
    max_context: usize,
}

impl Model {
    /// `ceil(kv_len / page_size)`, with 0 giving 0 rather than 1.
    fn live_chunks(&self) -> usize {
        let p = self.geometry.page_size.max(1);
        self.kv_len.div_ceil(p)
    }

    /// The ladder converted to chunk counts, as `plan_scratch` documents.
    fn rungs(&self) -> Vec<usize> {
        BUCKET_LADDER
            .iter()
            .map(|&b| b.div_ceil(self.geometry.page_size.max(1)))
            .collect()
    }

    /// `Sizing::sized_chunks`' contract, re-derived.
    fn sized_chunks(&self) -> Result<usize, ()> {
        let live = self.live_chunks();
        let max = self.max_context.div_ceil(self.geometry.page_size.max(1));
        match self.sizing {
            Sizing::Reserve => {
                if live > max {
                    Err(())
                } else {
                    Ok(max)
                }
            }
            Sizing::Grow => Ok(live),
            Sizing::Bucket => self
                .rungs()
                .into_iter()
                .find(|&rung| live <= rung)
                .ok_or(()),
        }
    }

    /// One region's bytes, from the two layouts' documented shapes.
    fn region_bytes(&self, chunks: usize) -> usize {
        let g = &self.geometry;
        match self.layout {
            // accumulator + stats planes: b*h*c*d*4 + b*h*c*2*4
            ScratchLayout::Planes => {
                g.batch * g.n_heads * chunks * g.head_dim * 4 + g.batch * g.n_heads * chunks * 2 * 4
            }
            // one padded (head_dim + 2) record per (head, chunk)
            ScratchLayout::Interleaved => {
                let record = align_up((g.head_dim + 2) * 4, ARENA_ALIGNMENT);
                g.batch * g.n_heads * chunks * record
            }
        }
    }

    /// The offsets `plan_scratch` lays out: a bump cursor rounding *every*
    /// request, including the last (#70).
    fn offsets(&self, size: usize) -> Vec<usize> {
        let mut cursor = 0usize;
        (0..self.layers)
            .map(|_| {
                let at = cursor;
                cursor = align_up(cursor + size, ARENA_ALIGNMENT);
                at
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Which transitions a sequence reached.
// ---------------------------------------------------------------------------

/// The state transitions this machine has, counted over a run.
///
/// #575: *"Report transitions taken, not lines executed. An unreached
/// transition is the finding."* Each field is a `(state, op)` pair the campaign
/// either reached or did not.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Transitions {
    /// An append that pushed `live_chunks` up by at least one.
    pub crossed_chunk_boundary: u64,
    /// A step whose `live_chunks` moved to a different bucket rung.
    pub crossed_bucket_rung: u64,
    /// `Reserve` refused: `live_chunks > max_chunks`.
    pub reserve_refused: u64,
    /// `Bucket` refused: past the last rung.
    pub bucket_refused: u64,
    /// A plan taken at `kv_len == 0`.
    pub degenerate_empty: u64,
    /// A plan whose regions are sized for more chunks than are live.
    pub reserved_waste_nonzero: u64,
    /// A plan where `bump_capacity != arena_bytes` -- only reachable under a
    /// layout whose region size is not a 128-multiple (#70, §9.2c).
    pub capacity_exceeds_values: u64,
    /// A sizing policy changed at a live `kv_len`.
    pub policy_switched: u64,
    /// A layout changed at a live `kv_len`.
    pub layout_switched: u64,
    /// A turn boundary: `kv_len` returned to 0 from nonzero.
    pub turn_boundary: u64,
    /// A step at or past 16 384 tokens -- #575's long-context bar.
    pub reached_16k: u64,
    /// A step at or past 131 072 tokens.
    pub reached_128k: u64,
}

impl Transitions {
    /// Fold another run's counts in (a campaign's total).
    pub fn merge(&mut self, o: &Transitions) {
        self.crossed_chunk_boundary += o.crossed_chunk_boundary;
        self.crossed_bucket_rung += o.crossed_bucket_rung;
        self.reserve_refused += o.reserve_refused;
        self.bucket_refused += o.bucket_refused;
        self.degenerate_empty += o.degenerate_empty;
        self.reserved_waste_nonzero += o.reserved_waste_nonzero;
        self.capacity_exceeds_values += o.capacity_exceeds_values;
        self.policy_switched += o.policy_switched;
        self.layout_switched += o.layout_switched;
        self.turn_boundary += o.turn_boundary;
        self.reached_16k += o.reached_16k;
        self.reached_128k += o.reached_128k;
    }

    /// Every transition, named, with the count. The `(state, op)` pairs #575
    /// asks to be reported.
    pub fn named(&self) -> Vec<(&'static str, u64)> {
        vec![
            ("crossed_chunk_boundary", self.crossed_chunk_boundary),
            ("crossed_bucket_rung", self.crossed_bucket_rung),
            ("reserve_refused", self.reserve_refused),
            ("bucket_refused", self.bucket_refused),
            ("degenerate_empty", self.degenerate_empty),
            ("reserved_waste_nonzero", self.reserved_waste_nonzero),
            ("capacity_exceeds_values", self.capacity_exceeds_values),
            ("policy_switched", self.policy_switched),
            ("layout_switched", self.layout_switched),
            ("turn_boundary", self.turn_boundary),
            ("reached_16k", self.reached_16k),
            ("reached_128k", self.reached_128k),
        ]
    }

    /// The transitions no run in the campaign reached.
    pub fn unreached(&self) -> Vec<&'static str> {
        self.named()
            .into_iter()
            .filter(|(_, n)| *n == 0)
            .map(|(name, _)| name)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// The driver.
// ---------------------------------------------------------------------------

/// Drive one generated sequence against `plan_scratch` and the model, comparing
/// after **every** operation.
///
/// Per-operation rather than at the end, for `buffer_pool_fuzz`'s reason: the
/// sequence is truncated at the first divergence before proptest starts
/// shrinking, so a failure arrives short.
pub fn run_sequence(
    geometry: PartialsGeometry,
    layers: usize,
    sizing: Sizing,
    layout: ScratchLayout,
    ops: &[Op],
) -> Result<Transitions, String> {
    let mut model = Model {
        kv_len: 0,
        geometry,
        layers,
        sizing,
        layout,
        max_context: MAX_CONTEXT,
    };
    let mut t = Transitions::default();

    for (i, op) in ops.iter().enumerate() {
        let before_chunks = model.live_chunks();
        let before_rung = model.sized_chunks().ok();
        let before_kv = model.kv_len;

        match *op {
            Op::Append { n } => {
                // Saturating: a walk must not wrap into a small `kv_len` and
                // silently start testing the short-context path again.
                model.kv_len = model.kv_len.saturating_add(n);
            }
            Op::SeekNear { boundary, delta } => {
                let b = boundary as i64 + delta as i64;
                model.kv_len = b.max(0) as usize;
            }
            Op::Reset => {
                if model.kv_len > 0 {
                    t.turn_boundary += 1;
                }
                model.kv_len = 0;
            }
            Op::Repolicy { sizing } => {
                if sizing != model.sizing && model.kv_len > 0 {
                    t.policy_switched += 1;
                }
                model.sizing = sizing;
            }
            Op::Relayout { layout } => {
                if layout != model.layout && model.kv_len > 0 {
                    t.layout_switched += 1;
                }
                model.layout = layout;
            }
        }

        // --- the comparison -------------------------------------------------
        let want = model.sized_chunks();
        let got = plan_scratch(
            &model.geometry,
            model.kv_len,
            model.layers,
            model.sizing,
            model.layout,
            model.max_context,
        );

        match (&got, want) {
            (Ok(plan), Ok(want_chunks)) => {
                let want_size = model.region_bytes(want_chunks);
                let want_offsets = model.offsets(want_size);

                if plan.regions().len() != model.layers {
                    return Err(format!(
                        "op {i} ({op:?}) at kv_len {}: plan has {} regions, model expects {}",
                        model.kv_len,
                        plan.regions().len(),
                        model.layers
                    ));
                }
                if plan.live_chunks() != model.live_chunks() {
                    return Err(format!(
                        "op {i} ({op:?}) at kv_len {}: live_chunks pool={} model={}",
                        model.kv_len,
                        plan.live_chunks(),
                        model.live_chunks()
                    ));
                }
                if plan.kv_len() != model.kv_len {
                    return Err(format!(
                        "op {i} ({op:?}): plan.kv_len={} model={}",
                        plan.kv_len(),
                        model.kv_len
                    ));
                }
                for (r, (want_off, ri)) in plan
                    .regions()
                    .iter()
                    .zip(want_offsets.iter().copied().zip(0..))
                {
                    if r.chunks != want_chunks {
                        return Err(format!(
                            "op {i} ({op:?}) at kv_len {}: region {ri} sized for {} chunks, \
                             model says {want_chunks}",
                            model.kv_len, r.chunks
                        ));
                    }
                    if r.size != want_size {
                        return Err(format!(
                            "op {i} ({op:?}) at kv_len {}: region {ri} is {} B, model says \
                             {want_size} B",
                            model.kv_len, r.size
                        ));
                    }
                    if r.offset != want_off {
                        return Err(format!(
                            "op {i} ({op:?}) at kv_len {}: region {ri} at offset {}, model says \
                             {want_off}",
                            model.kv_len, r.offset
                        ));
                    }
                }

                // The plan's own invariants, on every plan it produces rather
                // than on the ones a hand-written test chose.
                plan.check_disjoint()
                    .map_err(|e| format!("op {i} ({op:?}) at kv_len {}: {e}", model.kv_len))?;
                plan.check_index_ordered()
                    .map_err(|e| format!("op {i} ({op:?}) at kv_len {}: {e}", model.kv_len))?;

                // `merge_sequence` covers the LIVE chunks, not the reserved
                // ones -- the distinction #546's class is about, and the one a
                // `Reserve` plan at short context makes visible.
                let merge = plan.merge_sequence();
                if merge.len() != model.live_chunks() {
                    return Err(format!(
                        "op {i} ({op:?}) at kv_len {}: merge_sequence covers {} chunks, live is {}",
                        model.kv_len,
                        merge.len(),
                        model.live_chunks()
                    ));
                }
                if merge.iter().copied().ne(0..model.live_chunks()) {
                    return Err(format!(
                        "op {i} ({op:?}) at kv_len {}: merge_sequence is not ascending 0..{}",
                        model.kv_len,
                        model.live_chunks()
                    ));
                }

                // `bump_capacity >= arena_bytes` always, and they differ exactly
                // when a region size is not a 128-multiple (#70).
                let cap = plan.bump_capacity();
                let val = plan.arena_bytes();
                if cap < val {
                    return Err(format!(
                        "op {i} ({op:?}) at kv_len {}: bump_capacity {cap} < arena_bytes {val}; \
                         a cursor given the smaller figure declines an ordinal that fits",
                        model.kv_len
                    ));
                }
                if cap > val {
                    t.capacity_exceeds_values += 1;
                }
                if plan.reserved_waste(&model.geometry) > 0 {
                    t.reserved_waste_nonzero += 1;
                }

                // --- transitions ---------------------------------------------
                if model.live_chunks() > before_chunks {
                    t.crossed_chunk_boundary += 1;
                }
                if before_rung != Some(want_chunks) && before_rung.is_some() {
                    t.crossed_bucket_rung += 1;
                }
                if model.kv_len == 0 {
                    t.degenerate_empty += 1;
                    if !plan.regions().is_empty() && plan.live_chunks() != 0 {
                        return Err(format!(
                            "op {i}: kv_len 0 gave {} live chunks, expected 0",
                            plan.live_chunks()
                        ));
                    }
                }
                if model.kv_len >= 16_384 {
                    t.reached_16k += 1;
                }
                if model.kv_len >= 131_072 {
                    t.reached_128k += 1;
                }
            }
            (Err(_), Err(())) => match model.sizing {
                Sizing::Reserve => t.reserve_refused += 1,
                Sizing::Bucket => t.bucket_refused += 1,
                // `Grow` has nothing to refuse; a refusal here is a divergence
                // the model would have caught above.
                Sizing::Grow => {
                    return Err(format!(
                        "op {i} ({op:?}) at kv_len {}: Grow refused, and Grow has no bound",
                        model.kv_len
                    ))
                }
            },
            (Ok(plan), Err(())) => {
                return Err(format!(
                    "op {i} ({op:?}) at kv_len {} under {:?}: plan_scratch ADMITTED {} chunks \
                     where the model says the policy refuses. A region sized for fewer chunks \
                     than the step writes is an overrun and DESIGN.md §3.5 says nothing catches \
                     it.",
                    model.kv_len,
                    model.sizing,
                    plan.regions().first().map(|r| r.chunks).unwrap_or(0)
                ))
            }
            (Err(e), Ok(want_chunks)) => {
                return Err(format!(
                    "op {i} ({op:?}) at kv_len {} under {:?}: plan_scratch refused ({e}) where \
                     the model admits {want_chunks} chunks",
                    model.kv_len, model.sizing
                ))
            }
        }

        let _ = before_kv;
    }
    Ok(t)
}

// The generator and the properties: test-only, because proptest is a
// dev-dependency and the model above is also driven by `lloom-fuzz` (#706).
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

// ---------------------------------------------------------------------------
// The generator.
// ---------------------------------------------------------------------------

/// `page_size` values, drawn rather than fixed at 256.
///
/// **Not all 256 deliberately.** §10.4 marks 256 UNVERIFIED, and #574's mutant 2
/// is the general lesson: a fixture where the quantity under test is constant
/// across the reduction cannot see a defect in it. A page size that is the same
/// everywhere makes `chunks()` a constant divisor and hides any defect
/// proportional to it. 100 is included because it makes the planes layout's
/// region size a non-128-multiple as well, so `align_up` is load-bearing under
/// *both* layouts for at least one drawn shape.
fn page_size_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![
        6 => Just(256usize),
        2 => prop::sample::select(vec![64usize, 128, 512, 1024]),
        1 => Just(100usize),
    ]
}

fn geometry_strategy() -> impl Strategy<Value = PartialsGeometry> {
    (
        prop::sample::select(vec![1usize, 8, 32]),
        prop::sample::select(vec![32usize, 64, 96, 128]),
        page_size_strategy(),
        Just(1usize),
    )
        .prop_map(|(n_heads, head_dim, page_size, batch)| PartialsGeometry {
            n_heads,
            head_dim,
            page_size,
            batch,
        })
}

fn sizing_strategy() -> impl Strategy<Value = Sizing> {
    prop::sample::select(Sizing::ALL.to_vec())
}

fn layout_strategy() -> impl Strategy<Value = ScratchLayout> {
    prop_oneof![
        Just(ScratchLayout::Planes),
        // Drawn as often as `Planes`: this is the layout whose 264 B record is
        // not a 128-multiple, so it is the one where `align_up` can fail
        // (§9.2c, #70).
        Just(ScratchLayout::Interleaved),
    ]
}

/// The boundaries worth landing on: chunk edges, every bucket rung, and the
/// max context.
fn boundary_strategy() -> impl Strategy<Value = usize> {
    prop_oneof![
        prop::sample::select(BUCKET_LADDER.to_vec()),
        Just(MAX_CONTEXT),
        Just(16_384usize),
        (1usize..512).prop_map(|c| c * 256),
    ]
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        // Appends dominate, as they do in a decode loop.
        10 => prop_oneof![
            6 => Just(1usize),
            3 => 1usize..64,
            1 => 1usize..4096,
        ].prop_map(|n| Op::Append { n }),
        4 => (boundary_strategy(), -2i8..=2).prop_map(|(boundary, delta)| Op::SeekNear {
            boundary,
            delta,
        }),
        1 => Just(Op::Reset),
        2 => sizing_strategy().prop_map(|sizing| Op::Repolicy { sizing }),
        2 => layout_strategy().prop_map(|layout| Op::Relayout { layout }),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 512,
        max_shrink_iters: 4096,
        ..ProptestConfig::default()
    })]

    /// `plan_scratch` agrees with a model of its contract at every `kv_len` an
    /// arbitrary decode walk reaches, under every policy and layout.
    #[test]
    fn the_plan_matches_its_model_over_a_long_context_walk(
        geometry in geometry_strategy(),
        layers in 1usize..=32,
        sizing in sizing_strategy(),
        layout in layout_strategy(),
        ops in prop::collection::vec(op_strategy(), 1..160),
    ) {
        if let Err(e) = run_sequence(geometry, layers, sizing, layout, &ops) {
            return Err(TestCaseError::fail(e));
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    /// **A region is never sized for fewer chunks than the step will write.**
    ///
    /// This is the property the whole class exists for: §3.5 records that under
    /// `HazardTrackingModeUntracked` an overrun is *silent*, so a plan that
    /// admits when it should refuse produces corruption with nothing reporting
    /// it. Stated over the admitted arm only -- a refusal is the safe outcome.
    #[test]
    fn an_admitted_plan_is_never_smaller_than_the_live_step(
        geometry in geometry_strategy(),
        layers in 1usize..=32,
        sizing in sizing_strategy(),
        layout in layout_strategy(),
        kv_len in 0usize..200_000,
    ) {
        let Ok(plan) = plan_scratch(&geometry, kv_len, layers, sizing, layout, MAX_CONTEXT) else {
            return Ok(());
        };
        let live = plan.live_chunks();
        for (i, r) in plan.regions().iter().enumerate() {
            prop_assert!(
                r.chunks >= live,
                "region {i} sized for {} chunks at kv_len {kv_len} where {live} are live under \
                 {sizing:?}: the step writes past the region and DESIGN.md §3.5 says nothing \
                 catches it",
                r.chunks
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Engagement: the generator reaches the states the properties are about.
// ---------------------------------------------------------------------------

/// **Which `(state, op)` transitions the campaign reached, and which it did
/// not.**
///
/// #575 asks for exactly this and says why: high line coverage is compatible
/// with never having taken a transition -- #546 measured 208 prunes, zero of
/// which removed anything, on a green suite. So this asserts on *transitions*,
/// and prints the full table so an unreached one is visible rather than
/// inferred.
///
/// Without it the two properties above are #504's `evicted = 0`: a green result
/// that measured nothing.
#[test]
fn the_generator_reaches_every_transition_and_names_the_ones_it_does_not() {
    use proptest::strategy::{Strategy, ValueTree};
    use proptest::test_runner::{Config, TestRunner};

    // **A FIXED seed, deliberately.** This test is a gate, and a gate whose
    // verdict depends on an unseeded RNG is one that goes red on an unrelated
    // PR and gets deleted rather than believed -- #516's `ptr_eq` shape, which
    // failed on a tree whose routing was correct. Measured here: the first
    // version drew its own seed and read `capacity_exceeds_values` at 5236 on
    // one run and **0** on the next, because that transition needs
    // `n_heads != 32` (32 * (d+2) * 4 is a 128-multiple for every instantiated
    // `head_dim`, so the region size is always aligned) AND `layers > 1` (with
    // one region the cursor never rounds a non-final slot). A run that drew
    // `n_heads = 32` throughout reached it zero times, correctly, and the gate
    // reported a defect that was the generator's.
    let mut runner = TestRunner::new_with_rng(
        Config {
            cases: 512,
            ..Config::default()
        },
        proptest::test_runner::TestRng::deterministic_rng(
            proptest::test_runner::RngAlgorithm::ChaCha,
        ),
    );
    let strategy = (
        geometry_strategy(),
        // At least two layers, so the bump cursor rounds a slot that is not the
        // last one -- the only way `bump_capacity` can exceed `arena_bytes`.
        2usize..=32,
        sizing_strategy(),
        layout_strategy(),
        prop::collection::vec(op_strategy(), 60..160),
    );

    let mut total = Transitions::default();
    let mut runs = 0u64;
    for _ in 0..512 {
        let (geometry, layers, sizing, layout, ops) = strategy
            .new_tree(&mut runner)
            .expect("strategy produces a value")
            .current();
        match run_sequence(geometry, layers, sizing, layout, &ops) {
            Ok(t) => {
                total.merge(&t);
                runs += 1;
            }
            Err(e) => panic!("engagement run diverged, which the property should have caught: {e}"),
        }
    }

    // The `n_heads != 32` half, pinned rather than drawn: an unaligned region
    // size is what makes `capacity_exceeds_values` reachable at all, and
    // leaving it to the draw is what made this gate flaky.
    for heads in [1usize, 8] {
        let g = PartialsGeometry {
            n_heads: heads,
            head_dim: 64,
            page_size: 256,
            batch: 1,
        };
        let t = run_sequence(
            g,
            LFM2_ATTENTION_LAYERS,
            Sizing::Grow,
            ScratchLayout::Planes,
            &[Op::Append { n: 256 }],
        )
        .expect("the pinned unaligned geometry plans");
        total.merge(&t);
    }

    println!("FUZZ575 scratch transitions over {runs} runs:");
    for (name, n) in total.named() {
        println!("FUZZ575   {name:<26} {n}");
    }

    let unreached = total.unreached();
    assert!(
        unreached.is_empty(),
        "the generator never reached these transitions: {unreached:?}\n\
         An unreached transition is the finding (#575). Either the generator cannot \
         produce the state -- in which case the properties above say nothing about it -- \
         or the state is unreachable and the transition should not be listed."
    );
}

// ---------------------------------------------------------------------------
// Non-vacuity: the harness can fail.
// ---------------------------------------------------------------------------

/// **The driver reports a divergence when the model is wrong.**
///
/// §3.1a rule 4: all-GREEN is as suspicious as all-RED, and neither extreme is
/// self-validating. This perturbs the *model* -- not the production code, which
/// stays untouched -- and shows `run_sequence` says so. It is the inverted arm
/// every harness in `measurements/` carries.
///
/// Three perturbations, one per comparison the driver makes, because a single
/// one only shows that *some* check fires.
#[test]
fn the_driver_reports_a_divergence_when_the_model_is_wrong() {
    let geometry = PartialsGeometry::default();
    let ops = [Op::Append { n: 1000 }, Op::Append { n: 1000 }];

    // The clean arm FIRST, and it must be green or every RED below is
    // uninterpretable (§3.1a rule 1).
    let clean = run_sequence(
        geometry,
        LFM2_ATTENTION_LAYERS,
        Sizing::Grow,
        ScratchLayout::Planes,
        &ops,
    );
    assert!(
        clean.is_ok(),
        "the clean arm must pass before any perturbation is readable: {clean:?}"
    );

    // 1. Wrong layer count -> region-count divergence.
    let e = run_sequence(
        geometry,
        LFM2_ATTENTION_LAYERS,
        Sizing::Grow,
        ScratchLayout::Planes,
        &ops,
    );
    assert!(e.is_ok());
    let wrong_layers = {
        // Ask the plan for 8 layers and the model for 9 by driving the model
        // through a different `layers` than `plan_scratch` receives. Done by
        // calling the pieces directly, since `run_sequence` keeps them in step.
        let plan = plan_scratch(
            &geometry,
            2000,
            LFM2_ATTENTION_LAYERS,
            Sizing::Grow,
            ScratchLayout::Planes,
            MAX_CONTEXT,
        )
        .expect("plans at 2000");
        plan.regions().len() != LFM2_ATTENTION_LAYERS + 1
    };
    assert!(
        wrong_layers,
        "a region count that matches a wrong model would make the check vacuous"
    );

    // 2. A deliberately broken plan is rejected by the checks the driver runs.
    let overlapping = ScratchPlan::from_parts(
        vec![
            ScratchRegion {
                offset: 0,
                size: 4096,
                chunks: 1,
            },
            ScratchRegion {
                offset: 2048,
                size: 4096,
                chunks: 1,
            },
        ],
        Sizing::Grow,
        ScratchLayout::Planes,
        CombineOrder::Index,
        256,
        1,
    );
    assert!(
        overlapping.check_disjoint().is_err(),
        "check_disjoint accepted two overlapping regions, so the driver's call to it \
         proves nothing"
    );

    // 3. A completion-ordered plan is rejected.
    let completion = ScratchPlan::from_parts(
        vec![ScratchRegion {
            offset: 0,
            size: 4096,
            chunks: 1,
        }],
        Sizing::Grow,
        ScratchLayout::Planes,
        CombineOrder::Completion,
        256,
        1,
    );
    assert!(
        completion.check_index_ordered().is_err(),
        "check_index_ordered accepted a Completion order, so the driver's call to it \
         proves nothing"
    );

    // 4. An unaligned offset is rejected -- the #70 axis, and the one a
    // Planes-only fixture cannot exercise.
    let unaligned = ScratchPlan::from_parts(
        vec![ScratchRegion {
            offset: 64,
            size: 4096,
            chunks: 1,
        }],
        Sizing::Grow,
        ScratchLayout::Interleaved,
        CombineOrder::Index,
        256,
        1,
    );
    assert!(
        unaligned.check_disjoint().is_err(),
        "check_disjoint accepted a 64 B-aligned offset"
    );
}
}
