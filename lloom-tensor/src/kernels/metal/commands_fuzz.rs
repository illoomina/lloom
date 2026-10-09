//! A stateful model-based fuzzer over `Commands`' submission surface, and the
//! boundedness oracle nothing had exercised (issue #508).
//!
//! # What this target is, and why it is not the pool's
//!
//! #494/#504 fuzzed the **buffer pool**: an allocator whose state is counters,
//! a `BTreeMap` free list and an epoch clock. This is the **command-buffer
//! queue** one layer up, and the two differ in the property under test.
//!
//! `Commands::flush_and_prune` (`commands.rs:641`) exists because
//! `DESIGN.md` §11.5h found that removing the per-token readback removed the
//! only thing pruning `EntryState::in_flight`:
//!
//! > `in_flight` is a `Vec<CommandBuffer>` that shrinks on exactly two paths --
//! > `wait_until_completed`'s `mem::take` and `flush_and_wait_current`'s
//! > `retain` -- both of which block.
//!
//! Its repair **returns the surviving count so a caller can assert the queue is
//! bounded rather than assume it** (§2.4). §11.5j then observed that count from
//! the harness -- `max_in_flight` flat at 13-15 across a 7.8x span of `n`,
//! against **63 at `n` = 64** with the prune removed. **That assertion is an
//! oracle, and until now the only sequence it had ever been checked against is
//! the one the decode loop happens to produce**: `commit -> prune -> read flag`,
//! the same three operations in the same order, every token.
//!
//! # The shape: one shrink site against one growth site
//!
//! §17's registered prediction for #508 rests on this asymmetry, so it is worth
//! stating from the source rather than from the issue.
//!
//! **Growth** is `commit_swap_locked` (`:713`), which ends with
//! `state.in_flight.push(old_cb)` unconditionally. It is reached from
//! `command_encoder` (on a cut), `flush`, `flush_and_wait`,
//! `flush_and_wait_current` and `flush_and_prune` -- five callers, one push.
//!
//! **Shrink** is three sites with three different predicates:
//!
//! | site | predicate | blocks? |
//! |---|---|---|
//! | `flush_and_wait` | `mem::take` -- unconditional, takes everything | yes |
//! | `flush_and_wait_current` | `retain(status != Completed)` after waiting on the last | yes |
//! | `flush_and_prune` | `retain(status != Completed)`, **no wait** | **no** |
//!
//! **The third one's bound depends on a caller doing the right thing**, which is
//! the shape #493 already caught once by reading source: it prunes whatever the
//! GPU *happens* to have finished at the instant it is called. Nothing inside
//! `Commands` bounds the queue; the bound is a *joint* property of the prune and
//! of the GPU making progress, and a generated sequence is what tests a joint
//! property that one loop shape cannot.
//!
//! # This is a GPU target and it is still lease-free
//!
//! #504 established the pool's alphabet reaches the allocator and never the
//! execution units, which is what made it lease-free. **That argument does not
//! transfer and is not reused.** This alphabet commits real command buffers to
//! a real queue: the growth site *is* `create_command_buffer`, and a model that
//! stubbed it would be modelling something else.
//!
//! What makes it cheap anyway is that the work encoded is deliberately trivial
//! -- a small `cos` over 64 f32 -- so a command buffer's *execution* is
//! microseconds while its *submission bookkeeping* is what the property is
//! about. Measured in `the_generator_actually_reaches_the_interesting_states`
//! below, and the counters are printed rather than asserted-in-the-dark.
//!
//! # The model is written from the contract
//!
//! Per #504: a model mirroring the implementation cannot find a bug in it. This
//! one tracks what the queue's length **should** be from the doc comments on
//! the five entry points, and its central claim is not an equality but a
//! **bound** -- which is the honest shape, because `retain(status !=
//! Completed)` depends on the GPU's progress and is therefore not a
//! deterministic function of the operation sequence.
//!
//! So the model predicts:
//!
//! * an **exact** length after every blocking operation (they are deterministic
//!   -- `flush_and_wait` takes everything, `flush_and_wait_current` waits then
//!   retains, so both leave a length the sequence determines);
//! * an **upper bound** after a non-blocking one, since the GPU may or may not
//!   have finished any given buffer;
//! * and the **invariant that outranks both**: the queue never exceeds the
//!   number of commits since the last blocking drain. That is the property
//!   §11.5h's return value exists to let a caller assert, and it is the one a
//!   fuzzer can hold across arbitrary interleavings.

#![cfg(any(test, feature = "fuzzing"))]

use super::commands::Commands;
use super::residency_set::ResidencySet;
use crate::kernels::{call_unary_contiguous, unary, BufferOffset, Device, Kernels};
use std::sync::Arc;

/// One symbol of the submission alphabet.
///
/// This is `Commands`' public surface that touches `in_flight`, minus the
/// accessors (`clock`, `executor`, `cut_policy`, `compute_per_buffer`) which
/// are pure reads, and minus `blit_command_encoder` -- which reaches the same
/// `commit_swap_locked` through the same lock and would add a second encoder
/// kind without adding a transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Encode `n` trivial dispatches. Each takes `command_encoder()`, which
    /// increments `compute_count` and cuts a new command buffer when the policy
    /// says so -- so this is the operation that reaches the growth site
    /// *without* the caller asking for a commit.
    Encode { dispatches: usize },
    /// `flush()` -- commit without waiting and without pruning. The purest
    /// growth operation: it pushes and never pops.
    Flush,
    /// `flush_and_prune()` -- the non-blocking commit under test. Records the
    /// returned surviving count.
    FlushAndPrune,
    /// `flush_and_wait_current()` -- commit, wait on the last, retain the
    /// unfinished.
    FlushAndWaitCurrent,
    /// `flush_and_wait()` -- commit and take everything.
    FlushAndWait,
    /// Sleep briefly, letting the GPU retire what is in flight without the host
    /// waiting on it.
    ///
    /// **This is not padding and it is the operation that makes the target
    /// interesting.** `flush_and_prune`'s `retain` is a function of GPU
    /// progress, so a sequence with no time in it prunes nothing and the
    /// non-blocking path is never observed doing its job -- which is exactly
    /// #504's `evicted=0` trap in this alphabet. The engagement assertion below
    /// is what proves it fires.
    Settle,
}

/// What the queue's length should be, tracked from the documented contract.
///
/// Deliberately does not hold `CommandBuffer`s: the model has no business
/// knowing about Metal objects, and one that shared the implementation's data
/// structure would share its bugs.
#[derive(Debug, Default)]
struct Model {
    /// Commits since the last operation that emptied the queue. The upper bound
    /// on `in_flight.len()`, and the quantity the boundedness claim is about.
    committed_since_drain: usize,
    /// Dispatches encoded into the current (uncommitted) command buffer.
    compute_count: usize,
    /// Total commits, for the engagement counters.
    commits: u64,
}

impl Model {
    /// A commit pushes one buffer, unconditionally (`commit_swap_locked:713`).
    fn commit(&mut self) {
        self.committed_since_drain += 1;
        self.compute_count = 0;
        self.commits += 1;
    }

    /// `flush`, `flush_and_prune`, `flush_and_wait` and `flush_and_wait_current`
    /// each commit **only if** there is encoded work -- except
    /// `flush_and_wait_current`, which commits unconditionally.
    ///
    /// Read from source rather than assumed: `flush` (`:607`), `flush_and_wait`
    /// (`:555`) and `flush_and_prune` (`:641`) all guard on
    /// `compute_count.load() > 0`, while `flush_and_wait_current` (`:590`) calls
    /// `commit_swap_locked` with no guard. **That asymmetry is real and is what
    /// lets a `flush_and_wait_current` on an idle queue still push a buffer.**
    fn commit_if_work(&mut self) {
        if self.compute_count > 0 {
            self.commit();
        }
    }
}

/// Encode `n` trivial dispatches through the real encoder path.
///
/// The work is deliberately tiny: the property is about submission bookkeeping,
/// not about GPU time, and a large kernel would make the suite slow without
/// making the queue behave differently.
fn encode(
    device: &Device,
    kernels: &Kernels,
    cmds: &Commands,
    src: &crate::kernels::Buffer,
    dst: &crate::kernels::Buffer,
    n: usize,
) -> Result<(), String> {
    for _ in 0..n {
        let guard = cmds
            .command_encoder()
            .map_err(|e| format!("command_encoder: {e:?}"))?;
        call_unary_contiguous(
            device,
            &guard,
            kernels,
            unary::contiguous::cos::FLOAT,
            std::mem::size_of::<f32>(),
            64,
            BufferOffset::zero_offset(src),
            dst,
        )
        .map_err(|e| format!("call_unary_contiguous: {e:?}"))?;
    }
    Ok(())
}

/// Run one generated sequence against the real `Commands` and its model.
///
/// Returns the failure description on divergence, which is what proptest
/// shrinks. Per #494/#504: a 400-operation failing sequence is not a bug report
/// and a 4-operation one is.
pub fn run_sequence(dev: &Device, ops: &[Op]) -> Result<Engagement, String> {
    // An autorelease pool per sequence. `commandBuffer`, the encoders and the
    // device return autoreleased objects; on a thread with no pool they are
    // never released ("autoreleased with no pool in place - just leaking",
    // `OBJC_DEBUG_MISSING_POOLS=YES`), and in a debug build -- where objc2's
    // retain of an autoreleased return does not take the claim fast path --
    // ~460 sequences of encoded work exhaust the process: the next commit
    // returns `Internal Error (00000206)` (lloom #706 follow-up).
    objc2::rc::autoreleasepool(|_| run_sequence_inner(dev, ops))
}

fn run_sequence_inner(dev: &Device, ops: &[Op]) -> Result<Engagement, String> {
    let queue = dev
        .new_command_queue()
        .map_err(|e| format!("new_command_queue: {e:?}"))?;
    let residency_set = Arc::new(ResidencySet::new(dev));
    let cmds = Commands::new(queue, residency_set).map_err(|e| format!("Commands::new: {e:?}"))?;
    // One pipeline cache per process, not per sequence: a fresh `Kernels`
    // per run compiles the `cos` pipeline again for every sequence, and in a
    // debug build ~460 of those exhaust the process (lloom #706 follow-up:
    // `Internal Error (00000206)` on the next commit; release runs 1200 clean).
    static KERNELS: std::sync::OnceLock<Kernels> = std::sync::OnceLock::new();
    let kernels = KERNELS.get_or_init(Kernels::new);

    let options = crate::kernels::RESOURCE_OPTIONS;
    let input = [1.0f32; 64];
    let bytes = std::mem::size_of_val(&input);
    let src = dev
        .new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, bytes, options)
        .map_err(|e| format!("new_buffer_with_data: {e:?}"))?;
    let dst = dev
        .new_buffer(bytes, options)
        .map_err(|e| format!("new_buffer: {e:?}"))?;

    let mut model = Model::default();
    let mut eng = Engagement::default();
    let period = cmds.compute_per_buffer();

    for (i, op) in ops.iter().enumerate() {
        let at = |what: &str| format!("after op {i} ({op:?}): {what}");

        match op {
            Op::Encode { dispatches } => {
                encode(dev, kernels, &cmds, &src, &dst, *dispatches)
                    .map_err(|e| at(&format!("encoding: {e}")))?;
                // The encoder cuts when the dispatch counter reaches the
                // period, so `n` dispatches produce as many commits as the
                // counter crosses it. Modelled by stepping one at a time
                // rather than by dividing, because the counter is not reset to
                // zero on a cut -- `commit_swap_locked(.., 1)` resets it to
                // **one**, the dispatch that caused the cut.
                for _ in 0..*dispatches {
                    if model.compute_count >= period {
                        model.commit();
                        model.compute_count = 1;
                    } else {
                        model.compute_count += 1;
                    }
                }
                eng.encoded += *dispatches as u64;
            }
            Op::Flush => {
                cmds.flush().map_err(|e| at(&format!("flush: {e:?}")))?;
                model.commit_if_work();
            }
            Op::FlushAndPrune => {
                // Read the length before and after, so "did it prune" is a
                // measurement of this call rather than an inference from the
                // model. The `before` read is what makes the engagement counter
                // honest: the model's bound is an upper bound, so
                // `surviving < bound` would count a prune whenever the GPU had
                // finished anything, including work an earlier operation
                // committed.
                let before = cmds
                    .in_flight_len()
                    .map_err(|e| at(&format!("reading in_flight before the prune: {e:?}")))?;
                let surviving = cmds
                    .flush_and_prune()
                    .map_err(|e| at(&format!("flush_and_prune: {e:?}")))?;
                let committed = model.compute_count > 0;
                model.commit_if_work();
                eng.prunes += 1;
                eng.max_surviving = eng.max_surviving.max(surviving);
                // This call pushed at most one buffer (iff it committed), so
                // anything below `before + pushed` was removed by the `retain`.
                let pushed = usize::from(committed);
                if surviving < before + pushed {
                    eng.pruned_something += 1;
                }

                // **The oracle.** `flush_and_prune` returns the surviving count
                // so a caller can assert boundedness rather than assume it
                // (§11.5h). The bound is the number of commits since the last
                // operation that emptied the queue: the prune can only remove
                // buffers, never add, and the growth site pushes exactly one
                // per commit.
                //
                // Stated as `<=` rather than `==` deliberately. Whether a given
                // buffer has reached `Completed` at the instant `retain` runs
                // is a property of the GPU's progress, not of the operation
                // sequence -- so an equality here would be a test of the
                // scheduler, which `DESIGN.md` §3.1b names as the thing not to
                // assert. What is a genuine invariant is that the queue cannot
                // exceed what was pushed into it.
                if surviving > model.committed_since_drain {
                    return Err(at(&format!(
                        "flush_and_prune returned {surviving} surviving command buffers, \
                         but only {} commits have happened since the queue was last emptied. \
                         The queue grew without a commit, or a drain did not empty it.",
                        model.committed_since_drain
                    )));
                }
                // The prune observed the queue: adopt its count as the new
                // ceiling, since anything it retained is what remains.
                model.committed_since_drain = surviving;
            }
            Op::FlushAndWaitCurrent => {
                cmds.flush_and_wait_current()
                    .map_err(|e| at(&format!("flush_and_wait_current: {e:?}")))?;
                // Commits unconditionally (no `compute_count` guard at `:590`),
                // then waits on the last and retains the unfinished. Metal's
                // queue is FIFO, so everything committed before the one waited
                // on is complete -- the retain therefore empties the queue.
                model.commit();
                model.committed_since_drain = 0;
                model.compute_count = 0;
                eng.blocking_drains += 1;
            }
            Op::FlushAndWait => {
                cmds.flush_and_wait()
                    .map_err(|e| at(&format!("flush_and_wait: {e:?}")))?;
                // `mem::take` is unconditional: whatever was in flight is gone.
                model.commit_if_work();
                model.committed_since_drain = 0;
                model.compute_count = 0;
                eng.blocking_drains += 1;
            }
            Op::Settle => {
                std::thread::sleep(std::time::Duration::from_millis(2));
                eng.settles += 1;
            }
        }

        // **The standing invariant, checked after every operation.** The
        // queue is bounded by the commits since it was last emptied, whatever
        // the interleaving. This is the claim §11.5h's return value exists to
        // support, and the one #513 observed on a single loop shape.
        //
        // **Read through `in_flight_len`, which does not prune, and that is a
        // repair rather than a detail.** The first version of this observer
        // called `flush_and_prune` -- the only length-reporting call that
        // existed -- reasoning that its `compute_count > 0` guard makes it a
        // pure observation on an idle queue. The guard does stop it
        // *committing*; it does not stop it *pruning*, and `retain` runs
        // unconditionally. So the observation consumed the event it existed to
        // observe: every generated `FlushAndPrune` found a queue this line had
        // already emptied, and the engagement counter read
        // **`pruned_something = 0` across 208 prunes**.
        //
        // That is §15.1a's VACUOUS class in the observer, and it is #504's
        // `evicted = 0` reproduced in a second alphabet -- caught the same way,
        // by an engagement assertion that was run first and made to fail.
        let observed = cmds
            .in_flight_len()
            .map_err(|e| at(&format!("observing the queue: {e:?}")))?;
        if observed > model.committed_since_drain {
            return Err(at(&format!(
                "in_flight holds {observed} command buffers against a bound of {}",
                model.committed_since_drain
            )));
        }
        eng.max_observed = eng.max_observed.max(observed);
    }

    // Leave the queue drained so the next case starts clean and no command
    // buffer outlives the `Commands` that owns it.
    cmds.flush_and_wait()
        .map_err(|e| format!("final drain: {e:?}"))?;
    eng.commits = model.commits;
    Ok(eng)
}

/// What a run reached, so a green result can be shown to have measured
/// something (§2.4, and #504's own `evicted=0`).
#[derive(Debug, Default, Clone)]
pub struct Engagement {
    pub encoded: u64,
    pub commits: u64,
    pub prunes: u64,
    pub blocking_drains: u64,
    pub settles: u64,
    /// Prunes that actually removed a buffer. **The counter that matters**: a
    /// prune that never prunes is this target's `evicted=0`.
    pub pruned_something: u64,
    pub max_surviving: usize,
    pub max_observed: usize,
}

// The generator and the properties: test-only, because proptest is a
// dev-dependency and the model above is also driven by `lloom-fuzz` (#706).
#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseError;

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            // Encoding is what grows the queue without asking, so it is weighted
            // highest. The range spans the default cut period (10) so a single
            // `Encode` can cross it more than once.
            6 => (1usize..24).prop_map(|dispatches| Op::Encode { dispatches }),
            3 => Just(Op::FlushAndPrune),
            2 => Just(Op::Flush),
            2 => Just(Op::Settle),
            1 => Just(Op::FlushAndWaitCurrent),
            1 => Just(Op::FlushAndWait),
        ]
    }

    /// ONE device per process, not one per case. `Device::system_default()` per
    /// proptest case logged "Context leak detected, CoreAnalytics returned false"
    /// once per case (lloom #706). Recorded beside it, because it is NOT fixed by
    /// this: in a DEBUG build the commands proptest at 4096 cases fails on the
    /// ~476th case with `flush_and_wait` returning `Internal Error (00000206)`,
    /// with or without this cache, while the same 4096 cases pass in `--release`
    /// and the release LibAFL harness runs 1 200 sequences in one process clean
    /// (`issue-706-raw/commands-proptest-4096*.log`). The 48-case default never
    /// reaches it; the debug-only exhaustion is open.
    fn device() -> Device {
        static DEVICE: std::sync::OnceLock<Device> = std::sync::OnceLock::new();
        DEVICE
            .get_or_init(|| Device::system_default().expect("no Metal device"))
            .clone()
    }

    proptest! {
        // Smaller than #504's 256 x 120 because every operation here submits real
        // command buffers: the alphabet reaches the GPU where the pool's reached
        // only the allocator. Sized against the same criterion -- a suite that runs
        // every time is worth more than one that is skipped.
        #![proptest_config(ProptestConfig {
            cases: 48,
            max_shrink_iters: 2048,
            ..ProptestConfig::default()
        })]

        /// `in_flight` stays bounded by the commits since it was last emptied,
        /// across arbitrary commit / prune / wait interleavings.
        #[test]
        fn in_flight_stays_bounded(ops in prop::collection::vec(op_strategy(), 1..40)) {
            let dev = device();
            if let Err(e) = run_sequence(&dev, &ops) {
                return Err(TestCaseError::fail(e));
            }
        }
    }

    /// The generator reaches the states the property is about.
    ///
    /// **#504's first probe read 5 894 163 ops/s and `evicted=0`** -- a trim every
    /// 4096 operations kept the free list under budget, so the eviction path never
    /// ran and it was timing a state machine with a third of its transitions dead.
    /// The equivalent here is a `flush_and_prune` that never prunes: if the GPU has
    /// finished nothing when `retain` runs, the non-blocking path is exercised only
    /// on its no-op arm and a green suite has observed nothing.
    #[test]
    fn the_generator_actually_reaches_the_interesting_states() {
        use proptest::strategy::{Strategy, ValueTree};
        use proptest::test_runner::{Config, TestRunner};

        let dev = device();
        let mut runner = TestRunner::new(Config {
            cases: 32,
            ..Config::default()
        });
        let strategy = prop::collection::vec(op_strategy(), 20..40);

        let mut total = Engagement::default();
        for _ in 0..32 {
            let ops = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            let e = run_sequence(&dev, &ops).expect("engagement run must not fail");
            total.encoded += e.encoded;
            total.commits += e.commits;
            total.prunes += e.prunes;
            total.blocking_drains += e.blocking_drains;
            total.settles += e.settles;
            total.pruned_something += e.pruned_something;
            total.max_surviving = total.max_surviving.max(e.max_surviving);
            total.max_observed = total.max_observed.max(e.max_observed);
        }

        println!(
            "FUZZ508 engagement encoded={} commits={} prunes={} pruned_something={} \
         blocking_drains={} settles={} max_surviving={} max_observed={}",
            total.encoded,
            total.commits,
            total.prunes,
            total.pruned_something,
            total.blocking_drains,
            total.settles,
            total.max_surviving,
            total.max_observed,
        );

        assert!(total.commits > 0, "the generator never committed a buffer");
        assert!(
            total.prunes > 0,
            "the generator never called flush_and_prune, which is the operation under test"
        );
        assert!(
        total.blocking_drains > 0,
        "the generator never took a blocking drain, so the two shrink paths were never compared"
    );
        assert!(
            total.max_observed > 0,
            "the queue was never non-empty: every observation saw a drained queue, so the \
         boundedness invariant was checked only against zero"
        );
        // **The one that earns its place**, and the analogue of #504's `evicted=0`.
        assert!(
            total.pruned_something > 0,
            "flush_and_prune never removed a command buffer: the non-blocking prune was exercised \
         only on its no-op arm, so a green suite says nothing about whether it prunes"
        );
    }

    /// **One shrink site against one growth site, checked in source.**
    ///
    /// §17's registered prediction for #508 rests on this asymmetry, and a runtime
    /// test cannot see it: whether `in_flight` has a second unconditional push is a
    /// property of the *text*, and a sequence that never reaches a new push cannot
    /// report one. Same reasoning, and the same mechanism, as
    /// `pool_counters_holds_accumulators_only` (#507) and `Buffer`'s
    /// `black_box` guard (#454).
    ///
    /// Mutation-tested: adding a `state.in_flight.push(..)` outside
    /// `commit_swap_locked`, or removing the `retain` from `flush_and_prune`, makes
    /// this fail.
    ///
    /// # It reads the whole file, and that is a correction
    ///
    /// The first version sliced off everything after the first `#[cfg(test)]`,
    /// copying `pool_counters_holds_accumulators_only`'s guard against a check
    /// reading its own description. `commands.rs` has **no test module** -- so the
    /// only `#[cfg(test)]` in it is [`Commands::in_flight_len`], the accessor added
    /// for this very fuzzer, and the slice ended **57 lines before the push it was
    /// counting**. It read `found 0` and would have read `found 0` just as happily
    /// if the growth site had been deleted.
    ///
    /// **A borrowed guard is not a free one**: the reason that split is right in
    /// `buffer_pool_fuzz` is that `buffer_pool.rs` has a test module quoting the
    /// defective line, and neither half of that is true here. The repair is to read
    /// the file and assert the *counts* are the ones a reader can check against the
    /// line numbers below -- plus a non-vacuity arm, because a pattern that matches
    /// nothing and a file that contains nothing look identical (§15.1a).
    #[test]
    fn in_flight_has_one_growth_site_and_three_shrink_sites() {
        let src = include_str!("commands.rs");

        // Non-vacuity: the file was found and is the one this test is about.
        // Without it every count below is trivially satisfied by an empty string,
        // which is #457's `nm | grep -c` shape -- the reader exits printing nothing
        // and every pattern counts 0.
        assert!(
            src.contains("pub fn flush_and_prune"),
            "commands.rs does not contain flush_and_prune: this test is reading the wrong text, \
         and every count below would be vacuous"
        );

        let sites = |needle: &str| -> Vec<(usize, &str)> {
            src.lines()
                .enumerate()
                .filter(|(_, l)| l.contains(needle))
                .map(|(i, l)| (i + 1, l.trim()))
                .collect()
        };

        // Growth: `push` onto `in_flight`. The fuzzer's bound counts commits, so it
        // is only meaningful if a commit is the only thing that can grow the queue.
        let pushes = sites("in_flight.push");
        assert_eq!(
        pushes.len(),
        1,
        "expected exactly one site pushing onto in_flight (commit_swap_locked's), found {:#?}. \
         A second growth site invalidates the fuzzer's bound.",
        pushes
    );

        // Shrink, counted per mechanism rather than per line, so a repair that
        // removes the prune -- #513's own mutation, which read 63 in flight at
        // n=64 -- fails here as well as in a run.
        let takes = sites("mem::take(&mut state.in_flight)");
        assert_eq!(
            takes.len(),
            1,
            "expected exactly one mem::take of in_flight (flush_and_wait's), found {takes:#?}"
        );

        // `retain` appears in prose too, so count executable occurrences only: a
        // line that is a comment cannot shrink a queue. The one retain is
        // `prune_in_flight`'s, which drops completed buffers and reports a
        // failed one; the three paths that shrink by retaining call it.
        let retains: Vec<(usize, &str)> = sites(".retain(")
            .into_iter()
            .filter(|(_, l)| !l.starts_with("//") && !l.starts_with("///"))
            .collect();
        assert_eq!(
            retains.len(),
            1,
            "expected exactly one retain on in_flight (prune_in_flight's), found {retains:#?}"
        );
        let prunes = sites("self.prune_in_flight(&mut state,");
        assert_eq!(
            prunes.len(),
            3,
            "expected exactly three retain-based shrink sites on in_flight \
         (flush_and_wait_current's, flush_and_prune's and wait_marker's), found {prunes:#?}. \
         Losing one is the unbounded-queue defect #493 repaired."
        );

        // The asymmetry §17's prediction rests on, stated as the number a reader
        // can check: one unconditional push against four conditional removals, of
        // which exactly one does not block.
        assert_eq!(
            pushes.len() + takes.len() + prunes.len(),
            5,
            "the growth/shrink census moved; the fuzzer's model is written against it"
        );
    }
}
