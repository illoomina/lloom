//! A stateful model-based fuzzer over the buffer pool's public surface
//! (issue #494).
//!
//! # Why a fuzzer, when there are already 30 tests
//!
//! The 30 tests in `buffer_pool::tests` are 30 *chosen* paths through a state
//! machine with ~14 mutating operations, an epoch clock, a free budget, a
//! residual cap and an eviction observer. **Three of the last five findings in
//! this project were not arithmetic** -- `clear_arena` leaking a residency-set
//! entry (#489), a stale poison reporting the driver's own `bzero` (#480), and
//! `checksum == 0.0` passing a NaN (#479). Every one was a path nothing
//! exercised, reachable by an operation *sequence* nobody had written down.
//! That is the shape a generator reaches and a hand-written test does not.
//!
//! # This runs without a GPU lease, and that is a measured claim
//!
//! `BufferPool`'s bookkeeping never touches the GPU: `adopt` takes a `Buffer`
//! and a size and updates counters, the free list is a `BTreeMap`, and the
//! epoch clock is two atomics driven here by hand. What it *cannot* avoid is
//! allocating a real `MTLBuffer`, because `Buffer` wraps
//! `Retained<ProtocolObject<dyn MTLBuffer>>` -- there is no seam for a fake
//! handle and this issue does not take one.
//!
//! Measured for #494 (`examples/pool_lease_probe.rs`, three runs **while
//! another agent held an exclusive `measure` lease**): 200 000 pool operations
//! in 0.33-0.53 s, ~380-600 k ops/s, with reuse, eviction, draining and trim
//! all engaged. So the allocator is not the GPU's execution units, and a
//! generator over this alphabet is CI-cheap rather than lease-bound.
//!
//! # The model is the instrument, so it is written from the contract
//!
//! A model that mirrors the implementation cannot find a bug in it. This one is
//! derived from the *documented* contract -- the doc comments on `acquire`,
//! `release`, `drain_completed`, `evict_over_budget` and `trim` -- and tracks
//! what the counters and occupancy **should** be. Any divergence is the
//! finding; the harness reports the operation sequence that produced it, and
//! proptest shrinks it, which is the property that makes a failure a bug report
//! rather than a 400-operation transcript (#494's own argument for proptest
//! over a coverage-guided fuzzer).

#![cfg(any(test, feature = "fuzzing"))]

use super::buffer_pool::{BufferHandle, BufferPool, GpuClock};
use super::device::Device;
use crate::retire::Epoch;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

/// One symbol of the pool's alphabet.
///
/// This is the public surface #494 names, minus the two operations that are
/// *destructors* rather than calls: the last `BufferHandle`'s drop is reached
/// by [`Op::DropLive`] and `PoolInner::drop` by the end of a run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// `acquire(size)`, falling back to a fresh allocation + `adopt` on a miss
    /// -- the shape `MetalDevice` uses and the shape the tests' `alloc` uses.
    Alloc { size: usize },
    /// Drop the live handle at `index % live.len()`, running the last
    /// `BufferHandle`'s drop (the node goes back with its buffer).
    DropLive { index: usize },
    /// `commit_epoch()` -- the host committing a command buffer.
    CommitEpoch,
    /// `mark_completed(epoch)` -- Metal's completion handler, on a driver
    /// thread. `back` selects an *older* epoch, since handlers may be delivered
    /// out of order (the clock takes a max for exactly this reason).
    MarkCompleted { back: usize },
    /// `drain_completed()`.
    Drain,
    /// `trim()`, dropping what it returns.
    Trim,
    /// `set_free_budget(bytes)`, which evicts immediately if already exceeded.
    SetFreeBudget { bytes: usize },
    /// `set_residual_cap(limit, planned)`.
    SetResidualCap { limit: usize, planned: usize },
    /// `clear_residual_cap()`.
    ClearResidualCap,
    /// `check_residual(size)`, whose verdict the model predicts.
    CheckResidual { size: usize },
    /// `reset_counters()`.
    ResetCounters,
}

/// What the pool's counters and occupancy **should** be, tracked independently.
///
/// Deliberately a re-derivation from the contract rather than a copy of
/// `PoolState`: it keeps sizes, not buffers, because the model has no business
/// knowing about `MTLBuffer` -- and because a model that shares the
/// implementation's data structures shares its bugs.
#[derive(Debug, Default)]
struct Model {
    /// size -> count of free buffers of that size.
    free: BTreeMap<usize, usize>,
    /// Sizes in the order they entered the free list, including stale entries
    /// that `acquire` left behind -- the implementation does the same and
    /// `evict_over_budget` skips them.
    free_order: VecDeque<usize>,
    /// Sizes of the buffers callers currently hold.
    live: Vec<usize>,
    /// `(size, epoch)` for released buffers awaiting GPU completion.
    pending: VecDeque<(usize, u64)>,
    free_bytes: usize,
    free_budget: usize,
    residual: Option<(usize, usize)>,
    // Counters, in the order `PoolCounters` declares them.
    lookups: u64,
    hits: u64,
    releases: u64,
    allocations: u64,
    trimmed: u64,
    trims: u64,
    deferred: u64,
    drained: u64,
    evicted: u64,
    // The clock.
    submitted: u64,
    completed: u64,
}

impl Model {
    fn new() -> Self {
        Self {
            free_budget: 256 * 1024 * 1024,
            ..Default::default()
        }
    }

    fn free_buffers(&self) -> usize {
        self.free.values().sum()
    }

    fn free_buckets(&self) -> usize {
        self.free.values().filter(|c| **c > 0).count()
    }

    /// The lookup half of `acquire`'s contract: exact bucket first, then the
    /// smallest bucket strictly larger. Returns the bucket size served, if any.
    /// `acquire` drains first; the caller runs [`Self::drain`] before this.
    fn acquire(&mut self, size: usize) -> Option<usize> {
        self.lookups += 1;
        let bucket = if self.free.get(&size).is_some_and(|c| *c > 0) {
            Some(size)
        } else {
            self.free
                .range((std::ops::Bound::Excluded(size), std::ops::Bound::Unbounded))
                .find(|(_, c)| **c > 0)
                .map(|(k, _)| *k)
        }?;
        let count = self.free.get_mut(&bucket).expect("bucket just found");
        *count -= 1;
        if *count == 0 {
            self.free.remove(&bucket);
        }
        self.free_bytes = self.free_bytes.saturating_sub(bucket);
        self.hits += 1;
        self.live.push(bucket);
        Some(bucket)
    }

    fn adopt(&mut self, size: usize) {
        self.allocations += 1;
        self.live.push(size);
    }

    /// `release`: always parks in `pending`, stamped with the *open* epoch.
    fn release(&mut self, size: usize) {
        self.releases += 1;
        self.deferred += 1;
        self.pending.push_back((size, self.submitted));
    }

    fn is_complete(&self, epoch: u64) -> bool {
        self.completed > epoch
    }

    /// `drain_completed`: a prefix walk that stops at the first outstanding
    /// epoch, followed by eviction if anything moved.
    fn drain(&mut self) {
        let mut moved = 0u64;
        while let Some((size, epoch)) = self.pending.front().copied() {
            if !self.is_complete(epoch) {
                break;
            }
            self.pending.pop_front();
            self.free_bytes += size;
            self.free_order.push_back(size);
            *self.free.entry(size).or_insert(0) += 1;
            moved += 1;
        }
        if moved > 0 {
            self.drained += moved;
            self.evict_over_budget();
        }
    }

    /// `evict_over_budget`: oldest-first by `free_order`, skipping stale
    /// entries, until the free list is inside its budget.
    fn evict_over_budget(&mut self) {
        while self.free_bytes > self.free_budget {
            let Some(size) = self.free_order.pop_front() else {
                break;
            };
            let Some(count) = self.free.get_mut(&size) else {
                continue;
            };
            if *count == 0 {
                continue;
            }
            *count -= 1;
            if *count == 0 {
                self.free.remove(&size);
            }
            self.free_bytes = self.free_bytes.saturating_sub(size);
            self.evicted += 1;
        }
    }

    /// `trim`: drains first, then destroys everything free. Leaves `pending`
    /// alone -- the GPU may still be using it.
    fn trim(&mut self) {
        self.drain();
        let freed: usize = self.free.values().sum();
        self.free.clear();
        self.free_order.clear();
        self.free_bytes = 0;
        self.trims += 1;
        self.trimmed += freed as u64;
    }

    fn pending_bytes(&self) -> usize {
        self.pending.iter().map(|(s, _)| *s).sum()
    }

    fn live_bytes(&self) -> usize {
        self.live.iter().sum()
    }

    /// `unplanned_bytes`: `live + free + pending − planned`, saturating.
    fn unplanned(&self) -> Option<usize> {
        let (_, planned) = self.residual?;
        Some((self.live_bytes() + self.free_bytes + self.pending_bytes()).saturating_sub(planned))
    }
}

/// Drives one generated sequence against the real pool and the model, and
/// compares after **every** operation.
///
/// Comparing per-operation rather than at the end is what makes a failure
/// short: the sequence is truncated at the first divergence before proptest
/// even begins shrinking.
pub fn run_sequence(dev: &Device, ops: &[Op]) -> Result<(), String> {
    let clock = Arc::new(GpuClock::new());
    let pool = BufferPool::with_clock(Arc::clone(&clock));
    let mut model = Model::new();
    let mut live: Vec<BufferHandle> = Vec::new();

    for (i, op) in ops.iter().enumerate() {
        match op {
            Op::Alloc { size } => {
                let size = (*size).max(1);
                // `acquire` drains what the clock says completed before it
                // looks: completion only advances the clock (§6.4e).
                model.drain();
                match pool.acquire(size) {
                    Some(b) => {
                        let served = model
                            .acquire(size)
                            .ok_or_else(|| format!("op {i}: pool hit {size} but model missed"))?;
                        if b.size() != served {
                            return Err(format!(
                                "op {i}: acquire({size}) served bucket {} but model says {served}",
                                b.size()
                            ));
                        }
                        live.push(b);
                    }
                    None => {
                        if let Some(served) = model.acquire(size) {
                            return Err(format!(
                                "op {i}: pool missed acquire({size}) but model says bucket \
                                 {served} was free"
                            ));
                        }
                        let raw = dev
                            .new_buffer(size, crate::kernels::RESOURCE_OPTIONS)
                            .map_err(|e| format!("op {i}: allocation failed: {e:?}"))?;
                        live.push(pool.adopt(raw, size));
                        model.adopt(size);
                    }
                }
            }
            Op::DropLive { index } => {
                if live.is_empty() {
                    continue;
                }
                let idx = index % live.len();
                let b = live.remove(idx);
                let size = b.size();
                // Only the *last* handle triggers the return. Every handle here
                // is unique, so this always does.
                drop(b);
                let m = model.live.iter().position(|s| *s == size).ok_or_else(|| {
                    format!("op {i}: dropped a {size} B handle the model does not hold")
                })?;
                model.live.remove(m);
                model.release(size);
                // `release` drains too (§6.4e).
                model.drain();
            }
            Op::CommitEpoch => {
                let closed = clock.commit_epoch().0;
                if closed != model.submitted {
                    return Err(format!(
                        "op {i}: commit_epoch returned {closed}, model expected {}",
                        model.submitted
                    ));
                }
                model.submitted += 1;
            }
            Op::MarkCompleted { back } => {
                // An older epoch, as an out-of-order completion handler would
                // report. `submitted` is the open one, so the newest *closed*
                // epoch is `submitted - 1`.
                if model.submitted == 0 {
                    continue;
                }
                let newest = model.submitted - 1;
                let epoch = newest - (*back as u64 % (newest + 1));
                clock.mark_completed(Epoch(epoch));
                model.completed = model.completed.max(epoch + 1);
            }
            Op::Drain => {
                pool.drain_completed();
                model.drain();
            }
            Op::Trim => {
                let freed = pool.trim();
                // `trim` DRAINS FIRST ("a trim after a synchronize reclaims what
                // that synchronize made safe"), so the buffers it returns
                // include anything the drain just moved. A first version of this
                // harness snapshotted the model's free count *before*
                // `model.trim()` and compared against that -- which ignored the
                // internal drain and reported a 5-operation "divergence" that
                // was the harness's, not the pool's. The count that respects the
                // contract is the one the model's own trim computes.
                let before_trimmed = model.trimmed;
                model.trim();
                let want = model.trimmed - before_trimmed;
                if freed.len() as u64 != want {
                    return Err(format!(
                        "op {i}: trim returned {} buffers, model expected {want}",
                        freed.len()
                    ));
                }
                drop(freed);
            }
            Op::SetFreeBudget { bytes } => {
                pool.set_free_budget(*bytes);
                model.free_budget = *bytes;
                model.evict_over_budget();
            }
            Op::SetResidualCap { limit, planned } => {
                pool.set_residual_cap(*limit, *planned);
                model.residual = Some((*limit, *planned));
            }
            Op::ClearResidualCap => {
                pool.clear_residual_cap();
                model.residual = None;
            }
            Op::CheckResidual { size } => {
                let got = pool.check_residual(*size);
                let expected = match model.residual {
                    None => None,
                    Some((limit, _)) => {
                        let unplanned = model.unplanned().expect("cap installed");
                        (unplanned.saturating_add(*size) > limit).then_some(unplanned)
                    }
                };
                match (&got, expected) {
                    (Ok(()), None) => {}
                    (Err(e), Some(unplanned)) => {
                        if e.unplanned != unplanned {
                            return Err(format!(
                                "op {i}: check_residual({size}) reported unplanned={} but model \
                                 says {unplanned}",
                                e.unplanned
                            ));
                        }
                    }
                    (Ok(()), Some(unplanned)) => {
                        return Err(format!(
                            "op {i}: check_residual({size}) admitted, but model says it exceeds \
                             the cap (unplanned={unplanned})"
                        ))
                    }
                    (Err(e), None) => {
                        return Err(format!(
                            "op {i}: check_residual({size}) refused ({e:?}) with no cap installed \
                             or inside the cap"
                        ))
                    }
                }
            }
            Op::ResetCounters => {
                pool.reset_counters();
                model.lookups = 0;
                model.hits = 0;
                model.releases = 0;
                model.allocations = 0;
                model.trimmed = 0;
                model.trims = 0;
                model.deferred = 0;
                model.drained = 0;
                model.evicted = 0;
                // Nothing else to teach the model. `reset_counters` zeroes the
                // accumulators and touches no live figure, because #507 moved
                // the one gauge out of `PoolCounters` entirely.
                //
                // This arm used to reproduce a DEFECT on purpose -- the model
                // set `pending_gauge_zeroed` so that one known finding did not
                // mask every later operation in every generated sequence -- and
                // the accommodation was written with its own expiry, per
                // §2.6a. The repair landed, so the accommodation is gone rather
                // than left behind to become a permanent lie.
            }
        }
        compare(&pool, &clock, &model, i, op)?;
    }
    Ok(())
}

/// Every counter and every occupancy field, after every operation.
///
/// Checking all of them rather than the ones an operation "should" touch is the
/// point: a defect that moves a counter nothing looked at is exactly the class
/// #494 exists to find.
fn compare(
    pool: &BufferPool,
    clock: &GpuClock,
    model: &Model,
    i: usize,
    op: &Op,
) -> Result<(), String> {
    let c = pool.counters();
    let occ = pool.occupancy();
    let mut bad = Vec::new();

    let mut check = |name: &str, got: u64, want: u64| {
        if got != want {
            bad.push(format!("{name}: pool={got} model={want}"));
        }
    };
    check("lookups", c.lookups, model.lookups);
    check("hits", c.hits, model.hits);
    check("releases", c.releases, model.releases);
    check("allocations", c.allocations, model.allocations);
    check("trimmed", c.trimmed, model.trimmed);
    check("trims", c.trims, model.trims);
    check("deferred", c.deferred, model.deferred);
    check("drained", c.drained, model.drained);
    check("evicted", c.evicted, model.evicted);
    // How many buffers are parked is an OCCUPANCY figure and is checked as one
    // below (`pending_buffers`). Until #507 it was also mirrored into
    // `PoolCounters`, and the mirror is what `reset_counters` could zero.
    check(
        "live_buffers",
        occ.live_buffers as u64,
        model.live.len() as u64,
    );
    check(
        "live_bytes",
        occ.live_bytes as u64,
        model.live_bytes() as u64,
    );
    check(
        "free_buffers",
        occ.free_buffers as u64,
        model.free_buffers() as u64,
    );
    check("free_bytes", occ.free_bytes as u64, model.free_bytes as u64);
    check(
        "free_buckets",
        occ.free_buckets as u64,
        model.free_buckets() as u64,
    );
    check(
        "pending_buffers",
        occ.pending_buffers as u64,
        model.pending.len() as u64,
    );
    check(
        "pending_bytes",
        occ.pending_bytes as u64,
        model.pending_bytes() as u64,
    );
    // Read off the real clock, not off the model. A first version of this
    // compared `model.completed` against itself and routed `submitted` through
    // a helper that returned the model's own field -- two gates that could not
    // fail, which is #479's class exactly, in the harness written to find it.
    check("clock.submitted", clock.current_epoch().0, model.submitted);
    check(
        "clock.completed",
        clock.completed_count().0,
        model.completed,
    );

    if let Some(want) = model.unplanned() {
        let got = pool.unplanned_bytes();
        if got != Some(want) {
            bad.push(format!("unplanned_bytes: pool={got:?} model=Some({want})"));
        }
    } else if pool.unplanned_bytes().is_some() {
        bad.push("unplanned_bytes: pool reports a figure with no cap installed".to_string());
    }

    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "after op {i} ({op:?}):\n    {}",
            bad.join("\n    ")
        ))
    }
}

// The generators and the properties: test-only, because proptest is a
// dev-dependency and the model above is also driven by `lloom-fuzz` (#706).
#[cfg(test)]
mod proptests {
    use super::*;

    // ---------------------------------------------------------------------------
    // The generator, and the properties.
    // ---------------------------------------------------------------------------

    use proptest::prelude::*;

    /// Sizes drawn from a small set rather than uniformly over `usize`.
    ///
    /// Uniform sizes would essentially never collide, so the free list would hold
    /// one buffer per bucket and `acquire`'s exact-match path -- the decode steady
    /// state, and the path the eviction ordering depends on -- would almost never
    /// fire. A handful of repeated sizes plus a growing one is the shape a real
    /// workload has (§6.3b: a KV cache asks for a slightly larger size each step).
    fn size_strategy() -> impl Strategy<Value = usize> {
        prop_oneof![
            // Repeated sizes: exercise exact-bucket hits and multi-buffer buckets.
            8 => prop::sample::select(vec![512usize, 1024, 2048, 4096, 8192]),
            // Nearby sizes: exercise the range probe and the smallest-fit rule.
            3 => 1usize..16384,
            // Large: exercise the free budget and eviction.
            1 => 64usize * 1024..1024 * 1024,
        ]
    }

    fn op_strategy() -> impl Strategy<Value = Op> {
        prop_oneof![
            6 => size_strategy().prop_map(|size| Op::Alloc { size }),
            5 => any::<u16>().prop_map(|index| Op::DropLive { index: index as usize }),
            3 => Just(Op::CommitEpoch),
            3 => any::<u8>().prop_map(|back| Op::MarkCompleted { back: back as usize }),
            3 => Just(Op::Drain),
            1 => Just(Op::Trim),
            1 => prop_oneof![
                Just(0usize),
                Just(4096usize),
                Just(64 * 1024usize),
                Just(1024 * 1024usize),
            ]
            .prop_map(|bytes| Op::SetFreeBudget { bytes }),
            1 => (0usize..4 * 1024 * 1024, 0usize..1024 * 1024)
                .prop_map(|(limit, planned)| Op::SetResidualCap { limit, planned }),
            1 => Just(Op::ClearResidualCap),
            1 => size_strategy().prop_map(|size| Op::CheckResidual { size }),
            1 => Just(Op::ResetCounters),
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
        // 256 cases of up to 120 operations. Sized to land inside #494's "a minute
        // per target" on this machine rather than to be as large as possible: the
        // argument for proptest over a coverage-guided fuzzer is that SHRINKING
        // matters more than throughput here, and a case count that makes the suite
        // slow enough to skip is worth less than one that runs every time.
        #![proptest_config(ProptestConfig {
            cases: 256,
            max_shrink_iters: 4096,
            ..ProptestConfig::default()
        })]

        /// The pool's counters and occupancy agree with a model of its contract,
        /// after every operation of an arbitrary sequence.
        #[test]
        fn pool_matches_its_model(ops in prop::collection::vec(op_strategy(), 1..120)) {
            let dev = device();
            if let Err(e) = run_sequence(&dev, &ops) {
                return Err(TestCaseError::fail(e));
            }
        }
    }

    /// The generator reaches the states the property is about.
    ///
    /// Without this the suite above is #454's `to_cpu_fast=0` sweep and #69's
    /// vacuous determinism run: a green result that measured nothing. Asserted on
    /// the *counters*, so it fails if a future change to the strategy stops
    /// producing reuse, eviction, draining or trimming -- rather than on a coverage
    /// figure nobody reads.
    #[test]
    fn the_generator_actually_reaches_the_interesting_states() {
        use proptest::strategy::{Strategy, ValueTree};
        use proptest::test_runner::{Config, TestRunner};

        let dev = device();
        let mut runner = TestRunner::new(Config {
            cases: 256,
            ..Config::default()
        });
        let strategy = prop::collection::vec(op_strategy(), 40..120);

        let (mut hits, mut evicted, mut drained, mut trims, mut refusals) =
            (0u64, 0u64, 0u64, 0u64, 0u64);
        for _ in 0..256 {
            let ops = strategy
                .new_tree(&mut runner)
                .expect("strategy produces a value")
                .current();
            // Re-run the sequence against a fresh pool, counting what it reached.
            let clock = Arc::new(GpuClock::new());
            let pool = BufferPool::with_clock(Arc::clone(&clock));
            let mut live: Vec<BufferHandle> = Vec::new();
            let mut submitted = 0u64;
            for op in &ops {
                match op {
                    Op::Alloc { size } => {
                        let size = (*size).max(1);
                        let b = pool.acquire(size).unwrap_or_else(|| {
                            let raw = dev
                                .new_buffer(size, crate::kernels::RESOURCE_OPTIONS)
                                .expect("allocation");
                            pool.adopt(raw, size)
                        });
                        live.push(b);
                    }
                    Op::DropLive { index } => {
                        if !live.is_empty() {
                            let i = index % live.len();
                            drop(live.remove(i));
                        }
                    }
                    Op::CommitEpoch => {
                        clock.commit_epoch();
                        submitted += 1;
                    }
                    Op::MarkCompleted { back } => {
                        if submitted > 0 {
                            let newest = submitted - 1;
                            clock.mark_completed(Epoch(newest - (*back as u64 % (newest + 1))));
                        }
                    }
                    Op::Drain => pool.drain_completed(),
                    Op::Trim => drop(pool.trim()),
                    Op::SetFreeBudget { bytes } => pool.set_free_budget(*bytes),
                    Op::SetResidualCap { limit, planned } => {
                        pool.set_residual_cap(*limit, *planned)
                    }
                    Op::ClearResidualCap => pool.clear_residual_cap(),
                    Op::CheckResidual { size } => {
                        if pool.check_residual(*size).is_err() {
                            refusals += 1;
                        }
                    }
                    Op::ResetCounters => {}
                }
            }
            let c = pool.counters();
            hits += c.hits;
            evicted += c.evicted;
            drained += c.drained;
            trims += c.trims;
        }

        println!(
            "FUZZ494 engagement hits={hits} evicted={evicted} drained={drained} \
         trims={trims} residual_refusals={refusals}"
        );
        assert!(hits > 0, "the generator never produced a pool hit");
        assert!(drained > 0, "the generator never drained a completed epoch");
        assert!(
            evicted > 0,
            "the generator never evicted: the budget never bound"
        );
        assert!(trims > 0, "the generator never trimmed");
        assert!(
            refusals > 0,
            "the residual cap never refused: check_residual was exercised only on its Ok arm, \
         which is CONTRIBUTING.md §1.1's both-bounds rule (a check that refuses nothing is \
         not a check)"
        );
    }

    /// **`PoolCounters` holds accumulators only**, checked in source (#507).
    ///
    /// This replaces `the_known_pending_defect_is_still_present`, which existed to
    /// stop the model's accommodation for #494 outliving the bug. The repair landed,
    /// the accommodation is gone, and what is worth keeping is the *class* being
    /// closed rather than the instance: `reset_counters` zeroes the whole struct, so
    /// **any** gauge added to it is silently zeroed while the state it describes is
    /// unchanged. That is what made #494's finding possible, and it is a property of
    /// the struct rather than of the field that happened to carry it.
    ///
    /// **It has to read the source, because the defect is a WRITE SHAPE.** A runtime
    /// test can only catch a gauge that some sequence disturbs -- which is the test
    /// in `buffer_pool::tests` and is the right check for the field that existed.
    /// The generalisation is *"nothing here is assigned"*, and an assignment among
    /// the `+=` is visible in the text and invisible to any pool state. Same
    /// reasoning, and the same mechanism, as `Buffer`'s
    /// `load_bearing_contents_sites_keep_their_black_box`.
    ///
    /// Mutation-tested: adding `state.counters.<field> = ...` to `buffer_pool.rs`
    /// makes this fail.
    #[test]
    fn pool_counters_holds_accumulators_only() {
        let src = include_str!("buffer_pool.rs");

        // Only the production half. This module's own doc comments quote the
        // defective `counters.pending =` line, and a check that reads its own
        // description reports on the wrong text -- §15.1a's vacuous class, in the
        // guard written to prevent a regression.
        let production = src
            .split("#[cfg(test)]")
            .next()
            .expect("split always yields one element");

        let writes: Vec<&str> = production
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with("//") && !l.starts_with("///"))
            .filter(|l| l.contains("counters."))
            .filter(|l| l.contains('=') && !l.contains("+=") && !l.contains("=="))
            .collect();

        assert!(
            production.contains("counters.releases += 1"),
            "no counter increment found in buffer_pool.rs -- was the struct renamed? \
         A guard that finds nothing to check passes for the wrong reason."
        );

        assert!(
            writes.is_empty(),
            "a field of `PoolCounters` is ASSIGNED rather than incremented:\n    {}\n\
         That makes it a GAUGE -- a figure describing the pool's state now -- and \
         `reset_counters` zeroes the whole struct, so it will be zeroed while the \
         state it describes is unchanged. That is lloom #494's defect, repaired in \
         #507 by moving `pending` into `PoolOccupancySnapshot`. Live figures belong \
         in `occupancy()`, derived from the state they describe; `PoolCounters` is \
         events since the last reset. DESIGN.md §6.7 L2.",
            writes.join("\n    ")
        );
    }

    /// The hole the check above cannot see: a field borrowed mutably OUT of the
    /// struct, and written through the borrow.
    ///
    /// `pool_counters_holds_accumulators_only` reads for `counters.<field> =` with
    /// no `+=`. **A `let x = &mut state.counters.<field>;` followed by a write
    /// through `x` defeats it**, because the write names `x` and never names
    /// `counters` -- so the gauge class it exists to close stays reachable through
    /// one extra line.
    ///
    /// Found while adding §9.2u's bands (#47): the natural spelling for a per-band
    /// update *is* that borrow, and it made the first guard fail on code that was a
    /// pure accumulator. **The guard was right that the shape is dangerous and
    /// wrong about that instance** -- the two-sided form of `CONTRIBUTING.md`
    /// §3.1c's *"a pattern written to find a thing is a thing the pattern finds"*.
    /// The repair is not to loosen the needle but to name the escape route it
    /// cannot follow.
    ///
    /// The rule is narrow on purpose: a mutable borrow of a `counters` field must
    /// not leave `impl PoolCounters`. Inside it the borrow is local to a method
    /// whose whole body a reader is already checking; outside it the write happens
    /// where the first guard has no line to read.
    #[test]
    fn no_counter_field_is_mutably_borrowed_out_of_the_struct() {
        let src = include_str!("buffer_pool.rs");

        let production = src
            .split("#[cfg(test)]")
            .next()
            .expect("split always yields one element");

        let borrows: Vec<&str> = production
            .lines()
            .map(str::trim)
            .filter(|l| !l.starts_with("//") && !l.starts_with("///"))
            .filter(|l| l.contains("&mut") && l.contains("counters."))
            .collect();

        // Non-vacuity: the in-struct borrow must still be findable, or an empty
        // result means the spelling moved rather than that the property holds
        // (§3.1c -- a zero is not evidence without a nonzero from the same run).
        assert!(
            production.contains("&mut self.bands["),
            "no in-struct mutable borrow found -- `record_band` was renamed or \
         removed, so this guard is matching against code that no longer exists \
         and its empty result means nothing."
        );

        assert!(
            borrows.is_empty(),
            "a `PoolCounters` field is mutably borrowed outside `impl PoolCounters`:\n    {}\n\
         A write through that borrow never spells `counters.<field> =`, so \
         `pool_counters_holds_accumulators_only` cannot see it and the gauge class \
         it closes is reachable again. Put the update in a method on \
         `PoolCounters` -- `record_band` is the pattern -- so the borrow stays \
         where a reader checking the accumulator property is already looking.",
            borrows.join("\n    ")
        );
    }

    // ---------------------------------------------------------------------------
    // Target 4: `GpuClock` epoch ordering, in isolation.
    // ---------------------------------------------------------------------------

    /// One symbol of the clock's alphabet.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum ClockOp {
        Commit,
        /// `mark_completed(epoch)`, where `epoch` is chosen freely -- including
        /// epochs that were never committed and epochs long past. Metal delivers
        /// completion handlers on driver threads and makes no ordering promise
        /// between them, so an arbitrary arrival order is the real contract.
        MarkCompleted(u64),
    }

    proptest! {
        #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

        /// The clock's two invariants, over arbitrary interleavings:
        ///
        /// 1. **`completed_count` never decreases.** It is a `fetch_max` precisely
        ///    because a late handler for an older command buffer may arrive after a
        ///    newer one. A store would move it backwards and free every buffer
        ///    parked in between -- issue #19's corruption, by a different route.
        /// 2. **Zero means "nothing has completed", not "epoch 0 has".** The
        ///    `epoch + 1` encoding exists to keep those distinguishable, and an
        ///    off-by-one here is `release_is_deferred_even_when_the_gpu_is_idle`'s
        ///    unreachable branch becoming reachable.
        #[test]
        fn clock_is_monotone_and_zero_means_nothing_completed(
            ops in prop::collection::vec(
                prop_oneof![
                    2 => Just(ClockOp::Commit),
                    3 => (0u64..8).prop_map(ClockOp::MarkCompleted),
                ],
                1..64,
            )
        ) {
            let clock = GpuClock::new();
            prop_assert_eq!(clock.current_epoch(), Epoch(0));
            prop_assert_eq!(clock.completed_count(), crate::retire::Completed(0));

            let mut last_completed = 0u64;
            let mut submitted = 0u64;
            let mut highest_marked: Option<u64> = None;

            for op in &ops {
                match op {
                    ClockOp::Commit => {
                        let closed = clock.commit_epoch().0;
                        prop_assert_eq!(
                            closed, submitted,
                            "commit_epoch returned the wrong closed epoch"
                        );
                        submitted += 1;
                        prop_assert_eq!(clock.current_epoch(), Epoch(submitted));
                    }
                    ClockOp::MarkCompleted(e) => {
                        clock.mark_completed(Epoch(*e));
                        highest_marked = Some(highest_marked.map_or(*e, |h: u64| h.max(*e)));
                    }
                }
                let now = clock.completed_count().0;
                prop_assert!(
                    now >= last_completed,
                    "completed_count went BACKWARDS ({last_completed} -> {now}): a late \
                     handler moved the clock back, which frees every buffer parked in \
                     between -- issue #19 by another route"
                );
                last_completed = now;
                // The encoding: `completed_count` is `highest marked + 1`, so zero
                // is reachable only while nothing has been marked at all.
                match highest_marked {
                    None => prop_assert_eq!(
                        now, 0,
                        "completed_count is nonzero before anything was marked"
                    ),
                    Some(h) => prop_assert_eq!(
                        now,
                        h + 1,
                        "completed_count is not `highest marked + 1`; zero and \
                         'epoch 0 completed' have become indistinguishable"
                    ),
                }
            }
        }
    }
}
