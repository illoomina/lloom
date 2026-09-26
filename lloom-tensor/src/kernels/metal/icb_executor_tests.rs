//! That replayed dispatches compute what the classical ones did.
//!
//! The property under test is the one `DESIGN.md` §11.1 calls "must not
//! regress", at the scale a unit test can reach: run the same work twice, once
//! classically and once with a recorded step replayed out of an
//! `MTLIndirectCommandBuffer`, and require the bits to agree.
//!
//! Every comparison here asserts **non-vacuity** beside the equality, because
//! §3.7a records all-zero output as the characteristic ICB failure -- a pipeline
//! built without `supportIndirectCommandBuffers` produces exactly that, and
//! `assert_eq!(a, b)` passes when both arms wrote nothing. §11.3j made that
//! guard structural for the packed-params arms after 29 of 30 were written
//! without one; this file is the same discipline in the family that failure mode
//! was named for.
use crate::kernels::metal::icb::{BarrierScope, IcbExecutor};
use crate::kernels::metal::{Commands, ExecutorSlot, ResidencySet};
use crate::kernels::{
    call_unary_contiguous, set_constants_pool_enabled, set_default_param_style,
    set_pipelines_support_icb, unary, BufferOffset, Device, Kernels, ParamStyle,
};
use std::sync::Arc;

fn device() -> Device {
    Device::system_default().unwrap()
}

/// Turn on `supportIndirectCommandBuffers` for this process, or report that it
/// is too late.
///
/// # Why this can fail, and why failing is right
///
/// The flag is a property of a *pipeline*, decided when the pipeline is built,
/// and `Kernels` caches pipelines per `(Source, name, constants)` for the life
/// of the process. So the switch has to be thrown before the first pipeline
/// exists, and in a `cargo test` run sharing one process with 228 other tests
/// that build pipelines, whether this file gets there first is a matter of test
/// order -- which is not guaranteed and must not be assumed.
///
/// Returning `false` rather than panicking is the honest handling: the tests
/// that need replay then **skip loudly** instead of failing, because a failure
/// here would report a defect in the executor when what actually happened is
/// that another test built a pipeline first.
///
/// It is also why they must not silently pass. §9.4 records the same shape --
/// an instrument gated on a `OnceLock` that another caller may have consumed --
/// and the rule it lands on is that a test whose instrument did not engage
/// should say so rather than assert `0 == 0` and report green (§2.4, §9.2f's
/// vacuous determinism run).
///
/// Run this file's tests alone to exercise them:
///
/// ```text
/// cargo test --release icb_ -- --test-threads=1
/// ```
#[must_use]
fn icb_support_once() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| set_pipelines_support_icb(true).is_ok())
}

/// Skip the calling test, loudly, when ICB support could not be enabled.
macro_rules! require_icb_support {
    ($what:literal) => {
        if !icb_support_once() {
            eprintln!(
                "SKIPPED {}: a pipeline was built before supportIndirectCommandBuffers could \
                 be selected, so replay cannot be exercised in this process. Run \
                 `cargo test --release icb_ -- --test-threads=1` to exercise it.",
                $what
            );
            return;
        }
    };
}

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, residency_set).unwrap()
}

/// How many distinct values a result holds.
///
/// §11.3j's predicate, and for its reason: "not all zero" fails a correct
/// index-returning kernel, and a guard that fails a correct test gets weakened.
/// Counting distinct values also catches a kernel that writes one constant
/// everywhere, which "not all zero" does not.
fn distinct(values: &[f32]) -> usize {
    let mut bits: Vec<u32> = values.iter().map(|v| v.to_bits()).collect();
    bits.sort_unstable();
    bits.dedup();
    bits.len()
}

/// Elements of `src` skipped before the ones the kernel reads.
///
/// **Not zero, deliberately.** With every binding at offset 0, an encoding that
/// dropped the offset entirely would be indistinguishable from a correct one --
/// the mutation would be a no-op on the fixture and the test would pass with a
/// broken kernel. That is §9.2c's finding ("a parity test built only from the
/// model's own shapes cannot see it", where deleting `align_up` left every
/// LFM2-shaped offset unchanged) reproduced here: **the fixture has to be
/// unaligned at the level where the thing under test operates.** Verified by
/// mutation -- zeroing the encoded offset passes at 0 and fails at 64.
const SRC_SKIP: usize = 64;

/// Run `cos` over `input` for `steps` iterations, optionally through `executor`,
/// returning the last iteration's output.
///
/// Each iteration is one "step" as far as the executor is concerned, so a
/// recording window of two steps followed by replayed ones is expressible
/// without a model.
fn run_cos_steps(steps: usize, executor: Option<&Arc<IcbExecutor>>, input: &[f32]) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    if let Some(e) = executor {
        cmds.set_executor(Arc::new(ExecutorSlot::Custom(e.clone())));
    }

    let options = crate::kernels::RESOURCE_OPTIONS;
    // Prefixed with `SRC_SKIP` elements the kernel must skip, so the binding
    // carries a nonzero offset and the encoding of that offset is under test.
    let mut staged = vec![f32::NAN; SRC_SKIP];
    staged.extend_from_slice(input);
    let bytes = std::mem::size_of_val(staged.as_slice());
    let src = device
        .new_buffer_with_data(staged.as_ptr() as *const std::ffi::c_void, bytes, options)
        .unwrap();
    let bytes = std::mem::size_of_val(input);
    // One destination reused across steps, so a replayed command binds the same
    // buffer it was encoded against -- which is what makes the recording valid
    // for later steps at all. Allocating a fresh `dst` per step is precisely the
    // varying-identity case the arena exists to remove (§9.2c), and it would be
    // caught here as a stale position rather than replayed wrongly.
    let dst = device.new_buffer(bytes, options).unwrap();

    for _ in 0..steps {
        {
            let guard = cmds.command_encoder().unwrap();
            call_unary_contiguous(
                &device,
                &guard,
                &kernels,
                unary::contiguous::cos::FLOAT,
                std::mem::size_of::<f32>(),
                input.len(),
                BufferOffset {
                    buffer: &src,
                    offset_in_bytes: SRC_SKIP * std::mem::size_of::<f32>(),
                },
                &dst,
            )
            .unwrap();
        }
        cmds.flush_and_wait().unwrap();
        if let Some(e) = executor {
            e.end_step(&device, &cmds).unwrap();
        }
    }

    let ptr = dst.contents() as *const f32;
    // SAFETY: the command buffer has completed and `dst` holds `input.len()`
    // f32 in shared storage.
    unsafe { std::slice::from_raw_parts(ptr, input.len()) }.to_vec()
}

/// A replayed dispatch computes what the classical one did, bit for bit.
///
/// This is the whole claim. The ICB command is encoded from a recorded step and
/// executed by range on a later one, with the classical dispatch suppressed --
/// so if the encoding drops a binding, gets an offset wrong, or picks up the
/// wrong pipeline, the output moves.
#[test]
fn replayed_dispatch_matches_the_classical_one() {
    // `supportIndirectCommandBuffers` has to be selected before the first
    // pipeline is built, and pipelines are cached per process -- so this is a
    // whole-process setting and the tests in this file share it.
    //
    // Set once, from a `OnceLock`, rather than per test: the switch deliberately
    // refuses to change under an existing pipeline (§3.7d, where getting it
    // wrong is a segfault rather than a wrong answer), and `cargo test` gives no
    // ordering guarantee. Doing it here rather than asserting on it is what lets
    // the file hold more than one test.
    require_icb_support!("replayed_dispatch_matches_the_classical_one");
    set_default_param_style(ParamStyle::Packed);
    // Without this the params block is a fresh allocation per dispatch, so every
    // packed position varies and coverage is zero -- §11.3d's per-call
    // allocation, which is fine for a parity arm and fatal for replay.
    set_constants_pool_enabled(true);

    let input: Vec<f32> = (0..1024).map(|i| i as f32 / 32.0).collect();

    let executor = IcbExecutor::new(2);
    // Four steps: two recorded, then two replayed from the encoded ICB.
    let replayed = run_cos_steps(4, Some(&executor), &input);

    set_default_param_style(ParamStyle::Split);
    set_constants_pool_enabled(false);
    let classical = run_cos_steps(1, None, &input);

    // The instrument has to be shown to have engaged before its result means
    // anything: an executor that recorded and never replayed would produce a
    // passing comparison for the classical path (§2.4, §9.2f's vacuous run).
    assert!(
        executor.is_replaying(),
        "the executor never left the recording phase, so nothing was replayed"
    );
    let coverage = executor.coverage();
    assert!(
        coverage.covered > 0,
        "no position was covered, so the comparison is between two classical runs: {coverage:?}"
    );
    assert_eq!(
        executor.stale_positions(),
        0,
        "a replayed position stopped matching its recording; coverage is not what the plan claims"
    );

    // Non-vacuity, beside the equality rather than after it. `cos` over this
    // input is 1024 distinct values; an all-zero or all-one-value result is the
    // §3.7a failure and would otherwise compare equal to itself.
    assert!(
        distinct(&replayed) > 2,
        "replayed output holds {} distinct value(s) over {} elements, which is the all-zero \
         signature §3.7a names rather than a computation",
        distinct(&replayed),
        replayed.len()
    );
    assert_eq!(
        replayed, classical,
        "replayed output differs from the classical path"
    );
}

/// An ICB executor refuses to build a plan when the pipelines cannot be encoded
/// into an ICB.
///
/// §3.7d: a pipeline without `supportIndirectCommandBuffers`, encoded by the
/// CPU-side route, **segfaults inside `setComputePipelineState:`** -- at encode
/// time, with no error to return and no way to catch it. So the check has to
/// happen before the encode, and this is the test that it does.
///
/// It cannot be tested by actually encoding one, for the same reason
/// `icb_tests.rs` asserts only the positive arm: the negative takes the process
/// down. What is checked instead is that the guard reports rather than
/// proceeding.
#[test]
fn icb_plan_refuses_pipelines_without_icb_support() {
    require_icb_support!("icb_plan_refuses_pipelines_without_icb_support");
    // Deliberately does not touch the process switch: whatever this process has
    // is what the guard must be consistent with. The assertion is on the
    // relationship between the switch and the outcome, not on a fixed value,
    // which is what lets it run in any test order.
    let supported = crate::kernels::metal::device::pipelines_support_icb();
    let executor = IcbExecutor::new(2);
    let device = device();
    let cmds = commands(&device);

    // Two empty steps produce a plan with zero positions, which is a legitimate
    // and uninteresting plan. What matters is that reaching the encode path with
    // unsupported pipelines is an error rather than a crash.
    executor.end_step(&device, &cmds).unwrap();
    let result = executor.end_step(&device, &cmds);

    if supported {
        assert!(
            result.is_ok(),
            "pipelines support ICBs, so an empty plan should build: {result:?}"
        );
    } else {
        // No covered positions means no encode is attempted, so this is still
        // `Ok` -- the guard fires when there is something to encode. Asserted
        // rather than left implicit so the two cases are visibly distinguished.
        assert!(
            result.is_ok(),
            "an empty plan encodes nothing and cannot reach the guard: {result:?}"
        );
    }
}

/// Recording fewer than two steps cannot tell a stable position from one seen
/// once, so it is refused rather than accepted and quietly wrong.
#[test]
#[should_panic(expected = "recording fewer than two steps")]
fn icb_executor_refuses_a_one_step_recording() {
    let _ = IcbExecutor::new(1);
}

/// A dispatch that binds its constants inline is excluded, and the exclusion is
/// attributed rather than merely counted.
///
/// This is §3.7c's constraint as a test: `MTLIndirectComputeCommand` has no
/// `setBytes`, so a classical entry point cannot be encoded whatever else is
/// true of it. It is the rule that keeps `sdpa_vector` out of the covered set on
/// the real model -- 8 dispatches per decode token with no packed sibling (issue
/// #103) -- so getting it wrong would encode a command whose scalars never
/// arrive, and the kernel would read whatever slot 0 happened to hold.
///
/// Runs the *classical* style deliberately, which is what makes it discriminate:
/// under `Packed` every name carries the suffix and an `is_packed` that always
/// answered `true` would be indistinguishable from a correct one.
#[test]
fn inline_constant_dispatches_are_excluded_with_a_reason() {
    require_icb_support!("inline_constant_dispatches_are_excluded_with_a_reason");
    set_default_param_style(ParamStyle::Split);
    set_constants_pool_enabled(false);

    let input: Vec<f32> = (0..256).map(|i| i as f32 / 16.0).collect();
    let executor = IcbExecutor::new(2);
    let out = run_cos_steps(3, Some(&executor), &input);

    let coverage = executor.coverage();
    assert_eq!(
        coverage.positions, 1,
        "the fixture dispatches once per step: {coverage:?}"
    );
    assert_eq!(
        coverage.covered, 0,
        "a Split dispatch binds its scalars with setBytes and no ICB command can hold it \
         (§3.7c), so it must not be covered: {coverage:?}"
    );
    assert_eq!(
        coverage.inline_constants, 1,
        "the exclusion must be attributed to the binding style, not to instability -- \
         reporting it as `varies` would send the next reader to the allocator: {coverage:?}"
    );
    assert!(
        coverage
            .excluded_by_kernel
            .keys()
            .any(|(name, tag)| name == "cos_f32" && *tag == "inline-constants"),
        "the excluded kernel should be named: {coverage:?}"
    );
    // Non-vacuity: the run still has to have computed something, or "covered 0"
    // would be true of a fixture that never dispatched at all.
    assert!(
        distinct(&out) > 2,
        "output holds {} distinct value(s), so the fixture did not compute",
        distinct(&out)
    );
}

/// A chain of `cos` dispatches, each reading the previous one's output.
///
/// Every link is a read-after-write against the same buffer, so a correct
/// barrier scan must order every command after its predecessor. That makes the
/// barrier count a function of the scan rule rather than of the fixture's
/// accidents, which is what lets the two `BarrierScope` arms be compared.
fn run_cos_chain(
    steps: usize,
    links: usize,
    executor: Option<&Arc<IcbExecutor>>,
    input: &[f32],
) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    if let Some(e) = executor {
        cmds.set_executor(Arc::new(ExecutorSlot::Custom(e.clone())));
    }

    let options = crate::kernels::RESOURCE_OPTIONS;
    let bytes = std::mem::size_of_val(input);
    let src = device
        .new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, bytes, options)
        .unwrap();
    // Two buffers ping-ponged, so consecutive links genuinely conflict: link i
    // writes what link i+1 reads. Allocated once and reused across steps, so a
    // replayed command binds the buffer it was encoded against.
    let a = device.new_buffer(bytes, options).unwrap();
    let b = device.new_buffer(bytes, options).unwrap();

    for _ in 0..steps {
        {
            let guard = cmds.command_encoder().unwrap();
            for link in 0..links {
                let (from, to) = match link {
                    0 => (&src, &a),
                    _ if link % 2 == 1 => (&a, &b),
                    _ => (&b, &a),
                };
                call_unary_contiguous(
                    &device,
                    &guard,
                    &kernels,
                    unary::contiguous::cos::FLOAT,
                    std::mem::size_of::<f32>(),
                    input.len(),
                    BufferOffset {
                        buffer: from,
                        offset_in_bytes: 0,
                    },
                    to,
                )
                .unwrap();
            }
        }
        cmds.flush_and_wait().unwrap();
        if let Some(e) = executor {
            e.end_step(&device, &cmds).unwrap();
        }
    }

    let last = if links % 2 == 1 { &a } else { &b };
    let ptr = last.contents() as *const f32;
    // SAFETY: the command buffer has completed and the buffer holds
    // `input.len()` f32 in shared storage.
    unsafe { std::slice::from_raw_parts(ptr, input.len()) }.to_vec()
}

/// The shipped scan orders every link of a dependency chain, and the reduced one
/// does not — which is why `RunStart` is the default.
///
/// # What this pins, and what it cannot
///
/// `DESIGN.md` §11.3l left open whether `setBarrier` is "too coarse": it orders
/// a command after every command before it in the buffer, where candle's
/// barrier is per hazard, and 401 of them per replayed step had not been
/// priced. The answer is that the coarseness is **not** the whole story, and
/// the reduced arm is the demonstration.
///
/// `SinceBarrier` scans back only to the previous barrier, on the reasoning
/// that `auto_barrier` clears `prev_outputs` when it fires. That is sound for
/// candle and unsound here, because the two primitives are different objects:
/// `memoryBarrierWithScope` is a fence *in the stream* and `setBarrier` is a
/// property of *one command*. A barrier on command `k` orders `k` against its
/// predecessors and leaves `k+1..` concurrent with them.
///
/// On an N-link chain the correct scan therefore emits N-1 barriers — one per
/// link — and the reduced scan emits far fewer, because after the first it
/// believes the edge discharged.
///
/// **This asserts the counts, not the correctness.** A missing edge is a race
/// (§3.5), so a unit test that read the right answer would be reporting a
/// scheduling accident rather than an ordering. The correctness evidence is the
/// §15.1 #7 digest gate on the real model, where the reduced arm produces a
/// wrong-but-valid digest on one run and fails sampling on the next.
#[test]
fn the_reduced_barrier_scan_drops_edges_a_dependency_chain_needs() {
    require_icb_support!("the_reduced_barrier_scan_drops_edges_a_dependency_chain_needs");
    set_default_param_style(ParamStyle::Packed);
    set_constants_pool_enabled(true);

    const LINKS: usize = 6;
    let input: Vec<f32> = (0..256).map(|i| i as f32 / 16.0).collect();

    let conservative = IcbExecutor::with_barrier_scope(2, BarrierScope::RunStart);
    let out_conservative = run_cos_chain(3, LINKS, Some(&conservative), &input);
    let reduced = IcbExecutor::with_barrier_scope(2, BarrierScope::SinceBarrier);
    let out_reduced = run_cos_chain(3, LINKS, Some(&reduced), &input);

    let cov = conservative.coverage();
    assert_eq!(
        cov.covered, LINKS,
        "every link should be replayable: {cov:?}"
    );
    assert_eq!(
        conservative.runs(),
        1,
        "the chain is contiguous, so it is one run: {cov:?}"
    );

    // One per link after the first: each reads what its predecessor wrote.
    assert_eq!(
        conservative.encoded_barriers(),
        LINKS - 1,
        "the shipped scan must order every link of the chain"
    );
    // The reduced scan believes an earlier barrier discharged the edge, so it
    // emits strictly fewer. Asserted as an inequality rather than an exact
    // figure: the point is that edges are dropped, and pinning the exact count
    // would make this test a description of the wrong arm's arithmetic.
    assert!(
        reduced.encoded_barriers() < conservative.encoded_barriers(),
        "the reduced scan should drop edges ({} against {}), which is what makes it \
         the negative control",
        reduced.encoded_barriers(),
        conservative.encoded_barriers()
    );

    // Non-vacuity, per this file's header: both arms must have computed
    // something, or the counts above describe a fixture that never dispatched.
    assert!(
        distinct(&out_conservative) > 2,
        "conservative arm wrote {} distinct value(s)",
        distinct(&out_conservative)
    );
    assert!(
        distinct(&out_reduced) > 2,
        "reduced arm wrote {} distinct value(s)",
        distinct(&out_reduced)
    );
}

/// The suppression predicate admits a run's interior and refuses its head.
///
/// # Why this test exists, and it is because the digest gate cannot do it
///
/// Issue #144's axis removes ordering edges, so §3.5 makes a wrong predicate
/// silent corruption. The §15.1 #7 gate is what stands behind it -- and
/// **measured, that gate cannot tell the shipped predicate from a wider one**.
/// Two mutations on production source, each run three times against the ICB
/// arm's canonical pair:
///
/// * widening the predicate to include **run heads** -- the case §11.3p says
///   must survive, 30 of the 505 -- moves `suppressed` 29945 -> 32225 and
///   leaves the digest at `a2d13650…`/`5b517964…`, **unmoved**;
/// * the naive *"the position is covered"* predicate §11.3p warns against is
///   the same set, and likewise passes.
///
/// Suppressing *everything* while replaying does corrupt, with §11.3l's
/// signature ("Error: sampling"), so the gate is not blind in general -- it is
/// blind to exactly this distinction. That is §2.4's rule arriving in a new
/// quantity: **a wrongly-dropped edge leaves the digest plausible**, and here it
/// leaves it *identical*. Full runs in
/// `measurements/issue-144-raw/mutation-*.txt`.
///
/// So the narrowing is asserted **mechanically, on the predicate itself**,
/// rather than being left to an end-to-end check that demonstrably does not
/// discriminate. A test that cannot fail is not a test (§15.1 #2), and the
/// digest gate is that test for this property.
#[test]
fn suppression_admits_a_run_interior_and_refuses_its_head() {
    if !icb_support_once() {
        return;
    }
    use crate::kernels::metal::executor::{Disposition, Executor};
    use crate::kernels::metal::icb::ReplayBarriers;

    let input: Vec<f32> = (0..64).map(|i| i as f32 * 0.5).collect();

    // `Always` must never license suppression, whatever the executor is doing.
    // This is the default arm and the one every recorded digest belongs to.
    let always = IcbExecutor::configured(2, BarrierScope::RunStart, ReplayBarriers::Always);
    let out_always = run_cos_steps(4, Some(&always), &input);
    assert!(
        always.is_replaying(),
        "the fixture must reach the replaying phase, or the assertion below is vacuous"
    );
    assert_eq!(
        always.disposition(),
        Disposition::Unknown,
        "ReplayBarriers::Always must never report a suppressible disposition"
    );

    // `SkipReplayed` on the same fixture. The executor is between steps here --
    // `end_step` reset `position` and `suppress_until` to 0 -- so no run is in
    // flight and the answer must still be `Unknown`. That is the run-head case
    // in its purest form: position 0 of a step is a head if it is covered at
    // all, and a head is never inside a suppression window.
    let skip = IcbExecutor::configured(2, BarrierScope::RunStart, ReplayBarriers::SkipReplayed);
    let out_skip = run_cos_steps(4, Some(&skip), &input);
    assert!(
        skip.is_replaying(),
        "the fixture must reach the replaying phase"
    );
    assert_eq!(
        skip.disposition(),
        Disposition::Unknown,
        "with no run in flight -- which is where every run HEAD is offered -- suppression          must not be licensed. §11.3p: a head's ICB scan slice is empty by construction, so          candle's fence is the only thing ordering a gap dispatch into it, and 30 of the 505          fire at heads with every one required."
    );

    // And the arm still computes what the classical path did. Both halves are
    // needed: an executor that suppressed nothing would pass the assertions
    // above trivially, and one that corrupted would pass them too.
    let classical = run_cos_steps(1, None, &input);
    assert_eq!(
        out_always, classical,
        "Always must be bit-identical to classical"
    );
    assert_eq!(
        out_skip, classical,
        "SkipReplayed must be bit-identical to classical"
    );

    // Non-vacuity, per this file's header: an all-zero fixture would satisfy
    // every equality above (§3.7a's characteristic ICB failure).
    assert!(
        distinct(&classical) > 2,
        "fixture wrote {} distinct value(s); the comparisons above are vacuous",
        distinct(&classical)
    );
}

/// The axis is a no-op on the classical path, by construction rather than by
/// measurement.
///
/// `ExecutorSlot::Classical` answers `Disposition::Unknown` without a virtual
/// call, so with no executor installed there is nothing a suppression decision
/// could be made against -- which is the property that makes this safe to land
/// dark (§9.2e's shape for `HazardKey`). Asserted here so a future change that
/// gave `Classical` a non-default answer would fail rather than silently alter
/// the default path.
#[test]
fn classical_never_reports_a_replayed_disposition() {
    use crate::kernels::metal::executor::Disposition;
    let slot = ExecutorSlot::Classical;
    assert_eq!(
        slot.disposition(),
        Disposition::Unknown,
        "Classical must never license suppression: with no executor installed there is no \
         second ordering source, so the axis has nothing to act on"
    );
}

/// Five `cos` dispatches with a classical one in the middle, so the covered
/// positions form two runs: `[0, 1]` and `[3, 4]`.
///
/// Positions 0 and 1 bind `src -> a -> b`; positions 3 and 4 bind `a -> b -> a`.
/// So `src` is bound by the first run only, which is what lets the test tell a
/// per-run residency set from a plan-wide one by an absolute reference rather
/// than by a count alone (`CONTRIBUTING.md` §3.1b). Returns the last output and
/// the three buffers' addresses.
fn run_two_runs(
    steps: usize,
    executor: Option<&Arc<IcbExecutor>>,
    input: &[f32],
) -> (Vec<f32>, [usize; 3]) {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    if let Some(e) = executor {
        cmds.set_executor(Arc::new(ExecutorSlot::Custom(e.clone())));
    }
    let options = crate::kernels::RESOURCE_OPTIONS;
    let bytes = std::mem::size_of_val(input);
    let src = device
        .new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, bytes, options)
        .unwrap();
    let a = device.new_buffer(bytes, options).unwrap();
    let b = device.new_buffer(bytes, options).unwrap();
    let packed = executor.is_some();

    for _ in 0..steps {
        {
            let guard = cmds.command_encoder().unwrap();
            let links: [(
                &crate::kernels::metal::Buffer,
                &crate::kernels::metal::Buffer,
            ); 5] = [(&src, &a), (&a, &b), (&b, &a), (&a, &b), (&b, &a)];
            for (i, (from, to)) in links.iter().enumerate() {
                // Position 2 binds its scalars inline, so no ICB command can
                // hold it (§3.7c) and the covered positions split around it.
                if packed {
                    set_default_param_style(if i == 2 {
                        ParamStyle::Split
                    } else {
                        ParamStyle::Packed
                    });
                }
                call_unary_contiguous(
                    &device,
                    &guard,
                    &kernels,
                    unary::contiguous::cos::FLOAT,
                    std::mem::size_of::<f32>(),
                    input.len(),
                    BufferOffset {
                        buffer: from,
                        offset_in_bytes: 0,
                    },
                    to,
                )
                .unwrap();
            }
        }
        cmds.flush_and_wait().unwrap();
        if let Some(e) = executor {
            e.end_step(&device, &cmds).unwrap();
        }
    }

    let ptr = a.contents() as *const f32;
    // SAFETY: the command buffer has completed and `a` holds `input.len()` f32
    // in shared storage -- the fifth link wrote it.
    let out = unsafe { std::slice::from_raw_parts(ptr, input.len()) }.to_vec();
    (out, [src.raw_addr(), a.raw_addr(), b.raw_addr()])
}

/// Each run's residency set is the buffers its own commands bind -- not the
/// plan's.
///
/// `DESIGN.md` §11.3w: declaring the whole plan at every run head is
/// `runs x |plan|` `useResource` calls per step, 28.5x what the runs touch. The
/// absolute references here are the fixture's own geometry: run 0 binds
/// `{src, a, b}` plus its two params blocks, run 1 binds `{a, b}` plus its two,
/// the plan binds seven distinct buffers, and a plan-wide set would put `src`
/// in run 1's declaration where its commands never bind it.
#[test]
fn each_run_declares_only_the_buffers_its_own_commands_bind() {
    require_icb_support!("each_run_declares_only_the_buffers_its_own_commands_bind");
    set_default_param_style(ParamStyle::Packed);
    set_constants_pool_enabled(true);

    let input: Vec<f32> = (0..512).map(|i| i as f32 / 16.0).collect();
    let executor = IcbExecutor::new(2);
    let (replayed, [src, a, b]) = run_two_runs(4, Some(&executor), &input);

    set_default_param_style(ParamStyle::Split);
    set_constants_pool_enabled(false);
    let (classical, _) = run_two_runs(1, None, &input);

    assert!(executor.is_replaying(), "nothing was replayed");
    let coverage = executor.coverage();
    assert_eq!(
        (
            coverage.positions,
            coverage.covered,
            coverage.inline_constants
        ),
        (5, 4, 1),
        "the fixture is four packed positions around one classical: {coverage:?}"
    );
    assert_eq!(
        executor.runs(),
        2,
        "the classical position splits the covered set in two"
    );
    assert_eq!(executor.stale_positions(), 0);

    let runs = executor.run_residency();
    assert_eq!(runs.len(), 2);
    assert_eq!(
        runs[0].len(),
        5,
        "run 0: src, a, b and two params blocks: {:?}",
        runs[0]
    );
    assert_eq!(
        runs[1].len(),
        4,
        "run 1: a, b and two params blocks: {:?}",
        runs[1]
    );
    assert_eq!(
        executor.plan_resident(),
        7,
        "src, a, b and four params blocks"
    );
    assert_eq!(
        executor.residency_per_step(),
        9,
        "5 + 4, against a plan-wide 2 x 7 = 14"
    );
    for run in &runs {
        assert!(
            run.contains(&a) && run.contains(&b),
            "every command binds a or b: {run:?}"
        );
    }
    assert!(
        runs[0].contains(&src),
        "run 0's first command reads src: {:?}",
        runs[0]
    );
    assert!(
        !runs[1].contains(&src),
        "run 1 never binds src; declaring it there is the plan-wide set: {:?}",
        runs[1]
    );

    // The replay still computes what the classical path did, with the narrower
    // declaration. Non-vacuity beside the equality, per §3.7a.
    assert!(
        distinct(&replayed) > 2,
        "{} distinct values",
        distinct(&replayed)
    );
    assert_eq!(
        replayed, classical,
        "replayed output differs from the classical path"
    );
}

/// One step of [`run_chain_shaped`]: a cos chain of `links` dispatches, with
/// link `moved` (if any) writing a third buffer instead of the one the
/// recording saw, so its binding identity -- and every later link's -- moves.
#[derive(Clone, Copy)]
struct ChainStep {
    links: usize,
    moved: Option<usize>,
}

/// Run a cos chain whose shape may change from step to step, returning the
/// last link's output after the final step.
///
/// The fixture for replay under binding movement: the first steps are the
/// steady state the executor records, a later step is the one whose operands
/// move (a buffer reissued, a dispatch that vanished), and the property under
/// test is that the executor computes what the classical path computes on
/// *that* step -- not on the one it recorded.
fn run_chain_shaped(
    shape: &[ChainStep],
    executor: Option<&Arc<IcbExecutor>>,
    input: &[f32],
) -> Vec<f32> {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    if let Some(e) = executor {
        cmds.set_executor(Arc::new(ExecutorSlot::Custom(e.clone())));
    }
    let options = crate::kernels::RESOURCE_OPTIONS;
    let bytes = std::mem::size_of_val(input);
    let src = device
        .new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, bytes, options)
        .unwrap();
    let a = device.new_buffer(bytes, options).unwrap();
    let b = device.new_buffer(bytes, options).unwrap();
    let c = device.new_buffer(bytes, options).unwrap();

    let mut last = &a;
    for step in shape {
        {
            let guard = cmds.command_encoder().unwrap();
            let mut from = &src;
            for link in 0..step.links {
                let to = if step.moved == Some(link) {
                    &c
                } else if link % 2 == 0 {
                    &a
                } else {
                    &b
                };
                call_unary_contiguous(
                    &device,
                    &guard,
                    &kernels,
                    unary::contiguous::cos::FLOAT,
                    std::mem::size_of::<f32>(),
                    input.len(),
                    BufferOffset {
                        buffer: from,
                        offset_in_bytes: 0,
                    },
                    to,
                )
                .unwrap();
                from = to;
                last = to;
            }
        }
        // The step closes before its work is committed, as the decode loop
        // closes it: a range the executor still owes at the boundary lands on
        // the queue ahead of this commit.
        if let Some(e) = executor {
            e.end_step(&device, &cmds).unwrap();
        }
        cmds.flush_and_wait().unwrap();
    }
    let ptr = last.contents() as *const f32;
    // SAFETY: the command buffer has completed and the buffer holds
    // `input.len()` f32 in shared storage.
    unsafe { std::slice::from_raw_parts(ptr, input.len()) }.to_vec()
}

/// A member whose operand moved runs classically, the members before it run
/// from the ICB, and the step computes what the classical path computes.
///
/// The run is validated position by position and issued at its tail, so the
/// command encoded for the moved member never executes: the validated prefix
/// is issued as a range, the moved member encodes as itself, and the rest of
/// the run follows it classically. The counters say which path each took --
/// one stale position, one prefix flush, one re-recording at the step's close
/// -- and the equality says none of them ran twice or from the recording.
///
/// Mutation-verified: issuing the range at the head (the previous form) makes
/// the moved link's recorded command write the buffer the recording saw, and
/// the classical dispatch write the moved one, so the chain's later links read
/// a value one cos short and the equality fails.
#[test]
fn a_member_whose_operand_moved_runs_classically_after_its_validated_prefix() {
    require_icb_support!(
        "a_member_whose_operand_moved_runs_classically_after_its_validated_prefix"
    );
    set_default_param_style(ParamStyle::Packed);
    set_constants_pool_enabled(true);

    let input: Vec<f32> = (0..512).map(|i| i as f32 / 16.0).collect();
    let steady = ChainStep {
        links: 5,
        moved: None,
    };
    let moved = ChainStep {
        links: 5,
        moved: Some(2),
    };
    let shape = [steady, steady, steady, moved];
    let executor = IcbExecutor::new(2);
    let replayed = run_chain_shaped(&shape, Some(&executor), &input);
    set_default_param_style(ParamStyle::Split);
    set_constants_pool_enabled(false);
    let classical = run_chain_shaped(&shape, None, &input);

    assert!(executor.is_replaying() || executor.rerecordings() > 0);
    assert_eq!(
        executor.stale_positions(),
        1,
        "the moved member, and only it"
    );
    assert_eq!(
        executor.prefix_flushes(),
        1,
        "its validated prefix was issued once"
    );
    assert_eq!(
        executor.rerecordings(),
        1,
        "the plan is dropped at the stale step's close"
    );
    assert!(distinct(&classical) > 1, "vacuous fixture");
    assert_eq!(
        replayed, classical,
        "the moved step computes what the classical path computes"
    );
}

/// A step that ends inside a run issues the validated prefix at the boundary.
///
/// The members offered before the stream ended were suppressed on the promise
/// of a range issued at the tail; the tail never came, so `end_step` issues
/// what was validated on the queue it is handed. Without that the step's last
/// dispatches would simply not run, and a read of their output would return
/// the previous step's -- a plausible value, not an error (§3.5).
#[test]
fn a_step_that_ends_inside_a_run_issues_its_validated_prefix_at_the_boundary() {
    require_icb_support!(
        "a_step_that_ends_inside_a_run_issues_its_validated_prefix_at_the_boundary"
    );
    set_default_param_style(ParamStyle::Packed);
    set_constants_pool_enabled(true);

    let input: Vec<f32> = (0..512).map(|i| i as f32 / 16.0).collect();
    let steady = ChainStep {
        links: 5,
        moved: None,
    };
    let short = ChainStep {
        links: 3,
        moved: None,
    };
    let shape = [steady, steady, steady, short];
    let executor = IcbExecutor::new(2);
    let replayed = run_chain_shaped(&shape, Some(&executor), &input);
    set_default_param_style(ParamStyle::Split);
    set_constants_pool_enabled(false);
    let classical = run_chain_shaped(&shape, None, &input);

    assert_eq!(
        executor.stale_positions(),
        0,
        "every offered position matched"
    );
    assert_eq!(
        executor.prefix_flushes(),
        1,
        "the prefix was issued at the step's close"
    );
    assert!(distinct(&classical) > 1, "vacuous fixture");
    assert_eq!(
        replayed, classical,
        "the short step computes its three links"
    );
}

/// One fused-epilogue GEMV per step over fixed buffers, optionally through
/// `executor`, returning the last step's output as f32 (lloom #732).
///
/// `orient`: `Weight` = `(1, n_out, 1, k)` with the weight on the left,
/// `Model` = `(1, 1, n_out, k)` with the vector on the left and the weight's
/// transposed strides `[1, k]` — the call `nn::ops::gemv_fused_metal` makes.
/// The operand at slot 15 is a row vector for `Residual` and `Mul`, a second
/// `[n_out, k]` weight for `Swiglu`.
#[derive(Clone, Copy)]
enum Orient {
    Weight,
    Model,
}

fn run_gemv_epilogue_steps(
    steps: usize,
    executor: Option<&Arc<IcbExecutor>>,
    epilogue: crate::kernels::kernels::mlx_gemm::GemvEpilogue,
    (k, n_out): (usize, usize),
    dtype: crate::kernels::kernels::mlx_gemm::GemmDType,
    orient: Orient,
) -> Vec<f32> {
    use crate::kernels::kernels::mlx_gemm::{call_mlx_gemv_epilogue, GemmDType, GemvEpilogue};
    use half::f16;
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    if let Some(e) = executor {
        cmds.set_executor(Arc::new(ExecutorSlot::Custom(e.clone())));
    }
    let options = crate::kernels::RESOURCE_OPTIONS;
    let up: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
        .collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
    let operand: Vec<f32> = match epilogue {
        GemvEpilogue::Swiglu => (0..n_out * k)
            .map(|i| ((i % 19) as f32 - 9.0) / 32.0)
            .collect(),
        _ => (0..n_out).map(|i| ((i % 7) as f32 - 3.0) / 4.0).collect(),
    };
    let mk = |v: &[f32]| -> crate::kernels::metal::Buffer {
        match dtype {
            GemmDType::F16 => {
                let h: Vec<f16> = v.iter().map(|x| f16::from_f32(*x)).collect();
                device
                    .new_buffer_with_data(
                        h.as_ptr() as *const std::ffi::c_void,
                        h.len() * 2,
                        options,
                    )
                    .unwrap()
            }
            _ => device
                .new_buffer_with_data(
                    v.as_ptr() as *const std::ffi::c_void,
                    std::mem::size_of_val(v),
                    options,
                )
                .unwrap(),
        }
    };
    let elt = if matches!(dtype, GemmDType::F16) {
        2usize
    } else {
        4
    };
    let (up_buf, vec_buf, op_buf) = (mk(&up), mk(&vec), mk(&operand));
    let out = device.new_buffer(n_out * elt, options).unwrap();
    let style = if executor.is_some() {
        ParamStyle::Packed
    } else {
        ParamStyle::Split
    };
    for _ in 0..steps {
        {
            let guard = cmds.command_encoder().unwrap();
            match orient {
                Orient::Weight => call_mlx_gemv_epilogue(
                    &device,
                    &guard,
                    &kernels,
                    dtype,
                    (1, n_out, 1, k),
                    &[n_out * k, k, 1],
                    0,
                    &up_buf,
                    &[k, 1, 1],
                    0,
                    &vec_buf,
                    &out,
                    style,
                    None,
                    epilogue,
                    Some((&op_buf, 0)),
                )
                .unwrap(),
                Orient::Model => call_mlx_gemv_epilogue(
                    &device,
                    &guard,
                    &kernels,
                    dtype,
                    (1, 1, n_out, k),
                    &[k, k, 1],
                    0,
                    &vec_buf,
                    &[1, k],
                    0,
                    &up_buf,
                    &out,
                    style,
                    None,
                    epilogue,
                    Some((&op_buf, 0)),
                )
                .unwrap(),
            }
        }
        cmds.flush_and_wait().unwrap();
        if let Some(e) = executor {
            e.end_step(&device, &cmds).unwrap();
        }
    }
    // SAFETY: completed, shared storage, `n_out` elements of `dtype`.
    match dtype {
        GemmDType::F16 => {
            unsafe { std::slice::from_raw_parts(out.contents() as *const f16, n_out) }
                .iter()
                .map(|h| h.to_f32())
                .collect()
        }
        _ => unsafe { std::slice::from_raw_parts(out.contents() as *const f32, n_out) }.to_vec(),
    }
}

/// A replayed fused-epilogue GEMV computes what the classical one did, bit
/// for bit — every operand-carrying epilogue, both dtypes, both
/// orientations, at the model's tiles including its FFN shape (lloom #732).
///
/// Six steps, so each command runs at most four times: this pins what one
/// replay computes. What a command does over many executions is
/// `a_command_replayed_every_step_matches_classical_on_every_step`.
#[test]
fn replayed_fused_gemv_epilogues_match_the_classical_ones() {
    use crate::kernels::kernels::mlx_gemm::{GemmDType, GemvEpilogue};
    require_icb_support!("replayed_fused_gemv_epilogues_match_the_classical_ones");
    for dtype in [GemmDType::F32, GemmDType::F16] {
        for &(epilogue, label) in &[
            (GemvEpilogue::Residual, "residual"),
            (GemvEpilogue::Mul, "mul"),
            (GemvEpilogue::Swiglu, "swiglu"),
        ] {
            for &shape in &[(2048usize, 512usize), (2048, 6144), (2048, 10752)] {
                for orient in [Orient::Weight, Orient::Model] {
                    set_default_param_style(ParamStyle::Packed);
                    set_constants_pool_enabled(true);
                    let executor = IcbExecutor::new(2);
                    let replayed =
                        run_gemv_epilogue_steps(6, Some(&executor), epilogue, shape, dtype, orient);
                    set_default_param_style(ParamStyle::Split);
                    set_constants_pool_enabled(false);
                    let classical =
                        run_gemv_epilogue_steps(1, None, epilogue, shape, dtype, orient);
                    assert!(
                        executor.is_replaying(),
                        "{label} {dtype:?} {shape:?}: never replayed"
                    );
                    let coverage = executor.coverage();
                    assert!(
                        coverage.covered > 0,
                        "{label} {dtype:?} {shape:?}: nothing covered: {coverage:?}"
                    );
                    assert_eq!(
                        executor.stale_positions(),
                        0,
                        "{label} {dtype:?} {shape:?}: stale"
                    );
                    assert!(
                        distinct(&replayed) > 2,
                        "{label} {dtype:?} {shape:?}: vacuous replayed output"
                    );
                    let differing = replayed
                        .iter()
                        .zip(classical.iter())
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    assert_eq!(
                        differing, 0,
                        "{label} {dtype:?} {shape:?}: replayed output differs from classical on {differing} of {} rows",
                        classical.len()
                    );
                }
            }
        }
    }
}

/// The quarantine list is empty: the swiglu it held is replayed, and the
/// model-level digest gate (`replay-adaptive/gate.sh`) is what shows that is
/// right.
#[test]
fn the_quarantine_list_is_empty() {
    use crate::kernels::metal::icb::is_quarantined;
    assert!(!is_quarantined(
        "gemv_float16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0_swiglu_packed"
    ));
    assert!(!is_quarantined(
        "gemv_float16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0_mul_packed"
    ));
    assert!(!is_quarantined(
        "gemv_float16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0_residual_packed"
    ));
    assert!(!is_quarantined(
        "gemv_float16_bm8_bn1_sm1_sn32_tm4_tn4_nc0_axpby0_packed"
    ));
    assert!(!is_quarantined("rmsnorm_f16_packed"));
}

/// How many steps [`run_mlp_steps`] runs: enough that each replayed command
/// is executed well past the 7th time, the first execution
/// `icb::Generation` guards against (lloom #740).
const MLP_STEPS: usize = 40;

/// The model's MLP, thirty `swiglu -> w2` pairs a step, [`MLP_STEPS`] steps:
/// f16, a `StorageModePrivate` arena, the swiglu reading x at +512000 and
/// writing h at +0, `w2` reading h and writing its output (plus a residual)
/// INTO x's slot. Every step starts from the same x (a classical copy) and
/// ends by copying its output into its own row of the result, so every step
/// computes the same thing and a wrong step is found by index.
///
/// `consumer` is how `w2` runs under the executor; see [`Consumer`]. Without
/// an executor both kernels bind inline (`Split`): the reference.
///
/// `pacing` says when the host waits for the GPU; see [`Pacing`].
fn run_mlp_steps(
    executor: Option<&Arc<IcbExecutor>>,
    consumer: Consumer,
    pacing: Pacing,
) -> Vec<f32> {
    use crate::kernels::kernels::mlx_gemm::{call_mlx_gemv_epilogue, GemmDType, GemvEpilogue};
    use crate::kernels::kernels::unary::{call_copy2d_with, copy2d};
    use half::f16;
    let (k, n_out, reps, steps) = (2048usize, 10752usize, 30usize, MLP_STEPS);
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    if let Some(e) = executor {
        cmds.set_executor(Arc::new(ExecutorSlot::Custom(e.clone())));
    }
    let shared = crate::kernels::RESOURCE_OPTIONS;
    let private = objc2_metal::MTLResourceOptions(
        objc2_metal::MTLResourceOptions::StorageModePrivate.0
            | objc2_metal::MTLResourceOptions::HazardTrackingModeUntracked.0,
    );
    let mk = |v: &[f32]| -> crate::kernels::metal::Buffer {
        let h: Vec<f16> = v.iter().map(|x| f16::from_f32(*x)).collect();
        device
            .new_buffer_with_data(h.as_ptr() as *const std::ffi::c_void, h.len() * 2, shared)
            .unwrap()
    };
    let up: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
        .collect();
    let gate: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 19) as f32 - 9.0) / 32.0)
        .collect();
    let w2: Vec<f32> = (0..n_out * k)
        .map(|i| ((i % 23) as f32 - 11.0) / 256.0)
        .collect();
    let vec: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) / 16.0).collect();
    let res: Vec<f32> = (0..k).map(|i| ((i % 5) as f32 - 2.0) / 8.0).collect();
    let (up_buf, gate_buf, w2_buf, x0) = (mk(&up), mk(&gate), mk(&w2), mk(&vec));
    let staging = device.new_buffer(1 << 20, shared).unwrap();
    // SAFETY: shared storage of 1 MiB, nothing in flight; both copies are
    // `k` f16 values inside it.
    unsafe {
        let r: Vec<f16> = res.iter().map(|x| f16::from_f32(*x)).collect();
        for at in [772096, 776192] {
            std::ptr::copy_nonoverlapping(
                r.as_ptr() as *const u8,
                staging.contents().add(at),
                k * 2,
            );
        }
    }
    let arena = device.new_buffer(1 << 20, private).unwrap();
    {
        let mut blit = cmds.blit_command_encoder().unwrap();
        blit.copy_from_buffer(&staging, 0, &arena, 0, 1 << 20);
    }
    cmds.flush_and_wait().unwrap();
    let x = arena.view(512000, 256000);
    let h = arena.view(0, 512000);
    let o = arena.view(512000, 256000);
    // Two identical copies: a consumer that alternates between them computes
    // the same thing and moves its binding offset every step.
    let residuals = [arena.view(772096, 4096), arena.view(776192, 4096)];
    let results = device.new_buffer(steps * k * 2, shared).unwrap();
    let (producer_style, consumer_style) = match (executor.is_some(), consumer) {
        (false, _) => (ParamStyle::Split, ParamStyle::Split),
        (true, Consumer::Replayed | Consumer::PackedMoving) => {
            (ParamStyle::Packed, ParamStyle::Packed)
        }
        (true, Consumer::Inline) => (ParamStyle::Packed, ParamStyle::Split),
    };
    let copy =
        |src: &crate::kernels::metal::Buffer, dst: &crate::kernels::metal::Buffer, dst_off| {
            let guard = cmds.command_encoder().unwrap();
            call_copy2d_with(
                &device,
                &guard,
                &kernels,
                copy2d::HALF,
                src,
                dst,
                1,
                k,
                k,
                k,
                0,
                dst_off,
                ParamStyle::Split,
            )
            .unwrap();
        };
    for step in 0..steps {
        let residual = match consumer {
            Consumer::PackedMoving => &residuals[step % 2],
            _ => &residuals[0],
        };
        copy(&x0, &x, 0);
        for _ in 0..reps {
            {
                let guard = cmds.command_encoder().unwrap();
                call_mlx_gemv_epilogue(
                    &device,
                    &guard,
                    &kernels,
                    GemmDType::F16,
                    (1, 1, n_out, k),
                    &[k, k, 1],
                    0,
                    &x,
                    &[1, k],
                    0,
                    &up_buf,
                    &h,
                    producer_style,
                    None,
                    GemvEpilogue::Swiglu,
                    Some((&gate_buf, 0)),
                )
                .unwrap();
            }
            {
                let guard = cmds.command_encoder().unwrap();
                call_mlx_gemv_epilogue(
                    &device,
                    &guard,
                    &kernels,
                    GemmDType::F16,
                    (1, 1, k, n_out),
                    &[n_out, n_out, 1],
                    0,
                    &h,
                    &[1, n_out],
                    0,
                    &w2_buf,
                    &o,
                    consumer_style,
                    None,
                    GemvEpilogue::Residual,
                    Some((residual, 0)),
                )
                .unwrap();
            }
        }
        copy(&o, &results, step * k * 2);
        let wait = match pacing {
            Pacing::Waited => true,
            Pacing::Bounded(depth) => (step + 1) % depth == 0,
            Pacing::Unbounded => false,
        };
        if wait {
            cmds.flush_and_wait().unwrap();
        } else {
            cmds.flush().unwrap();
        }
        if let Some(e) = executor {
            e.end_step(&device, &cmds).unwrap();
        }
    }
    cmds.flush_and_wait().unwrap();
    // SAFETY: completed, shared storage; `steps * k` f16 values.
    unsafe { std::slice::from_raw_parts(results.contents() as *const f16, steps * k) }
        .iter()
        .map(|v| v.to_f32())
        .collect()
}

/// When the host waits for the GPU between steps.
#[derive(Clone, Copy, Debug)]
enum Pacing {
    /// After every step: each generation's last use has completed by the next
    /// step, so the executor reuses one after setting its pipelines again.
    Waited,
    /// Every `n` steps, as the decode loop's pacer bounds the steps in
    /// flight: the executor encodes new generations while earlier ones run and
    /// reuses them once they complete.
    Bounded(usize),
    /// Never: the host encodes every step before the GPU finishes the first,
    /// so the generations reach their cap and later steps run classically.
    Unbounded,
}

/// How `w2`, the swiglu's consumer, runs while the swiglu is replayed.
#[derive(Clone, Copy, Debug)]
enum Consumer {
    /// Packed and stable, so replayed: each pair is one ICB run and `w2` is
    /// ordered after the swiglu by its command's `setBarrier`.
    Replayed,
    /// Bound inline, which the plan excludes (`Excluded::InlineConstants`):
    /// the swiglu is a run of one and `w2` a classical dispatch ordered after
    /// the range by the encoder's memory barrier.
    Inline,
    /// The packed kernel, excluded because its residual binding moves every
    /// step (`Excluded::Varies`): a classical dispatch of the kernel the model
    /// runs, reading its constants from the pool.
    PackedMoving,
}

/// Steps of `run`, `k` values each, whose output differs from `reference`'s.
fn differing_steps(run: &[f32], reference: &[f32], k: usize) -> Vec<usize> {
    (0..reference.len() / k)
        .filter(|s| {
            run[s * k..(s + 1) * k]
                .iter()
                .zip(&reference[s * k..(s + 1) * k])
                .any(|(a, b)| a.to_bits() != b.to_bits())
        })
        .collect()
}

/// Replay the MLP with `w2` as `consumer` against the classical reference and
/// require every step to agree bit for bit.
///
/// Before lloom #740 the swiglu's command went wrong on its 7th, 13th, 20th
/// and 26th execution (steps 8, 14, 21, 27 here): the trailing simdgroups of
/// each threadgroup never ran. The step count is what reaches it; eight steps,
/// which this harness ran first, stop one execution short.
fn assert_mlp_replays_like_classical(consumer: Consumer, pacing: Pacing, what: &str) {
    use crate::kernels::metal::encoder::{set_hazard_key, HazardKey};
    let k = 2048;
    set_hazard_key(HazardKey::Range);
    set_default_param_style(ParamStyle::Packed);
    set_constants_pool_enabled(true);
    let executor = IcbExecutor::new(2);
    let replayed = run_mlp_steps(Some(&executor), consumer, pacing);
    set_default_param_style(ParamStyle::Split);
    set_constants_pool_enabled(false);
    let classical = run_mlp_steps(None, consumer, pacing);
    set_hazard_key(HazardKey::Pointer);
    let cov = executor.coverage();
    eprintln!(
        "[{what}] covered={} inline={} varies={} runs={} barriers={} stale={} \
         generations_built={} pipeline_refreshes={} skipped_steps={}",
        cov.covered,
        cov.inline_constants,
        cov.varies,
        executor.runs(),
        executor.encoded_barriers(),
        executor.stale_positions(),
        executor.generations_built(),
        executor.pipeline_refreshes(),
        executor.skipped_steps(),
    );
    assert!(executor.is_replaying(), "{what}: never replayed");
    assert_eq!(cov.quarantined, 0, "{what}: the swiglu was quarantined");
    assert!(
        cov.covered >= 30,
        "{what}: the swiglu was not covered: {cov:?}"
    );
    // The reference must be a reference: every classical step computes the
    // same thing, so every one equals the first.
    let classical_first = classical[..k].repeat(MLP_STEPS);
    assert!(
        differing_steps(&classical, &classical_first, k).is_empty(),
        "{what}: the classical steps disagree with each other"
    );
    assert!(distinct(&replayed) > 2, "{what}: vacuous replayed output");
    let bad = differing_steps(&replayed, &classical, k);
    eprintln!("[{what}] steps whose output differs from classical: {bad:?}");
    assert!(
        bad.is_empty(),
        "{what}: {} replayed steps differ from classical, first {:?}",
        bad.len(),
        &bad[..bad.len().min(8)]
    );
    // Each pacing must engage the path it exists for, or its pass reads
    // nothing: a bounded run never falls back and both builds and reuses; an
    // unbounded one reaches the cap.
    match pacing {
        Pacing::Waited | Pacing::Bounded(_) => {
            assert_eq!(
                executor.skipped_steps(),
                0,
                "{what}: a paced step fell back"
            );
            assert!(
                executor.pipeline_refreshes() > 0,
                "{what}: no generation reused"
            );
        }
        Pacing::Unbounded => {
            assert!(
                executor.skipped_steps() > 0,
                "{what}: the cap was never reached"
            );
        }
    }
    if let Pacing::Bounded(_) = pacing {
        assert!(
            executor.generations_built() > 0,
            "{what}: no generation built"
        );
    }
}

/// The swiglu and `w2` both replayed, four steps in flight at a time.
#[test]
fn a_replayed_long_command_with_the_previous_step_in_flight() {
    require_icb_support!("a_replayed_long_command_with_the_previous_step_in_flight");
    assert_mlp_replays_like_classical(Consumer::Replayed, Pacing::Bounded(4), "replayed consumer");
}

/// The swiglu replayed and `w2` classical (bound inline).
#[test]
fn a_classical_consumer_of_a_replayed_long_command() {
    require_icb_support!("a_classical_consumer_of_a_replayed_long_command");
    assert_mlp_replays_like_classical(Consumer::Inline, Pacing::Bounded(4), "inline consumer");
}

/// The swiglu replayed and `w2` the packed kernel run classically -- the
/// model's shape when the swiglu is the only replayed position.
#[test]
fn a_classical_packed_consumer_of_a_replayed_long_command() {
    require_icb_support!("a_classical_packed_consumer_of_a_replayed_long_command");
    assert_mlp_replays_like_classical(
        Consumer::PackedMoving,
        Pacing::Bounded(4),
        "packed classical consumer",
    );
}

/// Every step waited for: each generation's last use has completed by the
/// next step, so the executor reuses one and sets its pipelines again rather
/// than encoding new ones -- the path a quiet GPU takes.
#[test]
fn a_command_replayed_every_step_matches_classical_on_every_step() {
    require_icb_support!("a_command_replayed_every_step_matches_classical_on_every_step");
    assert_mlp_replays_like_classical(Consumer::Inline, Pacing::Waited, "waited steps");
}

/// No host wait at all: the generations reach their cap while every one is
/// still in flight, and the steps after run classically rather than execute a
/// command a second time without its pipeline set again.
#[test]
fn a_run_ahead_past_the_generation_cap_falls_back_to_classical() {
    require_icb_support!("a_run_ahead_past_the_generation_cap_falls_back_to_classical");
    assert_mlp_replays_like_classical(Consumer::Inline, Pacing::Unbounded, "unbounded run-ahead");
}
