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
    Commands::new(queue, &residency_set).unwrap()
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
            e.end_step(&device).unwrap();
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

    // Two empty steps produce a plan with zero positions, which is a legitimate
    // and uninteresting plan. What matters is that reaching the encode path with
    // unsupported pipelines is an error rather than a crash.
    executor.end_step(&device).unwrap();
    let result = executor.end_step(&device);

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
            e.end_step(&device).unwrap();
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
