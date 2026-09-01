//! The GPU bump allocator, against the CPU plan it must reproduce (issue #70).
//!
//! `DESIGN.md` §9.2d case 1 says GPU-computed offsets into one arena buffer are
//! the case we build, and §11.3c verified the mechanism in a probe. These tests
//! are the difference between "the mechanism works" and "it produces the same
//! answer as the path it replaces", which is the acceptance bar #70 sets:
//! bit-identical, not close.
//!
//! The central test is [`gpu_offsets_match_the_non_aliasing_plan`]. Everything
//! else exists to stop that one from passing for a bad reason -- a vacuous
//! comparison, a kernel that never ran, an ordering that happens to hold on a
//! quiet machine.

use crate::metal::{
    arena::{plan_from_intervals, ArenaLayout, StepPlan, ARENA_ALIGNMENT},
    ArenaCursor, ArenaOffsets, Commands, ResidencySet,
};
use crate::{
    call_arena_alloc_report, call_arena_bump, call_arena_bump_concurrent, call_arena_reset, Device,
    Kernels, ARENA_DECLINED,
};
use std::sync::Arc;

fn device() -> Device {
    Device::system_default().expect("no Metal device")
}

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, &residency_set).unwrap()
}

/// Run the sequential bump allocator over `plan` and return the cursor state.
///
/// Waits for completion before reading, because the offsets live in shared
/// storage and reading them mid-flight is a race the host cannot detect.
fn run_bump(plan: &StepPlan) -> ArenaCursor {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    let sizes = plan.request_sizes();
    let cursor = ArenaCursor::new(&device, &sizes, plan.bump_capacity().max(1)).unwrap();

    {
        let guard = cmds.command_encoder().unwrap();
        call_arena_bump(
            &device,
            &guard,
            &kernels,
            cursor.cursor_buffer(),
            cursor.sizes_buffer(),
            cursor.offsets_buffer(),
            cursor.len(),
            cursor.capacity(),
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    cursor
}

/// A plan with one slot per value, laid out in ordinal order -- §9.3's
/// non-aliasing reference, and the layout a forward-only cursor can reproduce.
fn reference_plan(sizes: &[usize], excluded: &[bool]) -> StepPlan {
    let first: Vec<usize> = (0..sizes.len()).collect();
    let last: Vec<usize> = (0..sizes.len()).collect();
    plan_from_intervals(sizes, &first, &last, excluded, ArenaLayout::NonAliasing)
}

/// **The acceptance bar for issue #70**: a kernel bump-allocating over the
/// arena assigns every ordinal the byte the CPU planner would have.
///
/// This is an equality over the whole offset table, not a tolerance and not a
/// spot check. If every ordinal resolves to the same byte, every kernel binds
/// what it bound before, so the activations are **bit-identical by
/// construction** rather than by measurement -- which is the only argument
/// available under `HazardTrackingModeUntracked`, where a wrong offset is
/// silent corruption rather than an error (§3.5, §9.3).
///
/// The sizes are LFM2's decode shapes (§5.9): the MLP intermediate at 21504 B,
/// the residual stream at 4096, a kv-head value at 1024, and a declined ordinal
/// standing for session state.
///
/// **One deliberately unaligned size (300 B) is included, and it is load-bearing
/// rather than decorative.** Every real LFM2 decode size is a multiple of 128,
/// so a mutation that drops the allocator's `align_up` entirely leaves the
/// offsets for those shapes *unchanged* -- this test passed under exactly that
/// mutation until the 300 was added. A parity test built only from the shapes
/// the model happens to use cannot see an alignment defect at all, which is a
/// gap worth stating: the model's own sizes are the weakest possible input for
/// this check.
#[test]
fn gpu_offsets_match_the_non_aliasing_plan() {
    let sizes = [4096usize, 21504, 1024, 300, 21504, 4096];
    let excluded = [false, false, true, false, false, false];
    let plan = reference_plan(&sizes, &excluded);

    assert!(
        plan.is_bump_reproducible(),
        "the reference plan is not monotone, so no cursor could reproduce it"
    );

    let cursor = run_bump(&plan);
    let expected = plan.expected_offsets();

    cursor
        .verify_against(&expected)
        .expect("GPU offsets disagree with the CPU plan");

    // Non-vacuity, per §15.1 #1 and #53: two paths that both produce nothing
    // agree perfectly. A declined ordinal is legitimately the sentinel, so the
    // guard counts *served* ordinals that got a real offset, and requires more
    // than one distinct value -- an allocator that returned 0 for everything
    // would otherwise pass.
    let got = cursor.offsets();
    let served: Vec<u32> = got
        .iter()
        .copied()
        .filter(|&o| o != ARENA_DECLINED)
        .collect();
    assert_eq!(
        served.len(),
        5,
        "expected 5 served ordinals, got {served:?}"
    );
    let distinct: std::collections::HashSet<u32> = served.iter().copied().collect();
    assert!(
        distinct.len() > 1,
        "every served ordinal got the same offset {served:?}; the comparison is vacuous"
    );

    // The declined ordinal consumed no bytes: ordinal 3 sits where it would
    // have had ordinal 2 never existed. This is what keeps an exclusion from
    // renumbering the ordinals after it (§9.1).
    assert_eq!(got[2], ARENA_DECLINED, "session state was served a slot");
    assert_eq!(
        got[3] as usize,
        expected[3].unwrap(),
        "a declined ordinal shifted the ordinals after it"
    );
}

/// **A bump allocator needs more bytes than the plan's `arena_bytes`**, and
/// giving it the wrong figure silently costs arena coverage.
///
/// `arena_bytes` is where the last *value* ends; a cursor rounds every request
/// including the last, so it ends where the last *slot* ends. Found by this
/// test failing on `[100, 300, 5000]`: the plan reports 5512 and the cursor
/// needs 5632, so the final ordinal -- which fits the plan exactly -- was
/// declined for want of 120 bytes of tail padding.
///
/// The failure mode is quiet rather than corrupting: a declined ordinal falls
/// through to the pool, so the model stays correct and the GPU path just serves
/// fewer ordinals than the CPU one, with nothing pointing at the cause. Pinned
/// because the two figures differ only when the last size is not a multiple of
/// 128, so a plan of round sizes hides it entirely -- and LFM2's decode sizes
/// (21504, 4096, 1024) are all multiples of 128.
#[test]
fn bump_capacity_exceeds_arena_bytes_when_the_last_size_is_unrounded() {
    let unrounded = reference_plan(&[100, 300, 5000], &[false; 3]);
    assert_eq!(unrounded.arena_bytes(), 5512);
    assert_eq!(unrounded.bump_capacity(), 5632);
    assert!(
        unrounded.bump_capacity() > unrounded.arena_bytes(),
        "the cursor's capacity must cover the last slot's padding"
    );

    // LFM2's own shapes are all multiples of the alignment, so the two agree
    // there -- which is exactly why this needed a test rather than a run.
    let lfm2 = reference_plan(&[21504, 4096, 1024], &[false; 3]);
    assert_eq!(
        lfm2.arena_bytes(),
        lfm2.bump_capacity(),
        "aligned sizes should need no extra tail"
    );
}

/// Every offset the allocator hands out is 128 B aligned.
///
/// §9.2: the alignment covers every Metal dtype *and* is `hw.cachelinesize`, so
/// lowering it would silently introduce false sharing between adjacent slots.
/// A GPU allocator that aligned differently from the CPU one would produce a
/// layout that still looked plausible, which is why this is asserted on the
/// kernel's own output rather than inferred from the constant.
#[test]
fn gpu_offsets_are_cache_line_aligned() {
    // Sizes deliberately not multiples of 128, so the kernel has to round.
    let sizes = [100usize, 300, 5000];
    let plan = reference_plan(&sizes, &[false, false, false]);
    let cursor = run_bump(&plan);

    for (i, off) in cursor.offsets().iter().enumerate() {
        assert_ne!(*off, ARENA_DECLINED, "ordinal {i} was declined");
        assert_eq!(
            *off as usize % ARENA_ALIGNMENT,
            0,
            "ordinal {i} got offset {off}, which is not {ARENA_ALIGNMENT} B aligned"
        );
    }
}

/// Two runs of the allocator over the same plan produce the same offsets.
///
/// The determinism §2.3 requires, at the allocator. It is not free: a bump
/// allocator whose increments race would hand ordinals different offsets each
/// run, and §2.3.2 says a nondeterministic layout is indistinguishable from a
/// missing fence. The sequential kernel earns this by taking the ordering out
/// of the atomic; `concurrent_bump_does_not_fix_an_ordinal_to_an_offset` is the
/// control showing the alternative genuinely fails it.
#[test]
fn the_allocator_is_deterministic_across_runs() {
    let sizes = [4096usize, 21504, 1024, 21504];
    let plan = reference_plan(&sizes, &[false; 4]);

    let first = run_bump(&plan).offsets();
    for run in 1..8 {
        let again = run_bump(&plan).offsets();
        assert_eq!(again, first, "run {run} produced different offsets");
    }
}

/// **The negative control for the choice of a single-threaded allocator.**
///
/// A concurrent `atomic_fetch_add` gives every thread a *disjoint* slice --
/// that much is guaranteed -- but fixes no mapping from ordinal to offset,
/// because the order in which threads reach the atomic is unspecified. So the
/// same plan can produce a different layout each run.
///
/// This asserts the disjointness (the property that does hold) and reports the
/// ordering (the property that does not), rather than asserting that the
/// ordering fails -- which would be a test of the scheduler and could pass or
/// fail for reasons unrelated to the allocator. A run where the concurrent
/// version happens to agree is not a failure; the point is that nothing
/// guarantees it, which is why the decode path does not use it.
///
/// Without this, "the allocator is single-threaded" reads as caution. With it,
/// it is a choice with evidence under it.
#[test]
fn concurrent_bump_does_not_fix_an_ordinal_to_an_offset() {
    let sizes = [4096usize; 32];
    let plan = reference_plan(&sizes, &[false; 32]);

    let device = device();
    let kernels = Kernels::new();
    let mut layouts = Vec::new();

    for _ in 0..8 {
        let cmds = commands(&device);
        let cursor =
            ArenaCursor::new(&device, &plan.request_sizes(), plan.bump_capacity().max(1)).unwrap();
        {
            let guard = cmds.command_encoder().unwrap();
            call_arena_bump_concurrent(
                &device,
                &guard,
                &kernels,
                cursor.cursor_buffer(),
                cursor.sizes_buffer(),
                cursor.offsets_buffer(),
                cursor.len(),
                cursor.capacity(),
            )
            .unwrap();
        }
        cmds.wait_until_completed().unwrap();
        layouts.push(cursor.offsets());
    }

    // The property that DOES hold: claims are disjoint, so no two ordinals
    // share a byte. This is what makes a concurrent bump allocator *safe* and
    // still unsuitable.
    for (run, offsets) in layouts.iter().enumerate() {
        let distinct: std::collections::HashSet<u32> = offsets.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            offsets.len(),
            "run {run} handed two ordinals the same offset: {offsets:?}"
        );
    }

    // The property that does NOT hold is reported rather than asserted, for the
    // reason in the doc comment.
    let all_same = layouts.iter().all(|l| l == &layouts[0]);
    eprintln!(
        "concurrent bump: {} of {} runs produced the first run's layout \
         (agreement here is luck, not a guarantee -- the decode path uses \
         arena_bump_sequential)",
        layouts.iter().filter(|l| *l == &layouts[0]).count(),
        layouts.len(),
    );
    if all_same {
        eprintln!("note: all runs agreed this time; nothing guarantees they will");
    }
}

/// An allocation that would run past the arena is declined, not wrapped.
///
/// Wrapping would hand out an offset addressing another slot's bytes, and under
/// `HazardTrackingModeUntracked` there is no safety net (§3.5) -- the two values
/// would simply share memory and corrupt each other intermittently. Declining
/// is visible to the host, which can report it.
#[test]
fn an_allocation_past_the_arena_is_declined() {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);

    // Three 4096 B requests against an arena with room for two.
    let sizes = [4096u32, 4096, 4096];
    let capacity = 2 * 4096;
    let cursor = ArenaCursor::new(&device, &sizes, capacity).unwrap();
    {
        let guard = cmds.command_encoder().unwrap();
        call_arena_bump(
            &device,
            &guard,
            &kernels,
            cursor.cursor_buffer(),
            cursor.sizes_buffer(),
            cursor.offsets_buffer(),
            cursor.len(),
            cursor.capacity(),
        )
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();

    let offs = cursor.offsets();
    assert_eq!(offs[0], 0);
    assert_eq!(offs[1], 4096);
    assert_eq!(
        offs[2], ARENA_DECLINED,
        "an allocation past the arena was served offset {}",
        offs[2]
    );
}

/// The reset returns the cursor to 0, so the next step reproduces the first
/// step's offsets exactly.
///
/// That equality *is* the per-step reset's correctness condition: the arena
/// exists to give a dispatch position the same bytes on every token (§9.2c's
/// "674 varying → 0"), and a reset that left the cursor anywhere else would
/// walk the layout forward every step.
///
/// The ordering of this reset against the previous step's reads is the separate
/// and harder question -- see
/// [`reset_orders_against_the_previous_steps_arena_reads`].
#[test]
fn the_reset_makes_every_step_reproduce_the_first() {
    let device = device();
    let kernels = Kernels::new();
    let sizes = [4096usize, 21504, 1024];
    let plan = reference_plan(&sizes, &[false; 3]);
    let cursor = ArenaCursor::new(&device, &plan.request_sizes(), plan.bump_capacity()).unwrap();
    let arena = device
        .new_buffer(plan.arena_bytes().max(1), crate::RESOURCE_OPTIONS)
        .unwrap();

    let mut per_step = Vec::new();
    for _ in 0..6 {
        let cmds = commands(&device);
        {
            let guard = cmds.command_encoder().unwrap();
            // The reset opens the step, exactly as the decode path would.
            call_arena_reset(&device, &guard, &kernels, cursor.cursor_buffer(), &arena).unwrap();
            call_arena_bump(
                &device,
                &guard,
                &kernels,
                cursor.cursor_buffer(),
                cursor.sizes_buffer(),
                cursor.offsets_buffer(),
                cursor.len(),
                cursor.capacity(),
            )
            .unwrap();
        }
        cmds.wait_until_completed().unwrap();
        per_step.push(cursor.offsets());
    }

    let first = &per_step[0];
    for (i, step) in per_step.iter().enumerate() {
        assert_eq!(step, first, "step {i} produced different offsets: {step:?}");
    }
    // And it is the plan's layout, not merely a stable one.
    cursor
        .verify_against(&plan.expected_offsets())
        .expect("the repeated layout is not the planned one");
}

/// **The reset is ordered against the previous step's arena reads, and the
/// ordering is candle's barrier rather than the kernel's fence alone.**
///
/// The fence inside `arena_reset_cursor` orders that thread's own device memory
/// operations. It cannot order a *different dispatch*: dispatches within one
/// encoder overlap and the GPU does not drain between them (§3.5). So the reset
/// additionally needs a `memoryBarrierWithScope(Buffers)` separating it from
/// the previous step's arena readers, and `call_arena_reset` obtains one by
/// binding the arena as an output -- a write-after-read against every dispatch
/// that read the arena, which is exactly what `auto_barrier` emits on.
///
/// This test asserts that the barrier is actually emitted, by counting them
/// around the reset. The mutation it is proof against is dropping the arena
/// binding from `call_arena_reset`: the reset would still set the cursor
/// correctly in isolation, the offsets would still be right, every other test
/// here would still pass -- and the reset would race the previous step's reads,
/// which under `HazardTrackingModeUntracked` is silent corruption rather than
/// an error (§9.3, §2.3.2).
///
/// It is worth being precise about what this does and does not establish. It
/// establishes that the ordering edge is *emitted*. It does not establish that
/// the hardware honours it, which is §2.3.8's territory and is why the
/// acceptance evidence for this issue is the LFM2 determinism gate as well as
/// this test.
#[test]
fn reset_orders_against_the_previous_steps_arena_reads() {
    use crate::metal::trace;

    if !trace::trace_requested() {
        // The barrier counter only records under CANDLE_METAL_TRACE, so without
        // it this test cannot observe anything. Reported rather than silently
        // passing: an instrument that cannot be shown to have engaged has not
        // measured anything (§2.4, and #69's vacuous determinism run).
        eprintln!(
            "skipping barrier assertion: CANDLE_METAL_TRACE is unset, so no barrier \
             is recorded. Run with CANDLE_METAL_TRACE=1 to exercise it."
        );
        return;
    }

    // Count the barriers a reset emits, with and without a prior arena reader.
    //
    // **The difference between the two arms is the measurement**, and a bare
    // count is not. The reset also rewrites the cursor, which every allocator
    // dispatch binds as an output, so a write-after-write on the *cursor*
    // emits a barrier whichever way the arena is bound -- a test asserting
    // `barriers > 0` therefore passes with the arena binding deleted. That is
    // not hypothetical: it is what the first version of this test did, and the
    // mutation below is what caught it.
    let barriers_with = |read_arena: bool| -> usize {
        let device = device();
        let kernels = Kernels::new();
        let cmds = commands(&device);
        let arena = device.new_buffer(4096, crate::RESOURCE_OPTIONS).unwrap();
        let cursor = ArenaCursor::new(&device, &[128], 4096).unwrap();

        let _ = trace::take_dispatches();
        trace::set_recording(true);
        {
            let guard = cmds.command_encoder().unwrap();
            if read_arena {
                // Stand in for the previous decode step: a **completed
                // dispatch** that read the arena's bytes. It has to be a real
                // dispatch, not just a binding: candle moves a binding from
                // `next_inputs` to `prev_inputs` inside `auto_barrier`, which
                // runs per dispatch, so a bind with no dispatch after it is
                // never in the set the next conflict check consults.
                let enc: &crate::metal::ComputeCommandEncoder = guard.as_ref();
                enc.set_input_buffer(5, Some(&arena), 0);
                call_arena_alloc_report(&device, &guard, &kernels, cursor.offsets_buffer())
                    .unwrap();
            }
            call_arena_reset(&device, &guard, &kernels, cursor.cursor_buffer(), &arena).unwrap();
        }
        cmds.wait_until_completed().unwrap();
        trace::set_recording(false);
        trace::take_dispatches()
            .iter()
            .filter(|d| d.barrier)
            .count()
    };

    let without_reader = barriers_with(false);
    let with_reader = barriers_with(true);

    assert!(
        with_reader > without_reader,
        "the reset emitted {with_reader} barriers after an arena read and \
         {without_reader} without one. Equal counts mean the reset is NOT ordered \
         against the previous step's arena reads -- under \
         HazardTrackingModeUntracked that is silent corruption, not an error \
         (DESIGN.md §3.5, §9.3)."
    );
}

/// The kernel's alignment and sentinel agree with the Rust side's.
///
/// Checked across the boundary rather than asserted on each side: a
/// `static_assert` in MSL proves only that MSL agrees with itself, and the
/// failure being guarded against is the two disagreeing. §11.3d makes the same
/// argument for struct layouts and found a real width mismatch by it.
#[test]
fn arena_alloc_reports_alignment() {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    let out = device.new_buffer(8, crate::RESOURCE_OPTIONS).unwrap();

    {
        let guard = cmds.command_encoder().unwrap();
        call_arena_alloc_report(&device, &guard, &kernels, &out).unwrap();
    }
    cmds.wait_until_completed().unwrap();

    // SAFETY: shared storage, two u32, and the command buffer has completed.
    let got = unsafe { std::slice::from_raw_parts(out.contents() as *const u32, 2) };
    assert_eq!(
        got[0] as usize, ARENA_ALIGNMENT,
        "arena_alloc.metal aligns to {} where Rust aligns to {ARENA_ALIGNMENT}",
        got[0]
    );
    assert_eq!(
        got[1], ARENA_DECLINED,
        "the kernel's declined sentinel is {}, Rust's is {ARENA_DECLINED}",
        got[1]
    );
}

/// A packed plan is **not** bump-reproducible, and the code says so rather than
/// discovering it at a wrong bind.
///
/// Packing reuses slots, so ordinal 7 can resolve to an offset earlier than
/// ordinal 3's -- which a forward-only cursor cannot produce. This is the
/// constraint that decides which layout the GPU path serves, so it is pinned
/// rather than left to the layout enum's name.
#[test]
fn a_packed_plan_is_not_bump_reproducible() {
    // Two values whose intervals are disjoint, so packing gives them one slot,
    // plus a third that overlaps both.
    let sizes = [4096usize, 4096, 4096];
    let first = [0usize, 0, 2];
    let last = [1usize, 2, 2];
    let packed = plan_from_intervals(&sizes, &first, &last, &[false; 3], ArenaLayout::Packed);
    let reference =
        plan_from_intervals(&sizes, &first, &last, &[false; 3], ArenaLayout::NonAliasing);

    assert!(
        reference.is_bump_reproducible(),
        "the non-aliasing reference must be monotone"
    );
    assert!(
        !packed.is_bump_reproducible(),
        "a packed plan that reuses a slot cannot be monotone: {:?}",
        packed.expected_offsets()
    );
}

/// **An arena refuses GPU offsets that disagree with its plan**, rather than
/// binding them.
///
/// The check runs *before* the switch, so a wrong offset can never reach a
/// bind. That ordering is the whole safety argument for the GPU path: §9.3 says
/// aliasing correctness rests entirely on this arithmetic and that
/// `HazardTrackingModeUntracked` gives no safety net, so an offset wrong by one
/// slot does not fail -- it silently overlaps another value and corrupts
/// intermittently (§2.3.2). Refusing costs nothing, because the CPU path it
/// falls back to is the default anyway.
///
/// Three ways to disagree are checked, because they are separate code paths and
/// a fix for one would not catch the others: a served ordinal at the wrong
/// offset, a declined ordinal handed a real offset, and a table of the wrong
/// length.
#[test]
fn an_arena_refuses_gpu_offsets_that_disagree_with_the_plan() {
    let device = device();
    let sizes = [4096usize, 21504, 1024];
    let excluded = [false, true, false];
    let plan = reference_plan(&sizes, &excluded);
    let base = device
        .new_buffer(plan.bump_capacity().max(1), crate::RESOURCE_OPTIONS)
        .unwrap();
    let arena = crate::metal::Arena::new(base, plan.clone(), ArenaLayout::NonAliasing).unwrap();

    let correct: Vec<u32> = plan
        .expected_offsets()
        .iter()
        .map(|e| e.map_or(ARENA_DECLINED, |o| o as u32))
        .collect();

    // Control: the correct table is adopted, so the rejections below are not
    // "this arena refuses everything".
    assert_eq!(arena.offsets(), ArenaOffsets::Cpu, "default is not the CPU");
    arena
        .adopt_gpu_offsets(&correct)
        .expect("the correct table was refused");
    assert_eq!(
        arena.offsets(),
        ArenaOffsets::Gpu,
        "adoption did not record that the GPU path engaged"
    );
    assert_eq!(arena.counters().gpu_verified, 1);

    // 1. A served ordinal at the wrong offset -- off by one slot, the shape a
    //    real arithmetic slip produces.
    let mut wrong = correct.clone();
    wrong[0] += ARENA_ALIGNMENT as u32;
    let err = arena
        .adopt_gpu_offsets(&wrong)
        .expect_err("a wrong offset was adopted");
    assert!(err.contains("ordinal 0"), "unexpected error: {err}");

    // 2. A declined ordinal handed a real offset -- session state pulled into
    //    the arena, which §9.1 says must never happen.
    let mut intruding = correct.clone();
    intruding[1] = 0;
    let err = arena
        .adopt_gpu_offsets(&intruding)
        .expect_err("session state was admitted");
    assert!(err.contains("ordinal 1"), "unexpected error: {err}");

    // 3. A table of the wrong length, which would otherwise index past the plan.
    let err = arena
        .adopt_gpu_offsets(&correct[..2])
        .expect_err("a short table was adopted");
    assert!(err.contains("ordinals"), "unexpected error: {err}");
}

/// The offset source defaults to the CPU, so an unconfigured process is #69's
/// path byte for byte.
///
/// The same property `HazardKey::Pointer` and `ArenaLayout::Packed` preserve
/// for their axes: a new mechanism is opt-in, and the default is what shipped.
#[test]
fn the_default_offset_source_is_the_cpu() {
    assert_eq!(ArenaOffsets::default(), ArenaOffsets::Cpu);
    assert!(!ArenaOffsets::default().is_gpu());
    assert!(ArenaOffsets::Gpu.is_gpu());
}
