//! The scratch class on the device: disjointness, the fence, and the merge
//! order **as executed** (`DESIGN.md` §9.4, §10.4, issue #71).
//!
//! `scratch_tests.rs` covers the arithmetic, which needs no GPU. These are the
//! three properties that can only be established by running:
//!
//! 1. **N partials writing disjoint regions need no fences between them**, and
//!    the disjointness is *our* assertion rather than the driver's (§3.5). So it
//!    is checked by execution -- run them, read every byte back, and show no
//!    region was touched by another's dispatch.
//! 2. **The KV append must be fenced before the partials read** (§9.4). Easy to
//!    forget "because it is not between the two things one is thinking about",
//!    so the test is written to **fail without the fence**, mechanistically
//!    rather than by sampling a race.
//! 3. **The combine merges in index order** (§10.4). Asserted against the order
//!    the kernel actually walked, recorded by the kernel itself -- an ordering
//!    that cannot be observed cannot be asserted.

use crate::kernels::kernels::{ScratchKernel, ScratchParams};
use crate::kernels::metal::scratch::*;
use crate::kernels::metal::{Commands, ResidencySet};
use crate::kernels::{
    call_scratch_combine, call_scratch_partials, call_scratch_report, Device, Kernels, Source,
};
use objc2_metal::MTLResourceOptions;
use std::sync::Arc;

const SHARED: MTLResourceOptions = MTLResourceOptions(
    MTLResourceOptions::StorageModeShared.0 | MTLResourceOptions::HazardTrackingModeUntracked.0,
);

fn device() -> Device {
    Device::system_default().expect("no Metal device")
}

fn commands(device: &Device) -> Commands {
    let queue = device.new_command_queue().unwrap();
    let residency_set = Arc::new(ResidencySet::new(device));
    Commands::new(queue, residency_set).unwrap()
}

fn read_f32(buf: &crate::kernels::Buffer, n: usize) -> Vec<f32> {
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let p = unsafe { buf.contents() } as *const f32;
    assert!(!p.is_null(), "buffer has no CPU mapping");
    // SAFETY: shared storage holding at least `n` f32, and the caller has waited
    // for the command buffer.
    unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
}

fn read_u32(buf: &crate::kernels::Buffer, n: usize) -> Vec<u32> {
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let p = unsafe { buf.contents() } as *const u32;
    assert!(!p.is_null(), "buffer has no CPU mapping");
    // SAFETY: as `read_f32`.
    unsafe { std::slice::from_raw_parts(p, n) }.to_vec()
}

fn write_params(buf: &crate::kernels::Buffer, p: ScratchParams) {
    // SAFETY: the test owns every buffer bound or mapped here and reads its
    // results only after waiting for the command buffer that writes them.
    let dst = unsafe { buf.contents() } as *mut ScratchParams;
    assert!(!dst.is_null(), "params buffer has no CPU mapping");
    // SAFETY: shared storage sized for one `ScratchParams`.
    unsafe { dst.write(p) };
}

/// A small geometry whose interleaved record is **not** 128-aligned, so the
/// device tests exercise `align_up` rather than the identity function.
///
/// `(8 + 2) * 4` = 40 B, padded to 128. #70's warning, applied to the fixture
/// rather than only to the arithmetic tests: a device test built from
/// LFM2-shaped sizes would be blind in the same way its acceptance test was.
fn small() -> PartialsGeometry {
    PartialsGeometry {
        n_heads: 4,
        head_dim: 8,
        page_size: 256,
        batch: 1,
    }
}

/// Whether the barrier instrument is live, and a loud skip if it is not.
///
/// `trace::record_barrier` is gated on `CANDLE_METAL_TRACE`, read **once**
/// through a `OnceLock`. That is the shape §9.2f records as having produced a
/// vacuous determinism run in #69: the harness consumed the `OnceLock` guarding
/// an environment switch, both arms ran the default, and the "changed" arm
/// reported a passing result for the unchanged path.
///
/// So the two barrier tests below do **not** trust the flag. They ask whether
/// the instrument recorded anything at all, and if it did not they say so rather
/// than asserting `0 == 0` and reporting green. §2.4: *an instrument that cannot
/// be shown to have engaged has not measured anything.*
///
/// Run them with `CANDLE_METAL_TRACE=1`; without it they are skipped, visibly.
fn barrier_instrument_live() -> bool {
    if crate::kernels::metal::trace::trace_requested() {
        return true;
    }
    eprintln!(
        "SKIPPED: CANDLE_METAL_TRACE is not set, so auto_barrier's emissions are not \n\
         observable and this test would assert 0 == 0. Re-run with CANDLE_METAL_TRACE=1."
    );
    false
}

/// **The compiled kernel's constants agree with the Rust side's.**
///
/// §11.3d: a `static_assert` proves only that one side agrees with itself, and
/// the failure being guarded is the two sides disagreeing. #38 found a real
/// width mismatch this way (`usize` bound to a `constant uint &`).
///
/// The last pair is the interesting one: the unpadded record against the padded
/// stride. Their difference **is** what `align_up` does, on a shape that is not
/// a 128-multiple, reported by the kernel that will address it.
#[test]
fn scratch_reports_its_constants() {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    let g = PartialsGeometry::default(); // LFM2: record 264 -> stride 384

    let out = device.new_buffer(5 * 4, SHARED).unwrap();
    let params = device
        .new_buffer(std::mem::size_of::<ScratchParams>(), SHARED)
        .unwrap();
    write_params(
        &params,
        ScratchParams {
            n_heads: g.n_heads as u32,
            head_dim: g.head_dim as u32,
            live_chunks: 1,
            sized_chunks: 1,
            interleaved: 1,
            seed: 1,
        },
    );

    {
        let guard = cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe { call_scratch_report(&device, &guard, &kernels, &out, &params) }.unwrap();
    }
    cmds.wait_until_completed().unwrap();

    let got = read_u32(&out, 5);
    assert_eq!(
        got[0] as usize,
        super::ARENA_ALIGNMENT,
        "the kernel's alignment constant disagrees with ARENA_ALIGNMENT"
    );
    assert_eq!(
        got[1] as usize, PARTIAL_STATS,
        "the kernel's stats count disagrees with PARTIAL_STATS"
    );
    assert_eq!(
        got[2] as usize,
        std::mem::size_of::<ScratchParams>(),
        "sizeof(ScratchParams) differs across the boundary"
    );
    // 264 unpadded, 384 padded. Both reported, so the padding is visible rather
    // than inferred -- and 264 is the number that makes this fixture able to
    // fail at all.
    assert_eq!(got[4], 264, "LFM2's interleaved record is not 264 B");
    assert_eq!(got[3], 384, "the padded stride is not 3 cache lines");
    assert_ne!(
        got[4] as usize % super::ARENA_ALIGNMENT,
        0,
        "the reported record is 128-aligned, so this test cannot see a bad align_up"
    );
}

/// Every declared `[[host_name]]` loads from the compiled library.
///
/// §8.1b's checked registry. Catches a rename on either side and a policy added
/// to the Rust list but not instantiated in Metal -- the absent-variant class
/// that has now fired four times (§11.3h).
#[test]
fn scratch_names_resolve() {
    let device = device();
    let kernels = Kernels::new();
    for (_, name) in ScratchKernel::PARTIALS
        .iter()
        .chain(ScratchKernel::COMBINE.iter())
    {
        kernels
            .load_pipeline(&device, Source::Scratch, *name)
            .unwrap_or_else(|e| panic!("{name} does not resolve: {e:?}"));
    }
    kernels
        .load_pipeline(&device, Source::Scratch, "scratch_report")
        .expect("scratch_report does not resolve");
}

/// Each declared name equals `<stem>_<policy suffix>`.
///
/// A separate test from resolution because they catch different things. §8.1b's
/// `max_pool2d` -> `avg_pool2d` case: a row pairing one family's suffix with
/// **another family's valid name** resolves fine and would silently dispatch the
/// wrong kernel. Resolution alone cannot see it; #64 reproduced the same case in
/// `indexing` and the consequence there was `scatter` accumulating instead of
/// assigning.
///
/// Here the wrong pairing would run a *different sizing policy* than the one
/// selected -- so the A/B would compare a policy against itself and report a
/// null, which is the most expensive shape of wrong answer this issue could
/// produce.
#[test]
fn scratch_names_match_their_stem_and_policy() {
    for (stem, family) in ScratchKernel::STEMS {
        for (sizing, name) in family {
            assert_eq!(
                *name,
                format!("{stem}_{}", sizing.suffix()),
                "{name} is not {stem} + {:?}'s suffix",
                sizing
            );
        }
    }
}

/// Run the partials stub and the combine for one policy, returning
/// `(combined, order_walked)`.
fn run_layer(
    sizing: Sizing,
    g: &PartialsGeometry,
    kv_len: usize,
    layout: ScratchLayout,
    max_context: usize,
    seed: u32,
) -> (Vec<f32>, Vec<u32>) {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);

    let plan = plan_scratch(g, kv_len, 1, sizing, layout, max_context).expect("plan");
    plan.check_index_ordered()
        .expect("plan is not index-ordered");
    let live = plan.live_chunks();
    let sized = plan.regions()[0].chunks;

    let partials = device
        .new_buffer(plan.bump_capacity().max(1), SHARED)
        .unwrap();
    let out = device
        .new_buffer(g.n_heads * g.head_dim * 4, SHARED)
        .unwrap();
    let order = device
        .new_buffer((g.n_heads * live.max(1)) * 4, SHARED)
        .unwrap();
    let params = device
        .new_buffer(std::mem::size_of::<ScratchParams>(), SHARED)
        .unwrap();
    write_params(
        &params,
        ScratchParams {
            n_heads: g.n_heads as u32,
            head_dim: g.head_dim as u32,
            live_chunks: live as u32,
            sized_chunks: sized as u32,
            interleaved: u32::from(layout == ScratchLayout::Interleaved),
            seed,
        },
    );

    {
        let guard = cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_scratch_partials(
                &device,
                &guard,
                &kernels,
                sizing,
                &partials,
                &params,
                g.n_heads as u32,
                live as u32,
            )
        }
        .unwrap();
        // One barrier before the combine, and only one -- §9.4. It falls out of
        // `auto_barrier`: the combine binds `partials` as an input where the
        // stub bound it as an output, which is the read-after-write the hazard
        // tracker is looking for.
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_scratch_combine(
                &device,
                &guard,
                &kernels,
                sizing,
                &partials,
                &out,
                &params,
                &order,
                g.n_heads as u32,
            )
        }
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();

    (
        read_f32(&out, g.n_heads * g.head_dim),
        read_u32(&order, g.n_heads * live.max(1)),
    )
}

/// **The combine walked ascending chunk indices**, per the kernel's own record.
///
/// §10.4 calls a completion-ordered merge here the single most likely place for
/// nondeterminism to enter the whole design. Asserting the source says index
/// order proves nothing about what ran; the kernel writes the index it consumed
/// at each step and this compares against that.
#[test]
fn the_combine_walks_chunks_in_ascending_index_order() {
    let g = small();
    let kv = 2_000; // 8 chunks at page 256
    for sizing in Sizing::ALL {
        let (_, order) = run_layer(sizing, &g, kv, ScratchLayout::Planes, 131_072, 7);
        let live = g.chunks(kv);
        assert_eq!(order.len(), g.n_heads * live);
        for head in 0..g.n_heads {
            let walked: Vec<u32> = order[head * live..(head + 1) * live].to_vec();
            let expected: Vec<u32> = (0..live as u32).collect();
            assert_eq!(
                walked, expected,
                "{sizing:?} head {head} merged out of index order: {walked:?}"
            );
        }
    }
}

/// **The merge is bit-stable across runs**, which is what index order buys.
///
/// §2.3.6: a single comparison is a weak test, so this runs the same merge eight
/// times and requires one distinct result. A completion-ordered merge would
/// pass this sometimes -- which is exactly why §10.4 says the symptom is
/// indistinguishable from a missing fence and why the order assertion above is
/// the primary evidence and this is the corroboration.
#[test]
fn the_merge_is_bit_stable_across_runs() {
    let g = small();
    let mut seen: Vec<Vec<u32>> = Vec::new();
    for _ in 0..8 {
        let (out, _) = run_layer(
            Sizing::Reserve,
            &g,
            2_000,
            ScratchLayout::Planes,
            131_072,
            11,
        );
        // Compare bits, not values: two f32 that print the same can differ.
        seen.push(out.iter().map(|f| f.to_bits()).collect());
    }
    let first = &seen[0];
    for (i, s) in seen.iter().enumerate() {
        assert_eq!(s, first, "run {i} produced different bits");
    }
    // And the output is not degenerate, or the stability above is vacuous --
    // §15.1 #1 and #53: two kernels that both write nothing agree perfectly.
    let distinct: std::collections::BTreeSet<u32> = first.iter().copied().collect();
    assert!(
        distinct.len() > 4,
        "the combine wrote {} distinct values across {} elements -- the comparison is vacuous",
        distinct.len(),
        first.len()
    );
}

/// **The three policies compute identical bits.**
///
/// This is the acceptance bar the issue sets: *LFM2 digests unchanged under
/// every policy -- a policy that changes numerics is a bug, not a tradeoff.*
/// At this level the equivalent statement is that the policies differ in what
/// they reserve and in nothing else, so their outputs agree bit for bit.
///
/// If they ever disagree, the sizing axis has leaked into the addressing and it
/// is no longer an A/B: two arms computing different things cannot be compared
/// on time.
#[test]
fn every_policy_computes_the_same_bits() {
    let g = small();
    for layout in [ScratchLayout::Planes, ScratchLayout::Interleaved] {
        let mut arms: Vec<(Sizing, Vec<u32>)> = Vec::new();
        for sizing in Sizing::ALL {
            let (out, _) = run_layer(sizing, &g, 2_000, layout, 131_072, 23);
            arms.push((sizing, out.iter().map(|f| f.to_bits()).collect()));
        }
        let (_, first) = &arms[0];
        for (sizing, bits) in &arms[1..] {
            assert_eq!(
                bits, first,
                "{sizing:?} computes different bits from {:?} under {layout:?}",
                arms[0].0
            );
        }
        let distinct: std::collections::BTreeSet<u32> = first.iter().copied().collect();
        assert!(
            distinct.len() > 4,
            "the comparison is vacuous under {layout:?}"
        );
    }
}

/// **The two layouts compute the same bits too.**
///
/// Interleaved padding changes *where* a record lives, never what it holds. If
/// the two disagree, the padded addressing is wrong -- and under
/// `HazardTrackingModeUntracked` that is a silent wrong answer rather than an
/// error (§3.5). This is §9.3's parity argument applied to the scratch class:
/// an execution comparison, not a size comparison.
#[test]
fn the_two_layouts_compute_the_same_bits() {
    let g = small();
    let (planes, _) = run_layer(Sizing::Grow, &g, 2_000, ScratchLayout::Planes, 131_072, 31);
    let (inter, _) = run_layer(
        Sizing::Grow,
        &g,
        2_000,
        ScratchLayout::Interleaved,
        131_072,
        31,
    );
    let a: Vec<u32> = planes.iter().map(|f| f.to_bits()).collect();
    let b: Vec<u32> = inter.iter().map(|f| f.to_bits()).collect();
    assert_eq!(
        a, b,
        "the interleaved layout computes different bits from the planes layout"
    );
}

/// **N partials write disjoint regions**, checked by execution rather than
/// argued.
///
/// §9.4 says they need no fences between them *because* they are disjoint, and
/// §3.5 says the disjointness is our assertion and not the driver's. So the test
/// runs every chunk's write, then reads every byte of the region back and
/// checks that each (head, chunk) record holds what **its own** dispatch wrote
/// and nothing else.
///
/// The failure this catches is a stride error: an addressing bug that has two
/// chunks land on one record produces a plausible answer, and nothing in the
/// driver reports it.
#[test]
fn scratch_partials_write_disjoint_regions() {
    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    let g = small();
    let kv = 2_000;
    let seed = 5u32;

    let plan = plan_scratch(&g, kv, 1, Sizing::Grow, ScratchLayout::Planes, 131_072).expect("plan");
    let live = plan.live_chunks();
    let bytes = plan.bump_capacity().max(1);

    let partials = device.new_buffer(bytes, SHARED).unwrap();
    // Poison the whole region first, so "untouched" is distinguishable from
    // "written zero" -- the ambiguity §13.1 records for the counter API and
    // §3.7a for all-zero output.
    {
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        let p = unsafe { partials.contents() } as *mut f32;
        for i in 0..bytes / 4 {
            // SAFETY: shared storage of `bytes` bytes.
            unsafe { p.add(i).write(f32::from_bits(0xDEAD_BEEF)) };
        }
    }

    let params = device
        .new_buffer(std::mem::size_of::<ScratchParams>(), SHARED)
        .unwrap();
    write_params(
        &params,
        ScratchParams {
            n_heads: g.n_heads as u32,
            head_dim: g.head_dim as u32,
            live_chunks: live as u32,
            sized_chunks: plan.regions()[0].chunks as u32,
            interleaved: 0,
            seed,
        },
    );

    {
        let guard = cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_scratch_partials(
                &device,
                &guard,
                &kernels,
                Sizing::Grow,
                &partials,
                &params,
                g.n_heads as u32,
                live as u32,
            )
        }
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();

    let got = read_f32(&partials, bytes / 4);
    let stats_off = g.n_heads * live * g.head_dim;

    // Every accumulator slot holds its own (head, chunk, lane) value, so no two
    // dispatches landed on the same bytes.
    let mut touched = vec![false; got.len()];
    for head in 0..g.n_heads {
        for chunk in 0..live {
            let base = (head * live + chunk) * g.head_dim;
            for lane in 0..g.head_dim {
                let want = stub_value(head as u32, chunk as u32, lane as u32, seed);
                assert_eq!(
                    got[base + lane],
                    want,
                    "head {head} chunk {chunk} lane {lane}: another dispatch's value \
                     or an untouched byte"
                );
                touched[base + lane] = true;
            }
            let l_at = stats_off + (head * live + chunk) * PARTIAL_STATS + 1;
            assert_eq!(
                got[l_at],
                1.0 + chunk as f32,
                "l at head {head} chunk {chunk}"
            );
            touched[stats_off + (head * live + chunk) * PARTIAL_STATS] = true;
            touched[l_at] = true;
        }
    }

    // And nothing outside the planned extent was written: the poison survives
    // wherever the plan says nothing lives. That is the other half of
    // disjointness -- not merely that regions do not collide with each other,
    // but that none runs past what was reserved for it.
    let planned = stats_off + g.n_heads * live * PARTIAL_STATS;
    for (i, &t) in touched.iter().enumerate().skip(planned) {
        assert!(!t, "index {i} inside the padding was marked touched");
        assert_eq!(
            got[i].to_bits(),
            0xDEAD_BEEF,
            "a write ran past the planned extent at index {i}"
        );
    }
}

/// The Rust-side mirror of `stub_value` in `scratch.metal`.
///
/// Duplicated deliberately and checked against the kernel by the test above:
/// the point is that the host can predict every byte the kernel writes, which is
/// what makes "each record holds its own dispatch's value" checkable at all. A
/// shared implementation would make the comparison vacuous -- it would check
/// that one function equals itself.
fn stub_value(head: u32, chunk: u32, lane: u32, seed: u32) -> f32 {
    let mixed = head.wrapping_mul(2654435761)
        ^ chunk.wrapping_mul(40503)
        ^ lane.wrapping_mul(2246822519)
        ^ seed;
    ((mixed & 0xFFFF) as i32 - 32768) as f32 / 32768.0
}

/// **The KV append is fenced before the partials read** (§9.4), and this test
/// fails without it.
///
/// # Why this is the easy one to forget
///
/// §9.4 says it in those words: *easy to forget because it is not between the
/// two things one is thinking about.* The two things one is thinking about are
/// the partials and the combine, and the fence between **those** falls out of
/// `auto_barrier` because the combine reads what the partials wrote. The KV
/// append is upstream of both, and if the partials read KV that the append has
/// not finished writing, the answer is wrong intermittently -- §2.3.2's most
/// expensive bug class, and §3.5 says there is no safety net.
///
/// # Mechanistic, not stochastic
///
/// A test that ran the two dispatches and looked for corruption would be
/// sampling a race, and §6.3b records that a flake rate is **not usable as
/// evidence on its own**: the same unfixed binary measured 12/30, then 2/60,
/// then 16/60. So this asserts on the **mechanism** instead -- the barrier
/// `auto_barrier` emits, observed at its emission site by `trace::record_barrier`
/// (§9.2f: observed rather than simulated, because a simulation of this quantity
/// cannot be calibrated).
///
/// The mutation is the whole test: the same two dispatches with the KV buffer
/// **not** declared as an input to the partials emit no barrier, and the
/// assertion catches it.
#[test]
fn the_kv_append_is_fenced_before_the_partials_read() {
    if !barrier_instrument_live() {
        return;
    }

    let g = small();
    let kv = 2_000;
    let live = g.chunks(kv);

    // Runs the append and the partials in one encoder. `declare_kv_input` is
    // the mutation point: with it the partials declare they read what the
    // append wrote, and `auto_barrier` orders them; without it the read is
    // undeclared and nothing does.
    let run = |declare_kv_input: bool| -> usize {
        let device = device();
        let kernels = Kernels::new();
        let cmds = commands(&device);

        let kv_buf = device.new_buffer(4096, SHARED).unwrap();
        let partials = device.new_buffer(64 * 1024, SHARED).unwrap();
        let params = device
            .new_buffer(std::mem::size_of::<ScratchParams>(), SHARED)
            .unwrap();
        write_params(
            &params,
            ScratchParams {
                n_heads: g.n_heads as u32,
                head_dim: g.head_dim as u32,
                live_chunks: live as u32,
                sized_chunks: live as u32,
                interleaved: 0,
                seed: 3,
            },
        );

        // This queue's own trace window: the process-wide trace would also
        // hold whatever the lib's other tests encode meanwhile (§15.1n).
        let window = cmds
            .dispatch_trace()
            .open()
            .expect("a trace window on a fresh queue");
        {
            let guard = cmds.command_encoder().unwrap();

            // Stand-in for the KV append: a dispatch that *writes* the KV
            // buffer. The real one is Phase 5 item 15 (in-place append); what
            // matters here is the hazard it creates, not the arithmetic.
            {
                use crate::kernels::{
                    debug_group, set_params, ComputeCommandEncoder, EncoderProvider, Output,
                };
                let pipeline = kernels
                    .load_pipeline(&device, Source::Scratch, "scratch_partials_grow")
                    .unwrap();
                let gref = &guard;
                let enc = gref.encoder();
                let enc: &ComputeCommandEncoder = enc.as_ref();
                enc.set_compute_pipeline_state(&pipeline);
                debug_group!(enc, "kv_append_stub");
                // SAFETY: the test owns every buffer bound or mapped here and reads its
                // results only after waiting for the command buffer that writes them.
                unsafe {
                    set_params!(enc, (Output::new(&kv_buf), &params));
                }
                enc.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            }

            // The partials. They read KV -- and whether they *say* so is the
            // mutation.
            {
                use crate::kernels::{
                    debug_group, set_params, ComputeCommandEncoder, EncoderProvider, Output,
                };
                let pipeline = kernels
                    .load_pipeline(&device, Source::Scratch, "scratch_partials_grow")
                    .unwrap();
                let gref = &guard;
                let enc = gref.encoder();
                let enc: &ComputeCommandEncoder = enc.as_ref();
                enc.set_compute_pipeline_state(&pipeline);
                debug_group!(enc, "partials_reading_kv");
                // SAFETY: the test owns every buffer bound or mapped here and reads its
                // results only after waiting for the command buffer that writes them.
                unsafe {
                    set_params!(enc, (Output::new(&partials), &params));
                }
                if declare_kv_input {
                    // The declaration §9.4 requires. Index past the kernel's
                    // arguments, which Metal permits: the binding exists for its
                    // ordering effect, exactly as `call_arena_reset`'s arena
                    // binding does (#70).
                    // SAFETY: the test owns every buffer bound or mapped here and reads its
                    // results only after waiting for the command buffer that writes them.
                    unsafe { enc.set_input_buffer(4, Some(&kv_buf), 0) };
                }
                enc.dispatch_thread_groups(
                    objc2_metal::MTLSize {
                        width: g.n_heads,
                        height: live,
                        depth: 1,
                    },
                    objc2_metal::MTLSize {
                        width: 32,
                        height: 1,
                        depth: 1,
                    },
                );
            }
        }
        cmds.wait_until_completed().unwrap();
        window.close().iter().filter(|d| d.barrier).count()
    };

    let fenced = run(true);
    let unfenced = run(false);

    assert_eq!(
        fenced, 1,
        "the KV append was not fenced before the partials read: expected exactly one \
         barrier (DESIGN.md 9.4), observed {fenced}"
    );
    // The mutation. Without this the assertion above would pass on a build that
    // emits a barrier for some unrelated reason, and the test would be evidence
    // of nothing -- §11.3j's control arm, in a barrier count.
    assert_eq!(
        unfenced, 0,
        "the unfenced arm emitted {unfenced} barriers, so the fenced arm's count is \
         not evidence that the declaration caused it"
    );
}

/// **N disjoint partials need no fences between them, and one before the
/// combine** -- the fence budget §9.4 states, observed.
///
/// §9.4's honest count is **2 per attention layer per decode step**: one after
/// the KV append, one before the combine. The partials themselves are one
/// dispatch over a `(heads, chunks)` grid precisely because they are disjoint;
/// if they were chained they would need one per tile.
///
/// This measures the second of the two. The first is the test above.
#[test]
fn the_partials_need_one_barrier_before_the_combine_and_none_between() {
    if !barrier_instrument_live() {
        return;
    }

    let device = device();
    let kernels = Kernels::new();
    let cmds = commands(&device);
    let g = small();
    let kv = 2_000;
    let live = g.chunks(kv);

    let plan = plan_scratch(&g, kv, 1, Sizing::Grow, ScratchLayout::Planes, 131_072).expect("plan");
    let partials = device
        .new_buffer(plan.bump_capacity().max(1), SHARED)
        .unwrap();
    let out = device
        .new_buffer(g.n_heads * g.head_dim * 4, SHARED)
        .unwrap();
    let order = device.new_buffer(g.n_heads * live * 4, SHARED).unwrap();
    let params = device
        .new_buffer(std::mem::size_of::<ScratchParams>(), SHARED)
        .unwrap();
    write_params(
        &params,
        ScratchParams {
            n_heads: g.n_heads as u32,
            head_dim: g.head_dim as u32,
            live_chunks: live as u32,
            sized_chunks: plan.regions()[0].chunks as u32,
            interleaved: 0,
            seed: 13,
        },
    );

    let window = cmds
        .dispatch_trace()
        .open()
        .expect("a trace window on a fresh queue");
    {
        let guard = cmds.command_encoder().unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_scratch_partials(
                &device,
                &guard,
                &kernels,
                Sizing::Grow,
                &partials,
                &params,
                g.n_heads as u32,
                live as u32,
            )
        }
        .unwrap();
        // SAFETY: the test owns every buffer bound or mapped here and reads its
        // results only after waiting for the command buffer that writes them.
        unsafe {
            call_scratch_combine(
                &device,
                &guard,
                &kernels,
                Sizing::Grow,
                &partials,
                &out,
                &params,
                &order,
                g.n_heads as u32,
            )
        }
        .unwrap();
    }
    cmds.wait_until_completed().unwrap();
    let dispatches = window.close();

    assert_eq!(dispatches.len(), 2, "expected two dispatches");
    assert!(
        !dispatches[0].barrier,
        "a barrier was emitted before the partials, which have nothing to order against"
    );
    assert!(
        dispatches[1].barrier,
        "no barrier before the combine: it reads what the partials wrote (DESIGN.md 9.4)"
    );
    let total = dispatches.iter().filter(|d| d.barrier).count();
    assert_eq!(total, 1, "expected exactly one barrier, observed {total}");
}
