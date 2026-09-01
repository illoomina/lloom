//! Dispatching the GPU-side arena bump allocator (`DESIGN.md` §9.2d, issue #70).
//!
//! Three entry points, and the asymmetry between them is deliberate:
//! [`call_arena_bump`] is the one the install path uses,
//! [`call_arena_bump_concurrent`] exists only so a test can demonstrate why the
//! first one is single-threaded, and [`call_arena_reset`] is the reset whose
//! ordering is the subtle part of the whole issue.
//!
//! # None of the three runs per step, and this file used to say one did (#526)
//!
//! The text here read *"[`call_arena_reset`] is the per-step reset"*, and
//! `metal_backend/device.rs` said *"what **is** per step is the reset"*. **Both
//! were comments and neither was ever true of a shipped configuration.**
//!
//! `install_gpu_arena_offsets` bump-allocates **once at install** (§9.2g,
//! deliberately -- *"doing it once at install rather than per token is what
//! keeps the cost off the per-token path entirely"*), reads the offsets back,
//! and hands them to `Arena::adopt_gpu_offsets`, which verifies them
//! element-wise and copies them into the plan. **Binding then reads the plan.**
//! The [`ArenaCursor`](crate::ArenaCursor) is a local in that function and is
//! dropped when it returns; no cursor outlives install, on the device or
//! anywhere else.
//!
//! So a per-step reset caller does not merely happen to be absent -- **there is
//! no cursor for it to reset.** That is what separates this from
//! `clear_arena` (§9.2m, #489), where a caller genuinely should have existed
//! and the function was wrong in two ways when one finally ran. Here the
//! design moved and the comments did not: this is the **`ArenaOffsets::Gpu`
//! path as built**, not a gap in it.
//!
//! **What would make the reset live** is the GPU-driven token loop (§11.5),
//! where an encoding kernel consumes offsets without a host round-trip. Then
//! the cursor is per-step state, the reset opens each step, and the ordering
//! this file documents becomes load-bearing rather than latent. The kernel and
//! its two ordering tests are kept for that consumer -- §15.2 #11's *keep as
//! evidence*, with the reason now written down rather than implied by a
//! comment that misdescribed it.

use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, EncoderProvider, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

/// One thread, one workgroup: the grid every allocator dispatch uses.
///
/// The allocator is single-threaded by design (see `arena_alloc.metal`), so the
/// grid is a constant rather than a computed split. `linear_split` would size it
/// from the request count and hand the kernel a wider grid than it wants, which
/// the `tid != 0` guard tolerates but which would misrepresent the intent.
fn one_thread() -> (MTLSize, MTLSize) {
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    (one, one)
}

/// Bump-allocate `n` slices in ordinal order.
///
/// `sizes` holds one `u32` byte-size per ordinal, 0 marking an ordinal the
/// arena does not serve; `out_offs` receives one `u32` offset per ordinal, or
/// [`ARENA_DECLINED`]. `capacity` is the arena's byte length, and an allocation
/// that would run past it is declined rather than wrapped -- wrapping would
/// hand out an offset addressing another slot's bytes, which is silent
/// corruption under `HazardTrackingModeUntracked` (§3.5).
#[allow(clippy::too_many_arguments)]
pub fn call_arena_bump(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    cursor: &Buffer,
    sizes: &Buffer,
    out_offs: &Buffer,
    n: u32,
    capacity: u32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::ArenaAlloc, "arena_bump_sequential")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "arena_bump_sequential n={n}");
    // The cursor is bound as an Output: it is read-modify-written, and binding
    // it as an input would let a write-after-read against the previous step's
    // readers go unordered. See `call_arena_reset`.
    set_params!(
        encoder,
        (
            Output::new(cursor),
            sizes,
            Output::new(out_offs),
            n,
            capacity
        )
    );
    let (grid, group) = one_thread();
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}

/// The naive parallel bump allocator. **Not for the decode path.**
///
/// Dispatched only by `concurrent_bump_does_not_fix_an_ordinal_to_an_offset`,
/// which uses it to show that a concurrent `atomic_fetch_add` fixes no mapping
/// from ordinal to offset -- so the sequential form is chosen on evidence
/// rather than on caution. §2.3.2: a nondeterministic layout is
/// indistinguishable from a missing fence, so this must not reach a model.
#[allow(clippy::too_many_arguments)]
pub fn call_arena_bump_concurrent(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    cursor: &Buffer,
    sizes: &Buffer,
    out_offs: &Buffer,
    n: u32,
    capacity: u32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::ArenaAlloc, "arena_bump_concurrent")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "arena_bump_concurrent n={n}");
    set_params!(
        encoder,
        (
            Output::new(cursor),
            sizes,
            Output::new(out_offs),
            n,
            capacity
        )
    );
    let group = MTLSize {
        width: (n as usize).clamp(1, 64),
        height: 1,
        depth: 1,
    };
    let grid = MTLSize {
        width: n.max(1) as usize,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_threads(grid, group);
    Ok(())
}

/// Reset the cursor, opening a new step.
///
/// **Nothing on a model path calls this** -- see the module header. The
/// shipping `ArenaOffsets::Gpu` path bump-allocates once at install and binds
/// through the adopted table, so no cursor survives to be reset. Its consumer
/// is §11.5's GPU-driven loop; what is below is the ordering argument that
/// consumer will need, and it is tested today (#526).
///
/// # The ordering, which is the point of this function
///
/// The reset is a write-after-read against the whole previous step: it declares
/// every arena byte reusable, and the previous step's kernels were reading
/// those bytes. Two mechanisms order it, and both are required because they
/// order different things.
///
/// **Inside the kernel**, `arena_reset_cursor` issues an
/// `atomic_thread_fence(mem_device, relaxed, thread_scope_device)` before the
/// store. Without acquire/release -- which MSL does not have in device space
/// (§9.3) -- a relaxed store carries no ordering of its own, so that fence is
/// the only place the ordering can be expressed at all.
///
/// **Between dispatches**, the fence is not enough, and this is the half that
/// lives here rather than in MSL. Dispatches within one encoder overlap
/// (§3.5, `MTLDispatchType::Concurrent`) -- the GPU does not drain between them
/// -- and a fence inside a kernel orders that kernel's own memory operations,
/// never another dispatch's. So the reset must additionally be separated from
/// the previous step's arena readers by a `memoryBarrierWithScope(Buffers)`.
///
/// That barrier is obtained rather than assumed: `arena` is bound as an
/// **output** here, and candle's `auto_barrier` emits a barrier when a binding
/// conflicts with a previously-bound one (`prev_inputs` for write-after-read).
/// The previous step's dispatches bound the arena as an input, so binding it as
/// an output now is exactly the write-after-read the hazard tracker is looking
/// for, and the barrier falls out of the mechanism candle already runs.
///
/// It is bound and never written, which is unusual enough to state: the binding
/// exists **for its ordering effect**, not for its data. Removing it would leave
/// a kernel that still resets the cursor correctly in isolation and races the
/// previous step in an encoder -- the silent case §9.3 says has no safety net,
/// and `reset_orders_against_the_previous_steps_arena_reads` is what pins it.
pub fn call_arena_reset(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    cursor: &Buffer,
    arena: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::ArenaAlloc, "arena_reset_cursor")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "arena_reset_cursor");
    set_params!(encoder, (Output::new(cursor)));
    // Bound for ordering, not for data -- see the doc comment. The index is
    // past the kernel's declared arguments, which Metal permits, and the kernel
    // never reads it.
    encoder.set_output_buffer(1, Some(arena), 0);
    let (grid, group) = one_thread();
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}

/// Ask the compiled kernel what alignment and sentinel it was built with.
///
/// The cross-boundary check §11.3d argues for: a `static_assert` in MSL proves
/// only that MSL agrees with itself, and the failure being guarded against is
/// the two sides disagreeing.
pub fn call_arena_alloc_report(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::ArenaAlloc, "arena_alloc_report")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "arena_alloc_report");
    set_params!(encoder, (Output::new(out)));
    let (grid, group) = one_thread();
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}
