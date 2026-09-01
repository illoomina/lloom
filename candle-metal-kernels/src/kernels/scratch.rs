//! Dispatching the FlashDecoding partials stub and its combine (issue #71).
//!
//! Three entry points and a registry. The registry is the part that matters:
//! §8.1b's argument is that a name resolved against **the compiled metallib the
//! GPU will be asked for** is a strictly stronger oracle than two lists
//! generated from one source, because it also proves the Metal side compiles
//! that name. It has caught an absent-variant defect four times now (§11.3h),
//! once only visible across a crate boundary.
//!
//! **This does not implement FlashDecoding** -- see `scratch.metal`'s header.
//! It writes the partials in the right shape and merges them in the order §10.4
//! requires, so the memory behaviour can be tested before the arithmetic exists.
//!
//! # The arithmetic now exists, and what that leaves this file doing (#526)
//!
//! **The real kernel superseded most of this and nothing told it.** These
//! entry points read **zero production callers** on #525's caller census, and
//! the reason is not that a wiring was forgotten:
//!
//! - **The sizing axis is wired, to the real kernel.** §7.1b's prerequisite --
//!   *"connect `ScratchSizing` to FlashDecoding's allocation"* -- was
//!   discharged by **#234**. `candle-nn/src/ops.rs` calls
//!   [`Sizing::sized_chunks`](crate::metal::scratch::Sizing::sized_chunks),
//!   the same function [`plan_scratch`](crate::plan_scratch) calls, so the two
//!   cannot disagree about what an arm means. §9.1f's three arms were measured
//!   through that seam.
//! - **The addressing distinction is wired too, as a parameter.**
//!   `SIZING::fixed_offsets` selects between the sized and the live chunk
//!   count for the write stride; `flash_decoding.metal`'s
//!   `FlashPartialParams::chunk_capacity` is that same choice made at run time
//!   -- *"the partial stride must be the reserved count"*.
//! - **The properties are re-pinned on the real path.**
//!   `flash_reserving_capacity_is_bit_identical_to_grow`,
//!   `combine_walks_chunks_in_ascending_index_order` and
//!   `flash_is_bit_stable_across_runs` assert on the real kernel what
//!   `every_policy_computes_the_same_bits`, `the_combine_walks_chunks_in_...`
//!   and `the_merge_is_bit_stable_across_runs` assert here.
//!
//! **What is NOT duplicated is the layout axis, and that is why this stays.**
//! [`ScratchLayout::Interleaved`](crate::ScratchLayout) has no consumer outside
//! this stub: `flash_decoding.metal` is hardcoded to the **planes** layout and
//! takes no layout parameter. §9.1d's **+45.5 % per record** for the
//! interleaved arm was measured *here*, and this stub is the only instrument
//! that can measure it -- so retiring the file would delete the evidence for a
//! recorded finding and the means of re-deriving it (§15.2 #11, *keep as
//! evidence*).
//!
//! So: **kept deliberately, and no longer described as a stub awaiting a
//! wiring that already happened.** What would consume it is a FlashDecoding
//! kernel that takes the layout as an axis, which §9.1d's figure is the
//! argument for and which nothing has yet been asked to build.

use crate::metal::scratch::Sizing;
use crate::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, EncoderProvider, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

/// Parameters, matching `ScratchParams` in `scratch.metal` field for field.
///
/// `#[repr(C)]` and six `u32`, so there is no padding on either side and
/// `sizeof` is 24. That agreement is **checked across the boundary** by
/// `scratch_reports_its_constants` rather than asserted on each side: §11.3d
/// records that a `static_assert` proves only that one side agrees with itself,
/// and #40 found the first structs whose `sizeof` is not the sum of their field
/// widths precisely because the check ships the numbers across.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScratchParams {
    /// Query heads. 32 for LFM2, **not** the 8 KV heads -- a partial is
    /// downstream of GQA's register broadcast (§8.3 item 2).
    pub n_heads: u32,
    pub head_dim: u32,
    /// Chunks the current `kv_len` needs.
    pub live_chunks: u32,
    /// Chunks the region is sized for. Equal to `live_chunks` under
    /// [`Sizing::Grow`]; larger under the other two.
    pub sized_chunks: u32,
    /// 0 = separate planes, 1 = one padded record per (head, chunk).
    ///
    /// The interleaved layout's record is 264 B on LFM2's shapes, which is not a
    /// 128-multiple -- the shape #70's warning says our own model cannot
    /// otherwise produce (§9.2c).
    pub interleaved: u32,
    /// Varies the stub's written values so a parity comparison is not vacuous
    /// (§15.1 #1, #53).
    pub seed: u32,
}

/// Every `[[host_name]]` `scratch.metal` instantiates.
///
/// Declared once here and resolved against the compiled library by
/// `scratch_names_resolve`. Two tests, and they catch different things --
/// §8.1b's `max_pool2d` -> `avg_pool2d` case, which resolves fine and would
/// silently dispatch the wrong kernel:
///
/// - resolution: every declared name loads.
/// - spelling: each name equals `<stem>_<policy suffix>`, so a row pairing one
///   policy's suffix with another policy's valid name is caught.
pub struct ScratchKernel;

impl ScratchKernel {
    /// The partials stub, per policy.
    pub const PARTIALS: [(Sizing, &'static str); 3] = [
        (Sizing::Reserve, "scratch_partials_reserve"),
        (Sizing::Grow, "scratch_partials_grow"),
        (Sizing::Bucket, "scratch_partials_bucket"),
    ];

    /// The combine, per policy.
    pub const COMBINE: [(Sizing, &'static str); 3] = [
        (Sizing::Reserve, "scratch_combine_reserve"),
        (Sizing::Grow, "scratch_combine_grow"),
        (Sizing::Bucket, "scratch_combine_bucket"),
    ];

    /// The stem each family's names are built from, for the spelling test.
    pub const STEMS: [(&'static str, &'static [(Sizing, &'static str)]); 2] = [
        ("scratch_partials", &Self::PARTIALS),
        ("scratch_combine", &Self::COMBINE),
    ];

    fn partials_name(sizing: Sizing) -> &'static str {
        Self::PARTIALS
            .iter()
            .find(|(s, _)| *s == sizing)
            .map(|(_, n)| *n)
            .expect("every Sizing has a partials variant; ALL and PARTIALS are length-checked")
    }

    fn combine_name(sizing: Sizing) -> &'static str {
        Self::COMBINE
            .iter()
            .find(|(s, _)| *s == sizing)
            .map(|(_, n)| *n)
            .expect("every Sizing has a combine variant; ALL and COMBINE are length-checked")
    }
}

/// A threadgroup of 32 -- one simdgroup.
///
/// The unit of divergence (§3.1), and the width the stub's strided writes and
/// the combine's per-lane accumulator are both written against.
const LANES: usize = 32;

/// Write every chunk's partials for one layer.
///
/// # No fences between the chunks
///
/// §9.4: N partials writing **disjoint** output regions need no fences between
/// them, only one before the combine. That is why this is a single dispatch over
/// a `(heads, chunks)` grid rather than one dispatch per chunk with barriers
/// between -- and why the disjointness is checked by execution in
/// `scratch_partials_write_disjoint_regions` rather than argued: §3.5 says
/// disjointness is *our* assertion, not the driver's, and under
/// `HazardTrackingModeUntracked` a wrong one is silent corruption.
#[allow(clippy::too_many_arguments)]
pub fn call_scratch_partials(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    sizing: Sizing,
    out: &Buffer,
    params: &Buffer,
    n_heads: u32,
    live_chunks: u32,
) -> Result<(), MetalKernelError> {
    let name = ScratchKernel::partials_name(sizing);
    let pipeline = kernels.load_pipeline(device, Source::Scratch, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "{name} heads={n_heads} chunks={live_chunks}");
    set_params!(encoder, (Output::new(out), params));
    let grid = MTLSize {
        width: n_heads.max(1) as usize,
        height: live_chunks.max(1) as usize,
        depth: 1,
    };
    let group = MTLSize {
        width: LANES,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}

/// Merge the partials into one output per head, **in ascending chunk index**.
///
/// `order` receives the chunk indices in the order they were actually walked,
/// so a test compares against what ran rather than against the source. §10.4
/// calls a completion-ordered merge here the single most likely place for
/// nondeterminism to enter the design; asserting the order from the kernel's own
/// trace is what makes "merges in index order" a measurement rather than a
/// promise.
///
/// The caller is responsible for the fence between the partials and this: one
/// barrier, per §9.4. `Output::new(out)` against the partials' `Output` gives it
/// through candle's `auto_barrier`, the same mechanism `call_arena_reset` uses
/// for its ordering binding (#70).
#[allow(clippy::too_many_arguments)]
pub fn call_scratch_combine(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    sizing: Sizing,
    partials: &Buffer,
    out: &Buffer,
    params: &Buffer,
    order: &Buffer,
    n_heads: u32,
) -> Result<(), MetalKernelError> {
    let name = ScratchKernel::combine_name(sizing);
    let pipeline = kernels.load_pipeline(device, Source::Scratch, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "{name} heads={n_heads}");
    set_params!(
        encoder,
        (partials, Output::new(out), params, Output::new(order))
    );
    let grid = MTLSize {
        width: n_heads.max(1) as usize,
        height: 1,
        depth: 1,
    };
    let group = MTLSize {
        width: LANES,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, group);
    Ok(())
}

/// Ask the compiled kernel what constants and layout it was built with.
///
/// The cross-boundary check §11.3d argues for. Writes five `u32`: alignment,
/// stats count, `sizeof(ScratchParams)`, the padded interleaved record stride,
/// and the unpadded record size -- the last two being the pair whose difference
/// is what `align_up` does on a shape LFM2 can actually produce.
pub fn call_scratch_report(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    out: &Buffer,
    params: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Scratch, "scratch_report")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "scratch_report");
    set_params!(encoder, (Output::new(out), params));
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(one, one);
    Ok(())
}
