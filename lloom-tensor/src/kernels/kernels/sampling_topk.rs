//! The device sampler's dispatch: the selection, the multinomial draw and
//! the filtered Gumbel draw, each a short pipeline of small dispatches so a
//! row's work covers every core (dispatch §11.2j). Every stage reads and
//! writes a row-local scratch block (`sampling_work_words`) and a per-row id
//! scratch of `vocab` words; the kernels' layout constants and the ones
//! here are one set (the draw's compact capacity, 1024, is the kernels'
//! alone).
use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

/// Ids per histogram tile.
pub(crate) const SAMPLING_TILE: usize = 4096;
/// Histogram bins: the key's top eleven bits.
pub(crate) const SAMPLING_BINS: usize = 2048;
/// Scratch words per row ahead of the tile and histogram regions.
pub(crate) const SAMPLING_WORDS: usize = 64;
/// Threads per threadgroup; the kernels' threadgroup arrays are sized to it.
const SAMPLING_WIDTH: usize = 256;

pub(crate) fn sampling_tiles(vocab: usize) -> usize {
    vocab.div_ceil(SAMPLING_TILE)
}

/// Scratch words for `rows` rows of `vocab`: per row the words, four per
/// tile, the per-tile histograms, the reduced histogram, the 1024 lane sums
/// and two per tile for the Gumbel tiles' best keys.
pub(crate) fn sampling_work_words(rows: usize, vocab: usize) -> Option<usize> {
    let tiles = sampling_tiles(vocab);
    let per_row = SAMPLING_WORDS
        .checked_add(tiles.checked_mul(4)?)?
        .checked_add(tiles.checked_mul(SAMPLING_BINS)?)?
        .checked_add(SAMPLING_BINS)?
        .checked_add(1024)?
        .checked_add(tiles.checked_mul(2)?)?;
    rows.checked_mul(per_row)
}

struct Stages<'a> {
    device: &'a Device,
    kernels: &'a Kernels,
    rows: usize,
    vocab: u32,
}

impl Stages<'_> {
    fn pipeline(
        &self,
        name: &'static str,
    ) -> Result<(crate::kernels::ComputePipeline, usize), MetalKernelError> {
        use objc2_metal::{MTLComputePipelineState as _, MTLDevice as _};
        let pipeline = self
            .kernels
            .load_pipeline(self.device, Source::Random, name)?;
        let width = pipeline
            .max_total_threads_per_threadgroup()
            .min(SAMPLING_WIDTH);
        if width == 0
            || pipeline.as_ref().staticThreadgroupMemoryLength()
                > self.device.as_ref().maxThreadgroupMemoryLength()
        {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "{name}: native threadgroup limits exceeded"
            )));
        }
        Ok((pipeline, width))
    }

    /// Dispatch `name` over `groups_per_row` threadgroups per row, or over
    /// `vocab` threads per row when `groups_per_row` is `None`.
    fn dispatch(
        &self,
        encoder: &ComputeCommandEncoder,
        name: &'static str,
        groups_per_row: Option<usize>,
        bind: impl FnOnce(&ComputeCommandEncoder),
    ) -> Result<(), MetalKernelError> {
        let (pipeline, width) = self.pipeline(name)?;
        encoder.set_compute_pipeline_state(&pipeline);
        bind(encoder);
        let groups = groups_per_row.unwrap_or_else(|| (self.vocab as usize).div_ceil(width));
        encoder.dispatch_thread_groups(
            MTLSize {
                width: groups,
                height: self.rows,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }
}

/// Encode the selection over contiguous processed f32 rows: the status words
/// (code, first bad id, finite count, |K|) and, in `work`, K's cutoff key
/// and the compact set the draw reads. `filter_p` asks for the nucleus's
/// compact set when k is off.
///
/// # Safety
/// `vocab` is positive. `scores` has a four-byte-aligned offset and covers
/// `rows * vocab` processed f32 values at that offset; `status` covers
/// `4 * rows` u32 values, `work` `sampling_work_words(rows, vocab)` and
/// `scratch_ids` `rows * vocab`. The outputs are disjoint from each other and
/// the input. All buffers retain allocation reuse authority through GPU
/// completion and any captured replay. Byte spans fit the native address space.
#[allow(
    clippy::too_many_arguments,
    reason = "private dispatch mirrors the kernel's buffers and scalar bindings"
)]
pub(crate) unsafe fn call_sampling_select(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    scores: (&Buffer, usize),
    rows: u32,
    vocab: u32,
    k: u32,
    filter_p: u32,
    work: &Buffer,
    scratch_ids: &Buffer,
    status: &Buffer,
) -> Result<(), MetalKernelError> {
    if rows == 0 {
        return Ok(());
    }
    let stages = Stages {
        device,
        kernels,
        rows: rows as usize,
        vocab,
    };
    let tiles = sampling_tiles(vocab as usize);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    debug_group!(encoder, "sampling_select rows={rows} vocab={vocab} k={k}");
    // SAFETY: the caller provides the disjoint, sized, retained allocations above.
    unsafe {
        stages.dispatch(
            encoder,
            "sampling_select_hist_f32",
            Some(tiles),
            |encoder| {
                set_params!(encoder, (scores, vocab, Output::new(work)));
            },
        )?;
        stages.dispatch(
            encoder,
            "sampling_select_reduce_f32",
            Some(SAMPLING_BINS.div_ceil(SAMPLING_WIDTH)),
            |encoder| {
                set_params!(encoder, (vocab, Output::new(work)));
            },
        )?;
        stages.dispatch(encoder, "sampling_select_scan_f32", Some(1), |encoder| {
            set_params!(
                encoder,
                (
                    scores,
                    vocab,
                    k,
                    filter_p,
                    Output::new(work),
                    Output::new(status)
                )
            );
        })?;
        stages.dispatch(
            encoder,
            "sampling_select_compact_f32",
            Some(tiles),
            |encoder| {
                set_params!(
                    encoder,
                    (
                        scores,
                        vocab,
                        Output::new(work),
                        Output::new(work),
                        Output::new(scratch_ids)
                    )
                );
            },
        )?;
        stages.dispatch(encoder, "sampling_select_rank_f32", Some(1), |encoder| {
            set_params!(encoder, (scores, vocab, Output::new(work), scratch_ids));
        })?;
    }
    Ok(())
}

/// Encode K as a dense u32 mask from a completed selection.
///
/// # Safety
/// As [`call_sampling_select`], on the same `scores` and `work`; `mask`
/// covers `rows * vocab` u32 values.
#[allow(clippy::too_many_arguments, reason = "private kernel ABI")]
pub(crate) unsafe fn call_sampling_select_mask(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    scores: (&Buffer, usize),
    rows: u32,
    vocab: u32,
    work: &Buffer,
    mask: &Buffer,
) -> Result<(), MetalKernelError> {
    if rows == 0 {
        return Ok(());
    }
    let stages = Stages {
        device,
        kernels,
        rows: rows as usize,
        vocab,
    };
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    // SAFETY: the caller provides the disjoint, sized, retained allocations above.
    unsafe {
        stages.dispatch(encoder, "sampling_select_mask_f32", None, |encoder| {
            set_params!(encoder, (scores, vocab, work, Output::new(mask)));
        })
    }
}

/// The selection as a dense mask: the select stages, then the mask.
///
/// # Safety
/// As [`call_sampling_select`] and [`call_sampling_select_mask`].
#[allow(clippy::too_many_arguments, reason = "private kernel ABI")]
pub(crate) unsafe fn call_sampling_topk_mask(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    scores: (&Buffer, usize),
    rows: u32,
    vocab: u32,
    k: u32,
    mask: &Buffer,
    status: &Buffer,
    work: &Buffer,
    scratch_ids: &Buffer,
) -> Result<(), MetalKernelError> {
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    // SAFETY: the caller's contract is this function's.
    unsafe {
        call_sampling_select(
            device,
            encoder,
            kernels,
            scores,
            rows,
            vocab,
            k,
            0,
            work,
            scratch_ids,
            status,
        )?;
        call_sampling_select_mask(device, encoder, kernels, scores, rows, vocab, work, mask)
    }
}

/// The weights of K and the contract's lane sums, shared by both draws.
#[allow(clippy::too_many_arguments, reason = "private kernel ABI")]
unsafe fn encode_weights(
    stages: &Stages<'_>,
    encoder: &ComputeCommandEncoder,
    scores: (&Buffer, usize),
    inv_t: f32,
    gumbel: u32,
    noise: (&Buffer, usize),
    work: &Buffer,
    scratch_ids: &Buffer,
    weights: &Buffer,
) -> Result<(), MetalKernelError> {
    let vocab = stages.vocab;
    // SAFETY: the caller provides the disjoint, sized, retained allocations.
    unsafe {
        stages.dispatch(encoder, "sampling_weights_f32", None, |encoder| {
            set_params!(
                encoder,
                (
                    scores,
                    vocab,
                    inv_t.to_bits(),
                    gumbel,
                    noise,
                    Output::new(work),
                    Output::new(work),
                    Output::new(scratch_ids),
                    Output::new(weights)
                )
            );
        })?;
        stages.dispatch(
            encoder,
            "sampling_lanes_f32",
            Some(1024usize.div_ceil(SAMPLING_WIDTH)),
            |encoder| {
                set_params!(encoder, (vocab, Output::new(work), weights));
            },
        )
    }
}

/// Encode the private contract draw after the selection.
///
/// # Safety
/// Positive `vocab`, finite nonnegative `inv_t`, `p` in [0, 1], `filter_p`
/// either 0 or 1. Scores and uniforms have aligned offsets and validated
/// spans of `rows * vocab` f32 and `rows` f32 respectively. `work` and
/// `scratch_ids` hold this score row's unchanged selection. Weights and
/// support cover `rows * vocab` f32/u32, ids `rows` u32, status `4 * rows`
/// u32. Outputs/scratch are mutually disjoint and disjoint from inputs.
/// All byte products/spans fit usize. Every buffer retains original
/// allocation reuse authority through GPU completion and future replay.
#[allow(clippy::too_many_arguments, reason = "private kernel ABI")]
pub(crate) unsafe fn call_sampling_multinomial(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    scores: (&Buffer, usize),
    rows: u32,
    vocab: u32,
    inv_t: f32,
    p: f32,
    filter_p: u32,
    uniforms: (&Buffer, usize),
    work: &Buffer,
    scratch_ids: &Buffer,
    weights: &Buffer,
    support: &Buffer,
    ids: &Buffer,
    status: &Buffer,
) -> Result<(), MetalKernelError> {
    if rows == 0 {
        return Ok(());
    }
    let stages = Stages {
        device,
        kernels,
        rows: rows as usize,
        vocab,
    };
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    debug_group!(encoder, "sampling_multinomial rows={rows} vocab={vocab}");
    // SAFETY: caller supplies the sized, disjoint, retained allocations above.
    unsafe {
        encode_weights(
            &stages,
            encoder,
            scores,
            inv_t,
            0,
            uniforms,
            work,
            scratch_ids,
            weights,
        )?;
        stages.dispatch(encoder, "sampling_draw_f32", Some(1), |encoder| {
            set_params!(
                encoder,
                (
                    scores,
                    vocab,
                    p.to_bits(),
                    filter_p,
                    uniforms,
                    Output::new(work),
                    scratch_ids,
                    weights,
                    Output::new(ids),
                    Output::new(status)
                )
            );
        })?;
        stages.dispatch(encoder, "sampling_support_f32", None, |encoder| {
            set_params!(
                encoder,
                (
                    scores,
                    vocab,
                    work,
                    Output::new(support),
                    Output::new(weights)
                )
            );
        })
    }
}

/// Encode the private centered-f32 Gumbel draw after the selection.
///
/// # Safety
/// As [`call_sampling_multinomial`], with `variates` an aligned, validated
/// span of `rows * vocab` f32 in original token-id order.
#[allow(clippy::too_many_arguments, reason = "private kernel ABI")]
pub(crate) unsafe fn call_sampling_filtered_gumbel_f32(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    scores: (&Buffer, usize),
    rows: u32,
    vocab: u32,
    inv_t: f32,
    p: f32,
    filter_p: u32,
    variates: (&Buffer, usize),
    work: &Buffer,
    scratch_ids: &Buffer,
    weights: &Buffer,
    support: &Buffer,
    ids: &Buffer,
    status: &Buffer,
) -> Result<(), MetalKernelError> {
    if rows == 0 {
        return Ok(());
    }
    let stages = Stages {
        device,
        kernels,
        rows: rows as usize,
        vocab,
    };
    let tiles = sampling_tiles(vocab as usize);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    debug_group!(
        encoder,
        "sampling_filtered_gumbel_f32 rows={rows} vocab={vocab}"
    );
    // SAFETY: caller supplies the sized, disjoint, retained allocations above.
    unsafe {
        encode_weights(
            &stages,
            encoder,
            scores,
            inv_t,
            1,
            variates,
            work,
            scratch_ids,
            weights,
        )?;
        stages.dispatch(encoder, "sampling_gumbel_support_f32", Some(1), |encoder| {
            set_params!(
                encoder,
                (
                    scores,
                    vocab,
                    p.to_bits(),
                    filter_p,
                    Output::new(work),
                    scratch_ids,
                    weights
                )
            );
        })?;
        stages.dispatch(
            encoder,
            "sampling_gumbel_tile_f32",
            Some(tiles),
            |encoder| {
                set_params!(
                    encoder,
                    (
                        scores,
                        vocab,
                        inv_t.to_bits(),
                        variates,
                        Output::new(work),
                        Output::new(weights),
                        Output::new(support)
                    )
                );
            },
        )?;
        stages.dispatch(encoder, "sampling_gumbel_pick_f32", Some(1), |encoder| {
            set_params!(
                encoder,
                (vocab, work, Output::new(ids), Output::new(status))
            );
        })
    }
}

/// The dispatch labels of each pipeline, for census assertions.
#[cfg(test)]
pub(crate) const SELECT_KERNELS: &[&str] = &[
    "sampling_select_hist_f32",
    "sampling_select_reduce_f32",
    "sampling_select_scan_f32",
    "sampling_select_compact_f32",
    "sampling_select_rank_f32",
];
#[cfg(test)]
pub(crate) const MULTINOMIAL_KERNELS: &[&str] = &[
    "sampling_weights_f32",
    "sampling_lanes_f32",
    "sampling_draw_f32",
    "sampling_support_f32",
];
#[cfg(test)]
pub(crate) const GUMBEL_KERNELS: &[&str] = &[
    "sampling_weights_f32",
    "sampling_lanes_f32",
    "sampling_gumbel_support_f32",
    "sampling_gumbel_tile_f32",
    "sampling_gumbel_pick_f32",
];
