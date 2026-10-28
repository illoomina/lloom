use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};

/// Encode the private radix candidate-mask arm over contiguous processed f32 rows.
///
/// # Safety
/// `vocab` is positive. `scores` has a four-byte-aligned offset and covers
/// `rows * vocab` processed f32 values at that offset; `mask` covers
/// as many u32 values and `status` covers `4 * rows` u32 values. The outputs
/// are disjoint from each other and the input. All buffers retain allocation
/// reuse authority through GPU completion and any captured replay.
/// The native pipeline supports a positive threadgroup width (clamped here to
/// at most 256) and its static threadgroup-memory requirements on this device.
/// Byte spans and products must fit the native address space without overflow.
#[allow(
    clippy::too_many_arguments,
    reason = "private dispatch mirrors the kernel's buffers and scalar bindings"
)]
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
) -> Result<(), MetalKernelError> {
    if rows == 0 {
        return Ok(());
    }
    let pipeline = kernels.load_pipeline(device, Source::Random, "sampling_topk_mask_f32")?;
    let width = pipeline.max_total_threads_per_threadgroup().min(256);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "sampling_topk_mask rows={rows} vocab={vocab} k={k}"
    );
    // SAFETY: the caller provides the disjoint, sized, retained allocations above.
    unsafe {
        set_params!(
            encoder,
            (scores, vocab, k, Output::new(mask), Output::new(status))
        );
    }
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: rows as usize,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Encode the private contract draw after the existing top-k stage.
///
/// # Safety
/// Positive `vocab`, finite nonnegative `inv_t`, `p` in [0, 1], `filter_p`
/// either 0 or 1. Scores and uniforms have aligned offsets and validated
/// spans of `rows * vocab` f32 and `rows` f32 respectively. `kept` and
/// `selection_status` are this score row's unchanged top-k outputs. Scratch
/// and support cover `rows * vocab` f32/u32, ids `rows` u32, status `4 * rows`
/// u32. Outputs/scratch are mutually disjoint and disjoint from inputs.
/// Scratch is explicitly read/write within this dispatch after its device
/// barrier. All byte products/spans fit usize. Every buffer retains original
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
    kept: &Buffer,
    selection_status: &Buffer,
    weights: &Buffer,
    support: &Buffer,
    ids: &Buffer,
    status: &Buffer,
) -> Result<(), MetalKernelError> {
    use objc2_metal::{MTLComputePipelineState as _, MTLDevice as _};
    if rows == 0 {
        return Ok(());
    }
    let pipeline = kernels.load_pipeline(device, Source::Random, "sampling_multinomial_f32")?;
    let width = pipeline.max_total_threads_per_threadgroup().min(256);
    if width == 0
        || pipeline.as_ref().staticThreadgroupMemoryLength()
            > device.as_ref().maxThreadgroupMemoryLength()
    {
        return Err(MetalKernelError::LoadLibraryError(
            "sampling_multinomial: native threadgroup limits exceeded".into(),
        ));
    }
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "sampling_multinomial rows={rows} vocab={vocab}");
    // SAFETY: caller supplies the sized, disjoint, retained allocations above.
    unsafe {
        set_params!(
            encoder,
            (
                scores,
                vocab,
                inv_t.to_bits(),
                p.to_bits(),
                filter_p,
                uniforms,
                kept,
                selection_status,
                Output::new(weights),
                Output::new(support),
                Output::new(ids),
                Output::new(status)
            )
        );
    }
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: rows as usize,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Encode the private centered-f32 Gumbel candidate using shared preparation.
///
/// # Safety
/// Positive `vocab`, finite nonnegative `inv_t`, `p` in [0, 1], `filter_p`
/// either 0 or 1. Scores and variates have aligned offsets and validated
/// spans of `rows * vocab` f32 each, in original token-id order. `kept` and
/// `selection_status` are this score row's unchanged top-k outputs. Scratch
/// and support cover `rows * vocab` f32/u32, ids `rows` u32, status `4 * rows`
/// u32. Outputs/scratch are mutually disjoint and disjoint from inputs.
/// Scratch is explicitly read/write within this dispatch after its device
/// barrier. All byte products/spans fit usize. Every buffer retains original
/// allocation reuse authority through GPU completion and future replay.
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
    kept: &Buffer,
    selection_status: &Buffer,
    weights: &Buffer,
    support: &Buffer,
    ids: &Buffer,
    status: &Buffer,
) -> Result<(), MetalKernelError> {
    use objc2_metal::{MTLComputePipelineState as _, MTLDevice as _};
    if rows == 0 {
        return Ok(());
    }
    let pipeline = kernels.load_pipeline(device, Source::Random, "sampling_filtered_gumbel_f32")?;
    let width = pipeline.max_total_threads_per_threadgroup().min(256);
    if width == 0
        || pipeline.as_ref().staticThreadgroupMemoryLength()
            > device.as_ref().maxThreadgroupMemoryLength()
    {
        return Err(MetalKernelError::LoadLibraryError(
            "sampling_filtered_gumbel_f32: native threadgroup limits exceeded".into(),
        ));
    }
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "sampling_filtered_gumbel_f32 rows={rows} vocab={vocab}"
    );
    // SAFETY: caller supplies the sized, disjoint, retained allocations above.
    unsafe {
        set_params!(
            encoder,
            (
                scores,
                vocab,
                inv_t.to_bits(),
                p.to_bits(),
                filter_p,
                variates,
                kept,
                selection_status,
                Output::new(weights),
                Output::new(support),
                Output::new(ids),
                Output::new(status)
            )
        );
    }
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: rows as usize,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
