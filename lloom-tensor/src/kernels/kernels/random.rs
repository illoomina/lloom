use crate::kernels::linear_split;
use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_random_uniform(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    min: f32,
    max: f32,
    length: usize,
    seed: &Buffer,
    buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    if min >= max {
        return Err(MetalKernelError::LoadLibraryError(
            "min must be less than max".to_string(),
        ));
    }
    let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    // One thread per element (lloom #345).
    //
    // This dispatched `length / 2 + odd` threads and had each write two
    // elements from consecutive states of one `HybridTaus` stream, which made
    // the elements of a vector pairwise dependent -- see the comment on
    // `rand_uniform` in `metal_src/random.metal`. Independence within a vector
    // is the property GPU sampling rests on, and it costs one thread per
    // element to have.
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "rand_uniform {name} elems={length}");

    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (length, min, max, Output::new(seed), Output::new(buffer))
        );
    }

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_random_normal(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    mean: f32,
    stddev: f32,
    length: usize,
    seed: &Buffer,
    buffer: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Random, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    // One thread per element, as `call_random_uniform` above (lloom #345).
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "rand_normal {name} elems={length}");

    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (length, mean, stddev, Output::new(seed), Output::new(buffer))
        );
    }

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// The multinomial draw by inverse CDF over `rows` rows of `vocab` f32
/// logits at temperature `1 / inv_t`, one host-supplied uniform per row in
/// `u`, the drawn index per row into `out` (`sample_inverse_cdf_f32`). One
/// threadgroup per row; the width is the kernel's shared arrays' (1024) or
/// the pipeline's maximum, a power of two either way.
#[allow(clippy::too_many_arguments)]
pub fn call_sample_inverse_cdf(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    logits: (&Buffer, usize),
    rows: usize,
    vocab: usize,
    inv_t: f32,
    u: (&Buffer, usize),
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Random, "sample_inverse_cdf_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "sample_inverse_cdf rows={rows} vocab={vocab}");
    set_params!(encoder, (logits, vocab as u32, inv_t, u, Output::new(out)));
    let max = pipeline.max_total_threads_per_threadgroup().min(1024);
    // The scan and the tree assume a power of two.
    let width = if max.is_power_of_two() {
        max
    } else {
        max.next_power_of_two() / 2
    };
    encoder.dispatch_thread_groups(
        objc2_metal::MTLSize {
            width: rows,
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
