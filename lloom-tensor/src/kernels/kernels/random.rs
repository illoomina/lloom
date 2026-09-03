use crate::kernels::linear_split;
use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};

#[allow(clippy::too_many_arguments)]
pub fn call_random_uniform(
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

    set_params!(
        encoder,
        (length, min, max, Output::new(seed), Output::new(buffer))
    );

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_random_normal(
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

    set_params!(
        encoder,
        (length, mean, stddev, Output::new(seed), Output::new(buffer))
    );

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
