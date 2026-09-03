use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, CastParams, CastStridedParams, ParamStyle,
};
use crate::kernels::utils::{BufferOffset, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use crate::kernels::{get_tile_size, linear_split};

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `cast_params_layout_matches_metal` is what proves those
/// mirrors agree with `cast.metal`.
const CAST_PARAMS_ALIGN: usize = core::mem::align_of::<CastParams>();
const CAST_STRIDED_PARAMS_ALIGN: usize = core::mem::align_of::<CastStridedParams>();

#[allow(clippy::too_many_arguments)]
pub fn call_cast_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_cast_contiguous_with(
        device,
        ep,
        kernels,
        kernel_name,
        dtype_size,
        length,
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_cast_contiguous`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_cast_contiguous_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Cast, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "cast {kernel_name} elems={length}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(encoder, (length, &input, Output::new(output)));
    let _staged = finish_packed_params(device, encoder, style, CAST_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_cast_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_cast_strided_with(
        device,
        ep,
        kernels,
        kernel_name,
        shape,
        input,
        input_strides,
        output,
        ParamStyle::default(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn call_cast_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_strides: &[usize],
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Cast, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    let length: usize = shape.iter().product();
    debug_group!(encoder, "cast_strided {kernel_name} elems={length}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length,
            shape.len(),
            shape,
            input_strides,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, CAST_STRIDED_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
