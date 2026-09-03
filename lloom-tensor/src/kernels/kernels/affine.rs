use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, AffineParams, AffineStridedParams, ParamStyle,
    ScaleParams, ScaleStridedParams,
};
use crate::kernels::utils::{BufferOffset, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use crate::kernels::{get_tile_size, linear_split};

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `affine_params_layout_matches_metal` is what proves those
/// mirrors agree with `affine.metal`.
///
/// `powf` and `elu` bind one float where `affine` binds two, so they use the
/// `Scale*` blocks: a shared struct would be longer than the bytes the capture
/// produces, and the kernel would read a field that was never written.
const AFFINE_PARAMS_ALIGN: usize = core::mem::align_of::<AffineParams>();
const AFFINE_STRIDED_PARAMS_ALIGN: usize = core::mem::align_of::<AffineStridedParams>();
const SCALE_PARAMS_ALIGN: usize = core::mem::align_of::<ScaleParams>();
const SCALE_STRIDED_PARAMS_ALIGN: usize = core::mem::align_of::<ScaleStridedParams>();

#[allow(clippy::too_many_arguments)]
pub fn call_affine(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dtype_size: usize,
    size: usize,
    input: BufferOffset,
    output: &Buffer,
    mul: f32,
    add: f32,
) -> Result<(), MetalKernelError> {
    call_affine_with(
        device,
        ep,
        kernels,
        name,
        dtype_size,
        size,
        input,
        output,
        mul,
        add,
        ParamStyle::default(),
    )
}

/// As [`call_affine`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_affine_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dtype_size: usize,
    size: usize,
    input: BufferOffset,
    output: &Buffer,
    mul: f32,
    add: f32,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name);
    let pipeline = kernels.load_pipeline(device, Source::Affine, kernel)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "affine {name} elems={size}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(encoder, (size, mul, add, &input, Output::new(output)));
    let _staged = finish_packed_params(device, encoder, style, AFFINE_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = size.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_affine_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_stride: &[usize],
    output: &Buffer,
    mul: f32,
    add: f32,
) -> Result<(), MetalKernelError> {
    call_affine_strided_with(
        device,
        ep,
        kernels,
        name,
        shape,
        input,
        input_stride,
        output,
        mul,
        add,
        ParamStyle::default(),
    )
}

/// As [`call_affine_strided`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_affine_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_stride: &[usize],
    output: &Buffer,
    mul: f32,
    add: f32,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name);
    let pipeline = kernels.load_pipeline(device, Source::Affine, kernel)?;
    let size: usize = shape.iter().product();

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "affine_strided {name} elems={size}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            size,
            shape.len(),
            shape,
            input_stride,
            mul,
            add,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, AFFINE_STRIDED_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_powf(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dtype_size: usize,
    size: usize,
    input: BufferOffset,
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    call_powf_with(
        device,
        ep,
        kernels,
        name,
        dtype_size,
        size,
        input,
        output,
        mul,
        ParamStyle::default(),
    )
}

/// As [`call_powf`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_powf_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dtype_size: usize,
    size: usize,
    input: BufferOffset,
    output: &Buffer,
    mul: f32,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name);
    let pipeline = kernels.load_pipeline(device, Source::Affine, kernel)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "powf {name} elems={size}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(encoder, (size, mul, &input, Output::new(output)));
    let _staged = finish_packed_params(device, encoder, style, SCALE_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = size.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_powf_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_stride: &[usize],
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    call_powf_strided_with(
        device,
        ep,
        kernels,
        name,
        shape,
        input,
        input_stride,
        output,
        mul,
        ParamStyle::default(),
    )
}

/// As [`call_powf_strided`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_powf_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_stride: &[usize],
    output: &Buffer,
    mul: f32,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name);
    let pipeline = kernels.load_pipeline(device, Source::Affine, kernel)?;
    let size: usize = shape.iter().product();

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "powf_strided {name} elems={size}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            size,
            shape.len(),
            shape,
            input_stride,
            mul,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, SCALE_STRIDED_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_elu(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dtype_size: usize,
    size: usize,
    input: BufferOffset,
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    call_elu_with(
        device,
        ep,
        kernels,
        name,
        dtype_size,
        size,
        input,
        output,
        mul,
        ParamStyle::default(),
    )
}

/// As [`call_elu`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_elu_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dtype_size: usize,
    size: usize,
    input: BufferOffset,
    output: &Buffer,
    mul: f32,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name);
    let pipeline = kernels.load_pipeline(device, Source::Affine, kernel)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "elu {name} elems={size}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(encoder, (size, mul, &input, Output::new(output)));
    let _staged = finish_packed_params(device, encoder, style, SCALE_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = size.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_elu_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_stride: &[usize],
    output: &Buffer,
    mul: f32,
) -> Result<(), MetalKernelError> {
    call_elu_strided_with(
        device,
        ep,
        kernels,
        name,
        shape,
        input,
        input_stride,
        output,
        mul,
        ParamStyle::default(),
    )
}

/// As [`call_elu_strided`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_elu_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    input: BufferOffset,
    input_stride: &[usize],
    output: &Buffer,
    mul: f32,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name);
    let pipeline = kernels.load_pipeline(device, Source::Affine, kernel)?;
    let size: usize = shape.iter().product();

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "elu_strided {name} elems={size}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            size,
            shape.len(),
            shape,
            input_stride,
            mul,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, SCALE_STRIDED_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
