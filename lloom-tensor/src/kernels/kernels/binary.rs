use crate::kernels::kernels::macros::ops;
use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, BinaryParams, BinaryStridedParams, ParamStyle,
};
use crate::kernels::utils::{BufferOffset, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use crate::kernels::{get_tile_size, linear_split};

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `binary_params_layout_matches_metal` is what proves those
/// mirrors agree with `binary.metal`.
const BINARY_PARAMS_ALIGN: usize = core::mem::align_of::<BinaryParams>();
const BINARY_STRIDED_PARAMS_ALIGN: usize = core::mem::align_of::<BinaryStridedParams>();

ops!(badd, bsub, bmul, bdiv, bminimum, bmaximum, eq, ne, le, lt, ge, gt);

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_binary_contiguous<S: ToString>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: S,
    dtype_size: usize,
    length: usize,
    left: BufferOffset,
    right: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_binary_contiguous_with(
            device,
            ep,
            kernels,
            kernel_name,
            dtype_size,
            length,
            left,
            right,
            output,
            ParamStyle::default(),
        )
    }
}

/// As [`call_binary_contiguous`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_binary_contiguous_with<S: ToString>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: S,
    dtype_size: usize,
    length: usize,
    left: BufferOffset,
    right: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel_name = kernel_name.to_string();
    let name = style.kernel_name_owned(kernel_name.clone());
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        encode_binary_contiguous(
            device,
            ep,
            kernels,
            &kernel_name,
            name,
            dtype_size,
            length,
            left,
            right,
            output,
            style,
        )
    }
}

/// As [`call_binary_contiguous`], for a classical name that is already a
/// `&'static str` (`MetalStorage::binary`'s interned names): the `Split`
/// name is borrowed rather than copied, so the dispatch builds no `String`
/// (#1235 step 1: 60 allocations a served LFM2 token at the `to_string` and
/// its clone in [`call_binary_contiguous_with`]).
///
/// # Safety
/// As [`call_binary_contiguous`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_binary_contiguous_static(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    dtype_size: usize,
    length: usize,
    left: BufferOffset,
    right: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let style = ParamStyle::default();
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        encode_binary_contiguous(
            device,
            ep,
            kernels,
            kernel_name,
            style.kernel_name(kernel_name),
            dtype_size,
            length,
            left,
            right,
            output,
            style,
        )
    }
}

/// The one body both contiguous entry points share: `display` is the
/// classical name for the debug group, `name` the styled pipeline name.
///
/// # Safety
/// As [`call_binary_contiguous`].
#[allow(clippy::too_many_arguments)]
unsafe fn encode_binary_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    display: &str,
    name: crate::kernels::kernel::KernelName,
    dtype_size: usize,
    length: usize,
    left: BufferOffset,
    right: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    // `display` is read only by `debug_group!` (a no-op without `debug-labels`).
    let _ = display;
    let pipeline = kernels.load_pipeline(device, Source::Binary, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "binary {display} elems={length}");

    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(encoder, (length, &left, &right, Output::new(output)));
    }
    let _staged = finish_packed_params(device, encoder, style, BINARY_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_binary_strided<S: ToString>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: S,
    dtype_size: usize,
    shape: &[usize],
    left_input: BufferOffset,
    left_strides: &[usize],
    right_input: BufferOffset,
    right_strides: &[usize],
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_binary_strided_with(
            device,
            ep,
            kernels,
            kernel_name,
            dtype_size,
            shape,
            left_input,
            left_strides,
            right_input,
            right_strides,
            output,
            ParamStyle::default(),
        )
    }
}

/// As [`call_binary_strided`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_binary_strided_with<S: ToString>(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: S,
    dtype_size: usize,
    shape: &[usize],
    left_input: BufferOffset,
    left_strides: &[usize],
    right_input: BufferOffset,
    right_strides: &[usize],
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel_name = kernel_name.to_string();
    let name = style.kernel_name_owned(kernel_name.clone());
    let pipeline = kernels.load_pipeline(device, Source::Binary, name)?;

    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let length: usize = shape.iter().product();
    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "binary_strided {kernel_name} elems={length}");
    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                length,
                num_dims,
                shape,
                left_strides,
                right_strides,
                &left_input,
                &right_input,
                Output::new(output)
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, BINARY_STRIDED_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
