use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, Col2im1dParams, Conv1dDepthwiseKParams,
    Conv1dDepthwiseParams, ConvTranspose1dParams, ConvTranspose2dParams, Im2col1dParams,
    Im2colParams, ParamStyle, Pool2dParams, ShortconvRingParams, ShortconvStepParams,
    UpsampleBilinear2dParams, UpsampleNearest2dParams,
};
use crate::kernels::linear_split;
use crate::kernels::utils::{BufferOffset, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `conv_params_layout_matches_metal` is what proves those
/// mirrors agree with `conv.metal`.
const IM2COL_PARAMS_ALIGN: usize = core::mem::align_of::<Im2colParams>();
const COL2IM1D_PARAMS_ALIGN: usize = core::mem::align_of::<Col2im1dParams>();
const IM2COL1D_PARAMS_ALIGN: usize = core::mem::align_of::<Im2col1dParams>();
const UPSAMPLE_NEAREST2D_PARAMS_ALIGN: usize = core::mem::align_of::<UpsampleNearest2dParams>();
const UPSAMPLE_BILINEAR2D_PARAMS_ALIGN: usize = core::mem::align_of::<UpsampleBilinear2dParams>();
const POOL2D_PARAMS_ALIGN: usize = core::mem::align_of::<Pool2dParams>();
const CONV_TRANSPOSE1D_PARAMS_ALIGN: usize = core::mem::align_of::<ConvTranspose1dParams>();
const CONV_TRANSPOSE2D_PARAMS_ALIGN: usize = core::mem::align_of::<ConvTranspose2dParams>();
const CONV1D_DEPTHWISE_PARAMS_ALIGN: usize = core::mem::align_of::<Conv1dDepthwiseParams>();
const CONV1D_DEPTHWISE_K_PARAMS_ALIGN: usize = core::mem::align_of::<Conv1dDepthwiseKParams>();
const SHORTCONV_STEP_PARAMS_ALIGN: usize = core::mem::align_of::<ShortconvStepParams>();
const SHORTCONV_RING_PARAMS_ALIGN: usize = core::mem::align_of::<ShortconvRingParams>();

// A note on widths, since #38 found a latent mismatch of exactly this kind in
// `reduce.metal`'s callers and predicted it in the remaining six files.
//
// `conv.metal` declares every scalar as `constant size_t &` -- 8 bytes -- and
// every `call_*` below binds a `usize`, which is also 8. So the mismatch #38
// describes (a `usize` bound to a `constant uint &`, benign under `setBytes`
// and wrong once packed) **does not occur in this family**. Checked rather than
// assumed: the two `bool` and two `f32` in `call_upsample_bilinear_2d` are the
// only non-`usize` binds, and `conv.metal` declares them `bool` and `float` to
// match. The parity tests are what would have caught it either way.

#[allow(clippy::too_many_arguments)]
pub fn call_im2col1d_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_im2col1d_strided_with(
        device,
        ep,
        kernels,
        name,
        shape,
        strides,
        (k_size, stride, padding, dilation),
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_im2col1d_strided`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_im2col1d_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    let l_out = (shape[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    let dst_el = shape[0] * l_out * shape[1] * k_size;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "im2col1d {name} dst_el={dst_el}");
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            k_size,
            stride,
            padding,
            dilation,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, IM2COL1D_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Fused depthwise 1D convolution.
///
/// Computes the whole layer in one dispatch, writing directly in `(b, c, l_out)`
/// layout. The generic path instead builds an im2col matrix, runs a matmul and
/// transposes the result, and candle splits grouped convolutions into one
/// convolution per group — so a depthwise layer over N channels costs roughly
/// 3N dispatches there against one here.
#[allow(clippy::too_many_arguments)]
pub fn call_conv1d_depthwise(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: BufferOffset,
    weight: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_conv1d_depthwise_with(
        device,
        ep,
        kernels,
        name,
        shape,
        strides,
        (k_size, stride, padding, dilation),
        input,
        weight,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_conv1d_depthwise`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_conv1d_depthwise_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (k_size, stride, padding, dilation): (usize, usize, usize, usize),
    input: BufferOffset,
    weight: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    let l_out = (shape[2] + 2 * padding - dilation * (k_size - 1) - 1) / stride + 1;
    // One thread per output element, in destination order.
    let dst_el = shape[0] * shape[1] * l_out;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "conv1d_depthwise {name} dst_el={dst_el}");
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            k_size,
            stride,
            padding,
            dilation,
            shape,
            strides,
            &input,
            &weight,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, CONV1D_DEPTHWISE_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Fused depthwise 1D convolution, specialized for `stride == 1`,
/// `dilation == 1` and a contiguous source.
///
/// Same result as [`call_conv1d_depthwise`]; the difference is that `k_size` is
/// baked into the pipeline (so the tap loop unrolls) and the addressing is
/// direct rather than going through `src_strides[]`. The caller picks the name
/// via `ConvKernel::conv1d_depthwise_k(k_size)`, which returns `None` when no
/// variant is instantiated for that `k` — that is the signal to use the generic
/// entry point, not an error.
///
/// The preconditions are the caller's to enforce; this function cannot check
/// them, because the layout is not passed in. `MetalStorage::conv1d_depthwise`
/// is the single caller and checks all three.
#[allow(clippy::too_many_arguments)]
pub fn call_conv1d_depthwise_k(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    (k_size, padding): (usize, usize),
    input: BufferOffset,
    weight: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_conv1d_depthwise_k_with(
        device,
        ep,
        kernels,
        name,
        shape,
        (k_size, padding),
        input,
        weight,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_conv1d_depthwise_k`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_conv1d_depthwise_k_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    (k_size, padding): (usize, usize),
    input: BufferOffset,
    weight: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    // stride == 1 and dilation == 1 are preconditions, so this reduces to
    // l_in + 2 * padding - (k_size - 1).
    let l_out = shape[2] + 2 * padding - (k_size - 1);
    let dst_el = shape[0] * shape[1] * l_out;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "conv1d_depthwise_k {name} dst_el={dst_el}");
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            padding,
            shape,
            &input,
            &weight,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, CONV1D_DEPTHWISE_K_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// The `_k` fold with a per-output `scale` multiplied into the store — LFM2's
/// `*C` conv gate as an epilogue (`DESIGN.md` §12.2 row 1). `scale` has the same
/// `(b, c, l_out)` layout as `output`, so the kernel indexes it by the same flat
/// `tid`; the caller passes the `C` tensor. Same preconditions as
/// [`call_conv1d_depthwise_k`] (stride 1, dilation 1, contiguous source), plus
/// `scale` contiguous in destination layout — the single caller enforces both.
#[allow(clippy::too_many_arguments)]
pub fn call_conv1d_depthwise_k_scaled(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    (k_size, padding): (usize, usize),
    input: BufferOffset,
    weight: BufferOffset,
    scale: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_conv1d_depthwise_k_scaled_with(
        device,
        ep,
        kernels,
        name,
        shape,
        (k_size, padding),
        input,
        weight,
        scale,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_conv1d_depthwise_k_scaled`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_conv1d_depthwise_k_scaled_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    (k_size, padding): (usize, usize),
    input: BufferOffset,
    weight: BufferOffset,
    scale: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    let l_out = shape[2] + 2 * padding - (k_size - 1);
    let dst_el = shape[0] * shape[1] * l_out;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "conv1d_depthwise_k_scaled {name} dst_el={dst_el}");
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            padding,
            shape,
            &input,
            &weight,
            &scale,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, CONV1D_DEPTHWISE_K_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// The fused decode shortconv step (§12.2 row 1): `bx = b*x`, shift into the
/// `k`-wide `state_in`, fold, `*C`, write the shifted state to `state_out`. One
/// thread per output channel (`dst_numel == b * hidden`). `b`, `x`, `c` share
/// one `(stride_b, stride_c)` pair — they are sibling narrows of one `in_proj`
/// output — and their base offsets ride on the `BufferOffset`s. Preconditions:
/// contiguous `state_in`/`state_out` in `[b, hidden, k]`, contiguous `weight` in
/// `[hidden, k]`, decode shape (`l_out == 1`), and a `k` with an instantiated
/// variant. The single caller enforces them.
#[allow(clippy::too_many_arguments)]
pub fn call_shortconv_step(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    (dst_numel, hidden, proj_stride_b, proj_stride_c): (usize, usize, usize, usize),
    b_in: BufferOffset,
    x_in: BufferOffset,
    c_in: BufferOffset,
    state_in: BufferOffset,
    weight: BufferOffset,
    state_out: &Buffer,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    call_shortconv_step_with(
        device,
        ep,
        kernels,
        name,
        (dst_numel, hidden, proj_stride_b, proj_stride_c),
        b_in,
        x_in,
        c_in,
        state_in,
        weight,
        state_out,
        dst,
        ParamStyle::default(),
    )
}

/// As [`call_shortconv_step`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_shortconv_step_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    (dst_numel, hidden, proj_stride_b, proj_stride_c): (usize, usize, usize, usize),
    b_in: BufferOffset,
    x_in: BufferOffset,
    c_in: BufferOffset,
    state_in: BufferOffset,
    weight: BufferOffset,
    state_out: &Buffer,
    dst: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_numel);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "shortconv_step {name} dst_numel={dst_numel}");
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_numel,
            hidden,
            proj_stride_b,
            proj_stride_c,
            &b_in,
            &x_in,
            &c_in,
            &state_in,
            &weight,
            Output::new(state_out),
            Output::new(dst)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, SHORTCONV_STEP_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// The RotatingRing conv block over `seq_len` positions in one dispatch (the
/// speculative verify pass's per-position ring writer, or the K=1 decode
/// step on the same ring). `b_in`/`x_in`/`c_in` share one
/// `(stride_b, stride_c, stride_l)` triple — sibling narrows of one `in_proj`
/// output — with base offsets on the `BufferOffset`s. Preconditions:
/// contiguous `state_in`/`state_out` in `[b, hidden, width]`, contiguous
/// `weights` in `[width, hidden, width]` (the per-phase permuted table),
/// `1 <= seq_len <= width`; `dst` is written contiguous `[b, seq_len, hidden]`.
/// The single caller enforces them.
#[allow(clippy::too_many_arguments)]
pub fn call_shortconv_ring(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    params: ShortconvRingParams,
    b_in: BufferOffset,
    x_in: BufferOffset,
    c_in: BufferOffset,
    // The ring's state, read AND written in place — see the kernel's note on
    // why aliasing is sound (one `(batch, channel)` row per thread).
    state: BufferOffset,
    weights: BufferOffset,
    dst: &Buffer,
) -> Result<(), MetalKernelError> {
    call_shortconv_ring_with(
        device,
        ep,
        kernels,
        name,
        params,
        b_in,
        x_in,
        c_in,
        state,
        weights,
        dst,
        ParamStyle::default(),
    )
}

/// As [`call_shortconv_ring`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_shortconv_ring_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    params: ShortconvRingParams,
    b_in: BufferOffset,
    x_in: BufferOffset,
    c_in: BufferOffset,
    state: BufferOffset,
    weights: BufferOffset,
    dst: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let dst_numel = params.dst_numel as usize;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_numel);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "shortconv_ring {name} dst_numel={dst_numel} seq_len={}",
        params.seq_len
    );
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            params.dst_numel as usize,
            params.hidden as usize,
            params.seq_len as usize,
            params.width as usize,
            params.phase_last as usize,
            params.proj_stride_b as usize,
            params.proj_stride_c as usize,
            params.proj_stride_l as usize,
            &b_in,
            &x_in,
            &c_in,
            // `Output`, not a plain bind: the state is written, and the
            // encoder's hazard tracker must see the write to order the next
            // pass's read of the same range against it (§9.2e).
            Output::from_buffer_offset(&state),
            &weights,
            Output::new(dst)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, SHORTCONV_RING_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_col2im1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    k_size: usize,
    stride: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_col2im1d_with(
        device,
        ep,
        kernels,
        name,
        shape,
        k_size,
        stride,
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_col2im1d`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_col2im1d_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    k_size: usize,
    stride: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    let l_in = shape[1];
    let c_out = shape[2];
    let l_out = (l_in - 1) * stride + k_size;
    let dst_el = shape[0] * c_out * l_out;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "col2im1d {name} dst_el={dst_el}");
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            l_out,
            l_in,
            c_out,
            k_size,
            stride,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, COL2IM1D_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_im2col_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_im2col_strided_with(
        device,
        ep,
        kernels,
        name,
        shape,
        strides,
        (h_k, w_k, stride, padding, dilation),
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_im2col_strided`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_im2col_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    (h_k, w_k, stride, padding, dilation): (usize, usize, usize, usize, usize),
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;

    let h = shape[2];
    let w = shape[3];
    let h_out = (h + 2 * padding - dilation * (h_k - 1) - 1) / stride + 1;
    let w_out = (w + 2 * padding - dilation * (w_k - 1) - 1) / stride + 1;

    let dst_el = shape[0] * h_out * w_out * shape[1] * h_k * w_k;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "im2col {name} dst_el={dst_el}");
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            dst_el,
            h_out,
            w_out,
            h_k,
            w_k,
            stride,
            padding,
            dilation,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, IM2COL_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_upsample_nearest_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_upsample_nearest_2d_with(
        device,
        ep,
        kernels,
        name,
        shape,
        strides,
        out_w,
        out_h,
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_upsample_nearest_2d`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_upsample_nearest_2d_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let scale_w = shape[2] as f32 / out_w as f32;
    let scale_h = shape[3] as f32 / out_h as f32;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "upsample_nearest2d {name} {out_w}x{out_h}");
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            out_w,
            out_h,
            scale_w,
            scale_h,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, UPSAMPLE_NEAREST2D_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_upsample_bilinear_2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_upsample_bilinear_2d_with(
        device,
        ep,
        kernels,
        name,
        shape,
        strides,
        out_w,
        out_h,
        align_corners,
        scale_h,
        scale_w,
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_upsample_bilinear_2d`], choosing how the scalars are bound.
///
/// This is the family whose packed block exercises both layout hazards issue
/// #38 names at once: three `bool` (1 byte in MSL as in Rust) and two `f32`
/// between two `size_t`, so five of the seven fields land at an offset the
/// padding rule decides. It is also the only conv family whose classical entry
/// point has pinned `[[buffer(N)]]` indices, which the packed one deliberately
/// does *not* carry forward -- seven scalars leave the argument list, so the
/// remaining bindings renumber down to 0..4. See `conv.metal`.
#[allow(clippy::too_many_arguments)]
pub fn call_upsample_bilinear_2d_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    align_corners: bool,
    scale_h: Option<f64>,
    scale_w: Option<f64>,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    let dst_el = out_w * out_h * shape[0] * shape[1];

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "upsample_bilinear2d {name} {out_w}x{out_h}");

    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            out_w,
            out_h,
            align_corners,
            scale_h.is_some(),
            scale_h.unwrap_or(0.0) as f32,
            scale_w.is_some(),
            scale_w.unwrap_or(0.0) as f32,
            shape,
            strides,
            &input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, UPSAMPLE_BILINEAR2D_PARAMS_ALIGN)?;

    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_pool2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    w_k: usize,
    h_k: usize,
    w_stride: usize,
    h_stride: usize,
    input: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_pool2d_with(
        device,
        ep,
        kernels,
        name,
        shape,
        strides,
        out_w,
        out_h,
        w_k,
        h_k,
        w_stride,
        h_stride,
        input,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_pool2d`], choosing how the scalars are bound.
///
/// Serves both `max_pool2d` and `avg_pool2d`, which bind the same four scalars
/// and differ only in their accumulator — a template parameter on the Metal
/// side, so it is not part of the binding and is untouched here. The integer
/// `avg_pool2d` instantiations accumulate in their own type rather than
/// widening (`DESIGN.md` §8.1c), and that is preserved by construction.
#[allow(clippy::too_many_arguments)]
pub fn call_pool2d_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_w: usize,
    out_h: usize,
    w_k: usize,
    h_k: usize,
    w_stride: usize,
    h_stride: usize,
    input: &Buffer,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let dst_el = out_w * out_h * shape[0] * shape[1];
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "pool2d {name} {out_w}x{out_h} k={w_k}x{h_k}");
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            w_k,
            h_k,
            w_stride,
            h_stride,
            shape,
            strides,
            input,
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, POOL2D_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose1d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    c_out: usize,
    l_out: usize,
    b_size: usize,
    src_shape: &[usize],
    src_strides: &[usize],
    kernel_shape: &[usize],
    kernel_strides: &[usize],
    input: &Buffer,
    input_offset: usize,
    kernel: &Buffer,
    kernel_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_conv_transpose1d_with(
        device,
        ep,
        kernels,
        name,
        dilation,
        stride,
        padding,
        out_padding,
        c_out,
        l_out,
        b_size,
        src_shape,
        src_strides,
        kernel_shape,
        kernel_strides,
        input,
        input_offset,
        kernel,
        kernel_offset,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_conv_transpose1d`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose1d_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    dilation: usize,
    stride: usize,
    padding: usize,
    out_padding: usize,
    c_out: usize,
    l_out: usize,
    b_size: usize,
    src_shape: &[usize],
    src_strides: &[usize],
    kernel_shape: &[usize],
    kernel_strides: &[usize],
    input: &Buffer,
    input_offset: usize,
    kernel: &Buffer,
    kernel_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let dst_el = c_out * l_out * b_size;
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "conv_transpose1d {name} c_out={c_out} l_out={l_out} b={b_size}"
    );
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            l_out,
            stride,
            padding,
            out_padding,
            dilation,
            src_shape,
            src_strides,
            kernel_shape,
            kernel_strides,
            (input, input_offset),
            (kernel, kernel_offset),
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, CONV_TRANSPOSE1D_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

pub struct CallConvTranspose2dCfg<'a> {
    pub dilation: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub c_out: usize,
    pub out_w: usize,
    pub out_h: usize,
    pub b_size: usize,
    pub input_dims: &'a [usize],
    pub input_stride: &'a [usize],
    pub kernel_dims: &'a [usize],
    pub kernel_stride: &'a [usize],
    pub input_offset: usize,
    pub kernel_offset: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    cfg: CallConvTranspose2dCfg,
    input: &Buffer,
    kernel: &Buffer,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_conv_transpose2d_with(
        device,
        ep,
        kernels,
        name,
        cfg,
        input,
        kernel,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_conv_transpose2d`], choosing how the scalars are bound.
#[allow(clippy::too_many_arguments)]
pub fn call_conv_transpose2d_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    cfg: CallConvTranspose2dCfg,
    input: &Buffer,
    kernel: &Buffer,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let dst_el = cfg.c_out * cfg.out_w * cfg.out_h * cfg.b_size;
    let pipeline = kernels.load_pipeline(device, Source::Conv, style.kernel_name(name))?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, dst_el);
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "conv_transpose2d {name} c_out={} {}x{} b={}",
        cfg.c_out,
        cfg.out_w,
        cfg.out_h,
        cfg.b_size
    );
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            cfg.out_w,
            cfg.out_h,
            cfg.stride,
            cfg.padding,
            cfg.output_padding,
            cfg.dilation,
            cfg.input_dims,
            cfg.input_stride,
            cfg.kernel_dims,
            cfg.kernel_stride,
            (input, cfg.input_offset),
            (kernel, cfg.kernel_offset),
            Output::new(output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, CONV_TRANSPOSE2D_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
