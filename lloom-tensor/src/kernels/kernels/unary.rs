use crate::kernels::kernels::macros::ops;
use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, Copy2dParams, ParamStyle, UnaryParams,
    UnaryStridedParams,
};
use crate::kernels::utils::{BufferOffset, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, EncoderParam, Kernels,
    MetalKernelError, Output, Source,
};
use crate::kernels::{get_block_dims, get_tile_size, linear_split};
use objc2_metal::MTLSize;

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `unary_params_layout_matches_metal` is what proves those
/// mirrors agree with `unary.metal`.
const UNARY_PARAMS_ALIGN: usize = core::mem::align_of::<UnaryParams>();
const UNARY_STRIDED_PARAMS_ALIGN: usize = core::mem::align_of::<UnaryStridedParams>();
const COPY2D_PARAMS_ALIGN: usize = core::mem::align_of::<Copy2dParams>();

ops!(
    cos, sin, exp, sqr, sqrt, neg, log, gelu, abs, ceil, floor, relu, round, erf, gelu_erf, tanh,
    recip, silu, sign, sigmoid, const_set
);

#[allow(clippy::too_many_arguments)]
pub fn call_unary_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    call_unary_contiguous_with(
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

/// As [`call_unary_contiguous`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_unary_contiguous_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "unary {} elems={length}", kernel_name.0);

    let _capture = begin_packed_params(encoder, style);
    set_params!(encoder, (length, &input, Output::new(output)));
    let _staged = finish_packed_params(device, encoder, style, UNARY_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: BufferOffset,
    strides: &[usize],
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    call_unary_strided_with(
        device,
        ep,
        kernels,
        name,
        shape,
        input,
        strides,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_unary_strided`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_unary_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: BufferOffset,
    strides: &[usize],
    output: BufferOffset,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel)?;

    let length: usize = shape.iter().product();
    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "unary_strided {} elems={length}", name.0);
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            strides,
            &input,
            Output::from_buffer_offset(&output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, UNARY_STRIDED_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_const_set_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: impl EncoderParam,
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    call_const_set_contiguous_with(
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

/// As [`call_const_set_contiguous`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_const_set_contiguous_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: impl EncoderParam,
    output: BufferOffset,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "const_set {} elems={length}", kernel_name.0);
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (length, input, Output::from_buffer_offset(&output))
    );
    let _staged = finish_packed_params(device, encoder, style, UNARY_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_const_set_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: impl EncoderParam,
    strides: &[usize],
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    call_const_set_strided_with(
        device,
        ep,
        kernels,
        name,
        shape,
        input,
        strides,
        output,
        ParamStyle::default(),
    )
}

/// As [`call_const_set_strided`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_const_set_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: impl EncoderParam,
    strides: &[usize],
    output: BufferOffset,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel)?;

    let length: usize = shape.iter().product();
    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "const_set_strided {} elems={length}", name.0);
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            strides,
            input,
            Output::from_buffer_offset(&output)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, UNARY_STRIDED_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

pub mod copy2d {
    pub struct Kernel(pub &'static str);
    pub const FLOAT: Kernel = Kernel("copy2d_f32");
    pub const HALF: Kernel = Kernel("copy2d_f16");
    pub const BFLOAT: Kernel = Kernel("copy2d_bf16");
    pub const I64: Kernel = Kernel("copy2d_i64");
    pub const I32: Kernel = Kernel("copy2d_i32");
    pub const I16: Kernel = Kernel("copy2d_i16");
    pub const U32: Kernel = Kernel("copy2d_u32");
    pub const U8: Kernel = Kernel("copy2d_u8");
}

#[allow(clippy::too_many_arguments)]
pub fn call_copy2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: copy2d::Kernel,
    input: &Buffer,
    output: &Buffer,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src_o_in_bytes: usize,
    dst_o_in_bytes: usize,
) -> Result<(), MetalKernelError> {
    call_copy2d_with(
        device,
        ep,
        kernels,
        name,
        input,
        output,
        d1,
        d2,
        src_s,
        dst_s,
        src_o_in_bytes,
        dst_o_in_bytes,
        ParamStyle::default(),
    )
}

/// As [`call_copy2d`], choosing how the scalars are bound.
///
/// The classical entry point above delegates here with [`ParamStyle::Split`],
/// so there is one body rather than two and the styles cannot drift in what
/// they bind -- which is the property that makes the bit-identical test
/// meaningful.
#[allow(clippy::too_many_arguments)]
pub fn call_copy2d_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: copy2d::Kernel,
    input: &Buffer,
    output: &Buffer,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src_o_in_bytes: usize,
    dst_o_in_bytes: usize,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let kernel = style.kernel_name(name.0);
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "copy2d {} d1={d1} d2={d2}", name.0);
    let _capture = begin_packed_params(encoder, style);
    set_params!(
        encoder,
        (
            d1 as i64,
            d2 as i64,
            src_s as i64,
            dst_s as i64,
            (input, src_o_in_bytes),
            Output::with_offset(output, dst_o_in_bytes)
        )
    );
    let _staged = finish_packed_params(device, encoder, style, COPY2D_PARAMS_ALIGN)?;

    let grid_dims = MTLSize {
        width: d1,
        height: d2,
        depth: 1,
    };
    let group_dims = get_block_dims(d1, d2, 1);
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

///////////////////////////////////////////////////////////////////////////////
/// The SiLU variant axis (issue #599, `DESIGN.md` §12.2l)
///////////////////////////////////////////////////////////////////////////////

/// Which SiLU implementation the `usilu` dispatch resolves to.
///
/// # What this axis is for
///
/// #599 asks *how much error the model absorbs before it degenerates*, and that
/// question needs SiLU arms separated by a known, wide range of error. #604
/// (`DESIGN.md` §12.2k) measured the five spellings already in the tree and
/// found them bit-identical at f32 and 0-3 ulp apart at f16 -- a spread far too
/// narrow to locate a threshold, and the reason the arms below were built.
///
/// # Why a process switch rather than a parameter
///
/// `MetalStorage::unary_impl` resolves `("usilu", dtype)` to a `&'static str`
/// through a hardcoded match (`metal_backend/mod.rs:892`), and `candle-core`
/// reaches it from `Tensor::silu` with no argument to thread down from the
/// model. That is exactly [`crate::kernels::ParamStyle`]'s situation, and this follows
/// its shape for its reasons: overriding the resolution in one place leaves
/// every signature and the classical behaviour untouched, and a harness that
/// cannot select an arm cannot gate it.
///
/// **`Exact` is the default and is byte-identical to what shipped.** An
/// unconfigured process resolves `silu_f16`, the name it always resolved, so
/// no default is flipped here (`DESIGN.md` §7.1a).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SiluImpl {
    /// `x / (1 + exp(-x))` -- `unary.metal:326`, what ships. The correctness bar.
    #[default]
    Exact,
    /// `x * recip(1 + exp(-x))` -- `unary.metal:308`'s sigmoid spelling.
    ///
    /// **The internal control, not a variant.** #604 measured it bit-identical
    /// to `Exact` at f32 and f16, so it must read zero everywhere; a sweep in
    /// which every arm differs cannot show that the instrument reports no
    /// difference when there is none (§3.1a rule 4).
    Recip,
    /// `x / (1.0f + exp(-x))` -- `quantized.metal:1563`'s spelling.
    ///
    /// The smallest non-zero perturbation available from a real site in this
    /// tree: 1 ulp at f16 over 440 of 4096 points, bit-identical at f32,
    /// measured by #604. The mechanism is the `1.0f` suffix promoting the
    /// addition to float, not the algebra.
    Literal,
    /// `exp2` with its argument rounded to 4 fractional bits.
    ///
    /// Peak absolute error ~4.6e-3 against true SiLU -- three orders of
    /// magnitude coarser than [`Self::Literal`] and an order finer than
    /// [`Self::Coarse1`]. The error is a genuine approximation error: it
    /// decays to zero in the tail exactly as SiLU does.
    Coarse4,
    /// `exp2` with its argument rounded to 1 fractional bit.
    ///
    /// Peak absolute error ~3.9e-2. The same mechanism as [`Self::Coarse4`]
    /// with the knob turned, which is what makes the pair a controlled step on
    /// the ladder rather than two unrelated approximations.
    Coarse1,
    /// `x * clamp(x/6 + 0.5, 0, 1)` -- MobileNetV3's hard-swish.
    ///
    /// Piecewise-linear, exp-free, peak absolute error ~0.14, and exactly zero
    /// for `x <= -3` where SiLU is small but non-zero.
    HardSwish,
    /// `x < 0 ? 0 : x` — ReLU. **Not an approximation to SiLU: a different
    /// activation.**
    ///
    /// Peak absolute error ~0.28, about twice [`Self::HardSwish`]'s, and it
    /// abandons SiLU's shape rather than approximating it — exactly zero for
    /// all negative x, where SiLU has its characteristic dip. The top of the
    /// ladder, added because the first sweep saw every arm below it pass the
    /// text gate.
    Relu,
}

/// Which arm [`SiluImpl::default`] yields. `0` is [`SiluImpl::Exact`], so an
/// unconfigured process is byte-for-byte what it was.
static SILU_IMPL: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Select the SiLU arm every subsequent `usilu` dispatch resolves to.
///
/// Takes effect at the next dispatch. See [`SiluImpl`] for why the choice lives
/// here rather than at the call sites.
pub fn set_silu_impl(impl_: SiluImpl) {
    SILU_IMPL.store(impl_ as u8, std::sync::atomic::Ordering::Relaxed);
}

/// The arm currently selected.
pub fn silu_impl() -> SiluImpl {
    match SILU_IMPL.load(std::sync::atomic::Ordering::Relaxed) {
        1 => SiluImpl::Recip,
        2 => SiluImpl::Literal,
        3 => SiluImpl::Coarse4,
        4 => SiluImpl::Coarse1,
        5 => SiluImpl::HardSwish,
        6 => SiluImpl::Relu,
        _ => SiluImpl::Exact,
    }
}

impl SiluImpl {
    /// The name segment this arm appends after `silu` and before the dtype.
    ///
    /// `Exact` contributes nothing, which is what makes the default resolve the
    /// pre-existing `silu_f16` rather than a renamed sibling.
    pub const fn suffix(self) -> &'static str {
        match self {
            SiluImpl::Exact => "",
            SiluImpl::Recip => "_recip",
            SiluImpl::Literal => "_literal",
            SiluImpl::Coarse4 => "_coarse4",
            SiluImpl::Coarse1 => "_coarse1",
            SiluImpl::HardSwish => "_hardswish",
            SiluImpl::Relu => "_relu",
        }
    }

    /// Every arm, in ladder order. The sweep iterates this rather than a list
    /// written beside it -- `DESIGN.md` §11.3h's recurring lesson.
    pub const ALL: [SiluImpl; 7] = [
        SiluImpl::Exact,
        SiluImpl::Recip,
        SiluImpl::Literal,
        SiluImpl::Coarse4,
        SiluImpl::Coarse1,
        SiluImpl::HardSwish,
        SiluImpl::Relu,
    ];

    /// Parse an arm from its CLI spelling, which is the suffix without the
    /// leading underscore: `exact`, `recip`, `literal`, `coarse4`, `coarse1`,
    /// `hardswish`, `relu`.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "exact" => Some(SiluImpl::Exact),
            "recip" => Some(SiluImpl::Recip),
            "literal" => Some(SiluImpl::Literal),
            "coarse4" => Some(SiluImpl::Coarse4),
            "coarse1" => Some(SiluImpl::Coarse1),
            "hardswish" => Some(SiluImpl::HardSwish),
            "relu" => Some(SiluImpl::Relu),
            _ => None,
        }
    }

    /// The CLI spelling, and the arm name a `RESULT` line carries.
    pub const fn name(self) -> &'static str {
        match self {
            SiluImpl::Exact => "exact",
            SiluImpl::Recip => "recip",
            SiluImpl::Literal => "literal",
            SiluImpl::Coarse4 => "coarse4",
            SiluImpl::Coarse1 => "coarse1",
            SiluImpl::HardSwish => "hardswish",
            SiluImpl::Relu => "relu",
        }
    }

    /// The contiguous `[[host_name]]` for this arm at `dtype`.
    ///
    /// Returns `&'static str` so the per-token path allocates nothing: the
    /// pipeline cache is keyed on the name and this is read once per dispatch
    /// (`DESIGN.md` §15.2 #10).
    pub const fn contiguous_name(self, dtype: SiluDType) -> &'static str {
        match (self, dtype) {
            (SiluImpl::Exact, SiluDType::F16) => "silu_f16",
            (SiluImpl::Exact, SiluDType::F32) => "silu_f32",
            (SiluImpl::Exact, SiluDType::BF16) => "silu_bf16",
            (SiluImpl::Recip, SiluDType::F16) => "silu_recip_f16",
            (SiluImpl::Recip, SiluDType::F32) => "silu_recip_f32",
            (SiluImpl::Recip, SiluDType::BF16) => "silu_recip_bf16",
            (SiluImpl::Literal, SiluDType::F16) => "silu_literal_f16",
            (SiluImpl::Literal, SiluDType::F32) => "silu_literal_f32",
            (SiluImpl::Literal, SiluDType::BF16) => "silu_literal_bf16",
            (SiluImpl::Coarse4, SiluDType::F16) => "silu_coarse4_f16",
            (SiluImpl::Coarse4, SiluDType::F32) => "silu_coarse4_f32",
            (SiluImpl::Coarse4, SiluDType::BF16) => "silu_coarse4_bf16",
            (SiluImpl::Coarse1, SiluDType::F16) => "silu_coarse1_f16",
            (SiluImpl::Coarse1, SiluDType::F32) => "silu_coarse1_f32",
            (SiluImpl::Coarse1, SiluDType::BF16) => "silu_coarse1_bf16",
            (SiluImpl::HardSwish, SiluDType::F16) => "silu_hardswish_f16",
            (SiluImpl::HardSwish, SiluDType::F32) => "silu_hardswish_f32",
            (SiluImpl::HardSwish, SiluDType::BF16) => "silu_hardswish_bf16",
            (SiluImpl::Relu, SiluDType::F16) => "silu_relu_f16",
            (SiluImpl::Relu, SiluDType::F32) => "silu_relu_f32",
            (SiluImpl::Relu, SiluDType::BF16) => "silu_relu_bf16",
        }
    }

    /// The strided `[[host_name]]` for this arm at `dtype`.
    pub const fn strided_name(self, dtype: SiluDType) -> &'static str {
        match (self, dtype) {
            (SiluImpl::Exact, SiluDType::F16) => "silu_f16_strided",
            (SiluImpl::Exact, SiluDType::F32) => "silu_f32_strided",
            (SiluImpl::Exact, SiluDType::BF16) => "silu_bf16_strided",
            (SiluImpl::Recip, SiluDType::F16) => "silu_recip_f16_strided",
            (SiluImpl::Recip, SiluDType::F32) => "silu_recip_f32_strided",
            (SiluImpl::Recip, SiluDType::BF16) => "silu_recip_bf16_strided",
            (SiluImpl::Literal, SiluDType::F16) => "silu_literal_f16_strided",
            (SiluImpl::Literal, SiluDType::F32) => "silu_literal_f32_strided",
            (SiluImpl::Literal, SiluDType::BF16) => "silu_literal_bf16_strided",
            (SiluImpl::Coarse4, SiluDType::F16) => "silu_coarse4_f16_strided",
            (SiluImpl::Coarse4, SiluDType::F32) => "silu_coarse4_f32_strided",
            (SiluImpl::Coarse4, SiluDType::BF16) => "silu_coarse4_bf16_strided",
            (SiluImpl::Coarse1, SiluDType::F16) => "silu_coarse1_f16_strided",
            (SiluImpl::Coarse1, SiluDType::F32) => "silu_coarse1_f32_strided",
            (SiluImpl::Coarse1, SiluDType::BF16) => "silu_coarse1_bf16_strided",
            (SiluImpl::HardSwish, SiluDType::F16) => "silu_hardswish_f16_strided",
            (SiluImpl::HardSwish, SiluDType::F32) => "silu_hardswish_f32_strided",
            (SiluImpl::HardSwish, SiluDType::BF16) => "silu_hardswish_bf16_strided",
            (SiluImpl::Relu, SiluDType::F16) => "silu_relu_f16_strided",
            (SiluImpl::Relu, SiluDType::F32) => "silu_relu_f32_strided",
            (SiluImpl::Relu, SiluDType::BF16) => "silu_relu_bf16_strided",
        }
    }
}

/// The three float dtypes `init_unary_float` instantiates.
///
/// A local enum rather than `candle_core::DType` because this crate does not
/// depend on `candle-core` -- the same reason `GemmDType` exists here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiluDType {
    F16,
    F32,
    BF16,
}
