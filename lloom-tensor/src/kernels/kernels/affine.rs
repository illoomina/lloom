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

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_affine(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_affine`], choosing how the scalars are bound.
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
pub unsafe fn call_affine_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(encoder, (size, mul, add, &input, Output::new(output)));
    }
    let _staged = finish_packed_params(device, encoder, style, AFFINE_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = size.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_affine_strided(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_affine_strided`], choosing how the scalars are bound.
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
pub unsafe fn call_affine_strided_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
    }
    let _staged = finish_packed_params(device, encoder, style, AFFINE_STRIDED_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_powf(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_powf`], choosing how the scalars are bound.
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
pub unsafe fn call_powf_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(encoder, (size, mul, &input, Output::new(output)));
    }
    let _staged = finish_packed_params(device, encoder, style, SCALE_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = size.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_powf_strided(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_powf_strided`], choosing how the scalars are bound.
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
pub unsafe fn call_powf_strided_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
    }
    let _staged = finish_packed_params(device, encoder, style, SCALE_STRIDED_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_elu(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_elu`], choosing how the scalars are bound.
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
pub unsafe fn call_elu_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(encoder, (size, mul, &input, Output::new(output)));
    }
    let _staged = finish_packed_params(device, encoder, style, SCALE_PARAMS_ALIGN)?;

    let tile_size = get_tile_size(dtype_size);
    let tiles = size.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_elu_strided(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_elu_strided`], choosing how the scalars are bound.
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
pub unsafe fn call_elu_strided_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
    }
    let _staged = finish_packed_params(device, encoder, style, SCALE_STRIDED_PARAMS_ALIGN)?;

    let (thread_group_count, thread_group_size) = linear_split(&pipeline, size);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

// Plan-owned counterpart for the six actual affine/powf/elu backend routes.
// Existing public raw APIs and both parameter styles remain unchanged here.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AffineOperation {
    Affine,
    Powf,
    Elu,
}
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum AffineDType {
    U8,
    U32,
    I64,
    F16,
    F32,
    BF16,
}
impl AffineDType {
    fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::F16 | Self::BF16 => 2,
            Self::U32 | Self::F32 => 4,
            Self::I64 => 8,
        }
    }
    fn names(self, operation: AffineOperation) -> Option<(&'static str, &'static str)> {
        Some(match (operation, self) {
            (AffineOperation::Affine, Self::U8) => ("affine_u8", "affine_u8_strided"),
            (AffineOperation::Affine, Self::U32) => ("affine_u32", "affine_u32_strided"),
            (AffineOperation::Affine, Self::I64) => ("affine_i64", "affine_i64_strided"),
            (AffineOperation::Affine, Self::F16) => ("affine_f16", "affine_f16_strided"),
            (AffineOperation::Affine, Self::F32) => ("affine_f32", "affine_f32_strided"),
            (AffineOperation::Affine, Self::BF16) => ("affine_bf16", "affine_bf16_strided"),
            (AffineOperation::Powf, Self::F16) => ("powf_f16", "powf_f16_strided"),
            (AffineOperation::Powf, Self::F32) => ("powf_f32", "powf_f32_strided"),
            (AffineOperation::Powf, Self::BF16) => ("powf_bf16", "powf_bf16_strided"),
            (AffineOperation::Elu, Self::F16) => ("elu_f16", "elu_f16_strided"),
            (AffineOperation::Elu, Self::F32) => ("elu_f32", "elu_f32_strided"),
            (AffineOperation::Elu, Self::BF16) => ("elu_bf16", "elu_bf16_strided"),
            _ => return None,
        })
    }
}
#[derive(Clone, Copy)]
pub(crate) enum AffineValue {
    Affine { mul: f32, add: f32 },
    Powf(f32),
    Elu(f32),
}
impl AffineValue {
    fn operation(self) -> AffineOperation {
        match self {
            Self::Affine { .. } => AffineOperation::Affine,
            Self::Powf(_) => AffineOperation::Powf,
            Self::Elu(_) => AffineOperation::Elu,
        }
    }
    fn scalars(self) -> (f32, Option<f32>) {
        match self {
            Self::Affine { mul, add } => (mul, Some(add)),
            Self::Powf(v) | Self::Elu(v) => (v, None),
        }
    }
}
pub(crate) struct PreparedAffinePipeline {
    pipeline: crate::kernels::ComputePipeline,
    device: crate::metal_backend::DeviceId,
    dtype_size: usize,
    dtype: AffineDType,
    style: ParamStyle,
    operation: AffineOperation,
    strided: bool,
}
#[derive(Debug)]
pub(crate) enum PreparedAffineError {
    Native(MetalKernelError),
    Backend(crate::Error),
    Params(crate::kernels::kernels::params::PreparedParamError),
}
impl PreparedAffinePipeline {
    pub(crate) fn native_pipeline(&self) -> &crate::kernels::ComputePipeline {
        &self.pipeline
    }
    /// # Safety
    /// Sealed operation/dtype choose the actual shader and width together.
    /// This cold catalog owns pipeline/style/EOS policy and full
    /// predicted constructor terms; disposal occurs outside EntryState/EDIT.
    /// No operation creates a new pipeline or capsule on the warm path.
    pub(crate) unsafe fn cold(
        device: &crate::metal_backend::MetalDevice,
        dtype: AffineDType,
        style: ParamStyle,
        operation: AffineOperation,
        strided: bool,
    ) -> Result<Self, MetalKernelError> {
        if core::mem::size_of::<usize>() != 8 {
            return Err(MetalKernelError::QueueUnavailable(
                "affine plan scalar schema mismatch",
            ));
        }
        let names = dtype
            .names(operation)
            .ok_or(MetalKernelError::QueueUnavailable(
                "affine operation/dtype unsupported",
            ))?;
        let name = if strided { names.1 } else { names.0 };
        let dtype_size = dtype.bytes();
        let pipeline = device.kernels().load_pipeline(
            device.device(),
            Source::Affine,
            style.kernel_name(name),
        )?;
        Ok(Self {
            pipeline,
            device: device.id(),
            dtype_size,
            dtype,
            style,
            operation,
            strided,
        })
    }
}

/// # Safety
/// Genuine planned rich input/output originate from the actual catalog/context;
/// initialized reads and mutation exclusion hold. All terminal/future-emission
/// users (including ICB/IO and implicit EOS flag) have corresponding custody.
/// Existing uint loop i += ceil(dim/W), its final increment and padded grid IDs
/// cannot wrap. The safe factory must prove this or select a validated widened
/// variant; usize extent checks below do NOT certify shader integer semantics.
/// This is not a default Legacy refusal/selector or safe raw interoperability.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_affine_contiguous_prepared(
    device: &crate::metal_backend::MetalDevice,
    plan: &PreparedAffinePipeline,
    value: AffineValue,
    length: usize,
    input: &crate::kernels::metal::BufferHandle,
    input_offset: usize,
    output: &crate::kernels::metal::BufferHandle,
    output_offset: usize,
    params: &mut crate::kernels::kernels::params::PreparedParamSlot,
    context: &crate::memory_admission::AllocationContext,
) -> Result<(), PreparedAffineError> {
    use crate::kernels::metal::encoder::InlineArgument as Scalar;
    use crate::kernels::utils::{dispatch_prepared_arguments, TrackedBufferArgument as Arg};
    let invalid = |s| PreparedAffineError::Native(MetalKernelError::QueueUnavailable(s));
    if plan.strided || plan.device != device.id() || plan.operation != value.operation() {
        return Err(invalid("affine operation/plan mismatch"));
    }
    context.check_open().map_err(|e| {
        PreparedAffineError::Params(
            crate::kernels::kernels::params::PreparedParamError::Admission(e),
        )
    })?;
    let extent = length
        .checked_mul(plan.dtype_size)
        .ok_or_else(|| invalid("affine extent overflow"))?;
    if input_offset
        .checked_add(extent)
        .is_none_or(|n| n > input.length())
        || output_offset
            .checked_add(extent)
            .is_none_or(|n| n > output.length())
    {
        return Err(invalid("affine extent exceeds originating view"));
    }
    if length == 0 {
        return Ok(());
    }
    let tiles = length.div_ceil(get_tile_size(plan.dtype_size));
    let width = plan.pipeline.max_total_threads_per_threadgroup().min(tiles);
    if width == 0 {
        return Err(invalid("affine pipeline width is zero"));
    }
    let groups = crate::kernels::MTLSize {
        width: tiles.div_ceil(width),
        height: 1,
        depth: 1,
    };
    let threads = crate::kernels::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let n = length.to_ne_bytes();
    let (mul, add) = value.scalars();
    let mul = mul.to_ne_bytes();
    let add = add.map(f32::to_ne_bytes);
    // All padding initialized, including ScaleParams' trailing four bytes.
    let mut block = [0u8; 16];
    block[..8].copy_from_slice(&n);
    block[8..12].copy_from_slice(&mul);
    if let Some(add) = add {
        block[12..16].copy_from_slice(&add);
    }
    let packed = if plan.style == ParamStyle::Packed {
        Some(
            params
                .prepare(&block, context, device.commands())
                .map_err(PreparedAffineError::Params)?,
        )
    } else {
        None
    };
    // Last-created guard drops before temporary prepared owners on every path.
    let encoder = device
        .command_encoder()
        .map_err(PreparedAffineError::Backend)?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    if let Some(block) = packed.as_ref() {
        // Actual Affine/Scale packed schema: block0,input1,output2.
        unsafe {
            dispatch_prepared_arguments(
                encoder,
                &plan.pipeline,
                [
                    Arg {
                        index: 0,
                        owner: block,
                        offset: 0,
                        write: false,
                    },
                    Arg {
                        index: 1,
                        owner: input,
                        offset: input_offset,
                        write: false,
                    },
                    Arg {
                        index: 2,
                        owner: output,
                        offset: output_offset,
                        write: true,
                    },
                ],
                &[],
                groups,
                threads,
            )
        }
        .map_err(PreparedAffineError::Native)
    } else {
        let mut scalars = [
            Scalar {
                index: 0,
                bytes: &n,
            },
            Scalar {
                index: 1,
                bytes: &mul,
            },
            Scalar {
                index: 2,
                bytes: &[],
            },
        ];
        if let Some(add) = add.as_ref() {
            scalars[2].bytes = add;
        }
        let data = if add.is_some() { 3 } else { 2 };
        unsafe {
            dispatch_prepared_arguments(
                encoder,
                &plan.pipeline,
                [
                    Arg {
                        index: data,
                        owner: input,
                        offset: input_offset,
                        write: false,
                    },
                    Arg {
                        index: data + 1,
                        owner: output,
                        offset: output_offset,
                        write: true,
                    },
                ],
                &scalars[..data],
                groups,
                threads,
            )
        }
        .map_err(PreparedAffineError::Native)
    }
}

/// # Safety
/// Same real catalog/terminal/emission custody as contiguous counterpart. Shader
/// tid and get_strided_index are uint: factory must validate rank/length/max-index
/// and dispatch IDs against that exact shader or use a separately validated
/// widened variant. Both styles and oversized Split arrays remain supported.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_affine_strided_prepared(
    device: &crate::metal_backend::MetalDevice,
    plan: &PreparedAffinePipeline,
    value: AffineValue,
    shape: &[usize],
    strides: &[usize],
    input: &crate::kernels::metal::BufferHandle,
    input_offset: usize,
    output: &crate::kernels::metal::BufferHandle,
    output_offset: usize,
    params: &mut crate::kernels::kernels::params::PreparedParamSlot,
    dims_slot: &mut crate::kernels::kernels::params::PreparedParamSlot,
    strides_slot: &mut crate::kernels::kernels::params::PreparedParamSlot,
    context: &crate::memory_admission::AllocationContext,
) -> Result<(), PreparedAffineError> {
    use crate::kernels::metal::encoder::InlineArgument as Scalar;
    use crate::kernels::utils::{dispatch_prepared_arguments, TrackedBufferArgument as Arg};
    let invalid = |s| PreparedAffineError::Native(MetalKernelError::QueueUnavailable(s));
    if !plan.strided
        || plan.device != device.id()
        || plan.operation != value.operation()
        || shape.len() != strides.len()
    {
        return Err(invalid("strided affine operation/plan mismatch"));
    }
    context.check_open().map_err(|e| {
        PreparedAffineError::Params(
            crate::kernels::kernels::params::PreparedParamError::Admission(e),
        )
    })?;
    let length = shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| invalid("strided affine length overflow"))?;
    if length == 0 {
        return Ok(());
    }
    let max_index = shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |n, (&d, &s)| {
            (d - 1).checked_mul(s).and_then(|v| n.checked_add(v))
        })
        .ok_or_else(|| invalid("strided affine index overflow"))?;
    let input_bytes = max_index
        .checked_add(1)
        .and_then(|n| n.checked_mul(plan.dtype_size))
        .ok_or_else(|| invalid("strided affine input extent overflow"))?;
    let output_bytes = length
        .checked_mul(plan.dtype_size)
        .ok_or_else(|| invalid("strided affine output extent overflow"))?;
    if input_offset
        .checked_add(input_bytes)
        .is_none_or(|n| n > input.length())
        || output_offset
            .checked_add(output_bytes)
            .is_none_or(|n| n > output.length())
    {
        return Err(invalid("strided affine extent exceeds originating view"));
    }
    let width = plan
        .pipeline
        .max_total_threads_per_threadgroup()
        .min(length);
    if width == 0 {
        return Err(invalid("strided affine pipeline width is zero"));
    }
    let groups = crate::kernels::MTLSize {
        width: length.div_ceil(width),
        height: 1,
        depth: 1,
    };
    let threads = crate::kernels::MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let n = length.to_ne_bytes();
    let rank = shape.len().to_ne_bytes();
    let (mul, add) = value.scalars();
    let mul = mul.to_ne_bytes();
    let add = add.map(f32::to_ne_bytes);
    // Initialized native-endian64 host arrays, not Shared mapped aliases.
    let dims = unsafe {
        std::slice::from_raw_parts(shape.as_ptr().cast::<u8>(), std::mem::size_of_val(shape))
    };
    let strides = unsafe {
        std::slice::from_raw_parts(
            strides.as_ptr().cast::<u8>(),
            std::mem::size_of_val(strides),
        )
    };
    let mut block = [0u8; 24];
    block[..8].copy_from_slice(&n);
    block[8..16].copy_from_slice(&rank);
    block[16..20].copy_from_slice(&mul);
    if let Some(add) = add {
        block[20..].copy_from_slice(&add);
    }
    let packed = if plan.style == ParamStyle::Packed {
        Some(
            params
                .prepare(&block, context, device.commands())
                .map_err(PreparedAffineError::Params)?,
        )
    } else {
        None
    };
    let arrays = if plan.style == ParamStyle::Packed || dims.is_empty() || dims.len() >= 4096 {
        Some((
            dims_slot
                .prepare(dims, context, device.commands())
                .map_err(PreparedAffineError::Params)?,
            strides_slot
                .prepare(strides, context, device.commands())
                .map_err(PreparedAffineError::Params)?,
        ))
    } else {
        None
    };
    let encoder = device
        .command_encoder()
        .map_err(PreparedAffineError::Backend)?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    if let Some(block) = packed.as_ref() {
        let (dims, strides) = arrays
            .as_ref()
            .expect("packed arrays prepared before EntryState");
        unsafe {
            dispatch_prepared_arguments(
                encoder,
                &plan.pipeline,
                [
                    Arg {
                        index: 0,
                        owner: block,
                        offset: 0,
                        write: false,
                    },
                    Arg {
                        index: 1,
                        owner: dims,
                        offset: 0,
                        write: false,
                    },
                    Arg {
                        index: 2,
                        owner: strides,
                        offset: 0,
                        write: false,
                    },
                    Arg {
                        index: 3,
                        owner: input,
                        offset: input_offset,
                        write: false,
                    },
                    Arg {
                        index: 4,
                        owner: output,
                        offset: output_offset,
                        write: true,
                    },
                ],
                &[],
                groups,
                threads,
            )
        }
        .map_err(PreparedAffineError::Native)
    } else {
        let mut scalars = [
            Scalar {
                index: 0,
                bytes: &n,
            },
            Scalar {
                index: 1,
                bytes: &rank,
            },
            Scalar {
                index: 4,
                bytes: &mul,
            },
            Scalar {
                index: 5,
                bytes: &[],
            },
            Scalar {
                index: 2,
                bytes: dims,
            },
            Scalar {
                index: 3,
                bytes: strides,
            },
        ];
        if let Some(add) = add.as_ref() {
            scalars[3].bytes = add;
        }
        let scalar_count = if add.is_some() { 4 } else { 3 };
        let data = if add.is_some() { 6 } else { 5 };
        if let Some((dims, strides)) = arrays.as_ref() {
            unsafe {
                dispatch_prepared_arguments(
                    encoder,
                    &plan.pipeline,
                    [
                        Arg {
                            index: 2,
                            owner: dims,
                            offset: 0,
                            write: false,
                        },
                        Arg {
                            index: 3,
                            owner: strides,
                            offset: 0,
                            write: false,
                        },
                        Arg {
                            index: data,
                            owner: input,
                            offset: input_offset,
                            write: false,
                        },
                        Arg {
                            index: data + 1,
                            owner: output,
                            offset: output_offset,
                            write: true,
                        },
                    ],
                    &scalars[..scalar_count],
                    groups,
                    threads,
                )
            }
            .map_err(PreparedAffineError::Native)
        } else {
            // Scale has one fewer scalar; move arrays into contiguous initialized
            // stack prefix without reevaluating any caller expression.
            if add.is_none() {
                scalars[3] = Scalar {
                    index: 2,
                    bytes: dims,
                };
                scalars[4] = Scalar {
                    index: 3,
                    bytes: strides,
                };
            }
            let scalar_count = if add.is_some() { 6 } else { 5 };
            unsafe {
                dispatch_prepared_arguments(
                    encoder,
                    &plan.pipeline,
                    [
                        Arg {
                            index: data,
                            owner: input,
                            offset: input_offset,
                            write: false,
                        },
                        Arg {
                            index: data + 1,
                            owner: output,
                            offset: output_offset,
                            write: true,
                        },
                    ],
                    &scalars[..scalar_count],
                    groups,
                    threads,
                )
            }
            .map_err(PreparedAffineError::Native)
        }
    }
}

/// Actual initialized-rich input -> authentic output -> terminal-owned plan route.
/// This opt-in constructor does not reinterpret legacy Tensor storage.
/// # Safety
/// Input/position originate on this SAME device/namespace/context and their cold
/// full pricing and native/raw/foreign/IO/CPU-use closure contracts hold. Existing
/// capture/ICB users are not activated by this route; all escaped emissions retain
/// the original generation page. No concurrent CPU/native mutation of input.
/// Complete default/supplied activation remains a factory obligation.
pub(crate) unsafe fn call_affine_catalog_f32(
    device: &crate::metal_backend::MetalDevice,
    input: &crate::metal_backend::PreparedAffineInput,
    position: &crate::metal_backend::DirectOutputPosition,
    context: &crate::memory_admission::AllocationContext,
    mul: f32,
    add: f32,
) -> Result<crate::kernels::metal::BufferHandle, crate::Error> {
    unsafe {
        call_affine_catalog_f32_under(
            device,
            input,
            position,
            &context.operation_context(),
            mul,
            add,
        )
    }
}
/// # Safety
/// Same originating cold-price/closed-use/initialized-range obligations as
/// the AllocationContext compatibility wrapper; warm permission alone is
/// neither a native-use receipt nor a write-before-read certificate.
pub(crate) unsafe fn call_affine_catalog_f32_under(
    device: &crate::metal_backend::MetalDevice,
    input: &crate::metal_backend::PreparedAffineInput,
    position: &crate::metal_backend::DirectOutputPosition,
    context: &crate::memory_admission::OperationContext<'_>,
    mul: f32,
    add: f32,
) -> Result<crate::kernels::metal::BufferHandle, crate::Error> {
    use crate::kernels::metal::encoder::InlineArgument;
    let invalid = |message| crate::Error::wrap(MetalKernelError::QueueUnavailable(message));
    let length = input.elements();
    let extent = length
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or_else(|| invalid("prepared affine extent overflow"))?;
    if extent > input.handle().length() || extent > position.capacity() {
        return Err(invalid("prepared affine originating extent refused"));
    }
    // The authentic operation lease/root is owned before preparation and becomes
    // the installed readonly Plan; no method-return writer release is possible.
    let mut operation = position
        .try_operation_under(context, device.commands())
        .map_err(|e| crate::Error::Msg(format!("affine operation: {e:?}")))?;
    let output = unsafe { operation.checkout_output_under(context, device.commands()) }
        .map_err(|e| crate::Error::Msg(format!("affine output: {e:?}")))?;
    // Empty logical output performs no read, shader/grid/pipeline/EOS bind or
    // native recording. Genuine output custody still names the selected payer.
    // The uninstalled operation owner releases only its writer/root count; its
    // output rich header follows the actual never-submitted cleanup path.
    if length == 0 {
        return Ok(output);
    }
    let (style, groups, threads, packed) = {
        let mut preparation = operation
            .affine_preparation(None)
            .map_err(|e| crate::Error::Msg(format!("affine preparation: {e:?}")))?;
        let shader = preparation.shader();
        if shader.device != device.id()
            || shader.dtype != AffineDType::F32
            || shader.operation != AffineOperation::Affine
            || shader.strided
        {
            return Err(invalid("prepared affine F32 schema mismatch"));
        }
        let tiles = length.div_ceil(get_tile_size(shader.dtype_size));
        let width = shader
            .pipeline
            .max_total_threads_per_threadgroup()
            .min(tiles);
        if width == 0 {
            return Err(invalid("prepared affine pipeline width is zero"));
        }
        let groups = crate::kernels::MTLSize {
            width: tiles.div_ceil(width),
            height: 1,
            depth: 1,
        };
        let threads = crate::kernels::MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        // Exact existing MSL: dim/step and i/tid are uint. The final loop add is
        // evaluated even for the last valid element. Preserve it without wrap;
        // larger legal workloads need a separately validated widened shader.
        let grid = groups
            .width
            .checked_mul(width)
            .ok_or_else(|| invalid("affine grid overflow"))?;
        if length > u32::MAX as usize
            || grid == 0
            || grid - 1 > u32::MAX as usize
            || (length - 1)
                .checked_add(tiles)
                .is_none_or(|n| n > u32::MAX as usize)
        {
            return Err(invalid("affine uint shader needs widened geometry"));
        }
        let style = shader.style;
        let packed = if style == ParamStyle::Packed {
            let mut block = [0u8; 16];
            block[..8].copy_from_slice(&length.to_ne_bytes());
            block[8..12].copy_from_slice(&mul.to_ne_bytes());
            block[12..16].copy_from_slice(&add.to_ne_bytes());
            Some(
                preparation
                    .prepare_params_under(&block, context, device.commands())
                    .map_err(|e| crate::Error::Msg(format!("affine params: {e:?}")))?,
            )
        } else {
            None
        };
        (style, groups, threads, packed)
    }; // all mutable payload borrows end BEFORE consuming the writer
    let plan = operation.into_plan();
    let n = length.to_ne_bytes();
    let mul = mul.to_ne_bytes();
    let add = add.to_ne_bytes();
    // Created last: EntryState guard ends before local rich owners on refusal.
    let encoder = device.command_encoder()?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    if style == ParamStyle::Packed {
        let block = packed
            .as_ref()
            .expect("packed block prepared before encoder");
        // SAFETY: exact sealed AffineParams schema: block0,input1,output2;
        // all three counted Buffer pins plus WHOLE Plan install before effects.
        unsafe {
            encoder.dispatch_affine_arguments_tracked::<3>(
                plan,
                &[
                    (0, &**block, 0, false),
                    (1, &**input.handle(), 0, false),
                    (2, &*output, 0, true),
                ],
                &[],
                groups,
                threads,
            )
        }
        .map_err(crate::Error::wrap)?;
    } else {
        // SAFETY: exact Split schema size0,mul1,add2,input3,output4.
        unsafe {
            encoder.dispatch_affine_arguments_tracked::<2>(
                plan,
                &[(3, &**input.handle(), 0, false), (4, &*output, 0, true)],
                &[
                    InlineArgument {
                        index: 0,
                        bytes: &n,
                    },
                    InlineArgument {
                        index: 1,
                        bytes: &mul,
                    },
                    InlineArgument {
                        index: 2,
                        bytes: &add,
                    },
                ],
                groups,
                threads,
            )
        }
        .map_err(crate::Error::wrap)?;
    }
    Ok(output)
}
