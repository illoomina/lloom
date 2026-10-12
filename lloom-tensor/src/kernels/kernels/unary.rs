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

/// Plan-owned shader/style resolution. Packed naming and pipeline-cache/native
/// setup happen ONCE at the cold plan boundary, never in the warm dispatch.
pub(crate) struct PreparedUnaryPipeline {
    pipeline: crate::kernels::ComputePipeline,
    device: crate::metal_backend::DeviceId,
    dtype_size: usize,
    style: ParamStyle,
    kind: PreparedUnaryKind,
}
#[derive(PartialEq, Eq)]
enum PreparedUnaryKind {
    Contiguous,
    Strided,
}
impl PreparedUnaryPipeline {
    /// # Safety
    /// Exact name resolves a unary_body (not const_set/copy2d), with this dtype
    /// width. Plan/native/control prices and finalization are cold-owned.
    pub(crate) unsafe fn cold_contiguous(
        device: &crate::metal_backend::MetalDevice,
        name: contiguous::Kernel,
        dtype_size: usize,
        style: ParamStyle,
    ) -> Result<Self, MetalKernelError> {
        unsafe {
            Self::cold(
                device,
                name.0,
                dtype_size,
                style,
                PreparedUnaryKind::Contiguous,
            )
        }
    }
    /// # Safety
    /// Exact name resolves matching unary_strided_body, with this dtype width;
    /// same cold price/finalization obligations as cold_contiguous.
    pub(crate) unsafe fn cold_strided(
        device: &crate::metal_backend::MetalDevice,
        name: strided::Kernel,
        dtype_size: usize,
        style: ParamStyle,
    ) -> Result<Self, MetalKernelError> {
        unsafe {
            Self::cold(
                device,
                name.0,
                dtype_size,
                style,
                PreparedUnaryKind::Strided,
            )
        }
    }
    unsafe fn cold(
        device: &crate::metal_backend::MetalDevice,
        name: &'static str,
        dtype_size: usize,
        style: ParamStyle,
        kind: PreparedUnaryKind,
    ) -> Result<Self, MetalKernelError> {
        if !matches!(dtype_size, 1 | 2 | 4 | 8) || core::mem::size_of::<usize>() != 8 {
            return Err(MetalKernelError::QueueUnavailable(
                "prepared unary native scalar schema mismatch",
            ));
        }
        let pipeline = device.kernels().load_pipeline(
            device.device(),
            Source::Unary,
            style.kernel_name(name),
        )?;
        Ok(Self {
            pipeline,
            device: device.id(),
            dtype_size,
            style,
            kind,
        })
    }
}

/// Inactive complete-owner counterpart of the existing contiguous unary path.
/// Both Split and Packed remain expressible; no old kernel/ICB route is removed.
/// # Safety
/// The typed kernel name and dtype_size describe the SAME unary shader, including
/// element width. Existing uint shader variant requires dim representable as
/// uint, and every i += ceil(dim/W) (including the final loop update) and grid
/// tid representable without wrap. A widened variant needs separate validation;
/// usize bounds below do not discharge this shader arithmetic obligation.
/// Both originating handles, the backend namespace, plan storage
/// and its predicted metadata/native terms belong to the caller's closed use
/// route. Real M3/M4 and escaped ICB/IO owners keep all counted uses to terminal
/// plus future-emission closure. Default selection is NOT enabled by this method.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_unary_contiguous_prepared(
    device: &crate::metal_backend::MetalDevice,
    plan: &PreparedUnaryPipeline,
    length: usize,
    input: &crate::kernels::metal::BufferHandle,
    input_offset: usize,
    output: &crate::kernels::metal::BufferHandle,
    params: &mut crate::kernels::kernels::params::PreparedParamSlot,
    context: &crate::memory_admission::AllocationContext,
) -> Result<(), PreparedUnaryError> {
    use crate::kernels::metal::encoder::InlineArgument;
    use crate::kernels::utils::{dispatch_prepared_arguments, TrackedBufferArgument};
    let invalid = |reason| PreparedUnaryError::Native(MetalKernelError::QueueUnavailable(reason));
    if plan.kind != PreparedUnaryKind::Contiguous
        || plan.device != device.id()
        || core::mem::size_of::<usize>() != 8
    {
        return Err(invalid("prepared unary plan/native scalar schema mismatch"));
    }
    context.check_open().map_err(|e| {
        PreparedUnaryError::Preparation(
            crate::kernels::kernels::params::PreparedParamError::Admission(e),
        )
    })?;
    let dtype_size = plan.dtype_size;
    let style = plan.style;
    let extent = length
        .checked_mul(dtype_size)
        .ok_or_else(|| invalid("prepared unary extent overflow"))?;
    if input_offset
        .checked_add(extent)
        .is_none_or(|end| end > input.length())
        || extent > output.length()
    {
        return Err(invalid("prepared unary extent exceeds originating view"));
    }
    if length == 0 {
        return Ok(());
    }
    let pipeline = &plan.pipeline;
    let tiles = length.div_ceil(get_tile_size(dtype_size));
    let width = pipeline.max_total_threads_per_threadgroup().min(tiles);
    if width == 0 {
        return Err(invalid("prepared unary pipeline has zero thread width"));
    }
    let groups = MTLSize {
        width: tiles.div_ceil(width),
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let length_bytes = length.to_ne_bytes();
    // Genuine staged creator preparation is BEFORE CommandsGuard/EntryState and
    // before original encoder EDIT. Typed Busy/Full reaches the plan/refill caller
    // without hiding its recovery type or consuming native generation credits.
    let packed = if style == ParamStyle::Packed {
        Some(
            params
                .prepare(&length_bytes, context, device.commands())
                .map_err(PreparedUnaryError::Preparation)?,
        )
    } else {
        None
    };
    // Declared after packed: on every refusal/drop/unwind the native EntryState
    // guard is released BEFORE this temporary rich Handle is released.
    let encoder = device
        .command_encoder()
        .map_err(PreparedUnaryError::Backend)?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    if let Some(packed) = packed.as_ref() {
        // SAFETY: exact _packed unary signature: block0={size_t dim}, input1,
        // output2; native padding is initialized by PreparedStagingSlot. Same
        // caller-validated shader width and checked extents/geometry above.
        unsafe {
            dispatch_prepared_arguments(
                encoder,
                pipeline,
                [
                    TrackedBufferArgument {
                        index: 0,
                        owner: packed,
                        offset: 0,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 1,
                        owner: input,
                        offset: input_offset,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 2,
                        owner: output,
                        offset: 0,
                        write: true,
                    },
                ],
                &[],
                groups,
                threads,
            )
        }
        .map_err(PreparedUnaryError::Native)
    } else {
        // SAFETY: exact Split unary signature: size_t dim0, input1, output2;
        // no arbitrary EncoderParam evaluation or captured temporary exists.
        unsafe {
            dispatch_prepared_arguments(
                encoder,
                pipeline,
                [
                    TrackedBufferArgument {
                        index: 1,
                        owner: input,
                        offset: input_offset,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 2,
                        owner: output,
                        offset: 0,
                        write: true,
                    },
                ],
                &[InlineArgument {
                    index: 0,
                    bytes: &length_bytes,
                }],
                groups,
                threads,
            )
        }
        .map_err(PreparedUnaryError::Native)
    }
}
/// Complete array/staging counterpart: dims and strides are independent native
/// users and join the same exact EDIT tuple as the packed block and data.
/// # Safety
/// Same creator/shader/native-use contract as call_unary_contiguous_prepared.
/// Shader is the matching strided unary body, not const_set/copy2d; dtype_size
/// and all element/address arithmetic agree with that body's types. Existing
/// uint tid/get_strided_index requires representable length/rank/max index and
/// grid IDs; usize extent checks are NOT proof of that shader-width contract.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_unary_strided_prepared(
    device: &crate::metal_backend::MetalDevice,
    plan: &PreparedUnaryPipeline,
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
) -> Result<(), PreparedUnaryError> {
    use crate::kernels::metal::encoder::InlineArgument;
    use crate::kernels::utils::{dispatch_prepared_arguments, TrackedBufferArgument};
    let invalid = |reason| PreparedUnaryError::Native(MetalKernelError::QueueUnavailable(reason));
    if plan.kind != PreparedUnaryKind::Strided
        || plan.device != device.id()
        || shape.len() != strides.len()
        || core::mem::size_of::<usize>() != 8
    {
        return Err(invalid("prepared strided unary schema/shape mismatch"));
    }
    context.check_open().map_err(|e| {
        PreparedUnaryError::Preparation(
            crate::kernels::kernels::params::PreparedParamError::Admission(e),
        )
    })?;
    let dtype_size = plan.dtype_size;
    let style = plan.style;
    let length = shape
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim))
        .ok_or_else(|| invalid("prepared strided unary element count overflow"))?;
    if length == 0 {
        return Ok(());
    }
    let max_index = shape
        .iter()
        .zip(strides)
        .try_fold(0usize, |index, (&dim, &stride)| {
            (dim - 1)
                .checked_mul(stride)
                .and_then(|term| index.checked_add(term))
        })
        .ok_or_else(|| invalid("prepared strided unary input offset overflow"))?;
    let input_extent = max_index
        .checked_add(1)
        .and_then(|n| n.checked_mul(dtype_size))
        .ok_or_else(|| invalid("prepared strided unary input extent overflow"))?;
    let output_extent = length
        .checked_mul(dtype_size)
        .ok_or_else(|| invalid("prepared strided unary output extent overflow"))?;
    if input_offset
        .checked_add(input_extent)
        .is_none_or(|end| end > input.length())
        || output_offset
            .checked_add(output_extent)
            .is_none_or(|end| end > output.length())
    {
        return Err(invalid(
            "prepared strided unary extent exceeds originating view",
        ));
    }
    let pipeline = &plan.pipeline;
    let width = pipeline.max_total_threads_per_threadgroup().min(length);
    if width == 0 {
        return Err(invalid("prepared strided unary zero thread width"));
    }
    let groups = MTLSize {
        width: length.div_ceil(width),
        height: 1,
        depth: 1,
    };
    let threads = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let length_bytes = length.to_ne_bytes();
    let rank_bytes = shape.len().to_ne_bytes();
    // SAFETY: usize slices contain only initialized, padding-free integers;
    // native endian 64-bit representation matches these exact MSL size_t arrays.
    // These are host borrows, never an unsafe view into staged Shared storage.
    let dims_bytes = unsafe {
        std::slice::from_raw_parts(shape.as_ptr().cast::<u8>(), std::mem::size_of_val(shape))
    };
    let strides_bytes = unsafe {
        std::slice::from_raw_parts(
            strides.as_ptr().cast::<u8>(),
            std::mem::size_of_val(strides),
        )
    };
    let mut block = [0u8; 16];
    block[..8].copy_from_slice(&length_bytes);
    block[8..].copy_from_slice(&rank_bytes);
    // ALL staging preparations precede CommandsGuard/EDIT. A later member's
    // refusal retains truthful earlier prepared memo/shadow but submits nothing.
    // Busy/Full is returned typed for admitted spill/refill outside those guards.
    let packed = if style == ParamStyle::Packed {
        Some(
            params
                .prepare(&block, context, device.commands())
                .map_err(PreparedUnaryError::Preparation)?,
        )
    } else {
        None
    };
    // Apple setBytes is for data SMALLER than 4KiB. Preserve Split's shader:
    // larger (or empty) constant arrays bind genuine staged buffers at its
    // existing indices, rather than refusing the workload or changing style.
    let arrays = if style == ParamStyle::Packed || dims_bytes.is_empty() || dims_bytes.len() >= 4096
    {
        Some((
            dims_slot
                .prepare(dims_bytes, context, device.commands())
                .map_err(PreparedUnaryError::Preparation)?,
            strides_slot
                .prepare(strides_bytes, context, device.commands())
                .map_err(PreparedUnaryError::Preparation)?,
        ))
    } else {
        None
    };
    let encoder = device
        .command_encoder()
        .map_err(PreparedUnaryError::Backend)?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    if let Some(block) = packed.as_ref() {
        let (dims, strides) = arrays
            .as_ref()
            .expect("packed arrays prepared before guard");
        // SAFETY: actual unary_kernel_strided_packed signature: block0,
        // dims1, strides2, input3, output4. All five handles join one batch.
        unsafe {
            dispatch_prepared_arguments(
                encoder,
                pipeline,
                [
                    TrackedBufferArgument {
                        index: 0,
                        owner: block,
                        offset: 0,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 1,
                        owner: dims,
                        offset: 0,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 2,
                        owner: strides,
                        offset: 0,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 3,
                        owner: input,
                        offset: input_offset,
                        write: false,
                    },
                    TrackedBufferArgument {
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
        .map_err(PreparedUnaryError::Native)
    } else if let Some((dims, strides)) = arrays.as_ref() {
        // SAFETY: unchanged Split shader constant pointers accept MTLBuffer
        // data. Exact indices2/3 replace oversized inline arrays; scalar0/1 and
        // input4/output5 stay unchanged. All FOUR counted pins precede effects.
        unsafe {
            dispatch_prepared_arguments(
                encoder,
                pipeline,
                [
                    TrackedBufferArgument {
                        index: 2,
                        owner: dims,
                        offset: 0,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 3,
                        owner: strides,
                        offset: 0,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 4,
                        owner: input,
                        offset: input_offset,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 5,
                        owner: output,
                        offset: output_offset,
                        write: true,
                    },
                ],
                &[
                    InlineArgument {
                        index: 0,
                        bytes: &length_bytes,
                    },
                    InlineArgument {
                        index: 1,
                        bytes: &rank_bytes,
                    },
                ],
                groups,
                threads,
            )
        }
        .map_err(PreparedUnaryError::Native)
    } else {
        // SAFETY: actual Split signature: dim0, rank1, dims2, strides3,
        // input4, output5. Native setBytes consumes these initialized borrows
        // synchronously; no GPU reference to the host slices is retained.
        unsafe {
            dispatch_prepared_arguments(
                encoder,
                pipeline,
                [
                    TrackedBufferArgument {
                        index: 4,
                        owner: input,
                        offset: input_offset,
                        write: false,
                    },
                    TrackedBufferArgument {
                        index: 5,
                        owner: output,
                        offset: output_offset,
                        write: true,
                    },
                ],
                &[
                    InlineArgument {
                        index: 0,
                        bytes: &length_bytes,
                    },
                    InlineArgument {
                        index: 1,
                        bytes: &rank_bytes,
                    },
                    InlineArgument {
                        index: 2,
                        bytes: dims_bytes,
                    },
                    InlineArgument {
                        index: 3,
                        bytes: strides_bytes,
                    },
                ],
                groups,
                threads,
            )
        }
        .map_err(PreparedUnaryError::Native)
    }
}

#[derive(Debug)]
pub(crate) enum PreparedUnaryError {
    Preparation(crate::kernels::kernels::params::PreparedParamError),
    Native(MetalKernelError),
    Backend(crate::Error),
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_unary_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_unary_contiguous`], choosing how the scalars are bound.
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
pub unsafe fn call_unary_contiguous_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(encoder, (length, &input, Output::new(output)));
    }
    let _staged = finish_packed_params(device, encoder, style, UNARY_PARAMS_ALIGN)?;

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
pub unsafe fn call_unary_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: BufferOffset,
    strides: &[usize],
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_unary_strided`], choosing how the scalars are bound.
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
pub unsafe fn call_unary_strided_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
    }
    let _staged = finish_packed_params(device, encoder, style, UNARY_STRIDED_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_const_set_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: impl EncoderParam,
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_const_set_contiguous`], choosing how the scalars are bound.
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
pub unsafe fn call_const_set_contiguous_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (length, input, Output::from_buffer_offset(&output))
        );
    }
    let _staged = finish_packed_params(device, encoder, style, UNARY_PARAMS_ALIGN)?;

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
pub unsafe fn call_const_set_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: impl EncoderParam,
    strides: &[usize],
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_const_set_strided`], choosing how the scalars are bound.
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
pub unsafe fn call_const_set_strided_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
    }
    let _staged = finish_packed_params(device, encoder, style, UNARY_STRIDED_PARAMS_ALIGN)?;
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

pub mod copy_nd {
    pub struct Kernel(pub &'static str);
    pub const FLOAT: Kernel = Kernel("copy_nd_f32");
    pub const HALF: Kernel = Kernel("copy_nd_f16");
    pub const BFLOAT: Kernel = Kernel("copy_nd_bf16");
    pub const I64: Kernel = Kernel("copy_nd_i64");
    pub const I32: Kernel = Kernel("copy_nd_i32");
    pub const I16: Kernel = Kernel("copy_nd_i16");
    pub const U32: Kernel = Kernel("copy_nd_u32");
    pub const U8: Kernel = Kernel("copy_nd_u8");
}

/// `output[dst_strides · i] = input[src_strides · i]` over `shape`: a copy
/// between two strided layouts of one shape, one thread per element. The
/// offsets ride in the two `BufferOffset`s. See `copy_nd` in `unary.metal`.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_copy_nd(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: copy_nd::Kernel,
    shape: &[usize],
    input: BufferOffset,
    src_strides: &[usize],
    output: BufferOffset,
    dst_strides: &[usize],
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_copy_nd_with(
            device,
            ep,
            kernels,
            name,
            shape,
            input,
            src_strides,
            output,
            dst_strides,
            ParamStyle::default(),
        )
    }
}

/// As [`call_copy_nd`], choosing how the scalars are bound.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_copy_nd_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: copy_nd::Kernel,
    shape: &[usize],
    input: BufferOffset,
    src_strides: &[usize],
    output: BufferOffset,
    dst_strides: &[usize],
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
    debug_group!(encoder, "copy_nd {} elems={length}", name.0);
    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                length,
                num_dims,
                shape,
                src_strides,
                dst_strides,
                &input,
                Output::from_buffer_offset(&output)
            )
        );
    }
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

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_copy2d(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
}

/// As [`call_copy2d`], choosing how the scalars are bound.
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
pub unsafe fn call_copy2d_with(
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
    // SAFETY: this function's contract (# Safety above).
    unsafe {
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
    }
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
// The SiLU variant axis.
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
/// A local enum rather than `lloom_tensor::DType` because this crate does not
/// depend on `candle-core` -- the same reason `GemmDType` exists here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiluDType {
    F16,
    F32,
    BF16,
}

/// Cold sealed byte-transfer plan. Never resolves a dynamic scalar/type name.
/// Three direct params positions and optional overlap scratch come from actual
/// backend admission, not a raw Device/global payer lookup or legacy pool hit.
pub(crate) struct PreparedByteCopyPlan {
    pipeline: crate::kernels::ComputePipeline,
    device: crate::metal_backend::DeviceId,
    style: ParamStyle,
    params: [crate::kernels::kernels::params::PreparedParamSlot; 3],
    scratch: Option<crate::kernels::metal::PreparedPoolSlot>,
}
#[derive(Debug)]
pub(crate) enum PreparedByteCopyError {
    Native(MetalKernelError),
    Backend(crate::Error),
    Params(crate::kernels::kernels::params::PreparedParamError),
    Scratch(crate::kernels::metal::StagingError),
}
impl PreparedByteCopyPlan {
    /// # Safety
    /// Preadmitted positions use the SAME actual backend/context; prediction
    /// includes their owned terms once and preserves terminal/emission custody.
    /// Scratch placement, when present, is independent from every eligible data
    /// allocation (actual allocator range ownership, not pointer inequality).
    /// Final plan/pipeline disposal occurs off engine/EDIT guards. This private
    /// inactive plan is NOT a default selector or complete factory-price claim.
    pub(crate) unsafe fn cold(
        device: &crate::metal_backend::MetalDevice,
        style: ParamStyle,
        params: [crate::kernels::kernels::params::PreparedParamSlot; 3],
        scratch: Option<crate::kernels::metal::PreparedPoolSlot>,
    ) -> Result<Self, MetalKernelError> {
        if core::mem::size_of::<usize>() != 8 {
            return Err(MetalKernelError::QueueUnavailable(
                "byte copy requires checked64 scalar schema",
            ));
        }
        // Copy is the original byte-blit semantic, even while decode EOS gating
        // is armed. Explicit constant=false avoids an implicit index30 native
        // user and prevents compute fallback silently skipping a legal blit.
        let constants = crate::kernels::ConstantValues::new(vec![(
            crate::kernels::metal::eos_gate::EOS_GATE_FUNCTION_CONSTANT,
            crate::kernels::metal::library::Value::Bool(false),
        )]);
        let pipeline = device.kernels().load_pipeline_with_constants(
            device.device(),
            Source::Unary,
            style.kernel_name("copy2d_u8"),
            Some(constants),
        )?;
        Ok(Self {
            pipeline,
            device: device.id(),
            style,
            params,
            scratch,
        })
    }
}

/// Concrete full-width byte-copy caller: aligned legal blit OR exact compute,
/// with independent scratch for overlap. Neither branch rounds a view/range.
/// # Safety
/// Source/destination are genuine rich creator handles in this closed tracked
/// route; source range is initialized and concurrent mutation is excluded by
/// the operation owner. Distinct native objects have actual disjoint allocator
/// placement unless the operation supplies an appropriate overlap scratch.
/// Plan's exact placement/cold-price/emission contracts hold. This is not safe
/// coercion of a default legacy Handle, nor an activated Tensor selector.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn call_byte_copy_prepared(
    device: &crate::metal_backend::MetalDevice,
    plan: &mut PreparedByteCopyPlan,
    source: &crate::kernels::metal::BufferHandle,
    source_offset: usize,
    destination: &crate::kernels::metal::BufferHandle,
    destination_offset: usize,
    bytes: usize,
    allow_blit: bool,
    context: &crate::memory_admission::AllocationContext,
) -> Result<(), PreparedByteCopyError> {
    use crate::kernels::metal::encoder::ByteCopyStep;
    use crate::native_binding_preflight::{byte_copy_route, byte_rectangles, ByteCopyRoute};
    let invalid =
        |reason| PreparedByteCopyError::Native(MetalKernelError::QueueUnavailable(reason));
    if plan.device != device.id() {
        return Err(invalid("byte-copy plan belongs to another device"));
    }
    context.check_open().map_err(|e| {
        PreparedByteCopyError::Params(
            crate::kernels::kernels::params::PreparedParamError::Admission(e),
        )
    })?;
    let route = byte_copy_route(
        source.base_offset(),
        source.length(),
        source_offset,
        destination.base_offset(),
        destination.length(),
        destination_offset,
        bytes,
        source.raw_ptr() == destination.raw_ptr(),
    )
    .ok_or_else(|| invalid("byte copy originating extent overflow"))?;
    let overlap = match route {
        ByteCopyRoute::Empty => return Ok(()),
        ByteCopyRoute::AlignedBlit { .. } if allow_blit => {
            // This genuine caller retains both owners through guarded alignment
            // validation/mints/installation/native copy; no legacy raw wrapper.
            let mut blit = device
                .blit_command_encoder()
                .map_err(PreparedByteCopyError::Backend)?;
            return blit
                .copy_from_handles_aligned(
                    source,
                    source_offset,
                    destination,
                    destination_offset,
                    bytes,
                )
                .map_err(PreparedByteCopyError::Native);
        }
        ByteCopyRoute::AlignedBlit { .. } => false,
        ByteCopyRoute::Compute { overlap, .. } => overlap,
    };
    let rectangles = byte_rectangles(bytes)
        .ok_or_else(|| invalid("byte-copy rectangle scalar schema refused"))?;
    // Acquire actual uninitialized rich scratch BEFORE EntryState/EDIT. Busy or
    // header Full reaches caller's preadmitted spill/refill, not permanent U8 cap.
    let scratch = if overlap {
        let slot = plan
            .scratch
            .as_ref()
            .ok_or_else(|| invalid("overlap needs admitted independent scratch"))?;
        if bytes > slot.capacity() {
            return Err(invalid("overlap scratch needs admitted larger tranche"));
        }
        Some(
            unsafe { slot.try_checkout(context, device.commands()) }
                .map_err(PreparedByteCopyError::Scratch)?,
        )
    } else {
        None
    };
    let mut packed: [Option<crate::kernels::metal::BufferHandle>; 3] =
        std::array::from_fn(|_| None);
    let mut count = 0;
    for (position, rectangle) in rectangles.iter().enumerate() {
        let Some(rectangle) = rectangle else {
            continue;
        };
        count += 1;
        if plan.style == ParamStyle::Packed {
            let values = [
                rectangle.rows as i64,
                rectangle.columns as i64,
                rectangle.columns as i64,
                rectangle.columns as i64,
            ];
            let mut block = [0u8; 32];
            for (i, value) in values.iter().enumerate() {
                block[i * 8..i * 8 + 8].copy_from_slice(&value.to_ne_bytes());
            }
            packed[position] = Some(
                plan.params[position]
                    .prepare(&block, context, device.commands())
                    .map_err(PreparedByteCopyError::Params)?,
            );
        }
    }
    let mut steps: [Option<ByteCopyStep>; 6] = std::array::from_fn(|_| None);
    let packed_base = if overlap { 3 } else { 2 };
    for (position, rectangle) in rectangles.iter().enumerate() {
        let Some(rectangle) = rectangle else {
            continue;
        };
        let parameter = (plan.style == ParamStyle::Packed).then_some(packed_base + position);
        let input_offset = source_offset
            .checked_add(rectangle.offset)
            .ok_or_else(|| invalid("copy source chunk offset overflow"))?;
        let output_offset = destination_offset
            .checked_add(rectangle.offset)
            .ok_or_else(|| invalid("copy destination chunk offset overflow"))?;
        steps[position] = Some(ByteCopyStep {
            source: 0,
            destination: if overlap { 2 } else { 1 },
            source_offset: input_offset,
            destination_offset: if overlap {
                rectangle.offset
            } else {
                output_offset
            },
            rows: rectangle.rows,
            columns: rectangle.columns,
            packed: parameter,
        });
        if overlap {
            steps[count + position] = Some(ByteCopyStep {
                source: 2,
                destination: 1,
                source_offset: rectangle.offset,
                destination_offset: output_offset,
                rows: rectangle.rows,
                columns: rectangle.columns,
                packed: parameter,
            });
        }
    }
    // Fixed initialized step storage, no per-operation Vec/Arc allocation. Keep
    // actual order: ALL source->scratch rectangles then ALL scratch->destination.
    let steps = steps.map(|step| {
        step.unwrap_or(ByteCopyStep {
            source: 0,
            destination: 1,
            source_offset: 0,
            destination_offset: 0,
            rows: 0,
            columns: 0,
            packed: None,
        })
    });
    let steps = &steps[..count * if overlap { 2 } else { 1 }];
    // Declared after all prepared owners: guard drops before those handles on
    // every refusal/unwind. No staging CPU rewrite can occur inside EDIT.
    let encoder = device
        .command_encoder()
        .map_err(PreparedByteCopyError::Backend)?;
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let result = match (scratch.as_ref(), plan.style) {
        (None, ParamStyle::Split) => unsafe {
            encoder.dispatch_byte_copy_steps(&plan.pipeline, &[source, destination], steps)
        },
        (Some(scratch), ParamStyle::Split) => unsafe {
            encoder.dispatch_byte_copy_steps(&plan.pipeline, &[source, destination, scratch], steps)
        },
        // Exactly count Packed pins, no fabricated unused native users. Array
        // sizes derive from <=3 rectangles plus2 data plus optional scratch.
        (None, ParamStyle::Packed) => match count {
            1 => unsafe {
                encoder.dispatch_byte_copy_steps(
                    &plan.pipeline,
                    &[source, destination, packed[0].as_ref().unwrap()],
                    steps,
                )
            },
            2 => unsafe {
                encoder.dispatch_byte_copy_steps(
                    &plan.pipeline,
                    &[
                        source,
                        destination,
                        packed[0].as_ref().unwrap(),
                        packed[1].as_ref().unwrap(),
                    ],
                    steps,
                )
            },
            _ => unsafe {
                encoder.dispatch_byte_copy_steps(
                    &plan.pipeline,
                    &[
                        source,
                        destination,
                        packed[0].as_ref().unwrap(),
                        packed[1].as_ref().unwrap(),
                        packed[2].as_ref().unwrap(),
                    ],
                    steps,
                )
            },
        },
        (Some(scratch), ParamStyle::Packed) => match count {
            1 => unsafe {
                encoder.dispatch_byte_copy_steps(
                    &plan.pipeline,
                    &[source, destination, scratch, packed[0].as_ref().unwrap()],
                    steps,
                )
            },
            2 => unsafe {
                encoder.dispatch_byte_copy_steps(
                    &plan.pipeline,
                    &[
                        source,
                        destination,
                        scratch,
                        packed[0].as_ref().unwrap(),
                        packed[1].as_ref().unwrap(),
                    ],
                    steps,
                )
            },
            _ => unsafe {
                encoder.dispatch_byte_copy_steps(
                    &plan.pipeline,
                    &[
                        source,
                        destination,
                        scratch,
                        packed[0].as_ref().unwrap(),
                        packed[1].as_ref().unwrap(),
                        packed[2].as_ref().unwrap(),
                    ],
                    steps,
                )
            },
        },
    };
    result.map_err(PreparedByteCopyError::Native)
}
