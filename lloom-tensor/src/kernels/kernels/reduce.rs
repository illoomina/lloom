use crate::kernels::linear_split;
use crate::kernels::utils::{BufferOffset, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, ComputePipeline, Device, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

use crate::kernels::kernels::params::{
    begin_packed_params, finish_packed_params, NormExtParams, NormParams, ParamStyle, ReduceParams,
    RopeIParams, RopeParams, RopeThdParams, SoftmaxParams,
};

/// Trailing alignment of each packed block, so its length matches the `sizeof`
/// the kernel sees. Taken from the Rust mirrors rather than written as
/// literals, and `reduce_params_layout_matches_metal` is what proves those
/// mirrors agree with `reduce.metal`.
const REDUCE_PARAMS_ALIGN: usize = core::mem::align_of::<ReduceParams>();
const SOFTMAX_PARAMS_ALIGN: usize = core::mem::align_of::<SoftmaxParams>();
const NORM_PARAMS_ALIGN: usize = core::mem::align_of::<NormParams>();
const NORM_EXT_PARAMS_ALIGN: usize = core::mem::align_of::<NormExtParams>();
const ROPE_I_PARAMS_ALIGN: usize = core::mem::align_of::<RopeIParams>();
const ROPE_PARAMS_ALIGN: usize = core::mem::align_of::<RopeParams>();
const ROPE_THD_PARAMS_ALIGN: usize = core::mem::align_of::<RopeThdParams>();

// A width mismatch this change surfaced, recorded because it is latent on the
// classical path rather than introduced here.
//
// `softmax_kernel`, `rms_norm_kernel` and `layer_norm_kernel` declare their
// scalars as `constant uint &` -- 4 bytes -- but `call_last_softmax`,
// `call_rms_norm` and `call_layer_norm` bound them as `usize`, which is 8. The
// classical path survives that only because `setBytes` writes 8 bytes into the
// argument slot and a little-endian `uint` read takes the low 4, which are the
// right ones for any length that fits in 32 bits.
//
// Packing turns it into a real error rather than a benign one: the block is
// built from each value's own width, so two `usize` fields occupy 16 bytes
// where the struct is 8, and every field after the first lands wrong. The
// parity test caught it on the first run, which is the argument for having it.
//
// The fix is to bind what the kernel declares -- `as u32`, which is what the
// reduce and arg-reduce families already did. That is a no-op for the classical
// path (the same low four bytes reach the same slot), and it is why the LFM2
// digests are unchanged.

// `ParamStyle`, `begin_packed_params` and `finish_packed_params` were declared
// here when `reduce.metal` was the only family carrying both styles. They moved
// to `kernels/params.rs` once other families followed (issues #40, #41, #42),
// because the capture protocol is identical and a copy per family would be the
// hand-sync `DESIGN.md` §8.1b exists to remove. `kernels/mod.rs` re-exports
// `ParamStyle` from there, so callers naming it through this module still
// resolve.

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_reduce_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_reduce_contiguous_with(
            device,
            ep,
            kernels,
            kernel_name,
            shape,
            out_length,
            input,
            output,
            ParamStyle::default(),
        )
    }
}

/// As [`call_reduce_contiguous`], choosing how the scalars are bound.
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
pub unsafe fn call_reduce_contiguous_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "reduce {kernel_name} length={length} out_length={out_length}"
    );

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                length as u32,
                num_dims as u32,
                shape.as_slice(),
                work_per_threadgroup as u32,
                &input,
                Output::new(output)
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, REDUCE_PARAMS_ALIGN)?;

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_reduce_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_reduce_strided_with(
            device,
            ep,
            kernels,
            kernel_name,
            shape,
            strides,
            out_length,
            input,
            output,
            ParamStyle::default(),
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_reduce_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    out_length: usize,
    input: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let length: usize = shape.iter().product();
    let num_dims = shape.len();
    let work_per_threadgroup = length / out_length;

    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "reduce_strided {kernel_name} length={length} out_length={out_length}"
    );

    let shape: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    let strides: Vec<u32> = strides.iter().map(|&x| x as u32).collect();
    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                length as u32,
                num_dims as u32,
                shape.as_slice(),
                strides.as_slice(),
                work_per_threadgroup as u32,
                &input,
                Output::new(output)
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, REDUCE_PARAMS_ALIGN)?;

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    encoder.dispatch_thread_groups(
        MTLSize {
            width: out_length,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_last_softmax(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_last_softmax_with(
            device,
            ep,
            kernels,
            kernel_name,
            length,
            elements,
            input,
            input_offset,
            output,
            ParamStyle::default(),
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_last_softmax_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements: usize,
    input: &Buffer,
    input_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let work_per_threadgroup = elements;

    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "softmax {kernel_name} length={length} elements={elements}"
    );

    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                length as u32,
                work_per_threadgroup as u32,
                (input, input_offset),
                Output::new(output)
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, SOFTMAX_PARAMS_ALIGN)?;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rms_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_rms_norm_with(
            device,
            ep,
            kernels,
            kernel_name,
            length,
            elements_to_sum,
            eps,
            input,
            input_offset,
            alpha,
            alpha_offset,
            output,
            ParamStyle::default(),
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rms_norm_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "rms_norm {kernel_name} length={length} elements_to_sum={elements_to_sum}"
    );

    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                length as u32,
                elements_to_sum as u32,
                (input, input_offset),
                Output::new(output),
                (alpha, alpha_offset),
                eps
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, NORM_PARAMS_ALIGN)?;
    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// The threadgroup width `call_rms_norm` dispatches for a row of `last_dim`
/// elements: `min(max_threads, (last_dim / 2).next_power_of_two())`. Stated
/// once, because the rotated form below requires it to equal `last_dim / 2`.
fn rms_norm_width(pipeline: &ComputePipeline, last_dim: usize) -> usize {
    std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (last_dim / 2).next_power_of_two(),
    )
}

/// The rows of a strided source: `shape` and `strides` describe the source as
/// the caller sees it; the last dimension is the row. Returns
/// `(numel, last_dim)`.
fn rows_of(shape: &[usize]) -> Result<(usize, usize), MetalKernelError> {
    let numel: usize = shape.iter().product();
    let last_dim = match shape.last() {
        Some(&d) if d > 0 => d,
        _ => {
            return Err(MetalKernelError::InvalidInput(format!(
                "rms_norm over a strided source needs a non-empty last dimension, got {shape:?}"
            )))
        }
    };
    Ok((numel, last_dim))
}

/// RMSNorm over a strided source (`rms_norm_kernel_strided`), the source read
/// through its own `shape`/`strides` and the normalised rows written
/// contiguously. Same reduction, block size and rounding as [`call_rms_norm`];
/// only the source addressing differs, so the output is the contiguous
/// kernel's over a copy, bit for bit.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rms_norm_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    eps: f32,
    input: BufferOffset,
    alpha: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_rms_norm_strided_with(
            device,
            ep,
            kernels,
            kernel_name,
            shape,
            strides,
            eps,
            input,
            alpha,
            output,
            ParamStyle::default(),
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rms_norm_strided_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: &[usize],
    eps: f32,
    input: BufferOffset,
    alpha: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let (numel, last_dim) = rows_of(shape)?;
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "rms_norm_strided {kernel_name} shape={shape:?} strides={strides:?}"
    );

    let shape_u32: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    let strides_u32: Vec<u32> = strides.iter().map(|&x| x as u32).collect();
    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                numel as u32,
                shape.len() as u32,
                shape_u32.as_slice(),
                strides_u32.as_slice(),
                last_dim as u32,
                0u32,
                eps,
                &input,
                Output::new(output),
                &alpha
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, NORM_EXT_PARAMS_ALIGN)?;

    let width = rms_norm_width(&pipeline, last_dim);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: numel / last_dim,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// RMSNorm and RoPE in one dispatch (`rms_norm_rope_kernel[_strided]`): each
/// row of `[b, h, t, d]` is normalised by `alpha` and rotated by the
/// `[t, d/2]` `cos`/`sin` rows at its own position, `rope`'s pairing
/// `(i, i + d/2)`. `strides` is `None` for a contiguous source.
///
/// Refused unless `d` is a power of two whose half fits one threadgroup: the
/// kernel's thread `tid` must hold exactly the pair the rotation reads, which
/// is the geometry `call_rms_norm` dispatches at those sizes and no other.
/// A refusal is an `Err`, so a caller falls back to the two-dispatch path
/// rather than reading a row the kernel did not write.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rms_norm_rope(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: Option<&[usize]>,
    t_len: usize,
    eps: f32,
    input: BufferOffset,
    alpha: BufferOffset,
    cos: BufferOffset,
    sin: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_rms_norm_rope_with(
            device,
            ep,
            kernels,
            kernel_name,
            shape,
            strides,
            t_len,
            eps,
            input,
            alpha,
            cos,
            sin,
            output,
            ParamStyle::default(),
        )
    }
}

/// Whether [`call_rms_norm_rope`] serves a row of `last_dim` elements on this
/// device: a power of two whose half is one threadgroup.
pub fn rms_norm_rope_applies(
    device: &Device,
    kernels: &Kernels,
    kernel_name: &'static str,
    last_dim: usize,
) -> Result<bool, MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    Ok(last_dim >= 2
        && last_dim.is_power_of_two()
        && rms_norm_width(&pipeline, last_dim) * 2 == last_dim)
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rms_norm_rope_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    shape: &[usize],
    strides: Option<&[usize]>,
    t_len: usize,
    eps: f32,
    input: BufferOffset,
    alpha: BufferOffset,
    cos: BufferOffset,
    sin: BufferOffset,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let (numel, last_dim) = rows_of(shape)?;
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let width = rms_norm_width(&pipeline, last_dim);
    if t_len == 0 || !last_dim.is_power_of_two() || width * 2 != last_dim {
        return Err(MetalKernelError::InvalidInput(format!(
            "{kernel_name}: a row of {last_dim} at t_len {t_len} is not a power of two whose \
             half is one threadgroup ({width}); take the two-dispatch path"
        )));
    }
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "rms_norm_rope {kernel_name} shape={shape:?} strides={strides:?} t_len={t_len}"
    );

    let shape_u32: Vec<u32> = shape.iter().map(|&x| x as u32).collect();
    let _capture = begin_packed_params(encoder, style);
    match strides {
        Some(strides) => {
            let strides_u32: Vec<u32> = strides.iter().map(|&x| x as u32).collect();
            // SAFETY: this function's contract (# Safety above).
            unsafe {
                set_params!(
                    encoder,
                    (
                        numel as u32,
                        shape.len() as u32,
                        shape_u32.as_slice(),
                        strides_u32.as_slice(),
                        last_dim as u32,
                        t_len as u32,
                        eps,
                        &input,
                        Output::new(output),
                        &alpha,
                        &cos,
                        &sin
                    )
                );
            }
        }
        None => {
            // SAFETY: this function's contract (# Safety above).
            unsafe {
                set_params!(
                    encoder,
                    (
                        numel as u32,
                        shape.len() as u32,
                        last_dim as u32,
                        t_len as u32,
                        eps,
                        &input,
                        Output::new(output),
                        &alpha,
                        &cos,
                        &sin
                    )
                );
            }
        }
    }
    let _staged = finish_packed_params(device, encoder, style, NORM_EXT_PARAMS_ALIGN)?;

    encoder.dispatch_thread_groups(
        MTLSize {
            width: numel / last_dim,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// RMSNorm's reduction half only: writes `rsqrt(mean(x^2) + eps)`, one `f32`
/// per row, and no normalized vector (issue #266, `DESIGN.md` §12.2 #2).
///
/// The scaling half is applied by the consuming matmul's prologue
/// ([`crate::kernels::GemvRmsNorm`]), so the normalized activation is never written to
/// memory. That round-trip is what §12.2 #2 removes and what §12.1 names as
/// the real win.
///
/// **The dispatch geometry is `call_rms_norm`'s, deliberately.** Same
/// `work_per_threadgroup`, same `out_length`, same
/// `min(max_total_threads_per_threadgroup, (work/2).next_power_of_two())`
/// width — so the kernel selects the same `BLOCKSIZE` case and reduces over
/// the same tree in the same index order. The scale it writes is therefore
/// bit-identical to the `total` the shipping kernel computes internally, which
/// keeps §2.3.3 #1 out of the reduction entirely and confines the numerical
/// question to where the scaling multiply happens.
///
/// `alpha` is not taken: the weight is the consumer's to apply.
///
/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rms_norm_scale(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    output_scale: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Reduce, kernel_name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "rms_norm_scale {kernel_name} length={length} elements_to_sum={elements_to_sum}"
    );

    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                length as u32,
                elements_to_sum as u32,
                (input, input_offset),
                Output::new(output_scale),
                eps
            )
        );
    }

    let work_per_threadgroup = elements_to_sum;
    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };
    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );
    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_layer_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    beta: &Buffer,
    beta_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_layer_norm_with(
            device,
            ep,
            kernels,
            kernel_name,
            length,
            elements_to_sum,
            eps,
            input,
            input_offset,
            alpha,
            alpha_offset,
            beta,
            beta_offset,
            output,
            ParamStyle::default(),
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_layer_norm_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    length: usize,
    elements_to_sum: usize,
    eps: f32,
    input: &Buffer,
    input_offset: usize,
    alpha: &Buffer,
    alpha_offset: usize,
    beta: &Buffer,
    beta_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Reduce, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "layer_norm {kernel_name} length={length} elements_to_sum={elements_to_sum}"
    );

    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                length as u32,
                elements_to_sum as u32,
                (input, input_offset),
                Output::new(output),
                (alpha, alpha_offset),
                (beta, beta_offset),
                eps
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, NORM_PARAMS_ALIGN)?;

    let work_per_threadgroup = elements_to_sum;

    let out_length = length / work_per_threadgroup;

    let thread_group_count = MTLSize {
        width: out_length,
        height: 1,
        depth: 1,
    };

    let width = std::cmp::min(
        pipeline.max_total_threads_per_threadgroup(),
        (work_per_threadgroup / 2).next_power_of_two(),
    );

    let thread_group_size = MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rope_i(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_rope_i_with(
            device,
            ep,
            kernels,
            kernel_name,
            bh,
            td,
            stride_b,
            src,
            src_offset,
            cos,
            cos_offset,
            sin,
            sin_offset,
            output,
            ParamStyle::default(),
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rope_i_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Rope, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "rope_i {kernel_name} bh={bh} td={td}");

    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                bh,
                td,
                stride_b,
                (src, src_offset),
                (cos, cos_offset),
                (sin, sin_offset),
                Output::new(output)
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, ROPE_I_PARAMS_ALIGN)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rope_thd(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    b: usize,
    t: usize,
    h: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_rope_thd_with(
            device,
            ep,
            kernels,
            kernel_name,
            b,
            t,
            h,
            d,
            stride_b,
            src,
            src_offset,
            cos,
            cos_offset,
            sin,
            sin_offset,
            output,
            ParamStyle::default(),
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rope_thd_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    b: usize,
    t: usize,
    h: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Rope, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "rope_thd {kernel_name} b={b} t={t} h={h} d={d}");

    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                b,
                t,
                h,
                d,
                stride_b,
                (src, src_offset),
                (cos, cos_offset),
                (sin, sin_offset),
                Output::new(output)
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, ROPE_THD_PARAMS_ALIGN)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (b * t * h * d) / 2);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rope(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        call_rope_with(
            device,
            ep,
            kernels,
            kernel_name,
            bh,
            td,
            d,
            stride_b,
            src,
            src_offset,
            cos,
            cos_offset,
            sin,
            sin_offset,
            output,
            ParamStyle::default(),
        )
    }
}

/// # Safety
/// Every buffer bound here stays unreused and unwritten by the host until
/// the command buffer it is encoded into completes, and while an ICB that
/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_rope_with(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: &'static str,
    bh: usize,
    td: usize,
    d: usize,
    stride_b: usize,
    src: &Buffer,
    src_offset: usize,
    cos: &Buffer,
    cos_offset: usize,
    sin: &Buffer,
    sin_offset: usize,
    output: &Buffer,
    style: ParamStyle,
) -> Result<(), MetalKernelError> {
    let name = style.kernel_name(kernel_name);
    let pipeline = kernels.load_pipeline(device, Source::Rope, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "rope {kernel_name} bh={bh} td={td} d={d}");

    let _capture = begin_packed_params(encoder, style);
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(
            encoder,
            (
                bh,
                td,
                d,
                stride_b,
                (src, src_offset),
                (cos, cos_offset),
                (sin, sin_offset),
                Output::new(output)
            )
        );
    }
    let _staged = finish_packed_params(device, encoder, style, ROPE_PARAMS_ALIGN)?;
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, (bh * td) / 2);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
