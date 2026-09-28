//! The EOS gate's kernel (lloom #735 step 4): see `metal_src/eos_gate.metal`
//! and [`crate::kernels::metal::DispatchGate`], which owns the flag this
//! kernel raises and encodes the dispatch.

use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, BufferOffset, ComputeCommandEncoder, Device, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

/// Encode the gate: `flag[0] = 1` if `token[0] ∈ eos_ids`. One thread.
#[allow(clippy::too_many_arguments)]
pub fn call_eos_gate(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    token: &Buffer,
    token_offset: usize,
    eos_ids: &Buffer,
    eos_offset: usize,
    n_eos: usize,
    flag: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::EosGate, "eos_gate_u32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "eos_gate n_eos={n_eos}");

    set_params!(
        encoder,
        (
            (token, token_offset),
            (eos_ids, eos_offset),
            n_eos as u32,
            Output::new(flag)
        )
    );

    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(one, one);
    Ok(())
}
/// `words_publish`: `n` u32 words from `src` into the mapped `out`, then the
/// `generation` stamp into `step` behind a device barrier, unless
/// `generation` is 0. The barrier orders the two for the GPU, not for a host
/// polling the stamp: such a caller passes 0 and stamps with
/// [`call_ring_stamp`] certifying `out`. One threadgroup.
#[allow(clippy::too_many_arguments)]
pub fn call_words_publish(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    src: BufferOffset,
    n: usize,
    out: BufferOffset,
    step: BufferOffset,
    generation: u32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::EosGate, "words_publish")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "words_publish n={n} gen={generation}");
    set_params!(
        encoder,
        (
            &src,
            n as u32,
            Output::from_buffer_offset(&out),
            Output::from_buffer_offset(&step),
            generation
        )
    );
    let width = 256usize.min(pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
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

/// `ring_stamp`: `generation` into `step`, ordered after the last writers of
/// `certified` (bound, never read). The ordering is the command stream's: the
/// binds are what make the hazard tracker emit the barrier or the fence wait.
pub fn call_ring_stamp(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    step: BufferOffset,
    generation: u32,
    certified: [BufferOffset; 2],
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::EosGate, "ring_stamp")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "ring_stamp gen={generation}");
    let [a, b] = certified;
    set_params!(
        encoder,
        (Output::from_buffer_offset(&step), generation, &a, &b)
    );
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(one, one);
    Ok(())
}

/// The fused publish (lloom #735 step 5): see `argmax_publish_body` in
/// `metal_src/eos_gate.metal`. `kernel` is `argmax_publish_f16` or
/// `argmax_publish_f32`; every offset in bytes. `n_eos == 0` skips the flag
/// (pass any buffer). One threadgroup, as wide as the pipeline allows up to
/// 1024, a power of two (the reduction tree).
#[allow(clippy::too_many_arguments)]
pub fn call_argmax_publish(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel: &'static str,
    logits: BufferOffset,
    n: usize,
    token_slot: BufferOffset,
    history_slot: BufferOffset,
    eos_ids: BufferOffset,
    n_eos: usize,
    flag: &Buffer,
    step: BufferOffset,
    generation: u32,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::EosGate, kernel)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "argmax_publish n={n} n_eos={n_eos} gen={generation}"
    );

    set_params!(
        encoder,
        (
            &logits,
            n as u32,
            Output::from_buffer_offset(&token_slot),
            Output::from_buffer_offset(&history_slot),
            &eos_ids,
            n_eos as u32,
            Output::new(flag),
            Output::from_buffer_offset(&step),
            generation
        )
    );

    let width = 1024usize
        .min(pipeline.max_total_threads_per_threadgroup())
        .next_power_of_two()
        .min(1024);
    let width = if width.is_power_of_two() {
        width
    } else {
        width / 2
    };
    encoder.dispatch_thread_groups(
        MTLSize {
            width: 1,
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
