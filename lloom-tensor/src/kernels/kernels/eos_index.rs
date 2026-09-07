use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

/// The batched EOS-index reduction: over the `n` freshly-sampled token ids in
/// `tokens`, write to `out[0]` the FIRST index whose token is one of the `n_eos`
/// stop ids, or `n` if none matched. ONE dispatch per commit window — the
/// replacement for the poll arm's per-token `broadcast_eq → max_all` train.
///
/// A single threadgroup covers the window (a wake cadence of a handful to a few
/// tens of tokens), and the width is the smallest power of two `>= n` so the
/// in-kernel tree reduction spans every real lane; lanes past `n` propose the
/// sentinel and drop out. `n` must be `<= 1024` (the kernel's shared array and a
/// threadgroup's ceiling) — the caller sizes the window well under that.
#[allow(clippy::too_many_arguments)]
pub fn call_eos_index(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    tokens: &Buffer,
    tokens_offset: usize,
    n: usize,
    eos_ids: &Buffer,
    eos_offset: usize,
    n_eos: usize,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::EosIndex, "eos_index_u32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "eos_index n={n} n_eos={n_eos}");

    set_params!(
        encoder,
        (
            (tokens, tokens_offset),
            n as u32,
            (eos_ids, eos_offset),
            n_eos as u32,
            Output::new(out)
        )
    );

    // One threadgroup; width = next_pow2(n), clamped to the kernel's shared array
    // (1024) and to the pipeline's own max. The reduction reads `width` lanes, so
    // it must cover every real token — a width below `n` would drop the tail.
    let want = n.max(1).next_power_of_two();
    let width = want
        .min(1024)
        .min(pipeline.max_total_threads_per_threadgroup());
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
