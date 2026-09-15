//! The EOS gate's kernel (lloom #735 step 4): see `metal_src/eos_gate.metal`
//! and [`crate::kernels::metal::DispatchGate`], which owns the flag this
//! kernel raises and encodes the dispatch.

use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
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
