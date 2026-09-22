//! The Qwen4-exp block kernels (`metal_src/qwen4exp.metal`): dispatch
//! entry points, one per kernel. See the model crate
//! (`lloom::models::qwen4exp::kernels`) for the ops that call them and the
//! scalar references they are tested against.

use crate::kernels::utils::EncoderProvider;
use crate::kernels::{
    debug_group, set_params, Buffer, BufferOffset, ComputeCommandEncoder, Device, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

/// `(I + A)⁻¹` for `batch` strictly-lower `[n, n]` f32 matrices at `a`, into
/// `out` (`batch · n · n` f32, contiguous). One threadgroup of `min(n, max)`
/// threads per matrix.
pub fn call_unit_lower_inverse(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    batch: usize,
    n: usize,
    a: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "unit_lower_inverse_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "unit_lower_inverse batch={batch} n={n}");
    set_params!(encoder, (&a, Output::new(out), n as u32));
    let width = n.max(1).min(pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(
        MTLSize {
            width: batch.max(1),
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
