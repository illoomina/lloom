use crate::kernels::linear_split;
use crate::kernels::{
    debug_group, set_params, Buffer, ComputeCommandEncoder, Device, EncoderParam, EncoderProvider,
    Kernels, MetalKernelError, Output, Source,
};

/// # Safety

/// Every buffer bound here stays unreused and unwritten by the host until

/// the command buffer it is encoded into completes, and while an ICB that

/// captured it can replay ([`crate::kernels::utils::EncoderParam`]).

pub unsafe fn call_const_fill(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    length: usize,
    output: &Buffer,
    v: impl EncoderParam,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Fill, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "const_fill {name} elems={length}");
    // SAFETY: this function's contract (# Safety above).
    unsafe {
        set_params!(encoder, (Output::new(output), v, length));
    }
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

// Additive closed typed route. Existing generic/custom EncoderParam path remains
// unchanged until explicit unsafe legacy/caller migration; NOT hidden raw proof.
mod tracked_value {
    pub trait Sealed {}
}
pub(crate) trait TrackedFillValue: tracked_value::Sealed {
    fn fill_shader(&self) -> &'static str;
    fn fill_bytes(self) -> ([u8; 8], usize, usize);
}
macro_rules! tracked_fill_value {
    ($(($ty:ty, $shader:literal)),+) => { $(
        impl tracked_value::Sealed for $ty {}
        impl TrackedFillValue for $ty {
            fn fill_shader(&self) -> &'static str { $shader }
            fn fill_bytes(self) -> ([u8; 8], usize, usize) {
                let bytes = self.to_ne_bytes();
                let mut inline = [0u8; 8];
                inline[..bytes.len()].copy_from_slice(&bytes);
                (inline, bytes.len(), core::mem::align_of::<Self>())
            }
        }
    )+ };
}
// Exact instantiated producer set in metal_src/fill.metal. No independent
// name/type pair: shader and scalar width are fixed by sealed Rust type.
tracked_fill_value!(
    (u8, "fill_u8"),
    (u32, "fill_u32"),
    (i64, "fill_i64"),
    (f32, "fill_f32"),
    (half::f16, "fill_f16"),
    (half::bf16, "fill_bf16")
);

/// Closed primitive adapter into actual generation-owned operation lowering.
/// No arbitrary EncoderParam executes during tuple preflight. Args are already
/// evaluated once at this Rust call boundary; scalar conversion is sealed.
/// Actual safe Tensor route NOT switched until default/supplied rich creators,
/// cold inventory/page/packed capacity placement and raw/capture closure exist.
pub(crate) fn call_const_fill_tracked(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    length: usize,
    output: &Buffer,
    value: impl TrackedFillValue,
) -> Result<(), MetalKernelError> {
    let name = value.fill_shader();
    let (value, scalar_len, scalar_align) = value.fill_bytes();
    let pipeline = kernels.load_pipeline(device, Source::Fill, name)?;
    let encoder_owner = ep.encoder();
    let encoder = encoder_owner.as_ref();
    // Match existing thread geometry; refuse zero pipeline width before native
    // effects instead of arithmetic panic. Existing legacy entry is unchanged.
    let width = pipeline.max_total_threads_per_threadgroup().min(length);
    if width == 0 {
        return Err(MetalKernelError::QueueUnavailable(
            "fill requires nonzero planned thread width",
        ));
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
    encoder.dispatch_fill_tracked(
        &pipeline,
        output,
        &value[..scalar_len],
        scalar_align,
        length,
        groups,
        threads,
    )
}
