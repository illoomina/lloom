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

/// One threadgroup per `groups` items of `width` threads.
fn dispatch_groups(encoder: &ComputeCommandEncoder, groups: usize, width: usize) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: groups.max(1),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

/// `hc_norm_f32`: the grouped RMS norm of `rows` carries of `n` streams of
/// `d`, γ over `n·d`. One threadgroup per (row, stream).
#[allow(clippy::too_many_arguments)]
pub fn call_hc_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    n: usize,
    d: usize,
    eps: f32,
    x: BufferOffset,
    gamma: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "hc_norm_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "hc_norm rows={rows} n={n} d={d}");
    set_params!(
        encoder,
        (&x, &gamma, Output::new(out), n as u32, d as u32, eps)
    );
    let width = 256usize
        .min(pipeline.max_total_threads_per_threadgroup())
        .min(d.max(32).next_power_of_two());
    dispatch_groups(encoder, rows * n, width);
    Ok(())
}

/// `hc_collapse_f32`: `out [rows, d] = mean_s(xn · gate)` over the `n` streams.
#[allow(clippy::too_many_arguments)]
pub fn call_hc_collapse(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    n: usize,
    d: usize,
    xn: BufferOffset,
    gate: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "hc_collapse_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "hc_collapse rows={rows} n={n} d={d}");
    set_params!(
        encoder,
        (
            &xn,
            &gate,
            Output::new(out),
            n as u32,
            d as u32,
            rows as u32
        )
    );
    let total = rows * d;
    let width = 256usize.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, total.div_ceil(width), width);
    Ok(())
}

/// `hc_combine_f32`: `out [rows, n·d] = h + bo ⊗ 2·sigmoid(inject / n)`.
#[allow(clippy::too_many_arguments)]
pub fn call_hc_combine(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    n: usize,
    d: usize,
    h: BufferOffset,
    bo: BufferOffset,
    inject: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "hc_combine_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "hc_combine rows={rows} n={n} d={d}");
    set_params!(
        encoder,
        (
            &h,
            &bo,
            &inject,
            Output::new(out),
            n as u32,
            d as u32,
            rows as u32
        )
    );
    let total = rows * n * d;
    let width = 256usize.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, total.div_ceil(width), width);
    Ok(())
}

/// `moe_topk_f32`: per row of `n_experts` logits, the top-`k` ids (u32) and
/// their renormalised softmax weights. `n_experts` ≤ 1024.
#[allow(clippy::too_many_arguments)]
pub fn call_moe_topk(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    n_experts: usize,
    k: usize,
    logits: BufferOffset,
    ids: &Buffer,
    w: &Buffer,
) -> Result<(), MetalKernelError> {
    if n_experts > 1024 || k > n_experts || k == 0 {
        return Err(MetalKernelError::LoadFunctionError(format!(
            "moe_topk: n_experts {n_experts} (<= 1024), k {k}"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "moe_topk_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "moe_topk rows={rows} e={n_experts} k={k}");
    set_params!(
        encoder,
        (
            &logits,
            Output::new(ids),
            Output::new(w),
            n_experts as u32,
            k as u32
        )
    );
    let width = 256usize
        .min(pipeline.max_total_threads_per_threadgroup())
        .max(k.next_power_of_two().max(32));
    dispatch_groups(encoder, rows, width);
    Ok(())
}

/// `silu_mul_f32`: `out = silu(g) · u` over `n` elements.
pub fn call_silu_mul(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    n: usize,
    g: BufferOffset,
    u: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "silu_mul_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "silu_mul n={n}");
    set_params!(encoder, (&g, &u, Output::new(out), n as u32));
    let width = 256usize.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, n.div_ceil(width), width);
    Ok(())
}

/// `moe_combine_f32`: `out [rows, d] = Σ_k w · y + shared · sigmoid(gate)`.
#[allow(clippy::too_many_arguments)]
pub fn call_moe_combine(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    k: usize,
    d: usize,
    y: BufferOffset,
    w: BufferOffset,
    shared: BufferOffset,
    gate: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "moe_combine_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "moe_combine rows={rows} k={k} d={d}");
    set_params!(
        encoder,
        (
            &y,
            &w,
            &shared,
            &gate,
            Output::new(out),
            k as u32,
            d as u32,
            rows as u32
        )
    );
    let total = rows * d;
    let width = 256usize.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, total.div_ceil(width), width);
    Ok(())
}

/// `gdn_prologue_f32`: `qkv [conv_ch]` + the conv window (in place) →
/// `heads [3, h_v, d]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_prologue(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    d: usize,
    h_k: usize,
    h_v: usize,
    kk: usize,
    eps: f32,
    q_scale: f32,
    qkv: BufferOffset,
    conv_w: BufferOffset,
    win: BufferOffset,
    heads: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "gdn_prologue_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_prologue d={d} h_k={h_k} h_v={h_v}");
    set_params!(
        encoder,
        (
            &qkv,
            &conv_w,
            Output::from_buffer_offset(&win),
            Output::new(heads),
            d as u32,
            h_k as u32,
            h_v as u32,
            kk as u32,
            eps,
            q_scale
        )
    );
    let width = d
        .max(32)
        .next_power_of_two()
        .min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, 2 * h_k + h_v, width);
    Ok(())
}

/// `gdn_step_f32`: the recurrence on `s [d, d, h_v]` in place → `o [h_v, d]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_step(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    d: usize,
    h_v: usize,
    heads: BufferOffset,
    alpha: BufferOffset,
    beta: BufferOffset,
    dt_bias: BufferOffset,
    ssm_a: BufferOffset,
    s: BufferOffset,
    o: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "gdn_step_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_step d={d} h_v={h_v}");
    set_params!(
        encoder,
        (
            &heads,
            &alpha,
            &beta,
            &dt_bias,
            &ssm_a,
            Output::from_buffer_offset(&s),
            Output::new(o),
            d as u32,
            h_v as u32
        )
    );
    let width = h_v.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, d, width);
    Ok(())
}

/// `gdn_gated_norm_f32`: per-head RMS norm of `o [h_v, d]` × γ × sigmoid(z).
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_gated_norm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    d: usize,
    h_v: usize,
    eps: f32,
    o: BufferOffset,
    gamma: BufferOffset,
    z: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "gdn_gated_norm_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_gated_norm d={d} h_v={h_v}");
    set_params!(encoder, (&o, &gamma, &z, Output::new(out), d as u32, eps));
    let width = d
        .max(32)
        .next_power_of_two()
        .min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, h_v, width);
    Ok(())
}
