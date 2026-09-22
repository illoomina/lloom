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

/// `qsa_prep_f32`: the q/gate split, the per-head norms and the partial rope
/// at `pos` → `out [2·nh + nkv, hd]`.
#[allow(clippy::too_many_arguments)]
pub fn call_qsa_prep(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    hd: usize,
    n_rot: usize,
    nh: usize,
    nkv: usize,
    pos: usize,
    theta: f32,
    eps: f32,
    qg: BufferOffset,
    k: BufferOffset,
    q_gamma: BufferOffset,
    k_gamma: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    if hd > 1024 || n_rot > hd || !n_rot.is_multiple_of(2) {
        return Err(MetalKernelError::LoadFunctionError(format!(
            "qsa_prep: head_dim {hd} (<= 1024), n_rot {n_rot} (even, <= head_dim)"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "qsa_prep_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "qsa_prep hd={hd} nh={nh} nkv={nkv} pos={pos}");
    set_params!(
        encoder,
        (
            &qg,
            &k,
            &q_gamma,
            &k_gamma,
            Output::new(out),
            hd as u32,
            n_rot as u32,
            nh as u32,
            nkv as u32,
            pos as u32,
            theta,
            eps
        )
    );
    let width = hd
        .max(32)
        .next_power_of_two()
        .min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, nh + nkv, width);
    Ok(())
}

/// `sigmoid_mul_f32`: `out = sigmoid(a) · b` over `n` elements.
pub fn call_sigmoid_mul(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    n: usize,
    a: BufferOffset,
    b: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "sigmoid_mul_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "sigmoid_mul n={n}");
    set_params!(encoder, (&a, &b, Output::new(out), n as u32));
    let width = 256usize.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, n.div_ceil(width), width);
    Ok(())
}

fn dispatch_groups_2d(encoder: &ComputeCommandEncoder, gx: usize, gy: usize, width: usize) {
    encoder.dispatch_thread_groups(
        MTLSize {
            width: gx.max(1),
            height: gy.max(1),
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
}

fn head_width(d: usize, pipeline: &crate::kernels::ComputePipeline) -> usize {
    d.max(32)
        .next_power_of_two()
        .min(pipeline.max_total_threads_per_threadgroup())
}

/// `gdn_gate_f32`: `alpha`, `beta_raw` `[seq, h_v]` → `[2, h_v, nc, cs]` (g, β).
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_gate(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    seq: usize,
    h_v: usize,
    nc: usize,
    cs: usize,
    alpha: BufferOffset,
    beta_raw: BufferOffset,
    dt_bias: BufferOffset,
    ssm_a: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "gdn_gate_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_gate seq={seq} h_v={h_v} nc={nc}");
    set_params!(
        encoder,
        (
            &alpha,
            &beta_raw,
            &dt_bias,
            &ssm_a,
            Output::new(out),
            seq as u32,
            h_v as u32,
            nc as u32,
            cs as u32
        )
    );
    let n = h_v * nc * cs;
    let width = 256usize.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, n.div_ceil(width), width);
    Ok(())
}

/// `gdn_prefill_prologue_f32`: `qkv [seq, C]`, the window, the taps →
/// `[3, h_v, nc, cs, d]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_prefill_prologue(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    seq: usize,
    d: usize,
    h_k: usize,
    h_v: usize,
    kk: usize,
    cs: usize,
    nc: usize,
    eps: f32,
    q_scale: f32,
    qkv: BufferOffset,
    win: BufferOffset,
    conv_w: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "gdn_prefill_prologue_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "gdn_prefill_prologue seq={seq} d={d} h_k={h_k} h_v={h_v}"
    );
    set_params!(
        encoder,
        (
            &qkv,
            &win,
            &conv_w,
            Output::new(out),
            seq as u32,
            d as u32,
            h_k as u32,
            h_v as u32,
            kk as u32,
            cs as u32,
            nc as u32,
            eps,
            q_scale
        )
    );
    dispatch_groups_2d(encoder, 2 * h_k + h_v, nc * cs, head_width(d, &pipeline));
    Ok(())
}

/// `gdn_conv_window_f32`: the window after the chunk → a new `[C, kk]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_conv_window(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    seq: usize,
    channels: usize,
    kk: usize,
    win: BufferOffset,
    qkv: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "gdn_conv_window_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_conv_window seq={seq} C={channels} kk={kk}");
    set_params!(
        encoder,
        (
            &win,
            &qkv,
            Output::new(out),
            seq as u32,
            channels as u32,
            kk as u32
        )
    );
    let n = channels * kk;
    let width = 256usize.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, n.div_ceil(width), width);
    Ok(())
}

/// `gdn_chunk_decay_f32`: `g [h_v, nc, cs]` → the flat decay pack (see the kernel).
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_chunk_decay(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    h_v: usize,
    nc: usize,
    cs: usize,
    g: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    if cs > 256 || cs == 0 {
        return Err(MetalKernelError::LoadFunctionError(format!(
            "gdn_chunk_decay: chunk {cs} (1..=256)"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "gdn_chunk_decay_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_chunk_decay h_v={h_v} nc={nc} cs={cs}");
    set_params!(
        encoder,
        (&g, Output::new(out), h_v as u32, nc as u32, cs as u32)
    );
    dispatch_groups(encoder, h_v * nc, head_width(cs, &pipeline));
    Ok(())
}

/// `gdn_chunk_scale_f32`: the chunked q/k/v `[3, n, d]`, the gate pack, the
/// decay pack → the five scaled operands, flat `[5, n, d]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_chunk_scale(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    n: usize,
    d: usize,
    qkv: BufferOffset,
    gb: BufferOffset,
    small: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "gdn_chunk_scale_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_chunk_scale n={n} d={d}");
    set_params!(
        encoder,
        (&qkv, &gb, &small, Output::new(out), n as u32, d as u32)
    );
    let total = n * d;
    let width = 256usize.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, total.div_ceil(width), width);
    Ok(())
}

/// `gdn_chunk_epilogue_f32`: `o [h_v, padded, d]`, γ, `z [seq, h_v·d]` →
/// `[seq, h_v·d]`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_chunk_epilogue(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    seq: usize,
    padded: usize,
    d: usize,
    h_v: usize,
    eps: f32,
    o: BufferOffset,
    gamma: BufferOffset,
    z: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "gdn_chunk_epilogue_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_chunk_epilogue seq={seq} d={d} h_v={h_v}");
    set_params!(
        encoder,
        (
            &o,
            &gamma,
            &z,
            Output::new(out),
            padded as u32,
            d as u32,
            h_v as u32,
            eps
        )
    );
    dispatch_groups_2d(encoder, h_v, seq, head_width(d, &pipeline));
    Ok(())
}

/// `qsa_prep_rows_f32`: over a chunk from `pos0` → the flat pack (see the kernel).
#[allow(clippy::too_many_arguments)]
pub fn call_qsa_prep_rows(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    hd: usize,
    n_rot: usize,
    nh: usize,
    nkv: usize,
    seq: usize,
    pos0: usize,
    theta: f32,
    eps: f32,
    qg: BufferOffset,
    k: BufferOffset,
    v: BufferOffset,
    q_gamma: BufferOffset,
    k_gamma: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    if hd > 1024 || n_rot > hd || !n_rot.is_multiple_of(2) {
        return Err(MetalKernelError::LoadFunctionError(format!(
            "qsa_prep_rows: head_dim {hd} (<= 1024), n_rot {n_rot} (even, <= head_dim)"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "qsa_prep_rows_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qsa_prep_rows hd={hd} nh={nh} nkv={nkv} seq={seq} pos0={pos0}"
    );
    set_params!(
        encoder,
        (
            &qg,
            &k,
            &v,
            &q_gamma,
            &k_gamma,
            Output::new(out),
            hd as u32,
            n_rot as u32,
            nh as u32,
            nkv as u32,
            seq as u32,
            pos0 as u32,
            theta,
            eps
        )
    );
    dispatch_groups_2d(encoder, nh + nkv, seq, head_width(hd, &pipeline));
    Ok(())
}

/// `moe_slot_map_u32`: expert ids through the residency table → the flat
/// `[3·n]` plan (slot ids | ids | miss flags).
#[allow(clippy::too_many_arguments)]
pub fn call_moe_slot_map(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    n: usize,
    n_experts: usize,
    ids: BufferOffset,
    table: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "moe_slot_map_u32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "moe_slot_map n={n}");
    set_params!(
        encoder,
        (&ids, &table, Output::new(out), n as u32, n_experts as u32)
    );
    let width = 64usize.min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, n.div_ceil(width), width);
    Ok(())
}
