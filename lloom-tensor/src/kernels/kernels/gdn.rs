//! The Gated DeltaNet block's kernels (`metal_src/gdn/gdn.metal`): dispatch
//! entry points, one per kernel. The block's ops (`lloom::nn::gdn`, the
//! Qwen4-exp mixer `lloom::models::qwen4exp::deltanet_block`) call them.

use crate::kernels::utils::{dispatch_groups, dispatch_groups_2d, head_width, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, BufferOffset, ComputeCommandEncoder, Device, Kernels,
    MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

/// The kernel `base` at the output gate: `sigmoid(z)` (`_f32`) or `silu(z)`
/// (`_silu_f32`, Qwen3.5's `output_gate_type: swish`). The gate is the
/// block's (`lloom::nn::gdn::OutGate`); the kernels are instantiated per
/// gate. A static name: no allocation per dispatch.
fn gated_kernel(base: &str, silu_gate: bool) -> &'static str {
    match (base, silu_gate) {
        ("gdn_gated_norm", false) => "gdn_gated_norm_f32",
        ("gdn_gated_norm", true) => "gdn_gated_norm_silu_f32",
        ("gdn_gated_norm_rows", false) => "gdn_gated_norm_rows_f32",
        ("gdn_gated_norm_rows", true) => "gdn_gated_norm_rows_silu_f32",
        ("gdn_chunk_epilogue", false) => "gdn_chunk_epilogue_f32",
        ("gdn_chunk_epilogue", true) => "gdn_chunk_epilogue_silu_f32",
        _ => unreachable!("no gated kernel named {base}"),
    }
}

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
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "unit_lower_inverse_f32")?;
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
    win_out: BufferOffset,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "gdn_prologue_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_prologue d={d} h_k={h_k} h_v={h_v}");
    set_params!(
        encoder,
        (
            &qkv,
            &conv_w,
            &win,
            Output::new(heads),
            d as u32,
            h_k as u32,
            h_v as u32,
            kk as u32,
            eps,
            q_scale,
            Output::from_buffer_offset(&win_out)
        )
    );
    let width = d
        .max(32)
        .next_power_of_two()
        .min(pipeline.max_total_threads_per_threadgroup());
    dispatch_groups(encoder, 2 * h_k + h_v, width);
    Ok(())
}

/// `gdn_step_f32`: the recurrence on `s [d, d, h_v]` → `o [h_v, d]` and the
/// slab after the step into `s_out`. `register` holds each thread's slice of
/// `S` in registers (the slab read once) where an instantiated array bound
/// covers `ceil(d / split)`; otherwise, and when `register` is false, the loop
/// form.
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
    s_out: BufferOffset,
    register: bool,
) -> Result<(), MetalKernelError> {
    let (pipeline, split) = gdn_step_pipeline(device, kernels, StepShape::Row, d, h_v, register)?;
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
            &s,
            Output::new(o),
            d as u32,
            h_v as u32,
            Output::from_buffer_offset(&s_out)
        )
    );
    dispatch_groups(encoder, d, h_v * split);
    Ok(())
}

/// The step kernels' shapes: one row, the rows of a pool, the positions of
/// one row.
#[derive(Debug, Clone, Copy)]
enum StepShape {
    Row,
    Rows,
    Seq,
}

/// The pipeline for `shape` and the `split` it runs at. `h_v × split`
/// threads: `split` slices of `dk` per head, as many as the loop form's
/// threadgroup takes up to the cap (the reductions close over `split`
/// partials in threadgroup memory; `h_v × split` ≤ 1024). `register` takes
/// the register form where an instantiated array bound covers `ceil(d /
/// split)` and its pipeline takes the same threadgroup; otherwise the loop
/// form. Both forms run at the loop form's `split`, so each form computes one
/// arithmetic whatever the shape.
fn gdn_step_pipeline(
    device: &Device,
    kernels: &Kernels,
    shape: StepShape,
    d: usize,
    h_v: usize,
    register: bool,
) -> Result<(crate::kernels::ComputePipeline, usize), MetalKernelError> {
    let (loop_name, reg16, reg32) = match shape {
        StepShape::Row => ("gdn_step_f32", "gdn_step_r16_f32", "gdn_step_r32_f32"),
        StepShape::Rows => (
            "gdn_step_rows_f32",
            "gdn_step_rows_r16_f32",
            "gdn_step_rows_r32_f32",
        ),
        StepShape::Seq => (
            "gdn_step_seq_f32",
            "gdn_step_seq_r16_f32",
            "gdn_step_seq_r32_f32",
        ),
    };
    let pipeline = kernels.load_pipeline(device, Source::Gdn, loop_name)?;
    let split = gdn_split(h_v, &pipeline);
    if !register || split == 0 {
        return Ok((pipeline, split));
    }
    let name = match d.div_ceil(split) {
        0..=16 => reg16,
        17..=32 => reg32,
        _ => return Ok((pipeline, split)),
    };
    let reg = kernels.load_pipeline(device, Source::Gdn, name)?;
    if reg.max_total_threads_per_threadgroup() >= h_v * split {
        Ok((reg, split))
    } else {
        Ok((pipeline, split))
    }
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
    silu_gate: bool,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(
        device,
        Source::Gdn,
        gated_kernel("gdn_gated_norm", silu_gate),
    )?;
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
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "gdn_gate_f32")?;
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
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "gdn_prefill_prologue_f32")?;
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
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "gdn_conv_window_f32")?;
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
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "gdn_chunk_decay_f32")?;
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
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "gdn_chunk_scale_f32")?;
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

/// `gdn_chunk_scan_f32`: the inter-chunk recurrence of a DeltaNet prefill
/// as one launch, a threadgroup per head (`w`, `u`, `qg`, `kg` `[h_v, nc,
/// cs, d]`; `kq [h_v, nc, cs, cs]`; `gl [h_v, nc]`; `s_in [h_v, d, d]`) →
/// `o [h_v, nc·cs, d]` and `s_out [h_v, d, d]`; `vn [h_v, cs, d]` scratch.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_chunk_scan_at(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    (h_v, nc, cs, d): (usize, usize, usize, usize),
    splits: usize,
    w: BufferOffset,
    u: BufferOffset,
    qg: BufferOffset,
    kq: BufferOffset,
    kg: BufferOffset,
    gl: BufferOffset,
    s_in: BufferOffset,
    o: &Buffer,
    s_out: BufferOffset,
    vn: &Buffer,
) -> Result<(), MetalKernelError> {
    let splits = splits.clamp(1, d / 8);
    if !cs.is_multiple_of(8) || !d.is_multiple_of(8) || cs == 0 || d == 0 {
        return Err(MetalKernelError::LoadFunctionError(format!(
            "gdn_chunk_scan: cs {cs} and d {d} must be multiples of 8"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "gdn_chunk_scan_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "gdn_chunk_scan h={h_v} nc={nc} cs={cs} d={d} splits={splits}"
    );
    set_params!(
        encoder,
        (
            &w,
            &u,
            &qg,
            &kq,
            &kg,
            &gl,
            &s_in,
            Output::new(o),
            Output::with_offset(s_out.buffer, s_out.offset_in_bytes),
            Output::new(vn),
            nc as u32,
            cs as u32,
            d as u32,
            splits as u32
        )
    );
    dispatch_groups(encoder, h_v * splits, 128);
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
    silu_gate: bool,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(
        device,
        Source::Gdn,
        gated_kernel("gdn_chunk_epilogue", silu_gate),
    )?;
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

/// `gdn_prologue_rows_f32`: `B` rows of `qkv [B, C]` against the window
/// pool (`slots` / `phases` per row) → `heads [B, 3, h_v, d]`, each row's
/// window after the step written to its next phase.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_prologue_rows(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    d: usize,
    h_k: usize,
    h_v: usize,
    kk: usize,
    eps: f32,
    q_scale: f32,
    n_slots: usize,
    n_phases: usize,
    qkv: BufferOffset,
    conv_w: BufferOffset,
    win_pool: BufferOffset,
    slots: BufferOffset,
    phases: BufferOffset,
    heads: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "gdn_prologue_rows_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "gdn_prologue_rows rows={rows} d={d} h_k={h_k} h_v={h_v}"
    );
    set_params!(
        encoder,
        (
            &qkv,
            &conv_w,
            &win_pool,
            &slots,
            &phases,
            Output::new(heads),
            d as u32,
            h_k as u32,
            h_v as u32,
            kk as u32,
            eps,
            q_scale,
            n_slots as u32,
            n_phases as u32,
            Output::from_buffer_offset(&win_pool)
        )
    );
    dispatch_groups_2d(encoder, 2 * h_k + h_v, rows, head_width(d, &pipeline));
    Ok(())
}

/// `gdn_prologue_seq_f32`: `T` positions of one row (`slot` at `phase`) →
/// `heads [T, 3, h_v, d]`, the window after each position at phase
/// `phase + 1 + t`.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_prologue_seq(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    steps: usize,
    d: usize,
    h_k: usize,
    h_v: usize,
    kk: usize,
    eps: f32,
    q_scale: f32,
    n_slots: usize,
    n_phases: usize,
    slot: usize,
    phase: usize,
    qkv: BufferOffset,
    conv_w: BufferOffset,
    win_pool: BufferOffset,
    heads: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Gdn, "gdn_prologue_seq_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "gdn_prologue_seq steps={steps} d={d} h_k={h_k} h_v={h_v}"
    );
    set_params!(
        encoder,
        (
            &qkv,
            &conv_w,
            &win_pool,
            Output::new(heads),
            d as u32,
            h_k as u32,
            h_v as u32,
            kk as u32,
            eps,
            q_scale,
            n_slots as u32,
            n_phases as u32,
            slot as u32,
            phase as u32,
            Output::from_buffer_offset(&win_pool)
        )
    );
    dispatch_groups_2d(encoder, 2 * h_k + h_v, steps, head_width(d, &pipeline));
    Ok(())
}

fn gdn_split(h_v: usize, pipeline: &crate::kernels::ComputePipeline) -> usize {
    let max = pipeline.max_total_threads_per_threadgroup().min(1024);
    (max / h_v.max(1)).clamp(1, gdn_split_cap())
}

/// The most `dk` slices a head's threads take in `gdn_step` (`LLOOM_GDN_SPLIT`,
/// read once): more slices, more threads in flight per threadgroup against a
/// longer cross-slice reduction.
fn gdn_split_cap() -> usize {
    crate::tuning::tuning().gdn.split
}

/// `gdn_step_rows_f32`: `B` rows of the recurrence over the `S` pool, in the
/// form `register` names (as [`call_gdn_step`]).
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_step_rows(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    d: usize,
    h_v: usize,
    n_slots: usize,
    n_phases: usize,
    heads: BufferOffset,
    alpha: BufferOffset,
    beta: BufferOffset,
    dt_bias: BufferOffset,
    ssm_a: BufferOffset,
    s_pool: BufferOffset,
    slots: BufferOffset,
    phases: BufferOffset,
    o: &Buffer,
    register: bool,
) -> Result<(), MetalKernelError> {
    let (pipeline, split) = gdn_step_pipeline(device, kernels, StepShape::Rows, d, h_v, register)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_step_rows rows={rows} d={d} h_v={h_v}");
    set_params!(
        encoder,
        (
            &heads,
            &alpha,
            &beta,
            &dt_bias,
            &ssm_a,
            &s_pool,
            &slots,
            &phases,
            Output::new(o),
            d as u32,
            h_v as u32,
            n_slots as u32,
            n_phases as u32,
            Output::from_buffer_offset(&s_pool)
        )
    );
    dispatch_groups_2d(encoder, d, rows, h_v * split);
    Ok(())
}

/// `gdn_step_seq_f32`: `T` positions of one row's recurrence in one launch,
/// every intermediate `S` kept at phases `phase + 1 ..= phase + T`, in the
/// form `register` names (as [`call_gdn_step`]).
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_step_seq(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    steps: usize,
    d: usize,
    h_v: usize,
    n_slots: usize,
    n_phases: usize,
    slot: usize,
    phase: usize,
    heads: BufferOffset,
    alpha: BufferOffset,
    beta: BufferOffset,
    dt_bias: BufferOffset,
    ssm_a: BufferOffset,
    s_pool: BufferOffset,
    o: &Buffer,
    register: bool,
) -> Result<(), MetalKernelError> {
    let (pipeline, split) = gdn_step_pipeline(device, kernels, StepShape::Seq, d, h_v, register)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_step_seq steps={steps} d={d} h_v={h_v}");
    set_params!(
        encoder,
        (
            &heads,
            &alpha,
            &beta,
            &dt_bias,
            &ssm_a,
            &s_pool,
            Output::new(o),
            d as u32,
            h_v as u32,
            n_slots as u32,
            n_phases as u32,
            slot as u32,
            phase as u32,
            steps as u32,
            Output::from_buffer_offset(&s_pool)
        )
    );
    dispatch_groups(encoder, d, h_v * split);
    Ok(())
}

/// `gdn_gated_norm_rows_f32`: `B` rows of the gated norm.
#[allow(clippy::too_many_arguments)]
pub fn call_gdn_gated_norm_rows(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    d: usize,
    h_v: usize,
    eps: f32,
    o: BufferOffset,
    gamma: BufferOffset,
    z: BufferOffset,
    out: &Buffer,
    silu_gate: bool,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(
        device,
        Source::Gdn,
        gated_kernel("gdn_gated_norm_rows", silu_gate),
    )?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "gdn_gated_norm_rows rows={rows} d={d} h_v={h_v}");
    set_params!(
        encoder,
        (&o, &gamma, &z, Output::new(out), d as u32, h_v as u32, eps)
    );
    dispatch_groups_2d(encoder, h_v, rows, head_width(d, &pipeline));
    Ok(())
}
