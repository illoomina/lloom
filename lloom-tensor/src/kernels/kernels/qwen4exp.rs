//! The Qwen4-exp model's kernels (`metal_src/qwen4exp.metal`): dispatch
//! entry points, one per kernel. See the model crate
//! (`lloom::models::qwen4exp::kernels`) for the ops that call them and the
//! scalar references they are tested against. The Gated DeltaNet block's are
//! in `gdn.rs`.

use crate::kernels::utils::{dispatch_groups, dispatch_groups_2d, head_width, EncoderProvider};
use crate::kernels::{
    debug_group, set_params, Buffer, BufferOffset, ComputeCommandEncoder, Device, Kernels,
    MetalKernelError, Output, Source,
};

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

/// `hc_mix_a_f32`: the HC prologue's first launch — the grouped norm, the
/// down projection (SiLU, `scale`) and the inject projection of `rows`
/// carries, into one flat buffer `[rows·n·d] xn | [rows·lr] lo | [rows·n]
/// inject`. `n_inj` is `n` with an inject weight and 0 without (the inject
/// block is then zeros; `w_inject` is unread). Threadgroups `(⌈(lr+n)/8⌉, rows)`
/// of 256 threads, a simdgroup per output row. `n` ≤ 8 (the threadgroup
/// stream scales).
#[allow(clippy::too_many_arguments)]
pub fn call_hc_mix_a(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    n: usize,
    d: usize,
    lr: usize,
    n_inj: usize,
    eps: f32,
    scale: f32,
    h: BufferOffset,
    gamma: BufferOffset,
    w_down: BufferOffset,
    w_inject: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    if n > HC_MIX_MAX_N {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "hc_mix_a: {n} streams above the threadgroup scales of {HC_MIX_MAX_N}"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "hc_mix_a_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "hc_mix_a rows={rows} n={n} d={d} lr={lr}");
    set_params!(
        encoder,
        (
            &h,
            &gamma,
            &w_down,
            &w_inject,
            Output::new(out),
            n as u32,
            d as u32,
            lr as u32,
            n_inj as u32,
            rows as u32,
            eps,
            scale
        )
    );
    let gx = (lr + n).div_ceil(HC_MIX_SG * HC_MIX_OUT);
    dispatch_groups_2d(encoder, gx, rows, HC_MIX_SG * 32);
    Ok(())
}

/// `hc_mix_b_f32`: the HC prologue's second launch — the up projection
/// (sigmoid) and the gated collapse over the streams, `out [rows, d]`, from
/// `hc_mix_a`'s buffer. Threadgroups `(⌈d/8⌉, rows)`, a simdgroup per column.
/// `lr` ≤ 1 024 (the threadgroup staging of `lo`).
#[allow(clippy::too_many_arguments)]
pub fn call_hc_mix_b(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    n: usize,
    d: usize,
    lr: usize,
    inv_n: f32,
    packed: BufferOffset,
    w_up: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    if lr > HC_MIX_MAX_LR {
        return Err(MetalKernelError::LoadLibraryError(format!(
            "hc_mix_b: lr {lr} above the threadgroup staging of {HC_MIX_MAX_LR}"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "hc_mix_b_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(encoder, "hc_mix_b rows={rows} n={n} d={d} lr={lr}");
    set_params!(
        encoder,
        (
            &packed,
            &w_up,
            Output::new(out),
            n as u32,
            d as u32,
            lr as u32,
            rows as u32,
            inv_n
        )
    );
    let gx = d.div_ceil(HC_MIX_SG);
    dispatch_groups_2d(encoder, gx, rows, HC_MIX_SG * 32);
    Ok(())
}

/// Simdgroups per threadgroup of the `hc_mix_*` kernels (`HC_MIX_SG`).
const HC_MIX_SG: usize = 8;
/// Output rows per simdgroup in `hc_mix_a` (`HC_MIX_OUT`).
const HC_MIX_OUT: usize = 1;
/// The most `lr` `hc_mix_b` stages in threadgroup memory (`HC_MIX_MAX_LR`).
const HC_MIX_MAX_LR: usize = 1024;
/// The most streams `hc_mix_a` holds a scale for in threadgroup memory
/// (`HC_MAX_N`).
const HC_MIX_MAX_N: usize = 8;

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

/// `qsa_prep_batch_f32`: [`call_qsa_prep`] over `rows` rows in one launch —
/// `qg [rows, nh, 2·hd]`, `k [rows, nkv, hd]`, `pos [rows]` (u32) → `out`
/// as three row-major regions `q [rows, nh, hd] | gate [rows, nh, hd] | k
/// [rows, nkv, hd]`, row for row the single kernel's result.
#[allow(clippy::too_many_arguments)]
pub fn call_qsa_prep_batch(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    rows: usize,
    hd: usize,
    n_rot: usize,
    nh: usize,
    nkv: usize,
    theta: f32,
    eps: f32,
    qg: BufferOffset,
    k: BufferOffset,
    pos: BufferOffset,
    q_gamma: BufferOffset,
    k_gamma: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    if hd > 1024 || n_rot > hd || !n_rot.is_multiple_of(2) {
        return Err(MetalKernelError::LoadFunctionError(format!(
            "qsa_prep_batch: head_dim {hd} (<= 1024), n_rot {n_rot} (even, <= head_dim)"
        )));
    }
    let pipeline = kernels.load_pipeline(device, Source::Qwen4Exp, "qsa_prep_batch_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    debug_group!(
        encoder,
        "qsa_prep_batch rows={rows} hd={hd} nh={nh} nkv={nkv}"
    );
    set_params!(
        encoder,
        (
            &qg,
            &k,
            &pos,
            &q_gamma,
            &k_gamma,
            Output::new(out),
            hd as u32,
            n_rot as u32,
            nh as u32,
            nkv as u32,
            theta,
            eps,
            rows as u32
        )
    );
    dispatch_groups_2d(encoder, nh + nkv, rows, head_width(hd, &pipeline));
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
