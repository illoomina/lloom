//! Gated DeltaNet — the gated delta-rule linear-attention recurrence, as a
//! model-parameterized block.
//!
//! Seven current models share this recurrence (llama.cpp `llm_build_delta_net_base`:
//! Qwen3.5 / 3.5-MoE / 3-Next / 4-exp, Kimi-Linear / K3, Bailing-MoE3), so it is
//! built once here parameterized by dims + the gate variant, not re-hand-rolled
//! per model (§3.6a). Qwen3.8-Flash-Next is the first consumer.
//!
//! The recurrence is **pure tensor-op composition** — the reference's non-fused
//! arms are ggml ops, every one of which candle has, so there is no novel Metal
//! kernel here; the ops stay visible to the fusion optimizer (§3.6, #732). The
//! `S` accumulation is fp32 and mandatory (`mamba_ssm_dtype: float32`); a fused
//! f16 epilogue on that path would corrupt, so it is an optimizer exclusion.
//!
//! State `S ∈ [d_k, d_v, H_v]` per sequence is fixed-size (no context growth)
//! and lives in a [`RecurrentState`] channel — same migrate/rollback/snapshot
//! lifecycle as LFM2's conv window.
//!
//! Reference: `qwen4exp.cpp build_layer_attn_linear` +
//! `delta-net-base.cpp build_delta_net_autoregressive` (decode) /
//! `build_delta_net_chunking` (prefill).

use lloom_tensor::{Result, Tensor, D};

/// The gated delta-rule state and dims for one delta-net layer stack.
///
/// Head geometry (Qwen4-exp): `d_k = d_v = ssm_d_state (128)`,
/// `h_k = ssm_n_group (16)`, `h_v = ssm_dt_rank (48)`, GQA-repeat `h_k -> h_v`.
#[derive(Debug, Clone, Copy)]
pub struct DeltaNetDims {
    /// Key/query head dim (`d_k = d_v`).
    pub d_k: usize,
    /// Value head dim (`= d_k` for this family).
    pub d_v: usize,
    /// Number of key/query heads (`ssm_n_group`).
    pub h_k: usize,
    /// Number of value heads (`ssm_dt_rank`); the state carries `h_v` matrices.
    pub h_v: usize,
}

/// The gated delta-rule decode step (Kernel A), for ONE sequence, ONE token.
///
/// Inputs are already conv'd, gated, L2-normed and GQA-repeated to `h_v` heads:
/// - `q`, `k`: `[h_v, d_k]` (post-l2norm, post-repeat, q pre-scaled here)
/// - `v`: `[h_v, d_v]`
/// - `g`: `[h_v]` the per-head log-decay (`exp` applied here)
/// - `beta`: `[h_v]` the per-head write strength (already `sigmoid`'d)
/// - `state`: `[d_k, d_v, h_v]` fp32, the carried `S` (or `None` = zero state)
///
/// Returns `(o [h_v, d_v], new_state)` where `new_state`'s layout is chosen by
/// `STATE_HEAD_FIRST` (see below). All math in the state's dtype (fp32). Mirrors
/// `build_delta_net_autoregressive` step for step —
/// `g=exp(g); S=g·S; r=k̃ᵀS; d=β(v−r); S += k̃⊗d; o=q̃ᵀS` — with the two
/// read-outs as batched matvecs and the outer product as a matmul.
///
/// # Geometry (const-generic, flash_decoding style — compile-time, no branch)
///
/// `STATE_HEAD_FIRST` selects the `S` storage layout, a geometry axis lfm2
/// measured worth tuning (tiling/stride pays at B>1 + speculative):
/// - `false` (the reference-faithful DEFAULT, the digest anchor): `S` is stored
///   head-LAST `[d_k, d_v, h_v]`, matching the reference recurrent-cache row
///   (`n_embd_s = d_k·d_v·h_v` contiguous). Costs a permute in/out per step.
/// - `true`: `S` is stored head-FIRST `[h_v, d_k, d_v]` — the recurrence's own
///   broadcast layout — so the per-step permute disappears. Our own `S` bytes,
///   not the reference cache's; the channel is shape-agnostic so it does not care.
///
/// The two produce bit-identical OUTPUT (same fp32 ops, only the storage
/// transpose differs), so `false` is the parity/digest baseline and `true` is a
/// perf variant selectable per regime.
pub fn decode_step<const STATE_HEAD_FIRST: bool>(
    dims: DeltaNetDims,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
    let dev = q.device();
    let dt = lloom_tensor::DType::F32; // S is fp32 (mamba_ssm_dtype)

    // Bring the carried state into the recurrence's head-first working layout
    // `[h_v, d_k, d_v]`, whatever it was stored as; zero when unwritten.
    let s = match state {
        Some(s) if STATE_HEAD_FIRST => s.to_dtype(dt)?, // already [h_v, d_k, d_v]
        Some(s) => s.to_dtype(dt)?.permute((2, 0, 1))?.contiguous()?, // [d_k,d_v,h_v]->[h_v,d_k,d_v]
        None => Tensor::zeros((h_v, d_k, d_v), dt, dev)?,
    };

    // Everything to fp32 for the recurrence.
    let q = q.to_dtype(dt)?; // [h_v, d_k]
    let k = k.to_dtype(dt)?; // [h_v, d_k]
    let v = v.to_dtype(dt)?; // [h_v, d_v]
    let g = g.to_dtype(dt)?; // [h_v]
    let beta = beta.to_dtype(dt)?; // [h_v]

    // Decay: S <- exp(g) * S, per-head scalar broadcast over [d_k, d_v].
    let g_exp = g.exp()?.reshape((h_v, 1, 1))?; // [h_v,1,1]
    let s = s.broadcast_mul(&g_exp)?; // [h_v, d_k, d_v]

    let k_col = k.reshape((h_v, d_k, 1))?; // [h_v, d_k, 1]
    let beta_b = beta.reshape((h_v, 1))?; // [h_v,1]
    let (s, o) = if !delta_by_reduce() {
        // The two read-outs as batched matvecs and the rank-1 update as an
        // outer-product matmul — the model doc's "two matvecs + one rank-1
        // update". This is the shipping form: the broadcast-and-reduce form
        // below diverged run to run on the real model at long context (its
        // strided middle-dim `sum(1)` is the implicated kernel), this one is
        // bit-identical across runs.
        let k_row = k.reshape((h_v, 1, d_k))?; // [h_v, 1, d_k]
        let r = k_row.matmul(&s)?.reshape((h_v, d_v))?; // k̃ᵀ S -> [h_v, d_v]
        let d = (v - &r)?.broadcast_mul(&beta_b)?; // [h_v, d_v]
        let kd = k_col.matmul(&d.reshape((h_v, 1, d_v))?)?; // [h_v, d_k, d_v]
        let s = (s + kd)?;
        let q_row = q.reshape((h_v, 1, d_k))?;
        let o = q_row.matmul(&s)?.reshape((h_v, d_v))?; // q̃ᵀ S
        (s, o)
    } else {
        // The broadcast-and-reduce form, kept as the recorded A/B arm
        // (`QWEN4EXP_DELTA_REDUCE=1`), never the default.
        // r = Sᵀ k̃  (read-out): sum over d_k of S[.,dk,dv] * k[.,dk] -> [h_v, d_v].
        let r = s.broadcast_mul(&k_col)?.sum(1)?; // [h_v, d_v]

        // delta d = β (v − r), broadcasting the per-head scalar β over d_v.
        let d = (v - &r)?.broadcast_mul(&beta_b)?; // [h_v, d_v]

        // Rank-1 update: S <- S + k̃ ⊗ d  (outer over d_k × d_v).
        let d_row = d.reshape((h_v, 1, d_v))?; // [h_v,1,d_v]
        let kd = k_col.broadcast_mul(&d_row)?; // [h_v, d_k, d_v]
        let s = (s + kd)?; // [h_v, d_k, d_v]

        // o = Sᵀ q̃ : sum over d_k of S[.,dk,dv] * q[.,dk] -> [h_v, d_v].
        let q_col = q.reshape((h_v, d_k, 1))?;
        let o = s.broadcast_mul(&q_col)?.sum(1)?; // [h_v, d_v]
        (s, o)
    };

    // Store `S` back in the chosen layout.
    let new_state = if STATE_HEAD_FIRST {
        s // [h_v, d_k, d_v]
    } else {
        s.permute((1, 2, 0))?.contiguous()? // [d_k, d_v, h_v]
    };
    Ok((o, new_state))
}

/// Whether Kernel A's read-outs run as broadcast-multiply + strided reduce
/// instead of batched matmuls (`QWEN4EXP_DELTA_REDUCE=1`, read once). The
/// recorded A/B arm: on the real model's 3749-token prompt it diverged run to
/// run (4 of 4 executions) where the matmul form was bit-identical; never the
/// shipping path.
fn delta_by_reduce() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.delta_reduce)
}

/// GQA head expansion `h_k -> h_v` before the recurrence, const-generic over the
/// mapping style (a geometry axis; lfm2 measured strided/tiled head geometries
/// worth tuning at B>1 + speculative).
///
/// `INTERLEAVE`:
/// - `false` (reference-faithful DEFAULT, the digest anchor): TILE, exactly
///   `ggml_repeat` — the whole `[.., h_k]` block repeated `h_v/h_k` times, so
///   v-head `j` reads k-head `j % h_k` (verified from ggml's repeat CPU kernel).
/// - `true`: INTERLEAVE — each k-head repeated `h_v/h_k` consecutively, v-head
///   `j` reads k-head `j / (h_v/h_k)`. A DIFFERENT operator (pairs every v-head
///   with a different k-head); the reference does NOT use it, so it is a
///   deliberate variant, never the parity baseline.
///
/// `x` is `[h_k, d]`, returns `[h_v, d]`.
pub fn gqa_expand<const INTERLEAVE: bool>(x: &Tensor, h_k: usize, h_v: usize) -> Result<Tensor> {
    if h_k == h_v {
        return Ok(x.clone());
    }
    debug_assert_eq!(h_v % h_k, 0, "h_v must be a multiple of h_k");
    let rep = h_v / h_k;
    let d = x.dim(1)?;
    if INTERLEAVE {
        // [h_k, d] -> [h_k, 1, d] -> broadcast [h_k, rep, d] -> [h_v, d]
        x.reshape((h_k, 1, d))?
            .broadcast_as((h_k, rep, d))?
            .reshape((h_v, d))
    } else {
        // TILE (ggml_repeat): [h_k, d] -> [1, h_k, d] -> [rep, h_k, d] -> [h_v, d]
        x.reshape((1, h_k, d))?
            .broadcast_as((rep, h_k, d))?
            .reshape((h_v, d))
    }
}

/// Pre-scale `q` by `1/sqrt(d_k)` as the reference does before the recurrence.
pub fn q_scale(q: &Tensor, d_k: usize) -> Result<Tensor> {
    q.affine(1.0 / (d_k as f64).sqrt(), 0.0)
}

/// L2-normalize the last dim (used on q and k before the recurrence — L2, NOT
/// RMS: `x / sqrt(sum(x^2) + eps)`, no learned weight).
pub fn l2_norm_last(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(lloom_tensor::DType::F32)?;
    let ss = xf.sqr()?.sum_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(ss + eps)?.sqrt()?)?;
    normed.to_dtype(dt)
}

/// Numerically-stable `softplus(x) = log(1 + exp(x))`, via
/// `max(x,0) + log1p(exp(-|x|))` so a large `x` does not overflow `exp`. lloom
/// has no softplus op; this composes it from ops the optimizer can see.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let max0 = x.maximum(&zeros)?; // max(x, 0)
    let neg_abs = x.abs()?.neg()?; // -|x|
                                   // log1p(exp(-|x|)) = log(1 + exp(-|x|)); exp(-|x|) ∈ (0,1] so no overflow.
    let log1p = (neg_abs.exp()? + 1.0)?.log()?;
    max0 + log1p
}

/// Compute the gated-delta gate `g` and write-strength `beta` from the layer's
/// pre-conv input `x [b, seq, d]` and the SSM parameters.
///
/// `beta = sigmoid(ssm_beta · x)`; `alpha = ssm_alpha · x`;
/// `g = ssm_a · softplus(alpha + ssm_dt_bias)`  (`ssm_a` holds `−exp(A_log)`, so
/// `g` is the reference's `−exp(A_log)·softplus(a + dt_bias)`, the log-decay).
/// Returns `(g, beta)` each `[b, seq, h_v]`, in fp32 (the recurrence's dtype).
pub fn gate_and_beta(
    x: &Tensor,
    ssm_beta: &lloom_tensor::quantized::QMatMul,
    ssm_alpha: &lloom_tensor::quantized::QMatMul,
    ssm_dt_bias: &Tensor, // [h_v]
    ssm_a: &Tensor,       // [h_v]  (= −exp(A_log))
) -> Result<(Tensor, Tensor)> {
    use lloom_tensor::Module;
    let f32 = lloom_tensor::DType::F32;
    let beta = crate::nn::ops::sigmoid(&ssm_beta.forward(x)?.to_dtype(f32)?)?;
    let alpha = ssm_alpha.forward(x)?.to_dtype(f32)?; // [b,seq,h_v]
    let a_biased = alpha.broadcast_add(&ssm_dt_bias.to_dtype(f32)?)?;
    let a_sp = softplus(&a_biased)?;
    let g = a_sp.broadcast_mul(&ssm_a.to_dtype(f32)?)?;
    Ok((g, beta))
}

/// The conv's references are the short-conv block's shift window: the step
/// and the chunk, position by position the same (`nn::short_conv`).
pub use crate::nn::short_conv::{
    shift_prefill_reference as conv_prefill, shift_step_reference as conv_decode_step,
};

/// The chunk size Kernel B tiles a sequence into (the reference's `CS` for the
/// gated-delta / GDA case, `delta-net-base.cpp`; FLA's default).
pub const CHUNK_SIZE: usize = 64;

/// The gated delta-rule prefill (Kernel B): the chunkwise WY form of the
/// recurrence over `seq` positions at once, for ONE sequence.
///
/// Inputs are conv'd, gated, L2-normed and GQA-repeated to `h_v` heads, laid
/// out position-major:
/// - `q`, `k`: `[seq, h_v, d_k]` (`q` pre-scaled by `1/√d_k`)
/// - `v`: `[seq, h_v, d_v]`
/// - `g`: `[seq, h_v]` the per-position log-decay (`exp` applied here)
/// - `beta`: `[seq, h_v]` the write strength (already `sigmoid`'d)
/// - `state`: `[d_k, d_v, h_v]` fp32 head-last (the channel's layout; `None` = zero)
///
/// Returns `(o [seq, h_v, d_v], new_state [d_k, d_v, h_v])`, all fp32.
///
/// Per chunk of `chunk` positions (reference `build_delta_net_chunking`, FLA
/// `chunk_gated_delta_rule`): with `G = cumsum(g)` inside the chunk and
/// `decay[b, a] = exp(G[b] − G[a])` for `a ≤ b`,
/// `A = tril₋₁(diag(β) K Kᵀ ⊙ decay)`, `T = (I + A)⁻¹` (unit-lower, so the
/// nilpotent product `∏ (I + (−A)^{2ⁱ})`), `U = T·diag(β)V`,
/// `W = T·(diag(β)K ⊙ exp(G))`; then chunk by chunk `V' = U − W·S`,
/// `O = (Q ⊙ exp(G))·S + (Q Kᵀ ⊙ decay)·V'`, `S ← exp(G_last)·S + (K ⊙
/// exp(G_last − G))ᵀ·V'`. Positions past `seq` are zero-padded (`g = 0`,
/// `k = v = 0`: no decay, no write). Pure tensor-op composition, fp32.
#[allow(clippy::too_many_arguments)]
pub fn chunk_forward(
    dims: DeltaNetDims,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
    chunk: usize,
) -> Result<(Tensor, Tensor)> {
    let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
    let dev = q.device();
    let dt = lloom_tensor::DType::F32;
    let (seq, hq, dq) = q.dims3()?;
    if hq != h_v || dq != d_k || chunk == 0 {
        lloom_tensor::bail!(
            "chunk_forward: q {:?} vs dims {dims:?} chunk {chunk}",
            q.shape()
        )
    }
    let pad = (chunk - seq % chunk) % chunk;
    let padded = seq + pad;
    let nc = padded / chunk;

    // Head-first, chunked: [h_v, nc, cs, d].
    let to_chunks = |t: &Tensor, d: usize| -> Result<Tensor> {
        let t = t.to_dtype(dt)?.transpose(0, 1)?.contiguous()?; // [h_v, seq, d]
        let t = if pad > 0 {
            t.pad_with_zeros(1, 0, pad)?
        } else {
            t
        };
        t.reshape((h_v, nc, chunk, d))
    };
    let q = to_chunks(q, d_k)?;
    let k = to_chunks(k, d_k)?;
    let v = to_chunks(v, d_v)?;
    let g = to_chunks(&g.unsqueeze(2)?, 1)?.squeeze(3)?; // [h_v, nc, cs]
    let b = to_chunks(&beta.unsqueeze(2)?, 1)?; // [h_v, nc, cs, 1]

    let s0 = match state {
        Some(s) => s.to_dtype(dt)?.permute((2, 0, 1))?.contiguous()?, // [h_v, d_k, d_v]
        None => Tensor::zeros((h_v, d_k, d_v), dt, dev)?,
    };

    // Cumulative log-decay inside each chunk, and the pairwise decay matrix
    // exp(G[b] − G[a]) masked to a ≤ b (the masked entries are set to a large
    // negative BEFORE the exp, so no inf·0).
    let g_cs = g.cumsum(2)?; // [h_v, nc, cs]
    let gi = g_cs.unsqueeze(3)?.broadcast_as((h_v, nc, chunk, chunk))?; // row b
    let gj = g_cs.unsqueeze(2)?.broadcast_as((h_v, nc, chunk, chunk))?; // col a
    let diff = (gi - gj)?;
    let lower = Tensor::tril2(chunk, lloom_tensor::DType::U8, dev)?
        .broadcast_as((h_v, nc, chunk, chunk))?
        .contiguous()?;
    let strict = (Tensor::tril2(chunk, lloom_tensor::DType::U8, dev)?
        - Tensor::eye(chunk, lloom_tensor::DType::U8, dev)?)?
    .broadcast_as((h_v, nc, chunk, chunk))?
    .contiguous()?;
    let neg = Tensor::full(-1e30f32, (h_v, nc, chunk, chunk), dev)?;
    let decay = lower.where_cond(&diff, &neg)?.exp()?; // 0 above the diagonal
    let zero = Tensor::zeros((h_v, nc, chunk, chunk), dt, dev)?;

    let k_beta = k.broadcast_mul(&b)?; // [h_v, nc, cs, d_k]
    let v_beta = v.broadcast_mul(&b)?; // [h_v, nc, cs, d_v]
    let k_t = k.transpose(2, 3)?.contiguous()?; // [h_v, nc, d_k, cs]

    // A = tril₋₁((β K) Kᵀ ⊙ decay); T = (I + A)⁻¹ by forward substitution.
    let a = k_beta.matmul(&k_t)?.mul(&decay)?; // [h_v, nc, cs(b), cs(a)]
    let a = strict.where_cond(&a, &zero)?;
    let t = unit_lower_inverse(&a)?;

    let g_exp = g_cs.exp()?.unsqueeze(3)?; // [h_v, nc, cs, 1]
    let u = t.matmul(&v_beta)?; // [h_v, nc, cs, d_v]
    let w = t.matmul(&k_beta.broadcast_mul(&g_exp)?)?; // [h_v, nc, cs, d_k]
    let q_g = q.broadcast_mul(&g_exp)?; // [h_v, nc, cs, d_k]
    let kq = q.matmul(&k_t)?.mul(&decay)?; // [h_v, nc, cs(b), cs(a)], a ≤ b
    let g_last = g_cs.narrow(2, chunk - 1, 1)?; // [h_v, nc, 1]
    let g_diff = g_last.broadcast_sub(&g_cs)?.exp()?.unsqueeze(3)?; // [h_v, nc, cs, 1]
    let kg_t = k.broadcast_mul(&g_diff)?.transpose(2, 3)?.contiguous()?; // [h_v, nc, d_k, cs]
    let g_last_exp = g_last.exp()?; // [h_v, nc, 1]

    let mut s = s0;
    let mut outs = Vec::with_capacity(nc);
    for c in 0..nc {
        let w_c = w.narrow(1, c, 1)?.squeeze(1)?; // [h_v, cs, d_k]
        let u_c = u.narrow(1, c, 1)?.squeeze(1)?; // [h_v, cs, d_v]
        let v_new = (u_c - w_c.matmul(&s)?)?; // [h_v, cs, d_v]
        let q_g_c = q_g.narrow(1, c, 1)?.squeeze(1)?;
        let kq_c = kq.narrow(1, c, 1)?.squeeze(1)?;
        let o_c = (q_g_c.matmul(&s)? + kq_c.matmul(&v_new)?)?; // [h_v, cs, d_v]
        outs.push(o_c);
        let kg_c = kg_t.narrow(1, c, 1)?.squeeze(1)?; // [h_v, d_k, cs]
        let gl = g_last_exp.narrow(1, c, 1)?.reshape((h_v, 1, 1))?;
        s = (s.broadcast_mul(&gl)? + kg_c.matmul(&v_new)?)?;
    }
    let o = Tensor::cat(&outs, 1)?; // [h_v, padded, d_v]
    let o = o.narrow(1, 0, seq)?.transpose(0, 1)?.contiguous()?; // [seq, h_v, d_v]
    let new_state = s.permute((1, 2, 0))?.contiguous()?; // [d_k, d_v, h_v]
    Ok((o, new_state))
}

/// `(I + A)⁻¹` for a strictly-lower-triangular `A [.., n, n]` (batched over the
/// leading dims), by forward substitution: row `i` of the inverse is
/// `e_i − Σ_{j<i} A[i, j] · T[j, :]`, each row from the rows before it.
///
/// This is the reference's `solve_tri` / FLA's row loop, and it is the
/// numerically stable form. The nilpotent identity `(I + A)⁻¹ = ∏ (I +
/// (−A)^{2ⁱ})` is exact in arithmetic and catastrophic in fp32 on real keys:
/// within a chunk the L2-normed keys are near-identical, `β ≈ 1` and the
/// decay ≈ 1, so `A` is close to the all-ones strictly-lower matrix, whose
/// powers reach `C(63, 31) ≈ 4.6e17` while the inverse is bidiagonal `±1`;
/// summing those terms in fp32 leaves garbage of ~1e11 that overflows the
/// state two chunks later (measured: NaN at layer 0 on a 643-token prompt).
pub fn unit_lower_inverse(a: &Tensor) -> Result<Tensor> {
    // One launch on Metal (`kernels::UnitLowerInverse`, the same substitution
    // row by row); the composed loop below is the CPU path and its reference.
    if super::kernels::unit_lower_inverse_applies(a) {
        return super::kernels::unit_lower_inverse(a);
    }
    unit_lower_inverse_composed(a)
}

/// [`unit_lower_inverse`] as tensor ops: an `n`-row host loop, each row a
/// `narrow`/`cat`/`matmul` — the reference, and the CPU path.
pub fn unit_lower_inverse_composed(a: &Tensor) -> Result<Tensor> {
    let dims = a.dims().to_vec();
    let n = dims[dims.len() - 1];
    if dims[dims.len() - 2] != n {
        lloom_tensor::bail!("unit_lower_inverse: not square {:?}", a.shape())
    }
    let dt = a.dtype();
    let dev = a.device();
    let lead: Vec<usize> = dims[..dims.len() - 2].to_vec();
    let neg_a = a.neg()?;
    // rows[i] is [.., 1, n]
    let mut rows: Vec<Tensor> = Vec::with_capacity(n);
    let mut row_shape = lead.clone();
    row_shape.extend([1, n]);
    let e = Tensor::eye(n, dt, dev)?; // [n, n]
    for i in 0..n {
        let e_i = e
            .narrow(0, i, 1)?
            .broadcast_as(row_shape.as_slice())?
            .contiguous()?;
        if i == 0 {
            rows.push(e_i);
            continue;
        }
        // T[i, :] = e_i + (−A)[i, :i] · T[:i, :]
        let coeffs = neg_a
            .narrow(dims.len() - 2, i, 1)?
            .narrow(dims.len() - 1, 0, i)?; // [.., 1, i]
        let prev = Tensor::cat(&rows, dims.len() - 2)?; // [.., i, n]
        let t_i = (e_i + coeffs.matmul(&prev)?)?; // [.., 1, n]
        rows.push(t_i);
    }
    Tensor::cat(&rows, dims.len() - 2)
}

/// [`gqa_expand`] over a position-major `[seq, h_k, d]` → `[seq, h_v, d]`,
/// the same head map (tile: v-head `j` reads k-head `j % h_k`).
pub fn gqa_expand_seq<const INTERLEAVE: bool>(
    x: &Tensor,
    h_k: usize,
    h_v: usize,
) -> Result<Tensor> {
    if h_k == h_v {
        return Ok(x.clone());
    }
    let (seq, _, d) = x.dims3()?;
    let rep = h_v / h_k;
    if INTERLEAVE {
        x.reshape((seq, h_k, 1, d))?
            .broadcast_as((seq, h_k, rep, d))?
            .reshape((seq, h_v, d))
    } else {
        x.reshape((seq, 1, h_k, d))?
            .broadcast_as((seq, rep, h_k, d))?
            .reshape((seq, h_v, d))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::{Device, IndexOp};

    /// A reference decode step computed with plain nested loops, to pin
    /// `decode_step` against the delta-rule math directly (§3.1 parity — the
    /// reference is an independent scalar implementation, not a restatement).
    fn ref_step(
        dims: DeltaNetDims,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        s: &mut [f32], // [h_v * d_k * d_v], layout [h][dk][dv]
    ) -> Vec<f32> {
        let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
        let mut o = vec![0f32; h_v * d_v];
        for h in 0..h_v {
            let ge = g[h].exp();
            let sb = h * d_k * d_v;
            // S <- ge * S
            for x in &mut s[sb..sb + d_k * d_v] {
                *x *= ge;
            }
            // r[dv] = sum_dk S[dk,dv] * k[dk]
            let mut r = vec![0f32; d_v];
            for dk in 0..d_k {
                let kk = k[h * d_k + dk];
                for dv in 0..d_v {
                    r[dv] += s[sb + dk * d_v + dv] * kk;
                }
            }
            // d[dv] = beta * (v - r)
            let mut dd = vec![0f32; d_v];
            for dv in 0..d_v {
                dd[dv] = beta[h] * (v[h * d_v + dv] - r[dv]);
            }
            // S += k outer d
            for dk in 0..d_k {
                let kk = k[h * d_k + dk];
                for dv in 0..d_v {
                    s[sb + dk * d_v + dv] += kk * dd[dv];
                }
            }
            // o[dv] = sum_dk S[dk,dv] * q[dk]
            for dk in 0..d_k {
                let qq = q[h * d_k + dk];
                for dv in 0..d_v {
                    o[h * d_v + dv] += s[sb + dk * d_v + dv] * qq;
                }
            }
        }
        o
    }

    #[test]
    fn decode_step_matches_scalar_reference() {
        let dev = Device::Cpu;
        let dims = DeltaNetDims {
            d_k: 3,
            d_v: 3,
            h_k: 2,
            h_v: 2,
        };
        let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
        // Deterministic pseudo-inputs.
        let mk = |n: usize, seed: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * 0.37 + seed).sin()).collect()
        };
        let q = mk(h_v * d_k, 0.1);
        let k = mk(h_v * d_k, 0.2);
        let v = mk(h_v * d_v, 0.3);
        let g = mk(h_v, 0.4);
        let beta = mk(h_v, 0.5);

        // Scalar reference over an initial nonzero state.
        let s0: Vec<f32> = mk(h_v * d_k * d_v, 0.6);
        let mut s_ref = s0.clone();
        let o_ref = ref_step(dims, &q, &k, &v, &g, &beta, &mut s_ref);

        // Tensor path: state as [d_k, d_v, h_v] (storage layout).
        let mut s_store = vec![0f32; h_v * d_k * d_v];
        for h in 0..h_v {
            for dk in 0..d_k {
                for dv in 0..d_v {
                    s_store[(dk * d_v + dv) * h_v + h] = s0[h * d_k * d_v + dk * d_v + dv];
                }
            }
        }
        let s_t = Tensor::from_vec(s_store, (d_k, d_v, h_v), &dev).unwrap();
        let qt = Tensor::from_vec(q, (h_v, d_k), &dev).unwrap();
        let kt = Tensor::from_vec(k, (h_v, d_k), &dev).unwrap();
        let vt = Tensor::from_vec(v, (h_v, d_v), &dev).unwrap();
        let gt = Tensor::from_vec(g, h_v, &dev).unwrap();
        let bt = Tensor::from_vec(beta, h_v, &dev).unwrap();

        let (o, _new_s) = decode_step::<false>(dims, &qt, &kt, &vt, &gt, &bt, Some(&s_t)).unwrap();
        let o_got = o.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in o_got.iter().zip(o_ref.iter()) {
            assert!((a - b).abs() < 1e-5, "got {o_got:?} ref {o_ref:?}");
        }
    }

    /// The two `STATE_HEAD_FIRST` geometries produce bit-identical OUTPUT (only
    /// the storage transpose differs) — the property that lets `true` be a perf
    /// variant and `false` the digest anchor. Feed head-first the head-last
    /// state permuted, and check `o` matches.
    #[test]
    fn state_layout_variants_agree_on_output() {
        let dev = Device::Cpu;
        let dims = DeltaNetDims {
            d_k: 3,
            d_v: 3,
            h_k: 2,
            h_v: 2,
        };
        let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
        let mk = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * 0.29 + s).cos()).collect()
        };
        let qt = Tensor::from_vec(mk(h_v * d_k, 0.1), (h_v, d_k), &dev).unwrap();
        let kt = Tensor::from_vec(mk(h_v * d_k, 0.2), (h_v, d_k), &dev).unwrap();
        let vt = Tensor::from_vec(mk(h_v * d_v, 0.3), (h_v, d_v), &dev).unwrap();
        let gt = Tensor::from_vec(mk(h_v, 0.4), h_v, &dev).unwrap();
        let bt = Tensor::from_vec(mk(h_v, 0.5), h_v, &dev).unwrap();
        // head-last state, and its head-first permutation.
        let s_last = Tensor::from_vec(mk(d_k * d_v * h_v, 0.6), (d_k, d_v, h_v), &dev).unwrap();
        let s_first = s_last.permute((2, 0, 1)).unwrap().contiguous().unwrap();

        let (o_last, ns_last) =
            decode_step::<false>(dims, &qt, &kt, &vt, &gt, &bt, Some(&s_last)).unwrap();
        let (o_first, ns_first) =
            decode_step::<true>(dims, &qt, &kt, &vt, &gt, &bt, Some(&s_first)).unwrap();
        let a = o_last.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = o_first.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "outputs differ: {a:?} vs {b:?}");
        }
        // And the stored states are transposes of each other.
        assert_eq!(ns_last.dims(), &[d_k, d_v, h_v]);
        assert_eq!(ns_first.dims(), &[h_v, d_k, d_v]);
        let nl = ns_last
            .permute((2, 0, 1))
            .unwrap()
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let nf = ns_first.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in nl.iter().zip(&nf) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    /// softplus matches log(1+exp(x)) including at large |x| (no overflow).
    #[test]
    fn softplus_stable() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![-40f32, -1.0, 0.0, 1.0, 40.0], 5, &dev).unwrap();
        let got = softplus(&x).unwrap().to_vec1::<f32>().unwrap();
        let want = [
            (1f32 + (-40f32).exp()).ln(),
            (1f32 + (-1f32).exp()).ln(),
            std::f32::consts::LN_2,
            (1f32 + 1f32.exp()).ln(),
            40.0 + (1f32 + (-40f32).exp()).ln(), // large x: ~x
        ];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "softplus {got:?} vs {want:?}");
        }
        // no NaN/inf at the extreme.
        assert!(got.iter().all(|v| v.is_finite()));
    }

    /// GQA tile (reference) vs interleave are DIFFERENT operators: v-head j reads
    /// k-head j%h_k (tile) vs j/(h_v/h_k) (interleave). Pin both mappings.
    #[test]
    fn gqa_tile_vs_interleave() {
        let dev = Device::Cpu;
        let (h_k, h_v, d) = (2usize, 4usize, 2usize);
        // heads 0,1 distinguishable: head0=[1,1], head1=[2,2].
        let x = Tensor::from_vec(vec![1f32, 1.0, 2.0, 2.0], (h_k, d), &dev).unwrap();
        let tile = gqa_expand::<false>(&x, h_k, h_v).unwrap();
        let inter = gqa_expand::<true>(&x, h_k, h_v).unwrap();
        // tile: [h0,h1,h0,h1] -> heads = 1,2,1,2 (per first col)
        let tv = tile.i((.., 0)).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(tv, vec![1.0, 2.0, 1.0, 2.0]);
        // interleave: [h0,h0,h1,h1] -> heads = 1,1,2,2
        let iv = inter.i((.., 0)).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(iv, vec![1.0, 1.0, 2.0, 2.0]);
    }

    /// Mutation: a broken decay (skip `exp`) must fail the parity check — proves
    /// the test can fail (§3.1 mutation).
    #[test]
    fn parity_is_not_vacuous() {
        // Sanity that the reference itself is sensitive to the gate: two
        // different g give different outputs, so a test comparing them is live.
        let dev = Device::Cpu;
        let dims = DeltaNetDims {
            d_k: 2,
            d_v: 2,
            h_k: 1,
            h_v: 1,
        };
        let q = Tensor::from_vec(vec![1f32, 0.5], (1, 2), &dev).unwrap();
        let k = Tensor::from_vec(vec![0.3f32, 0.7], (1, 2), &dev).unwrap();
        let v = Tensor::from_vec(vec![1f32, -1.0], (1, 2), &dev).unwrap();
        let b = Tensor::from_vec(vec![0.9f32], 1, &dev).unwrap();
        let s = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (2, 2, 1), &dev).unwrap();
        let g_a = Tensor::from_vec(vec![0.0f32], 1, &dev).unwrap();
        let g_b = Tensor::from_vec(vec![-2.0f32], 1, &dev).unwrap();
        let (oa, _) = decode_step::<false>(dims, &q, &k, &v, &g_a, &b, Some(&s)).unwrap();
        let (ob, _) = decode_step::<false>(dims, &q, &k, &v, &g_b, &b, Some(&s)).unwrap();
        let va = oa.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = ob.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(va.iter().zip(&vb).any(|(a, b)| (a - b).abs() > 1e-4));
    }

    /// Kernel B equals Kernel A replayed: the chunked prefill's outputs and
    /// final state match the sequential decode over the same positions, with
    /// a nonzero initial state, at a sequence that is not a multiple of the
    /// chunk (padding) and spans several chunks. fp32 reorder only.
    #[test]
    fn chunk_forward_matches_sequential_decode() {
        let dev = Device::Cpu;
        let dims = DeltaNetDims {
            d_k: 6,
            d_v: 5,
            h_k: 2,
            h_v: 4,
        };
        let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
        for (seq, chunk) in [(1usize, 4usize), (3, 4), (4, 4), (11, 4), (23, 8), (7, 64)] {
            let mk = |n: usize, seed: f32, scale: f32| -> Vec<f32> {
                (0..n)
                    .map(|i| ((i as f32 * 0.731 + seed).sin()) * scale)
                    .collect()
            };
            let q = Tensor::from_vec(mk(seq * h_v * d_k, 0.1, 0.8), (seq, h_v, d_k), &dev).unwrap();
            let k = Tensor::from_vec(mk(seq * h_v * d_k, 0.2, 0.8), (seq, h_v, d_k), &dev).unwrap();
            let v = Tensor::from_vec(mk(seq * h_v * d_v, 0.3, 1.0), (seq, h_v, d_v), &dev).unwrap();
            // log-decay in (−3, 0), β in (0, 1)
            let g = Tensor::from_vec(
                mk(seq * h_v, 0.4, 1.0)
                    .iter()
                    .map(|x| -1.5 - 1.4 * x)
                    .collect::<Vec<_>>(),
                (seq, h_v),
                &dev,
            )
            .unwrap();
            let beta = Tensor::from_vec(
                mk(seq * h_v, 0.5, 1.0)
                    .iter()
                    .map(|x| 0.5 + 0.45 * x)
                    .collect::<Vec<_>>(),
                (seq, h_v),
                &dev,
            )
            .unwrap();
            let s0 =
                Tensor::from_vec(mk(d_k * d_v * h_v, 0.6, 0.3), (d_k, d_v, h_v), &dev).unwrap();

            // Sequential reference: Kernel A per position.
            let mut s_ref = s0.clone();
            let mut o_ref = Vec::with_capacity(seq);
            for t in 0..seq {
                let (o, ns) = decode_step::<false>(
                    dims,
                    &q.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                    &k.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                    &v.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                    &g.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                    &beta.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                    Some(&s_ref),
                )
                .unwrap();
                o_ref.push(o);
                s_ref = ns;
            }
            let o_ref = Tensor::stack(&o_ref, 0).unwrap();

            let (o, s) = chunk_forward(dims, &q, &k, &v, &g, &beta, Some(&s0), chunk).unwrap();
            assert_eq!(o.dims(), &[seq, h_v, d_v]);
            assert_eq!(s.dims(), &[d_k, d_v, h_v]);
            let (a, b) = (
                o.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                o_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            );
            let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            let maxdiff = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                maxdiff <= 2e-5 * scale,
                "seq {seq} chunk {chunk}: outputs differ by {maxdiff} at scale {scale}"
            );
            let (sa, sb) = (
                s.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                s_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            );
            let sscale = sb.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            let smax = sa
                .iter()
                .zip(&sb)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                smax <= 2e-5 * sscale,
                "seq {seq} chunk {chunk}: states differ by {smax} at scale {sscale}"
            );
            assert!(scale > 1e-3 && sscale > 1e-3, "degenerate fixture");
        }
    }

    /// The real-model regime that broke the first Kernel B: near-identical
    /// L2-normed keys, β ≈ 1, decay ≈ 1, a full 64-chunk. Kernel B must still
    /// equal Kernel A replayed (fp32 reorder only).
    #[test]
    fn chunk_forward_is_stable_on_near_identical_keys() {
        let dev = Device::Cpu;
        let dims = DeltaNetDims {
            d_k: 8,
            d_v: 8,
            h_k: 1,
            h_v: 1,
        };
        let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
        let seq = 130;
        // keys: a unit vector plus a tiny per-position perturbation, renormalized.
        let mut kv_ = Vec::with_capacity(seq * d_k);
        for t in 0..seq {
            let mut v: Vec<f32> = (0..d_k)
                .map(|i| 1.0 + 0.01 * ((t * 7 + i) as f32 * 0.3).sin())
                .collect();
            let nrm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            for x in &mut v {
                *x /= nrm;
            }
            kv_.extend(v);
        }
        let k = Tensor::from_vec(kv_.clone(), (seq, h_v, d_k), &dev).unwrap();
        let q = Tensor::from_vec(
            kv_.iter().map(|x| x * 0.9).collect::<Vec<_>>(),
            (seq, h_v, d_k),
            &dev,
        )
        .unwrap();
        let v = Tensor::from_vec(
            (0..seq * d_v)
                .map(|i| (i as f32 * 0.11).cos())
                .collect::<Vec<_>>(),
            (seq, h_v, d_v),
            &dev,
        )
        .unwrap();
        let g = Tensor::from_vec(vec![-0.001f32; seq * h_v], (seq, h_v), &dev).unwrap();
        let beta = Tensor::from_vec(vec![0.999f32; seq * h_v], (seq, h_v), &dev).unwrap();
        let mut s_ref: Option<Tensor> = None;
        let mut o_ref = Vec::with_capacity(seq);
        for t in 0..seq {
            let (o, ns) = decode_step::<false>(
                dims,
                &q.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                &k.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                &v.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                &g.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                &beta.narrow(0, t, 1).unwrap().squeeze(0).unwrap(),
                s_ref.as_ref(),
            )
            .unwrap();
            o_ref.push(o);
            s_ref = Some(ns);
        }
        let o_ref = Tensor::stack(&o_ref, 0).unwrap();
        let (o, s) = chunk_forward(dims, &q, &k, &v, &g, &beta, None, 64).unwrap();
        let (a, b) = (
            o.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            o_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        assert!(a.iter().all(|x| x.is_finite()), "non-finite output");
        let scale = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
        let maxdiff = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            maxdiff <= 1e-4 * scale,
            "outputs differ by {maxdiff} at scale {scale}"
        );
        let (sa, sb) = (
            s.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            s_ref
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
        );
        let sscale = sb.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
        let smax = sa
            .iter()
            .zip(&sb)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max);
        assert!(
            smax <= 1e-4 * sscale,
            "states differ by {smax} at scale {sscale}"
        );
    }

    /// Forward substitution inverts `I + A` for a strictly-lower `A`, batched,
    /// and stays exact on the hard case — the all-ones strictly-lower matrix
    /// at n = 64, whose inverse is bidiagonal (1 on the diagonal, −1 below)
    /// while its Neumann terms reach ~1e17. The nilpotent doubling product
    /// this replaced fails this fixture by ~1e11 (the NaN on the real model).
    /// The fused inverse (one launch) equals the composed loop on the hard
    /// case's shape — near-identical keys, β ≈ 1, decay ≈ 1, `A` close to the
    /// all-ones strictly-lower matrix — at the chunk size, batched over heads.
    #[cfg(feature = "metal")]
    #[test]
    fn unit_lower_inverse_fused_matches_composed() {
        let gpu = Device::new_metal(0).unwrap();
        let (b, n) = (6usize, CHUNK_SIZE);
        let mut v = vec![0f32; b * n * n];
        for bi in 0..b {
            for i in 0..n {
                for j in 0..i {
                    let x = ((i * 7 + j * 3 + bi) as f32 * 0.37).sin();
                    v[bi * n * n + i * n + j] = 0.97 + 0.03 * x;
                }
            }
        }
        let a = Tensor::from_vec(v, (b, n, n), &gpu).unwrap();
        let fused = unit_lower_inverse(&a).unwrap();
        let composed = unit_lower_inverse_composed(&a).unwrap();
        let (x, y) = (
            fused.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            composed.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
        let scale = y.iter().fold(0f32, |m, v| m.max(v.abs()));
        let maxdiff = x
            .iter()
            .zip(&y)
            .map(|(p, q)| (p - q).abs())
            .fold(0f32, f32::max);
        // The inverse of a near-all-ones strictly-lower matrix is near
        // bidiagonal ±1: the scale is ~1 and the off-diagonal is live.
        assert!(scale >= 1.0, "a degenerate inverse: max |T| = {scale}");
        let off: f32 = (1..n).map(|i| y[i * n + i - 1].abs()).sum();
        assert!(
            off > 0.5 * (n - 1) as f32,
            "the sub-diagonal is dead: {off}"
        );
        assert!(
            maxdiff <= 2e-5 * scale,
            "fused differs from composed by {maxdiff} at scale {scale}"
        );
        // non-vacuity: the fused path ran (it refuses a non-f32 / cpu input)
        assert!(super::super::kernels::unit_lower_inverse_applies(&a));
    }

    #[test]
    fn unit_lower_inverse_is_stable_on_the_hard_case() {
        let dev = Device::Cpu;
        let n = 64usize;
        let dt = lloom_tensor::DType::F32;
        let strict =
            (Tensor::tril2(n, dt, &dev).unwrap() - Tensor::eye(n, dt, &dev).unwrap()).unwrap();
        // batched: [2, n, n] with the all-ones case and a random case
        let dense = Tensor::from_vec(
            (0..n * n)
                .map(|i| (i as f32 * 0.37).sin())
                .collect::<Vec<_>>(),
            (n, n),
            &dev,
        )
        .unwrap();
        let random = (dense * &strict).unwrap();
        let a = Tensor::stack(&[&strict, &random], 0).unwrap();
        let t = unit_lower_inverse(&a).unwrap();
        assert_eq!(t.dims(), &[2, n, n]);
        let eye = Tensor::eye(n, dt, &dev).unwrap();
        let ones_t = t.get(0).unwrap().to_vec2::<f32>().unwrap();
        for i in 0..n {
            for j in 0..n {
                let want = if i == j {
                    1.0
                } else if j + 1 == i {
                    -1.0
                } else {
                    0.0
                };
                assert!(
                    (ones_t[i][j] - want).abs() < 1e-6,
                    "all-ones inverse at ({i},{j}): {}",
                    ones_t[i][j]
                );
            }
        }
        let prod = (&eye + a.get(1).unwrap())
            .unwrap()
            .matmul(&t.get(1).unwrap())
            .unwrap();
        let p = prod.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let e = eye.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in p.iter().zip(&e) {
            assert!((x - y).abs() < 1e-4, "(I+A)·T is not I");
        }
    }
}
