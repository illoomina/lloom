//! The `GatedDeltaNet` weight-bearing block: loads one linear-attention layer's
//! weights and runs its decode forward, composing the tested primitives
//! (`deltanet::*`) and the engine's two GDN blocks — the short conv
//! ([`ShortConv`]) and the recurrence step ([`GdnStepArm`]) — with the
//! [`RecurrentState`] channels for `S` and the conv window.
//!
//! Reference dataflow: `qwen4exp.cpp build_layer_attn_linear`. The block is
//! model-parameterized by [`Config`] so a second delta-net model reuses it.

use super::config::Config;
use super::deltanet::{
    chunk_forward, conv_decode_step, conv_prefill, decode_step, gate_and_beta, gqa_expand,
    gqa_expand_seq, l2_norm_last, q_scale, DeltaNetDims, CHUNK_SIZE,
};
use crate::nn::gdn::{GdnStepArm, GdnStepForm, OutGate};
use crate::nn::short_conv::{Shift, ShortConv};
use crate::nn::{ops, RecurrentState};
use crate::profile::Profile;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Module, Result, Tensor, D};
use std::marker::PhantomData;

/// One Gated DeltaNet layer's weights.
///
/// `QMatMul` holds each projection (quantized IQ/Q2_0 or BF16 — it dequantizes
/// on the fly), so the block reaches quant weights through the one seam
/// (`call_quantized_matmul_mv_t`), never a bespoke dequant.
#[derive(Debug, Clone)]
pub struct GatedDeltaNet<P: Profile> {
    profile: PhantomData<P>,
    /// Fused QKV in-proj → `[conv_channels]` (`attn_qkv`).
    wqkv: QMatMul,
    /// Output-gate in-proj → `[value_dim]` (`attn_gate`).
    wqkv_gate: QMatMul,
    /// The short conv over the concat-QKV channels (`ssm_conv1d`, `[conv_channels,
    /// k]`, k=4), in the shift layout its window channel carries.
    conv: ShortConv<Shift>,
    /// The recurrence step, in the form the construction tier chose.
    step: GdnStepArm,
    /// `-exp(A_log)` per value-head `[h_v]` (`ssm_a`).
    ssm_a: Tensor,
    /// dt bias `[h_v]` (`ssm_dt.bias`).
    ssm_dt: Tensor,
    /// `ssm_alpha` in-proj → `[h_v]`.
    ssm_alpha: QMatMul,
    /// `ssm_beta` in-proj → `[h_v]`.
    ssm_beta: QMatMul,
    /// GatedRMSNorm weight `[d_v]` (`ssm_norm`).
    ssm_norm: Tensor,
    /// out-proj `[value_dim → hidden]` (`ssm_out`).
    ssm_out: QMatMul,
    /// The four input stems (`wqkv | wqkv_gate | ssm_alpha | ssm_beta`) as
    /// one f16 matrix when the loader could stack them (the parts above are
    /// row views of it): a decode step runs one gemv and splits the row.
    stem_cat: Option<QMatMul>,
    /// The two gate stems (`ssm_alpha | ssm_beta`) as one f16 matrix when
    /// only those stack (the projections quantized): one gemv for the pair.
    gate_stems_cat: Option<QMatMul>,
    dims: DeltaNetDims,
    conv_channels: usize,
    value_dim: usize,
    key_dim: usize,
    norm_eps: f64,
    /// The gated norm's output gate (`OutGate`): sigmoid on Qwen4-exp, SiLU
    /// on Qwen3.5.
    gate: OutGate,
}

/// What a Gated DeltaNet layer's blocks need of a family's config: the head
/// geometry, the conv width, the norm epsilon and the output gate. A family
/// builds one from its own config ([`From<&Config>`] is Qwen4-exp's).
#[derive(Debug, Clone, Copy)]
pub struct GdnGeometry {
    pub dims: DeltaNetDims,
    pub conv_kernel: usize,
    pub norm_eps: f64,
    pub gate: OutGate,
}

impl GdnGeometry {
    /// The concat-QKV conv channel count, `key_dim · 2 + value_dim`.
    pub fn conv_channels(&self) -> usize {
        self.key_dim() * 2 + self.value_dim()
    }
    pub fn value_dim(&self) -> usize {
        self.dims.d_v * self.dims.h_v
    }
    pub fn key_dim(&self) -> usize {
        self.dims.d_k * self.dims.h_k
    }
}

impl From<&Config> for GdnGeometry {
    fn from(cfg: &Config) -> Self {
        Self {
            dims: cfg.deltanet_dims(),
            conv_kernel: cfg.ssm_conv_kernel,
            norm_eps: cfg.norm_eps,
            gate: OutGate::Sigmoid,
        }
    }
}

impl<P: Profile> GatedDeltaNet<P> {
    /// The per-layer geometry, from the config.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wqkv: QMatMul,
        wqkv_gate: QMatMul,
        conv_weight: Tensor,
        ssm_a: Tensor,
        ssm_dt: Tensor,
        ssm_alpha: QMatMul,
        ssm_beta: QMatMul,
        ssm_norm: Tensor,
        ssm_out: QMatMul,
        cfg: &Config,
    ) -> Self {
        Self::with_geometry(
            wqkv,
            wqkv_gate,
            conv_weight,
            ssm_a,
            ssm_dt,
            ssm_alpha,
            ssm_beta,
            ssm_norm,
            ssm_out,
            &GdnGeometry::from(cfg),
        )
    }

    /// [`Self::new`] from the geometry alone, for a family whose config is
    /// not Qwen4-exp's.
    #[allow(clippy::too_many_arguments)]
    pub fn with_geometry(
        wqkv: QMatMul,
        wqkv_gate: QMatMul,
        conv_weight: Tensor,
        ssm_a: Tensor,
        ssm_dt: Tensor,
        ssm_alpha: QMatMul,
        ssm_beta: QMatMul,
        ssm_norm: Tensor,
        ssm_out: QMatMul,
        geo: &GdnGeometry,
    ) -> Self {
        let dims = geo.dims;
        Self {
            profile: PhantomData,
            wqkv,
            wqkv_gate,
            conv: ShortConv::<Shift>::new(conv_weight, geo.conv_kernel),
            step: GdnStepArm::configured(dims.d_k, dims.h_v, &ssm_dt, &ssm_a),
            ssm_a,
            ssm_dt,
            ssm_alpha,
            ssm_beta,
            ssm_norm,
            ssm_out,
            stem_cat: None,
            gate_stems_cat: None,
            dims,
            conv_channels: geo.conv_channels(),
            value_dim: geo.value_dim(),
            key_dim: geo.key_dim(),
            norm_eps: geo.norm_eps,
            gate: geo.gate,
        }
    }

    /// The gated norm's output gate.
    pub fn out_gate(&self) -> OutGate {
        self.gate
    }

    /// This layer with its recurrence step in `form`, where the construction
    /// tier's choice ([`GdnStepArm::configured`]) is not the one wanted: an
    /// A/B arm, or a fixture pinned to one form.
    pub fn with_step_form(mut self, form: GdnStepForm) -> Self {
        let (d, h_v) = (self.dims.d_k, self.dims.h_v);
        self.step = GdnStepArm::new(form, d, h_v, &self.ssm_dt, &self.ssm_a);
        self
    }

    /// The form of this layer's recurrence step.
    pub fn step_form(&self) -> GdnStepForm {
        self.step.form()
    }

    /// One decode step: `x [1, 1, hidden]` in, `[1, 1, hidden]` out, threading
    /// this layer's `S` (state channel `l`) and conv window (conv channel `l`).
    ///
    /// Composes, per `build_layer_attn_linear`: qkvz → gate(α,g,β) → conv →
    /// silu → split q/k/v → l2norm(q,k) → GQA tile → decode_step → write S →
    /// One decode step: `x [1, 1, hidden]` → `[1, 1, hidden]` against the
    /// layer's `S` and conv channels — [`Self::decode_rows`] over one row.
    pub fn decode(
        &self,
        x: &Tensor,
        l: usize,
        s_state: &mut RecurrentState,
        conv_state: &mut RecurrentState,
    ) -> Result<Tensor> {
        self.decode_rows(x, l, &mut [(s_state, conv_state)])
    }

    /// One decode step for `B` independent sequences at once: `x [B, 1,
    /// hidden]`, row `r` against `states[r]` (its `S` and conv channels), →
    /// `[B, 1, hidden]`. The weight-bearing ops (the stems, the out-proj) run
    /// once over all rows (`m = B`); the state ops run per row against that
    /// row's own slabs through the kernels a row of one sees, so a row
    /// computes exactly what `decode` of that row computes (`docs/
    /// batched-decode-design.md` §10.2). A single row IS `decode`.
    pub fn decode_rows(
        &self,
        x: &Tensor,
        l: usize,
        states: &mut [(&mut RecurrentState, &mut RecurrentState)],
    ) -> Result<Tensor> {
        let (b, seq, _) = x.dims3()?;
        if seq != 1 || b != states.len() || b == 0 {
            lloom_tensor::bail!(
                "deltanet decode_rows: x {:?} for {} rows (one token per row)",
                x.shape(),
                states.len()
            )
        }
        if let Some(out) = self.decode_rows_fused(x, l, states)? {
            return Ok(out);
        }
        let mut outs = Vec::with_capacity(b);
        for (r, (s_state, conv_state)) in states.iter_mut().enumerate() {
            outs.push(self.decode_composed(&x.narrow(0, r, 1)?, l, s_state, conv_state)?);
        }
        if b == 1 {
            Ok(outs.pop().expect("one row"))
        } else {
            Tensor::cat(&outs, 0)
        }
    }

    /// The rows as one launch per kernel over their shared pools
    /// (`kernels::GdnPrologueRows`, the step block's rows,
    /// `kernels::GdnGatedNormRows`): the row tables `slots` / `phases`
    /// uploaded per layer, the pools' tensors bound, each row read at its
    /// phase and written at the next, then advanced. `None` unless every row's `S` and window channels are
    /// pooled — in one pool each — and written (a fresh row takes the
    /// per-row path once, which zeroes and writes its slots).
    #[allow(clippy::too_many_arguments)]
    fn decode_rows_pooled(
        &self,
        qkv: &Tensor,
        z: &Tensor,
        alpha: &Tensor,
        beta: &Tensor,
        l: usize,
        states: &mut [(&mut RecurrentState, &mut RecurrentState)],
        gamma: &Tensor,
    ) -> Result<Option<Tensor>> {
        use super::kernels::{GdnGatedNormRows, GdnPrologueRows};
        let dims = self.dims;
        let (d, h_k, h_v) = (dims.d_k, dims.h_k, dims.h_v);
        let b = states.len();
        let dev = qkv.device();
        let mut s_slots = Vec::with_capacity(b);
        let mut s_phases = Vec::with_capacity(b);
        let mut w_slots = Vec::with_capacity(b);
        let mut w_phases = Vec::with_capacity(b);
        let mut s_pool: Option<crate::nn::StatePool> = None;
        let mut w_pool: Option<crate::nn::StatePool> = None;
        for (s_state, conv_state) in states.iter() {
            let (Some((sp, ss, sph, sw)), Some((wp, ws, wph, ww))) =
                (s_state.pool_slot(l), conv_state.pool_slot(l))
            else {
                return Ok(None);
            };
            if !sw || !ww {
                return Ok(None);
            }
            match &s_pool {
                Some(p) if !p.same(sp) => return Ok(None),
                Some(_) => {}
                None => s_pool = Some(sp.clone()),
            }
            match &w_pool {
                Some(p) if !p.same(wp) => return Ok(None),
                Some(_) => {}
                None => w_pool = Some(wp.clone()),
            }
            s_slots.push(ss as u32);
            s_phases.push(sph as u32);
            w_slots.push(ws as u32);
            w_phases.push(wph as u32);
        }
        let (s_pool, w_pool) = (s_pool.expect("rows"), w_pool.expect("rows"));
        let up = |v: &[u32]| Tensor::from_vec(v.to_vec(), v.len(), dev);
        let heads = qkv.apply_op1_no_bwd(&GdnPrologueRows {
            rows: b,
            d,
            h_k,
            h_v,
            eps: self.norm_eps as f32,
            q_scale: 1.0 / (d as f32).sqrt(),
            conv: self.conv.clone(),
            win: w_pool.pool_layer(l)?,
            slots: up(&w_slots)?,
            phases: up(&w_phases)?,
        })?; // [B, 3, h_v, d]
        let o = self.step.rows(
            &heads,
            alpha,
            beta,
            s_pool.pool_layer(l)?,
            up(&s_slots)?,
            up(&s_phases)?,
        )?; // [B, h_v, d]
        for (s_state, conv_state) in states.iter_mut() {
            conv_state.flip_phase(l)?;
            s_state.flip_phase(l)?;
        }
        let gated = o.apply_op3_no_bwd(
            gamma,
            z,
            &GdnGatedNormRows {
                rows: b,
                d,
                h_v,
                eps: self.norm_eps as f32,
                gate: self.gate,
            },
        )?; // [B, value_dim]
        Ok(Some(gated))
    }

    /// `T` POSITIONS of one sequence at once (a speculative verify window):
    /// `x [T, 1, hidden]` → `[T, 1, hidden]`, the state advanced `T` steps
    /// with every intermediate state kept in the channels' phase rings, so
    /// the caller can accept any prefix by phase (`RecurrentState::
    /// accept_window`). The stems and the out-proj run once over the `T`
    /// rows; the recurrence runs the positions in sequence — as one launch
    /// per kernel on Metal (`kernels::GdnPrologueSeq`, the step block's seq,
    /// the arithmetic of the per-position kernels step for step), per position
    /// through the composed path elsewhere. Requires pooled channels with
    /// `T + 1` phases.
    pub fn decode_seq(
        &self,
        x: &Tensor,
        l: usize,
        s_state: &mut RecurrentState,
        conv_state: &mut RecurrentState,
    ) -> Result<Tensor> {
        let (t, seq, _) = x.dims3()?;
        if seq != 1 || t == 0 {
            lloom_tensor::bail!(
                "deltanet decode_seq: x {:?} (positions on dim 0)",
                x.shape()
            )
        }
        if !s_state.ring_holds(t) || !conv_state.ring_holds(t) {
            lloom_tensor::bail!(
                "deltanet decode_seq: {t} positions need pooled channels with {} phases",
                t + 1
            )
        }
        if let Some(out) = self.decode_seq_fused(x, l, s_state, conv_state)? {
            return Ok(out);
        }
        let mut outs = Vec::with_capacity(t);
        for p in 0..t {
            outs.push(self.decode_rows(
                &x.narrow(0, p, 1)?,
                l,
                &mut [(&mut *s_state, &mut *conv_state)],
            )?);
        }
        if t == 1 {
            Ok(outs.pop().expect("one position"))
        } else {
            Tensor::cat(&outs, 0)
        }
    }

    fn decode_seq_fused(
        &self,
        x: &Tensor,
        l: usize,
        s_state: &mut RecurrentState,
        conv_state: &mut RecurrentState,
    ) -> Result<Option<Tensor>> {
        use super::kernels::{fused_off, GdnGatedNormRows, GdnPrologueSeq};
        let dims = self.dims;
        let (d, h_k, h_v) = (dims.d_k, dims.h_k, dims.h_v);
        let t = x.dim(0)?;
        if fused_off()
            || !x.device().is_metal()
            || x.dtype() != lloom_tensor::DType::F32
            || self.conv.weight().dtype() != lloom_tensor::DType::F32
            || dims.d_v != d
        {
            return Ok(None);
        }
        let f32 = lloom_tensor::DType::F32;
        let dev = x.device();
        // A fresh row: zero its slots through the per-row path's first step
        // (which allocates and zeroes), then the rest in sequence.
        let (Some((sp, s_slot, s_phase, s_written)), Some((wp, w_slot, w_phase, w_written))) =
            (s_state.pool_slot(l), conv_state.pool_slot(l))
        else {
            return Ok(None);
        };
        if !s_written || !w_written {
            if s_state.get(l).is_none() {
                s_state.write(l, Tensor::zeros((d, d, h_v), f32, dev)?)?;
            }
            if conv_state.get(l).is_none() {
                conv_state.write(
                    l,
                    Tensor::zeros((self.conv_channels, self.conv.k()), f32, dev)?,
                )?;
            }
            return self.decode_seq_fused(x, l, s_state, conv_state);
        }
        let s_layer = sp.pool_layer(l)?;
        let w_layer = wp.pool_layer(l)?;
        let (qkv, z, alpha, beta) = self.decode_stems(x, h_v)?; // [T, ·]
        let heads = qkv.apply_op1_no_bwd(&GdnPrologueSeq {
            steps: t,
            d,
            h_k,
            h_v,
            eps: self.norm_eps as f32,
            q_scale: 1.0 / (d as f32).sqrt(),
            conv: self.conv.clone(),
            win: w_layer,
            slot: w_slot,
            phase: w_phase,
        })?; // [T, 3, h_v, d]
        let o = self
            .step
            .seq(&heads, &alpha, &beta, s_layer, s_slot, s_phase)?; // [T, h_v, d]
        conv_state.advance_phase(l, t)?;
        s_state.advance_phase(l, t)?;
        let gated = o.apply_op3_no_bwd(
            &self.ssm_norm.to_dtype(f32)?,
            &z,
            &GdnGatedNormRows {
                rows: t,
                d,
                h_v,
                eps: self.norm_eps as f32,
                gate: self.gate,
            },
        )?; // [T, value_dim]
        Ok(Some(self.ssm_out.forward(&gated.reshape((
            t,
            1,
            self.value_dim,
        ))?)?))
    }

    /// The composed decode of one row: the CPU path and the reference.
    fn decode_composed(
        &self,
        x: &Tensor,
        l: usize,
        s_state: &mut RecurrentState,
        conv_state: &mut RecurrentState,
    ) -> Result<Tensor> {
        let dims = self.dims;
        let (d_k, h_k, h_v) = (dims.d_k, dims.h_k, dims.h_v);

        // qkvz: fused QKV + the Z output gate.
        let qkv = self.wqkv.forward(x)?; // [1,1,conv_channels]
        let z = self.wqkv_gate.forward(x)?; // [1,1,value_dim]

        // gate: β = σ(ssm_beta·x); g = ssm_a·softplus(ssm_alpha·x + dt_bias).
        let (g, beta) = gate_and_beta(
            x,
            &self.ssm_beta,
            &self.ssm_alpha,
            &self.ssm_dt,
            &self.ssm_a,
        )?;
        let g = g.reshape(h_v)?; // [h_v]
        let beta = beta.reshape(h_v)?; // [h_v]

        // conv over the concat-QKV channels (k=4), then SiLU. Decode = one
        // shift-and-fold against the layer's conv window.
        let qkv_flat = qkv.reshape(self.conv_channels)?; // [conv_channels]
        let (conv_out, new_win) = conv_decode_step(
            &qkv_flat,
            self.conv.weight(),
            conv_state.get(l),
            self.conv_channels,
            self.conv.k(),
        )?;
        conv_state.write(l, new_win)?;
        let conv_out = ops::silu(&conv_out)?; // [conv_channels]

        // split into Q, K [key_dim] and V [value_dim].
        let q = conv_out.narrow(0, 0, self.key_dim)?; // [key_dim]
        let k = conv_out.narrow(0, self.key_dim, self.key_dim)?;
        let v = conv_out.narrow(0, 2 * self.key_dim, self.value_dim)?;
        // reshape to heads: q,k [h_k, d_k], v [h_v, d_v].
        let q = q.reshape((h_k, d_k))?;
        let k = k.reshape((h_k, d_k))?;
        let v = v.reshape((h_v, dims.d_v))?;

        // l2norm q,k (L2, not RMS), then GQA tile h_k → h_v (reference default).
        let q = l2_norm_last(&q, self.norm_eps)?;
        let k = l2_norm_last(&k, self.norm_eps)?;
        let q = gqa_expand::<false>(&q, h_k, h_v)?; // [h_v, d_k]
        let k = gqa_expand::<false>(&k, h_k, h_v)?;
        let q = q_scale(&q, d_k)?;

        // the recurrence; state channel `l` (head-last layout = reference default).
        let (o, new_s) = decode_step::<false>(dims, &q, &k, &v, &g, &beta, s_state.get(l))?;
        s_state.write(l, new_s)?;

        // gated norm: rms_norm(o, ssm_norm) · gate(z). o is [h_v, d_v].
        let o_flat = o.reshape((1, 1, self.value_dim))?; // back to [1,1,value_dim]
        let normed = self.gated_rms_norm(&o_flat)?;
        let z_gate = self.gate.apply(&z.to_dtype(normed.dtype())?)?;
        let gated = (normed * z_gate)?;

        // out-proj → [1,1,hidden].
        self.ssm_out.forward(&gated)
    }

    /// [`Self::decode_rows`] as launches (Metal, f32, `d_k == d_v`): the stems
    /// once over all rows, then per row `kernels::GdnPrologue`, the step
    /// block's step and `kernels::GdnGatedNorm` against the row's slabs (the
    /// next window and `S` written out of place into the channels' spares and
    /// swapped in — no copy, and a speculative window's base stays intact),
    /// the gated rows gathered, the out-proj once. `None` where the composed path applies.
    fn decode_rows_fused(
        &self,
        x: &Tensor,
        l: usize,
        states: &mut [(&mut RecurrentState, &mut RecurrentState)],
    ) -> Result<Option<Tensor>> {
        use super::kernels::{fused_off, GdnGatedNorm, GdnPrologue};
        let dims = self.dims;
        let (d, h_k, h_v) = (dims.d_k, dims.h_k, dims.h_v);
        let b = x.dim(0)?;
        if fused_off()
            || !x.device().is_metal()
            || x.dtype() != lloom_tensor::DType::F32
            || self.conv.weight().dtype() != lloom_tensor::DType::F32
            || dims.d_v != d
        {
            return Ok(None);
        }
        let dev = x.device();
        let f32 = lloom_tensor::DType::F32;
        let gamma = self.ssm_norm.to_dtype(f32)?;
        // The projections once over all rows: `[B, C] | [B, value_dim] | [B, h_v] | [B, h_v]`.
        let (qkv, z, alpha, beta) = self.decode_stems(x, h_v)?;
        // The batched arm: every row a slot of ONE state pool (and written
        // there) — the three kernels run once over the rows with the slot
        // and phase tables, and every row flips its phase.
        if b > 1 {
            if let Some(gated) =
                self.decode_rows_pooled(&qkv, &z, &alpha, &beta, l, states, &gamma)?
            {
                let out = self
                    .ssm_out
                    .forward(&gated.reshape((b, 1, self.value_dim))?)?;
                return Ok(Some(out));
            }
        }
        let mut gated_rows = Vec::with_capacity(b);
        for (r, (s_state, conv_state)) in states.iter_mut().enumerate() {
            // The persistent buffers: allocate the zero state once (the first
            // write takes an own-storage copy) so the kernels always have a
            // slab to read and a spare to write.
            if conv_state.get(l).is_none() {
                conv_state.write(
                    l,
                    Tensor::zeros((self.conv_channels, self.conv.k()), f32, dev)?,
                )?;
            }
            if s_state.get(l).is_none() {
                s_state.write(l, Tensor::zeros((d, d, h_v), f32, dev)?)?;
            }
            let win = conv_state.get(l).expect("written above").clone();
            let slab = s_state.get(l).expect("written above").clone();
            if win.dtype() != f32
                || slab.dtype() != f32
                || !win.is_contiguous()
                || !slab.is_contiguous()
            {
                return Ok(None);
            }
            let win_out = conv_state.take_spare(l)?;
            let slab_out = s_state.take_spare(l)?;
            let row = |t: &Tensor| -> Result<Tensor> { t.narrow(0, r, 1)?.flatten_all() };
            let heads = row(&qkv)?.apply_op1_no_bwd(&GdnPrologue {
                d,
                h_k,
                h_v,
                eps: self.norm_eps as f32,
                q_scale: 1.0 / (d as f32).sqrt(),
                conv: self.conv.clone(),
                win: win.clone(),
                win_out: win_out.clone(),
            })?; // [3, h_v, d]
            let o = self
                .step
                .step(&heads, &row(&alpha)?, &row(&beta)?, &slab, &slab_out)?; // [h_v, d]
            conv_state.swap_in(l, win_out)?;
            s_state.swap_in(l, slab_out)?;
            let gated = o.apply_op3_no_bwd(
                &gamma,
                &row(&z)?,
                &GdnGatedNorm {
                    d,
                    h_v,
                    eps: self.norm_eps as f32,
                    gate: self.gate,
                },
            )?; // [value_dim]
            gated_rows.push(gated.reshape((1, self.value_dim))?);
        }
        let gated = if b == 1 {
            gated_rows.pop().expect("one row")
        } else {
            Tensor::cat(&gated_rows, 0)?
        }; // [B, value_dim]
        let out = self
            .ssm_out
            .forward(&gated.reshape((b, 1, self.value_dim))?)?;
        Ok(Some(out))
    }

    /// A whole chunk `x [1, seq, hidden]` → `[1, seq, hidden]` (Kernel B), the
    /// same composition as [`Self::decode`] with the conv over the chunk and
    /// the chunkwise recurrence; leaves the state channels where `seq` decode
    /// steps would (position by position, to fp32 reorder).
    pub fn prefill(
        &self,
        x: &Tensor,
        l: usize,
        s_state: &mut RecurrentState,
        conv_state: &mut RecurrentState,
    ) -> Result<Tensor> {
        if let Some(out) = self.prefill_fused(x, l, s_state, conv_state)? {
            return Ok(out);
        }
        let dims = self.dims;
        let (d_k, h_k, h_v) = (dims.d_k, dims.h_k, dims.h_v);
        let (b, seq, _) = x.dims3()?;
        if b != 1 {
            lloom_tensor::bail!("deltanet prefill takes one sequence, got {:?}", x.shape())
        }
        let qkv = self.wqkv.forward(x)?; // [1, seq, conv_channels]
        let z = self.wqkv_gate.forward(x)?; // [1, seq, value_dim]
        let (g, beta) = gate_and_beta(
            x,
            &self.ssm_beta,
            &self.ssm_alpha,
            &self.ssm_dt,
            &self.ssm_a,
        )?;
        let g = g.reshape((seq, h_v))?;
        let beta = beta.reshape((seq, h_v))?;

        let qkv = qkv.reshape((seq, self.conv_channels))?;
        let (conv_out, new_win) = conv_prefill(
            &qkv,
            self.conv.weight(),
            conv_state.get(l),
            self.conv_channels,
            self.conv.k(),
        )?;
        conv_state.write(l, new_win)?;
        let conv_out = ops::silu(&conv_out)?; // [seq, conv_channels]

        let q = conv_out
            .narrow(1, 0, self.key_dim)?
            .reshape((seq, h_k, d_k))?;
        let k = conv_out
            .narrow(1, self.key_dim, self.key_dim)?
            .reshape((seq, h_k, d_k))?;
        let v = conv_out
            .narrow(1, 2 * self.key_dim, self.value_dim)?
            .reshape((seq, h_v, dims.d_v))?;
        let q = l2_norm_last(&q, self.norm_eps)?;
        let k = l2_norm_last(&k, self.norm_eps)?;
        let q = gqa_expand_seq::<false>(&q, h_k, h_v)?;
        let k = gqa_expand_seq::<false>(&k, h_k, h_v)?;
        let q = q_scale(&q, d_k)?;

        let (o, new_s) = chunk_forward(dims, &q, &k, &v, &g, &beta, s_state.get(l), CHUNK_SIZE)?;
        s_state.write(l, new_s)?;

        let o_flat = o.reshape((1, seq, self.value_dim))?;
        let normed = self.gated_rms_norm(&o_flat)?;
        let z_gate = self.gate.apply(&z.to_dtype(normed.dtype())?)?;
        let gated = (normed * z_gate)?;
        self.ssm_out.forward(&gated)
    }

    /// GatedRMSNorm over the value-head dim with the `ssm_norm` weight. `o` is
    /// `[1, 1, value_dim]`; normalize per value-head (over d_v) then scale.
    /// [`Self::prefill`] as launches (Metal, f32, `d_k == d_v`): the four
    /// projections; `kernels::GdnGate` (g, β chunked); `kernels::GdnPrefillPrologue`
    /// (the conv against the window, SiLU, the split, the L2 norms, the GQA
    /// tile, the scale — straight into the chunked head-first layout, no
    /// transposes, no pads); `kernels::GdnConvWindow` (the window after the
    /// chunk); `kernels::GdnChunkDecay` (the cumsum and both decay matrices);
    /// `kernels::GdnChunkScale` (the five β / decay scalings); the WY matmuls
    /// on transposed VIEWS (the gemm takes the stride); the fused inverse;
    /// the chunk loop as `chunk_forward`'s; `kernels::GdnChunkEpilogue` (the
    /// gated RMS norm read straight from the head-first output). `None` where
    /// the composed path applies — which stays the reference.
    fn prefill_fused(
        &self,
        x: &Tensor,
        l: usize,
        s_state: &mut RecurrentState,
        conv_state: &mut RecurrentState,
    ) -> Result<Option<Tensor>> {
        use super::deltanet::{unit_lower_inverse, CHUNK_SIZE};
        use super::kernels::{
            fused_off, GdnChunkDecay, GdnChunkEpilogue, GdnChunkScale, GdnConvWindow, GdnGate,
            GdnPrefillPrologue,
        };
        let dims = self.dims;
        let (d, h_k, h_v) = (dims.d_k, dims.h_k, dims.h_v);
        let (b, seq, _) = x.dims3()?;
        if fused_off()
            || !x.device().is_metal()
            || x.dtype() != lloom_tensor::DType::F32
            || self.conv.weight().dtype() != lloom_tensor::DType::F32
            || dims.d_v != d
            || b != 1
            || seq == 0
        {
            return Ok(None);
        }
        let dev = x.device();
        let f32 = lloom_tensor::DType::F32;
        let cs = CHUNK_SIZE;
        let nc = seq.div_ceil(cs);
        let padded = nc * cs;
        let n = h_v * nc * cs;
        let kk = self.conv.k();
        let win = match conv_state.get(l) {
            Some(w) => w.to_dtype(f32)?.contiguous()?,
            None => Tensor::zeros((self.conv_channels, kk), f32, dev)?,
        };

        // The projections.
        let qkv = self
            .wqkv
            .forward(x)?
            .to_dtype(f32)?
            .reshape((seq, self.conv_channels))?; // [seq, C]
        let z = self
            .wqkv_gate
            .forward(x)?
            .to_dtype(f32)?
            .reshape((seq, self.value_dim))?; // [seq, h_v·d]
        let alpha = self
            .ssm_alpha
            .forward(x)?
            .to_dtype(f32)?
            .reshape((seq, h_v))?;
        let beta = self
            .ssm_beta
            .forward(x)?
            .to_dtype(f32)?
            .reshape((seq, h_v))?;

        // The chunked operands.
        let gb = alpha.apply_op2_no_bwd(
            &beta,
            &GdnGate {
                seq,
                h_v,
                nc,
                cs,
                dt_bias: self.ssm_dt.to_dtype(f32)?,
                ssm_a: self.ssm_a.to_dtype(f32)?,
            },
        )?; // [2, h_v, nc, cs]
        let g_ch = gb.narrow(0, 0, 1)?.reshape((h_v, nc, cs))?;
        let pack = qkv.apply_op3_no_bwd(
            &win,
            self.conv.weight(),
            &GdnPrefillPrologue {
                seq,
                d,
                h_k,
                h_v,
                kk,
                cs,
                nc,
                eps: self.norm_eps as f32,
                q_scale: 1.0 / (d as f32).sqrt(),
            },
        )?; // [3, h_v, nc, cs, d]
        let new_win = win.apply_op1_no_bwd(&GdnConvWindow {
            seq,
            channels: self.conv_channels,
            kk,
            qkv: qkv.clone(),
        })?;
        conv_state.write(l, new_win)?;
        let decay_op = GdnChunkDecay { h_v, nc, cs };
        let small = g_ch.apply_op1_no_bwd(&decay_op)?; // flat
        let (_, _, o_last, o_decay, o_strict, _) = decay_op.offsets();
        let g_last_exp = small.narrow(0, o_last, h_v * nc)?.reshape((h_v, nc))?;
        let decay = small
            .narrow(0, o_decay, h_v * nc * cs * cs)?
            .reshape((h_v, nc, cs, cs))?;
        let decay_strict = small
            .narrow(0, o_strict, h_v * nc * cs * cs)?
            .reshape((h_v, nc, cs, cs))?;
        let scaled = pack.apply_op3_no_bwd(&gb, &small, &GdnChunkScale { n, d })?; // [5, n, d]
        let part =
            |i: usize| -> Result<Tensor> { scaled.narrow(0, i, 1)?.reshape((h_v, nc, cs, d)) };
        let (k_beta, w_in, q_g, kg, v_beta) = (part(0)?, part(1)?, part(2)?, part(3)?, part(4)?);
        let q = pack.narrow(0, 0, 1)?.reshape((h_v, nc, cs, d))?;
        let k = pack.narrow(0, 1, 1)?.reshape((h_v, nc, cs, d))?;
        let k_t = k.transpose(2, 3)?; // a view; the gemm takes the stride

        // The WY form (as `chunk_forward`).
        let a = k_beta.matmul(&k_t)?.mul(&decay_strict)?; // [h_v, nc, cs, cs]
        let t = unit_lower_inverse(&a)?;
        let u = t.matmul(&v_beta)?; // [h_v, nc, cs, d]
        let w = t.matmul(&w_in)?; // [h_v, nc, cs, d]
        let kq = q.matmul(&k_t)?.mul(&decay)?; // [h_v, nc, cs, cs]
        let kg_t = kg.transpose(2, 3)?; // [h_v, nc, d, cs] view

        let mut s = match s_state.get(l) {
            Some(s) => s.to_dtype(f32)?.permute((2, 0, 1))?.contiguous()?, // [h_v, d, d]
            None => Tensor::zeros((h_v, d, d), f32, dev)?,
        };
        // The inter-chunk recurrence as one launch (`kernels::GdnChunkScan`);
        // the host loop below is its A/B arm and the CPU's.
        let scanned =
            if super::kernels::chunk_scan_on() && cs.is_multiple_of(8) && d.is_multiple_of(8) {
                let flat = w.contiguous()?.apply_op3_no_bwd(
                    &u.contiguous()?,
                    &q_g.contiguous()?,
                    &super::kernels::GdnChunkScan {
                        h_v,
                        nc,
                        cs,
                        d,
                        kq: kq.contiguous()?,
                        kg: kg.contiguous()?,
                        g_last: g_last_exp.contiguous()?,
                        s_in: s.clone(),
                    },
                )?;
                let n_o = h_v * nc * cs * d;
                let o = flat.narrow(0, 0, n_o)?.reshape((h_v, padded, d))?;
                s = flat.narrow(0, n_o, h_v * d * d)?.reshape((h_v, d, d))?;
                Some(o)
            } else {
                None
            };
        let mut outs = Vec::with_capacity(nc);
        for c in 0..nc {
            if scanned.is_some() {
                break;
            }
            let w_c = w.narrow(1, c, 1)?.squeeze(1)?; // [h_v, cs, d]
            let u_c = u.narrow(1, c, 1)?.squeeze(1)?;
            let v_new = (u_c - w_c.matmul(&s)?)?; // [h_v, cs, d]
            let q_g_c = q_g.narrow(1, c, 1)?.squeeze(1)?;
            let kq_c = kq.narrow(1, c, 1)?.squeeze(1)?;
            let o_c = (q_g_c.matmul(&s)? + kq_c.matmul(&v_new)?)?; // [h_v, cs, d]
            outs.push(o_c);
            let kg_c = kg_t.narrow(1, c, 1)?.squeeze(1)?; // [h_v, d, cs]
            let gl = g_last_exp.narrow(1, c, 1)?.reshape((h_v, 1, 1))?;
            s = (s.broadcast_mul(&gl)? + kg_c.matmul(&v_new)?)?;
        }
        let o = match scanned {
            Some(o) => o,
            None if nc == 1 => outs.pop().expect("one chunk"),
            None => Tensor::cat(&outs, 1)?,
        }; // [h_v, padded, d]
        let out = o.apply_op2_no_bwd(
            &z,
            &GdnChunkEpilogue {
                seq,
                padded,
                d,
                h_v,
                eps: self.norm_eps as f32,
                gamma: self.ssm_norm.to_dtype(f32)?,
                gate: self.gate,
            },
        )?; // [seq, h_v·d]
        s_state.write(l, s.permute((1, 2, 0))?.contiguous()?)?;
        Ok(Some(self.ssm_out.forward(&out.reshape((
            1,
            seq,
            self.value_dim,
        ))?)?))
    }

    fn gated_rms_norm(&self, o: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = o.dims3()?;
        let (h_v, d_v) = (self.dims.h_v, self.dims.d_v);
        let o = o.reshape((b, seq, h_v, d_v))?;
        let normed = ops::rms_norm(
            &o.to_dtype(lloom_tensor::DType::F32)?,
            &self.ssm_norm.to_dtype(lloom_tensor::DType::F32)?,
            self.norm_eps as f32,
        )?;
        normed.reshape((b, seq, h_v * d_v))
    }

    /// The head dim, for the caller (used nowhere yet; keeps the API discoverable).
    pub fn value_dim(&self) -> usize {
        self.value_dim
    }

    /// One layer's state on its own phase rings (`StatePool::with_phases`,
    /// one slot, `phases` phases): what a verify window of `phases − 1`
    /// positions runs through as one pass (`Self::decode_seq`) and accepts by
    /// phase (`RecurrentState::accept_window`) — the Qwen4-exp `spec_pools`
    /// shape, built per layer for a `Decoder` at the `Ring` profile.
    pub fn pooled_state(&self, dev: &lloom_tensor::Device, phases: usize) -> Result<GdnState> {
        let f32 = lloom_tensor::DType::F32;
        let s = crate::nn::StatePool::with_phases(
            1,
            phases,
            1,
            &[self.dims.d_k, self.dims.d_v, self.dims.h_v],
            f32,
            dev,
        );
        let conv = crate::nn::StatePool::with_phases(
            1,
            phases,
            1,
            &[self.conv_channels, self.conv.k()],
            f32,
            dev,
        );
        Ok(GdnState {
            layer: 0,
            s: RecurrentState::in_pool(&s)?,
            conv: RecurrentState::in_pool(&conv)?,
        })
    }

    /// Attach the stacked input stems (`[conv_channels + value_dim + 2·h_v,
    /// hidden]` f16, the parts' rows in that order) for the one-gemv decode.
    pub fn with_stem_cat(mut self, cat: Tensor) -> Self {
        self.stem_cat = Some(QMatMul::TensorF16(cat));
        self
    }

    /// Attach the stacked gate stems (`[2·h_v, hidden]` f16, `α | β` rows).
    pub fn with_gate_stems_cat(mut self, cat: Tensor) -> Self {
        self.gate_stems_cat = Some(QMatMul::TensorF16(cat));
        self
    }

    /// The decode step's four projections of `x [1, 1, hidden]` as f32 rows
    /// — one gemv over the stacked stems when they are stacked, four else
    /// (three when only the gate stems are).
    fn decode_stems(&self, x: &Tensor, h_v: usize) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let f32 = lloom_tensor::DType::F32;
        let b = x.dim(0)?;
        let rows = |t: Tensor, n: usize| -> Result<Tensor> { t.to_dtype(f32)?.reshape((b, n)) };
        if let Some(cat) = &self.stem_cat {
            let (c, v) = (self.conv_channels, self.value_dim);
            let all = rows(cat.forward(x)?, c + v + 2 * h_v)?;
            return Ok((
                all.narrow(1, 0, c)?.contiguous()?,
                all.narrow(1, c, v)?.contiguous()?,
                all.narrow(1, c + v, h_v)?.contiguous()?,
                all.narrow(1, c + v + h_v, h_v)?.contiguous()?,
            ));
        }
        if let Some(ab) = &self.gate_stems_cat {
            let ab = rows(ab.forward(x)?, 2 * h_v)?;
            return Ok((
                rows(self.wqkv.forward(x)?, self.conv_channels)?,
                rows(self.wqkv_gate.forward(x)?, self.value_dim)?,
                ab.narrow(1, 0, h_v)?.contiguous()?,
                ab.narrow(1, h_v, h_v)?.contiguous()?,
            ));
        }
        Ok((
            rows(self.wqkv.forward(x)?, self.conv_channels)?,
            rows(self.wqkv_gate.forward(x)?, self.value_dim)?,
            rows(self.ssm_alpha.forward(x)?, h_v)?,
            rows(self.ssm_beta.forward(x)?, h_v)?,
        ))
    }

    /// Silence the unused-import lint for `D` until prefill (Kernel B) uses it.
    #[allow(dead_code)]
    fn _uses_d() -> D {
        D::Minus1
    }
}

/// A linear-attention layer's decode step ([`GatedDeltaNet::decode_rows`]'s
/// launches at one row): the input stems as the loader stacked them, the
/// prologue over the conv window, the recurrence step over `S`, the gated
/// norm, the output projection. The window, `S` and the two kernels that
/// read and rewrite them are layer `l`'s recurrence. `In` is the bracket's
/// output and the layer.
#[cfg(not(target_arch = "wasm32"))]
/// A layer's recurrent state as the Gated DeltaNet's block state: its
/// index in the family's state (the pooled channels are addressed by
/// layer), its `S` and its conv window.
#[derive(Debug, Clone)]
pub struct GdnState {
    pub layer: usize,
    pub s: RecurrentState,
    pub conv: RecurrentState,
}

/// The Gated DeltaNet as a level-0 block (`nn::blocks::Mixer`): one
/// position is a decode step, more are a chunk (the chunkwise recurrence,
/// the state left where `seq` steps would leave it).
impl<P: Profile> crate::nn::blocks::Mixer for GatedDeltaNet<P> {
    type State = GdnState;
    fn execute(&self, x: &Tensor, st: &mut GdnState) -> Result<Tensor> {
        if x.dim(1)? == 1 {
            self.decode(x, st.layer, &mut st.s, &mut st.conv)
        } else {
            self.prefill(x, st.layer, &mut st.s, &mut st.conv)
        }
    }
}

impl<P: Profile> crate::optimizer::Visit for GatedDeltaNet<P> {
    type In = (crate::optimizer::NodeId, usize);
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, (x, l): Self::In) -> Self::Out {
        use crate::optimizer::{Region, Shape};
        let f32 = lloom_tensor::DType::F32;
        let (d, h_v) = (self.dims.d_k, self.dims.h_v);
        let (c, v) = (self.conv_channels, self.value_dim);
        let narrow =
            |w: &mut crate::optimizer::Walker, t, n: usize| w.view("narrow", t, Shape::of(&[1, n]));
        let (qkv, z, alpha, beta) = if let Some(cat) = &self.stem_cat {
            let all = w.project_qmat(
                "attn_qkv.weight+attn_gate.weight+ssm_alpha.weight+ssm_beta.weight",
                cat,
                x,
            );
            (
                narrow(w, all, c),
                narrow(w, all, v),
                narrow(w, all, h_v),
                narrow(w, all, h_v),
            )
        } else {
            let qkv = w.project_qmat("attn_qkv.weight", &self.wqkv, x);
            let z = w.project_qmat("attn_gate.weight", &self.wqkv_gate, x);
            let (qkv, z) = (
                w.view("reshape", qkv, Shape::of(&[1, c])),
                w.view("reshape", z, Shape::of(&[1, v])),
            );
            match &self.gate_stems_cat {
                Some(ab) => {
                    let ab = w.project_qmat("ssm_alpha.weight+ssm_beta.weight", ab, x);
                    (qkv, z, narrow(w, ab, h_v), narrow(w, ab, h_v))
                }
                None => {
                    let a = w.project_qmat("ssm_alpha.weight", &self.ssm_alpha, x);
                    let b = w.project_qmat("ssm_beta.weight", &self.ssm_beta, x);
                    (
                        qkv,
                        z,
                        w.view("reshape", a, Shape::of(&[1, h_v])),
                        w.view("reshape", b, Shape::of(&[1, h_v])),
                    )
                }
            }
        };
        let taps = self.conv.weight();
        let conv_w = w.weight("ssm_conv1d.weight", Shape::of(taps.dims()), taps.dtype());
        let window = w.input(
            &format!("gdn.conv@{l}"),
            Shape::of(&[c, self.conv.k()]),
            f32,
        );
        let heads = w.opaque(
            "gdn_prologue",
            &[qkv, conv_w, window],
            Shape::of(&[3, h_v, d]),
            f32,
        );
        let s_in = w.input(&format!("gdn.s@{l}"), Shape::of(&[d, d, h_v]), f32);
        let dt = w.weight(
            "ssm_dt.bias",
            Shape::of(self.ssm_dt.dims()),
            self.ssm_dt.dtype(),
        );
        let a = w.weight("ssm_a", Shape::of(self.ssm_a.dims()), self.ssm_a.dtype());
        let o = w.opaque(
            "gdn_step",
            &[heads, alpha, beta, s_in, dt, a],
            Shape::of(&[h_v, d]),
            f32,
        );
        w.mark(&[window, heads, s_in, o], Region::Ssm(l));
        let gamma = w.weight(
            "ssm_norm.weight",
            Shape::of(self.ssm_norm.dims()),
            self.ssm_norm.dtype(),
        );
        let gated = w.opaque("gdn_gated_norm", &[o, gamma, z], Shape::of(&[v]), f32);
        let gated = w.view("reshape", gated, Shape::of(&[1, 1, v]));
        w.project_qmat("ssm_out.weight", &self.ssm_out, gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::CrownExact;
    use lloom_tensor::{DType, Device};

    // A QMatMul over an f32 weight of shape [out, in] (a plain matmul), the shape
    // the BF16/quant HC+SSM weights hold. `QMatMul::forward` computes x·Wᵀ.
    fn lin(out: usize, inp: usize, dev: &Device) -> QMatMul {
        // small deterministic values so the recurrence stays finite.
        let n = out * inp;
        let v: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.017).sin()) * 0.1).collect();
        QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev).unwrap())
    }
    fn vecw(n: usize, dev: &Device, f: impl Fn(usize) -> f32) -> Tensor {
        Tensor::from_vec((0..n).map(f).collect::<Vec<f32>>(), n, dev).unwrap()
    }

    /// A full DeltaNet layer decodes end to end at tiny dims: shapes thread, no
    /// panic, and the `S` state accumulates across steps (the output at step 2
    /// differs from a fresh step 1 on the same token — the recurrence is live).
    #[test]
    fn decode_threads_end_to_end_and_is_stateful() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let d = cfg.hidden_size;
        let conv_ch = cfg.conv_channels();
        let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
        let h_v = cfg.ssm_dt_rank;
        let d_v = cfg.ssm_state_size;

        let blk = GatedDeltaNet::<CrownExact>::new(
            lin(conv_ch, d, &dev),   // wqkv
            lin(value_dim, d, &dev), // wqkv_gate
            vecw(conv_ch * cfg.ssm_conv_kernel, &dev, |i| {
                ((i as f32 * 0.03).cos()) * 0.2
            })
            .reshape((conv_ch, cfg.ssm_conv_kernel))
            .unwrap(), // conv_weight [conv_ch, k]
            vecw(h_v, &dev, |i| -0.5 - 0.01 * i as f32), // ssm_a = -exp(A_log) (negative)
            vecw(h_v, &dev, |_| 0.0), // ssm_dt bias
            lin(h_v, d, &dev),       // ssm_alpha
            lin(h_v, d, &dev),       // ssm_beta
            vecw(d_v, &dev, |_| 1.0), // ssm_norm (unit γ)
            lin(d, value_dim, &dev), // ssm_out
            &cfg,
        );

        let mut s_state = RecurrentState::new(cfg.n_layers);
        let mut conv_state = RecurrentState::new(cfg.n_layers);

        let tok = |seed: f32| {
            Tensor::from_vec(
                (0..d)
                    .map(|i| ((i as f32 * 0.05 + seed).sin()) * 0.3)
                    .collect::<Vec<f32>>(),
                (1, 1, d),
                &dev,
            )
            .unwrap()
        };

        // Step on layer 0 twice with the SAME token; a stateful recurrence gives
        // different outputs (S carries over).
        let x = tok(0.0);
        let o1 = blk.decode(&x, 0, &mut s_state, &mut conv_state).unwrap();
        assert_eq!(o1.dims(), &[1, 1, d]);
        let o1v = o1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(o1v.iter().all(|x| x.is_finite()));

        let o2 = blk.decode(&x, 0, &mut s_state, &mut conv_state).unwrap();
        let o2v = o2.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Different from step 1 → the S state (and conv window) carried over.
        assert!(
            o1v.iter().zip(&o2v).any(|(a, b)| (a - b).abs() > 1e-6),
            "state did not accumulate: step1 == step2"
        );
        // The state channel holds this layer's S now.
        assert!(s_state.get(0).is_some());
        assert!(conv_state.get(0).is_some());
    }

    /// The chunk path equals the per-token path on a whole block (conv window,
    /// gates, recurrence, gated norm, out-proj): same outputs per position and
    /// the same state channels afterwards, from a mid-sequence state, at a
    /// length past one chunk. The block's own composition is the reference —
    /// `chunk_forward`'s parity against Kernel A is pinned in `deltanet`.
    #[test]
    fn prefill_matches_decode_replay() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let d = cfg.hidden_size;
        let conv_ch = cfg.conv_channels();
        let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
        let h_v = cfg.ssm_dt_rank;
        let d_v = cfg.ssm_state_size;
        let blk = GatedDeltaNet::<CrownExact>::new(
            lin(conv_ch, d, &dev),
            lin(value_dim, d, &dev),
            vecw(conv_ch * cfg.ssm_conv_kernel, &dev, |i| {
                ((i as f32 * 0.03).cos()) * 0.2
            })
            .reshape((conv_ch, cfg.ssm_conv_kernel))
            .unwrap(),
            vecw(h_v, &dev, |i| -0.5 - 0.01 * i as f32),
            vecw(h_v, &dev, |i| 0.1 * i as f32),
            lin(h_v, d, &dev),
            lin(h_v, d, &dev),
            vecw(d_v, &dev, |i| 1.0 + 0.05 * i as f32),
            lin(d, value_dim, &dev),
            &cfg,
        );
        let tok = |seed: f32| -> Vec<f32> {
            (0..d)
                .map(|i| ((i as f32 * 0.05 + seed).sin()) * 0.6)
                .collect()
        };
        // Warm both paths with the same 3 decode tokens, then compare a
        // 70-position chunk (more than CHUNK_SIZE = 64) against 70 decode steps.
        let (mut s_a, mut c_a) = (RecurrentState::new(1), RecurrentState::new(1));
        let (mut s_b, mut c_b) = (RecurrentState::new(1), RecurrentState::new(1));
        for t in 0..3 {
            let x = Tensor::from_vec(tok(t as f32), (1, 1, d), &dev).unwrap();
            blk.decode(&x, 0, &mut s_a, &mut c_a).unwrap();
            blk.decode(&x, 0, &mut s_b, &mut c_b).unwrap();
        }
        let seq = 70;
        let flat: Vec<f32> = (0..seq).flat_map(|t| tok(10.0 + t as f32)).collect();
        let x = Tensor::from_vec(flat, (1, seq, d), &dev).unwrap();
        let out = blk.prefill(&x, 0, &mut s_a, &mut c_a).unwrap();
        assert_eq!(out.dims(), &[1, seq, d]);
        let mut rows = Vec::with_capacity(seq);
        for t in 0..seq {
            rows.push(
                blk.decode(&x.narrow(1, t, 1).unwrap(), 0, &mut s_b, &mut c_b)
                    .unwrap(),
            );
        }
        let out_ref = Tensor::cat(&rows, 1).unwrap();
        let (a, b) = (
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            out_ref.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
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
        for (name, ca, cb) in [("S", &s_a, &s_b), ("conv", &c_a, &c_b)] {
            let (va, vb) = (
                ca.get(0)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                cb.get(0)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
            );
            let sc = vb.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            let md = va
                .iter()
                .zip(&vb)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                md <= 1e-4 * sc,
                "{name} state differs by {md} at scale {sc}"
            );
        }
    }

    /// The fused decode (three launches, the slabs in place) matches the
    /// composed path — the CPU's — at the real head geometry over 40 steps
    /// from a fresh state: every output and both state channels.
    /// The fused prefill (Kernel B as launches) matches the composed path —
    /// the CPU's — at the real head geometry over two chunks: a 70-position
    /// chunk (two 64-chunks, the second padded) from a fresh state, then a
    /// 30-position chunk against the carried window and `S`; the outputs and
    /// both state channels within 2e-4 relative.
    /// The stacked f16 stems (one gemv, the row split) decode as the four
    /// stems do — the same weights as row views of one matrix — at the real
    /// head geometry over 12 steps, within the mixed gemv's tile reorder.
    #[cfg(feature = "metal")]
    #[test]
    fn stacked_stems_decode_as_the_four_stems() {
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let d = cfg.hidden_size;
        let conv_ch = cfg.conv_channels();
        let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
        let h_v = cfg.ssm_dt_rank;
        let d_v = cfg.ssm_state_size;
        let f16 = |out: usize, seed: f32| {
            Tensor::from_vec(
                (0..out * d)
                    .map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.05)
                    .collect::<Vec<_>>(),
                (out, d),
                &gpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
        };
        let parts = [
            f16(conv_ch, 1.0),
            f16(value_dim, 2.0),
            f16(h_v, 4.0),
            f16(h_v, 5.0),
        ];
        let cat = Tensor::cat(&parts.iter().collect::<Vec<_>>(), 0).unwrap();
        let mut off = 0;
        let views: Vec<Tensor> = parts
            .iter()
            .map(|p| {
                let v = cat.narrow(0, off, p.dim(0).unwrap()).unwrap();
                off += p.dim(0).unwrap();
                v
            })
            .collect();
        let vecw = |n: usize, f: &dyn Fn(usize) -> f32| {
            Tensor::from_vec((0..n).map(f).collect::<Vec<f32>>(), n, &gpu).unwrap()
        };
        let build = |stack: Option<Tensor>| {
            let blk = GatedDeltaNet::<CrownExact>::new(
                QMatMul::TensorF16(views[0].clone()),
                QMatMul::TensorF16(views[1].clone()),
                vecw(conv_ch * cfg.ssm_conv_kernel, &|i| {
                    ((i as f32 * 0.03).cos()) * 0.2
                })
                .reshape((conv_ch, cfg.ssm_conv_kernel))
                .unwrap(),
                vecw(h_v, &|i| -0.5 - 0.01 * i as f32),
                vecw(h_v, &|i| 0.1 * (i % 7) as f32),
                QMatMul::TensorF16(views[2].clone()),
                QMatMul::TensorF16(views[3].clone()),
                vecw(d_v, &|i| 1.0 + 0.01 * (i % 5) as f32),
                QMatMul::TensorF16(
                    Tensor::from_vec(
                        (0..d * value_dim)
                            .map(|i| ((i as f32 * 0.013 + 6.0).sin()) * 0.05)
                            .collect::<Vec<_>>(),
                        (d, value_dim),
                        &gpu,
                    )
                    .unwrap()
                    .to_dtype(DType::F16)
                    .unwrap(),
                ),
                &cfg,
            );
            match stack {
                Some(c) => blk.with_stem_cat(c),
                None => blk,
            }
        };
        let (four, one) = (build(None), build(Some(cat.clone())));
        // The gate-stems-only stack (α | β), the checkpoint's case.
        let ab = Tensor::cat(&[&parts[2], &parts[3]], 0).unwrap();
        let two = build(None).with_gate_stems_cat(ab);
        let (mut s2, mut c2) = (RecurrentState::new(1), RecurrentState::new(1));
        // Non-vacuity: a stack with the gate rows zeroed must decode
        // differently — the stacked path is the one that runs.
        let zeroed = Tensor::cat(
            &[
                &parts[0],
                &Tensor::zeros(parts[1].shape(), DType::F16, &gpu).unwrap(),
                &parts[2],
                &parts[3],
            ],
            0,
        )
        .unwrap();
        let bad = build(Some(zeroed));
        let (mut sb, mut cb) = (RecurrentState::new(1), RecurrentState::new(1));
        let (mut s4, mut c4) = (RecurrentState::new(1), RecurrentState::new(1));
        let (mut s1, mut c1) = (RecurrentState::new(1), RecurrentState::new(1));
        let mut worst = 0f32;
        for t in 0..12 {
            let x = Tensor::from_vec(
                (0..d)
                    .map(|i| ((i as f32 * 0.05 + t as f32).sin()) * 0.6)
                    .collect::<Vec<f32>>(),
                (1, 1, d),
                &gpu,
            )
            .unwrap();
            let a: Vec<f32> = four
                .decode(&x, 0, &mut s4, &mut c4)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let b: Vec<f32> = one
                .decode(&x, 0, &mut s1, &mut c1)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let bad_out: Vec<f32> = bad
                .decode(&x, 0, &mut sb, &mut cb)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_ne!(a, bad_out, "step {t}: the stacked path did not run");
            assert!(b.iter().all(|v| v.is_finite()), "step {t}: non-finite");
            let ab_out: Vec<f32> = two
                .decode(&x, 0, &mut s2, &mut c2)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let diff = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            let diff_ab = a
                .iter()
                .zip(&ab_out)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            worst = worst.max(diff / scale).max(diff_ab / scale);
            assert!(
                diff <= 1e-4 * scale,
                "step {t}: stacked differs by {diff} at {scale}"
            );
            assert!(
                diff_ab <= 1e-4 * scale,
                "step {t}: the gate-stems stack differs by {diff_ab} at {scale}"
            );
        }
        println!("stacked vs four stems worst relative diff: {worst:e}");
    }

    #[cfg(feature = "metal")]
    #[test]
    fn prefill_fused_matches_composed_at_real_geometry() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let d = cfg.hidden_size;
        let conv_ch = cfg.conv_channels();
        let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
        let h_v = cfg.ssm_dt_rank;
        let d_v = cfg.ssm_state_size;
        let build = |dev: &Device| {
            let lin = |out: usize, inp: usize, seed: f32| {
                QMatMul::Tensor(
                    Tensor::from_vec(
                        (0..out * inp)
                            .map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.05)
                            .collect::<Vec<_>>(),
                        (out, inp),
                        dev,
                    )
                    .unwrap(),
                )
            };
            let vecw = |n: usize, f: &dyn Fn(usize) -> f32| {
                Tensor::from_vec((0..n).map(f).collect::<Vec<f32>>(), n, dev).unwrap()
            };
            GatedDeltaNet::<CrownExact>::new(
                lin(conv_ch, d, 1.0),
                lin(value_dim, d, 2.0),
                vecw(conv_ch * cfg.ssm_conv_kernel, &|i| {
                    ((i as f32 * 0.03).cos()) * 0.2
                })
                .reshape((conv_ch, cfg.ssm_conv_kernel))
                .unwrap(),
                vecw(h_v, &|i| -0.5 - 0.01 * i as f32),
                vecw(h_v, &|i| 0.1 * (i % 7) as f32),
                lin(h_v, d, 4.0),
                lin(h_v, d, 5.0),
                vecw(d_v, &|i| 1.0 + 0.01 * (i % 5) as f32),
                lin(d, value_dim, 6.0),
                &cfg,
            )
        };
        let (blk_cpu, blk_gpu) = (build(&cpu), build(&gpu));
        let (mut s_c, mut c_c) = (RecurrentState::new(1), RecurrentState::new(1));
        let (mut s_g, mut c_g) = (RecurrentState::new(1), RecurrentState::new(1));
        let close = |name: &str, a: &Tensor, b: &Tensor| -> f32 {
            let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
            let bv: Vec<f32> = b
                .to_device(&Device::Cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(av.len(), bv.len(), "{name}: sizes");
            assert!(bv.iter().all(|v| v.is_finite()), "{name}: non-finite fused");
            let scale = av.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let diff = av
                .iter()
                .zip(&bv)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            assert!(
                diff <= 2e-4 * scale,
                "{name}: fused differs by {diff} at scale {scale}"
            );
            diff / scale
        };
        let mut worst = 0f32;
        for (round, seq) in [70usize, 30].into_iter().enumerate() {
            let x = Tensor::from_vec(
                (0..seq * d)
                    .map(|i| ((i as f32 * 0.05 + round as f32).sin()) * 0.6)
                    .collect::<Vec<f32>>(),
                (1, seq, d),
                &cpu,
            )
            .unwrap();
            let a = blk_cpu.prefill(&x, 0, &mut s_c, &mut c_c).unwrap();
            let b = blk_gpu
                .prefill(&x.to_device(&gpu).unwrap(), 0, &mut s_g, &mut c_g)
                .unwrap();
            worst = worst.max(close(&format!("round {round} output"), &a, &b));
            worst = worst.max(close(
                &format!("round {round} S"),
                s_c.get(0).unwrap(),
                s_g.get(0).unwrap(),
            ));
            worst = worst.max(close(
                &format!("round {round} window"),
                c_c.get(0).unwrap(),
                c_g.get(0).unwrap(),
            ));
        }
        println!("prefill fused vs composed worst relative diff: {worst:e}");

        // The chunk scan against the host loop it replaces (`GdnChunkScan`,
        // `set_chunk_scan`): four chunks, within the same tolerance, and
        // the scan bit-stable run to run.
        let seq = 200usize;
        let x = Tensor::from_vec(
            (0..seq * d)
                .map(|i| ((i as f32 * 0.021 + 3.0).sin()) * 0.6)
                .collect::<Vec<f32>>(),
            (1, seq, d),
            &gpu,
        )
        .unwrap();
        let run = |on: bool| -> (Vec<f32>, Vec<f32>) {
            super::super::kernels::set_chunk_scan(on);
            let (mut s, mut c) = (RecurrentState::new(1), RecurrentState::new(1));
            let out = blk_gpu.prefill(&x, 0, &mut s, &mut c).unwrap();
            let o: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
            let st: Vec<f32> = s.get(0).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            (o, st)
        };
        let (o_loop, s_loop) = run(false);
        let (o_scan, s_scan) = run(true);
        let (o_scan2, s_scan2) = run(true);
        super::super::kernels::set_chunk_scan(true);
        assert_eq!(o_scan, o_scan2, "the scan is not bit-stable (output)");
        assert_eq!(s_scan, s_scan2, "the scan is not bit-stable (state)");
        for (name, a, b) in [("output", &o_loop, &o_scan), ("state", &s_loop, &s_scan)] {
            assert!(b.iter().all(|v| v.is_finite()), "{name}: non-finite scan");
            let scale = a.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let diff = a
                .iter()
                .zip(b.iter())
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            assert!(
                diff <= 2e-4 * scale,
                "{name}: the scan differs from the loop by {diff} at scale {scale}"
            );
            println!("chunk scan vs loop {name}: {:e}", diff / scale);
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn decode_fused_matches_composed_at_real_geometry() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let d = cfg.hidden_size;
        let conv_ch = cfg.conv_channels();
        let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
        let h_v = cfg.ssm_dt_rank;
        let d_v = cfg.ssm_state_size;
        let build = |dev: &Device| {
            let lin = |out: usize, inp: usize, seed: f32| {
                QMatMul::Tensor(
                    Tensor::from_vec(
                        (0..out * inp)
                            .map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.05)
                            .collect::<Vec<_>>(),
                        (out, inp),
                        dev,
                    )
                    .unwrap(),
                )
            };
            let vecw = |n: usize, f: &dyn Fn(usize) -> f32| {
                Tensor::from_vec((0..n).map(f).collect::<Vec<f32>>(), n, dev).unwrap()
            };
            GatedDeltaNet::<CrownExact>::new(
                lin(conv_ch, d, 1.0),
                lin(value_dim, d, 2.0),
                vecw(conv_ch * cfg.ssm_conv_kernel, &|i| {
                    ((i as f32 * 0.03).cos()) * 0.2
                })
                .reshape((conv_ch, cfg.ssm_conv_kernel))
                .unwrap(),
                vecw(h_v, &|i| -0.5 - 0.01 * i as f32),
                vecw(h_v, &|i| 0.1 * (i % 7) as f32),
                lin(h_v, d, 4.0),
                lin(h_v, d, 5.0),
                vecw(d_v, &|i| 1.0 + 0.01 * (i % 5) as f32),
                lin(d, value_dim, 6.0),
                &cfg,
            )
        };
        let (blk_cpu, blk_gpu) = (build(&cpu), build(&gpu));
        let (mut s_c, mut c_c) = (RecurrentState::new(1), RecurrentState::new(1));
        let (mut s_g, mut c_g) = (RecurrentState::new(1), RecurrentState::new(1));
        let steps = 40;
        let mut worst = 0f32;
        for t in 0..steps {
            let x = Tensor::from_vec(
                (0..d)
                    .map(|i| ((i as f32 * 0.05 + t as f32).sin()) * 0.6)
                    .collect::<Vec<f32>>(),
                (1, 1, d),
                &cpu,
            )
            .unwrap();
            let a = blk_cpu.decode(&x, 0, &mut s_c, &mut c_c).unwrap();
            let b = blk_gpu
                .decode(&x.to_device(&gpu).unwrap(), 0, &mut s_g, &mut c_g)
                .unwrap();
            let av: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
            let bv: Vec<f32> = b
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let scale = av.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let diff = av
                .iter()
                .zip(&bv)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            worst = worst.max(diff / scale);
            assert!(
                diff <= 2e-4 * scale,
                "step {t}: fused differs by {diff} at scale {scale}"
            );
        }
        for (name, ca, cg) in [("S", &s_c, &s_g), ("conv", &c_c, &c_g)] {
            let va: Vec<f32> = ca.get(0).unwrap().flatten_all().unwrap().to_vec1().unwrap();
            let vg: Vec<f32> = cg
                .get(0)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            assert_eq!(va.len(), vg.len(), "{name}: shapes");
            let sc = va.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let md = va
                .iter()
                .zip(&vg)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            assert!(
                md <= 2e-4 * sc,
                "{name} state differs by {md} at scale {sc}"
            );
        }
        // the fused arm ran: the slab is the same buffer step after step
        assert!(
            worst > 0.0 || steps == 0,
            "identical to the bit is suspicious for a reordered sum"
        );
        println!("fused vs composed worst relative diff over {steps} steps: {worst:e}");
    }

    /// The Metal chunk path matches the CPU chunk path on the real head
    /// geometry (16 k-heads → 48 v-heads, d 128, conv 10240 channels) at a
    /// small hidden size: same weights on both devices, 70 positions (past one
    /// chunk), outputs and both state channels equal within f32 reduction noise.
    #[cfg(feature = "metal")]
    #[test]
    fn prefill_metal_matches_cpu() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let d = cfg.hidden_size;
        let conv_ch = cfg.conv_channels();
        let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
        let h_v = cfg.ssm_dt_rank;
        let d_v = cfg.ssm_state_size;
        let build = |dev: &Device| {
            let lin = |out: usize, inp: usize, seed: f32| {
                QMatMul::Tensor(
                    Tensor::from_vec(
                        (0..out * inp)
                            .map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.05)
                            .collect::<Vec<_>>(),
                        (out, inp),
                        dev,
                    )
                    .unwrap(),
                )
            };
            let vecw = |n: usize, f: &dyn Fn(usize) -> f32| {
                Tensor::from_vec((0..n).map(f).collect::<Vec<f32>>(), n, dev).unwrap()
            };
            GatedDeltaNet::<CrownExact>::new(
                lin(conv_ch, d, 1.0),
                lin(value_dim, d, 2.0),
                vecw(conv_ch * cfg.ssm_conv_kernel, &|i| {
                    ((i as f32 * 0.03).cos()) * 0.2
                })
                .reshape((conv_ch, cfg.ssm_conv_kernel))
                .unwrap(),
                vecw(h_v, &|i| -0.5 - 0.01 * i as f32),
                vecw(h_v, &|i| 0.1 * (i % 7) as f32),
                lin(h_v, d, 4.0),
                lin(h_v, d, 5.0),
                vecw(d_v, &|i| 1.0 + 0.01 * (i % 5) as f32),
                lin(d, value_dim, 6.0),
                &cfg,
            )
        };
        let (b_cpu, b_gpu) = (build(&cpu), build(&gpu));
        let seq = 70;
        let flat: Vec<f32> = (0..seq * d)
            .map(|i| ((i as f32 * 0.05).sin()) * 0.6)
            .collect();
        let x_cpu = Tensor::from_vec(flat.clone(), (1, seq, d), &cpu).unwrap();
        let x_gpu = Tensor::from_vec(flat, (1, seq, d), &gpu).unwrap();
        let (mut sc, mut cc) = (RecurrentState::new(1), RecurrentState::new(1));
        let (mut sg, mut cg) = (RecurrentState::new(1), RecurrentState::new(1));
        let oc = b_cpu.prefill(&x_cpu, 0, &mut sc, &mut cc).unwrap();
        let og = b_gpu.prefill(&x_gpu, 0, &mut sg, &mut cg).unwrap();
        let cmp = |name: &str, a: &Tensor, b: &Tensor, tol: f32| {
            let va = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            let vb = b
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            assert_eq!(va.len(), vb.len());
            assert_eq!(
                vb.iter().filter(|x| !x.is_finite()).count(),
                0,
                "{name}: non-finite on Metal"
            );
            let scale = va.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
            let md = va
                .iter()
                .zip(&vb)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            assert!(
                md <= tol * scale,
                "{name}: Metal/CPU differ by {md} at scale {scale}"
            );
        };
        cmp("output", &oc, &og, 2e-3);
        cmp("S", sc.get(0).unwrap(), sg.get(0).unwrap(), 2e-3);
        cmp("conv", cc.get(0).unwrap(), cg.get(0).unwrap(), 1e-4);
    }

    /// Run-to-run identity of the decode path at the real geometry (16 → 48
    /// heads × 128, 10240 conv channels) over 1500 steps, twice from fresh
    /// state channels: every step's output bit-identical. The state is
    /// rewritten in place every step (`RecurrentState::write`), so a missing
    /// read-before-write fence on the slab would show here as a step whose
    /// two passes disagree.
    #[cfg(feature = "metal")]
    #[test]
    fn decode_is_repeatable_at_real_geometry() {
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 128;
        let d = cfg.hidden_size;
        let conv_ch = cfg.conv_channels();
        let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
        let h_v = cfg.ssm_dt_rank;
        let d_v = cfg.ssm_state_size;
        let lin = |out: usize, inp: usize, seed: f32| {
            QMatMul::Tensor(
                Tensor::from_vec(
                    (0..out * inp)
                        .map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.05)
                        .collect::<Vec<_>>(),
                    (out, inp),
                    &gpu,
                )
                .unwrap(),
            )
        };
        let vecw = |n: usize, f: &dyn Fn(usize) -> f32| {
            Tensor::from_vec((0..n).map(f).collect::<Vec<f32>>(), n, &gpu).unwrap()
        };
        let blk = GatedDeltaNet::<CrownExact>::new(
            lin(conv_ch, d, 1.0),
            lin(value_dim, d, 2.0),
            vecw(conv_ch * cfg.ssm_conv_kernel, &|i| {
                ((i as f32 * 0.03).cos()) * 0.2
            })
            .reshape((conv_ch, cfg.ssm_conv_kernel))
            .unwrap(),
            vecw(h_v, &|i| -0.5 - 0.01 * i as f32),
            vecw(h_v, &|i| 0.1 * (i % 7) as f32),
            lin(h_v, d, 4.0),
            lin(h_v, d, 5.0),
            vecw(d_v, &|i| 1.0 + 0.01 * (i % 5) as f32),
            lin(d, value_dim, 6.0),
            &cfg,
        );
        let steps = 1500usize;
        let xs: Vec<Tensor> = (0..steps)
            .map(|t| {
                Tensor::from_vec(
                    (0..d)
                        .map(|i| ((i as f32 * 0.05 + t as f32).sin()) * 0.6)
                        .collect::<Vec<f32>>(),
                    (1, 1, d),
                    &gpu,
                )
                .unwrap()
            })
            .collect();
        let run = || -> Vec<Vec<f32>> {
            let (mut s, mut c) = (RecurrentState::new(1), RecurrentState::new(1));
            xs.iter()
                .map(|x| {
                    blk.decode(x, 0, &mut s, &mut c)
                        .unwrap()
                        .to_device(&Device::Cpu)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap()
                })
                .collect()
        };
        let a = run();
        let b = run();
        for (t, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(x.iter().all(|v| v.is_finite()), "step {t}: non-finite");
            let n_diff = x
                .iter()
                .zip(y)
                .filter(|(p, q)| p.to_bits() != q.to_bits())
                .count();
            assert_eq!(
                n_diff,
                0,
                "step {t}: {n_diff} of {} values differ between two identical passes",
                x.len()
            );
        }
    }

    /// The block runs on a `Config::tiny` FULL-attn interval layer index too
    /// (the block itself is linear-only, but the dims are shared) — a smoke that
    /// the geometry is consistent across layer indices.
    #[test]
    fn decode_runs_on_multiple_layers() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let d = cfg.hidden_size;
        let conv_ch = cfg.conv_channels();
        let value_dim = cfg.ssm_state_size * cfg.ssm_dt_rank;
        let blk = GatedDeltaNet::<CrownExact>::new(
            lin(conv_ch, d, &dev),
            lin(value_dim, d, &dev),
            Tensor::zeros((conv_ch, cfg.ssm_conv_kernel), DType::F32, &dev).unwrap(),
            vecw(cfg.ssm_dt_rank, &dev, |_| -0.5),
            vecw(cfg.ssm_dt_rank, &dev, |_| 0.0),
            lin(cfg.ssm_dt_rank, d, &dev),
            lin(cfg.ssm_dt_rank, d, &dev),
            vecw(cfg.ssm_state_size, &dev, |_| 1.0),
            lin(d, value_dim, &dev),
            &cfg,
        );
        let mut s = RecurrentState::new(cfg.n_layers);
        let mut c = RecurrentState::new(cfg.n_layers);
        let x = Tensor::zeros((1, 1, d), DType::F32, &dev).unwrap();
        for l in [0usize, 1, 2] {
            let o = blk.decode(&x, l, &mut s, &mut c).unwrap();
            assert_eq!(o.dims(), &[1, 1, d]);
        }
    }
}
