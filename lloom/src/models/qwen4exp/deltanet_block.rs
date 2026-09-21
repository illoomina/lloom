//! The `GatedDeltaNet` weight-bearing block: loads one linear-attention layer's
//! weights and runs its decode forward, composing the tested primitives
//! (`deltanet::*`) with the [`RecurrentState`] channels for `S` and the conv
//! window.
//!
//! Reference dataflow: `qwen4exp.cpp build_layer_attn_linear`. The block is
//! model-parameterized by [`Config`] so a second delta-net model reuses it.

use super::config::Config;
use super::deltanet::{
    conv_decode_step, decode_step, gate_and_beta, gqa_expand, l2_norm_last, q_scale, DeltaNetDims,
};
use crate::nn::{ops, RecurrentState};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Module, Result, Tensor, D};

/// One Gated DeltaNet layer's weights.
///
/// `QMatMul` holds each projection (quantized IQ/Q2_0 or BF16 — it dequantizes
/// on the fly), so the block reaches quant weights through the one seam
/// (`call_quantized_matmul_mv_t`), never a bespoke dequant.
#[derive(Debug, Clone)]
pub struct GatedDeltaNet {
    /// Fused QKV in-proj → `[conv_channels]` (`attn_qkv`).
    wqkv: QMatMul,
    /// Output-gate in-proj → `[value_dim]` (`attn_gate`).
    wqkv_gate: QMatMul,
    /// Depthwise conv kernel `[conv_channels, k]` (`ssm_conv1d`), k=4.
    conv_weight: Tensor,
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
    dims: DeltaNetDims,
    conv_k: usize,
    conv_channels: usize,
    value_dim: usize,
    key_dim: usize,
    norm_eps: f64,
}

impl GatedDeltaNet {
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
        let dims = cfg.deltanet_dims();
        Self {
            wqkv,
            wqkv_gate,
            conv_weight,
            ssm_a,
            ssm_dt,
            ssm_alpha,
            ssm_beta,
            ssm_norm,
            ssm_out,
            dims,
            conv_k: cfg.ssm_conv_kernel,
            conv_channels: cfg.conv_channels(),
            value_dim: cfg.ssm_state_size * cfg.ssm_dt_rank,
            key_dim: cfg.ssm_state_size * cfg.ssm_n_group,
            norm_eps: cfg.norm_eps,
        }
    }

    /// One decode step: `x [1, 1, hidden]` in, `[1, 1, hidden]` out, threading
    /// this layer's `S` (state channel `l`) and conv window (conv channel `l`).
    ///
    /// Composes, per `build_layer_attn_linear`: qkvz → gate(α,g,β) → conv →
    /// silu → split q/k/v → l2norm(q,k) → GQA tile → decode_step → write S →
    /// gated-norm(·sigmoid(z)) → out-proj.
    pub fn decode(
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
            &self.conv_weight,
            conv_state.get(l),
            self.conv_channels,
            self.conv_k,
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

        // gated norm: rms_norm(o, ssm_norm) · sigmoid(z). o is [h_v, d_v].
        let o_flat = o.reshape((1, 1, self.value_dim))?; // back to [1,1,value_dim]
        let normed = self.gated_rms_norm(&o_flat)?;
        let z_gate = ops::sigmoid(&z.to_dtype(normed.dtype())?)?;
        let gated = (normed * z_gate)?;

        // out-proj → [1,1,hidden].
        self.ssm_out.forward(&gated)
    }

    /// GatedRMSNorm over the value-head dim with the `ssm_norm` weight. `o` is
    /// `[1, 1, value_dim]`; normalize per value-head (over d_v) then scale.
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

    /// Silence the unused-import lint for `D` until prefill (Kernel B) uses it.
    #[allow(dead_code)]
    fn _uses_d() -> D {
        D::Minus1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let blk = GatedDeltaNet::new(
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
        let blk = GatedDeltaNet::new(
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
