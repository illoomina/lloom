//! The `GatedDeltaNet` weight-bearing block: loads one linear-attention layer's
//! weights and runs its decode forward, composing the tested primitives
//! (`deltanet::*`) with the [`RecurrentState`] channels for `S` and the conv
//! window.
//!
//! Reference dataflow: `qwen4exp.cpp build_layer_attn_linear`. The block is
//! model-parameterized by [`Config`] so a second delta-net model reuses it.

use super::config::Config;
use super::deltanet::{
    chunk_forward, conv_decode_step, conv_prefill, decode_step, gate_and_beta, gqa_expand,
    gqa_expand_seq, l2_norm_last, q_scale, DeltaNetDims, CHUNK_SIZE,
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
            &self.conv_weight,
            conv_state.get(l),
            self.conv_channels,
            self.conv_k,
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
        let z_gate = ops::sigmoid(&z.to_dtype(normed.dtype())?)?;
        let gated = (normed * z_gate)?;
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
        let blk = GatedDeltaNet::new(
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
            GatedDeltaNet::new(
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
                    (0..out * inp).map(|i| ((i as f32 * 0.017 + seed).sin()) * 0.05).collect::<Vec<_>>(),
                    (out, inp),
                    &gpu,
                )
                .unwrap(),
            )
        };
        let vecw = |n: usize, f: &dyn Fn(usize) -> f32| Tensor::from_vec((0..n).map(f).collect::<Vec<f32>>(), n, &gpu).unwrap();
        let blk = GatedDeltaNet::new(
            lin(conv_ch, d, 1.0),
            lin(value_dim, d, 2.0),
            vecw(conv_ch * cfg.ssm_conv_kernel, &|i| ((i as f32 * 0.03).cos()) * 0.2)
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
                    (0..d).map(|i| ((i as f32 * 0.05 + t as f32).sin()) * 0.6).collect::<Vec<f32>>(),
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
            let n_diff = x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count();
            assert_eq!(n_diff, 0, "step {t}: {n_diff} of {} values differ between two identical passes", x.len());
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
