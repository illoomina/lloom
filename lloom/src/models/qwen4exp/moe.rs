//! MoE FFN block behind the `ExpertResidency` seam (§20.6).
//!
//! `MoeLayer` is generic over an `ExpertResidency` strategy
//! (`nn::expert_stream`, the model-independent seam) so a `Resident` model
//! (all experts in RAM) monomorphizes to a no-loader path (§3.6 HARD RULE,
//! static dispatch — no `dyn` on decode) and `Streamed` plugs in with the same
//! forward: the strategy groups the routed rows and resolves their expert ids
//! to what the kernel reads; the dispatch and the combine are shared.
//!
//! Routing (reference `build_moe_ffn`, softmax gating): `probs = softmax(router·x)`
//! over all experts → top-k → renormalize the top-k weights → weighted sum of the
//! selected experts' SwiGLU. Plus a shared expert gated by `sigmoid(shared_gate·x)`.
//! Router top-k reuses candle `arg_sort` (§3.6a — a selection of existing blocks);
//! the selected experts' GEMVs are ONE routed dispatch over the stacked,
//! packed-quantized expert weight (`QTensor::indexed_moe_forward`, the
//! `mul_mv_id` kernels on Metal — dequant in-kernel, no per-expert slice, no
//! host readback of the ids).

use super::config::Config;
use crate::nn::ops;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{DType, Module, Result, Tensor, D};

pub use crate::nn::expert_stream::{ExpertResidency, Resident, RoutedGroup, Streamed};

/// One layer's MoE, generic over the residency strategy (static dispatch).
#[derive(Debug, Clone)]
pub struct MoeLayer<R: ExpertResidency> {
    /// Router `hidden -> n_experts` (`ffn_gate_inp`).
    router: QMatMul,
    /// The routed experts, behind the residency seam.
    experts: R,
    /// Shared expert `up`/`gate`/`down` (`ffn_*_shexp`).
    sh_gate: QMatMul,
    sh_up: QMatMul,
    sh_down: QMatMul,
    /// Shared-expert gate `hidden -> 1` (`ffn_gate_inp_shexp`).
    sh_router: QMatMul,
    n_experts_per_tok: usize,
    /// The activation dtype of the routed dispatch — its `(activation, f32,
    /// store)` triple (§15.2a): `F32` is the byte-identical `_f32` routed arm,
    /// `F16` the `(f16, f32, f16)` arm (issue #748). The combine over slots
    /// stays f32 either way.
    routed_act: DType,
}

impl<R: ExpertResidency> MoeLayer<R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        router: QMatMul,
        experts: R,
        sh_gate: QMatMul,
        sh_up: QMatMul,
        sh_down: QMatMul,
        sh_router: QMatMul,
        cfg: &Config,
    ) -> Self {
        Self {
            router,
            experts,
            sh_gate,
            sh_up,
            sh_down,
            sh_router,
            n_experts_per_tok: cfg.n_experts_per_tok,
            routed_act: DType::F32,
        }
    }

    /// Select the routed dispatch's activation dtype (`F32` or `F16`).
    pub fn set_routed_act(&mut self, dtype: DType) {
        self.routed_act = dtype;
    }

    pub fn routed_act(&self) -> DType {
        self.routed_act
    }

    /// The residency strategy behind the routed experts.
    pub fn experts(&self) -> &R {
        &self.experts
    }

    /// One decode token through the MoE. `x` is `[1, 1, hidden]`, returns
    /// `[1, 1, hidden]` = routed top-k combine + gated shared expert.
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        self.forward(x)
    }

    /// `x [1, seq, hidden]` → `[1, seq, hidden]`, every position routed on
    /// its own: routed top-k combine + gated shared expert.
    ///
    /// Everything stays on the device: the top-k ids feed the routed dispatch
    /// directly (`seq` planes of `topk` slots) and the weighted sum over slots
    /// is a broadcast-multiply and a reduction. Correctness-first: the routing
    /// is a softmax over all experts, top-k, renormalize (reference
    /// `build_moe_ffn`).
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(self.forward_inner(x, false)?.0)
    }

    /// [`Self::forward`] with the routed experts planned ON THE DEVICE when
    /// the residency can (`ExpertResidency::plan_device`): no readback at
    /// the layer; the plan's record comes back with the output and the
    /// caller settles it at the end of the pass (`settle`), replaying the
    /// pass if it computed on a missing expert. `None` record = the host
    /// plan ran (nothing to settle).
    pub fn decode_planned(&self, x: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        self.forward_inner(x, true)
    }

    fn forward_inner(&self, x: &Tensor, device_plan: bool) -> Result<(Tensor, Option<Tensor>)> {
        let (b, seq, hidden) = x.dims3()?;
        if b != 1 {
            lloom_tensor::bail!("moe takes one sequence, got {:?}", x.shape())
        }
        let topk = self.n_experts_per_tok;
        let x1 = x.reshape((seq, hidden))?; // [seq, hidden]

        // Router: softmax over all experts, then top-k select + renormalize —
        // one launch on Metal (`kernels::MoeTopk`), the composed six elsewhere.
        let logits = self.router.forward(&x1)?.to_dtype(DType::F32)?; // [seq, n_experts]
        super::trace("moe.logits", usize::MAX, &logits);
        let (ids, w) = match super::kernels::moe_topk(&logits, topk) {
            Some(r) => r?,
            None => {
                let probs = ops::softmax_last_dim(&logits)?; // [seq, n_experts]
                let ids = probs
                    .arg_sort_last_dim(false)?
                    .narrow(D::Minus1, 0, topk)?
                    .contiguous()?; // [seq, topk] u32
                let w = probs.gather(&ids, D::Minus1)?; // [seq, topk]
                let w = w.broadcast_div(&w.sum_keepdim(D::Minus1)?)?;
                (ids, w)
            }
        };
        if super::trace_enabled() {
            eprintln!(
                "TRACE moe.ids={:?} w={:?}",
                ids.flatten_all()?.to_vec1::<u32>()?,
                w.flatten_all()?.to_vec1::<f32>()?
            );
        }

        // The selected experts' SwiGLU, one routed dispatch per projection
        // per group, in the routed arm's activation dtype. `Resident` plans
        // one group over the ids as they are; `Streamed` groups the rows to
        // its slot pool and resolves each group's ids to slots first.
        let x3 = x1.to_dtype(self.routed_act)?.reshape((seq, 1, hidden))?; // [seq, 1, hidden]
        let device_plan = if device_plan {
            self.experts.plan_device(&ids)?
        } else {
            None
        };
        let mut record = None;
        let y = match device_plan {
            // One group over the slot ids the device resolved.
            Some((slot_ids, rec)) => {
                record = Some(rec);
                let (g, u) = self.experts.gate_up(&x3, &slot_ids)?;
                let h = match super::kernels::silu_mul(&g, &u) {
                    Some(h) => h?,
                    None => (ops::silu(&g)? * u)?,
                };
                self.experts.down(&h, &slot_ids)?
            }
            None => {
                let groups = self.experts.plan(&ids)?;
                let mut ys = Vec::with_capacity(groups.len());
                for group in &groups {
                    let slot_ids = self.experts.ensure(group)?;
                    let xg = if groups.len() == 1 {
                        x3.clone()
                    } else {
                        x3.narrow(0, group.rows.start, group.rows.len())?
                    };
                    let (g, u) = self.experts.gate_up(&xg, &slot_ids)?; // [rows, topk, n_ff] each
                    super::trace("moe.g", usize::MAX, &g);
                    super::trace("moe.u", usize::MAX, &u);
                    let h = match super::kernels::silu_mul(&g, &u) {
                        Some(h) => h?,
                        None => (ops::silu(&g)? * u)?,
                    };
                    ys.push(self.experts.down(&h, &slot_ids)?); // [rows, topk, hidden]
                }
                if ys.len() == 1 {
                    ys.pop().expect("one group")
                } else {
                    Tensor::cat(&ys, 0)?
                }
            }
        };
        super::trace("moe.y", usize::MAX, &y);

        // Shared expert: SwiGLU gated by sigmoid(sh_router·x), always on.
        let shared = swiglu(&x1, &self.sh_gate, &self.sh_up, &self.sh_down)?;
        let sh_logit = self.sh_router.forward(&x1)?.to_dtype(shared.dtype())?; // [seq, 1]

        // The routed combine over the top-k plus the gated shared expert: one
        // launch on Metal (`kernels::MoeCombine`), the composed five elsewhere.
        let y32 = y.to_dtype(DType::F32)?;
        let out = match super::kernels::moe_combine(&y32, &w, &shared, &sh_logit, topk, hidden) {
            Some(o) => o?.to_dtype(x1.dtype())?,
            None => {
                let routed = y32
                    .broadcast_mul(&w.reshape((seq, topk, 1))?)?
                    .sum(1)?
                    .to_dtype(x1.dtype())?; // [seq, hidden]
                let sh_gate_v = ops::sigmoid(&sh_logit)?;
                let shared = shared.broadcast_mul(&sh_gate_v)?;
                (routed + shared)?
            }
        };
        Ok((out.reshape((1, seq, hidden))?, record))
    }
}

/// SwiGLU: `down( silu(gate(x)) * up(x) )`.
fn swiglu(x: &Tensor, gate: &QMatMul, up: &QMatMul, down: &QMatMul) -> Result<Tensor> {
    let g = gate.forward(x)?;
    let u = up.forward(x)?;
    let h = match super::kernels::silu_mul(&g, &u) {
        Some(h) => h?,
        None => (ops::silu(&g)? * u)?,
    };
    down.forward(&h)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::quantized::{GgmlDType, QTensor};
    use lloom_tensor::Device;
    use std::sync::Arc;

    fn lin(out: usize, inp: usize, seed: f32, dev: &Device) -> QMatMul {
        let n = out * inp;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.1)
            .collect();
        QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev).unwrap())
    }

    /// A stacked `[e, out, inp]` expert weight as an F32-typed `QTensor` (the
    /// routed path's storage; F32 blocks so the CPU arm is exact).
    pub(crate) fn stacked(
        e: usize,
        out: usize,
        inp: usize,
        seed: f32,
        dev: &Device,
    ) -> Arc<QTensor> {
        let n = e * out * inp;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.011 + seed).cos()) * 0.1)
            .collect();
        let t = Tensor::from_vec(v, (e, out, inp), dev).unwrap();
        Arc::new(QTensor::quantize(&t, GgmlDType::F32).unwrap())
    }

    fn resident(cfg: &Config, dev: &Device) -> Resident {
        let (h, ff, ne) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_experts);
        Resident::new(
            stacked(ne, ff, h, 1.0, dev),
            stacked(ne, ff, h, 2.0, dev),
            stacked(ne, h, ff, 3.0, dev),
        )
    }

    /// The MoE decodes end to end (Resident): shapes thread, output finite, and
    /// the seam monomorphizes to the no-loader path.
    #[test]
    fn moe_resident_decodes() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let h = cfg.hidden_size;
        let moe = MoeLayer::new(
            lin(cfg.n_experts, h, 1.0, &dev), // router
            resident(&cfg, &dev),
            lin(cfg.shared_expert_intermediate_size, h, 2.0, &dev), // sh_gate
            lin(cfg.shared_expert_intermediate_size, h, 3.0, &dev), // sh_up
            lin(h, cfg.shared_expert_intermediate_size, 4.0, &dev), // sh_down
            lin(1, h, 5.0, &dev),                                   // sh_router (1 value/token)
            &cfg,
        );
        let x = Tensor::from_vec(
            (0..h)
                .map(|i| ((i as f32) * 0.05).cos() * 0.3)
                .collect::<Vec<f32>>(),
            (1, 1, h),
            &dev,
        )
        .unwrap();
        let o = moe.decode(&x).unwrap();
        assert_eq!(o.dims(), &[1, 1, h]);
        let ov = o.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(ov.iter().all(|v| v.is_finite()));
        // Not degenerate zero (the shared expert is always on).
        assert!(ov.iter().any(|&v| v.abs() > 1e-9));
    }

    /// The routed combine equals an explicit per-expert reference: softmax the
    /// router, take the top-k, renormalize, and sum `w_e · down_e(silu(gate_e x)
    /// · up_e x)` with the dense per-expert matrices. Pins the routed dispatch
    /// (ids → stacked slices) and the on-device weighted sum against a value
    /// computed by a different route (§3.1b: a reference, not a shape).
    #[test]
    fn routed_combine_matches_per_expert_reference() {
        let dev = Device::Cpu;
        let mut cfg = Config::tiny();
        cfg.n_experts = 6;
        cfg.n_experts_per_tok = 3;
        let (h, ff, ne, k) = (
            cfg.hidden_size,
            cfg.moe_intermediate_size,
            cfg.n_experts,
            cfg.n_experts_per_tok,
        );
        let (gate, up, down) = (
            stacked(ne, ff, h, 1.0, &dev),
            stacked(ne, ff, h, 2.0, &dev),
            stacked(ne, h, ff, 3.0, &dev),
        );
        let router = lin(ne, h, 1.0, &dev);
        // zero shared expert so only the routed half is compared
        let zero =
            |o: usize, i: usize| QMatMul::Tensor(Tensor::zeros((o, i), DType::F32, &dev).unwrap());
        let moe = MoeLayer::new(
            router.clone(),
            Resident::new(gate.clone(), up.clone(), down.clone()),
            zero(ff, h),
            zero(ff, h),
            zero(h, ff),
            zero(1, h),
            &cfg,
        );
        let x = Tensor::from_vec(
            (0..h)
                .map(|i| ((i as f32) * 0.07).sin() * 0.4)
                .collect::<Vec<f32>>(),
            (1, 1, h),
            &dev,
        )
        .unwrap();
        let got = moe
            .decode(&x)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        // reference
        let x1 = x.reshape((1, h)).unwrap();
        let probs = ops::softmax_last_dim(&router.forward(&x1).unwrap()).unwrap();
        let pv = probs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let mut order: Vec<usize> = (0..ne).collect();
        order.sort_by(|&a, &b| pv[b].partial_cmp(&pv[a]).unwrap());
        let sel = &order[..k];
        let wsum: f32 = sel.iter().map(|&e| pv[e]).sum();
        let (gd, ud, dd) = (
            gate.dequantize(&dev).unwrap(),
            up.dequantize(&dev).unwrap(),
            down.dequantize(&dev).unwrap(),
        );
        let mut want = vec![0f32; h];
        for &e in sel {
            let g = x1.matmul(&gd.get(e).unwrap().t().unwrap()).unwrap();
            let u = x1.matmul(&ud.get(e).unwrap().t().unwrap()).unwrap();
            let hh = (ops::silu(&g).unwrap() * u).unwrap();
            let y = hh.matmul(&dd.get(e).unwrap().t().unwrap()).unwrap();
            let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
            for (w, yv) in want.iter_mut().zip(yv) {
                *w += pv[e] / wsum * yv;
            }
        }
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() < 1e-5, "mismatch at {i}: got {g} want {w}");
        }
        assert!(want.iter().any(|w| w.abs() > 1e-6), "degenerate reference");
    }

    /// A chunk through `forward` equals its positions through `decode` one by
    /// one (per-position routing, the `seq`-plane routed dispatch, the
    /// per-row combine).
    #[test]
    fn forward_over_seq_matches_per_position_decode() {
        let dev = Device::Cpu;
        let mut cfg = Config::tiny();
        cfg.n_experts = 6;
        cfg.n_experts_per_tok = 3;
        let (h, ff, ne) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_experts);
        let moe = MoeLayer::new(
            lin(ne, h, 1.0, &dev),
            Resident::new(
                stacked(ne, ff, h, 1.0, &dev),
                stacked(ne, ff, h, 2.0, &dev),
                stacked(ne, h, ff, 3.0, &dev),
            ),
            lin(ff, h, 2.0, &dev),
            lin(ff, h, 3.0, &dev),
            lin(h, ff, 4.0, &dev),
            lin(1, h, 5.0, &dev),
            &cfg,
        );
        let seq = 5;
        let x = Tensor::from_vec(
            (0..seq * h)
                .map(|i| ((i as f32) * 0.07).sin() * 0.4)
                .collect::<Vec<f32>>(),
            (1, seq, h),
            &dev,
        )
        .unwrap();
        let all = moe.forward(&x).unwrap();
        assert_eq!(all.dims(), &[1, seq, h]);
        for t in 0..seq {
            let one = moe.decode(&x.narrow(1, t, 1).unwrap()).unwrap();
            let (a, b) = (
                all.narrow(1, t, 1)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap(),
                one.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            );
            for (x, y) in a.iter().zip(&b) {
                assert!((x - y).abs() < 1e-5, "position {t}: {a:?} vs {b:?}");
            }
        }
    }

    /// Router selects exactly `n_experts_per_tok` experts and the path runs
    /// with k=2 over 4 experts to the right shape.
    #[test]
    fn router_selects_topk_renormalized() {
        let dev = Device::Cpu;
        let mut cfg = Config::tiny();
        cfg.n_experts = 4;
        cfg.n_experts_per_tok = 2;
        let (h, ff) = (cfg.hidden_size, cfg.moe_intermediate_size);
        let zero =
            |o: usize, i: usize| QMatMul::Tensor(Tensor::zeros((o, i), DType::F32, &dev).unwrap());
        let moe = MoeLayer::new(
            lin(4, h, 1.0, &dev),
            resident(&cfg, &dev),
            zero(ff, h),
            zero(ff, h),
            zero(h, ff),
            zero(1, h),
            &cfg,
        );
        let x = Tensor::ones((1, 1, h), DType::F32, &dev).unwrap();
        let o = moe.decode(&x).unwrap();
        assert_eq!(o.dims(), &[1, 1, h]);
    }
}
