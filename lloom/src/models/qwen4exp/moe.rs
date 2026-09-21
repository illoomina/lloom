//! MoE FFN block behind the `ExpertResidency` seam (§20.6).
//!
//! `MoeLayer` is generic over an `ExpertResidency` strategy so a `Resident`
//! model (all experts in RAM) monomorphizes to a no-loader path (§3.6 HARD RULE,
//! static dispatch — no `dyn` on decode), and issue 3's `Streamed` impl plugs in
//! without a forward-pass edit. This issue builds only `Resident`.
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
use lloom_tensor::quantized::{QMatMul, QTensor};
use lloom_tensor::{DType, Module, Result, Tensor, D};
use std::sync::Arc;

/// How a layer's routed experts are held and applied. `Resident` = all in RAM
/// (this issue); a `Streamed` impl (issue 3) selects the same seam by type.
///
/// The contract is the routed matmul's: the strategy receives the activation
/// and the selected expert ids and returns the per-slot projections, so a
/// `Streamed` impl can ensure residency for exactly `ids` (and prefetch from
/// them) before dispatching the same kernel. The residency query is resolved
/// at plan time (here, trivially — every expert is present), never an O(N)
/// scan on the per-token path (§3.3).
pub trait ExpertResidency {
    /// `gate` and `up` projections of `x [batch, 1, hidden]` through experts
    /// `ids [batch, topk]`: each `[batch, topk, n_ff]`.
    fn gate_up(&self, x: &Tensor, ids: &Tensor) -> Result<(Tensor, Tensor)>;
    /// `down` projection of `h [batch, topk, n_ff]` through experts `ids`:
    /// `[batch, topk, hidden]`.
    fn down(&self, h: &Tensor, ids: &Tensor) -> Result<Tensor>;
}

/// All experts resident in RAM: the small-model / big-RAM case, zero streaming
/// overhead. Holds the GGUF's stacked expert tensors as they come —
/// `ffn_{gate,up}_exps [n_expert, n_ff, hidden]`, `ffn_down_exps [n_expert,
/// hidden, n_ff]` — packed; the routed kernel reads the selected experts'
/// blocks in place.
#[derive(Debug, Clone)]
pub struct Resident {
    gate: Arc<QTensor>,
    up: Arc<QTensor>,
    down: Arc<QTensor>,
}

impl Resident {
    pub fn new(gate: Arc<QTensor>, up: Arc<QTensor>, down: Arc<QTensor>) -> Self {
        Self { gate, up, down }
    }
}

impl ExpertResidency for Resident {
    #[inline]
    fn gate_up(&self, x: &Tensor, ids: &Tensor) -> Result<(Tensor, Tensor)> {
        Ok((
            self.gate.indexed_moe_forward(x, ids)?,
            self.up.indexed_moe_forward(x, ids)?,
        ))
    }
    #[inline]
    fn down(&self, h: &Tensor, ids: &Tensor) -> Result<Tensor> {
        self.down.indexed_moe_forward(h, ids)
    }
}

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
        }
    }

    /// One decode token through the MoE. `x` is `[1, 1, hidden]`, returns
    /// `[1, 1, hidden]` = routed top-k combine + gated shared expert.
    ///
    /// Everything stays on the device: the top-k ids feed the routed dispatch
    /// directly and the weighted sum over slots is a broadcast-multiply and a
    /// reduction. Correctness-first: the routing is a softmax over all experts,
    /// top-k, renormalize (reference `build_moe_ffn`).
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.dim(D::Minus1)?;
        let topk = self.n_experts_per_tok;
        let x1 = x.reshape((1, hidden))?; // [1, hidden]

        // Router: softmax over all experts, then top-k select + renormalize.
        let logits = self.router.forward(&x1)?.to_dtype(DType::F32)?; // [1, n_experts]
        super::trace("moe.logits", usize::MAX, &logits);
        let probs = ops::softmax_last_dim(&logits)?; // [1, n_experts]
        let ids = probs
            .arg_sort_last_dim(false)?
            .narrow(D::Minus1, 0, topk)?
            .contiguous()?; // [1, topk] u32
        let w = probs.gather(&ids, D::Minus1)?; // [1, topk]
        let w = w.broadcast_div(&w.sum_keepdim(D::Minus1)?)?;
        if super::trace_enabled() {
            eprintln!(
                "TRACE moe.ids={:?} w={:?}",
                ids.flatten_all()?.to_vec1::<u32>()?,
                w.flatten_all()?.to_vec1::<f32>()?
            );
        }

        // The selected experts' SwiGLU, one routed dispatch per projection.
        let x3 = x1.to_dtype(DType::F32)?.reshape((1, 1, hidden))?; // [1, 1, hidden]
        let (g, u) = self.experts.gate_up(&x3, &ids)?; // [1, topk, n_ff] each
        super::trace("moe.g", usize::MAX, &g);
        super::trace("moe.u", usize::MAX, &u);
        let h = (ops::silu(&g)? * u)?;
        let y = self.experts.down(&h, &ids)?; // [1, topk, hidden]
        super::trace("moe.y", usize::MAX, &y);
        let routed = y
            .broadcast_mul(&w.reshape((1, topk, 1))?)?
            .sum(1)?
            .to_dtype(x1.dtype())?; // [1, hidden]

        // Shared expert: SwiGLU gated by sigmoid(sh_router·x), always on.
        let shared = swiglu(&x1, &self.sh_gate, &self.sh_up, &self.sh_down)?;
        let sh_gate_v = ops::sigmoid(&self.sh_router.forward(&x1)?.to_dtype(shared.dtype())?)?; // [1, 1]
        let shared = shared.broadcast_mul(&sh_gate_v)?;

        let out = (routed + shared)?;
        out.reshape((1, 1, hidden))
    }
}

/// SwiGLU: `down( silu(gate(x)) * up(x) )`.
fn swiglu(x: &Tensor, gate: &QMatMul, up: &QMatMul, down: &QMatMul) -> Result<Tensor> {
    let g = ops::silu(&gate.forward(x)?)?;
    let u = up.forward(x)?;
    down.forward(&(g * u)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::quantized::GgmlDType;
    use lloom_tensor::Device;

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
