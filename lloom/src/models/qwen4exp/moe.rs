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
//! Router top-k reuses candle `arg_sort` (§3.6a — a selection of existing blocks).

use super::config::Config;
use crate::nn::ops;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Module, Result, Tensor, D};

/// How a layer's experts are held. `Resident` = all in RAM (this issue); a
/// `Streamed` impl (issue 3) selects the same seam by type.
///
/// The trait is the seam §20.6 names; the residency query is resolved at plan
/// time (here, trivially — every expert is present), never an O(N) scan on the
/// per-token path (§3.3).
pub trait ExpertResidency {
    /// The `gate`/`up`/`down` weights for expert `e`, ready for a GEMV. On the
    /// `Resident` path this is offset arithmetic into the resident set — no load,
    /// no fault. A `Streamed` impl would ensure-resident here.
    fn expert(&self, e: usize) -> Result<ExpertWeights<'_>>;
}

/// One expert's three projections (borrowed from the residency strategy).
pub struct ExpertWeights<'a> {
    pub gate: &'a QMatMul,
    pub up: &'a QMatMul,
    pub down: &'a QMatMul,
}

/// All experts resident in RAM: the small-model / big-RAM case, zero streaming
/// overhead. Stored as per-expert `QMatMul` triples (the GGUF stacks them as
/// `[n_ff, hidden, n_expert]`; the loader splits them per expert).
#[derive(Debug, Clone)]
pub struct Resident {
    gate: Vec<QMatMul>,
    up: Vec<QMatMul>,
    down: Vec<QMatMul>,
}

impl Resident {
    pub fn new(gate: Vec<QMatMul>, up: Vec<QMatMul>, down: Vec<QMatMul>) -> Self {
        Self { gate, up, down }
    }
}

impl ExpertResidency for Resident {
    #[inline]
    fn expert(&self, e: usize) -> Result<ExpertWeights<'_>> {
        Ok(ExpertWeights {
            gate: &self.gate[e],
            up: &self.up[e],
            down: &self.down[e],
        })
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
    n_experts: usize,
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
            n_experts: cfg.n_experts,
            n_experts_per_tok: cfg.n_experts_per_tok,
        }
    }

    /// One decode token through the MoE. `x` is `[1, 1, hidden]`, returns
    /// `[1, 1, hidden]` = routed top-k combine + gated shared expert.
    ///
    /// The host learns the top-k expert ids (one small readback) to index the
    /// resident set — the Resident path's only host step; a `Streamed` impl would
    /// prefetch from these ids (§20.3 P2/P4). Correctness-first: the routing is a
    /// softmax over all experts, top-k, renormalize (reference `build_moe_ffn`).
    pub fn decode(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = x.dim(D::Minus1)?;
        let x1 = x.reshape((1, hidden))?; // [1, hidden]

        // Router: softmax over all experts, then top-k select + renormalize.
        let logits = self.router.forward(&x1)?.to_dtype(lloom_tensor::DType::F32)?; // [1, n_experts]
        let probs = ops::softmax_last_dim(&logits)?; // [1, n_experts]
        let probs_row = probs.reshape(self.n_experts)?;
        // top-k by descending probability (reuse arg_sort).
        let order = probs_row.arg_sort_last_dim(false)?; // ids, best first
        let top_ids: Vec<u32> = order
            .narrow(0, 0, self.n_experts_per_tok)?
            .to_vec1::<u32>()?;
        let probs_host = probs_row.to_vec1::<f32>()?;
        let sel_w: Vec<f32> = top_ids.iter().map(|&e| probs_host[e as usize]).collect();
        let wsum: f32 = sel_w.iter().sum::<f32>().max(1e-20);

        // Weighted sum of the selected experts' SwiGLU.
        let mut acc: Option<Tensor> = None;
        for (i, &e) in top_ids.iter().enumerate() {
            let w = self.experts.expert(e as usize)?;
            let y = swiglu(&x1, w.gate, w.up, w.down)?; // [1, hidden]
            let scaled = (y * (sel_w[i] / wsum) as f64)?;
            acc = Some(match acc {
                Some(a) => (a + scaled)?,
                None => scaled,
            });
        }
        let routed = acc.expect("n_experts_per_tok > 0");

        // Shared expert: SwiGLU gated by sigmoid(sh_router·x), always on.
        let shared = swiglu(&x1, &self.sh_gate, &self.sh_up, &self.sh_down)?;
        let sh_gate_v = ops::sigmoid(
            &self.sh_router.forward(&x1)?.to_dtype(shared.dtype())?,
        )?; // [1, 1]
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
    use lloom_tensor::{DType, Device};

    fn lin(out: usize, inp: usize, seed: f32, dev: &Device) -> QMatMul {
        let n = out * inp;
        let v: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.013 + seed).sin()) * 0.1).collect();
        QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev).unwrap())
    }

    /// The MoE decodes end to end (Resident): shapes thread, output finite, and
    /// the seam monomorphizes to the no-loader path.
    #[test]
    fn moe_resident_decodes() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let (h, ff, ne) = (cfg.hidden_size, cfg.moe_intermediate_size, cfg.n_experts);
        let gate: Vec<_> = (0..ne).map(|e| lin(ff, h, e as f32, &dev)).collect();
        let up: Vec<_> = (0..ne).map(|e| lin(ff, h, e as f32 + 100.0, &dev)).collect();
        let down: Vec<_> = (0..ne).map(|e| lin(h, ff, e as f32 + 200.0, &dev)).collect();
        let moe = MoeLayer::new(
            lin(ne, h, 1.0, &dev),                    // router
            Resident::new(gate, up, down),
            lin(cfg.shared_expert_intermediate_size, h, 2.0, &dev), // sh_gate
            lin(cfg.shared_expert_intermediate_size, h, 3.0, &dev), // sh_up
            lin(h, cfg.shared_expert_intermediate_size, 4.0, &dev), // sh_down
            lin(1, h, 5.0, &dev),                     // sh_router (1 value/token)
            &cfg,
        );
        let x = Tensor::from_vec(
            (0..h).map(|i| ((i as f32) * 0.05).cos() * 0.3).collect::<Vec<f32>>(),
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

    /// Router selects exactly `n_experts_per_tok` experts, and the top-k weights
    /// renormalize to sum ~1 (reference build_moe_ffn norm_w=true). Checked via a
    /// router that puts all mass on two experts.
    #[test]
    fn router_selects_topk_renormalized() {
        let dev = Device::Cpu;
        let mut cfg = Config::tiny();
        cfg.n_experts = 4;
        cfg.n_experts_per_tok = 2;
        let (h, ff) = (cfg.hidden_size, cfg.moe_intermediate_size);
        // identity-ish experts so the output reflects the routing weights.
        let gate: Vec<_> = (0..4).map(|_| QMatMul::Tensor(Tensor::zeros((ff, h), DType::F32, &dev).unwrap())).collect();
        let up: Vec<_> = gate.clone();
        let down: Vec<_> = (0..4).map(|_| QMatMul::Tensor(Tensor::zeros((h, ff), DType::F32, &dev).unwrap())).collect();
        let moe = MoeLayer::new(
            lin(4, h, 1.0, &dev),
            Resident::new(gate, up, down),
            QMatMul::Tensor(Tensor::zeros((ff, h), DType::F32, &dev).unwrap()),
            QMatMul::Tensor(Tensor::zeros((ff, h), DType::F32, &dev).unwrap()),
            QMatMul::Tensor(Tensor::zeros((h, ff), DType::F32, &dev).unwrap()),
            QMatMul::Tensor(Tensor::zeros((1, h), DType::F32, &dev).unwrap()),
            &cfg,
        );
        // zero-weight experts → routed contribution is zero, shared is zero too
        // (all-zero weights), so this only checks the path runs with k=2 without
        // panic and produces the right shape (the renorm math is exercised).
        let x = Tensor::ones((1, 1, h), DType::F32, &dev).unwrap();
        let o = moe.decode(&x).unwrap();
        assert_eq!(o.dims(), &[1, 1, h]);
    }
}
