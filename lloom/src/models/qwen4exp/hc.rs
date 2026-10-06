//! HyperConnection (Gated Residual) — the prologue/epilogue pair that replaces
//! RMSNorm+residual on every Qwen4-exp layer.
//!
//! The hidden is carried as `n = hc_count` parallel streams `H ∈ [b, seq, n*d]`.
//! Per sub-block (token-mixer, then MoE), a `mix` collapses the streams to the
//! sub-block's `[b, seq, d]` input and predicts an injection weight, the
//! sub-block runs once, and a `combine` scatters its output back across the
//! streams. `mix`/`combine` are pure transforms on the carried state — the same
//! shape LFM2's `input_layernorm` + residual-add occupy, n-wide — so the layer
//! holds two `HyperConnection`s (`hc_attn`, `hc_ffn`) as fields and threads `H`
//! through them explicitly, rather than a wrapper owning the inner block.
//!
//! Ported from the reference graph (llama.cpp `qwen4exp.cpp` build_hc_mix /
//! build_hc_combine); the math is mirrored, the layout is candle row-major.

use crate::nn::ops;
use lloom_tensor::quantized::gemv_mixed::MixedEpilogue;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Module, Result, Tensor};

/// One HyperConnection module: the low-rank dynamic prediction weights plus the
/// grouped norm, for one bracket (attn or ffn) of one layer.
#[derive(Debug, Clone)]
pub struct HyperConnection {
    /// The grouped-RMSNorm γ, `[n*d]`. The reference normalizes PER STREAM (over
    /// `d`) and then scales all streams by this `[n*d]` γ (converter folded to
    /// `1+w`) — so it is kept as a raw tensor and applied after a per-stream
    /// normalize, not as a whole-`n*d` `RmsNorm`.
    norm_w: Tensor,
    /// Unit alpha `[d]` for the per-stream RMS (normalize only; γ is applied
    /// separately over the full `n*d`). Cached to avoid rebuilding per token.
    unit_d: Tensor,
    /// `n*d -> hc_lowrank` (DHC prediction down-projection).
    down: QMatMul,
    /// `hc_lowrank -> n*d` (up-projection; feeds the collapse gate).
    up: QMatMul,
    /// `n*d -> n` (the per-stream injection weight predictor). `None` on the
    /// head mixer (`output_hc_*` has no inject: its collapse feeds `lm_head`
    /// and nothing is scattered back).
    inject: Option<QMatMul>,
    /// Whether the down and up projections may carry their activation at
    /// the store (`QMatMul::gemv_mixed`'s epilogue): both on unless a
    /// derived plan says otherwise (`qwen4exp.fusion = derived`).
    down_epilogue: bool,
    up_epilogue: bool,
    /// Whether the grouped norm may run inside its consumers at a one-row
    /// step ([`Self::mix_parts_norm_gemv`]): `QWEN4EXP_HC_NORM_GEMV` at
    /// construction. The products keep their epilogue switches under it.
    norm_gemv: bool,
    /// `n = hc_count`.
    n: usize,
    /// `d = hidden_size`.
    d: usize,
    /// RMS epsilon.
    eps: f32,
}

/// `QWEN4EXP_HC_NORM_GEMV`: [`HyperConnection::mix_parts_norm_gemv`]. Read once.
fn hc_norm_gemv_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.hc_norm_gemv)
}

/// `QWEN4EXP_ABLATE` names `hcnorm`: a one-row step skips the HC brackets'
/// grouped norm (an instrument pricing a fusion's upper bound). Read once.
fn ablate_hc_norm() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        crate::tuning::tuning()
            .qwen4exp
            .ablate
            .split(',')
            .any(|s| s.trim() == "hcnorm")
    })
}

/// Grouped RMSNorm over an n-stream carry `x [b, seq, n·d]`: normalize PER
/// STREAM (over `d`, with the unit alpha `unit_d [d]`), then scale all streams
/// by the `[n·d]` γ (the converter folds each γ to `1 + w`). The HC brackets and
/// the PLE's three norms share this one definition.
pub(crate) fn grouped_rms_norm(
    x: &Tensor,
    unit_d: &Tensor,
    gamma: &Tensor,
    n: usize,
    d: usize,
    eps: f32,
) -> Result<Tensor> {
    let (b, seq, _) = x.dims3()?;
    let x_s = x.reshape((b, seq, n, d))?;
    let xn = ops::rms_norm(&x_s, unit_d, eps)?;
    xn.reshape((b, seq, n * d))?.broadcast_mul(gamma)
}

impl HyperConnection {
    pub fn new(
        norm_w: Tensor,
        down: QMatMul,
        up: QMatMul,
        inject: Option<QMatMul>,
        n: usize,
        d: usize,
        eps: f32,
    ) -> Result<Self> {
        let unit_d = Tensor::ones(d, norm_w.dtype(), norm_w.device())?;
        Ok(Self {
            norm_w,
            unit_d,
            down,
            up,
            inject,
            down_epilogue: true,
            up_epilogue: true,
            norm_gemv: hc_norm_gemv_on(),
            n,
            d,
            eps,
        })
    }

    /// Whether the down and up projections may carry their activations at
    /// the store (a derived plan's per-bracket verdict).
    pub fn set_epilogues(&mut self, down: bool, up: bool) {
        self.down_epilogue = down;
        self.up_epilogue = up;
    }

    pub fn epilogues(&self) -> (bool, bool) {
        (self.down_epilogue, self.up_epilogue)
    }

    /// Whether the grouped norm may run inside its consumers (the
    /// `QWEN4EXP_HC_NORM_GEMV` arm, per bracket).
    pub fn set_norm_gemv(&mut self, on: bool) {
        self.norm_gemv = on;
    }

    pub fn norm_gemv(&self) -> bool {
        self.norm_gemv
    }

    /// Whether the arm's products serve a one-row step at this bracket's
    /// shapes: `down` (and `inject`, when wanted) the f16-weight mixed GEMV
    /// at the tile the grouped-norm kernel is instantiated for. The shape
    /// half of the forward's applicability check, which the walk reads.
    #[cfg(not(target_arch = "wasm32"))]
    fn norm_gemv_serves(&self, want_inject: bool) -> bool {
        use crate::optimizer::{walk::qmat_operand, Core, Dim};
        let serves = |q: &QMatMul| match qmat_operand(q) {
            (Core::Mixed, shape, _) => match shape.0.as_slice() {
                [Dim::Const(n_out), Dim::Const(k)] => {
                    lloom_tensor::gemv_tile::gemv_mixed_gnorm_serves(*n_out, *k, self.n)
                }
                _ => false,
            },
            _ => false,
        };
        self.norm_w.dtype() == lloom_tensor::DType::F32
            && serves(&self.down)
            && (!want_inject || self.inject.as_ref().is_none_or(serves))
    }

    /// Prologue: collapse the `n` streams of `h [b, seq, n*d]` to the sub-block's
    /// input `[b, seq, d]`, and predict the injection tensor `[b, seq, n]` the
    /// matching [`Self::combine`] scatters with.
    ///
    /// Reference (build_hc_mix): `xn = rms_norm(h)·γ`; `lo = silu(down·xn / n)`;
    /// `gate = sigmoid(up·lo)`; `mixed = mean_streams(xn · gate)`;
    /// `inject = inject·xn`.
    pub fn mix(&self, h: &Tensor) -> Result<(Tensor, Tensor)> {
        let (mixed, inject) = self.mix_parts(h, true)?;
        Ok((mixed, inject.expect("the injection was asked for")))
    }

    /// The collapsed input alone: [`Self::mix`] for a caller with no
    /// [`Self::combine`] to feed (the head, a tap), which makes no injection
    /// (the composed arm's zeros when there is no `inject`, a gemv when there
    /// is) only to drop it.
    pub fn collapse(&self, h: &Tensor) -> Result<Tensor> {
        Ok(self.mix_parts(h, false)?.0)
    }

    fn mix_parts(&self, h: &Tensor, want_inject: bool) -> Result<(Tensor, Option<Tensor>)> {
        let (b, seq, _) = h.dims3()?;
        if b * seq == 1 && self.norm_gemv {
            if let Some(out) = self.mix_parts_norm_gemv(h, want_inject)? {
                return Ok(out);
            }
        }
        // The whole prologue as two launches on Metal at a decode's rows
        // (`kernels::hc_mix_fused`: norm + down·SiLU + inject, then
        // up·sigmoid + collapse), opt-in: measured slower than the five
        // launches below, whose gemvs run at bandwidth (P29 (3)).
        if let Some(out) = super::kernels::hc_mix_fused(
            h,
            &self.norm_w,
            &self.down,
            &self.up,
            self.inject.as_ref(),
            self.n,
            self.d,
            self.eps,
        ) {
            return out.map(|(m, i)| (m, Some(i)));
        }
        // The grouped norm as one launch on Metal (`kernels::HcNorm`); the
        // composed pair (`rms_norm` per stream, `broadcast_mul` γ) elsewhere.
        // Under the `hcnorm` ablation at a one-row step the carry is passed
        // through instead (an instrument: the output is wrong, every other
        // launch is unchanged).
        let xn = if b * seq == 1 && ablate_hc_norm() {
            h.clone()
        } else {
            match super::kernels::hc_norm(h, &self.norm_w, self.n, self.d, self.eps) {
                Some(xn) => xn?,
                None => grouped_rms_norm(h, &self.unit_d, &self.norm_w, self.n, self.d, self.eps)?,
            }
        }; // [b, seq, n*d]

        // Low-rank gate prediction on the normed carry. On Metal at decode
        // row counts the two projections take the mixed arm with the scale
        // and the activation as the store's epilogue (`QMatMul::gemv_mixed`:
        // one launch each, f32 in and out); elsewhere the composed chain.
        let inv_n = 1.0 / self.n as f64;
        let lo = match self
            .down_epilogue
            .then(|| self.down.gemv_mixed(&xn, MixedEpilogue::Silu, inv_n as f32))
            .flatten()
        {
            Some(lo) => lo?,
            None => ops::silu(&(self.down.forward(&xn)? * inv_n)?)?,
        }; // [b, seq, hc_lr]
        let gate = match self
            .up_epilogue
            .then(|| self.up.gemv_mixed(&lo, MixedEpilogue::Sigmoid, 1.0))
            .flatten()
        {
            Some(g) => g?,
            None => ops::sigmoid(&self.up.forward(&lo)?)?,
        }; // [b, seq, n*d]

        // Gate the streams, then collapse them by their mean: one launch on
        // Metal (`kernels::HcCollapse`), the composed three elsewhere.
        let mixed = match super::kernels::hc_collapse(&xn, &gate, self.n, self.d) {
            Some(m) => m?,
            None => {
                let gated = (xn.clone() * &gate)?; // [b, seq, n*d]
                let gated = gated.reshape((b, seq, self.n, self.d))?;
                let mixed = gated.sum(2)?; // [b, seq, d]
                (mixed * (1.0 / self.n as f64))?
            }
        };

        if !want_inject {
            return Ok((mixed, None));
        }
        let inject = match &self.inject {
            Some(inject) => inject.forward(&xn)?, // [b, seq, n]
            None => Tensor::zeros((b, seq, self.n), xn.dtype(), xn.device())?,
        };
        Ok((mixed, Some(inject)))
    }

    /// [`Self::mix_parts`] with the grouped norm computed inside its consumers
    /// (`QWEN4EXP_HC_NORM_GEMV`, `docs/tuning.md` §13.5u): `down` and `inject`
    /// normalize `h` in their own launches (`QMatMul::gemv_mixed_gnorm`),
    /// `down` also writing the `n` stream scales, and the collapse forms
    /// `xn` from `h`, γ and those scales — `hc_norm_f32`'s launch and its
    /// `[n·d]` round trip removed. `Ok(None)` when any of the three would not
    /// take its fused form: decided before anything is dispatched, so the
    /// bracket is fused whole or composed whole.
    fn mix_parts_norm_gemv(
        &self,
        h: &Tensor,
        want_inject: bool,
    ) -> Result<Option<(Tensor, Option<Tensor>)>> {
        let (b, seq, _) = h.dims3()?;
        let inject_ok = !want_inject
            || self.inject.as_ref().is_none_or(|w| {
                w.gemv_mixed_gnorm_applies(h, &self.norm_w, self.n, MixedEpilogue::None, false)
            });
        // The products' epilogues are the bracket's switches, as on the
        // composed path: off, the activation is its own launch.
        let down_epi = if self.down_epilogue {
            MixedEpilogue::Silu
        } else {
            MixedEpilogue::None
        };
        if !(self
            .down
            .gemv_mixed_gnorm_applies(h, &self.norm_w, self.n, down_epi, true)
            && inject_ok
            && super::kernels::hc_collapse_scaled_applies(h, &self.norm_w))
        {
            return Ok(None);
        }
        let checked = || {
            lloom_tensor::Error::Msg(
                "hc_norm_gemv: an applicability check passed and the op declined".into(),
            )
        };
        // `down` with the norm, `silu(lo / n)` as the composed path's scale,
        // and the stream scales after its `hc_lowrank` outputs.
        let inv_n = 1.0 / self.n as f64;
        let lo_s = self
            .down
            .gemv_mixed_gnorm(
                h,
                &self.norm_w,
                self.eps,
                self.n,
                down_epi,
                if self.down_epilogue {
                    inv_n as f32
                } else {
                    1.0
                },
                true,
            )
            .ok_or_else(checked)??;
        let lr = lo_s.dim(lloom_tensor::D::Minus1)? - self.n;
        let lo = lo_s.narrow(lloom_tensor::D::Minus1, 0, lr)?;
        let scales = lo_s.narrow(lloom_tensor::D::Minus1, lr, self.n)?;
        let lo = if self.down_epilogue {
            lo
        } else {
            ops::silu(&(lo * inv_n)?)?
        };
        let gate = match self
            .up_epilogue
            .then(|| self.up.gemv_mixed(&lo, MixedEpilogue::Sigmoid, 1.0))
            .flatten()
        {
            Some(g) => g?,
            None => ops::sigmoid(&self.up.forward(&lo)?)?,
        };
        let mixed =
            super::kernels::hc_collapse_scaled(h, &self.norm_w, &scales, &gate, self.n, self.d)
                .ok_or_else(checked)??;
        if !want_inject {
            return Ok(Some((mixed, None)));
        }
        let inject = match &self.inject {
            Some(w) => w
                .gemv_mixed_gnorm(
                    h,
                    &self.norm_w,
                    self.eps,
                    self.n,
                    MixedEpilogue::None,
                    1.0,
                    false,
                )
                .ok_or_else(checked)??,
            None => Tensor::zeros((b, seq, self.n), h.dtype(), h.device())?,
        };
        Ok(Some((mixed, Some(inject))))
    }

    /// [`Self::combine`]'s decode step for the walk: one `hc_combine` over the
    /// carry, the block's output and the injection.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn visit_combine(
        &self,
        w: &mut crate::optimizer::Walker,
        carry: crate::optimizer::NodeId,
        block: crate::optimizer::NodeId,
        inject: crate::optimizer::NodeId,
    ) -> crate::optimizer::NodeId {
        use crate::optimizer::Shape;
        let nd = self.n * self.d;
        w.opaque(
            "hc_combine",
            &[carry, block, inject],
            Shape::of(&[1, 1, nd]),
            lloom_tensor::DType::F32,
        )
    }

    /// Epilogue: scatter `block_out [b, seq, d]` back across the `n` streams of
    /// the carry `h [b, seq, n*d]`, weighted by the prediction from `mix`.
    ///
    /// Reference (build_hc_combine): `w = 2·sigmoid(inject / n)` (centred on 1,
    /// so a zero injection is a plain residual add); `H' = H + broadcast(block_out) · w`.
    pub fn combine(&self, h: &Tensor, block_out: &Tensor, inject: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = h.dims3()?;
        // One launch on Metal (`kernels::HcCombine`); the composed five elsewhere.
        if let Some(out) = super::kernels::hc_combine(h, block_out, inject, self.n, self.d) {
            return out;
        }
        // Per-stream scatter weight, centred on 1.
        let w = ops::sigmoid(&(inject * (1.0 / self.n as f64))?)?; // [b, seq, n]
        let w = (w * 2.0)?;
        let w = w.reshape((b, seq, self.n, 1))?; // [b, seq, n, 1]

        // Broadcast the sub-block output across the n streams.
        let bo = block_out.reshape((b, seq, 1, self.d))?; // [b, seq, 1, d]
        let scattered = bo.broadcast_mul(&w)?; // [b, seq, n, d]
        let scattered = scattered.reshape((b, seq, self.n * self.d))?;

        h + scattered
    }
}

/// One bracket's prologue at a decode step ([`HyperConnection::mix`] when
/// the injection is wanted, [`HyperConnection::collapse`] when not): the
/// grouped norm; `silu(down·xn / n)` and `sigmoid(up·lo)` unfused — the
/// products' epilogues are what a derivation over the walk decides; the
/// collapse; the injection's product. `In` is the carry, the bracket's
/// checkpoint name (`hc_attn`, `output_hc`) and whether the injection is
/// wanted; `Out` the collapsed input and the injection.
#[cfg(not(target_arch = "wasm32"))]
impl crate::optimizer::Visit for HyperConnection {
    type In = (crate::optimizer::NodeId, &'static str, bool);
    type Out = (crate::optimizer::NodeId, Option<crate::optimizer::NodeId>);
    fn visit(
        &self,
        w: &mut crate::optimizer::Walker,
        (carry, name, want_inject): Self::In,
    ) -> Self::Out {
        use crate::optimizer::Shape;
        let (n, d) = (self.n, self.d);
        let f32 = lloom_tensor::DType::F32;
        let gamma = w.weight(
            &format!("{name}_norm.weight"),
            Shape::of(&[n * d]),
            self.norm_w.dtype(),
        );
        if self.norm_gemv && self.norm_gemv_serves(want_inject) {
            return self.visit_norm_gemv(w, carry, gamma, name, want_inject);
        }
        let xn = w.opaque("hc_norm", &[carry, gamma], Shape::of(&[1, 1, n * d]), f32);
        let lo = w.project_qmat(&format!("{name}_down.weight"), &self.down, xn);
        let lo = w.scale(lo, (1.0 / n as f64) as f32);
        let lo = w.silu(lo);
        let gate = w.project_qmat(&format!("{name}_up.weight"), &self.up, lo);
        let gate = w.sigmoid(gate);
        let mixed = w.opaque("hc_collapse", &[xn, gate], Shape::of(&[1, 1, d]), f32);
        let inject = want_inject.then(|| match &self.inject {
            Some(q) => w.project_qmat(&format!("{name}_inject.weight"), q, xn),
            // `Tensor::zeros`: a fresh zero buffer, no kernel.
            None => w.input(&format!("{name}.zero_inject"), Shape::of(&[1, 1, n]), f32),
        });
        (mixed, inject)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl HyperConnection {
    /// The bracket under the grouped-norm arm, as it dispatches: `down` and
    /// `inject` normalizing the carry in their own launches (`down` writing
    /// the stream scales after its outputs), the collapse forming the
    /// normalized carry from the scales — no `hc_norm`. The products'
    /// epilogues follow the bracket's switches. The derivation cannot
    /// produce this composition (`optimizer::fragments::MIXED_GNORM_REFUSAL`),
    /// so it is walked as the kernels it runs.
    fn visit_norm_gemv(
        &self,
        w: &mut crate::optimizer::Walker,
        carry: crate::optimizer::NodeId,
        gamma: crate::optimizer::NodeId,
        name: &'static str,
        want_inject: bool,
    ) -> (crate::optimizer::NodeId, Option<crate::optimizer::NodeId>) {
        use crate::optimizer::{walk::qmat_operand, Dim, Shape};
        let (n, d) = (self.n, self.d);
        let f32 = lloom_tensor::DType::F32;
        let weight = |w: &mut crate::optimizer::Walker, part: &str, q: &QMatMul| {
            let (_, shape, dtype) = qmat_operand(q);
            let rows = match shape.0.first() {
                Some(Dim::Const(r)) => *r,
                _ => unreachable!("a bracket's product has a constant width"),
            };
            (
                w.weight(&format!("{name}_{part}.weight"), shape, dtype),
                rows,
            )
        };
        let (wd, lr) = weight(w, "down", &self.down);
        let family = if self.down_epilogue {
            "gemv_mixed_gnorm_silu"
        } else {
            "gemv_mixed_gnorm"
        };
        let lo_s = w.opaque(family, &[wd, carry, gamma], Shape::of(&[1, 1, lr + n]), f32);
        let lo = w.view("narrow", lo_s, Shape::of(&[1, 1, lr]));
        let scales = w.view("narrow", lo_s, Shape::of(&[1, 1, n]));
        let lo = if self.down_epilogue {
            lo
        } else {
            let s = w.scale(lo, (1.0 / n as f64) as f32);
            w.silu(s)
        };
        let gate = w.project_qmat(&format!("{name}_up.weight"), &self.up, lo);
        let gate = w.sigmoid(gate);
        let mixed = w.opaque(
            "hc_collapse_scaled",
            &[carry, gamma, scales, gate],
            Shape::of(&[1, 1, d]),
            f32,
        );
        let inject = want_inject.then(|| match &self.inject {
            Some(q) => {
                let (wi, _) = weight(w, "inject", q);
                w.opaque(
                    "gemv_mixed_gnorm",
                    &[wi, carry, gamma],
                    Shape::of(&[1, 1, n]),
                    f32,
                )
            }
            None => w.input(&format!("{name}.zero_inject"), Shape::of(&[1, 1, n]), f32),
        });
        (mixed, inject)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::{DType, Device};

    /// The three fused launches (`hc_norm`, `hc_collapse`, `hc_combine`)
    /// equal the composed bracket, at the real geometry (n 4, d 2560) on a
    /// decode row and a 5-row chunk, against the CPU path.
    #[cfg(feature = "metal")]
    #[test]
    fn fused_bracket_matches_composed() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let (n, d, lr) = (4usize, 2560usize, 320usize);
        let build = |dev: &Device| {
            let lin = |out: usize, inp: usize, seed: f32| {
                QMatMul::Tensor(
                    Tensor::from_vec(
                        (0..out * inp)
                            .map(|i| ((i as f32 * 0.0117 + seed).sin()) * 0.03)
                            .collect::<Vec<_>>(),
                        (out, inp),
                        dev,
                    )
                    .unwrap(),
                )
            };
            let gamma = Tensor::from_vec(
                (0..n * d)
                    .map(|i| 1.0 + 0.1 * ((i as f32) * 0.3).sin())
                    .collect::<Vec<f32>>(),
                n * d,
                dev,
            )
            .unwrap();
            HyperConnection::new(
                gamma,
                lin(lr, n * d, 1.0),
                lin(n * d, lr, 2.0),
                Some(lin(n, n * d, 3.0)),
                n,
                d,
                1e-6,
            )
            .unwrap()
        };
        let (hc_cpu, hc_gpu) = (build(&cpu), build(&gpu));
        for &seq in &[1usize, 5] {
            let h = Tensor::from_vec(
                (0..seq * n * d)
                    .map(|i| ((i as f32 * 0.021).cos()) * 0.8)
                    .collect::<Vec<f32>>(),
                (1, seq, n * d),
                &cpu,
            )
            .unwrap();
            let bo = Tensor::from_vec(
                (0..seq * d)
                    .map(|i| ((i as f32 * 0.031).sin()) * 0.5)
                    .collect::<Vec<f32>>(),
                (1, seq, d),
                &cpu,
            )
            .unwrap();
            let (m_cpu, i_cpu) = hc_cpu.mix(&h).unwrap();
            let (m_gpu, i_gpu) = hc_gpu.mix(&h.to_device(&gpu).unwrap()).unwrap();
            let c_cpu = hc_cpu.combine(&h, &bo, &i_cpu).unwrap();
            let c_gpu = hc_gpu
                .combine(
                    &h.to_device(&gpu).unwrap(),
                    &bo.to_device(&gpu).unwrap(),
                    &i_gpu,
                )
                .unwrap();
            for (name, a, b) in [
                ("mixed", &m_cpu, &m_gpu),
                ("inject", &i_cpu, &i_gpu),
                ("combine", &c_cpu, &c_gpu),
            ] {
                let x: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
                let y: Vec<f32> = b
                    .to_device(&cpu)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1()
                    .unwrap();
                assert_eq!(x.len(), y.len(), "{name} seq {seq}");
                let scale = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
                let diff = x
                    .iter()
                    .zip(&y)
                    .map(|(p, q)| (p - q).abs())
                    .fold(0f32, f32::max);
                assert!(
                    diff <= 3e-5 * scale,
                    "{name} seq {seq}: {diff} at scale {scale}"
                );
                assert!(scale > 1e-3, "{name} seq {seq}: degenerate");
            }
        }
    }

    /// The fused two-launch prologue (`hc_mix_a` / `hc_mix_b`, f16 weights on
    /// Metal) against the CPU's composed chain over the same f16-rounded
    /// weights, with and without an inject weight, at 1 and 3 rows — and the
    /// composed Metal chain (`QWEN4EXP_FUSED_OFF` is process-wide, so the
    /// composed arm here is the CPU's). The pair is an arm, off by default:
    /// at one row it is called through `kernels::hc_mix_pair` so the test
    /// reaches it (through `mix` it read the composed chain and a mutation of
    /// either kernel survived, #749 P33 (6b)); at three rows the pair
    /// declines (`HC_MIX_MAX_ROWS`) and `mix` is the composed Metal chain.
    #[cfg(feature = "metal")]
    #[test]
    fn fused_prologue_matches_composed_f16() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let (n, d, lr) = (4usize, 2560usize, 320usize);
        let f16_rounded = |out: usize, inp: usize, seed: f32| -> Tensor {
            Tensor::from_vec(
                (0..out * inp)
                    .map(|i| ((i as f32 * 0.0117 + seed).sin()) * 0.03)
                    .collect::<Vec<_>>(),
                (out, inp),
                &cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
        };
        let gamma = Tensor::from_vec(
            (0..n * d)
                .map(|i| 1.0 + 0.1 * ((i as f32) * 0.3).sin())
                .collect::<Vec<f32>>(),
            n * d,
            &cpu,
        )
        .unwrap();
        for with_inject in [true, false] {
            let (wd, wu, wi) = (
                f16_rounded(lr, n * d, 1.0),
                f16_rounded(n * d, lr, 2.0),
                f16_rounded(n, n * d, 3.0),
            );
            let hc_cpu = HyperConnection::new(
                gamma.clone(),
                QMatMul::Tensor(wd.to_dtype(DType::F32).unwrap()),
                QMatMul::Tensor(wu.to_dtype(DType::F32).unwrap()),
                with_inject.then(|| QMatMul::Tensor(wi.to_dtype(DType::F32).unwrap())),
                n,
                d,
                1e-6,
            )
            .unwrap();
            let hc_gpu = HyperConnection::new(
                gamma.to_device(&gpu).unwrap(),
                QMatMul::TensorF16(wd.to_device(&gpu).unwrap()),
                QMatMul::TensorF16(wu.to_device(&gpu).unwrap()),
                with_inject.then(|| QMatMul::TensorF16(wi.to_device(&gpu).unwrap())),
                n,
                d,
                1e-6,
            )
            .unwrap();
            for &seq in &[1usize, 3] {
                let h = Tensor::from_vec(
                    (0..seq * n * d)
                        .map(|i| ((i as f32 * 0.021).cos()) * 0.8)
                        .collect::<Vec<f32>>(),
                    (1, seq, n * d),
                    &cpu,
                )
                .unwrap();
                let (m_cpu, i_cpu) = hc_cpu.mix(&h).unwrap();
                let h_gpu = h.to_device(&gpu).unwrap();
                let pair = || {
                    super::super::kernels::hc_mix_pair(
                        &h_gpu,
                        &hc_gpu.norm_w,
                        &hc_gpu.down,
                        &hc_gpu.up,
                        hc_gpu.inject.as_ref(),
                        n,
                        d,
                        1e-6,
                    )
                };
                let fused = pair();
                assert_eq!(
                    fused.is_some(),
                    seq <= super::super::kernels::HC_MIX_MAX_ROWS,
                    "seq {seq}: the pair must take a decode row and decline more"
                );
                let (m_gpu, i_gpu) = match fused {
                    Some(r) => r.unwrap(),
                    None => hc_gpu.mix(&h_gpu).unwrap(),
                };
                assert_eq!(m_gpu.dims(), &[1, seq, d]);
                assert_eq!(i_gpu.dims(), &[1, seq, n]);
                for (name, a, b, tol) in [
                    ("mixed", &m_cpu, &m_gpu, 3e-5f32),
                    ("inject", &i_cpu, &i_gpu, 3e-5f32),
                ] {
                    let x: Vec<f32> = a.flatten_all().unwrap().to_vec1().unwrap();
                    let y: Vec<f32> = b
                        .to_device(&cpu)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap();
                    assert_eq!(x.len(), y.len(), "{name} seq {seq}");
                    let scale = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
                    let diff = x
                        .iter()
                        .zip(&y)
                        .map(|(p, q)| (p - q).abs())
                        .fold(0f32, f32::max);
                    assert!(
                        diff <= tol * scale,
                        "{name} inject={with_inject} seq {seq}: {diff} at scale {scale}"
                    );
                    if name == "inject" && !with_inject {
                        assert!(y.iter().all(|v| *v == 0.0), "no-inject rows must be zero");
                    } else {
                        assert!(scale > 1e-3, "{name} seq {seq}: degenerate");
                    }
                }
                // Determinism: the same launch twice, bit for bit.
                let (m2, _) = match pair() {
                    Some(r) => r.unwrap(),
                    None => hc_gpu.mix(&h_gpu).unwrap(),
                };
                let a: Vec<f32> = m_gpu.flatten_all().unwrap().to_vec1().unwrap();
                let b2: Vec<f32> = m2.flatten_all().unwrap().to_vec1().unwrap();
                assert_eq!(a, b2, "fused prologue not bit-stable at seq {seq}");
            }
        }
    }

    /// The bracket with its grouped norm inside its consumers
    /// (`mix_parts_norm_gemv`, `QWEN4EXP_HC_NORM_GEMV`) against the composed
    /// chain on the device (`hc_norm_f32`, the mixed GEMVs, `hc_collapse_f32`)
    /// and the CPU's, at the checkpoint's shapes: the per-stream scales are
    /// reduced in `hc_norm_f32`'s tree and the coefficients formed in its
    /// order, so the outputs are the composed chain's, bit for bit (asserted),
    /// and within the CPU tolerance of the CPU's chain. One row only (the
    /// scales' form); three rows decline. Bit-stable run to run.
    #[cfg(feature = "metal")]
    #[test]
    fn norm_gemv_bracket_matches_composed() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let (n, d, lr) = (4usize, 2560usize, 320usize);
        let f16_rounded = |out: usize, inp: usize, seed: f32| -> Tensor {
            Tensor::from_vec(
                (0..out * inp)
                    .map(|i| ((i as f32 * 0.0117 + seed).sin()) * 0.03)
                    .collect::<Vec<_>>(),
                (out, inp),
                &cpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
        };
        let gamma = Tensor::from_vec(
            (0..n * d)
                .map(|i| 1.0 + 0.1 * ((i as f32) * 0.3).sin())
                .collect::<Vec<f32>>(),
            n * d,
            &cpu,
        )
        .unwrap();
        for with_inject in [true, false] {
            let (wd, wu, wi) = (
                f16_rounded(lr, n * d, 1.0),
                f16_rounded(n * d, lr, 2.0),
                f16_rounded(n, n * d, 3.0),
            );
            let hc_cpu = HyperConnection::new(
                gamma.clone(),
                QMatMul::Tensor(wd.to_dtype(DType::F32).unwrap()),
                QMatMul::Tensor(wu.to_dtype(DType::F32).unwrap()),
                with_inject.then(|| QMatMul::Tensor(wi.to_dtype(DType::F32).unwrap())),
                n,
                d,
                1e-6,
            )
            .unwrap();
            let hc_gpu = HyperConnection::new(
                gamma.to_device(&gpu).unwrap(),
                QMatMul::TensorF16(wd.to_device(&gpu).unwrap()),
                QMatMul::TensorF16(wu.to_device(&gpu).unwrap()),
                with_inject.then(|| QMatMul::TensorF16(wi.to_device(&gpu).unwrap())),
                n,
                d,
                1e-6,
            )
            .unwrap();
            for &seq in &[1usize, 3] {
                let h = Tensor::from_vec(
                    (0..seq * n * d)
                        .map(|i| ((i as f32 * 0.021).cos()) * 0.8)
                        .collect::<Vec<f32>>(),
                    (1, seq, n * d),
                    &cpu,
                )
                .unwrap();
                let h_gpu = h.to_device(&gpu).unwrap();
                let fused = hc_gpu.mix_parts_norm_gemv(&h_gpu, true).unwrap();
                if seq != 1 {
                    assert!(
                        fused.is_none(),
                        "seq {seq}: the scales' form is one row; it must decline"
                    );
                    continue;
                }
                let (m_f, i_f) = fused.expect("one row: the fused bracket must take it");
                let i_f = i_f.unwrap();
                // The composed chain on the device (the arm is off in tests).
                let (m_c, i_c) = hc_gpu.mix(&h_gpu).unwrap();
                let (m_cpu, i_cpu) = hc_cpu.mix(&h).unwrap();
                let v = |t: &Tensor| -> Vec<f32> {
                    t.to_device(&cpu)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1()
                        .unwrap()
                };
                for (name, f, c, r) in [
                    ("mixed", v(&m_f), v(&m_c), v(&m_cpu)),
                    ("inject", v(&i_f), v(&i_c), v(&i_cpu)),
                ] {
                    assert_eq!(f.len(), c.len(), "{name}");
                    assert!(
                        f.iter().all(|x| x.is_finite()),
                        "{name}: a non-finite output"
                    );
                    let differ = f
                        .iter()
                        .zip(&c)
                        .filter(|(a, b)| a.to_bits() != b.to_bits())
                        .count();
                    let scale = r.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
                    let d_dev = f
                        .iter()
                        .zip(&c)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    let d_cpu = f
                        .iter()
                        .zip(&r)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0f32, f32::max);
                    println!(
                        "norm_gemv bracket inject={with_inject} {name}: {differ} of {} differ from the composed chain bit-wise, max |d| {d_dev:.3e}; vs CPU {d_cpu:.3e} at scale {scale:.3e}",
                        f.len()
                    );
                    // The scales are `hc_norm_f32`'s own tree and the
                    // coefficients its own product, so the GEMVs and the
                    // collapse see the composed chain's inputs exactly:
                    // measured 0 differing elements, pinned here.
                    assert_eq!(differ, 0, "{name} inject={with_inject}: {differ} elements differ from the device's composed chain (max |d| {d_dev})");
                    assert!(
                        d_cpu <= 3e-5 * scale,
                        "{name} inject={with_inject}: {d_cpu} from the CPU"
                    );
                    if name == "inject" && !with_inject {
                        assert!(f.iter().all(|x| *x == 0.0), "no-inject rows must be zero");
                    } else {
                        assert!(scale > 1e-3, "{name}: degenerate");
                    }
                }
                let (m2, _) = hc_gpu.mix_parts_norm_gemv(&h_gpu, true).unwrap().unwrap();
                assert_eq!(v(&m_f), v(&m2), "the fused bracket is not bit-stable");
            }
        }
    }

    /// Past eight streams the fused prologue must decline: `hc_mix_a` keeps
    /// one scale per stream in a threadgroup array of eight, and a carry of
    /// more streams wrote past it and returned the wrong mix (the shape
    /// campaign found it at `hc_count` 10–48, rel 0.1–0.6 against the CPU).
    /// At 8 the pair runs; above, `mix`'s composed chain does — each against
    /// the CPU's composed chain over the same f16-rounded weights.
    #[cfg(feature = "metal")]
    #[test]
    fn fused_prologue_declines_past_its_stream_scales() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let (d, lr) = (256usize, 64usize);
        let mut bad = Vec::new();
        for n in [8usize, 9, 10, 16] {
            let f16_rounded = |out: usize, inp: usize, seed: f32| -> Tensor {
                Tensor::from_vec(
                    (0..out * inp)
                        .map(|i| ((i as f32 * 0.0117 + seed).sin()) * 0.03)
                        .collect::<Vec<_>>(),
                    (out, inp),
                    &cpu,
                )
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap()
            };
            let gamma = Tensor::from_vec(
                (0..n * d)
                    .map(|i| 1.0 + 0.1 * ((i as f32) * 0.3).sin())
                    .collect::<Vec<f32>>(),
                n * d,
                &cpu,
            )
            .unwrap();
            let (wd, wu, wi) = (
                f16_rounded(lr, n * d, 1.0),
                f16_rounded(n * d, lr, 2.0),
                f16_rounded(n, n * d, 3.0),
            );
            let hc_cpu = HyperConnection::new(
                gamma.clone(),
                QMatMul::Tensor(wd.to_dtype(DType::F32).unwrap()),
                QMatMul::Tensor(wu.to_dtype(DType::F32).unwrap()),
                Some(QMatMul::Tensor(wi.to_dtype(DType::F32).unwrap())),
                n,
                d,
                1e-6,
            )
            .unwrap();
            let hc_gpu = HyperConnection::new(
                gamma.to_device(&gpu).unwrap(),
                QMatMul::TensorF16(wd.to_device(&gpu).unwrap()),
                QMatMul::TensorF16(wu.to_device(&gpu).unwrap()),
                Some(QMatMul::TensorF16(wi.to_device(&gpu).unwrap())),
                n,
                d,
                1e-6,
            )
            .unwrap();
            let h = Tensor::from_vec(
                (0..n * d)
                    .map(|i| ((i as f32 * 0.021).cos()) * 0.8)
                    .collect::<Vec<f32>>(),
                (1, 1, n * d),
                &cpu,
            )
            .unwrap();
            let (m_cpu, _) = hc_cpu.mix(&h).unwrap();
            let h_gpu = h.to_device(&gpu).unwrap();
            let fused = super::super::kernels::hc_mix_pair(
                &h_gpu,
                &hc_gpu.norm_w,
                &hc_gpu.down,
                &hc_gpu.up,
                hc_gpu.inject.as_ref(),
                n,
                d,
                1e-6,
            );
            let took = fused.is_some();
            let (m_gpu, _) = match fused {
                Some(r) => r.unwrap(),
                None => hc_gpu.mix(&h_gpu).unwrap(),
            };
            let x: Vec<f32> = m_cpu.flatten_all().unwrap().to_vec1().unwrap();
            let y: Vec<f32> = m_gpu
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1()
                .unwrap();
            let scale = x.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-3);
            let diff = x
                .iter()
                .zip(&y)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            eprintln!("n {n}: fused pair taken {took}, |gpu - cpu| {diff:e} at scale {scale:e}");
            if diff > 3e-5 * scale {
                bad.push(format!(
                    "n {n} (pair taken: {took}): {diff:e} at scale {scale:e}"
                ));
            }
            if took != (n <= super::super::kernels::HC_MIX_MAX_N) {
                bad.push(format!(
                    "n {n}: pair taken {took}; it takes up to HC_MIX_MAX_N streams"
                ));
            }
        }
        assert!(bad.is_empty(), "{bad:#?}");
    }

    /// `cargo test -p lloom --release --features metal --lib hc::tests::prologue_bench -- --ignored --nocapture`
    /// (under `lloom-arb measure`): µs per call of the composed Metal
    /// prologue (norm, down·SiLU, up·sigmoid, collapse, inject — five
    /// launches) against the fused pair, and each fused launch alone, at the
    /// real geometry and a decode's rows.
    #[cfg(feature = "metal")]
    #[test]
    #[ignore]
    fn prologue_bench() {
        use super::super::kernels::{HcMixA, HcMixB};
        let gpu = Device::new_metal(0).unwrap();
        let (n, d, lr) = (4usize, 2560usize, 320usize);
        let w = |out: usize, inp: usize, seed: f32| -> Tensor {
            Tensor::from_vec(
                (0..out * inp)
                    .map(|i| ((i as f32 * 0.0117 + seed).sin()) * 0.03)
                    .collect::<Vec<_>>(),
                (out, inp),
                &gpu,
            )
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
        };
        let gamma = Tensor::ones(n * d, DType::F32, &gpu).unwrap();
        // SETS distinct brackets cycled in turn (SETS × 13 MB, past the
        // system cache): the model streams every bracket's weights from
        // memory once a token, and a bench over one bracket measures the
        // cache instead (12.7 µs cached against ~40 in the model, P29).
        const SETS: usize = 32;
        let brackets: Vec<(Tensor, Tensor, Tensor, HyperConnection)> = (0..SETS)
            .map(|k| {
                let seed = k as f32 * 0.7;
                let (wd, wu, wi) = (
                    w(lr, n * d, 1.0 + seed),
                    w(n * d, lr, 2.0 + seed),
                    w(n, n * d, 3.0 + seed),
                );
                let hc = HyperConnection::new(
                    gamma.clone(),
                    QMatMul::TensorF16(wd.clone()),
                    QMatMul::TensorF16(wu.clone()),
                    Some(QMatMul::TensorF16(wi.clone())),
                    n,
                    d,
                    1e-6,
                )
                .unwrap();
                (wd, wu, wi, hc)
            })
            .collect();
        for rows in [1usize, 4] {
            let h = Tensor::randn(0f32, 0.8, (1, rows, n * d), &gpu).unwrap();
            let composed = |k: usize| -> Tensor {
                let hc = &brackets[k].3;
                let xn = super::super::kernels::hc_norm(&h, &gamma, n, d, 1e-6)
                    .unwrap()
                    .unwrap();
                let lo = hc
                    .down
                    .gemv_mixed(&xn, MixedEpilogue::Silu, 0.25)
                    .unwrap()
                    .unwrap();
                let gate = hc
                    .up
                    .gemv_mixed(&lo, MixedEpilogue::Sigmoid, 1.0)
                    .unwrap()
                    .unwrap();
                let mixed = super::super::kernels::hc_collapse(&xn, &gate, n, d)
                    .unwrap()
                    .unwrap();
                let _inject = hc.inject.as_ref().unwrap().forward(&xn).unwrap();
                mixed
            };
            let a_only = |k: usize| -> Tensor {
                h.apply_op1_no_bwd(&HcMixA {
                    gamma: gamma.clone(),
                    w_down: brackets[k].0.clone(),
                    w_inject: Some(brackets[k].2.clone()),
                    n,
                    d,
                    eps: 1e-6,
                    scale: 0.25,
                })
                .unwrap()
            };
            let packed = a_only(0);
            let b_only = |k: usize| -> Tensor {
                packed
                    .apply_op1_no_bwd(&HcMixB {
                        w_up: brackets[k].1.clone(),
                        n,
                        d,
                        lr,
                    })
                    .unwrap()
            };
            let fused = |k: usize| -> Tensor { brackets[k].3.mix(&h).unwrap().0 };
            // Under `CANDLE_METAL_PROFILE=1` the GPU-busy union over the timed
            // loop is the kernels' own time (the wall of a one-launch loop is
            // the host's encode); each loop runs one kind of launch.
            let time = |name: &str, f: &dyn Fn(usize) -> Tensor| {
                use lloom_tensor::kernels::metal::profile;
                for k in 0..SETS {
                    let _ = f(k);
                }
                gpu.synchronize().unwrap();
                let iters = 8 * SETS;
                profile::reset();
                let t = std::time::Instant::now();
                let mut last = None;
                for i in 0..iters {
                    last = Some(f(i % SETS));
                }
                gpu.synchronize().unwrap();
                let us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;
                let gpu_us = if profile::enabled() {
                    profile::snapshot().gpu_busy_union_s * 1e6 / iters as f64
                } else {
                    f64::NAN
                };
                let _ = last;
                println!("HCBENCH rows={rows} {name} wall {us:.1} us/call gpu {gpu_us:.1} us/call ({SETS} brackets cycled)");
            };
            let xn0 = super::super::kernels::hc_norm(&h, &gamma, n, d, 1e-6)
                .unwrap()
                .unwrap();
            let lo0 = brackets[0]
                .3
                .down
                .gemv_mixed(&xn0, MixedEpilogue::Silu, 0.25)
                .unwrap()
                .unwrap();
            let gate0 = brackets[0]
                .3
                .up
                .gemv_mixed(&lo0, MixedEpilogue::Sigmoid, 1.0)
                .unwrap()
                .unwrap();
            time("composed(5 launches)", &composed);
            time("  hc_norm", &|_k| {
                super::super::kernels::hc_norm(&h, &gamma, n, d, 1e-6)
                    .unwrap()
                    .unwrap()
            });
            time("  down gemv+silu", &|k| {
                brackets[k]
                    .3
                    .down
                    .gemv_mixed(&xn0, MixedEpilogue::Silu, 0.25)
                    .unwrap()
                    .unwrap()
            });
            time("  up gemv+sigmoid", &|k| {
                brackets[k]
                    .3
                    .up
                    .gemv_mixed(&lo0, MixedEpilogue::Sigmoid, 1.0)
                    .unwrap()
                    .unwrap()
            });
            time("  hc_collapse", &|_k| {
                super::super::kernels::hc_collapse(&xn0, &gate0, n, d)
                    .unwrap()
                    .unwrap()
            });
            time("  inject", &|k| {
                brackets[k]
                    .3
                    .inject
                    .as_ref()
                    .unwrap()
                    .forward(&xn0)
                    .unwrap()
            });
            time("fused(2 launches)", &fused);
            time("mix_a", &a_only);
            time("mix_b", &b_only);
        }
    }

    fn qmm(rows: usize, cols: usize, dev: &Device) -> QMatMul {
        // A dense (unquantized) weight, the shape a QMatMul::from would hold:
        // QMatMul over an f32 Tensor is a plain matmul, which is what the BF16
        // HC weights are in the real model.
        let w = Tensor::zeros((rows, cols), DType::F32, dev).unwrap();
        QMatMul::Tensor(w)
    }

    /// `combine` with a zero injection is a plain residual add spread over the
    /// streams: `w = 2·sigmoid(0) = 1`, so each stream gains the full block
    /// output. The reference centres the scatter on 1 for exactly this.
    #[test]
    fn combine_zero_injection_is_residual_add() {
        let dev = Device::Cpu;
        let (b, seq, n, d) = (1usize, 1usize, 4usize, 3usize);
        let hc = HyperConnection {
            norm_w: Tensor::ones(n * d, DType::F32, &dev).unwrap(),
            unit_d: Tensor::ones(d, DType::F32, &dev).unwrap(),
            down: qmm(1, 1, &dev),
            up: qmm(1, 1, &dev),
            inject: Some(qmm(1, 1, &dev)),
            down_epilogue: true,
            up_epilogue: true,
            norm_gemv: false,
            n,
            d,
            eps: 1e-6,
        };
        let h = Tensor::ones((b, seq, n * d), DType::F32, &dev).unwrap();
        let block_out = Tensor::full(2.0f32, (b, seq, d), &dev).unwrap();
        let inject = Tensor::zeros((b, seq, n), DType::F32, &dev).unwrap();
        let out = hc.combine(&h, &block_out, &inject).unwrap();
        let v = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // Every element: 1 (residual) + 2 (block_out) * 1 (weight) = 3.
        assert!(v.iter().all(|&x| (x - 3.0).abs() < 1e-5), "{v:?}");
    }

    /// `mix` collapses `n` identical streams of a value back to that value's
    /// row (mean of equal streams), after the per-stream normalize + a unit γ.
    /// With `down`/`up` zero, `gate = sigmoid(0) = 0.5`, so the collapse is
    /// `mean(0.5 · norm(stream))` = `0.5 · norm(stream)` for identical streams.
    #[test]
    fn mix_collapses_identical_streams() {
        let dev = Device::Cpu;
        let (b, seq, n, d) = (1usize, 1usize, 4usize, 3usize);
        let hc = HyperConnection {
            norm_w: Tensor::ones(n * d, DType::F32, &dev).unwrap(),
            unit_d: Tensor::ones(d, DType::F32, &dev).unwrap(),
            down: qmm(2, n * d, &dev), // hc_lr=2, out zeros
            up: qmm(n * d, 2, &dev),
            inject: Some(qmm(n, n * d, &dev)),
            down_epilogue: true,
            up_epilogue: true,
            norm_gemv: false,
            n,
            d,
            eps: 1e-6,
        };
        // n identical streams, each = [1, 2, 3].
        let one_stream = [1.0f32, 2.0, 3.0];
        let mut flat = Vec::new();
        for _ in 0..n {
            flat.extend_from_slice(&one_stream);
        }
        let h = Tensor::from_vec(flat, (b, seq, n * d), &dev).unwrap();
        let (mixed, inject) = hc.mix(&h).unwrap();
        assert_eq!(mixed.dims(), &[b, seq, d]);
        assert_eq!(inject.dims(), &[b, seq, n]);
        // Per-stream RMS-normalize [1,2,3] (unit γ), then ·0.5 gate, mean over
        // identical streams = the same normalized·0.5 row.
        let rms = ((1.0f32 + 4.0 + 9.0) / 3.0 + 1e-6).sqrt();
        let expect: Vec<f32> = one_stream.iter().map(|&x| 0.5 * x / rms).collect();
        let got = mixed.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (g, e) in got.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-4, "got {got:?} expect {expect:?}");
        }
    }
}
