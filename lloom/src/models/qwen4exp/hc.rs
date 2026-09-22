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
    /// `n = hc_count`.
    n: usize,
    /// `d = hidden_size`.
    d: usize,
    /// RMS epsilon.
    eps: f32,
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
            n,
            d,
            eps,
        })
    }

    /// Prologue: collapse the `n` streams of `h [b, seq, n*d]` to the sub-block's
    /// input `[b, seq, d]`, and predict the injection tensor `[b, seq, n]` the
    /// matching [`Self::combine`] scatters with.
    ///
    /// Reference (build_hc_mix): `xn = rms_norm(h)·γ`; `lo = silu(down·xn / n)`;
    /// `gate = sigmoid(up·lo)`; `mixed = mean_streams(xn · gate)`;
    /// `inject = inject·xn`.
    pub fn mix(&self, h: &Tensor) -> Result<(Tensor, Tensor)> {
        let (b, seq, _) = h.dims3()?;
        // The grouped norm as one launch on Metal (`kernels::HcNorm`); the
        // composed pair (`rms_norm` per stream, `broadcast_mul` γ) elsewhere.
        let xn = match super::kernels::hc_norm(h, &self.norm_w, self.n, self.d, self.eps) {
            Some(xn) => xn?,
            None => grouped_rms_norm(h, &self.unit_d, &self.norm_w, self.n, self.d, self.eps)?,
        }; // [b, seq, n*d]

        // Low-rank gate prediction on the normed carry. On Metal at decode
        // row counts the two projections take the mixed arm with the scale
        // and the activation as the store's epilogue (`QMatMul::gemv_mixed`:
        // one launch each, f32 in and out); elsewhere the composed chain.
        let inv_n = 1.0 / self.n as f64;
        let lo = match self.down.gemv_mixed(&xn, MixedEpilogue::Silu, inv_n as f32) {
            Some(lo) => lo?,
            None => ops::silu(&(self.down.forward(&xn)? * inv_n)?)?,
        }; // [b, seq, hc_lr]
        let gate = match self.up.gemv_mixed(&lo, MixedEpilogue::Sigmoid, 1.0) {
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

        let inject = match &self.inject {
            Some(inject) => inject.forward(&xn)?, // [b, seq, n]
            None => Tensor::zeros((b, seq, self.n), xn.dtype(), xn.device())?,
        };
        Ok((mixed, inject))
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
