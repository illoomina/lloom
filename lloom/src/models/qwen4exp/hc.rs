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
    /// `n*d -> n` (the per-stream injection weight predictor).
    inject: QMatMul,
    /// `n = hc_count`.
    n: usize,
    /// `d = hidden_size`.
    d: usize,
    /// RMS epsilon.
    eps: f32,
}

impl HyperConnection {
    pub fn new(
        norm_w: Tensor,
        down: QMatMul,
        up: QMatMul,
        inject: QMatMul,
        n: usize,
        d: usize,
        eps: f32,
    ) -> Result<Self> {
        let unit_d = Tensor::ones(d, norm_w.dtype(), norm_w.device())?;
        Ok(Self { norm_w, unit_d, down, up, inject, n, d, eps })
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
        // Grouped RMSNorm: normalize PER STREAM (over d), then scale all streams
        // by the [n*d] γ. Reshape to [b, seq, n, d], RMS over d with a unit
        // alpha, reshape back, multiply by γ.
        let h_s = h.reshape((b, seq, self.n, self.d))?;
        let xn = ops::rms_norm(&h_s, &self.unit_d, self.eps)?;
        let xn = xn.reshape((b, seq, self.n * self.d))?;
        let xn = xn.broadcast_mul(&self.norm_w)?; // [b, seq, n*d]

        // Low-rank gate prediction on the normed carry.
        let lo = self.down.forward(&xn)?; // [b, seq, hc_lr]
        let lo = ops::silu(&(lo * (1.0 / self.n as f64))?)?;
        let gate = ops::sigmoid(&self.up.forward(&lo)?)?; // [b, seq, n*d]

        // Gate the streams, then collapse them by their mean.
        let gated = (xn.clone() * gate)?; // [b, seq, n*d]
        let gated = gated.reshape((b, seq, self.n, self.d))?;
        let mixed = gated.sum(2)?; // [b, seq, d]
        let mixed = (mixed * (1.0 / self.n as f64))?;

        let inject = self.inject.forward(&xn)?; // [b, seq, n]
        Ok((mixed, inject))
    }

    /// Epilogue: scatter `block_out [b, seq, d]` back across the `n` streams of
    /// the carry `h [b, seq, n*d]`, weighted by the prediction from `mix`.
    ///
    /// Reference (build_hc_combine): `w = 2·sigmoid(inject / n)` (centred on 1,
    /// so a zero injection is a plain residual add); `H' = H + broadcast(block_out) · w`.
    pub fn combine(&self, h: &Tensor, block_out: &Tensor, inject: &Tensor) -> Result<Tensor> {
        let (b, seq, _) = h.dims3()?;
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
            inject: qmm(1, 1, &dev),
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
            down: qmm(2, n * d, &dev),  // hc_lr=2, out zeros
            up: qmm(n * d, 2, &dev),
            inject: qmm(n, n * d, &dev),
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
