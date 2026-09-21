//! Qwen Sparse Attention (QSA) — the full-attention layers (every 4th).
//!
//! For context ≤ `indexer_budget` (2048) QSA is BYTE-IDENTICAL to dense QK-norm
//! GQA attention: the reference's Lightning indexer returns
//! `width = min(n_kv, budget + r − 1)`, which is every cell when `n_kv ≤ budget`
//! (`qwen4exp.cpp:665`). So this builds the dense path — QK-norm, partial-mRoPE,
//! output-gate, `wo` — which is correct (not approximate) for ≤2048 tokens;
//! the indexer sparsification (the risky kernel) is the long-context path, and is
//! deferred (`design/docs/models/qwen3.8-flash-next.md`, "the #28734 lesson").
//!
//! Reference: `qwen4exp.cpp build_layer_attn`. Composes lloom's attention
//! primitives (softmax + matmul, `rope_i`) — a selection of existing blocks
//! (§3.6a), no novel kernel here.

use super::config::Config;
use crate::nn::{ops, RmsNorm};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Module, Result, Tensor, D};

/// Per-layer growing KV state for one sequence (full-attn layers only). Unlike
/// DeltaNet's fixed `S`, KV grows with context, so this is a simple append cache
/// (the paged/pooled KV is the perf path, deferred with caching).
#[derive(Debug, Clone, Default)]
pub struct KvCache {
    /// `[n_kv_heads, seq, head_dim]`, `None` before the first token.
    k: Option<Tensor>,
    v: Option<Tensor>,
}

impl KvCache {
    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
    pub fn len(&self) -> usize {
        self.k.as_ref().map(|k| k.dim(1).unwrap_or(0)).unwrap_or(0)
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One QSA (full-attention) layer's weights.
#[derive(Debug, Clone)]
pub struct QsaAttention {
    /// Fused Q proj → `[q | gate]` interleaved per head (`attn_q`, 2×head_dim/head).
    wq: QMatMul,
    /// K, V projs (`attn_k`, `attn_v`).
    wk: QMatMul,
    wv: QMatMul,
    /// Output proj (`attn_output`).
    wo: QMatMul,
    /// QK-norm (RMS over head_dim), `attn_q_norm` / `attn_k_norm`.
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    /// Rotary dims = `head_dim * partial_rotary_factor` (64 of 256).
    n_rot: usize,
    rope_theta: f64,
}

impl QsaAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: QMatMul,
        wk: QMatMul,
        wv: QMatMul,
        wo: QMatMul,
        q_norm: RmsNorm,
        k_norm: RmsNorm,
        cfg: &Config,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            n_rot: (cfg.head_dim as f64 * cfg.partial_rotary_factor) as usize,
            rope_theta: cfg.rope_theta,
        }
    }

    /// Precompute per-rotary-pair (cos, sin) for `pos`, each `[n_rot/2]`.
    /// Text-only decode: mRoPE reduces to standard RoPE (all sections share the
    /// text position), so partial RoPE over the first `n_rot` dims is correct.
    fn rope_cs(&self, pos: usize, dev: &lloom_tensor::Device) -> Result<(Tensor, Tensor)> {
        let half = self.n_rot / 2;
        let freqs: Vec<f32> = (0..half)
            .map(|i| {
                let inv = 1f32 / (self.rope_theta as f32).powf(2.0 * i as f32 / self.n_rot as f32);
                inv * pos as f32
            })
            .collect();
        let cos = Tensor::from_vec(freqs.iter().map(|f| f.cos()).collect::<Vec<_>>(), half, dev)?;
        let sin = Tensor::from_vec(freqs.iter().map(|f| f.sin()).collect::<Vec<_>>(), half, dev)?;
        Ok((cos, sin))
    }

    /// Apply partial INTERLEAVED RoPE to the first `n_rot` dims of `x [h, head_dim]`,
    /// leaving the tail unrotated. Interleaved pairs `(x[2i], x[2i+1])`:
    /// `y[2i] = x[2i]·cos − x[2i+1]·sin`, `y[2i+1] = x[2i]·sin + x[2i+1]·cos`.
    /// A direct composition (broadcast over heads), CPU-clear and testable.
    fn partial_rope(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (h, hd) = x.dims2()?;
        let half = self.n_rot / 2;
        // rot part [h, n_rot] → view even/odd pairs.
        let rot = x.narrow(1, 0, self.n_rot)?.reshape((h, half, 2))?;
        let x_even = rot.narrow(2, 0, 1)?.reshape((h, half))?;
        let x_odd = rot.narrow(2, 1, 1)?.reshape((h, half))?;
        let cos_b = cos.reshape((1, half))?;
        let sin_b = sin.reshape((1, half))?;
        let y_even = (x_even.broadcast_mul(&cos_b)? - x_odd.broadcast_mul(&sin_b)?)?;
        let y_odd = (x_even.broadcast_mul(&sin_b)? + x_odd.broadcast_mul(&cos_b)?)?;
        // re-interleave: stack on a new last axis then flatten.
        let y_rot = Tensor::stack(&[&y_even, &y_odd], 2)?.reshape((h, self.n_rot))?;
        if self.n_rot == hd {
            return Ok(y_rot);
        }
        let pass = x.narrow(1, self.n_rot, hd - self.n_rot)?;
        Tensor::cat(&[&y_rot, &pass], 1)
    }

    /// One decode step: `x [1, 1, hidden]` in, `[1, 1, hidden]` out, appending to
    /// this layer's `kv`. Dense QK-norm GQA attention (= QSA for ctx ≤ budget).
    pub fn decode(&self, x: &Tensor, kv: &mut KvCache) -> Result<Tensor> {
        let hidden = x.dim(D::Minus1)?;
        let dev = x.device();
        let (hd, nh, nkv) = (self.head_dim, self.n_heads, self.n_kv_heads);
        let x1 = x.reshape((1, hidden))?; // [1, hidden]

        // wq → [q | gate] interleaved per head: layout is
        // [head0_q(hd), head0_gate(hd), head1_q(hd), ...]. Split per head.
        let qg = self.wq.forward(&x1)?.reshape((nh, 2 * hd))?; // [nh, 2*hd]
        let q = qg.narrow(1, 0, hd)?.contiguous()?; // [nh, hd]
        let gate = qg.narrow(1, hd, hd)?.contiguous()?; // [nh, hd]

        let k = self.wk.forward(&x1)?.reshape((nkv, hd))?; // [nkv, hd]
        let v = self.wv.forward(&x1)?.reshape((nkv, hd))?; // [nkv, hd]

        // QK-norm (RMS over head_dim).
        let q = self.q_norm.forward(&q)?; // [nh, hd]
        let k = self.k_norm.forward(&k)?; // [nkv, hd]

        // partial mRoPE at this position.
        let pos = kv.len();
        let (cos, sin) = self.rope_cs(pos, dev)?;
        let q = self.partial_rope(&q, &cos, &sin)?; // [nh, hd]
        let k = self.partial_rope(&k, &cos, &sin)?; // [nkv, hd]

        // Append k, v to the cache: [nkv, seq, hd].
        let k_step = k.reshape((nkv, 1, hd))?;
        let v_step = v.reshape((nkv, 1, hd))?;
        let (k_all, v_all) = match (&kv.k, &kv.v) {
            (Some(kc), Some(vc)) => (
                Tensor::cat(&[kc, &k_step], 1)?,
                Tensor::cat(&[vc, &v_step], 1)?,
            ),
            _ => (k_step.clone(), v_step.clone()),
        };
        kv.k = Some(k_all.clone());
        kv.v = Some(v_all.clone());
        let seq = k_all.dim(1)?;

        // GQA expand kv heads nkv → nh (tile, the reference repeat), then dense
        // attention: scores = q·kᵀ·scale, softmax, ·v. No mask needed at decode
        // (the single query attends to all past keys, all causal).
        let rep = nh / nkv;
        let k_exp = k_all
            .reshape((nkv, 1, seq, hd))?
            .broadcast_as((nkv, rep, seq, hd))?
            .reshape((nh, seq, hd))?;
        let v_exp = v_all
            .reshape((nkv, 1, seq, hd))?
            .broadcast_as((nkv, rep, seq, hd))?
            .reshape((nh, seq, hd))?;

        let scale = 1.0 / (hd as f64).sqrt();
        let q_h = q.reshape((nh, 1, hd))?.to_dtype(lloom_tensor::DType::F32)?; // [nh,1,hd]
        let k_h = k_exp.to_dtype(lloom_tensor::DType::F32)?; // [nh,seq,hd]
        let scores = (q_h.matmul(&k_h.transpose(1, 2)?.contiguous()?)? * scale)?; // [nh,1,seq]
        let probs = ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v_exp.to_dtype(lloom_tensor::DType::F32)?)?; // [nh,1,hd]
        let ctx = ctx.reshape((1, nh * hd))?.to_dtype(x.dtype())?;

        // output gate: · sigmoid(gate), then wo.
        let gate_flat = gate.reshape((1, nh * hd))?;
        let gated = (ctx * ops::sigmoid(&gate_flat)?)?;
        self.wo.forward(&gated)?.reshape((1, 1, hidden))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::{DType, Device};

    fn lin(out: usize, inp: usize, seed: f32, dev: &Device) -> QMatMul {
        let n = out * inp;
        let v: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.011 + seed).sin()) * 0.1)
            .collect();
        QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev).unwrap())
    }
    fn rms(dim: usize, dev: &Device) -> RmsNorm {
        RmsNorm::new(Tensor::ones(dim, DType::F32, dev).unwrap(), 1e-6)
    }

    /// QSA decodes end to end and its KV grows: shapes thread, output finite, and
    /// the cache length advances one per step (context accumulates).
    #[test]
    fn qsa_decodes_and_kv_grows() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let attn = QsaAttention::new(
            lin(nh * hd * 2, h, 1.0, &dev), // wq → [q|gate]
            lin(nkv * hd, h, 2.0, &dev),    // wk
            lin(nkv * hd, h, 3.0, &dev),    // wv
            lin(h, nh * hd, 4.0, &dev),     // wo
            rms(hd, &dev),
            rms(hd, &dev),
            &cfg,
        );
        let mut kv = KvCache::default();
        let tok = |s: f32| {
            Tensor::from_vec(
                (0..h)
                    .map(|i| ((i as f32 * 0.05 + s).sin()) * 0.3)
                    .collect::<Vec<f32>>(),
                (1, 1, h),
                &dev,
            )
            .unwrap()
        };
        assert_eq!(kv.len(), 0);
        let o0 = attn.decode(&tok(0.0), &mut kv).unwrap();
        assert_eq!(o0.dims(), &[1, 1, h]);
        assert_eq!(kv.len(), 1);
        let o1 = attn.decode(&tok(1.0), &mut kv).unwrap();
        assert_eq!(kv.len(), 2);
        let o1v = o1.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(o1v.iter().all(|x| x.is_finite()));
        // step 1 attends to 2 keys (grows); a nonzero output.
        assert!(o1v.iter().any(|&x| x.abs() > 1e-9));
    }

    /// Partial RoPE rotates only the first n_rot dims, leaving the tail
    /// unchanged (partial_rotary_factor 0.25 → 64 of 256; tiny: 4 of 16).
    #[test]
    fn partial_rope_leaves_tail_unrotated() {
        let dev = Device::Cpu;
        let cfg = Config::tiny();
        let attn = QsaAttention::new(
            lin(4, 4, 0.0, &dev),
            lin(4, 4, 0.0, &dev),
            lin(4, 4, 0.0, &dev),
            lin(4, 4, 0.0, &dev),
            rms(cfg.head_dim, &dev),
            rms(cfg.head_dim, &dev),
            &cfg,
        );
        let hd = cfg.head_dim;
        let n_rot = (hd as f64 * cfg.partial_rotary_factor) as usize;
        let x = Tensor::ones((1, hd), DType::F32, &dev).unwrap(); // [h=1, hd]
                                                                  // pos 0: cos=1, sin=0 → rope is identity, whole vector unchanged.
        let (c0, s0) = attn.rope_cs(0, &dev).unwrap();
        let y0 = attn.partial_rope(&x, &c0, &s0).unwrap();
        let y0v = y0.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            y0v.iter().all(|&v| (v - 1.0).abs() < 1e-5),
            "pos0 rope must be identity"
        );
        // pos 5: the tail dims (>= n_rot) stay exactly 1.0 (unrotated).
        let (c, s) = attn.rope_cs(5, &dev).unwrap();
        let y = attn.partial_rope(&x, &c, &s).unwrap();
        let yv = y.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for &v in &yv[n_rot..] {
            assert!((v - 1.0).abs() < 1e-5, "tail dim rotated: {yv:?}");
        }
    }
    /// The Metal decode matches the CPU decode over three positions on the
    /// real head geometry (24/2 heads, head_dim 256, 64 rotated dims): same
    /// weights on both devices, per-step outputs finite and equal within f32
    /// reduction noise. Position 0 is rope-blind (sin = 0), so the later
    /// positions are the ones that see the KV growth, the rope and the
    /// softmax over more than one key.
    #[cfg(feature = "metal")]
    #[test]
    fn qsa_metal_matches_cpu_over_three_tokens() {
        let cpu = Device::Cpu;
        let gpu = Device::new_metal(0).unwrap();
        let mut cfg = Config::flash_next();
        cfg.hidden_size = 256; // small in-proj, real head geometry
        let (h, hd, nh, nkv) = (cfg.hidden_size, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        let build = |dev: &Device| {
            QsaAttention::new(
                lin(nh * hd * 2, h, 1.0, dev),
                lin(nkv * hd, h, 2.0, dev),
                lin(nkv * hd, h, 3.0, dev),
                lin(h, nh * hd, 4.0, dev),
                rms(hd, dev),
                rms(hd, dev),
                &cfg,
            )
        };
        let (a_cpu, a_gpu) = (build(&cpu), build(&gpu));
        let (mut kv_cpu, mut kv_gpu) = (KvCache::default(), KvCache::default());
        for step in 0..3 {
            let x: Vec<f32> = (0..h)
                .map(|i| ((i as f32 * 0.05 + step as f32).sin()) * 0.3)
                .collect();
            let xc = Tensor::from_vec(x.clone(), (1, 1, h), &cpu).unwrap();
            let xg = Tensor::from_vec(x, (1, 1, h), &gpu).unwrap();
            let oc = a_cpu
                .decode(&xc, &mut kv_cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let og = a_gpu
                .decode(&xg, &mut kv_gpu)
                .unwrap()
                .to_device(&cpu)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let nonfinite = og.iter().filter(|v| !v.is_finite()).count();
            assert_eq!(
                nonfinite, 0,
                "step {step}: {nonfinite} non-finite values on Metal"
            );
            let maxdiff = oc
                .iter()
                .zip(&og)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            let scale = oc.iter().map(|a| a.abs()).fold(0f32, f32::max);
            assert!(
                maxdiff <= 1e-3 * scale.max(1e-3),
                "step {step}: Metal/CPU max diff {maxdiff} at scale {scale}"
            );
        }
    }
}
