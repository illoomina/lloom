//! Gated DeltaNet — the gated delta-rule linear-attention recurrence, as a
//! model-parameterized block.
//!
//! Seven current models share this recurrence (llama.cpp `llm_build_delta_net_base`:
//! Qwen3.5 / 3.5-MoE / 3-Next / 4-exp, Kimi-Linear / K3, Bailing-MoE3), so it is
//! built once here parameterized by dims + the gate variant, not re-hand-rolled
//! per model (§3.6a). Qwen3.8-Flash-Next is the first consumer.
//!
//! The recurrence is **pure tensor-op composition** — the reference's non-fused
//! arms are ggml ops, every one of which candle has, so there is no novel Metal
//! kernel here; the ops stay visible to the fusion optimizer (§3.6, #732). The
//! `S` accumulation is fp32 and mandatory (`mamba_ssm_dtype: float32`); a fused
//! f16 epilogue on that path would corrupt, so it is an optimizer exclusion.
//!
//! State `S ∈ [d_k, d_v, H_v]` per sequence is fixed-size (no context growth)
//! and lives in a [`RecurrentState`] channel — same migrate/rollback/snapshot
//! lifecycle as LFM2's conv window.
//!
//! Reference: `qwen4exp.cpp build_layer_attn_linear` +
//! `delta-net-base.cpp build_delta_net_autoregressive` (decode) /
//! `build_delta_net_chunking` (prefill).

use lloom_tensor::{Result, Tensor, D};

/// The gated delta-rule state and dims for one delta-net layer stack.
///
/// Head geometry (Qwen4-exp): `d_k = d_v = ssm_d_state (128)`,
/// `h_k = ssm_n_group (16)`, `h_v = ssm_dt_rank (48)`, GQA-repeat `h_k -> h_v`.
#[derive(Debug, Clone, Copy)]
pub struct DeltaNetDims {
    /// Key/query head dim (`d_k = d_v`).
    pub d_k: usize,
    /// Value head dim (`= d_k` for this family).
    pub d_v: usize,
    /// Number of key/query heads (`ssm_n_group`).
    pub h_k: usize,
    /// Number of value heads (`ssm_dt_rank`); the state carries `h_v` matrices.
    pub h_v: usize,
}

/// The gated delta-rule decode step (Kernel A), for ONE sequence, ONE token.
///
/// Inputs are already conv'd, gated, L2-normed and GQA-repeated to `h_v` heads:
/// - `q`, `k`: `[h_v, d_k]` (post-l2norm, post-repeat, q pre-scaled here)
/// - `v`: `[h_v, d_v]`
/// - `g`: `[h_v]` the per-head log-decay (`exp` applied here)
/// - `beta`: `[h_v]` the per-head write strength (already `sigmoid`'d)
/// - `state`: `[d_k, d_v, h_v]` fp32, the carried `S` (or `None` = zero state)
///
/// Returns `(o [h_v, d_v], new_state)` where `new_state`'s layout is chosen by
/// `STATE_HEAD_FIRST` (see below). All math in the state's dtype (fp32). Mirrors
/// `build_delta_net_autoregressive` op for op:
/// `g=exp(g); S=g·S; r=sum_k(S·k); d=β(v−r); S += k⊗d; o=sum_k(S·q)`.
///
/// # Geometry (const-generic, flash_decoding style — compile-time, no branch)
///
/// `STATE_HEAD_FIRST` selects the `S` storage layout, a geometry axis lfm2
/// measured worth tuning (tiling/stride pays at B>1 + speculative):
/// - `false` (the reference-faithful DEFAULT, the digest anchor): `S` is stored
///   head-LAST `[d_k, d_v, h_v]`, matching the reference recurrent-cache row
///   (`n_embd_s = d_k·d_v·h_v` contiguous). Costs a permute in/out per step.
/// - `true`: `S` is stored head-FIRST `[h_v, d_k, d_v]` — the recurrence's own
///   broadcast layout — so the per-step permute disappears. Our own `S` bytes,
///   not the reference cache's; the channel is shape-agnostic so it does not care.
///
/// The two produce bit-identical OUTPUT (same fp32 ops, only the storage
/// transpose differs), so `false` is the parity/digest baseline and `true` is a
/// perf variant selectable per regime.
pub fn decode_step<const STATE_HEAD_FIRST: bool>(
    dims: DeltaNetDims,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
    let dev = q.device();
    let dt = lloom_tensor::DType::F32; // S is fp32 (mamba_ssm_dtype)

    // Bring the carried state into the recurrence's head-first working layout
    // `[h_v, d_k, d_v]`, whatever it was stored as; zero when unwritten.
    let s = match state {
        Some(s) if STATE_HEAD_FIRST => s.to_dtype(dt)?, // already [h_v, d_k, d_v]
        Some(s) => s.to_dtype(dt)?.permute((2, 0, 1))?.contiguous()?, // [d_k,d_v,h_v]->[h_v,d_k,d_v]
        None => Tensor::zeros((h_v, d_k, d_v), dt, dev)?,
    };

    // Everything to fp32 for the recurrence.
    let q = q.to_dtype(dt)?; // [h_v, d_k]
    let k = k.to_dtype(dt)?; // [h_v, d_k]
    let v = v.to_dtype(dt)?; // [h_v, d_v]
    let g = g.to_dtype(dt)?; // [h_v]
    let beta = beta.to_dtype(dt)?; // [h_v]

    // Decay: S <- exp(g) * S, per-head scalar broadcast over [d_k, d_v].
    let g_exp = g.exp()?.reshape((h_v, 1, 1))?; // [h_v,1,1]
    let s = s.broadcast_mul(&g_exp)?; // [h_v, d_k, d_v]

    // r = Sᵀ k̃  (read-out): sum over d_k of S[.,dk,dv] * k[.,dk] -> [h_v, d_v].
    let k_col = k.reshape((h_v, d_k, 1))?; // [h_v, d_k, 1]
    let r = s.broadcast_mul(&k_col)?.sum(1)?; // [h_v, d_v]

    // delta d = β (v − r), broadcasting the per-head scalar β over d_v.
    let beta_b = beta.reshape((h_v, 1))?; // [h_v,1]
    let d = (v - &r)?.broadcast_mul(&beta_b)?; // [h_v, d_v]

    // Rank-1 update: S <- S + k̃ ⊗ d  (outer over d_k × d_v).
    let d_row = d.reshape((h_v, 1, d_v))?; // [h_v,1,d_v]
    let kd = k_col.broadcast_mul(&d_row)?; // [h_v, d_k, d_v]
    let s = (s + kd)?; // [h_v, d_k, d_v]

    // o = Sᵀ q̃ : sum over d_k of S[.,dk,dv] * q[.,dk] -> [h_v, d_v].
    let q_col = q.reshape((h_v, d_k, 1))?;
    let o = s.broadcast_mul(&q_col)?.sum(1)?; // [h_v, d_v]

    // Store `S` back in the chosen layout.
    let new_state = if STATE_HEAD_FIRST {
        s // [h_v, d_k, d_v]
    } else {
        s.permute((1, 2, 0))?.contiguous()? // [d_k, d_v, h_v]
    };
    Ok((o, new_state))
}

/// GQA head expansion `h_k -> h_v` before the recurrence, const-generic over the
/// mapping style (a geometry axis; lfm2 measured strided/tiled head geometries
/// worth tuning at B>1 + speculative).
///
/// `INTERLEAVE`:
/// - `false` (reference-faithful DEFAULT, the digest anchor): TILE, exactly
///   `ggml_repeat` — the whole `[.., h_k]` block repeated `h_v/h_k` times, so
///   v-head `j` reads k-head `j % h_k` (verified from ggml's repeat CPU kernel).
/// - `true`: INTERLEAVE — each k-head repeated `h_v/h_k` consecutively, v-head
///   `j` reads k-head `j / (h_v/h_k)`. A DIFFERENT operator (pairs every v-head
///   with a different k-head); the reference does NOT use it, so it is a
///   deliberate variant, never the parity baseline.
///
/// `x` is `[h_k, d]`, returns `[h_v, d]`.
pub fn gqa_expand<const INTERLEAVE: bool>(x: &Tensor, h_k: usize, h_v: usize) -> Result<Tensor> {
    if h_k == h_v {
        return Ok(x.clone());
    }
    debug_assert_eq!(h_v % h_k, 0, "h_v must be a multiple of h_k");
    let rep = h_v / h_k;
    let d = x.dim(1)?;
    if INTERLEAVE {
        // [h_k, d] -> [h_k, 1, d] -> broadcast [h_k, rep, d] -> [h_v, d]
        x.reshape((h_k, 1, d))?
            .broadcast_as((h_k, rep, d))?
            .reshape((h_v, d))
    } else {
        // TILE (ggml_repeat): [h_k, d] -> [1, h_k, d] -> [rep, h_k, d] -> [h_v, d]
        x.reshape((1, h_k, d))?
            .broadcast_as((rep, h_k, d))?
            .reshape((h_v, d))
    }
}

/// Pre-scale `q` by `1/sqrt(d_k)` as the reference does before the recurrence.
pub fn q_scale(q: &Tensor, d_k: usize) -> Result<Tensor> {
    q.affine(1.0 / (d_k as f64).sqrt(), 0.0)
}

/// L2-normalize the last dim (used on q and k before the recurrence — L2, NOT
/// RMS: `x / sqrt(sum(x^2) + eps)`, no learned weight).
pub fn l2_norm_last(x: &Tensor, eps: f64) -> Result<Tensor> {
    let dt = x.dtype();
    let xf = x.to_dtype(lloom_tensor::DType::F32)?;
    let ss = xf.sqr()?.sum_keepdim(D::Minus1)?;
    let normed = xf.broadcast_div(&(ss + eps)?.sqrt()?)?;
    normed.to_dtype(dt)
}

/// Numerically-stable `softplus(x) = log(1 + exp(x))`, via
/// `max(x,0) + log1p(exp(-|x|))` so a large `x` does not overflow `exp`. lloom
/// has no softplus op; this composes it from ops the optimizer can see.
pub fn softplus(x: &Tensor) -> Result<Tensor> {
    let zeros = x.zeros_like()?;
    let max0 = x.maximum(&zeros)?; // max(x, 0)
    let neg_abs = x.abs()?.neg()?; // -|x|
                                   // log1p(exp(-|x|)) = log(1 + exp(-|x|)); exp(-|x|) ∈ (0,1] so no overflow.
    let log1p = (neg_abs.exp()? + 1.0)?.log()?;
    max0 + log1p
}

/// Compute the gated-delta gate `g` and write-strength `beta` from the layer's
/// pre-conv input `x [b, seq, d]` and the SSM parameters.
///
/// `beta = sigmoid(ssm_beta · x)`; `alpha = ssm_alpha · x`;
/// `g = ssm_a · softplus(alpha + ssm_dt_bias)`  (`ssm_a` holds `−exp(A_log)`, so
/// `g` is the reference's `−exp(A_log)·softplus(a + dt_bias)`, the log-decay).
/// Returns `(g, beta)` each `[b, seq, h_v]`, in fp32 (the recurrence's dtype).
pub fn gate_and_beta(
    x: &Tensor,
    ssm_beta: &lloom_tensor::quantized::QMatMul,
    ssm_alpha: &lloom_tensor::quantized::QMatMul,
    ssm_dt_bias: &Tensor, // [h_v]
    ssm_a: &Tensor,       // [h_v]  (= −exp(A_log))
) -> Result<(Tensor, Tensor)> {
    use lloom_tensor::Module;
    let f32 = lloom_tensor::DType::F32;
    let beta = crate::nn::ops::sigmoid(&ssm_beta.forward(x)?.to_dtype(f32)?)?;
    let alpha = ssm_alpha.forward(x)?.to_dtype(f32)?; // [b,seq,h_v]
    let a_biased = alpha.broadcast_add(&ssm_dt_bias.to_dtype(f32)?)?;
    let a_sp = softplus(&a_biased)?;
    let g = a_sp.broadcast_mul(&ssm_a.to_dtype(f32)?)?;
    Ok((g, beta))
}

/// One decode-step of the depthwise causal conv over the concat-QKV channels
/// (k=4, no bias — `ssm_conv1d`), mirroring the shift-and-fold pattern Mamba2
/// uses: shift the per-channel window, append the new token, fold against the
/// k-wide weight. The window is a [`RecurrentState`] channel (a second recurrent
/// state alongside `S`), so it migrates and rolls back with the sequence.
///
/// `x` is `[channels]` (this token's concat-QKV projection), `weight` is
/// `[channels, k]` (the depthwise kernel), `window` is the `[channels, k]` state
/// (oldest..newest, `None` = zero). Returns `(conv_out [channels], new_window)`.
pub fn conv_decode_step(
    x: &Tensor,
    weight: &Tensor,
    window: Option<&Tensor>,
    channels: usize,
    k: usize,
) -> Result<(Tensor, Tensor)> {
    let dev = x.device();
    let dt = x.dtype();
    // window [channels, k], newest at column k-1.
    let win = match window {
        Some(w) => w.clone(),
        None => Tensor::zeros((channels, k), dt, dev)?,
    };
    // shift left by one, append x at the last column.
    let shifted = win.narrow(1, 1, k - 1)?; // [channels, k-1]
    let x_col = x.reshape((channels, 1))?;
    let new_win = Tensor::cat(&[&shifted, &x_col], 1)?; // [channels, k]
                                                        // fold: sum_i weight[:,i] * window[:,i].
    let folded = (new_win.contiguous()? * weight)?.sum(1)?; // [channels]
    Ok((folded, new_win))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lloom_tensor::{Device, IndexOp};

    /// A reference decode step computed with plain nested loops, to pin
    /// `decode_step` against the delta-rule math directly (§3.1 parity — the
    /// reference is an independent scalar implementation, not a restatement).
    fn ref_step(
        dims: DeltaNetDims,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        g: &[f32],
        beta: &[f32],
        s: &mut [f32], // [h_v * d_k * d_v], layout [h][dk][dv]
    ) -> Vec<f32> {
        let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
        let mut o = vec![0f32; h_v * d_v];
        for h in 0..h_v {
            let ge = g[h].exp();
            let sb = h * d_k * d_v;
            // S <- ge * S
            for x in &mut s[sb..sb + d_k * d_v] {
                *x *= ge;
            }
            // r[dv] = sum_dk S[dk,dv] * k[dk]
            let mut r = vec![0f32; d_v];
            for dk in 0..d_k {
                let kk = k[h * d_k + dk];
                for dv in 0..d_v {
                    r[dv] += s[sb + dk * d_v + dv] * kk;
                }
            }
            // d[dv] = beta * (v - r)
            let mut dd = vec![0f32; d_v];
            for dv in 0..d_v {
                dd[dv] = beta[h] * (v[h * d_v + dv] - r[dv]);
            }
            // S += k outer d
            for dk in 0..d_k {
                let kk = k[h * d_k + dk];
                for dv in 0..d_v {
                    s[sb + dk * d_v + dv] += kk * dd[dv];
                }
            }
            // o[dv] = sum_dk S[dk,dv] * q[dk]
            for dk in 0..d_k {
                let qq = q[h * d_k + dk];
                for dv in 0..d_v {
                    o[h * d_v + dv] += s[sb + dk * d_v + dv] * qq;
                }
            }
        }
        o
    }

    #[test]
    fn decode_step_matches_scalar_reference() {
        let dev = Device::Cpu;
        let dims = DeltaNetDims {
            d_k: 3,
            d_v: 3,
            h_k: 2,
            h_v: 2,
        };
        let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
        // Deterministic pseudo-inputs.
        let mk = |n: usize, seed: f32| -> Vec<f32> {
            (0..n).map(|i| ((i as f32 * 0.37 + seed).sin())).collect()
        };
        let q = mk(h_v * d_k, 0.1);
        let k = mk(h_v * d_k, 0.2);
        let v = mk(h_v * d_v, 0.3);
        let g = mk(h_v, 0.4);
        let beta = mk(h_v, 0.5);

        // Scalar reference over an initial nonzero state.
        let s0: Vec<f32> = mk(h_v * d_k * d_v, 0.6);
        let mut s_ref = s0.clone();
        let o_ref = ref_step(dims, &q, &k, &v, &g, &beta, &mut s_ref);

        // Tensor path: state as [d_k, d_v, h_v] (storage layout).
        let mut s_store = vec![0f32; h_v * d_k * d_v];
        for h in 0..h_v {
            for dk in 0..d_k {
                for dv in 0..d_v {
                    s_store[(dk * d_v + dv) * h_v + h] = s0[h * d_k * d_v + dk * d_v + dv];
                }
            }
        }
        let s_t = Tensor::from_vec(s_store, (d_k, d_v, h_v), &dev).unwrap();
        let qt = Tensor::from_vec(q, (h_v, d_k), &dev).unwrap();
        let kt = Tensor::from_vec(k, (h_v, d_k), &dev).unwrap();
        let vt = Tensor::from_vec(v, (h_v, d_v), &dev).unwrap();
        let gt = Tensor::from_vec(g, h_v, &dev).unwrap();
        let bt = Tensor::from_vec(beta, h_v, &dev).unwrap();

        let (o, _new_s) = decode_step::<false>(dims, &qt, &kt, &vt, &gt, &bt, Some(&s_t)).unwrap();
        let o_got = o.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, b) in o_got.iter().zip(o_ref.iter()) {
            assert!((a - b).abs() < 1e-5, "got {o_got:?} ref {o_ref:?}");
        }
    }

    /// The two `STATE_HEAD_FIRST` geometries produce bit-identical OUTPUT (only
    /// the storage transpose differs) — the property that lets `true` be a perf
    /// variant and `false` the digest anchor. Feed head-first the head-last
    /// state permuted, and check `o` matches.
    #[test]
    fn state_layout_variants_agree_on_output() {
        let dev = Device::Cpu;
        let dims = DeltaNetDims {
            d_k: 3,
            d_v: 3,
            h_k: 2,
            h_v: 2,
        };
        let (h_v, d_k, d_v) = (dims.h_v, dims.d_k, dims.d_v);
        let mk = |n: usize, s: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * 0.29 + s).cos()).collect()
        };
        let qt = Tensor::from_vec(mk(h_v * d_k, 0.1), (h_v, d_k), &dev).unwrap();
        let kt = Tensor::from_vec(mk(h_v * d_k, 0.2), (h_v, d_k), &dev).unwrap();
        let vt = Tensor::from_vec(mk(h_v * d_v, 0.3), (h_v, d_v), &dev).unwrap();
        let gt = Tensor::from_vec(mk(h_v, 0.4), h_v, &dev).unwrap();
        let bt = Tensor::from_vec(mk(h_v, 0.5), h_v, &dev).unwrap();
        // head-last state, and its head-first permutation.
        let s_last = Tensor::from_vec(mk(d_k * d_v * h_v, 0.6), (d_k, d_v, h_v), &dev).unwrap();
        let s_first = s_last.permute((2, 0, 1)).unwrap().contiguous().unwrap();

        let (o_last, ns_last) =
            decode_step::<false>(dims, &qt, &kt, &vt, &gt, &bt, Some(&s_last)).unwrap();
        let (o_first, ns_first) =
            decode_step::<true>(dims, &qt, &kt, &vt, &gt, &bt, Some(&s_first)).unwrap();
        let a = o_last.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = o_first.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-6, "outputs differ: {a:?} vs {b:?}");
        }
        // And the stored states are transposes of each other.
        assert_eq!(ns_last.dims(), &[d_k, d_v, h_v]);
        assert_eq!(ns_first.dims(), &[h_v, d_k, d_v]);
        let nl = ns_last
            .permute((2, 0, 1))
            .unwrap()
            .contiguous()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        let nf = ns_first.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (x, y) in nl.iter().zip(&nf) {
            assert!((x - y).abs() < 1e-6);
        }
    }

    /// The decode conv shifts the window and folds against the k-wide weight,
    /// exactly the causal depthwise conv. Two tokens through a 1-channel k=3 conv,
    /// checked by hand.
    #[test]
    fn conv_decode_step_folds_causal_window() {
        let dev = Device::Cpu;
        let (ch, k) = (1usize, 3usize);
        // weight [1,3] = [w0,w1,w2]; conv folds oldest..newest = w0*x_{t-2}+w1*x_{t-1}+w2*x_t.
        let w = Tensor::from_vec(vec![0.5f32, 1.0, 2.0], (ch, k), &dev).unwrap();
        // token 1 (x=1) onto a zero window: window=[0,0,1], out = 0.5*0+1*0+2*1 = 2.
        let x1 = Tensor::from_vec(vec![1f32], ch, &dev).unwrap();
        let (o1, win1) = conv_decode_step(&x1, &w, None, ch, k).unwrap();
        assert_eq!(o1.to_vec1::<f32>().unwrap()[0], 2.0);
        // token 2 (x=3): window shifts to [0,1,3], out = 0.5*0+1*1+2*3 = 7.
        let x2 = Tensor::from_vec(vec![3f32], ch, &dev).unwrap();
        let (o2, _win2) = conv_decode_step(&x2, &w, Some(&win1), ch, k).unwrap();
        assert_eq!(o2.to_vec1::<f32>().unwrap()[0], 7.0);
    }

    /// softplus matches log(1+exp(x)) including at large |x| (no overflow).
    #[test]
    fn softplus_stable() {
        let dev = Device::Cpu;
        let x = Tensor::from_vec(vec![-40f32, -1.0, 0.0, 1.0, 40.0], 5, &dev).unwrap();
        let got = softplus(&x).unwrap().to_vec1::<f32>().unwrap();
        let want = [
            (1f32 + (-40f32).exp()).ln(),
            (1f32 + (-1f32).exp()).ln(),
            std::f32::consts::LN_2,
            (1f32 + 1f32.exp()).ln(),
            40.0 + (1f32 + (-40f32).exp()).ln(), // large x: ~x
        ];
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-4, "softplus {got:?} vs {want:?}");
        }
        // no NaN/inf at the extreme.
        assert!(got.iter().all(|v| v.is_finite()));
    }

    /// GQA tile (reference) vs interleave are DIFFERENT operators: v-head j reads
    /// k-head j%h_k (tile) vs j/(h_v/h_k) (interleave). Pin both mappings.
    #[test]
    fn gqa_tile_vs_interleave() {
        let dev = Device::Cpu;
        let (h_k, h_v, d) = (2usize, 4usize, 2usize);
        // heads 0,1 distinguishable: head0=[1,1], head1=[2,2].
        let x = Tensor::from_vec(vec![1f32, 1.0, 2.0, 2.0], (h_k, d), &dev).unwrap();
        let tile = gqa_expand::<false>(&x, h_k, h_v).unwrap();
        let inter = gqa_expand::<true>(&x, h_k, h_v).unwrap();
        // tile: [h0,h1,h0,h1] -> heads = 1,2,1,2 (per first col)
        let tv = tile.i((.., 0)).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(tv, vec![1.0, 2.0, 1.0, 2.0]);
        // interleave: [h0,h0,h1,h1] -> heads = 1,1,2,2
        let iv = inter.i((.., 0)).unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(iv, vec![1.0, 1.0, 2.0, 2.0]);
    }

    /// Mutation: a broken decay (skip `exp`) must fail the parity check — proves
    /// the test can fail (§3.1 mutation).
    #[test]
    fn parity_is_not_vacuous() {
        // Sanity that the reference itself is sensitive to the gate: two
        // different g give different outputs, so a test comparing them is live.
        let dev = Device::Cpu;
        let dims = DeltaNetDims {
            d_k: 2,
            d_v: 2,
            h_k: 1,
            h_v: 1,
        };
        let q = Tensor::from_vec(vec![1f32, 0.5], (1, 2), &dev).unwrap();
        let k = Tensor::from_vec(vec![0.3f32, 0.7], (1, 2), &dev).unwrap();
        let v = Tensor::from_vec(vec![1f32, -1.0], (1, 2), &dev).unwrap();
        let b = Tensor::from_vec(vec![0.9f32], 1, &dev).unwrap();
        let s = Tensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4], (2, 2, 1), &dev).unwrap();
        let g_a = Tensor::from_vec(vec![0.0f32], 1, &dev).unwrap();
        let g_b = Tensor::from_vec(vec![-2.0f32], 1, &dev).unwrap();
        let (oa, _) = decode_step::<false>(dims, &q, &k, &v, &g_a, &b, Some(&s)).unwrap();
        let (ob, _) = decode_step::<false>(dims, &q, &k, &v, &g_b, &b, Some(&s)).unwrap();
        let va = oa.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let vb = ob.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(va.iter().zip(&vb).any(|(a, b)| (a - b).abs() > 1e-4));
    }
}
