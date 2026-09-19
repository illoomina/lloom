//! The Qwen3.8-Flash-Next (Qwen4-exp) model assembly: the 48-layer stack wiring
//! HyperConnection + (DeltaNet | QSA) + MoE per layer, through the observer and
//! recurrent-state seams, into embed → layers → final-HC-collapse → lm_head.
//!
//! Reference graph: `qwen4exp.cpp` (the `graph::graph` ctor, lines 344-430):
//! per layer, `hc_mix(attn) → mixer → hc_combine → hc_mix(ffn) → moe →
//! hc_combine`; final `hc_mix(head) → lm_head` (NO separate output_norm — the
//! head mixer carries it). The n-stream HC carry `res_hc [b,seq,n*d]` is the true
//! residual, threaded across layers; the observer taps it (`HcStatePoint`) and
//! the collapsed per-layer value (`LayerOutputPoint`).

use super::config::Config;
use super::deltanet_block::GatedDeltaNet;
use super::hc::HyperConnection;
use super::moe::{ExpertResidency, MoeLayer};
use super::qsa::{KvCache, QsaAttention};
use crate::nn::observer::{ForwardObserver, HcStatePoint, LayerOutputPoint};
use crate::nn::RecurrentState;
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Module, Result, Tensor, D};

/// One layer's token-mixer (static dispatch — an enum, not `dyn`, per §3.6).
#[derive(Debug, Clone)]
pub enum Mixer {
    Linear(GatedDeltaNet),
    Full(QsaAttention),
}

/// One decoder layer: two HC brackets around the token-mixer and the MoE.
#[derive(Debug, Clone)]
pub struct Layer<R: ExpertResidency> {
    hc_attn: HyperConnection,
    mixer: Mixer,
    hc_ffn: HyperConnection,
    moe: MoeLayer<R>,
}

/// The per-sequence state the forward threads: DeltaNet `S`, the conv window,
/// and per-QSA-layer growing KV. The recurrent channels carry the migrate/
/// rollback/snapshot discipline; KV is a simple append cache (paged KV deferred).
#[derive(Debug, Clone)]
pub struct Cache {
    /// DeltaNet `S` state, one slot per layer (linear layers use it).
    pub s_state: RecurrentState,
    /// DeltaNet conv window, one slot per layer.
    pub conv_state: RecurrentState,
    /// Per-layer QSA KV (only full-attn layers populate their slot).
    pub kv: Vec<KvCache>,
    /// Decode position (number of tokens seen).
    pub pos: usize,
}

impl Cache {
    pub fn new(cfg: &Config) -> Self {
        Self {
            s_state: RecurrentState::new(cfg.n_layers),
            conv_state: RecurrentState::new(cfg.n_layers),
            kv: vec![KvCache::default(); cfg.n_layers],
            pos: 0,
        }
    }
    pub fn reset(&mut self) {
        self.s_state.reset();
        self.conv_state.reset();
        for k in &mut self.kv {
            k.reset();
        }
        self.pos = 0;
    }
}

/// The assembled model.
#[derive(Debug, Clone)]
pub struct Qwen4Exp<R: ExpertResidency> {
    embed_tokens: crate::nn::Embedding,
    layers: Vec<Layer<R>>,
    /// Final HC collapse (the head mixer that carries the output norm).
    hc_head: HyperConnection,
    /// lm_head (`output.weight`, untied).
    lm_head: QMatMul,
    cfg: Config,
}

impl<R: ExpertResidency> Qwen4Exp<R> {
    pub fn new(
        embed_tokens: crate::nn::Embedding,
        layers: Vec<Layer<R>>,
        hc_head: HyperConnection,
        lm_head: QMatMul,
        cfg: Config,
    ) -> Self {
        Self { embed_tokens, layers, hc_head, lm_head, cfg }
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// One decode token: `input_id [1,1]` → logits `[1, vocab]`, threading the
    /// state and calling the observer at each layer's carry. `obs = &mut ()` on
    /// the crown path compiles to zero observation cost.
    pub fn decode<O: ForwardObserver>(
        &self,
        input_id: &Tensor,
        cache: &mut Cache,
        obs: &mut O,
    ) -> Result<Tensor> {
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;

        // embed → broadcast into the n-stream HC carry `res_hc [1, 1, n*d]`.
        // The reference inits the streams from the embed broadcast across n.
        let h = self.embed_tokens.forward(input_id)?; // [1, 1, d]
        let mut res_hc = h.broadcast_as((1, 1, n, d))?.reshape((1, 1, n * d))?.contiguous()?;

        for (l, layer) in self.layers.iter().enumerate() {
            // observe the n-stream carry entering the layer (HcState) — zero cost
            // on the crown path.
            if obs.wants::<HcStatePoint>(l) {
                obs.at::<HcStatePoint>(l, &res_hc)?;
            }

            // HC prologue (attn bracket): collapse streams → mixer input.
            let (cur, inject) = layer.hc_attn.mix(&res_hc)?; // cur [1,1,d]
            let mixed = match &layer.mixer {
                Mixer::Linear(dn) => dn.decode(&cur, l, &mut cache.s_state, &mut cache.conv_state)?,
                Mixer::Full(qsa) => qsa.decode(&cur, &mut cache.kv[l])?,
            };
            // HC epilogue: scatter the mixer output back across the streams.
            res_hc = layer.hc_attn.combine(&res_hc, &mixed, &inject)?;

            // HC prologue (ffn bracket) → MoE → HC epilogue.
            let (cur, inject) = layer.hc_ffn.mix(&res_hc)?;
            let moe_out = layer.moe.decode(&cur)?;
            res_hc = layer.hc_ffn.combine(&res_hc, &moe_out, &inject)?;

            // observe the collapsed layer output (d-wide) — the DSpark/MTP tap side.
            if obs.wants::<LayerOutputPoint>(l) {
                let (collapsed, _) = layer.hc_ffn.mix(&res_hc)?;
                obs.at::<LayerOutputPoint>(l, &collapsed)?;
            }
        }

        // Final HC collapse (carries the output norm) → lm_head.
        let (collapsed, _) = self.hc_head.mix(&res_hc)?; // [1, 1, d]
        cache.pos += 1;
        let logits = self.lm_head.forward(&collapsed)?; // [1, 1, vocab]
        logits.reshape((1, self.cfg.vocab_size))?.to_dtype(lloom_tensor::DType::F32)
    }

    /// Number of layers, for the caller.
    pub fn n_layers(&self) -> usize {
        self.layers.len()
    }

    #[allow(dead_code)]
    fn _uses_d() -> D {
        D::Minus1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::config::LayerKind;
    use super::super::deltanet_block::GatedDeltaNet;
    use super::super::hc::HyperConnection;
    use super::super::moe::{MoeLayer, Resident};
    use super::super::qsa::QsaAttention;
    use crate::nn::{Embedding, RmsNorm};
    use lloom_tensor::{DType, Device};

    fn lin(out: usize, inp: usize, seed: f32, dev: &Device) -> QMatMul {
        let v: Vec<f32> = (0..out * inp).map(|i| ((i as f32 * 0.007 + seed).sin()) * 0.08).collect();
        QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev).unwrap())
    }
    fn ones(n: usize, dev: &Device) -> Tensor {
        Tensor::ones(n, DType::F32, dev).unwrap()
    }
    fn wt(shape: (usize, usize), seed: f32, dev: &Device) -> Tensor {
        let (r, c) = shape;
        Tensor::from_vec(
            (0..r * c).map(|i| ((i as f32 * 0.009 + seed).cos()) * 0.05).collect::<Vec<f32>>(),
            (r, c),
            dev,
        )
        .unwrap()
    }

    fn hc(cfg: &Config, seed: f32, dev: &Device) -> HyperConnection {
        let hc_dim = cfg.hc_dim();
        HyperConnection::new(
            ones(hc_dim, dev),                          // norm_w
            lin(cfg.hc_lowrank, hc_dim, seed, dev),     // down
            lin(hc_dim, cfg.hc_lowrank, seed + 1.0, dev), // up
            lin(cfg.hc_count, hc_dim, seed + 2.0, dev), // inject → n
            cfg.hc_count,
            cfg.hidden_size,
            cfg.norm_eps as f32,
        )
        .unwrap()
    }

    fn deltanet(cfg: &Config, dev: &Device) -> GatedDeltaNet {
        let d = cfg.hidden_size;
        let cc = cfg.conv_channels();
        let vd = cfg.ssm_state_size * cfg.ssm_dt_rank;
        let hv = cfg.ssm_dt_rank;
        GatedDeltaNet::new(
            lin(cc, d, 1.0, dev),
            lin(vd, d, 2.0, dev),
            wt((cc, cfg.ssm_conv_kernel), 3.0, dev),
            Tensor::from_vec((0..hv).map(|_| -0.5f32).collect::<Vec<_>>(), hv, dev).unwrap(),
            Tensor::zeros(hv, DType::F32, dev).unwrap(),
            lin(hv, d, 4.0, dev),
            lin(hv, d, 5.0, dev),
            ones(cfg.ssm_state_size, dev),
            lin(d, vd, 6.0, dev),
            cfg,
        )
    }

    fn qsa(cfg: &Config, dev: &Device) -> QsaAttention {
        let d = cfg.hidden_size;
        let (hd, nh, nkv) = (cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
        QsaAttention::new(
            lin(nh * hd * 2, d, 1.0, dev),
            lin(nkv * hd, d, 2.0, dev),
            lin(nkv * hd, d, 3.0, dev),
            lin(d, nh * hd, 4.0, dev),
            RmsNorm::new(ones(hd, dev), cfg.norm_eps),
            RmsNorm::new(ones(hd, dev), cfg.norm_eps),
            cfg,
        )
    }

    fn moe(cfg: &Config, dev: &Device) -> MoeLayer<Resident> {
        let d = cfg.hidden_size;
        let ff = cfg.moe_intermediate_size;
        let ne = cfg.n_experts;
        let sh = cfg.shared_expert_intermediate_size;
        MoeLayer::new(
            lin(ne, d, 1.0, dev),
            Resident::new(
                (0..ne).map(|e| lin(ff, d, e as f32, dev)).collect(),
                (0..ne).map(|e| lin(ff, d, e as f32 + 100.0, dev)).collect(),
                (0..ne).map(|e| lin(d, ff, e as f32 + 200.0, dev)).collect(),
            ),
            lin(sh, d, 2.0, dev),
            lin(sh, d, 3.0, dev),
            lin(d, sh, 4.0, dev),
            lin(1, d, 5.0, dev),
            cfg,
        )
    }

    /// The WHOLE model decodes end to end at tiny dims: embed → 4 layers
    /// ([L,L,L,F] — DeltaNet ×3 + QSA ×1, each wrapped by HC, each with MoE) →
    /// final HC collapse → lm_head → [1, vocab]. Proves every block wires
    /// together through the HC carry and the state channels, crown observer
    /// (&mut ()) inlines to nothing.
    #[test]
    fn full_model_decodes_end_to_end() {
        let dev = Device::Cpu;
        let cfg = Config::tiny(); // [L, L, L, F]
        // Non-zero embedding table (a zero table gives zero input → zero logits).
        let embed_w = wt((cfg.vocab_size, cfg.hidden_size), 0.5, &dev);
        let embed = Embedding::new(embed_w, cfg.hidden_size);

        let layers: Vec<Layer<Resident>> = (0..cfg.n_layers)
            .map(|l| {
                let mixer = match cfg.layer_types[l] {
                    LayerKind::Linear => Mixer::Linear(deltanet(&cfg, &dev)),
                    LayerKind::Full => Mixer::Full(qsa(&cfg, &dev)),
                };
                Layer {
                    hc_attn: hc(&cfg, l as f32, &dev),
                    mixer,
                    hc_ffn: hc(&cfg, l as f32 + 50.0, &dev),
                    moe: moe(&cfg, &dev),
                }
            })
            .collect();

        let model = Qwen4Exp::new(
            embed,
            layers,
            hc(&cfg, 999.0, &dev),                  // hc_head
            lin(cfg.vocab_size, cfg.hidden_size, 7.0, &dev), // lm_head
            cfg.clone(),
        );

        let mut cache = Cache::new(&cfg);
        let id = Tensor::from_vec(vec![3u32], (1, 1), &dev).unwrap();

        // crown path: &mut () observer inlines to nothing.
        let logits = model.decode(&id, &mut cache, &mut ()).unwrap();
        assert_eq!(logits.dims(), &[1, cfg.vocab_size]);
        let lv = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(lv.iter().all(|x| x.is_finite()), "non-finite logits");
        assert!(lv.iter().any(|&x| x.abs() > 1e-9), "degenerate zero logits");
        assert_eq!(cache.pos, 1);

        // a second token advances state (both the DeltaNet S/conv and QSA KV).
        let id2 = Tensor::from_vec(vec![5u32], (1, 1), &dev).unwrap();
        let logits2 = model.decode(&id2, &mut cache, &mut ()).unwrap();
        assert_eq!(logits2.dims(), &[1, cfg.vocab_size]);
        assert_eq!(cache.pos, 2);
        // the QSA layer's KV grew to 2.
        let full_layer = cfg.layer_types.iter().position(|k| *k == LayerKind::Full).unwrap();
        assert_eq!(cache.kv[full_layer].len(), 2);
    }
}
