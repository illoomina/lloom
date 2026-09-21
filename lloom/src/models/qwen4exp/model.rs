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
use super::moe::Resident;
use super::moe::{ExpertResidency, MoeLayer};
use super::ple::{NgramHash, NgramWindow, PleLayer, PleTable};
use super::qsa::{KvCache, QsaAttention};
use super::trace;
use crate::nn::model::{CacheReset, CausalLM, Model as ModelTrait};
use crate::nn::observer::{ForwardObserver, HcStatePoint, LayerOutputPoint};
use crate::nn::var_builder::VarBuilder;
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

impl<R: ExpertResidency> Layer<R> {
    pub fn new(
        hc_attn: HyperConnection,
        mixer: Mixer,
        hc_ffn: HyperConnection,
        moe: MoeLayer<R>,
    ) -> Self {
        Self {
            hc_attn,
            mixer,
            hc_ffn,
            moe,
        }
    }
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
    /// The PLE's dilated-conv history, one slot per layer (only the PLE layer's
    /// is written).
    pub ple_conv: RecurrentState,
    /// The last `ngram_size − 1` token ids, for the PLE hash.
    pub ngram: NgramWindow,
    /// Token ids of the positions the in-flight pass will consume, front
    /// first; set by [`CausalLM::forward`], empty when the caller only had
    /// embeddings (an image), in which case the hash uses the image stand-in.
    pub pending_tokens: std::collections::VecDeque<u32>,
    /// Decode position (number of tokens seen).
    pub pos: usize,
}

impl Cache {
    pub fn new(cfg: &Config) -> Self {
        Self {
            s_state: RecurrentState::new(cfg.n_layers),
            conv_state: RecurrentState::new(cfg.n_layers),
            kv: vec![KvCache::default(); cfg.n_layers],
            ple_conv: RecurrentState::new(cfg.n_layers),
            ngram: NgramWindow::new(cfg.ple_ngram_size.saturating_sub(1)),
            pending_tokens: std::collections::VecDeque::new(),
            pos: 0,
        }
    }
    pub fn reset(&mut self) {
        self.s_state.reset();
        self.conv_state.reset();
        for k in &mut self.kv {
            k.reset();
        }
        self.ple_conv.reset();
        self.ngram.reset();
        self.pending_tokens.clear();
        self.pos = 0;
    }

    /// A checksum (FNV over the f32 bit patterns) and a non-finite count per
    /// state channel and layer, for a probe comparing two runs' states: which
    /// channel diverges or goes non-finite first. Reads every state back — an
    /// instrument, never on the decode path.
    pub fn checksums(&self) -> Result<Vec<(String, u64, usize)>> {
        fn fnv(t: &Tensor) -> Result<(u64, usize)> {
            let v = t.flatten_all()?.to_dtype(lloom_tensor::DType::F32)?.to_vec1::<f32>()?;
            let mut h = 0xcbf2_9ce4_8422_2325u64;
            let mut nonfinite = 0usize;
            for x in v {
                h ^= x.to_bits() as u64;
                h = h.wrapping_mul(0x0100_0000_01b3);
                if !x.is_finite() {
                    nonfinite += 1;
                }
            }
            Ok((h, nonfinite))
        }
        let mut out = Vec::new();
        for l in 0..self.s_state.n_layers() {
            if let Some(t) = self.s_state.get(l) {
                let (h, n) = fnv(t)?;
                out.push((format!("S[{l}]"), h, n));
            }
            if let Some(t) = self.conv_state.get(l) {
                let (h, n) = fnv(t)?;
                out.push((format!("conv[{l}]"), h, n));
            }
            if let Some(t) = self.ple_conv.get(l) {
                let (h, n) = fnv(t)?;
                out.push((format!("ple_conv[{l}]"), h, n));
            }
        }
        for (l, kv) in self.kv.iter().enumerate() {
            for (name, t) in ["k", "v", "raw", "blocks"].iter().zip(kv.state_views()?) {
                if let Some(t) = t {
                    let (h, n) = fnv(&t)?;
                    out.push((format!("{name}[{l}]"), h, n));
                }
            }
        }
        Ok(out)
    }

    /// Name the token ids of the positions about to be run, so the PLE hash
    /// sees them (a caller entering through `forward_embeds` alone leaves
    /// this empty).
    pub fn set_pending_tokens(&mut self, tokens: &[u32]) {
        self.pending_tokens.clear();
        self.pending_tokens.extend(tokens.iter().copied());
    }
}

/// How a multi-token pass (a prompt) is run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillMode {
    /// Whole chunks through Kernel B / dense chunk attention / per-row
    /// routing (the reference's prefill graph). Chunks of at most
    /// [`PREFILL_CHUNK`] positions.
    #[default]
    Chunk,
    /// The decode recurrence replayed one position at a time (#747's path;
    /// the per-token math, bit for bit, at prefill cost).
    Replay,
}

/// Positions per chunk on the [`PrefillMode::Chunk`] path.
pub const PREFILL_CHUNK: usize = 512;

/// The N-gram PLE at its one layer.
#[derive(Debug)]
pub struct Ple {
    pub layer: usize,
    pub hash: NgramHash,
    pub table: PleTable,
    pub block: PleLayer,
}

/// The assembled model.
#[derive(Debug)]
pub struct Qwen4Exp<R: ExpertResidency> {
    /// `token_embd.weight` `[vocab, hidden]` — a `QMatMul` so the real IQ3_S
    /// table is gathered by the quantized `get_rows` kernel, never dequantized
    /// whole (2.5 GB at f32).
    embed_tokens: QMatMul,
    layers: Vec<Layer<R>>,
    /// Final HC collapse (the head mixer that carries the output norm).
    hc_head: HyperConnection,
    /// lm_head (`output.weight`, untied).
    lm_head: QMatMul,
    /// The N-gram PLE, when its table was loaded (`LoadOptions::ple_shard`).
    ple: Option<Ple>,
    prefill_mode: PrefillMode,
    cfg: Config,
}

impl<R: ExpertResidency> Qwen4Exp<R> {
    pub fn new(
        embed_tokens: QMatMul,
        layers: Vec<Layer<R>>,
        hc_head: HyperConnection,
        lm_head: QMatMul,
        cfg: Config,
    ) -> Self {
        Self {
            embed_tokens,
            layers,
            hc_head,
            lm_head,
            ple: None,
            prefill_mode: PrefillMode::default(),
            cfg,
        }
    }

    /// Select how multi-token passes run (see [`PrefillMode`]).
    pub fn set_prefill_mode(&mut self, mode: PrefillMode) {
        self.prefill_mode = mode;
    }

    /// Select every layer's routed-expert activation dtype (`F32` or `F16`;
    /// `MoeLayer::set_routed_act`).
    pub fn set_routed_act(&mut self, dtype: lloom_tensor::DType) {
        for l in &mut self.layers {
            l.moe.set_routed_act(dtype);
        }
    }

    pub fn prefill_mode(&self) -> PrefillMode {
        self.prefill_mode
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Attach the N-gram PLE at `layer`.
    pub fn set_ple(
        &mut self,
        layer: usize,
        hash: NgramHash,
        table: PleTable,
        block: PleLayer,
    ) -> Result<()> {
        if layer >= self.layers.len() {
            lloom_tensor::bail!("ple layer {layer} out of {} layers", self.layers.len())
        }
        if table.head_dim()? * hash.n_heads() == 0 {
            lloom_tensor::bail!("ple: empty table or hash")
        }
        self.ple = Some(Ple {
            layer,
            hash,
            table,
            block,
        });
        Ok(())
    }

    /// Whether the N-gram PLE is attached.
    pub fn has_ple(&self) -> bool {
        self.ple.is_some()
    }

    /// The PLE delta for the positions of `res_hc [1, seq, n·d]`: hash each
    /// position's token window, gather the rows, run the block against the
    /// carry. Consumes `seq` pending tokens and advances the window.
    fn ple_delta(&self, ple: &Ple, res_hc: &Tensor, cache: &mut Cache) -> Result<Tensor> {
        let seq = res_hc.dim(1)?;
        let n_heads = ple.hash.n_heads();
        let mut ids = vec![0i64; seq * n_heads];
        for t in 0..seq {
            let token = cache.pending_tokens.pop_front();
            let token = token.or(self.cfg.ple_image_token_id);
            let window = cache.ngram.window(token);
            ple.hash
                .row_ids(&window, &mut ids[t * n_heads..(t + 1) * n_heads]);
            cache.ngram.push(token);
        }
        let emb = ple.table.gather(&ids, res_hc.device())?; // [seq·n_heads·head_dim]
        let emb = emb.reshape((seq, emb.elem_count() / seq))?;
        ple.block
            .forward(res_hc, &emb, &mut cache.ple_conv, ple.layer)
    }

    /// Embed `tokens [b, seq]` → `[b, seq, hidden]` without running a layer.
    pub fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.embed_tokens.embedding(tokens)
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
        let h = self.embed(input_id)?; // [1, 1, d]
        self.decode_embed(&h, cache, obs)
    }

    /// Run a sequence `embeds [1, seq, hidden]` from `cache.pos`, returning the
    /// LAST position's logits `[1, vocab]`. One position takes the decode
    /// path; more take [`PrefillMode`]: chunks of [`PREFILL_CHUNK`] through
    /// [`Self::prefill_embeds`], or the decode recurrence replayed.
    pub fn forward_embeds<O: ForwardObserver>(
        &self,
        embeds: &Tensor,
        cache: &mut Cache,
        obs: &mut O,
    ) -> Result<Tensor> {
        let (b, seq, _) = embeds.dims3()?;
        if b != 1 || seq == 0 {
            lloom_tensor::bail!(
                "qwen4exp forward takes one sequence of >= 1 tokens, got {:?}",
                embeds.shape()
            )
        }
        let mut logits = None;
        match self.prefill_mode {
            PrefillMode::Replay => {
                for t in 0..seq {
                    let h = embeds.narrow(1, t, 1)?;
                    logits = Some(self.decode_embed(&h, cache, obs)?);
                }
            }
            PrefillMode::Chunk => {
                // A chunk attends densely, which is QSA only while the context
                // stays within the indexer's dense range; positions past it
                // run one at a time through the selection.
                let dense_limit = self.dense_limit();
                let mut start = 0;
                while start < seq {
                    let mut len = (seq - start).min(PREFILL_CHUNK);
                    if let Some(limit) = dense_limit {
                        len = len.min(limit.saturating_sub(cache.pos)).max(1);
                    }
                    let h = embeds.narrow(1, start, len)?;
                    logits = Some(if len == 1 {
                        self.decode_embed(&h, cache, obs)?
                    } else {
                        self.prefill_embeds(&h, cache, obs)?
                    });
                    start += len;
                }
            }
        }
        Ok(logits.expect("seq >= 1"))
    }

    /// One chunk of positions from their embeddings `h [1, seq, hidden]` →
    /// the last position's logits `[1, vocab]`: the layer stack over the
    /// whole chunk (DeltaNet Kernel B, dense chunk attention, per-row MoE,
    /// the PLE over the chunk), the observer seeing each layer's chunk.
    pub fn prefill_embeds<O: ForwardObserver>(
        &self,
        h: &Tensor,
        cache: &mut Cache,
        obs: &mut O,
    ) -> Result<Tensor> {
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;
        let (_, seq, _) = h.dims3()?;
        let mut res_hc = h
            .reshape((1, seq, 1, d))?
            .broadcast_as((1, seq, n, d))?
            .reshape((1, seq, n * d))?
            .contiguous()?;

        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(ple) = &self.ple {
                if ple.layer == l {
                    let delta = self.ple_delta(ple, &res_hc, cache)?;
                    trace("ple", l, &delta);
                    res_hc = (res_hc + delta)?;
                }
            }
            if obs.wants::<HcStatePoint>(l) {
                obs.at::<HcStatePoint>(l, &res_hc)?;
            }
            let (cur, inject) = layer.hc_attn.mix(&res_hc)?; // [1, seq, d]
            let mixed = match &layer.mixer {
                Mixer::Linear(dn) => {
                    dn.prefill(&cur, l, &mut cache.s_state, &mut cache.conv_state)?
                }
                Mixer::Full(qsa) => qsa.prefill(&cur, &mut cache.kv[l])?,
            };
            trace("mixer", l, &mixed);
            res_hc = layer.hc_attn.combine(&res_hc, &mixed, &inject)?;
            let (cur, inject) = layer.hc_ffn.mix(&res_hc)?;
            let moe_out = layer.moe.forward(&cur)?;
            trace("moe", l, &moe_out);
            res_hc = layer.hc_ffn.combine(&res_hc, &moe_out, &inject)?;
            if obs.wants::<LayerOutputPoint>(l) {
                let (collapsed, _) = layer.hc_ffn.mix(&res_hc)?;
                obs.at::<LayerOutputPoint>(l, &collapsed)?;
            }
        }

        // Only the last position feeds the head.
        let last = res_hc.narrow(1, seq - 1, 1)?;
        let (collapsed, _) = self.hc_head.mix(&last)?; // [1, 1, d]
        cache.pos += seq;
        let logits = self.lm_head.forward(&collapsed)?;
        logits
            .reshape((1, self.cfg.vocab_size))?
            .to_dtype(lloom_tensor::DType::F32)
    }

    /// One position from its embedding `h [1, 1, hidden]` → logits `[1, vocab]`.
    pub fn decode_embed<O: ForwardObserver>(
        &self,
        h: &Tensor,
        cache: &mut Cache,
        obs: &mut O,
    ) -> Result<Tensor> {
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;

        // broadcast the embedding into the n-stream HC carry `res_hc [1, 1, n*d]`.
        // The reference inits the streams from the embed broadcast across n.
        let mut res_hc = h
            .broadcast_as((1, 1, n, d))?
            .reshape((1, 1, n * d))?
            .contiguous()?;

        for (l, layer) in self.layers.iter().enumerate() {
            // The N-gram PLE adds to the carry at the start of its layer.
            if let Some(ple) = &self.ple {
                if ple.layer == l {
                    let delta = self.ple_delta(ple, &res_hc, cache)?;
                    trace("ple", l, &delta);
                    res_hc = (res_hc + delta)?;
                }
            }

            // observe the n-stream carry entering the layer (HcState) — zero cost
            // on the crown path.
            if obs.wants::<HcStatePoint>(l) {
                obs.at::<HcStatePoint>(l, &res_hc)?;
            }

            // HC prologue (attn bracket): collapse streams → mixer input.
            let (cur, inject) = layer.hc_attn.mix(&res_hc)?; // cur [1,1,d]
            trace("attn.mix", l, &cur);
            trace("attn.inject", l, &inject);
            let mixed = match &layer.mixer {
                Mixer::Linear(dn) => {
                    dn.decode(&cur, l, &mut cache.s_state, &mut cache.conv_state)?
                }
                Mixer::Full(qsa) => qsa.decode(&cur, &mut cache.kv[l])?,
            };
            trace("mixer", l, &mixed);
            // HC epilogue: scatter the mixer output back across the streams.
            res_hc = layer.hc_attn.combine(&res_hc, &mixed, &inject)?;
            trace("attn.combine", l, &res_hc);

            // HC prologue (ffn bracket) → MoE → HC epilogue.
            let (cur, inject) = layer.hc_ffn.mix(&res_hc)?;
            trace("ffn.mix", l, &cur);
            let moe_out = layer.moe.decode(&cur)?;
            trace("moe", l, &moe_out);
            res_hc = layer.hc_ffn.combine(&res_hc, &moe_out, &inject)?;
            trace("ffn.combine", l, &res_hc);

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
        logits
            .reshape((1, self.cfg.vocab_size))?
            .to_dtype(lloom_tensor::DType::F32)
    }

    /// The context past which the full-attention layers are sparse (the
    /// indexer's `budget + ratio − 1`), `None` when no layer carries an indexer.
    pub fn dense_limit(&self) -> Option<usize> {
        self.layers.iter().find_map(|l| match &l.mixer {
            Mixer::Full(q) => q.dense_limit(),
            Mixer::Linear(_) => None,
        })
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

impl ModelTrait for Qwen4Exp<Resident> {
    type Config = Config;

    /// The checkpoint is GGUF (a non-uniform per-tensor quant); the safetensors
    /// `VarBuilder` path has no weights to offer. Load with
    /// [`Qwen4Exp::from_gguf`](super::loader).
    fn from_weights(_vb: VarBuilder, _cfg: &Self::Config) -> Result<Self> {
        lloom_tensor::bail!("qwen4exp loads from GGUF: use Qwen4Exp::from_gguf")
    }
}

impl CausalLM for Qwen4Exp<Resident> {
    type Cache = Cache;

    fn new_cache(&self, cfg: &Self::Config, _dev: &lloom_tensor::Device) -> Result<Self::Cache> {
        Ok(Cache::new(cfg))
    }

    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        Qwen4Exp::embed(self, tokens)
    }

    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        if pos != cache.pos {
            lloom_tensor::bail!("qwen4exp: pos {pos} != cache position {}", cache.pos)
        }
        Qwen4Exp::forward_embeds(self, embeds, cache, &mut ())
    }

    /// Embed and run; the token ids are recorded first so the PLE hash reads
    /// them (the trait default would run `forward_embeds` with no ids).
    fn forward(&self, tokens: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        let ids: Vec<u32> = tokens.flatten_all()?.to_vec1::<u32>()?;
        cache.set_pending_tokens(&ids);
        let embeds = self.embed(tokens)?;
        <Self as CausalLM>::forward_embeds(self, &embeds, pos, cache)
    }

    fn vocab_size(&self) -> usize {
        self.cfg.vocab_size
    }
}

impl CacheReset for Cache {
    fn reset(&mut self) {
        Cache::reset(self)
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::LayerKind;
    use super::super::deltanet_block::GatedDeltaNet;
    use super::super::hc::HyperConnection;
    use super::super::moe::{MoeLayer, Resident};
    use super::super::qsa::QsaAttention;
    use super::*;
    use crate::nn::RmsNorm;
    use lloom_tensor::quantized::{GgmlDType, QTensor};
    use lloom_tensor::{DType, Device};
    use std::sync::Arc;

    /// A stacked `[e, out, inp]` expert weight as an F32-typed `QTensor`.
    fn stacked(e: usize, out: usize, inp: usize, seed: f32, dev: &Device) -> Arc<QTensor> {
        let v: Vec<f32> = (0..e * out * inp)
            .map(|i| ((i as f32 * 0.011 + seed).cos()) * 0.1)
            .collect();
        let t = Tensor::from_vec(v, (e, out, inp), dev).unwrap();
        Arc::new(QTensor::quantize(&t, GgmlDType::F32).unwrap())
    }

    fn lin(out: usize, inp: usize, seed: f32, dev: &Device) -> QMatMul {
        let v: Vec<f32> = (0..out * inp)
            .map(|i| ((i as f32 * 0.007 + seed).sin()) * 0.08)
            .collect();
        QMatMul::Tensor(Tensor::from_vec(v, (out, inp), dev).unwrap())
    }
    fn ones(n: usize, dev: &Device) -> Tensor {
        Tensor::ones(n, DType::F32, dev).unwrap()
    }
    fn wt(shape: (usize, usize), seed: f32, dev: &Device) -> Tensor {
        let (r, c) = shape;
        Tensor::from_vec(
            (0..r * c)
                .map(|i| ((i as f32 * 0.009 + seed).cos()) * 0.05)
                .collect::<Vec<f32>>(),
            (r, c),
            dev,
        )
        .unwrap()
    }

    fn hc(cfg: &Config, seed: f32, dev: &Device) -> HyperConnection {
        let hc_dim = cfg.hc_dim();
        HyperConnection::new(
            ones(hc_dim, dev),                                // norm_w
            lin(cfg.hc_lowrank, hc_dim, seed, dev),           // down
            lin(hc_dim, cfg.hc_lowrank, seed + 1.0, dev),     // up
            Some(lin(cfg.hc_count, hc_dim, seed + 2.0, dev)), // inject → n
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
                stacked(ne, ff, d, 1.0, dev),
                stacked(ne, ff, d, 100.0, dev),
                stacked(ne, d, ff, 200.0, dev),
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
        let embed = QMatMul::Tensor(wt((cfg.vocab_size, cfg.hidden_size), 0.5, &dev));

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
            hc(&cfg, 999.0, &dev),                           // hc_head
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

        // The chunk prefill path agrees with the replayed decode path on the
        // whole model (state channels and logits), with the PLE attached to a
        // dense table so its chunk path is exercised too.
        {
            let ple_layer = 1usize;
            let hash = super::super::ple::NgramHash::new(
                vec![1_000_003, 999_983, 7],
                vec![13, 17, 19, 23],
                vec![0, 13, 30, 49],
                2,
                cfg.eos_token_id,
            )
            .unwrap();
            let rows = hash.total_rows() as usize;
            let head_dim = cfg.ple_head_dim;
            let ple_dim = head_dim * hash.n_heads();
            let table = super::super::ple::PleTable::Dense(wt((rows, head_dim), 3.0, &dev));
            let hc_dim = cfg.hc_dim();
            let block = super::super::ple::PleLayer::new(
                lin(hc_dim, ple_dim, 21.0, &dev),
                lin(cfg.hidden_size, ple_dim, 22.0, &dev),
                ones(hc_dim, &dev),
                ones(hc_dim, &dev),
                ones(hc_dim, &dev),
                wt((hc_dim, 4), 23.0, &dev),
                cfg.hc_count,
                cfg.hidden_size,
                cfg.ple_ngram_size,
                cfg.norm_eps as f32,
            )
            .unwrap();
            let mut with_ple = Qwen4Exp::new(
                QMatMul::Tensor(wt((cfg.vocab_size, cfg.hidden_size), 0.5, &dev)),
                (0..cfg.n_layers)
                    .map(|l| {
                        let mixer = match cfg.layer_types[l] {
                            LayerKind::Linear => Mixer::Linear(deltanet(&cfg, &dev)),
                            LayerKind::Full => Mixer::Full(qsa(&cfg, &dev)),
                        };
                        Layer::new(
                            hc(&cfg, l as f32, &dev),
                            mixer,
                            hc(&cfg, l as f32 + 50.0, &dev),
                            moe(&cfg, &dev),
                        )
                    })
                    .collect(),
                hc(&cfg, 999.0, &dev),
                lin(cfg.vocab_size, cfg.hidden_size, 7.0, &dev),
                cfg.clone(),
            );
            with_ple.set_ple(ple_layer, hash, table, block).unwrap();
            let toks: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
            let mut run = |mode: PrefillMode| -> (Vec<f32>, Vec<f32>, usize) {
                with_ple.set_prefill_mode(mode);
                let mut cache = Cache::new(&cfg);
                let input = Tensor::from_vec(toks.clone(), (1, toks.len()), &dev).unwrap();
                let logits = with_ple.forward(&input, 0, &mut cache).unwrap();
                // one decode step after the prompt, so the carried state matters
                let next = Tensor::from_vec(vec![4u32], (1, 1), &dev).unwrap();
                let logits2 = with_ple.forward(&next, cache.pos, &mut cache).unwrap();
                (
                    logits.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    logits2.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                    cache.pos,
                )
            };
            let (ra, rb, rpos) = run(PrefillMode::Replay);
            let (ca, cb, cpos) = run(PrefillMode::Chunk);
            assert_eq!(rpos, toks.len() + 1);
            assert_eq!(cpos, rpos);
            for (name, r, c) in [("prompt", &ra, &ca), ("next", &rb, &cb)] {
                let scale = r.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-3);
                let md = r
                    .iter()
                    .zip(c)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0f32, f32::max);
                assert!(
                    md <= 1e-4 * scale,
                    "{name} logits: chunk vs replay differ by {md} at scale {scale}"
                );
            }
        }

        // a second token advances state (both the DeltaNet S/conv and QSA KV).
        let id2 = Tensor::from_vec(vec![5u32], (1, 1), &dev).unwrap();
        let logits2 = model.decode(&id2, &mut cache, &mut ()).unwrap();
        assert_eq!(logits2.dims(), &[1, cfg.vocab_size]);
        assert_eq!(cache.pos, 2);
        // the QSA layer's KV grew to 2.
        let full_layer = cfg
            .layer_types
            .iter()
            .position(|k| *k == LayerKind::Full)
            .unwrap();
        assert_eq!(cache.kv[full_layer].len(), 2);
    }
}
