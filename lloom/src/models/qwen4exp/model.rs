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
use super::ple::{NgramHash, NgramWindow, PleLayer, PleTable};
use super::qsa::{KvCache, KvPool, QsaAttention};
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

    pub fn mixer(&self) -> &Mixer {
        &self.mixer
    }

    pub fn moe(&self) -> &MoeLayer<R> {
        &self.moe
    }

    pub fn moe_mut(&mut self) -> &mut MoeLayer<R> {
        &mut self.moe
    }

    /// The `[.., d]` collapse of a carry through this layer's ffn bracket —
    /// the per-layer value the observer reports (`LayerOutputPoint`).
    pub fn collapsed(&self, res_hc: &Tensor) -> Result<Tensor> {
        Ok(self.hc_ffn.mix(res_hc)?.0)
    }

    /// One position through the layer: `res_hc [1, 1, n·d]` → the new carry,
    /// plus the token-mixer's and the MoE's outputs (the non-finite watch reads
    /// them). State channel / KV slot `l` of `cache`.
    pub fn decode_parts(
        &self,
        res_hc: &Tensor,
        l: usize,
        cache: &mut Cache,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (a, b, c, _) = self.decode_parts_planned(res_hc, l, cache, false)?;
        Ok((a, b, c))
    }

    /// [`Self::decode_parts`], the MoE's routed experts planned on the
    /// device when `device_plan` (`MoeLayer::decode_planned`); the plan's
    /// record, when one was made, is the caller's to settle at the end of
    /// the pass.
    pub fn decode_parts_planned(
        &self,
        res_hc: &Tensor,
        l: usize,
        cache: &mut Cache,
        device_plan: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        self.decode_parts_rows(res_hc, l, &mut [cache], device_plan)
    }

    /// [`Self::decode_parts_planned`] over `B` rows: `res_hc [B, 1, n·d]`,
    /// row `r` against `caches[r]`. The HC brackets and the MoE take the
    /// rows as one (`m = B`); the mixer runs its state per row
    /// (`GatedDeltaNet::decode_rows`, `QsaAttention::decode_rows`).
    pub fn decode_parts_rows(
        &self,
        res_hc: &Tensor,
        l: usize,
        caches: &mut [&mut Cache],
        device_plan: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        // HC prologue (attn bracket): collapse streams → mixer input.
        let (cur, inject) = self.hc_attn.mix(res_hc)?; // cur [B,1,d]
        trace("attn.mix", l, &cur);
        trace("attn.inject", l, &inject);
        let mixed = match &self.mixer {
            Mixer::Linear(dn) => {
                let mut states: Vec<(&mut RecurrentState, &mut RecurrentState)> = caches
                    .iter_mut()
                    .map(|c| (&mut c.s_state, &mut c.conv_state))
                    .collect();
                dn.decode_rows(&cur, l, &mut states)?
            }
            Mixer::Full(qsa) => {
                let mut kvs: Vec<&mut KvCache> = caches.iter_mut().map(|c| &mut c.kv[l]).collect();
                qsa.decode_rows(&cur, &mut kvs)?
            }
        };
        trace("mixer", l, &mixed);
        // HC epilogue: scatter the mixer output back across the streams.
        let res_hc = self.hc_attn.combine(res_hc, &mixed, &inject)?;
        trace("attn.combine", l, &res_hc);

        // HC prologue (ffn bracket) → MoE → HC epilogue.
        let (cur, inject) = self.hc_ffn.mix(&res_hc)?;
        trace("ffn.mix", l, &cur);
        let (moe_out, record) = if device_plan {
            self.moe.decode_planned(&cur)?
        } else {
            (self.moe.decode(&cur)?, None)
        };
        trace("moe", l, &moe_out);
        let res_hc = self.hc_ffn.combine(&res_hc, &moe_out, &inject)?;
        trace("ffn.combine", l, &res_hc);
        Ok((res_hc, mixed, moe_out, record))
    }

    /// `T` positions of ONE sequence through the layer as one pass (a
    /// verify window): the brackets and the MoE over the `T` rows, the
    /// mixer's state advanced position by position into its phase ring
    /// (`GatedDeltaNet::decode_seq`, `QsaAttention::decode_seq`). Returns
    /// the carry, the MoE's plan record, and — on a full-attention layer —
    /// the KV lengths after every position.
    pub fn decode_parts_seq(
        &self,
        res_hc: &Tensor,
        l: usize,
        cache: &mut Cache,
        device_plan: bool,
    ) -> Result<SeqParts> {
        let (cur, inject) = self.hc_attn.mix(res_hc)?; // [T, 1, d]
        let (mixed, lens) = match &self.mixer {
            Mixer::Linear(dn) => (
                dn.decode_seq(&cur, l, &mut cache.s_state, &mut cache.conv_state)?,
                None,
            ),
            Mixer::Full(qsa) => {
                let (m, lens) = qsa.decode_seq(&cur, &mut cache.kv[l])?;
                (m, Some(lens))
            }
        };
        let res_hc = self.hc_attn.combine(res_hc, &mixed, &inject)?;
        let (cur, inject) = self.hc_ffn.mix(&res_hc)?;
        let (moe_out, record) = if device_plan {
            self.moe.decode_planned(&cur)?
        } else {
            (self.moe.decode(&cur)?, None)
        };
        let res_hc = self.hc_ffn.combine(&res_hc, &moe_out, &inject)?;
        Ok((res_hc, record, lens))
    }

    /// One position through the layer ([`Self::decode_parts`]'s carry).
    pub fn decode(&self, res_hc: &Tensor, l: usize, cache: &mut Cache) -> Result<Tensor> {
        Ok(self.decode_parts(res_hc, l, cache)?.0)
    }

    /// A chunk through the layer: `res_hc [1, seq, n·d]` → the new carry
    /// (DeltaNet Kernel B / dense chunk attention, per-row MoE).
    pub fn prefill(&self, res_hc: &Tensor, l: usize, cache: &mut Cache) -> Result<Tensor> {
        let (cur, inject) = self.hc_attn.mix(res_hc)?; // [1, seq, d]
        let mixed = match &self.mixer {
            Mixer::Linear(dn) => dn.prefill(&cur, l, &mut cache.s_state, &mut cache.conv_state)?,
            Mixer::Full(qsa) => qsa.prefill(&cur, &mut cache.kv[l])?,
        };
        trace("mixer", l, &mixed);
        let res_hc = self.hc_attn.combine(res_hc, &mixed, &inject)?;
        let (cur, inject) = self.hc_ffn.mix(&res_hc)?;
        let moe_out = self.moe.forward(&cur)?;
        trace("moe", l, &moe_out);
        self.hc_ffn.combine(&res_hc, &moe_out, &inject)
    }
}

/// What `Layer::decode_parts_seq` returns: the carry, the MoE's device-plan
/// record, and — on a full-attention layer — the KV lengths after every
/// position.
pub type SeqParts = (Tensor, Option<Tensor>, Option<Vec<(usize, usize)>>);

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
    /// `pending_tokens` was filled by the driver (`CausalLM::note_input_ids`)
    /// for the NEXT forward, which consumes the note instead of reading its
    /// input ids back from the device. Cleared by that forward.
    pub pending_noted: bool,
    /// Decode position (number of tokens seen).
    pub pos: usize,
    /// Retain every forward's pre-final-mixer carry `[1, seq, n·d]` (the
    /// `NextnHidden` probe an MTP drafter reads) until [`Self::take_nextn`].
    /// Off, a forward keeps nothing.
    pub want_nextn: bool,
    nextn: Vec<Tensor>,
    /// Retained carries a verify pass found untaken and discarded.
    nextn_drops: u64,
    /// The open speculative window's pre-window bookkeeping, if any.
    window: Option<CacheWindow>,
    /// Decode passes replayed because a device-planned routed expert was
    /// not resident (`ExpertResidency::settle`) — the streamed tier's miss
    /// cost under the device plan.
    pub redos: u64,
    /// Whether the next decode pass plans on the device: armed by a pass
    /// that completed with every expert resident, disarmed by a pass that
    /// missed (the cold phase runs the host plan — one token and its loads,
    /// no replay — and the warm phase the device plan).
    pub device_plan_armed: bool,
    /// Decode passes that ran the host plan while the device plan was
    /// disarmed (the cold phase's length, in tokens).
    pub host_plan_passes: u64,
    /// The `(pos, seq)` the last `prepare_pass` sized and admitted, so the
    /// same pass prepared twice (driver, then `forward`'s chunk loop) reads
    /// the device once.
    prepared: Option<(usize, usize)>,
}

/// What a speculative window must put back besides the recurrent channels'
/// own snapshots: per-layer KV lengths, the n-gram window, the pending token
/// ids and the position.
#[derive(Debug, Clone)]
struct CacheWindow {
    kv: Vec<(usize, usize)>,
    ngram: NgramWindow,
    pending_tokens: std::collections::VecDeque<u32>,
    pos: usize,
    /// A verify window's bookkeeping per position: the tokens fed and every
    /// layer's KV lengths after each position (`Qwen4Exp::verify_window`),
    /// for an accept by position (`Cache::accept_window`).
    tokens: Vec<u32>,
    kv_after: Vec<Vec<(usize, usize)>>,
}

impl Cache {
    pub fn new(cfg: &Config) -> Self {
        Self::with_layers(cfg, cfg.n_layers)
    }

    /// A cache whose DeltaNet `S` and conv window live in `pools` (a batch's
    /// shared, slot-indexed state — `StatePool`s from
    /// [`Qwen4Exp::state_pools`]); a slot of each is leased here and freed
    /// when the cache drops. The PLE history and the KV stay the cache's own.
    pub fn in_pools(cfg: &Config, pools: &StatePools) -> Result<Self> {
        let mut cache = Self::new(cfg);
        cache.s_state = RecurrentState::in_pool(&pools.s)?;
        cache.conv_state = RecurrentState::in_pool(&pools.conv)?;
        if let Some(p) = &pools.ple {
            cache.ple_conv = RecurrentState::in_pool(p)?;
        }
        if let Some(p) = &pools.kv {
            let lease = p.lease()?;
            for (l, kv) in cache.kv.iter_mut().enumerate() {
                *kv = KvCache::in_pool(p, &lease, l);
            }
        }
        Ok(cache)
    }

    /// Whether a verify window of `steps` positions can be accepted by
    /// position on this cache: every recurrent channel pooled with a ring of
    /// `steps + 1` phases (`Qwen4Exp::spec_pools`).
    pub fn ring_holds(&self, steps: usize, with_ple: bool) -> bool {
        self.s_state.ring_holds(steps)
            && self.conv_state.ring_holds(steps)
            && (!with_ple || self.ple_conv.ring_holds(steps))
    }

    /// Accept the first `a` positions of the open verify window and close
    /// it: every recurrent channel to phase `base + a`, every KV to its
    /// length after position `a`, the n-gram window and the position as
    /// after `a` tokens, the retained carries cut to `a`. `a = 0` is the
    /// rewind. The verify's positions past `a` cost nothing.
    pub fn accept_window(&mut self, a: usize) -> Result<()> {
        let w = match self.window.take() {
            Some(w) => w,
            None => lloom_tensor::bail!("qwen4exp cache: no speculative window is open"),
        };
        if a > w.tokens.len() {
            lloom_tensor::bail!(
                "qwen4exp cache: accepting {a} of a {}-position window",
                w.tokens.len()
            )
        }
        self.s_state.accept_window(a)?;
        self.conv_state.accept_window(a)?;
        if self.ple_conv.is_pooled() {
            self.ple_conv.accept_window(a)?;
        } else if a == w.tokens.len() {
            self.ple_conv.resolve_window()?;
        } else {
            self.ple_conv.rewind_to_window()?;
        }
        let lens: &[(usize, usize)] = if a == 0 { &w.kv } else { &w.kv_after[a - 1] };
        for (k, &(tokens, blocks)) in self.kv.iter_mut().zip(lens) {
            k.truncate(tokens, blocks);
        }
        self.ngram = w.ngram;
        for &t in &w.tokens[..a] {
            self.ngram.push(Some(t));
        }
        self.pending_tokens = w.pending_tokens;
        self.pending_noted = false;
        self.pos = w.pos + a;
        if self.nextn.len() > a {
            self.nextn.truncate(a);
        }
        Ok(())
    }

    /// A cache for `n_layers` layers of `cfg`'s geometry (an MTP head's single
    /// layer shares the model's shapes, not its depth).
    pub fn with_layers(cfg: &Config, n_layers: usize) -> Self {
        Self {
            s_state: RecurrentState::new(n_layers),
            conv_state: RecurrentState::new(n_layers),
            kv: vec![KvCache::default(); n_layers],
            ple_conv: RecurrentState::new(n_layers),
            ngram: NgramWindow::new(cfg.ple_ngram_size.saturating_sub(1)),
            pending_tokens: std::collections::VecDeque::new(),
            pending_noted: false,
            pos: 0,
            want_nextn: false,
            nextn: Vec::new(),
            nextn_drops: 0,
            window: None,
            redos: 0,
            device_plan_armed: false,
            host_plan_passes: 0,
            prepared: None,
        }
    }

    pub fn reset(&mut self) {
        self.prepared = None;
        self.s_state.reset();
        self.conv_state.reset();
        for k in &mut self.kv {
            k.reset();
        }
        self.ple_conv.reset();
        self.ngram.reset();
        self.pending_tokens.clear();
        self.pending_noted = false;
        self.pos = 0;
        self.nextn.clear();
        self.window = None;
    }

    /// The carries retained since the last take, concatenated along the
    /// sequence: `[1, positions, n·d]`, or `None` if nothing was retained.
    pub fn take_nextn(&mut self) -> Result<Option<Tensor>> {
        let rows = std::mem::take(&mut self.nextn);
        match rows.len() {
            0 => Ok(None),
            1 => Ok(rows.into_iter().next()),
            _ => Ok(Some(Tensor::cat(&rows, 1)?)),
        }
    }

    fn keep_nextn(&mut self, res_hc: &Tensor) {
        if self.want_nextn {
            self.nextn.push(res_hc.clone());
        }
    }

    /// Discard whatever is retained, counting it as a dropped probe.
    pub(crate) fn drop_nextn(&mut self) {
        if !self.nextn.is_empty() {
            self.nextn_drops += 1;
            self.nextn.clear();
        }
    }

    /// Retained carries discarded untaken so far (`Probes::probe_drops`).
    pub fn nextn_drops(&self) -> u64 {
        self.nextn_drops
    }

    /// Open a speculative window: snapshot every state channel and the
    /// lengths a verify pass will advance, so a partially-accepted window can
    /// be put back (§10.2a). One window at a time.
    pub fn open_window(&mut self) -> Result<()> {
        if self.window.is_some() {
            lloom_tensor::bail!("qwen4exp cache: a speculative window is already open")
        }
        self.s_state.open_window()?;
        self.conv_state.open_window()?;
        self.ple_conv.open_window()?;
        self.window = Some(CacheWindow {
            kv: self.kv.iter().map(|k| k.lens()).collect(),
            ngram: self.ngram.clone(),
            pending_tokens: self.pending_tokens.clone(),
            pos: self.pos,
            tokens: Vec::new(),
            kv_after: Vec::new(),
        });
        Ok(())
    }

    /// Put the cache back to where the window was opened: the recurrent
    /// channels from their snapshots, the KV caches by length, the n-gram
    /// window, the pending ids and the position. The caller replays the
    /// accepted prefix afterwards (the decode recurrence is replayable, so a
    /// checkpoint + replay is the rollback, §10.2a).
    pub fn rewind_window(&mut self) -> Result<()> {
        let w = match self.window.take() {
            Some(w) => w,
            None => lloom_tensor::bail!("qwen4exp cache: no speculative window is open"),
        };
        self.s_state.rewind_to_window()?;
        self.conv_state.rewind_to_window()?;
        self.ple_conv.rewind_to_window()?;
        for (k, &(tokens, blocks)) in self.kv.iter_mut().zip(&w.kv) {
            k.truncate(tokens, blocks);
        }
        self.ngram = w.ngram;
        self.pending_tokens = w.pending_tokens;
        self.pending_noted = false;
        self.pos = w.pos;
        Ok(())
    }

    /// Close the window keeping everything the pass wrote.
    pub fn resolve_window(&mut self) -> Result<()> {
        if self.window.take().is_none() {
            lloom_tensor::bail!("qwen4exp cache: no speculative window is open")
        }
        self.s_state.resolve_window()?;
        self.conv_state.resolve_window()?;
        self.ple_conv.resolve_window()?;
        Ok(())
    }

    pub fn window_open(&self) -> bool {
        self.window.is_some()
    }

    /// A checksum (FNV over the f32 bit patterns) and a non-finite count per
    /// state channel and layer, for a probe comparing two runs' states: which
    /// channel diverges or goes non-finite first. Reads every state back — an
    /// instrument, never on the decode path.
    pub fn checksums(&self) -> Result<Vec<(String, u64, usize)>> {
        fn fnv(t: &Tensor) -> Result<(u64, usize)> {
            let v = t
                .flatten_all()?
                .to_dtype(lloom_tensor::DType::F32)?
                .to_vec1::<f32>()?;
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
        self.pending_noted = false;
    }
}

/// Whether every decode position checks each layer's mixer and MoE output for
/// non-finite values and stops at the first (`QWEN4EXP_WATCH_NAN`, read once):
/// one small reduce per sub-block and one readback per token — an instrument
/// behind a gate (§3.3a), never the shipping path.
fn watch_nan() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("QWEN4EXP_WATCH_NAN").is_some())
}

/// A `[1]` f32 that is finite iff every element of `t` is: `sum(x − x)`
/// (`x − x` is 0 for finite `x` and NaN for an inf or NaN — the subtraction
/// of a tensor from itself is the point).
#[allow(clippy::eq_op)]
fn nonfinite_probe(t: &Tensor) -> Result<Tensor> {
    let f = t.to_dtype(lloom_tensor::DType::F32)?;
    (&f - &f)?.sum_all()?.reshape(1)
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
    /// The MTP head, when loaded (`LoadOptions::mtp`); an `MtpDrafter`
    /// borrows it beside `embed_tokens` and `lm_head`.
    mtp: Option<super::mtp::MtpHead<R>>,
    prefill_mode: PrefillMode,
    /// Some layer's residency plans on the device (`Streamed`): a decode
    /// pass runs under a window and settles at its end (`decode_embed`).
    device_plan: bool,
    /// The per-pass memory check (`prepare_pass`, §9.5p's shape): the
    /// fraction of `recommendedMaxWorkingSetSize` a pass may take the
    /// process to, and whether a pass over it is refused or reported.
    memory_fraction: f64,
    mem_check: MemCheck,
    cfg: Config,
}

/// What `prepare_pass` does with a pass the budget cannot take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemCheck {
    /// No device read, no check (the pre-#749-P18 behaviour).
    Off,
    /// Read and report on stderr (`MEMCHECK` lines), never refuse — the arm
    /// the transient term is measured under (`PREFILL_TRANSIENT_BYTES_PER_TOKEN`).
    Report,
    /// Refuse the pass with the arithmetic, before any state moves.
    Refuse,
}

impl MemCheck {
    /// `QWEN4EXP_MEM` = `off` | `report` | `refuse`, read once; `refuse`
    /// by default — flipped from `report` with the transient term measured
    /// (`PREFILL_TRANSIENT_BYTES_PER_TOKEN`, DESIGN §7.1a).
    pub fn from_env() -> Self {
        static ARM: std::sync::OnceLock<MemCheck> = std::sync::OnceLock::new();
        *ARM.get_or_init(|| match std::env::var("QWEN4EXP_MEM").ok().as_deref() {
            Some("off") => MemCheck::Off,
            Some("report") => MemCheck::Report,
            Some("refuse") | None => MemCheck::Refuse,
            Some(other) => {
                eprintln!("QWEN4EXP_MEM={other}: off | report | refuse; using refuse");
                MemCheck::Refuse
            }
        })
    }
}

/// Bytes one position of a multi-token pass adds to the process's Metal
/// allocation while the pass is in flight — the activation class of §9.5n
/// for this family, per token, over the whole layer stack (the class sums
/// over layers because a buffer returns only when its command buffer
/// completes, §6.3b).
///
/// **Measured 2026-09-23** (issue #749 P18, `qwen4exp-dispatches-raw/
/// p18-stream-long1.txt`): a 937-token prompt in 512-token tiles paced at
/// depth 2, `--experts stream`, the gauge sampling at 10 ms — the pool's
/// `pending` peaks at **1.34 GB** and `device_allocated` at **+1.51 GB**
/// over the load, with two tiles in flight: **~2.6 MB a position**. Rounded
/// up to 3 MB. A pass is admitted against `PREFILL_TRANSIENT_BYTES_PER_TOKEN
/// × seq` plus its KV grow. Re-measure when the prefill path's allocation
/// shape moves (a fused chunk kernel, `--moe-act f16`, a wider tile).
pub const PREFILL_TRANSIENT_BYTES_PER_TOKEN: usize = 3 * 1000 * 1000;

/// What a pass of `seq` positions at `pos` would allocate before it runs:
/// the KV grow (slab or pool) and the transient term. Sized, not performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassPlan {
    pub pos: usize,
    pub seq: usize,
    /// Bytes the QSA layers' KV grows by for this pass (0 when every
    /// layer's slab or pool already holds `pos + seq`).
    pub kv_grow: usize,
    /// `PREFILL_TRANSIENT_BYTES_PER_TOKEN × seq` for `seq > 1`, else 0.
    pub transient: usize,
}

impl PassPlan {
    pub fn bytes(&self) -> usize {
        self.kv_grow + self.transient
    }
}

impl<R: ExpertResidency> Qwen4Exp<R> {
    pub fn new(
        embed_tokens: QMatMul,
        layers: Vec<Layer<R>>,
        hc_head: HyperConnection,
        lm_head: QMatMul,
        cfg: Config,
    ) -> Self {
        let device_plan = layers.iter().any(|l| l.moe().experts().plans_on_device());
        Self {
            embed_tokens,
            layers,
            hc_head,
            lm_head,
            ple: None,
            mtp: None,
            prefill_mode: PrefillMode::default(),
            device_plan,
            memory_fraction: super::loader::DEFAULT_MEMORY_FRACTION,
            mem_check: MemCheck::from_env(),
            cfg,
        }
    }

    /// The fraction of `recommendedMaxWorkingSetSize` a pass may take the
    /// process to (`prepare_pass`); the load's own (`LoadOptions::
    /// memory_fraction`) by default. 0 turns the per-pass check off.
    pub fn set_memory_fraction(&mut self, fraction: f64) {
        self.memory_fraction = fraction.max(0.0);
    }

    pub fn memory_fraction(&self) -> f64 {
        self.memory_fraction
    }

    /// What a pass the budget cannot take does (the env's arm by default).
    pub fn set_mem_check(&mut self, arm: MemCheck) {
        self.mem_check = arm;
    }

    pub fn mem_check(&self) -> MemCheck {
        self.mem_check
    }

    /// Size the session allocations a pass of `seq` positions from
    /// `cache.pos` would make: every full-attention layer's KV grow (a slab
    /// widened, or the batch pool's next page slab) and the transient term.
    /// No side effects.
    pub fn plan_pass(&self, cache: &Cache, seq: usize) -> PassPlan {
        let (nkv, hd) = (self.cfg.n_kv_heads, self.cfg.head_dim);
        let elem = lloom_tensor::DType::F32.size_in_bytes();
        let mut kv_grow = 0usize;
        for (l, layer) in self.layers.iter().enumerate() {
            if matches!(layer.mixer, Mixer::Full(_)) {
                kv_grow += cache.kv[l].grow_bytes_for(seq, nkv, hd, elem);
            }
        }
        let transient = if seq > 1 {
            PREFILL_TRANSIENT_BYTES_PER_TOKEN.saturating_mul(seq)
        } else {
            0
        };
        PassPlan {
            pos: cache.pos,
            seq,
            kv_grow,
            transient,
        }
    }

    /// `Model::prepare_pass` for this family (§9.5p's per-pass device
    /// read): size the pass, and when it allocates — a KV grow, a
    /// multi-token pass with a measured transient — read the device and
    /// refuse (or report) BEFORE anything moves: `currentAllocatedSize +
    /// bytes ≤ memory_fraction × recommendedMaxWorkingSetSize`. A decode
    /// step inside its slab reads nothing (§9.5e: no per-token Metal call).
    /// Idempotent per `(pos, seq)`: a driver prepares, then `forward`'s own
    /// chunk loop prepares the same pass and finds it done.
    pub fn prepare_pass_inner(&self, cache: &mut Cache, seq: usize) -> Result<()> {
        if seq == 0 || cache.prepared == Some((cache.pos, seq)) {
            return Ok(());
        }
        let plan = self.plan_pass(cache, seq);
        cache.prepared = Some((cache.pos, seq));
        if plan.bytes() == 0 || self.mem_check == MemCheck::Off || self.memory_fraction <= 0.0 {
            return Ok(());
        }
        #[cfg(feature = "metal")]
        if let Ok(m) = self.device().as_metal_device() {
            use crate::models::lfm2::{kv_grow_budget, kv_grow_fits};
            let dev = m.device();
            let allocated = dev.current_allocated_size();
            let working_set = dev.recommended_max_working_set_size();
            let fits = kv_grow_fits(allocated, working_set, self.memory_fraction, plan.bytes());
            let gb = |b: usize| b as f64 / 1e9;
            let line = format!(
                "pos={} seq={} kv_grow={:.3} GB transient={:.3} GB allocated={:.2} GB budget={:.2} GB ({:.2} x working_set {:.2} GB) fits={fits}",
                plan.pos,
                plan.seq,
                gb(plan.kv_grow),
                gb(plan.transient),
                gb(allocated),
                gb(kv_grow_budget(working_set, self.memory_fraction)),
                self.memory_fraction,
                gb(working_set),
            );
            match self.mem_check {
                MemCheck::Report => eprintln!("MEMCHECK {line}"),
                MemCheck::Refuse if !fits => {
                    cache.prepared = None;
                    lloom_tensor::bail!(
                        "qwen4exp pass refused {line}: the pass stops here, before anything is \
                         written, rather than risk the machine (DESIGN.md 9.5, 3.5): a smaller \
                         prefill tile, `--memory-fraction`, free memory, or bound the run."
                    )
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Attach the MTP head.
    pub fn set_mtp(&mut self, head: super::mtp::MtpHead<R>) {
        self.mtp = Some(head);
    }

    pub fn mtp_head(&self) -> Option<&super::mtp::MtpHead<R>> {
        self.mtp.as_ref()
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
        if let Some(m) = &mut self.mtp {
            m.layer_mut().moe_mut().set_routed_act(dtype);
        }
    }

    pub fn prefill_mode(&self) -> PrefillMode {
        self.prefill_mode
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The device the weights live on.
    pub fn device(&self) -> lloom_tensor::Device {
        match &self.lm_head {
            QMatMul::QTensor(q) => q.device(),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => t.device().clone(),
        }
    }

    /// The token embedding (`token_embd`), shared with an MTP drafter.
    pub fn embed_tokens(&self) -> &QMatMul {
        &self.embed_tokens
    }

    /// The `lm_head`, shared with an MTP drafter.
    pub fn lm_head(&self) -> &QMatMul {
        &self.lm_head
    }

    pub fn layers(&self) -> &[Layer<R>] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [Layer<R>] {
        &mut self.layers
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

    /// The PLE hot-row cache's `(hits, misses)`, if a cached table is attached.
    pub fn ple_cache_stats(&self) -> Option<(u64, u64)> {
        match &self.ple {
            Some(p) => match &p.table {
                PleTable::Cached(_) => Some(p.table.cache_stats()),
                _ => None,
            },
            None => None,
        }
    }

    /// Every layer's expert-residency counters summed (`Streamed` only).
    pub fn stream_stats(&self) -> crate::nn::expert_stream::StreamStats
    where
        R: crate::nn::expert_stream::HasStreamStats,
    {
        let mut s = crate::nn::expert_stream::StreamStats::default();
        for l in &self.layers {
            s += l.moe().experts().stream_stats();
        }
        s
    }

    /// The staging rings' counters summed, a ring shared across layers
    /// counted once (`Streamed` on Metal; zeros otherwise).
    pub fn staging_stats(&self) -> crate::nn::expert_stream::StagingStats
    where
        R: crate::nn::expert_stream::HasStreamStats,
    {
        let mut t = crate::nn::expert_stream::StagingStats::default();
        let mut seen = Vec::new();
        for l in &self.layers {
            let e = l.moe().experts();
            if let Some(id) = e.staging_id() {
                if seen.contains(&id) {
                    continue;
                }
                seen.push(id);
            }
            let s = e.staging_stats();
            t.buffers += s.buffers;
            t.bytes += s.bytes;
            t.allocations += s.allocations;
            t.reuses += s.reuses;
            t.overflows += s.overflows;
        }
        t
    }

    /// Every layer's expert usage counts (`Streamed`; empty rows for
    /// `Resident`) — the profile a hotlist is made of.
    pub fn stream_usage(&self) -> Vec<Vec<u32>>
    where
        R: crate::nn::expert_stream::HasStreamStats,
    {
        self.layers
            .iter()
            .map(|l| l.moe().experts().usage())
            .collect()
    }

    /// Warm every streamed layer's pool from a hotlist — the head of each
    /// layer's profile into its free slots (`Streamed::warm`; the MTP head's
    /// layer too when it is streamed, from the profile's last row). Returns
    /// the experts loaded.
    pub fn warm_from_hotlist(&self, hot: &super::hotlist::Hotlist) -> Result<usize>
    where
        R: crate::nn::expert_stream::HasStreamStats,
    {
        let mut loaded = 0;
        for (l, layer) in self.layers.iter().enumerate() {
            let e = layer.moe().experts();
            let n = e.slots();
            if n == 0 {
                continue;
            }
            loaded += e.warm(&hot.top(l, n))?;
        }
        Ok(loaded)
    }

    /// The streamed tier's prefetch counters (shared by its layers; zeros
    /// for `Resident` or without a prefetcher).
    pub fn prefetch_stats(&self) -> crate::nn::expert_stream::PrefetchStats {
        self.layers
            .first()
            .map(|l| l.moe().experts().prefetch_stats())
            .unwrap_or_default()
    }

    /// Slots per layer per projection (`Streamed`), 0 for `Resident`.
    pub fn slots_per_layer(&self) -> usize
    where
        R: crate::nn::expert_stream::HasStreamStats,
    {
        self.layers
            .first()
            .map(|l| l.moe().experts().slots())
            .unwrap_or(0)
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
                    // Each chunk is a pass of its own: admitted here when the
                    // driver handed the whole prompt over untiled.
                    self.prepare_pass_inner(cache, len)?;
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

        // The streamed tier's prefetch (§20.3 P1): layer 0's likely misses
        // before the stack starts, every layer's the moment the layer
        // before it is encoded — the reads overlap that layer's compute.
        if let Some(first) = self.layers.first() {
            first.moe().experts().prefetch_ahead()?;
        }
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(next) = self.layers.get(l + 1) {
                next.moe().experts().prefetch_ahead()?;
            }
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
            res_hc = layer.prefill(&res_hc, l, cache)?;
            if obs.wants::<LayerOutputPoint>(l) {
                obs.at::<LayerOutputPoint>(l, &layer.collapsed(&res_hc)?)?;
            }
        }
        cache.keep_nextn(&res_hc);

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
    ///
    /// Under the streamed tier's device plan (`QWEN4EXP_PLAN=device`, the
    /// default; `host` is the per-layer readback): the pass runs with no
    /// host sync at any layer, under a speculative window; at its end every
    /// layer's plan is settled at once (the one readback) and, if any layer
    /// computed on an expert that was not resident, the window is rewound
    /// and the pass replayed — the experts are resident now, the replay is
    /// the same math on the same bytes (`ExpertResidency::settle`). Off when
    /// a window is already open (an MTP verify) or an observer taps a layer
    /// (a replay would tap twice).
    pub fn decode_embed<O: ForwardObserver>(
        &self,
        h: &Tensor,
        cache: &mut Cache,
        obs: &mut O,
    ) -> Result<Tensor> {
        let planned = self.device_plan
            && plan_on_device()
            && !cache.window_open()
            && !(0..self.layers.len())
                .any(|l| obs.wants::<HcStatePoint>(l) || obs.wants::<LayerOutputPoint>(l));
        if !planned {
            return self.decode_embed_pass(h, cache, obs, false).map(|(l, _)| l);
        }
        // The cold phase: a pool that missed on the last pass runs the host
        // plan — one token and its loads, no replay — until a pass completes
        // with every expert resident; then the device plan, until it misses.
        if !cache.device_plan_armed {
            let before = self.stream_misses();
            let logits = self
                .decode_embed_pass(h, cache, obs, false)
                .map(|(l, _)| l)?;
            cache.host_plan_passes += 1;
            if self.stream_misses() == before {
                cache.device_plan_armed = true;
            }
            return Ok(logits);
        }
        // A pass that computed on a missing expert at layer L was right up to
        // L only (its ids past L came from a wrong carry), so a device-planned
        // replay could miss again one layer further, up to the layer count of
        // replays on a cold pool. The replay is therefore the HOST plan — the
        // per-layer readback loads each layer's true experts as it reaches
        // them: one replay per missed token, and the experts it loaded are
        // resident for the tokens after it.
        cache.open_window()?;
        let (logits, records) = self.decode_embed_pass(h, cache, obs, true)?;
        let mut redo = false;
        for (l, rec) in &records {
            if self.layers[*l].moe().experts().settle(rec)? {
                redo = true;
            }
        }
        if !redo {
            cache.resolve_window()?;
            return Ok(logits);
        }
        cache.rewind_window()?;
        cache.redos += 1;
        cache.device_plan_armed = false;
        self.decode_embed_pass(h, cache, obs, false).map(|(l, _)| l)
    }

    /// The streamed tier's misses so far, summed over the layers (0 for
    /// `Resident`): the cold-phase probe.
    fn stream_misses(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.moe().experts().misses_so_far())
            .sum()
    }

    fn decode_embed_pass<O: ForwardObserver>(
        &self,
        h: &Tensor,
        cache: &mut Cache,
        obs: &mut O,
        device_plan: bool,
    ) -> Result<(Tensor, Vec<(usize, Tensor)>)> {
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;
        let mut records: Vec<(usize, Tensor)> = Vec::new();

        // broadcast the embedding into the n-stream HC carry `res_hc [1, 1, n*d]`.
        // The reference inits the streams from the embed broadcast across n.
        let mut res_hc = h
            .broadcast_as((1, 1, n, d))?
            .reshape((1, 1, n * d))?
            .contiguous()?;

        let watch = watch_nan();
        let mut watched: Vec<Tensor> = Vec::new();
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

            let (next, mixed, moe_out, record) =
                layer.decode_parts_planned(&res_hc, l, cache, device_plan)?;
            if let Some(r) = record {
                records.push((l, r));
            }
            if watch {
                watched.push(nonfinite_probe(&mixed)?);
                watched.push(nonfinite_probe(&moe_out)?);
            }
            res_hc = next;

            // observe the collapsed layer output (d-wide) — the DSpark/MTP tap side.
            if obs.wants::<LayerOutputPoint>(l) {
                obs.at::<LayerOutputPoint>(l, &layer.collapsed(&res_hc)?)?;
            }
        }
        cache.keep_nextn(&res_hc);

        if watch && !watched.is_empty() {
            // One readback per token: the first non-finite (layer, sub-block).
            let flags = Tensor::cat(&watched, 0)?.to_vec1::<f32>()?;
            if let Some(i) = flags.iter().position(|f| !f.is_finite()) {
                lloom_tensor::bail!(
                    "NONFINITE pos={} layer={} block={}",
                    cache.pos,
                    i / 2,
                    if i % 2 == 0 { "mixer" } else { "moe" }
                )
            }
        }

        // Final HC collapse (carries the output norm) → lm_head.
        let (collapsed, _) = self.hc_head.mix(&res_hc)?; // [1, 1, d]
        cache.pos += 1;
        let logits = self.lm_head.forward(&collapsed)?; // [1, 1, vocab]
        let logits = logits
            .reshape((1, self.cfg.vocab_size))?
            .to_dtype(lloom_tensor::DType::F32)?;
        Ok((logits, records))
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

impl<R: ExpertResidency> ModelTrait for Qwen4Exp<R> {
    type Config = Config;

    /// The checkpoint is GGUF (a non-uniform per-tensor quant); the safetensors
    /// `VarBuilder` path has no weights to offer. Load with
    /// [`Qwen4Exp::from_gguf`](super::loader).
    fn from_weights(_vb: VarBuilder, _cfg: &Self::Config) -> Result<Self> {
        lloom_tensor::bail!("qwen4exp loads from GGUF: use Qwen4Exp::from_gguf")
    }
}

impl<R: ExpertResidency> CausalLM for Qwen4Exp<R> {
    type Cache = Cache;

    fn new_cache(&self, cfg: &Self::Config, _dev: &lloom_tensor::Device) -> Result<Self::Cache> {
        Ok(Cache::new(cfg))
    }

    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> Result<()> {
        self.prepare_pass_inner(cache, seq_len)
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
    /// them (the trait default would run `forward_embeds` with no ids). The
    /// ids come from the driver's note when it holds them on the host
    /// ([`CausalLM::note_input_ids`] — the blocking arm, a prompt chunk) and
    /// are read back from `tokens` otherwise (a ring arm's device-resident
    /// token: correct, and a wait on the queue's tail per step).
    fn forward(&self, tokens: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        let seq = tokens.elem_count();
        if cache.pending_noted && cache.pending_tokens.len() == seq {
            cache.pending_noted = false;
        } else {
            let ids: Vec<u32> = tokens.flatten_all()?.to_vec1::<u32>()?;
            cache.set_pending_tokens(&ids);
        }
        let embeds = self.embed(tokens)?;
        <Self as CausalLM>::forward_embeds(self, &embeds, pos, cache)
    }

    fn note_input_ids(&self, cache: &mut Self::Cache, ids: &[u32]) -> Result<()> {
        cache.set_pending_tokens(ids);
        cache.pending_noted = true;
        Ok(())
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

impl<R: ExpertResidency> Qwen4Exp<R> {
    /// One decode step for `B` independent sequences (`BatchedCausalLM::
    /// forward_rows`): `tokens [B, 1]`, row `r` fed to `rows[r]` at its
    /// position, → `[B, vocab]`. The weight-bearing ops run once over all
    /// rows (the HC brackets' gemvs, the mixers' stems and out-projs, the
    /// MoE's routed and shared experts, the head); the state runs per row
    /// against that row's cache through the kernels a row of one sees
    /// (`Layer::decode_parts_rows`), so a row computes what `forward` of
    /// that token against that cache computes. The streamed tier's device
    /// plan runs when every row is armed for it: a window per row, one
    /// settle for all, and a miss in any row replays the whole step as the
    /// host plan (`decode_embed`'s discipline over the rows).
    pub fn forward_rows(
        &self,
        tokens: &Tensor,
        rows: &mut [crate::nn::model::Row<'_, Cache>],
    ) -> Result<Tensor> {
        let (b, seq) = tokens.dims2()?;
        if seq != 1 {
            lloom_tensor::bail!(
                "qwen4exp forward_rows: one token per row, got seq_len {seq}; prefill each \
                 sequence through `forward`"
            )
        }
        if b != rows.len() || b == 0 {
            lloom_tensor::bail!(
                "qwen4exp forward_rows: {b} token rows for {} caches",
                rows.len()
            )
        }
        for (r, row) in rows.iter().enumerate() {
            if row.pos != row.cache.pos {
                lloom_tensor::bail!(
                    "qwen4exp forward_rows: row {r} at pos {} but its cache is at {}",
                    row.pos,
                    row.cache.pos
                )
            }
            if row.cache.window_open() {
                lloom_tensor::bail!("qwen4exp forward_rows: row {r} has an open speculative window")
            }
        }
        // The ids, for the PLE's hash per row (one readback; the driver's
        // note covers only the single-sequence path).
        let ids: Vec<u32> = tokens.flatten_all()?.to_vec1::<u32>()?;
        for (row, &id) in rows.iter_mut().zip(&ids) {
            row.cache.set_pending_tokens(&[id]);
        }
        let h = self.embed(tokens)?; // [B, 1, hidden]
        let mut caches: Vec<&mut Cache> = rows.iter_mut().map(|r| &mut *r.cache).collect();
        let planned =
            self.device_plan && plan_on_device() && caches.iter().all(|c| c.device_plan_armed);
        if !planned {
            let before = self.stream_misses();
            let logits = self.decode_rows_pass(&h, &mut caches, false)?.0;
            let clean = self.stream_misses() == before;
            for c in caches.iter_mut() {
                c.host_plan_passes += 1;
                if clean {
                    c.device_plan_armed = true;
                }
            }
            return Ok(logits);
        }
        for c in caches.iter_mut() {
            c.open_window()?;
        }
        let (logits, records) = self.decode_rows_pass(&h, &mut caches, true)?;
        let mut redo = false;
        for (l, rec) in &records {
            if self.layers[*l].moe().experts().settle(rec)? {
                redo = true;
            }
        }
        if !redo {
            for c in caches.iter_mut() {
                c.resolve_window()?;
            }
            return Ok(logits);
        }
        for c in caches.iter_mut() {
            c.rewind_window()?;
            c.redos += 1;
            c.device_plan_armed = false;
        }
        Ok(self.decode_rows_pass(&h, &mut caches, false)?.0)
    }

    /// The layer stack over `B` rows from their embeddings `h [B, 1,
    /// hidden]` → logits `[B, vocab]` and the device plans' records.
    fn decode_rows_pass(
        &self,
        h: &Tensor,
        caches: &mut [&mut Cache],
        device_plan: bool,
    ) -> Result<(Tensor, Vec<(usize, Tensor)>)> {
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;
        let b = h.dim(0)?;
        let mut records: Vec<(usize, Tensor)> = Vec::new();
        let mut res_hc = h
            .reshape((b, 1, 1, d))?
            .broadcast_as((b, 1, n, d))?
            .reshape((b, 1, n * d))?
            .contiguous()?;
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(ple) = &self.ple {
                if ple.layer == l {
                    // The PLE per row: each row's own token window and history.
                    let mut deltas = Vec::with_capacity(b);
                    for (r, cache) in caches.iter_mut().enumerate() {
                        deltas.push(self.ple_delta(ple, &res_hc.narrow(0, r, 1)?, cache)?);
                    }
                    let delta = if b == 1 {
                        deltas.pop().expect("one row")
                    } else {
                        Tensor::cat(&deltas, 0)?
                    };
                    res_hc = (res_hc + delta)?;
                }
            }
            let (next, _, _, record) = layer.decode_parts_rows(&res_hc, l, caches, device_plan)?;
            if let Some(r) = record {
                records.push((l, r));
            }
            res_hc = next;
        }
        for (r, cache) in caches.iter_mut().enumerate() {
            cache.keep_nextn(&res_hc.narrow(0, r, 1)?);
            cache.pos += 1;
        }
        let (collapsed, _) = self.hc_head.mix(&res_hc)?; // [B, 1, d]
        let logits = self.lm_head.forward(&collapsed)?; // [B, 1, vocab]
        let logits = logits
            .reshape((b, self.cfg.vocab_size))?
            .to_dtype(lloom_tensor::DType::F32)?;
        Ok((logits, records))
    }
}

/// The pools a batch's rows keep their recurrent state in: the DeltaNet
/// `S` (`[d_k, d_v, h_v]` a row) and the conv window (`[conv_channels,
/// kernel]` a row), one slot per row over every layer.
#[derive(Debug, Clone)]
pub struct StatePools {
    pub s: crate::nn::StatePool,
    pub conv: crate::nn::StatePool,
    /// The PLE's dilated-conv history, when a PLE is attached.
    pub ple: Option<crate::nn::StatePool>,
    /// The QSA layers' KV in one page pool with per-row tables (§12.2), for
    /// a batch of rows; `None` keeps every row's own slab (a single row).
    pub kv: Option<KvPool>,
}

impl<R: ExpertResidency> Qwen4Exp<R> {
    /// The pools for `rows` rows on `dev` (f32, the fused kernels' dtype),
    /// two phases (a decode's current and next).
    pub fn state_pools(&self, rows: usize, dev: &lloom_tensor::Device) -> StatePools {
        self.state_pools_with(rows, 2, dev)
    }

    /// The pools for a speculative window of `k` drafts: `k + 2` phases, so
    /// the `k + 1` verify positions' states are all held and an accept is
    /// a phase.
    pub fn spec_pools(&self, rows: usize, k: usize, dev: &lloom_tensor::Device) -> StatePools {
        self.state_pools_with(rows, k + 2, dev)
    }

    pub fn state_pools_with(
        &self,
        rows: usize,
        phases: usize,
        dev: &lloom_tensor::Device,
    ) -> StatePools {
        let dims = self.cfg.deltanet_dims();
        let f32 = lloom_tensor::DType::F32;
        let n = self.layers.len();
        StatePools {
            s: crate::nn::StatePool::with_phases(
                rows,
                phases,
                n,
                &[dims.d_k, dims.d_v, dims.h_v],
                f32,
                dev,
            ),
            conv: crate::nn::StatePool::with_phases(
                rows,
                phases,
                n,
                &[self.cfg.conv_channels(), self.cfg.ssm_conv_kernel],
                f32,
                dev,
            ),
            ple: self.ple.as_ref().map(|p| {
                crate::nn::StatePool::with_phases(
                    rows,
                    phases,
                    n,
                    &[self.cfg.hc_dim(), p.block.history_len()],
                    f32,
                    dev,
                )
            }),
            kv: (rows > 1)
                .then(|| KvPool::new(rows, n, self.cfg.n_kv_heads, self.cfg.head_dim, f32, dev)),
        }
    }

    /// A speculative verify window: the `T` tokens `[anchor, proposed…]`
    /// through the model as ONE pass — the weights swept once (the HC
    /// brackets, the stems, the MoE's shared and routed experts, the head
    /// over `T` rows), the state advanced position by position with every
    /// intermediate kept in the channels' phase rings — under an open
    /// window; returns each position's argmax. The caller accepts a prefix
    /// with [`Cache::accept_window`] (no replay). Every step is the
    /// arithmetic of a decode step (the row gemvs, the per-position state
    /// kernels), so the argmaxes are the sequential decode's. Needs
    /// `cache.ring_holds(T, ple)`.
    pub fn verify_window(&self, tokens: &[u32], cache: &mut Cache) -> Result<Vec<u32>> {
        let t = tokens.len();
        if t == 0 {
            lloom_tensor::bail!("qwen4exp verify_window: no tokens")
        }
        if !cache.ring_holds(t, self.ple.is_some()) {
            lloom_tensor::bail!(
                "qwen4exp verify_window: {t} positions need pooled channels with {} phases \
                 (`Cache::in_pools` over `spec_pools`)",
                t + 1
            )
        }
        cache.open_window()?;
        cache.set_pending_tokens(tokens);
        if let Some(w) = &mut cache.window {
            w.tokens = tokens.to_vec();
        }
        let dev = self.device();
        if !dev.is_metal() {
            // The reference path: the positions one by one through the
            // decode step (the CPU's gemm sums in an order that depends on
            // the row count, so a rows pass would not be the sequential
            // math there); the ring channels take the writes phase by phase.
            let mut argmax = Vec::with_capacity(t);
            let base_lens: Vec<(usize, usize)> = cache.kv.iter().map(|k| k.lens()).collect();
            let mut kv_after = Vec::with_capacity(t);
            let _ = base_lens;
            for &tok in tokens {
                let input = Tensor::from_vec(vec![tok], (1, 1), &dev)?;
                let logits = self
                    .decode_embed_pass(&self.embed(&input)?, cache, &mut (), false)?
                    .0;
                argmax.push(
                    logits
                        .flatten_all()?
                        .argmax(D::Minus1)?
                        .to_scalar::<u32>()?,
                );
                kv_after.push(cache.kv.iter().map(|k| k.lens()).collect::<Vec<_>>());
            }
            if let Some(w) = &mut cache.window {
                w.kv_after = kv_after;
            }
            return Ok(argmax);
        }
        let input = Tensor::from_vec(tokens.to_vec(), (t, 1), &dev)?;
        let h = self.embed(&input)?; // [T, 1, hidden]
        let planned = self.device_plan && plan_on_device() && cache.device_plan_armed;
        let (logits, records) = self.verify_pass(&h, cache, planned)?;
        let mut logits = logits;
        if planned {
            let mut redo = false;
            for (l, rec) in &records {
                if self.layers[*l].moe().experts().settle(rec)? {
                    redo = true;
                }
            }
            if redo {
                // The host plan over the same window from the base.
                let tokens_v = cache
                    .window
                    .as_ref()
                    .map(|w| w.tokens.clone())
                    .unwrap_or_default();
                cache.accept_window(0)?;
                cache.redos += 1;
                cache.device_plan_armed = false;
                cache.open_window()?;
                cache.set_pending_tokens(&tokens_v);
                if let Some(w) = &mut cache.window {
                    w.tokens = tokens_v;
                }
                logits = self.verify_pass(&h, cache, false)?.0;
            }
        }
        let argmax = logits.argmax(D::Minus1)?.to_vec1::<u32>()?; // [T]
        Ok(argmax)
    }

    /// The layer stack over the window's `T` positions → logits `[T, vocab]`
    /// and the device plans' records; the KV lengths after each position
    /// recorded in the window.
    fn verify_pass(
        &self,
        h: &Tensor,
        cache: &mut Cache,
        device_plan: bool,
    ) -> Result<(Tensor, Vec<(usize, Tensor)>)> {
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;
        let t = h.dim(0)?;
        let n_layers = self.layers.len();
        let mut records: Vec<(usize, Tensor)> = Vec::new();
        let mut kv_after: Vec<Vec<(usize, usize)>> = (0..t)
            .map(|_| cache.kv.iter().map(|k| k.lens()).collect())
            .collect();
        let mut res_hc = h
            .reshape((t, 1, 1, d))?
            .broadcast_as((t, 1, n, d))?
            .reshape((t, 1, n * d))?
            .contiguous()?;
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(ple) = &self.ple {
                if ple.layer == l {
                    // Per position: its own token window and history phase.
                    let mut deltas = Vec::with_capacity(t);
                    for p in 0..t {
                        deltas.push(self.ple_delta(ple, &res_hc.narrow(0, p, 1)?, cache)?);
                    }
                    let delta = if t == 1 {
                        deltas.pop().expect("one position")
                    } else {
                        Tensor::cat(&deltas, 0)?
                    };
                    res_hc = (res_hc + delta)?;
                }
            }
            let (next, record, lens) = layer.decode_parts_seq(&res_hc, l, cache, device_plan)?;
            if let Some(r) = record {
                records.push((l, r));
            }
            if let Some(lens) = lens {
                for (p, ln) in lens.into_iter().enumerate() {
                    for row in kv_after.iter_mut().skip(p) {
                        row[l] = ln;
                    }
                }
                let _ = n_layers;
            }
            res_hc = next;
        }
        for p in 0..t {
            cache.keep_nextn(&res_hc.narrow(0, p, 1)?);
        }
        cache.pos += t;
        if let Some(w) = &mut cache.window {
            w.kv_after = kv_after;
        }
        let (collapsed, _) = self.hc_head.mix(&res_hc)?; // [T, 1, d]
        let logits = self.lm_head.forward(&collapsed)?; // [T, 1, vocab]
        let logits = logits
            .reshape((t, self.cfg.vocab_size))?
            .to_dtype(lloom_tensor::DType::F32)?;
        Ok((logits, records))
    }
}

impl<R: ExpertResidency> crate::nn::model::BatchedCausalLM for Qwen4Exp<R> {
    fn forward_rows(
        &self,
        tokens: &Tensor,
        rows: &mut [crate::nn::model::Row<'_, Self::Cache>],
    ) -> Result<Tensor> {
        Qwen4Exp::forward_rows(self, tokens, rows)
    }
}

/// No prefix sharing between rows yet: every row prefills its own prompt
/// (the recurrent channels have no page to share by reference).
impl crate::nn::model::CacheShare for Cache {
    fn share_prefix_from(&mut self, _src: &Self, _n_tokens: usize) -> Result<usize> {
        Ok(0)
    }
    fn share_granularity(&self) -> usize {
        0
    }
}

/// Whether the streamed tier's routed experts are planned on the device
/// (`QWEN4EXP_PLAN`, read once: `device` — the default — or `host`, the
/// per-layer readback; the A/B arm).
fn plan_on_device() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("QWEN4EXP_PLAN")
            .map(|v| v != "host")
            .unwrap_or(true)
    })
}

/// No rewindable arm yet: the recurrent channels roll back only through a
/// speculative window (`open_window`/`rewind_window`, the MTP verify's), so a
/// driver that overran an EOS (a ring arm's lookahead) resets the sequence
/// rather than rewinding it (`Session::prepare`), and a request that reuses
/// a prefix extends an exactly-ending cache (the blocking arm's shape).
impl crate::nn::model::CacheRewind for Cache {
    fn rewind_horizon(&self) -> usize {
        0
    }

    fn rewind(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        lloom_tensor::bail!(
            "qwen4exp cache: no rewindable arm (horizon 0); a rewind of {n} positions is refused"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::LayerKind;
    use super::super::moe::Resident;
    use super::super::testing::{deltanet, hc, lin, moe, ones, qsa, wt};
    use super::*;
    use lloom_tensor::quantized::QMatMul;
    use lloom_tensor::Device;

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
