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
use super::moe::{ExpertResidency, MoeLayer, WindowPlan, WindowPlanKind, WindowRecord};
use super::ple::{NgramHash, NgramWindow, PleLayer, PleTable};
use super::qsa::{KvCache, KvPool, QsaAttention};
use super::trace;
use crate::nn::model::{CacheReset, CausalLM, Model as ModelTrait};
use crate::nn::observer::{ForwardObserver, HcStatePoint, LayerOutputPoint};
use crate::nn::var_builder::VarBuilder;
use crate::nn::RecurrentState;
#[cfg(feature = "metal")]
use lloom_tensor::metal_backend::{PassKey, PassKind, PassOpener};
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
        self.hc_ffn.collapse(res_hc)
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
        let (a, b, c, r) = self.decode_parts_rows_with(
            res_hc,
            l,
            &mut [cache],
            RowPlan::of(device_plan),
            &mut (),
        )?;
        Ok((a, b, c, r.and_then(WindowPlan::into_device)))
    }

    /// [`Self::decode_parts_planned`] with `hook` at the MoE's join points
    /// (`moe::PlanHook`) when the routed experts are planned on the device.
    pub fn decode_parts_planned_with<H: super::moe::PlanHook>(
        &self,
        res_hc: &Tensor,
        l: usize,
        cache: &mut Cache,
        device_plan: bool,
        hook: &mut H,
    ) -> Result<(Tensor, Tensor, Tensor, Option<Tensor>)> {
        let (a, b, c, r) =
            self.decode_parts_rows_with(res_hc, l, &mut [cache], RowPlan::of(device_plan), hook)?;
        Ok((a, b, c, r.and_then(WindowPlan::into_device)))
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
        let (a, b, c, r) =
            self.decode_parts_rows_with(res_hc, l, caches, RowPlan::of(device_plan), &mut ())?;
        Ok((a, b, c, r.and_then(WindowPlan::into_device)))
    }

    fn decode_parts_rows_with<H: super::moe::PlanHook>(
        &self,
        res_hc: &Tensor,
        l: usize,
        caches: &mut [&mut Cache],
        plan: RowPlan,
        hook: &mut H,
    ) -> Result<(Tensor, Tensor, Tensor, Option<WindowPlan>)> {
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
        let (moe_out, record) = match plan {
            RowPlan::Host => (self.moe.decode(&cur)?, None),
            RowPlan::Device => {
                let (o, r) = self.moe.decode_planned_with(&cur, hook)?;
                (o, r.map(WindowPlan::Device))
            }
            RowPlan::WindowRow { real, loaded } => self.moe.decode_window(
                &cur,
                WindowPlanKind::Host {
                    real_rows: real as usize,
                    loaded_rows: (real || loaded) as usize,
                },
            )?,
        };
        trace("moe", l, &moe_out);
        let res_hc = self.hc_ffn.combine(&res_hc, &moe_out, &inject)?;
        trace("ffn.combine", l, &res_hc);
        Ok((res_hc, mixed, moe_out, record))
    }

    /// One window position through the layer, its routed experts planned as
    /// `RowPlan::WindowRow` (row 0 of a verify window loaded and counted, a
    /// draft row loaded beside it or resolved against what is resident):
    /// the carry and the plan record (the verify's reference path, position
    /// by position).
    pub fn decode_parts_window_row(
        &self,
        res_hc: &Tensor,
        l: usize,
        cache: &mut Cache,
        real: bool,
        loaded: bool,
    ) -> Result<(Tensor, Option<WindowPlan>)> {
        let (a, _, _, r) = self.decode_parts_rows_with(
            res_hc,
            l,
            &mut [cache],
            RowPlan::WindowRow { real, loaded },
            &mut (),
        )?;
        Ok((a, r))
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
        plan: super::moe::WindowPlanKind,
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
        let (moe_out, record) = self.moe.decode_window(&cur, plan)?;
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
        let (res_hc, cur, inject) = self.prefill_mix(res_hc, l, cache)?;
        let moe_out = self.moe.forward(&cur)?;
        self.prefill_finish(&res_hc, &moe_out, &inject, l)
    }

    /// [`Self::prefill`] with the MoE planned as a window (`MoeLayer::
    /// decode_window`): every row real, the plan's record returned for the
    /// caller to settle after its readback (the MTP head's observe).
    pub fn prefill_window(
        &self,
        res_hc: &Tensor,
        l: usize,
        cache: &mut Cache,
        kind: WindowPlanKind,
    ) -> Result<(Tensor, Option<WindowPlan>)> {
        let (res_hc, cur, inject) = self.prefill_mix(res_hc, l, cache)?;
        let (moe_out, record) = self.moe.decode_window(&cur, kind)?;
        Ok((self.prefill_finish(&res_hc, &moe_out, &inject, l)?, record))
    }

    /// The mixer half of [`Self::prefill`]: the carry after the mixer's
    /// bracket, and the FFN bracket's `(cur, inject)` for the MoE. The
    /// pipelined prefill (`Qwen4Exp::prefill_embeds_pair`) runs this, the
    /// MoE's route, then its plan and execute, for two tiles interleaved.
    pub fn prefill_mix(
        &self,
        res_hc: &Tensor,
        l: usize,
        cache: &mut Cache,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (cur, inject) = self.hc_attn.mix(res_hc)?; // [1, seq, d]
        let mixed = match &self.mixer {
            Mixer::Linear(dn) => dn.prefill(&cur, l, &mut cache.s_state, &mut cache.conv_state)?,
            Mixer::Full(qsa) => qsa.prefill(&cur, &mut cache.kv[l])?,
        };
        trace("mixer", l, &mixed);
        let res_hc = self.hc_attn.combine(res_hc, &mixed, &inject)?;
        let (cur, inject) = self.hc_ffn.mix(&res_hc)?;
        Ok((res_hc, cur, inject))
    }

    /// The other half: the FFN bracket's combine over the MoE's output.
    pub fn prefill_finish(
        &self,
        res_hc: &Tensor,
        moe_out: &Tensor,
        inject: &Tensor,
        l: usize,
    ) -> Result<Tensor> {
        trace("moe", l, moe_out);
        self.hc_ffn.combine(res_hc, moe_out, inject)
    }
}

/// How one decode position's routed experts are planned
/// (`Layer::decode_parts_rows_with`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RowPlan {
    /// The host plan: the ids read back at the layer, every row's misses
    /// loaded.
    Host,
    /// The device plan: no readback, the record settled after the pass.
    Device,
    /// A verify window's position on the host plan: `real` (the next
    /// token: loaded, counted), a draft `loaded` beside it (loaded, counted
    /// only if accepted) or a draft resolved against what is resident (a
    /// miss flagged) — `MoeLayer::decode_window`.
    WindowRow { real: bool, loaded: bool },
}

impl RowPlan {
    fn of(device_plan: bool) -> Self {
        if device_plan {
            RowPlan::Device
        } else {
            RowPlan::Host
        }
    }
}

/// A verify window's pass, run and read back, awaiting its accept
/// (`Qwen4Exp::verify_window_run` → `verify_window_settle`).
#[derive(Debug)]
pub struct VerifyPass {
    /// Each window position's argmax.
    pub argmax: Vec<u32>,
    /// Per layer (in layer order), the plan record over the window's rows.
    records: Vec<(usize, WindowRecord)>,
    /// The window ran on the device plan.
    pub planned: bool,
    t0: std::time::Instant,
    tokens: Vec<u32>,
    /// The window's embeddings, for a redo (Metal; the reference path
    /// re-embeds).
    h: Option<Tensor>,
    /// A host-planned window that loaded for its draft rows too.
    drafts_loaded: bool,
}

/// What a settled verify window is worth (`Qwen4Exp::verify_window_settle`).
#[derive(Debug, Clone)]
pub struct WindowVerdict {
    /// Each position's argmax (the redo's, if one ran).
    pub argmax: Vec<u32>,
    /// The rows that stand: the accept's count, cut at the first flagged
    /// draft row.
    pub accepted: usize,
    pub planned: bool,
    pub redo: bool,
    pub first_flagged: Option<usize>,
    /// Flags on the rows past `accepted`, over the layers.
    pub rejected_flags: u64,
}

/// What `Layer::decode_parts_seq` returns: the carry, the MoE's plan record
/// (device or host, settled after the accept), and — on a full-attention
/// layer — the KV lengths after every position.
pub type SeqParts = (
    Tensor,
    Option<super::moe::WindowPlan>,
    Option<Vec<(usize, usize)>>,
);

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
    /// The PLE's host fill (`CausalLM::host_feed`).
    host: HostFill,
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
    /// The open window is the STANDING one: the last decode step's, kept
    /// open after the step so the step can be undone (`CacheRewind`, horizon
    /// = the tokens fed since it opened — one). Settled (resolved) by the
    /// next forward or by any `open_window`; invisible to `window_open()`,
    /// which answers for a verify's window only.
    standing: bool,
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
    /// Host nanoseconds in the replays (`redos`) and in the disarmed
    /// host-plan passes (`host_plan_passes`): the miss regime's two pass
    /// kinds, timed only when they run.
    pub replay_ns: u64,
    pub host_pass_ns: u64,
    /// Experts a settle loaded for layers past the first layer that missed:
    /// their ids were routed from a carry already wrong at that layer.
    pub settle_after_first: u64,
    /// Verify windows whose accepted prefix was cut at a draft row that
    /// computed on a missing expert (no redo: the rows before it stand).
    pub window_truncations: u64,
    /// Flags on the rows a verify rejected, summed over layers: the
    /// wrong-path misses no settle loaded (`expert-streaming.md` §20.16).
    pub window_rejected_flags: u64,
    /// Verify windows run: host-planned, device-planned, and the device
    /// plan's redos (a flag on row 0, the real next token) — `redos`
    /// counts those too.
    pub window_host: u64,
    pub window_device: u64,
    pub window_redos: u64,
    /// Whether the next host-planned verify window loads its drafts' misses
    /// beside row 0's (`QWEN4EXP_MTP_DRAFT_LOADS=auto`): on while the real
    /// row's loads a window (`window_real_loads_ema`, a moving average over
    /// the host-planned windows) say the pool is missing the stream.
    pub window_draft_loads: bool,
    /// The real row's loads per host-planned window, averaged (×1/4 the
    /// last window, ×3/4 the average before it): the regime the drafts'
    /// loads follow — ≥ 2 turns them on, < 1 turns them off.
    pub window_real_loads_ema: f32,
    /// The `(pos, seq)` the last `prepare_pass` sized and admitted, so the
    /// same pass prepared twice (driver, then `forward`'s chunk loop) reads
    /// the device once.
    prepared: Option<(usize, usize)>,
}

/// A sequence's PLE host fill: the run in progress, if a decode loop armed
/// one — a single-position forward with no noted id then waits on the GPU for
/// rows a producer fills, and `Cache::ngram` stays at the run's first decode
/// step until `host_feed_end` catches it up — and the staging slots and event
/// it runs over, kept across runs. A clone of the cache shares neither: two
/// sequences filling one ring would overwrite each other's rows.
#[derive(Debug, Default)]
struct HostFill {
    run: Option<std::sync::Arc<crate::nn::ngram_memory::FeedLink>>,
    #[cfg(feature = "metal")]
    ring: Option<std::sync::Arc<lloom_tensor::metal_backend::HostFedRing>>,
}

impl Clone for HostFill {
    fn clone(&self) -> Self {
        Self::default()
    }
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
        if pools.kv.is_some() && cfg.kv_precision != super::config::DEFAULT_KV_PRECISION {
            lloom_tensor::bail!(
                "qwen4exp: kv-quant = {:?} has no arm on the rows' paged KV pool (f32): \
                 give every row its own cache (the unpooled rows) or keep kv-quant off",
                cfg.kv_precision
            )
        }
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
            kv: (0..n_layers)
                .map(|_| {
                    KvCache::with_precision(super::qsa::KV_SLAB_TOKENS, cfg.kv_precision)
                        .expect("a KvPrecision's quantized arm is never Off")
                })
                .collect(),
            ple_conv: RecurrentState::new(n_layers),
            ngram: NgramWindow::new(cfg.ple_ngram_size.saturating_sub(1)),
            pending_tokens: std::collections::VecDeque::new(),
            pending_noted: false,
            host: HostFill::default(),
            pos: 0,
            want_nextn: false,
            nextn: Vec::new(),
            nextn_drops: 0,
            window: None,
            standing: false,
            redos: 0,
            device_plan_armed: false,
            host_plan_passes: 0,
            replay_ns: 0,
            host_pass_ns: 0,
            settle_after_first: 0,
            window_truncations: 0,
            window_rejected_flags: 0,
            window_host: 0,
            window_device: 0,
            window_redos: 0,
            window_draft_loads: false,
            window_real_loads_ema: 0.0,
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
        self.host.run = None;
        self.pos = 0;
        self.nextn.clear();
        self.window = None;
        self.standing = false;
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
        self.settle_standing()?;
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
        self.standing = false;
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
        self.standing = false;
        self.s_state.resolve_window()?;
        self.conv_state.resolve_window()?;
        self.ple_conv.resolve_window()?;
        Ok(())
    }

    /// Whether a verify's window is open (a standing one does not count).
    pub fn window_open(&self) -> bool {
        self.window.is_some() && !self.standing
    }

    /// Keep the open window as the standing one: the decode step just run
    /// stays undoable until the next forward settles it (P29 (7), the
    /// `CacheRewind` horizon the driver's ring arm and a multi-turn chat
    /// reconcile through instead of resetting).
    pub fn keep_standing(&mut self) {
        if self.window.is_some() {
            self.standing = true;
        }
    }

    /// Resolve the standing window, if any: the last step is final.
    pub fn settle_standing(&mut self) -> Result<()> {
        if self.standing {
            self.resolve_window()?;
        }
        Ok(())
    }

    /// The tokens the standing window would undo: 0 without one.
    pub fn standing_horizon(&self) -> usize {
        match (&self.window, self.standing) {
            (Some(w), true) => self.pos.saturating_sub(w.pos),
            _ => 0,
        }
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
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.watch_nan)
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

static PREFILL_CHUNK_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// The chunk in force: [`PREFILL_CHUNK`] unless `QWEN4EXP_PREFILL_CHUNK`
/// (read once) or [`set_prefill_chunk`] (tests) says otherwise.
pub fn prefill_chunk() -> usize {
    let v = PREFILL_CHUNK_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed);
    if v != 0 {
        return v;
    }
    let c = match crate::tuning::tuning().qwen4exp.prefill_chunk {
        0 => PREFILL_CHUNK,
        c => c,
    };
    PREFILL_CHUNK_OVERRIDE.store(c, std::sync::atomic::Ordering::Relaxed);
    c
}

pub fn set_prefill_chunk(chunk: usize) {
    PREFILL_CHUNK_OVERRIDE.store(chunk.max(1), std::sync::atomic::Ordering::Relaxed);
}

static PREFILL_PAIR: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(u8::MAX);

/// Whether a multi-chunk prefill runs its chunks two at a time, interleaved
/// a layer apart on the one queue (`DESIGN.md` §11.5z-ix item 2;
/// `QWEN4EXP_PREFILL_PAIR`, `0` = chunk after chunk, read once;
/// [`set_prefill_pair`] for tests). Byte-identical either way: the same
/// dispatches in the same order per tile.
pub fn prefill_pair_on() -> bool {
    let v = PREFILL_PAIR.load(std::sync::atomic::Ordering::Relaxed);
    if v != u8::MAX {
        return v != 0;
    }
    let on = crate::tuning::tuning().qwen4exp.prefill_pair;
    PREFILL_PAIR.store(on as u8, std::sync::atomic::Ordering::Relaxed);
    on
}

pub fn set_prefill_pair(on: bool) {
    PREFILL_PAIR.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

pub const DEFAULT_PREFILL_PAIR: bool = true;

static PREFILL_TILES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Chunks a pipelined pass interleaves (`QWEN4EXP_PREFILL_TILES`, default
/// 2; read once; [`set_prefill_tiles`] for tests). Under
/// [`prefill_pair_on`] only; a read's lead is `tiles − 1` stages.
pub fn prefill_tiles() -> usize {
    let v = PREFILL_TILES.load(std::sync::atomic::Ordering::Relaxed);
    if v != 0 {
        return v;
    }
    let t = crate::tuning::tuning().qwen4exp.prefill_tiles.max(2);
    PREFILL_TILES.store(t, std::sync::atomic::Ordering::Relaxed);
    t
}

pub fn set_prefill_tiles(tiles: usize) {
    PREFILL_TILES.store(tiles.max(2), std::sync::atomic::Ordering::Relaxed);
}

pub const DEFAULT_PREFILL_TILES: usize = 2;

static DECODE_HINTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// `QWEN4EXP_DECODE_HINT=1` (read once; off by default): at decode under
/// the host plan, layer l+2's router on the carry entering layer l names
/// the experts to prefetch (recall 0.56, P27) — the loads at the layer,
/// not the syncs, are what it can cut.
pub fn decode_hint_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.decode_hint)
}

/// The PLE-window instrument's host stamps: a decode step's entry and the
/// cut at its PLE layer, in the clock of the Metal profile's command-buffer
/// records (`lloom_tensor::kernels::metal::profile::now_s`).
#[derive(Debug, Default)]
#[cfg_attr(not(feature = "metal"), allow(dead_code))]
struct PleWindowMarks {
    entry: Option<f64>,
    marks: Vec<(f64, f64)>,
}

/// The N-gram PLE at its one layer.
#[derive(Debug)]
pub struct Ple {
    pub layer: usize,
    pub hash: NgramHash,
    /// Shared with a run's host-fill producer (`CausalLM::host_feed`).
    pub table: std::sync::Arc<PleTable>,
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
    /// The PLE-window instrument, when armed (`arm_ple_window`).
    ple_window: Option<parking_lot::Mutex<PleWindowMarks>>,
    /// Whether [`CausalLM::host_feed`] offers the PLE's host fill to a loop
    /// whose tokens stay on the device (`set_host_fill`).
    host_fill: bool,
    /// The MTP head, when loaded (`LoadOptions::mtp`); an `MtpDrafter`
    /// borrows it beside `embed_tokens` and `lm_head`.
    mtp: Option<super::mtp::MtpHead<R>>,
    prefill_mode: PrefillMode,
    /// Some layer's residency plans on the device (`Streamed`): a decode
    /// pass runs under a window and settles at its end (`decode_embed`).
    device_plan: bool,
    /// The hint-recall probe's tallies per lead k = 0..3: `(predicted ∩
    /// real, real)` over the decode passes (`hint_recall`).
    hint_recall: parking_lot::Mutex<[(u64, u64); 4]>,
    /// The per-pass memory check (`prepare_pass`, §9.5p's shape): the
    /// fraction of `recommendedMaxWorkingSetSize` a pass may take the
    /// process to, and whether a pass over it is refused or reported.
    memory_fraction: f64,
    mem_check: MemCheck,
    /// How the streamed tier's prefetch hint is formed on a prefill pass
    /// (§20.3): 0 = the layer's usage profile (what it does not hold, most
    /// used first); 1 = residual speculation — layer `l+2`'s router on the
    /// carry entering layer `l`, encoded before `l` runs and read back
    /// after it, so the reads overlap layer `l+1` whole (a hint just before
    /// the layer itself measured worse than no hint: the reads were
    /// cancelled by the take, P20). `QWEN4EXP_SPEC_AHEAD` by default.
    spec_ahead: usize,
    /// Spike 0's event (`JoinHook::Wait`), made at first use.
    #[cfg(feature = "metal")]
    spike0: std::sync::OnceLock<Option<std::sync::Arc<lloom_tensor::metal_backend::HostFedRing>>>,
    /// The device plan's settle reads every layer's record back into this
    /// Shared buffer rather than a pool buffer per step (`DESIGN.md` §9.2aa):
    /// grown to the largest record set, kept for the model's life.
    #[cfg(feature = "metal")]
    settle_dst: parking_lot::Mutex<Option<lloom_tensor::kernels::metal::BufferHandle>>,
    /// A one-row decode step's replay input, in memory the model owns rather
    /// than a pool buffer per step (`replay_input_owned`), and the epoch of
    /// the command buffer its last reader was encoded into.
    #[cfg(feature = "metal")]
    replay_h: parking_lot::Mutex<Option<(Tensor, Option<u64>)>>,
    /// A one-position step's PLE rows, gathered on the host into a Shared
    /// buffer the model keeps rather than uploaded per step
    /// (`ple_rows_owned`), and the epoch of its last reader.
    #[cfg(feature = "metal")]
    ple_rows: parking_lot::Mutex<
        Option<(
            lloom_tensor::kernels::metal::BufferHandle,
            Tensor,
            Option<u64>,
        )>,
    >,
    cfg: Config,
}

/// Layers the usage hint leads the running layer by on a prefill pass:
/// `QWEN4EXP_PREFETCH_LEAD`, read once, default 2 — a layer's misses
/// (~330 MB at 512 rows) take ~50 ms at the SSD's eight-thread rate against
/// a layer-tile's ~46 ms of GPU, so one layer of lead stalls (P25).
pub fn prefetch_lead() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| crate::tuning::tuning().stream.prefetch_lead.max(1))
}

/// `QWEN4EXP_SPEC_AHEAD`, read once: 0 | 1 | 2 (see `Qwen4Exp::set_spec_ahead`).
fn spec_ahead_from_env() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| crate::tuning::tuning().qwen4exp.spec_ahead.min(1))
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
        *ARM.get_or_init(|| match crate::tuning::tuning().qwen4exp.mem_check.as_str() {
            "off" => MemCheck::Off,
            "report" => MemCheck::Report,
            // `refuse`; the parser (`tuning.rs`) refused anything else.
            _ => MemCheck::Refuse,
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
            ple_window: None,
            host_fill: true,
            mtp: None,
            prefill_mode: PrefillMode::default(),
            device_plan,
            hint_recall: parking_lot::Mutex::new([(0, 0); 4]),
            memory_fraction: super::loader::DEFAULT_MEMORY_FRACTION,
            mem_check: MemCheck::from_env(),
            spec_ahead: spec_ahead_from_env(),
            #[cfg(feature = "metal")]
            spike0: std::sync::OnceLock::new(),
            #[cfg(feature = "metal")]
            settle_dst: parking_lot::Mutex::new(None),
            #[cfg(feature = "metal")]
            replay_h: parking_lot::Mutex::new(None),
            #[cfg(feature = "metal")]
            ple_rows: parking_lot::Mutex::new(None),
            cfg,
        }
    }

    /// Select the prefill prefetch hint (see the `spec_ahead` field).
    pub fn set_spec_ahead(&mut self, ahead: usize) {
        self.spec_ahead = ahead.min(1);
    }

    pub fn spec_ahead(&self) -> usize {
        self.spec_ahead
    }

    /// The residual-speculation hint for `layer`: its router on `carry`
    /// (the residual stream as it stands, before that layer's mixer) →
    /// the predicted top-k ids on the device, encoded now and read back by
    /// [`Self::hint_land`] after the layer between has run — the host plan
    /// of that layer has synced past this router by then, so the read
    /// waits for nothing. The real routing runs unchanged when the layer
    /// does. `None` for a tier with no device plan (`Resident`).
    fn hint_encode(&self, layer: usize, carry: &Tensor) -> Result<Option<(usize, Tensor)>> {
        let l = &self.layers[layer];
        if !l.moe().experts().plans_on_device() {
            return Ok(None);
        }
        let cur = l.collapsed(carry)?; // [1, seq, d]
        Ok(Some((layer, l.moe().route_ids(&cur)?)))
    }

    /// Open a planned pass for `shape` (`DESIGN.md` §9.2aa: the pass's
    /// activations served by ordinal from one base, no allocation on a
    /// cached key) — when the device has pass plans enabled and no driver
    /// holds a pass already. Returns whether this call opened it. The pass
    /// is keyed as the model's own: it opens after the embedding a driver's
    /// pass of the same shape holds, so the two are different sequences.
    fn open_pass(&self, shape: crate::nn::decode::PassShape) -> Result<bool> {
        self.open_pass_variant(shape, 0)
    }

    /// [`Self::open_pass`] for the allocation sequence `variant` of `shape`
    /// ([`Self::decode_variant`]).
    fn open_pass_variant(&self, shape: crate::nn::decode::PassShape, variant: u32) -> Result<bool> {
        #[cfg(feature = "metal")]
        if let Ok(m) = self.device().as_metal_device() {
            if !m.pass_plans_enabled() || m.pass_open() {
                return Ok(false);
            }
            let Some(key) = Self::pass_key(shape, PassOpener::Model) else {
                return Ok(false);
            };
            m.begin_pass(key.with_variant(variant))?;
            return Ok(true);
        }
        let _ = (shape, variant);
        Ok(false)
    }

    /// Which of this model's decode sequences a step at `pos` runs, as its
    /// pass key's variant (`PassKey::variant`, `DESIGN.md` §9.2aa): bit 0,
    /// the step completes an indexer block (`(pos + 1) % ratio == 0`), which
    /// adds the block's pooling, norm, rope and append; bit 1, the routing
    /// is resolved on the device (the device plan and its settle) rather
    /// than read back per layer. Two steps of one variant allocate the same
    /// ordinals, so each is served its own plan instead of the one recorded
    /// at the other phase drifting on it.
    /// Which of this model's prefill sequences a chunk of `w` positions at
    /// `pos` runs, as its pass key's variant: 0 inside the QSA indexer's
    /// dense range (dense attention), 1 crossing it (a dense part, then the
    /// selection), 2 wholly past it (the selection alone, as wide as the
    /// context's blocks). A plan recorded at one is not an allocation
    /// sequence of another (`pass-plan-design.md` "what would move this").
    /// Chunks past the range still differ in size with the context's
    /// blocks — the drift guard re-records those; the kind keeps one
    /// sequence from being served another's.
    fn prefill_variant(&self, pos: usize, w: usize) -> u32 {
        match self.dense_limit() {
            None => 0,
            Some(limit) if pos + w <= limit => 0,
            Some(limit) if pos >= limit => 2,
            Some(_) => 1,
        }
    }

    fn decode_variant(&self, pos: usize, device_plan: bool) -> u32 {
        let ratio = self.layers.iter().find_map(|l| match &l.mixer {
            Mixer::Full(a) => a.indexer().map(|i| i.ratio()),
            Mixer::Linear(_) => None,
        });
        let block = ratio.is_some_and(|r| (pos + 1).is_multiple_of(r));
        block as u32 | (device_plan as u32) << 1
    }

    /// The plan-cache key of a pass shape this model prepares or opens — a
    /// decode step or a prefill chunk — bracketed by `opener`. `None` for
    /// the shapes only a driver opens.
    #[cfg(feature = "metal")]
    fn pass_key(shape: crate::nn::decode::PassShape, opener: PassOpener) -> Option<PassKey> {
        use crate::nn::decode::PassShape;
        let kind = match shape {
            PassShape::Decode { b } => PassKind::Decode {
                b: b.min(u16::MAX as usize) as u16,
            },
            PassShape::Prefill { w, fresh } => PassKind::Prefill {
                w: w.min(u32::MAX as usize) as u32,
                fresh,
            },
            PassShape::Verify { .. } | PassShape::Ragged { .. } => return None,
        };
        Some(PassKey::new(kind).opened_by(opener))
    }

    fn close_pass(&self, opened: bool) -> Result<()> {
        if opened {
            crate::nn::decode::end_pass(&self.device())?;
        }
        Ok(())
    }

    /// Decode hints handed to the prefetchers so far (the process).
    pub fn decode_hints() -> u64 {
        DECODE_HINTS.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// [`Self::hint_encode`] for a layer that may be past the last.
    fn hint_encode_at(&self, layer: usize, carry: &Tensor) -> Result<Option<(usize, Tensor)>> {
        if layer >= self.layers.len() {
            return Ok(None);
        }
        let l = &self.layers[layer];
        let cur = l.collapsed(carry)?;
        Ok(Some((layer, l.moe().route_ids(&cur)?)))
    }

    /// The hint-recall probe's tallies per lead k = 0..3: `(predicted ∩
    /// real, real)` over the decode passes so far (`QWEN4EXP_HINT_RECALL=1`).
    pub fn hint_recall(&self) -> [(u64, u64); 4] {
        *self.hint_recall.lock()
    }

    fn hint_land(&self, pending: Option<(usize, Tensor)>) -> Result<()> {
        if let Some((layer, ids)) = pending {
            let host: Vec<u32> = ids.flatten_all()?.to_vec1()?;
            self.layers[layer].moe().experts().prefetch_ids(&host)?;
        }
        Ok(())
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
        let mut kv_grow = 0usize;
        for (l, layer) in self.layers.iter().enumerate() {
            if matches!(layer.mixer, Mixer::Full(_)) {
                kv_grow += cache.kv[l].grow_bytes_for(seq, nkv, hd);
            }
        }
        // The constant was measured as the peak over a 512-tile WITH a second
        // tile in flight (P18): two tiles' worth per chunk of positions. A
        // pipelined pass holds `prefill_tiles` tiles at once — each beyond
        // the pair another half of the constant per chunk — and past that
        // the transient does not grow with the pass.
        let transient = if seq > 1 {
            let chunk = prefill_chunk();
            let tiles = if prefill_pair_on() {
                prefill_tiles()
            } else {
                2
            };
            let in_flight = chunk.saturating_mul(tiles).max(chunk);
            let base = PREFILL_TRANSIENT_BYTES_PER_TOKEN.saturating_mul(seq.min(chunk));
            // Positions past the two tiles the constant already covers.
            let further = seq.min(in_flight).saturating_sub(2 * chunk);
            base + PREFILL_TRANSIENT_BYTES_PER_TOKEN.saturating_mul(further) / 2
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
        self.prepare_pass_as(cache, seq, false)
    }

    /// [`Self::prepare_pass_inner`] for the pass about to run: a driver's,
    /// or with `own` the one this model opens itself when no driver holds a
    /// pass — whose plan, if any, is a different key.
    fn prepare_pass_as(&self, cache: &mut Cache, seq: usize, own: bool) -> Result<()> {
        if seq == 0 || cache.prepared == Some((cache.pos, seq)) {
            return Ok(());
        }
        let mut plan = self.plan_pass(cache, seq);
        cache.prepared = Some((cache.pos, seq));
        // A pass a cached plan serves allocates its activations from the
        // base, which is already allocated and counted: it owes no transient.
        #[cfg(feature = "metal")]
        if seq > 1 {
            if let Ok(m) = self.device().as_metal_device() {
                if m.pass_plans_enabled() {
                    let opener = if own && !m.pass_open() {
                        PassOpener::Model
                    } else {
                        PassOpener::Driver
                    };
                    let shape = crate::nn::decode::PassShape::Prefill {
                        w: seq,
                        fresh: cache.pos == 0,
                    };
                    let variant = self.prefill_variant(cache.pos, seq);
                    if Self::pass_key(shape, opener)
                        .is_some_and(|k| m.pass_plan_cached(&k.with_variant(variant)))
                    {
                        plan.transient = 0;
                    }
                }
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = own;
        if plan.bytes() == 0 || self.mem_check == MemCheck::Off || self.memory_fraction <= 0.0 {
            return Ok(());
        }
        #[cfg(feature = "metal")]
        if let Ok(m) = self.device().as_metal_device() {
            use crate::models::lfm2::{kv_grow_budget, kv_grow_fits};
            let dev = m.device();
            let allocated = dev.current_allocated_size();
            let working_set = dev.recommended_max_working_set_size();
            // The transients come from the heap chunks first (§6.3k), whose
            // whole size `allocated` already counts: what the chunks have
            // free is transient the pass will NOT allocate.
            let (hs, hp) = m.heap_stats();
            let heap_free = (hs.size + hp.size).saturating_sub(hs.used + hp.used);
            let need = plan.kv_grow + plan.transient.saturating_sub(heap_free);
            let fits = kv_grow_fits(allocated, working_set, self.memory_fraction, need);
            let gb = |b: usize| b as f64 / 1e9;
            let line = format!(
                "pos={} seq={} kv_grow={:.3} GB transient={:.3} GB heap_free={:.3} GB need={:.3} GB allocated={:.2} GB budget={:.2} GB ({:.2} x working_set {:.2} GB) fits={fits}",
                plan.pos,
                plan.seq,
                gb(plan.kv_grow),
                gb(plan.transient),
                gb(heap_free),
                gb(need),
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

    /// Whether the streamed tier's routed experts may be planned on the
    /// device (`ExpertResidency::plan_device`; on when any layer's tier
    /// can). Off, every pass takes the host plan — the A/B arm a test picks
    /// per model where `QWEN4EXP_PLAN=host` would pick it per process.
    pub fn set_device_plan(&mut self, on: bool) {
        self.device_plan = on
            && self
                .layers
                .iter()
                .any(|l| l.moe().experts().plans_on_device());
    }

    pub fn device_plan(&self) -> bool {
        self.device_plan
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

    /// Select whether every layer's f16 routed step fuses its SwiGLU and its
    /// combine's read (`MoeLayer::set_f16_fuse`; `qwen4exp.f16_fuse` by
    /// default).
    pub fn set_f16_fuse(&mut self, on: bool) {
        for l in &mut self.layers {
            l.moe.set_f16_fuse(on);
        }
        if let Some(m) = &mut self.mtp {
            m.layer_mut().moe_mut().set_f16_fuse(on);
        }
    }

    /// Select whether every layer's f16 routed step folds its activation
    /// cast into gate|up (`MoeLayer::set_f16_fold`; `qwen4exp.f16_fold` by
    /// default).
    pub fn set_f16_fold(&mut self, on: bool) {
        for l in &mut self.layers {
            l.moe.set_f16_fold(on);
        }
        if let Some(m) = &mut self.mtp {
            m.layer_mut().moe_mut().set_f16_fold(on);
        }
    }

    /// The routed experts' activation dtype (the first layer's; every layer
    /// is set together, [`Self::set_routed_act`]).
    pub fn routed_act(&self) -> lloom_tensor::DType {
        self.layers
            .first()
            .map_or(lloom_tensor::DType::F32, |l| l.moe.routed_act())
    }

    pub fn prefill_mode(&self) -> PrefillMode {
        self.prefill_mode
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Apply the runtime axes (`--kv-quant`, …) to the model's config: every
    /// `Cache` built from `config()` afterwards takes them. Refused on a
    /// device the arm has no kernels on (the quantized KV is Metal only).
    pub fn apply_axes(&mut self, args: &super::config::ConfigArgs) -> Result<()> {
        let mut cfg = self.cfg.clone();
        args.apply(&mut cfg).map_err(lloom_tensor::Error::msg)?;
        self.set_kv_precision(cfg.kv_precision)
    }

    /// Set the KV store every `Cache` built from `config()` afterwards uses
    /// (`DESIGN.md` §10.7).
    pub fn set_kv_precision(&mut self, p: crate::nn::ops::KvPrecision) -> Result<()> {
        if p.quant().is_on() && !self.device().is_metal() {
            lloom_tensor::bail!(
                "qwen4exp: kv-quant = {p:?} is Metal-only (the quantized append and reads are \
                 Metal kernels)"
            )
        }
        self.cfg.kv_precision = p;
        Ok(())
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
            table: std::sync::Arc::new(table),
            block,
        });
        Ok(())
    }

    /// Offer the PLE's host fill to a decode loop whose tokens stay on the
    /// device (`CausalLM::host_feed`, on by default). Off, such a loop's
    /// forwards read their input ids back instead: the A/B arm.
    pub fn set_host_fill(&mut self, on: bool) {
        self.host_fill = on;
    }

    pub fn host_fill(&self) -> bool {
        self.host_fill
    }

    /// Arm the PLE-window instrument (`design/docs/expert-streaming.md`
    /// §20.13, the overlap window): every single-position forward stamps its
    /// entry and cuts its command buffer at the PLE layer, so a profile's
    /// command-buffer records (`CANDLE_METAL_PROFILE=1`) split into the work
    /// before the PLE (embedding + the layers under it) and the rest. The cut
    /// is one commit a token; nothing it computes changes. Metal only.
    pub fn arm_ple_window(&mut self) {
        self.ple_window = Some(parking_lot::Mutex::new(PleWindowMarks::default()));
    }

    /// `(entry, cut)` per decode step since `arm_ple_window` or the last
    /// take, drained.
    pub fn take_ple_window_marks(&self) -> Vec<(f64, f64)> {
        self.ple_window
            .as_ref()
            .map(|w| std::mem::take(&mut w.lock().marks))
            .unwrap_or_default()
    }

    fn ple_window_entry(&self, seq: usize) {
        #[cfg(feature = "metal")]
        if let (Some(w), 1) = (&self.ple_window, seq) {
            w.lock().entry = Some(lloom_tensor::kernels::metal::profile::now_s());
        }
        #[cfg(not(feature = "metal"))]
        let _ = seq;
    }

    fn ple_window_cut(&self, res_hc: &Tensor) -> Result<()> {
        #[cfg(feature = "metal")]
        if let (Some(w), lloom_tensor::Device::Metal(m)) = (&self.ple_window, res_hc.device()) {
            if res_hc.dim(1)? == 1 {
                m.flush_marker()?;
                let cut = lloom_tensor::kernels::metal::profile::now_s();
                let mut w = w.lock();
                if let Some(entry) = w.entry.take() {
                    w.marks.push((entry, cut));
                }
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = res_hc;
        Ok(())
    }

    /// The N-gram PLE's table, if attached (its read policy and counters).
    pub fn ple_table(&self) -> Option<&PleTable> {
        self.ple.as_ref().map(|p| &*p.table)
    }

    /// The N-gram PLE's hash, if attached.
    pub fn ple_hash(&self) -> Option<&NgramHash> {
        self.ple.as_ref().map(|p| &p.hash)
    }

    /// Whether the N-gram PLE is attached.
    pub fn has_ple(&self) -> bool {
        self.ple.is_some()
    }

    /// The PLE hot-row cache's `(hits, misses)`, if a cached table is attached.
    pub fn ple_cache_stats(&self) -> Option<(u64, u64)> {
        match &self.ple {
            Some(p) => match &*p.table {
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

    /// Pin the head of each layer's profile — its `n` most-used experts
    /// (fewer where the pool is smaller: a pin leaves at least a free
    /// slot) — so no pass evicts them (`Streamed::pin`, §20.3 R2): the
    /// decode's hot set survives a long prompt's sweep, which otherwise
    /// churns the whole pool and leaves the decode on the per-layer host
    /// plan (measured 123 ms a token after a 937-token prompt, #749 P26).
    /// Returns the experts pinned.
    pub fn pin_from_hotlist(&self, hot: &super::hotlist::Hotlist, n: usize) -> Result<usize>
    where
        R: crate::nn::expert_stream::HasStreamStats,
    {
        let mut pinned = 0;
        for (l, layer) in self.layers.iter().enumerate() {
            let e = layer.moe().experts();
            let slots = e.slots();
            if slots == 0 || n == 0 {
                continue;
            }
            let take = n.min(slots.saturating_sub(1));
            pinned += e.pin(&hot.top(l, take))?;
        }
        Ok(pinned)
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
        self.ple_window_cut(res_hc)?;
        let seq = res_hc.dim(1)?;
        if seq == 1 && cache.pending_tokens.is_empty() {
            if let Some(link) = &cache.host.run {
                // The host fill: wait on the GPU for the rows a producer
                // fills from this position's token (`nn::ngram_memory`).
                let emb = link.encode_read(cache.pos)?;
                let emb = emb.reshape((1, emb.elem_count()))?;
                return ple
                    .block
                    .forward(res_hc, &emb, &mut cache.ple_conv, ple.layer);
            }
        }
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
        if seq == 1 {
            if let Some(emb) = self.ple_rows_owned(ple, &ids, res_hc.device())? {
                let emb = emb.reshape((1, emb.elem_count()))?;
                let out = ple
                    .block
                    .forward(res_hc, &emb, &mut cache.ple_conv, ple.layer);
                self.ple_rows_read_encoded();
                return out;
            }
        }
        let emb = ple.table.gather(&ids, res_hc.device())?; // [seq·n_heads·head_dim]
        let emb = emb.reshape((seq, emb.elem_count() / seq))?;
        ple.block
            .forward(res_hc, &emb, &mut cache.ple_conv, ple.layer)
    }

    /// One position's PLE rows gathered on the host straight into the
    /// model's Shared buffer, once the step that last read it has retired;
    /// `None` — the caller uploads them, as always — while it may still run,
    /// off Metal, or for a table that cannot gather into a slice.
    fn ple_rows_owned(
        &self,
        ple: &Ple,
        ids: &[i64],
        dev: &lloom_tensor::Device,
    ) -> Result<Option<Tensor>> {
        #[cfg(feature = "metal")]
        if let lloom_tensor::Device::Metal(m) = dev {
            let Ok(hd) = ple.table.head_dim() else {
                return Ok(None);
            };
            let n = ids.len() * hd;
            let clock = m.gpu_clock();
            let mut g = self.ple_rows.lock();
            let fits = matches!(g.as_ref(), Some((_, t, _)) if t.elem_count() == n);
            if !fits {
                // Outlives every pass: no plan serves it.
                let _declared = lloom_tensor::EscapeGuard::new();
                let buf = m.new_persistent_shared(n * std::mem::size_of::<f32>())?;
                if buf.contents().is_null() {
                    return Ok(None);
                }
                let storage = lloom_tensor::Storage::Metal(lloom_tensor::MetalStorage::new(
                    buf.clone(),
                    m.clone(),
                    n,
                    lloom_tensor::DType::F32,
                ));
                let t =
                    Tensor::from_storage(storage, n, lloom_tensor::op::BackpropOp::none(), false);
                *g = Some((buf, t, None));
            }
            let (buf, t, read) = g.as_mut().expect("made above");
            if !read.is_none_or(|e| clock.is_complete(e)) {
                return Ok(None);
            }
            // SAFETY: a CPU-mapped Shared buffer of `n` f32 (checked when
            // made); the step that last read it has completed (above), and
            // nothing on the device writes it.
            let out = unsafe { std::slice::from_raw_parts_mut(buf.contents() as *mut f32, n) };
            if ple.table.gather_into(ids, out).is_err() {
                return Ok(None);
            }
            *read = None;
            return Ok(Some(t.clone()));
        }
        let _ = (ple, ids, dev);
        Ok(None)
    }

    /// The PLE block has encoded its reads of the model's rows buffer: stamp
    /// the epoch they ride (`ple_rows_owned`).
    fn ple_rows_read_encoded(&self) {
        #[cfg(feature = "metal")]
        if let Ok(m) = self.device().as_metal_device() {
            if let Some((_, _, read)) = self.ple_rows.lock().as_mut() {
                *read = Some(m.gpu_clock().current_epoch());
            }
        }
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
        // The last decode step is final once another forward begins.
        cache.settle_standing()?;
        let mut logits = None;
        match self.prefill_mode {
            PrefillMode::Replay => {
                for t in 0..seq {
                    let h = embeds.narrow(1, t, 1)?;
                    logits = Some(self.decode_embed(&h, cache, obs)?);
                }
            }
            PrefillMode::Chunk => {
                // A chunk is QSA at every position: past the indexer's dense
                // range each query attends to its own selection.
                let chunk = prefill_chunk();
                let chunk_len = |start: usize| -> usize { (seq - start).min(chunk) };
                let pair_ok = prefill_pair_on()
                    && self.spec_ahead == 0
                    && !(0..self.layers.len())
                        .any(|l| obs.wants::<HcStatePoint>(l) || obs.wants::<LayerOutputPoint>(l));
                let mut start = 0;
                while start < seq {
                    let len = chunk_len(start);
                    // Up to `prefill_tiles` multi-position chunks in a row:
                    // one pass, the tiles interleaved a layer apart.
                    let mut lens = vec![len];
                    if pair_ok && len > 1 {
                        let mut at = start + len;
                        while lens.len() < prefill_tiles() && at < seq {
                            let l = chunk_len(at);
                            if l <= 1 {
                                break;
                            }
                            lens.push(l);
                            at += l;
                        }
                    }
                    if lens.len() > 1 {
                        let mut tiles = Vec::with_capacity(lens.len());
                        let mut at = start;
                        for &l in &lens {
                            tiles.push(embeds.narrow(1, at, l)?);
                            at += l;
                        }
                        let total: usize = lens.iter().sum();
                        self.prepare_pass_as(cache, total, true)?;
                        let bracket = self.open_pass_variant(
                            crate::nn::decode::PassShape::Prefill {
                                w: total,
                                fresh: cache.pos == 0,
                            },
                            self.prefill_variant(cache.pos, total),
                        )?;
                        logits = Some(self.prefill_embeds_pipelined(&tiles, cache)?);
                        self.close_pass(bracket)?;
                        start += total;
                        continue;
                    }
                    let h = embeds.narrow(1, start, len)?;
                    // Each chunk is a pass of its own: admitted here when the
                    // driver handed the whole prompt over untiled.
                    self.prepare_pass_as(cache, len, true)?;
                    logits = Some(if len == 1 {
                        self.decode_embed(&h, cache, obs)?
                    } else {
                        let bracket = self.open_pass_variant(
                            crate::nn::decode::PassShape::Prefill {
                                w: len,
                                fresh: cache.pos == 0,
                            },
                            self.prefill_variant(cache.pos, len),
                        )?;
                        let l = self.prefill_embeds(&h, cache, obs)?;
                        self.close_pass(bracket)?;
                        l
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

        // The streamed tier's prefetch (§20.3 P1 / P2): layer 0's likely
        // misses before the stack starts, every layer's as early as the
        // hint allows — the reads overlap the compute between the hint and
        // the layer's routed dispatch.
        let n_layers = self.layers.len();
        let spec = self.spec_ahead > 0;
        // The usage hint does not depend on the carry, so it can lead by
        // any number of layers; the staged-bytes cap is what bounds the
        // lead (`prefetch_lead`: a layer's misses are ~330 MB of a 1 GiB cap).
        let lead = prefetch_lead();
        if spec {
            // Layers 0 and 1 from the embedding's carry, read back now: no
            // layer runs before them to hide the read behind.
            if n_layers > 0 {
                self.hint_land(self.hint_encode(0, &res_hc)?)?;
            }
            if n_layers > 1 {
                self.hint_land(self.hint_encode(1, &res_hc)?)?;
            }
        } else {
            for l in 0..lead.min(n_layers) {
                self.layers[l].moe().experts().prefetch_ahead()?;
            }
        }
        for (l, layer) in self.layers.iter().enumerate() {
            // Under speculation: layer l+2's router on the carry entering
            // layer l, encoded before l runs and landed after it (its host
            // plan synced past the router); its reads overlap l+1 whole.
            let pending = if spec && l + 2 < n_layers {
                self.hint_encode(l + 2, &res_hc)?
            } else {
                None
            };
            if !spec {
                if let Some(ahead) = self.layers.get(l + lead) {
                    ahead.moe().experts().prefetch_ahead()?;
                }
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
            self.hint_land(pending)?;
            if obs.wants::<LayerOutputPoint>(l) {
                obs.at::<LayerOutputPoint>(l, &layer.collapsed(&res_hc)?)?;
            }
        }
        cache.keep_nextn(&res_hc);

        // Only the last position feeds the head.
        let last = res_hc.narrow(1, seq - 1, 1)?;
        let collapsed = self.hc_head.collapse(&last)?; // [1, 1, d]
        cache.pos += seq;
        let logits = self.lm_head.forward(&collapsed)?;
        logits
            .reshape((1, self.cfg.vocab_size))?
            .to_dtype(lloom_tensor::DType::F32)
    }

    /// Two consecutive chunks as one pass, interleaved a layer apart
    /// (`prefill_embeds_pipelined` over two tiles).
    pub fn prefill_embeds_pair(
        &self,
        ha: &Tensor,
        hb: &Tensor,
        cache: &mut Cache,
    ) -> Result<Tensor> {
        self.prefill_embeds_pipelined(&[ha.clone(), hb.clone()], cache)
    }

    /// `N` consecutive chunks `tiles[t] [1, seq_t, hidden]` as ONE pass,
    /// interleaved a layer apart on the one queue (`DESIGN.md` §11.5z-ix
    /// item 2): tile t runs layer l at step s = l + t, every step encoding
    /// each tile's X stage (the mixer bracket and the router, its ids'
    /// readback staged) and then each tile's M stage (the plan — the
    /// readback's wait — and the experts), so at every wait the GPU has the
    /// other tiles' stages queued:
    ///
    /// ```text
    /// step s:  X(0,s) X(1,s−1) … X(N−1,s−N+1)   then   M(0,s) M(1,s−1) … M(N−1,s−N+1)
    /// ```
    ///
    /// Tile t's layer l comes after tile t−1's layer l in the stream (its M
    /// was the previous step's), so it attends over the earlier tile's keys
    /// and starts from its recurrent state as a chunk after a chunk does;
    /// the PLE pops the tiles' tokens in order. The same dispatches in the
    /// same order per tile as `N` `prefill_embeds`: byte-identical. The
    /// lead a miss's read gets is the other tiles' stages — `N − 1` of them
    /// — which is what an arm whose loads the GPU waits for (the `io` arm's
    /// DMA at the layer) needs to not stall. Off Metal the readback is a
    /// plain one and the tiles still interleave.
    pub fn prefill_embeds_pipelined(&self, tiles: &[Tensor], cache: &mut Cache) -> Result<Tensor> {
        let n_tiles = tiles.len();
        if n_tiles == 0 {
            lloom_tensor::bail!("prefill_embeds_pipelined: no tiles")
        }
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;
        let mut seqs = Vec::with_capacity(n_tiles);
        let mut res: Vec<Tensor> = Vec::with_capacity(n_tiles);
        for h in tiles {
            let (_, seq, _) = h.dims3()?;
            seqs.push(seq);
            res.push(
                h.reshape((1, seq, 1, d))?
                    .broadcast_as((1, seq, n, d))?
                    .reshape((1, seq, n * d))?
                    .contiguous()?,
            );
        }
        let n_layers = self.layers.len();
        let lead = prefetch_lead();
        for l in 0..lead.min(n_layers) {
            self.layers[l].moe().experts().prefetch_ahead()?;
        }
        // A tile's X stage at layer l: the PLE's delta if this is its layer,
        // the mixer bracket, the router — and the ids' readback STAGED (a
        // blit into a CPU-visible copy, its command buffer committed with a
        // marker), so the M stage waits for that buffer alone.
        struct Staged {
            res: Tensor,
            inject: Tensor,
            routed: super::moe::PrefillRouted,
            ids: Option<super::moe::StagedIds>,
        }
        let stage_x = |res: &mut Tensor, l: usize, cache: &mut Cache| -> Result<Staged> {
            if let Some(ple) = &self.ple {
                if ple.layer == l {
                    let delta = self.ple_delta(ple, res, cache)?;
                    trace("ple", l, &delta);
                    *res = (&*res + delta)?;
                }
            }
            let layer = &self.layers[l];
            let (res_after, cur, inject) = layer.prefill_mix(res, l, cache)?;
            let routed = layer.moe().prefill_route(&cur)?;
            let ids = layer.moe().prefill_stage_ids(&routed)?;
            Ok(Staged {
                res: res_after,
                inject,
                routed,
                ids,
            })
        };
        // A tile's M stage: the plan from the staged copy (waits for its
        // marker; the queue runs on), the experts, the FFN bracket's combine.
        let stage_m = |st: Staged, l: usize| -> Result<Tensor> {
            let layer = &self.layers[l];
            let groups = layer.moe().prefill_plan_staged(&st.routed, st.ids)?;
            let moe_out = layer.moe().prefill_execute(&st.routed, &groups)?;
            layer.prefill_finish(&st.res, &moe_out, &st.inject, l)
        };

        let mut staged: Vec<Option<Staged>> = (0..n_tiles).map(|_| None).collect();
        for step in 0..n_layers + n_tiles - 1 {
            if let Some(ahead) = self.layers.get(step + lead) {
                ahead.moe().experts().prefetch_ahead()?;
            }
            for t in 0..n_tiles {
                if step >= t && step - t < n_layers {
                    staged[t] = Some(stage_x(&mut res[t], step - t, cache)?);
                }
            }
            for t in 0..n_tiles {
                if let Some(st) = staged[t].take() {
                    res[t] = stage_m(st, step - t)?;
                }
            }
        }

        for r in &res {
            cache.keep_nextn(r);
        }
        // Only the last position feeds the head.
        let last_seq = seqs[n_tiles - 1];
        let last = res[n_tiles - 1].narrow(1, last_seq - 1, 1)?;
        let collapsed = self.hc_head.collapse(&last)?; // [1, 1, d]
        cache.pos += seqs.iter().sum::<usize>();
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
            // Inside a verify's window the step is the verify's to undo;
            // otherwise it gets a standing window of its own.
            let own = !cache.window_open();
            if own {
                cache.open_window()?;
            }
            let logits = self
                .decode_embed_pass(h, cache, obs, RowPlan::Host)
                .map(|(l, _)| l)?;
            if own {
                cache.keep_standing();
            }
            return Ok(logits);
        }
        // The cold phase: a pool that missed on the last pass runs the host
        // plan — one token and its loads, no replay — until a pass completes
        // with every expert resident; then the device plan, until it misses.
        if !cache.device_plan_armed {
            let before = self.stream_misses();
            let t0 = std::time::Instant::now();
            cache.open_window()?;
            let logits = self
                .decode_embed_pass(h, cache, obs, RowPlan::Host)
                .map(|(l, _)| l)?;
            cache.keep_standing();
            cache.host_plan_passes += 1;
            cache.host_pass_ns += t0.elapsed().as_nanos() as u64;
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
        let h = &self.replay_input_owned(h)?;
        cache.open_window()?;
        let (logits, records) = self.decode_embed_pass(h, cache, obs, RowPlan::Device)?;
        self.replay_read_encoded(h);
        let records: Vec<(usize, Tensor)> = records
            .into_iter()
            .filter_map(|(l, r)| r.into_device().map(|t| (l, t)))
            .collect();
        let (redo, after_first) = self.settle_records(&records)?;
        if !redo {
            // The step's window stays open as the standing one.
            cache.keep_standing();
            return Ok(logits);
        }
        cache.settle_after_first += after_first;
        let t0 = std::time::Instant::now();
        self.note_replay();
        cache.rewind_window()?;
        cache.redos += 1;
        cache.device_plan_armed = false;
        cache.open_window()?;
        let logits = self
            .decode_embed_pass(h, cache, obs, RowPlan::Host)
            .map(|(l, _)| l)?;
        self.replay_read_encoded(h);
        cache.keep_standing();
        cache.replay_ns += t0.elapsed().as_nanos() as u64;
        Ok(logits)
    }

    /// Settle a device-planned pass's records: every layer's `[3·n]` record
    /// in ONE readback (a `cat` after the pass and one wait on its tail),
    /// then each layer's slice settled in layer order. Reading them one by
    /// one was 48 commits and waits a token on retired work (#749 P29).
    /// `true` when any layer computed on a missing expert, and the experts
    /// loaded for layers past the first one that did.
    fn settle_records(&self, records: &[(usize, Tensor)]) -> Result<(bool, u64)> {
        if records.is_empty() {
            return Ok((false, 0));
        }
        let owned = self.read_records_host(records)?;
        let parts: Vec<(&R, &[u32])> = records
            .iter()
            .zip(&owned)
            .map(|((l, _), v)| (self.layers[*l].moe().experts(), v.as_slice()))
            .collect();
        Self::redo_of(R::settle_batch(&parts)?)
    }

    /// Every layer's device-plan record on the host, in ONE wait: read in
    /// place from the tiers' host-read slots when they hold them (a one-row
    /// step, #814 part 5), else a `cat` and one readback.
    fn read_records_host(&self, records: &[(usize, Tensor)]) -> Result<Vec<Vec<u32>>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "metal")]
        if let Ok(m) = records[0].1.device().as_metal_device() {
            let marker = m.flush_marker()?;
            m.wait_marker(&marker)?;
            let mut owned: Vec<Vec<u32>> = Vec::with_capacity(records.len());
            for (l, r) in records {
                match self.layers[*l].moe().experts().read_record_in_place(r)? {
                    Some(v) => owned.push(v),
                    None => break,
                }
            }
            if owned.len() == records.len() {
                return Ok(owned);
            }
        }
        let flat: Vec<Tensor> = records
            .iter()
            .map(|(_, r)| r.flatten_all())
            .collect::<Result<_>>()?;
        let all: Vec<u32> = if flat.len() == 1 {
            self.read_records(&flat[0])?
        } else {
            let refs: Vec<&Tensor> = flat.iter().collect();
            self.read_records(&Tensor::cat(&refs, 0)?)?
        };
        let mut owned = Vec::with_capacity(records.len());
        let mut at = 0usize;
        for ((l, _), f) in records.iter().zip(&flat) {
            let n = f.elem_count();
            let slice = all.get(at..at + n).ok_or_else(|| {
                lloom_tensor::Error::Msg(format!(
                    "settle: the records' cat is {} long, layer {l} needs {at}..{}",
                    all.len(),
                    at + n
                ))
            })?;
            owned.push(slice.to_vec());
            at += n;
        }
        Ok(owned)
    }

    /// A settle's verdicts as `(redo, misses after the first)`.
    fn redo_of(settled: Vec<(bool, u64)>) -> Result<(bool, u64)> {
        let mut redo = false;
        let mut after_first = 0u64;
        for (missed, n) in settled {
            if missed {
                if redo {
                    after_first += n;
                }
                redo = true;
            }
        }
        Ok((redo, after_first))
    }

    /// The settle's one readback of the records `t` (u32), through the
    /// model's own Shared destination on Metal so a step asks no pool for
    /// it; `to_vec1` elsewhere. The same bytes either way.
    fn read_records(&self, t: &Tensor) -> Result<Vec<u32>> {
        #[cfg(feature = "metal")]
        if let Ok(m) = t.device().as_metal_device() {
            let t = t.contiguous()?;
            let bytes = t.elem_count() * std::mem::size_of::<u32>();
            let dst = {
                let mut g = self.settle_dst.lock();
                match g.as_ref() {
                    Some(b) if b.length() >= bytes => b.clone(),
                    _ => {
                        let _declared = lloom_tensor::EscapeGuard::new();
                        let b = m.new_persistent_shared(bytes)?;
                        *g = Some(b.clone());
                        b
                    }
                }
            };
            return lloom_tensor::metal_backend::stage_readback_into(&t, &dst)?.finish();
        }
        t.to_vec1()
    }

    /// The streamed tier's misses so far, summed over the layers (0 for
    /// `Resident`): the cold-phase probe.
    fn stream_misses(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.moe().experts().misses_so_far())
            .sum()
    }

    /// The streamed tier's misses and evictions so far, summed over the
    /// layers (0 for `Resident`): the per-round trace's counters.
    pub(crate) fn stream_misses_total(&self) -> u64 {
        self.stream_misses()
    }

    pub(crate) fn stream_evictions_total(&self) -> u64 {
        self.layers
            .iter()
            .map(|l| l.moe().experts().evictions_so_far())
            .sum()
    }

    /// `h` as the device-planned pass and its replay read it. A miss
    /// rewinds the window and replays the pass from `h` after the planned
    /// pass has run, inside the pass a driver holds; a pass plan recorded
    /// from a pass that did not replay ends `h`'s interval at its first
    /// reader and serves its slot to the next value, so the replay would
    /// read that value. While the device may reuse activations, `h` is
    /// copied into memory no plan serves (`DESIGN.md` §9.2aa, escapes are
    /// declared); `h` itself otherwise.
    fn replay_input(h: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Ok(m) = h.device().as_metal_device() {
            if m.activations_may_be_reused() {
                let _declared = lloom_tensor::EscapeGuard::new();
                return crate::nn::recurrent::own_copy(h);
            }
        }
        Ok(h.clone())
    }

    /// [`Self::replay_input`] for a one-row decode step, into a buffer the
    /// model keeps: the copy asks no pool for its destination. The previous
    /// step's replay (if it ran) must have retired first, since it read the
    /// same buffer; otherwise, and for a shape the buffer is not, the copy
    /// goes to the pool as `replay_input` makes it.
    fn replay_input_owned(&self, h: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Ok(m) = h.device().as_metal_device() {
            if m.activations_may_be_reused() {
                let clock = m.gpu_clock();
                let mut g = self.replay_h.lock();
                let fits = matches!(g.as_ref(), Some((t, _)) if t.dims() == h.dims() && t.dtype() == h.dtype());
                if !fits {
                    // Outlives every pass: no plan serves it.
                    let _declared = lloom_tensor::EscapeGuard::new();
                    *g = Some((Tensor::zeros(h.dims(), h.dtype(), h.device())?, None));
                }
                let (t, read) = g.as_mut().expect("made above");
                if read.is_none_or(|e| clock.is_complete(e)) {
                    // The copy as `replay_input` makes it, minus the fill.
                    let _declared = lloom_tensor::EscapeGuard::new();
                    t.slice_set(&h.contiguous()?, 0, 0)?;
                    *read = None;
                    return Ok(t.clone());
                }
            }
        }
        Self::replay_input(h)
    }

    /// A pass has encoded its reads of `h` (the planned computation, or the
    /// replay): when `h` is the model's own replay input, stamp the epoch
    /// they ride, so the next step's copy waits for them to retire
    /// (`replay_input_owned`).
    fn replay_read_encoded(&self, h: &Tensor) {
        #[cfg(feature = "metal")]
        if let Ok(m) = h.device().as_metal_device() {
            let mut g = self.replay_h.lock();
            if let Some((t, read)) = g.as_mut() {
                if t.id() == h.id() {
                    *read = Some(m.gpu_clock().current_epoch());
                }
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = h;
    }

    /// The pass in progress replays: its recording holds the replay, which
    /// a pass that settles clean never runs, so it must not become its key's
    /// plan (`MetalDevice::pass_diverged`).
    fn note_replay(&self) {
        #[cfg(feature = "metal")]
        if let Ok(m) = self.device().as_metal_device() {
            m.pass_diverged();
        }
    }

    /// The hook a device-planned decode pass runs at each layer's join
    /// points: nothing, or spike 0's wait (`QWEN4EXP_SPIKE0=wait`).
    fn join_hook(&self) -> Result<JoinHook> {
        #[cfg(feature = "metal")]
        if spike0_wait() {
            if let Ok(m) = self.device().as_metal_device() {
                let ring = self
                    .spike0
                    .get_or_init(|| {
                        lloom_tensor::metal_backend::HostFedRing::new(m, 1, 1)
                            .map(std::sync::Arc::new)
                            .ok()
                    })
                    .clone()
                    .ok_or_else(|| lloom_tensor::Error::Msg("spike 0: no event".into()))?;
                let next = ring.base();
                return Ok(JoinHook::Wait { ring, next });
            }
        }
        Ok(JoinHook::Off)
    }

    fn decode_embed_pass<O: ForwardObserver>(
        &self,
        h: &Tensor,
        cache: &mut Cache,
        obs: &mut O,
        plan: RowPlan,
    ) -> Result<(Tensor, Vec<(usize, WindowPlan)>)> {
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;
        let device_plan = plan == RowPlan::Device;
        let mut records: Vec<(usize, WindowPlan)> = Vec::new();
        let bracket = self.open_pass_variant(
            crate::nn::decode::PassShape::Decode { b: 1 },
            self.decode_variant(cache.pos, device_plan),
        )?;

        // broadcast the embedding into the n-stream HC carry `res_hc [1, 1, n*d]`.
        // The reference inits the streams from the embed broadcast across n.
        let mut res_hc = h
            .broadcast_as((1, 1, n, d))?
            .reshape((1, 1, n * d))?
            .contiguous()?;

        let watch = watch_nan();
        let mut watched: Vec<Tensor> = Vec::new();
        // The hint-recall probe: layer l+k's router on the carry entering
        // layer l, k = 0..3, scored against layer l+k's real routing after
        // the pass.
        let probe = super::moe::hint_recall_probe();
        let mut predicted: Vec<(usize, usize, Tensor)> = Vec::new(); // (k, layer, ids)
                                                                     // The decode hint (`decode_hint_on`, the host plan only): layer
                                                                     // l+2's router on the carry entering layer l — 56 % of its experts
                                                                     // (P27) — its ids' readback staged before this layer's own plan
                                                                     // readback drains the queue, finished after it (the marker is past),
                                                                     // and handed to layer l+2's prefetcher: its misses read during
                                                                     // layers l+1 and l+2's mixer, landed at l+2's resolve.
        let hint = decode_hint_on() && !device_plan;
        let mut join = self.join_hook()?;
        for (l, layer) in self.layers.iter().enumerate() {
            if probe {
                for k in 0..4 {
                    if let Some((target, ids)) = self.hint_encode_at(l + k, &res_hc)? {
                        predicted.push((k, target, ids));
                    }
                }
            }
            #[cfg(feature = "metal")]
            let staged_hint = if hint && l + 2 < self.layers.len() {
                match self.hint_encode(l + 2, &res_hc)? {
                    // Deferred: the layer's own plan readback drains the
                    // queue past this blit, so no cut of its own (a cut is
                    // a residency commit on the per-buffer arm, ~0.7 ms —
                    // 46 a token undid the hint's gain, P27).
                    Some((target, ids)) if ids.device().is_metal() => Some((
                        target,
                        lloom_tensor::metal_backend::stage_readback_deferred(&ids)?,
                    )),
                    _ => None,
                }
            } else {
                None
            };
            #[cfg(not(feature = "metal"))]
            let _ = hint;
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
                layer.decode_parts_rows_with(&res_hc, l, &mut [cache], plan, &mut join)?;
            if let Some(r) = record {
                records.push((l, r));
            }
            #[cfg(feature = "metal")]
            if let Some((target, st)) = staged_hint {
                let ids: Vec<u32> = st.finish()?;
                // The named ten alone: the prefill hint's fill from the
                // usage profile (up to the pool) is 300 reads a layer a
                // token here — measured 167 000 wasted over 64 tokens and
                // the decode 35 % slower (P27).
                self.layers[target].moe().experts().prefetch_named(&ids)?;
                DECODE_HINTS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        self.close_pass(bracket)?;
        if probe && !predicted.is_empty() {
            let mut tally = self.hint_recall.lock();
            for (k, target, pred) in predicted {
                let Some(real) = self.layers[target].moe().probe_ids() else {
                    continue;
                };
                let pred: Vec<u32> = pred.flatten_all()?.to_vec1()?;
                let real: Vec<u32> = real.flatten_all()?.to_vec1()?;
                let hit = real.iter().filter(|e| pred.contains(e)).count() as u64;
                tally[k].0 += hit;
                tally[k].1 += real.len() as u64;
            }
        }

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
        let collapsed = self.hc_head.collapse(&res_hc)?; // [1, 1, d]
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
        #[cfg(feature = "metal")]
        if let Ok(m) = self.device().as_metal_device() {
            if m.pass_plans_enabled() {
                let variant = if seq_len == 1 {
                    let device_plan = self.device_plan
                        && plan_on_device()
                        && !cache.window_open()
                        && cache.device_plan_armed;
                    self.decode_variant(cache.pos, device_plan)
                } else {
                    self.prefill_variant(cache.pos, seq_len)
                };
                m.set_next_pass_variant(variant);
            }
        }
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
        self.ple_window_entry(seq);
        if cache.pending_noted && cache.pending_tokens.len() == seq {
            cache.pending_noted = false;
        } else if seq == 1 && cache.host.run.is_some() {
            // The host fill supplies this position's rows (`host_feed`).
            cache.set_pending_tokens(&[]);
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

    /// The PLE's host fill (`expert-streaming.md` §20.13): the rows of each
    /// decode position are filled by the loop's producer from the token
    /// before it and waited for on the GPU at the PLE layer. Metal only, and
    /// only with the PLE attached and `set_host_fill` on.
    fn host_feed(
        &self,
        cache: &mut Self::Cache,
    ) -> Result<Option<crate::nn::ngram_memory::HostFeed>> {
        #[cfg(feature = "metal")]
        {
            use crate::nn::ngram_memory::{FeedLink, HostFeed, HOST_FILL_SLOTS};
            use lloom_tensor::metal_backend::HostFedRing;
            use std::sync::Arc;
            let Some(ple) = self.ple.as_ref().filter(|_| self.host_fill) else {
                return Ok(None);
            };
            let lloom_tensor::Device::Metal(m) = ple.block.device() else {
                return Ok(None);
            };
            let len = ple.hash.n_heads() * ple.table.head_dim()?;
            let ring = match &cache.host.ring {
                Some(r) if r.len() == len => Arc::clone(r),
                _ => {
                    let r = Arc::new(HostFedRing::new(m, HOST_FILL_SLOTS, len)?);
                    cache.host.ring = Some(Arc::clone(&r));
                    r
                }
            };
            let link = Arc::new(FeedLink::new(ring, cache.pos));
            cache.host.run = Some(Arc::clone(&link));
            Ok(Some(HostFeed::new(
                link,
                Arc::clone(&ple.table),
                ple.hash.clone(),
                cache.ngram.clone(),
            )))
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = cache;
            Ok(None)
        }
    }

    /// Disarm the host fill and catch the n-gram window up: every fed
    /// forward consumed its token without pushing it, so the window (and a
    /// standing window's snapshot of it) still reads the run's first decode
    /// step.
    fn host_feed_end(&self, cache: &mut Self::Cache, published: &[u32]) -> Result<()> {
        let Some(link) = cache.host.run.take() else {
            return Ok(());
        };
        let fed = cache.pos.saturating_sub(link.pos0());
        if published.len() < fed && !link.released() {
            lloom_tensor::bail!(
                "qwen4exp host feed: {} tokens filled for {fed} fed forwards",
                published.len()
            )
        }
        let consumed = &published[..fed.min(published.len())];
        let start = cache.ngram.clone();
        let after = |n: usize| {
            let mut w = start.clone();
            for &t in &consumed[..n.min(consumed.len())] {
                w.push(Some(t));
            }
            w
        };
        // A window a fed forward opened snapshots the window it started from;
        // one opened before the run already holds the right one.
        if let Some(w) = cache.window.as_mut().filter(|w| w.pos >= link.pos0()) {
            w.ngram = after(w.pos - link.pos0());
        }
        cache.ngram = after(fed);
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
        for (r, row) in rows.iter_mut().enumerate() {
            row.cache.settle_standing()?;
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
            // Timed only where a tier streams (the miss regime's pass).
            let t0 = self.device_plan.then(std::time::Instant::now);
            let logits = self.decode_rows_pass(&h, &mut caches, false)?.0;
            let clean = self.stream_misses() == before;
            let ns = t0.map_or(0, |t| t.elapsed().as_nanos() as u64);
            for c in caches.iter_mut() {
                c.host_plan_passes += 1;
                c.host_pass_ns += ns;
                if clean {
                    c.device_plan_armed = true;
                }
            }
            return Ok(logits);
        }
        let h = Self::replay_input(&h)?;
        for c in caches.iter_mut() {
            c.open_window()?;
        }
        let (logits, records) = self.decode_rows_pass(&h, &mut caches, true)?;
        let (redo, after_first) = self.settle_records(&records)?;
        if !redo {
            for c in caches.iter_mut() {
                c.resolve_window()?;
            }
            return Ok(logits);
        }
        let t0 = std::time::Instant::now();
        self.note_replay();
        for c in caches.iter_mut() {
            c.rewind_window()?;
            c.redos += 1;
            c.device_plan_armed = false;
        }
        let logits = self.decode_rows_pass(&h, &mut caches, false)?.0;
        let ns = t0.elapsed().as_nanos() as u64;
        for c in caches.iter_mut() {
            c.replay_ns += ns;
            c.settle_after_first += after_first;
        }
        Ok(logits)
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
    ///
    /// Every row is settled as real here (a streamed tier loads what any
    /// row lacks, and a row that computed on a missing expert redoes the
    /// window). A speculative verify, whose rows past the accept are not
    /// real, takes [`Self::verify_window_run`] and
    /// [`Self::verify_window_settle`] instead.
    pub fn verify_window(&self, tokens: &[u32], cache: &mut Cache) -> Result<Vec<u32>> {
        let t = tokens.len();
        let pass = self.verify_window_run(tokens, cache)?;
        let v = self.verify_window_settle(pass, |_| t, false, cache)?;
        Ok(v.argmax)
    }

    /// [`Self::verify_window`]'s pass alone: the window opened and run, the
    /// argmax read back, every layer's plan record on the host — settled
    /// by [`Self::verify_window_settle`] once the accept is known. On the
    /// streamed tier a device-planned window flags a missing expert per
    /// row; a host-planned one loads row 0's (the real next token) and
    /// flags a draft row's instead of loading it (`QWEN4EXP_MTP_DRAFT_LOADS`
    /// loads those too, the A/B arm).
    pub fn verify_window_run(&self, tokens: &[u32], cache: &mut Cache) -> Result<VerifyPass> {
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
        let t0 = std::time::Instant::now();
        let dev = self.device();
        if !dev.is_metal() {
            // The reference path: the positions one by one through the
            // decode step (the CPU's gemm sums in an order that depends on
            // the row count, so a rows pass would not be the sequential
            // math there); the ring channels take the writes phase by phase.
            let drafts = cache.window_loads_drafts();
            let (argmax, records) = self.verify_rows_reference(tokens, cache, false, drafts)?;
            return Ok(VerifyPass {
                argmax,
                records,
                planned: false,
                t0,
                tokens: tokens.to_vec(),
                h: None,
                drafts_loaded: drafts,
            });
        }
        let input = Tensor::from_vec(tokens.to_vec(), (t, 1), &dev)?;
        let h = self.embed(&input)?; // [T, 1, hidden]
        let planned = self.device_plan && plan_on_device() && cache.device_plan_armed;
        let h = if planned { Self::replay_input(&h)? } else { h };
        let drafts = !planned && cache.window_loads_drafts();
        let kind = if planned {
            WindowPlanKind::Device
        } else {
            WindowPlanKind::Host {
                real_rows: 1,
                loaded_rows: if drafts { t } else { 1 },
            }
        };
        let (logits, plans) = self.verify_pass(&h, cache, kind)?;
        let argmax = logits.argmax(D::Minus1)?; // [T]
        let (argmax, records) = self.window_read(argmax, plans, t)?;
        Ok(VerifyPass {
            argmax,
            records,
            planned,
            t0,
            tokens: tokens.to_vec(),
            h: Some(h),
            drafts_loaded: drafts,
        })
    }

    /// The window's argmax `[T]` and its device plans' records in ONE
    /// readback (their `cat`, through the model's own destination): the
    /// round's one wait on the pass. A host plan's records are on the
    /// host already.
    fn window_read(
        &self,
        argmax: Tensor,
        plans: Vec<(usize, WindowPlan)>,
        rows: usize,
    ) -> Result<(Vec<u32>, Vec<(usize, WindowRecord)>)> {
        let topk = self.cfg.n_experts_per_tok;
        let mut device: Vec<(usize, Tensor)> = Vec::new();
        let mut out: Vec<(usize, WindowRecord)> = Vec::with_capacity(plans.len());
        for (l, p) in plans {
            match p {
                WindowPlan::Host(r) => out.push((l, r)),
                WindowPlan::Device(t) => device.push((l, t.flatten_all()?)),
            }
        }
        if device.is_empty() {
            return Ok((argmax.to_vec1::<u32>()?, out));
        }
        let mut refs: Vec<&Tensor> = device.iter().map(|(_, t)| t).collect();
        refs.push(&argmax);
        let all = self.read_records(&Tensor::cat(&refs, 0)?)?;
        let mut at = 0usize;
        for (l, t) in &device {
            let n = t.elem_count();
            let slice = all.get(at..at + n).ok_or_else(|| {
                lloom_tensor::Error::Msg(format!(
                    "window read: the cat is {} long, layer {l} needs {at}..{}",
                    all.len(),
                    at + n
                ))
            })?;
            out.push((*l, WindowRecord::device(slice.to_vec(), rows, topk)?));
            at += n;
        }
        let am = all.get(at..at + rows).ok_or_else(|| {
            lloom_tensor::Error::Msg(format!(
                "window read: the cat is {} long, the argmax needs {at}..{}",
                all.len(),
                at + rows
            ))
        })?;
        out.sort_by_key(|(l, _)| *l);
        Ok((am.to_vec(), out))
    }

    /// Settle [`Self::verify_window_run`]'s pass once `accept` has read its
    /// argmax (the accepted row count, `1..=T`; row 0 is always real), and
    /// decide what the window is worth (`expert-streaming.md` §20.16):
    ///
    /// - a flag on row 0 (the real next token computed on a missing expert)
    ///   redoes the window on the host plan — row 0's experts loaded as the
    ///   pass reaches each layer — and disarms the device plan;
    /// - a flag on a draft row `p ≥ 1` the accept would have kept cuts the
    ///   accepted prefix at `p` when `truncate` (the rows before it stand:
    ///   nothing after `p` is read by them) and disarms the device plan;
    ///   without `truncate` (every row real) it redoes the window with every
    ///   row's experts loaded;
    /// - a flag on a rejected row costs nothing: its experts are neither
    ///   loaded, nor touched, nor counted.
    ///
    /// The residency is then touched and the usage counted for the accepted
    /// rows alone. A host-planned window that loaded nothing for row 0 and
    /// was not cut arms the device plan for the windows after it.
    pub fn verify_window_settle(
        &self,
        pass: VerifyPass,
        accept: impl Fn(&[u32]) -> usize,
        truncate: bool,
        cache: &mut Cache,
    ) -> Result<WindowVerdict> {
        let VerifyPass {
            mut argmax,
            mut records,
            planned,
            t0,
            tokens,
            h,
            drafts_loaded,
        } = pass;
        let mut drafts_loaded = drafts_loaded;
        let t = tokens.len();
        if planned {
            cache.window_device += 1;
        } else {
            cache.window_host += 1;
            cache.host_plan_passes += 1;
            cache.host_pass_ns += t0.elapsed().as_nanos() as u64;
        }
        let mut a = accept(&argmax).clamp(1, t);
        let mut first_flagged = Self::first_flagged(&records);
        let redo = match first_flagged {
            Some(0) => true,
            Some(p) => !truncate && p < a,
            None => false,
        };
        if redo {
            // The plans closed (nothing accepted of them), the window put
            // back, the pass again on the host plan from the base: row 0's
            // experts loaded as it reaches each layer (every row's, when
            // every row is real).
            let parts: Vec<(&R, &WindowRecord)> = records
                .iter()
                .map(|(l, r)| (self.layers[*l].moe().experts(), r))
                .collect();
            R::settle_window(&parts, 0)?;
            let t1 = std::time::Instant::now();
            self.note_replay();
            cache.accept_window(0)?;
            cache.redos += 1;
            cache.window_redos += 1;
            cache.device_plan_armed = false;
            cache.open_window()?;
            cache.set_pending_tokens(&tokens);
            if let Some(w) = &mut cache.window {
                w.tokens = tokens.clone();
            }
            let all_loaded = !truncate || cache.window_loads_drafts();
            drafts_loaded = all_loaded;
            let (am, recs) = match &h {
                Some(h) => {
                    let kind = WindowPlanKind::Host {
                        real_rows: if truncate { 1 } else { t },
                        loaded_rows: if all_loaded { t } else { 1 },
                    };
                    let (logits, plans) = self.verify_pass(h, cache, kind)?;
                    self.window_read(logits.argmax(D::Minus1)?, plans, t)?
                }
                None => self.verify_rows_reference(&tokens, cache, !truncate, all_loaded)?,
            };
            argmax = am;
            records = recs;
            cache.replay_ns += t1.elapsed().as_nanos() as u64;
            a = accept(&argmax).clamp(1, t);
            first_flagged = Self::first_flagged(&records);
        }
        let accepted = match first_flagged {
            Some(p) if p < a => p.max(1),
            _ => a,
        };
        let parts: Vec<(&R, &WindowRecord)> = records
            .iter()
            .map(|(l, r)| (self.layers[*l].moe().experts(), r))
            .collect();
        let rejected_flags = R::settle_window(&parts, accepted)?;
        cache.window_rejected_flags += rejected_flags;
        let truncated = accepted < a;
        if truncated {
            cache.window_truncations += 1;
        }
        // The device plan: a window it computed whole stays armed; one cut
        // at a draft the accept would have kept names a token whose
        // experts are not all held — the host plan's round.
        if planned && !redo && truncated {
            cache.device_plan_armed = false;
        }
        // The host plan: a window whose real row(s) loaded nothing and that
        // was not cut found its experts resident — the device plan's round.
        // (The drafts loaded beside row 0 do not count: they are not the
        // next round's row 0.)
        let real_loads: u64 = records.iter().map(|(_, r)| r.real_loads).sum();
        let draft_loads: u64 = records.iter().map(|(_, r)| r.draft_loads).sum();
        // The drafts' loads under `auto` follow the real row's loads a
        // host-planned window (a redo's replay is one): the pool's regime.
        if !planned || redo {
            cache.window_real_loads_ema =
                0.75 * cache.window_real_loads_ema + 0.25 * real_loads as f32;
            if cache.window_real_loads_ema >= 2.0 {
                cache.window_draft_loads = true;
            } else if cache.window_real_loads_ema < 1.0 {
                cache.window_draft_loads = false;
            }
        }
        let _ = (drafts_loaded, draft_loads);
        if (!planned || redo)
            && self.device_plan
            && plan_on_device()
            && !truncated
            && real_loads == 0
        {
            cache.device_plan_armed = true;
        }
        Ok(WindowVerdict {
            argmax,
            accepted,
            planned,
            redo,
            first_flagged,
            rejected_flags,
        })
    }

    /// The first row of the window any layer flagged: the window is valid
    /// up to it.
    fn first_flagged(records: &[(usize, WindowRecord)]) -> Option<usize> {
        records
            .iter()
            .filter_map(|(_, r)| r.first_flagged_row())
            .min()
    }

    /// The window position by position through the decode step (the
    /// reference path): each position's argmax, and per layer one record
    /// over the window's rows — row 0 real (every row, when `all_real`),
    /// the drafts loaded beside it when `all_loaded`, else resolved
    /// against what is resident.
    fn verify_rows_reference(
        &self,
        tokens: &[u32],
        cache: &mut Cache,
        all_real: bool,
        all_loaded: bool,
    ) -> Result<(Vec<u32>, Vec<(usize, WindowRecord)>)> {
        let dev = self.device();
        let t = tokens.len();
        let mut argmax = Vec::with_capacity(t);
        let mut kv_after = Vec::with_capacity(t);
        let mut per_layer: Vec<Vec<WindowRecord>> = vec![Vec::new(); self.layers.len()];
        for (p, &tok) in tokens.iter().enumerate() {
            let input = Tensor::from_vec(vec![tok], (1, 1), &dev)?;
            let plan = RowPlan::WindowRow {
                real: p == 0 || all_real,
                loaded: p == 0 || all_loaded,
            };
            let (logits, recs) =
                self.decode_embed_pass(&self.embed(&input)?, cache, &mut (), plan)?;
            argmax.push(
                logits
                    .flatten_all()?
                    .argmax(D::Minus1)?
                    .to_scalar::<u32>()?,
            );
            kv_after.push(cache.kv.iter().map(|k| k.lens()).collect::<Vec<_>>());
            for (l, r) in recs {
                match r {
                    WindowPlan::Host(rec) => per_layer[l].push(rec),
                    WindowPlan::Device(_) => {
                        lloom_tensor::bail!(
                            "qwen4exp verify_window: a device plan on the reference path"
                        )
                    }
                }
            }
        }
        if let Some(w) = &mut cache.window {
            w.kv_after = kv_after;
        }
        let mut records = Vec::new();
        for (l, recs) in per_layer.into_iter().enumerate() {
            if !recs.is_empty() {
                records.push((l, WindowRecord::concat_rows(&recs)?));
            }
        }
        Ok((argmax, records))
    }

    /// The layer stack over the window's `T` positions → logits `[T, vocab]`
    /// and the layers' plan records; the KV lengths after each position
    /// recorded in the window.
    fn verify_pass(
        &self,
        h: &Tensor,
        cache: &mut Cache,
        kind: WindowPlanKind,
    ) -> Result<(Tensor, Vec<(usize, WindowPlan)>)> {
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;
        let t = h.dim(0)?;
        let mut records: Vec<(usize, WindowPlan)> = Vec::new();
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
            let (next, record, lens) = layer.decode_parts_seq(&res_hc, l, cache, kind)?;
            if let Some(r) = record {
                records.push((l, r));
            }
            if let Some(lens) = lens {
                for (p, ln) in lens.into_iter().enumerate() {
                    for row in kv_after.iter_mut().skip(p) {
                        row[l] = ln;
                    }
                }
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

/// Spike 0 of fetch-and-wait (`expert-streaming.md` §20.14): at every
/// device-planned layer's join point — after the routed and the shared
/// experts, before the combine, where the pass would wait for its misses —
/// a GPU wait on an event the host has already raised. What the hit path
/// pays for the wait alone, before any host round trip. An instrument
/// (`QWEN4EXP_SPIKE0=wait`); `Off` compiles to nothing.
enum JoinHook {
    Off,
    #[cfg(feature = "metal")]
    Wait {
        ring: std::sync::Arc<lloom_tensor::metal_backend::HostFedRing>,
        next: u64,
    },
}

impl super::moe::PlanHook for JoinHook {
    fn at_join(&mut self) -> Result<()> {
        match self {
            JoinHook::Off => Ok(()),
            #[cfg(feature = "metal")]
            JoinHook::Wait { ring, next } => {
                *next += 1;
                ring.signal(*next);
                ring.encode_wait(*next)
            }
        }
    }
}

/// `QWEN4EXP_SPIKE0=wait` (read once): spike 0's arm.
#[cfg(feature = "metal")]
fn spike0_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.spike0 == "wait")
}

/// Whether a host-planned verify window loads the DRAFT rows' missing
/// experts beside row 0's (`QWEN4EXP_MTP_DRAFT_LOADS`, read once).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DraftLoads {
    /// While the real row's loads a window say the pool is missing the
    /// stream (≥ 2 on average: the 171-slot regime, code and math at
    /// `auto`, §20.16a — a kept draft would be cut every round and the
    /// loads are what the verify needs to accept anything); off where the
    /// pool holds it (< 1: prose and count at `auto` — a wrong draft's
    /// loads are the cost §20.16 removes).
    Auto,
    On,
    Off,
}

fn mtp_draft_loads() -> DraftLoads {
    static ON: std::sync::OnceLock<DraftLoads> = std::sync::OnceLock::new();
    *ON.get_or_init(
        || match crate::tuning::tuning().qwen4exp.mtp_draft_loads.as_str() {
            "on" => DraftLoads::On,
            "off" => DraftLoads::Off,
            _ => DraftLoads::Auto,
        },
    )
}

impl Cache {
    /// Whether the next host-planned verify window loads its draft rows'
    /// misses (`DraftLoads`): under `Auto`, while the real row's loads a
    /// window average 2 or more (off again under 1).
    fn window_loads_drafts(&self) -> bool {
        match mtp_draft_loads() {
            DraftLoads::On => true,
            DraftLoads::Off => false,
            DraftLoads::Auto => self.window_draft_loads,
        }
    }
}

/// Whether the streamed tier's routed experts are planned on the device
/// (`QWEN4EXP_PLAN`, read once: `device` — the default — or `host`, the
/// per-layer readback; the A/B arm).
fn plan_on_device() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.plan != "host")
}

/// The rewindable arm is the standing window (P29 (7)): every decode step
/// leaves its window open until the next forward, so the driver can undo
/// exactly that step — the ring arm's lookahead overrun, a one-token
/// divergence of a reused prefix (`Session::prepare`). A prefill leaves no
/// horizon (its tiles are not windowed), nor does a step run inside a
/// verify's window; those reset, as before.
impl crate::nn::model::CacheRewind for Cache {
    fn rewind_horizon(&self) -> usize {
        self.standing_horizon()
    }

    fn rewind(&mut self, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let horizon = self.standing_horizon();
        if n != horizon {
            lloom_tensor::bail!(
                "qwen4exp cache: a rewind of {n} positions against a standing horizon of {horizon}: \
                 only the whole last step can be undone"
            )
        }
        self.rewind_window()
    }
}

#[cfg(test)]
mod tests {
    /// The KV store reaches the caches built from the config, and the
    /// shapes it has no arm on are refused by name: a quantized store on a
    /// device without the kernels (CPU), and any non-default store on the
    /// rows' paged pool (f32 only). f16 is served on the CPU — it is a dtype.
    #[test]
    fn the_kv_store_reaches_the_cache_and_its_missing_arms_are_refused() {
        use crate::nn::ops::{KvPrecision, KvQuant};
        use lloom_tensor::DType;
        let dev = lloom_tensor::Device::Cpu;
        let cfg = Config::tiny();
        let mut model = super::super::testing::resident_model(&cfg, &dev);
        let int8 = KvPrecision::quantized(KvQuant::Int8).unwrap();
        let err = model.set_kv_precision(int8).unwrap_err().to_string();
        assert!(err.contains("Metal-only"), "{err}");
        assert_eq!(
            model.config().kv_precision,
            super::super::config::DEFAULT_KV_PRECISION
        );
        model
            .set_kv_precision(KvPrecision::dense(DType::F16))
            .unwrap();
        let cache = Cache::new(model.config());
        let full = cfg
            .layer_types
            .iter()
            .position(|k| *k == LayerKind::Full)
            .unwrap();
        assert_eq!(cache.kv[full].precision(), KvPrecision::dense(DType::F16));
        let pools = model.state_pools(2, &dev);
        let err = Cache::in_pools(model.config(), &pools)
            .unwrap_err()
            .to_string();
        assert!(err.contains("paged KV pool"), "{err}");
        let mut f32_cfg = model.config().clone();
        f32_cfg.kv_precision = super::super::config::DEFAULT_KV_PRECISION;
        Cache::in_pools(&f32_cfg, &pools).unwrap();
        // One row keeps its own caches: every store is served there.
        let one = model.state_pools(1, &dev);
        let mut q_cfg = f32_cfg.clone();
        q_cfg.kv_precision = int8;
        Cache::in_pools(&q_cfg, &one).unwrap();
    }

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

    /// The device plan's replay input is the model's own buffer, reused only
    /// once the epoch its last reader rode has completed (#814). A read
    /// stamped in the OPEN epoch, which cannot be complete, sends the next
    /// copy to the pool. After a wait, the buffer is reused and holds the
    /// new copy. (A blocking driver retires every step before the next copy,
    /// so no decode test reaches the refusing arm. A lookahead driver does.)
    #[cfg(feature = "metal")]
    #[test]
    fn the_replay_input_is_reused_only_once_its_readers_retired() -> lloom_tensor::Result<()> {
        use lloom_tensor::{DType, Device, Tensor};
        let dev = Device::new_metal(0)?;
        let Device::Metal(m) = &dev else {
            unreachable!("a Metal device")
        };
        // What makes an activation's buffer reusable at all: a plan cache.
        m.enable_pass_plans(lloom_tensor::metal_backend::ArenaLayout::Packed);
        let cfg = super::super::config::Config::tiny();
        let model = super::super::testing::resident_model(&cfg, &dev);
        let h = Tensor::ones((1, 1, cfg.hidden_size), DType::F32, &dev)?;
        let a = model.replay_input_owned(&h)?;
        model.replay_read_encoded(&a);
        let b = model.replay_input_owned(&h)?;
        assert_ne!(
            b.id(),
            a.id(),
            "the replay input was overwritten under its readers"
        );
        dev.synchronize()?;
        let h2 = (Tensor::ones((1, 1, cfg.hidden_size), DType::F32, &dev)? * 2.0)?;
        let c = model.replay_input_owned(&h2)?;
        assert_eq!(c.id(), a.id(), "not reused once its readers retired");
        assert_eq!(
            c.flatten_all()?.to_vec1::<f32>()?,
            h2.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }
}
