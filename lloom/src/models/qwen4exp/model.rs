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
use super::deltanet_block::{GatedDeltaNet, GdnCheckpoint};
use super::hc::HyperConnection;
use super::moe::{ExpertResidency, MoeLayer, WindowPlan, WindowPlanKind, WindowRecord};
use super::ple::{NgramHash, NgramWindow, PleLayer, PleTable};
use super::qsa::{KvCache, KvPool, QsaAttention};
use super::trace;
use crate::nn::decoder::WindowMark;
use crate::nn::model::{CacheReset, CausalLM, Model as ModelTrait};
use crate::nn::observer::{ForwardObserver, HcStatePoint, LayerOutputPoint};
use crate::nn::turn_checkpoint::CheckpointRing;
use crate::nn::var_builder::VarBuilder;
use crate::nn::RecurrentState;
use crate::profile::{NoRewind, One, Own, Paged, Profile, Rewind, Ring, Rows};
#[cfg(feature = "metal")]
use lloom_tensor::metal_backend::{PassKey, PassKind, PassOpener};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Module, Result, Tensor, D};
use std::marker::PhantomData;

/// One layer's token-mixer (static dispatch — an enum, not `dyn`, per §3.6).
#[derive(Debug, Clone)]
pub enum Mixer<P: Profile> {
    Linear(GatedDeltaNet<P>),
    Full(QsaAttention<P>),
}

/// One decoder layer: two HC brackets around the token-mixer and the MoE.
#[derive(Debug, Clone)]
pub struct Layer<P: Profile, R: ExpertResidency> {
    hc_attn: HyperConnection,
    mixer: Mixer<P>,
    hc_ffn: HyperConnection,
    moe: MoeLayer<P, R>,
}

impl<P: Profile, R: ExpertResidency> Layer<P, R> {
    pub fn new(
        hc_attn: HyperConnection,
        mixer: Mixer<P>,
        hc_ffn: HyperConnection,
        moe: MoeLayer<P, R>,
    ) -> Self {
        Self {
            hc_attn,
            mixer,
            hc_ffn,
            moe,
        }
    }

    pub fn mixer(&self) -> &Mixer<P> {
        &self.mixer
    }

    pub fn moe(&self) -> &MoeLayer<P, R> {
        &self.moe
    }

    pub fn moe_mut(&mut self) -> &mut MoeLayer<P, R> {
        &mut self.moe
    }

    /// The `[.., d]` collapse of a carry through this layer's ffn bracket —
    /// the per-layer value the observer reports (`LayerOutputPoint`).
    pub fn collapsed(&self, res_hc: &Tensor) -> Result<Tensor> {
        self.hc_ffn.collapse(res_hc)
    }

    /// The `[.., d]` input this layer's mixer reads from the carry entering
    /// it — its attention bracket's mix — which is what a cache tapping
    /// this layer retains (`Cache::set_tap_layers`).
    pub fn attn_input(&self, res_hc: &Tensor) -> Result<Tensor> {
        self.hc_attn.collapse(res_hc)
    }

    /// One position through the layer: `res_hc [1, 1, n·d]` → the new carry,
    /// plus the token-mixer's and the MoE's outputs (the non-finite watch reads
    /// them). State channel / KV slot `l` of `cache`.
    pub fn decode_parts(
        &self,
        res_hc: &Tensor,
        l: usize,
        cache: &mut State<P>,
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
        cache: &mut State<P>,
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
        cache: &mut State<P>,
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
        caches: &mut [&mut State<P>],
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
        caches: &mut [&mut State<P>],
        plan: RowPlan,
        hook: &mut H,
    ) -> Result<(Tensor, Tensor, Tensor, Option<WindowPlan>)> {
        // HC prologue (attn bracket): collapse streams → mixer input.
        let (cur, inject) = self.hc_attn.mix(res_hc)?; // cur [B,1,d]
        trace("attn.mix", l, &cur);
        trace("attn.inject", l, &inject);
        if let [cache] = caches {
            cache.keep_tap(l, &cur)?;
        } else {
            for (r, c) in caches.iter_mut().enumerate() {
                if c.taps_layer(l) {
                    c.keep_tap(l, &cur.narrow(0, r, 1)?)?;
                }
            }
        }
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
        cache: &mut State<P>,
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
        cache: &mut State<P>,
        plan: super::moe::WindowPlanKind,
    ) -> Result<SeqParts> {
        let (cur, inject) = self.hc_attn.mix(res_hc)?; // [T, 1, d]
        cache.keep_tap(l, &cur)?;
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
    pub fn decode(&self, res_hc: &Tensor, l: usize, cache: &mut State<P>) -> Result<Tensor> {
        Ok(self.decode_parts(res_hc, l, cache)?.0)
    }

    /// A chunk through the layer: `res_hc [1, seq, n·d]` → the new carry
    /// (DeltaNet Kernel B / dense chunk attention, per-row MoE).
    pub fn prefill(&self, res_hc: &Tensor, l: usize, cache: &mut State<P>) -> Result<Tensor> {
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
        cache: &mut State<P>,
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
        cache: &mut State<P>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (cur, inject) = self.hc_attn.mix(res_hc)?; // [1, seq, d]
        cache.keep_tap(l, &cur)?;
        // The turn checkpoint inside the pass (`State::ckpt_take`): the
        // DeltaNet's recurrence a second time over the tile's prefix from
        // the same `S`; a full layer's KV lengths at the position.
        let take = cache.ckpt_take(l)?;
        let mixed = match &self.mixer {
            Mixer::Linear(dn) => match take {
                Some((rows, Some(frame))) => {
                    let (ns, nc) = dn.checkpoint_parts();
                    let s_view = frame.narrow(0, 0, ns)?;
                    let c_view = frame.narrow(0, ns, nc)?;
                    let ck = GdnCheckpoint {
                        s: &s_view,
                        conv: &c_view,
                    };
                    dn.prefill_checkpointing(
                        &cur,
                        l,
                        &mut cache.s_state,
                        &mut cache.conv_state,
                        rows,
                        &ck,
                    )?
                }
                Some((_, None)) => {
                    lloom_tensor::bail!("qwen4exp: layer {l}'s turn checkpoint needs a ring frame")
                }
                None => dn.prefill(&cur, l, &mut cache.s_state, &mut cache.conv_state)?,
            },
            Mixer::Full(qsa) => {
                let before = cache.kv[l].lens();
                let out = qsa.prefill(&cur, &mut cache.kv[l])?;
                if let Some((rows, _)) = take {
                    let seq = cur.dim(1)?;
                    let after = cache.kv[l].lens();
                    let tokens = before.0 + rows;
                    // The indexer's compressed blocks complete within
                    // `tokens` (`ratio` keys a block, `truncate_attention`'s
                    // rule); a full tile's are the pass's own.
                    let blocks = if rows == seq {
                        after.1
                    } else {
                        match qsa.indexer() {
                            Some(ix) => after.1.min(tokens / ix.ratio().max(1)),
                            None => before.1,
                        }
                    };
                    cache.ckpt_mark(l, WindowMark { tokens, blocks });
                }
                out
            }
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

/// One layer's decode step ([`Layer::decode_parts_rows_with`]): the attn
/// bracket's prologue, the mixer, its combine; the ffn bracket's prologue,
/// the MoE, its combine. `In` is the carry, the layer and the KV store's
/// precision.
#[cfg(not(target_arch = "wasm32"))]
impl<P: Profile, R: ExpertResidency> crate::optimizer::Visit for Layer<P, R> {
    type In = (crate::optimizer::NodeId, usize, crate::nn::ops::KvPrecision);
    type Out = crate::optimizer::NodeId;
    fn visit(
        &self,
        w: &mut crate::optimizer::Walker,
        (carry, l, precision): Self::In,
    ) -> Self::Out {
        let (cur, inject) = self.hc_attn.visit(w, (carry, "hc_attn", true));
        let inject = inject.expect("the injection was asked for");
        let mixed = match &self.mixer {
            Mixer::Linear(dn) => dn.visit(w, (cur, l)),
            Mixer::Full(qsa) => qsa.visit(w, (cur, l, precision)),
        };
        let carry = self.hc_attn.visit_combine(w, carry, mixed, inject);
        let (cur, inject) = self.hc_ffn.visit(w, (carry, "hc_ffn", true));
        let moe = self.moe.visit(w, cur);
        let inject = inject.expect("the injection was asked for");
        self.hc_ffn.visit_combine(w, carry, moe, inject)
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<P: Profile, R: ExpertResidency> Layer<P, R> {
    /// One row through [`Self::prefill`] (the MTP head's pass): the brackets
    /// as the decode step's, the attention's prefill row
    /// ([`QsaAttention::visit_row`]), the MoE planned on the host
    /// ([`MoeLayer::forward`]). A full-attention layer's; a linear layer's
    /// prefill row is not walked.
    pub(crate) fn visit_row(
        &self,
        w: &mut crate::optimizer::Walker,
        carry: crate::optimizer::NodeId,
        l: usize,
        precision: crate::nn::ops::KvPrecision,
    ) -> crate::optimizer::NodeId {
        use crate::optimizer::Visit;
        let Mixer::Full(qsa) = &self.mixer else {
            unreachable!("only a full-attention layer's prefill row is walked")
        };
        let (cur, inject) = self.hc_attn.visit(w, (carry, "hc_attn", true));
        let inject = inject.expect("the injection was asked for");
        let mixed = qsa.visit_row(w, cur, l, precision);
        let carry = self.hc_attn.visit_combine(w, carry, mixed, inject);
        let (cur, inject) = self.hc_ffn.visit(w, (carry, "hc_ffn", true));
        let moe = self.moe.visit_planned(w, cur, false);
        let inject = inject.expect("the injection was asked for");
        self.hc_ffn.visit_combine(w, carry, moe, inject)
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
pub struct VerifyPass<P: Profile> {
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
    /// The MTP head run over every window row inside the pass
    /// (`QWEN4EXP_MTP_FOLD`), its per-row drafts read back with the argmax.
    pub(crate) fold: Option<super::mtp::Folded<Own<P>>>,
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
pub struct State<P: Profile> {
    profile: PhantomData<P>,
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
    /// This cache's KV row on the device's ledger (`ledger.rs`), registered
    /// by `new_cache`; a clone shares it, a drop removes it.
    #[cfg(feature = "metal")]
    pub(crate) ledger: Option<std::sync::Arc<lloom_tensor::kernels::metal::ledger::Gauge>>,
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
    nextn: RowLog,
    /// Retained carries a verify pass found untaken and discarded.
    nextn_drops: u64,
    /// The `LayerTaps` probe a hidden-state drafter reads (DFlash/DSpark):
    /// per tapped layer, the d-wide input its attention bracket's mix hands
    /// the mixer, every position, until [`Self::take_taps`]. `tap_slot[l]`
    /// is layer `l`'s index into `taps` (the drafter's consumer order);
    /// empty, a forward taps nothing.
    tap_slot: Vec<Option<usize>>,
    taps: Vec<RowLog>,
    /// Retained taps a verify pass found untaken and discarded.
    tap_drops: u64,
    /// The open speculative window's pre-window bookkeeping, if any.
    window: Option<CacheWindow>,
    /// The turn checkpoints (`sequence-reuse-design.md` §2.3b, the
    /// `Decoder`'s mechanism on this family's own state): a ring of every
    /// linear layer's `S` and conv window, the full layers' KV lengths in
    /// the marks; `Qwen4Exp::arm_turn_checkpoints` sizes it.
    pub(crate) ckpt: CheckpointRing,
    /// The checkpoint the pass in flight takes inside itself.
    ckpt_pass: Option<PassCkpt>,
    /// Per ring slot, the PLE's n-gram window at the checkpoint (its conv
    /// history is the ring frame's last entry).
    ckpt_ngram: Vec<Option<NgramWindow>>,
    /// A decode step's window HELD behind the standing one (`hold_standing`):
    /// a provisional step settled one step behind (`Provisional::Behind`,
    /// a lookahead ring) is still undone after the next step was encoded —
    /// the newer step dropped (`drop_newer`), then this one rewound. Every
    /// recurrent channel holds its base (`RecurrentState::hold_window`).
    held: Option<CacheWindow>,
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
    /// Replays run on the DEVICE plan (`QWEN4EXP_REPLAY=device`, the arm
    /// that keeps the plan armed through a miss): one settle readback a
    /// replay, no per-layer drain; `replay_fallbacks` counts the replays
    /// that exhausted the bound (one per layer) and fell back to the host
    /// plan.
    pub device_replays: u64,
    pub replay_fallbacks: u64,
    /// Whether the next decode pass plans on the device: armed by a pass
    /// that completed with every expert resident, disarmed by a pass that
    /// missed (the cold phase runs the host plan — one token and its loads,
    /// no replay — and the warm phase the device plan).
    pub device_plan_armed: bool,
    /// The plan armed by the arm (`QWEN4EXP_REPLAY=device`) after a
    /// host-planned pass that loaded, not by a clean one: the decode's
    /// tokens run the device plan and replay there; a verify window keeps
    /// the default's rule (a window planned on a pool that misses is cut
    /// or redone, and its accounting is the host plan's — the MTP tests
    /// pin it), so the forced arming does not reach it.
    pub armed_through_misses: bool,
    /// The tier's miss count at the last device-planned token's hint: the
    /// hotlist's prediction (`QWEN4EXP_PREDICT=hotlist`) stages only after
    /// a token that loaded — a warm pool stages nothing (the third A/B:
    /// eight spare slots a layer, always free, had the hint stage 384
    /// experts a token on a pool that missed nothing, +10 % warm).
    pub predict_misses_seen: u64,
    /// Decode passes that ran the host plan while the device plan was
    /// disarmed (the cold phase's length, in tokens).
    pub host_plan_passes: u64,
    /// The fetch inside the token (`QWEN4EXP_FETCH=inside`): whether the
    /// next decode step fetches its misses (each layer waits on the device
    /// for the worker that loads them) — on from a fresh cache and from a
    /// step that missed, off after `fetch_exit` fetching steps in a row
    /// loaded nothing.
    pub fetch_mode: bool,
    /// Fetching steps in a row that loaded nothing.
    pub fetch_clean: usize,
    /// Fetching steps run; entries into and exits from the fetching
    /// regime; fetching steps redone (an expert the worker found no slot
    /// for). (The wait for the commands before a fetching step is the
    /// worker's now — `FetchStats::floor_waits`.)
    pub fetch_passes: u64,
    pub fetch_entries: u64,
    pub fetch_exits: u64,
    pub fetch_redos: u64,
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
    /// Retained admission-planner scratch, never an authoritative cached total.
    /// Growth/State clone can allocate host metadata; steady collection reuses it.
    #[cfg(feature = "metal")]
    heap_credit: lloom_tensor::kernels::metal::backing::ReuseCredit,
    /// A driver owns the settles of this cache's device-planned decode steps
    /// (`CausalLM::provisional`): a step returns its logits unsettled, and
    /// the driver settles it once it has read them back
    /// (`CausalLM::settle_step`), redrawing from the redone step's logits
    /// when it computed on a missing expert. Off outside such a run.
    #[cfg(feature = "metal")]
    pub(crate) provisional: crate::nn::model::Provisional,
    /// Provisional steps the driver's settle found had computed on a
    /// missing expert, redone before their token was handed out (`redos`
    /// counts them too).
    pub provisional_redos: u64,
}

/// Request-owned recurrent copies and append-only KV marks.
pub struct RetainedState {
    s: crate::nn::recurrent::RecurrentRetained,
    conv: crate::nn::recurrent::RecurrentRetained,
    ple: crate::nn::recurrent::RecurrentRetained,
    kv: Vec<WindowMark>,
    ngram: Vec<Option<u32>>,
    pos: usize,
}

impl<P: Profile> crate::nn::branch::RetainState for State<P> {
    type Retained = RetainedState;
    fn retain_state(&mut self) -> Result<Self::Retained> {
        self.settle_standing()?;
        if self.window.is_some() || self.held.is_some() || self.host.run.is_some() {
            lloom_tensor::bail!("qwen retention requires settled windows and no host-fill run")
        }
        let mut kv = crate::nn::branch::retained_vec(self.kv.len())?;
        for layer in &mut self.kv {
            kv.push(layer.retain_state()?);
        }
        Ok(RetainedState {
            s: self.s_state.retain_state()?,
            conv: self.conv_state.retain_state()?,
            ple: self.ple_conv.retain_state()?,
            ngram: self.ngram.retain_state()?,
            kv,
            pos: self.pos,
        })
    }
    fn restore_retained(&mut self, saved: &Self::Retained) -> Result<()> {
        self.settle_standing()?;
        if self.window.is_some()
            || self.held.is_some()
            || self.host.run.is_some()
            || saved.pos > self.pos
            || saved.kv.len() != self.kv.len()
        {
            lloom_tensor::bail!("qwen retention: invalid window, host fill, position or layers")
        }
        for (layer, mark) in self.kv.iter_mut().zip(&saved.kv) {
            layer.restore_retained(mark)?;
        }
        self.s_state.restore_retained(&saved.s)?;
        self.conv_state.restore_retained(&saved.conv)?;
        self.ple_conv.restore_retained(&saved.ple)?;
        self.ngram.restore_retained(&saved.ngram)?;
        self.pos = saved.pos;
        self.nextn.clear();
        for t in &mut self.taps {
            t.clear();
        }
        self.pending_tokens.clear();
        self.pending_noted = false;
        self.prepared = None;
        self.ckpt_pass = None;
        self.ckpt.forget_past(saved.pos);
        Ok(())
    }
}

/// A sequence's PLE host fill: the run in progress, if a decode loop armed
/// one — a single-position forward with no noted id then waits on the GPU for
/// rows a producer fills, and `State::<P>::ngram` stays at the run's first decode
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
    /// for an accept by position (`State::<P>::accept_window`).
    tokens: Vec<u32>,
    kv_after: Vec<Vec<(usize, usize)>>,
    /// The rows the retained carries and taps held when the window opened
    /// (every tap holds the same rows): a rewind cuts what the window's
    /// passes retained, so a replayed pass retains its positions once.
    retained: (usize, usize),
}

use crate::nn::row_log::RowLog;

/// The turn checkpoint a pass of `seq` positions from the state's position
/// takes, as bits 8..32 of the pass key's variant
/// (`CheckpointRing::pass_variant`): a pass that copies the state out, or
/// runs the recurrence a second time over its prefix, allocates another
/// sequence than one that does not.
fn ckpt_variant<P: Profile>(cache: &State<P>, seq: usize) -> u32 {
    (cache.ckpt.pass_variant(cache.pos, seq, cache.window_open()) & 0x00ff_ffff) << 8
}

/// A checkpoint a pass takes inside itself: the tile (of a pipelined pass)
/// the hinted position falls in, the rows into that tile, the ring slot,
/// and per layer how many tiles the layer has run (every layer sees the
/// tiles in order), with the full layers' marks as they are recorded.
#[derive(Debug, Clone)]
struct PassCkpt {
    tile: usize,
    rows: usize,
    slot: usize,
    seen: Vec<usize>,
    marks: Vec<WindowMark>,
    /// The n-gram window after the checkpoint's rows (the PLE's host
    /// state), taken by `Qwen4Exp::ple_delta` in the checkpoint's tile.
    ngram: Option<NgramWindow>,
}

/// The one-sequence, no-rewind shape (`No × One`): every channel and the
/// KV the state's own, the standing window a snapshot.
impl<P: Profile<Rewind = NoRewind, Rows = One>> State<P> {
    pub fn new(cfg: &Config) -> Self {
        Self::with_layers(cfg, cfg.n_layers)
    }

    /// [`State::new`] for `n_layers` layers of `cfg`'s geometry: an MTP
    /// head's lane cache is one layer of the model's shapes, its own
    /// channels and KV (`MtpLanes`), which is why it is this shape's.
    pub fn for_layers(cfg: &Config, n_layers: usize) -> Self {
        Self::with_layers(cfg, n_layers)
    }
}

/// The rewindable shape (`Ring × One`): the recurrent channels in `pools`'
/// phase rings (`Qwen4Exp::spec_pools`, `k + 2` phases for a `k`-draft
/// window), so a verify window is accepted by position; the KV the
/// state's own slab.
impl<P: Profile<Rewind = Ring, Rows = One>> State<P> {
    pub fn in_ring(cfg: &Config, pools: &StatePools) -> Result<Self> {
        if pools.kv.is_some() {
            lloom_tensor::bail!(
                "qwen4exp: a `Ring × One` state keeps its own KV; `pools` carries the rows' paged \
                 KV pool (`Qwen4Exp::spec_pools(1, k, ..)` makes the rewindable one)"
            )
        }
        if pools.s.phases() < 3 {
            lloom_tensor::bail!(
                "qwen4exp: a `Ring × One` state needs a ring of at least 3 phases (one draft), \
                 `pools` has {} (`Qwen4Exp::spec_pools`)",
                pools.s.phases()
            )
        }
        Self::leased(cfg, pools)
    }
}

/// The rows' shapes (`Paged`): the recurrent channels slot-indexed in
/// `pools` and the KV a slot of the rows' paged pool. `No × Paged` is
/// `Qwen4Exp::state_pools` (two phases: a decode's current and next);
/// `Ring × Paged` is `Qwen4Exp::spec_pools` over the rows (`k + 2` phases,
/// a verify window accepted by phase on every row).
impl<P: Profile<Rows = Paged>> State<P> {
    pub fn in_pools(cfg: &Config, pools: &StatePools) -> Result<Self> {
        let Some(p) = &pools.kv else {
            lloom_tensor::bail!(
                "qwen4exp: a {} state is a slot of the rows' paged KV pool; `pools` \
                 carries none (`Qwen4Exp::state_pools(rows > 1, ..)` makes it)",
                P::KEY
            )
        };
        let ring = pools.s.phases() >= 3;
        if ring != P::Rewind::REWINDABLE {
            lloom_tensor::bail!(
                "qwen4exp: a {} state over pools of {} phases: a `Ring` shape takes \
                 `spec_pools` (k + 2 phases), a `No` shape `state_pools` (two)",
                P::KEY,
                pools.s.phases()
            )
        }
        if cfg.kv_precision != super::config::DEFAULT_KV_PRECISION {
            lloom_tensor::bail!(
                "qwen4exp: kv-quant = {:?} has no arm on the rows' paged KV pool (f32): \
                 give every row its own cache (the unpooled rows) or keep kv-quant off",
                cfg.kv_precision
            )
        }
        let mut cache = Self::leased(cfg, pools)?;
        let lease = p.lease()?;
        for (l, kv) in cache.kv.iter_mut().enumerate() {
            *kv = KvCache::in_pool(p, &lease, l);
        }
        Ok(cache)
    }
}

impl<P: Profile> State<P> {
    /// [`State::new`] where `P` is `No × One`; the other shapes are leased
    /// from pools and refused here by name.
    pub fn own(cfg: &Config) -> Result<Self> {
        let key = P::KEY;
        if key.rewind != NoRewind::NAME || key.rows != One::NAME {
            lloom_tensor::bail!(
                "qwen4exp: a {key} state is leased from the driver's pools \
                 (`State::in_ring` over `spec_pools` on Ring, `State::in_pools` over \
                 `state_pools` on Paged), not built on its own"
            )
        }
        Ok(Self::with_layers(cfg, cfg.n_layers))
    }

    /// A state whose DeltaNet `S`, conv window and PLE history live in
    /// `pools` (a slot of each leased here, freed when the state drops);
    /// the KV its own until a shape says otherwise.
    fn leased(cfg: &Config, pools: &StatePools) -> Result<Self> {
        let mut cache = Self::with_layers(cfg, cfg.n_layers);
        cache.s_state = RecurrentState::in_pool(&pools.s)?;
        cache.conv_state = RecurrentState::in_pool(&pools.conv)?;
        if let Some(p) = &pools.ple {
            cache.ple_conv = RecurrentState::in_pool(p)?;
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
        self.ckpt.forget_past(self.pos);
        self.keep_first_rows(a)?;
        Ok(())
    }

    /// Cut the retained probes (the carries and every tap) to their first
    /// `a` positions.
    fn keep_first_rows(&mut self, a: usize) -> Result<()> {
        self.nextn.keep_first(a)?;
        for t in &mut self.taps {
            t.keep_first(a)?;
        }
        Ok(())
    }

    /// A cache for `n_layers` layers of `cfg`'s geometry (an MTP head's single
    /// layer shares the model's shapes, not its depth).
    pub(crate) fn with_layers(cfg: &Config, n_layers: usize) -> Self {
        Self {
            profile: PhantomData,
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
            ckpt: CheckpointRing::none(),
            ckpt_pass: None,
            ckpt_ngram: Vec::new(),
            pending_tokens: std::collections::VecDeque::new(),
            #[cfg(feature = "metal")]
            ledger: None,
            pending_noted: false,
            host: HostFill::default(),
            pos: 0,
            want_nextn: false,
            nextn: RowLog::default(),
            nextn_drops: 0,
            tap_slot: Vec::new(),
            taps: Vec::new(),
            tap_drops: 0,
            window: None,
            held: None,
            standing: false,
            redos: 0,
            device_plan_armed: false,
            armed_through_misses: false,
            predict_misses_seen: 0,
            device_replays: 0,
            replay_fallbacks: 0,
            host_plan_passes: 0,
            fetch_mode: false,
            fetch_clean: 0,
            fetch_passes: 0,
            fetch_entries: 0,
            fetch_exits: 0,
            fetch_redos: 0,
            #[cfg(feature = "metal")]
            provisional: crate::nn::model::Provisional::Off,
            provisional_redos: 0,
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
            #[cfg(feature = "metal")]
            heap_credit: Default::default(),
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
        for t in &mut self.taps {
            t.clear();
        }
        self.window = None;
        self.held = None;
        self.standing = false;
        self.ckpt.forget_all();
        self.ckpt_pass = None;
    }

    /// Keep `slots` turn checkpoints, each `elems[l]` f32s for layer `l`
    /// (`Qwen4Exp::arm_turn_checkpoints`).
    pub fn set_turn_checkpoints(&mut self, elems: &[usize], slots: usize) {
        self.ckpt = CheckpointRing::new(elems, slots);
        self.ckpt_ngram = vec![None; slots];
    }

    pub fn checkpoint_slots(&self) -> usize {
        self.ckpt.slots()
    }

    pub fn checkpoint_bytes(&self) -> usize {
        self.ckpt.bytes()
    }

    /// The positions the valid turn checkpoints stand at.
    pub fn checkpoint_positions(&self) -> Vec<usize> {
        self.ckpt.positions()
    }

    /// Have the passes that follow leave their checkpoint at `pos`
    /// (`CausalLM::checkpoint_at`).
    pub fn checkpoint_at(&mut self, pos: usize) {
        self.ckpt.hint(pos);
    }

    /// Open a request whose prefill ends at `end`
    /// (`CheckpointRing::open_request`).
    pub fn open_checkpoint_request(&mut self, policy: crate::nn::CheckpointPolicy, end: usize) {
        self.ckpt.open_request(policy, end);
    }

    /// Close the open request (`CheckpointRing::close_request`).
    pub fn close_checkpoint_request(&mut self) {
        self.ckpt.close_request();
    }

    /// The policy of the open request, if one is open.
    pub fn checkpoint_request(&self) -> Option<crate::nn::CheckpointPolicy> {
        self.ckpt.request_policy()
    }

    /// The checkpoint `Layer::prefill_mix` of layer `l` takes in the pass
    /// in flight, when this call is the hinted tile: the rows into the tile
    /// and the layer's frame slice (`None` for a full layer, whose mark is
    /// its lengths).
    pub(crate) fn ckpt_take(&mut self, l: usize) -> Result<Option<(usize, Option<Tensor>)>> {
        let Some(p) = &mut self.ckpt_pass else {
            return Ok(None);
        };
        let t = p.seen[l];
        p.seen[l] += 1;
        if t != p.tile {
            return Ok(None);
        }
        let frame = self.ckpt.frame(p.slot, l)?;
        Ok(Some((p.rows, frame)))
    }

    /// The rows into the chunk the PLE at layer `l` runs that the pass in
    /// flight checkpoints at: `Some` when this call (before layer `l`'s
    /// mixer, which counts the tile) is the checkpoint's tile.
    pub(crate) fn ckpt_ple_rows(&self, l: usize) -> Option<usize> {
        let p = self.ckpt_pass.as_ref()?;
        (p.seen[l] == p.tile).then_some(p.rows)
    }

    /// The PLE's state at the checkpoint's rows: the conv history into the
    /// frame's last entry, the n-gram window beside it.
    pub(crate) fn ckpt_ple_store(&mut self, hist: &Tensor, ngram: NgramWindow) -> Result<()> {
        let Some(p) = &mut self.ckpt_pass else {
            return Ok(());
        };
        p.ngram = Some(ngram);
        let slot = p.slot;
        if let Some(frame) = self.ckpt.frame(slot, self.kv.len())? {
            let h = hist
                .to_dtype(lloom_tensor::DType::F32)?
                .contiguous()?
                .flatten_all()?;
            frame.slice_set(&h, 0, 0)?;
        }
        Ok(())
    }

    pub(crate) fn ckpt_mark(&mut self, l: usize, mark: WindowMark) {
        if let Some(p) = &mut self.ckpt_pass {
            p.marks[l] = mark;
        }
    }

    /// Whether [`Self::restore`] can put the state back at exactly `pos`: a
    /// checkpoint stands there, no verify window is open, none is held.
    pub fn can_restore(&self, pos: usize) -> bool {
        !self.window_open()
            && self.held.is_none()
            && pos <= self.pos
            && self.ckpt.index(pos).is_some()
    }

    /// The latest checkpoint at or before `pos` this state can be put back
    /// at (`CacheRewind::restore_point`).
    pub fn restore_point(&self, pos: usize) -> Option<usize> {
        if self.window_open() || self.held.is_some() {
            return None;
        }
        self.ckpt
            .latest_at_or_before(pos.min(self.pos))
            .map(|(_, p)| p)
    }

    /// Put the state back where the checkpoint at `pos` was taken: every
    /// linear layer's `S` and conv window from the frame (in place, or the
    /// pool's next phase), every full layer's KV to its lengths, the
    /// position; the standing window settled first, what the discarded
    /// positions retained gone.
    pub fn restore(&mut self, pos: usize) -> Result<()> {
        self.settle_standing()?;
        let Some(idx) = self.ckpt.index(pos).filter(|_| self.can_restore(pos)) else {
            lloom_tensor::bail!(
                "qwen4exp cache: no turn checkpoint stands at {pos} (position {}, checkpoints at {:?})",
                self.pos,
                self.ckpt.positions()
            )
        };
        let marks = self.ckpt.marks(idx).to_vec();
        let n = self.kv.len();
        if marks.len() != n || self.ckpt.layers() != n + 1 {
            lloom_tensor::bail!(
                "qwen4exp cache: a checkpoint of {} layers for {n} layers",
                marks.len()
            )
        }
        for (l, &m) in marks.iter().enumerate() {
            match self.ckpt.frame(idx, l)? {
                Some(frame) => {
                    let (Some(s_cur), Some(c_cur)) = (self.s_state.get(l), self.conv_state.get(l))
                    else {
                        lloom_tensor::bail!(
                            "qwen4exp cache: layer {l} restores over an unwritten state"
                        )
                    };
                    let (s_dims, s_dt) = (s_cur.dims().to_vec(), s_cur.dtype());
                    let (c_dims, c_dt) = (c_cur.dims().to_vec(), c_cur.dtype());
                    let ns: usize = s_dims.iter().product();
                    let nc: usize = c_dims.iter().product();
                    if frame.elem_count() != ns + nc {
                        lloom_tensor::bail!(
                            "qwen4exp cache: layer {l}'s frame holds {} elements for a state of {}",
                            frame.elem_count(),
                            ns + nc
                        )
                    }
                    let s = frame
                        .narrow(0, 0, ns)?
                        .reshape(s_dims)?
                        .to_dtype(s_dt)?
                        .contiguous()?;
                    let w = frame
                        .narrow(0, ns, nc)?
                        .reshape(c_dims)?
                        .to_dtype(c_dt)?
                        .contiguous()?;
                    self.s_state.write(l, s)?;
                    self.conv_state.write(l, w)?;
                }
                None => {
                    if m.tokens > self.kv[l].len() {
                        lloom_tensor::bail!(
                            "qwen4exp cache: layer {l}'s KV of {} positions cannot be put back at {}",
                            self.kv[l].len(),
                            m.tokens
                        )
                    }
                    self.kv[l].truncate(m.tokens, m.blocks);
                }
            }
        }
        // The PLE: its conv history from the frame's last entry, its n-gram
        // window from the slot's copy.
        if let Some(frame) = self.ckpt.frame(idx, n)? {
            let Some(ngram) = self.ckpt_ngram.get(idx).cloned().flatten() else {
                lloom_tensor::bail!(
                    "qwen4exp cache: the checkpoint at {pos} keeps no n-gram window"
                )
            };
            let Some(l) = (0..n).find(|&l| self.ple_conv.get(l).is_some()) else {
                lloom_tensor::bail!(
                    "qwen4exp cache: the PLE's history restores over an unwritten state"
                )
            };
            let cur = self.ple_conv.get(l).expect("found");
            let h = frame
                .reshape(cur.dims())?
                .to_dtype(cur.dtype())?
                .contiguous()?;
            self.ple_conv.write(l, h)?;
            self.ngram = ngram;
        }
        self.pos = pos;
        self.nextn.clear();
        for t in &mut self.taps {
            t.clear();
        }
        self.pending_tokens.clear();
        self.pending_noted = false;
        self.prepared = None;
        self.ckpt.forget_past(pos);
        Ok(())
    }

    /// The carries retained since the last take, concatenated along the
    /// sequence: `[1, positions, n·d]`, or `None` if nothing was retained.
    pub fn take_nextn(&mut self) -> Result<Option<Tensor>> {
        self.nextn.take()
    }

    /// Tap the attention bracket's mix output of `layers` (in this order,
    /// the drafter's consumer order) on every forward from now on. Each is
    /// the input layer `l`'s mixer reads — the HC-contracted residual the
    /// layers before `l` left, d wide. Empty taps nothing; a layer this
    /// cache does not have, or one named twice, is refused.
    pub fn set_tap_layers(&mut self, layers: &[usize]) -> Result<()> {
        let n = self.kv.len();
        let mut slot = vec![None; if layers.is_empty() { 0 } else { n }];
        for (i, &l) in layers.iter().enumerate() {
            if l >= n {
                lloom_tensor::bail!(
                    "qwen4exp cache: tap layer {l} past the {n} layers (a tap on the last layer's \
                     output is the head mixer's, not a layer's)"
                )
            }
            if slot[l].replace(i).is_some() {
                lloom_tensor::bail!("qwen4exp cache: tap layer {l} named twice")
            }
        }
        self.tap_slot = slot;
        self.taps = vec![RowLog::default(); layers.len()];
        Ok(())
    }

    /// The tapped layers, in consumer order.
    pub fn tap_layers(&self) -> Vec<usize> {
        let mut out = vec![0; self.taps.len()];
        for (l, s) in self.tap_slot.iter().enumerate() {
            if let Some(i) = s {
                out[*i] = l;
            }
        }
        out
    }

    /// The taps retained since the last take, one `[1, positions, d]` F32
    /// per tapped layer in consumer order; `None` when nothing was retained
    /// or no layer is tapped.
    pub fn take_taps(&mut self) -> Result<Option<Vec<Tensor>>> {
        if self.taps.iter().all(RowLog::is_empty) {
            return Ok(None);
        }
        let rows = self.taps[0].rows();
        let mut out = Vec::with_capacity(self.taps.len());
        for (i, t) in self.taps.iter_mut().enumerate() {
            let r = t.rows();
            if r != rows {
                lloom_tensor::bail!(
                    "qwen4exp cache: tap {i} holds {r} positions, tap 0 holds {rows}"
                )
            }
            out.push(t.take()?.expect("non-empty: every tap holds the same rows"));
        }
        Ok(Some(out))
    }

    /// Whether layer `l` is tapped.
    pub(crate) fn taps_layer(&self, l: usize) -> bool {
        matches!(self.tap_slot.get(l), Some(Some(_)))
    }

    /// Retain layer `l`'s attention-bracket input `cur` (`[rows, 1, d]` or
    /// `[1, rows, d]`, positions in order) when `l` is tapped: an F32 copy
    /// outside every pass plan, since the drafter reads it after the pass.
    pub(crate) fn keep_tap(&mut self, l: usize, cur: &Tensor) -> Result<()> {
        let Some(&Some(i)) = self.tap_slot.get(l) else {
            return Ok(());
        };
        let d = cur.dim(D::Minus1)?;
        let rows = cur.elem_count() / d.max(1);
        let f = cur.to_dtype(lloom_tensor::DType::F32)?;
        self.taps[i].push(declared_copy(&f)?.reshape((1, rows, d))?);
        Ok(())
    }

    /// Discard whatever taps are retained, counting it as a dropped probe.
    pub(crate) fn drop_taps(&mut self) {
        if self.taps.iter().any(|t| !t.is_empty()) {
            self.tap_drops += 1;
            for t in &mut self.taps {
                t.clear();
            }
        }
    }

    /// Retained taps discarded untaken so far (`Probes::probe_drops`).
    pub fn tap_drops(&self) -> u64 {
        self.tap_drops
    }

    /// Retain a forward's carries `res_hc` when they are wanted. The drafter
    /// reads them after the pass that produced them, so they are kept as a
    /// declared escape ([`declared_copy`]).
    fn keep_nextn(&mut self, res_hc: &Tensor) -> Result<()> {
        if self.want_nextn {
            self.nextn.push(declared_copy(res_hc)?);
        }
        Ok(())
    }

    /// [`Self::keep_nextn`] for a window's `res_hc [rows, 1, n·d]`, kept
    /// as one `[1, rows, n·d]`: an accept by position keeps a view of its
    /// first `a` rows, so taking them allocates nothing whatever the accept
    /// (the verify's pass runs one sequence for every accepted length).
    fn keep_nextn_rows(&mut self, res_hc: &Tensor, rows: usize) -> Result<()> {
        if self.want_nextn {
            let kept = declared_copy(res_hc)?;
            let nd = kept.dim(D::Minus1)?;
            self.nextn.push(kept.reshape((1, rows, nd))?);
        }
        Ok(())
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
            retained: (self.nextn.rows(), self.taps.first().map_or(0, RowLog::rows)),
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
        self.ckpt.forget_past(self.pos);
        // What the window's passes retained goes with them.
        self.nextn.keep_first(w.retained.0)?;
        for t in &mut self.taps {
            t.keep_first(w.retained.1)?;
        }
        Ok(())
    }

    /// Cut an attention-only cache (an MTP head's layer) back to its first
    /// `tokens` positions: every KV, the indexer's blocks complete within
    /// them (`ratio` keys a block) and the position. Refused while a window
    /// is open or a recurrent channel holds state, which a length cannot put
    /// back.
    pub fn truncate_attention(&mut self, tokens: usize, ratio: usize) -> Result<()> {
        if self.window.is_some() {
            lloom_tensor::bail!("qwen4exp cache: truncating under an open window")
        }
        let recurrent = (0..self.s_state.n_layers()).any(|l| {
            self.s_state.get(l).is_some()
                || self.conv_state.get(l).is_some()
                || self.ple_conv.get(l).is_some()
        });
        if recurrent {
            lloom_tensor::bail!("qwen4exp cache: a recurrent channel cannot be truncated")
        }
        if tokens > self.pos {
            lloom_tensor::bail!(
                "qwen4exp cache: truncating {} positions to {tokens}",
                self.pos
            )
        }
        for k in self.kv.iter_mut().filter(|k| k.len() > tokens) {
            let blocks = k.n_blocks().min(tokens / ratio.max(1));
            k.truncate(tokens, blocks);
        }
        self.pos = tokens;
        self.ckpt.forget_past(tokens);
        self.nextn.clear();
        for t in &mut self.taps {
            t.clear();
        }
        self.prepared = None;
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

    /// Hold the standing window behind the next step's (a provisional step
    /// settled one step behind): the step stays undoable while the next one
    /// runs. Nothing to hold without a standing window; refused under a
    /// verify's window or with a window already held.
    #[cfg(feature = "metal")]
    pub(crate) fn hold_standing(&mut self) -> Result<()> {
        if self.window.is_none() {
            return Ok(());
        }
        if !self.standing {
            lloom_tensor::bail!("qwen4exp cache: holding a verify's window")
        }
        if self.held.is_some() {
            lloom_tensor::bail!("qwen4exp cache: a window is already held")
        }
        self.s_state.hold_window()?;
        self.conv_state.hold_window()?;
        self.ple_conv.hold_window()?;
        self.held = self.window.take();
        self.standing = false;
        Ok(())
    }

    /// The held step stands: release its window.
    #[cfg(feature = "metal")]
    pub(crate) fn resolve_held(&mut self) -> Result<()> {
        if self.held.take().is_some() {
            self.s_state.resolve_held()?;
            self.conv_state.resolve_held()?;
            self.ple_conv.resolve_held()?;
        }
        Ok(())
    }

    /// Undo the standing step and make the held one standing again: the cache
    /// is as it was right after the held step, which a rewind now undoes.
    #[cfg(feature = "metal")]
    pub(crate) fn drop_newer(&mut self) -> Result<()> {
        if self.held.is_none() {
            lloom_tensor::bail!("qwen4exp cache: no held window to fall back to")
        }
        if self.window.is_some() {
            if !self.standing {
                lloom_tensor::bail!("qwen4exp cache: dropping a verify's window")
            }
            self.rewind_window()?;
        }
        self.s_state.unhold_window()?;
        self.conv_state.unhold_window()?;
        self.ple_conv.unhold_window()?;
        self.window = self.held.take();
        self.standing = true;
        Ok(())
    }

    /// Whether a window is held behind the standing one.
    pub fn window_held(&self) -> bool {
        self.held.is_some()
    }

    /// Whether every recurrent channel can hold a window (none is pooled).
    #[cfg(feature = "metal")]
    pub(crate) fn holds_windows(&self) -> bool {
        !self.s_state.is_pooled() && !self.conv_state.is_pooled() && !self.ple_conv.is_pooled()
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

/// `t` in memory no pass plan serves while the device may reuse activations,
/// `t` itself otherwise. For a value read after the pass that produced it: a
/// plan ends a value's interval at its last read inside the pass and serves
/// the slot to the next value, and no recorder sees a read made between
/// passes, so such a value is a declared escape (`DESIGN.md` §9.2aa).
fn declared_copy(t: &Tensor) -> Result<Tensor> {
    #[cfg(feature = "metal")]
    if let Ok(m) = t.device().as_metal_device() {
        let m = m.execution_device()?;
        if m.activations_may_be_reused() {
            let _declared = lloom_tensor::EscapeGuard::new();
            return crate::nn::recurrent::own_copy(t);
        }
    }
    Ok(t.clone())
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

static PREFILL_PAIR_FORWARD: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(u8::MAX);

/// Whether a driver tile of one chunk is widened to [`prefill_tiles`] chunks
/// a forward (`prefill.pair_forward`, read once; [`set_prefill_pair_forward`]
/// for tests). `forward_embeds` pairs chunks only within one forward, and the
/// served driver hands it one tile a forward (`docs/expert-streaming.md`
/// §20.5).
pub fn prefill_pair_forward() -> bool {
    let v = PREFILL_PAIR_FORWARD.load(std::sync::atomic::Ordering::Relaxed);
    if v != u8::MAX {
        return v != 0;
    }
    let on = crate::tuning::tuning().qwen4exp.prefill_pair_forward;
    PREFILL_PAIR_FORWARD.store(on as u8, std::sync::atomic::Ordering::Relaxed);
    on
}

pub fn set_prefill_pair_forward(on: bool) {
    PREFILL_PAIR_FORWARD.store(on as u8, std::sync::atomic::Ordering::Relaxed);
}

/// The prefill tile a driver hands one forward, for a configured `tile`:
/// [`prefill_tiles`] chunks when [`prefill_pair_forward`] and the paired
/// prefill are on and `tile` is exactly one chunk, so the forward pairs its
/// chunks; `tile` otherwise. A tile between one chunk and two already pairs
/// inside its forward, a tile below one chunk is the caller's bound on the
/// prefill's working set, and `0` (the whole prompt) pairs already.
pub fn forward_tile(tile: usize) -> usize {
    if tile != 0 && tile == prefill_chunk() && prefill_pair_forward() && prefill_pair_on() {
        tile * prefill_tiles()
    } else {
        tile
    }
}

static PIPELINED_PASSES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Pipelined prefill passes (`prefill_embeds_pipelined`) this process has
/// run: the paired prefill's reach, read by tests and reports.
pub fn pipelined_passes() -> u64 {
    PIPELINED_PASSES.load(std::sync::atomic::Ordering::Relaxed)
}

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
pub struct Ple<P: Profile> {
    pub layer: usize,
    pub hash: NgramHash,
    /// Shared with a run's host-fill producer (`CausalLM::host_feed`).
    pub table: std::sync::Arc<PleTable>,
    pub block: PleLayer<P>,
}

/// The assembled model.
/// The streamed expert pool as the ledger's elastic tenant
/// (`profiles-and-memory-design.md` §4.2, class `State<P>`): `topk + 1` slots
/// as the floor, every expert as the want, one slot as the unit — the one
/// tenant type every weight-group tier is (`nn::group_tenant`, #892 A6b;
/// the dense layers' tier claims the same way). `resize` records the grant
/// as the slots wanted in service; the model applies it to its layers'
/// tiers (`Qwen4Exp::apply_expert_grant`) — the tiers evict through their
/// quarantine and give the parked slots' pages back.
pub type ExpertTenant = crate::nn::group_tenant::GroupTenant;

fn expert_tenant(
    bytes_per_slot: usize,
    slots: usize,
    n_experts: usize,
    topk: usize,
) -> Result<ExpertTenant> {
    let floor = topk.checked_add(1).ok_or_else(|| {
        lloom_tensor::Error::Msg("qwen4exp expert tenant: topk + 1 exceeds usize".into())
    })?;
    ExpertTenant::new(
        "qwen4exp-experts",
        bytes_per_slot,
        slots,
        floor,
        n_experts,
        slots,
    )
}

#[derive(Debug)]
pub struct Qwen4Exp<P: Profile, R: ExpertResidency> {
    /// `token_embd.weight` `[vocab, hidden]` — a `QMatMul` so the real IQ3_S
    /// table is gathered by the quantized `get_rows` kernel, never dequantized
    /// whole (2.5 GB at f32).
    embed_tokens: QMatMul,
    layers: Vec<Layer<P, R>>,
    /// Final HC collapse (the head mixer that carries the output norm).
    hc_head: HyperConnection,
    /// lm_head (`output.weight`, untied).
    lm_head: QMatMul,
    /// The N-gram PLE, when its table was loaded (`LoadOptions::ple_shard`).
    ple: Option<Ple<P>>,
    /// Turn checkpoints a state built from this model keeps
    /// (`State::ckpt`, `sequence-reuse-design.md` §2.3b): 2 by default,
    /// the engine admits them beside the experts' budget
    /// (`set_turn_checkpoints`); 0 keeps none.
    turn_checkpoints: usize,
    /// The PLE-window instrument, when armed (`arm_ple_window`).
    ple_window: Option<parking_lot::Mutex<PleWindowMarks>>,
    /// Whether [`CausalLM::host_feed`] offers the PLE's host fill to a loop
    /// whose tokens stay on the device (`set_host_fill`).
    host_fill: bool,
    /// The MTP head, when loaded (`LoadOptions::mtp`); an `MtpDrafter`
    /// borrows it beside `embed_tokens` and `lm_head`.
    mtp: Option<super::mtp::MtpHead<Own<P>, R>>,
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
    /// The expert tier's row on the device's ledger (`ledger.rs`), set by
    /// the streamed loader; dropped with the model.
    #[cfg(feature = "metal")]
    ledger_gauge: Option<std::sync::Arc<ExpertTenant>>,
    /// What the load reserved on the machine (`lloom_tensor::machine`),
    /// released when the model drops.
    reservation: Option<std::sync::Arc<lloom_tensor::machine::Holding>>,
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
    /// The fetch worker (`JoinHook::Fetch`, `QWEN4EXP_FETCH=inside`), made
    /// at the first fetching step.
    #[cfg(feature = "metal")]
    fetch_lane: std::sync::OnceLock<Option<std::sync::Arc<crate::nn::expert_fetch::FetchLane>>>,
    /// A fetching step whose settle trails its return — the next step's
    /// encode (`QWEN4EXP_FETCH_SETTLE=behind`) or the driver's settle (a
    /// provisional step): its records, its pass, the misses before it. Held
    /// by the model, not the cache — the settle is the tiers' bookkeeping,
    /// and a cache rewound or reset meanwhile does not change it — and
    /// settled by the driver (`CausalLM::settle_step`, `settle_pending` at a
    /// run's end) or at the model's next entry ([`Self::settle_trailing`]).
    #[cfg(feature = "metal")]
    pending: parking_lot::Mutex<std::collections::VecDeque<Pending>>,
    /// The device plan's settle reads every layer's record back into this
    /// Shared buffer rather than a pool buffer per step (`DESIGN.md` §9.2aa):
    /// grown to the largest record set, kept for the model's life.
    #[cfg(feature = "metal")]
    settle_dst: parking_lot::Mutex<Option<lloom_tensor::kernels::metal::BufferHandle>>,
    /// A one-row decode step's replay input, in memory the model owns rather
    /// than a pool buffer per step (`replay_input_owned`), and the epoch of
    /// the command buffer its last reader was encoded into — two, alternating
    /// (`ReplayInputs`).
    #[cfg(feature = "metal")]
    replay_h: parking_lot::Mutex<ReplayInputs>,
    /// A one-position step's PLE rows, gathered on the host into a Shared
    /// buffer the model keeps rather than uploaded per step
    /// (`ple_rows_owned`), and the epoch of its last reader.
    #[cfg(feature = "metal")]
    ple_rows: parking_lot::Mutex<
        Option<(
            lloom_tensor::kernels::metal::BufferHandle,
            Tensor,
            Option<lloom_tensor::kernels::metal::Epoch>,
        )>,
    >,
    /// What `qwen4exp.fusion = derived` decided at load (`derive_fusion`).
    fusion_report: Option<String>,
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
    /// a pass's transient is observed under beside its derivation
    /// ([`super::transient`]).
    Report,
    /// Refuse the pass with the arithmetic, before any state moves.
    Refuse,
}

impl MemCheck {
    /// `QWEN4EXP_MEM` = `off` | `report` | `refuse`, read once; `refuse`
    /// by default (`memory.md` §9.5q).
    pub fn from_env() -> Self {
        static ARM: std::sync::OnceLock<MemCheck> = std::sync::OnceLock::new();
        *ARM.get_or_init(
            || match crate::tuning::tuning().qwen4exp.mem_check.as_str() {
                "off" => MemCheck::Off,
                "report" => MemCheck::Report,
                // `refuse`; the parser (`tuning.rs`) refused anything else.
                _ => MemCheck::Refuse,
            },
        )
    }
}

/// What a pass of `seq` positions at `pos` would allocate before it runs:
/// the KV grow (slab or pool) and the transient term. Sized, not performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassPlan {
    pub pos: usize,
    pub seq: usize,
    /// Bytes the QSA layers' KV grows by for this pass (0 when every
    /// layer's slab or pool already holds `pos + seq`).
    pub kv_grow: usize,
    /// What the pass's route keeps of its allocations at once
    /// ([`super::transient::PrefillTransient::forward`]) for `seq > 1`,
    /// else 0.
    pub transient: usize,
}

impl PassPlan {
    pub fn bytes(&self) -> usize {
        self.kv_grow + self.transient
    }
}

/// What Qwen4-exp's switches can express, added to `policy`: the bracket
/// epilogues ride the brackets' own products only (`mixed_silu` the `down`
/// projection, `mixed_sigmoid` the `up`), the quantized store blocks the
/// shared expert's (`quant_silu` its gate, `quant_mul` its up), and no
/// switch puts a norm in a product's load. A derivation over the walk under this policy proposes
/// only what [`Qwen4Exp::derive_fusion_with`] can set.
#[cfg(not(target_arch = "wasm32"))]
pub fn fusion_policy(policy: &crate::optimizer::Policy) -> crate::optimizer::Policy {
    let mut policy = policy.clone();
    for f in ["norm", "rmsnorm"] {
        if !policy.unavailable.iter().any(|(n, _)| n == f) {
            policy
                .unavailable
                .push((f.to_string(), "no Qwen4-exp switch selects it".to_string()));
        }
    }
    for (f, site) in [
        ("mixed_silu", "_down.weight"),
        ("mixed_sigmoid", "_up.weight"),
        ("quant_silu", "ffn_gate_shexp.weight"),
        ("quant_mul", "ffn_up_shexp.weight"),
    ] {
        if !policy.sites.iter().any(|(n, _)| n == f) {
            policy.sites.push((f.to_string(), vec![site.to_string()]));
        }
    }
    policy
}

impl<P: Profile, R: ExpertResidency> Qwen4Exp<P, R> {
    pub fn new(
        embed_tokens: QMatMul,
        layers: Vec<Layer<P, R>>,
        hc_head: HyperConnection,
        lm_head: QMatMul,
        cfg: Config,
    ) -> Result<Self> {
        // The process is held to the regime the type names (`profile::hold`):
        // a family at `Exact` never computes under an inherited `fast`.
        crate::profile::hold::<P::Regime>("qwen4exp")?;
        let device_plan = layers.iter().any(|l| l.moe().experts().plans_on_device());
        Ok(Self {
            embed_tokens,
            layers,
            hc_head,
            lm_head,
            ple: None,
            turn_checkpoints: 2,
            ple_window: None,
            host_fill: true,
            mtp: None,
            prefill_mode: PrefillMode::default(),
            device_plan,
            hint_recall: parking_lot::Mutex::new([(0, 0); 4]),
            memory_fraction: super::loader::DEFAULT_MEMORY_FRACTION,
            #[cfg(feature = "metal")]
            ledger_gauge: None,
            reservation: None,
            mem_check: MemCheck::from_env(),
            spec_ahead: spec_ahead_from_env(),
            #[cfg(feature = "metal")]
            spike0: std::sync::OnceLock::new(),
            #[cfg(feature = "metal")]
            fetch_lane: std::sync::OnceLock::new(),
            #[cfg(feature = "metal")]
            pending: parking_lot::Mutex::new(std::collections::VecDeque::new()),
            #[cfg(feature = "metal")]
            settle_dst: parking_lot::Mutex::new(None),
            #[cfg(feature = "metal")]
            replay_h: parking_lot::Mutex::new(ReplayInputs::default()),
            #[cfg(feature = "metal")]
            ple_rows: parking_lot::Mutex::new(None),
            fusion_report: None,
            cfg,
        })
    }

    /// `qwen4exp.fusion = derived`: the optimizer's plan over the walk of the
    /// model as built (`optimizer::walk`), set on the brackets it decides —
    /// whether each bracket's down and up projections carry their activation
    /// at the store. Once, at load; the forward reads the bracket's flags as
    /// it did. Returns the report line.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn derive_fusion(&mut self) -> Result<String> {
        self.derive_fusion_with(&crate::optimizer::Policy::default())
    }

    /// [`Self::derive_fusion`] under `policy` (a test's exclusions), with
    /// what the model's switches can express added ([`fusion_policy`]).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn derive_fusion_with(&mut self, policy: &crate::optimizer::Policy) -> Result<String> {
        if super::kernels::hc_mix_on() {
            lloom_tensor::bail!(
                "qwen4exp fusion=derived with qwen4exp.hc_mix: the pair kernels replace the \
                 brackets' projections, and the walk describes the brackets"
            )
        }
        if super::kernels::fused_off() {
            lloom_tensor::bail!(
                "qwen4exp fusion=derived with qwen4exp.fused_off: the walk describes the fused \
                 launches, not the composed step"
            )
        }
        if self.brackets().any(|hc| hc.norm_gemv()) {
            lloom_tensor::bail!(
                "qwen4exp fusion=derived with qwen4exp.hc_norm_gemv: the arm's grouped-norm \
                 prologue is {}",
                crate::optimizer::fragments::MIXED_GNORM_REFUSAL
            )
        }
        let graph = crate::optimizer::walk(&*self);
        let policy = fusion_policy(policy);
        let d = crate::optimizer::derive(&graph, &policy)?;
        let sites = &d.plan.sites;
        // The kernel fragments: the f32 SwiGLUs are one launch whatever the
        // switches say, so the plan must place every one; the f16 routed
        // step's SwiGLU and combine read are its `f16_fuse` switch, set as
        // the plan places them — all, or none.
        let placed = |k: &str| d.plan.kernels.get(k).copied().unwrap_or(0);
        let swiglus = |dt: lloom_tensor::DType| {
            graph
                .nodes()
                .iter()
                .filter(|n| {
                    n.op == crate::optimizer::Op::Mul
                        && n.inputs.iter().any(|&i| {
                            let s = graph.node(i);
                            s.op == crate::optimizer::Op::Silu && s.dtype == dt
                        })
                })
                .count()
        };
        let combines = graph
            .nodes()
            .iter()
            .filter(|n| {
                n.name == "moe_combine"
                    && n.inputs
                        .first()
                        .is_some_and(|&c| graph.node(c).name == "cast")
            })
            .count();
        let (f32_sites, f16_sites) = (
            swiglus(lloom_tensor::DType::F32),
            swiglus(lloom_tensor::DType::F16),
        );
        // The shared experts' SwiGLUs on their projections' stores: every
        // layer's, or none (one switch).
        let shared_sites = self.layers.len();
        let store_pairs = d.plan.sites_of("quant_mul").len();
        let shared_store = match (d.plan.sites_of("quant_silu").len(), store_pairs) {
            (0, 0) => false,
            (g, u) if g == shared_sites && u == shared_sites => true,
            (g, u) => lloom_tensor::bail!(
                "qwen4exp fusion=derived: the plan puts the shared SwiGLU on {g} gate and {u} up \
                 stores of {shared_sites} layers, which the one switch cannot express; plan: {}",
                d.plan
            ),
        };
        if placed("silu_mul_f32") + store_pairs != f32_sites {
            lloom_tensor::bail!(
                "qwen4exp fusion=derived: the plan leaves {} of {f32_sites} f32 SwiGLUs composed, \
                 which no switch selects; plan: {}",
                f32_sites - placed("silu_mul_f32") - store_pairs,
                d.plan
            )
        }
        let f16_fuse = match (placed("silu_mul_f16"), placed("combine_cast")) {
            _ if f16_sites == 0 && combines == 0 => None,
            (s, c) if s == f16_sites && c == combines => Some(true),
            (0, 0) => Some(false),
            (s, c) => lloom_tensor::bail!(
                "qwen4exp fusion=derived: the plan fuses {s} of {f16_sites} f16 SwiGLUs and {c} of \
                 {combines} combine reads, which the one `f16_fuse` switch cannot express; plan: {}",
                d.plan
            ),
        };
        for (site, fragment) in sites {
            let known = (fragment == "mixed_silu" && site.ends_with("_down.weight"))
                || (fragment == "mixed_sigmoid" && site.ends_with("_up.weight"))
                || (fragment == "quant_silu" && site.ends_with("ffn_gate_shexp.weight"))
                || (fragment == "quant_mul" && site.ends_with("ffn_up_shexp.weight"));
            if !known {
                lloom_tensor::bail!(
                    "qwen4exp fusion=derived: the plan fuses `{fragment}` onto `{site}`, a site \
                     no bracket flag selects; plan: {}",
                    d.plan
                )
            }
        }
        let fused =
            |site: &str, fragment: &str| sites.get(site).map(String::as_str) == Some(fragment);
        let set = |hc: &mut HyperConnection, p: &str| {
            hc.set_epilogues(
                fused(&format!("{p}_down.weight"), "mixed_silu"),
                fused(&format!("{p}_up.weight"), "mixed_sigmoid"),
            )
        };
        for (l, layer) in self.layers.iter_mut().enumerate() {
            set(&mut layer.hc_attn, &format!("blk.{l}.hc_attn"));
            set(&mut layer.hc_ffn, &format!("blk.{l}.hc_ffn"));
        }
        set(&mut self.hc_head, "output_hc");
        if let Some(on) = f16_fuse {
            self.set_f16_fuse(on);
        }
        self.set_shared_store(shared_store);
        let refused: Vec<String> = d
            .refused
            .iter()
            .map(|(f, why)| format!("{f}: {why}"))
            .collect();
        let unvalidated: Vec<String> = policy
            .unvalidated_in(&graph)
            .into_iter()
            .map(|(f, why)| format!("{f}: {why}"))
            .collect();
        let line = format!(
            "qwen4exp fusion=derived: {} ({} refused{}{}; {} unvalidated{}{})",
            d.plan,
            refused.len(),
            if refused.is_empty() { "" } else { ": " },
            refused.join("; "),
            unvalidated.len(),
            if unvalidated.is_empty() { "" } else { ": " },
            unvalidated.join("; "),
        );
        self.fusion_report = Some(line.clone());
        Ok(line)
    }

    /// The composition the model as configured runs, as a policy: the
    /// model's own ([`fusion_policy`]) with every block whose switch is off
    /// refused by that switch — the bracket epilogues where every bracket's
    /// is off, the shared expert's store blocks under
    /// `qwen4exp.shared_store` off (admitted, their measured loss
    /// notwithstanding, with it on), the f16 routed step's compositions under
    /// `qwen4exp.f16_fuse` off. A derivation over the walk under it is the
    /// step the forward dispatches (the census tests hold the two equal).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn switches_policy(&self) -> crate::optimizer::Policy {
        let mut p = fusion_policy(&crate::optimizer::Policy::default());
        let mut off = |f: &str, why: &str| p.unavailable.push((f.to_string(), why.to_string()));
        let epi = self.bracket_epilogues();
        if epi.iter().all(|e| !e.0) {
            off("mixed_silu", "every bracket's down epilogue is off");
        }
        if epi.iter().all(|e| !e.1) {
            off("mixed_sigmoid", "every bracket's up epilogue is off");
        }
        if self.layers.iter().all(|l| !l.moe().shared_store()) {
            off("quant_silu", "the switch qwen4exp.shared_store is off");
            off("quant_mul", "the switch qwen4exp.shared_store is off");
        } else {
            // The switch runs the blocks whatever their measurement.
            p.admit
                .extend(["quant_silu".to_string(), "quant_mul".to_string()]);
        }
        if self.layers.iter().all(|l| !l.moe().f16_fuse()) {
            off("silu_mul_f16", "the switch qwen4exp.f16_fuse is off");
            off("combine_cast", "the switch qwen4exp.f16_fuse is off");
        }
        p
    }

    /// Every HyperConnection bracket of the decode step: each layer's attn
    /// and ffn brackets in order, then the head's.
    fn brackets(&self) -> impl Iterator<Item = &HyperConnection> {
        self.layers
            .iter()
            .flat_map(|l| [&l.hc_attn, &l.hc_ffn])
            .chain(std::iter::once(&self.hc_head))
    }

    /// Whether every bracket's grouped norm may run inside its consumers
    /// (`QWEN4EXP_HC_NORM_GEMV`, per bracket; the products keep their
    /// epilogue switches). `fusion = derived` refuses the arm by name.
    pub fn set_hc_norm_gemv(&mut self, on: bool) {
        for l in &mut self.layers {
            l.hc_attn.set_norm_gemv(on);
            l.hc_ffn.set_norm_gemv(on);
        }
        self.hc_head.set_norm_gemv(on);
    }

    /// What `qwen4exp.fusion = derived` decided at load; `None` under `manual`.
    pub fn fusion_report(&self) -> Option<&str> {
        self.fusion_report.as_deref()
    }

    /// Every bracket's `(down, up)` epilogue arms
    /// ([`HyperConnection::epilogues`]): each layer's attn and ffn brackets in
    /// order, then the head's.
    pub fn bracket_epilogues(&self) -> Vec<(bool, bool)> {
        let mut out: Vec<(bool, bool)> = self
            .layers
            .iter()
            .flat_map(|l| [l.hc_attn.epilogues(), l.hc_ffn.epilogues()])
            .collect();
        out.push(self.hc_head.epilogues());
        out
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
    /// holds a pass already. Returns the pass when this call opened it, for
    /// [`Self::close_pass`]; dropped instead, after the body failed, it
    /// aborts the pass (`pass-plan-design.md` §24). The pass is keyed as the
    /// model's own: it opens after the embedding a driver's pass of the same
    /// shape holds, so the two are different sequences. It opens the
    /// allocation sequence `variant` of `shape` ([`Self::decode_variant`]);
    /// variant 0 is the shape's plain sequence.
    fn open_pass_variant(
        &self,
        shape: crate::nn::decode::PassShape,
        variant: u32,
    ) -> Result<Option<crate::nn::decode::OpenPass<lloom_tensor::Device>>> {
        #[cfg(feature = "metal")]
        {
            let device = self.device();
            if let Ok(m) = device.as_metal_device() {
                let m = m.execution_device()?;
                if !m.pass_plans_enabled() || m.pass_open() {
                    return Ok(None);
                }
                let Some(key) = Self::pass_key(shape, PassOpener::Model) else {
                    return Ok(None);
                };
                m.begin_pass(key.with_variant(variant))?;
                return Ok(Some(crate::nn::decode::OpenPass::adopt(device)));
            }
        }
        let _ = (shape, variant);
        Ok(None)
    }

    /// Which of this model's decode sequences a step at `pos` runs, as its
    /// pass key's variant (`PassKey::variant`, `DESIGN.md` §9.2aa): bit 0,
    /// the step completes an indexer block (`(pos + 1) % ratio == 0`), which
    /// adds the block's pooling, norm, rope and append; bit 1, the routing
    /// is resolved on the device (the device plan and its settle) rather
    /// than read back per layer; bits 2–3, the step inside or past the QSA
    /// dense range ([`Self::prefill_variant`] of one position): past it the
    /// indexer's selection adds its score, sort and cells to the step, so a
    /// plan recorded inside it is not a step past it shifted. Two steps of
    /// one variant allocate the same ordinals, so each is served its own
    /// plan instead of the one recorded at the other phase drifting on it.
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

    /// Bit 4: the step fetches its misses (`QWEN4EXP_FETCH=inside`) — the
    /// shared expert before the routed experts and a wait between, a
    /// sequence of its own.
    fn decode_variant(&self, pos: usize, device_plan: bool, fetch: bool) -> u32 {
        let ratio = self.layers.iter().find_map(|l| match &l.mixer {
            Mixer::Full(a) => a.indexer().map(|i| i.ratio()),
            Mixer::Linear(_) => None,
        });
        let block = ratio.is_some_and(|r| (pos + 1).is_multiple_of(r));
        block as u32
            | (device_plan as u32) << 1
            | self.prefill_variant(pos, 1) << 2
            | (fetch as u32) << 4
    }

    /// Which of this model's verify-window sequences a window of `t`
    /// positions at `cache.pos` runs, as its pass key's variant: bits
    /// `0..16`, the rows that complete an indexer block (the block's
    /// pooling, norm, rope and append); bit 16, the device plan (its
    /// records read back with the argmax) rather than a readback per layer;
    /// bit 17, the drafts' loads beside row 0 on the host plan; bits 18–19,
    /// the window inside / crossing / past the QSA dense range
    /// ([`Self::prefill_variant`]); and when a drafter handed the MTP head's
    /// lane to the window (`QWEN4EXP_MTP_FOLD`), the head's own: bits
    /// `20..28` its rows that complete a block, bit 28 its device plan,
    /// bits 29–30 its dense-range kind. Each is decided here as
    /// [`Self::verify_window_run`] decides it, before the pass.
    pub(crate) fn verify_variant(&self, cache: &State<P>, t: usize) -> u32 {
        let ratio = self.layers.iter().find_map(|l| match &l.mixer {
            Mixer::Full(a) => a.indexer().map(|i| i.ratio()),
            Mixer::Linear(_) => None,
        });
        let blocks = |pos: usize, n: usize| match ratio {
            Some(r) => (0..t.min(n))
                .filter(|row| (pos + row + 1).is_multiple_of(r))
                .fold(0u32, |m, row| m | 1 << row),
            None => 0,
        };
        let pos = cache.pos;
        let planned = self.device_plan && plan_on_device() && cache.device_plan_armed;
        let drafts = !planned && cache.window_loads_drafts();
        let target = blocks(pos, 16)
            | (planned as u32) << 16
            | (drafts as u32) << 17
            | self.prefill_variant(pos, t) << 18;
        let head = self.mtp.as_ref().and_then(|h| {
            h.pending_lane().map(|(hpos, armed)| {
                let device = h.layer().moe().experts().plans_on_device() && armed;
                blocks(hpos, 8) << 20 | (device as u32) << 28 | self.prefill_variant(hpos, t) << 29
            })
        });
        target | head.unwrap_or(0)
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

    fn close_pass(
        &self,
        opened: Option<crate::nn::decode::OpenPass<lloom_tensor::Device>>,
    ) -> Result<()> {
        match opened {
            Some(pass) => pass.finish(),
            None => Ok(()),
        }
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

    /// Register the expert tier on the device's ledger as a cache tenant:
    /// `topk + 1` slots as its floor, every expert resident as its want, one
    /// slot as its unit, the pool's slots as what it holds (§4.2, §4.8).
    pub fn register_expert_ledger(
        &mut self,
        device: &lloom_tensor::Device,
        bytes_per_slot: usize,
        slots: usize,
        n_experts: usize,
        topk: usize,
    ) -> Result<()> {
        let tenant = expert_tenant(bytes_per_slot, slots, n_experts, topk)?;
        #[cfg(feature = "metal")]
        if let lloom_tensor::Device::Metal(m) = device {
            use lloom_tensor::kernels::metal::ledger::Tenant;
            let tenant = std::sync::Arc::new(tenant);
            let dyn_tenant: std::sync::Arc<dyn Tenant> = tenant.clone();
            m.ledger().register(&dyn_tenant);
            self.ledger_gauge = Some(tenant);
        }
        #[cfg(not(feature = "metal"))]
        let _ = (device, tenant);
        Ok(())
    }

    /// Apply the expert tenant's grant to every layer's tier: shrink the
    /// slots in service (their pages parked, the ledger told) or grow them
    /// back. Returns the bytes parked (positive) or brought back (as a
    /// negative is not needed: the ledger's `parked` says). Nothing to do
    /// when the grant equals the slots in service.
    #[cfg(not(feature = "metal"))]
    pub fn apply_expert_grant(&self) -> Result<usize> {
        Ok(0)
    }

    #[cfg(feature = "metal")]
    pub fn apply_expert_grant(&self) -> Result<usize> {
        let Some(t) = &self.ledger_gauge else {
            return Ok(0);
        };
        // A plan open on the tiers refuses the shrink: a trailing fetching
        // settle closes it first.
        self.settle_trailing(None)?;
        let wanted = t.wanted_units();
        let mut parked = 0usize;
        #[cfg(feature = "metal")]
        if let lloom_tensor::Device::Metal(m) = self.device() {
            if self
                .layers
                .iter()
                .any(|l| wanted < l.moe.experts().active_slots())
            {
                // The one host wait of the yield, on the refusal path alone
                // (§21.13): the slots leaving service must have no reader
                // the clock has not completed. The tier never drains.
                m.wait_until_completed()?;
            }
        }
        for layer in &self.layers {
            let tier = layer.moe.experts();
            let active = tier.active_slots();
            if wanted < active {
                parked += tier.shrink_slots(wanted, t.floor_units())?;
            } else if wanted > active {
                tier.grow_slots(wanted);
            }
        }
        Ok(parked)
    }

    pub fn memory_fraction(&self) -> f64 {
        self.memory_fraction
    }

    /// Keep the load's machine reservation for the model's life.
    pub fn hold_reservation(
        &mut self,
        reservation: Option<std::sync::Arc<lloom_tensor::machine::Holding>>,
    ) {
        self.reservation = reservation;
    }

    pub fn reservation(&self) -> Option<&std::sync::Arc<lloom_tensor::machine::Holding>> {
        self.reservation.as_ref()
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
    pub fn plan_pass(&self, cache: &State<P>, seq: usize) -> PassPlan {
        let (nkv, hd) = (self.cfg.n_kv_heads, self.cfg.head_dim);
        let mut kv_grow = 0usize;
        for (l, layer) in self.layers.iter().enumerate() {
            if matches!(layer.mixer, Mixer::Full(_)) {
                kv_grow += cache.kv[l].grow_bytes_for(seq, nkv, hd);
            }
        }
        // The predecessors still pending are in the device's allocation the
        // admission reads; this pass owes what its own route keeps.
        let transient = if seq > 1 {
            let held = self.prefill_transient().forward(seq, cache.pos).held;
            usize::try_from(held).unwrap_or(usize::MAX)
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

    /// This model's prefill transient: its config, on its device, with its
    /// experts planned where they are (the streamed tier reads every
    /// layer's routing back on the host, which retires what came before).
    pub fn prefill_transient(&self) -> super::transient::PrefillTransient {
        let host_planned = self
            .layers
            .first()
            .is_some_and(|l| l.moe().experts().plans_on_device());
        let indexer = self.layers.iter().all(|l| match &l.mixer {
            Mixer::Full(q) => q.indexer().is_some(),
            Mixer::Linear(_) => true,
        });
        let route = super::transient::Route {
            mtp: self.mtp.is_some(),
            ple: self.ple.is_some(),
            indexer,
            routed_f16: self.routed_act() == lloom_tensor::DType::F16,
            ..super::transient::Route::of(&self.cfg, self.device().is_metal(), host_planned)
        };
        super::transient::PrefillTransient::new(&self.cfg, route)
    }

    /// `Model::prepare_pass` for this family (§9.5p's per-pass device
    /// read): size the pass, and when it allocates — a KV grow, a
    /// multi-token pass with a measured transient — read the device and
    /// refuse (or report) BEFORE anything moves: `currentAllocatedSize +
    /// bytes ≤ memory_fraction × recommendedMaxWorkingSetSize`. A decode
    /// step inside its slab reads nothing (§9.5e: no per-token Metal call).
    /// Idempotent per `(pos, seq)`: a driver prepares, then `forward`'s own
    /// chunk loop prepares the same pass and finds it done.
    pub fn prepare_pass_inner(&self, cache: &mut State<P>, seq: usize) -> Result<()> {
        self.prepare_pass_as(cache, seq, false)
    }

    /// [`Self::prepare_pass_inner`] for the pass about to run: a driver's,
    /// or with `own` the one this model opens itself when no driver holds a
    /// pass — whose plan, if any, is a different key.
    fn prepare_pass_as(&self, cache: &mut State<P>, seq: usize, own: bool) -> Result<()> {
        if seq == 0 || cache.prepared == Some((cache.pos, seq)) {
            return Ok(());
        }
        let mut plan = self.plan_pass(cache, seq);
        cache.prepared = Some((cache.pos, seq));
        // The ledger hears the KV row at the passes that allocate — a grow,
        // or a sequence's first — never on a decode step inside its slab.
        #[cfg(feature = "metal")]
        if plan.kv_grow > 0 || cache.pos == 0 {
            if let Some(g) = &cache.ledger {
                use lloom_tensor::kernels::metal::ledger::Tenant;
                let (nkv, hd) = (self.cfg.n_kv_heads, self.cfg.head_dim);
                let held: usize = cache
                    .kv
                    .iter()
                    .map(|k| k.bytes_held(nkv, hd))
                    .sum::<usize>()
                    + plan.kv_grow;
                let mut claim = g.claim();
                claim.floor = held;
                claim.want = held;
                g.set_claim(claim);
                g.set_held(held);
            }
        }
        // A pass a cached plan serves allocates its activations from the
        // base, which is already allocated and counted: it owes no transient.
        #[cfg(feature = "metal")]
        if seq > 1 {
            if let Ok(m) = self.device().as_metal_device() {
                let m = m.execution_device()?;
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
                    let variant = self.prefill_variant(cache.pos, seq) | ckpt_variant(cache, seq);
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
            let working_set = dev.recommended_max_working_set_size();
            // The transients come from the heap chunks first (§6.3k), whose
            // whole size the allocated figure already counts: what the chunks
            // have free is transient the pass will NOT allocate.
            // Legacy planner cannot spend foreign admitted-domain or not-yet-
            // published capacity. Admitted callers must pass their domain key.
            let heap_free = m.heap_reusable_bytes(None, &mut cache.heap_credit)?;
            let need = plan
                .kv_grow
                .checked_add(plan.transient.saturating_sub(heap_free))
                .ok_or_else(|| {
                    lloom_tensor::Error::Msg("pass demand exceeds usize after heap credit".into())
                })?;
            // Read with the free lists given back first when the first
            // figure would refuse (`MetalDevice::admit_bytes`); the arithmetic
            // is `kv_grow_fits`'.
            let mut a = m.admit_bytes(need, kv_grow_budget(working_set, self.memory_fraction));
            if a.needs_yield_retry() {
                // The tenants pledged a yield (the expert tier's slots):
                // apply it — park the slots, their pages given back — and
                // ask again on the figure after.
                let parked = self.apply_expert_grant()?;
                let (yielded, reclaimed) = (a.yield_report(), a.reclaimed);
                a = m.admit_bytes(need, kv_grow_budget(working_set, self.memory_fraction));
                a.merge_yield_report(yielded)?;
                a.reclaimed += reclaimed + parked;
            }
            let (allocated, fits) = (a.committed, a.fits);
            debug_assert_eq!(
                fits || a.machine.is_some(),
                kv_grow_fits(allocated, working_set, self.memory_fraction, need)
            );
            let gb = |b: usize| b as f64 / 1e9;
            let budget = kv_grow_budget(working_set, self.memory_fraction);
            // Formatted only when it is printed: never on an admitted pass.
            macro_rules! mem_line {
                () => {
                    format_args!(
                        "pos={} seq={} kv_grow={:.3} GB transient={:.3} GB heap_free={:.3} GB \
                         need={:.3} GB committed={:.2} GB reclaimed={:.3} GB budget={:.2} GB \
                         ({:.2} x working_set {:.2} GB) margin={} B fits={fits} pledged={} \
                         yield_contended={}",
                        plan.pos,
                        plan.seq,
                        gb(plan.kv_grow),
                        gb(plan.transient),
                        gb(heap_free),
                        gb(need),
                        gb(allocated),
                        gb(a.reclaimed),
                        gb(budget),
                        self.memory_fraction,
                        gb(working_set),
                        budget as i128 - allocated as i128 - need as i128,
                        a.pledged,
                        a.yield_contended,
                    )
                };
            }
            match self.mem_check {
                MemCheck::Report => eprintln!("MEMCHECK {}", mem_line!()),
                MemCheck::Refuse if !fits => {
                    cache.prepared = None;
                    let refusal = a.machine.unwrap_or(
                        lloom_tensor::memory_admission::AllocationError::Refused {
                            requested: need,
                            available: budget.saturating_sub(allocated),
                            domain: lloom_tensor::memory_admission::RefusalDomain::Budget,
                        },
                    );
                    m.report_refusal(
                        &format_args!(
                            "qwen4exp pass refused {}: the pass stops here, before anything is \
                             written, rather than risk the machine (DESIGN.md 9.5, 3.5): a smaller \
                             prefill tile, `--memory-fraction`, free memory, or bound the run",
                            mem_line!()
                        ),
                        &refusal,
                    );
                    return Err(refusal.into());
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Attach the MTP head.
    pub fn set_mtp(&mut self, head: super::mtp::MtpHead<Own<P>, R>) {
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

    pub fn mtp_head(&self) -> Option<&super::mtp::MtpHead<Own<P>, R>> {
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

    /// Select whether every layer's routed SwiGLU rides the gate|up launch's
    /// store (`MoeLayer::set_routed_glu`; `qwen4exp.routed_glu` by default).
    pub fn set_routed_glu(&mut self, on: bool) {
        for l in &mut self.layers {
            l.moe.set_routed_glu(on);
        }
        if let Some(m) = &mut self.mtp {
            m.layer_mut().moe_mut().set_routed_glu(on);
        }
    }

    /// Select whether every layer's shared expert runs its SwiGLU on its
    /// quantized projections' stores (`MoeLayer::set_shared_store`;
    /// `qwen4exp.shared_store` by default).
    pub fn set_shared_store(&mut self, on: bool) {
        for l in &mut self.layers {
            l.moe.set_shared_store(on);
        }
        if let Some(m) = &mut self.mtp {
            m.layer_mut().moe_mut().set_shared_store(on);
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

    /// Turn checkpoints a state built from this model keeps
    /// (`sequence-reuse-design.md` §2.3b); 0 keeps none.
    pub fn set_turn_checkpoints(&mut self, slots: usize) {
        self.turn_checkpoints = slots;
    }

    pub fn turn_checkpoints(&self) -> usize {
        self.turn_checkpoints
    }

    /// f32 elements one turn checkpoint holds per layer: a linear layer's
    /// `S` and conv window; a full layer's checkpoint is its KV lengths.
    fn checkpoint_elems(&self) -> Vec<usize> {
        let mut elems: Vec<usize> = self
            .layers
            .iter()
            .map(|l| match &l.mixer {
                Mixer::Linear(dn) => {
                    let (s, c) = dn.checkpoint_parts();
                    s + c
                }
                Mixer::Full(_) => 0,
            })
            .collect();
        // The PLE's conv history, the frame's last entry (0 without a PLE).
        elems.push(
            self.ple
                .as_ref()
                .map_or(0, |p| p.block.hc_dim() * p.block.history_len()),
        );
        elems
    }

    /// The bytes one turn checkpoint of a state takes.
    pub fn checkpoint_frame_bytes(&self) -> usize {
        self.checkpoint_elems().iter().sum::<usize>() * std::mem::size_of::<f32>()
    }

    /// Size `cache`'s checkpoint ring from this model: `turn_checkpoints`
    /// frames of the linear layers' state and the PLE's conv history.
    pub fn arm_turn_checkpoints(&self, cache: &mut State<P>) {
        cache.set_turn_checkpoints(&self.checkpoint_elems(), self.turn_checkpoints);
    }

    /// A turn checkpoint of the state as it stands, at its position: every
    /// linear layer's `S` and conv window copied into a ring frame, every
    /// full layer's KV lengths marked.
    fn checkpoint_here(&self, cache: &mut State<P>, dev: &lloom_tensor::Device) -> Result<()> {
        if cache.ckpt.slots() == 0 {
            return Ok(());
        }
        let slot = cache.ckpt.open(cache.pos, dev)?;
        let mut marks = Vec::with_capacity(self.layers.len());
        for (l, layer) in self.layers.iter().enumerate() {
            marks.push(match &layer.mixer {
                Mixer::Linear(dn) => {
                    let Some(frame) = cache.ckpt.frame(slot, l)? else {
                        lloom_tensor::bail!(
                            "qwen4exp: layer {l}'s turn checkpoint needs a ring frame"
                        )
                    };
                    let (ns, nc) = dn.checkpoint_parts();
                    let s_view = frame.narrow(0, 0, ns)?;
                    let c_view = frame.narrow(0, ns, nc)?;
                    dn.store_checkpoint(
                        l,
                        &cache.s_state,
                        &cache.conv_state,
                        dev,
                        &GdnCheckpoint {
                            s: &s_view,
                            conv: &c_view,
                        },
                    )?;
                    WindowMark::default()
                }
                Mixer::Full(_) => {
                    let (tokens, blocks) = cache.kv[l].lens();
                    WindowMark { tokens, blocks }
                }
            });
        }
        if let Some(ple) = &self.ple {
            if let Some(frame) = cache.ckpt.frame(slot, self.layers.len())? {
                let h = match cache.ple_conv.get(ple.layer) {
                    Some(h) => h.to_dtype(lloom_tensor::DType::F32)?,
                    None => Tensor::zeros(
                        (ple.block.hc_dim(), ple.block.history_len()),
                        lloom_tensor::DType::F32,
                        dev,
                    )?,
                };
                frame.slice_set(&h.contiguous()?.flatten_all()?, 0, 0)?;
            }
            cache.ckpt_ngram[slot] = Some(cache.ngram.clone());
        }
        cache.ckpt.close(slot, marks);
        Ok(())
    }

    /// Open the checkpoint a pass over `lens` tiles (in order, from
    /// `cache.pos`) takes inside itself at `rows` positions in — the tile
    /// the position falls in and the rows into that tile — for
    /// `Layer::prefill_mix` to take (`State::ckpt_take`); nothing when
    /// `rows` is `None`.
    fn ckpt_pass_open(
        &self,
        cache: &mut State<P>,
        lens: &[usize],
        rows: Option<usize>,
        dev: &lloom_tensor::Device,
    ) -> Result<()> {
        let Some(r) = rows else {
            return Ok(());
        };
        let mut off = 0;
        let mut found = None;
        for (t, &len) in lens.iter().enumerate() {
            if r <= off + len {
                found = Some((t, r - off));
                break;
            }
            off += len;
        }
        let Some((tile, rows)) = found else {
            lloom_tensor::bail!(
                "qwen4exp: a checkpoint {r} positions into a pass of {} positions",
                lens.iter().sum::<usize>()
            )
        };
        let slot = cache.ckpt.open(cache.pos + r, dev)?;
        let n = self.layers.len();
        cache.ckpt_pass = Some(PassCkpt {
            tile,
            rows,
            slot,
            seen: vec![0; n],
            marks: vec![WindowMark::default(); n],
            ngram: None,
        });
        Ok(())
    }

    /// Close the pass's checkpoint: every layer ran the tile; its marks.
    fn ckpt_pass_close(&self, cache: &mut State<P>) -> Result<()> {
        let Some(p) = cache.ckpt_pass.take() else {
            return Ok(());
        };
        if let Some(l) = p.seen.iter().position(|&seen| seen <= p.tile) {
            lloom_tensor::bail!(
                "qwen4exp: layer {l} did not run tile {} of the pass its checkpoint is in",
                p.tile
            )
        }
        if self.ple.is_some() {
            let Some(ngram) = p.ngram else {
                lloom_tensor::bail!("qwen4exp: the PLE did not run the tile its checkpoint is in")
            };
            cache.ckpt_ngram[p.slot] = Some(ngram);
        }
        cache.ckpt.close(p.slot, p.marks);
        Ok(())
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Apply the runtime axes (`--kv-quant`, …) to the model's config: every
    /// `State<P>` built from `config()` afterwards takes them. Refused on a
    /// device the arm has no kernels on (the quantized KV is Metal only).
    pub fn apply_axes(&mut self, args: &super::config::ConfigArgs) -> Result<()> {
        let mut cfg = self.cfg.clone();
        args.apply(&mut cfg).map_err(lloom_tensor::Error::msg)?;
        self.set_kv_precision(cfg.kv_precision)
    }

    /// Set the KV store every `State<P>` built from `config()` afterwards uses
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

    pub fn layers(&self) -> &[Layer<P, R>] {
        &self.layers
    }

    pub fn layers_mut(&mut self) -> &mut [Layer<P, R>] {
        &mut self.layers
    }

    /// Attach the N-gram PLE at `layer`.
    pub fn set_ple(
        &mut self,
        layer: usize,
        hash: NgramHash,
        table: PleTable,
        block: PleLayer<P>,
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
    fn ple_delta(&self, ple: &Ple<P>, res_hc: &Tensor, cache: &mut State<P>) -> Result<Tensor> {
        self.ple_window_cut(res_hc)?;
        let seq = res_hc.dim(1)?;
        // The turn checkpoint inside the pass: the n-gram window and the
        // conv history after its rows (the gather path, which sees them).
        let cap = cache.ckpt_ple_rows(ple.layer);
        if let Some(rows) = cap {
            let n_heads = ple.hash.n_heads();
            let mut ids = vec![0i64; seq * n_heads];
            let mut at = (rows == 0).then(|| cache.ngram.clone());
            for t in 0..seq {
                let token = cache.pending_tokens.pop_front();
                let token = token.or(self.cfg.ple_image_token_id);
                let window = cache.ngram.window(token);
                ple.hash
                    .row_ids(&window, &mut ids[t * n_heads..(t + 1) * n_heads]);
                cache.ngram.push(token);
                if t + 1 == rows {
                    at = Some(cache.ngram.clone());
                }
            }
            let emb = ple.table.gather(&ids, res_hc.device())?;
            let emb = emb.reshape((seq, emb.elem_count() / seq))?;
            let (out, hist) =
                ple.block
                    .forward_at(res_hc, &emb, &mut cache.ple_conv, ple.layer, Some(rows))?;
            let (Some(hist), Some(ngram)) = (hist, at) else {
                lloom_tensor::bail!("qwen4exp: the PLE's checkpoint at {rows} rows of {seq}")
            };
            cache.ckpt_ple_store(&hist, ngram)?;
            return Ok(out);
        }
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
        ple: &Ple<P>,
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
                // SAFETY: the pointer is only compared with null.
                if unsafe { buf.contents() }.is_null() {
                    return Ok(None);
                }
                let storage =
                    lloom_tensor::Storage::Metal(lloom_tensor::MetalStorage::new_allocated(
                        buf.clone(),
                        m.clone(),
                        n,
                        lloom_tensor::DType::F32,
                    )?);
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
        cache: &mut State<P>,
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
        cache: &mut State<P>,
        obs: &mut O,
    ) -> Result<Tensor> {
        let (b, seq, _) = embeds.dims3()?;
        if b != 1 || seq == 0 {
            lloom_tensor::bail!(
                "qwen4exp forward takes one sequence of >= 1 tokens, got {:?}",
                embeds.shape()
            )
        }
        // A trailing fetching settle runs before a multi-row pass.
        #[cfg(feature = "metal")]
        if seq > 1 {
            self.settle_trailing(Some(cache))?;
        }
        // The last decode step is final once another forward begins — or,
        // when its settle trails this step's encode (`Provisional::Behind`),
        // held behind this step's window until the driver settles it.
        #[cfg(feature = "metal")]
        if seq == 1 && cache.provisional == crate::nn::model::Provisional::Behind {
            // The driver settles behind every forward it encodes, which
            // releases the window held before it: one still held here is a
            // settle the driver skipped, and settling it now would settle
            // the newest step too, ahead of its time.
            if cache.window_held() {
                lloom_tensor::bail!(
                    "qwen4exp: a decode step's window still held at the next step's entry — its \
                     driver did not settle behind the forward it encoded (`CausalLM::settle_step`)"
                )
            }
            cache.hold_standing()?;
        } else {
            cache.settle_standing()?;
        }
        #[cfg(not(feature = "metal"))]
        cache.settle_standing()?;
        // The turn checkpoint this forward leaves (§2.3b): at the hinted
        // position when a pass covers it, else at the end of a multi-token
        // pass a caller never hinted; none under a verify window.
        cache.ckpt.drop_stale_hint(cache.pos);
        let windowed = cache.window_open();
        let dev = embeds.device().clone();
        let mut logits = None;
        match self.prefill_mode {
            PrefillMode::Replay => {
                let want = cache
                    .ckpt
                    .rows_for(cache.pos, seq, true, seq, windowed)
                    .map(|r| cache.pos + r);
                if want == Some(cache.pos) {
                    self.checkpoint_here(cache, &dev)?;
                }
                for t in 0..seq {
                    let h = embeds.narrow(1, t, 1)?;
                    // Only the last position's logits are returned; held across
                    // the next position's pass they would be an undeclared
                    // escape (`pass-plan-design.md` §26).
                    drop(logits.take());
                    logits = Some(self.decode_embed(&h, cache, obs)?);
                    if want == Some(cache.pos) {
                        self.checkpoint_here(cache, &dev)?;
                    }
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
                    // As above: only the last chunk's logits are returned.
                    drop(logits.take());
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
                        let rows = cache.ckpt.rows_for(
                            cache.pos,
                            total,
                            start + total == seq,
                            seq,
                            windowed,
                        );
                        let ckpt = ckpt_variant(cache, total);
                        self.ckpt_pass_open(cache, &lens, rows, &dev)?;
                        self.prepare_pass_as(cache, total, true)?;
                        let bracket = self.open_pass_variant(
                            crate::nn::decode::PassShape::Prefill {
                                w: total,
                                fresh: cache.pos == 0,
                            },
                            self.prefill_variant(cache.pos, total) | ckpt,
                        )?;
                        logits = Some(self.prefill_embeds_pipelined(&tiles, cache)?);
                        self.close_pass(bracket)?;
                        self.ckpt_pass_close(cache)?;
                        start += total;
                        continue;
                    }
                    let h = embeds.narrow(1, start, len)?;
                    let rows =
                        cache
                            .ckpt
                            .rows_for(cache.pos, len, start + len == seq, seq, windowed);
                    // Each chunk is a pass of its own: admitted here when the
                    // driver handed the whole prompt over untiled.
                    self.prepare_pass_as(cache, len, true)?;
                    logits = Some(if len == 1 {
                        // A decode step: the state before or after it.
                        if rows == Some(0) {
                            self.checkpoint_here(cache, &dev)?;
                        }
                        let l = self.decode_embed(&h, cache, obs)?;
                        if rows == Some(1) {
                            self.checkpoint_here(cache, &dev)?;
                        }
                        l
                    } else {
                        let ckpt = ckpt_variant(cache, len);
                        self.ckpt_pass_open(cache, &[len], rows, &dev)?;
                        let bracket = self.open_pass_variant(
                            crate::nn::decode::PassShape::Prefill {
                                w: len,
                                fresh: cache.pos == 0,
                            },
                            self.prefill_variant(cache.pos, len) | ckpt,
                        )?;
                        let l = self.prefill_embeds(&h, cache, obs)?;
                        self.close_pass(bracket)?;
                        self.ckpt_pass_close(cache)?;
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
        cache: &mut State<P>,
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
        cache.keep_nextn(&res_hc)?;

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
        cache: &mut State<P>,
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
    pub fn prefill_embeds_pipelined(
        &self,
        tiles: &[Tensor],
        cache: &mut State<P>,
    ) -> Result<Tensor> {
        let n_tiles = tiles.len();
        if n_tiles == 0 {
            lloom_tensor::bail!("prefill_embeds_pipelined: no tiles")
        }
        PIPELINED_PASSES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        let stage_x = |res: &mut Tensor, l: usize, cache: &mut State<P>| -> Result<Staged> {
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
            cache.keep_nextn(r)?;
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
        cache: &mut State<P>,
        obs: &mut O,
    ) -> Result<Tensor> {
        let planned = self.device_plan
            && plan_on_device()
            && !cache.window_open()
            && !(0..self.layers.len())
                .any(|l| obs.wants::<HcStatePoint>(l) || obs.wants::<LayerOutputPoint>(l));
        // A trailing fetching settle runs before any other kind of step; a
        // fetching step settles it after its own encode. A provisional step
        // the driver left unsettled is settled here, whatever comes next.
        #[cfg(feature = "metal")]
        {
            let behind = cache.provisional == crate::nn::model::Provisional::Behind;
            if behind && planned {
                // The step before this one settles after this one's encode,
                // by the driver; anything older is a contract the driver did
                // not keep.
                if self.pending.lock().len() > 1 {
                    self.settle_trailing(Some(cache))?;
                }
            } else if !(planned && self.fetches(cache)) {
                self.settle_trailing(Some(cache))?;
            } else {
                self.settle_trailing_devices()?;
            }
        }
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
        // The arm (`QWEN4EXP_FETCH=inside`): a fresh cache, or a pool that
        // is missing, fetches inside the step — each layer waits on the
        // device for its misses — instead of taking the host plan.
        #[cfg(feature = "metal")]
        if self.fetches(cache) {
            return self.decode_fetching(h, cache, obs, false);
        }
        // The arm (`QWEN4EXP_PREDICT=hotlist`): before a device-planned
        // token, every layer's tier stages the hotlist's experts not
        // resident; its plan lands them before it maps the pool (#849's D
        // with the prediction the hotlist gives).
        if cache.device_plan_armed && predict_hot() {
            let misses = self.stream_misses();
            if misses != cache.predict_misses_seen {
                cache.predict_misses_seen = misses;
                for layer in &self.layers {
                    layer.moe().experts().prefetch_hot()?;
                }
            }
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
            // The arm (`QWEN4EXP_REPLAY=device`): the plan arms after the
            // first host-planned pass whatever it loaded — in a steady miss
            // regime no pass is ever miss-free, so the default's switch
            // never arms and every token is host-planned (48 drains); the
            // arm puts those tokens on the device plan, their misses
            // settled and replayed there.
            let clean = self.stream_misses() == before;
            if clean || replay_on_device() {
                cache.device_plan_armed = true;
                cache.armed_through_misses = !clean;
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
        // A provisional step (`State::provisional`): the settle is the
        // driver's, once it has read the logits back — no wait here, so the
        // host is not standing between this pass and the draw after it.
        #[cfg(feature = "metal")]
        if cache.provisional != crate::nn::model::Provisional::Off && !records.is_empty() {
            let marker = self.device().as_metal_device()?.flush_marker()?;
            self.pending.lock().push_back(Pending::Device(PendingStep {
                records,
                marker,
                h: h.clone(),
            }));
            cache.keep_standing();
            return Ok(logits);
        }
        let (redo, after_first) = self.settle_records(&records)?;
        if !redo {
            // The step's window stays open as the standing one.
            cache.keep_standing();
            return Ok(logits);
        }
        self.redo_device_step(h, cache, obs, after_first)
    }

    /// A device-planned decode step whose settle found a layer computed on
    /// a missing expert, redone. `h` is the step's input (the model's own
    /// copy), the cache's open window the step's own (standing or not);
    /// `after_first` the experts the settle loaded past the first layer that
    /// missed.
    fn redo_device_step<O: ForwardObserver>(
        &self,
        h: &Tensor,
        cache: &mut State<P>,
        obs: &mut O,
        after_first: u64,
    ) -> Result<Tensor> {
        cache.settle_after_first += after_first;
        // The arm (`QWEN4EXP_FETCH=inside`): the step is redone fetching
        // (its later layers' misses loaded inside it, not one more replay
        // each), and the steps after it fetch until the pool holds again.
        #[cfg(feature = "metal")]
        if self.fetch_ready() {
            return self.decode_fetching(h, cache, obs, true);
        }
        let t0 = std::time::Instant::now();
        // The arm (`QWEN4EXP_REPLAY=device`): the plan stays armed through
        // the miss. The settle loaded what the pass missed; the replay runs
        // on the device plan again — right up to the layer past the loaded
        // one at worst, so it may miss once more one layer further, at
        // most once per layer (#849's "the device plan staying armed
        // through misses": one settle readback a replay instead of the
        // host plan's drain at every layer). The bound exhausted, the host
        // plan once, as the default does.
        if replay_on_device() {
            let bound = self.layers.len().max(1);
            let mut h = h.clone();
            for _ in 0..bound {
                self.note_replay()?;
                cache.rewind_window()?;
                cache.redos += 1;
                cache.device_replays += 1;
                cache.open_window()?;
                if predict_hot() {
                    for layer in &self.layers {
                        layer.moe().experts().prefetch_hot()?;
                    }
                }
                h = self.replay_input_owned(&h)?;
                let (logits, records) = self.decode_embed_pass(&h, cache, obs, RowPlan::Device)?;
                self.replay_read_encoded(&h);
                let records: Vec<(usize, Tensor)> = records
                    .into_iter()
                    .filter_map(|(l, r)| r.into_device().map(|t| (l, t)))
                    .collect();
                let (redo, after_first) = self.settle_records(&records)?;
                if !redo {
                    cache.keep_standing();
                    cache.replay_ns += t0.elapsed().as_nanos() as u64;
                    return Ok(logits);
                }
                cache.settle_after_first += after_first;
            }
            cache.replay_fallbacks += 1;
        }
        self.note_replay()?;
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
    /// step, #814 part 5), else a `cat` and one readback. The wait is for
    /// everything encoded so far — the settle of the pass just encoded.
    fn read_records_host(&self, records: &[(usize, Tensor)]) -> Result<Vec<Vec<u32>>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        #[cfg(feature = "metal")]
        if let Ok(m) = records[0].1.device().as_metal_device() {
            let marker = m.flush_marker()?;
            return self.read_records_at(records, &marker);
        }
        self.read_records_cat(records)
    }

    /// [`Self::read_records_host`] for a pass whose last commit is `marker`:
    /// the wait is for that pass only, so a pass encoded after it keeps
    /// running (the settle trailing the next encode), and its records are
    /// its own (each tier alternates its record slot per plan,
    /// `DecodeIo::record_slots`).
    #[cfg(feature = "metal")]
    fn read_records_at(
        &self,
        records: &[(usize, Tensor)],
        marker: &lloom_tensor::metal_backend::CommandMarker,
    ) -> Result<Vec<Vec<u32>>> {
        if records.is_empty() {
            return Ok(Vec::new());
        }
        let m = records[0].1.device().as_metal_device()?;
        m.wait_marker(marker)?;
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
        self.read_records_cat(records)
    }

    /// The records through a `cat` and one readback (a record not in its
    /// tier's slot: a multi-row plan, or off Metal).
    fn read_records_cat(&self, records: &[(usize, Tensor)]) -> Result<Vec<Vec<u32>>> {
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

    /// [`Self::settle_records`] for a pass whose last commit is `marker`.
    #[cfg(feature = "metal")]
    fn settle_records_at(
        &self,
        records: &[(usize, Tensor)],
        marker: &lloom_tensor::metal_backend::CommandMarker,
    ) -> Result<(bool, u64)> {
        if records.is_empty() {
            return Ok((false, 0));
        }
        let owned = self.read_records_at(records, marker)?;
        let parts: Vec<(&R, &[u32])> = records
            .iter()
            .zip(&owned)
            .map(|((l, _), v)| (self.layers[*l].moe().experts(), v.as_slice()))
            .collect();
        Self::redo_of(R::settle_batch(&parts)?)
    }

    /// [`Self::settle_records_at`] for a fetching pass: the bookkeeping
    /// alone (`ExpertResidency::settle_fetched_batch`) — it may run while
    /// the fetch worker serves the next pass, so it loads nothing.
    #[cfg(feature = "metal")]
    fn settle_records_fetched_at(
        &self,
        records: &[(usize, Tensor)],
        marker: &lloom_tensor::metal_backend::CommandMarker,
    ) -> Result<(bool, u64)> {
        if records.is_empty() {
            return Ok((false, 0));
        }
        let owned = self.read_records_at(records, marker)?;
        let parts: Vec<(&R, &[u32])> = records
            .iter()
            .zip(&owned)
            .map(|((l, _), v)| (self.layers[*l].moe().experts(), v.as_slice()))
            .collect();
        Self::redo_of(R::settle_fetched_batch(&parts)?)
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
        declared_copy(h)
    }

    /// [`Self::replay_input`] for a one-row decode step, into a buffer the
    /// model keeps: the copy asks no pool for its destination. The previous
    /// step's replay (if it ran) must have retired first, since it read the
    /// same buffer; otherwise, and for a shape the buffer is not, the copy
    /// goes to the pool as `replay_input` makes it.
    fn replay_input_owned(&self, h: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if let Ok(m) = h.device().as_metal_device() {
            let m = m.execution_device()?;
            if m.activations_may_be_reused() {
                let clock = m.gpu_clock();
                let mut g = self.replay_h.lock();
                // A replay of a step whose input is already the model's own:
                // the same buffer, read again.
                if g.bufs.iter().flatten().any(|(t, _)| t.id() == h.id()) {
                    return Ok(h.clone());
                }
                // The other buffer than the last step's: that one may still
                // be a pending step's input (`ReplayInputs`).
                let at = g.at ^ 1;
                let fits = matches!(&g.bufs[at], Some((t, _)) if t.dims() == h.dims() && t.dtype() == h.dtype());
                if !fits {
                    // Outlives every pass: no plan serves it.
                    let _declared = lloom_tensor::EscapeGuard::new();
                    g.bufs[at] = Some((Tensor::zeros(h.dims(), h.dtype(), h.device())?, None));
                }
                // Never the input a pending step's redo would replay — the
                // alternation keeps it the other buffer; this keeps it so.
                let named = {
                    let t = &g.bufs[at].as_ref().expect("made above").0;
                    self.pending.lock().iter().any(|p| match p {
                        Pending::Device(d) => d.h.id() == t.id(),
                        Pending::Fetch(f) => f.h.id() == t.id(),
                    })
                };
                let (t, read) = g.bufs[at].as_mut().expect("made above");
                if !named && read.is_none_or(|e| clock.is_complete(e)) {
                    // The copy as `replay_input` makes it, minus the fill.
                    let _declared = lloom_tensor::EscapeGuard::new();
                    t.slice_set(&h.contiguous()?, 0, 0)?;
                    *read = None;
                    let t = t.clone();
                    g.at = at;
                    return Ok(t);
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
            for (t, read) in g.bufs.iter_mut().flatten() {
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
    fn note_replay(&self) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Ok(m) = self.device().as_metal_device() {
            m.execution_device()?.pass_diverged();
        }
        // The tiers: the replay's resolves must not count the settled
        // token's experts into the hotlist again (#899's finding).
        for layer in &self.layers {
            layer.moe().experts().note_replay();
        }
        Ok(())
    }

    /// Whether the fetch arm (`QWEN4EXP_FETCH=inside`) can run on this
    /// model: every layer's tier fetches (`ExpertResidency::fetch_ready`)
    /// on a Metal device.
    fn fetch_ready(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            crate::nn::expert_stream::fetch_on()
                && self.device().is_metal()
                && !self.layers.is_empty()
                && self.layers.iter().all(|l| l.moe().experts().fetch_ready())
        }
        #[cfg(not(feature = "metal"))]
        false
    }

    /// Whether the next decode step on `cache` fetches its misses: the arm
    /// ready, and the cache in the fetching regime or not yet armed (a
    /// fresh cache's first step fetches instead of taking the host plan).
    fn fetches(&self, cache: &State<P>) -> bool {
        (cache.fetch_mode || !cache.device_plan_armed) && self.fetch_ready()
    }

    /// The model's fetch worker, made at first use.
    #[cfg(feature = "metal")]
    fn fetch_lane(&self) -> Result<Option<std::sync::Arc<crate::nn::expert_fetch::FetchLane>>> {
        let dev = self.device();
        let Ok(m) = dev.as_metal_device() else {
            return Ok(None);
        };
        if let Some(l) = self.fetch_lane.get() {
            return Ok(l.clone());
        }
        let t = &crate::tuning::tuning().stream;
        let lane = crate::nn::expert_fetch::FetchLane::new(
            m,
            std::time::Duration::from_micros(t.fetch_poll_us as u64),
            &t.fetch_qos,
        )?;
        Ok(self.fetch_lane.get_or_init(|| Some(lane)).clone())
    }

    /// The fetch worker's counters (zeros before the first fetching step,
    /// or off Metal), settled: every job registered so far is done before
    /// the counters are read (`FetchLane::settled_stats`), so a snapshot
    /// taken right after a decode counts the decode's last job.
    pub fn fetch_stats(&self) -> crate::nn::expert_stream::FetchStats {
        #[cfg(feature = "metal")]
        if let Some(Some(l)) = self.fetch_lane.get() {
            return l.settled_stats();
        }
        crate::nn::expert_stream::FetchStats::default()
    }

    /// Settle a fetching step whose settle trailed: its records read back
    /// once its own pass completed (one wait — none when the pass is done,
    /// and never on the steps encoded after it), its pass's jobs, and the
    /// regime's counters on `cache` when there is one. `Some(h)` — the
    /// step's input, for its redo — when it computed on a missing expert
    /// (the fetch worker's fail-safe fired: no slot or no routing within its
    /// bound).
    #[cfg(feature = "metal")]
    fn settle_fetch(
        &self,
        p: PendingFetch,
        cache: Option<&mut State<P>>,
    ) -> Result<Option<Tensor>> {
        let PendingFetch {
            records,
            pass,
            before,
            marker,
            h,
        } = p;
        let (redo, after_first) = self.settle_records_fetched_at(&records, &marker)?;
        pass.finish();
        if redo {
            if let Some(cache) = cache {
                cache.settle_after_first += after_first;
            }
            return Ok(Some(h));
        }
        if let Some(cache) = cache {
            // The decision lags a step: the step after the one that ends the
            // regime was already encoded fetching, so its settle finds the
            // regime off and is not a second exit.
            if self.stream_misses() == before {
                cache.fetch_clean += 1;
                if cache.fetch_mode && cache.fetch_clean >= fetch_exit() {
                    cache.fetch_mode = false;
                    cache.fetch_exits += 1;
                    cache.armed_through_misses = false;
                }
            } else {
                cache.fetch_clean = 0;
            }
        }
        Ok(None)
    }

    /// Settle whatever step's settle still trails — a fetching step held
    /// for the next step's encode, a provisional step the driver did not
    /// settle — oldest first, before work that must not run beside it (a
    /// multi-row pass, a verify window, a grant, a step the driver did not
    /// settle behind); then the window held behind the standing one, if
    /// `cache` has one. Their tokens were handed out already, so a step that
    /// computed on a missing expert is an error here, never a silent redo:
    /// only the driver's settle (`CausalLM::settle_step`) can still take a
    /// token back.
    #[cfg(feature = "metal")]
    pub(crate) fn settle_trailing(&self, mut cache: Option<&mut State<P>>) -> Result<()> {
        loop {
            let p = self.pending.lock().pop_front();
            let Some(p) = p else { break };
            let redo = match p {
                Pending::Fetch(f) => self.settle_fetch(f, cache.as_deref_mut())?.is_some(),
                Pending::Device(d) => self.settle_records_at(&d.records, &d.marker)?.0,
            };
            if redo {
                lloom_tensor::bail!(
                    "qwen4exp: a decode step computed on a missing expert after its token was \
                     handed out — a fetching step past the fetch worker's fail-safe \
                     (QWEN4EXP_FETCH_SETTLE=now settles each step before its token), or a \
                     provisional step its driver did not settle (`CausalLM::settle_step`)"
                )
            }
        }
        if let Some(cache) = cache {
            cache.resolve_held()?;
        }
        Ok(())
    }

    /// [`Self::settle_trailing`] for the provisional device-planned steps at
    /// the queue's head only: a fetching step's entry, whose own held settle
    /// runs after its encode.
    #[cfg(feature = "metal")]
    fn settle_trailing_devices(&self) -> Result<()> {
        loop {
            let p = {
                let mut q = self.pending.lock();
                match q.front() {
                    Some(Pending::Device(_)) => q.pop_front(),
                    _ => None,
                }
            };
            let Some(Pending::Device(d)) = p else { break };
            if self.settle_records_at(&d.records, &d.marker)?.0 {
                lloom_tensor::bail!(
                    "qwen4exp: a provisional decode step computed on a missing expert and reached \
                     the model's next entry unsettled — its driver hands a step's token out only \
                     after `CausalLM::settle_step`"
                )
            }
        }
        Ok(())
    }

    /// Whether any layer's record flags a pass that computed on a missing
    /// expert — read in place once the pass completed, nothing settled.
    #[cfg(feature = "metal")]
    fn records_flagged(
        &self,
        records: &[(usize, Tensor)],
        marker: &lloom_tensor::metal_backend::CommandMarker,
    ) -> Result<bool> {
        Ok(self.read_records_at(records, marker)?.iter().any(|v| {
            let n = v.len() / 3;
            v[2 * n..].iter().any(|&f| f != 0)
        }))
    }

    /// Everything encoded so far complete: a redo behind the next step
    /// undoes that step, which must have run (and the PLE host fill read
    /// its token) before it is taken back.
    #[cfg(feature = "metal")]
    fn sync_encoded(&self) -> Result<()> {
        let m = self.device();
        let m = m.as_metal_device()?;
        let marker = m.flush_marker()?;
        m.wait_marker(&marker)
    }

    /// A pending step that will never be settled — encoded after a step that
    /// is being redone: its plans close (`ExpertResidency::discard_plan`),
    /// its fetching pass's jobs end. Its pass has completed.
    #[cfg(feature = "metal")]
    fn discard_pending(&self, p: Pending) {
        for (l, _) in p.records() {
            self.layers[*l].moe().experts().discard_plan();
        }
        if let Pending::Fetch(f) = p {
            f.pass.finish();
        }
    }

    /// The driver's settle (`CausalLM::settle_step`): every pending step,
    /// oldest first — under `Provisional::Behind` every one but the newest,
    /// whose settle trails the next step's encode. A step that settles clean
    /// releases the window held behind the next one. A step that computed on
    /// a missing expert is redone: the steps encoded after it are dropped
    /// (their passes complete, their plans closed, the cache back to the
    /// redone step's window) and `Some(logits)` of the redo returned — the
    /// driver takes back the draws made after it.
    #[cfg(feature = "metal")]
    fn settle_step_inner(&self, cache: &mut State<P>) -> Result<Option<Tensor>> {
        use crate::nn::model::Provisional;
        let keep = usize::from(cache.provisional == Provisional::Behind);
        loop {
            let p = {
                let mut q = self.pending.lock();
                if q.len() <= keep {
                    None
                } else {
                    q.pop_front()
                }
            };
            let Some(p) = p else { break };
            // A step encoded after this one: its window is the standing one
            // and this one's is held.
            let mut newer = !self.pending.lock().is_empty();
            // A flagged step is redone, and every step after it ran on its
            // wrong output: they are dropped FIRST — their passes complete,
            // their plans closed, the cache back to this step's window — so
            // the settle's loads run with no pass beside them and no other
            // plan open (in place, a load may not evict under an open plan).
            if newer {
                let (records, marker) = match &p {
                    Pending::Device(d) => (&d.records, &d.marker),
                    Pending::Fetch(f) => (&f.records, &f.marker),
                };
                if self.records_flagged(records, marker)? {
                    self.sync_encoded()?;
                    let dropped: Vec<Pending> = self.pending.lock().drain(..).collect();
                    for d in dropped {
                        self.discard_pending(d);
                    }
                    cache.drop_newer()?;
                    newer = false;
                }
            }
            let redo = match p {
                Pending::Fetch(f) => self.settle_fetch(f, Some(cache))?.map(|h| (h, None)),
                Pending::Device(d) => {
                    let (redo, after_first) = self.settle_records_at(&d.records, &d.marker)?;
                    redo.then_some((d.h, Some(after_first)))
                }
            };
            let Some((h, after_first)) = redo else {
                if newer {
                    cache.resolve_held()?;
                }
                continue;
            };
            if newer {
                // Flagged nowhere and still redone: not a shape the settle
                // produces (a redo is a flag); refuse rather than redo under
                // a step encoded on its output.
                lloom_tensor::bail!(
                    "qwen4exp: a provisional step was redone without a flag while a later step \
                     ran on its output"
                )
            }
            cache.provisional_redos += 1;
            return match after_first {
                Some(n) => self.redo_device_step(&h, cache, &mut (), n).map(Some),
                None => self.decode_fetching(&h, cache, &mut (), true).map(Some),
            };
        }
        // The held step had nothing to settle (a host-planned or replayed
        // step, settled inline): it stands.
        if cache.window_held() && self.pending.lock().len() <= keep {
            cache.resolve_held()?;
        }
        Ok(None)
    }

    /// Whether a fetching step's settle is trailing — held for the next
    /// step, the driver's run end, or the model's next entry. Test surface.
    #[cfg(feature = "metal")]
    #[doc(hidden)]
    pub fn fetch_settle_trailing(&self) -> bool {
        self.pending
            .lock()
            .iter()
            .any(|p| matches!(p, Pending::Fetch(_)))
    }

    /// Whether a provisional device-planned step waits for its settle.
    /// Test surface.
    #[cfg(feature = "metal")]
    #[doc(hidden)]
    pub fn step_settle_trailing(&self) -> bool {
        self.pending
            .lock()
            .iter()
            .any(|p| matches!(p, Pending::Device(_)))
    }

    /// Open a fetching step at the current epoch, the pass's floor: the
    /// worker waits for every epoch before it to complete before it writes a
    /// slot (`expert_fetch::floor_settled`, its `floor_waits`), so this
    /// thread does not — the step before may still be running while this one
    /// is encoded. Then each tier's routing slot is filled with the sentinel
    /// the worker tells a new routing from.
    #[cfg(feature = "metal")]
    fn fetch_begin(&self, _cache: &mut State<P>) -> Result<crate::nn::expert_fetch::FetchPass> {
        let Some(lane) = self.fetch_lane()? else {
            lloom_tensor::bail!("qwen4exp: a fetching step without a fetch worker")
        };
        let m = self.device();
        let m = m.as_metal_device()?;
        let floor = m.gpu_clock().current_epoch();
        let topk = self.cfg.n_experts_per_tok;
        for layer in &self.layers {
            layer.moe().experts().fetch_prepare(topk)?;
        }
        Ok(lane.begin_pass(floor))
    }

    /// A decode step that fetches its misses (`QWEN4EXP_FETCH=inside`,
    /// `expert-streaming.md` §20.14c): the device plan with every layer's
    /// slot map waiting for the worker that loads what the router chose —
    /// no replay, no host plan. `replay`: the step is a redo of a device
    /// plan that computed on a missing expert (its settle loaded what it
    /// saw missing). A fetching step is redone only when the worker found
    /// no slot for an expert, at most once per layer; past that, the host
    /// plan once. The regime: on from here, off after `fetch_exit` steps in
    /// a row that loaded nothing.
    #[cfg(feature = "metal")]
    fn decode_fetching<O: ForwardObserver>(
        &self,
        h: &Tensor,
        cache: &mut State<P>,
        obs: &mut O,
        replay: bool,
    ) -> Result<Tensor> {
        if !cache.fetch_mode {
            cache.fetch_mode = true;
            cache.fetch_entries += 1;
            cache.fetch_clean = 0;
        }
        let t0 = std::time::Instant::now();
        // A replay's `h` is already the model's own copy.
        let owned;
        let h = if replay {
            h
        } else {
            owned = self.replay_input_owned(h)?;
            &owned
        };
        // The settle trails (`QWEN4EXP_FETCH_SETTLE` = behind): this step
        // is encoded, then the step before it settled — waiting for that
        // pass alone, which the GPU runs while the host encodes this one —
        // and this step's records wait for the next step. A provisional
        // step's settle is the driver's instead, after its draw and before
        // its token is handed out (the one before it was settled there). A
        // replay settles at once.
        let provisional = cache.provisional != crate::nn::model::Provisional::Off;
        if !replay && (provisional || fetch_settle_behind()) {
            let before = self.stream_misses();
            let mut pass = self.fetch_begin(cache)?;
            cache.open_window()?;
            let (logits, records) =
                self.decode_embed_pass_as(h, cache, obs, RowPlan::Device, true)?;
            self.replay_read_encoded(h);
            pass.seal();
            let marker = self.device().as_metal_device()?.flush_marker()?;
            let records: Vec<(usize, Tensor)> = records
                .into_iter()
                .filter_map(|(l, r)| r.into_device().map(|t| (l, t)))
                .collect();
            // A provisional step's settle is the driver's (before this one's
            // encode on the blocking arm, after it on a lookahead ring).
            if !provisional {
                self.settle_trailing(Some(cache))?;
            }
            self.pending.lock().push_back(Pending::Fetch(PendingFetch {
                records,
                pass,
                before,
                marker,
                h: h.clone(),
            }));
            cache.fetch_passes += 1;
            cache.device_plan_armed = true;
            cache.armed_through_misses = true;
            cache.keep_standing();
            return Ok(logits);
        }
        let bound = self.layers.len().max(1);
        for attempt in 0..bound {
            if replay || attempt > 0 {
                self.note_replay()?;
                cache.rewind_window()?;
                cache.redos += 1;
                if attempt > 0 {
                    cache.fetch_redos += 1;
                }
            }
            let before = self.stream_misses();
            let pass = self.fetch_begin(cache)?;
            cache.open_window()?;
            let (logits, records) =
                self.decode_embed_pass_as(h, cache, obs, RowPlan::Device, true)?;
            self.replay_read_encoded(h);
            let records: Vec<(usize, Tensor)> = records
                .into_iter()
                .filter_map(|(l, r)| r.into_device().map(|t| (l, t)))
                .collect();
            let (redo, after_first) = self.settle_records(&records)?;
            pass.finish();
            cache.fetch_passes += 1;
            cache.device_plan_armed = true;
            cache.armed_through_misses = true;
            if !redo {
                cache.keep_standing();
                if replay || attempt > 0 {
                    cache.replay_ns += t0.elapsed().as_nanos() as u64;
                }
                if self.stream_misses() == before && !replay {
                    cache.fetch_clean += 1;
                    if cache.fetch_clean >= fetch_exit() {
                        cache.fetch_mode = false;
                        cache.fetch_exits += 1;
                        cache.armed_through_misses = false;
                    }
                } else {
                    cache.fetch_clean = 0;
                }
                return Ok(logits);
            }
            cache.settle_after_first += after_first;
        }
        cache.replay_fallbacks += 1;
        self.note_replay()?;
        cache.rewind_window()?;
        cache.redos += 1;
        cache.open_window()?;
        let logits = self
            .decode_embed_pass(h, cache, obs, RowPlan::Host)
            .map(|(l, _)| l)?;
        self.replay_read_encoded(h);
        cache.keep_standing();
        cache.replay_ns += t0.elapsed().as_nanos() as u64;
        Ok(logits)
    }

    /// The hook a device-planned decode pass runs at each layer's join
    /// points: nothing, or spike 0's wait (`QWEN4EXP_SPIKE0=wait`).
    fn join_hook(&self, fetch: bool) -> Result<JoinHook> {
        #[cfg(feature = "metal")]
        if fetch {
            if let Some(lane) = self.fetch_lane()? {
                return Ok(JoinHook::Fetch(lane));
            }
        }
        #[cfg(not(feature = "metal"))]
        let _ = fetch;
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
        cache: &mut State<P>,
        obs: &mut O,
        plan: RowPlan,
    ) -> Result<(Tensor, Vec<(usize, WindowPlan)>)> {
        self.decode_embed_pass_as(h, cache, obs, plan, false)
    }

    /// [`Self::decode_embed_pass`]; `fetch` (a device plan only): every
    /// layer waits for the fetch worker before its slot map (`JoinHook::Fetch`).
    fn decode_embed_pass_as<O: ForwardObserver>(
        &self,
        h: &Tensor,
        cache: &mut State<P>,
        obs: &mut O,
        plan: RowPlan,
        fetch: bool,
    ) -> Result<(Tensor, Vec<(usize, WindowPlan)>)> {
        let n = self.cfg.hc_count;
        let d = self.cfg.hidden_size;
        let device_plan = plan == RowPlan::Device;
        let fetch = fetch && device_plan;
        let mut records: Vec<(usize, WindowPlan)> = Vec::new();
        let bracket = self.open_pass_variant(
            crate::nn::decode::PassShape::Decode { b: 1 },
            self.decode_variant(cache.pos, device_plan, fetch) | ckpt_variant(cache, 1),
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
        let mut join = self.join_hook(fetch)?;
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
        cache.keep_nextn(&res_hc)?;
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

/// The decode step's IR (`optimizer::walk`): the token's row gather, the
/// carry's expansion over the streams, the N-gram PLE at its layer, every
/// layer, the head bracket's collapse and `lm_head` — what one position of
/// [`Qwen4Exp::decode_embed`] dispatches on Metal, with the device plan
/// armed where the residency plans on the device. The decode loop's argmax
/// is the driver's, not the model's.
#[cfg(not(target_arch = "wasm32"))]
impl<P: Profile, R: ExpertResidency> crate::optimizer::Visit for Qwen4Exp<P, R> {
    type In = ();
    type Out = crate::optimizer::NodeId;
    fn visit(&self, w: &mut crate::optimizer::Walker, (): ()) -> Self::Out {
        use crate::optimizer::Shape;
        use lloom_tensor::DType;
        let (n, d, vocab) = (self.cfg.hc_count, self.cfg.hidden_size, self.cfg.vocab_size);
        let ids = w.input("input_ids", Shape::of(&[1, 1]), DType::U32);
        let (_, table, dt) = crate::optimizer::walk::qmat_operand(&self.embed_tokens);
        let table = w.weight("token_embd.weight", table, dt);
        // A packed table's rows are gathered by the quantized kernel; a dense
        // one's by `index_select`.
        let gather = match &self.embed_tokens {
            lloom_tensor::quantized::QMatMul::QTensor(_) => "get_rows",
            _ => "is_u32",
        };
        let h = w.opaque(gather, &[table, ids], Shape::of(&[1, 1, d]), DType::F32);
        // `broadcast_as((1, 1, n, d)).reshape(..)`: one strided copy.
        let mut carry = w.opaque("copy_strided", &[h], Shape::of(&[1, 1, n * d]), DType::F32);
        for (l, layer) in self.layers.iter().enumerate() {
            if let Some(ple) = self.ple.as_ref().filter(|p| p.layer == l) {
                let delta = w.scoped(format!("blk.{l}"), |w| ple.block.visit(w, (carry, l)));
                carry = w.add(carry, delta);
            }
            let precision = self.cfg.kv_precision;
            carry = w.scoped(format!("blk.{l}"), |w| {
                layer.visit(w, (carry, l, precision))
            });
        }
        let (collapsed, _) = self.hc_head.visit(w, (carry, "output_hc", false));
        let logits = w.project_qmat("output.weight", &self.lm_head, collapsed);
        w.view("reshape", logits, Shape::of(&[1, vocab]))
    }
}

/// The state a driver builds for its shape, from the model it drives
/// (`profiles-and-memory-design.md` §3): one method per thing a driver
/// needs, implemented per shape, so generic code names the shape once
/// (`P::single`, `P::rows`) and the shapes it cannot serve refuse by name.
pub trait Shape: Profile {
    /// One sequence's state: its own on `No × One`; on `Ring × One` the
    /// recurrent channels in a `k`-draft verify ring (`spec_pools`). The
    /// `Paged` shapes are rows (`rows`), not a single sequence.
    fn single<R: ExpertResidency>(
        model: &Qwen4Exp<Self, R>,
        dev: &lloom_tensor::Device,
        k: usize,
    ) -> Result<State<Self>>;
    /// `rows` rows' states over one set of pools: `No × Paged` over
    /// `state_pools`, `Ring × Paged` over `spec_pools(rows, k)`. The `One`
    /// shapes have no rows.
    fn rows<R: ExpertResidency>(
        model: &Qwen4Exp<Self, R>,
        dev: &lloom_tensor::Device,
        rows: usize,
        k: usize,
    ) -> Result<(StatePools, Vec<State<Self>>)>;
}

fn no_rows<P: Profile>() -> lloom_tensor::Error {
    lloom_tensor::Error::msg(format!(
        "qwen4exp: a {} model serves one sequence; rows are the `Paged` shapes",
        P::KEY
    ))
}

impl<G: crate::profile::Regime> Shape for crate::profile::P<G, NoRewind, One> {
    fn single<R: ExpertResidency>(
        model: &Qwen4Exp<Self, R>,
        dev: &lloom_tensor::Device,
        _k: usize,
    ) -> Result<State<Self>> {
        model.new_cache(model.config(), dev)
    }
    fn rows<R: ExpertResidency>(
        _model: &Qwen4Exp<Self, R>,
        _dev: &lloom_tensor::Device,
        _rows: usize,
        _k: usize,
    ) -> Result<(StatePools, Vec<State<Self>>)> {
        Err(no_rows::<Self>())
    }
}

impl<G: crate::profile::Regime> Shape for crate::profile::P<G, Ring, One> {
    fn single<R: ExpertResidency>(
        model: &Qwen4Exp<Self, R>,
        dev: &lloom_tensor::Device,
        k: usize,
    ) -> Result<State<Self>> {
        let pools = model.try_spec_pools(1, k.max(1), dev)?;
        let mut cache = State::<Self>::in_ring(model.config(), &pools)?;
        model.arm_turn_checkpoints(&mut cache);
        Ok(cache)
    }
    fn rows<R: ExpertResidency>(
        _model: &Qwen4Exp<Self, R>,
        _dev: &lloom_tensor::Device,
        _rows: usize,
        _k: usize,
    ) -> Result<(StatePools, Vec<State<Self>>)> {
        Err(no_rows::<Self>())
    }
}

impl<G: crate::profile::Regime> Shape for crate::profile::P<G, NoRewind, Paged> {
    /// One row of a one-row pool (the engine's single sequence at
    /// `--batch N` warms the rows' kernels on it).
    fn single<R: ExpertResidency>(
        model: &Qwen4Exp<Self, R>,
        dev: &lloom_tensor::Device,
        _k: usize,
    ) -> Result<State<Self>> {
        let pools = model.row_pools(1, 2, dev);
        State::<Self>::in_pools(model.config(), &pools)
    }
    fn rows<R: ExpertResidency>(
        model: &Qwen4Exp<Self, R>,
        dev: &lloom_tensor::Device,
        rows: usize,
        _k: usize,
    ) -> Result<(StatePools, Vec<State<Self>>)> {
        let pools = model.row_pools(rows, 2, dev);
        let states = (0..rows)
            .map(|_| State::<Self>::in_pools(model.config(), &pools))
            .collect::<Result<Vec<_>>>()?;
        Ok((pools, states))
    }
}

impl<G: crate::profile::Regime> Shape for crate::profile::P<G, Ring, Paged> {
    fn single<R: ExpertResidency>(
        model: &Qwen4Exp<Self, R>,
        dev: &lloom_tensor::Device,
        k: usize,
    ) -> Result<State<Self>> {
        let phases = crate::nn::speculator::checked_window_width(k.max(1), 2)?;
        let pools = model.row_pools(1, phases, dev);
        State::<Self>::in_pools(model.config(), &pools)
    }
    fn rows<R: ExpertResidency>(
        model: &Qwen4Exp<Self, R>,
        dev: &lloom_tensor::Device,
        rows: usize,
        k: usize,
    ) -> Result<(StatePools, Vec<State<Self>>)> {
        let phases = crate::nn::speculator::checked_window_width(k.max(1), 2)?;
        let pools = model.row_pools(rows, phases, dev);
        let states = (0..rows)
            .map(|_| State::<Self>::in_pools(model.config(), &pools))
            .collect::<Result<Vec<_>>>()?;
        Ok((pools, states))
    }
}

impl<P: Profile, R: ExpertResidency> crate::profile::Shaped for Qwen4Exp<P, R> {
    type Profile = P;
}

impl<P: Profile, R: ExpertResidency> ModelTrait for Qwen4Exp<P, R> {
    type Config = Config;

    /// The checkpoint is GGUF (a non-uniform per-tensor quant); the safetensors
    /// `VarBuilder` path has no weights to offer. Load with
    /// [`Qwen4Exp::from_gguf`](super::loader).
    fn from_weights(_vb: VarBuilder, _cfg: &Self::Config) -> Result<Self> {
        lloom_tensor::bail!("qwen4exp loads from GGUF: use Qwen4Exp::from_gguf")
    }
}

impl<P: Profile, R: ExpertResidency> CausalLM for Qwen4Exp<P, R> {
    type Cache = State<P>;

    /// The state of the family's shape: its own channels and KV on
    /// `No × One`; a `Ring` or `Paged` state is leased from pools the
    /// driver holds (`State::in_ring` over `spec_pools`, `State::in_pools`
    /// over `state_pools`), so here it is refused by name.
    fn new_cache(&self, cfg: &Self::Config, dev: &lloom_tensor::Device) -> Result<Self::Cache> {
        // Validate the unit before state construction, not after its allocations.
        let _slab_unit = super::loader::kv_slab_bytes(cfg)?;
        let mut cache = State::<P>::own(cfg)?;
        self.arm_turn_checkpoints(&mut cache);
        #[cfg(feature = "metal")]
        if let lloom_tensor::Device::Metal(m) = dev {
            use lloom_tensor::kernels::metal::ledger::{Claim, Class, Gauge, Tenant};
            // The slabs grow a slab at a time with no ceiling of their own:
            // the row's floor is what it holds, its unit one slab across the
            // full-attention layers; `prepare_pass` keeps it current.
            let gauge = Gauge::new(
                "qwen4exp-kv",
                Claim {
                    floor: 0,
                    want: 0,
                    unit: _slab_unit,
                    class: Class::Capability,
                },
                0,
            );
            let tenant: std::sync::Arc<dyn Tenant> = gauge.clone();
            m.ledger().register(&tenant);
            cache.ledger = Some(gauge);
        }
        Ok(cache)
    }

    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Ok(m) = self.device().as_metal_device() {
            let m = m.execution_device()?;
            if m.pass_plans_enabled() {
                let variant = if seq_len == 1 {
                    let planned = self.device_plan && plan_on_device() && !cache.window_open();
                    let fetch = planned && self.fetches(cache);
                    let device_plan = planned && (cache.device_plan_armed || fetch);
                    self.decode_variant(cache.pos, device_plan, fetch)
                } else {
                    self.prefill_variant(cache.pos, seq_len)
                };
                m.set_next_pass_variant(variant | ckpt_variant(cache, seq_len));
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
    fn settle_pending(&self, cache: &mut Self::Cache) -> Result<()> {
        #[cfg(feature = "metal")]
        self.settle_trailing(Some(cache))?;
        #[cfg(not(feature = "metal"))]
        let _ = cache;
        Ok(())
    }

    /// Provisional steps on the device plan's warm path and the fetching
    /// steps (Metal; `QWEN4EXP_PROVISIONAL=off` declines): their settle — the
    /// records' one wait, and a redo when a layer computed on a missing
    /// expert — moves from the end of the step to the driver's settle: after
    /// its readback (`Settled`), where the pass has completed and the wait
    /// costs nothing, or after the next step's encode (`Behind`), the host
    /// encoding one step while the GPU runs the one before. `Behind` holds a
    /// window behind the standing one, which a pooled recurrent channel does
    /// not; it declines there.
    fn provisional(
        &self,
        cache: &mut Self::Cache,
        mode: crate::nn::model::Provisional,
    ) -> Result<crate::nn::model::Provisional> {
        use crate::nn::model::Provisional;
        #[cfg(feature = "metal")]
        {
            let able = self.device().is_metal()
                && provisional_on()
                && self.device_plan
                && self
                    .layers
                    .iter()
                    .any(|l| l.moe().experts().plans_on_device());
            cache.provisional = match mode {
                Provisional::Off => Provisional::Off,
                Provisional::Settled if able => Provisional::Settled,
                Provisional::Behind if able && cache.holds_windows() => Provisional::Behind,
                _ => Provisional::Off,
            };
            Ok(cache.provisional)
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = (cache, mode);
            Ok(Provisional::Off)
        }
    }

    fn settle_step(&self, cache: &mut Self::Cache) -> Result<Option<Tensor>> {
        #[cfg(feature = "metal")]
        {
            self.settle_step_inner(cache)
        }
        #[cfg(not(feature = "metal"))]
        {
            let _ = cache;
            Ok(None)
        }
    }

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

    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> Result<()> {
        cache.checkpoint_at(pos);
        Ok(())
    }

    fn open_checkpoint_request(
        &self,
        cache: &mut Self::Cache,
        policy: crate::nn::CheckpointPolicy,
        end: usize,
    ) -> Result<()> {
        cache.open_checkpoint_request(policy, end);
        Ok(())
    }

    fn close_checkpoint_request(&self, cache: &mut Self::Cache) -> Result<()> {
        cache.close_checkpoint_request();
        Ok(())
    }
}

impl<P: Profile> CacheReset for State<P> {
    fn reset(&mut self) {
        State::<P>::reset(self)
    }
}

impl<P: Profile, R: ExpertResidency> Qwen4Exp<P, R> {
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
        rows: &mut [crate::nn::model::Row<'_, State<P>>],
    ) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        self.settle_trailing(None)?;
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
        let mut caches: Vec<&mut State<P>> = rows.iter_mut().map(|r| &mut *r.cache).collect();
        let planned = self.device_plan
            && plan_on_device()
            && caches
                .iter()
                .all(|c| c.device_plan_armed && !c.armed_through_misses);
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
        self.note_replay()?;
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
        caches: &mut [&mut State<P>],
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
            cache.keep_nextn(&res_hc.narrow(0, r, 1)?)?;
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

impl<P: Profile, R: ExpertResidency> Qwen4Exp<P, R> {
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

    /// Fallible speculative-pool construction: reject width overflow before
    /// creating pool metadata. Like [`Self::spec_pools`], raw `k = 0` means
    /// two phases; shape/app callers resolve their minimum draft count first.
    /// Tensor products, allocator failure and model-domain admission are
    /// separate obligations, not guaranteed by this addition check.
    pub fn try_spec_pools(
        &self,
        rows: usize,
        k: usize,
        dev: &lloom_tensor::Device,
    ) -> Result<StatePools> {
        let phases = crate::nn::speculator::checked_window_width(k, 2)?;
        Ok(self.state_pools_with(rows, phases, dev))
    }

    /// The pools for `rows` rows of the `Paged` shapes: the recurrent
    /// channels slot-indexed and the KV in the paged pool, whatever the
    /// count — a `Paged` state of one row (the engine's single sequence at
    /// `--batch N`) is a row of a one-row pool, never its own slab.
    pub fn row_pools(&self, rows: usize, phases: usize, dev: &lloom_tensor::Device) -> StatePools {
        let mut pools = self.state_pools_with(rows, phases, dev);
        if pools.kv.is_none() {
            let f32 = lloom_tensor::DType::F32;
            pools.kv = Some(KvPool::new(
                rows,
                self.layers.len(),
                self.cfg.n_kv_heads,
                self.cfg.head_dim,
                f32,
                dev,
            ));
        }
        pools
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
    /// with [`State::<P>::accept_window`] (no replay). Every step is the
    /// arithmetic of a decode step (the row gemvs, the per-position state
    /// kernels), so the argmaxes are the sequential decode's. Needs
    /// `cache.ring_holds(T, ple)`.
    ///
    /// Every row is settled as real here (a streamed tier loads what any
    /// row lacks, and a row that computed on a missing expert redoes the
    /// window). A speculative verify, whose rows past the accept are not
    /// real, takes [`Self::verify_window_run`] and
    /// [`Self::verify_window_settle`] instead.
    pub fn verify_window(&self, tokens: &[u32], cache: &mut State<P>) -> Result<Vec<u32>> {
        let t = tokens.len();
        let pass = self.verify_window_run(tokens, cache)?;
        let v = self.verify_window_settle(pass, |_| t, false, cache, None)?;
        Ok(v.argmax)
    }

    /// [`Self::verify_window`]'s pass alone: the window opened and run, the
    /// argmax read back, every layer's plan record on the host — settled
    /// by [`Self::verify_window_settle`] once the accept is known. On the
    /// streamed tier a device-planned window flags a missing expert per
    /// row; a host-planned one loads row 0's (the real next token) and
    /// flags a draft row's instead of loading it (`QWEN4EXP_MTP_DRAFT_LOADS`
    /// loads those too, the A/B arm).
    pub fn verify_window_run(&self, tokens: &[u32], cache: &mut State<P>) -> Result<VerifyPass<P>> {
        self.verify_window_run_drawn(tokens, cache, None)
    }

    /// [`Self::verify_window_run`] where each row's token is `draw`'s keyed
    /// draw rather than the row's argmax ([`window_tokens`]): the sampled
    /// verify. `None` is the argmax.
    pub(crate) fn verify_window_run_drawn(
        &self,
        tokens: &[u32],
        cache: &mut State<P>,
        mut draw: Option<&mut crate::generation::LogitsProcessor>,
    ) -> Result<VerifyPass<P>> {
        let t = tokens.len();
        if t == 0 {
            lloom_tensor::bail!("qwen4exp verify_window: no tokens")
        }
        if !cache.ring_holds(t, self.ple.is_some()) {
            lloom_tensor::bail!(
                "qwen4exp verify_window: {t} positions need pooled channels with {} phases \
                 (`State::<P>::in_pools` over `spec_pools`)",
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
            let (argmax, records) =
                self.verify_rows_reference(tokens, cache, false, drafts, draw.as_deref_mut())?;
            return Ok(VerifyPass {
                argmax,
                records,
                planned: false,
                t0,
                tokens: tokens.to_vec(),
                h: None,
                drafts_loaded: drafts,
                fold: None,
            });
        }
        let input = Tensor::from_vec(tokens.to_vec(), (t, 1), &dev)?;
        let h = self.embed(&input)?; // [T, 1, hidden]
        let planned = self.device_plan
            && plan_on_device()
            && cache.device_plan_armed
            && !cache.armed_through_misses;
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
        // The pass advances the position by `t`; row 0 draws the token
        // after the anchor.
        let first_pos = cache.pos as u64 + 1;
        let (logits, plans, carries) = self.verify_pass(&h, cache, kind)?;
        let argmax = window_tokens(&logits, tokens, first_pos, draw.as_deref_mut())?; // [T]
        let run = {
            let _window = lloom_tensor::tuning::VerifyWindow::open();
            self.fold_head(&carries, &argmax, t)?
        };
        let extras: Vec<&Tensor> = run.as_ref().map_or_else(Vec::new, |r| r.read_tensors());
        let (mut argmax, records, read) = self.window_read(argmax, plans, t, &extras)?;
        let mut fold = match run {
            Some(r) => Some(r.read(read, self.mtp_topk())?),
            None => None,
        };
        if let Some(s) = draw {
            // What each row feeds (a thinking budget's forced tail where it
            // applies); the head's folded rows were keyed from the draws, so
            // a row whose fed token differs from its draw cannot stand.
            let fed = s.window_fed(&argmax);
            if fed != argmax {
                if let Some(f) = fold.as_mut() {
                    f.invalidate();
                }
            }
            argmax = fed;
        }
        // Only a redo reads `h`, after the pass; a window with no records (a
        // resident tier) cannot redo, and holding plan memory past the pass
        // would be an undeclared escape (§26).
        let h = (!records.is_empty()).then_some(h);
        Ok(VerifyPass {
            argmax,
            records,
            planned,
            t0,
            tokens: tokens.to_vec(),
            h,
            drafts_loaded: drafts,
            fold,
        })
    }

    /// The window's argmax `[T]` and its device plans' records in ONE
    /// readback (their `cat`, through the model's own destination): the
    /// round's one wait on the pass. A host plan's records are on the
    /// host already.
    /// `extras` are further `u32` tensors read in the same readback, each
    /// returned flat after the records.
    fn window_read(
        &self,
        argmax: Tensor,
        plans: Vec<(usize, WindowPlan)>,
        rows: usize,
        extras: &[&Tensor],
    ) -> Result<WindowRead> {
        let topk = self.cfg.n_experts_per_tok;
        let mut device: Vec<(usize, Tensor)> = Vec::new();
        let mut out: Vec<(usize, WindowRecord)> = Vec::with_capacity(plans.len());
        for (l, p) in plans {
            match p {
                WindowPlan::Host(r) => out.push((l, r)),
                WindowPlan::Device(t) => device.push((l, t.flatten_all()?)),
            }
        }
        if device.is_empty() && extras.is_empty() {
            return Ok((argmax.to_vec1::<u32>()?, out, Vec::new()));
        }
        let extras: Vec<Tensor> = extras
            .iter()
            .map(|t| t.flatten_all())
            .collect::<Result<_>>()?;
        let mut refs: Vec<&Tensor> = device.iter().map(|(_, t)| t).collect();
        refs.push(&argmax);
        refs.extend(extras.iter());
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
        at += rows;
        let mut read = Vec::with_capacity(extras.len());
        for t in &extras {
            let n = t.elem_count();
            let slice = all.get(at..at + n).ok_or_else(|| {
                lloom_tensor::Error::Msg(format!(
                    "window read: the cat is {} long, an extra needs {at}..{}",
                    all.len(),
                    at + n
                ))
            })?;
            read.push(slice.to_vec());
            at += n;
        }
        out.sort_by_key(|(l, _)| *l);
        Ok((am.to_vec(), out, read))
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
        pass: VerifyPass<P>,
        accept: impl Fn(&[u32]) -> usize,
        truncate: bool,
        cache: &mut State<P>,
        mut draw: Option<&mut crate::generation::LogitsProcessor>,
    ) -> Result<WindowVerdict> {
        let VerifyPass {
            mut argmax,
            mut records,
            planned,
            t0,
            tokens,
            h,
            drafts_loaded,
            fold: _,
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
            self.note_replay()?;
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
            let first_pos = cache.pos as u64 + 1;
            let (am, recs) = match &h {
                Some(h) => {
                    let kind = WindowPlanKind::Host {
                        real_rows: if truncate { 1 } else { t },
                        loaded_rows: if all_loaded { t } else { 1 },
                    };
                    let (logits, plans, _) = self.verify_pass(h, cache, kind)?;
                    let drawn = window_tokens(&logits, &tokens, first_pos, draw.as_deref_mut())?;
                    let (am, recs, _) = self.window_read(drawn, plans, t, &[])?;
                    let am = match draw.as_deref() {
                        Some(s) => s.window_fed(&am),
                        None => am,
                    };
                    (am, recs)
                }
                None => self.verify_rows_reference(&tokens, cache, !truncate, all_loaded, draw)?,
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
        cache: &mut State<P>,
        all_real: bool,
        all_loaded: bool,
        draw: Option<&mut crate::generation::LogitsProcessor>,
    ) -> Result<(Vec<u32>, Vec<(usize, WindowRecord)>)> {
        #[cfg(feature = "metal")]
        self.settle_trailing(Some(cache))?;
        let dev = self.device();
        let t = tokens.len();
        let first_pos = cache.pos as u64 + 1;
        let mut argmax = Vec::with_capacity(t);
        // Under a draw the rows' logits, drawn together once every row ran.
        let mut rows: Vec<Tensor> = Vec::new();
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
            if draw.is_some() {
                rows.push(logits.flatten_all()?);
            } else {
                argmax.push(
                    logits
                        .flatten_all()?
                        .argmax(D::Minus1)?
                        .to_scalar::<u32>()?,
                );
            }
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
        if let Some(s) = draw {
            let drawn = window_tokens(&Tensor::stack(&rows, 0)?, tokens, first_pos, Some(&mut *s))?
                .to_vec1::<u32>()?;
            argmax = s.window_fed(&drawn);
        }
        let mut records = Vec::new();
        for (l, recs) in per_layer.into_iter().enumerate() {
            if !recs.is_empty() {
                records.push((l, WindowRecord::concat_rows(&recs)?));
            }
        }
        Ok((argmax, records))
    }

    /// The layer stack over the window's `T` positions → logits `[T, vocab]`,
    /// the layers' plan records and the pre-mixer carries `[T, 1, n·d]`; the
    /// KV lengths after each position recorded in the window. A verify window
    /// ([`lloom_tensor::tuning::VerifyWindow`]), the redo's too.
    fn verify_pass(
        &self,
        h: &Tensor,
        cache: &mut State<P>,
        kind: WindowPlanKind,
    ) -> Result<VerifyPassOut> {
        let _window = lloom_tensor::tuning::VerifyWindow::open();
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
        cache.keep_nextn_rows(&res_hc, t)?;
        cache.pos += t;
        if let Some(w) = &mut cache.window {
            w.kv_after = kv_after;
        }
        let (collapsed, _) = self.hc_head.mix(&res_hc)?; // [T, 1, d]
        let logits = self.lm_head.forward(&collapsed)?; // [T, 1, vocab]
        let logits = logits
            .reshape((t, self.cfg.vocab_size))?
            .to_dtype(lloom_tensor::DType::F32)?;
        Ok((logits, records, res_hc))
    }
}

/// [`Qwen4Exp::window_read`]: the argmax, the records, and the extras.
type WindowRead = (Vec<u32>, Vec<(usize, WindowRecord)>, Vec<Vec<u32>>);

/// `Qwen4Exp::verify_pass`: the logits, the layers' plan records, the carries.
type VerifyPassOut = (Tensor, Vec<(usize, WindowPlan)>, Tensor);

impl<P: Profile, R: ExpertResidency> crate::nn::model::BatchedCausalLM for Qwen4Exp<P, R> {
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
impl<P: Profile> crate::nn::model::CacheShare for State<P> {
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
/// Or the fetch inside the token (`QWEN4EXP_FETCH=inside`): each layer
/// waits for the worker that loads what its router chose.
enum JoinHook {
    Off,
    #[cfg(feature = "metal")]
    Wait {
        ring: std::sync::Arc<lloom_tensor::metal_backend::HostFedRing>,
        next: u64,
    },
    #[cfg(feature = "metal")]
    Fetch(std::sync::Arc<crate::nn::expert_fetch::FetchLane>),
}

impl super::moe::PlanHook for JoinHook {
    fn at_join(&mut self) -> Result<()> {
        match self {
            JoinHook::Off => Ok(()),
            #[cfg(feature = "metal")]
            JoinHook::Wait { ring, next } => {
                *next += 1;
                ring.signal(*next);
                ring.encode_satisfied_wait(*next)
            }
            #[cfg(feature = "metal")]
            JoinHook::Fetch(_) => Ok(()),
        }
    }

    fn fetching(&self) -> bool {
        #[cfg(feature = "metal")]
        if let JoinHook::Fetch(_) = self {
            return true;
        }
        false
    }

    fn fetch_join<E: ExpertResidency>(&mut self, experts: &E, ids: &Tensor) -> Result<bool> {
        match self {
            #[cfg(feature = "metal")]
            JoinHook::Fetch(lane) => experts.fetch_register(lane, ids),
            _ => {
                let _ = (experts, ids);
                Ok(false)
            }
        }
    }
}

/// `weights.experts.replay_arming = device` (read once): a decode step that missed replays
/// on the device plan, the plan staying armed (#849 D's base, #892's
/// follow-up); the default replays on the host plan and disarms, as
/// before — the maintainer decides the default on the numbers.
fn replay_on_device() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.replay == "device")
}

/// Whether a decode driver may take the device plan's settles
/// (`QWEN4EXP_PROVISIONAL`, `on` | `off`, default `on`; the A/B arm).
#[cfg(feature = "metal")]
fn provisional_on() -> bool {
    crate::tuning::tuning().stream.provisional == "on"
}

/// The fetch worker holds pointers into the tiers (`expert_fetch::TierRef`):
/// a trailing settle's pass and any job still queued end before the layers
/// do. Fields drop after this runs.
#[cfg(feature = "metal")]
impl<P: Profile, R: ExpertResidency> Drop for Qwen4Exp<P, R> {
    fn drop(&mut self) {
        self.pending.get_mut().clear();
        if let Some(Some(lane)) = self.fetch_lane.get() {
            lane.quiesce();
        }
    }
}

/// A fetching step whose settle trails its return: the next step's encode
/// (`QWEN4EXP_FETCH_SETTLE=behind`) or the driver's settle (a provisional
/// step).
#[cfg(feature = "metal")]
struct PendingFetch {
    records: Vec<(usize, Tensor)>,
    pass: crate::nn::expert_fetch::FetchPass,
    before: u64,
    /// The pass's own last commit: the settle waits for it, not for the
    /// steps encoded after it.
    marker: lloom_tensor::metal_backend::CommandMarker,
    /// The step's input, the model's own copy: a provisional step's redo
    /// runs it again.
    h: Tensor,
}

/// A decode step whose settle trails its return, oldest first in
/// `Qwen4Exp::pending`: a device-planned provisional step, or a fetching
/// step held for the next encode or the driver.
#[cfg(feature = "metal")]
#[derive(Debug)]
enum Pending {
    Device(PendingStep),
    Fetch(PendingFetch),
}

#[cfg(feature = "metal")]
impl Pending {
    fn records(&self) -> &[(usize, Tensor)] {
        match self {
            Pending::Device(d) => &d.records,
            Pending::Fetch(f) => &f.records,
        }
    }
}

/// A one-row decode step's replay inputs (`Qwen4Exp::replay_input_owned`):
/// two buffers, each step copying into the one the step before did not.
/// **Two, derived (§15.2 #10c):** a step's input is read by its own passes
/// (the epoch stamped here) and by its redo while its settle is pending; a
/// settle trails its step by one at most (`Provisional::Behind`), so the
/// pending step is always the last copy's, and the copy goes to the other.
#[cfg(feature = "metal")]
#[derive(Debug, Default)]
struct ReplayInputs {
    bufs: [Option<(Tensor, Option<lloom_tensor::kernels::metal::Epoch>)>; 2],
    at: usize,
}

/// A provisional device-planned decode step (`State::provisional`).
#[cfg(feature = "metal")]
struct PendingStep {
    records: Vec<(usize, Tensor)>,
    marker: lloom_tensor::metal_backend::CommandMarker,
    h: Tensor,
}

#[cfg(feature = "metal")]
impl std::fmt::Debug for PendingStep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "PendingStep {{ layers: {} }}", self.records.len())
    }
}

#[cfg(feature = "metal")]
impl std::fmt::Debug for PendingFetch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PendingFetch {{ layers: {}, misses_before: {} }}",
            self.records.len(),
            self.before
        )
    }
}

/// Whether a fetching step's settle trails the next step's encode
/// (`QWEN4EXP_FETCH_SETTLE=behind`, the default inside the fetch arm; read
/// once).
#[cfg(feature = "metal")]
fn fetch_settle_behind() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.fetch_settle == "behind")
}

/// Fetching steps in a row that load nothing before the waits stop
/// (`QWEN4EXP_FETCH_EXIT`, read once).
fn fetch_exit() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| crate::tuning::tuning().stream.fetch_exit.max(1))
}

/// `weights.experts.predict = hotlist` (read once): before a device-planned token
/// the tiers stage the hotlist's experts not resident and the plan lands
/// them (#849's D with the hotlist's prediction); off by default.
fn predict_hot() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().stream.predict == "hotlist")
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

impl<P: Profile> State<P> {
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
impl<P: Profile> crate::nn::model::CacheRewind for State<P> {
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
    fn can_restore(&self, pos: usize) -> bool {
        State::<P>::can_restore(self, pos)
    }
    fn restore_point(&self, pos: usize) -> Option<usize> {
        State::<P>::restore_point(self, pos)
    }
    fn restore(&mut self, pos: usize) -> Result<()> {
        State::<P>::restore(self, pos)
    }
}

/// A verify window's token per row from its logits `[T, vocab]`: the argmax,
/// or `draw`'s draw with row `i` keyed to `first_pos + i` (the position of
/// the token it draws) and processed against the history the window's
/// `tokens` give it (`LogitsProcessor::window_draw_device`). A `[T]` `U32`
/// device tensor either way.
pub(crate) fn window_tokens(
    logits: &Tensor,
    tokens: &[u32],
    first_pos: u64,
    draw: Option<&mut crate::generation::LogitsProcessor>,
) -> Result<Tensor> {
    match draw {
        None => logits.argmax(D::Minus1),
        Some(s) => {
            let inputs = Tensor::from_vec(tokens.to_vec(), tokens.len(), logits.device())?;
            s.window_draw_device(logits, first_pos, Some(&inputs))
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn numeric_expert_tenant_checks_the_actual_registration_input() {
        assert!(super::expert_tenant(1, 1, 1, usize::MAX).is_err());
        assert!(super::expert_tenant(usize::MAX, 2, 2, 0).is_err());
        assert!(super::expert_tenant(usize::MAX, 1, 2, 0).is_err());
        let tenant = super::expert_tenant(100, 8, 30, 2).unwrap();
        assert_eq!(
            (
                tenant.unit(),
                tenant.wanted_units(),
                tenant.floor_units(),
                tenant.max_units()
            ),
            (100, 8, 3, 8)
        );
        assert!(super::expert_tenant(0, usize::MAX, usize::MAX, usize::MAX - 1).is_ok());
    }

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
        let mut model = super::super::testing::resident_model::<CrownExact>(&cfg, &dev);
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
        let cache = State::<CrownExact>::new(model.config());
        let full = cfg
            .layer_types
            .iter()
            .position(|k| *k == LayerKind::Full)
            .unwrap();
        assert_eq!(cache.kv[full].precision(), KvPrecision::dense(DType::F16));
        let pools = model.state_pools(2, &dev);
        let err = State::<PoolExact>::in_pools(model.config(), &pools)
            .unwrap_err()
            .to_string();
        assert!(err.contains("paged KV pool"), "{err}");
        let mut f32_cfg = model.config().clone();
        f32_cfg.kv_precision = super::super::config::DEFAULT_KV_PRECISION;
        State::<PoolExact>::in_pools(&f32_cfg, &pools).unwrap();
        // The rewindable shape keeps its own KV: every store is served there.
        let one = model.spec_pools(1, 1, &dev);
        let mut q_cfg = f32_cfg.clone();
        q_cfg.kv_precision = int8;
        State::<RewindableExact>::in_ring(&q_cfg, &one).unwrap();
        // The shapes are types: a `No × One` state is never leased from pools,
        // a `Paged` one never built on its own.
        let err = State::<PoolExact>::in_pools(&f32_cfg, &one)
            .unwrap_err()
            .to_string();
        assert!(err.contains("carries none"), "{err}");
        let err = State::<RewindableExact>::in_ring(&f32_cfg, &pools)
            .unwrap_err()
            .to_string();
        assert!(err.contains("keeps its own KV"), "{err}");
    }

    use super::super::config::LayerKind;
    use super::super::moe::Resident;
    use super::super::testing::{deltanet, hc, lin, moe, ones, qsa, wt};
    use super::*;
    use crate::profile::{CrownExact, PoolExact, RewindableExact};
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

        let layers: Vec<Layer<CrownExact, Resident>> = (0..cfg.n_layers)
            .map(|l| {
                let mixer = match cfg.layer_types[l] {
                    LayerKind::Linear => Mixer::Linear(deltanet::<CrownExact>(&cfg, &dev)),
                    LayerKind::Full => Mixer::Full(qsa::<CrownExact>(&cfg, &dev)),
                };
                Layer {
                    hc_attn: hc(&cfg, l as f32, &dev),
                    mixer,
                    hc_ffn: hc(&cfg, l as f32 + 50.0, &dev),
                    moe: moe::<CrownExact>(&cfg, &dev),
                }
            })
            .collect();

        let model = Qwen4Exp::<CrownExact, _>::new(
            embed,
            layers,
            hc(&cfg, 999.0, &dev),                           // hc_head
            lin(cfg.vocab_size, cfg.hidden_size, 7.0, &dev), // lm_head
            cfg.clone(),
        )
        .unwrap();

        let mut cache = State::<CrownExact>::new(&cfg);
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
            let block = super::super::ple::PleLayer::<CrownExact>::new(
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
            let mut with_ple = Qwen4Exp::<CrownExact, _>::new(
                QMatMul::Tensor(wt((cfg.vocab_size, cfg.hidden_size), 0.5, &dev)),
                (0..cfg.n_layers)
                    .map(|l| {
                        let mixer = match cfg.layer_types[l] {
                            LayerKind::Linear => Mixer::Linear(deltanet::<CrownExact>(&cfg, &dev)),
                            LayerKind::Full => Mixer::Full(qsa::<CrownExact>(&cfg, &dev)),
                        };
                        Layer::new(
                            hc(&cfg, l as f32, &dev),
                            mixer,
                            hc(&cfg, l as f32 + 50.0, &dev),
                            moe::<CrownExact>(&cfg, &dev),
                        )
                    })
                    .collect(),
                hc(&cfg, 999.0, &dev),
                lin(cfg.vocab_size, cfg.hidden_size, 7.0, &dev),
                cfg.clone(),
            )
            .unwrap();
            with_ple.set_ple(ple_layer, hash, table, block).unwrap();
            let toks: Vec<u32> = vec![3, 5, 2, 8, 8, 1, 9];
            let mut run = |mode: PrefillMode| -> (Vec<f32>, Vec<f32>, usize) {
                with_ple.set_prefill_mode(mode);
                let mut cache = State::<CrownExact>::new(&cfg);
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
        let model = super::super::testing::resident_model::<CrownExact>(&cfg, &dev);
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
