//! MTP on the DSpark seam: the target's `NextnHidden` probe, the one-layer
//! MTP head, and the drafter that fills the seam's reserved MTP slot
//! (`design/docs/models/qwen3.8-flash-next.md` "MTP on the DSpark seam").
//!
//! Dataflow, from the reference draft model (vLLM `qwen4_exp/nvidia/mtp.py`):
//! the target's pre-final-mixer n-stream carry `H [T, n·d]` is RMS-normed over
//! `n·d` and projected per stream by `fc_hidden [d, d]`; the NEXT token's
//! embedding is RMS-normed over `d`, projected by `fc_embedding [d, d]` and
//! added to every stream (the layer's `prev_block_output` with no injection —
//! a plain residual); one full-attention layer (HC-bracketed QSA + MoE, its
//! own experts and KV) runs on the sum; the head mixer collapses the result
//! for the SHARED `lm_head`. The two projections are held as ONE matrix
//! `eh_proj [d, 2d] = [fc_embedding | fc_hidden]` applied to `[e | h_stream]`
//! — ds4's / llama.cpp's `nextn.eh_proj` tensor, the same sum as one dot.
//! The MTP module of the HF checkpoint has exactly one such layer
//! (`mtp.layers.0`), so a draft is one token per round.
//!
//! Verify is `K+1` sequential decode steps — greedy's own math, position by
//! position — because the chunk prefill reorders fp32 (Kernel B, the chunk
//! attention) and would not be byte-identical to greedy. Rollback is the
//! cache's window: snapshot at the window start, rewind on a partial accept,
//! replay the accepted prefix (the recurrence is replayable, §10.2a).

use super::config::Config;
use super::hc::HyperConnection;
use super::model::{Cache, Layer, Qwen4Exp, VerifyPass};
use super::moe::{ExpertResidency, WindowPlan, WindowPlanKind, WindowRecord};
use crate::nn::speculator::{
    accept_prefix, Anchor, Draft, DraftTokens, KvSource, NextnHidden, Probes, SeqId, Speculator,
    VerifyRowRef,
};
use crate::nn::{CausalLM, RmsNorm};
use lloom_tensor::quantized::QMatMul;
use lloom_tensor::{Device, Module, Result, Tensor, D};

/// The one-layer MTP head.
#[derive(Debug)]
pub struct MtpHead<R: ExpertResidency> {
    /// `pre_fc_norm_embedding` — RMS over `d` on the next token's embedding.
    norm_e: RmsNorm,
    /// `pre_fc_norm_hidden` — RMS over `n·d` on the target's carry.
    norm_h: RmsNorm,
    /// `eh_proj [d, 2d]` = `[fc_embedding | fc_hidden]`, applied per stream to
    /// `[normed embedding | normed stream]`.
    eh_proj: QMatMul,
    /// `layers.0` — a full-attention layer with its own MoE.
    layer: Layer<R>,
    /// `hyper_connection_mixer` — the head collapse (no injection).
    head_mix: HyperConnection,
    n: usize,
    pub(crate) d: usize,
}

impl<R: ExpertResidency> MtpHead<R> {
    pub fn new(
        norm_e: RmsNorm,
        norm_h: RmsNorm,
        eh_proj: QMatMul,
        layer: Layer<R>,
        head_mix: HyperConnection,
        cfg: &Config,
    ) -> Self {
        Self {
            norm_e,
            norm_h,
            eh_proj,
            layer,
            head_mix,
            n: cfg.hc_count,
            d: cfg.hidden_size,
        }
    }

    /// `hidden [1, a, n·d]` (the target's carries at `a` positions) with
    /// `embeds [1, a, d]` (the embeddings of the tokens that FOLLOW those
    /// positions) → `(sample [1, a, d], multi [1, a, n·d])`: the collapsed
    /// stream the `lm_head` reads and the pre-collapse carry a further draft
    /// step would continue from. Appends `a` positions to `cache` (the MTP
    /// layer's own KV).
    pub fn forward(
        &self,
        hidden: &Tensor,
        embeds: &Tensor,
        cache: &mut Cache,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_with(hidden, embeds, cache, false)
    }

    /// [`Self::forward`] with `swap_eh`: the two halves of `eh_proj`'s input
    /// exchanged (`[h_stream | e]` — an ablation for the probe below; the
    /// shipping path passes `false`).
    pub fn forward_with(
        &self,
        hidden: &Tensor,
        embeds: &Tensor,
        cache: &mut Cache,
        swap_eh: bool,
    ) -> Result<(Tensor, Tensor)> {
        let (sample, multi, _) = self.forward_planned(hidden, embeds, cache, swap_eh, None)?;
        Ok((sample, multi))
    }

    /// [`Self::forward_with`] with the layer's routed experts planned as a
    /// window when `kind` is given (`Layer::prefill_window`): the plan's
    /// record comes back for the caller to settle after its own readback
    /// (`MtpDrafter::observe`); `None` is the plain host plan.
    pub fn forward_planned(
        &self,
        hidden: &Tensor,
        embeds: &Tensor,
        cache: &mut Cache,
        swap_eh: bool,
        kind: Option<WindowPlanKind>,
    ) -> Result<(Tensor, Tensor, Option<WindowPlan>)> {
        let (b, a, nd) = hidden.dims3()?;
        let (n, d) = (self.n, self.d);
        if b != 1 || nd != n * d || embeds.dims3()? != (1, a, d) {
            lloom_tensor::bail!(
                "mtp head: hidden {:?} / embeds {:?} for n {n} d {d}",
                hidden.shape(),
                embeds.shape()
            )
        }
        // [e_normed | h_normed_stream] per stream → eh_proj → the layer's carry.
        let e = self
            .norm_e
            .forward(embeds)?
            .reshape((1, a, 1, d))?
            .broadcast_as((1, a, n, d))?; // [1, a, n, d]
        let hn = self.norm_h.forward(hidden)?.reshape((1, a, n, d))?;
        let eh = if swap_eh {
            Tensor::cat(&[&hn, &e], D::Minus1)?
        } else {
            Tensor::cat(&[&e, &hn], D::Minus1)?
        }
        .reshape((a * n, 2 * d))?;
        let h0 = self.eh_proj.forward(&eh)?.reshape((1, a, n * d))?;
        let (multi, record) = match kind {
            Some(kind) => self.layer.prefill_window(&h0, 0, cache, kind)?,
            None => (self.layer.prefill(&h0, 0, cache)?, None),
        };
        cache.pos += a;
        let (sample, _) = self.head_mix.mix(&multi)?;
        Ok((sample, multi, record))
    }

    pub fn layer(&self) -> &Layer<R> {
        &self.layer
    }

    pub fn layer_mut(&mut self) -> &mut Layer<R> {
        &mut self.layer
    }
}

/// The MTP drafter: one token per round from the target's carry at the last
/// accepted position, re-keyed from every accepted row. One [`MtpLane`] per
/// sequence it drafts for.
pub struct MtpDrafter<'a, R: ExpertResidency> {
    head: &'a MtpHead<R>,
    /// The target's `token_embd` (shared, `mtp_use_dedicated_embeddings false`).
    embed: &'a QMatMul,
    /// The target's `lm_head` (shared).
    lm_head: &'a QMatMul,
    device: Device,
    /// Tokens proposed per round: the first from the target's carry at the
    /// last accepted position, each further one from the MTP layer re-applied
    /// on its own `multi` output with the previous draft's embedding (vLLM's
    /// scheme for `mtp_num_hidden_layers` 1: the one layer, `k` times). The
    /// re-applications run under a window on the layer's own cache and are
    /// rewound: only observed rows key it.
    k: usize,
    /// The head's indexer keys per compressed block: a lane cut back to a
    /// reused prefix keeps the blocks complete within it.
    ratio: usize,
    lanes: Vec<MtpLane>,
    /// `QWEN4EXP_MTP_PROBE` over the generation rows of this drafter's run.
    probe: GenerationProbe,
}

/// The drafter's per-sequence state, held across requests by a caller that
/// keeps its target cache across them (`MtpDrafter::with_lanes`,
/// `MtpDrafter::into_lanes`): a lane keyed from a prefix the next request
/// reuses carries that prefix's rows into it.
pub struct MtpLanes(Vec<MtpLane>);

/// What the drafter keeps for one sequence: the MTP layer's cache and where
/// the sequence's drafting stands.
struct MtpLane {
    cache: Cache,
    prompt: Vec<u32>,
    /// The tokens the cache's rows were keyed from: row `j` pairs the
    /// target's carry at position `j` with the embedding of `keyed[j + 1]`,
    /// so `rows_seen + 1` of them once the lane is seeded.
    keyed: Vec<u32>,
    /// `begin_at`s that found a reused prefix this lane was not keyed from
    /// and reset it.
    resets: u64,
    seeded: bool,
    draft: Option<u32>,
    /// The MTP layer's pre-collapse carry after the last observed row — the
    /// state a second draft step continues from.
    multi: Option<Tensor>,
    rows_seen: usize,
    /// The observes planned on the head's device plan, on its host plan,
    /// and the device-planned ones redone on a miss.
    observe_device: u64,
    observe_host: u64,
    observe_redos: u64,
}

impl MtpLane {
    fn new(cfg: &Config) -> Self {
        Self {
            cache: Cache::with_layers(cfg, 1),
            prompt: Vec::new(),
            keyed: Vec::new(),
            resets: 0,
            seeded: false,
            draft: None,
            multi: None,
            rows_seen: 0,
            observe_device: 0,
            observe_host: 0,
            observe_redos: 0,
        }
    }
}

impl<'a, R: ExpertResidency> MtpDrafter<'a, R> {
    /// A drafter for one sequence (`SeqId(0)`).
    pub fn new(
        head: &'a MtpHead<R>,
        embed: &'a QMatMul,
        lm_head: &'a QMatMul,
        cfg: &Config,
        device: &Device,
    ) -> Self {
        Self::with_seqs(head, embed, lm_head, cfg, device, 1)
    }

    /// A drafter for `n` sequences, `SeqId(0..n)`, each with its own MTP
    /// cache — the batched speculative loop's drafter (`decode_spec_batched`).
    pub fn with_seqs(
        head: &'a MtpHead<R>,
        embed: &'a QMatMul,
        lm_head: &'a QMatMul,
        cfg: &Config,
        device: &Device,
        n: usize,
    ) -> Self {
        Self {
            head,
            embed,
            lm_head,
            device: device.clone(),
            k: 1,
            ratio: cfg.indexer_compress_ratio,
            lanes: (0..n.max(1)).map(|_| MtpLane::new(cfg)).collect(),
            probe: GenerationProbe::default(),
        }
    }

    /// Draft over `lanes`, a previous drafter's state for as many
    /// sequences, in place of fresh ones: each lane is kept for the prefix
    /// its next `begin_at` reuses, or reset there.
    pub fn with_lanes(mut self, lanes: Option<MtpLanes>) -> Self {
        if let Some(MtpLanes(l)) = lanes {
            if l.len() == self.lanes.len() {
                self.lanes = l;
            }
        }
        self
    }

    /// The lanes, for the next request's drafter.
    pub fn into_lanes(mut self) -> MtpLanes {
        MtpLanes(std::mem::take(&mut self.lanes))
    }

    /// `begin_at`s over every lane that reset a lane for a reused prefix it
    /// was not keyed from.
    pub fn lane_resets(&self) -> u64 {
        self.lanes.iter().map(|l| l.resets).sum()
    }

    /// The observes planned on the head's device plan, on its host plan,
    /// and the device-planned ones redone on a miss, over every lane.
    pub fn observe_plans(&self) -> (u64, u64, u64) {
        self.lanes.iter().fold((0, 0, 0), |(d, h, r), l| {
            (
                d + l.observe_device,
                h + l.observe_host,
                r + l.observe_redos,
            )
        })
    }

    /// Propose `k` tokens a round (`k ≥ 1`; the target verifies `k + 1`
    /// positions sequentially, byte-identical to greedy whatever `k`).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k.max(1);
        self
    }

    pub fn k(&self) -> usize {
        self.k
    }

    /// Positions the MTP layer's caches hold, over every sequence.
    pub fn rows_seen(&self) -> usize {
        self.lanes.iter().map(|l| l.rows_seen).sum()
    }

    /// Positions sequence `seq`'s MTP cache holds.
    pub fn seq_rows_seen(&self, seq: SeqId) -> usize {
        self.lanes.get(seq.0).map_or(0, |l| l.rows_seen)
    }

    /// Sequence 0's MTP-layer cache (its KV), for a probe.
    pub fn cache(&self) -> &Cache {
        &self.lanes[0].cache
    }

    fn lane(&mut self, seq: SeqId) -> Result<&mut MtpLane> {
        let n = self.lanes.len();
        self.lanes.get_mut(seq.0).ok_or_else(|| {
            lloom_tensor::Error::msg(format!(
                "MtpDrafter serves {n} sequence(s), got {seq:?}"
            ))
        })
    }

    /// One sequence's draft: the carry's token, then `k − 1` re-applications.
    fn draft_one(&mut self, seq: SeqId) -> Result<Vec<u32>> {
        let (head, embed, lm_head, k) = (self.head, self.embed, self.lm_head, self.k);
        let device = self.device.clone();
        let lane = self.lane(seq)?;
        // Before the first `observe` there is no carry to draft from; propose
        // id 0 — any proposal is corrected by the verify.
        let first = lane.draft.unwrap_or(0);
        let mut tokens = vec![first];
        if k > 1 {
            if let Some(multi0) = lane.multi.clone() {
                // The layer re-applied on its own carry with the previous
                // draft's embedding, `k − 1` times, under a window rewound
                // after: the layer's cache keeps only what `observe` keyed.
                lane.cache.open_window()?;
                let mut multi = multi0;
                let mut result = Ok(());
                for _ in 1..k {
                    let prev = *tokens.last().expect("one token");
                    let toks = Tensor::from_vec(vec![prev], (1, 1), &device)?;
                    let step = (|| -> Result<u32> {
                        let emb = embed.embedding(&toks)?;
                        let (sample, next_multi) = head.forward(&multi, &emb, &mut lane.cache)?;
                        let logits = lm_head.forward(&sample)?;
                        let next = logits
                            .flatten_all()?
                            .argmax(D::Minus1)?
                            .to_scalar::<u32>()?;
                        multi = next_multi;
                        Ok(next)
                    })();
                    match step {
                        Ok(next) => tokens.push(next),
                        Err(e) => {
                            result = Err(e);
                            break;
                        }
                    }
                }
                lane.cache.rewind_window()?;
                result?;
            }
        }
        Ok(tokens)
    }
}

impl<R: ExpertResidency> Speculator for MtpDrafter<'_, R> {
    type Probe = NextnHidden;

    fn n_seq(&self) -> usize {
        self.lanes.len()
    }

    fn kv_source(&self) -> KvSource {
        KvSource::Own
    }

    fn max_draft(&self) -> usize {
        self.k
    }

    fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<()> {
        self.begin_at(seq, prompt, &[])
    }

    /// The lane is kept for `kept` iff its rows were keyed from exactly
    /// those tokens and the one after them, `prompt[0]` (row `j` reads the
    /// embedding of token `j + 1`): it is cut back to `kept.len()` rows and
    /// `prompt` is keyed after them, at the target's positions. Otherwise it
    /// is reset and keys `prompt` from its own position 0 — the head's
    /// attention reads positions only relative to each other (RoPE), so the
    /// restart costs the reused prefix's context, which the acceptance rate
    /// pays for, and nothing else.
    fn begin_at(&mut self, seq: SeqId, prompt: &[u32], kept: &[u32]) -> Result<()> {
        let ratio = self.ratio;
        let lane = self.lane(seq)?;
        let at = kept.len();
        let follow = at > 0
            && lane.rows_seen >= at
            && lane.keyed.len() > at
            && lane.keyed[..at] == kept[..]
            && prompt.first() == Some(&lane.keyed[at]);
        // A lane left mid-round by a failed run (a window open) is refused
        // by the cut, and reset.
        if follow && lane.cache.truncate_attention(at, ratio).is_ok() {
            lane.keyed.truncate(at + 1);
            lane.rows_seen = at;
        } else {
            lane.cache.reset();
            lane.keyed.clear();
            lane.rows_seen = 0;
            if at > 0 {
                lane.resets += 1;
            }
        }
        lane.prompt = prompt.to_vec();
        lane.seeded = false;
        lane.draft = None;
        lane.multi = None;
        Ok(())
    }

    fn context_len(&self, seq: SeqId) -> Option<usize> {
        self.lanes.get(seq.0).map(|l| l.cache.pos)
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        active
            .iter()
            .map(|&seq| Ok(Draft::host(self.draft_one(seq)?)))
            .collect()
    }

    /// `probe [1, a, n·d]` = the target's carries at the `a` positions the
    /// round ran; `accepted` = the tokens the target kept at those positions
    /// (the token that follows each). Row `i` pairs with the token that follows
    /// it: after the prefill that is the prompt's next token for all but the
    /// last row, and the accepted anchor for the last.
    fn observe(&mut self, seq: SeqId, probe: &Tensor, accepted: &[u32]) -> Result<()> {
        let a = probe.dim(1)?;
        if a == 0 {
            return Ok(());
        }
        let (head, embed, lm_head) = (self.head, self.embed, self.lm_head);
        let device = self.device.clone();
        let lane = self.lane(seq)?;
        let seeding = !lane.seeded;
        let nexts: Vec<u32> = if seeding {
            lane.seeded = true;
            let mut v: Vec<u32> = lane.prompt.iter().skip(1).copied().collect();
            v.extend_from_slice(accepted);
            v
        } else {
            accepted.to_vec()
        };
        if nexts.len() != a {
            lloom_tensor::bail!(
                "MtpDrafter::observe: {a} probe rows but {} next tokens (prompt {}, accepted {})",
                nexts.len(),
                lane.prompt.len(),
                accepted.len()
            )
        }
        if mtp_probe() && seeding {
            self.probe_prefill_rows(seq, probe, &nexts)?;
        }
        let lane = self.lane(seq)?;
        let toks = Tensor::from_vec(nexts.clone(), (1, a), &device)?;
        let emb = embed.embedding(&toks)?; // [1, a, d]
                                           // The head's own streamed experts planned on the device when its
                                           // pool held the last round's whole (`QWEN4EXP_MTP_OBSERVE_PLAN`):
                                           // no readback at its layer; the record is settled after the draft's
                                           // argmax — a miss rewinds the head's cache and runs the rows again
                                           // on the host plan, as the target's decode step does.
        let tier = head.layer().moe().experts();
        // The seeding observe keys the prompt's rows — wider than a round's
        // and, on a streamed head, than its pool: the grouped host plan's.
        let planned = observe_on_device() && tier.plans_on_device() && !seeding;
        let kind = planned.then(|| {
            if lane.cache.device_plan_armed {
                WindowPlanKind::Device
            } else {
                WindowPlanKind::Host {
                    real_rows: a,
                    loaded_rows: a,
                }
            }
        });
        if planned {
            lane.cache.open_window()?;
        }
        let misses_before = tier.misses_so_far();
        let (mut sample, multi, record) =
            head.forward_planned(probe, &emb, &mut lane.cache, false, kind)?;
        let last = sample.narrow(1, a - 1, 1)?;
        let logits = lm_head.forward(&last)?; // [1, 1, vocab]
        let mut next = logits
            .flatten_all()?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;
        let mut multi = multi;
        if let Some(kind) = kind {
            let rec = match record {
                Some(WindowPlan::Host(r)) => Some(r),
                Some(WindowPlan::Device(t)) => {
                    // The pass is complete (the argmax above waited for it):
                    // the record in place from the tier's slot, else read.
                    let data = match tier.read_record_in_place(&t)? {
                        Some(v) => v,
                        None => t.flatten_all()?.to_vec1::<u32>()?,
                    };
                    Some(WindowRecord::device(data, a, head.layer().moe().topk())?)
                }
                None => None,
            };
            let flagged = rec.as_ref().and_then(|r| r.first_flagged_row()).is_some();
            if let Some(r) = &rec {
                // Every row is real: closed and touched whole, or closed
                // alone before the rows run again.
                R::settle_window(&[(tier, r)], if flagged { 0 } else { a })?;
            }
            if flagged {
                lane.cache.rewind_window()?;
                lane.cache.device_plan_armed = false;
                lane.cache.open_window()?;
                let kind = WindowPlanKind::Host {
                    real_rows: a,
                    loaded_rows: a,
                };
                let (s, m, record) =
                    head.forward_planned(probe, &emb, &mut lane.cache, false, Some(kind))?;
                if let Some(WindowPlan::Host(r)) = record {
                    R::settle_window(&[(tier, &r)], a)?;
                }
                sample = s;
                let last = sample.narrow(1, a - 1, 1)?;
                let logits = lm_head.forward(&last)?;
                next = logits
                    .flatten_all()?
                    .argmax(D::Minus1)?
                    .to_scalar::<u32>()?;
                multi = m;
                lane.observe_redos += 1;
            } else if kind != WindowPlanKind::Device && tier.misses_so_far() == misses_before {
                // A host-planned observe that found its experts resident
                // arms the device plan for the rounds after it.
                lane.cache.device_plan_armed = true;
            }
            lane.cache.resolve_window()?;
            if kind == WindowPlanKind::Device {
                lane.observe_device += 1;
            } else {
                lane.observe_host += 1;
            }
        }
        if lane.keyed.is_empty() {
            lane.keyed.push(lane.prompt.first().copied().unwrap_or(0));
        }
        lane.keyed.extend_from_slice(&nexts);
        let held = lane.draft.replace(next);
        lane.multi = Some(multi.narrow(1, a - 1, 1)?);
        lane.rows_seen += a;
        if mtp_probe() {
            let rows = if seeding {
                None
            } else {
                let logits = lm_head.forward(&sample)?; // [1, a, vocab]
                Some(logits.squeeze(0)?.argmax(D::Minus1)?.to_vec1::<u32>()?)
            };
            self.probe
                .score(held.filter(|_| !seeding), rows.as_deref(), &nexts);
        }
        Ok(())
    }
}

/// `QWEN4EXP_MTP_PROBE` on the generation rows: the head's teacher-forced
/// hit rate over the rows the rounds key, `hit2` as on the prompt rows (the
/// row's argmax is the token after its next token). A round's last row is
/// the one the draft is taken from, and its next token is only known at the
/// next round's observe, so it is scored there against the draft itself —
/// at K = 1 its rate is the first draft's acceptance; the rows before it
/// (an accepted draft's) are scored on the observe that keys them.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GenerationProbe {
    /// Last rows scored, and those whose draft was the next token's next.
    pub last_rows: u64,
    pub last_hits: u64,
    /// Earlier rows of a round scored, and their hits.
    pub inner_rows: u64,
    pub inner_hits: u64,
}

impl GenerationProbe {
    /// One observe's rows: `draft`, the previous observe's draft (scored
    /// against this observe's first next token); `arg`, this observe's
    /// per-row argmax (`None` on the seeding observe, whose rows are the
    /// prompt probe's).
    fn score(&mut self, draft: Option<u32>, arg: Option<&[u32]>, nexts: &[u32]) {
        if let (Some(d), Some(&first)) = (draft, nexts.first()) {
            self.last_rows += 1;
            self.last_hits += u64::from(d == first);
        }
        if let Some(arg) = arg {
            for i in 0..arg.len().saturating_sub(1) {
                self.inner_rows += 1;
                self.inner_hits += u64::from(arg[i] == nexts[i + 1]);
            }
        }
    }

    fn line(&self) -> String {
        let rate = |h: u64, n: u64| if n == 0 { 0.0 } else { h as f64 / n as f64 };
        let (rows, hits) = (
            self.last_rows + self.inner_rows,
            self.last_hits + self.inner_hits,
        );
        format!(
            "MTPPROBE generation rows={rows} hit2={hits} ({:.3}) last_rows={} last_hit2={} ({:.3}) inner_rows={} inner_hit2={} ({:.3})",
            rate(hits, rows),
            self.last_rows,
            self.last_hits,
            rate(self.last_hits, self.last_rows),
            self.inner_rows,
            self.inner_hits,
            rate(self.inner_hits, self.inner_rows),
        )
    }
}

impl<R: ExpertResidency> MtpDrafter<'_, R> {
    /// The generation-row probe so far (zeros unless `QWEN4EXP_MTP_PROBE`).
    pub fn generation_probe(&self) -> GenerationProbe {
        self.probe
    }
}

impl<R: ExpertResidency> Drop for MtpDrafter<'_, R> {
    /// The generation-row probe's summary, once a run.
    fn drop(&mut self) {
        if mtp_probe() && self.probe != GenerationProbe::default() {
            eprintln!("{}", self.probe.line());
        }
    }
}

/// `QWEN4EXP_MTP_OBSERVE_PLAN` (read once): `device` plans the MTP head's
/// own streamed experts on the device in the observe; `host` (the default)
/// is the per-round readback.
fn observe_on_device() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.mtp_observe_plan == "device")
}

impl<R: ExpertResidency> MtpDrafter<'_, R> {
    /// `QWEN4EXP_MTP_PROBE`: on the prompt rows, the head's teacher-forced
    /// hit rate under the shipping dataflow and under four ablations of it,
    /// each run under a window on the layer's cache and rewound (nothing is
    /// keyed). `hit2` = the row's argmax equals the token that follows the
    /// row's next token (what the head is trained to predict); `hit1` = it
    /// equals the row's next token itself (the embedding it was given). The
    /// discriminator for a wiring defect: the shipping dataflow must beat
    /// every ablation on `hit2`, and an ablation that beats it names the
    /// defect (`off-by-one`: the row's own token in place of its next;
    /// `swap-eh`: the `eh_proj` halves exchanged; `zero-embed` /
    /// `zero-hidden`: one input path silenced).
    fn probe_prefill_rows(&mut self, seq: SeqId, probe: &Tensor, nexts: &[u32]) -> Result<()> {
        let a = nexts.len();
        let prompt = self.lane(seq)?.prompt.clone();
        if a < 3 || a > prompt.len() {
            return Ok(());
        }
        let own: Vec<u32> = prompt[..a].to_vec(); // the rows' own tokens
        let zeros_e = Tensor::zeros((1, a, self.head.d), probe.dtype(), &self.device)?;
        let zeros_h = probe.zeros_like()?;
        let emb_of = |toks: &[u32]| -> Result<Tensor> {
            let t = Tensor::from_vec(toks.to_vec(), (1, a), &self.device)?;
            self.embed.embedding(&t)
        };
        let variants: Vec<(&str, Tensor, Tensor, bool)> = vec![
            ("shipping", probe.clone(), emb_of(nexts)?, false),
            ("off-by-one", probe.clone(), emb_of(&own)?, false),
            ("swap-eh", probe.clone(), emb_of(nexts)?, true),
            ("zero-embed", probe.clone(), zeros_e, false),
            ("zero-hidden", zeros_h, emb_of(nexts)?, false),
        ];
        let (head, lm_head) = (self.head, self.lm_head);
        for (name, hidden, emb, swap) in variants {
            let lane = self.lane(seq)?;
            lane.cache.open_window()?;
            let run = (|| -> Result<Vec<u32>> {
                let (sample, _) = head.forward_with(&hidden, &emb, &mut lane.cache, swap)?;
                let logits = lm_head.forward(&sample)?; // [1, a, vocab]
                logits.squeeze(0)?.argmax(D::Minus1)?.to_vec1::<u32>()
            })();
            lane.cache.rewind_window()?;
            let arg = run?;
            // Row i's target: the token after its next token = nexts[i + 1]
            // (the last row has none).
            let n = a - 1;
            let hit2 = (0..n).filter(|&i| arg[i] == nexts[i + 1]).count();
            let hit1 = (0..n).filter(|&i| arg[i] == nexts[i]).count();
            eprintln!(
                "MTPPROBE variant={name} rows={n} hit2={hit2} ({:.3}) hit1={hit1} ({:.3})",
                hit2 as f64 / n as f64,
                hit1 as f64 / n as f64
            );
        }
        Ok(())
    }
}

/// `QWEN4EXP_MTP_PROBE` (read once): the prompt-row ablation probe above.
fn mtp_probe() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.mtp_probe)
}

/// `QWEN4EXP_MTP_TRACE` (read once): print every verify round's window,
/// argmax and accepted prefix. An instrument, never the shipping path.
fn mtp_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::tuning::tuning().qwen4exp.mtp_trace)
}

/// A verify pass encoded and read back, awaiting the accept decision.
#[derive(Debug)]
pub struct MtpPending {
    /// The target's argmax at each of the `k + 1` window positions.
    argmax: Vec<u32>,
    /// The window's input tokens `[anchor, proposed…]`.
    tokens: Vec<u32>,
    want_was: bool,
    /// The window ran as one pass over the phase rings
    /// (`verify_window_run`): the accept is by phase, and the streamed
    /// tier's plans are settled once it is known (`verify_window_settle`).
    pass: Option<VerifyPass>,
}

impl<R: ExpertResidency> Qwen4Exp<R> {
    /// One decode step of `tok` at the cache's position, returning the
    /// device-side argmax (no readback).
    fn step_argmax(&self, tok: u32, cache: &mut Cache) -> Result<Tensor> {
        let input = Tensor::from_vec(vec![tok], (1, 1), &self.device())?;
        let logits = CausalLM::forward(self, &input, cache.pos, cache)?; // [1, vocab]
        logits.flatten_all()?.argmax(D::Minus1)?.reshape(1)
    }

    /// Replay `tokens` from the cache's position (after a rewind), retaining
    /// the carries.
    fn replay(&self, tokens: &[u32], cache: &mut Cache) -> Result<()> {
        for &t in tokens {
            let _ = self.step_argmax(t, cache)?;
        }
        Ok(())
    }
}

impl<R: ExpertResidency> Probes<NextnHidden> for Qwen4Exp<R> {
    fn verify_and_probe(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<Vec<u32>>, Tensor)> {
        if drafts.len() != 1 || anchors.len() != 1 {
            lloom_tensor::bail!(
                "qwen4exp verify_and_probe: one sequence; got {} drafts and {} anchors",
                drafts.len(),
                anchors.len()
            )
        }
        let pending = self.verify_enqueue(&drafts[0], &anchors[0], cache)?;
        let (accepted, probe) = self.verify_complete(pending, cache, eos_ids)?;
        Ok((vec![accepted], probe))
    }

    type Pending = MtpPending;
    /// One [`MtpPending`] per row, in row order.
    type PendingRows = Vec<MtpPending>;

    /// Every row's window through [`Self::verify_enqueue`] against its own
    /// cache, one after another: a row verifies exactly as it would alone, so
    /// each row's stream is its greedy stream. Not one weight sweep for the
    /// rows — the windows are separate passes (`batched-decode-design.md`
    /// §12.4 describes the ragged pass this is not).
    fn verify_rows_enqueue(
        &self,
        rows: &mut [VerifyRowRef<'_, Self::Cache>],
    ) -> Result<Vec<MtpPending>> {
        if rows.is_empty() {
            lloom_tensor::bail!("qwen4exp verify_rows_enqueue: no rows")
        }
        rows.iter_mut()
            .map(|r| self.verify_enqueue(r.draft, &r.anchor, r.cache))
            .collect()
    }

    fn verify_rows_complete(
        &self,
        pending: Vec<MtpPending>,
        caches: &mut [&mut Self::Cache],
        eos_sets: &[&[u32]],
    ) -> Result<Vec<(Vec<u32>, Tensor)>> {
        if pending.len() != caches.len() || pending.len() != eos_sets.len() {
            lloom_tensor::bail!(
                "qwen4exp verify_rows_complete: {} pending, {} caches, {} EOS sets",
                pending.len(),
                caches.len(),
                eos_sets.len()
            )
        }
        pending
            .into_iter()
            .zip(caches.iter_mut())
            .zip(eos_sets)
            .map(|((p, c), eos)| self.verify_complete(p, c, eos))
            .collect()
    }

    /// The window `[anchor, proposed…]` as `k + 1` sequential decode steps
    /// inside an open cache window, the per-position argmax read back once.
    fn verify_enqueue(
        &self,
        draft: &Draft,
        anchor: &Anchor,
        cache: &mut Self::Cache,
    ) -> Result<MtpPending> {
        if anchor.pos != cache.pos {
            lloom_tensor::bail!(
                "qwen4exp verify: anchor pos {} != cache pos {}",
                anchor.pos,
                cache.pos
            )
        }
        let proposed: Vec<u32> = match &draft.tokens {
            DraftTokens::Host(v) => v.clone(),
            DraftTokens::Device(t) => t.flatten_all()?.to_vec1::<u32>()?,
        };
        if proposed.is_empty() {
            lloom_tensor::bail!("qwen4exp verify: no proposed tokens")
        }
        let mut tokens = Vec::with_capacity(proposed.len() + 1);
        tokens.push(anchor.last);
        tokens.extend_from_slice(&proposed);
        cache.drop_nextn();
        let want_was = cache.want_nextn;
        cache.want_nextn = true;
        // The window as ONE pass with every position's state kept (the
        // phase rings, `verify_window_run`) when the cache holds them; else
        // the sequential steps with the replay on a partial accept.
        let by_phase = cache.ring_holds(tokens.len(), self.has_ple());
        let (argmax, pass) = if by_phase {
            let pass = self.verify_window_run(&tokens, cache)?;
            (pass.argmax.clone(), Some(pass))
        } else {
            cache.open_window()?;
            let mut argmaxes = Vec::with_capacity(tokens.len());
            for &t in &tokens {
                argmaxes.push(self.step_argmax(t, cache)?);
            }
            (Tensor::cat(&argmaxes, 0)?.to_vec1::<u32>()?, None)
        };
        Ok(MtpPending {
            argmax,
            tokens,
            want_was,
            pass,
        })
    }

    /// Accept the longest agreeing prefix; on a partial accept rewind the
    /// window and replay the accepted prefix. The probe is the carry at each
    /// accepted position. Over the phase rings the streamed tier's plans are
    /// settled AFTER the accept: a rejected draft's experts are neither
    /// loaded nor touched, a draft that computed on a missing expert cuts
    /// the prefix at itself, only a miss at the real next token redoes the
    /// window (`expert-streaming.md` §20.16).
    fn verify_complete(
        &self,
        pending: MtpPending,
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, Tensor)> {
        let MtpPending {
            argmax,
            tokens,
            want_was,
            pass,
        } = pending;
        let (argmax, a, verdict) = match pass {
            Some(pass) => {
                let proposed = &tokens[1..];
                let v = self.verify_window_settle(
                    pass,
                    |argmax| accept_prefix(argmax, proposed, eos_ids).len(),
                    true,
                    cache,
                )?;
                let a = v.accepted;
                (v.argmax.clone(), a, Some(v))
            }
            None => {
                let a = accept_prefix(&argmax, &tokens[1..], eos_ids).len();
                (argmax, a, None)
            }
        };
        let accepted: Vec<u32> = argmax[..a].to_vec();
        if mtp_trace() {
            let (planned, redo, first_flagged, rejected) = verdict
                .as_ref()
                .map(|v| (v.planned, v.redo, v.first_flagged, v.rejected_flags))
                .unwrap_or((false, false, None, 0));
            eprintln!(
                "MTPTRACE pos={} window={:?} argmax={:?} accepted={:?} planned={planned} redo={redo} first_flagged={first_flagged:?} rejected_flags={rejected} misses={} evictions={}",
                cache.pos - tokens.len(),
                tokens,
                argmax,
                accepted,
                self.stream_misses_total(),
                self.stream_evictions_total(),
            );
        }
        let probe = if verdict.is_some() {
            // The accepted prefix by phase: no replay.
            cache.accept_window(a)?;
            cache.take_nextn()?
        } else if a == tokens.len() {
            cache.resolve_window()?;
            cache.take_nextn()?
        } else {
            cache.rewind_window()?;
            cache.drop_nextn();
            self.replay(&tokens[..a], cache)?;
            cache.take_nextn()?
        };
        cache.want_nextn = want_was;
        let probe = match probe {
            Some(p) => p,
            None => lloom_tensor::bail!("qwen4exp verify: the pass retained no carry"),
        };
        if probe.dim(1)? != a {
            lloom_tensor::bail!(
                "qwen4exp verify: {} carries for {a} accepted positions",
                probe.dim(1)?
            )
        }
        Ok((accepted, probe))
    }

    fn take_probe(&self, cache: &mut Self::Cache) -> Result<Tensor> {
        let (n, d) = (self.config().hc_count, self.config().hidden_size);
        match cache.take_nextn()? {
            Some(t) => Ok(t),
            None => Tensor::zeros((1, 0, n * d), lloom_tensor::DType::F32, &self.device()),
        }
    }

    fn probe_drops(&self, cache: &Self::Cache) -> u64 {
        cache.nextn_drops()
    }
}
