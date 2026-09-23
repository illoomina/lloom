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
use super::model::{Cache, Layer, Qwen4Exp};
use super::moe::ExpertResidency;
use crate::nn::speculator::{
    accept_prefix, Anchor, Draft, DraftTokens, KvSource, NextnHidden, Probes, SeqId, Speculator,
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
        let multi = self.layer.prefill(&h0, 0, cache)?;
        cache.pos += a;
        let (sample, _) = self.head_mix.mix(&multi)?;
        Ok((sample, multi))
    }

    pub fn layer(&self) -> &Layer<R> {
        &self.layer
    }

    pub fn layer_mut(&mut self) -> &mut Layer<R> {
        &mut self.layer
    }
}

/// The MTP drafter: one token per round from the target's carry at the last
/// accepted position, re-keyed from every accepted row.
pub struct MtpDrafter<'a, R: ExpertResidency> {
    head: &'a MtpHead<R>,
    /// The target's `token_embd` (shared, `mtp_use_dedicated_embeddings false`).
    embed: &'a QMatMul,
    /// The target's `lm_head` (shared).
    lm_head: &'a QMatMul,
    cache: Cache,
    device: Device,
    prompt: Vec<u32>,
    seeded: bool,
    draft: Option<u32>,
    /// Tokens proposed per round: the first from the target's carry at the
    /// last accepted position, each further one from the MTP layer re-applied
    /// on its own `multi` output with the previous draft's embedding (vLLM's
    /// scheme for `mtp_num_hidden_layers` 1: the one layer, `k` times). The
    /// re-applications run under a window on the layer's own cache and are
    /// rewound: only observed rows key it.
    k: usize,
    /// The MTP layer's pre-collapse carry after the last observed row — the
    /// state a second draft step continues from.
    multi: Option<Tensor>,
    rows_seen: usize,
}

impl<'a, R: ExpertResidency> MtpDrafter<'a, R> {
    pub fn new(
        head: &'a MtpHead<R>,
        embed: &'a QMatMul,
        lm_head: &'a QMatMul,
        cfg: &Config,
        device: &Device,
    ) -> Self {
        Self {
            head,
            embed,
            lm_head,
            cache: Cache::with_layers(cfg, 1),
            device: device.clone(),
            prompt: Vec::new(),
            seeded: false,
            draft: None,
            k: 1,
            multi: None,
            rows_seen: 0,
        }
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

    /// Positions the MTP layer has been keyed with.
    pub fn rows_seen(&self) -> usize {
        self.rows_seen
    }

    /// The MTP layer's cache (its KV), for a probe.
    pub fn cache(&self) -> &Cache {
        &self.cache
    }
}

impl<R: ExpertResidency> Speculator for MtpDrafter<'_, R> {
    type Probe = NextnHidden;

    fn n_seq(&self) -> usize {
        1
    }

    fn kv_source(&self) -> KvSource {
        KvSource::Own
    }

    fn max_draft(&self) -> usize {
        self.k
    }

    fn begin(&mut self, seq: SeqId, prompt: &[u32]) -> Result<()> {
        if seq.0 != 0 {
            lloom_tensor::bail!("MtpDrafter serves one sequence (SeqId(0)), got {:?}", seq)
        }
        self.cache.reset();
        self.prompt = prompt.to_vec();
        self.seeded = false;
        self.draft = None;
        self.multi = None;
        self.rows_seen = 0;
        Ok(())
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        if active.len() != 1 {
            lloom_tensor::bail!(
                "MtpDrafter drafts for one sequence, asked for {}",
                active.len()
            )
        }
        // Before the first `observe` there is no carry to draft from; propose
        // id 0 — any proposal is corrected by the verify.
        let first = self.draft.unwrap_or(0);
        let mut tokens = vec![first];
        if self.k > 1 {
            if let Some(multi0) = self.multi.clone() {
                // The layer re-applied on its own carry with the previous
                // draft's embedding, `k − 1` times, under a window rewound
                // after: the layer's cache keeps only what `observe` keyed.
                self.cache.open_window()?;
                let mut multi = multi0;
                let mut result = Ok(());
                for _ in 1..self.k {
                    let prev = *tokens.last().expect("one token");
                    let toks = Tensor::from_vec(vec![prev], (1, 1), &self.device)?;
                    let step = (|| -> Result<u32> {
                        let emb = self.embed.embedding(&toks)?;
                        let (sample, next_multi) =
                            self.head.forward(&multi, &emb, &mut self.cache)?;
                        let logits = self.lm_head.forward(&sample)?;
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
                self.cache.rewind_window()?;
                result?;
            }
        }
        Ok(vec![Draft::host(tokens)])
    }

    /// `probe [1, a, n·d]` = the target's carries at the `a` positions the
    /// round ran; `accepted` = the tokens the target kept at those positions
    /// (the token that follows each). Row `i` pairs with the token that follows
    /// it: after the prefill that is the prompt's next token for all but the
    /// last row, and the accepted anchor for the last.
    fn observe(&mut self, _seq: SeqId, probe: &Tensor, accepted: &[u32]) -> Result<()> {
        let a = probe.dim(1)?;
        if a == 0 {
            return Ok(());
        }
        let nexts: Vec<u32> = if !self.seeded {
            self.seeded = true;
            let mut v: Vec<u32> = self.prompt.iter().skip(1).copied().collect();
            v.extend_from_slice(accepted);
            v
        } else {
            accepted.to_vec()
        };
        if nexts.len() != a {
            lloom_tensor::bail!(
                "MtpDrafter::observe: {a} probe rows but {} next tokens (prompt {}, accepted {})",
                nexts.len(),
                self.prompt.len(),
                accepted.len()
            )
        }
        if mtp_probe() && self.rows_seen == 0 {
            self.probe_prefill_rows(probe, &nexts)?;
        }
        let toks = Tensor::from_vec(nexts, (1, a), &self.device)?;
        let emb = self.embed.embedding(&toks)?; // [1, a, d]
        let (sample, multi) = self.head.forward(probe, &emb, &mut self.cache)?;
        self.multi = Some(multi.narrow(1, a - 1, 1)?);
        let last = sample.narrow(1, a - 1, 1)?;
        let logits = self.lm_head.forward(&last)?; // [1, 1, vocab]
        let next = logits
            .flatten_all()?
            .argmax(D::Minus1)?
            .to_scalar::<u32>()?;
        self.draft = Some(next);
        self.rows_seen += a;
        Ok(())
    }
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
    fn probe_prefill_rows(&mut self, probe: &Tensor, nexts: &[u32]) -> Result<()> {
        let a = nexts.len();
        if a < 3 || a > self.prompt.len() {
            return Ok(());
        }
        let own: Vec<u32> = self.prompt[..a].to_vec(); // the rows' own tokens
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
        for (name, hidden, emb, swap) in variants {
            self.cache.open_window()?;
            let run = (|| -> Result<Vec<u32>> {
                let (sample, _) = self
                    .head
                    .forward_with(&hidden, &emb, &mut self.cache, swap)?;
                let logits = self.lm_head.forward(&sample)?; // [1, a, vocab]
                logits.squeeze(0)?.argmax(D::Minus1)?.to_vec1::<u32>()
            })();
            self.cache.rewind_window()?;
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
    *ON.get_or_init(|| std::env::var_os("QWEN4EXP_MTP_PROBE").is_some())
}

/// `QWEN4EXP_MTP_TRACE` (read once): print every verify round's window,
/// argmax and accepted prefix. An instrument, never the shipping path.
fn mtp_trace() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("QWEN4EXP_MTP_TRACE").is_some())
}

/// A verify pass encoded and read back, awaiting the accept decision.
#[derive(Debug)]
pub struct MtpPending {
    /// The target's argmax at each of the `k + 1` window positions.
    argmax: Vec<u32>,
    /// The window's input tokens `[anchor, proposed…]`.
    tokens: Vec<u32>,
    want_was: bool,
    /// The window ran as one pass over the phase rings (`verify_window`):
    /// the accept is by phase.
    by_phase: bool,
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
    /// No batched verify (B=1 here); the trait's default refuses.
    type PendingRows = ();

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
        // phase rings, `verify_window`) when the cache holds them; else the
        // sequential steps with the replay on a partial accept.
        let by_phase = cache.ring_holds(tokens.len(), self.has_ple());
        let argmax = if by_phase {
            self.verify_window(&tokens, cache)?
        } else {
            cache.open_window()?;
            let mut argmaxes = Vec::with_capacity(tokens.len());
            for &t in &tokens {
                argmaxes.push(self.step_argmax(t, cache)?);
            }
            Tensor::cat(&argmaxes, 0)?.to_vec1::<u32>()?
        };
        Ok(MtpPending {
            argmax,
            tokens,
            want_was,
            by_phase,
        })
    }

    /// Accept the longest agreeing prefix; on a partial accept rewind the
    /// window and replay the accepted prefix. The probe is the carry at each
    /// accepted position.
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
            by_phase,
        } = pending;
        let accepted = accept_prefix(&argmax, &tokens[1..], eos_ids);
        let a = accepted.len();
        if mtp_trace() {
            eprintln!(
                "MTPTRACE pos={} window={:?} argmax={:?} accepted={:?}",
                cache.pos - tokens.len(),
                tokens,
                argmax,
                accepted
            );
        }
        let probe = if by_phase {
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
