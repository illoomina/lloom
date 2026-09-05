//! The decode loop — `Session::decode`, the baseline (K=1) body of
//! `design/docs/decode-driver-design.md`, stage 1.
//!
//! The loop names two seams and neither concrete arm: a [`Speculator`] for
//! *which* proposer (stage 1 admits only [`IdentityProposer`], the K=1 identity;
//! the verify body is stage 2 and is refused rather than approximated) and a
//! [`ReadbackStrategy`] for how a step feeds the next forward and learns EOS.
//! The harness selects both and never writes the body.
//!
//! # The invariant, and what it costs to hold
//!
//! The design's hot path *does not commit, does not block, does not read
//! back*: sample on the device, feed the ring, enqueue the forward, bump
//! counters. The one thing the loop must eventually learn per token is whether
//! it was EOS, and where that wait sits decides everything:
//!
//! - **`lookahead = 0`** waits for step `s`'s publish *before* encoding forward
//!   `s`. The cache never advances past an EOS, so the arm is safe for a
//!   multi-turn conversation on any cache — and the GPU idles from the publish
//!   to the first commit of the next forward, which is the non-GPU gap
//!   (§11.2b's 0.74 ms). The publish must be committed before the host can wait
//!   on it, so this arm commits per token whatever the budget says; the
//!   `commits` counter reports it.
//! - **`lookahead = 1`** encodes forward `s` first and waits for step `s`'s
//!   publish afterwards, while that forward runs. The GPU always has the next
//!   step queued, so the gap closes — and when step `s` turns out to be EOS,
//!   forward `s` has already been enqueued: **the cache is one position past
//!   the EOS**, reported as [`DecodeStats::overrun`]. A caller that continues
//!   the conversation from that cache needs a rewind the crown configuration
//!   (`KvAppend=Cat`, `ConvState=Shuffle`) cannot express, so this arm is for a
//!   run whose cache ends with it, or for a cache that can rewind (stage 2's
//!   `advance`/`resolve` window is the mechanism).
//!
//! The ordering across steps is candle's own fence chain and is unchanged by
//! the lookahead: forward `s+1`'s first encoder waits on the ring slot's last
//! writer (publish `s+1`), which waited on the argmax, which waited on forward
//! `s`'s logits — so the GPU runs the steps serially and only the *host* has
//! stopped standing between them. What the lookahead adds is that step `s+1`
//! is committed before step `s` retires.

use crate::generation::{CommitBudget, LogitsProcessor, ReadbackCounters, ReadbackStrategy};
use crate::nn::model::{CacheReset, CausalLM};
use crate::nn::speculator::{
    Anchor, Draft, NoProbe, Probes, RoundProbe, RoundTiming, SeqId, Session, Speculator,
    TargetProbe,
};
use lloom_tensor::{Device, Result, Tensor};

/// The K=1 identity proposer: drafts nothing, so the loop takes the baseline
/// body. Stage 1's only admitted family.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityProposer;

impl Speculator for IdentityProposer {
    type Probe = NoProbe;

    fn n_seq(&self) -> usize {
        1
    }

    fn max_draft(&self) -> usize {
        0
    }

    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        Ok(())
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        Ok(active.iter().map(|_| Draft::host(Vec::new())).collect())
    }

    fn observe(
        &mut self,
        _seq: SeqId,
        _probe: &<Self::Probe as TargetProbe>::Output,
        _accepted: &[u32],
    ) -> Result<()> {
        Ok(())
    }
}

/// The knobs of one decode run.
#[derive(Debug, Clone, Copy)]
pub struct DecodeOptions {
    /// Tokens to produce at most; EOS may stop the run earlier.
    pub max_tokens: usize,
    /// How often the loop commits and prunes the in-flight queue.
    pub budget: CommitBudget,
    /// Forward passes the host may enqueue before it has learned the previous
    /// step's EOS: `0` or `1`. See the module docs for what `1` costs.
    pub lookahead: usize,
    /// The most command buffers the host may leave in flight before it waits for
    /// the GPU to catch up — explicit backpressure, decoupled from the EOS
    /// mechanism. `0` is **off** (no cap).
    ///
    /// # Why this exists, and what it replaces
    ///
    /// On the poll arm the per-token EOS wait ([`ReadbackStrategy::eos_of_last`])
    /// doubles as pacing: the host blocks ~every token, so it never runs more
    /// than a handful of forwards ahead of the GPU. An arm that learns EOS off
    /// the hot path ([`crate::generation::EosRead::HostOnDrain`]) has no such
    /// wait, so nothing bounds how far the host runs ahead — measured, in-flight
    /// ratcheted to 63 against the poll arm's ~9, and the deep queue cost the
    /// tail. `flush_and_prune` only drops *completed* buffers; it never caps.
    ///
    /// This is the cap made explicit and per-arm: after each enqueue the loop
    /// asks the strategy to apply backpressure, which waits (by the same WFE
    /// mechanism the poll arm uses, on the published-generation counter) **only
    /// when** the host is more than `max_in_flight` generations ahead — a mapped
    /// read and a compare when it is not, so the GPU is free to run ahead up to
    /// the cap rather than being throttled to one forward per token.
    pub max_in_flight: usize,
    /// **Diagnostic, and it perturbs the thing it measures**: drain the GPU
    /// before each sample so the sample term can be timed apart from the
    /// forward pass (§11.2a). Off for every published number. Under
    /// [`Session::decode_speculative`] it drains the GPU at the end of each
    /// phase of a round instead — draft, verify, observe — so
    /// [`DecodeStats::rounds`] charges each phase its own GPU tail, and takes
    /// the profiler's per-phase counters when it is on.
    pub sample_split: bool,
}

/// One speculative round of [`Session::decode_speculative`], for the per-round
/// breakdown a B=1 verdict rests on.
#[derive(Debug, Clone, Default)]
pub struct RoundStats {
    /// Tokens the verify returned, the bonus included — before the EOS and
    /// `max_tokens` cuts, so the round's work is counted whole.
    pub accepted: usize,
    /// The three phases' brackets; see [`RoundTiming`].
    pub timing: RoundTiming,
}

impl DecodeOptions {
    /// Greedy defaults: `max_tokens`, per-token commit, no lookahead, no
    /// in-flight cap, no split.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            budget: CommitBudget::per_token(),
            lookahead: 0,
            max_in_flight: 0,
            sample_split: false,
        }
    }
}

/// What one decode run counted, for the `RESULT` line.
#[derive(Debug, Clone, Default)]
pub struct DecodeStats {
    /// Tokens returned, EOS excluded.
    pub produced: usize,
    /// Loop iterations, EOS step included.
    pub steps: u64,
    /// Submission time of the prompt's forward pass — encode, not execution.
    pub prefill_ns: u64,
    /// Per-step host time, one entry per returned token. Under `lookahead = 1`
    /// the wait sits after the forward's encode, so the entry is still one
    /// token's cadence and not a submit time.
    pub step_ns: Vec<u64>,
    /// Per-step sample time, only under `DecodeOptions::sample_split`.
    pub sample_ns: Vec<u64>,
    /// `flush_and_prune` calls made by the loop.
    pub commits: u64,
    /// The largest in-flight count `flush_and_prune` returned.
    pub max_in_flight: usize,
    /// Forward passes enqueued past the last token this run returned — the
    /// positions the cache holds beyond the sequence's token log. One under
    /// `lookahead = 1` when EOS ended the run, up to a commit period under
    /// `EosRead::HostOnDrain`, and whatever a speculative round accepted past a
    /// mid-block EOS. Computed from the cache position, so a loop cannot
    /// under-report it. The session carries it as [`Session::overrun`] until
    /// [`Session::reconcile`] discards it.
    pub overrun: usize,
    /// The strategy's own counters.
    pub readback: ReadbackCounters,
    /// One entry per speculative round, empty on the baseline body.
    pub rounds: Vec<RoundStats>,
}

/// The outcome of one decode run.
#[derive(Debug, Clone)]
pub struct DecodeOut {
    /// The produced ids, EOS excluded.
    pub ids: Vec<u32>,
    /// Whether the run stopped on an EOS rather than on `max_tokens`.
    pub eos: bool,
    pub stats: DecodeStats,
}

/// Commit without waiting and prune the in-flight queue, on the backends that
/// have one. `None` where there is no queue to prune — a CPU run reports no
/// count, which is honest rather than zero (§14.1).
fn flush_and_prune(device: &Device) -> Result<Option<usize>> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return Ok(Some(m.flush_and_prune()?));
    }
    let _ = device;
    Ok(None)
}

impl<'m, M: CausalLM> Session<'m, M> {
    /// Decode from `prompt` against this session's cache, advancing `pos`.
    ///
    /// The prompt's forward pass is one blocking-free submission; the loop
    /// then runs `sample → feed → forward` per token with the wait placed by
    /// `opts.lookahead` (module docs). Returns the ids, whether EOS ended the
    /// run, and the counters the harness asserts.
    ///
    /// Refused rather than approximated: a proposer with `max_draft() > 0`
    /// (the verify body is stage 2), a lookahead the strategy cannot use, a
    /// lookahead above 1, and a sampler the strategy cannot serve without a
    /// host round trip it would not report.
    /// `on_tokens` is called with the ids produced since the last call, at each
    /// commit and once at the end — the STREAMING hook, and it is streaming
    /// *without re-serializing*. On the ring arms the ids live device-side and a
    /// commit is where the GPU has already retired up to `fed` (the host is
    /// `lookahead` steps ahead, `max_in_flight` deep), so draining
    /// `ReadbackStrategy::take_ids` there reads work that is already done — one
    /// batched readback per commit, not the per-token round trip §11.5a deleted.
    /// A caller that does not stream passes a no-op; the tail drain at the end is
    /// unchanged either way. Streaming appears in commit-sized chunks, which is
    /// the price of not blocking: a token cannot be shown before the GPU makes
    /// it, and the batch is what keeps that observation off the per-token path.
    pub fn decode<S, R>(
        &mut self,
        spec: &mut S,
        readback: &mut R,
        sampler: &mut LogitsProcessor,
        prompt: &[u32],
        opts: &DecodeOptions,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<()>,
    ) -> Result<DecodeOut>
    where
        S: Speculator,
        R: ReadbackStrategy,
    {
        self.decode_until(spec, readback, sampler, prompt, opts, &mut |batch| {
            on_tokens(batch)?;
            Ok(true)
        })
    }

    /// [`Self::decode`] whose streaming hook can end the run: a `false` from
    /// `on_tokens` stops after the batch it was handed. The serving seam's stop
    /// strings are decided on detokenized text the loop never sees, so the
    /// consumer decides. A run stopped this way leaves no overrun — every
    /// returned id was fed before its batch was streamed — and holds the last
    /// logits, so it resumes like a run that reached `max_tokens`.
    pub fn decode_until<S, R>(
        &mut self,
        spec: &mut S,
        readback: &mut R,
        sampler: &mut LogitsProcessor,
        prompt: &[u32],
        opts: &DecodeOptions,
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<bool>,
    ) -> Result<DecodeOut>
    where
        S: Speculator,
        R: ReadbackStrategy,
    {
        if spec.max_draft() > 0 {
            lloom_tensor::bail!(
                "Session::decode: a proposer with max_draft {} needs the speculative body \
                 (`Session::decode_speculative`), whose bounds are `M: Speculative + \
                 Probes<S::Probe>` — stronger than this method's `M: CausalLM`. Call \
                 `decode_speculative` directly; the K=1 identity path stays here.",
                spec.max_draft()
            );
        }
        if opts.lookahead > 1 {
            lloom_tensor::bail!(
                "Session::decode: lookahead {} — the ring holds one input slot and the \
                 overrun past an EOS is bounded at one forward only for lookahead <= 1",
                opts.lookahead
            );
        }
        if opts.lookahead > 0 && !readback.admits_lookahead() {
            lloom_tensor::bail!(
                "Session::decode: lookahead {} on a strategy that learns EOS at feed time; \
                 there is nothing to overlap and the extra forward would only pollute the cache",
                opts.lookahead
            );
        }
        if self.seq.overrun > 0 {
            lloom_tensor::bail!(
                "Session::decode: the cache is {} positions past the sequence's last token \
                 (an earlier run's overrun); `Session::reconcile` or `Session::reset` first",
                self.seq.overrun
            );
        }
        // An empty prompt is a resume: continue from the logits the previous
        // run left, which exist iff it ended with the cache one prediction
        // ahead of the token log (`Session::prepare`'s `Resume`).
        let resume = prompt.is_empty();
        if resume && self.seq.last_logits.is_none() {
            lloom_tensor::bail!("Session::decode: empty prompt and no held logits to resume from");
        }
        readback.check_sampler(sampler.sampling())?;
        spec.begin(SeqId(0), prompt)?;

        let device = readback.device().clone();
        let mut stats = DecodeStats {
            step_ns: Vec::with_capacity(opts.max_tokens),
            sample_ns: Vec::with_capacity(if opts.sample_split {
                opts.max_tokens
            } else {
                0
            }),
            ..DecodeStats::default()
        };

        // Prefill: the whole prompt in one pass. Its GPU tail lands in the
        // first decode steps; what is timed here is the submission.
        let pos_before = self.seq.pos;
        let prefill_start = std::time::Instant::now();
        let mut logits = match self.seq.last_logits.take() {
            Some(held) if resume => held,
            _ => {
                let input = Tensor::new(prompt, &device)?.unsqueeze(0)?;
                let logits = self
                    .model
                    .forward(&input, self.seq.pos, &mut self.seq.cache)?
                    .squeeze(0)?;
                self.seq.pos += prompt.len();
                self.seq.tokens.extend_from_slice(prompt);
                logits
            }
        };
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;

        let mut produced = 0usize;
        let mut since_commit = 0usize;
        let mut eos = false;
        // Every id the loop drains, streamed or not, accumulates here so the
        // returned `ids` is complete and the reconciliation below counts the
        // whole run. Draining for the stream advances the strategy's cursor, so
        // the final `take_ids` only sees the tail; the total is what matches
        // `produced`, not either part alone.
        let mut all_ids: Vec<u32> = Vec::with_capacity(opts.max_tokens);

        loop {
            if produced >= opts.max_tokens {
                break;
            }
            stats.steps += 1;
            let step_start = std::time::Instant::now();
            if opts.sample_split {
                device.synchronize()?;
            }
            let sample_start = std::time::Instant::now();

            let feed = readback.sample_and_feed(sampler, &logits)?;
            produced += 1;

            // Whether this step's commit is the loop's to make at all: the
            // blocking arm's own readback commits and waits every step, and a
            // loop-side commit beside it would be a second command buffer per
            // token the incumbent never cut.
            let loop_commits = readback.loop_commits();
            let mut committed_this_step = false;

            if opts.lookahead == 0 {
                // The publish must reach the GPU before the host waits on it.
                // Structural on this arm, not a budget choice, and counted.
                if loop_commits {
                    self.commit(&device, &mut stats)?;
                    since_commit = 0;
                    committed_this_step = true;
                }
                let is_eos = readback.eos_of_last()?;
                if opts.sample_split {
                    stats
                        .sample_ns
                        .push(sample_start.elapsed().as_nanos() as u64);
                }
                if is_eos {
                    produced -= 1;
                    eos = true;
                    break;
                }
                logits = self
                    .model
                    .forward(&feed, self.seq.pos, &mut self.seq.cache)?
                    .squeeze(0)?;
                self.seq.pos += 1;
            } else {
                if opts.sample_split {
                    stats
                        .sample_ns
                        .push(sample_start.elapsed().as_nanos() as u64);
                }
                // ENQUEUE, then learn. The GPU has this forward queued before
                // it retires the previous one.
                logits = self
                    .model
                    .forward(&feed, self.seq.pos, &mut self.seq.cache)?
                    .squeeze(0)?;
                self.seq.pos += 1;
                // Backpressure AFTER the enqueue: the host has just added a
                // command buffer, so cap how far it may run ahead of the GPU.
                // A no-op on arms that pace themselves; on the host-on-drain arm
                // it is what replaces the pacing the per-token EOS wait used to
                // provide, and it only waits when actually more than `cap` ahead.
                if opts.max_in_flight > 0 {
                    readback.apply_backpressure(opts.max_in_flight)?;
                }
                if readback.eos_of_last()? {
                    produced -= 1;
                    eos = true;
                    break;
                }
            }

            // The periodic machinery: the only place the lookahead arm
            // commits. A step that already committed before its wait is not
            // committed twice.
            let mut committed_now = committed_this_step;
            if loop_commits && !committed_this_step {
                since_commit += 1;
                if opts.budget.due(since_commit) {
                    self.commit(&device, &mut stats)?;
                    since_commit = 0;
                    committed_now = true;
                }
            }

            // Stream what is confirmed. This step's token has passed its EOS
            // check (an EOS `break`s above and is never streamed), so a drain
            // here carries only good ids.
            //
            // WHEN to drain is the whole of keeping this non-blocking. On the
            // ring arm (`loop_commits`) the ids live device-side and a readback
            // is a sync, so the drain is gated to commit steps: that is where the
            // GPU has retired up to `fed`, so the read is of finished work rather
            // than a wait, and it is one batched readback per commit instead of
            // the per-token round trip §11.5a deleted. On the blocking arm
            // (`!loop_commits`) the id is already host-side after this step's own
            // readback, so there is nothing to gate — it streams per token, which
            // is what that arm is.
            let stream_now = if loop_commits { committed_now } else { true };
            if stream_now {
                let batch = readback.take_ids()?;
                if !batch.is_empty() {
                    // The host-side EOS check (`EosRead::HostOnDrain`): this arm
                    // does not learn EOS per token, so a drained batch may
                    // contain the EOS token and the overrun the loop enqueued
                    // after it. Scan here, BEFORE streaming, so `on_tokens` still
                    // only ever sees good ids (the invariant the per-token arms
                    // hold by `break`ing before they stream). On other arms
                    // `eos_in_drained` is `None` and the batch streams whole.
                    if let Some(k) = readback.eos_in_drained(&batch) {
                        // `batch[..k]` are good; `batch[k]` is EOS (excluded, as
                        // the per-token retraction excludes it); `batch[k+1..]`
                        // are overrun (discarded), and so is everything the
                        // strategy still holds undrained — told to it, so a later
                        // call on the same strategy does not drain them as new
                        // tokens. Reconcile `produced` down to the good count so
                        // it matches `all_ids`.
                        let good = &batch[..k];
                        if !good.is_empty() {
                            // The run is over either way; the hook's answer
                            // changes nothing past an EOS.
                            let _ = on_tokens(good)?;
                            all_ids.extend_from_slice(good);
                        }
                        readback.note_eos_truncation(batch.len() - k - 1);
                        produced = all_ids.len();
                        eos = true;
                        stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                        break;
                    }
                    let go_on = on_tokens(&batch)?;
                    all_ids.extend_from_slice(&batch);
                    if !go_on {
                        stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                        break;
                    }
                }
            }
            stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
        }

        // After the clock: EVERY id not yet drained — `drain_all`, not the
        // incremental `take_ids`, because the poll arm may still have the last
        // forwards in flight and its incremental drain refuses to read past what
        // the GPU has published. `drain_all` synchronises up to the final slot,
        // so the tail is complete; streamed too, so a caller sees the last
        // tokens, then folded into the full `ids`. A host-on-drain arm that
        // truncated at EOS above has already discarded its undrained slots, so
        // its tail is empty and this costs it nothing.
        //
        // The tail is scanned for EOS too, for `EosRead::HostOnDrain`: if the run
        // ended on `max_tokens` (not an EOS `break` above) the EOS may sit in the
        // never-committed tail, and streaming it whole would emit tokens past it.
        // On the per-token arms `eos_in_drained` is `None`, so the tail streams
        // whole as before.
        let tail = readback.drain_all()?;
        if !tail.is_empty() {
            if let Some(k) = readback.eos_in_drained(&tail) {
                let good = &tail[..k];
                if !good.is_empty() {
                    let _ = on_tokens(good)?;
                    all_ids.extend_from_slice(good);
                }
                readback.note_eos_truncation(tail.len() - k - 1);
                produced = all_ids.len();
                eos = true;
            } else {
                let _ = on_tokens(&tail)?;
                all_ids.extend_from_slice(&tail);
            }
        }
        if all_ids.len() != produced {
            lloom_tensor::bail!(
                "Session::decode: the strategy returned {} ids for {} produced tokens — the \
                 count and the history disagree",
                all_ids.len(),
                produced
            );
        }

        // The sequence's bookkeeping, from the cache position rather than from
        // the arm's own accounting: every returned id was fed before its
        // successor's forward, so the cache holds the prompt, the returned ids
        // and — on an arm that learns EOS late — the positions past them.
        let fed = self.seq.pos - pos_before - prompt.len();
        if fed < produced {
            lloom_tensor::bail!(
                "Session::decode: {fed} forwards ran for {produced} returned tokens — a \
                 returned token whose forward never ran"
            );
        }
        let overrun = fed - produced;
        self.seq.tokens.extend_from_slice(&all_ids);
        self.seq.overrun = overrun;
        // The last forward's logits predict the token after the last returned
        // one iff nothing overran; they are what a continuation of exactly this
        // sequence resumes from. Past an EOS they predict nothing useful, but
        // they are still the right logits for that position.
        self.seq.last_logits = if overrun == 0 { Some(logits) } else { None };
        stats.overrun = overrun;
        stats.produced = produced;
        stats.readback = readback.counters();
        Ok(DecodeOut {
            ids: all_ids,
            eos,
            stats,
        })
    }

    fn commit(&self, device: &Device, stats: &mut DecodeStats) -> Result<()> {
        stats.commits += 1;
        if let Some(in_flight) = flush_and_prune(device)? {
            stats.max_in_flight = stats.max_in_flight.max(in_flight);
        }
        Ok(())
    }

    /// Decode from `prompt` with a real speculative proposer — stage 2 of
    /// `design/docs/decode-driver-design.md` ("the speculative body").
    ///
    /// This is the arm [`Self::decode`] refuses: a proposer with
    /// `max_draft() > 0` (DSpark). The bound is stronger than `decode`'s
    /// `M: CausalLM` — `M: Speculative` for the `forward_all`/`advance`/`resolve`
    /// rollback window and `M: Probes<S::Probe>` for the pairing contract
    /// (`speculator.rs`: an unsatisfiable target+drafter pair is a **compile
    /// error**, not a runtime bail). It is a separate method rather than a branch
    /// inside `decode` for exactly that reason: the K=1 identity path stays
    /// callable with only `CausalLM`, so widening the bound there would refuse a
    /// caller that has no drafter at all (`design/docs/decode-driver-design.md`,
    /// "two loop bodies, not one").
    ///
    /// # The round, and why it is output-identical to greedy
    ///
    /// Per round: draft up to `K` tokens, run the target's `forward_all` over
    /// `[anchor, draft…]`, accept the longest prefix whose argmax matches the
    /// draft **plus the target's own next token** (the bonus), roll the cache
    /// back to the accepted length, re-key the drafter from the accepted rows.
    /// The whole round is [`Self::speculative_step`], which is `draft →
    /// verify_and_probe → observe` — the verify accepts a draft token iff it
    /// equals `argmax(target logits)`, so the emitted stream is **byte-identical
    /// to non-speculative greedy decoding by construction** (`lfm2.rs`
    /// `verify_step`'s "greedy needs no epsilon"; DESIGN §10.2n). A drafter that
    /// proposes garbage costs acceptance rate and nothing else; it cannot change
    /// the output. That is the property the correctness gate asserts.
    ///
    /// # What this owns that `speculative_step` does not
    ///
    /// `speculative_step` runs one round across a batch and returns the accepted
    /// ids; it threads no decode state. The loop owns the decode cursor —
    /// [`Anchor`] (`last` = the last committed token, `pos` = the target position
    /// the window starts at), advancing `pos` by the accepted length and `last`
    /// to the accepted tail each round — plus EOS detection, the `max_tokens`
    /// cap (which may fall MID-block, so a round publishes 1..K+1 tokens), and
    /// the streaming hand-off.
    ///
    /// # Streaming, EOS and the cache
    ///
    /// The accepted tokens are already host-side — `verify_step` reads the argmax
    /// back to compare it (`design/docs/decode-driver-design.md`: the
    /// speculative body's one readback per round is inherent, not the ring's
    /// device-resident path). So this arm does NOT feed through the
    /// [`ReadbackStrategy`]'s device seam; it streams the accepted ids straight
    /// to `on_tokens` and tests EOS host-side against `eos_ids`. The strategy is
    /// still taken so the signature matches [`Self::decode`] and a future arm can
    /// publish the accepted block through the ring for a B>1 driver (#535's "the
    /// ring's real consumer is the speculation loop"); today it is used only for
    /// its device handle and its commit cadence.
    ///
    /// An EOS inside a block ends the run at that token: the tokens before it are
    /// streamed, the EOS and anything after it are not. The cache has advanced
    /// past the accepted prefix (`resolve` kept `accepted.len()` positions), so
    /// like `decode`'s `lookahead = 1` arm it can end one region past the EOS —
    /// which is safe for a single-turn run whose cache ends with it
    /// (`lfm2-chat`'s shape) and is why speculation needs the rewindable cache
    /// (`RotatingRing` + `InPlace`) `advance`/`resolve` require.
    ///
    /// # The drafter context is seeded from the prompt before round 0
    ///
    /// The prefill's probe (`Probes::take_probe`, every prompt position) is
    /// handed to `observe` before the first draft, with the prefill's argmax as
    /// the accepted tail — the `lfm2-dspark-verify` example's `--prefill-ctx
    /// all`, which is what llama.cpp's `speculative.cpp::process` does. An
    /// earlier form seeded lazily (round 0 drafted with an empty context and
    /// the verify taps keyed it) on the claim that this cost only round-0
    /// acceptance; measured, the drafter's context then never contained the
    /// prompt at all, and acceptance on prompts a draft copies from fell by up
    /// to half (`design/docs/speculation.md` §10.2w). Output-identical either
    /// way: the target verifies every token.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_speculative<S, R>(
        &mut self,
        spec: &mut S,
        readback: &mut R,
        _sampler: &mut LogitsProcessor,
        prompt: &[u32],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<()>,
    ) -> Result<DecodeOut>
    where
        S: Speculator,
        M: Probes<S::Probe>,
        R: ReadbackStrategy,
    {
        if spec.max_draft() == 0 {
            lloom_tensor::bail!(
                "Session::decode_speculative: the proposer has max_draft 0 (the K=1 identity); \
                 it belongs on `Session::decode`, which has the device-resident ring path this \
                 method deliberately does not."
            );
        }
        if eos_ids.is_empty() {
            lloom_tensor::bail!(
                "Session::decode_speculative: eos_ids is empty, so the loop would never terminate"
            );
        }
        if prompt.is_empty() {
            lloom_tensor::bail!("Session::decode_speculative: empty prompt");
        }
        if self.seq.overrun > 0 {
            lloom_tensor::bail!(
                "Session::decode_speculative: the cache is {} positions past the sequence's \
                 last token (an earlier run's overrun); `Session::reconcile` or \
                 `Session::reset` first",
                self.seq.overrun
            );
        }
        // A speculative run never leaves resumable logits (the verify reads
        // its argmax back and keeps no per-position logits), so a held one from
        // an earlier K=1 run is stale here.
        self.seq.last_logits = None;

        let device = readback.device().clone();
        let seq = SeqId(0);
        let mut stats = DecodeStats {
            step_ns: Vec::with_capacity(opts.max_tokens),
            ..DecodeStats::default()
        };

        // Prefill: the whole prompt in one pass. Its argmax is the first anchor —
        // the token whose logits produced the first draft — exactly as the
        // `lfm2-dspark-verify` example takes `logits.argmax` when it does not hold
        // the last prompt token back. `forward` taps (the tap hooks are woven into
        // both entry points, `lfm2.rs`), so the cache carries the prefill's taps
        // for `take_probe` below.
        let pos_before = self.seq.pos;
        let prefill_start = std::time::Instant::now();
        let input = Tensor::new(prompt, &device)?.unsqueeze(0)?;
        let logits = self
            .model
            .forward(&input, self.seq.pos, &mut self.seq.cache)?;
        self.seq.pos += prompt.len();
        self.seq.tokens.extend_from_slice(prompt);
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;

        let mut anchor_tok = logits
            .squeeze(0)?
            .argmax(lloom_tensor::D::Minus1)?
            .to_scalar::<u32>()?;

        // Seed the drafter's per-seq state: `begin` clears the context, then the
        // prefill's probe keys it from every prompt position and the prefill's
        // own argmax becomes the block anchor (the accepted tail) — the same
        // state the first round would see after a fully-accepted prompt.
        spec.begin(seq, prompt)?;
        let prefill_probe = self.model.take_probe(&mut self.seq.cache)?;
        spec.observe(seq, &prefill_probe, std::slice::from_ref(&anchor_tok))?;

        // The prefill's own argmax is the first generated token — it is what
        // non-speculative greedy would emit as token 0 (the prefill logits' argmax),
        // and the verify loop's anchor. Stream it and count it, unless it is EOS.
        let mut produced = 0usize;
        let mut eos = false;
        let mut all_ids: Vec<u32> = Vec::with_capacity(opts.max_tokens);

        if eos_ids.contains(&anchor_tok) {
            // The model wanted to stop immediately (an empty reply). Nothing is
            // produced and the run ends before any round.
            eos = true;
        } else if produced < opts.max_tokens {
            on_tokens(std::slice::from_ref(&anchor_tok))?;
            all_ids.push(anchor_tok);
            produced += 1;
        }

        // The verify window is `[anchor, draft…]` at this position; `pos` advances
        // by the accepted length each round (`lfm2-dspark-verify`'s `index_pos`).
        let mut since_commit = 0usize;

        while !eos && produced < opts.max_tokens {
            stats.steps += 1;
            let step_start = std::time::Instant::now();

            // One speculative round: draft → verify (accept longest argmax-match
            // prefix + bonus, roll back) → observe (re-key drafter from the
            // accepted rows). `p_min = 0.0` disables the confidence early-exit
            // (`speculative.cpp:796`; the driver owns the knob). `speculative_step`
            // returns the accepted tokens, INCLUDING the bonus, and has already
            // advanced the drafter's own anchor via `observe`.
            let anchor = Anchor {
                last: anchor_tok,
                pos: self.seq.pos,
            };
            // The census is one round's worth of allocation; take it once,
            // past the warm-up rounds, so the breakdown carries which kernels
            // each phase dispatches and not only how many.
            let probe = RoundProbe {
                sync: if opts.sample_split {
                    Some(&device)
                } else {
                    None
                },
                census: opts.sample_split && stats.steps == CENSUS_ROUND,
            };
            let (accepted, timing) = self.speculative_round(spec, &[seq], &[anchor], 0.0, probe)?;
            let accepted = accepted.into_iter().next().ok_or_else(|| {
                lloom_tensor::Error::msg("decode_speculative: verify returned no sequence")
            })?;
            stats.rounds.push(RoundStats {
                accepted: accepted.len(),
                timing,
            });
            if accepted.is_empty() {
                lloom_tensor::bail!(
                    "decode_speculative: a verify round accepted zero tokens; the target's own \
                     prediction at position 0 is always correct, so this is a broken verify"
                );
            }

            // The cache advanced by the accepted length; the next window starts
            // there. The loop's anchor follows the accepted tail — which is what
            // `observe` set the drafter's own anchor to, kept in step here so the
            // next `Anchor` matches.
            self.seq.pos += accepted.len();
            anchor_tok = *accepted.last().unwrap();

            // Publish the accepted tokens, stopping at the first EOS or at
            // `max_tokens`, whichever comes first — a round may straddle either.
            // An EOS is not streamed and retracts from the count, exactly as the
            // baseline loop drops an EOS token.
            let mut batch: Vec<u32> = Vec::with_capacity(accepted.len());
            for &id in &accepted {
                if eos_ids.contains(&id) {
                    eos = true;
                    break;
                }
                if produced + batch.len() >= opts.max_tokens {
                    break;
                }
                batch.push(id);
            }
            if !batch.is_empty() {
                on_tokens(&batch)?;
                all_ids.extend_from_slice(&batch);
                produced += batch.len();
            }

            // Periodic commit/prune. The verify pass already read the argmax back
            // (a host sync), so the round is naturally drained; the commit bounds
            // the in-flight queue on the ring-owning strategies and is a no-op
            // elsewhere. Gated on the budget like the baseline loop.
            if readback.loop_commits() {
                since_commit += 1;
                if opts.budget.due(since_commit) {
                    self.commit(&device, &mut stats)?;
                    since_commit = 0;
                }
            }

            stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
        }

        // The sequence's bookkeeping. The cache holds each round's window —
        // the anchor and the proposals the target agreed with — so it runs one
        // token BEHIND the stream while the run is going (the final anchor is
        // streamed and not yet fed) and AHEAD of it when an EOS fell mid-block
        // (the accepted tokens after it were fed before the host saw it). Both
        // are read off the position: the fed ids are the stream's prefix of
        // that length, and anything fed past the stream is overrun.
        let fed = self.seq.pos - pos_before - prompt.len();
        let in_cache = fed.min(produced);
        self.seq.tokens.extend_from_slice(&all_ids[..in_cache]);
        let overrun = fed - in_cache;
        self.seq.overrun = overrun;
        stats.overrun = overrun;

        stats.produced = produced;
        stats.readback = readback.counters();
        Ok(DecodeOut {
            ids: all_ids,
            eos,
            stats,
        })
    }

    /// Compile the Metal pipelines the first real forward would otherwise
    /// compile *inside its own timed window*, then reset so the session is
    /// untouched.
    ///
    /// # What this moves, and why it is not a compute change
    ///
    /// `DESIGN.md` §13.1i measures pipeline creation at **~2 ms cold, ~0.01 ms
    /// warm** and library compilation at seconds, both lazily cached in the
    /// device's `Kernels`. The decode step (`m == 1`) routes to GEMV **before**
    /// `select_tile_config` (`mlx_gemm.rs`), so the prefill GEMM-tile pipelines
    /// — and the F32 generic-attention and batched-`Conv1d` pipelines the
    /// `seq_len > 1` path takes — are **prefill-exclusive**: none is touched by
    /// any decode step, so all are cold on the first real prefill. That is the
    /// fixed cost a single-shot TTFT carries and `llama.cpp` does not (it
    /// compiles every kernel up front). A warmup pays it here, folded into model
    /// load, where no clock the user reads is running.
    ///
    /// # Why a real forward rather than an enumerated pipeline list
    ///
    /// The GEMM pipeline key varies with `M` through both the tile name and
    /// `align_m = m % bm`, so a hand-written list would be incomplete by
    /// construction and a near-miss on the constants keys a *different* cache
    /// entry — the real forward would still compile cold. Running the production
    /// `forward` cannot miss a pipeline that same `forward` will later use for
    /// the same shape bucket. The buckets are coarse (§13.5c: `m < 16` → 32×32,
    /// `m >= 16` → the 64-tile), so one prefill across the `m >= 16` boundary
    /// plus one decode step covers the set; [`WARMUP_PREFILL_LEN`] is chosen to
    /// cross it.
    ///
    /// Token id `0` is always a valid vocabulary index. The forward's output is
    /// discarded — only its compilation side effect is wanted — and
    /// `M::Cache: CacheReset` returns the cache to its pre-prefill state
    /// afterwards, so the next real prompt starts exactly as it would have.
    /// `synchronize` forces compilation to *complete* here rather than leak into
    /// the first real forward.
    ///
    /// Returns the device's `(libraries, pipelines)` cache sizes after warming,
    /// so a caller can report engagement — a nonzero pipeline count is what
    /// distinguishes a warmed cache from a probe that compiled nothing
    /// (§14.5d).
    pub fn warmup(&mut self, device: &Device) -> Result<(usize, usize)>
    where
        M::Cache: CacheReset,
    {
        // Prefill across the m >= 16 tile boundary (§13.5c), then one decode
        // step — the same two shape buckets every real run takes.
        let prefill_tokens = vec![0u32; WARMUP_PREFILL_LEN];
        let prefill = Tensor::new(prefill_tokens.as_slice(), device)?.unsqueeze(0)?;
        let _ = self.model.forward(&prefill, 0, &mut self.seq.cache)?;

        let decode = Tensor::new(&[0u32], device)?.unsqueeze(0)?;
        let _ = self
            .model
            .forward(&decode, WARMUP_PREFILL_LEN, &mut self.seq.cache)?;

        // Force the driver to finish compiling before the clock the caller is
        // about to stop; a cold pipeline creation left in flight would surface
        // on the first real forward instead.
        device.synchronize()?;

        // Leave nothing behind: the warmup wrote KV, conv state and the position
        // counter, and the next call is the user's real prefill from position 0.
        self.reset();

        Ok(cache_sizes(device))
    }
}

/// The speculative round whose per-phase dispatch census is kept under
/// `DecodeOptions::sample_split`: past the first rounds' pipeline warm-up,
/// early enough that a short run still reaches it.
const CENSUS_ROUND: u64 = 6;

/// Prefill length for [`Session::warmup`], chosen to cross the
/// `select_tile_config` `m >= 16` boundary (`DESIGN.md` §13.5c) so the warmup
/// compiles the prefill GEMM tile the real path will use, not a smaller one.
pub const WARMUP_PREFILL_LEN: usize = 32;

/// The device's `(libraries, pipelines)` cache sizes, or `(0, 0)` off Metal.
fn cache_sizes(device: &Device) -> (usize, usize) {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return m.kernels().cache_sizes();
    }
    let _ = device;
    (0, 0)
}
