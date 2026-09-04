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
use crate::nn::model::CausalLM;
use crate::nn::speculator::{Draft, NoProbe, SeqId, Session, Speculator, TargetProbe};
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
        Ok(active
            .iter()
            .map(|_| Draft {
                tokens: Vec::new(),
                confidence: Vec::new(),
            })
            .collect())
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
    /// **Diagnostic, and it perturbs the thing it measures**: drain the GPU
    /// before each sample so the sample term can be timed apart from the
    /// forward pass (§11.2a). Off for every published number.
    pub sample_split: bool,
}

impl DecodeOptions {
    /// Greedy defaults: `max_tokens`, per-token commit, no lookahead, no split.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            budget: CommitBudget::per_token(),
            lookahead: 0,
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
    /// Forward passes enqueued past an EOS (0 or 1).
    pub overrun: usize,
    /// The strategy's own counters.
    pub readback: ReadbackCounters,
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
    pub fn decode<S, R>(
        &mut self,
        spec: &mut S,
        readback: &mut R,
        sampler: &mut LogitsProcessor,
        prompt: &[u32],
        opts: &DecodeOptions,
    ) -> Result<DecodeOut>
    where
        S: Speculator,
        R: ReadbackStrategy,
    {
        if spec.max_draft() > 0 {
            lloom_tensor::bail!(
                "Session::decode: a proposer with max_draft {} needs the speculative body, \
                 which is stage 2 of the decode-driver design and is not built",
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
        if prompt.is_empty() {
            lloom_tensor::bail!("Session::decode: empty prompt");
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
        let prefill_start = std::time::Instant::now();
        let input = Tensor::new(prompt, &device)?.unsqueeze(0)?;
        let mut logits = self
            .model
            .forward(&input, self.pos, &mut self.cache)?
            .squeeze(0)?;
        self.pos += prompt.len();
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;

        let mut produced = 0usize;
        let mut since_commit = 0usize;
        let mut eos = false;

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
                    .forward(&feed, self.pos, &mut self.cache)?
                    .squeeze(0)?;
                self.pos += 1;
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
                    .forward(&feed, self.pos, &mut self.cache)?
                    .squeeze(0)?;
                self.pos += 1;
                if readback.eos_of_last()? {
                    produced -= 1;
                    eos = true;
                    stats.overrun = 1;
                    break;
                }
            }

            // The periodic machinery: the only place the lookahead arm
            // commits. A step that already committed before its wait is not
            // committed twice.
            if loop_commits && !committed_this_step {
                since_commit += 1;
                if opts.budget.due(since_commit) {
                    self.commit(&device, &mut stats)?;
                    since_commit = 0;
                }
            }
            stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
        }

        // After the clock: the ring's one readback recovers the ids.
        let ids = readback.take_ids()?;
        if ids.len() != produced {
            lloom_tensor::bail!(
                "Session::decode: the strategy returned {} ids for {} produced tokens — the \
                 count and the history disagree",
                ids.len(),
                produced
            );
        }
        stats.produced = produced;
        stats.readback = readback.counters();
        Ok(DecodeOut { ids, eos, stats })
    }

    fn commit(&self, device: &Device, stats: &mut DecodeStats) -> Result<()> {
        stats.commits += 1;
        if let Some(in_flight) = flush_and_prune(device)? {
            stats.max_in_flight = stats.max_in_flight.max(in_flight);
        }
        Ok(())
    }
}
