//! The decode loop — `Session::decode`, the baseline (K=1) body of
//! `design/docs/decode-driver-design.md`, stage 1.
//!
//! The loop names two seams and neither concrete arm: a [`Speculator`] for
//! *which* proposer (stage 1 admits only [`IdentityProposer`], the K=1 identity;
//! the verify body is stage 2 and is refused rather than approximated) and a
//! [`ReadbackStrategy`] for how a step feeds the next forward and learns EOS.
//! The harness selects both and never writes the body.
//!
//! # Two cadences, both kept
//!
//! The ring arms run one of two cadences, chosen by
//! [`DecodeOptions::adaptive_commit`]. Neither replaces the other: the flow is
//! the default where the engine argues it, and the fixed cadence is the arm
//! for a backend, a model or a consumer the flow does not serve. Both are
//! pinned to one token stream (`decode_driver_tests`, `docs/dispatch.md`
//! §11.5z-xv).
//!
//! - **The flow** (`adaptive_commit`): the host encodes step after step and
//!   closes its command buffer only when the GPU has started every buffer
//!   before it, or at the `run_ahead` bound; nothing on this thread waits on a
//!   token. A `RingReader` thread is the drain and the stream, per token, and
//!   raises [`DecodeOptions::stop`] on EOS; the EOS gate makes the forwards
//!   queued past it write nothing. Needs a CPU-mapped ring (Metal),
//!   `lookahead = 1` and an EOS read other than the flag.
//! - **The fixed cadence** (`adaptive_commit` off — `lloom --readback ring-fixed`
//!   and `ring-la0`, the drain and batched-EOS reads without the adaptive
//!   commit, and every ring off Metal, where there is no reader): each forward
//!   is committed at its end, and the pacer waits on the marker
//!   [`DecodeOptions::pacer_depth`] forwards back; EOS is learned on this
//!   thread, per step (`eos_of_last`) or from the drained ids on the deferred
//!   reads; and the loop drains the ring into `on_tokens` at each commit —
//!   every step for the first budget period (the TTFT ramp, so the first
//!   token is not held for a drain), then once a period. Its two lookaheads
//!   are below.
//!
//! # The invariant, and what it costs to hold
//!
//! The design's hot path *does not commit, does not block, does not read
//! back*: sample on the device, feed the ring, enqueue the forward, bump
//! counters. The one thing the loop must eventually learn per token is whether
//! it was EOS, and where that wait sits decides everything on the fixed
//! cadence:
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

use crate::generation::{
    CommitBudget, HotThread, HotThreadReport, LogitsProcessor, ReadbackCounters, ReadbackStrategy,
    Sampling,
};
use crate::nn::batched::BatchDecoder;
use crate::nn::model::{BatchedCausalLM, CacheReset, CausalLM};
use crate::nn::speculator::{
    readbacks, Anchor, Draft, NoProbe, Probes, RoundProbe, RoundTiming, SeqId, Session, Speculator,
    TargetProbe,
};
use lloom_tensor::{Device, Result, Tensor};

mod prefill_work;
#[cfg(test)]
mod speculative_prefill_tests;
pub(crate) use prefill_work::PrefillWork;

/// The K=1 identity proposer: drafts nothing, so the loop takes the baseline
/// body. Stage 1's only admitted family.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityProposer;

impl Speculator for IdentityProposer {
    type Probe = NoProbe;
    type Family = crate::nn::speculator::Identity;

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

/// The bodies a decode runs through — one loop, arms by what a round feeds
/// (`docs/decode-driver-design.md`, the one loop). A round is a verify
/// window: the anchor and the drafts behind it. Non-speculative decoding is
/// the window of the anchor alone, and [`Body::Step`] is that window's own
/// arm: the drawn token fed on the device, never read back on a ring arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Body {
    /// The anchor alone a step (`Session::decode_until`); `ring` when the
    /// token rides the device-resident ring rather than the blocking
    /// readback.
    Step { ring: bool },
    /// The anchor and its drafts a round (`Session::decode_speculative`;
    /// with a proposer that drafts nothing, the zero-draft window).
    Window,
    /// Several sequences' steps in one forward (`Session::decode_batched`,
    /// the batch scheduler).
    BatchedStep,
    /// Several sequences' windows (`Session::decode_speculative_pipelined`,
    /// the pool rows' `SpecBatchDecoder`).
    BatchedWindow,
}

/// What a request asks of the body that serves it.
#[derive(Debug, Clone, Copy)]
pub struct Asks<'a> {
    pub sampling: &'a Sampling,
    /// Logits processors (bias, penalties, min-p).
    pub processors: bool,
    /// A thinking budget.
    pub budget: bool,
    /// The logits digest.
    pub digest: bool,
    /// Per-token logprobs (a host read of every step's row): `Some(k)` is
    /// each drawn id's log-probability and the `k` most likely ids beside
    /// it, `k` at most [`MAX_TOP_LOGPROBS`].
    pub logprobs: Option<usize>,
}

/// The most candidates a per-token logprobs ask may name beside the drawn
/// id: the bound of the API's shape. The chat route's one-forward decision
/// scoring serves up to 64, so the bound is that rather than OpenAI's 20.
pub const MAX_TOP_LOGPROBS: usize = 64;

impl<'a> Asks<'a> {
    /// What `sampler` asks, with or without a thinking budget.
    pub fn of(sampler: &'a LogitsProcessor, budget: bool) -> Self {
        Asks {
            sampling: sampler.sampling(),
            processors: sampler.processors().is_some(),
            budget,
            digest: sampler.digests(),
            logprobs: None,
        }
    }
}

/// The ONE table of what a body cannot serve, refused by name before any
/// state moves. Every driver asks it; nothing is served under a label it
/// does not run.
///
/// - The ring step draws on the device: the host-sort arms (top-k, top-p)
///   sort on the host, and the digest reads every step's logits there.
/// - The window serves every sampler arm and feature: each row is drawn
///   keyed to its position against the history its position has
///   (`LogitsProcessor::window_draw_device`). It reads back the ids a round
///   keeps, never a row, so the per-token logprobs are the K=1 body's.
/// - The batched bodies draw per row without processors, feed no forced
///   token and keep no logprobs; the batched windows verify by the argmax.
pub fn refuse(body: Body, asks: &Asks<'_>) -> Result<()> {
    let host_sort = matches!(
        asks.sampling,
        Sampling::TopK { .. } | Sampling::TopP { .. } | Sampling::TopKThenTopP { .. }
    );
    if let Some(k) = asks.logprobs.filter(|&k| k > MAX_TOP_LOGPROBS) {
        lloom_tensor::bail!(
            "per-token logprobs name at most {MAX_TOP_LOGPROBS} candidates beside the drawn id, \
             not {k}"
        );
    }
    match body {
        Body::Step { ring: true } if host_sort => lloom_tensor::bail!(
            "the ring needs a device-resident sampler (argmax, multinomial or gumbel-max); \
             {:?} takes a host sort and cannot ride it (DESIGN.md §11.2c)",
            asks.sampling
        ),
        Body::Step { ring: true } if asks.digest => lloom_tensor::bail!(
            "the ring draws on the device and the logits digest reads every step's logits on \
             the host: the digest runs on the blocking arm"
        ),
        Body::Step { ring: true } if asks.logprobs.is_some() => lloom_tensor::bail!(
            "the ring draws on the device and the per-token logprobs read every step's row on \
             the host: they run on the blocking arm"
        ),
        Body::Window if asks.logprobs.is_some() => lloom_tensor::bail!(
            "the window reads back the ids a round keeps, never a row of logits: the per-token \
             logprobs are the non-speculative body's"
        ),
        Body::Step { .. } | Body::Window => Ok(()),
        Body::BatchedStep | Body::BatchedWindow if asks.processors => lloom_tensor::bail!(
            "a batched body does not run logits processors (logit_bias, repetition/presence/\
             frequency penalties, min_p): its rows draw through the batched driver's per-row \
             sampler or one device argmax over the batch (serve them at a batch of 1)"
        ),
        Body::BatchedStep | Body::BatchedWindow if asks.budget => lloom_tensor::bail!(
            "a batched body does not carry a thinking budget: its rows feed no forced token \
             (serve it at a batch of 1)"
        ),
        Body::BatchedStep | Body::BatchedWindow if asks.logprobs.is_some() => lloom_tensor::bail!(
            "a batched body does not keep per-token logprobs: its rows draw on the device and \
             read the ids back (serve it at a batch of 1)"
        ),
        Body::BatchedWindow if *asks.sampling != Sampling::ArgMax => lloom_tensor::bail!(
            "a batched speculative body verifies by the argmax: {:?} is refused rather than \
             served as argmax (serve it at a batch of 1, where the window draws each row)",
            asks.sampling
        ),
        Body::BatchedStep | Body::BatchedWindow => Ok(()),
    }
}

/// The knobs of one decode run.
// Not `Copy`: `prefill_schedule` is an owned `Vec`. `DecodeOptions` is passed
// by reference everywhere (`opts: &DecodeOptions`), so this costs the hot path
// nothing; the two struct-update sites (`..opts.clone()`) clone explicitly.
#[derive(Debug, Clone)]
pub struct DecodeOptions {
    /// Tokens to produce at most; EOS may stop the run earlier. **`0` means
    /// UNBOUNDED** — run until EOS with no ceiling (the token-ring history wraps,
    /// so an unbounded run streams at any length). A positive value is a hard
    /// cap. Use [`Self::reached`]/[`Self::unbounded`] rather than comparing
    /// directly, so the `0`-is-unbounded convention is honoured everywhere.
    pub max_tokens: usize,
    /// How often the loop commits and prunes the in-flight queue.
    pub budget: CommitBudget,
    /// Forward passes the host may enqueue before it has learned the previous
    /// step's EOS: `0` or `1`. See the module docs for what `1` costs.
    pub lookahead: usize,
    /// **Diagnostic, and it perturbs the thing it measures**: drain the GPU
    /// before each sample so the sample term can be timed apart from the
    /// forward pass (§11.2a). Off for every published number. Under
    /// [`Session::decode_speculative`] it drains the GPU at the end of each
    /// phase of a round instead — draft, verify, observe — so
    /// [`DecodeStats::rounds`] charges each phase its own GPU tail, and takes
    /// the profiler's per-phase counters when it is on.
    pub sample_split: bool,
    /// How many forwards the host may run ahead of the GPU (the pacer's bound).
    /// **`0` means AUTO** — computed from the host/GPU balance measured at
    /// warmup ([`compute_in_flight_depth`]), floored at [`IN_FLIGHT_FLOOR`]. A
    /// positive value overrides it (the tunable knob, e.g. an env A/B). Set from
    /// [`Session::warmup`]'s return, or left 0 to fall back to the floor when no
    /// warmup ran (CPU, or warmup skipped).
    pub in_flight_depth: usize,
    /// The spark: how many prompt tokens the FIRST prefill forward takes.
    /// **`0` means the whole prompt in one pass** — the incumbent one-shot. A
    /// positive value splits the prompt into TWO forwards, the spark and then
    /// the whole remainder (not `prompt_len / ubatch` chunks: the weight sweep
    /// is per forward, so the bulk is fed once), the first through the SAME
    /// in-flight pacer the decode loop uses so the first token streams off the
    /// spark while the bulk runs. It is the TTFT-vs-throughput dial: a wider
    /// spark keeps the matmul units fuller and approaches one-shot as it
    /// reaches the prompt length. Honoured by [`Session::decode`] and by both
    /// speculative drivers ([`Session::decode_speculative`],
    /// [`Session::decode_speculative_pipelined`], which seed the drafter from
    /// each unit's probe as it lands); [`Session::decode_batched`] prefills
    /// one-shot. Correctness across the unit boundary needs the conv's
    /// left-context carry (`Config::conv_prefill_ctx`), which the streamed
    /// path turns on; a single unit is identical either way (a cold cache has
    /// no left context to carry). [`DecodeStats::prefill_units`] reports the
    /// forwards a run actually took.
    pub prefill_ubatch: usize,
    /// An explicit prefill schedule: the exact widths of the forwards the
    /// prompt is fed through, in order. **`None` (the default) keeps the
    /// incumbent `prefill_ubatch` ramp** (one-shot, or spark-then-remainder) —
    /// byte-identical to before this field existed. `Some(units)` overrides it:
    /// the prompt is fed as forwards of `units[0], units[1], …` tokens, each
    /// clamped to the remaining prompt, and any prompt left after the schedule
    /// is exhausted is covered by `seq_len == 1` forwards (the tail).
    ///
    /// The point is to keep every prefill forward on an *aligned* GEMM tile
    /// variant. `Model::forward` selects a tile by the row count `m` (= the
    /// forward's `seq_len`) and compiles an M-boundary-guarded variant unless
    /// `m % bm == 0` (`mlx_gemm::select_tile_config`, `align_m`); a one-shot
    /// forward at an arbitrary `m = prompt.len()` lands on whatever variant that
    /// `m` picks, frequently the slower guarded one on the largest forward of
    /// the request. A schedule of aligned buckets ({16, 32} on the 16-row tile,
    /// 64 on the 64-row tile) keeps each forward on the no-guard variant, and a
    /// `1` unit is the `seq_len == 1` decode path. The CLI's `--spark` builds
    /// this; see that flag.
    ///
    /// **Read once, in the prefill prologue** — the steady decode loop never
    /// consults it, so a schedule adds no per-token cost on the hot path.
    pub prefill_schedule: Option<Vec<usize>>,
    /// The **bulk tile** width the prefill remainder is chopped into, after the
    /// [`Self::prefill_schedule`] ramp is exhausted. `0` (the default) means the
    /// remainder is a `seq_len == 1` tail (one forward per leftover token — the
    /// pathological many-forwards case on a long prompt); a positive aligned
    /// width (16 / 32 / 64) chops the remainder into that-wide forwards, each on
    /// the no-guard GEMM variant, collapsing the forward count to
    /// `remainder / bulk`. Only meaningful with `prefill_schedule = Some`.
    pub prefill_bulk: usize,
    /// The pacer's in-flight bound **during prefill** — how many prefill
    /// forwards the host may run ahead of the GPU. `0` (the default) uses the
    /// same depth as decode ([`Self::pacer_depth`]). A positive value runs
    /// prefill deeper: a chunked / many-small-forward prefill is compute-bound
    /// and serializes at the decode depth of 2 (each forward waits on the one
    /// two back), so a deeper bound lets the forwards overlap. Restored to the
    /// decode depth before the step loop, so it has no effect on steady decode.
    pub prefill_pacer_depth: usize,
    /// Stream tokens through a caller-owned [`RingReader`] instead of the hot
    /// loop's commit-cadence drain. `false` (default) is the incumbent: on a
    /// ring arm the loop drains the device ring at each commit and hands the
    /// batch to `on_tokens` — which batches visibility into commit-sized bursts.
    /// `true` (ring arms only) means the loop does **not** drain or call
    /// `on_tokens` per commit at all: the ids already live on the device ring,
    /// the caller reads them per-token off a
    /// [`RingReader`](crate::generation::readback::RingReader) (no sync, no
    /// wait), and the loop materialises the host id vec **once** from the ring
    /// at end-of-run (for `DecodeOut::ids` and the sequence's token log). This
    /// removes the visibility serialization (tokens stream per-token, not
    /// per-commit-burst) AND the redundant per-commit host mirror — the ring is
    /// the record.
    ///
    /// No effect on the blocking arm (its ids are host-side per step; it streams
    /// per token already) or on [`Self::decode_until`]'s stop return (a caller
    /// that needs stop-strings under a reader thread signals the loop itself).
    pub stream_via_reader: bool,
    /// How the host fill's producer waits between tokens for this run; `None`
    /// takes `LLOOM_HOST_FEED_POLL` ([`FeedWait`]).
    pub host_feed_wait: Option<FeedWait>,
    /// Whether the fixed cadence's EOS wait parks on the ring's publish event
    /// for this run; `None` takes `LLOOM_RING_EOS_PARK`.
    pub eos_park: Option<bool>,
    /// Under `adaptive_commit`: the loop watches the ring itself for a stop
    /// id (#993, `sequence-reuse-design.md` §2.3e). Before each step it reads
    /// what the GPU has published since the last (one load of the stamp and
    /// the new slots, no wait) and stops at a stop id among them, as it
    /// stops on [`Self::stop`]. The forwards past the EOS are then bounded by
    /// the pacer, not by the reader: `S − E ≤ adaptive_lead + 1`, however late
    /// the reader raises the flag. `None` takes `decode.ring.eos_watch`.
    pub eos_watch: Option<bool>,
    /// The flow's adaptive commit; off is the fixed cadence (module docs, "Two
    /// cadences, both kept"). The loop closes its command buffer only when the
    /// GPU has started every buffer committed before it, on no cadence of its
    /// own — a buffer holds whatever the host encoded while the GPU ran the
    /// previous one. Requires `stream_via_reader` (the drain is the reader's,
    /// per token, off this thread), `lookahead == 1`, and a ring whose EOS
    /// read is not the flag: `poll`, `drain` or `gpubatch*`. The loop does not
    /// call `eos_of_last` on this cadence — a `poll` would wait on a stamp
    /// inside an uncommitted buffer — and learns EOS from the reader's
    /// [`Self::stop`] flag, or from the drained ids when the run ends first.
    /// The cut period is held at `usize::MAX` for the run, so the only cuts
    /// are the loop's.
    pub adaptive_commit: bool,
    /// Under `adaptive_commit`: the most steps the host may encode into one
    /// buffer before closing it regardless (0 = 8). Not a cadence — a buffer
    /// closes earlier whenever the GPU has caught up — but the bound on how
    /// much work can sit queued past an EOS until the kernels gate on it.
    pub run_ahead: usize,
    /// Under `adaptive_commit`: a flag the reader thread raises when it has
    /// seen EOS (`RingReader::ended`); the loop stops encoding at the next
    /// step and reconciles the overrun. `None` runs to `max_tokens`. Also
    /// read by [`Session::decode_speculative`] between rounds as a caller's
    /// cancellation: a raised flag ends the run after the round in flight,
    /// with the sequence's bookkeeping intact.
    pub stop: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// Under `adaptive_commit`: the EOS gate (lloom #735 step 4,
    /// `Device::eos_gate_begin`). The decode kernels run in a variant that
    /// returns at once while a flag a tiny gate kernel raises on the sampled
    /// token being a stop token is set, so the forwards queued past an EOS
    /// write nothing: no cache write past the EOS, and
    /// the overrun is reconciled by moving counters back
    /// (`CacheRewind::rewind_unwritten`) rather than rewinding contents —
    /// which the crown's cache (`Shuffle`, in-place KV) cannot do at all.
    /// Needs a strategy with a device copy of the stop set
    /// (`ReadbackStrategy::eos_ids_tensor`) and no ICB executor.
    pub eos_gate: bool,
    /// Under `adaptive_commit`: publish each step with the fused
    /// argmax·eos·stamp kernel (lloom #735 step 5, `Device::argmax_publish`)
    /// where the strategy can (`ReadbackStrategy::set_fused_publish`). On by
    /// default; off is the six-launch chain, kept as the A/B arm.
    pub fused_publish: bool,
    /// The TILE: the most prompt tokens one prefill forward may take
    /// (llama.cpp's `n_ubatch`). **`0` means unbounded** — every forward is
    /// as wide as the ramp or the schedule makes it, the incumbent. A
    /// positive value splits any wider forward into sequential forwards of
    /// at most this many tokens (the last one narrower), so the prompt's
    /// activation working set — §9.5n's `a·C² + b·C`, the class admission
    /// sums — is bounded by the tile rather than by the prompt, and the
    /// buffers one tile's forward released are what the next tile's forward
    /// reuses. Composes with the spark and the schedule: it bounds whatever
    /// forwards they name. Read by every prefill this crate runs
    /// ([`Session::decode`], [`Session::prefill`], [`Session::decode_batched`])
    /// and by neither speculative driver, whose seed is `prefill_streamed`'s
    /// two forwards. A tiled prefill onto a warm cache needs the conv's
    /// left-context carry (`Config::conv_prefill_ctx`), as any multi-forward
    /// prefill does.
    pub prefill_tile: usize,
    /// The CHUNK under [`Session::decode_batched`]: the most prompt tokens one
    /// pending row is prefilled by per batched step, interleaved with the
    /// live rows' decode (vLLM's chunked prefill, its token budget). **`0`
    /// means every prompt is prefilled whole at its arrival** — the
    /// incumbent, no interleave. A positive value feeds at most that many
    /// prompt tokens per step over the pending rows in order, **as extra rows
    /// of the step's own batched forward** ([`BatchedCausalLM::forward_ragged`])
    /// — one weight sweep for the live rows' decode and the prompt pieces
    /// together, no second forward — each row joining the batch at the step
    /// after its last piece; a row's prompt KV is built incrementally and its
    /// stream is what a whole prefill gives it. The tile bounds a piece too.
    /// The time-sliced driver's chunk is the caller's — it feeds
    /// [`Session::prefill`] per visit — and the single-sequence loop has no
    /// other sequence to interleave with, so neither reads this.
    pub prefill_chunk: usize,
    /// How the widths of the prompt PIECES a batched step feeds are chosen,
    /// so the set of ragged pass shapes — and with it the plan cache's key
    /// set (`DESIGN.md` §9.2aa, I4) — is bounded. An axis with two arms, to
    /// be measured: [`PassWidths::Lru`] feeds each piece as wide as the chunk
    /// and the tile allow (the incumbent schedule; every distinct layout is a
    /// key, the cache bounds how many it keeps), [`PassWidths::Ladder`]
    /// rounds a piece DOWN to a rung of `{top, top/2, …, 1}` (`top` = the
    /// tile bounded by the chunk, else the chunk) so a prompt's pieces are
    /// drawn from `O(log top)` widths. Read by [`Session::decode_batched`]
    /// only; the single-sequence loop and the time-sliced driver feed whole
    /// chunks (the harness applies the ladder to its visits itself).
    pub pass_widths: PassWidths,
}

/// The arm of [`DecodeOptions::pass_widths`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PassWidths {
    /// Pieces as wide as the chunk and the tile allow; the plan cache keeps
    /// a bounded number of ragged layouts (`lru`).
    #[default]
    Lru,
    /// Pieces rounded down to a rung of `{top, top/2, …, 1}` (`ladder`).
    Ladder,
}

impl std::str::FromStr for PassWidths {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        Self::parse(s)
    }
}

impl std::fmt::Display for PassWidths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

lloom_tensor::serde_via_str!(PassWidths);

impl PassWidths {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "lru" => Ok(Self::Lru),
            "ladder" => Ok(Self::Ladder),
            other => Err(format!(
                "unknown pass widths `{other}`; want `lru` or `ladder`"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Lru => "lru",
            Self::Ladder => "ladder",
        }
    }
}

/// The widest rung of `{top, top/2, …, 1}` not above `take` — the
/// [`PassWidths::Ladder`] width of a piece of at most `take` tokens. `top`
/// or `take` of 0 leaves `take` as it is.
pub fn ladder_width(take: usize, top: usize) -> usize {
    if take == 0 || top == 0 {
        return take;
    }
    let mut rung = top;
    while rung > take {
        rung = (rung / 2).max(1);
    }
    rung
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
    /// Host time from the run's first prefill to this round's start, so rounds
    /// of different sequences can be laid on one timeline.
    pub start_ns: u64,
}

/// A resolved output ceiling; zero leaves EOS as the only token boundary.
#[derive(Clone, Copy)]
struct TokenLimit(usize);

impl TokenLimit {
    fn reached(self, produced: usize) -> bool {
        self.0 != 0 && produced >= self.0
    }

    fn remaining(self, produced: usize) -> usize {
        if self.0 == 0 {
            usize::MAX
        } else {
            self.0.saturating_sub(produced)
        }
    }
}

impl DecodeOptions {
    /// Whether the run has no token ceiling (`max_tokens == 0`) — it stops only
    /// on EOS. See [`Self::max_tokens`].
    pub fn unbounded(&self) -> bool {
        self.max_tokens == 0
    }

    /// Whether `produced` tokens have reached the cap. Always `false` when
    /// [`Self::unbounded`] (the `0`-is-unbounded convention), so the loops run
    /// until EOS. Use this instead of `produced >= max_tokens`.
    pub fn reached(&self, produced: usize) -> bool {
        TokenLimit(self.max_tokens).reached(produced)
    }

    /// The remaining token budget from `produced`, or `usize::MAX` when
    /// unbounded (for `min`-style clamps that want "no limit").
    pub fn remaining(&self, produced: usize) -> usize {
        TokenLimit(self.max_tokens).remaining(produced)
    }

    /// The pacer depth: the explicit override if set, else the floor. (The
    /// AUTO path sets `in_flight_depth` from warmup; this reads it back with the
    /// `0 = auto -> floor` convention applied.)
    /// Under `adaptive_commit`: how many steps the host may be ahead of the
    /// oldest incomplete buffer — two run-ahead windows queued plus the open
    /// one; 0 otherwise. What the ring's staged-through tiles are sized for.
    pub fn adaptive_lead(&self) -> usize {
        if self.adaptive_commit {
            3 * if self.run_ahead == 0 {
                8
            } else {
                self.run_ahead
            }
        } else {
            0
        }
    }

    pub fn pacer_depth(&self) -> usize {
        if self.in_flight_depth == 0 {
            IN_FLIGHT_FLOOR
        } else {
            self.in_flight_depth
        }
    }

    /// Greedy defaults: `max_tokens`, per-token commit, no lookahead, no split,
    /// auto in-flight depth.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            budget: CommitBudget::per_token(),
            lookahead: 0,
            sample_split: false,
            in_flight_depth: 0,
            prefill_ubatch: 0,
            prefill_schedule: None,
            prefill_bulk: 0,
            prefill_pacer_depth: 0,
            stream_via_reader: false,
            host_feed_wait: None,
            eos_park: None,
            eos_watch: None,
            adaptive_commit: false,
            run_ahead: 0,
            stop: None,
            eos_gate: false,
            fused_publish: true,
            prefill_tile: 0,
            prefill_chunk: 0,
            pass_widths: PassWidths::Lru,
        }
    }

    /// The spark width, or the whole prompt in one pass when `0` (the incumbent
    /// one-shot). See the `prefill_ubatch` field.
    pub fn prefill_ubatch(&self) -> usize {
        self.prefill_ubatch
    }

    /// The widest input one prefill forward can stage under these options:
    /// what a ring's input tile must hold for every forward to stage
    /// zero-copy. The tile when set (it bounds every forward), else the spark
    /// (one-shot's whole-prompt forward takes the allocating fallback either
    /// way), else the widest of the schedule's units and the bulk.
    pub fn max_prefill_width(&self) -> usize {
        if self.prefill_tile > 0 {
            return self.prefill_tile;
        }
        let widest_unit = self
            .prefill_schedule
            .as_ref()
            .and_then(|u| u.iter().copied().max())
            .unwrap_or(0);
        self.prefill_ubatch.max(widest_unit).max(self.prefill_bulk)
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
    /// Forward passes the prompt took: 1 one-shot, 2 streamed (the spark, then
    /// the remainder — `prefill_ubatch` names the spark's width, not a chunk
    /// count), 0 when the run resumed from held logits. Read it off the run
    /// rather than deriving it from the option: a unit count that is only ever
    /// computed from `prompt_len / ubatch` is not a measurement.
    pub prefill_units: usize,
    /// Host time from the loop's exit to the run's return: the final drain
    /// (`ReadbackStrategy::drain_all`, which waits for the GPU) and the
    /// bookkeeping after it. Inside a caller's wall clock and outside the loop.
    pub tail_ns: u64,
    /// Compute time to the first token: prefill + the first decode step, stamped
    /// ONCE (at step 0) so it survives release without any per-step clock read —
    /// the fair, llama-bench-comparable TTFT (their synchronous `llama_decode` of
    /// the prompt → first logits), measured here WITHOUT a forced sync (the async
    /// pipeline is the design). Distinct from the CLI's `first_batch_at` (when the
    /// host DRAINS the first token, which adds the commit cadence) and
    /// `first_visible_at` (+ the consumer/print). 0 if the run produced nothing.
    pub ttft_compute_ns: u64,
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
    /// Step markers taken: one per forward the loop committed at its end
    /// ([`IN_FLIGHT_STEPS`]).
    pub step_markers: u64,
    /// Markers the loop had to wait on before encoding the next step -- the
    /// host had run [`IN_FLIGHT_STEPS`] forwards ahead of the GPU. Zero on an
    /// arm the per-token EOS wait already paces.
    pub step_waits: u64,
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
    /// At the end of a speculative run, `(the drafter's own context length,
    /// the target's position)` — equal when the drafter followed the target
    /// through the request's prefix reuse (issue #894). `None` on the
    /// baseline body and for a family with no state of its own.
    pub drafter_ctx: Option<(usize, usize)>,
    /// The first [`FIRST_STEPS_TRACED`] steps' host time, by phase, on the
    /// `lookahead = 1` arm; empty on `lookahead = 0` and the speculative bodies.
    pub first_steps: Vec<StepPhases>,
    /// Every step's host time by phase, summed, on the `lookahead = 1` arm in a
    /// debug or `profile` build (zero in release, where the clock reads are
    /// compiled out): the steady-state split of the hot thread between encoding
    /// and waiting for the GPU. `phase_steps` is the number of steps summed.
    pub phase_totals: StepPhases,
    pub phase_steps: u64,
    /// The decode thread's scheduling class for this run: what it was, what
    /// the loop claimed (`generation::HotThread`), and whether that engaged.
    pub hot_thread: HotThreadReport,
    /// Under `adaptive_commit`: buffers the loop closed because the GPU had
    /// started the previous one, buffers closed at the `run_ahead` bound, the
    /// most steps one buffer held, and whether the reader's stop flag ended
    /// the run.
    pub adaptive_commits: u64,
    pub run_ahead_closes: u64,
    pub max_steps_per_buffer: usize,
    pub stopped_by_flag: bool,
    /// The loop's own watch saw a stop id published and stopped the run
    /// ([`DecodeOptions::eos_watch`]).
    pub stopped_by_eos_watch: bool,
    /// Under `eos_gate`: what the gate did (`Device::eos_gate_end`).
    pub eos_gate: Option<lloom_tensor::EosGateReport>,
    /// The overrun's forwards launched nothing (the gate fired), so the
    /// session reconciles it without a rewind horizon.
    pub overrun_unwritten: bool,
    /// The prefill's dispatch census, `(kernel, count)`, under
    /// `CANDLE_METAL_PROFILE=1`; empty otherwise and off Metal.
    pub prefill_census: Vec<(String, u64)>,
    /// Step 0's dispatch census, on the same terms.
    pub step0_census: Vec<(String, u64)>,
    /// Under [`Session::decode_batched`], the number of live rows in each
    /// step this sequence took part in — one entry per `step_ns` entry. Empty
    /// on the single-sequence bodies.
    pub batch_rows: Vec<usize>,
    /// The model's host fill (`CausalLM::host_feed`), when the run took it:
    /// its engagement counters.
    pub host_feed: Option<HostFeedReport>,
    /// Whether the run's steps were provisional (`CausalLM::provisional`):
    /// the loop settled each after its readback, before its token was
    /// handed out.
    pub provisional: bool,
    /// Draws taken back and made again from a redone step's logits (a
    /// provisional step that computed on something it lacked).
    pub redraws: u64,
}

/// What a run's host fill did (`CausalLM::host_feed`).
#[derive(Debug, Clone, Default)]
pub struct HostFeedReport {
    /// Slots the producer filled and signalled: one per token the loop fed.
    pub fills: u64,
    /// The decode forwards that waited for a filled slot.
    pub forwards: u64,
    /// Host time from seeing a token's stamp to signalling its slot, per
    /// fill, in a debug or `profile` build (empty in release, where the
    /// producer reads no clock).
    pub fill_ns: Vec<u64>,
    /// The producer's scheduling class for the run (`LLOOM_HOST_FEED_QOS`),
    /// and the errno of a refused claim (0 if none).
    pub qos: crate::generation::QosClass,
    pub qos_errno: i32,
    /// How the producer waited between tokens (`LLOOM_HOST_FEED_POLL`).
    pub poll: FeedWait,
    /// The producer's waits: spinning, sleeping, parked on the publish event.
    pub spun: u64,
    pub slept: u64,
    pub parked: u64,
    /// Wakes the forwards posted to the producer as they demanded a value.
    pub demand_wakes: u64,
    /// Under `LLOOM_HOST_FEED_TRACE`: per-fill host times and the
    /// producer's scheduling, for the park a forward's wait can see.
    pub trace: Option<HostFeedTrace>,
}

/// How the host fill's producer waits for the next token
/// (`LLOOM_HOST_FEED_POLL`).
///
/// A wait that spins holds a core, and while a core spins through the
/// streamed tier's host-planned pass the GPU runs that pass at low clocks
/// (`expert-streaming.md` §20.14a). [`FeedWait::Event`] parks, woken by the
/// ring's publish event (signalled right behind each stamp, delivered by a
/// listener block) or by a forward demanding a value (`FeedLink::encode_read`),
/// and spins (`WFE`) only for an imminent token: while a forward already
/// waits for a value the producer has not filled, and for at most
/// [`FEED_SPIN_BOUND`] retirements (§20.14b). In a host-planned pass the host
/// encodes a forward's wait only after it has waited for the token itself, so
/// the producer never spins there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedWait {
    /// One poll arm throughout.
    Fixed(crate::generation::poll::PollMechanism),
    /// [`crate::generation::poll::PollMechanism::DEFAULT`] while a wait is
    /// pending on an unfilled value, a sleep of `idle_us` otherwise.
    OnDemand { idle_us: u64 },
    /// Park on the publish event; the bounded `WFE` above while a wait is
    /// pending; a sleep of `idle_us` where the run armed no event.
    Event { idle_us: u64 },
}

/// The producer's `WFE` retirements (~1 µs each, `poll.rs`) for one imminent
/// token before it parks instead: past the longest demand-to-token window
/// measured on the pipelined decode (6.4 ms, `expert-streaming.md` §20.14b),
/// a fail-safe for a stalled stream rather than a latency budget.
pub const FEED_SPIN_BOUND: u64 = 16_384;

/// The longest one park lasts: a fail-safe, since the token's publish, a
/// forward's demand and the run's end each wake the producer.
pub const FEED_PARK_MS: u64 = 50;

/// What the producer does for one wait.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FeedStep {
    /// Wait on the stamp word by this arm.
    Poll(crate::generation::poll::PollMechanism),
    /// Park on the ring's publish event.
    Park,
}

impl Default for FeedWait {
    /// `on-demand:50`: measured faster than [`FeedWait::Event`] wherever the
    /// streamed pool misses (`expert-streaming.md` §20.14b).
    fn default() -> Self {
        FeedWait::OnDemand { idle_us: 50 }
    }
}

impl FeedWait {
    /// The next wait, given the highest value a forward of the run waits for,
    /// the values filled so far, the `WFE` waits spent on the token now
    /// awaited, and whether the run armed a publish event.
    pub fn step(self, demanded: u64, filled: u64, spins: u64, parks: bool) -> FeedStep {
        use crate::generation::poll::PollMechanism;
        let idle = |us| FeedStep::Poll(PollMechanism::Sleep(std::time::Duration::from_micros(us)));
        match self {
            FeedWait::Fixed(m) => FeedStep::Poll(m),
            FeedWait::OnDemand { .. } if demanded > filled => {
                FeedStep::Poll(PollMechanism::DEFAULT)
            }
            FeedWait::OnDemand { idle_us } => idle(idle_us),
            FeedWait::Event { .. } if demanded > filled && spins < FEED_SPIN_BOUND => {
                FeedStep::Poll(PollMechanism::DEFAULT)
            }
            FeedWait::Event { .. } if parks => FeedStep::Park,
            FeedWait::Event { idle_us } => idle(idle_us),
        }
    }
}

impl std::fmt::Display for FeedWait {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FeedWait::Fixed(m) => write!(f, "{m}"),
            FeedWait::OnDemand { idle_us } => write!(f, "on-demand:{idle_us}"),
            FeedWait::Event { idle_us } => write!(f, "event:{idle_us}"),
        }
    }
}

impl std::str::FromStr for FeedWait {
    type Err = String;

    /// `event` / `event:<µs>` (the sleep where no event is armed),
    /// `on-demand` / `on-demand:<µs>`, or any
    /// [`crate::generation::poll::PollMechanism`] spelling for one arm
    /// throughout.
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        let s = s.trim();
        let us = |rest: &str| {
            rest.parse::<u64>()
                .map_err(|e| format!("host feed wait {s:?}: bad microseconds: {e}"))
        };
        match s {
            "event" => return Ok(FeedWait::Event { idle_us: 50 }),
            "on-demand" => return Ok(FeedWait::OnDemand { idle_us: 50 }),
            _ => {}
        }
        if let Some(rest) = s.strip_prefix("event:") {
            return us(rest).map(|idle_us| FeedWait::Event { idle_us });
        }
        if let Some(rest) = s.strip_prefix("on-demand:") {
            return us(rest).map(|idle_us| FeedWait::OnDemand { idle_us });
        }
        s.parse::<crate::generation::poll::PollMechanism>()
            .map(FeedWait::Fixed)
    }
}

/// One traced run of the host fill's producer (`LLOOM_HOST_FEED_TRACE`), on
/// the clock of [`crate::nn::ngram_memory::trace_now_ns`]. Index `k - 1` is
/// run value `k`: the forward that waits for it, the token it is filled
/// from.
#[derive(Debug, Clone, Default)]
pub struct HostFeedTrace {
    /// When the waiting forward's wait was first encoded on the host.
    pub encoded_ns: Vec<Option<u64>>,
    /// When the producer read the token's stamp.
    pub observed_ns: Vec<u64>,
    /// When the producer raised the event for it.
    pub signalled_ns: Vec<u64>,
    /// Passes of the producer's loop, and the longest wall time between two
    /// of them: a gap far above the poll's own wait is time the thread was
    /// not running.
    pub iterations: u64,
    pub max_gap_ns: u64,
    /// Gaps over 1 ms: how many, and their total.
    pub gaps_over_1ms: u64,
    pub gap_over_1ms_ns: u64,
    /// The producer thread's CPU time and wall time over the run.
    pub cpu_ns: u64,
    pub wall_ns: u64,
    /// The prefetcher's reader threads' and the PLE read pool's classes.
    pub prefetch_qos: crate::generation::QosClass,
    pub read_pool_qos: crate::generation::QosClass,
}

impl HostFeedTrace {
    /// One line for a harness log: the classes, the loop, and the three
    /// per-fill intervals (µs) the park is read from.
    pub fn line(&self, qos: crate::generation::QosClass, poll: &str) -> String {
        fn dist(mut v: Vec<i64>) -> String {
            if v.is_empty() {
                return "n=0".into();
            }
            v.sort_unstable();
            let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize] as f64 / 1e3;
            format!(
                "n={} p50={:.1} p90={:.1} p99={:.1} max={:.1}",
                v.len(),
                q(0.5),
                q(0.9),
                q(0.99),
                q(1.0)
            )
        }
        let n = self.observed_ns.len().min(self.signalled_ns.len());
        let mut late = Vec::new();
        let mut park = Vec::new();
        let mut fill = Vec::new();
        let mut parked = 0usize;
        let mut park_sum = 0i64;
        for i in 0..n {
            let (o, s) = (self.observed_ns[i] as i64, self.signalled_ns[i] as i64);
            fill.push(s - o);
            if let Some(Some(e)) = self.encoded_ns.get(i) {
                let e = *e as i64;
                late.push(o - e);
                park.push(s - e);
                if s > e {
                    parked += 1;
                    park_sum += s - e;
                }
            }
        }
        format!(
            "producer_qos={} poll={poll} prefetch_qos={} read_pool_qos={} cpu_ms={:.1} wall_ms={:.1} \
             cpu_wall={:.3} iterations={} max_gap_us={:.1} gaps_over_1ms={} gap_over_1ms_ms={:.1} \
             signalled_after_encode={parked} signal_minus_encode_sum_ms={:.1} \
             observed_minus_encode_us=[{}] signal_minus_encode_us=[{}] signal_minus_observed_us=[{}]",
            qos.as_str(),
            self.prefetch_qos.as_str(),
            self.read_pool_qos.as_str(),
            self.cpu_ns as f64 / 1e6,
            self.wall_ns as f64 / 1e6,
            self.cpu_ns as f64 / self.wall_ns.max(1) as f64,
            self.iterations,
            self.max_gap_ns as f64 / 1e3,
            self.gaps_over_1ms,
            self.gap_over_1ms_ns as f64 / 1e6,
            park_sum as f64 / 1e6,
            dist(late),
            dist(park),
            dist(fill),
        )
    }
}

/// The per-kernel dispatch counts so far, under `CANDLE_METAL_PROFILE=1`.
/// `None` when the census is off, which is every timed run.
fn census_snapshot() -> Option<std::collections::HashMap<String, u64>> {
    #[cfg(feature = "metal")]
    {
        use lloom_tensor::kernels::metal::profile;
        if profile::enabled() {
            return Some(profile::snapshot().by_label.into_iter().collect());
        }
    }
    None
}

/// The dispatches between two census snapshots, sorted by count.
fn census_delta(
    before: &Option<std::collections::HashMap<String, u64>>,
    after: Option<std::collections::HashMap<String, u64>>,
) -> Vec<(String, u64)> {
    let (Some(before), Some(after)) = (before, after) else {
        return Vec::new();
    };
    let mut out: Vec<(String, u64)> = after
        .into_iter()
        .filter_map(|(k, n)| {
            let d = n - before.get(&k).copied().unwrap_or(0);
            (d > 0).then_some((k, d))
        })
        .collect();
    out.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}

/// Where one decode step's host time went, for the first few steps of a run.
///
/// TTFT is `prefill_ns + step_ns[0]`, and the first step is not a steady-state
/// step: under `lookahead = 1` its EOS wait is where the prefill's GPU tail
/// lands, and its forward is the first at the decode shape. The four brackets
/// say which term carries a first-token cost; the sum is `step_ns[i]` minus
/// the loop's own bookkeeping.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepPhases {
    /// `sample_and_feed`: the argmax and the ring publish, encode only.
    pub sample_ns: u64,
    /// The forward pass, encode only.
    pub forward_ns: u64,
    /// `apply_backpressure`: zero unless the host outran the GPU.
    pub backpressure_ns: u64,
    /// `eos_of_last`: the wait for this step's token to be published.
    pub eos_ns: u64,
}

impl StepPhases {
    /// Accumulate `other` into this, saturating.
    pub fn add(&mut self, other: &StepPhases) {
        self.sample_ns = self.sample_ns.saturating_add(other.sample_ns);
        self.forward_ns = self.forward_ns.saturating_add(other.forward_ns);
        self.backpressure_ns = self.backpressure_ns.saturating_add(other.backpressure_ns);
        self.eos_ns = self.eos_ns.saturating_add(other.eos_ns);
    }
}

/// How many leading steps [`DecodeStats::first_steps`] brackets.
pub const FIRST_STEPS_TRACED: u64 = 4;

/// What one [`Session::prefill`] took: the counters a scheduler reports per
/// visit. Zero on every field for an empty prompt.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillOut {
    /// Forward passes the prompt took ([`DecodeStats::prefill_units`]).
    pub units: usize,
    /// Submission time of the forwards — encode, not execution.
    pub prefill_ns: u64,
    /// Step markers the pacer took, one per forward.
    pub step_markers: u64,
    /// Markers the pacer had to wait on before encoding a forward.
    pub step_waits: u64,
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

/// One sequence's cursor through [`Session::decode_speculative_pipelined`].
struct Lane<P> {
    limit: TokenLimit,
    anchor_tok: u32,
    produced: usize,
    eos: bool,
    all_ids: Vec<u32>,
    pos_before: usize,
    prompt_len: usize,
    stats: DecodeStats,
    /// The verify in flight, if one is.
    pending: Option<P>,
    /// The in-flight round's draft and enqueue brackets, completed when its
    /// verify is read back.
    in_flight: RoundTiming,
    round_start_ns: u64,
    done: bool,
}

/// Commit without waiting and prune the in-flight queue, on the backends that
/// have one. `None` where there is no queue to prune — a CPU run reports no
/// count, which is honest rather than zero (§14.1).
/// Close the installed dispatch executor's step, on the backend that has one.
///
/// The executor cannot see a token boundary from inside the encoder, so the
/// loop that owns the step declares it -- after the prefill, and after each
/// step's or round's last dispatch, so the window an ICB executor records is
/// one whole step and the same shape every time (`icb::IcbExecutor::end_step`).
/// A no-op with no executor installed, and off Metal.
pub(crate) fn end_executor_step(device: &Device) -> Result<()> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        m.end_executor_step()?;
    }
    let _ = device;
    Ok(())
}

/// Bracket one decode step's forward pass for the activation arena
/// (`DESIGN.md` §9.2): the ordinal that ties an allocation to a slot restarts
/// here, and an armed recording (`MetalDevice::arm_arena_recording`) observes
/// the step and installs the arena when it has seen enough. No-ops with no
/// arena installed or armed, and off Metal. The sampling before the forward and
/// the drain after it stay outside, as the harnesses that record by hand keep
/// them.
/// The shape of a forward pass, for the pass-keyed plan cache
/// (`DESIGN.md` §9.2aa). Off-metal, and with the cache off, a bracket is the
/// decode-step bracket it always was.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassShape {
    Decode {
        b: usize,
    },
    /// `fresh`: the first pass on an empty cache (`pos == 0`).
    Prefill {
        w: usize,
        fresh: bool,
    },
    Verify {
        k: usize,
    },
    /// A batched step of `live` decode rows and `pieces` ragged prompt
    /// pieces whose `(width, fresh)`, in order, hash to `layout` (see
    /// [`ragged_layout`]).
    Ragged {
        live: u16,
        pieces: u16,
        layout: u64,
    },
}

/// FNV-1a over the pieces' `(width, fresh)` in order — the `layout` of a
/// [`PassShape::Ragged`]. `fresh` is the piece being its row's first pass
/// (the row's session state is allocated in it, and the conv has no left
/// context to carry — a different dispatch sequence, as `Prefill { fresh }`
/// is a different key). Backend-neutral so every driver keys the same way.
pub fn ragged_layout(pieces: impl IntoIterator<Item = (usize, bool)>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for (w, fresh) in pieces {
        for b in ((w as u64) << 1 | fresh as u64).to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

#[cfg(feature = "metal")]
fn pass_key(shape: PassShape) -> lloom_tensor::metal_backend::PassKey {
    use lloom_tensor::metal_backend::{PassKey, PassKind};
    let kind = match shape {
        PassShape::Decode { b } => PassKind::Decode {
            b: b.min(u16::MAX as usize) as u16,
        },
        PassShape::Prefill { w, fresh } => PassKind::Prefill {
            w: w.min(u32::MAX as usize) as u32,
            fresh,
        },
        PassShape::Ragged {
            live,
            pieces,
            layout,
        } => PassKind::Ragged {
            live,
            pieces,
            layout,
        },
        PassShape::Verify { k } => PassKind::Verify {
            k: k.min(u16::MAX as usize) as u16,
        },
    };
    PassKey::new(kind)
}

/// This pass shape, as the backend-neutral key the `BackendDevice` hook
/// takes (`DESIGN.md` §9.2aa step 8).
fn shape_key(shape: PassShape) -> lloom_tensor::backend::PassShapeKey {
    use lloom_tensor::backend::PassShapeKey as K;
    match shape {
        PassShape::Decode { b } => K::Decode {
            b: b.min(u16::MAX as usize) as u16,
        },
        PassShape::Prefill { w, fresh } => K::Prefill {
            w: w.min(u32::MAX as usize) as u32,
            fresh,
        },
        PassShape::Verify { k } => K::Verify {
            k: k.min(u16::MAX as usize) as u16,
        },
        PassShape::Ragged {
            live,
            pieces,
            layout,
        } => K::Ragged {
            live,
            pieces,
            layout,
        },
    }
}

/// Open a planned pass of `shape` on `device`, through the backend seam
/// (`BackendDevice::begin_pass`). A device that declines allocates exactly
/// as it did before, so every driver brackets unconditionally and no call
/// site carries a `#[cfg]` (`DESIGN.md` §9.2aa step 8).
///
/// On Metal with the cache off this still runs the legacy decode-step
/// bracket the arena needs, which is why the Metal arm is not simply the
/// hook: `MetalDevice::begin_pass` falls back to `begin_decode_step`.
pub(crate) fn begin_pass(device: &Device, shape: PassShape) -> Result<()> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        // The legacy arena path lives under the same bracket, so Metal goes
        // through its own entry point rather than the hook's early decline.
        m.begin_pass(pass_key(shape))?;
        return Ok(());
    }
    device.begin_pass(shape_key(shape))?;
    Ok(())
}

/// Whether `device` may hand a later pass the memory an earlier pass's
/// activation lives in (`MetalDevice::activations_may_be_reused`).
fn activations_may_be_reused(device: &Device) -> Result<bool> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return Ok(m.execution_device()?.activations_may_be_reused());
    }
    let _ = device;
    Ok(false)
}

pub(crate) fn end_pass(device: &Device) -> Result<()> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        m.finish_pass()?;
        return Ok(());
    }
    device.finish_pass()
}

fn begin_decode_step(device: &Device) {
    // The decode step is the `Decode { b: 1 }` pass; with the cache off this
    // is the legacy bracket, which cannot fail.
    let _ = begin_pass(device, PassShape::Decode { b: 1 });
}

fn end_decode_step(device: &Device) -> Result<()> {
    end_pass(device)
}

fn flush_and_prune(device: &Device) -> Result<Option<usize>> {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return Ok(Some(m.flush_and_prune()?));
    }
    let _ = device;
    Ok(None)
}

/// The FLOOR on how far the host may run ahead of the GPU -- the minimum the
/// lookahead-1 dependency already implies (host waits for step `s`'s publish
/// after encoding `s + 1`, so `s - 1` retires before `s + 2`). The ACTUAL depth
/// is computed from the host/GPU balance at warmup ([`compute_in_flight_depth`])
/// and is `>= this`; a constant `2` was only ever right for a config where the
/// GPU step dwarfs the host encode (this model on this machine). See
/// [`compute_in_flight_depth`] for why a static value caps other models/machines.
pub const IN_FLIGHT_FLOOR: usize = 2;

/// How long the loop waits for the host fill's producer to take a draw back
/// (`FeedLink::retract`): a fail-safe, the producer's take-back is a few
/// microseconds once it wakes. The same bound as the fetch worker's.
const FEED_RETRACT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Back-compat alias for [`IN_FLIGHT_FLOOR`]: callers that sized a buffer to the
/// old constant get the floor, which is a safe lower bound (the real depth is
/// `>=` it). Prefer `Session::measured_in_flight_depth` / `DecodeOptions::
/// in_flight_depth` for the actual computed depth.
pub const IN_FLIGHT_STEPS: usize = IN_FLIGHT_FLOOR;

/// The in-flight depth that keeps the GPU fed without the host running
/// pointlessly far ahead -- a *quantitative* bound, not an arbitrary constant.
///
/// # The two times must be DISJOINT
///
/// `host_step_ns` is the host's encode+submit; `gpu_step_ns` is the GPU's own
/// time for that step, **excluding** the host's. A caller that passes
/// `host + gpu` as `gpu_step_ns` makes this function unable to return more
/// than [`IN_FLIGHT_FLOOR`] -- `2·h / (h + t) <= 2` for every `t >= 0` -- and
/// the calibration silently degrades to the constant it replaced. That was a
/// real defect in [`Session::warmup`], caught by reading the arithmetic
/// rather than by any test, which is why the disjointness is stated here and
/// pinned by a unit test rather than left to the call site to remember.
///
/// The host must stay enqueued far enough that the GPU never starves waiting for
/// the next step: `depth = ceil(host_step / gpu_step) x safety`, floored at
/// [`IN_FLIGHT_FLOOR`] and capped by `mem_cap` (a memory-derived ceiling, so a
/// pathological ratio cannot pile up unbounded in-flight buffers). Both times
/// are MEASURED at warmup (a real decode step already runs there), so this
/// self-calibrates to the actual (model, machine, thermal) balance:
///
/// * **big model / slow GPU** (2.6B on M1 Max: gpu ~ 15 ms, host ~ 1.5 ms ->
///   ratio 0.1) -> `depth = 2` (the floor) -- identical to the old constant, so
///   no regression where the constant happened to be right.
/// * **small model** (350M: gpu ~ 2 ms, host ~ 1.5 ms -> ratio 0.75) -> `depth
///   ~ 3-4` -- the host runs deeper to feed the fast-draining GPU, which a
///   static `2` capped badly on the very same machine.
/// * **faster GPU** (M4 Max: higher bandwidth -> smaller gpu_step) -> deeper.
///
/// The old rationale ("a deeper queue buys the GPU nothing, the steps run
/// serially") holds ONLY when `gpu_step >> host_step`; when the GPU can outrun
/// the host, depth is exactly what stops it starving.
pub fn compute_in_flight_depth(host_step_ns: u64, gpu_step_ns: u64, mem_cap: usize) -> usize {
    if gpu_step_ns == 0 {
        return IN_FLIGHT_FLOOR;
    }
    // ceil(host/gpu) with a x2 safety, so a host step that occasionally spikes
    // toward the GPU step still has a queued forward to hide behind.
    let ratio_x2 = (2 * host_step_ns).div_ceil(gpu_step_ns) as usize;
    ratio_x2
        .max(IN_FLIGHT_FLOOR)
        .min(mem_cap.max(IN_FLIGHT_FLOOR))
}

/// The depth `Session::warmup` installs, from the two times it measured.
///
/// A named function rather than an inline call so the CALLER's operands are
/// testable: the defect this file carried was not in
/// [`compute_in_flight_depth`] — which is correct — but in what `warmup`
/// passed it (`host + tail` for the GPU time, which pins the result at
/// [`IN_FLIGHT_FLOOR`] for every balance). A correct function with wrong
/// operands is invisible to a test of the function, so the operand contract
/// gets its own seam here.
///
/// `gpu_ns` MUST be the GPU tail alone. `warm_depth_rejects_host_inclusive_times`
/// pins that by construction.
pub fn warm_depth(host_ns: u64, gpu_ns: u64, mem_cap: usize) -> usize {
    compute_in_flight_depth(host_ns, gpu_ns, mem_cap)
}

/// The memory ceiling on in-flight depth: the most forwards' worth of retained
/// command buffers + their buffers the host may hold ahead of the GPU, so a
/// pathological host/GPU ratio (a tiny model on a huge GPU) cannot pile up
/// unbounded in-flight state (the 63-deep drain-arm tail this pacer replaced).
///
/// A generous constant for now — 64 steps is far above any measured healthy
/// depth (~2-4) yet bounds a runaway; a config where the computed ratio wants
/// more than this is one whose memory should be checked first. A future refinement
/// derives it from `recommendedMaxWorkingSetSize` / per-step in-flight bytes
/// (the §9.5 admission machinery), so it scales with the machine — left as the
/// seam, not built, since the ratio floors at 2 on the shipping config anyway.
pub fn in_flight_mem_cap(_device: &Device) -> usize {
    64
}

/// The host's bound on how far it runs ahead of the GPU, as a wait on the
/// command buffer that carried a step rather than on anything the GPU writes.
///
/// A marker is a committed command buffer; waiting on it is Metal's own
/// completion, which retires or errors and cannot wait for a value that is
/// never published. That is the whole difference from a cap spun on the ring's
/// generation counter: a counter a stale replay stops advancing is a livelock,
/// Metal can report a command-buffer error, but an indefinitely scheduled
/// buffer can still block its native wait indefinitely. When the GPU is
/// within the bound the marker is already complete and the wait is a status
/// read.
pub(crate) struct StepPacer {
    #[cfg(feature = "metal")]
    pending: PacerQueue<PacerMarker>,
    depth: usize,
    taken: u64,
    waited: u64,
}

/// One circular queue owns both the completion marker and its step count.
/// Retiring either cadence therefore cannot leave unmatched size entries.
/// Capacity is reserved before prefill/decode, not on the token path.
#[cfg(any(feature = "metal", test))]
struct PacerQueue<M> {
    entries: std::collections::VecDeque<(M, usize)>,
    queued_steps: usize,
}

#[cfg(any(feature = "metal", test))]
impl<M> PacerQueue<M> {
    fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::VecDeque::with_capacity(capacity),
            queued_steps: 0,
        }
    }

    fn push(&mut self, marker: M, steps: usize) {
        debug_assert!(steps > 0);
        debug_assert!(
            self.entries.len() < self.entries.capacity(),
            "pacer setup must reserve its bound"
        );
        self.entries.push_back((marker, steps));
        self.queued_steps += steps;
    }

    /// Keep the entry if waiting fails: an error must not masquerade as retirement.
    fn retire(&mut self, wait: impl FnOnce(&M) -> Result<bool>) -> Result<bool> {
        let (marker, _) = self.entries.front().expect("non-empty pacer");
        let waited = wait(marker)?;
        let (_, steps) = self.entries.pop_front().expect("same entry");
        self.queued_steps -= steps;
        Ok(waited)
    }
}

impl StepPacer {
    pub(crate) fn new(depth: usize) -> Self {
        let depth = depth.max(IN_FLIGHT_FLOOR);
        Self {
            #[cfg(feature = "metal")]
            pending: PacerQueue::new(depth),
            depth,
            taken: 0,
            waited: 0,
        }
    }

    /// Reserve the largest bound this run can use before encoding starts.
    pub(crate) fn for_decode(opts: &DecodeOptions) -> Self {
        let capacity =
            opts.pacer_depth()
                .max(opts.prefill_pacer_depth)
                .max(if opts.adaptive_commit {
                    2 * if opts.run_ahead == 0 {
                        8
                    } else {
                        opts.run_ahead
                    }
                } else {
                    0
                });
        let mut pacer = Self::new(capacity);
        pacer.set_depth(opts.pacer_depth());
        pacer
    }

    fn before_forward_bounded(&mut self, device: &Device, max_queued: usize) -> Result<()> {
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            while self.pending.queued_steps >= max_queued && !self.pending.entries.is_empty() {
                self.waited += self.pending.retire(|oldest| {
                    let waited = !m.marker_complete(oldest);
                    m.wait_marker(oldest)?;
                    Ok(waited)
                })? as u64;
            }
        }
        let _ = (device, max_queued);
        Ok(())
    }

    /// Setup-only depth changes reuse the capacity reserved by `for_decode`.
    pub(crate) fn set_depth(&mut self, depth: usize) {
        self.depth = depth.max(IN_FLIGHT_FLOOR);
        #[cfg(test)]
        speculative_prefill_tests::note(speculative_prefill_tests::Event::Depth(self.depth));
        #[cfg(feature = "metal")]
        debug_assert!(self.depth <= self.pending.entries.capacity());
    }

    pub(crate) fn counters(&self) -> (u64, u64) {
        (self.taken, self.waited)
    }

    pub(crate) fn before_forward(&mut self, device: &Device) -> Result<()> {
        #[cfg(test)]
        speculative_prefill_tests::note(speculative_prefill_tests::Event::Before(self.depth));
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            while self.pending.entries.len() >= self.depth {
                self.waited += self.pending.retire(|oldest| {
                    let waited = !m.marker_complete(oldest);
                    m.wait_marker(oldest)?;
                    Ok(waited)
                })? as u64;
            }
        }
        let _ = device;
        Ok(())
    }

    pub(crate) fn after_forward(&mut self, device: &Device) -> Result<()> {
        #[cfg(test)]
        speculative_prefill_tests::note(speculative_prefill_tests::Event::After(self.depth));
        #[cfg(feature = "metal")]
        if let Device::Metal(m) = device {
            self.push(m.flush_marker()?, 1);
        }
        let _ = device;
        Ok(())
    }

    fn push(&mut self, marker: PacerMarker, steps: usize) {
        #[cfg(feature = "metal")]
        self.pending.push(marker, steps);
        #[cfg(not(feature = "metal"))]
        let _ = (marker, steps);
        self.taken += 1;
    }
}

#[cfg(feature = "metal")]
type PacerMarker = lloom_tensor::metal_backend::CommandMarker;
#[cfg(not(feature = "metal"))]
type PacerMarker = ();

/// The host fill's producer (`CausalLM::host_feed`, `expert-streaming.md`
/// §20.13): a thread that reads every token the ring stamps, in order, and
/// fills the slot of the forward that consumes it. It stops once the loop's
/// GPU work is complete and every value a forward waited for is filled.
/// Dropped before [`Self::finish`] — an error or a panic on the loop — it
/// releases every wait the run encoded, so no command buffer stays parked.
struct FeedThread {
    link: std::sync::Arc<crate::nn::ngram_memory::FeedLink>,
    done: std::sync::Arc<std::sync::atomic::AtomicBool>,
    handle: Option<std::thread::JoinHandle<Result<Produced>>>,
}

/// What the producer hands back: its fill (the tokens it filled), each
/// fill's host time, its class and its trace.
type Produced = (crate::nn::ngram_memory::NgramFill, ProducerRun);

/// The producer's own account of a run.
#[derive(Default)]
struct ProducerRun {
    fill_ns: Vec<u64>,
    qos: crate::generation::QosClass,
    qos_errno: i32,
    poll: FeedWait,
    spun: u64,
    slept: u64,
    parked: u64,
    trace: Option<HostFeedTrace>,
}

impl FeedThread {
    fn start(
        feed: crate::nn::ngram_memory::HostFeed,
        mut reader: crate::generation::RingReader,
        wait: FeedWait,
    ) -> Result<Self> {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        let (link, mut fill) = feed.into_parts();
        let done = Arc::new(AtomicBool::new(false));
        reader.resume_at_published();
        let (link_t, done_t) = (Arc::clone(&link), Arc::clone(&done));
        let handle = std::thread::Builder::new()
            .name("host-feed".into())
            .spawn(
                move || match produce(&mut fill, &mut reader, &link_t, &done_t, wait) {
                    Ok(run) => Ok((fill, run)),
                    Err(e) => {
                        link_t.fail(&e);
                        Err(e)
                    }
                },
            )
            .map_err(|e| lloom_tensor::Error::Msg(format!("host feed: spawn: {e}")))?;
        Ok(Self {
            link,
            done,
            handle: Some(handle),
        })
    }

    fn link(&self) -> &crate::nn::ngram_memory::FeedLink {
        &self.link
    }

    /// The loop's GPU work is complete (its final drain returned): let the
    /// producer fill what is left and stop. The tokens it filled, in order,
    /// and its counters.
    fn finish(mut self) -> Result<(Vec<u32>, HostFeedReport)> {
        self.done.store(true, std::sync::atomic::Ordering::Release);
        self.link.wake().wake();
        let handle = self.handle.take().expect("finished once");
        let (fill, run) = handle
            .join()
            .map_err(|_| lloom_tensor::Error::Msg("host feed: the producer panicked".into()))??;
        let poll = run.poll;
        let trace = run.trace.map(|mut t| {
            t.encoded_ns = (1..=t.observed_ns.len() as u64)
                .map(|k| self.link.encoded_ns(k))
                .collect();
            eprintln!(
                "HOSTFEED {} spun={} slept={} parked={}",
                t.line(run.qos, &poll.to_string()),
                run.spun,
                run.slept,
                run.parked
            );
            t
        });
        let report = HostFeedReport {
            fills: fill.filled(),
            forwards: self.link.demanded(),
            fill_ns: run.fill_ns,
            qos: run.qos,
            qos_errno: run.qos_errno,
            poll,
            spun: run.spun,
            slept: run.slept,
            parked: run.parked,
            demand_wakes: self.link.demand_wakes(),
            trace,
        };
        Ok((fill.into_consumed(), report))
    }
}

impl Drop for FeedThread {
    fn drop(&mut self) {
        if let Some(handle) = self.handle.take() {
            self.link.release();
            self.done.store(true, std::sync::atomic::Ordering::Release);
            self.link.wake().wake();
            let _ = handle.join();
        }
    }
}

/// How the producer waits between tokens (`LLOOM_HOST_FEED_POLL`, read once).
fn feed_poll() -> FeedWait {
    static POLL: std::sync::OnceLock<FeedWait> = std::sync::OnceLock::new();
    *POLL.get_or_init(|| {
        crate::tuning::tuning()
            .ring
            .feed_poll
            .parse()
            // The knob's parser has already refused anything else.
            .unwrap_or_default()
    })
}

/// The calling thread's CPU time (0 where the clock is unavailable).
fn thread_cpu_ns() -> u64 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid clock id and a pointer to a live local.
    let rc = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut ts) };
    if rc == 0 {
        ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64
    } else {
        0
    }
}

/// The producer's loop: fill every token the ring stamps until `done` and
/// every demanded value is filled (or the run released its waits). The GPU
/// waits on this thread, so it takes its stated class first
/// (`docs/invariants.md` §15.2 #10b), parks on the ring's publish event and
/// spins only for an imminent token ([`FeedWait`]). Each fill's host time in a debug or
/// `profile` build; the trace under `LLOOM_HOST_FEED_TRACE`.
fn produce(
    fill: &mut crate::nn::ngram_memory::NgramFill,
    reader: &mut crate::generation::RingReader,
    link: &crate::nn::ngram_memory::FeedLink,
    done: &std::sync::atomic::AtomicBool,
    wait: FeedWait,
) -> Result<ProducerRun> {
    use crate::nn::ngram_memory::trace_now_ns;
    let (qos, qos_errno) =
        crate::generation::hot_thread::claim_for_spawned(&crate::tuning::tuning().ring.feed_qos);
    let mut run = ProducerRun {
        qos,
        qos_errno,
        poll: wait,
        ..ProducerRun::default()
    };
    let mut trace = link.traced().then(|| HostFeedTrace {
        cpu_ns: thread_cpu_ns(),
        wall_ns: trace_now_ns(),
        ..HostFeedTrace::default()
    });
    let mut last = trace.as_ref().map_or(0, |t| t.wall_ns);
    let mut tokens: Vec<u32> = Vec::with_capacity(64);
    let parks = reader.parks();
    let mut spins = 0u64;
    let mut notified = 0u64;
    loop {
        if let Some(t) = trace.as_mut() {
            let now = trace_now_ns();
            let gap = now - last;
            last = now;
            t.iterations += 1;
            t.max_gap_ns = t.max_gap_ns.max(gap);
            if gap > 1_000_000 {
                t.gaps_over_1ms += 1;
                t.gap_over_1ms_ns += gap;
            }
        }
        // The loop took draws back (`FeedLink::retract`): the tokens they
        // were filled from leave the window, and the reader reads their
        // slots again under the stamps the draws made again carry.
        while link.retracts_pending() > 0 {
            fill.rewind_one()?;
            reader.retract_one();
            link.ack_retract();
        }
        tokens.clear();
        reader
            .read_all_new(&mut tokens)
            .map_err(|lap| lloom_tensor::Error::Msg(format!("host feed: {lap}")))?;
        if tokens.is_empty() {
            if done.load(std::sync::atomic::Ordering::Acquire)
                && (fill.filled() >= link.demanded() || link.released())
            {
                if let Some(mut t) = trace.take() {
                    t.cpu_ns = thread_cpu_ns().saturating_sub(t.cpu_ns);
                    t.wall_ns = trace_now_ns() - t.wall_ns;
                    t.prefetch_qos = prefetch_worker_qos();
                    t.read_pool_qos = crate::nn::ngram_memory::read_pool_qos();
                    run.trace = Some(t);
                }
                return Ok(run);
            }
            match wait.step(link.demanded(), fill.filled(), spins, parks) {
                FeedStep::Park => {
                    run.parked += 1;
                    let wake = link.wake();
                    let seen = wake.epoch();
                    // The GPU's wake for the next token, once per token.
                    let next = reader.cursor() as u64 + 1;
                    if notified != next {
                        let w = std::sync::Arc::clone(wake);
                        reader.notify_on_publish(next, move || w.wake());
                        notified = next;
                    }
                    // Looked at after the epoch: a wake from here on is not lost.
                    if !reader.has_new()
                        && link.demanded() <= fill.filled()
                        && !done.load(std::sync::atomic::Ordering::Acquire)
                    {
                        wake.park(seen, std::time::Duration::from_millis(FEED_PARK_MS));
                    }
                }
                FeedStep::Poll(m @ crate::generation::poll::PollMechanism::Sleep(_)) => {
                    run.slept += 1;
                    reader.wait_for_publish(m);
                }
                FeedStep::Poll(m) => {
                    run.spun += 1;
                    spins += 1;
                    reader.wait_for_publish(m);
                }
            }
            continue;
        }
        spins = 0;
        let observed = if trace.is_some() { trace_now_ns() } else { 0 };
        for &t in &tokens {
            #[cfg(any(debug_assertions, feature = "profile"))]
            let t0 = std::time::Instant::now();
            fill.fill(t)?;
            #[cfg(any(debug_assertions, feature = "profile"))]
            run.fill_ns.push(t0.elapsed().as_nanos() as u64);
            if let Some(tr) = trace.as_mut() {
                tr.observed_ns.push(observed);
                tr.signalled_ns.push(trace_now_ns());
            }
        }
    }
}

/// The prefetcher's reader threads' class, where the tier exists.
fn prefetch_worker_qos() -> crate::generation::QosClass {
    #[cfg(feature = "metal")]
    {
        crate::nn::expert_prefetch::worker_qos()
    }
    #[cfg(not(feature = "metal"))]
    {
        crate::generation::QosClass::Unknown
    }
}

/// Disarms the EOS gate when dropped (`Device::eos_gate_end`), so no exit
/// path of `decode_until` leaves it armed; `finish` takes the report.
struct EosGateGuard<'a>(&'a Device);

impl EosGateGuard<'_> {
    fn finish(self) -> lloom_tensor::EosGateReport {
        let report = self.0.eos_gate_end().unwrap_or_default();
        std::mem::forget(self);
        report
    }
}

impl Drop for EosGateGuard<'_> {
    fn drop(&mut self) {
        let _ = self.0.eos_gate_end();
    }
}

/// The adaptive commit's two primitives; `None` / `()` off Metal, where the
/// ring arms `adaptive_commit` requires do not exist.
fn device_flush_marker_if_started(device: &Device) -> Result<Option<PacerMarker>> {
    #[cfg(feature = "metal")]
    {
        if let Device::Metal(m) = device {
            return m.flush_marker_if_started();
        }
        Ok(None)
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = device;
        Ok(None)
    }
}

fn device_flush_marker(device: &Device) -> Result<PacerMarker> {
    #[cfg(feature = "metal")]
    {
        if let Device::Metal(m) = device {
            return m.flush_marker();
        }
        lloom_tensor::bail!("adaptive_commit: not a Metal device")
    }
    #[cfg(not(feature = "metal"))]
    {
        let _ = device;
        Ok(())
    }
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
    /// `lookahead` steps ahead, [`IN_FLIGHT_STEPS`] deep), so draining
    /// `ReadbackStrategy::take_ids` there reads work that is already done — one
    /// batched readback per commit, not the per-token round trip §11.5a deleted.
    /// A caller that does not stream passes a no-op; the tail drain at the end is
    /// unchanged either way. Streaming appears in commit-sized chunks, which is
    /// the price of not blocking: a token cannot be shown before the GPU makes
    /// it, and the batch is what keeps that observation off the per-token path.
    /// That is the fixed cadence; on a reader-owned run (the flow, or
    /// `stream_via_reader`) the reader is the stream and `on_tokens` is not
    /// called.
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
        let out = self.decode_until_run(spec, readback, sampler, prompt, opts, on_tokens);
        if out.is_err() {
            // A run that failed mid-way may leave a provisional step: settled
            // (whatever it finds, the run's error is the answer) and the
            // cache's steps the model's own again.
            let _ = self
                .model
                .provisional(&mut self.seq.cache, crate::nn::model::Provisional::Settled);
            let _ = self.model.settle_step(&mut self.seq.cache);
            let _ = self
                .model
                .provisional(&mut self.seq.cache, crate::nn::model::Provisional::Off);
        }
        out
    }

    fn decode_until_run<S, R>(
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
        if opts.adaptive_commit {
            if !opts.stream_via_reader {
                lloom_tensor::bail!(
                    "Session::decode: adaptive_commit needs stream_via_reader — the drain is \
                     the reader's, per token; the loop's own drain rides commits it no longer makes"
                );
            }
            if opts.lookahead != 1 {
                lloom_tensor::bail!(
                    "Session::decode: adaptive_commit needs lookahead 1 (got {}): the loop \
                     encodes the next forward before anything is read back",
                    opts.lookahead
                );
            }
            if matches!(readback.eos_read_arm(), "flag" | "host") {
                lloom_tensor::bail!(
                    "Session::decode: adaptive_commit on the `{}` EOS arm reads the token back \
                     per step; use a ring arm (`poll`, `drain`, `gpubatch`, \
                     `gpubatch-nonblocking`)",
                    readback.eos_read_arm()
                );
            }
        }
        if opts.eos_gate {
            if !opts.adaptive_commit {
                lloom_tensor::bail!(
                    "Session::decode: eos_gate needs adaptive_commit — it gates the forwards \
                     that loop queues past an EOS"
                );
            }
            if readback.eos_ids_tensor().is_none() {
                lloom_tensor::bail!(
                    "Session::decode: eos_gate on the `{}` strategy, which holds no device copy \
                     of the stop set for the gate kernel to compare against",
                    readback.eos_read_arm()
                );
            }
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
        spec.begin_at(SeqId(0), prompt, &self.seq.tokens)?;
        readback.begin_run();

        let device = readback.device().clone();
        let terminal_run = crate::nn::terminal_trace::run_begin(
            || readback.published_count(),
            std::mem::take(&mut self.terminal_prepare),
            readback.terminal_observers(),
        );
        // The hot thread's scheduling class, held for the run and restored on
        // return (`generation::hot_thread`): two syscalls a generation, none a
        // step.
        let hot_thread = HotThread::claim();
        let mut stats = DecodeStats {
            step_ns: Vec::with_capacity(opts.max_tokens),
            sample_ns: Vec::with_capacity(if opts.sample_split {
                opts.max_tokens
            } else {
                0
            }),
            hot_thread: hot_thread.report(),
            ..DecodeStats::default()
        };

        // The in-flight bound, created BEFORE the prefill so the prompt-steps and
        // the generate-steps share ONE pacer — the pipeline spans prefill→decode
        // with no drain at the boundary, and the pacer's depth lets the host encode
        // the next unit while the GPU runs the current one (host↔GPU overlap). The
        // blocking arm's readback drains the queue every step, so its stream stays
        // the incumbent's; depth is the computed host/GPU balance
        // (`opts.in_flight_depth`, from warmup) or the floor — not a constant.
        let mut pacer = StepPacer::for_decode(opts);
        let pace = readback.loop_commits();
        // The caller streams off its own `RingReader`: the ring arm skips the
        // hot-loop drain and materialises the host id vec once at end-of-run.
        // `pace == loop_commits`, so this is the ring-arm condition. Only the ring
        // arms carry a device-resident ring to read; the blocking arm is
        // host-side per step and this is a no-op for it.
        let reader_owned = opts.stream_via_reader && pace;
        // Run prefill deeper than decode when asked: prefill is compute-bound and
        // a chunked / many-small-forward prefill serializes at the decode depth of
        // 2 (each forward waits on the one two back). A deeper bound lets the
        // prefill forwards overlap. Restored to the decode depth after the prefill
        // block, so steady decode is unaffected. `0` = same as decode.
        if opts.prefill_pacer_depth > 0 {
            pacer.set_depth(opts.prefill_pacer_depth);
        }

        // Prefill: the state machine STEPS OVER THE PROMPT before it steps over
        // its own generated tokens — same stepping, different input source. Rather
        // than one monolithic `forward(whole_prompt)` the host blocks on, the
        // prompt flows in as `prefill_ubatch`-wide units through the shared pacer:
        // each unit is staged into the ring's input buffer (zero-copy) and read as
        // a narrow, exactly as a decode step feeds its one token. A unit does NOT
        // sample (the tokens are given) and produces no output (nothing generated
        // yet); only the LAST unit's last-position logits seed the first generate
        // step. `prefill_ubatch == 0` means one unit over the whole prompt — the
        // incumbent one-shot, so the default path is byte-identical.
        //
        // Correctness across unit boundaries: a `seq_len>1` conv writer zero-pads
        // its left, exact only for a cold cache; a unit onto a WARM cache must
        // carry the previous window (`Config::conv_prefill_ctx`). The caller turns
        // that on when it streams (`prefill_ubatch>0`); a single unit is identical
        // either way (a cold cache has no left context to carry).
        let pos_before = self.seq.pos;
        let census_before = census_snapshot();
        let prefill_start = std::time::Instant::now();
        let mut prefill_units = 0usize;
        let mut logits = match self.seq.last_logits.take() {
            Some(held) if resume => held,
            _ => {
                let (last, units) =
                    self.prefill_forwards(readback, prompt, opts, &mut pacer, pace)?;
                prefill_units = units;
                last.expect("a non-empty prompt steps at least once")
            }
        };
        end_executor_step(&device)?;
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;
        stats.prefill_units = prefill_units;
        let census_prefill = census_snapshot();
        stats.prefill_census = census_delta(&census_before, census_prefill.clone());
        // Restore the decode in-flight bound: prefill's deeper bound (if any) is
        // done. A lower depth here is honoured on the next `before_forward`,
        // which retires markers down to it before the first decode step encodes.
        if opts.prefill_pacer_depth > 0 {
            pacer.set_depth(opts.pacer_depth());
        }

        // Provisional steps (`CausalLM::provisional`), where the loop can take
        // a draw back (`ReadbackStrategy::retract_draw`) and the sampler can
        // draw again as if it had not (no processor history):
        // - the blocking arm (`Settled`): the model's step returns without
        //   settling and the loop settles it after the readback — the pass
        //   is complete, the settle waits for nothing — redrawing from the
        //   redone step when it computed on something it lacked;
        // - the ring on the fixed cadence at `lookahead = 1` (`Behind`): the
        //   loop settles each step after it has encoded the next, while the
        //   GPU runs it; a redone step undoes the forward after it, whose
        //   draw is taken back and drawn again and whose forward is encoded
        //   again. Not under the adaptive commit or a streaming reader: a
        //   token the ring stamped there may be streamed before its step is
        //   settled.
        // Armed after the prefill, whose passes settle as they always did.
        let provisional = {
            use crate::nn::model::Provisional;
            let takes_back = readback.retracts_draw() && sampler.mark().is_some();
            let want = if !takes_back {
                Provisional::Off
            } else if !readback.arms().token_ring && opts.lookahead == 0 {
                Provisional::Settled
            } else if readback.arms().token_ring
                && opts.lookahead == 1
                && !opts.adaptive_commit
                && !opts.stream_via_reader
                && !opts.eos_gate
            {
                Provisional::Behind
            } else {
                Provisional::Off
            };
            if want == Provisional::Off {
                Provisional::Off
            } else {
                self.model.provisional(&mut self.seq.cache, want)?
            }
        };
        stats.provisional = provisional != crate::nn::model::Provisional::Off;

        // The model's host fill (`CausalLM::host_feed`): on an arm whose
        // reader sees every token as it is stamped, a producer thread fills
        // each decode forward's host-side input from the token before it and
        // the GPU waits for it, so no forward reads its input back. Armed
        // after the prefill, whose ids are the host's already.
        let mut host_feed = None;
        let feed = if readback.stamps_each_token(opts.adaptive_commit) {
            self.model.host_feed(&mut self.seq.cache)?
        } else {
            None
        };
        // A host thread that waits on the ring's stamp parks on its publish
        // event: armed for this run before the producer's reader is taken,
        // for a producer that parks ([`FeedWait::Event`]) and for the fixed
        // cadence's EOS wait.
        let wait = opts.host_feed_wait.unwrap_or_else(feed_poll);
        let producer_parks = feed.is_some() && matches!(wait, FeedWait::Event { .. });
        let eos_park = opts
            .eos_park
            .unwrap_or(crate::tuning::tuning().ring.eos_park);
        if let Err(e) = readback.arm_publish_event(producer_parks, eos_park, opts.adaptive_commit) {
            if let Some(f) = &feed {
                f.link().release();
                self.model.host_feed_end(&mut self.seq.cache, &[])?;
            }
            return Err(e);
        }
        {
            if let Some(f) = feed {
                let link = std::sync::Arc::clone(f.link());
                let started = match readback.reader() {
                    Some(reader) => FeedThread::start(f, reader, wait).map(Some),
                    None => Ok(None),
                };
                match started {
                    Ok(Some(t)) => host_feed = Some(t),
                    Ok(None) => self.model.host_feed_end(&mut self.seq.cache, &[])?,
                    Err(e) => {
                        link.release();
                        self.model.host_feed_end(&mut self.seq.cache, &[])?;
                        return Err(e);
                    }
                }
            }
        }

        // The loop's own EOS watch (`DecodeOptions::eos_watch`): a reader of
        // its own, taken before this run's first stamp, read before each step.
        let watch_on = opts.adaptive_commit
            && opts
                .eos_watch
                .unwrap_or(crate::tuning::tuning().ring.eos_watch);
        let mut watch = if watch_on {
            readback.reader().map(|mut r| {
                r.resume_at_published();
                r.mark_watch();
                r
            })
        } else {
            None
        };
        let mut watch_ids: Vec<u32> = Vec::with_capacity(if watch.is_some() {
            opts.adaptive_lead() + 1
        } else {
            0
        });

        let mut produced = 0usize;
        let mut since_commit = 0usize;
        let mut eos = false;
        // Every id the loop drains, streamed or not, accumulates here so the
        // returned `ids` is complete and the reconciliation below counts the
        // whole run. Draining for the stream advances the strategy's cursor, so
        // the final `take_ids` only sees the tail; the total is what matches
        // `produced`, not either part alone.
        let mut all_ids: Vec<u32> = Vec::with_capacity(opts.max_tokens);

        // With a lookahead the host encodes step `s+1` while the GPU runs step
        // `s`, so a command-buffer cut inside a step buys no overlap and costs
        // a seam: the step loop runs at the pipelined cut period (512, one
        // interior cut a step at most) and the default (50) is restored when
        // the loop ends. Without a lookahead the default is the overlap and
        // stays (`Device::pipelined_cut_period`; lloom #715).
        let _pipelined_period = if opts.adaptive_commit {
            // Only the loop cuts: no counter cut inside a step or across them.
            Some(device.cut_period_hold(usize::MAX))
        } else {
            (opts.lookahead >= 1).then(|| device.pipelined_cut_period())
        };
        let run_ahead = if opts.run_ahead == 0 {
            8
        } else {
            opts.run_ahead
        };
        let mut steps_open = 0usize;
        // Armed here, disarmed on every exit path (the guard's `Drop`): an
        // armed gate outliving the run would route the next prefill through
        // a stale table.
        let eos_gate = if opts.eos_gate {
            device.eos_gate_begin()?;
            Some(EosGateGuard(&device))
        } else {
            None
        };
        // The fused publish (lloom #735 step 5) where the strategy can take it.
        readback.set_fused_publish(opts.adaptive_commit && opts.fused_publish);

        // Forwards fed past the prompt so far, for the terminal diagnostic.
        let fed_now = |s: &Self| s.seq.pos - pos_before - prompt.len();
        loop {
            if opts.reached(produced) {
                crate::nn::terminal_trace::producer_stop(
                    &terminal_run,
                    crate::nn::terminal_trace::StopBy::Length,
                    || (fed_now(self), readback.published_count()),
                );
                break;
            }
            // One autorelease pool per step, dropped last: the step's encoders
            // and command buffers are released here instead of never (the
            // release loop leaked ~200 autoreleased objects a token with no
            // pool on the thread -- `AutoreleaseScope` docs).
            if let Some(stop) = &opts.stop {
                if stop.load(std::sync::atomic::Ordering::Acquire) {
                    stats.stopped_by_flag = true;
                    crate::nn::terminal_trace::producer_stop(
                        &terminal_run,
                        crate::nn::terminal_trace::StopBy::Flag,
                        || (fed_now(self), readback.published_count()),
                    );
                    break;
                }
            }
            if let Some(w) = watch.as_mut() {
                watch_ids.clear();
                if let Err(lap) = w.read_new(&mut watch_ids) {
                    // The watch reads every step and the host is at most
                    // `adaptive_lead` steps ahead of what the GPU publishes,
                    // so a lap means the bound the watch rests on is broken.
                    lloom_tensor::bail!(
                        "Session::decode: the EOS watch fell behind the ring ({lap})"
                    )
                }
                if w.ended() {
                    stats.stopped_by_eos_watch = true;
                    crate::nn::terminal_trace::producer_stop(
                        &terminal_run,
                        crate::nn::terminal_trace::StopBy::Watch,
                        || (fed_now(self), readback.published_count()),
                    );
                    break;
                }
            }
            let _pool = lloom_tensor::AutoreleaseScope::new();
            stats.steps += 1;
            if stats.steps == 1 {
                crate::nn::terminal_trace::decode_begin(&terminal_run);
            }
            // Under `CANDLE_METAL_PROFILE=1` only: tell the profile recorder
            // which step the command buffers committed from here on belong to,
            // so an idle gap between two buffers can be laid at a step (issue
            // #702). A cached-bool check and nothing else on every other run.
            #[cfg(feature = "metal")]
            {
                use lloom_tensor::kernels::metal::profile;
                if profile::enabled() {
                    profile::set_step(stats.steps);
                }
            }
            // Only the first `FIRST_STEPS_TRACED` steps record per-phase timings —
            // and only in a debug/profile build. In release `traced` is a constant
            // `false`, so every `traced.then(Instant::now())` below is `None` and NO
            // phase clock read (`mach_absolute_time()`) reaches even the first steps:
            // the release decode path is free of per-step instrumentation entirely.
            // `timed`: every step's phases are bracketed and summed into
            // `phase_totals` (the steady-state split); `traced`: the first few
            // are also kept individually. Both are constant `false` in release.
            #[cfg(any(debug_assertions, feature = "profile"))]
            let timed = true;
            #[cfg(not(any(debug_assertions, feature = "profile")))]
            let timed = false;
            let traced = timed && stats.steps <= FIRST_STEPS_TRACED;
            // The per-step median clock read — DEBUG/PROFILE ONLY. In a plain
            // release build this `mach_absolute_time()` (and the `step_ns.push`es it
            // feeds) is compiled OUT, so the steady decode path takes no per-step
            // clock read at all; release reports coarse tokens ÷ wall-clock instead
            // (measured host-side, off the loop). `any(debug_assertions, profile)`.
            #[cfg(any(debug_assertions, feature = "profile"))]
            let step_start = std::time::Instant::now();
            if opts.sample_split {
                device.synchronize()?;
            }
            // `sample_start` feeds only the phase trace (`t_sample`) and the
            // sample-split split; skip the clock read on the steady path.
            let sample_start = (timed || opts.sample_split).then(std::time::Instant::now);

            let mark = if provisional != crate::nn::model::Provisional::Off {
                sampler.mark()
            } else {
                None
            };
            // The draw is for the token at `seq.pos`, and its noise is keyed
            // to that position rather than to how many draws preceded it
            // (`LogitsProcessor::at_position`): a sampled speculative run
            // draws the same positions a window at a time and must land on
            // the same tokens at one seed (#943 §2). Only the gumbel-max
            // arms read the key; argmax and the host-sort arms are unmoved.
            sampler.at_position(Some(self.seq.pos as u64));
            let mut feed = readback.sample_and_feed(sampler, &logits)?;
            // The step behind these logits settled, now that its pass is done
            // and before its token goes anywhere: a redone step's logits are
            // drawn from again, the first draw taken back.
            if provisional == crate::nn::model::Provisional::Settled {
                if let Some(redone) = self.model.settle_step(&mut self.seq.cache)? {
                    readback.retract_draw()?;
                    if let Some(m) = mark.clone() {
                        sampler.rewind(m);
                    }
                    logits = redone.squeeze(0)?;
                    sampler.at_position(Some(self.seq.pos as u64));
                    feed = readback.sample_and_feed(sampler, &logits)?;
                    stats.redraws += 1;
                }
            }
            if let Some(f) = &host_feed {
                f.link().note_fed();
            }
            // An arm that holds the fed id host-side hands it to the model
            // (`CausalLM::note_input_ids`): a no-op for every model whose
            // forward reads nothing back, the readback it would otherwise
            // pay for a model that hashes its input ids on the host.
            if let Some(ids) = readback.fed_host_ids() {
                self.model.note_input_ids(&mut self.seq.cache, ids)?;
            }
            if opts.adaptive_commit && !readback.publishes_fused() {
                // The reader drains on the stamp: the deferred arms write it
                // only at commits, so under the adaptive commit it rides every
                // step — HERE, before the next forward is encoded, so token g
                // is readable as soon as its argmax has run, not after
                // forward g+1 (one dispatch, as the poll arm's stamp is).
                readback.publish_marker(true)?;
                if opts.eos_gate {
                    // The gate on the token this forward consumes, then the
                    // forward's dispatches through the table it may zero.
                    let eos_ids = readback
                        .eos_ids_tensor()
                        .ok_or_else(|| lloom_tensor::Error::Msg("eos_gate: no stop set".into()))?;
                    device.eos_gate_step(&feed, eos_ids)?;
                }
            }
            produced += 1;
            // A step whose commit is due below closes a drain window: let the
            // strategy enqueue what that drain will read BEFORE the lookahead
            // forward, so it completes with the window rather than a step later
            // (`ReadbackStrategy::close_window`; a no-op on every arm but the
            // non-blocking batched-EOS one). The condition mirrors the commit
            // block's: every step under `lookahead = 0`, else the TTFT ramp or
            // the budget's period.
            if readback.loop_commits()
                && (opts.lookahead == 0
                    || produced <= opts.budget.period()
                    || opts.budget.due(since_commit + 1))
            {
                readback.close_window(produced <= opts.budget.period())?;
            }

            // The fair, llama-bench-comparable TTFT, stamped ONCE (step 0) — the
            // first output token is the argmax of the prefill's last-position
            // logits, sampled here, so `prefill_start.elapsed()` now is the host
            // time to submit prefill + sample the first token. No forced sync (the
            // async pipeline is the design — this is the honest difference from
            // llama-bench, which blocks on `llama_decode`), no drain cadence, no
            // consumer/print. ONE clock read per generation, so release-safe.
            if stats.steps == 1 {
                stats.ttft_compute_ns = prefill_start.elapsed().as_nanos() as u64;
            }

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
                    // The completion marker for the window closing at this commit,
                    // encoded BEFORE the commit so it rides the committing buffer;
                    // it certifies the window's slots, the ordering the drain's
                    // clamp rests on (§3.5). A no-op on the arms that stamp per
                    // step or read back per step. `force_live` during the
                    // TTFT ramp so token 0 never waits `live_every` commits.
                    readback.publish_marker(produced <= opts.budget.period())?;
                    self.commit(&device, &mut stats)?;
                    since_commit = 0;
                    committed_this_step = true;
                }
                let is_eos = readback.eos_of_last()?;
                if let Some(s) = sample_start {
                    if opts.sample_split {
                        stats.sample_ns.push(s.elapsed().as_nanos() as u64);
                    }
                }
                if is_eos {
                    produced -= 1;
                    eos = true;
                    crate::nn::terminal_trace::producer_stop(
                        &terminal_run,
                        crate::nn::terminal_trace::StopBy::EosOfLast,
                        || (fed_now(self), readback.published_count()),
                    );
                    break;
                }
                if pace {
                    pacer.before_forward(&device)?;
                }
                self.model.prepare_pass(&mut self.seq.cache, 1)?;
                begin_decode_step(&device);
                logits = self
                    .model
                    .forward(&feed, self.seq.pos, &mut self.seq.cache)?
                    .squeeze(0)?;
                end_decode_step(&device)?;
                self.seq.pos += 1;
                if pace {
                    pacer.after_forward(&device)?;
                }
            } else {
                if let Some(s) = sample_start {
                    if opts.sample_split {
                        stats.sample_ns.push(s.elapsed().as_nanos() as u64);
                    }
                }
                // ENQUEUE, then learn. The GPU has this forward queued before
                // it retires the previous one. The pacer bounds the enqueue:
                // an arm with no per-token wait would otherwise run ahead
                // without limit (#690's marker pacer, replacing the removed
                // `apply_backpressure` spin).
                //
                // The per-PHASE timings (sample / forward / backpressure / eos) are
                // recorded ONLY for the first `FIRST_STEPS_TRACED` steps (`traced`,
                // set at the loop top). Every clock read is a `mach_absolute_time()`
                // on the hot thread — a call the OS scheduler sees — so reading them
                // on all 2382 steps and discarding them past step 4 was pure
                // hot-path noise. Gated behind `traced`, so the steady path takes
                // NONE; the one clock read it still needs is `step_ns` (the median).
                let t_sample = sample_start.map_or(0, |s| s.elapsed().as_nanos() as u64);
                if opts.adaptive_commit {
                    // Steps queued in incomplete buffers, not buffers: two
                    // run-ahead windows deep, so one is running while the
                    // other waits and the host is never the one being waited on.
                    pacer.before_forward_bounded(&device, 2 * run_ahead)?;
                } else if pace {
                    pacer.before_forward(&device)?;
                }
                let t = timed.then(std::time::Instant::now);
                self.model.prepare_pass(&mut self.seq.cache, 1)?;
                begin_decode_step(&device);
                logits = self
                    .model
                    .forward(&feed, self.seq.pos, &mut self.seq.cache)?
                    .squeeze(0)?;
                end_decode_step(&device)?;
                self.seq.pos += 1;
                let t_forward = t.map_or(0, |t| t.elapsed().as_nanos() as u64);
                // The in-flight bound AFTER the enqueue: retire the oldest
                // marker if the host would exceed `IN_FLIGHT_STEPS` ahead. This
                // is #690's pacer (Metal completion), which replaced the
                // unbounded counter-spin `apply_backpressure` livelocked on.
                // `backpressure_ns` reports what that bound cost this step.
                let t = timed.then(std::time::Instant::now);
                if opts.adaptive_commit {
                    // The adaptive commit: close the buffer when the queue is
                    // down to the running one, or at the run-ahead bound.
                    steps_open += 1;
                    let marker = match device_flush_marker_if_started(&device)? {
                        Some(m) => {
                            stats.adaptive_commits += 1;
                            Some(m)
                        }
                        None if steps_open >= run_ahead => {
                            stats.run_ahead_closes += 1;
                            Some(device_flush_marker(&device)?)
                        }
                        None => None,
                    };
                    if let Some(m) = marker {
                        stats.max_steps_per_buffer = stats.max_steps_per_buffer.max(steps_open);
                        pacer.push(m, steps_open);
                        steps_open = 0;
                        // The reader streams; the ids the sequence keeps are
                        // taken off the ring here, so a run longer than the
                        // ring does not lap it. Never waits (§11.5z-x).
                        readback.take_ids_into(&mut all_ids)?;
                    }
                } else if pace {
                    pacer.after_forward(&device)?;
                }
                // Behind (`Provisional::Behind`): the step before this one
                // settles now, while the GPU runs this one. If it is redone,
                // the draw this forward consumed came from wrong logits: the
                // model has undone this forward and redone the step; the
                // draw is taken back (the ring's slot, the host fill's
                // token), drawn again from the redone logits, and the forward
                // encoded again. Before `eos_of_last` and every drain, so no
                // taken-back token is ever streamed or read as EOS.
                if provisional == crate::nn::model::Provisional::Behind {
                    // Settled again after a forward encoded again: the redone
                    // step's window, held behind it, is released there.
                    while let Some(redone) = self.model.settle_step(&mut self.seq.cache)? {
                        readback.retract_draw()?;
                        if let Some(f) = &host_feed {
                            f.link().retract(FEED_RETRACT_TIMEOUT)?;
                        }
                        if let Some(m) = mark.clone() {
                            sampler.rewind(m);
                        }
                        self.seq.pos -= 1;
                        sampler.at_position(Some(self.seq.pos as u64));
                        let again = readback.sample_and_feed(sampler, &redone.squeeze(0)?)?;
                        if let Some(f) = &host_feed {
                            f.link().note_fed();
                        }
                        if pace {
                            pacer.before_forward(&device)?;
                        }
                        self.model.prepare_pass(&mut self.seq.cache, 1)?;
                        begin_decode_step(&device);
                        logits = self
                            .model
                            .forward(&again, self.seq.pos, &mut self.seq.cache)?
                            .squeeze(0)?;
                        end_decode_step(&device)?;
                        self.seq.pos += 1;
                        if pace {
                            pacer.after_forward(&device)?;
                        }
                        stats.redraws += 1;
                    }
                }
                let t_backpressure = t.map_or(0, |t| t.elapsed().as_nanos() as u64);
                let t = timed.then(std::time::Instant::now);
                // Under the adaptive commit nothing on this thread waits on the
                // token: the poll arm's `eos_of_last` would WFE on a stamp inside
                // an uncommitted buffer, and EOS is the reader's flag (its device
                // EOS byte still lands every step — the gate's input).
                let is_eos = if opts.adaptive_commit {
                    false
                } else {
                    readback.eos_of_last()?
                };
                if timed {
                    let phases = StepPhases {
                        sample_ns: t_sample,
                        forward_ns: t_forward,
                        backpressure_ns: t_backpressure,
                        eos_ns: t.map_or(0, |t| t.elapsed().as_nanos() as u64),
                    };
                    stats.phase_totals.add(&phases);
                    stats.phase_steps += 1;
                    if traced {
                        stats.first_steps.push(phases);
                    }
                }
                if is_eos {
                    produced -= 1;
                    eos = true;
                    crate::nn::terminal_trace::producer_stop(
                        &terminal_run,
                        crate::nn::terminal_trace::StopBy::EosOfLast,
                        || (fed_now(self), readback.published_count()),
                    );
                    break;
                }
            }

            // The periodic machinery: the only place the lookahead arm
            // commits. A step that already committed before its wait is not
            // committed twice.
            //
            // EARLY-COMMIT RAMP (TTFT). The steady commit cadence is `period`
            // (e.g. 8), which batches the drain — good for throughput. But the
            // drain is the ONLY place a token streams to the consumer, so at
            // `period = 8` the FIRST token the user sees waits for step 8: measured
            // ~161 ms of user-visible first-token latency while the token was
            // computed at ~48 ms. That is the reported TTFT made moot. So for the
            // first `period` tokens, commit EVERY step — each reads finished work
            // (the drain never waits, §11.5a), so it costs a commit, not a stall —
            // and the first token streams as soon as it is computed. After the
            // ramp the steady period takes over and throughput is unchanged. The
            // ramp only SHORTENS early commits, so it cannot delay token 0; it is
            // TTFT-safe by construction. `period == 1` (per-token) already commits
            // every step, so the ramp is a no-op there.
            let in_ttft_ramp = produced <= opts.budget.period();
            let mut committed_now = committed_this_step;
            if loop_commits && !committed_this_step && !opts.adaptive_commit {
                since_commit += 1;
                if in_ttft_ramp || opts.budget.due(since_commit) {
                    // The completion marker for the window closing here, encoded
                    // before the commit so it rides the committing buffer; it
                    // certifies the window's slots (§3.5). No-op on the per-step
                    // arms.
                    // `force_live` during the TTFT ramp so early tokens stream at
                    // once regardless of the wake cadence.
                    readback.publish_marker(in_ttft_ramp)?;
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
            // When the caller streams through its own `RingReader`, the ring arm
            // does NOT drain here: the ids live on the device ring (read
            // per-token off the reader, no sync) and the host id vec is
            // materialised once at end-of-run. `produced` and EOS are per-step
            // signals (`produced += 1`, `eos_of_last`), independent of the drain,
            // so the loop is correct without it. The blocking arm ignores this
            // (its ids are host-side per step; nothing to defer). `reader_owned`
            // is hoisted above the loop.
            let stream_now = !reader_owned && if loop_commits { committed_now } else { true };
            if reader_owned && committed_now {
                readback.take_ids_into(&mut all_ids)?;
            }
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
                        #[cfg(any(debug_assertions, feature = "profile"))]
                        stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                        break;
                    }
                    let go_on = on_tokens(&batch)?;
                    all_ids.extend_from_slice(&batch);
                    if !go_on {
                        crate::nn::terminal_trace::producer_stop(
                            &terminal_run,
                            crate::nn::terminal_trace::StopBy::Hook,
                            || (fed_now(self), readback.published_count()),
                        );
                        #[cfg(any(debug_assertions, feature = "profile"))]
                        stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                        break;
                    }
                }
            }
            end_executor_step(&device)?;
            // The per-step median + the step-0 census: DEBUG/PROFILE ONLY. Release
            // takes neither clock read nor census on the steady path.
            #[cfg(any(debug_assertions, feature = "profile"))]
            {
                stats.step_ns.push(step_start.elapsed().as_nanos() as u64);
                if stats.steps == 1 {
                    stats.step0_census = census_delta(&census_prefill, census_snapshot());
                }
            }
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
        let tail_start = std::time::Instant::now();
        if reader_owned {
            // The reader streamed every token off the device ring already; the
            // loop kept the ids at each buffer it closed. The rest come here,
            // once: `drain_all` synchronises and reads the remaining published
            // range — on the poll arm the trailing EOS token (published before
            // the loop broke), on a deferred arm any overrun. `produced` is the
            // per-step count with a retracted EOS excluded, so the first
            // `produced` ids are the sequence, in order. No `on_tokens`: the
            // reader is the stream.
            all_ids.extend(readback.drain_all()?);
            all_ids.truncate(produced);
            // An arm that learns EOS late has it in the ids: the reader stopped
            // the run at an EOS it saw, or `max_tokens` arrived first with the
            // EOS inside the last window. The ids past it are the overrun,
            // reconciled below from the cache position. Scanned on every
            // reader-owned run; the arms that retract per step find none.
            if let Some(k) = readback.eos_in_drained(&all_ids) {
                readback.note_eos_truncation(all_ids.len() - k - 1);
                all_ids.truncate(k);
                produced = all_ids.len();
                eos = true;
            }
        } else {
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
        }
        // Every wait a forward encoded is satisfiable once what the loop
        // encoded is committed (the final drain synchronised it unless there
        // was nothing left to drain); the producer fills what is left of them
        // and stops. The model then brings the host-side state the fed
        // forwards skipped up to date.
        if let Some(f) = host_feed.take() {
            flush_and_prune(&device)?;
            let (published, report) = f.finish()?;
            self.model.host_feed_end(&mut self.seq.cache, &published)?;
            stats.host_feed = Some(report);
        }
        // The last step, when provisional, settled before its logits are
        // held: a redone step's are the ones a continuation resumes from.
        if provisional != crate::nn::model::Provisional::Off {
            // Behind, the newest step's settle waited for a next forward that
            // never came: every step settles now.
            self.model
                .provisional(&mut self.seq.cache, crate::nn::model::Provisional::Settled)?;
            if let Some(redone) = self.model.settle_step(&mut self.seq.cache)? {
                logits = redone.squeeze(0)?;
            }
            self.model
                .provisional(&mut self.seq.cache, crate::nn::model::Provisional::Off)?;
        }
        // Work the model let trail the last step (a fetching settle behind
        // the next encode) is the run's: settled here, after the host fill
        // could satisfy every wait the forwards encoded.
        self.model.settle_pending(&mut self.seq.cache)?;
        readback.arm_publish_event(false, false, true)?;
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
        // Back to the running stream for whatever else this sampler serves.
        sampler.at_position(None);
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
        self.seq.last_logits = if overrun == 0 {
            Some(self.hold_logits(logits)?)
        } else {
            None
        };
        stats.overrun = overrun;
        if let Some(guard) = eos_gate {
            let report = guard.finish();
            if report.fired && overrun > 0 {
                // Every forward past the EOS launched nothing: the cache's
                // contents end at the EOS; only the counters are ahead.
                self.seq.overrun_unwritten = true;
                stats.overrun_unwritten = true;
            }
            stats.eos_gate = Some(report);
        }
        stats.produced = produced;
        stats.tail_ns = tail_start.elapsed().as_nanos() as u64;
        crate::nn::terminal_trace::run_end(terminal_run, produced, overrun, eos, &device);
        stats.step_markers = pacer.taken;
        stats.step_waits = pacer.waited;
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

    /// The logits a later call resumes from, in memory this sequence owns.
    ///
    /// A pass's output lives in the pass's memory: under the pass cache (or
    /// the arena) that is a slot of one base every pass on the device reuses,
    /// so logits held across passes — this sequence's next `decode` with an
    /// empty prompt, a batched resume row, another session's passes in
    /// between — would be read after their slot was overwritten. This copies
    /// them into a per-sequence buffer allocated once, outside every pass (so
    /// no plan ever serves it), by one dispatch encoded before any other pass
    /// is: the barrier chain orders the read against the next writer. When
    /// the device serves activations from the pool only, the held buffer is
    /// its own and is returned as is (`DESIGN.md` §9.2aa, escapes are
    /// declared).
    pub(crate) fn hold_logits(&mut self, logits: Tensor) -> Result<Tensor> {
        if !activations_may_be_reused(logits.device())? {
            return Ok(logits);
        }
        let hold = match &self.seq.logits_hold {
            Some(h) if h.shape() == logits.shape() && h.dtype() == logits.dtype() => h.clone(),
            _ => {
                let h = Tensor::zeros(logits.shape(), logits.dtype(), logits.device())?;
                self.seq.logits_hold = Some(h.clone());
                h
            }
        };
        hold.slice_set(&logits, 0, 0)?;
        Ok(hold)
    }

    /// Feed `prompt` through the model as the forwards [`prefill_widths`]
    /// names, each paced by `pacer` exactly as a decode step is, and return
    /// the last forward's last-position logits and the number of forwards.
    ///
    /// The state machine steps over the prompt before it steps over its own
    /// generated tokens — same stepping, different input source. A unit is
    /// staged into the strategy's input (zero-copy on the ring arm when it
    /// fits the input tile, a fresh tensor otherwise), does not sample (the
    /// tokens are given) and produces no output; only the last unit's logits
    /// seed the first generated token. One unit over the whole prompt is the
    /// incumbent one-shot pass, so the default is byte-identical.
    ///
    /// Correctness across unit boundaries is the conv writer's left-context
    /// carry (`Config::conv_prefill_ctx`): a `seq_len > 1` pass onto a warm
    /// cache zero-pads its left without it. The caller turns it on whenever
    /// more than one forward can run; a single unit is identical either way.
    ///
    /// Each forward is a prefill window (`tuning::VerifyWindow::prefill`):
    /// under `deterministic` its half matmuls take the GEMM, so a prompt's
    /// bits are its tiling's, identical run to run for the same widths
    /// (`architecture.md` §2.3.5b).
    fn prefill_forwards<R>(
        &mut self,
        readback: &mut R,
        prompt: &[u32],
        opts: &DecodeOptions,
        pacer: &mut StepPacer,
        pace: bool,
    ) -> Result<(Option<Tensor>, usize)>
    where
        R: ReadbackStrategy,
    {
        let work = PrefillWork::resolve(opts, self.seq.pos, prompt.len(), pace)?;
        self.prefill_forwards_prepared(readback, prompt, &work, pacer)
    }

    /// Execute the resolved outer forwards without rescheduling. This seam
    /// carries no admission promise; ordinary callers resolve work locally.
    fn prefill_forwards_prepared<R>(
        &mut self,
        readback: &mut R,
        prompt: &[u32],
        work: &PrefillWork,
        pacer: &mut StepPacer,
    ) -> Result<(Option<Tensor>, usize)>
    where
        R: ReadbackStrategy,
    {
        if work.start() != self.seq.pos
            || work.end().checked_sub(work.start()) != Some(prompt.len())
        {
            lloom_tensor::bail!("prepared prefill work no longer matches the sequence");
        }
        let device = readback.device().clone();
        if let Some(depth) = work.active_depth() {
            pacer.set_depth(depth);
        }
        let pace = work.active_depth().is_some();
        let mut last_logits = None;
        let mut units = 0usize;
        let mut off = 0usize;
        for span in work.forwards() {
            let w = span.width;
            // Drains this unit's autoreleased encoders and command buffers
            // when the iteration ends (`AutoreleaseScope` docs).
            let _pool = lloom_tensor::AutoreleaseScope::new();
            let chunk = &prompt[off..off + w];
            let input = readback.stage_prompt_chunk(chunk)?;
            // The prompt's ids are host-side on every arm.
            self.model.note_input_ids(&mut self.seq.cache, chunk)?;
            if pace {
                pacer.before_forward(&device)?;
            }
            // The KV grow (`prepare_pass`) runs inside the model's forward,
            // before layer 0, so it is inside this bracket: a session
            // allocation the recorder excludes and no plan serves.
            // Session allocations (the KV slab, a bucket grow) happen here,
            // before the bracket, so the planned pass allocates nothing.
            self.model.prepare_pass(&mut self.seq.cache, w)?;
            begin_pass(
                &device,
                PassShape::Prefill {
                    w,
                    fresh: span.past == 0,
                },
            )?;
            let out = {
                let _prefill = lloom_tensor::tuning::VerifyWindow::prefill();
                self.model
                    .forward(&input, span.past, &mut self.seq.cache)?
                    .squeeze(0)?
            };
            end_pass(&device)?;
            if pace {
                pacer.after_forward(&device)?;
            }
            self.seq.pos += w;
            self.seq.tokens.extend_from_slice(chunk);
            last_logits = Some(out);
            off += w;
            units += 1;
        }
        Ok((last_logits, units))
    }

    /// Prefill `prompt` onto this sequence and produce nothing: the logical
    /// CHUNK seam. The cache and the token log advance by the prompt and the
    /// last position's logits are held, so a later [`Self::decode`] with an
    /// empty prompt resumes from them (`Session::prepare` reports `Resume`),
    /// and a later call with more of the sequence extends it. A scheduler
    /// that interleaves one request's prompt with other requests' decode
    /// feeds each request `≤ budget` prompt tokens per visit through this and
    /// visits the others between — the prompt's KV is built incrementally and
    /// the final state is what one pass over the prompt leaves, given the
    /// conv writer's left-context carry.
    ///
    /// The forwards taken are [`prefill_widths`]'s for `opts` (the tile bound
    /// applies here too), paced by their own [`StepPacer`]; nothing waits for
    /// the GPU at the end, so the next visit's work queues behind this one.
    /// Refused on an unreconciled overrun, as `decode` is. An empty prompt is
    /// a no-op that takes no forward.
    pub fn prefill<R>(
        &mut self,
        readback: &mut R,
        prompt: &[u32],
        opts: &DecodeOptions,
    ) -> Result<PrefillOut>
    where
        R: ReadbackStrategy,
    {
        if self.seq.overrun > 0 {
            lloom_tensor::bail!(
                "Session::prefill: the cache is {} positions past the sequence's last token \
                 (an earlier run's overrun); `Session::reconcile` or `Session::reset` first",
                self.seq.overrun
            );
        }
        if prompt.is_empty() {
            return Ok(PrefillOut::default());
        }
        // A failed forward may already mutate the cache. Previous logits no
        // longer describe it; failure requires caller recovery/reset, not a
        // stale prediction. This invalidates the result, not the transaction.
        self.seq.last_logits = None;
        let device = readback.device().clone();
        // The prefill's own in-flight bound when one is set, as `decode` runs
        // its prefill block at (`prefill_pacer_depth`); else the decode depth.
        let depth = if opts.prefill_pacer_depth > 0 {
            opts.prefill_pacer_depth
        } else {
            opts.pacer_depth()
        };
        let mut pacer = StepPacer::new(depth);
        let pace = readback.loop_commits();
        let start = std::time::Instant::now();
        let (last, units) = self.prefill_forwards(readback, prompt, opts, &mut pacer, pace)?;
        end_executor_step(&device)?;
        // The held logits are the resumable state: a decode with an empty
        // prompt continues from them, and a further prefill discards them.
        self.seq.last_logits = match last {
            Some(l) => Some(self.hold_logits(l)?),
            None => None,
        };
        Ok(PrefillOut {
            units,
            prefill_ns: start.elapsed().as_nanos() as u64,
            step_markers: pacer.taken,
            step_waits: pacer.waited,
        })
    }

    /// Seed from the resolved outer forwards, retaining the existing target
    /// forward's pass ownership (no new driver plan bracket). Each non-final
    /// probe is observed before the unit's pacing marker, so that marker also
    /// covers the drafter's queued reads. The last probe is returned unobserved
    /// with its logits: a pipeline may enqueue other lanes before reading any
    /// anchor. Producers own probe escapes; pipelined callers hold logits.
    ///
    /// `begin_at` follows the reused prefix. Stale probes are drained before
    /// counting drops; a dropped probe within this seed still refuses the run.
    /// Each resolved unit is a prefill window, as in ordinary prefill. The
    /// third result is the number of forwards (`DecodeStats::prefill_units`).
    /// This descriptor is an execution schedule, not an admission capability.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_streamed<S>(
        &mut self,
        spec: &mut S,
        seq: SeqId,
        prompt: &[u32],
        work: &PrefillWork,
        pacer: &mut StepPacer,
        device: &Device,
    ) -> Result<(Tensor, <S::Probe as TargetProbe>::Output, usize)>
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
        if prompt.is_empty()
            || work.start() != self.seq.pos
            || work.end() - work.start() != prompt.len()
        {
            lloom_tensor::bail!("speculative prefill work does not match the non-empty suffix")
        }
        if let Some(depth) = work.active_depth() {
            pacer.set_depth(depth);
        }
        spec.begin_at(seq, prompt, &self.seq.tokens)?;
        let _stale = self.model.take_probe(&mut self.seq.cache)?;
        let drops_before = self.model.probe_drops(&self.seq.cache);
        let mut off = 0usize;
        for (unit, span) in work.forwards().enumerate() {
            let _pool = lloom_tensor::AutoreleaseScope::new();
            let w = span.width;
            let chunk = &prompt[off..off + w];
            let input = Tensor::new(chunk, device)?.unsqueeze(0)?;
            if work.active_depth().is_some() {
                pacer.before_forward(device)?;
            }
            // Keep the incumbent model-owned/pool pass route. Composing this
            // through ordinary prefill would newly plan retained probes on
            // targets whose producer has not declared that escape.
            let logits = {
                let _prefill = lloom_tensor::tuning::VerifyWindow::prefill();
                self.model.forward(&input, span.past, &mut self.seq.cache)?
            };
            self.seq.pos += w;
            self.seq.tokens.extend_from_slice(chunk);
            off += w;
            let probe = self.model.take_probe(&mut self.seq.cache)?;
            let drops = self.model.probe_drops(&self.seq.cache);
            if drops != drops_before {
                lloom_tensor::bail!(
                    "Session: the target dropped {} probe set(s) during the prefill — a forward \
                     ran between two probes, so the drafter would be seeded from part of the \
                     prompt (design/docs/speculation.md §10.2w)",
                    drops - drops_before
                );
            }
            if off < prompt.len() {
                drop(logits);
                spec.observe(seq, &probe, &[])?;
                if work.active_depth().is_some() {
                    pacer.after_forward(device)?;
                }
            } else {
                if work.active_depth().is_some() {
                    pacer.after_forward(device)?;
                }
                return Ok((logits, probe, unit + 1));
            }
        }
        unreachable!("non-empty resolved work covers the suffix")
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
    /// streamed, the EOS is not. The accepted prefix itself ends at the EOS
    /// (`accept_prefix` takes `eos_ids`), so the cache keeps exactly the
    /// streamed tokens' positions, the drafter is keyed with no position past
    /// the EOS, and the run leaves no overrun — the conservative K=1 arm's
    /// shape, not the lookahead's one-past-EOS.
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
        _rewind: crate::profile::RewindProof<M::Profile>,
        spec: &mut S,
        readback: &mut R,
        sampler: &mut LogitsProcessor,
        prompt: &[u32],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        on_tokens: &mut dyn FnMut(&[u32]) -> Result<()>,
    ) -> Result<DecodeOut>
    where
        S: Speculator,
        M: Probes<S::Probe> + crate::profile::Shaped,
        R: ReadbackStrategy,
    {
        // The verify draws the target's token per window position with
        // this sampler whenever it does more than read the raw argmax
        // (`Probes::verify_and_probe_sampled`): a temperature on any arm, or
        // processors on any arm (each row against its own history,
        // `LogitsProcessor::window_draw_device`). Every draw is keyed by the
        // position of the token it draws, so a rejected row moves nothing.
        let sampled = sampler.draws_windows();
        // A proposer that drafts nothing runs the zero-draft window: the
        // anchor alone a round, which is a decode step on a target whose
        // verify runs that window. The device-resident K=1 body
        // (`Session::decode_until`) is that same window's own arm, the token
        // fed on the device instead of read back.
        if spec.max_draft() == 0 && !Probes::<S::Probe>::verifies_anchor_alone(self.model) {
            lloom_tensor::bail!(
                "Session::decode_speculative: the proposer drafts nothing and this target's \
                 verify needs a draft; its zero-draft window is the K=1 body \
                 (`Session::decode_until`)"
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
        // The draft's early stop on the drafter's confidence
        // (`speculative.p_min`, the driver's knob, §3.4.4); 0 is off.
        let p_min = crate::tuning::tuning().dspark.p_min as f32;
        let hot_thread = HotThread::claim();
        let mut stats = DecodeStats {
            step_ns: Vec::with_capacity(opts.max_tokens),
            hot_thread: hot_thread.report(),
            ..DecodeStats::default()
        };

        // Prefill in the shared resolved units, each unit's probe keyed
        // into the drafter as it lands (`prefill_streamed`); the last unit's
        // argmax is the first anchor — the token whose logits produced the
        // first draft — exactly as the `lfm2-dspark-verify` example takes
        // `logits.argmax` when it does not hold the last prompt token back.
        let pos_before = self.seq.pos;
        let prefill_start = std::time::Instant::now();
        let work = PrefillWork::resolve(opts, self.seq.pos, prompt.len(), true)?;
        let mut prefill_pacer = StepPacer::for_decode(opts);
        let (logits, prefill_probe, prefill_units) =
            self.prefill_streamed(spec, seq, prompt, &work, &mut prefill_pacer, &device)?;
        stats.prefill_ns = prefill_start.elapsed().as_nanos() as u64;
        stats.prefill_units = prefill_units;
        (stats.step_markers, stats.step_waits) = prefill_pacer.counters();

        // The prefill's step closes before its token is drawn, as the K=1
        // loop's does: the draw's allocations are not the prefill pass's.
        end_executor_step(&device)?;
        // The prefill's token: the argmax, or the draw for the token at
        // `seq.pos`, keyed exactly as the K=1 loop keys its first; a window
        // of the one row, whose input the prompt's last token is.
        let mut anchor_tok = if sampled {
            let last = [*prompt.last().expect("checked non-empty above")];
            let input = Tensor::new(&last, logits.device())?;
            let drawn = sampler
                .window_draw_device(&logits, self.seq.pos as u64, Some(&input))?
                .flatten_all()?
                .to_vec1::<u32>()?;
            let fed = sampler.window_fed(&drawn);
            sampler.commit_window(&fed)?;
            fed[0]
        } else {
            logits
                .squeeze(0)?
                .argmax(lloom_tensor::D::Minus1)?
                .to_scalar::<u32>()?
        };

        // The last unit's probe, with the prefill's own argmax as the accepted
        // tail (the block anchor) — the same state the first round would see
        // after a fully-accepted prompt.
        spec.observe(seq, &prefill_probe, std::slice::from_ref(&anchor_tok))?;
        end_executor_step(&device)?;
        // As for intermediate units, observe has queued the probe's last
        // reader. Release these seed-only views before later rounds; native
        // custody still follows the queued uses, not these logical drops.
        drop(prefill_probe);
        drop(logits);
        // Dropping seed receipts adds no completion wait.
        #[cfg(feature = "metal")]
        drop(prefill_pacer);

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
        } else if !opts.reached(produced) {
            on_tokens(std::slice::from_ref(&anchor_tok))?;
            all_ids.push(anchor_tok);
            produced += 1;
        }

        // The verify window is `[anchor, draft…]` at this position; `pos` advances
        // by the accepted length each round (`lfm2-dspark-verify`'s `index_pos`).
        let mut since_commit = 0usize;

        let stopped = |opts: &DecodeOptions| {
            opts.stop
                .as_ref()
                .map(|f| f.load(std::sync::atomic::Ordering::Acquire))
                .unwrap_or(false)
        };
        while !eos && !opts.reached(produced) && !stopped(opts) {
            let _pool = lloom_tensor::AutoreleaseScope::new(); // per window, see the decode loop
            stats.steps += 1;
            let step_start = std::time::Instant::now();

            // One speculative round: draft → verify (accept the longest prefix
            // the target's token agrees with + its own at the first
            // disagreement, roll back) → observe (re-key the drafter from the
            // accepted rows). `p_min` cuts the draft at the first position
            // the drafter is less confident of (`speculative.cpp:796`; 0 is
            // off). The round returns the accepted tokens, the target's own
            // included, and has advanced the drafter's anchor via `observe`.
            let anchor = Anchor {
                last: anchor_tok,
                pos: self.seq.pos,
            };
            // The census is one round's worth of allocation; take it once,
            // past the warm-up rounds, so the breakdown carries which kernels
            // each phase dispatches and not only how many.
            let mut probe = RoundProbe {
                sync: if opts.sample_split {
                    Some(&device)
                } else {
                    None
                },
                census: false,
            };
            // This automatic enrichment must not turn host/scalar sample_split
            // into an unsupported explicit detailed-census request.
            probe.census = opts.sample_split
                && stats.steps == CENSUS_ROUND
                && probe.detailed_recorder_active();
            // The verify is a planned pass of `K + 1` positions
            // (`DESIGN.md` §9.2aa): the drafter's own forwards and the
            // observe are OUTSIDE it — they run on the drafter's model and
            // its own caches, and `speculative_round` brackets them for the
            // timing, not for the plan. `Verify { k }` is keyed on the
            // window the round actually proposes, which shrinks when a
            // confidence cut or the drafter's own bound truncates a draft,
            // so a run takes one key per distinct window (at most `K`).
            // The verify pass is planned (`DESIGN.md` §9.2aa): its window
            // is `K + 1` positions — the drafter's block plus the target's
            // bonus — and a round that drafts fewer (a confidence cut, a
            // drafter's own bound) is a different sequence and a different
            // key. The draft and the observe are outside the bracket: they
            // run on the drafter's model and its own caches.
            // The round keeps nothing past the budget (`fit_to_room`): a
            // cache fed past the stream is an overrun the next request's
            // reuse may not be able to take back.
            let room = opts.remaining(produced);
            let floor = usize::from(!Probes::<S::Probe>::verifies_anchor_alone(self.model));
            let k = (spec.max_draft().min(room.saturating_sub(1).max(floor)) + 1)
                .min(u16::MAX as usize) as u16;
            let (accepted, timing) = self.speculative_round_keyed(
                spec,
                &[seq],
                &[anchor],
                p_min,
                eos_ids,
                probe,
                Some((&device, PassShape::Verify { k: k as usize })),
                room,
                sampled.then_some(&mut *sampler),
            )?;
            let accepted = accepted.into_iter().next().ok_or_else(|| {
                lloom_tensor::Error::msg("decode_speculative: verify returned no sequence")
            })?;
            stats.rounds.push(RoundStats {
                accepted: accepted.len(),
                timing,
                start_ns: (step_start - prefill_start).as_nanos() as u64,
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
                // A minimum-width verify may predict beyond the output
                // ceiling. Its unpublished bonus cannot change the stop.
                if opts.reached(produced + batch.len()) {
                    break;
                }
                if eos_ids.contains(&id) {
                    eos = true;
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

            end_executor_step(&device)?;
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
        stats.drafter_ctx = spec.context_len(seq).map(|n| (n, self.seq.pos));
        self.model.settle_pending(&mut self.seq.cache)?;

        stats.produced = produced;
        stats.readback = readback.counters();
        Ok(DecodeOut {
            ids: all_ids,
            eos,
            stats,
        })
    }

    /// [`Self::decode_speculative`] over several sequences at once, their
    /// rounds software-pipelined so one sequence's verify runs on the GPU while
    /// another's draft and observe are encoded
    /// (`design/docs/speculative-pipeline-scheduler-design.md` §4a, §10).
    ///
    /// # What overlaps, and what cannot
    ///
    /// Within one sequence the round is dependency-locked: the next draft reads
    /// the context `observe` rebuilds from this verify's taps. Across sequences
    /// there is no such edge — each has its own cache and its own drafter slot
    /// — so the loop keeps one verify per live sequence enqueued and services
    /// them in turn: complete `i`'s verify, observe, publish, draft and enqueue
    /// `i`'s next verify, then move to `i + 1`, whose verify has meanwhile been
    /// running. The GPU is fed for as long as the host's turnaround on one
    /// sequence is shorter than the others' queued work.
    ///
    /// The readback that ends a verify waits on that verify's own command
    /// buffer ([`Probes::verify_complete`]), not on the queue's tail; a
    /// tail wait would drain the other sequences' verifies and serialise the
    /// whole thing back to the solo round.
    ///
    /// # Correctness
    ///
    /// Each sequence's stream is byte-identical to `decode_speculative` run
    /// solo — the drafter's `PerSeq` slots and the target caches are disjoint,
    /// and the accept rule is the same `accept_prefix`. The per-sequence
    /// bookkeeping (`pos`, `tokens`, `overrun`, EOS mid-block, the `max_tokens`
    /// cut) is the serial loop's. `sessions[i]` is driven with drafter slot
    /// `SeqId(i)` and prompt `prompts[i]`; `on_tokens` receives `(i, ids)`.
    ///
    /// Refused: fewer drafter slots than sessions, an empty prompt, a session
    /// with an unreconciled overrun, a proposer with `max_draft() == 0`.
    /// `opts.sample_split` has no per-phase meaning here and is ignored.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_speculative_pipelined<S>(
        sessions: &mut [Self],
        rewind: crate::profile::RewindProof<M::Profile>,
        spec: &mut S,
        prompts: &[&[u32]],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        device: &Device,
        on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
    ) -> Result<Vec<DecodeOut>>
    where
        S: Speculator,
        M: Probes<S::Probe> + crate::profile::Shaped,
    {
        let n = sessions.len();
        Self::decode_speculative_pipelined_limits(
            sessions,
            rewind,
            spec,
            prompts,
            opts,
            std::iter::repeat_n(opts.max_tokens, n),
            eos_ids,
            device,
            on_tokens,
        )
    }

    /// The pipelined driver with each lane's own output ceiling (`0` is
    /// unbounded until EOS). `max_tokens[i]` belongs to `sessions[i]` and
    /// overrides `opts.max_tokens`; the other options remain shared.
    ///
    /// Shape errors are refused before any session or drafter state moves.
    /// The uniform-cap entry point and this one execute the same decode loop.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_speculative_pipelined_with_caps<S>(
        sessions: &mut [Self],
        rewind: crate::profile::RewindProof<M::Profile>,
        spec: &mut S,
        prompts: &[&[u32]],
        opts: &DecodeOptions,
        max_tokens: &[usize],
        eos_ids: &[u32],
        device: &Device,
        on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
    ) -> Result<Vec<DecodeOut>>
    where
        S: Speculator,
        M: Probes<S::Probe> + crate::profile::Shaped,
    {
        Self::decode_speculative_pipelined_limits(
            sessions,
            rewind,
            spec,
            prompts,
            opts,
            max_tokens.iter().copied(),
            eos_ids,
            device,
            on_tokens,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_speculative_pipelined_limits<S>(
        sessions: &mut [Self],
        _rewind: crate::profile::RewindProof<M::Profile>,
        spec: &mut S,
        prompts: &[&[u32]],
        opts: &DecodeOptions,
        max_tokens: impl ExactSizeIterator<Item = usize>,
        eos_ids: &[u32],
        device: &Device,
        on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
    ) -> Result<Vec<DecodeOut>>
    where
        S: Speculator,
        M: Probes<S::Probe> + crate::profile::Shaped,
    {
        let n = sessions.len();
        if n == 0 {
            lloom_tensor::bail!("decode_speculative_pipelined: no sessions");
        }
        if prompts.len() != n {
            lloom_tensor::bail!(
                "decode_speculative_pipelined: {} sessions and {} prompts",
                n,
                prompts.len()
            );
        }
        if max_tokens.len() != n {
            lloom_tensor::bail!(
                "decode_speculative_pipelined: {} sessions and {} row caps",
                n,
                max_tokens.len()
            );
        }
        if spec.max_draft() == 0 {
            lloom_tensor::bail!(
                "decode_speculative_pipelined: the proposer has max_draft 0 (the K=1 identity); \
                 it belongs on `Session::decode`"
            );
        }
        if spec.n_seq() < n {
            lloom_tensor::bail!(
                "decode_speculative_pipelined: the proposer has {} sequence slots for {} sessions",
                spec.n_seq(),
                n
            );
        }
        if eos_ids.is_empty() {
            lloom_tensor::bail!(
                "decode_speculative_pipelined: eos_ids is empty, so the loop would never terminate"
            );
        }
        for (i, s) in sessions.iter().enumerate() {
            if prompts[i].is_empty() {
                lloom_tensor::bail!(
                    "decode_speculative_pipelined: sequence {i} has an empty prompt"
                );
            }
            if s.seq.overrun > 0 {
                lloom_tensor::bail!(
                    "decode_speculative_pipelined: sequence {i}'s cache is {} positions past its \
                     last token (an earlier run's overrun); `Session::reconcile` or \
                     `Session::reset` first",
                    s.seq.overrun
                );
            }
        }
        spec.want_confidence(false);
        let run_start = std::time::Instant::now();
        let hot_thread = HotThread::claim();

        let mut lanes: Vec<Lane<<M as Probes<S::Probe>>::Pending>> = Vec::with_capacity(n);

        // Prefill every sequence before reading any back, so the GPU has all
        // the prompts queued while the host waits on the first. Each prefill
        // runs in the shared resolved units and seeds its drafter slot
        // unit by unit (`prefill_streamed`); only the last unit's probe waits
        // here for its anchor.
        let mut logits = Vec::with_capacity(n);
        let mut prefill_pacer = StepPacer::for_decode(opts);
        for (i, (s, cap)) in sessions.iter_mut().zip(max_tokens).enumerate() {
            let limit = TokenLimit(cap);
            s.seq.last_logits = None;
            let pos_before = s.seq.pos;
            let prefill_start = std::time::Instant::now();
            let work = PrefillWork::resolve(opts, s.seq.pos, prompts[i].len(), true)?;
            let before = prefill_pacer.counters();
            let (l, probe, prefill_units) = s.prefill_streamed(
                spec,
                SeqId(i),
                prompts[i],
                &work,
                &mut prefill_pacer,
                device,
            )?;
            let after = prefill_pacer.counters();
            // Read after the other sequences' prefills: held outside them.
            logits.push((s.hold_logits(l)?, probe));
            lanes.push(Lane {
                limit,
                anchor_tok: 0,
                produced: 0,
                eos: false,
                all_ids: Vec::with_capacity(cap),
                pos_before,
                prompt_len: prompts[i].len(),
                stats: DecodeStats {
                    step_ns: Vec::with_capacity(cap),
                    prefill_ns: prefill_start.elapsed().as_nanos() as u64,
                    prefill_units,
                    step_markers: after.0 - before.0,
                    step_waits: after.1 - before.1,
                    hot_thread: hot_thread.report(),
                    ..DecodeStats::default()
                },
                pending: None,
                in_flight: RoundTiming::default(),
                round_start_ns: 0,
                done: false,
            });
        }

        // Anchors and the last unit's seed, as the serial loop: the prefill's
        // argmax is token 0 and the block anchor; the earlier units were keyed
        // as they landed (`design/docs/speculation.md` §10.2w).
        for (i, (l, prefill_probe)) in logits.into_iter().enumerate() {
            let lane = &mut lanes[i];
            let anchor_tok = l
                .squeeze(0)?
                .argmax(lloom_tensor::D::Minus1)?
                .to_scalar::<u32>()?;
            spec.observe(SeqId(i), &prefill_probe, std::slice::from_ref(&anchor_tok))?;
            lane.anchor_tok = anchor_tok;
            if eos_ids.contains(&anchor_tok) {
                lane.eos = true;
            } else if !lane.limit.reached(lane.produced) {
                on_tokens(i, std::slice::from_ref(&anchor_tok))?;
                lane.all_ids.push(anchor_tok);
                lane.produced += 1;
            }
            lane.done = lane.eos || lane.limit.reached(lane.produced);
        }

        // All anchors have consumed their seed logits; the round pipeline has
        // its own pending receipts. No terminal drain is introduced here.
        #[cfg(feature = "metal")]
        drop(prefill_pacer);

        // Prime: one verify in flight per live sequence.
        for i in 0..n {
            if !lanes[i].done {
                let room = lanes[i].limit.remaining(lanes[i].produced);
                Self::pipeline_enqueue(
                    &mut sessions[i],
                    &mut lanes[i],
                    spec,
                    i,
                    run_start,
                    device,
                    room,
                )?;
            }
        }

        // Steady state: service the in-flight verifies round-robin. Completing
        // lane `i` waits on its own buffer only; its next verify is enqueued
        // behind the others' before the loop moves on.
        loop {
            let _pool = lloom_tensor::AutoreleaseScope::new(); // per round, see the decode loop
            let mut progressed = false;
            for i in 0..n {
                let Some(pending) = lanes[i].pending.take() else {
                    continue;
                };
                progressed = true;
                let lane = &mut lanes[i];
                let s = &mut sessions[i];
                let mut timing = std::mem::take(&mut lane.in_flight);

                let t = std::time::Instant::now();
                let rb = readbacks();
                let (accepted, probe_out) =
                    s.model
                        .verify_complete(pending, &mut s.seq.cache, eos_ids)?;
                timing.verify_wait_ns = t.elapsed().as_nanos() as u64;
                timing.verify_ns = timing.verify_enqueue_ns + timing.verify_wait_ns;
                timing.verify_readbacks += readbacks() - rb;

                let t = std::time::Instant::now();
                let rb = readbacks();
                spec.observe(SeqId(i), &probe_out, &accepted)?;
                timing.observe_ns = t.elapsed().as_nanos() as u64;
                timing.observe_readbacks = readbacks() - rb;

                if accepted.is_empty() {
                    lloom_tensor::bail!(
                        "decode_speculative_pipelined: a verify round accepted zero tokens; the \
                         target's own prediction at position 0 is always correct, so this is a \
                         broken verify"
                    );
                }
                lane.stats.steps += 1;
                lane.stats.rounds.push(RoundStats {
                    accepted: accepted.len(),
                    timing,
                    start_ns: lane.round_start_ns,
                });
                s.seq.pos += accepted.len();
                lane.anchor_tok = *accepted.last().unwrap();

                let mut batch: Vec<u32> = Vec::with_capacity(accepted.len());
                for &id in &accepted {
                    // A minimum-width verify may predict beyond the output
                    // ceiling. Its unpublished bonus cannot change the stop.
                    if lane.limit.reached(lane.produced + batch.len()) {
                        break;
                    }
                    if eos_ids.contains(&id) {
                        lane.eos = true;
                        break;
                    }
                    batch.push(id);
                }
                if !batch.is_empty() {
                    on_tokens(i, &batch)?;
                    lane.all_ids.extend_from_slice(&batch);
                    lane.produced += batch.len();
                }
                lane.stats.step_ns.push(
                    (run_start.elapsed().as_nanos() as u64).saturating_sub(lane.round_start_ns),
                );
                lane.done = lane.eos || lane.limit.reached(lane.produced);
                if !lane.done {
                    let room = lane.limit.remaining(lane.produced);
                    Self::pipeline_enqueue(s, lane, spec, i, run_start, device, room)?;
                }
            }
            if !progressed {
                break;
            }
        }

        // Per-sequence bookkeeping, as the serial loop: the fed ids are the
        // stream's prefix of the cache's length, anything past it is overrun.
        let mut outs = Vec::with_capacity(n);
        for (i, lane) in lanes.into_iter().enumerate() {
            let s = &mut sessions[i];
            let fed = s.seq.pos - lane.pos_before - lane.prompt_len;
            let in_cache = fed.min(lane.produced);
            s.seq.tokens.extend_from_slice(&lane.all_ids[..in_cache]);
            let overrun = fed - in_cache;
            s.seq.overrun = overrun;
            let mut stats = lane.stats;
            stats.overrun = overrun;
            stats.produced = lane.produced;
            stats.drafter_ctx = spec.context_len(SeqId(i)).map(|n| (n, s.seq.pos));
            outs.push(DecodeOut {
                ids: lane.all_ids,
                eos: lane.eos,
                stats,
            });
        }
        Ok(outs)
    }

    /// Draft sequence `i` and enqueue its verify, leaving the receipt in the
    /// lane; the draft fitted to `room`, the tokens the lane may still
    /// publish (`fit_to_room`). The draft is one drafter forward; nothing
    /// here waits for the GPU.
    #[allow(clippy::too_many_arguments)]
    fn pipeline_enqueue<S>(
        s: &mut Self,
        lane: &mut Lane<<M as Probes<S::Probe>>::Pending>,
        spec: &mut S,
        i: usize,
        run_start: std::time::Instant,
        device: &Device,
        room: usize,
    ) -> Result<()>
    where
        S: Speculator,
        M: Probes<S::Probe>,
    {
        let anchor = Anchor {
            last: lane.anchor_tok,
            pos: s.seq.pos,
        };
        lane.round_start_ns = run_start.elapsed().as_nanos() as u64;
        let mut timing = RoundTiming::default();
        let t = std::time::Instant::now();
        let rb = readbacks();
        let drafts = spec.draft(&[SeqId(i)])?;
        let mut draft = drafts.into_iter().next().ok_or_else(|| {
            lloom_tensor::Error::msg("decode_speculative_pipelined: the drafter returned no block")
        })?;
        let floor = usize::from(!Probes::<S::Probe>::verifies_anchor_alone(s.model));
        crate::nn::speculator::fit_to_room(&mut draft, room, floor)?;
        timing.draft_ns = t.elapsed().as_nanos() as u64;
        timing.draft_readbacks = readbacks() - rb;

        let t = std::time::Instant::now();
        let rb = readbacks();
        // The planned pass is the ENCODE (`DESIGN.md` §9.2aa): a plan serves
        // allocations, and `verify_enqueue` is where this round's are made.
        // `verify_complete` reads a marker back and allocates nothing of the
        // pass, and this driver interleaves OTHER sequences' rounds between
        // the two halves — so a bracket spanning them would put another
        // sequence's passes inside this one's, which is exactly the drift
        // the key exists to refuse.
        let k = (draft.tokens.len() + 1).min(u16::MAX as usize);
        s.model.prepare_verify(&mut s.seq.cache, k)?;
        begin_pass(device, PassShape::Verify { k })?;
        let pending = s.model.verify_enqueue(&draft, &anchor, &mut s.seq.cache);
        end_pass(device)?;
        let pending = pending?;
        timing.verify_enqueue_ns = t.elapsed().as_nanos() as u64;
        timing.verify_readbacks = readbacks() - rb;
        lane.in_flight = timing;
        lane.pending = Some(pending);
        Ok(())
    }

    /// Decode `sessions.len()` independent sequences together, one token for
    /// every live sequence per forward pass — the batched-forward body
    /// (issue #694, `design/docs/batched-decode-design.md` §10).
    ///
    /// # What a step is
    ///
    /// Every live row's next token is sampled on the device from the `[B,
    /// vocab]` logits in one argmax, and the `[B, 1]` ids are fed to
    /// [`BatchedCausalLM::forward_rows`], which applies each weight matrix to
    /// all `B` rows as one `m = B` GEMM and everything per sequence — RoPE,
    /// the KV append and read, the conv window — per row against that row's
    /// own cache. So the 5.394 GB weight sweep is paid once per step for `B`
    /// tokens (§13.5), and a row computes what it would compute alone: its
    /// stream does not depend on its batch neighbours (§2.3.3 #7).
    ///
    /// The rows are independent sequences: any prompts, any lengths, each
    /// prefilled through its own `forward` before the loop. A row leaves the
    /// batch at its EOS or at `opts.max_tokens`; the step shrinks to the rows
    /// still live and the run ends when none is.
    ///
    /// # Where the host waits — `opts.lookahead`
    ///
    /// The one thing the host must learn per step is which rows ended.
    /// `lookahead = 0` reads the `B` ids back before it encodes the next
    /// forward (the GPU idles for the round trip, no row runs past its end).
    /// `lookahead = 1` feeds the device-resident ids straight into the next
    /// forward, encodes it, and only then reads the ids back through a staged
    /// read that waits on the argmax's own command buffer (#685's
    /// `Tensor::stage_read`) — so the next step is on the GPU while the host
    /// learns this one's EOS, at the price [`Session::decode`]'s own lookahead
    /// pays: a row that ended has had one more forward run for it, reported as
    /// its [`DecodeStats::overrun`]. A row that ends on `max_tokens` under
    /// `lookahead = 1` keeps that forward's logits as its resumable
    /// [`Sequence`](crate::nn::speculator::Sequence) state.
    ///
    /// Argmax only: the ids are sampled on the device for the whole batch, and
    /// a per-row temperature sample would need a per-row noise stream the
    /// host `StdRng` does not give (§2.3.3 #7's counter-based RNG is not
    /// built). Refused rather than approximated, as the ring refuses top-p.
    ///
    /// `on_tokens` receives `(row, ids)` as each row's token is confirmed.
    /// `opts.budget` and `opts.sample_split` have no meaning here: the
    /// lookahead arm commits once per step at its staged read, the blocking
    /// arm's readback drains the queue every step.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_batched(
        sessions: &mut [Self],
        sampler: &mut LogitsProcessor,
        prompts: &[&[u32]],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        device: &Device,
        on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
    ) -> Result<Vec<DecodeOut>>
    where
        M: BatchedCausalLM,
    {
        let arrivals = vec![0u64; sessions.len()];
        Self::decode_batched_admitted(
            sessions, sampler, prompts, &arrivals, opts, eos_ids, device, on_tokens,
        )
    }

    /// [`Self::decode_batched`] with each row ARRIVING at a step of the run:
    /// `arrivals[i]` is the first step (counting from 0) at which row `i`'s
    /// prompt may be fed. A row whose arrival is `0` is admitted before the
    /// first step, which is `decode_batched`'s shape; a later arrival joins a
    /// batch that is already decoding — the serving shape, where a request
    /// lands while others are mid-reply.
    ///
    /// # How a prompt is fed — `opts.prefill_chunk`, the CHUNK
    ///
    /// `prefill_chunk == 0` feeds an arrived row's whole prompt at its
    /// arrival step, in one forward (or in `opts.prefill_tile` forwards when
    /// the tile is set), before that step's batched decode is encoded: every
    /// live row's next token waits for the whole prompt. `prefill_chunk > 0`
    /// feeds at most that many prompt tokens per step, over the arrived rows
    /// still pending in row order, each piece bounded by the tile — **as
    /// extra rows of the step's own batched forward**
    /// ([`BatchedCausalLM::forward_ragged`]): the live rows' next tokens and
    /// the pieces are one `[1, T]` input, so the pieces ride the step's one
    /// weight sweep and add no forward, no sweep and no per-forward cost of
    /// their own (vLLM's chunked prefill: a scheduling of tokens into the
    /// batch, not a second pass). A row joins the batch at the step after its
    /// last piece, with that piece's logits. The prompt's KV is built
    /// incrementally, which is exact given the conv writer's left-context
    /// carry (`Config::conv_prefill_ctx`).
    ///
    /// A step with no live row and no arrived prompt to feed (every pending
    /// row's arrival is in the future) is an idle step: nothing is encoded,
    /// and the step count advances so the arrival can come due. The run ends
    /// when no row is live and none is pending.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_batched_admitted(
        sessions: &mut [Self],
        sampler: &mut LogitsProcessor,
        prompts: &[&[u32]],
        arrivals: &[u64],
        opts: &DecodeOptions,
        eos_ids: &[u32],
        device: &Device,
        on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
    ) -> Result<Vec<DecodeOut>>
    where
        M: BatchedCausalLM,
    {
        let n = sessions.len();
        if n == 0 {
            lloom_tensor::bail!("decode_batched: no sessions");
        }
        if prompts.len() != n {
            lloom_tensor::bail!(
                "decode_batched: {} sessions and {} prompts",
                n,
                prompts.len()
            );
        }
        if arrivals.len() != n {
            lloom_tensor::bail!(
                "decode_batched: {} sessions and {} arrivals",
                n,
                arrivals.len()
            );
        }
        if sampler.sampling() != &Sampling::ArgMax {
            lloom_tensor::bail!(
                "decode_batched: the batch is sampled on the device in one argmax; {:?} needs a \
                 per-row host sample this body does not take (admit rows with their own \
                 samplers through `BatchDecoder` for that)",
                sampler.sampling()
            );
        }
        // The closed batch is the open decoder with every row admitted up
        // front at its arrival step, stepped until nothing is live or pending
        // (`nn::batched::BatchDecoder`). A slot index is the row's index.
        let mut decoder: BatchDecoder<'m, M, &mut Self> =
            BatchDecoder::new(opts.clone(), eos_ids, device.clone())?;
        let mut ids = Vec::with_capacity(n);
        for (i, s) in sessions.iter_mut().enumerate() {
            let id = decoder
                .admit_at(s, prompts[i].to_vec(), Sampling::ArgMax, 0, arrivals[i])
                .map_err(|e| lloom_tensor::Error::msg(format!("sequence {i}: {e}")))?;
            debug_assert_eq!(id.slot(), i);
            ids.push(id);
        }
        while !decoder.is_idle() {
            decoder.step(&mut |row, toks| on_tokens(row.slot(), toks))?;
        }
        let mut outs: Vec<Option<DecodeOut>> = (0..n).map(|_| None).collect();
        for (id, _session, out) in decoder.take_finished() {
            outs[id.slot()] = Some(out);
        }
        outs.into_iter()
            .enumerate()
            .map(|(i, o)| {
                o.ok_or_else(|| {
                    lloom_tensor::Error::msg(format!("decode_batched: row {i} never retired"))
                })
            })
            .collect()
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
    /// the same shape bucket. The buckets are the row tiles `select_tile_config`
    /// picks by `m` (§13.5o: 16 rows to `m` = 32, 64 rows above; `m == 1` is
    /// the GEMV), each in an aligned (`m % bm == 0`) and an unaligned variant —
    /// four pipelines a prefill can land on, none of which a decode step
    /// touches. [`WARMUP_PREFILL_LENS`] is one prefill per variant, each a
    /// prefill window as a real one is, then the decode step is the GEMV:
    /// the set every real run actually takes.
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
        // One prefill per GEMM pipeline variant a prompt can select (each row
        // tile, aligned and unaligned), then one decode step for the GEMV. A
        // variant left cold here is compiled inside the first prompt's own timed
        // prefill (~2 ms, §13.1i), a fresh cost every process since the
        // pipeline cache does not outlive one.
        let mut pos = 0;
        for &len in &WARMUP_PREFILL_LENS {
            let tokens = vec![0u32; len];
            let prefill = Tensor::new(tokens.as_slice(), device)?.unsqueeze(0)?;
            let _prefill = lloom_tensor::tuning::VerifyWindow::prefill();
            let _ = self.model.forward(&prefill, pos, &mut self.seq.cache)?;
            pos += len;
        }

        // **Drain the prefills before timing the decode step.** They are still
        // in flight here — the forwards above return once submitted — so a
        // `synchronize` after the decode step waits out ALL FIVE, and the
        // "GPU step" it measures is the whole warmup's tail rather than one
        // decode step's. Measured on this machine: 144 ms against a real
        // steady step of 15.9 ms, a 9x overstatement, which drives the ratio
        // `2·host/gpu` to zero and pins the depth at the floor for exactly the
        // configs the calibration exists to serve. Same defect class as the
        // host-inclusive `gpu_ns` above it — a correct formula fed a wrong
        // operand — one layer out, and invisible for the same reason: on a
        // GPU-bound config the floor is right either way (measurements §10).
        device.synchronize()?;

        // Measure the host/GPU balance from a real decode step, to compute the
        // in-flight depth (`compute_in_flight_depth`) instead of a static cap.
        // `host_ns` = the CPU time to encode+submit the step; `gpu_ns` = that
        // step's GPU tail alone. Their ratio decides how far the host must run
        // ahead to keep the GPU fed.
        let decode = Tensor::new(&[0u32], device)?.unsqueeze(0)?;
        let t_host = std::time::Instant::now();
        let _ = self.model.forward(&decode, pos, &mut self.seq.cache)?;
        // Host submit is done here (the forward returns before the GPU finishes).
        let host_ns = t_host.elapsed().as_nanos() as u64;

        // Force the driver to finish compiling before the clock the caller is
        // about to stop; a cold pipeline creation left in flight would surface
        // on the first real forward instead. This sync also completes the decode
        // step's GPU tail, so the elapsed-from-submit is the GPU step time.
        let t_gpu = std::time::Instant::now();
        device.synchronize()?;
        // `gpu_ns` = the GPU TAIL alone — the time from submit-done to the
        // step being finished — and **not** `host_ns + tail`.
        //
        // It was `host_ns + tail`, and that made the whole calibration dead
        // code: the ratio is `2·host / gpu`, so putting `host` into `gpu`'s
        // own definition bounds it at `2·host / (host + tail) <= 2` for every
        // `tail >= 0`, which `ceil` then floors back to `IN_FLIGHT_FLOOR`.
        // Every config got 2 — including the ones this function replaced a
        // constant 2 to serve (a small model, a faster GPU, any config where
        // the host encode approaches the GPU step). The doc's own worked
        // examples ("ratio 0.75 -> depth ~3-4") were arithmetically
        // unreachable. `compute_in_flight_depth_is_the_floor_only_when_the_gpu_dwarfs_the_host`
        // pins it.
        let gpu_ns = t_gpu.elapsed().as_nanos() as u64;
        self.warm_in_flight_depth = warm_depth(host_ns, gpu_ns, in_flight_mem_cap(device));
        self.warm_balance_ns = (host_ns, gpu_ns);

        // Leave nothing behind: the warmup wrote KV, conv state and the position
        // counter, and the next call is the user's real prefill from position 0.
        self.reset();

        Ok(cache_sizes(device))
    }

    /// The in-flight depth warmup measured (`0` if warmup has not run). A caller
    /// threads this into `DecodeOptions::in_flight_depth` so the pacer uses the
    /// computed host/GPU balance rather than the floor.
    pub fn measured_in_flight_depth(&self) -> usize {
        self.warm_in_flight_depth
    }

    /// The `(host_ns, gpu_ns)` warmup measured, `(0, 0)` before it runs — so a
    /// test (or a harness) can check the two are DISJOINT rather than trusting
    /// the call site to have kept them so.
    pub fn measured_balance_ns(&self) -> (u64, u64) {
        self.warm_balance_ns
    }
}

/// The speculative round whose per-phase dispatch census is kept under
/// `DecodeOptions::sample_split`: past the first rounds' pipeline warm-up,
/// early enough that a short run still reaches it.
const CENSUS_ROUND: u64 = 6;

/// The prefill lengths [`Session::warmup`] runs, one per GEMM pipeline variant
/// a prompt can select: the 16-row tile unaligned and aligned (`m` = 8, 16)
/// and the 64-row tile unaligned and aligned (40, 64) — `select_tile_config`
/// picks the row tile by `m` (§13.5o) and `align_m = m % bm` keys a second
/// pipeline per tile. A length here that stopped matching a variant would leave
/// that variant cold on the first prompt that selects it, the ~2 ms §13.1i
/// prices a pipeline creation at (measured 40 ms of `prefill_ns` at `m` = 64
/// before this list carried the aligned 64-row variant, §13.5o).
pub const WARMUP_PREFILL_LENS: [usize; 4] = [8, 16, 40, 64];

/// The forward widths a prefill of `prompt_len` tokens is fed through, in order.
///
/// This is the one place the prefill's forward shape is decided, so the
/// incumbent ramp and the explicit `--spark` schedule cannot drift.
///
/// * `schedule = None` — the **incumbent** `prefill_ubatch` ramp, reproduced
///   byte-for-byte: `ubatch == 0` is one-shot (`[prompt_len]`); a positive
///   `ubatch` is the spark then the whole remainder (`[ubatch, prompt_len −
///   ubatch]`), clamped so a spark wider than the prompt is a single forward.
/// * `schedule = Some(units)` — feed forwards of `units[0], units[1], …` (the
///   **ramp**), each clamped to what is left; then cover the remainder with
///   **`bulk`-wide forwards** (the bulk tile), and only the final `< bulk`
///   tokens as a `seq_len == 1` tail. `bulk == 0` (or `bulk == 1`) means the
///   whole remainder is the `seq_len == 1` tail — the pre-bulk behaviour.
///
///   The bulk tile is the fix for the many-tiny-forwards pathology: a ramp with
///   a `1`-token tail over a long prompt runs one full weight sweep per leftover
///   token (a 400-token prompt = ~400 forwards). Chopping the remainder at an
///   *aligned* bulk width (16 / 32 / 64) instead keeps every bulk forward on the
///   no-guard GEMM tile and collapses the forward count to `remainder / bulk`.
///
/// The TILE bound ([`tile_widths`]) is applied over the result by the callers.
///
/// Total is always exactly `prompt_len` (each width ≤ the remainder, and the
/// bulk + tail close any gap), so the caller's `off` walks the prompt exactly
/// once. An empty prompt yields no forwards.
fn prefill_widths(
    prompt_len: usize,
    ubatch: usize,
    schedule: &Option<Vec<usize>>,
    bulk: usize,
) -> Vec<usize> {
    if prompt_len == 0 {
        return Vec::new();
    }
    match schedule {
        None => {
            // The incumbent ramp: one-shot, or spark + remainder.
            if ubatch == 0 || ubatch >= prompt_len {
                vec![prompt_len]
            } else {
                vec![ubatch, prompt_len - ubatch]
            }
        }
        Some(units) => {
            let mut widths = Vec::new();
            let mut left = prompt_len;
            for &u in units {
                if left == 0 {
                    break;
                }
                let w = u.min(left);
                if w == 0 {
                    // A `0` in the schedule is a no-op unit, not a stall.
                    continue;
                }
                widths.push(w);
                left -= w;
            }
            // The remainder rides the bulk tile: `bulk`-wide forwards (each on
            // an aligned GEMM variant), then the final `< bulk` tokens one at a
            // time. `bulk <= 1` degenerates to the all-`seq_len==1` tail.
            let bulk = bulk.max(1);
            while left >= bulk && bulk > 1 {
                widths.push(bulk);
                left -= bulk;
            }
            // Whatever is left (`< bulk`, or all of it when `bulk <= 1`) as a
            // `seq_len == 1` tail — no arbitrary-width forward sneaks in.
            for _ in 0..left {
                widths.push(1);
            }
            widths
        }
    }
}

/// The TILE bound over [`prefill_widths`]'s result: every width above `tile`
/// is split into sequential forwards of `tile` tokens and a narrower last
/// one, so no forward is wider than the tile. `0` leaves the widths as they
/// are. The total is unchanged.
pub(crate) fn tile_widths(widths: Vec<usize>, tile: usize) -> Vec<usize> {
    if tile == 0 {
        return widths;
    }
    let mut tiled = Vec::with_capacity(widths.len());
    for w in widths {
        let mut left = w;
        while left > 0 {
            let t = left.min(tile);
            tiled.push(t);
            left -= t;
        }
    }
    tiled
}

#[cfg(test)]
mod in_flight_depth_tests {
    use super::{compute_in_flight_depth, warm_depth, StepPacer, IN_FLIGHT_FLOOR};

    /// The in-flight depth calibrates to the host/GPU balance — and the two
    /// times it is given must be DISJOINT for it to do so.
    ///
    /// This is the test that was missing when `Session::warmup` passed
    /// `host + tail` as `gpu_step_ns`: that makes the ratio `2h/(h+t) <= 2`
    /// for every `t >= 0`, so the function could never return more than the
    /// floor and the calibration was dead code on every config. The
    /// `host_inclusive` loop below reproduces the defect's arithmetic
    /// directly, so a caller that reintroduces it fails here rather than
    /// silently degrading to the constant this function replaced.
    #[test]
    fn the_in_flight_depth_calibrates_and_needs_disjoint_times() {
        const CAP: usize = 64;
        // The shipping config: the GPU step dwarfs the host encode, so the
        // floor is genuinely right — this case is NOT evidence the maths works.
        assert_eq!(compute_in_flight_depth(1_500_000, 15_000_000, CAP), 2);
        // The host OUTRUNS the GPU — the case a constant 2 caps badly.
        assert_eq!(compute_in_flight_depth(6_000_000, 2_000_000, CAP), 6);
        assert_eq!(compute_in_flight_depth(10_000_000, 1_000_000, CAP), 20);
        // The memory ceiling binds before a pathological ratio does.
        assert_eq!(compute_in_flight_depth(1_000_000_000, 1_000_000, 8), 8);
        // A GPU time of 0 cannot form a ratio; the floor stands.
        assert_eq!(compute_in_flight_depth(5_000_000, 0, CAP), 2);

        // **The defect, as arithmetic.** Passing `host + tail` for the GPU
        // time pins the result at the floor for every balance, including the
        // two above that must exceed it.
        for &(host, tail) in &[
            (1_500_000u64, 500_000u64),
            (6_000_000, 2_000_000),
            (10_000_000, 1_000_000),
        ] {
            assert_eq!(
                compute_in_flight_depth(host, host + tail, CAP),
                IN_FLIGHT_FLOOR,
                "host-inclusive gpu_ns collapses to the floor: that IS the defect \
                 (host={host} tail={tail})"
            );
        }
        // ...and the disjoint form does not, for the same numbers.
        assert!(compute_in_flight_depth(6_000_000, 2_000_000, CAP) > IN_FLIGHT_FLOOR);
        assert!(compute_in_flight_depth(10_000_000, 1_000_000, CAP) > IN_FLIGHT_FLOOR);
    }

    /// The CALLER's operands, not the function's: `Session::warmup` must pass
    /// the GPU tail alone as `gpu_step_ns`, and a host-inclusive value is
    /// detectable from outside because the two times it recorded would then
    /// satisfy `gpu >= host` by construction for every real measurement.
    ///
    /// This is the check the function-level test above cannot make — a
    /// correct function given wrong operands is invisible to a test of the
    /// function. `Session::measured_balance_ns` exists for this.
    #[test]
    fn warm_depth_rejects_host_inclusive_times() {
        const CAP: usize = 64;
        // A host-inclusive `gpu_ns` is `host + tail`, so it is >= host for
        // any tail >= 0 AND the depth it yields is always the floor. Both
        // together are the signature; either alone is not (a genuinely
        // GPU-bound step also has gpu > host, and legitimately floors).
        let host = 6_000_000u64;
        let tail = 2_000_000u64;
        assert_eq!(warm_depth(host, host + tail, CAP), IN_FLIGHT_FLOOR);
        assert!(warm_depth(host, tail, CAP) > IN_FLIGHT_FLOOR);
        // The shipping balance floors under BOTH forms, which is why this
        // defect survived: on the config it was measured against, the wrong
        // arithmetic and the right one agree.
        let (h, t) = (1_500_000u64, 15_000_000u64);
        assert_eq!(warm_depth(h, t, CAP), IN_FLIGHT_FLOOR);
        assert_eq!(warm_depth(h, h + t, CAP), IN_FLIGHT_FLOOR);
    }

    /// `StepPacer::new` cannot be given less than the floor, whatever the
    /// caller computed.
    #[test]
    fn the_pacer_floors_its_depth() {
        assert_eq!(StepPacer::new(0).depth, IN_FLIGHT_FLOOR);
        assert_eq!(StepPacer::new(1).depth, IN_FLIGHT_FLOOR);
        assert_eq!(StepPacer::new(7).depth, 7);
    }
}

#[cfg(test)]
mod pacer_queue_tests {
    use super::*;

    #[test]
    fn pacer_wraps_without_growth_and_retires_sizes_with_markers() -> Result<()> {
        let mut q = PacerQueue::new(8);
        let capacity = q.entries.capacity();
        // Fixed prefill, then adaptive multi-step commits in the same queue.
        for step in 0..10000 {
            if step < 100 {
                while q.entries.len() >= 2 {
                    q.retire(|_| Ok(false))?;
                }
                q.push(step, 1);
            } else {
                while q.queued_steps >= 8 {
                    q.retire(|_| Ok(true))?;
                }
                q.push(step, step % 4 + 1);
            }
            assert_eq!(
                q.queued_steps,
                q.entries.iter().map(|(_, n)| n).sum::<usize>()
            );
            assert_eq!(q.entries.capacity(), capacity);
        }
        while !q.entries.is_empty() {
            q.retire(|_| Ok(false))?;
        }
        assert_eq!(q.queued_steps, 0);
        Ok(())
    }

    #[test]
    fn pacer_failed_wait_preserves_the_entry_and_accounting() {
        let mut q = PacerQueue::new(2);
        q.push(42, 7);
        let result = q.retire(|_| Err(lloom_tensor::Error::Msg("injected wait failure".into())));
        assert!(result.is_err());
        assert_eq!(q.entries.front(), Some(&(42, 7)));
        assert_eq!(q.queued_steps, 7);
    }

    #[test]
    fn pacer_reserves_prefill_and_adaptive_bounds_before_decode() {
        let opts = DecodeOptions {
            prefill_pacer_depth: 24,
            adaptive_commit: true,
            run_ahead: 16,
            ..DecodeOptions::new(32)
        };
        let p = StepPacer::for_decode(&opts);
        assert_eq!(p.depth, opts.pacer_depth().max(IN_FLIGHT_FLOOR));
        #[cfg(feature = "metal")]
        assert!(p.pending.entries.capacity() >= 32);
    }
}

#[cfg(test)]
mod prefill_widths_tests {
    use super::{prefill_widths, tile_widths};

    // The `None` schedule must reproduce the incumbent ramp exactly — this is
    // the byte-identity contract the crown rests on. `bulk` is irrelevant on the
    // `None` path (no ramp), so pass 0.
    #[test]
    fn none_one_shot_is_the_whole_prompt() {
        assert_eq!(prefill_widths(400, 0, &None, 0), vec![400]);
        assert_eq!(prefill_widths(1, 0, &None, 0), vec![1]);
    }

    #[test]
    fn none_spark_is_spark_then_remainder() {
        assert_eq!(prefill_widths(400, 16, &None, 0), vec![16, 384]);
        assert_eq!(prefill_widths(400, 64, &None, 0), vec![64, 336]);
    }

    #[test]
    fn none_spark_wider_than_prompt_is_one_shot() {
        // A spark ≥ the prompt is a single forward, as the old `step.min` did.
        assert_eq!(prefill_widths(10, 16, &None, 0), vec![10]);
        assert_eq!(prefill_widths(16, 16, &None, 0), vec![16]);
    }

    #[test]
    fn none_empty_prompt_is_no_forwards() {
        assert_eq!(prefill_widths(0, 0, &None, 0), Vec::<usize>::new());
        assert_eq!(prefill_widths(0, 16, &None, 0), Vec::<usize>::new());
    }

    // With `bulk == 0` an explicit schedule feeds its units, then a `1`-token
    // tail — the pre-bulk behaviour.
    #[test]
    fn schedule_fixed_bucket_covers_with_tail() {
        // `--spark=16` on a 40-token prompt, no bulk: [16], then 24×1.
        let want: Vec<usize> = std::iter::once(16)
            .chain(std::iter::repeat(1).take(24))
            .collect();
        assert_eq!(prefill_widths(40, 0, &Some(vec![16]), 0), want);
    }

    #[test]
    fn schedule_progressive_ramps_then_tail() {
        // progressive:1,64 → units [1,16,32,64], prompt 200, no bulk: 113
        // consumed, 87 left as 1-token forwards.
        let widths = prefill_widths(200, 0, &Some(vec![1, 16, 32, 64]), 0);
        assert_eq!(&widths[..4], &[1, 16, 32, 64]);
        assert_eq!(widths.iter().sum::<usize>(), 200);
        assert_eq!(widths.len(), 4 + 87);
    }

    #[test]
    fn schedule_exhausted_exactly_has_no_tail() {
        // A schedule whose units sum to the prompt leaves no tail.
        assert_eq!(prefill_widths(64, 0, &Some(vec![64]), 0), vec![64]);
        assert_eq!(prefill_widths(48, 0, &Some(vec![16, 32]), 0), vec![16, 32]);
    }

    #[test]
    fn schedule_unit_wider_than_remainder_is_clamped() {
        // A `64` unit on a 40-token prompt clamps to 40, no negative left.
        assert_eq!(prefill_widths(40, 0, &Some(vec![64]), 0), vec![40]);
    }

    // The bulk tile: after the ramp, the remainder rides `bulk`-wide forwards,
    // then a `< bulk` tail. This is what collapses the many-tiny-forwards count.
    #[test]
    fn bulk_chops_remainder_into_bucket_forwards() {
        // ramp [1,16] on a 200-token prompt, bulk 32: 1+16=17 consumed, 183
        // left → 5×32=160, then 23 left as 1-token tail.
        let widths = prefill_widths(200, 0, &Some(vec![1, 16]), 32);
        assert_eq!(&widths[..2], &[1, 16]);
        // five 32-wide bulk forwards
        assert_eq!(widths.iter().filter(|&&w| w == 32).count(), 5);
        // 23-token tail of 1s
        assert_eq!(widths.iter().filter(|&&w| w == 1).count(), 1 + 23);
        assert_eq!(widths.iter().sum::<usize>(), 200);
        // FAR fewer forwards than the seq_len=1 tail would give (183 vs 2+5+23=30)
        assert_eq!(widths.len(), 2 + 5 + 23);
    }

    #[test]
    fn bulk_evenly_divides_has_no_tail() {
        // ramp [16] on a 400-token prompt, bulk 64: 16 consumed, 384 left =
        // 6×64 exactly, no tail.
        let widths = prefill_widths(400, 0, &Some(vec![16]), 64);
        assert_eq!(widths[0], 16);
        assert_eq!(widths.iter().filter(|&&w| w == 64).count(), 6);
        assert_eq!(widths.iter().filter(|&&w| w == 1).count(), 0);
        assert_eq!(widths.iter().sum::<usize>(), 400);
    }

    #[test]
    fn bulk_one_degenerates_to_seq_len_1_tail() {
        // bulk 1 (or 0) is the pre-bulk behaviour: all-1 tail.
        assert_eq!(
            prefill_widths(20, 0, &Some(vec![16]), 1),
            prefill_widths(20, 0, &Some(vec![16]), 0),
        );
    }

    #[test]
    fn bulk_larger_than_remainder_is_all_tail() {
        // ramp [16] on 40, bulk 64: 24 left < 64 → no bulk forward, 24×1 tail.
        let widths = prefill_widths(40, 0, &Some(vec![16]), 64);
        assert_eq!(widths[0], 16);
        assert_eq!(widths.iter().filter(|&&w| w == 1).count(), 24);
        assert_eq!(widths.iter().sum::<usize>(), 40);
    }

    #[test]
    fn schedule_total_is_always_prompt_len() {
        for &plen in &[1usize, 7, 16, 33, 64, 100, 347, 512] {
            for units in [
                vec![1],
                vec![16],
                vec![64],
                vec![1, 16, 32, 64],
                vec![32, 32],
            ] {
                for bulk in [0usize, 1, 16, 32, 64] {
                    let widths = prefill_widths(plen, 0, &Some(units.clone()), bulk);
                    assert_eq!(
                        widths.iter().sum::<usize>(),
                        plen,
                        "schedule {units:?} bulk {bulk} on prompt {plen} must total the prompt",
                    );
                }
            }
        }
    }

    // The TILE bounds every forward the ramp, the schedule or the bulk names,
    // and changes nothing when it is 0 or wider than every forward.
    #[test]
    fn tile_splits_one_shot_into_sequential_tiles_with_a_narrower_last() {
        let tiled = |plen, tile| tile_widths(prefill_widths(plen, 0, &None, 0), tile);
        assert_eq!(tiled(2048, 512), vec![512, 512, 512, 512]);
        assert_eq!(tiled(1000, 512), vec![512, 488]);
        assert_eq!(tiled(10, 3), vec![3, 3, 3, 1]);
        // A tile at or above the prompt is one-shot, byte for byte.
        assert_eq!(tiled(400, 400), vec![400]);
        assert_eq!(tiled(400, 4096), vec![400]);
    }

    #[test]
    fn tile_composes_with_the_spark_the_schedule_and_the_bulk() {
        // The spark stays a spark; the remainder is tiled.
        assert_eq!(
            tile_widths(prefill_widths(400, 16, &None, 0), 128),
            vec![16, 128, 128, 128]
        );
        // A schedule unit wider than the tile is split; the 1-token tail is not.
        let widths = tile_widths(prefill_widths(100, 0, &Some(vec![64]), 0), 32);
        assert_eq!(&widths[..2], &[32, 32]);
        assert_eq!(widths.len(), 2 + 36);
        assert!(widths[2..].iter().all(|&w| w == 1));
        // A bulk wider than the tile is split too; a bulk at the tile is not.
        let widths = tile_widths(prefill_widths(200, 0, &Some(vec![16]), 64), 32);
        assert_eq!(widths.iter().sum::<usize>(), 200);
        assert!(widths.iter().all(|&w| w <= 32));
        assert_eq!(
            tile_widths(prefill_widths(200, 0, &Some(vec![16]), 64), 64),
            prefill_widths(200, 0, &Some(vec![16]), 64)
        );
    }

    #[test]
    fn tile_zero_is_the_incumbent_and_a_tile_never_widens_a_forward() {
        for &plen in &[1usize, 7, 16, 33, 64, 100, 347, 512, 2049] {
            for &ubatch in &[0usize, 16, 64] {
                let base = prefill_widths(plen, ubatch, &None, 0);
                assert_eq!(tile_widths(base.clone(), 0), base);
                for &tile in &[1usize, 3, 16, 100, 512, 4096] {
                    let tiled = tile_widths(base.clone(), tile);
                    assert_eq!(tiled.iter().sum::<usize>(), plen, "tile {tile} on {plen}");
                    assert!(
                        tiled.iter().all(|&w| w <= tile && w > 0),
                        "tile {tile} on {plen}: {tiled:?}"
                    );
                    // A tile wider than every base forward leaves the widths alone.
                    if base.iter().all(|&w| w <= tile) {
                        assert_eq!(tiled, base);
                    }
                }
            }
        }
    }

    #[test]
    fn max_prefill_width_is_the_tile_else_the_spark_else_the_widest_unit() {
        let mut o = super::DecodeOptions::new(1);
        assert_eq!(o.max_prefill_width(), 0);
        o.prefill_ubatch = 16;
        assert_eq!(o.max_prefill_width(), 16);
        o.prefill_schedule = Some(vec![1, 16, 32, 64]);
        assert_eq!(o.max_prefill_width(), 64);
        o.prefill_bulk = 128;
        assert_eq!(o.max_prefill_width(), 128);
        o.prefill_tile = 24;
        assert_eq!(o.max_prefill_width(), 24);
    }
}

/// The device's `(libraries, pipelines)` cache sizes, or `(0, 0)` off Metal.
fn cache_sizes(device: &Device) -> (usize, usize) {
    #[cfg(feature = "metal")]
    if let Device::Metal(m) = device {
        return m.kernels().cache_sizes();
    }
    let _ = device;
    (0, 0)
}
