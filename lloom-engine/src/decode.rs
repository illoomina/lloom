//! The decode loop, generic over `CausalLM` — the serving engine's core.
//!
//! Extracted from amber's `Lfm2::generate_raw` (RESTRUCTURING §4, Wave F): the
//! prompt rendering and tool parsing that wrapped it are the assistant's, but
//! the loop between them — prefill, sample, detokenize, advance — is the
//! engine's. Since #677 the loop itself is `lloom`'s `Session::decode` over the
//! `Blocking` readback strategy (the host-side sampler, every sampling arm), so
//! this crate carries no second loop (§11.5v named the one it used to carry);
//! what it adds is the serving side: the incremental detokenize, the stop
//! strings, and **content-addressed reuse** — a request is a whole token
//! sequence, and `Session::prepare` reuses whatever prefix the session's cache
//! already holds.
//!
//! Static dispatch only: the loop is generic over `M: CausalLM` so the
//! per-token `forward` monomorphizes and LTO inlines it. `DESIGN.md` §15.2 #9 /
//! `CONTRIBUTING.md` §3.6 — no `dyn` on the decode path. The `dyn`-per-request
//! boundary is the [`crate::seam`] trait, one level out, where it is allowed.

use lloom::generation::poll::PollMechanism;
use lloom::generation::readback::{RingLap, RING_HISTORY_SLOTS};
use lloom::generation::{
    Blocking, EosRead, LogitsProcessor, Logprobs, Processors, ReadbackStrategy, Ring, Sampling,
    ThinkingBudget, ThinkingReport,
};
use lloom::nn::{
    CacheReset, CacheRewind, CausalLM, DecodeOptions, DecodeOut, DecodeStats, IdentityProposer,
    ReuseAction, Sequence, Session,
};
use lloom_tensor::{Device, Result};

/// Execute the reader while its producer's stop signal is in scope.
/// The exit guard also covers a fallible detokenizer and unwinding: error
/// propagation alone does not wake/cancel the independent producer loop.
fn stop_on_reader_exit<T>(stop: &std::sync::atomic::AtomicBool, read: impl FnOnce() -> T) -> T {
    struct StopOnExit<'a>(&'a std::sync::atomic::AtomicBool);
    impl Drop for StopOnExit<'_> {
        fn drop(&mut self) {
            self.0.store(true, std::sync::atomic::Ordering::Release);
        }
    }
    let _guard = StopOnExit(stop);
    read()
}

/// Drain published batches, stopping the producer before delivering a batch
/// whose read found EOS. A hook on an earlier batch can still delay that read.
fn drain_ring_reader(
    stop: &std::sync::atomic::AtomicBool,
    stream_done: &std::sync::atomic::AtomicBool,
    mut read: impl FnMut(&mut Vec<u32>) -> (std::result::Result<usize, RingLap>, bool),
    mut hook: impl FnMut(&[u32]) -> Result<bool>,
) -> Result<()> {
    use std::sync::atomic::Ordering;
    stop_on_reader_exit(stop, || {
        let mut read_and_stop = |buf: &mut Vec<u32>| {
            let (result, ended) = read(buf);
            if result.is_ok() && ended {
                stop.store(true, Ordering::Release);
            }
            result
        };
        let mut buf = Vec::with_capacity(64);
        loop {
            buf.clear();
            match read_and_stop(&mut buf) {
                Ok(0) => {
                    if stream_done.load(Ordering::Acquire) {
                        buf.clear();
                        let n = read_and_stop(&mut buf).map_err(|lap| {
                            stop.store(true, Ordering::Release);
                            lloom_tensor::Error::msg(format!("ring reader final drain: {lap}"))
                        })?;
                        if n > 0 && !hook(&buf)? {
                            stop.store(true, Ordering::Release);
                        }
                        return Ok(());
                    }
                    std::thread::sleep(std::time::Duration::from_micros(200));
                }
                Ok(_) => {
                    if !hook(&buf)? {
                        return Ok(());
                    }
                }
                Err(lap) => {
                    stop.store(true, Ordering::Release);
                    lloom_tensor::bail!("ring reader: {lap}");
                }
            }
        }
    })
}

/// Why a decode run stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The model emitted an end-of-sequence token.
    Eos,
    /// A configured stop string appeared in the decoded text.
    StopString,
    /// The token budget was exhausted.
    Length,
    /// The consumer's fragment hook ended the run ([`DecodeSession::decode_until`]).
    Cancelled,
}

/// A ring arm's parameters: what `lfm2-chat`'s six `--readback` spellings,
/// `--lookahead`, `--commit-budget`, `--adaptive-commit` and `--wake-cadence`
/// select, as one value (`crate::options::ReadbackConfig` builds it).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RingArm {
    /// Forward passes the host enqueues before it has learned the previous
    /// step's EOS: `0` or `1`. `1` is the crown arm and leaves the cache one
    /// position past an EOS, which [`Session::prepare`] reconciles or resets
    /// on the next request.
    pub lookahead: usize,
    /// Steps between commits of the in-flight queue; `1` is per-token.
    pub commit_steps: usize,
    /// The adaptive commit (`lfm2-chat --adaptive-commit --stream-reader`):
    /// a buffer is closed when the GPU starts the previous one rather than
    /// every `commit_steps`, and the token drain is a reader thread's, off
    /// the loop's path. Measured +0.5 tok/s on the B=1 crown (#148, record
    /// §8.1). Needs a CPU-mapped ring, lookahead 1 and an EOS read other
    /// than the flag; the fixed cadence where it has not
    /// ([`RingArm::resolved`]).
    pub adaptive: bool,
    /// How the ring learns EOS (`ring` = a blocking flag read, `ring-nonblocking`
    /// = the poll, `ring-hostdrain`, `ring-gpubatch[-nonblocking]`).
    pub eos: EosRead,
    /// Commits between live completion-marker writes on the deferred-marker
    /// arms (`--wake-cadence`).
    pub wake: usize,
}

impl RingArm {
    /// The crown: the poll, one forward of lookahead, a commit every eight
    /// steps, the adaptive commit (#702).
    pub const CROWN: RingArm = RingArm {
        lookahead: 1,
        commit_steps: 8,
        adaptive: true,
        eos: EosRead::Poll(PollMechanism::DEFAULT),
        wake: lloom::generation::WAKE_LIVE_EVERY_DEFAULT,
    };

    /// The crown before the adaptive commit — the A/B control.
    pub const FIXED: RingArm = RingArm {
        adaptive: false,
        ..Self::CROWN
    };

    /// Why this arm cannot run the adaptive commit, or `None` if it can. The
    /// commit encodes the next forward before anything is read back, so it
    /// needs one forward of lookahead and an EOS read that does not read each
    /// token back: every ring read but the flag (`Session::decode`'s
    /// refusals). Every read stamps each token under the commit, for any
    /// sampler, so its reader streams whatever the run returns
    /// (`docs/dispatch.md` §11.5z-xvi).
    pub fn why_not_adaptive(&self) -> Option<&'static str> {
        if self.lookahead != 1 {
            Some("the adaptive commit needs lookahead 1")
        } else if matches!(self.eos, EosRead::Blocking) {
            Some("the `ring` EOS read (the flag) reads each token back per step")
        } else {
            None
        }
    }

    /// The arm that runs: this one, on the fixed cadence where it cannot
    /// carry the adaptive commit.
    pub fn resolved(self) -> RingArm {
        RingArm {
            adaptive: self.adaptive && self.why_not_adaptive().is_none(),
            ..self
        }
    }

    /// The report-line rendering: the `--readback` spelling this arm is, its
    /// lookahead and cadence, from the value that ran (§7.1a-i).
    pub fn name(&self) -> String {
        let spelling = match self.eos {
            EosRead::Blocking => "ring",
            EosRead::Poll(_) => "ring-nonblocking",
            EosRead::HostOnDrain => "ring-hostdrain",
            EosRead::GpuBatched => "ring-gpubatch",
            EosRead::GpuBatchedNonblocking => "ring-gpubatch-nonblocking",
        };
        format!(
            "{spelling} lookahead={} commit_steps={}{}",
            self.lookahead,
            self.commit_steps,
            if self.adaptive { " adaptive" } else { "" }
        )
    }
}

/// Which readback strategy a run takes — where the per-token round trip sits
/// (`design/docs/decode-driver-design.md`, seam A; `readback.rs`).
///
/// The serving crate used to hard-code [`Blocking`], the arm that reads every
/// token back to the host and waits (§11.5a's 441 µs seam). The crown decode
/// path is the device-resident [`Ring`] polled for EOS with one forward of
/// lookahead (§11.5g, §11.2b), which `lfm2-chat` selects by hand; this enum is
/// how a consumer selects it through the seam instead of re-writing that
/// selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReadbackArm {
    /// The ring when the sampler can ride it (argmax or plain temperature),
    /// the blocking arm otherwise — the choice [`Ring::check_sampler`]
    /// would refuse is made here rather than failing the request.
    Auto,
    /// The host reads every token back; any sampler, no lookahead.
    Blocking,
    /// The device-resident token ring: the token stays on the device, EOS is
    /// learned per [`RingArm::eos`], the loop paced per the arm's lookahead
    /// and commit cadence.
    Ring(RingArm),
    /// The speculative body ([`Session::decode_speculative`]): a DSpark
    /// drafter proposes `k` tokens a round and the target verifies them in
    /// one pass; the round's one readback is the verify's. Reported, never
    /// selected by a request — the drafter is the process's.
    Speculative {
        /// The draft block width.
        k: usize,
    },
    /// The batched driver (`Session::decode_batched`'s body, issue #148):
    /// the row's token came from one device argmax over the batch (or the
    /// per-row host sample) and a staged readback, beside `rows` other
    /// sequences on the same weight sweep. Reported, never selected.
    Batched {
        /// Rows live in the batch when this one joined it.
        rows: usize,
    },
}

impl ReadbackArm {
    /// The crown selection: ring, one forward of lookahead, commit every 8
    /// steps (`lfm2-chat`'s defaults, #520's flat queue bound).
    pub const CROWN: ReadbackArm = ReadbackArm::Ring(RingArm::CROWN);

    /// The ring arm at the fixed cadence — the crown before the adaptive
    /// commit (`--readback ring-fixed`): the A/B control, not a default.
    pub const RING_FIXED: ReadbackArm = ReadbackArm::Ring(RingArm::FIXED);

    /// Whether the ring arm can serve `sampling` without a host round trip
    /// it would not report (`Ring::check_sampler`).
    fn ring_admits(sampling: &Sampling) -> bool {
        matches!(
            sampling,
            Sampling::ArgMax | Sampling::All { .. } | Sampling::GumbelSoftmax { .. }
        )
    }

    /// The arm that runs for `sampling`: `Auto` resolves here — to
    /// `auto_ring` (the engine's configured ring, `DecodeConfig`) where the
    /// sampler admits a ring, the blocking arm otherwise; the explicit arms
    /// pass through.
    pub fn resolve(self, sampling: &Sampling, auto_ring: RingArm) -> ReadbackArm {
        match self {
            ReadbackArm::Auto if Self::ring_admits(sampling) => ReadbackArm::Ring(auto_ring),
            ReadbackArm::Auto => ReadbackArm::Blocking,
            other => other,
        }
    }

    /// The arm's name for a report line: `blocking` or `ring-nonblocking`
    /// with its lookahead and commit period, rendered from the resolved value
    /// rather than from the request (§7.1a-i).
    pub fn name(self) -> String {
        match self {
            ReadbackArm::Auto => "auto".to_string(),
            ReadbackArm::Blocking => "blocking".to_string(),
            ReadbackArm::Ring(r) => r.name(),
            // The drafter is the family's (`speculative.method`); the loop is one.
            ReadbackArm::Speculative { k } => format!("speculative k={k}"),
            ReadbackArm::Batched { rows } => format!("batched rows={rows}"),
        }
    }
}

/// Sampling and stopping knobs for one decode run. Engine-level: token ids and
/// counts, no conversation vocabulary (that is the consumer's — [`crate::seam`]).
#[derive(Debug, Clone)]
pub struct DecodeParams {
    pub seed: u64,
    /// `None` (or `< 1e-7`) is greedy argmax; `Some(t)` samples at temperature
    /// `t`, with `top_p` when set. Matches [`LogitsProcessor::new`].
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    /// Top-k, with `top_p` top-k-then-top-p (`Sampling::TopK` / `TopKThenTopP`).
    pub top_k: Option<usize>,
    /// Device gumbel-max at the temperature (`Sampling::GumbelSoftmax`).
    pub gumbel: bool,
    /// Logit bias, the repetition / presence / frequency penalties and
    /// min-p, run before the draw (`lloom::generation::processors`); the
    /// default is neutral.
    pub processors: Processors,
    pub max_tokens: usize,
    /// End-of-sequence ids; the first one sampled stops the run.
    pub eos_ids: Vec<u32>,
    /// Text substrings that stop the run once they appear in the output.
    pub stop_strings: Vec<String>,
    /// Which readback arm the run takes; [`ReadbackArm::Auto`] by default.
    pub readback: ReadbackArm,
    /// Hash every decode step's logits beside the tokens
    /// (`LogitsProcessor::with_logits_digest`, the harnesses' `logits_sha256`):
    /// a diagnostic that runs the blocking readback (the ring arms draw on
    /// the device and the host never sees the logits) and is refused by the
    /// speculative body (its verify keeps no per-position logits).
    pub logits_digest: bool,
    /// The TILE: the most prompt tokens one prefill forward takes
    /// (`DecodeOptions::prefill_tile`, lloom #728). `0` is one forward over
    /// the whole prompt, whose activation working set grows with the prompt
    /// (~3.5 MB per token on the crown); a positive tile bounds it to the
    /// tile's worth whatever the prompt. The model's `conv_prefill_ctx` must
    /// be on for a tiled prefill to compute what one pass does.
    pub prefill_tile: usize,
    /// The thinking budget, resolved against the family's markers for a
    /// prompt that leaves a thinking block open (`generation::budget`). The
    /// K=1 body carries it on every readback arm; the speculative body and
    /// the batch scheduler refuse it.
    pub thinking: Option<ThinkingBudget>,
    /// The generation prompt's length at the end of the prompt
    /// (`thinking::checkpoint_tail`, the load-time constant): the prefill
    /// leaves its turn checkpoint before it (`CausalLM::checkpoint_at`), so
    /// the next request, whose render keeps everything before it, is a
    /// restore there rather than a fresh prefill. 0: the checkpoint goes at
    /// the prompt's end (`turn::CheckpointPolicy::of`).
    pub turn_tail: usize,
    /// The owner keeps nothing of this request's state for the next one
    /// (prefix reuse off): its prefill leaves no turn checkpoint.
    pub no_checkpoint: bool,
    /// Keep each returned id's log-probability (`Decoded::logprobs`). A host
    /// read of every step's logits: the run takes the blocking arm, as the
    /// logits digest does, and the speculative body and the batch scheduler
    /// refuse it. `None` costs nothing.
    pub logprobs: Option<LogprobRequest>,
    /// Start this run on an empty session whatever the owner's prefix reuse:
    /// a benchmark cell that measures a cold prefill stays cold with reuse on.
    pub cold: bool,
}

/// What a run keeps of each step's distribution: the drawn id's
/// log-probability always, the `top` most likely, and with `rows` the whole
/// row (a trainer's use, never the inference path's).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LogprobRequest {
    pub top: usize,
    pub rows: bool,
}

impl DecodeParams {
    /// The sampler arm for these fields: argmax below 1e-7 of temperature,
    /// else gumbel-max, top-k-then-top-p, top-k, top-p, or plain temperature
    /// (`LogitsProcessor::new`'s rule, extended to the arms the sampler has).
    pub fn sampling_of(
        temperature: Option<f64>,
        top_p: Option<f64>,
        top_k: Option<usize>,
        gumbel: bool,
    ) -> Sampling {
        let temperature = match temperature {
            Some(t) if t >= 1e-7 => t,
            _ => return Sampling::ArgMax,
        };
        if gumbel {
            return Sampling::GumbelSoftmax { temperature };
        }
        match (top_k, top_p) {
            (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            (Some(k), None) => Sampling::TopK { k, temperature },
            (None, Some(p)) => Sampling::TopP { p, temperature },
            (None, None) => Sampling::All { temperature },
        }
    }

    /// These params under an owner whose prefix reuse is `reuse`: with reuse
    /// off nothing of the request is kept for the next one, so its prefill
    /// leaves no turn checkpoint.
    pub fn under_reuse(&self, reuse: bool) -> std::borrow::Cow<'_, Self> {
        if reuse || self.no_checkpoint {
            std::borrow::Cow::Borrowed(self)
        } else {
            std::borrow::Cow::Owned(Self {
                no_checkpoint: true,
                ..self.clone()
            })
        }
    }

    /// The sampler arm of this request.
    pub fn sampling(&self) -> Sampling {
        Self::sampling_of(self.temperature, self.top_p, self.top_k, self.gumbel)
    }

    /// The sampler of this request, seeded, with its processors.
    pub fn sampler(&self) -> LogitsProcessor {
        LogitsProcessor::from_sampling(self.seed, self.sampling())
            .with_processors(self.processors.clone())
            .with_logits_digest(self.logits_digest)
    }

    /// Refuse what `body` cannot serve of this request, by name — the one
    /// table every body asks (`lloom::nn::decode::refuse`); `path` names the
    /// caller in the message.
    pub fn refuse_on(&self, body: lloom::nn::Body, path: &str) -> Result<()> {
        let sampling = self.sampling();
        lloom::nn::refuse(
            body,
            &lloom::nn::Asks {
                sampling: &sampling,
                processors: self.processors.is_active(),
                budget: self.thinking.is_some(),
                digest: self.logits_digest,
                logprobs: self.logprobs.map(|r| r.top),
            },
        )
        .map_err(|e| lloom_tensor::Error::Msg(format!("{path}: {e}")))
    }
}

impl Default for DecodeParams {
    fn default() -> Self {
        DecodeParams {
            seed: 42,
            temperature: None,
            top_p: None,
            top_k: None,
            gumbel: false,
            processors: Processors::default(),
            max_tokens: 512,
            eos_ids: Vec::new(),
            stop_strings: Vec::new(),
            readback: ReadbackArm::Auto,
            logits_digest: false,
            prefill_tile: 0,
            thinking: None,
            turn_tail: 0,
            no_checkpoint: false,
            logprobs: None,
            cold: false,
        }
    }
}

/// The outcome of a decode run: the full text, the tokens generated, why it
/// stopped, and what the request reused of the session's cache.
#[derive(Debug, Clone)]
pub struct Decoded {
    pub text: String,
    pub tokens: Vec<u32>,
    pub stop: Stop,
    /// How the request was matched against the session's cached tokens.
    pub reuse: ReuseAction,
    /// Prompt tokens whose state the cache already held and kept.
    pub reused: usize,
    /// Prompt tokens actually prefilled — the request's length minus `reused`,
    /// except under [`ReuseAction::Resume`] where nothing is.
    pub prefilled: usize,
    /// What the request reused of the session, and why not more
    /// ([`crate::turn`]).
    pub turn: crate::turn::TurnReuse,
    /// The turn checkpoint the request's prefill was asked to leave, and the
    /// position it resolves to (`None` for none).
    pub checkpoint: crate::turn::CheckpointPolicy,
    pub checkpoint_at: Option<usize>,
    /// The readback arm that ran: resolved (never [`ReadbackArm::Auto`]), and
    /// on the fixed cadence wherever the adaptive commit did not run.
    pub readback: ReadbackArm,
    /// The run's logits digest, when `DecodeParams::logits_digest` asked for
    /// one.
    pub logits_sha256: Option<String>,
    /// The loop's own counters — prefill submit time, per-step host time,
    /// commits, ring engagement — so a consumer that reports throughput reads
    /// the loop's clock rather than bracketing the call.
    pub stats: DecodeStats,
    /// What the run did inside the thinking block, when a budget was armed.
    pub thinking: Option<ThinkingReport>,
    /// Per returned id, its log-probability (and what else
    /// `DecodeParams::logprobs` asked for).
    pub logprobs: Option<Logprobs>,
}

impl Decoded {
    /// The turn checkpoint the request asked for: `none`, `end@P` or `at@P`.
    pub fn checkpoint_name(&self) -> String {
        self.checkpoint.name(self.checkpoint_at.unwrap_or(0))
    }
}

/// A decode session: a borrowed target and the sequence it drives — the cache,
/// the ids whose state it holds, and the position. The model is borrowed rather
/// than owned so a session pins no weights alive beyond the run it drives.
///
/// A request is the WHOLE token sequence to continue, not the new tokens: the
/// session reuses the prefix its cache already holds (`Session::prepare`) and
/// prefills the rest. A caller that decodes several turns against one session
/// therefore passes the growing conversation each time and pays only for the
/// new turn; a fresh session with an empty cache prefills everything, which is
/// what every request did before #677.
/// The token rings a serving process keeps across requests, one per EOS set.
///
/// This pool calls `Ring::new` with `RING_HISTORY_SLOTS` (4,092 dense slots,
/// 16,368 bytes), not `Ring::with_geometry` with its configurable consumer
/// slack. Capacity bounds the live window, not the generation: the same mapped
/// ring wraps across long runs. Larger segmented rings can remain mapped too;
/// mapping is a per-segment property, not a total-capacity threshold.
///
/// A ring's EOS set and read mechanism are fixed at construction, so the pool
/// keys both. Built on first use, or ahead of it by [`RingPool::prepare`] at
/// load, it is reused across requests rather than zero-filled per request.
#[derive(Default)]
pub struct RingPool {
    rings: Vec<(Vec<u32>, EosRead, Ring)>,
}

// SAFETY: a `Ring` is `!Send` only through `SegmentedRing::seg_bases`
// (`readback.rs`), a cache of `*const u32` CPU mappings of Metal buffers the
// ring itself owns — a mapping is valid from any thread for as long as its
// buffer lives, and the buffer moves with the pool. Nothing in a ring is
// thread-affine (no TLS, no Metal object that must be used on its creating
// thread). A pool is driven by one thread at a time — a serving binary keeps
// it behind the engine's mutex and hands it to whichever request thread holds
// the lock — which is the same discipline the sequence beside it already has.
unsafe impl Send for RingPool {}

impl RingPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// The ring for `eos_ids` under `eos`, built if the pool has none.
    pub fn ring(&mut self, eos_ids: &[u32], device: &Device, eos: EosRead) -> Result<&mut Ring> {
        let key = Self::key(eos_ids);
        let at = match self
            .rings
            .iter()
            .position(|(k, e, _)| *k == key && *e == eos)
        {
            Some(i) => i,
            None => {
                let ring = Ring::new(&key, device, RING_HISTORY_SLOTS, eos)?;
                self.rings.push((key, eos, ring));
                self.rings.len() - 1
            }
        };
        Ok(&mut self.rings[at].2)
    }

    /// Build the ring for `eos_ids` now; `true` if this call built it.
    pub fn prepare(&mut self, eos_ids: &[u32], device: &Device, eos: EosRead) -> Result<bool> {
        let before = self.rings.len();
        self.ring(eos_ids, device, eos)?;
        Ok(self.rings.len() > before)
    }

    pub fn len(&self) -> usize {
        self.rings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rings.is_empty()
    }

    /// An EOS *set*: order and repeats do not change what the ring stops on.
    fn key(eos_ids: &[u32]) -> Vec<u32> {
        let mut k = eos_ids.to_vec();
        k.sort_unstable();
        k.dedup();
        k
    }
}

/// The EOS read the crown arm's ring takes: the poll, on the platform's
/// default mechanism (`Wfe` on aarch64).
pub fn ring_eos_read() -> EosRead {
    EosRead::Poll(PollMechanism::default())
}

pub struct DecodeSession<'m, M: CausalLM> {
    session: Session<'m, M>,
    device: Device,
    rings: RingPool,
    /// The loop's options (`--readback`, the cadence, the pacer, the
    /// prefill): the engine's, set once at load, resolved per request into
    /// `DecodeOptions`.
    config: crate::options::DecodeConfig,
}

impl<'m, M: CausalLM> DecodeSession<'m, M> {
    /// Pair a target with a cache it has already built ([`CausalLM::new_cache`]).
    /// `pos` is the number of positions already prefilled into that cache — 0
    /// for a fresh one. A non-zero `pos` describes a cache whose contents this
    /// session does not know, so the first request resets it (`Session::with_pos`).
    pub fn new(model: &'m M, cache: M::Cache, device: Device, pos: usize) -> Self {
        DecodeSession {
            session: Session::with_pos(model, cache, pos),
            device,
            rings: RingPool::new(),
            config: Default::default(),
        }
    }

    /// Drive a sequence a serving layer held across requests ([`crate::SlotPool`]).
    pub fn from_sequence(model: &'m M, seq: Sequence<M::Cache>, device: Device) -> Self {
        Self::from_parts(model, seq, device, RingPool::new())
    }

    /// Resume a sequence with the rings an earlier session built, so a serving
    /// process pays a ring's construction once, not once per request.
    pub fn from_parts(
        model: &'m M,
        seq: Sequence<M::Cache>,
        device: Device,
        rings: RingPool,
    ) -> Self {
        DecodeSession {
            session: Session::from_sequence(model, seq),
            device,
            rings,
            config: Default::default(),
        }
    }

    /// The loop's options for every run of this session (`DecodeConfig`).
    pub fn with_config(mut self, config: crate::options::DecodeConfig) -> Self {
        self.config = config;
        self
    }

    pub fn set_config(&mut self, config: crate::options::DecodeConfig) {
        self.config = config;
    }

    pub fn config(&self) -> &crate::options::DecodeConfig {
        &self.config
    }

    /// The sequence and the rings, to hand to the next session.
    pub fn into_parts(self) -> (Sequence<M::Cache>, RingPool) {
        (self.session.into_sequence(), self.rings)
    }

    pub fn rings(&mut self) -> &mut RingPool {
        &mut self.rings
    }

    /// Give the sequence back, to be held until the next request.
    pub fn into_sequence(self) -> Sequence<M::Cache> {
        self.session.into_sequence()
    }

    /// The cache, for a caller that inspects or seeds it between runs.
    pub fn cache_mut(&mut self) -> &mut M::Cache {
        self.session.cache_mut()
    }

    /// Positions currently in the cache, a previous run's overrun included.
    pub fn pos(&self) -> usize {
        self.session.pos()
    }

    /// Discard a previous run's overrun so the cache ends at the token log's
    /// last id, or reset the sequence where the cache cannot take it back —
    /// what [`Session::prepare`] does first. For a caller that reads the
    /// position or prefills before [`Self::decode_until`] prepares the run.
    pub fn reconcile(&mut self)
    where
        M::Cache: CacheRewind + CacheReset,
    {
        if self.session.overrun() > 0 && self.session.reconcile().is_err() {
            self.session.reset();
        }
    }

    /// Return the sequence to empty: cache reset, token log cleared. A serving
    /// layer that must not reuse state between requests — a benchmark cell
    /// that measures a cold prefill — calls this before each one; a layer that
    /// wants prefix reuse does not, and [`Session::prepare`] decides.
    pub fn reset(&mut self)
    where
        M::Cache: CacheReset,
    {
        self.session.reset()
    }

    /// Compile the pipelines the first real forward would otherwise compile
    /// inside its own timed window, then reset ([`Session::warmup`]). Returns
    /// the device's `(libraries, pipelines)` cache sizes.
    pub fn warmup(&mut self) -> Result<(usize, usize)>
    where
        M::Cache: CacheReset,
    {
        self.session.warmup(&self.device)
    }

    /// What warmup measured: the in-flight depth its host/GPU balance
    /// implies, and the `(host_ns, gpu_ns)` balance itself; zeros before it
    /// runs. The session's pacer takes `decode.in_flight` (0 = the floor),
    /// not this; a report names both.
    pub fn measured_in_flight(&self) -> (usize, (u64, u64)) {
        (
            self.session.measured_in_flight_depth(),
            self.session.measured_balance_ns(),
        )
    }

    /// The device this session decodes on.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The target this session drives.
    pub fn model(&self) -> &'m M {
        self.session.model()
    }

    /// The ids whose state the cache holds.
    pub fn tokens(&self) -> &[u32] {
        self.session.tokens()
    }

    /// Feed `prompt_ids` onto the session's cache and produce nothing
    /// (`Session::prefill`): the prompt in tiles of `params.prefill_tile`
    /// tokens, each forward paced against the GPU exactly as the decode loop
    /// paces its own — at most `IN_FLIGHT_FLOOR` forwards in flight, so the
    /// activation working set is two tiles' worth whatever the prompt (#728
    /// §2). The session's held logits are the resumable state afterwards.
    ///
    /// The prefill-throughput cell's entry: a tile loop written by hand
    /// around `forward` has no pacer, and the host then encodes every tile
    /// back to back with all of their activations `pending` at once
    /// (`issue-148-serving-batch.md` §7: 11.4 GB against 4.5 GB paced, same
    /// tile, same prompt).
    pub fn prefill(
        &mut self,
        prompt_ids: &[u32],
        params: &DecodeParams,
    ) -> Result<lloom::nn::PrefillOut> {
        if params.eos_ids.is_empty() {
            lloom_tensor::bail!("DecodeSession::prefill: eos_ids is empty; a run needs an EOS set");
        }
        let mut readback = Blocking::new(&params.eos_ids, &self.device)?;
        let opts = self.config.options(
            params.max_tokens.max(1),
            ReadbackArm::Blocking,
            params.prefill_tile,
        );
        self.session.prefill(&mut readback, prompt_ids, &opts)
    }

    /// Decode `prompt_ids` — the whole sequence to continue — streaming each
    /// completed text fragment to `on_fragment` as it is produced. Returns the
    /// full decoded run.
    ///
    /// The streaming primitive: a fragment is emitted only when the tokens
    /// since the last emission decode to complete text — a multi-byte character
    /// can span several tokens, so a run ending on a `U+FFFD` replacement char
    /// is held until the next token completes it. This fragment-boundary logic
    /// is the engine's; it lives here now (was amber's).
    ///
    /// `detokenize` maps a run of ids to text — the consumer supplies it so the
    /// engine names no tokenizer library (amber and the engine were on
    /// different `tokenizers` majors; the seam must not couple them). It is
    /// called on the not-yet-emitted tail each step, so it must decode a *slice*
    /// of ids, not assume it starts at position 0.
    ///
    /// `on_fragment` is a `FnMut(&str)` rather than a channel so the engine
    /// stays runtime-agnostic: the consumer wires whatever streaming transport
    /// it uses (amber's `spawn_blocking` + mpsc channel) around this call.
    pub fn decode(
        &mut self,
        prompt_ids: &[u32],
        detokenize: impl FnMut(&[u32]) -> Result<String> + Send,
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) + Send,
    ) -> Result<Decoded>
    where
        M::Cache: CacheRewind + CacheReset,
    {
        self.decode_until(prompt_ids, detokenize, params, |frag| {
            on_fragment(frag);
            true
        })
    }

    /// [`Self::decode`] whose fragment hook can end the run: a `false` stops
    /// after the fragment it was handed, reported as [`Stop::Cancelled`]. The
    /// serving layer's cancellation — a client that went away, a benchmark's
    /// early abort on a degenerate sample — rides the same seam a stop string
    /// does, so the loop needs no second path.
    pub fn decode_until(
        &mut self,
        prompt_ids: &[u32],
        mut detokenize: impl FnMut(&[u32]) -> Result<String> + Send,
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded>
    where
        M::Cache: CacheRewind + CacheReset,
    {
        if params.eos_ids.is_empty() {
            lloom_tensor::bail!("DecodeSession::decode: eos_ids is empty; a run needs an EOS set");
        }
        let (prepared, turn) = crate::turn::prepare(&mut self.session, prompt_ids, true)?;
        let (checkpoint, checkpoint_at) = self.establish_checkpoint(&prepared, params)?;

        let mut sampler = params.sampler();
        sampler.observe_prompt(prompt_ids);
        let arm = if params.logits_digest || params.logprobs.is_some() {
            ReadbackArm::Blocking
        } else {
            params
                .readback
                .resolve(sampler.sampling(), self.config.readback.ring_arm())
        };
        if let Err(e) = self.config.check() {
            lloom_tensor::bail!("DecodeSession::decode: {e}");
        }
        let mut spec = IdentityProposer;

        let mut stream = crate::stream::TextStream::new(params.max_tokens);
        let mut stopped_on_string = false;
        let mut cancelled = false;

        // The streaming hook, shared by every arm: decode only the
        // not-yet-emitted tokens, hold a run ending mid-character or on what
        // may begin a stop string, and let a stop string end the run
        // (`TextStream::publish`).
        let mut on_tokens = |batch: &[u32]| -> Result<bool> {
            let (chunk, stopped) = stream.publish(batch, &mut detokenize, &params.stop_strings)?;
            if let Some(chunk) = chunk {
                if !on_fragment(&chunk) {
                    stream.refuse_last();
                    cancelled = true;
                    return Ok(false);
                }
            }
            if stopped {
                stopped_on_string = true;
                return Ok(false);
            }
            Ok(true)
        };

        let mut logprobs: Option<Logprobs> = None;
        let (out, ran): (DecodeOut, ReadbackArm) = match arm {
            ReadbackArm::Blocking => {
                let mut readback = Blocking::new(&params.eos_ids, &self.device)?;
                if let Some(r) = params.logprobs {
                    readback = readback.with_logprobs(r.top, r.rows);
                }
                readback.set_thinking_budget(params.thinking.as_ref())?;
                let opts = self.config.options(
                    params.max_tokens,
                    ReadbackArm::Blocking,
                    params.prefill_tile,
                );
                let out = self.session.decode_until(
                    &mut spec,
                    &mut readback,
                    &mut sampler,
                    &prepared.suffix,
                    &opts,
                    &mut on_tokens,
                )?;
                logprobs = readback.take_logprobs();
                (out, ReadbackArm::Blocking)
            }
            ReadbackArm::Ring(ring) => {
                // The fused publish runs for the argmax sampler only, and
                // never under processors: its kernel draws the argmax itself,
                // past them.
                let fused = self.config.readback.fused_publish
                    && matches!(sampler.sampling(), Sampling::ArgMax)
                    && sampler.processors().is_none();
                let ring = ring.resolved();
                let RingArm {
                    adaptive,
                    eos,
                    wake,
                    ..
                } = ring;
                // The pool's ring for this EOS set and EOS read: built once
                // per process (or at load, `RingPool::prepare`) and reused by
                // every run.
                let readback = self.rings.ring(&params.eos_ids, &self.device, eos)?;
                readback.set_wake_cadence(wake);
                // Armed for this run whatever it holds (`None` disarms), so a
                // pooled ring never carries a budget into another run.
                readback.set_thinking_budget(params.thinking.as_ref())?;
                // The adaptive arm needs the ring's reader (a CPU-mapped ring);
                // without one it is the fixed cadence, reported as such. The
                // reader alone (`--stream-reader`) streams what the arm stamps:
                // per token on the poll and flag reads, per commit window on
                // the deferred reads' fixed cadence (`docs/dispatch.md`
                // §11.5z-xvi). The pool's ring served earlier runs and its
                // stamp runs across them, so the reader starts at this run's
                // first token.
                let reader = if adaptive || self.config.readback.stream_reader {
                    readback.reader().map(|mut r| {
                        r.resume_at_published();
                        r
                    })
                } else {
                    None
                };
                let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let mut opts = self.config.options(
                    params.max_tokens,
                    ReadbackArm::Ring(ring),
                    params.prefill_tile,
                );
                opts.fused_publish = fused;
                opts.stream_via_reader = reader.is_some();
                opts.adaptive_commit = reader.is_some() && adaptive;
                opts.eos_gate = self.config.readback.eos_gate && opts.adaptive_commit;
                opts.stop = reader.is_some().then(|| std::sync::Arc::clone(&stop_flag));
                let ran = ReadbackArm::Ring(RingArm {
                    adaptive: opts.adaptive_commit,
                    ..ring
                });
                let out = match reader {
                    None => self.session.decode_until(
                        &mut spec,
                        readback,
                        &mut sampler,
                        &prepared.suffix,
                        &opts,
                        &mut on_tokens,
                    )?,
                    Some(mut reader) => {
                        // The drain is the reader's (`lfm2-chat --stream-reader`):
                        // a scoped thread reads the ring as the GPU publishes and
                        // runs the same streaming hook; the loop neither reads
                        // tokens back nor commits on a cadence. A stop string or
                        // a cancelled consumer raises `stop_flag`, which the loop
                        // checks before each step; an EOS the ring has seen
                        // raises it too. After the loop returns, one last read
                        // takes what was published behind the reader's cursor.
                        let stream_done = std::sync::atomic::AtomicBool::new(false);
                        let (out, drained) = std::thread::scope(|s| {
                            let hook = &mut on_tokens;
                            let stop_flag = &stop_flag;
                            let stream_done = &stream_done;
                            let worker = s.spawn(move || {
                                drain_ring_reader(
                                    stop_flag,
                                    stream_done,
                                    |buf| {
                                        if reader.ended() {
                                            stop_flag
                                                .store(true, std::sync::atomic::Ordering::Release);
                                        }
                                        let result = reader.read_new(buf);
                                        (result, reader.ended())
                                    },
                                    hook,
                                )
                            });
                            let out = self.session.decode_until(
                                &mut spec,
                                readback,
                                &mut sampler,
                                &prepared.suffix,
                                &opts,
                                &mut |_: &[u32]| Ok(true),
                            );
                            stream_done.store(true, std::sync::atomic::Ordering::Release);
                            let drained = worker.join().map_err(|_| {
                                lloom_tensor::Error::msg("ring reader thread panicked")
                            });
                            (out, drained)
                        });
                        drained??;
                        out?
                    }
                };
                self.rings
                    .ring(&params.eos_ids, &self.device, eos)?
                    .set_thinking_budget(None)?;
                (out, ran)
            }
            ReadbackArm::Auto => unreachable!("resolved above"),
            ReadbackArm::Speculative { .. } | ReadbackArm::Batched { .. } => {
                lloom_tensor::bail!(
                    "DecodeSession::decode: {} is a reported arm, not a selectable one",
                    arm.name()
                )
            }
        };

        if !cancelled && !stopped_on_string {
            let (rest, stopped) = stream.flush(&mut detokenize, &params.stop_strings)?;
            stopped_on_string |= stopped;
            if let Some(rest) = rest {
                if !on_fragment(&rest) {
                    stream.refuse_last();
                    cancelled = true;
                }
            }
        }
        let stop = if cancelled {
            Stop::Cancelled
        } else if stopped_on_string {
            Stop::StopString
        } else if out.eos {
            Stop::Eos
        } else {
            Stop::Length
        };
        let (text, _) = stream.finish(&params.stop_strings);
        let thinking = params.thinking.as_ref().map(|b| b.report(&out.ids));

        Ok(Decoded {
            text,
            tokens: out.ids,
            stop,
            reuse: prepared.action,
            reused: prepared.reused,
            prefilled: prepared.suffix.len(),
            turn,
            checkpoint,
            checkpoint_at,
            readback: ran,
            logits_sha256: sampler.logits_digest(),
            stats: out.stats,
            thinking,
            logprobs,
        })
    }
}

impl<'m, M: CausalLM> DecodeSession<'m, M> {
    /// The turn checkpoint this request's prefill leaves
    /// ([`crate::turn::CheckpointPolicy::of`]), established on the cache
    /// whatever the request before it asked; returned with the position it
    /// resolves to.
    ///
    /// Until the checkpoint ring takes a request's policy itself
    /// (`sequence-reuse-design.md` §2.3b), it is expressed through the
    /// cache's one hint (`CausalLM::checkpoint_at`):
    /// `At` and `End` hint their position, which a pass covering it meets
    /// and no other tile does; `None` hints a position no pass reaches, which
    /// the next request's hint replaces. A cache without checkpoints ignores
    /// every hint.
    fn establish_checkpoint(
        &mut self,
        prepared: &lloom::nn::Prepared,
        params: &DecodeParams,
    ) -> Result<(crate::turn::CheckpointPolicy, Option<usize>)> {
        let pos = self.session.pos();
        let end = pos + prepared.suffix.len();
        let policy = crate::turn::CheckpointPolicy::of(
            pos,
            prepared.suffix.len(),
            params.turn_tail,
            !params.no_checkpoint,
        );
        let at = policy.position(end);
        let model = self.session.model();
        model.checkpoint_at(self.session.cache_mut(), at.unwrap_or(usize::MAX))?;
        Ok((policy, at))
    }

    /// [`Self::decode_until`] through the speculative body: `spec` drafts a
    /// block per round and the target verifies it
    /// ([`Session::decode_speculative`], `design/docs/speculation.md`
    /// §10.2t). The verify draws the target's token per window position with
    /// the request's sampler, keyed by position, so the run is the
    /// non-speculative one at the same seed whatever the drafter proposed:
    /// every sampler arm, the processors, the thinking budget and the logits
    /// digest. The fragment hook's `false` ends the run after the round in
    /// flight (`DecodeOptions::stop`), reported as [`Stop::Cancelled`]; a
    /// prompt the cache holds whole re-feeds its last token
    /// (`Session::prepare_with`).
    pub fn decode_speculative_until<S>(
        &mut self,
        rewind: lloom::profile::RewindProof<M::Profile>,
        spec: &mut S,
        prompt_ids: &[u32],
        mut detokenize: impl FnMut(&[u32]) -> Result<String> + Send,
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded>
    where
        S: lloom::nn::Speculator,
        M: lloom::nn::Probes<S::Probe> + lloom::profile::Shaped,
        M::Cache: CacheRewind + CacheReset,
    {
        if params.eos_ids.is_empty() {
            lloom_tensor::bail!("DecodeSession::decode: eos_ids is empty; a run needs an EOS set");
        }
        params.refuse_on(
            lloom::nn::Body::Window,
            "the speculative body (--dspark, --mtp)",
        )?;
        // The window draws every row with the request's sampler, keyed by
        // the position of the token it draws: any sampler arm, the
        // processors (each row against its own history), the thinking
        // budget (each row's fed token the budget's rule over the rows
        // before it) and the logits digest (over the rows the round keeps)
        // are the K=1 loop's, row for row (`LogitsProcessor::window_draw_device`).
        let mut sampler = params
            .sampler()
            .with_window_budget(params.thinking.as_ref());
        sampler.observe_prompt(prompt_ids);
        // A prompt the cache holds whole re-feeds its last token: the first
        // round needs the target's probe there, which held logits lack.
        let (prepared, turn) = crate::turn::prepare(&mut self.session, prompt_ids, false)?;
        let (checkpoint, checkpoint_at) = self.establish_checkpoint(&prepared, params)?;
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut stream = crate::stream::TextStream::new(params.max_tokens);
        let mut stopped_on_string = false;
        let mut cancelled = false;
        let mut on_tokens = |batch: &[u32]| -> Result<()> {
            if stop.load(std::sync::atomic::Ordering::Acquire) {
                return Ok(());
            }
            let (chunk, stopped) = stream.publish(batch, &mut detokenize, &params.stop_strings)?;
            if let Some(chunk) = chunk {
                if !on_fragment(&chunk) {
                    stream.refuse_last();
                    cancelled = true;
                    stop.store(true, std::sync::atomic::Ordering::Release);
                    return Ok(());
                }
            }
            if stopped {
                stopped_on_string = true;
                stop.store(true, std::sync::atomic::Ordering::Release);
            }
            Ok(())
        };
        let mut readback = Blocking::new(&params.eos_ids, &self.device)?;
        // The session's own options, as the K=1 arms take them: the phase
        // drain (`decode.phase_sync`) and the spark the drafter is seeded
        // unit by unit from are this body's knobs; the cadence ones are
        // read by nothing here.
        let opts = DecodeOptions {
            stop: Some(stop.clone()),
            ..self.config.options(
                params.max_tokens,
                ReadbackArm::Blocking,
                params.prefill_tile,
            )
        };
        let k = spec.max_draft();
        let out = self.session.decode_speculative(
            rewind,
            spec,
            &mut readback,
            &mut sampler,
            &prepared.suffix,
            &opts,
            &params.eos_ids,
            &mut on_tokens,
        )?;
        if !cancelled && !stopped_on_string {
            let (rest, stopped) = stream.flush(&mut detokenize, &params.stop_strings)?;
            stopped_on_string |= stopped;
            if let Some(rest) = rest {
                if !on_fragment(&rest) {
                    stream.refuse_last();
                    cancelled = true;
                }
            }
        }
        let stop_reason = if cancelled {
            Stop::Cancelled
        } else if stopped_on_string {
            Stop::StopString
        } else if out.eos {
            Stop::Eos
        } else {
            Stop::Length
        };
        let (text, _) = stream.finish(&params.stop_strings);
        let thinking = params.thinking.as_ref().map(|b| b.report(&out.ids));
        Ok(Decoded {
            text,
            tokens: out.ids,
            stop: stop_reason,
            reuse: prepared.action,
            reused: prepared.reused,
            prefilled: prepared.suffix.len(),
            turn,
            checkpoint,
            checkpoint_at,
            readback: ReadbackArm::Speculative { k },
            logits_sha256: sampler.logits_digest(),
            stats: out.stats,
            thinking,
            logprobs: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reader_eos_batch_stops_the_producer_before_its_hook() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = AtomicBool::new(false);
        let done = AtomicBool::new(true);
        let mut reads = 0;
        let mut hooks = 0;
        drain_ring_reader(
            &stop,
            &done,
            |buf| {
                reads += 1;
                if reads == 1 {
                    buf.extend_from_slice(&[7, 11]);
                }
                (Ok(buf.len()), true)
            },
            |buf| {
                assert_eq!(buf, &[7, 11]);
                assert!(stop.load(Ordering::Acquire), "EOS hook preceded stop");
                hooks += 1;
                Ok(true)
            },
        )?;
        assert_eq!(reads, 3, "normal read, empty read, final drain");
        assert_eq!(hooks, 1);
        Ok(())
    }

    #[test]
    fn reader_earlier_batch_hook_delays_the_eos_read_not_its_stop() -> Result<()> {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::mpsc::sync_channel;
        use std::time::Duration;
        let stop = AtomicBool::new(false);
        let done = AtomicBool::new(true);
        let reads = AtomicUsize::new(0);
        let (entered_tx, entered_rx) = sync_channel(1);
        let (release_tx, release_rx) = sync_channel(1);
        std::thread::scope(|scope| -> Result<()> {
            let (stop, done, reads) = (&stop, &done, &reads);
            let reader = scope.spawn(move || {
                let mut hooks = 0;
                drain_ring_reader(
                    stop,
                    done,
                    |buf| {
                        let ordinal = reads.fetch_add(1, Ordering::AcqRel);
                        match ordinal {
                            0 => {
                                buf.push(7);
                                (Ok(1), false)
                            }
                            1 => {
                                buf.push(11);
                                (Ok(1), true)
                            }
                            _ => (Ok(0), true),
                        }
                    },
                    |buf| {
                        hooks += 1;
                        if buf == [7] {
                            assert!(!stop.load(Ordering::Acquire));
                            entered_tx.send(()).unwrap();
                            release_rx.recv_timeout(Duration::from_secs(10)).unwrap();
                            assert_eq!(reads.load(Ordering::Acquire), 1);
                            assert!(!stop.load(Ordering::Acquire));
                        } else {
                            assert_eq!(buf, &[11]);
                            assert!(stop.load(Ordering::Acquire), "EOS hook preceded stop");
                        }
                        Ok(true)
                    },
                )?;
                assert_eq!(hooks, 2);
                Ok(())
            });
            entered_rx.recv_timeout(Duration::from_secs(10)).unwrap();
            assert_eq!(reads.load(Ordering::Acquire), 1, "EOS remains unread");
            assert!(!stop.load(Ordering::Acquire), "earlier batch invented EOS");
            release_tx.send(()).unwrap();
            reader.join().expect("reader control")
        })?;
        assert_eq!(reads.load(Ordering::Acquire), 4);
        assert!(stop.load(Ordering::Acquire));
        Ok(())
    }

    #[test]
    fn reader_final_drain_retains_delivery_and_error_context() -> Result<()> {
        use std::sync::atomic::{AtomicBool, Ordering};
        let done = AtomicBool::new(true);
        for (fail, eos) in [(false, false), (false, true), (true, false)] {
            let stop = AtomicBool::new(false);
            let mut reads = 0;
            let mut hooks = 0;
            let result = drain_ring_reader(
                &stop,
                &done,
                |buf| {
                    reads += 1;
                    if reads == 1 {
                        return (Ok(0), false);
                    }
                    assert_eq!(reads, 2, "final drain is not polled again");
                    if fail {
                        (
                            Err(RingLap {
                                cursor: 1,
                                published: 10,
                                capacity: 4,
                            }),
                            false,
                        )
                    } else {
                        buf.push(13);
                        (Ok(1), eos)
                    }
                },
                |buf| {
                    hooks += 1;
                    assert_eq!(buf, &[13]);
                    assert_eq!(stop.load(Ordering::Acquire), eos);
                    Ok(false)
                },
            );
            if fail {
                assert!(result
                    .unwrap_err()
                    .to_string()
                    .contains("ring reader final drain:"));
            } else {
                result?;
            }
            assert_eq!(hooks, usize::from(!fail));
            assert_eq!(reads, 2);
            assert!(stop.load(Ordering::Acquire));
        }
        Ok(())
    }

    #[test]
    fn reader_shared_drain_cancels_on_read_hook_error_and_unwind() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let done = AtomicBool::new(true);
        for failure in 0..3 {
            let stop = AtomicBool::new(false);
            let outcome = std::panic::catch_unwind(|| {
                drain_ring_reader(
                    &stop,
                    &done,
                    |buf| {
                        if failure == 0 {
                            (
                                Err(RingLap {
                                    cursor: 1,
                                    published: 10,
                                    capacity: 4,
                                }),
                                false,
                            )
                        } else {
                            buf.push(7);
                            (Ok(1), false)
                        }
                    },
                    |_| {
                        assert_ne!(failure, 0, "a failed read reached the hook");
                        if failure == 1 {
                            lloom_tensor::bail!("injected hook error");
                        }
                        panic!("injected hook panic");
                    },
                )
            });
            match failure {
                0 => assert!(outcome
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("ring reader:")),
                1 => assert!(outcome
                    .unwrap()
                    .unwrap_err()
                    .to_string()
                    .contains("hook error")),
                _ => assert!(outcome.is_err()),
            }
            assert!(stop.load(Ordering::Acquire));
        }
    }

    #[test]
    fn reader_error_and_panic_cancel_the_producer_before_join() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let stop = AtomicBool::new(false);
        let err: Result<()> =
            stop_on_reader_exit(&stop, || lloom_tensor::bail!("injected detokenizer error"));
        assert!(err.unwrap_err().to_string().contains("detokenizer"));
        assert!(
            stop.load(Ordering::Acquire),
            "reader returned but producer is live"
        );
        stop.store(false, Ordering::Release);
        let panicked = std::panic::catch_unwind(|| {
            stop_on_reader_exit(&stop, || panic!("injected reader panic"))
        });
        assert!(panicked.is_err());
        assert!(stop.load(Ordering::Acquire), "unwinding left producer live");
    }

    /// The pool keys on the EOS *set*: a run that names the ids in another
    /// order, or twice, reuses the ring built for the first.
    #[test]
    fn the_pool_keys_on_the_eos_set() {
        assert_eq!(RingPool::key(&[7, 2, 7, 2, 9]), vec![2, 7, 9]);
        assert_eq!(RingPool::key(&[9, 2, 7]), RingPool::key(&[2, 7, 9]));
        assert_ne!(RingPool::key(&[2, 7]), RingPool::key(&[2, 7, 9]));
        assert!(RingPool::new().is_empty());
    }

    /// The decode session drives any `CausalLM`. Reachability the way §3.1c
    /// asks: a real end-to-end run needs weights (that is the `lfm2-smoke`
    /// gate — a `lloom` example, run in this PR's evidence), so what a unit
    /// test proves is that the seam is *callable* against the production model
    /// — if `lfm2::Model` did not satisfy `CausalLM`, this would not compile.
    #[allow(dead_code)]
    fn drives<M: CausalLM>()
    where
        M::Cache: CacheRewind + CacheReset,
    {
    }

    #[test]
    fn decode_session_drives_lfm2() {
        drives::<lloom::models::lfm2::Model<lloom::profile::CrownExact>>();
    }

    /// `Auto` takes the crown ring for the device-resident samplers and the
    /// blocking arm for the ones that need a host sort (`Ring::check_sampler`),
    /// and never leaves `Auto` in the resolved value a consumer reports.
    #[test]
    fn auto_resolves_by_sampler() {
        let argmax = Sampling::ArgMax;
        let temp = Sampling::All { temperature: 0.6 };
        let top_p = Sampling::TopP {
            p: 0.9,
            temperature: 0.6,
        };
        let crown = RingArm::CROWN;
        assert_eq!(
            ReadbackArm::Auto.resolve(&argmax, crown),
            ReadbackArm::CROWN
        );
        assert_eq!(ReadbackArm::Auto.resolve(&temp, crown), ReadbackArm::CROWN);
        assert_eq!(
            ReadbackArm::Auto.resolve(&top_p, crown),
            ReadbackArm::Blocking
        );
        assert_eq!(
            ReadbackArm::Blocking.resolve(&argmax, crown),
            ReadbackArm::Blocking
        );
        let ring = ReadbackArm::Ring(RingArm {
            lookahead: 0,
            commit_steps: 1,
            adaptive: false,
            ..RingArm::CROWN
        });
        assert_eq!(ring.resolve(&top_p, crown), ring);
        assert_ne!(ReadbackArm::CROWN.name(), ReadbackArm::Blocking.name());
        assert!(ReadbackArm::CROWN.name().contains("lookahead=1"));
    }

    /// The stop reasons a caller matches on are exhaustive and distinct.
    #[test]
    fn stop_variants_are_distinct() {
        assert_ne!(Stop::Eos, Stop::Length);
        assert_ne!(Stop::StopString, Stop::Length);
        assert_ne!(Stop::Eos, Stop::StopString);
        assert_ne!(Stop::Cancelled, Stop::StopString);
    }

    /// Requests in a row through one sequence and one ring pool, as `lloom
    /// serve` carries them (`from_parts` / `into_parts`, `app::lfm2`).
    mod served {
        use super::*;
        use crate::options::{DecodeConfig, ReadbackConfig};
        use lloom::nn::{Model, VarBuilder};
        use lloom_tensor::{DType, Tensor};

        const VOCAB: usize = 64;
        const EOS: u32 = 42;

        fn next_of(tok: u32) -> u32 {
            (tok * 7 + 3) % VOCAB as u32
        }

        /// The ids a request from `start` returns, and whether it ends on EOS.
        fn chain(start: u32, max: usize) -> (Vec<u32>, bool) {
            let mut ids = Vec::new();
            let mut tok = start;
            while ids.len() < max {
                tok = next_of(tok);
                if tok == EOS {
                    return (ids, true);
                }
                ids.push(tok);
            }
            (ids, false)
        }

        fn text_of(ids: &[u32]) -> String {
            ids.iter().map(|t| format!("{t} ")).collect()
        }

        /// A position counter as the cache.
        struct Pos(usize);

        impl CacheReset for Pos {
            fn reset(&mut self) {
                self.0 = 0;
            }
        }

        impl CacheRewind for Pos {
            fn rewind_horizon(&self) -> usize {
                self.0
            }
            fn rewind(&mut self, n: usize) -> Result<()> {
                self.0 -= n;
                Ok(())
            }
        }

        /// A decoder whose logits peak at `next_of(last token)`, device-side.
        struct Chain {
            table: Tensor,
            eye: Tensor,
            forwards: std::sync::atomic::AtomicUsize,
        }

        impl Chain {
            fn new(device: &Device) -> Result<Self> {
                let table: Vec<f32> = (0..VOCAB as u32).map(|t| next_of(t) as f32).collect();
                Ok(Self {
                    table: Tensor::from_vec(table, VOCAB, device)?,
                    eye: Tensor::eye(VOCAB, DType::F32, device)?,
                    forwards: std::sync::atomic::AtomicUsize::new(0),
                })
            }
        }

        impl Model for Chain {
            type Config = ();
            fn from_weights(_vb: VarBuilder, _cfg: &()) -> Result<Self> {
                lloom_tensor::bail!("Chain has no weights")
            }
        }

        impl CausalLM for Chain {
            type Cache = Pos;
            fn new_cache(&self, _cfg: &(), _dev: &Device) -> Result<Pos> {
                Ok(Pos(0))
            }
            fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
                Ok(tokens.clone())
            }
            fn forward_embeds(
                &self,
                embeds: &Tensor,
                pos: usize,
                cache: &mut Pos,
            ) -> Result<Tensor> {
                assert_eq!(
                    cache.0, pos,
                    "forward at {pos} against a cache of {}",
                    cache.0
                );
                let (_b, seq) = embeds.dims2()?;
                self.forwards
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                cache.0 += seq;
                let last = embeds.narrow(1, seq - 1, 1)?.flatten_all()?;
                let next = self.table.index_select(&last, 0)?.to_dtype(DType::U32)?;
                self.eye.index_select(&next, 0)
            }
            fn vocab_size(&self) -> usize {
                VOCAB
            }
        }

        fn devices() -> Vec<Device> {
            let mut v = vec![Device::Cpu];
            #[cfg(feature = "metal")]
            v.push(
                Device::new_metal(0).expect("the `metal` feature is on and no Metal device opened"),
            );
            v
        }

        /// One request: on the sequence and pool `held` carries over from the
        /// previous request, or on a fresh session. Returns the reply, the
        /// text its fragments streamed, and the parts for the next request.
        fn request(
            model: &Chain,
            device: &Device,
            config: &DecodeConfig,
            held: Option<(Sequence<Pos>, RingPool)>,
            prompt: &[u32],
            max_tokens: usize,
        ) -> Result<(Decoded, String, (Sequence<Pos>, RingPool))> {
            let (reply, streamed, _, parts) =
                request_streamed(model, device, config, held, prompt, max_tokens)?;
            Ok((reply, streamed, parts))
        }

        /// Who streamed a reply: a ring reader calls the fragment hook from
        /// its own thread and nothing else calls it, the loop's drain from
        /// the caller's thread. So the thread the fragments arrived on is
        /// which of the two streamed, and `None` is a reply that streamed
        /// nothing.
        #[derive(Debug, Clone, Copy, PartialEq)]
        enum StreamedBy {
            Reader,
            Loop,
        }

        fn streamed_by(
            caller: std::thread::ThreadId,
            seen: &mut Option<StreamedBy>,
            mixed: &mut bool,
        ) {
            let now = if std::thread::current().id() == caller {
                StreamedBy::Loop
            } else {
                StreamedBy::Reader
            };
            *mixed |= seen.is_some_and(|s| s != now);
            *seen = Some(now);
        }

        /// [`request`], and who streamed it.
        #[allow(clippy::type_complexity)]
        fn request_streamed(
            model: &Chain,
            device: &Device,
            config: &DecodeConfig,
            held: Option<(Sequence<Pos>, RingPool)>,
            prompt: &[u32],
            max_tokens: usize,
        ) -> Result<(
            Decoded,
            String,
            Option<StreamedBy>,
            (Sequence<Pos>, RingPool),
        )> {
            let session = match held {
                Some((seq, rings)) => DecodeSession::from_parts(model, seq, device.clone(), rings),
                None => DecodeSession::new(model, Pos(0), device.clone(), 0),
            };
            let mut session = session.with_config(config.clone());
            let params = DecodeParams {
                max_tokens,
                eos_ids: vec![EOS],
                readback: config.readback.arm(),
                ..DecodeParams::default()
            };
            let mut streamed = String::new();
            let caller = std::thread::current().id();
            let (mut by, mut mixed) = (None, false);
            let reply = session.decode(
                prompt,
                |ids: &[u32]| Ok(text_of(ids)),
                &params,
                |frag: &str| {
                    streamed_by(caller, &mut by, &mut mixed);
                    streamed.push_str(frag)
                },
            )?;
            if mixed {
                lloom_tensor::bail!("the reader and the loop both streamed one reply");
            }
            Ok((reply, streamed, by, session.into_parts()))
        }

        /// Production decode integration: a final pending tokenizer run
        /// is flushed on both blocking and fixed-cadence batch delivery.
        #[test]
        fn served_terminal_pending_text_matches_the_full_detokenizer() -> Result<()> {
            let dev = Device::Cpu;
            let model = Chain::new(&dev)?;
            let ids = chain(3, 3).0;
            assert_eq!(ids.len(), 3);
            let terminal = *ids.last().unwrap();
            let detok = |xs: &[u32]| -> Result<String> {
                Ok(xs
                    .iter()
                    .map(|id| {
                        if *id == terminal {
                            "\u{FFFD}".to_string()
                        } else {
                            format!("<{id}>")
                        }
                    })
                    .collect())
            };
            let expected = detok(&ids)?;
            for spelling in ["blocking", "ring-fixed"] {
                for stop in [None, Some(expected.clone())] {
                    let mut config = DecodeConfig::default();
                    ReadbackConfig::apply_spelling(&mut config.readback, spelling).unwrap();
                    let mut session = DecodeSession::new(&model, Pos(0), dev.clone(), 0)
                        .with_config(config.clone());
                    let params = DecodeParams {
                        max_tokens: 3,
                        eos_ids: vec![u32::MAX],
                        readback: config.readback.arm(),
                        stop_strings: stop.iter().cloned().collect(),
                        ..DecodeParams::default()
                    };
                    let mut delivered = String::new();
                    let out = session.decode(&[3], detok, &params, |s| delivered.push_str(s))?;
                    let want = if stop.is_some() {
                        ""
                    } else {
                        expected.as_str()
                    };
                    assert_eq!(out.tokens, ids, "{spelling}: sampled ids unchanged");
                    assert_eq!(out.text, want, "{spelling}: full tokenizer oracle");
                    assert_eq!(delivered, want, "{spelling}: delivered frontier");
                    assert_eq!(
                        out.stop,
                        if stop.is_some() {
                            Stop::StopString
                        } else {
                            Stop::Length
                        }
                    );
                }
            }
            Ok(())
        }

        /// A real reader-thread detokenization error cancels the producer,
        /// rather than letting its finite safety budget be the stop signal.
        /// Count model forwards at the error, not elapsed time or an Err exit.
        #[cfg(feature = "metal")]
        #[test]
        fn a_reader_detokenizer_error_bounds_extra_forwards_and_next_request() -> Result<()> {
            use std::sync::atomic::{AtomicUsize, Ordering};
            let dev = Device::new_metal(0)?;
            let model = Chain::new(&dev)?;
            let mut config = DecodeConfig::default();
            ReadbackConfig::apply_spelling(&mut config.readback, "ring-nonblocking").unwrap();
            config.readback.eos_gate = false;
            let mut session =
                DecodeSession::new(&model, Pos(0), dev, 0).with_config(config.clone());
            let params = DecodeParams {
                max_tokens: 256,
                eos_ids: vec![u32::MAX],
                readback: config.readback.arm(),
                ..DecodeParams::default()
            };
            let at_error = AtomicUsize::new(usize::MAX);
            let mut calls = 0;
            let error = session
                .decode(
                    &[3],
                    |ids| {
                        calls += 1;
                        if calls == 3 {
                            at_error
                                .store(model.forwards.load(Ordering::Acquire), Ordering::Release);
                            lloom_tensor::bail!("injected reader detokenizer failure");
                        }
                        Ok(text_of(ids))
                    },
                    &params,
                    |_| {},
                )
                .unwrap_err();
            assert!(error
                .to_string()
                .contains("injected reader detokenizer failure"));
            let at_error = at_error.load(Ordering::Acquire);
            assert_ne!(at_error, usize::MAX, "the reader failure fired");
            let extra = model.forwards.load(Ordering::Acquire) - at_error;
            assert!(
                extra <= 64,
                "reader failure left producer running: {extra} forwards after error"
            );
            session.reset();
            let next = DecodeParams {
                max_tokens: 6,
                readback: ReadbackArm::Blocking,
                ..params
            };
            let reply = session.decode(&[5], |ids| Ok(text_of(ids)), &next, |_| {})?;
            assert_eq!(reply.tokens, chain(5, 6).0, "the next request's stream");
            eprintln!("READER-ERROR additional-forwards={extra}; next-request=reference");
            Ok(())
        }

        /// Each request on a pooled ring streams its own run, and nothing of
        /// the earlier requests': the text its fragments carried equals the
        /// text of its ids, which equal the chain from its prompt — what the
        /// same request reads on a fresh session. On `lloom serve`'s default
        /// arm (the adaptive commit, whose reader streams, and the EOS gate)
        /// and the same arm ungated, on the reader-owned fixed-cadence arms
        /// (whose loop retracts every EOS), and on the blocking arm, after
        /// requests that ended on EOS and requests that did not. The reader
        /// is Metal's; on the CPU the ring arms drain on the loop's cadence
        /// and the test holds them to the same references.
        #[test]
        fn each_request_on_a_pooled_ring_streams_only_its_own_run() -> Result<()> {
            // (name, spelling, stream reader, EOS gate)
            let arms: [(&str, Option<&str>, bool, bool); 6] = [
                ("default", None, false, true),
                ("default --eos-gate=false", None, false, false),
                ("ring-la0 --stream-reader", Some("ring-la0"), true, true),
                ("ring-fixed --stream-reader", Some("ring-fixed"), true, true),
                ("ring", Some("ring"), false, true),
                ("blocking", Some("blocking"), false, true),
            ];
            // (prompt, max_tokens): ends on EOS, on length, EOS twice in a
            // row, then length after an EOS.
            let requests: [(&[u32], usize); 5] = [
                (&[5, 1], 64),
                (&[5, 0], 24),
                (&[9, 1], 64),
                (&[1], 64),
                (&[0], 24),
            ];
            for device in devices() {
                let model = Chain::new(&device)?;
                for (arm, spelling, reader, gate) in arms {
                    let mut config = DecodeConfig::default();
                    if let Some(s) = spelling {
                        ReadbackConfig::apply_spelling(&mut config.readback, s)
                            .map_err(lloom_tensor::Error::msg)?;
                    }
                    config.readback.stream_reader = reader;
                    config.readback.eos_gate = gate;
                    let mut held = None;
                    for (i, (prompt, max)) in requests.into_iter().enumerate() {
                        let (want, hit) = chain(*prompt.last().expect("a prompt"), max);
                        let (fresh, fresh_streamed, _) =
                            request(&model, &device, &config, None, prompt, max)?;
                        let (got, streamed, parts) =
                            request(&model, &device, &config, held.take(), prompt, max)?;
                        held = Some(parts);
                        let at = format!("{device:?} {arm} request {i} ({})", got.readback.name());
                        assert_eq!(fresh.tokens, want, "{at}: the fresh session's ids");
                        assert_eq!(
                            fresh_streamed,
                            text_of(&want),
                            "{at}: the fresh session's text"
                        );
                        assert_eq!(got.tokens, want, "{at}: the ids");
                        assert_eq!(streamed, text_of(&want), "{at}: the streamed text");
                        assert_eq!(got.text, streamed, "{at}: the reply's text");
                        assert_eq!(
                            got.stop,
                            if hit { Stop::Eos } else { Stop::Length },
                            "{at}: why it stopped"
                        );
                    }
                }
            }
            Ok(())
        }

        /// `DecodeSession::reconcile` takes a run's overrun back before the
        /// next request reads the position — the image path's reuse check
        /// (`app::lfm2::Engine::complete_images_until`) compares it with the
        /// token log. After a request that ended on EOS with forwards queued
        /// past it (the gated flow's up to two windows on Metal, the
        /// lookahead's one on the CPU), the position equals the token log,
        /// the cache kept: the next request extends it.
        #[test]
        fn reconcile_brings_the_position_back_to_the_token_log() -> Result<()> {
            for device in devices() {
                let model = Chain::new(&device)?;
                let config = DecodeConfig::default();
                let (reply, _, (seq, rings)) =
                    request(&model, &device, &config, None, &[5, 1], 64)?;
                assert_eq!(reply.stop, Stop::Eos, "{device:?}");
                let mut s = DecodeSession::from_parts(&model, seq, device.clone(), rings)
                    .with_config(config.clone());
                let (pos, kept) = (s.pos(), s.tokens().len());
                assert!(
                    pos > kept,
                    "{device:?}: no overrun to reconcile ({pos} vs {kept})"
                );
                s.reconcile();
                assert_eq!((s.pos(), s.tokens().len()), (kept, kept), "{device:?}");
                s.reconcile();
                assert_eq!(s.pos(), kept, "{device:?}: a second call moved it");
                let mut prompt = vec![5, 1];
                prompt.extend_from_slice(&reply.tokens);
                prompt.push(2);
                let (next, _, _) =
                    request(&model, &device, &config, Some(s.into_parts()), &prompt, 8)?;
                assert_eq!(next.reuse, ReuseAction::Extend, "{device:?}");
                assert_eq!(next.reused, kept, "{device:?}");
            }
            Ok(())
        }

        /// The ids a greedy request from `prompt` returns under `p`: the
        /// chain's one-hot logits through vLLM's bias and penalties, over the
        /// prompt and the history so far, then the lowest-index argmax.
        fn penalized_chain(prompt: &[u32], p: &Processors, max: usize) -> (Vec<u32>, bool) {
            let mut ids: Vec<u32> = Vec::new();
            let mut last = *prompt.last().expect("a prompt");
            while ids.len() < max {
                let mut x = vec![0f32; VOCAB];
                x[next_of(last) as usize] = 1.0;
                for &(id, b) in &p.logit_bias {
                    x[id as usize] += b;
                }
                for (i, xi) in x.iter_mut().enumerate() {
                    let n = ids.iter().filter(|&&t| t as usize == i).count() as f32;
                    let r = p.repetition_penalty;
                    if r != 1.0 && (n > 0.0 || prompt.contains(&(i as u32))) {
                        *xi *= if *xi > 0.0 {
                            (1.0 / r) as f32
                        } else {
                            r as f32
                        };
                    }
                    *xi -= n * p.frequency_penalty as f32;
                    if n > 0.0 {
                        *xi -= p.presence_penalty as f32;
                    }
                }
                let mut tok = 0;
                for i in 1..VOCAB {
                    if x[i] > x[tok] {
                        tok = i;
                    }
                }
                let tok = tok as u32;
                if tok == EOS {
                    return (ids, true);
                }
                ids.push(tok);
                last = tok;
            }
            (ids, false)
        }

        /// Greedy with processors on every readback arm, the default flow
        /// with and without the fused publish among them: the ids equal the
        /// penalized reference and the stream equals the ids. The fused
        /// publish draws the raw argmax in its own kernel, so a request with
        /// processors must not take it — on Metal's default arm it would
        /// return the plain chain.
        #[test]
        fn penalties_reach_every_readback_arm() -> Result<()> {
            let p = Processors {
                repetition_penalty: 1.25,
                presence_penalty: 1.5,
                frequency_penalty: 0.25,
                min_p: 0.0,
                logit_bias: vec![(EOS, -2.0), (11, 0.125)],
            };
            let prompt: &[u32] = &[5, 1];
            const MAX: usize = 96;
            let (want, hit) = penalized_chain(prompt, &p, MAX);
            assert_ne!(
                (want.clone(), hit),
                chain(1, MAX),
                "the processors changed nothing on this fixture"
            );
            // Past the chain's cycle the history penalties pick the tokens, not
            // the bias alone: a run that never advanced its history differs.
            let bias_only = Processors {
                logit_bias: p.logit_bias.clone(),
                ..Processors::default()
            };
            assert_ne!(
                want,
                penalized_chain(prompt, &bias_only, MAX).0,
                "the history penalties changed nothing on this fixture"
            );
            let mut arms: Vec<(String, DecodeConfig)> =
                vec![("default".into(), DecodeConfig::default())];
            let mut unfused = DecodeConfig::default();
            unfused.readback.fused_publish = false;
            arms.push(("default --fused-publish=false".into(), unfused));
            for s in ReadbackConfig::SPELLINGS {
                let mut c = DecodeConfig::default();
                ReadbackConfig::apply_spelling(&mut c.readback, s)
                    .map_err(lloom_tensor::Error::msg)?;
                arms.push((s.to_string(), c));
            }
            for device in devices() {
                let model = Chain::new(&device)?;
                for (arm, config) in &arms {
                    let mut session = DecodeSession::new(&model, Pos(0), device.clone(), 0)
                        .with_config(config.clone());
                    let params = DecodeParams {
                        max_tokens: MAX,
                        eos_ids: vec![EOS],
                        readback: config.readback.arm(),
                        processors: p.clone(),
                        ..DecodeParams::default()
                    };
                    let mut streamed = String::new();
                    let got = session.decode(
                        prompt,
                        |ids: &[u32]| Ok(text_of(ids)),
                        &params,
                        |frag: &str| streamed.push_str(frag),
                    )?;
                    let at = format!("{device:?} {arm} ({})", got.readback.name());
                    assert_eq!(got.tokens, want, "{at}: the ids");
                    assert_eq!(streamed, text_of(&want), "{at}: the streamed text");
                    assert_eq!(got.stop, if hit { Stop::Eos } else { Stop::Length }, "{at}");
                    if device.is_metal() && arm == "default" {
                        assert!(
                            matches!(got.readback, ReadbackArm::Ring(_)),
                            "{at}: processors took the request off the ring"
                        );
                    }
                }
            }
            Ok(())
        }

        /// A thinking budget on every readback arm, the default flow with and
        /// without the fused publish among them. From prompt `[5, 1]` the
        /// chain is `10 9 2 17 58 25 50 33`, then EOS.
        ///
        /// - fires: the block (closed by 20 or 30) is open at the budget of
        ///   2, so steps 2 and 3 feed the tail `5 20` in place of the draws
        ///   `2 38`, and the chain continues from 20;
        /// - a tool call first: 58 closes the block at step 4, before the
        ///   budget of 6, so nothing is forced and the chain is untouched;
        /// - never reached: a budget of 100;
        /// - cut short: a budget of 6 on a run of 3 tokens, which ends with
        ///   the rule still live — the case a budget left armed on a pooled
        ///   ring would force into the next request (its third token).
        ///
        /// The streamed text is the fed ids' (the reader streams the forced
        /// tokens too), the report reads the ids back, and a request without
        /// a budget on the same pooled ring right after is the plain chain.
        #[test]
        fn a_thinking_budget_feeds_its_tail_on_every_readback_arm() -> Result<()> {
            use lloom::generation::{ThinkingBudget, ThinkingReport};
            const MAX: usize = 12;
            let fires = ThinkingBudget::new(2, vec![20, 30], vec![5, 20])?;
            let mut fired_want = vec![10, 9, 5, 20];
            let (rest, rest_eos) = chain(20, MAX - 4);
            fired_want.extend(rest);
            assert_eq!(chain(1, 3).0, vec![10, 9, 2], "the fixture's chain moved");
            let tool = ThinkingBudget::new(6, vec![20, 58], vec![5, 20])?;
            let never = ThinkingBudget::new(100, vec![20, 30], vec![5, 20])?;
            let short = ThinkingBudget::new(6, vec![20, 30], vec![5, 20])?;
            let (plain, plain_eos) = chain(1, MAX);
            assert_eq!(plain, vec![10, 9, 2, 17, 58, 25, 50, 33]);
            let cases = [
                (
                    "fires",
                    MAX,
                    &fires,
                    fired_want.clone(),
                    rest_eos,
                    ThinkingReport {
                        thought: 3,
                        fired: true,
                        closed_by: Some(20),
                    },
                ),
                (
                    "a tool call first",
                    MAX,
                    &tool,
                    plain.clone(),
                    plain_eos,
                    ThinkingReport {
                        thought: 4,
                        fired: false,
                        closed_by: Some(58),
                    },
                ),
                (
                    "never reached",
                    MAX,
                    &never,
                    plain.clone(),
                    plain_eos,
                    ThinkingReport {
                        thought: plain.len(),
                        fired: false,
                        closed_by: None,
                    },
                ),
                (
                    "cut short",
                    3,
                    &short,
                    plain[..3].to_vec(),
                    false,
                    ThinkingReport {
                        thought: 3,
                        fired: false,
                        closed_by: None,
                    },
                ),
            ];
            let mut arms: Vec<(String, DecodeConfig)> =
                vec![("default".into(), DecodeConfig::default())];
            let mut unfused = DecodeConfig::default();
            unfused.readback.fused_publish = false;
            arms.push(("default --fused-publish=false".into(), unfused));
            for s in ReadbackConfig::SPELLINGS {
                let mut c = DecodeConfig::default();
                ReadbackConfig::apply_spelling(&mut c.readback, s)
                    .map_err(lloom_tensor::Error::msg)?;
                arms.push((s.to_string(), c));
            }
            for device in devices() {
                let model = Chain::new(&device)?;
                for (arm, config) in &arms {
                    let mut held: Option<(Sequence<Pos>, RingPool)> = None;
                    for (name, max, budget, want, hit, report) in &cases {
                        let session = match held.take() {
                            Some((seq, rings)) => {
                                DecodeSession::from_parts(&model, seq, device.clone(), rings)
                            }
                            None => DecodeSession::new(&model, Pos(0), device.clone(), 0),
                        };
                        let mut session = session.with_config(config.clone());
                        session.reset();
                        let params = DecodeParams {
                            max_tokens: *max,
                            eos_ids: vec![EOS],
                            readback: config.readback.arm(),
                            thinking: Some((*budget).clone()),
                            ..DecodeParams::default()
                        };
                        let mut streamed = String::new();
                        let got = session.decode(
                            &[5, 1],
                            |ids: &[u32]| Ok(text_of(ids)),
                            &params,
                            |frag: &str| streamed.push_str(frag),
                        )?;
                        let at = format!("{device:?} {arm} {name} ({})", got.readback.name());
                        assert_eq!(&got.tokens, want, "{at}: the ids");
                        assert_eq!(streamed, text_of(want), "{at}: the streamed text");
                        assert_eq!(
                            got.stop,
                            if *hit { Stop::Eos } else { Stop::Length },
                            "{at}"
                        );
                        assert_eq!(got.thinking, Some(*report), "{at}: the report");
                        let parts = session.into_parts();
                        let (plain_got, _, parts) =
                            request(&model, &device, config, Some(parts), &[5, 1], MAX)?;
                        assert_eq!(
                            plain_got.tokens, plain,
                            "{at}: a request without a budget after it"
                        );
                        assert_eq!(plain_got.thinking, None, "{at}");
                        held = Some(parts);
                    }
                }
            }
            Ok(())
        }

        /// The one table, through a request: each body refuses what it
        /// cannot serve by name, the window serves everything, and a
        /// request asking nothing passes every body.
        #[test]
        fn each_body_refuses_by_name_and_the_window_serves_everything() {
            use lloom::nn::Body;
            let mut procs = DecodeParams {
                temperature: Some(0.7),
                ..DecodeParams::default()
            };
            procs.processors.min_p = 0.1;
            let e = procs
                .refuse_on(Body::BatchedStep, "the batch scheduler")
                .unwrap_err()
                .to_string();
            assert!(
                e.starts_with("the batch scheduler: a batched body does not run logits processors"),
                "{e}"
            );
            let hot = DecodeParams {
                temperature: Some(0.7),
                ..DecodeParams::default()
            };
            let e = hot
                .refuse_on(Body::BatchedWindow, "row 0")
                .unwrap_err()
                .to_string();
            assert!(e.contains("verifies by the argmax"), "{e}");
            let sort = DecodeParams {
                temperature: Some(0.7),
                top_p: Some(0.9),
                ..DecodeParams::default()
            };
            let e = sort
                .refuse_on(Body::Step { ring: true }, "ring")
                .unwrap_err()
                .to_string();
            assert!(e.contains("host sort"), "{e}");
            let digest = DecodeParams {
                logits_digest: true,
                ..DecodeParams::default()
            };
            assert!(digest.refuse_on(Body::Step { ring: true }, "ring").is_err());
            for p in [&procs, &hot, &sort, &digest] {
                p.refuse_on(Body::Window, "window").unwrap();
                p.refuse_on(Body::Step { ring: false }, "blocking").unwrap();
            }
            for body in [
                Body::Step { ring: true },
                Body::Step { ring: false },
                Body::Window,
                Body::BatchedStep,
                Body::BatchedWindow,
            ] {
                DecodeParams::default().refuse_on(body, "x").unwrap();
            }
        }

        /// The per-token logprobs are the blocking K=1 body's alone: the ring,
        /// the window and the batched bodies read back ids, never a row, and
        /// each refuses them by name.
        #[test]
        fn the_logprobs_are_served_by_the_blocking_step_alone() {
            use lloom::nn::Body;
            let lp = DecodeParams {
                logprobs: Some(LogprobRequest::default()),
                ..DecodeParams::default()
            };
            lp.refuse_on(Body::Step { ring: false }, "blocking")
                .unwrap();
            for (body, says) in [
                (Body::Step { ring: true }, "they run on the blocking arm"),
                (Body::Window, "the window reads back the ids a round keeps"),
                (
                    Body::BatchedStep,
                    "a batched body does not keep per-token logprobs",
                ),
                (
                    Body::BatchedWindow,
                    "a batched body does not keep per-token logprobs",
                ),
            ] {
                let e = lp.refuse_on(body, "path").unwrap_err().to_string();
                assert!(e.starts_with("path: ") && e.contains(says), "{body:?}: {e}");
            }
        }

        /// One request on a fresh session at `temperature` (and `top_p`), on
        /// the arm the options select: the reply, whatever the sampler drew,
        /// and the text its fragments streamed. (Its one caller is the
        /// `serve`-gated arm sweep.)
        #[cfg(feature = "serve")]
        fn sampled(
            model: &Chain,
            device: &Device,
            config: &DecodeConfig,
            temperature: f64,
            top_p: Option<f64>,
        ) -> Result<(Decoded, String, Option<StreamedBy>)> {
            let mut session =
                DecodeSession::new(model, Pos(0), device.clone(), 0).with_config(config.clone());
            let params = DecodeParams {
                max_tokens: 24,
                temperature: Some(temperature),
                top_p,
                eos_ids: vec![EOS],
                readback: config.readback.arm(),
                ..DecodeParams::default()
            };
            let mut streamed = String::new();
            let caller = std::thread::current().id();
            let (mut by, mut mixed) = (None, false);
            let reply = session.decode(
                &[3],
                |ids: &[u32]| Ok(text_of(ids)),
                &params,
                |frag: &str| {
                    streamed_by(caller, &mut by, &mut mixed);
                    streamed.push_str(frag)
                },
            )?;
            if mixed {
                lloom_tensor::bail!("the reader and the loop both streamed one reply");
            }
            Ok((reply, streamed, by))
        }

        /// Every `--readback` spelling (and none) × the adaptive commit (unset,
        /// on, off) × the lookahead (unset, 0, 1) × the commit budget (unset,
        /// per-token) × the EOS gate (unset, on, off) × the stream reader ×
        /// the fused publish (on, off), resolved through `lloom serve`'s
        /// option path and served the way it serves them: every combination
        /// starts, and every request is answered and streams what it returns.
        /// Every EOS read stamps on a grain its reader can stream
        /// (`docs/dispatch.md` §11.5z-xvi), so nothing here is refused, at
        /// startup or per request.
        ///
        /// Each reply reports the arm that ran: it names the adaptive commit
        /// iff the loop closed a buffer adaptively (the length request runs
        /// past `run_ahead`, so an adaptive run closes at least one), and the
        /// EOS gate armed iff the run was adaptive and the gate is on, fired
        /// iff the run ended on EOS. Where the device carries a reader, that
        /// is also the arm the options resolve to, which is what the load
        /// line renders.
        #[cfg(feature = "serve")]
        #[test]
        fn every_readback_spelling_and_commit_arm_serves() -> Result<()> {
            use clap::Parser;
            #[derive(Parser)]
            struct Cli {
                #[command(flatten)]
                common: crate::app::CommonArgs,
            }
            let mut combos: Vec<Vec<String>> = Vec::new();
            let spellings = std::iter::once(None).chain(ReadbackConfig::SPELLINGS.iter().map(Some));
            for spelling in spellings {
                for adaptive in [None, Some("true"), Some("false")] {
                    for lookahead in [None, Some("0"), Some("1")] {
                        for commit in [None, Some("per-token")] {
                            for gate in [None, Some("true"), Some("false")] {
                                for reader in [false, true] {
                                    let mut a: Vec<String> = Vec::new();
                                    if let Some(s) = spelling {
                                        a.extend(["--readback".into(), s.to_string()]);
                                    }
                                    if let Some(v) = adaptive {
                                        a.push(format!("--adaptive-commit={v}"));
                                    }
                                    if let Some(v) = lookahead {
                                        a.extend(["--lookahead".into(), v.to_string()]);
                                    }
                                    if let Some(v) = commit {
                                        a.extend(["--commit-budget".into(), v.to_string()]);
                                    }
                                    if let Some(v) = gate {
                                        a.push(format!("--eos-gate={v}"));
                                    }
                                    if reader {
                                        a.push("--stream-reader".into());
                                    }
                                    combos.push(a.clone());
                                    a.push("--fused-publish=false".into());
                                    combos.push(a);
                                }
                            }
                        }
                    }
                }
            }
            assert_eq!(combos.len(), 10 * 3 * 3 * 2 * 3 * 2 * 2);
            // What one served combination saw: ran adaptive, was clamped, greedy
            // replies gated, of those fired, replies streamed by a reader, by the loop.
            type Seen = (bool, bool, usize, usize, usize, usize);
            let mut failures: Vec<String> = Vec::new();
            let mut vacuous: Vec<String> = Vec::new();
            let mut refusals = std::collections::BTreeMap::new();
            for device in devices() {
                let model = Chain::new(&device)?;
                // (refused at startup, served, replies that ran adaptive,
                // served combinations whose flags asked for it and did not,
                // greedy replies the EOS gate armed / fired on)
                let (mut refused, mut served, mut adaptive_ran, mut clamped) = (0, 0, 0, 0);
                let (mut gated, mut gate_fired) = (0, 0);
                // replies streamed by a ring reader, by the loop's drain
                let (mut by_reader, mut by_loop) = (0, 0);
                let failed_before = failures.len();
                for argv in &combos {
                    let label = format!("{device:?} [{}]", argv.join(" "));
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || -> Result<std::result::Result<Seen, String>> {
                            let cli = Cli::parse_from(
                                std::iter::once("t".to_string()).chain(argv.iter().cloned()),
                            );
                            let config = match crate::app::resolve_options_uninstalled(
                                &cli.common,
                                &crate::app::no_front,
                            ) {
                                Err(e) => return Ok(Err(e.to_string())),
                                Ok(o) => o.decode,
                            };
                            let (mut ran_adaptive, mut was_clamped) = (false, false);
                            let (mut armed, mut fired) = (0usize, 0usize);
                            let (mut readers, mut loops) = (0usize, 0usize);
                            // Where the device maps its ring, a reader streams
                            // every ring arm that runs the adaptive commit or
                            // asks for one; nothing refuses or withholds it.
                            let mut check_streamer =
                                |at: &str, arm: ReadbackArm, by: Option<StreamedBy>| {
                                    let want = if device.is_metal()
                                        && matches!(arm, ReadbackArm::Ring(r)
                                            if r.adaptive || config.readback.stream_reader)
                                    {
                                        StreamedBy::Reader
                                    } else {
                                        StreamedBy::Loop
                                    };
                                    assert_eq!(by, Some(want), "{at}: who streamed the reply");
                                    match want {
                                        StreamedBy::Reader => readers += 1,
                                        StreamedBy::Loop => loops += 1,
                                    }
                                };
                            let mut held = None;
                            for (prompt, max) in [(&[5u32, 1][..], 64), (&[0u32][..], 24)] {
                                let (want, hit) = chain(*prompt.last().expect("a prompt"), max);
                                let (got, streamed, by, parts) = request_streamed(
                                    &model,
                                    &device,
                                    &config,
                                    held.take(),
                                    prompt,
                                    max,
                                )
                                .unwrap_or_else(|e| {
                                    panic!("{label}: a greedy request was refused: {e}")
                                });
                                held = Some(parts);
                                let at = format!("{label} ({}) max {max}", got.readback.name());
                                assert_eq!(got.tokens, want, "{at}: the ids");
                                assert_eq!(streamed, text_of(&want), "{at}: the streamed text");
                                if !streamed.is_empty() {
                                    check_streamer(&at, got.readback, by);
                                }
                                assert_eq!(
                                    got.stop,
                                    if hit { Stop::Eos } else { Stop::Length },
                                    "{at}: why it stopped"
                                );
                                let ran =
                                    got.stats.adaptive_commits + got.stats.run_ahead_closes > 0;
                                let says =
                                    matches!(got.readback, ReadbackArm::Ring(r) if r.adaptive);
                                assert!(
                                    !ran || says,
                                    "{at}: ran the adaptive commit and does not say so"
                                );
                                let gate_on = says && config.readback.eos_gate;
                                assert_eq!(
                                    got.stats.eos_gate.is_some(),
                                    gate_on,
                                    "{at}: the EOS gate armed={}, adaptive={says} eos_gate={}",
                                    got.stats.eos_gate.is_some(),
                                    config.readback.eos_gate
                                );
                                if let Some(report) = got.stats.eos_gate {
                                    assert_eq!(
                                        report.fired, hit,
                                        "{at}: the gate fired={}",
                                        report.fired
                                    );
                                    armed += 1;
                                    fired += report.fired as usize;
                                }
                                if !hit {
                                    assert_eq!(says, ran, "{at}: the name says adaptive={says}, the loop ran adaptive={ran}");
                                    ran_adaptive = ran;
                                    let asked = config.readback.adaptive_commit;
                                    was_clamped = asked
                                        && !ran
                                        && matches!(got.readback, ReadbackArm::Ring(_));
                                }
                                if device.is_metal() {
                                    let resolved = config
                                        .readback
                                        .arm()
                                        .resolve(&Sampling::ArgMax, config.readback.ring_arm());
                                    assert_eq!(got.readback, resolved, "{at}: the reply reports another arm than the options resolve to");
                                }
                            }
                            let host_sampler =
                                !matches!(config.readback.arm(), ReadbackArm::Ring(_));
                            for (t, p) in [(0.6, None), (0.6, Some(0.9))] {
                                if p.is_some() && !host_sampler {
                                    continue;
                                }
                                let (got, streamed, by) = sampled(&model, &device, &config, t, p)
                                    .unwrap_or_else(|e| {
                                        panic!("{label}: a request at temperature {t} top_p {p:?} was refused: {e}")
                                    });
                                let at = format!(
                                    "{label} ({}) temperature {t} top_p {p:?}",
                                    got.readback.name()
                                );
                                if device.is_metal() {
                                    let resolved = config.readback.arm().resolve(
                                        &DecodeParams::sampling_of(Some(t), p, None, false),
                                        config.readback.ring_arm(),
                                    );
                                    assert_eq!(got.readback, resolved, "{at}: the reply reports another arm than the options resolve to");
                                }
                                if !streamed.is_empty() {
                                    check_streamer(&at, got.readback, by);
                                }
                                assert!(
                                    got.tokens.len() <= 24,
                                    "{at}: {} tokens past max_tokens 24",
                                    got.tokens.len()
                                );
                                assert_eq!(
                                    streamed,
                                    text_of(&got.tokens),
                                    "{at}: the streamed text"
                                );
                                assert_eq!(got.text, streamed, "{at}: the reply's text");
                            }
                            Ok(Ok((
                                ran_adaptive,
                                was_clamped,
                                armed,
                                fired,
                                readers,
                                loops,
                            )))
                        },
                    ));
                    match outcome {
                        Ok(Ok(Err(e))) => {
                            refused += 1;
                            *refusals.entry(e).or_insert(0usize) += 1;
                        }
                        Ok(Ok(Ok((r, c, g, f, rd, lp)))) => {
                            served += 1;
                            adaptive_ran += r as usize;
                            clamped += c as usize;
                            gated += g;
                            gate_fired += f;
                            by_reader += rd;
                            by_loop += lp;
                        }
                        Ok(Err(e)) => failures.push(format!("{label}: {e}")),
                        Err(p) => failures.push(
                            p.downcast_ref::<String>()
                                .cloned()
                                .or_else(|| p.downcast_ref::<&str>().map(|m| m.to_string()))
                                .unwrap_or_default(),
                        ),
                    }
                }
                eprintln!("{device:?}: {served} served, {refused} refused at startup, {adaptive_ran} ran adaptive, {clamped} asked for it and ran the fixed cadence, {gated} greedy replies gated, {gate_fired} fired, {by_reader} replies streamed by a reader, {by_loop} by the loop");
                let failed = failures.len() - failed_before;
                eprintln!("{device:?}: {failed} failed");
                for f in &failures[failed_before..] {
                    eprintln!("FAIL {f}");
                }
                assert_eq!(served + refused + failed, combos.len());
                for (e, n) in std::mem::take(&mut refusals) {
                    eprintln!("REFUSED x{n}: {e}");
                    failures.push(format!(
                        "{device:?}: {n} combinations refused at startup: {e}"
                    ));
                }
                if device.is_metal() && (adaptive_ran == 0 || clamped == 0) {
                    vacuous.push(format!(
                        "{device:?}: the attribution check saw one arm only"
                    ));
                }
                if device.is_metal() && (by_reader == 0 || by_loop == 0) {
                    vacuous.push(format!(
                        "{device:?}: the streamer check saw {by_reader} reader replies, {by_loop} loop replies"
                    ));
                }
                if device.is_metal() && (gated == 0 || gate_fired == 0 || gated == gate_fired) {
                    vacuous.push(format!(
                        "{device:?}: the gate check saw {gated} gated replies, {gate_fired} fired"
                    ));
                }
            }
            assert!(
                failures.is_empty(),
                "{} combination runs failed (listed above)",
                failures.len()
            );
            assert!(vacuous.is_empty(), "{vacuous:?}");
            Ok(())
        }
    }
}
