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
use lloom::generation::readback::RING_HISTORY_SLOTS;
use lloom::generation::{Blocking, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::nn::{
    CacheReset, CacheRewind, CausalLM, DecodeOptions, DecodeOut, DecodeStats, IdentityProposer,
    ReuseAction, Sequence, Session,
};
use lloom_tensor::{Device, Result};

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
    /// §8.1). Needs a CPU-mapped ring, lookahead 1, an EOS read other than
    /// the flag and a per-token stamp; the fixed cadence where it has not
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
    /// commit encodes the next forward before anything is read back and its
    /// reader streams the ring's per-token stamp, so it needs one forward of
    /// lookahead, an EOS read that does not read each token back (every ring
    /// read but the flag, `Session::decode`'s refusals), and a stamp per
    /// token: the poll read's, or the fused argmax·eos·stamp publish, which
    /// runs for the argmax sampler with `fused_publish` on (`fused`).
    pub fn why_not_adaptive(&self, fused: bool) -> Option<&'static str> {
        if self.lookahead != 1 {
            Some("the adaptive commit needs lookahead 1")
        } else if matches!(self.eos, EosRead::Blocking) {
            Some("the `ring` EOS read (the flag) reads each token back per step")
        } else {
            self.why_not_stream_reader(fused)
        }
    }

    /// Why a ring reader cannot stream this arm, or `None` if it can: the
    /// reader streams what a per-token stamp certifies. The poll read stamps
    /// every token on any cadence; the other reads do only through the
    /// adaptive commit's fused publish (see [`Self::why_not_adaptive`]) —
    /// otherwise the flag read stamps nothing and the deferred reads stamp
    /// once per commit window.
    pub fn why_not_stream_reader(&self, fused: bool) -> Option<&'static str> {
        match self.eos {
            EosRead::Poll(_) => None,
            _ if self.adaptive && fused => None,
            EosRead::Blocking => {
                Some("the `ring` EOS read (the flag) stamps no token for a reader")
            }
            _ => Some(
                "a deferred EOS read stamps once per commit window, not per token, outside the \
                 fused publish (the argmax sampler, `fused_publish` on)",
            ),
        }
    }

    /// The arm that runs: this one, on the fixed cadence where it cannot
    /// carry the adaptive commit.
    pub fn resolved(self, fused: bool) -> RingArm {
        RingArm {
            adaptive: self.adaptive && self.why_not_adaptive(fused).is_none(),
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
    pub max_tokens: usize,
    /// End-of-sequence ids; the first one sampled stops the run.
    pub eos_ids: Vec<u32>,
    /// Text substrings that stop the run once they appear in the output.
    pub stop_strings: Vec<String>,
    /// Which readback arm the run takes; [`ReadbackArm::Auto`] by default.
    pub readback: ReadbackArm,
    /// The TILE: the most prompt tokens one prefill forward takes
    /// (`DecodeOptions::prefill_tile`, lloom #728). `0` is one forward over
    /// the whole prompt, whose activation working set grows with the prompt
    /// (~3.5 MB per token on the crown); a positive tile bounds it to the
    /// tile's worth whatever the prompt. The model's `conv_prefill_ctx` must
    /// be on for a tiled prefill to compute what one pass does.
    pub prefill_tile: usize,
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

    /// The sampler arm of this request.
    pub fn sampling(&self) -> Sampling {
        Self::sampling_of(self.temperature, self.top_p, self.top_k, self.gumbel)
    }

    /// The sampler of this request, seeded.
    pub fn sampler(&self) -> LogitsProcessor {
        LogitsProcessor::from_sampling(self.seed, self.sampling())
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
            max_tokens: 512,
            eos_ids: Vec::new(),
            stop_strings: Vec::new(),
            readback: ReadbackArm::Auto,
            prefill_tile: 0,
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
    /// The readback arm that ran: resolved (never [`ReadbackArm::Auto`]), and
    /// on the fixed cadence wherever the adaptive commit did not run.
    pub readback: ReadbackArm,
    /// The loop's own counters — prefill submit time, per-step host time,
    /// commits, ring engagement — so a consumer that reports throughput reads
    /// the loop's clock rather than bracketing the call.
    pub stats: DecodeStats,
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
/// A [`Ring`] is a wrapping history of `RING_HISTORY_SLOTS` plus the consumer
/// slack (32 MiB by default, ~2 050 zero-filled segments): building one is
/// ~35 ms of `Tensor::zeros` and a `synchronize` (issue #702, §11.5y's TTFT
/// finding), and its capacity is its live window, not the generation, so one
/// ring serves every run — mapped, which is also the fast drain (a history
/// sized past `RING_HISTORY_SLOTS` is not CPU-mapped and snapshots on every
/// drain). What a ring cannot change after construction is its EOS set
/// (uploaded once, read by the `eos_index` kernel), so the pool is keyed on
/// it; a serving binary holds two — the checkpoint's EOS ids and the
/// never-EOS set its throughput cells use.
///
/// Built on first use, or ahead of it by [`RingPool::prepare`] at load, so no
/// request pays the construction inside its own latency.
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

    /// Positions currently in the cache.
    pub fn pos(&self) -> usize {
        self.session.pos()
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
        let opts = self
            .config
            .options(params.max_tokens.max(1), ReadbackArm::Blocking, params.prefill_tile);
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
        let prepared = self.session.prepare(prompt_ids)?;

        let mut sampler = params.sampler();
        let arm = params
            .readback
            .resolve(sampler.sampling(), self.config.readback.ring_arm());
        if let Err(e) = self.config.check() {
            lloom_tensor::bail!("DecodeSession::decode: {e}");
        }
        let mut spec = IdentityProposer;

        let mut stream = crate::stream::TextStream::new(params.max_tokens);
        let mut stopped_on_string = false;
        let mut cancelled = false;

        // The streaming hook, shared by every arm: decode only the
        // not-yet-emitted tokens, hold a run ending mid-character, and let a
        // stop string end the run (`TextStream`).
        let mut on_tokens = |batch: &[u32]| -> Result<bool> {
            if let Some(chunk) = stream.push(batch, &mut detokenize)? {
                if !on_fragment(&chunk) {
                    cancelled = true;
                    return Ok(false);
                }
            }
            if stream.hit_stop(&params.stop_strings) {
                stopped_on_string = true;
                return Ok(false);
            }
            Ok(true)
        };

        let (out, ran): (DecodeOut, ReadbackArm) = match arm {
            ReadbackArm::Blocking => {
                let mut readback = Blocking::new(&params.eos_ids, &self.device)?;
                let opts = self
                    .config
                    .options(params.max_tokens, ReadbackArm::Blocking, params.prefill_tile);
                let out = self.session.decode_until(
                    &mut spec,
                    &mut readback,
                    &mut sampler,
                    &prepared.suffix,
                    &opts,
                    &mut on_tokens,
                )?;
                (out, ReadbackArm::Blocking)
            }
            ReadbackArm::Ring(ring) => {
                // The fused publish, which lets a non-poll read stamp per
                // token, runs for the argmax sampler only.
                let fused = self.config.readback.fused_publish
                    && matches!(sampler.sampling(), Sampling::ArgMax);
                let ring = ring.resolved(fused);
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
                // The adaptive arm needs the ring's reader (a CPU-mapped ring);
                // without one it is the fixed cadence, reported as such. The
                // reader alone (`--stream-reader`) streams per token on the
                // fixed cadence where the ring stamps per token, and a request
                // whose sampler rules out the fused publish takes no reader
                // (`RingArm::why_not_stream_reader`; the options refuse at
                // startup what no request could stream). The pool's ring
                // served earlier runs and its stamp runs across them, so the
                // reader starts at this run's first token.
                let reader = if (adaptive || self.config.readback.stream_reader)
                    && ring.why_not_stream_reader(fused).is_none()
                {
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
                            let worker = s.spawn(move || -> Result<()> {
                                let mut buf: Vec<u32> = Vec::with_capacity(64);
                                loop {
                                    buf.clear();
                                    if reader.ended() {
                                        stop_flag.store(true, std::sync::atomic::Ordering::Release);
                                    }
                                    match reader.read_new(&mut buf) {
                                        Ok(0) => {
                                            if stream_done.load(std::sync::atomic::Ordering::Acquire) {
                                                buf.clear();
                                                if let Ok(n) = reader.read_new(&mut buf) {
                                                    if n > 0 && !hook(&buf)? {
                                                        stop_flag.store(
                                                            true,
                                                            std::sync::atomic::Ordering::Release,
                                                        );
                                                    }
                                                }
                                                return Ok(());
                                            }
                                            std::thread::sleep(std::time::Duration::from_micros(200));
                                        }
                                        Ok(_) => {
                                            if !hook(&buf)? {
                                                stop_flag.store(true, std::sync::atomic::Ordering::Release);
                                            }
                                        }
                                        Err(lap) => {
                                            lloom_tensor::bail!("ring reader: {lap}");
                                        }
                                    }
                                }
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

        Ok(Decoded {
            text,
            tokens: out.ids,
            stop,
            reuse: prepared.action,
            reused: prepared.reused,
            prefilled: prepared.suffix.len(),
            readback: ran,
            stats: out.stats,
        })
    }
}

impl<'m, M: CausalLM> DecodeSession<'m, M> {
    /// [`Self::decode_until`] through the speculative body: `spec` drafts a
    /// block per round and the target verifies it
    /// ([`Session::decode_speculative`], `design/docs/speculation.md`
    /// §10.2t). Greedy by construction — the verify accepts the target's
    /// argmax — so a temperature on `params` is refused rather than served
    /// as argmax under a sampled label. The fragment hook's `false` ends the
    /// run after the round in flight (`DecodeOptions::stop`), reported as
    /// [`Stop::Cancelled`]; a fully cached prompt cannot be resumed by this
    /// body (its verify holds no per-position logits) and is refused, which
    /// a caller that resets between requests never meets.
    pub fn decode_speculative_until<S>(
        &mut self,
        spec: &mut S,
        prompt_ids: &[u32],
        mut detokenize: impl FnMut(&[u32]) -> Result<String> + Send,
        params: &DecodeParams,
        mut on_fragment: impl FnMut(&str) -> bool + Send,
    ) -> Result<Decoded>
    where
        S: lloom::nn::Speculator,
        M: lloom::nn::Probes<S::Probe>,
        M::Cache: CacheRewind + CacheReset,
    {
        if params.eos_ids.is_empty() {
            lloom_tensor::bail!("DecodeSession::decode: eos_ids is empty; a run needs an EOS set");
        }
        let mut sampler = params.sampler();
        if sampler.sampling() != &Sampling::ArgMax {
            lloom_tensor::bail!(
                "DecodeSession::decode_speculative: the speculative body decodes greedily (the \
                 verify accepts the target's argmax); a temperature ({:?}) is refused rather \
                 than served as argmax",
                sampler.sampling()
            );
        }
        let prepared = self.session.prepare(prompt_ids)?;
        if prepared.suffix.is_empty() {
            lloom_tensor::bail!(
                "DecodeSession::decode_speculative: the prompt is entirely cached ({:?}) and the \
                 speculative body cannot resume from held logits; reset the session first",
                prepared.action
            );
        }
        let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut stream = crate::stream::TextStream::new(params.max_tokens);
        let mut stopped_on_string = false;
        let mut cancelled = false;
        let mut on_tokens = |batch: &[u32]| -> Result<()> {
            if stop.load(std::sync::atomic::Ordering::Acquire) {
                return Ok(());
            }
            if let Some(chunk) = stream.push(batch, &mut detokenize)? {
                if !on_fragment(&chunk) {
                    cancelled = true;
                    stop.store(true, std::sync::atomic::Ordering::Release);
                    return Ok(());
                }
            }
            if stream.hit_stop(&params.stop_strings) {
                stopped_on_string = true;
                stop.store(true, std::sync::atomic::Ordering::Release);
            }
            Ok(())
        };
        let mut readback = Blocking::new(&params.eos_ids, &self.device)?;
        let opts = DecodeOptions {
            stop: Some(stop.clone()),
            ..DecodeOptions::new(params.max_tokens)
        };
        let k = spec.max_draft();
        let out = self.session.decode_speculative(
            spec,
            &mut readback,
            &mut sampler,
            &prepared.suffix,
            &opts,
            &params.eos_ids,
            &mut on_tokens,
        )?;
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
        Ok(Decoded {
            text,
            tokens: out.ids,
            stop: stop_reason,
            reuse: prepared.action,
            reused: prepared.reused,
            prefilled: prepared.suffix.len(),
            readback: ReadbackArm::Speculative { k },
            stats: out.stats,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        drives::<lloom::models::lfm2::Model>();
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
        assert_eq!(ReadbackArm::Auto.resolve(&argmax, crown), ReadbackArm::CROWN);
        assert_eq!(ReadbackArm::Auto.resolve(&temp, crown), ReadbackArm::CROWN);
        assert_eq!(ReadbackArm::Auto.resolve(&top_p, crown), ReadbackArm::Blocking);
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
        }

        impl Chain {
            fn new(device: &Device) -> Result<Self> {
                let table: Vec<f32> = (0..VOCAB as u32).map(|t| next_of(t) as f32).collect();
                Ok(Self {
                    table: Tensor::from_vec(table, VOCAB, device)?,
                    eye: Tensor::eye(VOCAB, DType::F32, device)?,
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
            let reply = session.decode(
                prompt,
                |ids: &[u32]| Ok(text_of(ids)),
                &params,
                |frag: &str| streamed.push_str(frag),
            )?;
            Ok((reply, streamed, session.into_parts()))
        }

        /// Each request on a pooled ring streams its own run, and nothing of
        /// the earlier requests': the text its fragments carried equals the
        /// text of its ids, which equal the chain from its prompt — what the
        /// same request reads on a fresh session. On `lloom serve`'s default
        /// arm (the adaptive commit, whose reader streams), on the
        /// reader-owned fixed-cadence arms (whose loop retracts every EOS),
        /// and on the blocking arm, after requests that ended on EOS and
        /// requests that did not. The reader is Metal's; on the CPU the ring
        /// arms drain on the loop's cadence and the test holds them to the
        /// same references.
        #[test]
        fn each_request_on_a_pooled_ring_streams_only_its_own_run() -> Result<()> {
            let arms: [(&str, Option<&str>, bool); 5] = [
                ("default", None, false),
                ("ring-la0 --stream-reader", Some("ring-la0"), true),
                ("ring-fixed --stream-reader", Some("ring-fixed"), true),
                ("ring", Some("ring"), false),
                ("blocking", Some("blocking"), false),
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
                for (arm, spelling, reader) in arms {
                    let mut config = DecodeConfig::default();
                    if let Some(s) = spelling {
                        ReadbackConfig::apply_spelling(&mut config.readback, s)
                            .map_err(lloom_tensor::Error::msg)?;
                    }
                    config.readback.stream_reader = reader;
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

        /// One request on a fresh session at `temperature` (and `top_p`), on
        /// the arm the options select: the reply, whatever the sampler drew,
        /// and the text its fragments streamed.
        fn sampled(
            model: &Chain,
            device: &Device,
            config: &DecodeConfig,
            temperature: f64,
            top_p: Option<f64>,
        ) -> Result<(Decoded, String)> {
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
            let reply = session.decode(
                &[3],
                |ids: &[u32]| Ok(text_of(ids)),
                &params,
                |frag: &str| streamed.push_str(frag),
            )?;
            Ok((reply, streamed))
        }

        /// Every `--readback` spelling (and none) × the adaptive commit (unset,
        /// on, off) × the lookahead (unset, 0, 1) × the commit budget (unset,
        /// per-token) × the EOS gate × the stream reader × the fused publish
        /// (on, off), resolved through `lloom serve`'s option path and served
        /// the way it serves them: the options are refused at startup naming
        /// the knob the arm cannot carry
        /// (the EOS gate, the stream reader), or every request is answered and
        /// streams what it returns. A request is never where a combination is
        /// refused.
        ///
        /// Each reply reports the arm that ran: it names the adaptive commit
        /// iff the loop closed a buffer adaptively (the length request runs
        /// past `run_ahead`, so an adaptive run closes at least one). Where the
        /// device carries a reader, that is also the arm the options resolve
        /// to, which is what the load line renders.
        #[cfg(feature = "serve")]
        #[test]
        fn every_readback_spelling_and_commit_arm_serves_or_is_refused_at_startup() -> Result<()> {
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
                            for gate in [false, true] {
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
                                    if gate {
                                        a.push("--eos-gate".into());
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
            assert_eq!(combos.len(), 10 * 3 * 3 * 2 * 2 * 2 * 2);
            let mut failures: Vec<String> = Vec::new();
            let mut vacuous: Vec<String> = Vec::new();
            let mut refusals = std::collections::BTreeMap::new();
            for device in devices() {
                let model = Chain::new(&device)?;
                // (refused at startup, served, replies that ran adaptive,
                // served combinations whose flags asked for it and did not)
                let (mut refused, mut served, mut adaptive_ran, mut clamped) = (0, 0, 0, 0);
                let failed_before = failures.len();
                for argv in &combos {
                    let label = format!("{device:?} [{}]", argv.join(" "));
                    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                        || -> Result<std::result::Result<(bool, bool), String>> {
                            let cli = Cli::parse_from(
                                std::iter::once("t".to_string()).chain(argv.iter().cloned()),
                            );
                            let config = match crate::app::resolve_options_uninstalled(&cli.common)
                            {
                                Err(e) => {
                                    // Only an explicit knob the arm cannot carry is
                                    // refused, and the refusal names it.
                                    let e = e.to_string();
                                    let names = |flag: &str, key: &str| {
                                        argv.iter().any(|a| a == flag) && e.contains(key)
                                    };
                                    assert!(
                                        names("--eos-gate", "eos_gate")
                                            || names("--stream-reader", "stream_reader"),
                                        "{label}: refused without naming a knob it set: {e}"
                                    );
                                    return Ok(Err(e));
                                }
                                Ok(o) => o.decode,
                            };
                            let (mut ran_adaptive, mut was_clamped) = (false, false);
                            let mut held = None;
                            for (prompt, max) in [(&[5u32, 1][..], 64), (&[0u32][..], 24)] {
                                let (want, hit) = chain(*prompt.last().expect("a prompt"), max);
                                let (got, streamed, parts) =
                                    request(&model, &device, &config, held.take(), prompt, max)
                                        .unwrap_or_else(|e| {
                                            panic!("{label}: a greedy request was refused: {e}")
                                        });
                                held = Some(parts);
                                let at = format!("{label} ({}) max {max}", got.readback.name());
                                assert_eq!(got.tokens, want, "{at}: the ids");
                                assert_eq!(streamed, text_of(&want), "{at}: the streamed text");
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
                                let (got, streamed) = sampled(&model, &device, &config, t, p)
                                    .unwrap_or_else(|e| {
                                        panic!("{label}: a request at temperature {t} top_p {p:?} was refused: {e}")
                                    });
                                let at = format!(
                                    "{label} ({}) temperature {t} top_p {p:?}",
                                    got.readback.name()
                                );
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
                            Ok(Ok((ran_adaptive, was_clamped)))
                        },
                    ));
                    match outcome {
                        Ok(Ok(Err(e))) => {
                            refused += 1;
                            *refusals.entry(e).or_insert(0usize) += 1;
                        }
                        Ok(Ok(Ok((r, c)))) => {
                            served += 1;
                            adaptive_ran += r as usize;
                            clamped += c as usize;
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
                eprintln!("{device:?}: {served} served, {refused} refused at startup, {adaptive_ran} ran adaptive, {clamped} asked for it and ran the fixed cadence");
                let failed = failures.len() - failed_before;
                eprintln!("{device:?}: {failed} failed");
                for f in &failures[failed_before..] {
                    eprintln!("FAIL {f}");
                }
                assert_eq!(served + refused + failed, combos.len());
                if served == 0 || refused == 0 {
                    vacuous.push(format!("{device:?}: one side of the matrix is empty"));
                }
                if device.is_metal() && (adaptive_ran == 0 || clamped == 0) {
                    vacuous.push(format!(
                        "{device:?}: the attribution check saw one arm only"
                    ));
                }
            }
            for (e, n) in &refusals {
                eprintln!("REFUSED x{n}: {e}");
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
