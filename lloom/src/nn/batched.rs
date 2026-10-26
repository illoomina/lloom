//! The step-wise batched decoder: [`Session::decode_batched`]'s body as a
//! state machine a scheduler drives (issue #148,
//! `design/docs/batched-decode-design.md` §11).
//!
//! `decode_batched_admitted` runs a CLOSED batch — every row known before the
//! first step, each with an arrival step. A serving loop is the open form of
//! the same thing: a request arrives from a socket while other rows are
//! mid-reply, joins the batch at the next step, and leaves it at its EOS
//! without waiting for anyone. The loop body is one and the same — the CHUNK
//! feeding a pending row's prompt as extra ragged rows of the live rows' own
//! forward, the one device argmax over `[B, vocab]`, the lookahead's staged
//! readback, the retirement at EOS or the cap — so it lives here once, as
//! [`BatchDecoder`], and the closed driver is a loop over it.
//!
//! Rows are `R: BorrowMut<Session>`: an owned [`Session`] for a scheduler
//! that keeps its sequences across requests, a `&mut Session` for the closed
//! driver over a caller's slice. A finished row is handed back through
//! [`BatchDecoder::take_finished`] with its [`DecodeOut`].
//!
//! # Sampling
//!
//! A batch whose live rows are all argmax is sampled on the device in one
//! `argmax` over `[B, vocab]`, and the lookahead feeds the device ids straight
//! into the next forward (the closed driver's incumbent path). A row admitted
//! with a temperature sampler takes the HOST arm: the step's logits come to
//! the host and every live row is sampled by its own seeded
//! [`LogitsProcessor`], so a row's draw depends on its own seed and its own
//! logits and never on which rows sit beside it (`DESIGN.md` §2.3.3 #7); the
//! next forward is then fed the host ids, which is the `lookahead = 0` order
//! whatever the option says. Mixed batches take the host arm for every row.

use std::borrow::{Borrow, BorrowMut};

use crate::generation::{HotThread, LogitsProcessor, Sampling};
use crate::nn::decode::{
    begin_pass, end_executor_step, end_pass, ladder_width, ragged_layout, tile_widths,
    DecodeOptions, DecodeOut, DecodeStats, PassShape, PassWidths, StepPacer,
};
use crate::nn::model::{BatchedCausalLM, RaggedRow, Row};
use crate::nn::speculator::Session;
use lloom_tensor::{Device, Result, Tensor};

/// A row's handle: its slot in the decoder and the admission that filled it,
/// so a handle kept past [`BatchDecoder::take_finished`] cannot name the
/// slot's next occupant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RowId {
    pub(crate) slot: usize,
    pub(crate) generation: u64,
}

impl RowId {
    /// The slot index, stable for the row's life.
    pub fn slot(&self) -> usize {
        self.slot
    }
}

/// One sequence's cursor through the batch.
struct Lane {
    produced: usize,
    eos: bool,
    done: bool,
    /// Ended by the caller: nothing more of its prompt is fed and nothing
    /// more streamed. `done` is set with it; a `done` row that is NOT
    /// cancelled (a zero cap) still has its prompt fed before it retires.
    cancelled: bool,
    /// Whether the row has been retired: its bookkeeping written to the
    /// sequence and its `DecodeOut` queued for `take_finished`.
    retired: bool,
    all_ids: Vec<u32>,
    pos_before: usize,
    /// The logits after the last fed token, when the row ended with every
    /// returned id fed (the lookahead arm at `max_tokens`).
    last_logits: Option<Tensor>,
    stats: DecodeStats,
    /// The step this row's prompt may start being fed (its arrival).
    arrival: u64,
    /// Prompt tokens fed so far; the row joins the batch once this reaches
    /// `prompt_len` (or at once, on a resume with held logits).
    prompt_off: usize,
    /// Whether the row has joined the batch — its logits are in `logits`.
    joined: bool,
    /// The wall clock when the row's first prompt chunk was fed, for
    /// `stats.prefill_ns` (the submit time from the first chunk to the last).
    prefill_start: Option<std::time::Instant>,
    /// The ids that end THIS row, and its own cap (`0` = produce nothing):
    /// the decoder's defaults unless the row was admitted with its own.
    eos_ids: Vec<u32>,
    max_tokens: usize,
}

impl Lane {
    fn reached(&self, produced: usize) -> bool {
        self.max_tokens != 0 && produced >= self.max_tokens
    }
}

struct Slot<'m, M: BatchedCausalLM, R: BorrowMut<Session<'m, M>>> {
    session: R,
    prompt: Vec<u32>,
    lane: Lane,
    sampler: LogitsProcessor,
    generation: u64,
    _model: std::marker::PhantomData<&'m M>,
}

/// What one [`BatchDecoder::step`] did.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StepReport {
    /// Rows whose next token this step sampled and fed.
    pub live: usize,
    /// Prompt pieces fed as ragged rows of this step's forward.
    pub pieces: usize,
    /// Prompt tokens those pieces carried.
    pub piece_tokens: usize,
    /// Rows retired by this step (EOS, cap or cancellation).
    pub retired: usize,
    /// Rows still pending admission — arrived but their prompt not yet fed.
    pub pending: usize,
    /// Live rows this step held back while the pieces were fed
    /// ([`PrefillDecode::Pause`]): neither sampled nor fed.
    pub held: usize,
}

/// One [`BatchDecoder::step`]'s host time by stage, in nanoseconds: read in
/// a debug or `profile` build, zero in release, where the clock reads are
/// compiled out (the single-sequence loop's `StepPhases` gate). The stages
/// partition the step, so their sum is its host time
/// (`batched-decode-design.md` §13).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BatchStepPhases {
    /// Admission: a whole prompt fed at its arrival, the step's pieces
    /// chosen, the held rows set aside, the resumed rows joined.
    pub admit_ns: u64,
    /// The batch argmax over `[B, vocab]`, encode only.
    pub argmax_ns: u64,
    /// Staging the ids' readback, encode only.
    pub stage_ns: u64,
    /// The forward's host setup: its input and every row's `prepare_pass`.
    pub prepare_ns: u64,
    /// The pacer's wait before the forward: zero unless the host was its
    /// in-flight bound ahead of the GPU.
    pub pace_ns: u64,
    /// The forward pass, encode only (`begin_pass` to `end_pass`).
    pub encode_ns: u64,
    /// The pacer after the forward and the rows' positions.
    pub after_ns: u64,
    /// The readback: the host blocked until the step's ids (or, on the host
    /// sampling arm, its logits, sampled there) are on the host.
    pub wait_ns: u64,
    /// Streaming the ids to their rows, the EOS test and the cap.
    pub stream_ns: u64,
    /// The rest: the kept rows' logits, the pieces' bookkeeping, the
    /// executor's step end, the next live set, the retirements.
    pub tail_ns: u64,
}

impl BatchStepPhases {
    /// The step's host time: every stage summed.
    pub fn total_ns(&self) -> u64 {
        self.admit_ns
            + self.argmax_ns
            + self.stage_ns
            + self.prepare_ns
            + self.pace_ns
            + self.encode_ns
            + self.after_ns
            + self.wait_ns
            + self.stream_ns
            + self.tail_ns
    }
}

/// Laps the host clock between a step's stages in a debug or `profile`
/// build; nothing in release.
struct StageClock {
    #[cfg(any(debug_assertions, feature = "profile"))]
    last: std::time::Instant,
}

impl StageClock {
    #[inline]
    fn start() -> Self {
        Self {
            #[cfg(any(debug_assertions, feature = "profile"))]
            last: std::time::Instant::now(),
        }
    }

    /// Charge the time since the previous lap to `into`.
    #[inline]
    fn lap(&mut self, into: &mut u64) {
        #[cfg(any(debug_assertions, feature = "profile"))]
        {
            let now = std::time::Instant::now();
            *into = into.saturating_add(now.duration_since(self.last).as_nanos() as u64);
            self.last = now;
        }
        #[cfg(not(any(debug_assertions, feature = "profile")))]
        let _ = into;
    }
}

/// What the live rows do in a step that feeds prompt pieces
/// (`batched-decode-design.md` §11.8).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum PrefillDecode {
    /// Every live row's next token rides the pieces' forward: one weight
    /// sweep for both.
    #[default]
    Ride,
    /// A live row that has streamed a token waits for the pieces: the
    /// forward is the pieces and the rows still owing their first token, so
    /// a row's first token is never held behind another row's prompt.
    Pause,
}

impl PrefillDecode {
    pub fn parse(s: &str) -> std::result::Result<Self, String> {
        match s {
            "ride" => Ok(Self::Ride),
            "pause" => Ok(Self::Pause),
            other => Err(format!(
                "unknown prefill decode `{other}`; want `ride` or `pause`"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Ride => "ride",
            Self::Pause => "pause",
        }
    }
}

impl std::str::FromStr for PrefillDecode {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, String> {
        Self::parse(s)
    }
}

impl std::fmt::Display for PrefillDecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

lloom_tensor::serde_via_str!(PrefillDecode);

/// An admission the driver refused: the row it did not take, with the
/// reason.
///
/// Admission is transactional. A driver takes the row's session only once
/// it holds a slot for it, so a refusal — a sampler the body does not run,
/// a sequence in a state no row can start from, an empty prepared suffix,
/// no drafter slot left — hands the session back to whoever owns the
/// sequences. Without that, every refusal past the caller's own validation
/// costs a scheduler one sequence of its pool, and enough of them leave
/// valid requests failing admission with nothing live.
///
/// `?` on an admission still yields the error alone, so a caller with
/// nothing to hand the session back to is unchanged.
pub struct Rejected<R> {
    /// The session the driver did not take.
    pub session: R,
    pub error: lloom_tensor::Error,
}

impl<R> Rejected<R> {
    /// The reason, dropping the returned session.
    pub fn into_error(self) -> lloom_tensor::Error {
        self.error
    }
}

impl<R> From<Rejected<R>> for lloom_tensor::Error {
    fn from(rejected: Rejected<R>) -> Self {
        rejected.error
    }
}

impl<R> std::fmt::Debug for Rejected<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Rejected")
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

impl<R> std::fmt::Display for Rejected<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.error.fmt(f)
    }
}

/// An admission's outcome: the row's handle, or the refusal with the
/// session the driver did not take.
pub type Admission<R> = std::result::Result<RowId, Rejected<R>>;

/// What a scheduler drives, step by step: rows admitted at any step,
/// cancelled, retired with their outcome — the K=1 batched loop
/// ([`BatchDecoder`]) or the speculative one
/// ([`SpecBatchDecoder`](crate::nn::SpecBatchDecoder)), one surface
/// (`batched-decode-design.md` §12.4).
pub trait RowDriver<'m, M: BatchedCausalLM + 'm, R: BorrowMut<Session<'m, M>>> {
    /// Admit a row, or refuse it and hand its session back
    /// ([`Rejected`]).
    #[allow(clippy::too_many_arguments)]
    fn admit_with_eos(
        &mut self,
        session: R,
        prompt: Vec<u32>,
        sampling: Sampling,
        seed: u64,
        eos_ids: &[u32],
        max_tokens: usize,
    ) -> Admission<R>;
    fn step(
        &mut self,
        on_tokens: &mut dyn FnMut(RowId, &[u32]) -> Result<()>,
    ) -> Result<StepReport>;
    fn cancel(&mut self, id: RowId) -> bool;
    fn take_finished(&mut self) -> Vec<(RowId, R, DecodeOut)>;
    fn drain_rows(&mut self) -> Vec<R>;
    fn is_idle(&self) -> bool;
    fn rows(&self) -> usize;
    fn live_rows(&self) -> usize;
    fn pending_rows(&self) -> usize;
    fn row_ids(&self) -> Vec<RowId>;
    fn row_source(&self, id: RowId) -> Option<(&[u32], &M::Cache, bool)>;
    fn row_prospective_log(&self, id: RowId) -> Option<Vec<u32>>;
    fn holds(&self, id: RowId) -> bool;
    /// The last step's host time by stage; zero where the driver keeps none.
    fn step_phases(&self) -> BatchStepPhases {
        BatchStepPhases::default()
    }
}

impl<'m, M: BatchedCausalLM + 'm, R: BorrowMut<Session<'m, M>>> RowDriver<'m, M, R>
    for BatchDecoder<'m, M, R>
{
    fn admit_with_eos(
        &mut self,
        session: R,
        prompt: Vec<u32>,
        sampling: Sampling,
        seed: u64,
        eos_ids: &[u32],
        max_tokens: usize,
    ) -> Admission<R> {
        BatchDecoder::admit_with_eos(self, session, prompt, sampling, seed, eos_ids, max_tokens)
    }
    fn step(
        &mut self,
        on_tokens: &mut dyn FnMut(RowId, &[u32]) -> Result<()>,
    ) -> Result<StepReport> {
        BatchDecoder::step(self, on_tokens)
    }
    fn cancel(&mut self, id: RowId) -> bool {
        BatchDecoder::cancel(self, id)
    }
    fn take_finished(&mut self) -> Vec<(RowId, R, DecodeOut)> {
        BatchDecoder::take_finished(self)
    }
    fn drain_rows(&mut self) -> Vec<R> {
        BatchDecoder::drain_rows(self)
    }
    fn is_idle(&self) -> bool {
        BatchDecoder::is_idle(self)
    }
    fn rows(&self) -> usize {
        BatchDecoder::rows(self)
    }
    fn live_rows(&self) -> usize {
        BatchDecoder::live_rows(self)
    }
    fn pending_rows(&self) -> usize {
        BatchDecoder::pending_rows(self)
    }
    fn row_ids(&self) -> Vec<RowId> {
        BatchDecoder::row_ids(self)
    }
    fn row_source(&self, id: RowId) -> Option<(&[u32], &M::Cache, bool)> {
        BatchDecoder::row_source(self, id)
    }
    fn row_prospective_log(&self, id: RowId) -> Option<Vec<u32>> {
        BatchDecoder::row_prospective_log(self, id)
    }
    fn holds(&self, id: RowId) -> bool {
        BatchDecoder::holds(self, id)
    }
    fn step_phases(&self) -> BatchStepPhases {
        self.phases
    }
}

/// The batched decode loop, one step at a time.
pub struct BatchDecoder<'m, M: BatchedCausalLM, R: BorrowMut<Session<'m, M>>> {
    opts: DecodeOptions,
    eos_ids: Vec<u32>,
    device: Device,
    slots: Vec<Option<Slot<'m, M, R>>>,
    free: Vec<usize>,
    /// The live rows' slots, in the order the logits' rows are in; `logits`
    /// is `[live.len(), vocab]` (unset while nothing is live).
    live: Vec<usize>,
    logits: Option<Tensor>,
    step: u64,
    generation: u64,
    /// Retired rows awaiting `take_finished`, with their outcome.
    finished: Vec<(usize, DecodeOut)>,
    /// The batch argmax, for the device arm (argmax needs no seed).
    device_sampler: LogitsProcessor,
    hot_thread: HotThread,
    /// The bound on how many forwards the host may have in flight, as the
    /// single-sequence loop bounds it (`opts.pacer_depth()`, the floor of 2).
    ///
    /// Without it a step with no live row — the prompt-only phase of a
    /// closed batch, every tile of a whole-prompt admission — has no readback
    /// to wait on, and the host encodes every tile back to back: their
    /// activations are all `pending` at once (§9.5o), bounded only by
    /// Metal's per-queue command-buffer cap. Measured before this bound
    /// (`issue-148-raw/ctx128k/ladder-tiled.log`): a 512-token tile at
    /// 8192 → 65536 tokens peaked at 11–13 GB of `pending`, 17–20 GB
    /// device-allocated, against the ~4 GB two tiles' worth the tile is
    /// meant to bound (#728 §2).
    pacer: StepPacer,
    /// How many pending rows' chunks one step may carry (at least 1).
    prefill_batch: usize,
    /// What the live rows do in a step that feeds pieces.
    prefill_decode: PrefillDecode,
    /// The last step's host time by stage (debug and `profile` builds).
    phases: BatchStepPhases,
}

impl<'m, M: BatchedCausalLM, R: BorrowMut<Session<'m, M>>> BatchDecoder<'m, M, R> {
    /// An empty batch under `opts`, stopping rows on `eos_ids`.
    ///
    /// Refused: an empty EOS set (a row would never terminate), a lookahead
    /// past one (a row runs at most one forward past its end).
    pub fn new(opts: DecodeOptions, eos_ids: &[u32], device: Device) -> Result<Self> {
        if eos_ids.is_empty() {
            lloom_tensor::bail!(
                "decode_batched: eos_ids is empty, so the loop would never terminate"
            );
        }
        if opts.lookahead > 1 {
            lloom_tensor::bail!(
                "decode_batched: lookahead {} — a row runs at most one forward past its end",
                opts.lookahead
            );
        }
        let depth = opts.pacer_depth();
        Ok(Self {
            opts,
            eos_ids: eos_ids.to_vec(),
            device,
            slots: Vec::new(),
            free: Vec::new(),
            live: Vec::new(),
            logits: None,
            step: 0,
            generation: 0,
            finished: Vec::new(),
            device_sampler: LogitsProcessor::from_sampling(0, Sampling::ArgMax),
            hot_thread: HotThread::claim(),
            pacer: StepPacer::new(depth),
            prefill_batch: 1,
            prefill_decode: PrefillDecode::Ride,
            phases: BatchStepPhases::default(),
        })
    }

    /// The last step's host time by stage; zero in release.
    pub fn last_step_phases(&self) -> BatchStepPhases {
        self.phases
    }

    /// Feed up to `n` pending rows' chunks in one step under a chunk budget:
    /// the step's prompt budget becomes `n × prefill_chunk` tokens, each
    /// row's piece still at most one chunk (and one tile), so the pending
    /// prompts are prefilled together in one pass of `m = Σ pieces` rather
    /// than one prompt's chunk a step. `n = 1` (the default) is that one
    /// chunk; `0` is taken as 1. No effect without a chunk budget.
    pub fn with_prefill_batch(mut self, n: usize) -> Self {
        self.prefill_batch = n.max(1);
        self
    }

    /// What the live rows do in a step that feeds pieces (the default rides).
    pub fn with_prefill_decode(mut self, arm: PrefillDecode) -> Self {
        self.prefill_decode = arm;
        self
    }

    /// The options every row runs under.
    pub fn options(&self) -> &DecodeOptions {
        &self.opts
    }

    /// The pending rows' chunks one step may carry.
    pub fn prefill_batch(&self) -> usize {
        self.prefill_batch
    }

    /// What the live rows do in a step that feeds pieces.
    pub fn prefill_decode(&self) -> PrefillDecode {
        self.prefill_decode
    }

    /// The step counter: how many [`Self::step`] calls have run.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Rows whose next token the next step samples.
    pub fn live_rows(&self) -> usize {
        self.live.len()
    }

    /// Rows admitted and not yet retired, live or pending.
    pub fn rows(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|s| !s.lane.retired)
            .count()
    }

    /// Rows arrived and not yet joined: their prompts are still being fed.
    pub fn pending_rows(&self) -> usize {
        self.slots
            .iter()
            .flatten()
            .filter(|s| !s.lane.joined && !s.lane.cancelled)
            .count()
    }

    /// Take every row out, live or pending, without a step: the sessions in
    /// slot order, their lanes discarded — for a caller recovering from a
    /// failed step whose decoder state is not to be trusted. What the rows
    /// produced is lost; the sessions' caches hold whatever was fed.
    pub fn drain_rows(&mut self) -> Vec<R> {
        let mut out = Vec::new();
        for slot in self.slots.iter_mut() {
            if let Some(s) = slot.take() {
                out.push(s.session);
            }
        }
        self.free.clear();
        self.slots.clear();
        self.live.clear();
        self.logits = None;
        self.finished.clear();
        out
    }

    /// Whether a step would do nothing: no row is live and none is pending.
    pub fn is_idle(&self) -> bool {
        self.live.is_empty() && self.pending_rows() == 0
    }

    /// Admit a row now: its prompt is fed from the next step on. See
    /// [`Self::admit_at`].
    pub fn admit(
        &mut self,
        session: R,
        prompt: Vec<u32>,
        sampling: Sampling,
        seed: u64,
    ) -> Admission<R> {
        let step = self.step;
        self.admit_at(session, prompt, sampling, seed, step)
    }

    /// [`Self::admit`] with the row's own EOS set and cap in place of the
    /// decoder's (a serving row's `ignore_eos`, its `max_tokens`). An empty
    /// `eos_ids` is refused.
    pub fn admit_with_eos(
        &mut self,
        session: R,
        prompt: Vec<u32>,
        sampling: Sampling,
        seed: u64,
        eos_ids: &[u32],
        max_tokens: usize,
    ) -> Admission<R> {
        if eos_ids.is_empty() {
            return Err(Rejected {
                session,
                error: lloom_tensor::Error::msg(
                    "decode_batched: a row with an empty EOS set would never end",
                ),
            });
        }
        let step = self.step;
        self.admit_row(
            session,
            prompt,
            sampling,
            seed,
            step,
            eos_ids.to_vec(),
            max_tokens,
        )
    }

    /// Admit a row whose prompt may be fed from step `arrival` on (the closed
    /// driver's shape; a scheduler passes the current step).
    ///
    /// An empty prompt resumes from the sequence's held logits. Refused: an
    /// unreconciled overrun on the sequence, an empty prompt with nothing to
    /// resume from, a sampler the host arm cannot run per row (top-{k,p}
    /// need no more than a seed, so every arm is admitted; the refusal is
    /// only for a sequence in a state no row can start from).
    pub fn admit_at(
        &mut self,
        session: R,
        prompt: Vec<u32>,
        sampling: Sampling,
        seed: u64,
        arrival: u64,
    ) -> Admission<R> {
        let eos_ids = self.eos_ids.clone();
        let max_tokens = self.opts.max_tokens;
        self.admit_row(
            session, prompt, sampling, seed, arrival, eos_ids, max_tokens,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn admit_row(
        &mut self,
        mut session: R,
        prompt: Vec<u32>,
        sampling: Sampling,
        seed: u64,
        arrival: u64,
        eos_ids: Vec<u32>,
        max_tokens: usize,
    ) -> Admission<R> {
        // Every refusal is decided before the slot is filled, and hands the
        // session back ([`Rejected`]): what the row would own is not taken
        // until the admission commits.
        let refused = crate::nn::decode::refuse(
            crate::nn::decode::Body::BatchedStep,
            &crate::nn::decode::Asks::sampling(&sampling),
        )
        .err()
        .or_else(|| {
            let s: &mut Session<'m, M> = session.borrow_mut();
            if s.seq.overrun > 0 {
                return Some(lloom_tensor::Error::msg(format!(
                    "decode_batched: the sequence's cache is {} positions past its last token \
                     (an earlier run's overrun); `Session::reconcile` or `Session::reset` first",
                    s.seq.overrun
                )));
            }
            if prompt.is_empty() && s.seq.last_logits.is_none() {
                return Some(lloom_tensor::Error::msg(
                    "decode_batched: an empty prompt and no held logits to resume from",
                ));
            }
            None
        });
        if let Some(error) = refused {
            return Err(Rejected { session, error });
        }
        self.generation += 1;
        let generation = self.generation;
        let pos_before = session.borrow_mut().seq.pos;
        let lane = Lane {
            produced: 0,
            eos: false,
            done: max_tokens == 0,
            cancelled: false,
            retired: false,
            all_ids: Vec::with_capacity(max_tokens),
            pos_before,
            last_logits: None,
            stats: DecodeStats {
                step_ns: Vec::with_capacity(max_tokens),
                hot_thread: self.hot_thread.report(),
                ..DecodeStats::default()
            },
            arrival,
            prompt_off: 0,
            joined: false,
            prefill_start: None,
            eos_ids,
            max_tokens,
        };
        let slot = Slot {
            session,
            prompt,
            lane,
            sampler: LogitsProcessor::from_sampling(seed, sampling),
            generation,
            _model: std::marker::PhantomData,
        };
        let idx = match self.free.pop() {
            Some(i) => {
                self.slots[i] = Some(slot);
                i
            }
            None => {
                self.slots.push(Some(slot));
                self.slots.len() - 1
            }
        };
        Ok(RowId {
            slot: idx,
            generation,
        })
    }

    fn slot_of(&self, id: RowId) -> Option<&Slot<'m, M, R>> {
        self.slots
            .get(id.slot)
            .and_then(|s| s.as_ref())
            .filter(|s| s.generation == id.generation)
    }

    /// The rows this decoder holds (live, pending or finished-not-taken).
    pub fn row_ids(&self) -> Vec<RowId> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                s.as_ref().map(|s| RowId {
                    slot: i,
                    generation: s.generation,
                })
            })
            .collect()
    }

    /// What row `id`'s cache holds so far — its token log: a prefix it
    /// reused or shared plus the prompt tokens fed since — with the cache
    /// and whether the whole prompt is in (a row still prefilling has more
    /// to come). For a caller sharing a cached prefix
    /// (`batched-decode-design.md` §12.3).
    pub fn row_source(&self, id: RowId) -> Option<(&[u32], &M::Cache, bool)> {
        let s = self.slot_of(id)?;
        let session: &Session<'m, M> = Borrow::borrow(&s.session);
        let fed = s.lane.prompt_off.min(s.prompt.len());
        Some((
            session.seq.tokens(),
            &session.seq.cache,
            fed == s.prompt.len(),
        ))
    }

    /// The token log row `id`'s cache will hold once its prompt is in: what
    /// it holds now plus the prompt tokens still to feed.
    pub fn row_prospective_log(&self, id: RowId) -> Option<Vec<u32>> {
        let s = self.slot_of(id)?;
        let session: &Session<'m, M> = Borrow::borrow(&s.session);
        let fed = s.lane.prompt_off.min(s.prompt.len());
        let mut log = session.seq.tokens().to_vec();
        log.extend_from_slice(&s.prompt[fed..]);
        Some(log)
    }

    /// Row `id`'s whole prompt.
    pub fn row_prompt(&self, id: RowId) -> Option<&[u32]> {
        self.slot_of(id).map(|s| s.prompt.as_slice())
    }

    /// Whether `id` names a row this decoder still holds.
    pub fn holds(&self, id: RowId) -> bool {
        self.slot_of(id).is_some()
    }

    /// End a row early: it is retired at the end of the step it is in (or
    /// at the next step if pending), its stream as produced so far. A `false`
    /// return means the row was not held.
    pub fn cancel(&mut self, id: RowId) -> bool {
        match self
            .slots
            .get_mut(id.slot)
            .and_then(|s| s.as_mut())
            .filter(|s| s.generation == id.generation)
        {
            Some(s) => {
                s.lane.done = true;
                s.lane.cancelled = true;
                true
            }
            None => false,
        }
    }

    /// The retired rows, in retirement order, with their sequences and
    /// outcomes; their slots are free for the next admission.
    pub fn take_finished(&mut self) -> Vec<(RowId, R, DecodeOut)> {
        let mut out = Vec::with_capacity(self.finished.len());
        for (idx, decoded) in std::mem::take(&mut self.finished) {
            let slot = self.slots[idx]
                .take()
                .expect("a retired row holds its slot");
            self.free.push(idx);
            out.push((
                RowId {
                    slot: idx,
                    generation: slot.generation,
                },
                slot.session,
                decoded,
            ));
        }
        out
    }

    /// Without a chunk budget an arrived row's whole remaining prompt is
    /// fed at its arrival step, before that step's decode is encoded (the
    /// tile still bounds each forward). Returns the rows that completed
    /// their prompt, with their logits, in slot order.
    fn admit_whole(&mut self) -> Result<Vec<(usize, Tensor)>> {
        let mut joined = Vec::new();
        let step = self.step;
        let device = self.device.clone();
        for i in 0..self.slots.len() {
            let Some(slot) = self.slots[i].as_mut() else {
                continue;
            };
            let lane = &mut slot.lane;
            if lane.joined || lane.cancelled || lane.arrival > step {
                continue;
            }
            let s: &mut Session<'m, M> = slot.session.borrow_mut();
            if slot.prompt.is_empty() {
                // A resume: the held logits are the row's state.
                let held = s.seq.last_logits.take().expect("checked at admission");
                lane.joined = true;
                joined.push((i, held.unsqueeze(0)?));
                continue;
            }
            let start = std::time::Instant::now();
            let mut last = None;
            let mut off = 0usize;
            for w in tile_widths(vec![slot.prompt.len()], self.opts.prefill_tile) {
                let _pool = lloom_tensor::AutoreleaseScope::new();
                let unit = &slot.prompt[off..off + w];
                let input = Tensor::new(unit, &device)?.unsqueeze(0)?;
                s.model.prepare_pass(&mut s.seq.cache, w)?;
                self.pacer.before_forward(&device)?;
                begin_pass(
                    &device,
                    PassShape::Prefill {
                        w,
                        fresh: s.seq.pos == 0,
                    },
                )?;
                let out = {
                    let _prefill = lloom_tensor::tuning::VerifyWindow::prefill();
                    s.model.forward(&input, s.seq.pos, &mut s.seq.cache)
                };
                end_pass(&device)?;
                self.pacer.after_forward(&device)?;
                last = Some(out?);
                s.seq.pos += w;
                s.seq.tokens.extend_from_slice(unit);
                off += w;
                lane.stats.prefill_units += 1;
            }
            lane.prompt_off = slot.prompt.len();
            lane.stats.prefill_ns = start.elapsed().as_nanos() as u64;
            lane.joined = true;
            joined.push((i, last.expect("a non-empty prompt steps at least once")));
        }
        Ok(joined)
    }

    /// With a chunk budget: the prompt PIECES this step feeds — at most
    /// `prefill_batch × prefill_chunk` tokens over the arrived rows still
    /// pending, in slot order, each piece bounded by the chunk and the tile —
    /// chosen here and fed INSIDE the step's batched forward as extra ragged
    /// rows, so they ride the live rows' weight sweep. Resumed rows (held
    /// logits) join at once. Returns `(pieces as (slot, offset, len),
    /// resumed rows)`.
    fn select_pieces(&mut self) -> (Vec<(usize, usize, usize)>, Vec<usize>) {
        let mut pieces = Vec::new();
        let mut resumed = Vec::new();
        let chunk = self.opts.prefill_chunk;
        let mut budget_left = chunk.saturating_mul(self.prefill_batch);
        let step = self.step;
        for i in 0..self.slots.len() {
            if budget_left == 0 {
                break;
            }
            let Some(slot) = self.slots[i].as_mut() else {
                continue;
            };
            let lane = &mut slot.lane;
            if lane.joined || lane.cancelled || lane.arrival > step {
                continue;
            }
            if slot.prompt.is_empty() {
                lane.joined = true;
                resumed.push(i);
                continue;
            }
            lane.prefill_start
                .get_or_insert_with(std::time::Instant::now);
            let remaining = slot.prompt.len() - lane.prompt_off;
            let mut take = remaining.min(budget_left).min(chunk);
            if self.opts.prefill_tile > 0 {
                take = take.min(self.opts.prefill_tile);
            }
            if self.opts.pass_widths == PassWidths::Ladder {
                let top = if self.opts.prefill_tile > 0 {
                    self.opts.prefill_tile.min(self.opts.prefill_chunk)
                } else {
                    self.opts.prefill_chunk
                };
                take = ladder_width(take, top);
            }
            pieces.push((i, lane.prompt_off, take));
            budget_left -= take;
        }
        (pieces, resumed)
    }

    /// The pieces' tokens, in piece order, for the ragged input.
    fn piece_tokens(&self, pieces: &[(usize, usize, usize)]) -> Vec<u32> {
        let mut v = Vec::with_capacity(pieces.iter().map(|p| p.2).sum());
        for &(i, off, take) in pieces {
            let slot = self.slots[i].as_ref().expect("a piece names a held row");
            v.extend_from_slice(&slot.prompt[off..off + take]);
        }
        v
    }

    /// Whether every live row samples by argmax, so the step's ids can be
    /// one device argmax fed straight into the next forward.
    fn device_arm(&self) -> bool {
        self.live.iter().all(|&i| {
            self.slots[i]
                .as_ref()
                .map(|s| s.sampler.sampling() == &Sampling::ArgMax)
                .unwrap_or(true)
        })
    }

    /// Sample every live row on the host, each by its own sampler.
    fn sample_host(&mut self, logits: &Tensor) -> Result<Vec<u32>> {
        let mut ids = Vec::with_capacity(self.live.len());
        for (k, &i) in self.live.iter().enumerate() {
            let row = logits.narrow(0, k, 1)?.squeeze(0)?;
            let slot = self.slots[i].as_mut().expect("a live row is held");
            ids.push(slot.sampler.sample(&row)?);
        }
        Ok(ids)
    }

    /// One batched step: feed the pending rows' prompts (whole, or up to
    /// `prefill_batch` chunks as ragged rows of the live rows' forward),
    /// sample every live row — under [`PrefillDecode::Pause`], while pieces
    /// are fed, only the rows still owing their first token — stream the
    /// ids through `on_tokens`, and retire the rows that ended. A step with
    /// nothing live and nothing to feed advances the step count and encodes
    /// nothing.
    pub fn step(
        &mut self,
        on_tokens: &mut dyn FnMut(RowId, &[u32]) -> Result<()>,
    ) -> Result<StepReport> {
        let _pool = lloom_tensor::AutoreleaseScope::new(); // per batched step, see the decode loop
        let device = self.device.clone();
        let step_start = std::time::Instant::now();
        let mut clock = StageClock::start();
        let mut ph = BatchStepPhases::default();
        let mut joined = if self.opts.prefill_chunk == 0 {
            self.admit_whole()?
        } else {
            Vec::new()
        };
        let (pieces, resumed) = if self.opts.prefill_chunk > 0 {
            self.select_pieces()
        } else {
            (Vec::new(), Vec::new())
        };
        // Under the pause arm the held rows sit this step out: the step below
        // runs over the rows that remain, and the held rows (with their
        // logits, untouched) go back in front of the live set at its end.
        let (held, held_logits) = self.hold_for_pieces(!pieces.is_empty())?;
        let b = self.live.len();
        for i in resumed {
            let slot = self.slots[i].as_mut().expect("a resumed row is held");
            let resumed_logits = slot
                .session
                .borrow_mut()
                .seq
                .last_logits
                .take()
                .expect("checked at admission");
            joined.push((i, resumed_logits.unsqueeze(0)?));
        }
        for &i in &self.live {
            let lane = &mut self.slots[i].as_mut().expect("live").lane;
            lane.stats.steps += 1;
            lane.stats.batch_rows.push(b);
        }
        let device_arm = self.device_arm();
        let lookahead = if device_arm { self.opts.lookahead } else { 0 };
        clock.lap(&mut ph.admit_ns);

        // One argmax over `[B, vocab]`: the ids stay on the device.
        let ids = match &self.logits {
            Some(l) if b > 0 && device_arm => Some(self.device_sampler.sample_argmax_on_device(l)?),
            _ => None,
        };
        clock.lap(&mut ph.argmax_ns);

        // `piece_logits` is `[pieces, vocab]` once the pieces have been
        // fed — by the step's forward (the lookahead arm feeds them with
        // the live rows' next tokens; the blocking arm with the kept
        // rows' after the readback) or alone when nothing is live.
        let mut piece_logits: Option<Tensor> = None;
        let (host_ids, next_logits) = match (&ids, lookahead) {
            (None, _) if b == 0 => {
                if !pieces.is_empty() {
                    let toks = self.piece_tokens(&pieces);
                    let input = Tensor::new(toks.as_slice(), &device)?.unsqueeze(0)?;
                    piece_logits = Some(self.forward_ragged_live(
                        &[],
                        &pieces,
                        &input,
                        (&mut clock, &mut ph),
                    )?);
                }
                (Vec::new(), None)
            }
            (None, _) => {
                // The host arm: every live row sampled by its own sampler,
                // the next forward fed the host ids (`lookahead = 0`).
                let logits = self.logits.clone().expect("live rows have logits");
                let host = self.sample_host(&logits)?;
                clock.lap(&mut ph.wait_ns);
                (host, None)
            }
            (Some(ids), 0) => {
                // Learn first, then feed: the readback waits on the queue's
                // tail, which is exactly this step's argmax.
                let host = ids.to_vec1::<u32>()?;
                clock.lap(&mut ph.wait_ns);
                (host, None)
            }
            (Some(ids), _) => {
                // Stage the readback BEFORE the next forward is encoded, so
                // the wait is on the argmax's command buffer and not on the
                // forward queued behind it; then feed the device ids — and
                // the pieces with them, as extra rows of the same forward.
                let staged = ids.stage_read()?;
                clock.lap(&mut ph.stage_ns);
                let live = self.live.clone();
                let next = if pieces.is_empty() {
                    let input = ids.reshape((b, 1))?;
                    self.forward_live(&live, &input, (&mut clock, &mut ph))?
                } else {
                    let toks = self.piece_tokens(&pieces);
                    let input = Tensor::cat(&[ids, &Tensor::new(toks.as_slice(), &device)?], 0)?
                        .unsqueeze(0)?;
                    let all =
                        self.forward_ragged_live(&live, &pieces, &input, (&mut clock, &mut ph))?;
                    piece_logits = Some(all.narrow(0, b, pieces.len())?);
                    all.narrow(0, 0, b)?
                };
                for &i in &self.live {
                    self.slots[i].as_mut().expect("live").lane.stats.commits += 1;
                }
                let host = staged.to_vec1::<u32>()?;
                clock.lap(&mut ph.wait_ns);
                (host, Some(next))
            }
        };
        if host_ids.len() != b {
            lloom_tensor::bail!(
                "decode_batched: {} ids read back for {b} live rows",
                host_ids.len()
            );
        }

        // Stream and retire. An EOS is not streamed and ends its row; a
        // row reaching `max_tokens` ends after its token is streamed.
        for (k, &i) in self.live.iter().enumerate() {
            let id = host_ids[k];
            let slot = self.slots[i].as_mut().expect("live");
            let lane = &mut slot.lane;
            if lane.cancelled {
                // Cancelled since its last step: nothing more is streamed.
                continue;
            }
            if lane.eos_ids.contains(&id) {
                lane.eos = true;
                lane.done = true;
                continue;
            }
            on_tokens(
                RowId {
                    slot: i,
                    generation: slot.generation,
                },
                std::slice::from_ref(&id),
            )?;
            lane.all_ids.push(id);
            lane.produced += 1;
            if lane.reached(lane.produced) {
                lane.done = true;
                // Under the lookahead this row's last token was fed, so
                // the logits after it exist and are the state a
                // continuation resumes from.
                if let Some(next) = &next_logits {
                    let row = next.narrow(0, k, 1)?.squeeze(0)?;
                    let s: &mut Session<'m, M> = slot.session.borrow_mut();
                    lane.last_logits = Some(s.hold_logits(row)?);
                }
            }
        }
        clock.lap(&mut ph.stream_ns);
        let step_ns = step_start.elapsed().as_nanos() as u64;
        for &i in &self.live {
            self.slots[i]
                .as_mut()
                .expect("live")
                .lane
                .stats
                .step_ns
                .push(step_ns);
        }

        let kept: Vec<usize> = self
            .live
            .iter()
            .copied()
            .filter(|&i| !self.slots[i].as_ref().expect("live").lane.done)
            .collect();
        let kept_logits = match next_logits {
            // The forward already ran for every row that was live; keep
            // the rows still live, in their order.
            Some(next) if kept.len() == self.live.len() => Some(next),
            Some(_) if kept.is_empty() => None,
            Some(next) => {
                let keep: Vec<u32> = self
                    .live
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| !self.slots[**i].as_ref().expect("live").lane.done)
                    .map(|(k, _)| k as u32)
                    .collect();
                Some(next.index_select(&Tensor::new(keep.as_slice(), &device)?, 0)?)
            }
            None if kept.is_empty() && pieces.is_empty() => None,
            None => {
                // Blocking arm: feed the ids the host just read, kept rows
                // only — and the pieces with them, in one forward.
                let mut next_ids: Vec<u32> = self
                    .live
                    .iter()
                    .enumerate()
                    .filter(|(_, i)| !self.slots[**i].as_ref().expect("live").lane.done)
                    .map(|(k, _)| host_ids[k])
                    .collect();
                if pieces.is_empty() {
                    let input =
                        Tensor::new(next_ids.as_slice(), &device)?.reshape((kept.len(), 1))?;
                    Some(self.forward_live(&kept, &input, (&mut clock, &mut ph))?)
                } else if b > 0 {
                    next_ids.extend(self.piece_tokens(&pieces));
                    let input = Tensor::new(next_ids.as_slice(), &device)?.unsqueeze(0)?;
                    let all =
                        self.forward_ragged_live(&kept, &pieces, &input, (&mut clock, &mut ph))?;
                    piece_logits = Some(all.narrow(0, kept.len(), pieces.len())?);
                    if kept.is_empty() {
                        None
                    } else {
                        Some(all.narrow(0, 0, kept.len())?)
                    }
                } else {
                    // `b == 0`: the pieces were fed above, alone.
                    None
                }
            }
        };
        // The pieces are fed: advance the rows' logs, and join the rows
        // whose prompt is complete with the piece's logits.
        let piece_tokens: usize = pieces.iter().map(|p| p.2).sum();
        for (j, &(i, off, take)) in pieces.iter().enumerate() {
            let slot = self.slots[i].as_mut().expect("a piece names a held row");
            let lane = &mut slot.lane;
            slot.session
                .borrow_mut()
                .seq
                .tokens
                .extend_from_slice(&slot.prompt[off..off + take]);
            lane.prompt_off += take;
            lane.stats.prefill_units += 1;
            if let Some(start) = lane.prefill_start {
                lane.stats.prefill_ns = start.elapsed().as_nanos() as u64;
            }
            if lane.prompt_off == slot.prompt.len() {
                lane.joined = true;
                let pl = piece_logits.as_ref().expect("a fed piece has logits");
                joined.push((i, pl.narrow(0, j, 1)?));
            }
        }
        end_executor_step(&device)?;

        // Next step's rows: the held rows, the kept rows, then the rows that
        // joined this step, each with its own logits row. A row joining with `done`
        // already set (`max_tokens == 0`, or cancelled while pending) is
        // retired at once.
        let n_held = held.len();
        let mut next_live = held;
        next_live.extend(kept);
        let mut rows: Vec<Tensor> = Vec::with_capacity(next_live.len() + joined.len());
        if let Some(l) = held_logits {
            rows.push(l);
        }
        if let Some(l) = kept_logits {
            rows.push(l);
        }
        for (i, l) in joined {
            if self.slots[i].as_ref().expect("joined").lane.done {
                continue;
            }
            next_live.push(i);
            rows.push(l);
        }
        self.logits = match rows.len() {
            0 => None,
            1 => Some(rows.remove(0)),
            _ => Some(Tensor::cat(&rows, 0)?),
        };
        self.live = next_live;
        self.step += 1;

        // Retire every row that is done and no longer live — the rows that
        // ended this step, and any pending row cancelled before it joined.
        let mut retired = 0usize;
        for i in 0..self.slots.len() {
            let done = match self.slots[i].as_ref() {
                Some(s) => s.lane.done && !s.lane.retired,
                None => false,
            };
            if done && !self.live.contains(&i) {
                self.retire(i);
                retired += 1;
            }
        }
        clock.lap(&mut ph.tail_ns);
        self.phases = ph;
        Ok(StepReport {
            live: b,
            pieces: pieces.len(),
            piece_tokens,
            retired,
            pending: self.pending_rows(),
            held: n_held,
        })
    }

    /// Under [`PrefillDecode::Pause`] with pieces to feed: take out of the
    /// live set the rows that have streamed a token and are not done, with
    /// their logits rows, leaving the rows that still owe their first token
    /// (and the done rows, which retire this step). Returns the held slots
    /// and their `[held, vocab]` logits, in live order; nothing is held
    /// otherwise.
    fn hold_for_pieces(&mut self, has_pieces: bool) -> Result<(Vec<usize>, Option<Tensor>)> {
        if self.prefill_decode != PrefillDecode::Pause || !has_pieces || self.live.is_empty() {
            return Ok((Vec::new(), None));
        }
        let mut ride: Vec<u32> = Vec::new();
        let mut hold: Vec<u32> = Vec::new();
        for (k, &i) in self.live.iter().enumerate() {
            let lane = &self.slots[i].as_ref().expect("a live row is held").lane;
            if lane.produced == 0 || lane.done {
                ride.push(k as u32);
            } else {
                hold.push(k as u32);
            }
        }
        if hold.is_empty() {
            return Ok((Vec::new(), None));
        }
        let logits = self.logits.take().expect("live rows have logits");
        let pick = |ks: &[u32]| -> Result<Tensor> {
            logits.index_select(&Tensor::new(ks, &self.device)?, 0)
        };
        let held_logits = pick(&hold)?;
        self.logits = if ride.is_empty() {
            None
        } else {
            Some(pick(&ride)?)
        };
        let held = hold.iter().map(|&k| self.live[k as usize]).collect();
        self.live = ride.iter().map(|&k| self.live[k as usize]).collect();
        Ok((held, Some(held_logits)))
    }

    /// Per-sequence bookkeeping, as the other bodies: the fed ids are the
    /// stream's prefix of the cache's length, anything fed past the stream
    /// is overrun. A row cancelled mid-prompt counts the prompt it fed.
    fn retire(&mut self, i: usize) {
        let slot = self.slots[i].as_mut().expect("a retiring row is held");
        let lane = &mut slot.lane;
        let s: &mut Session<'m, M> = slot.session.borrow_mut();
        let fed = s.seq.pos - lane.pos_before - lane.prompt_off;
        let in_cache = fed.min(lane.produced);
        s.seq.tokens.extend_from_slice(&lane.all_ids[..in_cache]);
        let overrun = fed - in_cache;
        s.seq.overrun = overrun;
        s.seq.last_logits = if overrun == 0 {
            lane.last_logits.take()
        } else {
            None
        };
        let mut stats = std::mem::take(&mut lane.stats);
        stats.overrun = overrun;
        stats.produced = lane.produced;
        lane.retired = true;
        let out = DecodeOut {
            ids: std::mem::take(&mut lane.all_ids),
            eos: lane.eos,
            stats,
        };
        self.finished.push((i, out));
    }

    /// One RAGGED batched forward: the rows named by `live` as one-token
    /// rows (their next token each), then `pieces` as `(slot, offset, len)`
    /// prompt pieces, in that order — `input` is `[1, T]` laid out the same
    /// way. Returns `[live + pieces, vocab]` and advances every row's
    /// position by what it fed. A row may appear once across both lists.
    fn forward_ragged_live(
        &mut self,
        live: &[usize],
        pieces: &[(usize, usize, usize)],
        input: &Tensor,
        (clock, ph): (&mut StageClock, &mut BatchStepPhases),
    ) -> Result<Tensor> {
        let mut taken: Vec<Option<&mut Slot<'m, M, R>>> =
            self.slots.iter_mut().map(|s| s.as_mut()).collect();
        let mut picked: Vec<(&mut Session<'m, M>, usize)> =
            Vec::with_capacity(live.len() + pieces.len());
        for &i in live {
            match taken[i].take() {
                Some(s) => picked.push((s.session.borrow_mut(), 1)),
                None => lloom_tensor::bail!("decode_batched: row {i} appears twice in a step"),
            }
        }
        for &(i, _, take) in pieces {
            match taken[i].take() {
                Some(s) => picked.push((s.session.borrow_mut(), take)),
                None => lloom_tensor::bail!("decode_batched: row {i} appears twice in a step"),
            }
        }
        let model = picked
            .first()
            .map(|(s, _)| s.model)
            .ok_or_else(|| lloom_tensor::Error::msg("decode_batched: a forward over no rows"))?;
        // Session grows (a KV bucket) before the bracket, per row at its
        // own width, so the planned pass allocates nothing that outlives it.
        for (s, len) in picked.iter_mut() {
            model.prepare_pass(&mut s.seq.cache, *len)?;
        }
        let shape = PassShape::Ragged {
            live: live.len() as u16,
            pieces: pieces.len() as u16,
            layout: ragged_layout(
                picked[live.len()..]
                    .iter()
                    .map(|(s, len)| (*len, s.seq.pos == 0)),
            ),
        };
        let mut rows: Vec<RaggedRow<'_, M::Cache>> = picked
            .iter_mut()
            .map(|(s, len)| RaggedRow {
                pos: s.seq.pos,
                len: *len,
                cache: &mut s.seq.cache,
            })
            .collect();
        clock.lap(&mut ph.prepare_ns);
        self.pacer.before_forward(input.device())?;
        clock.lap(&mut ph.pace_ns);
        begin_pass(input.device(), shape)?;
        let logits = {
            // The live rows lead the input and the head: under
            // `kernels.gemm.f16_rows_decode` they take a decode step's route
            // (`tuning::RaggedDecodeRows`).
            let _rows = (!live.is_empty()).then(|| {
                lloom_tensor::tuning::RaggedDecodeRows::open(
                    live.len(),
                    live.len() + pieces.iter().map(|p| p.2).sum::<usize>(),
                    live.len() + pieces.len(),
                )
            });
            model.forward_ragged(input, &mut rows)
        }?;
        end_pass(input.device())?;
        clock.lap(&mut ph.encode_ns);
        self.pacer.after_forward(input.device())?;
        drop(rows);
        for (s, len) in picked {
            s.seq.pos += len;
        }
        clock.lap(&mut ph.after_ns);
        Ok(logits)
    }

    /// One batched forward over the rows named by `live`, in that order,
    /// advancing each one's position.
    fn forward_live(
        &mut self,
        live: &[usize],
        input: &Tensor,
        (clock, ph): (&mut StageClock, &mut BatchStepPhases),
    ) -> Result<Tensor> {
        // Disjoint `&mut` borrows of the live rows, in `live`'s order: the
        // indices are distinct by construction (each row is live at most once).
        let mut taken: Vec<Option<&mut Slot<'m, M, R>>> =
            self.slots.iter_mut().map(|s| s.as_mut()).collect();
        let mut picked: Vec<&mut Session<'m, M>> = Vec::with_capacity(live.len());
        for &i in live {
            match taken[i].take() {
                Some(s) => picked.push(s.session.borrow_mut()),
                None => {
                    lloom_tensor::bail!("decode_batched: row {i} appears twice in the live set")
                }
            }
        }
        let model = picked
            .first()
            .map(|s| s.model)
            .ok_or_else(|| lloom_tensor::Error::msg("decode_batched: a forward over no rows"))?;
        for s in picked.iter_mut() {
            model.prepare_pass(&mut s.seq.cache, 1)?;
        }
        let mut rows: Vec<Row<'_, M::Cache>> = picked
            .iter_mut()
            .map(|s| Row {
                pos: s.seq.pos,
                cache: &mut s.seq.cache,
            })
            .collect();
        clock.lap(&mut ph.prepare_ns);
        self.pacer.before_forward(input.device())?;
        clock.lap(&mut ph.pace_ns);
        begin_pass(input.device(), PassShape::Decode { b: live.len() })?;
        let logits = {
            // One token per live row: the step `kernels.gemm.f16_rows_decode`
            // routes (`tuning::DecodeRows`).
            let _rows = lloom_tensor::tuning::DecodeRows::open();
            model.forward_rows(input, &mut rows)
        }?;
        end_pass(input.device())?;
        clock.lap(&mut ph.encode_ns);
        self.pacer.after_forward(input.device())?;
        drop(rows);
        for s in picked {
            s.seq.pos += 1;
        }
        clock.lap(&mut ph.after_ns);
        Ok(logits)
    }
}
