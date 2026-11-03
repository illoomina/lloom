//! Continuous batching: the serving scheduler over `lloom`'s
//! [`BatchDecoder`] (issue #148, `design/docs/batched-decode-design.md` §11).
//!
//! One thread owns the model, a [`SlotPool`] of sequences and the decoder.
//! Requests arrive on a queue from the request threads; each is routed to
//! the sequence sharing the longest prefix with it ([`SlotPool::acquire`]),
//! prepared ([`Session::prepare`] — the prefix the cache already holds is
//! reused, the rest is the row's prompt), and admitted to the decoder, which
//! feeds its prompt as ragged rows of the live rows' own forward (the CHUNK)
//! and decodes it beside them from then on. Its tokens go back to the
//! request thread through a channel as each step confirms them; a row leaves
//! the batch at its EOS, its cap, or a cancellation (the client went away,
//! an `/eval/abort`), and its sequence returns to the pool.
//!
//! The scheduler is the B>1 serving sink `decode-driver-design.md` deferred
//! to this issue: the engine owns the decode thread, the caller owns one
//! channel per request, and routing is the request thread's — a server fans
//! its N clients into N submissions and reads N receivers. The B=1 core
//! (per-sequence [`Session`], the device-resident ring for the single-session
//! loop) is untouched underneath: a batch row's token path is the batched
//! driver's, not the ring's.
//!
//! **Closed jobs.** A benchmark cell wants the engine to itself — every row
//! admitted at once, one bracket around the run, nothing else live. Such a
//! job is queued like a request and run on the scheduler thread when no row
//! is live ([`BatchHandle::run_closed`]); the pool, the model and the batch
//! options are handed to it ([`Idle`]).
//!
//! **Static batches.** A body that cannot admit a row mid-run — the
//! speculative pipelined driver, one drafter slot per row — is driven as
//! static batches: the waiting requests are taken up to the batch width and
//! run to completion together through the caller's [`StaticBody`], then the
//! next batch. Same queue, same channels, same pool.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::sync::Arc;

use lloom::generation::Sampling;
use lloom::nn::{
    BatchDecoder, BatchedCausalLM, CacheReset, CacheRewind, CacheShare, DecodeOptions, DecodeOut,
    DecodeStats, PrefillDecode, ReuseAction, RowDriver, RowId, Sequence, Session,
};
use lloom_tensor::{Device, Result};

use crate::decode::{DecodeParams, RingPool};
use crate::slots::SlotPool;

/// A request to the batch scheduler: the whole token sequence to continue
/// and the run's knobs. `readback` and `stop_strings` on the params are the
/// request thread's (a batch row has no ring; stop strings are text).
pub struct BatchRequest {
    pub prompt_ids: Vec<u32>,
    pub params: DecodeParams,
    /// Raised by the request thread to end the row early (a client that
    /// went away, a benchmark's abort); the row retires at the next step.
    pub cancel: Arc<AtomicBool>,
}

/// What a request thread receives for its row, in order: token batches as
/// the steps confirm them, then exactly one `Done` or `Error`.
#[derive(Debug)]
pub enum BatchEvent {
    Tokens(Vec<u32>),
    Done(Box<BatchDone>),
    Error(String),
}

/// A hook the scheduler calls after every step with its counters.
pub type CountersHook = Box<dyn Fn(&BatchCounters) + Send>;

/// The scheduler's continuous driver, boxed: the K=1 batched loop or the
/// speculative one, one surface (`RowDriver`).
pub type Driver<'m, M> = Box<dyn RowDriver<'m, M, Session<'m, M>> + 'm>;

/// What the scheduler's `extra` can build its continuous driver from: the
/// speculative loop when it holds a proposer, else nothing (the K=1
/// `BatchDecoder`). Built on the scheduler's thread, where the model borrow
/// lives.
pub trait MakeDriver<M: BatchedCausalLM> {
    fn make_driver<'m>(
        &mut self,
        model: &'m M,
        opts: &DecodeOptions,
        device: &Device,
    ) -> Result<Option<Driver<'m, M>>>;
}

impl<M: BatchedCausalLM> MakeDriver<M> for () {
    fn make_driver<'m>(
        &mut self,
        _model: &'m M,
        _opts: &DecodeOptions,
        _device: &Device,
    ) -> Result<Option<Driver<'m, M>>> {
        Ok(None)
    }
}

/// What the scheduler's `extra` contributes when a static batch or a
/// continuous step fails, before its sequences are reset and returned to the
/// pool: whether the failure touched their state at all, the fence that
/// waits out the failed work still on the device, then the drop of the
/// extra's own per-row state (a drafter's contexts). None of it runs on a
/// batch or step that succeeded.
pub trait BatchRecovery {
    /// Whether `error`, as the static body or the continuous step returned
    /// it, may have moved the state of the rows it ended
    /// ([`crate::recovery::failure_touched_state`]). When it did not, the
    /// rows go back to the pool with their cached prefix and nothing below
    /// runs.
    fn touched(&self, error: &lloom_tensor::Error) -> bool {
        crate::recovery::failure_touched_state(error)
    }

    /// Wait for the failed work still on the device. A refusal leaves the
    /// rows held, unreset, and the recovery is retried before the next
    /// batch or admission.
    fn fence(&mut self, device: &Device) -> Result<()> {
        crate::recovery::fence_failed_execution(device)
    }

    /// Forget per-row state the failure may have moved. Runs only after
    /// [`Self::fence`] succeeded.
    fn abort(&mut self) {}
}

impl BatchRecovery for () {}

/// A retired row's outcome.
#[derive(Debug, Clone)]
pub struct BatchDone {
    /// The generated ids, EOS excluded.
    pub tokens: Vec<u32>,
    pub eos: bool,
    pub cancelled: bool,
    pub stats: DecodeStats,
    pub reuse: ReuseAction,
    pub reused: usize,
    pub prefilled: usize,
    /// What the row reused of its sequence (`crate::turn`).
    pub turn: crate::turn::TurnReuse,
    /// Rows live in the batch when this row joined it.
    pub joined_beside: usize,
}

/// What a closed job is handed: the engine, idle.
pub struct Idle<'a, 'm, M: BatchedCausalLM, E> {
    pub model: &'m M,
    pub device: &'a Device,
    /// The sequences, all of them checked in.
    pub pool: &'a mut SlotPool<M::Cache>,
    pub rings: &'a mut RingPool,
    /// The scheduler's batch options — the tile and the chunk, the lookahead.
    pub opts: &'a DecodeOptions,
    /// The consumer's own state on the scheduler thread (a drafter, say).
    pub extra: &'a mut E,
    /// The scheduler's continuous driver, idle: a closed job may run its
    /// rows THROUGH it ([`drive_closed_rows`]) — the speculative driver's
    /// one proposer serves the closed cells too, where a second drafter did
    /// (`batched-decode-design.md` §12.9).
    pub driver: &'a mut Driver<'m, M>,
}

impl<'m, M: BatchedCausalLM, E> Idle<'_, 'm, M, E>
where
    M::Cache: lloom::nn::CacheReset,
{
    /// Check `n` sequences out of the pool for a closed job, each reset,
    /// with the slot each goes back to ([`Self::give_back_reset`]); `None`
    /// when the pool holds fewer, the ones already taken given back, so a
    /// short pool costs it no slot.
    pub fn take_reset(&mut self, n: usize) -> Option<(Vec<Session<'m, M>>, Vec<usize>)> {
        let mut sessions = Vec::with_capacity(n);
        let mut slots = Vec::with_capacity(n);
        for _ in 0..n {
            let Some((routed, seq)) = self.pool.acquire(&[]) else {
                self.give_back_reset(sessions, slots);
                return None;
            };
            let mut s = Session::from_sequence(self.model, seq);
            s.reset();
            sessions.push(s);
            slots.push(routed.slot);
        }
        Some((sessions, slots))
    }

    /// Reset each of `sessions` and return it to its slot.
    pub fn give_back_reset(&mut self, sessions: Vec<Session<'m, M>>, slots: Vec<usize>) {
        for (slot, mut s) in slots.into_iter().zip(sessions) {
            s.reset();
            self.pool.release(slot, s.into_sequence());
        }
    }
}

/// Run `sessions` as one closed batch through `driver`: every row admitted
/// at once (greedy, `eos_ids`, `max_tokens`), stepped until the driver is
/// idle, the sessions handed back in the order they came with each row's
/// outcome. On an error the driver is drained so no session is lost — a
/// failed cell must not cost the pool a slot.
#[allow(clippy::type_complexity)]
pub fn drive_closed_rows<'m, M: BatchedCausalLM + 'm>(
    driver: &mut Driver<'m, M>,
    sessions: Vec<Session<'m, M>>,
    prompts: &[&[u32]],
    eos_ids: &[u32],
    max_tokens: usize,
    on_tokens: &mut dyn FnMut(usize, &[u32]) -> Result<()>,
) -> (Vec<Session<'m, M>>, Result<Vec<DecodeOut>>) {
    let n = sessions.len();
    if prompts.len() != n {
        return (
            sessions,
            Err(lloom_tensor::Error::msg(format!(
                "drive_closed_rows: {n} sessions and {} prompts",
                prompts.len()
            ))),
        );
    }
    if !driver.is_idle() {
        return (
            sessions,
            Err(lloom_tensor::Error::msg(
                "drive_closed_rows: the driver holds rows; a closed job needs it idle",
            )),
        );
    }
    let mut ids: Vec<RowId> = Vec::with_capacity(n);
    // Admitted in order; a row the driver refuses comes back with the
    // refusal (`Rejected`), and the rows after it are handed back
    // untouched — a failed cell costs the caller no session.
    let mut rest: Vec<Session<'m, M>> = sessions;
    rest.reverse();
    while let Some(s) = rest.pop() {
        let i = ids.len();
        match driver.admit_with_eos(
            s,
            prompts[i].to_vec(),
            Sampling::ArgMax,
            0,
            eos_ids,
            max_tokens,
        ) {
            Ok(id) => ids.push(id),
            Err(rejected) => {
                let mut back = driver.drain_rows();
                back.push(rejected.session);
                rest.reverse();
                back.append(&mut rest);
                return (back, Err(rejected.error));
            }
        }
    }
    let run = (|| -> Result<Vec<Option<(Session<'m, M>, DecodeOut)>>> {
        let mut outs: Vec<Option<(Session<'m, M>, DecodeOut)>> = (0..n).map(|_| None).collect();
        while !driver.is_idle() {
            driver.step(&mut |row, toks| {
                let i = ids
                    .iter()
                    .position(|&id| id == row)
                    .expect("an admitted row");
                on_tokens(i, toks)
            })?;
            for (row, session, out) in driver.take_finished() {
                let i = ids
                    .iter()
                    .position(|&id| id == row)
                    .expect("an admitted row");
                outs[i] = Some((session, out));
            }
        }
        Ok(outs)
    })();
    match run {
        Ok(outs) => {
            let mut sessions = Vec::with_capacity(n);
            let mut results = Vec::with_capacity(n);
            for o in outs {
                let (s, out) = o.expect("every row finished");
                sessions.push(s);
                results.push(out);
            }
            (sessions, Ok(results))
        }
        Err(e) => (driver.drain_rows(), Err(e)),
    }
}

/// A job that runs on the scheduler thread with nothing else live.
pub type ClosedJob<M, E> = Box<dyn for<'a, 'm> FnOnce(&mut Idle<'a, 'm, M, E>) + Send>;

/// One row of a static batch: its sampler, its EOS set and its cap.
#[derive(Debug, Clone)]
pub struct RowSpec {
    pub sampling: Sampling,
    pub seed: u64,
    pub eos_ids: Vec<u32>,
    pub max_tokens: usize,
}

/// A body that decodes a STATIC batch of rows to completion: every row's
/// session (prepared, at its prompt's suffix), prompt and [`RowSpec`] given
/// at once, tokens streamed per `(row, ids)`. The speculative pipelined
/// driver's shape.
pub type StaticBody<M, E> = Box<
    dyn for<'a, 'm> FnMut(
            &mut Idle<'a, 'm, M, E>,
            &mut [Session<'m, M>],
            &[&[u32]],
            &[RowSpec],
            &mut dyn FnMut(usize, &[u32]) -> Result<()>,
        ) -> Result<Vec<DecodeOut>>
        + Send,
>;

/// How the scheduler decodes its rows.
pub enum BatchMode<M: BatchedCausalLM, E> {
    /// Rows join and leave the batch per step (`BatchDecoder`).
    Continuous,
    /// Rows run in static batches through the body, up to the width each.
    Static(StaticBody<M, E>),
}

enum Job<M: BatchedCausalLM, E> {
    Request(Box<BatchRequest>, Sender<BatchEvent>),
    Closed(ClosedJob<M, E>),
    Shutdown,
}

/// The request threads' handle: submit rows, queue closed jobs.
pub struct BatchHandle<M: BatchedCausalLM, E> {
    tx: Sender<Job<M, E>>,
}

impl<M: BatchedCausalLM, E> Clone for BatchHandle<M, E> {
    fn clone(&self) -> Self {
        Self {
            tx: self.tx.clone(),
        }
    }
}

impl<M: BatchedCausalLM, E> BatchHandle<M, E> {
    /// Queue a row; its events arrive on the returned receiver.
    pub fn submit(&self, req: BatchRequest) -> Receiver<BatchEvent> {
        let (tx, rx) = mpsc::channel();
        if self
            .tx
            .send(Job::Request(Box::new(req), tx.clone()))
            .is_err()
        {
            let _ = tx.send(BatchEvent::Error(
                "the batch scheduler has stopped".to_string(),
            ));
        }
        rx
    }

    /// Queue a closed job; it runs once nothing is live. `false` if the
    /// scheduler has stopped.
    pub fn run_closed(&self, job: ClosedJob<M, E>) -> bool {
        self.tx.send(Job::Closed(job)).is_ok()
    }

    /// Run a closed job and wait for its result.
    pub fn run_closed_blocking<T: Send + 'static>(
        &self,
        job: impl for<'a, 'm> FnOnce(&mut Idle<'a, 'm, M, E>) -> T + Send + 'static,
    ) -> std::result::Result<T, String> {
        let (tx, rx) = mpsc::channel();
        let queued = self.run_closed(Box::new(move |idle| {
            let _ = tx.send(job(idle));
        }));
        if !queued {
            return Err("the batch scheduler has stopped".to_string());
        }
        rx.recv()
            .map_err(|_| "the batch scheduler dropped the job".to_string())
    }

    /// Ask the scheduler thread to exit after the rows it holds retire.
    pub fn shutdown(&self) {
        let _ = self.tx.send(Job::Shutdown);
    }
}

/// A static batch row's routing, its `prepare` outcome, its sender and its
/// cancel flag, held while the body runs.
type StaticRowMeta = (
    usize,
    lloom::nn::Prepared,
    Sender<BatchEvent>,
    Arc<AtomicBool>,
    crate::turn::TurnReuse,
);

/// A request's error while a failure's recovery is refused.
fn recovery_pending(fence: &str) -> String {
    format!("batch scheduler: a failed batch's recovery is pending: {fence}")
}

/// Where a request's prefix is shared from: an idle slot of the pool or a
/// live row of the decoder, and the tokens (whole pages) it can give.
#[derive(Debug, Clone, Copy)]
enum ShareSource {
    Idle(usize, usize),
    Live(RowId, usize),
}

struct InFlight {
    tx: Sender<BatchEvent>,
    cancel: Arc<AtomicBool>,
    slot: usize,
    reuse: ReuseAction,
    reused: usize,
    prefilled: usize,
    turn: crate::turn::TurnReuse,
    joined_beside: usize,
}

/// The scheduler's counters, for a report line.
#[derive(Debug, Clone, Copy, Default)]
pub struct BatchCounters {
    pub steps: u64,
    pub admitted: u64,
    pub retired: u64,
    pub max_live: usize,
    pub closed_jobs: u64,
    pub static_batches: u64,
    /// Admissions deferred for a live row's prefix still being fed (§12.3).
    pub share_waits: u64,
    /// The most prompt pieces one step carried (`prefill_batch`, §11.8).
    pub max_pieces: usize,
    /// Steps that held a streaming row back while pieces were fed
    /// (`PrefillDecode::Pause`).
    pub held_steps: u64,
    /// Ticks whose intake hit its per-tick budget while work was live, so
    /// the jobs left in the channel waited for the next tick
    /// (`scheduler.intake_per_tick`).
    pub intake_deferred: u64,
}

/// The scheduler: built and run on one thread ([`spawn`]).
pub struct BatchScheduler<'m, M: BatchedCausalLM, E> {
    model: &'m M,
    device: Device,
    pool: SlotPool<M::Cache>,
    rings: RingPool,
    opts: DecodeOptions,
    max_rows: usize,
    reuse_prefix: bool,
    /// Share a request's prefix from another row's cache — an idle slot's or
    /// a live row's — by page (`CacheShare`, §12.3). Implies `reuse_prefix`.
    share_prefix: bool,
    mode: BatchMode<M, E>,
    extra: E,
    /// The most jobs one tick takes off the queue while work is live
    /// (`scheduler.intake_per_tick`): the fairness bound on intake.
    intake_per_tick: usize,
    decoder: Driver<'m, M>,
    inflight: HashMap<RowId, InFlight>,
    waiting: VecDeque<(BatchRequest, Sender<BatchEvent>)>,
    closed: VecDeque<ClosedJob<M, E>>,
    rx: Receiver<Job<M, E>>,
    counters: BatchCounters,
    /// A failed static batch's or continuous step's sequences, by pool slot,
    /// out of the pool and unreset until [`Self::recover_failed`] succeeds.
    /// Non-empty between batches or steps only while the fence refuses.
    failed: Vec<(usize, Sequence<M::Cache>)>,
    /// Under the static mode, the rows the current batch used, by pool slot.
    on_counters: Option<CountersHook>,
    /// The open profile window, under the profile build's recorder.
    #[cfg(feature = "metal")]
    window: Option<ProfileWindow>,
    #[cfg(feature = "metal")]
    windows: u64,
}

/// One batch window under the profile build's recorder (`CANDLE_METAL_PROFILE=1`):
/// from an admission onto an idle engine until nothing is live, every step's
/// composition with its host interval in the GPU's clock.
#[cfg(feature = "metal")]
struct ProfileWindow {
    started: std::time::Instant,
    steps: Vec<WindowStep>,
}

/// The profile window's per-step record and its decode-cadence summary,
/// split out so the arithmetic is testable without a GPU.
#[cfg(any(feature = "metal", test))]
mod steady {
    /// One step of a profile window: its host interval in the GPU's clock, its
    /// composition, its host time by stage and what it dispatched.
    #[derive(Debug, Clone, Copy)]
    pub(super) struct WindowStep {
        pub(super) step: u64,
        pub(super) t0: f64,
        pub(super) t1: f64,
        pub(super) report: lloom::nn::StepReport,
        pub(super) phases: lloom::nn::BatchStepPhases,
        pub(super) dispatches: u64,
    }

    /// One live-row count's decode cadence over a window: the steps that fed no
    /// piece and held no row, each timed from its start to the next step's
    /// start (the scheduler's own work between steps included).
    #[derive(Debug, Clone, PartialEq)]
    pub(super) struct SteadyRow {
        pub(super) live: usize,
        pub(super) steps: usize,
        /// Medians, milliseconds: the step's wall (start to the next start), the
        /// GPU's busy union inside it, the host stages, and the scheduler's time
        /// between the step's end and the next one's start.
        pub(super) wall_ms: f64,
        pub(super) gpu_busy_ms: f64,
        pub(super) stages_ms: [f64; 10],
        pub(super) between_ms: f64,
        pub(super) dispatches: f64,
    }

    #[cfg_attr(not(feature = "metal"), allow(dead_code))]
    pub(super) const STAGE_NAMES: [&str; 10] = [
        "admit", "argmax", "stage", "prepare", "pace", "encode", "after", "wait", "stream", "tail",
    ];

    pub(super) fn stage_array(p: &lloom::nn::BatchStepPhases) -> [u64; 10] {
        [
            p.admit_ns,
            p.argmax_ns,
            p.stage_ns,
            p.prepare_ns,
            p.pace_ns,
            p.encode_ns,
            p.after_ns,
            p.wait_ns,
            p.stream_ns,
            p.tail_ns,
        ]
    }

    /// The median of `v` (sorted here); NaN when empty.
    fn median(v: &mut [f64]) -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        v[(v.len() - 1) / 2]
    }

    /// The busy union of `intervals` (sorted by start) clipped to `[lo, hi)`.
    pub(super) fn busy_within(intervals: &[(f64, f64)], lo: f64, hi: f64) -> f64 {
        let mut busy = 0.0;
        let mut cursor = lo;
        for &(s, e) in intervals {
            if s >= hi {
                break;
            }
            let s = s.max(cursor);
            let e = e.min(hi);
            if e > s {
                busy += e - s;
                cursor = e;
            }
        }
        busy
    }

    /// One decode step's interval, milliseconds: the fields [`SteadyRow`]
    /// takes the medians of.
    struct Sample {
        wall: f64,
        busy: f64,
        stages: [f64; 10],
        between: f64,
        dispatches: f64,
    }

    /// The decode cadence per live-row count: a step counts when it fed no
    /// piece, held no row and the next step began with the same live rows, so
    /// its interval is one decode step at that width. `intervals` are the
    /// window's command buffers' GPU intervals.
    pub(super) fn steady_rows(steps: &[WindowStep], intervals: &[(f64, f64)]) -> Vec<SteadyRow> {
        let mut sorted = intervals.to_vec();
        sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut by_live: std::collections::BTreeMap<usize, Vec<Sample>> = Default::default();
        for w in steps.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if a.report.live == 0
                || a.report.pieces > 0
                || a.report.held > 0
                || b.report.live != a.report.live
                || b.step != a.step + 1
            {
                continue;
            }
            let ms = |ns: u64| ns as f64 / 1e6;
            let stages = stage_array(&a.phases).map(ms);
            by_live.entry(a.report.live).or_default().push(Sample {
                wall: 1e3 * (b.t0 - a.t0),
                busy: 1e3 * busy_within(&sorted, a.t0, b.t0),
                stages,
                between: 1e3 * (b.t0 - a.t1),
                dispatches: a.dispatches as f64,
            });
        }
        by_live
            .into_iter()
            .map(|(live, rows)| {
                let col = |f: &dyn Fn(&Sample) -> f64| {
                    let mut v: Vec<f64> = rows.iter().map(f).collect();
                    median(&mut v)
                };
                let mut stages_ms = [0.0; 10];
                for (k, s) in stages_ms.iter_mut().enumerate() {
                    *s = col(&|r| r.stages[k]);
                }
                SteadyRow {
                    live,
                    steps: rows.len(),
                    wall_ms: col(&|r| r.wall),
                    gpu_busy_ms: col(&|r| r.busy),
                    stages_ms,
                    between_ms: col(&|r| r.between),
                    dispatches: col(&|r| r.dispatches),
                }
            })
            .collect()
    }
}
#[cfg(feature = "metal")]
use steady::{stage_array, STAGE_NAMES};
#[cfg(any(feature = "metal", test))]
use steady::{steady_rows, WindowStep};

/// What [`spawn`] needs to build the scheduler on its thread.
pub struct BatchConfig<M: BatchedCausalLM, E> {
    pub device: Device,
    pub pool: SlotPool<M::Cache>,
    pub rings: RingPool,
    pub opts: DecodeOptions,
    /// The most rows live at once — the batch width `B`.
    pub max_rows: usize,
    /// Reuse a routed sequence's cached prefix (`Session::prepare`); off, a
    /// request starts from a reset sequence.
    pub reuse_prefix: bool,
    /// Share a request's prefix from another row's cache, by page (§12.3).
    pub share_prefix: bool,
    /// How many pending rows' chunks one step of the K=1 loop carries
    /// (`BatchDecoder::with_prefill_batch`); a driver from `extra` prefills
    /// its own way and does not read it.
    pub prefill_batch: usize,
    /// What the K=1 loop's live rows do in a step that feeds prompt pieces.
    pub prefill_decode: PrefillDecode,
    /// The most jobs one intake tick takes off the request queue while
    /// work is live (`scheduler.intake_per_tick`, at least 1); the rest
    /// stay queued in arrival order for the next tick. With nothing live
    /// the whole queue is drained whatever this says.
    pub intake_per_tick: usize,
    pub mode: BatchMode<M, E>,
    pub extra: E,
    /// Called after every step with the counters (a log line).
    pub on_counters: Option<CountersHook>,
}

impl<'m, M, E> BatchScheduler<'m, M, E>
where
    M: BatchedCausalLM + 'm,
    M::Cache: CacheReset + CacheRewind + CacheShare,
    E: MakeDriver<M> + BatchRecovery,
{
    fn new(model: &'m M, mut cfg: BatchConfig<M, E>, rx: Receiver<Job<M, E>>) -> Result<Self> {
        if cfg.max_rows == 0 {
            lloom_tensor::bail!("batch scheduler: max_rows must be positive");
        }
        if cfg.pool.len() < cfg.max_rows {
            lloom_tensor::bail!(
                "batch scheduler: {} sequences in the pool for a batch of {}",
                cfg.pool.len(),
                cfg.max_rows
            );
        }
        // Every row needs an EOS set of its own; the decoder's is a floor
        // the rows' params refine per request (`admit` re-checks), so the
        // decoder is built with a placeholder no id matches.
        let decoder: Driver<'m, M> = match cfg.extra.make_driver(model, &cfg.opts, &cfg.device)? {
            Some(d) => d,
            None => Box::new(
                BatchDecoder::new(cfg.opts.clone(), &[u32::MAX], cfg.device.clone())?
                    .with_prefill_batch(cfg.prefill_batch)
                    .with_prefill_decode(cfg.prefill_decode),
            ),
        };
        Ok(Self {
            model,
            device: cfg.device,
            pool: cfg.pool,
            rings: cfg.rings,
            opts: cfg.opts,
            max_rows: cfg.max_rows,
            reuse_prefix: cfg.reuse_prefix || cfg.share_prefix,
            share_prefix: cfg.share_prefix,
            mode: cfg.mode,
            extra: cfg.extra,
            intake_per_tick: cfg.intake_per_tick.max(1),
            decoder,
            inflight: HashMap::new(),
            waiting: VecDeque::new(),
            closed: VecDeque::new(),
            rx,
            counters: BatchCounters::default(),
            failed: Vec::new(),
            on_counters: cfg.on_counters,
            #[cfg(feature = "metal")]
            window: None,
            #[cfg(feature = "metal")]
            windows: 0,
        })
    }

    fn idle(&mut self) -> Idle<'_, 'm, M, E> {
        Idle {
            model: self.model,
            device: &self.device,
            pool: &mut self.pool,
            rings: &mut self.rings,
            opts: &self.opts,
            extra: &mut self.extra,
            driver: &mut self.decoder,
        }
    }

    fn nothing_live(&self) -> bool {
        self.inflight.is_empty() && self.decoder.is_idle()
    }

    /// Take the queue: block when there is nothing to do, else take at most
    /// [`Self::intake_budget`] jobs before returning to the rows.
    /// `false` on shutdown.
    ///
    /// The budget is what keeps a producer that holds the queue non-empty
    /// from postponing the live rows' next step indefinitely: with work
    /// live, intake is bounded per tick and the jobs it does not take stay
    /// queued, in arrival order, for the next tick. With nothing live there
    /// is no row to starve and the whole queue is drained, as it always
    /// was — the closed-job and shutdown ordering is the queue's order
    /// either way.
    fn take_jobs(&mut self) -> bool {
        let block = self.nothing_live() && self.waiting.is_empty() && self.closed.is_empty();
        if block {
            match self.rx.recv() {
                Ok(job) => {
                    if !self.enqueue(job) {
                        return false;
                    }
                }
                Err(_) => return false,
            }
        }
        let budget = if self.nothing_live() {
            usize::MAX
        } else {
            self.intake_per_tick
        };
        let mut taken = 0usize;
        while taken < budget {
            match self.rx.try_recv() {
                Ok(job) => {
                    taken += 1;
                    if !self.enqueue(job) {
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
        if taken == budget {
            self.counters.intake_deferred += 1;
        }
        true
    }

    fn enqueue(&mut self, job: Job<M, E>) -> bool {
        match job {
            Job::Request(req, tx) => self.waiting.push_back((*req, tx)),
            Job::Closed(job) => self.closed.push_back(job),
            Job::Shutdown => return false,
        }
        true
    }

    /// Route and prepare a request: the sequence sharing its longest prefix,
    /// reset unless prefixes are reused, prepared to the suffix to feed.
    fn prepare(
        &mut self,
        req: &BatchRequest,
    ) -> Result<(
        usize,
        Session<'m, M>,
        lloom::nn::Prepared,
        crate::turn::TurnReuse,
    )> {
        if req.params.eos_ids.is_empty() {
            lloom_tensor::bail!("batch scheduler: eos_ids is empty; a run needs an EOS set");
        }
        if req.prompt_ids.is_empty() {
            lloom_tensor::bail!("batch scheduler: empty prompt");
        }
        req.params
            .refuse_on(lloom::nn::Body::BatchedStep, "the batch scheduler")?;
        let (routed, seq) = self
            .pool
            .acquire(&req.prompt_ids)
            .ok_or_else(|| lloom_tensor::Error::msg("batch scheduler: no idle sequence"))?;
        let mut session = Session::from_sequence(self.model, seq);
        if !self.reuse_prefix || req.params.cold {
            session.reset();
        }
        // The share source, when the routed slot has nothing of its own: the
        // idle slot or live row holding the longest common prefix, at least
        // a page of it.
        let source = if self.share_prefix {
            self.share_source(&req.prompt_ids)
        } else {
            None
        };
        let boundary = session.tokens().len();
        let prepared = match source {
            Some(ShareSource::Idle(slot, lcp)) => {
                let src = self.pool.sequence(slot).expect("an idle slot");
                let log = &src.tokens()[..lcp];
                session
                    .prepare_shared(&req.prompt_ids, src.cache(), log)
                    .map(|p| crate::turn::TurnReuse::of_shared(boundary, p))
            }
            Some(ShareSource::Live(id, lcp)) => {
                let (fed, cache, _) = self.decoder.row_source(id).expect("a held row");
                session
                    .prepare_shared(&req.prompt_ids, cache, &fed[..lcp])
                    .map(|p| crate::turn::TurnReuse::of_shared(boundary, p))
            }
            None => crate::turn::prepare(&mut session, &req.prompt_ids, true),
        };
        let (prepared, turn) = match prepared {
            Ok(p) => p,
            Err(e) => {
                self.pool.release(routed.slot, session.into_sequence());
                return Err(e);
            }
        };
        Ok((routed.slot, session, prepared, turn))
    }

    /// The share unit, from any sequence's cache (0: sharing is impossible).
    fn share_page(&self) -> usize {
        self.pool
            .idle_sequences()
            .next()
            .map(|(_, s)| s.cache().share_granularity())
            .or_else(|| {
                self.decoder.row_ids().into_iter().find_map(|id| {
                    self.decoder
                        .row_source(id)
                        .map(|(_, c, _)| c.share_granularity())
                })
            })
            .unwrap_or(0)
    }

    /// The best source for `prompt`'s prefix: the idle slot or live row whose
    /// CACHED tokens share the longest prefix with it, if that is at least a
    /// page (rounded down to whole pages, and to `prompt.len() - 1`).
    fn share_source(&self, prompt: &[u32]) -> Option<ShareSource> {
        let page = self.share_page();
        if page == 0 || prompt.len() < 2 {
            return None;
        }
        let cap = prompt.len() - 1;
        let lcp_of = |log: &[u32]| -> usize {
            let n = log
                .iter()
                .zip(prompt.iter())
                .take_while(|(a, b)| a == b)
                .count();
            (n.min(cap) / page) * page
        };
        let mut best: Option<ShareSource> = None;
        let mut best_n = 0usize;
        for (slot, seq) in self.pool.idle_sequences() {
            if !seq.tokens_known() {
                continue;
            }
            let n = lcp_of(seq.tokens());
            if n >= page && n > best_n {
                best_n = n;
                best = Some(ShareSource::Idle(slot, n));
            }
        }
        for id in self.decoder.row_ids() {
            let Some((fed, _, _)) = self.decoder.row_source(id) else {
                continue;
            };
            let n = lcp_of(fed);
            if n >= page && n > best_n {
                best_n = n;
                best = Some(ShareSource::Live(id, n));
            }
        }
        best
    }

    /// Whether `prompt` should WAIT for a live row still feeding a prompt it
    /// shares more than a page with: the tokens the row has fed so far share
    /// less than the two prompts do, and the row's prompt is not yet in — a
    /// burst of requests on one long prefix then prefills it once (§12.3).
    fn share_pending(&self, prompt: &[u32]) -> bool {
        let page = self.share_page();
        if page == 0 || prompt.len() < 2 {
            return false;
        }
        let cap = prompt.len() - 1;
        let pages_of = |log: &[u32]| -> usize {
            let n = log
                .iter()
                .zip(prompt.iter())
                .take_while(|(a, b)| a == b)
                .count();
            n.min(cap) / page
        };
        // What the idle slots already give: a live row must beat it to be
        // worth waiting for.
        let idle_best = self
            .pool
            .idle_sequences()
            .filter(|(_, s)| s.tokens_known())
            .map(|(_, s)| pages_of(s.tokens()))
            .max()
            .unwrap_or(0);
        for id in self.decoder.row_ids() {
            let Some((fed, _, complete)) = self.decoder.row_source(id) else {
                continue;
            };
            if complete {
                continue;
            }
            let Some(full) = self.decoder.row_prospective_log(id) else {
                continue;
            };
            let could = pages_of(&full);
            // Until the row's prompt is fully in, the ring snapshot of its
            // last shared page may not exist yet (it is taken when the NEXT
            // pass begins): wait for the whole prompt, not for the pages.
            if could > idle_best {
                if lloom::tuning::tuning().lfm2.share_trace {
                    eprintln!(
                        "[share: waiting for row {id:?}: {could} pages of its {}-token prompt \
                         ({} fed), idle slots give {idle_best}]",
                        full.len(),
                        fed.len()
                    );
                }
                return true;
            }
        }
        false
    }

    /// The row's sampler arm, resolved before the row takes a slot: a
    /// refused field is the request's error, and nothing is prepared for it.
    fn sampling_of(params: &DecodeParams) -> std::result::Result<Sampling, String> {
        params
            .sampling()
            .map_err(|e| e.without_backtrace().to_string())
    }

    /// Admit waiting rows into the continuous decoder while the width allows.
    /// A failed step's held rows are recovered first; while that recovery is
    /// refused, every waiting row gets its error rather than a sequence
    /// beside work the fence could not see retire.
    fn admit_waiting(&mut self) {
        if let Err(e) = self.recover_failed() {
            for (_, tx) in self.waiting.drain(..) {
                let _ = tx.send(BatchEvent::Error(recovery_pending(&e)));
            }
            return;
        }
        while self.decoder.rows() < self.max_rows {
            let Some((req, tx)) = self.waiting.pop_front() else {
                break;
            };
            if req.cancel.load(Ordering::Acquire) {
                let _ = tx.send(BatchEvent::Done(Box::new(BatchDone {
                    tokens: Vec::new(),
                    eos: false,
                    cancelled: true,
                    stats: DecodeStats::default(),
                    reuse: ReuseAction::Fresh,
                    reused: 0,
                    prefilled: 0,
                    turn: Default::default(),
                    joined_beside: 0,
                })));
                continue;
            }
            // A request sharing more than a page with a live row's prompt
            // that is still being fed waits for the pages to exist, rather
            // than prefilling the same prefix beside it (§12.3). The queue
            // holds behind it: a burst on one prefix is one prefill.
            if self.share_prefix && self.share_pending(&req.prompt_ids) {
                self.counters.share_waits += 1;
                self.waiting.push_front((req, tx));
                break;
            }
            let sampling = match Self::sampling_of(&req.params) {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(BatchEvent::Error(e));
                    continue;
                }
            };
            let t_prep = std::time::Instant::now();
            let (slot, session, prepared, turn) = match self.prepare(&req) {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.send(BatchEvent::Error(e.to_string()));
                    continue;
                }
            };
            if lloom::tuning::tuning().lfm2.share_trace {
                eprintln!(
                    "[share: prepared slot {slot} {:?} reused {} suffix {} in {:.1} ms; live {} waiting {}]",
                    prepared.action,
                    prepared.reused,
                    prepared.suffix.len(),
                    t_prep.elapsed().as_secs_f64() * 1e3,
                    self.decoder.live_rows(),
                    self.waiting.len()
                );
            }
            let joined_beside = self.decoder.live_rows();
            // The row's own EOS set: the decoder stops a row on its own
            // `eos_ids`, so a row with the never-EOS set runs to its cap.
            let admitted = self.decoder.admit_with_eos(
                session,
                prepared.suffix.clone(),
                sampling,
                req.params.seed,
                &req.params.eos_ids,
                req.params.max_tokens,
            );
            match admitted {
                Ok(id) => {
                    self.counters.admitted += 1;
                    self.inflight.insert(
                        id,
                        InFlight {
                            tx,
                            cancel: req.cancel,
                            slot,
                            reuse: prepared.action,
                            reused: prepared.reused,
                            prefilled: prepared.suffix.len(),
                            turn,
                            joined_beside,
                        },
                    );
                }
                Err(rejected) => {
                    // The checkout commits only on an admitted row: a
                    // driver that refuses this request — its own body's
                    // validation, which is stricter than the scheduler's
                    // on the speculative loop, an empty prepared suffix,
                    // no drafter slot left — hands the session back, and
                    // its sequence returns to the pool. Otherwise every
                    // such refusal costs the pool one sequence and valid
                    // requests later fail admission with nothing live.
                    let _ = tx.send(BatchEvent::Error(rejected.error.to_string()));
                    self.pool.release(slot, rejected.session.into_sequence());
                }
            }
        }
    }

    /// One continuous step: cancellations, the step, the retirements.
    fn step_continuous(&mut self) -> Result<()> {
        for (id, row) in &self.inflight {
            if row.cancel.load(Ordering::Acquire) {
                self.decoder.cancel(*id);
            }
        }
        #[cfg(feature = "metal")]
        let t0 = self.window.as_ref().map(|_| {
            use lloom_tensor::kernels::metal::profile;
            profile::set_step(self.counters.steps);
            (profile::now_s(), profile::dispatch_count())
        });
        let inflight = &self.inflight;
        let report = self.decoder.step(&mut |row, ids| {
            if let Some(f) = inflight.get(&row) {
                if f.tx.send(BatchEvent::Tokens(ids.to_vec())).is_err() {
                    // The request thread is gone: end the row at the next step.
                    f.cancel.store(true, Ordering::Release);
                }
            }
            Ok(())
        })?;
        #[cfg(feature = "metal")]
        if let (Some(w), Some((t0, d0))) = (self.window.as_mut(), t0) {
            use lloom_tensor::kernels::metal::profile;
            let t1 = profile::now_s();
            w.steps.push(WindowStep {
                step: self.counters.steps,
                t0,
                t1,
                report,
                phases: self.decoder.step_phases(),
                dispatches: profile::dispatch_count().saturating_sub(d0),
            });
        }
        self.counters.steps += 1;
        self.counters.max_live = self.counters.max_live.max(report.live);
        self.counters.max_pieces = self.counters.max_pieces.max(report.pieces);
        if report.held > 0 {
            self.counters.held_steps += 1;
        }
        if lloom::tuning::tuning().lfm2.share_trace {
            eprintln!(
                "[share: step {}: live {} held {} pieces {} ({} tokens) retired {} pending {}]",
                self.counters.steps,
                report.live,
                report.held,
                report.pieces,
                report.piece_tokens,
                report.retired,
                report.pending
            );
        }
        for (id, session, out) in self.decoder.take_finished() {
            let Some(f) = self.inflight.remove(&id) else {
                continue;
            };
            self.counters.retired += 1;
            let cancelled = f.cancel.load(Ordering::Acquire) && !out.eos;
            let seq = session.into_sequence();
            self.pool.release(f.slot, seq);
            let _ = f.tx.send(BatchEvent::Done(Box::new(BatchDone {
                tokens: out.ids,
                eos: out.eos,
                cancelled,
                stats: out.stats,
                reuse: f.reuse,
                reused: f.reused,
                prefilled: f.prefilled,
                turn: f.turn,
                joined_beside: f.joined_beside,
            })));
        }
        if let Some(cb) = &self.on_counters {
            cb(&self.counters);
        }
        Ok(())
    }

    /// One static batch: the waiting rows up to the width, run to completion.
    /// A failed batch's rows go back with their cached prefixes when the
    /// body reports that the failure touched none of their state; otherwise
    /// they are recovered ([`Self::recover_failed`], `batched-decode-design.md`
    /// §11.10).
    fn run_static(&mut self) -> Result<()> {
        let mut rows: Vec<(BatchRequest, Sender<BatchEvent>)> = Vec::new();
        while rows.len() < self.max_rows {
            let Some(r) = self.waiting.pop_front() else {
                break;
            };
            rows.push(r);
        }
        if rows.is_empty() {
            return Ok(());
        }
        if let Err(e) = self.recover_failed() {
            for (_, tx) in rows {
                let _ = tx.send(BatchEvent::Error(recovery_pending(&e)));
            }
            return Ok(());
        }
        let mut sessions = Vec::with_capacity(rows.len());
        let mut meta: Vec<Option<StaticRowMeta>> = Vec::with_capacity(rows.len());
        let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(rows.len());
        let mut specs: Vec<RowSpec> = Vec::with_capacity(rows.len());
        for (req, tx) in rows {
            let sampling = match Self::sampling_of(&req.params) {
                Ok(s) => s,
                Err(e) => {
                    let _ = tx.send(BatchEvent::Error(e));
                    continue;
                }
            };
            match self.prepare(&req) {
                Ok((slot, session, prepared, turn)) => {
                    // A static body takes the whole prompt as its row: a
                    // resume (empty suffix) has nothing to give it.
                    if prepared.suffix.is_empty() {
                        self.pool.release(slot, session.into_sequence());
                        let _ = tx.send(BatchEvent::Error(
                            "batch scheduler: a static batch cannot resume a fully cached prompt"
                                .to_string(),
                        ));
                        continue;
                    }
                    sessions.push(session);
                    prompts.push(prepared.suffix.clone());
                    specs.push(RowSpec {
                        sampling,
                        seed: req.params.seed,
                        eos_ids: req.params.eos_ids.clone(),
                        max_tokens: req.params.max_tokens,
                    });
                    meta.push(Some((slot, prepared, tx, req.cancel, turn)));
                }
                Err(e) => {
                    let _ = tx.send(BatchEvent::Error(e.to_string()));
                }
            }
        }
        if sessions.is_empty() {
            return Ok(());
        }
        self.counters.static_batches += 1;
        self.counters.max_live = self.counters.max_live.max(sessions.len());
        let prompt_refs: Vec<&[u32]> = prompts.iter().map(|p| p.as_slice()).collect();
        let outcome = {
            let BatchMode::Static(body) = &mut self.mode else {
                unreachable!("run_static under the continuous mode")
            };
            let mut idle = Idle {
                model: self.model,
                device: &self.device,
                pool: &mut self.pool,
                rings: &mut self.rings,
                opts: &self.opts,
                extra: &mut self.extra,
                driver: &mut self.decoder,
            };
            let senders: Vec<&Sender<BatchEvent>> = meta
                .iter()
                .map(|m| &m.as_ref().expect("a kept row").2)
                .collect();
            body(
                &mut idle,
                &mut sessions,
                &prompt_refs,
                &specs,
                &mut |i, ids| {
                    let _ = senders[i].send(BatchEvent::Tokens(ids.to_vec()));
                    Ok(())
                },
            )
        };
        let outs = match outcome {
            Ok(outs) => outs,
            Err(e) => {
                let touched = self.extra.touched(&e);
                let msg = e.to_string();
                for (session, m) in sessions.into_iter().zip(meta) {
                    let (slot, _, tx, _, _) = m.expect("a kept row");
                    if touched {
                        self.failed.push((slot, session.into_sequence()));
                    } else {
                        self.pool.release(slot, session.into_sequence());
                    }
                    let _ = tx.send(BatchEvent::Error(msg.clone()));
                }
                if let Err(fence) = self.recover_failed() {
                    eprintln!(
                        "[batch: a failed static batch's fence refused; its {} rows are held \
                         until it succeeds: {fence}]",
                        self.failed.len()
                    );
                }
                return Ok(());
            }
        };
        for ((session, m), out) in sessions.into_iter().zip(meta).zip(outs) {
            let (slot, prepared, tx, cancel, turn) = m.expect("a kept row");
            self.counters.admitted += 1;
            self.counters.retired += 1;
            self.pool.release(slot, session.into_sequence());
            let _ = tx.send(BatchEvent::Done(Box::new(BatchDone {
                tokens: out.ids,
                eos: out.eos,
                cancelled: cancel.load(Ordering::Acquire) && !out.eos,
                stats: out.stats,
                reuse: prepared.action,
                reused: prepared.reused,
                prefilled: prepared.suffix.len(),
                turn,
                joined_beside: 0,
            })));
        }
        if let Some(cb) = &self.on_counters {
            cb(&self.counters);
        }
        Ok(())
    }

    /// A failed continuous step ends every row in it with the error. The
    /// driver's state is not trusted past the failure, so every row is
    /// drained, its log settled to what its cache holds. A failure that
    /// touched no row's state returns each sequence to its own slot as it
    /// is, its cached prefix kept; one that may have is recovered as a
    /// failed static batch is (fence, abort, reset) and held while the
    /// fence refuses (`batched-decode-design.md` §11.11).
    fn fail_continuous(&mut self, error: lloom_tensor::Error) {
        let msg = error.to_string();
        let touched = self.extra.touched(&error);
        let ids = self.decoder.row_ids();
        let sessions = self.decoder.drain_rows();
        for (id, session) in ids.into_iter().zip(sessions) {
            let row = self.inflight.remove(&id);
            debug_assert!(row.is_some(), "a driver row the scheduler did not admit");
            let Some(row) = row else { continue };
            let _ = row.tx.send(BatchEvent::Error(msg.clone()));
            let seq = session.into_sequence();
            if touched {
                self.failed.push((row.slot, seq));
            } else {
                self.pool.release(row.slot, seq);
            }
        }
        for (_, row) in self.inflight.drain() {
            let _ = row.tx.send(BatchEvent::Error(msg.clone()));
        }
        if let Err(fence) = self.recover_failed() {
            eprintln!(
                "[batch: a failed step's fence refused; its {} rows are held until it \
                 succeeds: {fence}]",
                self.failed.len()
            );
        }
    }

    /// Recover a failed static batch's or continuous step's sequences
    /// ([`Self::failed`]). The failure returned with work still on the
    /// device — the failed pass, the other rows' enqueued verifies — and with
    /// every row's state possibly moved: a pass may advance its cache before
    /// the part that failed, and the sequence's log and position say nothing
    /// of it. So the extra's fence waits that work out, the extra and the
    /// continuous driver forget their per-row state (the drafter's contexts
    /// live in whichever holds the drafter), and every sequence is reset
    /// before it returns to its slot. A refused fence resets nothing and
    /// keeps them held.
    fn recover_failed(&mut self) -> std::result::Result<(), String> {
        if self.failed.is_empty() {
            return Ok(());
        }
        self.extra.fence(&self.device).map_err(|e| e.to_string())?;
        self.extra.abort();
        self.decoder.abort();
        for (slot, seq) in self.failed.drain(..) {
            let mut session = Session::from_sequence(self.model, seq);
            session.reset();
            self.pool.release(slot, session.into_sequence());
        }
        Ok(())
    }

    /// Open a profile window when requests arrive on an idle engine and the
    /// profile build's recorder is on: the counters start from zero.
    #[cfg(feature = "metal")]
    fn open_window(&mut self) {
        use lloom_tensor::kernels::metal::profile;
        if self.window.is_some()
            || self.waiting.is_empty()
            || !self.nothing_live()
            || !profile::recorder_enabled()
        {
            return;
        }
        if self.device.synchronize().is_err() {
            return;
        }
        profile::reset();
        self.window = Some(ProfileWindow {
            started: std::time::Instant::now(),
            steps: Vec::new(),
        });
    }

    /// Close the window once nothing is live and nothing waits: the GPU's
    /// busy union against the window's wall on stderr, and under
    /// `diagnostics.gpu_profile.dump` the command buffers
    /// (`<dump>.batch<n>.csv`), the steps (`<dump>.batch<n>.steps.csv`) and,
    /// with the encoder counters on, every sampled encoder
    /// (`<dump>.batch<n>.encoders.csv`).
    #[cfg(feature = "metal")]
    fn close_window(&mut self) {
        use lloom_tensor::kernels::metal::profile;
        if self.window.is_none() || !self.nothing_live() || !self.waiting.is_empty() {
            return;
        }
        let w = self.window.take().expect("checked");
        let _ = self.device.synchronize();
        let wall = w.started.elapsed().as_secs_f64();
        let s = profile::snapshot();
        let idle: f64 = s.idle_gaps.iter().map(|g| g.idle).sum();
        let starved: f64 = s.idle_gaps.iter().map(|g| g.starved).sum();
        let n = self.windows;
        self.windows += 1;
        eprintln!(
            "[profile: batch window {n}: {} steps, {} with pieces, {} holding rows; gpu_busy_union {:.4} s /              wall {:.4} s = {:.4} busy/wall; {} command buffers, idle between them {:.4} s (starved {:.4} s)]",
            w.steps.len(),
            w.steps.iter().filter(|x| x.report.pieces > 0).count(),
            w.steps.iter().filter(|x| x.report.held > 0).count(),
            s.gpu_busy_union_s,
            wall,
            s.gpu_busy_union_s / wall.max(1e-9),
            s.timed_command_buffers,
            idle,
            starved,
        );
        let intervals: Vec<(f64, f64)> = s
            .command_buffer_records
            .iter()
            .map(|r| (r.interval.start, r.interval.end))
            .collect();
        for row in steady_rows(&w.steps, &intervals) {
            let stages: Vec<String> = STAGE_NAMES
                .iter()
                .zip(row.stages_ms)
                .map(|(name, ms)| format!("{name} {ms:.3}"))
                .collect();
            eprintln!(
                "[profile: batch window {n} decode at {} live: {} steps; median ms/step: wall {:.3}, \
                 gpu_busy {:.3}; host {}; between steps {:.3}; {:.0} dispatches/step]",
                row.live,
                row.steps,
                row.wall_ms,
                row.gpu_busy_ms,
                stages.join(", "),
                row.between_ms,
                row.dispatches,
            );
        }
        let Some(path) = lloom_tensor::tuning::tuning()
            .instrument
            .profile_dump
            .clone()
        else {
            return;
        };
        let mut cb = String::from("step,committed_s,start_s,end_s\n");
        for r in &s.command_buffer_records {
            cb.push_str(&format!(
                "{},{:.6},{:.6},{:.6}\n",
                r.step, r.committed, r.interval.start, r.interval.end
            ));
        }
        let mut st = String::from(
            "step,host_start_s,host_end_s,live,held,pieces,piece_tokens,retired,pending,dispatches",
        );
        for name in STAGE_NAMES {
            st.push_str(&format!(",{name}_ns"));
        }
        st.push('\n');
        for x in &w.steps {
            let r = &x.report;
            st.push_str(&format!(
                "{},{:.6},{:.6},{},{},{},{},{},{},{}",
                x.step,
                x.t0,
                x.t1,
                r.live,
                r.held,
                r.pieces,
                r.piece_tokens,
                r.retired,
                r.pending,
                x.dispatches
            ));
            for ns in stage_array(&x.phases) {
                st.push_str(&format!(",{ns}"));
            }
            st.push('\n');
        }
        let mut census = String::from("label,dispatches\n");
        for (label, count) in &s.by_label {
            census.push_str(&format!("{label},{count}\n"));
        }
        let mut files = vec![
            (format!("{path}.batch{n}.csv"), cb),
            (format!("{path}.batch{n}.steps.csv"), st),
            (format!("{path}.batch{n}.census.csv"), census),
        ];
        if !s.encoder_records.is_empty() {
            let mut enc = String::from("step,label,start_s,end_s,dispatches\n");
            for e in &s.encoder_records {
                enc.push_str(&format!(
                    "{},{},{:.6},{:.6},{}\n",
                    e.step,
                    e.label.as_deref().unwrap_or("-"),
                    e.start,
                    e.end,
                    e.dispatches
                ));
            }
            files.push((format!("{path}.batch{n}.encoders.csv"), enc));
        }
        for (p, body) in files {
            match std::fs::write(&p, body) {
                Ok(()) => eprintln!("[profile: wrote {p}]"),
                Err(e) => eprintln!("[profile: writing {p} failed: {e}]"),
            }
        }
    }

    /// The loop, until shutdown or every sender is dropped.
    fn run(mut self) {
        loop {
            if !self.take_jobs() {
                break;
            }
            // Closed jobs run with nothing live; while one is queued no new
            // row is admitted, so the batch drains toward it.
            if !self.closed.is_empty() && self.nothing_live() {
                while let Some(job) = self.closed.pop_front() {
                    self.counters.closed_jobs += 1;
                    let mut idle = self.idle();
                    job(&mut idle);
                }
                continue;
            }
            match self.mode {
                BatchMode::Continuous => {
                    if self.closed.is_empty() {
                        #[cfg(feature = "metal")]
                        self.open_window();
                        self.admit_waiting();
                    }
                    if !self.decoder.is_idle() {
                        if let Err(e) = self.step_continuous() {
                            self.fail_continuous(e);
                        }
                    }
                    #[cfg(feature = "metal")]
                    self.close_window();
                }
                BatchMode::Static(_) => {
                    if self.closed.is_empty() && !self.waiting.is_empty() {
                        if let Err(e) = self.run_static() {
                            eprintln!("[batch: static batch failed: {e}]");
                        }
                    }
                }
            }
        }
        // Drain: every waiting row learns the scheduler is gone.
        for (_, tx) in self.waiting.drain(..) {
            let _ = tx.send(BatchEvent::Error(
                "the batch scheduler has stopped".to_string(),
            ));
        }
    }
}

/// Start the scheduler on its own thread over `model`, which the thread
/// borrows for its life (`shared` keeps it alive; the thread holds a clone).
///
/// Returns the handle the request threads submit through. The thread exits
/// on [`BatchHandle::shutdown`] or when every handle is dropped.
pub fn spawn<S, M, E>(
    shared: Arc<S>,
    model_of: impl Fn(&S) -> &M + Send + 'static,
    cfg: BatchConfig<M, E>,
) -> std::result::Result<(BatchHandle<M, E>, std::thread::JoinHandle<()>), String>
where
    S: Send + Sync + 'static,
    M: BatchedCausalLM + Sync + 'static,
    M::Cache: CacheReset + CacheRewind + CacheShare + Send + 'static,
    E: MakeDriver<M> + BatchRecovery + Send + 'static,
{
    let (tx, rx) = mpsc::channel();
    let (ready_tx, ready_rx) = mpsc::channel::<std::result::Result<(), String>>();
    let thread = std::thread::Builder::new()
        .name("lloom-batch".to_string())
        .spawn(move || {
            let model: &M = model_of(&shared);
            match BatchScheduler::new(model, cfg, rx) {
                Ok(sched) => {
                    let _ = ready_tx.send(Ok(()));
                    sched.run();
                }
                Err(e) => {
                    let _ = ready_tx.send(Err(e.to_string()));
                }
            }
        })
        .map_err(|e| format!("spawning the batch thread: {e}"))?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok((BatchHandle { tx }, thread)),
        Ok(Err(e)) => Err(e),
        Err(_) => Err("the batch thread exited before it was ready".to_string()),
    }
}

#[cfg(test)]
mod tests {
    //! The scheduler over a device-only fake model: the `Chain` of lloom's
    //! `batched_decode_tests` — the next token is a fixed function of the
    //! last, so every row's expected stream derives from its prompt.
    use super::*;
    use lloom::nn::CausalLM;
    use lloom::nn::{CacheReset, CacheRewind, Model as ModelTrait, RaggedRow, Row, VarBuilder};
    use lloom_tensor::{DType, Tensor};
    use std::time::Duration;

    const VOCAB: usize = 64;
    const EOS: u32 = 42;

    fn next_of(tok: u32) -> u32 {
        (tok * 7 + 3) % VOCAB as u32
    }

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

    /// A position counter as the cache: reset to 0, rewound by n.
    struct Pos(usize);

    impl CacheReset for Pos {
        fn reset(&mut self) {
            self.0 = 0;
        }
    }

    impl CacheShare for Pos {
        fn share_prefix_from(&mut self, _src: &Self, _n: usize) -> lloom_tensor::Result<usize> {
            Ok(0)
        }
        fn share_granularity(&self) -> usize {
            0
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

        fn logits_of(&self, tokens: &Tensor) -> Result<Tensor> {
            let (b, seq) = tokens.dims2()?;
            let last = tokens.narrow(1, seq - 1, 1)?.reshape(b)?;
            let next = self.table.index_select(&last, 0)?.to_dtype(DType::U32)?;
            self.eye.index_select(&next, 0)
        }
    }

    impl ModelTrait for Chain {
        type Config = ();
        fn from_weights(_vb: VarBuilder, _cfg: &()) -> Result<Self> {
            unreachable!()
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
        fn forward_embeds(&self, embeds: &Tensor, pos: usize, cache: &mut Pos) -> Result<Tensor> {
            assert_eq!(
                cache.0, pos,
                "forward at {pos} against a cache of {}",
                cache.0
            );
            cache.0 += embeds.dim(1)?;
            self.logits_of(embeds)
        }
        fn vocab_size(&self) -> usize {
            VOCAB
        }
    }

    impl BatchedCausalLM for Chain {
        fn forward_rows(&self, tokens: &Tensor, rows: &mut [Row<'_, Pos>]) -> Result<Tensor> {
            for row in rows.iter_mut() {
                assert_eq!(row.cache.0, row.pos);
                row.cache.0 += 1;
            }
            self.logits_of(tokens)
        }
        fn forward_ragged(
            &self,
            tokens: &Tensor,
            rows: &mut [RaggedRow<'_, Pos>],
        ) -> Result<Tensor> {
            let mut lasts = Vec::with_capacity(rows.len());
            let mut off = 0;
            for row in rows.iter_mut() {
                assert_eq!(row.cache.0, row.pos);
                row.cache.0 += row.len;
                off += row.len;
                lasts.push((off - 1) as u32);
            }
            let last = tokens
                .squeeze(0)?
                .index_select(&Tensor::new(lasts.as_slice(), tokens.device())?, 0)?;
            let next = self.table.index_select(&last, 0)?.to_dtype(DType::U32)?;
            self.eye.index_select(&next, 0)
        }
    }

    fn params(max_tokens: usize) -> DecodeParams {
        DecodeParams {
            max_tokens,
            eos_ids: vec![EOS],
            ..DecodeParams::default()
        }
    }

    /// `scheduler.intake_per_tick`'s default, which these fixtures run at.
    const INTAKE_PER_TICK: usize = crate::options::DEFAULT_INTAKE_PER_TICK;

    fn start(
        max_rows: usize,
        chunk: usize,
        mode: BatchMode<Chain, ()>,
    ) -> (BatchHandle<Chain, ()>, std::thread::JoinHandle<()>) {
        start_with(max_rows, chunk, 1, PrefillDecode::Ride, mode, None)
    }

    fn start_with(
        max_rows: usize,
        chunk: usize,
        prefill_batch: usize,
        prefill_decode: PrefillDecode,
        mode: BatchMode<Chain, ()>,
        on_counters: Option<CountersHook>,
    ) -> (BatchHandle<Chain, ()>, std::thread::JoinHandle<()>) {
        // The option's default; the fairness test reads the same constant.
        let intake_per_tick = INTAKE_PER_TICK;
        let device = Device::Cpu;
        let core = Arc::new(Chain::new(&device).unwrap());
        let pool = SlotPool::new(max_rows, || Pos(0));
        let cfg = BatchConfig {
            device,
            pool,
            rings: RingPool::new(),
            opts: DecodeOptions {
                lookahead: 1,
                prefill_chunk: chunk,
                ..DecodeOptions::new(0)
            },
            max_rows,
            reuse_prefix: false,
            share_prefix: false,
            prefill_batch,
            prefill_decode,
            intake_per_tick,
            mode,
            extra: (),
            on_counters,
        };
        spawn(core, |c: &Chain| c, cfg).unwrap()
    }

    fn collect(rx: &Receiver<BatchEvent>) -> (Vec<u32>, BatchDone) {
        let mut streamed = Vec::new();
        loop {
            match rx.recv_timeout(Duration::from_secs(30)).expect("an event") {
                BatchEvent::Tokens(ids) => streamed.extend(ids),
                BatchEvent::Done(d) => return (streamed, *d),
                BatchEvent::Error(e) => panic!("row failed: {e}"),
            }
        }
    }

    /// Hold the scheduler thread in a closed job until the returned sender
    /// fires (or is dropped). Returns once the job is running: no row is
    /// admitted while a closed job is queued or running, so every request
    /// submitted meanwhile is waiting when the gate opens.
    fn hold<M: BatchedCausalLM, E>(handle: &BatchHandle<M, E>) -> Sender<()> {
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (held_tx, held_rx) = mpsc::channel::<()>();
        assert!(handle.run_closed(Box::new(move |_| {
            let _ = held_tx.send(());
            let _ = release_rx.recv();
        })));
        held_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("the gate job runs");
        release_tx
    }

    /// Requests submitted from several threads at once, more than the batch
    /// width, each stream its own chain; the pool serves them in turn and
    /// every sequence comes back.
    ///
    /// The scheduler is held until all four are queued. Without the gate, a
    /// row of this toy model can finish before the next thread submits, and
    /// whether any row ever decodes beside another is the threads' timing.
    #[test]
    fn concurrent_requests_each_decode_their_own_chain() {
        let (handle, thread) = start(2, 2, BatchMode::Continuous);
        let prompts: Vec<Vec<u32>> = vec![vec![3, 1], vec![0], vec![9, 9, 9, 5], vec![7, 7, 2]];
        let gate = hold(&handle);
        let (queued_tx, queued_rx) = mpsc::channel::<()>();
        let results: Vec<(Vec<u32>, BatchDone)> = std::thread::scope(|scope| {
            let hs: Vec<_> = prompts
                .iter()
                .map(|p| {
                    let h = handle.clone();
                    let queued = queued_tx.clone();
                    scope.spawn(move || {
                        let rx = h.submit(BatchRequest {
                            prompt_ids: p.clone(),
                            params: params(12),
                            cancel: Arc::new(AtomicBool::new(false)),
                        });
                        queued.send(()).unwrap();
                        collect(&rx)
                    })
                })
                .collect();
            for _ in &prompts {
                queued_rx
                    .recv_timeout(Duration::from_secs(30))
                    .expect("every request is queued");
            }
            gate.send(()).unwrap();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (p, (streamed, done)) in prompts.iter().zip(&results) {
            let (want, eos) = chain(*p.last().unwrap(), 12);
            assert_eq!(&done.tokens, &want, "prompt {p:?}");
            assert_eq!(streamed, &want);
            assert_eq!(done.eos, eos);
            assert!(!done.cancelled);
            assert_eq!(
                done.prefilled,
                p.len(),
                "a reset sequence prefills the whole prompt"
            );
        }
        // Non-vacuity: at width 2 with four rows queued, at least one row
        // joined a batch that already had a live row, whatever order the
        // four were queued in.
        assert!(
            results.iter().any(|(_, d)| d.joined_beside >= 1),
            "no row ever shared a step: {:?}",
            results
                .iter()
                .map(|(_, d)| d.joined_beside)
                .collect::<Vec<_>>()
        );
        handle.shutdown();
        thread.join().unwrap();
    }

    /// The scheduler hands its prefill policy to the K=1 loop: four requests
    /// queued at once at `prefill_batch` 4 are fed together (a step carries
    /// all four rows' pieces), and under the pause arm a row streaming while
    /// another prompt is fed is held; every stream is its chain.
    ///
    /// At one chunk a step and the pause arm, steps 1–2 feed row 0, 3–4 row
    /// 1, 5–6 row 2, 7–8 row 3; a row rides the step after it joins (its
    /// first token) and is held from the next, so steps 4–8 hold a row.
    #[test]
    fn queued_prompts_are_prefilled_together_and_each_decodes_its_own_chain() {
        for (k, arm, held_steps) in [
            (1, PrefillDecode::Ride, 0),
            (4, PrefillDecode::Ride, 0),
            (1, PrefillDecode::Pause, 5),
            (4, PrefillDecode::Pause, 0),
        ] {
            let seen = Arc::new(parking_lot::Mutex::new(BatchCounters::default()));
            let sink = Arc::clone(&seen);
            let hook: CountersHook = Box::new(move |c| *sink.lock() = *c);
            let (handle, thread) = start_with(4, 2, k, arm, BatchMode::Continuous, Some(hook));
            let prompts: Vec<Vec<u32>> = vec![
                vec![9, 9, 9, 1],
                vec![9, 9, 9, 0],
                vec![9, 9, 9, 5],
                vec![9, 9, 9, 7],
            ];
            let gate = hold(&handle);
            let rxs: Vec<Receiver<BatchEvent>> = prompts
                .iter()
                .map(|p| {
                    handle.submit(BatchRequest {
                        prompt_ids: p.clone(),
                        params: params(12),
                        cancel: Arc::new(AtomicBool::new(false)),
                    })
                })
                .collect();
            gate.send(()).unwrap();
            for (p, rx) in prompts.iter().zip(&rxs) {
                let (streamed, done) = collect(rx);
                let (want, eos) = chain(*p.last().unwrap(), 12);
                assert_eq!(done.tokens, want, "k {k} {arm}: prompt {p:?}");
                assert_eq!(streamed, want, "k {k} {arm}: prompt {p:?}");
                assert_eq!(done.eos, eos);
            }
            handle.shutdown();
            thread.join().unwrap();
            let c = *seen.lock();
            // At a chunk of two, one row's piece a step at k = 1, all four
            // rows' at k = 4 (they were queued before the first step).
            assert_eq!(c.max_pieces, k, "k {k} {arm}: {c:?}");
            assert_eq!(c.held_steps, held_steps, "k {k} {arm}: {c:?}");
        }
    }

    /// A row draws through the batched driver, never through the request's
    /// sampler, so a request carrying logits processors is refused by name
    /// rather than served without them; the next plain request still runs.
    #[test]
    fn a_request_with_processors_is_refused_by_name() {
        let (handle, thread) = start(2, 0, BatchMode::Continuous);
        let mut with = params(12);
        with.processors.presence_penalty = 1.0;
        let rx = handle.submit(BatchRequest {
            prompt_ids: vec![3, 1],
            params: with,
            cancel: Arc::new(AtomicBool::new(false)),
        });
        match rx.recv_timeout(Duration::from_secs(30)).expect("an event") {
            BatchEvent::Error(e) => assert!(
                e.starts_with("the batch scheduler: a batched body does not run logits processors"),
                "{e}"
            ),
            BatchEvent::Tokens(_) | BatchEvent::Done(_) => panic!("served with processors"),
        }
        let rx = handle.submit(BatchRequest {
            prompt_ids: vec![3, 1],
            params: params(12),
            cancel: Arc::new(AtomicBool::new(false)),
        });
        let (_, done) = collect(&rx);
        assert_eq!(done.tokens, chain(1, 12).0);
        handle.shutdown();
        thread.join().unwrap();
    }

    /// A cancelled row retires with what it produced; a closed job runs
    /// once nothing is live and sees every sequence back in the pool.
    #[test]
    fn a_cancel_retires_the_row_and_a_closed_job_waits_for_an_idle_engine() {
        let (handle, thread) = start(2, 0, BatchMode::Continuous);
        let cancel = Arc::new(AtomicBool::new(false));
        // Row 1's chain never reaches EOS: it runs to its cap. The cap is
        // far past what the toy steps between the fifth token's receipt and
        // the cancel below; at 200 an optimised build on a fast CPU ran the
        // chain out before the flag landed and the row finished uncancelled.
        let cap = 1_000_000;
        let rx = handle.submit(BatchRequest {
            prompt_ids: vec![0],
            params: params(cap),
            cancel: Arc::clone(&cancel),
        });
        let mut got = Vec::new();
        while got.len() < 5 {
            match rx.recv_timeout(Duration::from_secs(30)).unwrap() {
                BatchEvent::Tokens(ids) => got.extend(ids),
                other => panic!("unexpected {other:?}"),
            }
        }
        cancel.store(true, Ordering::Release);
        let (streamed, done) = collect(&rx);
        assert!(done.cancelled);
        assert!(!done.eos);
        assert!(done.tokens.len() < cap);
        assert_eq!(done.tokens.len(), got.len() + streamed.len());
        assert_eq!(&done.tokens[..5], &chain(0, 5).0);
        // The closed job runs with the pool whole.
        let idle_len = handle.run_closed_blocking(|idle| idle.pool.idle()).unwrap();
        assert_eq!(idle_len, 2);
        handle.shutdown();
        thread.join().unwrap();
    }

    /// A burst queued while a row is live is taken a bounded number of jobs
    /// per step, so the live row's next step is not postponed behind the
    /// whole queue; nothing in the burst is lost, only deferred.
    ///
    /// The scheduler is held inside its counters hook after a step with row
    /// A live, the burst queued meanwhile, then released: every request
    /// the next tick takes is answered before the step that follows it
    /// (each is cancelled, so `admit_waiting` answers it at once), so the
    /// answers counted at that step are the jobs taken in that tick.
    #[test]
    fn a_queued_burst_cannot_postpone_a_live_row_past_the_intake_budget() {
        const BURST: usize = 4096;
        const BUDGET: usize = INTAKE_PER_TICK;
        let (paused_tx, paused_rx) = mpsc::channel::<u64>();
        let (go_tx, go_rx) = mpsc::channel::<()>();
        let go_rx = parking_lot::Mutex::new(go_rx);
        let hook: CountersHook = Box::new(move |c| {
            if c.steps == 3 || c.steps == 4 {
                let _ = paused_tx.send(c.steps);
                let _ = go_rx.lock().recv();
            }
        });
        let (handle, thread) = start_with(
            2,
            0,
            1,
            PrefillDecode::Ride,
            BatchMode::Continuous,
            Some(hook),
        );
        let live_cancel = Arc::new(AtomicBool::new(false));
        let live = handle.submit(BatchRequest {
            prompt_ids: vec![0],
            params: params(1_000_000),
            cancel: Arc::clone(&live_cancel),
        });
        assert_eq!(paused_rx.recv_timeout(Duration::from_secs(30)).unwrap(), 3);
        let mut before = 0usize;
        while let Ok(BatchEvent::Tokens(ids)) = live.try_recv() {
            before += ids.len();
        }
        assert!(before > 0, "the row is live before the burst");
        let burst: Vec<Receiver<BatchEvent>> = (0..BURST)
            .map(|_| {
                handle.submit(BatchRequest {
                    prompt_ids: vec![5],
                    params: params(4),
                    cancel: Arc::new(AtomicBool::new(true)),
                })
            })
            .collect();
        go_tx.send(()).unwrap();
        assert_eq!(paused_rx.recv_timeout(Duration::from_secs(30)).unwrap(), 4);
        let answered = burst.iter().filter(|rx| rx.try_recv().is_ok()).count();
        let mut after = 0usize;
        while let Ok(BatchEvent::Tokens(ids)) = live.try_recv() {
            after += ids.len();
        }
        assert!(after > 0, "the live row stepped after the release");
        assert!(
            answered <= BUDGET,
            "the tick before the live row's step took {answered} of {BURST} queued jobs \
             (budget {BUDGET}): intake postponed the live row behind the queue"
        );
        live_cancel.store(true, Ordering::Release);
        go_tx.send(()).unwrap();
        // Deferred, not dropped: every queued request is still answered.
        let mut late = 0usize;
        for rx in &burst {
            match rx.recv_timeout(Duration::from_secs(30)) {
                Ok(BatchEvent::Done(d)) => {
                    assert!(d.cancelled);
                    late += 1;
                }
                Ok(other) => panic!("a cancelled request got {other:?}"),
                Err(_) => {}
            }
        }
        assert_eq!(answered + late, BURST, "every queued request is answered");
        let (_, done) = collect(&live);
        assert!(done.cancelled);
        handle.shutdown();
        thread.join().unwrap();
    }

    /// The decode cadence counts a step only when it is one decode step at
    /// its width — no piece, no held row, the next step at the same live
    /// count — and charges it the GPU's busy union inside its interval.
    #[test]
    fn the_decode_cadence_counts_only_whole_decode_steps_at_one_width() {
        let report = |live, pieces| lloom::nn::StepReport {
            live,
            pieces,
            ..Default::default()
        };
        let phases = lloom::nn::BatchStepPhases {
            encode_ns: 2_000_000,
            wait_ns: 7_000_000,
            ..Default::default()
        };
        let step = |step: u64, t0: f64, live, pieces| WindowStep {
            step,
            t0,
            t1: t0 + 0.009,
            report: report(live, pieces),
            phases,
            dispatches: 400,
        };
        // A piece step, three decode steps at 4 live, a step at 3 live
        // (a row retired), then one at 3 to close its interval.
        let steps = vec![
            step(0, 0.000, 4, 1),
            step(1, 0.030, 4, 0),
            step(2, 0.040, 4, 0),
            step(3, 0.050, 4, 0),
            step(4, 0.060, 3, 0),
            step(5, 0.069, 3, 0),
        ];
        // The GPU busy 8 ms of every 10 from 0.030 on, idle at its edges.
        let intervals: Vec<(f64, f64)> = (0..5)
            .map(|k| (0.031 + 0.010 * k as f64, 0.039 + 0.010 * k as f64))
            .collect();
        let rows = steady_rows(&steps, &intervals);
        assert_eq!(rows.len(), 2, "{rows:?}");
        let four = &rows[1];
        assert_eq!(
            (four.live, four.steps),
            (4, 2),
            "step 3's successor is at 3 live"
        );
        assert!((four.wall_ms - 10.0).abs() < 1e-9, "{four:?}");
        assert!((four.gpu_busy_ms - 8.0).abs() < 1e-9, "{four:?}");
        assert!((four.between_ms - 1.0).abs() < 1e-9, "{four:?}");
        assert!((four.stages_ms[5] - 2.0).abs() < 1e-12);
        assert!((four.stages_ms[7] - 7.0).abs() < 1e-12);
        assert_eq!(four.dispatches, 400.0);
        let three = &rows[0];
        assert_eq!((three.live, three.steps), (3, 1));
        assert!((three.wall_ms - 9.0).abs() < 1e-9, "{three:?}");
        assert!((three.gpu_busy_ms - 8.0).abs() < 1e-9, "{three:?}");
    }

    /// The busy union does not double-count overlapping buffers and clips
    /// at the interval's edges.
    #[test]
    fn the_busy_union_inside_an_interval_merges_and_clips() {
        let ivs = [(0.0, 2.0), (1.0, 3.0), (5.0, 6.0), (7.0, 9.0)];
        assert_eq!(super::steady::busy_within(&ivs, 0.5, 8.0), 2.5 + 1.0 + 1.0);
        assert_eq!(super::steady::busy_within(&ivs, 3.0, 5.0), 0.0);
    }

    /// The static mode runs the waiting rows through the body as one batch
    /// and hands each its outcome.
    #[test]
    fn a_static_batch_runs_the_body_once_for_every_waiting_row() {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let seen = Arc::clone(&calls);
        let body: StaticBody<Chain, ()> =
            Box::new(move |idle, sessions, prompts, specs, on_tokens| {
                seen.fetch_add(1, Ordering::AcqRel);
                assert_eq!(sessions.len(), prompts.len());
                let mut sampler =
                    lloom::generation::LogitsProcessor::from_sampling(0, Sampling::ArgMax);
                let opts = DecodeOptions {
                    lookahead: 1,
                    ..DecodeOptions::new(specs.iter().map(|s| s.max_tokens).max().unwrap())
                };
                Session::decode_batched(
                    sessions,
                    &mut sampler,
                    prompts,
                    &opts,
                    &specs[0].eos_ids,
                    idle.device,
                    on_tokens,
                )
            });
        let (handle, thread) = start(3, 0, BatchMode::Static(body));
        let prompts: Vec<Vec<u32>> = vec![vec![3, 1], vec![0], vec![9, 9, 9, 5]];
        let rxs: Vec<_> = prompts
            .iter()
            .map(|p| {
                handle.submit(BatchRequest {
                    prompt_ids: p.clone(),
                    params: params(12),
                    cancel: Arc::new(AtomicBool::new(false)),
                })
            })
            .collect();
        for (p, rx) in prompts.iter().zip(&rxs) {
            let (streamed, done) = collect(rx);
            let (want, eos) = chain(*p.last().unwrap(), 12);
            assert_eq!(done.tokens, want);
            assert_eq!(streamed, want);
            assert_eq!(done.eos, eos);
        }
        // Three rows submitted before the thread woke: one body call, or at
        // most three if the scheduler ran them one by one — never more.
        let n = calls.load(Ordering::Acquire);
        assert!((1..=3).contains(&n), "{n} body calls");
        handle.shutdown();
        thread.join().unwrap();
    }

    /// What [`Faulty`]'s next forward does.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Fault {
        None,
        /// Moves the cache past the fed positions, then fails: a pass whose
        /// state advances before its fallible part (LFM2's conv phase ahead
        /// of the layer walk).
        MutateThenFail,
        /// Refuses before the cache moves: an admission refusal ahead of
        /// every layer (LFM2's `prepare_pass`).
        RefuseUntouched,
        /// Left for a batched step (`forward_rows`, `forward_ragged`), which
        /// moves every row's cache by what it fed and then fails: a
        /// continuous step whose rows advanced before its fallible part.
        RowsMutateThenFail,
        /// Left for a batched step, which refuses before any row's cache
        /// moves: a continuous step whose failure touched no state.
        RowsRefuseUntouched,
    }

    impl Fault {
        fn for_rows(self) -> bool {
            matches!(self, Fault::RowsMutateThenFail | Fault::RowsRefuseUntouched)
        }
    }

    /// A failure reported as a model reports it: refused before any state
    /// moved, or after some did.
    fn untouched(msg: &str) -> lloom_tensor::Error {
        lloom_tensor::Error::msg(msg).with_state_touched(lloom_tensor::StateTouched::No)
    }
    fn touched(msg: &str) -> lloom_tensor::Error {
        lloom_tensor::Error::msg(msg).with_state_touched(lloom_tensor::StateTouched::Yes)
    }

    /// The chain over a cache that counts the positions it holds. A forward
    /// at `pos` onto a cache holding `held` reads the drift `held - pos`
    /// into its next token, so a stale cache decodes a different stream
    /// rather than panicking. Every forward logs its `(pos, held)` on entry.
    struct Faulty {
        chain: Chain,
        fault: parking_lot::Mutex<Fault>,
        /// Batched steps a rows fault lets through before it fires.
        rows_skip: parking_lot::Mutex<usize>,
        entries: parking_lot::Mutex<Vec<(usize, usize)>>,
    }

    impl Faulty {
        fn new(device: &Device) -> Result<Self> {
            Ok(Self {
                chain: Chain::new(device)?,
                fault: parking_lot::Mutex::new(Fault::None),
                rows_skip: parking_lot::Mutex::new(0),
                entries: parking_lot::Mutex::new(Vec::new()),
            })
        }

        fn arm(&self, fault: Fault) {
            *self.fault.lock() = fault;
        }

        /// Arm a rows fault for the batched step after the next `skip`.
        fn arm_rows(&self, fault: Fault, skip: usize) {
            assert!(fault.for_rows());
            *self.rows_skip.lock() = skip;
            *self.fault.lock() = fault;
        }

        fn entries(&self) -> Vec<(usize, usize)> {
            self.entries.lock().clone()
        }

        /// The fault a single-sequence forward consumes: a batched step's is
        /// left for the batched step.
        fn take_single(&self) -> Fault {
            let mut f = self.fault.lock();
            if f.for_rows() {
                return Fault::None;
            }
            std::mem::replace(&mut *f, Fault::None)
        }

        /// The rows fault the batched step about to run fires, if any.
        fn take_rows(&self) -> Fault {
            let mut f = self.fault.lock();
            if !f.for_rows() {
                return Fault::None;
            }
            let mut skip = self.rows_skip.lock();
            if *skip > 0 {
                *skip -= 1;
                return Fault::None;
            }
            std::mem::replace(&mut *f, Fault::None)
        }

        fn next_rows(&self, rows: &[(u32, usize, usize)]) -> Result<Tensor> {
            let next: Vec<u32> = rows
                .iter()
                .map(|&(last, pos, held)| {
                    let drift = held.wrapping_sub(pos) % VOCAB;
                    (next_of(last) + drift as u32) % VOCAB as u32
                })
                .collect();
            let ids = Tensor::new(next.as_slice(), self.chain.eye.device())?;
            self.chain.eye.index_select(&ids, 0)
        }
    }

    impl ModelTrait for Faulty {
        type Config = ();
        fn from_weights(_vb: VarBuilder, _cfg: &()) -> Result<Self> {
            unreachable!()
        }
    }

    impl CausalLM for Faulty {
        type Cache = Pos;
        fn new_cache(&self, _cfg: &(), _dev: &Device) -> Result<Pos> {
            Ok(Pos(0))
        }
        fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
            Ok(tokens.clone())
        }
        fn forward_embeds(&self, embeds: &Tensor, pos: usize, cache: &mut Pos) -> Result<Tensor> {
            let n = embeds.dim(1)?;
            let held = cache.0;
            self.entries.lock().push((pos, held));
            match self.take_single() {
                Fault::None | Fault::RowsMutateThenFail | Fault::RowsRefuseUntouched => {}
                Fault::MutateThenFail => {
                    cache.0 += n;
                    return Err(touched("injected: the pass moved the cache, then failed"));
                }
                Fault::RefuseUntouched => {
                    return Err(untouched(
                        "injected: the pass was refused before anything moved",
                    ));
                }
            }
            cache.0 += n;
            let ids = embeds.to_vec2::<u32>()?;
            let last = *ids[0].last().expect("a non-empty forward");
            self.next_rows(&[(last, pos, held)])
        }
        fn vocab_size(&self) -> usize {
            VOCAB
        }
    }

    impl BatchedCausalLM for Faulty {
        fn forward_rows(&self, tokens: &Tensor, rows: &mut [Row<'_, Pos>]) -> Result<Tensor> {
            let ids = tokens.to_vec2::<u32>()?;
            let fault = self.take_rows();
            if fault == Fault::RowsRefuseUntouched {
                return Err(untouched(
                    "injected: the step was refused before any row moved",
                ));
            }
            let mut lanes = Vec::with_capacity(rows.len());
            for (row, ids) in rows.iter_mut().zip(&ids) {
                self.entries.lock().push((row.pos, row.cache.0));
                lanes.push((ids[0], row.pos, row.cache.0));
                row.cache.0 += 1;
            }
            if fault == Fault::RowsMutateThenFail {
                return Err(touched(
                    "injected: the step moved every row's cache, then failed",
                ));
            }
            self.next_rows(&lanes)
        }
        fn forward_ragged(
            &self,
            tokens: &Tensor,
            rows: &mut [RaggedRow<'_, Pos>],
        ) -> Result<Tensor> {
            let ids = tokens.squeeze(0)?.to_vec1::<u32>()?;
            let fault = self.take_rows();
            if fault == Fault::RowsRefuseUntouched {
                return Err(untouched(
                    "injected: the step was refused before any row moved",
                ));
            }
            let mut lanes = Vec::with_capacity(rows.len());
            let mut off = 0;
            for row in rows.iter_mut() {
                self.entries.lock().push((row.pos, row.cache.0));
                off += row.len;
                lanes.push((ids[off - 1], row.pos, row.cache.0));
                row.cache.0 += row.len;
            }
            if fault == Fault::RowsMutateThenFail {
                return Err(touched(
                    "injected: the step moved every row's cache, then failed",
                ));
            }
            self.next_rows(&lanes)
        }
    }

    /// The static body over [`Faulty`]: the batched driver, as
    /// [`a_static_batch_runs_the_body_once_for_every_waiting_row`]'s.
    fn faulty_body<E>() -> StaticBody<Faulty, E> {
        Box::new(|idle, sessions, prompts, specs, on_tokens| {
            let mut sampler =
                lloom::generation::LogitsProcessor::from_sampling(0, Sampling::ArgMax);
            let opts = DecodeOptions {
                lookahead: 1,
                ..DecodeOptions::new(specs.iter().map(|s| s.max_tokens).max().unwrap())
            };
            Session::decode_batched(
                sessions,
                &mut sampler,
                prompts,
                &opts,
                &specs[0].eos_ids,
                idle.device,
                on_tokens,
            )
        })
    }

    /// A one-row static scheduler over `model` with prefix reuse ON: every
    /// request lands on the one sequence the previous request left.
    fn start_faulty<E>(
        model: Arc<Faulty>,
        extra: E,
    ) -> (BatchHandle<Faulty, E>, std::thread::JoinHandle<()>)
    where
        E: MakeDriver<Faulty> + BatchRecovery + Send + 'static,
    {
        let device = Device::Cpu;
        let cfg = BatchConfig {
            device,
            pool: SlotPool::new(1, || Pos(0)),
            rings: RingPool::new(),
            opts: DecodeOptions {
                lookahead: 1,
                ..DecodeOptions::new(0)
            },
            max_rows: 1,
            reuse_prefix: true,
            share_prefix: false,
            prefill_batch: 1,
            prefill_decode: PrefillDecode::Ride,
            intake_per_tick: INTAKE_PER_TICK,
            mode: BatchMode::Static(faulty_body()),
            extra,
            on_counters: None,
        };
        spawn(model, |m: &Faulty| m, cfg).unwrap()
    }

    fn submit_faulty<E>(
        handle: &BatchHandle<Faulty, E>,
        prompt: &[u32],
        max: usize,
    ) -> Receiver<BatchEvent> {
        handle.submit(BatchRequest {
            prompt_ids: prompt.to_vec(),
            params: params(max),
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Wait for the scheduler thread to finish what it is doing: a closed
    /// job runs only between two turns of its loop. A row's error is sent
    /// before the failed batch's recovery runs.
    fn settle<E>(handle: &BatchHandle<Faulty, E>) {
        handle
            .run_closed_blocking(|_| ())
            .expect("the scheduler runs the closed job");
    }

    fn expect_error(rx: &Receiver<BatchEvent>) -> String {
        loop {
            match rx.recv_timeout(Duration::from_secs(30)).expect("an event") {
                BatchEvent::Tokens(_) => {}
                BatchEvent::Done(d) => panic!("the row succeeded: {:?}", d.tokens),
                BatchEvent::Error(e) => return e,
            }
        }
    }

    /// A static batch whose forward moved its cache and then failed hands its
    /// sequence back reset. The client sees the error; the same prompt, sent
    /// again with prefix reuse on and no cold flag, starts from an empty
    /// cache and decodes what a fresh sequence decodes. Returned as it was,
    /// the sequence's log is empty and `prepare` calls it `Fresh` without
    /// resetting, so the retry ran on the failed pass's cache.
    #[test]
    fn a_failed_static_batch_hands_its_rows_back_reset() {
        let model = Arc::new(Faulty::new(&Device::Cpu).unwrap());
        let (handle, thread) = start_faulty(Arc::clone(&model), ());
        let prompt = [3u32, 1, 4];
        model.arm(Fault::MutateThenFail);
        let err = expect_error(&submit_faulty(&handle, &prompt, 8));
        assert!(err.contains("moved the cache, then failed"), "{err}");
        let seen = model.entries().len();
        let (streamed, done) = collect(&submit_faulty(&handle, &prompt, 8));
        assert_eq!(
            model.entries()[seen],
            (0, 0),
            "the retry's first forward as (pos, held): a cache holding positions past pos \
             is the failed batch's"
        );
        let (want, eos) = chain(*prompt.last().unwrap(), 8);
        assert_eq!(done.tokens, want);
        assert_eq!(streamed, want);
        assert_eq!(done.eos, eos);
        handle.shutdown();
        thread.join().unwrap();
    }

    /// As [`a_failed_static_batch_hands_its_rows_back_reset`], where the
    /// failed request EXTENDED a prefix a successful request cached: kept
    /// as it was, the retry extends the same log onto the failed pass's
    /// cache.
    #[test]
    fn a_failed_static_batch_that_extended_a_cached_prefix_is_not_extended_again() {
        let model = Arc::new(Faulty::new(&Device::Cpu).unwrap());
        let (handle, thread) = start_faulty(Arc::clone(&model), ());
        let first = [3u32, 1, 4];
        let (_, done) = collect(&submit_faulty(&handle, &first, 4));
        assert!(!done.eos, "the fixture wants a log to extend");
        let mut next: Vec<u32> = first.to_vec();
        next.extend(&done.tokens);
        next.extend([9, 2]);
        let seen = model.entries().len();
        model.arm(Fault::MutateThenFail);
        let err = expect_error(&submit_faulty(&handle, &next, 8));
        assert!(err.contains("moved the cache, then failed"), "{err}");
        assert!(
            model.entries()[seen].0 > 0,
            "the failed request extended the cached prefix: {:?}",
            &model.entries()[seen..]
        );
        let seen = model.entries().len();
        let (streamed, done) = collect(&submit_faulty(&handle, &next, 8));
        let (pos, held) = model.entries()[seen];
        assert_eq!(
            pos, held,
            "the retry's first forward: a cache holding positions past pos is the failed batch's"
        );
        let (want, eos) = chain(*next.last().unwrap(), 8);
        assert_eq!(done.tokens, want);
        assert_eq!(streamed, want);
        assert_eq!(done.eos, eos);
        handle.shutdown();
        thread.join().unwrap();
    }

    /// The static scheduler's `extra` with a recovery that counts its calls:
    /// a stand-in for the drafter whose contexts a failed batch must drop.
    /// `refusals` fences are refused before one succeeds.
    #[derive(Clone, Default)]
    struct Recorder {
        fences: Arc<std::sync::atomic::AtomicUsize>,
        aborts: Arc<std::sync::atomic::AtomicUsize>,
        refusals: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl Recorder {
        fn calls(&self) -> (usize, usize) {
            (
                self.fences.load(Ordering::Acquire),
                self.aborts.load(Ordering::Acquire),
            )
        }
    }

    impl MakeDriver<Faulty> for Recorder {
        fn make_driver<'m>(
            &mut self,
            _model: &'m Faulty,
            _opts: &DecodeOptions,
            _device: &Device,
        ) -> Result<Option<Driver<'m, Faulty>>> {
            Ok(None)
        }
    }

    impl BatchRecovery for Recorder {
        fn fence(&mut self, device: &Device) -> Result<()> {
            self.fences.fetch_add(1, Ordering::AcqRel);
            if self
                .refusals
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_sub(1))
                .is_ok()
            {
                lloom_tensor::bail!("injected: the fence was refused");
            }
            crate::recovery::fence_failed_execution(device)
        }
        fn abort(&mut self) {
            self.aborts.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// A successful request leaves its sequence for the next to extend, and
    /// no batch that succeeds runs the recovery. After a failure the
    /// recovery runs once — the fence, then the extra's abort — and the next
    /// request decodes from a reset sequence.
    #[test]
    fn a_successful_static_batch_keeps_its_prefix_and_only_a_failed_one_recovers() {
        let model = Arc::new(Faulty::new(&Device::Cpu).unwrap());
        let extra = Recorder::default();
        let (handle, thread) = start_faulty(Arc::clone(&model), extra.clone());
        let first = [3u32, 1, 4];
        let (_, done) = collect(&submit_faulty(&handle, &first, 4));
        assert!(!done.eos, "the fixture wants a log to extend");
        let mut next: Vec<u32> = first.to_vec();
        next.extend(&done.tokens);
        next.extend([9, 2]);
        let (streamed, done) = collect(&submit_faulty(&handle, &next, 8));
        assert_eq!(done.reuse, ReuseAction::Extend, "{done:?}");
        assert!(done.reused >= first.len(), "{done:?}");
        let (want, eos) = chain(*next.last().unwrap(), 8);
        assert_eq!(
            (streamed, done.tokens, done.eos),
            (want.clone(), want.clone(), eos)
        );
        assert_eq!(
            extra.calls(),
            (0, 0),
            "a batch that succeeded ran the recovery"
        );

        model.arm(Fault::MutateThenFail);
        expect_error(&submit_faulty(&handle, &next, 8));
        settle(&handle);
        assert_eq!(
            extra.calls(),
            (1, 1),
            "(fences, aborts) after one failed batch"
        );
        let seen = model.entries().len();
        let (streamed, done) = collect(&submit_faulty(&handle, &next, 8));
        assert_eq!(model.entries()[seen], (0, 0));
        assert_eq!(done.reuse, ReuseAction::Fresh);
        assert_eq!((streamed, done.tokens), (want.clone(), want));
        assert_eq!(
            extra.calls(),
            (1, 1),
            "the retry succeeded and recovered nothing"
        );
        handle.shutdown();
        thread.join().unwrap();
    }

    /// A static batch refused before its pass moved anything, reported as
    /// such on its error ([`lloom_tensor::StateTouched::No`]), runs no
    /// recovery and hands its sequence back as it was: the same request sent
    /// again EXTENDS the cached prefix the refused one was extending, and
    /// decodes what a fresh sequence decodes (decision 105,
    /// `batched-decode-design.md` §11.10).
    #[test]
    fn a_static_batch_refused_before_its_pass_moved_anything_keeps_its_prefix() {
        let model = Arc::new(Faulty::new(&Device::Cpu).unwrap());
        let extra = Recorder::default();
        let (handle, thread) = start_faulty(Arc::clone(&model), extra.clone());
        let first = [3u32, 1, 4];
        let (_, done) = collect(&submit_faulty(&handle, &first, 4));
        assert!(!done.eos, "the fixture wants a log to extend");
        let mut next: Vec<u32> = first.to_vec();
        next.extend(&done.tokens);
        let cached = next.len();
        next.extend([9, 2]);
        let seen = model.entries().len();
        model.arm(Fault::RefuseUntouched);
        let err = expect_error(&submit_faulty(&handle, &next, 8));
        assert!(err.contains("refused before anything moved"), "{err}");
        let (pos, held) = model.entries()[seen];
        assert!(
            pos > 0 && pos == held,
            "the refused pass extended a clean prefix: {pos} {held}"
        );
        settle(&handle);
        assert_eq!(
            extra.calls(),
            (0, 0),
            "an untouched failure ran the recovery"
        );
        let seen = model.entries().len();
        let (streamed, done) = collect(&submit_faulty(&handle, &next, 8));
        assert_eq!(
            (done.reuse, done.reused),
            (ReuseAction::Extend, cached),
            "the retry extends the prefix the refused request found: {done:?}"
        );
        let (pos, held) = model.entries()[seen];
        assert_eq!(
            (pos, held),
            (cached, cached),
            "the retry's first forward onto the kept prefix"
        );
        let (want, eos) = chain(*next.last().unwrap(), 8);
        assert_eq!((streamed, done.tokens, done.eos), (want.clone(), want, eos));
        assert_eq!(extra.calls(), (0, 0));
        handle.shutdown();
        thread.join().unwrap();
    }

    /// A closed job asking for more sequences than the pool holds gets none
    /// and costs the pool none: the ones it took before the pool ran short
    /// go back, reset. Dropped, their slots stayed checked out for the
    /// process's life.
    #[test]
    fn a_closed_job_on_a_short_pool_costs_the_pool_no_slot() {
        let model = Arc::new(Faulty::new(&Device::Cpu).unwrap());
        let (handle, thread) = start_faulty_continuous(Arc::clone(&model), 2, ());
        let (short, idle, len) = handle
            .run_closed_blocking(|idle| {
                let short = idle.take_reset(3).is_none();
                (short, idle.pool.idle(), idle.pool.len())
            })
            .expect("the scheduler runs the closed job");
        assert!(short, "three sequences from a pool of two");
        assert_eq!(
            (idle, len),
            (2, 2),
            "(idle slots, slots) after the short take"
        );
        let (taken, idle) = handle
            .run_closed_blocking(|idle| {
                let taken = idle.take_reset(2).map(|(sessions, slots)| {
                    let n = sessions.len();
                    idle.give_back_reset(sessions, slots);
                    n
                });
                (taken, idle.pool.idle())
            })
            .expect("the scheduler runs the closed job");
        assert_eq!(
            (taken, idle),
            (Some(2), 2),
            "the whole pool, taken and given back"
        );
        let (streamed, done) = collect(&submit_faulty(&handle, &[3u32, 1, 4], 8));
        let (want, eos) = chain(4, 8);
        assert_eq!((streamed, done.tokens, done.eos), (want.clone(), want, eos));
        handle.shutdown();
        thread.join().unwrap();
    }

    /// A refused fence resets nothing and keeps the failed batch's sequence
    /// out of the pool; the next batch is refused with the fence's error
    /// rather than run beside work the fence could not see retire. Once the
    /// fence succeeds the sequence comes back reset, and the request after
    /// decodes what a fresh sequence decodes.
    #[test]
    fn a_refused_fence_holds_the_failed_rows_until_it_succeeds() {
        let model = Arc::new(Faulty::new(&Device::Cpu).unwrap());
        let extra = Recorder::default();
        extra.refusals.store(2, Ordering::Release);
        let (handle, thread) = start_faulty(Arc::clone(&model), extra.clone());
        let prompt = [3u32, 1, 4];
        model.arm(Fault::MutateThenFail);
        expect_error(&submit_faulty(&handle, &prompt, 8));
        settle(&handle);
        assert_eq!(extra.calls(), (1, 0), "a refused fence aborted the extra");
        let seen = model.entries().len();
        let err = expect_error(&submit_faulty(&handle, &prompt, 8));
        assert!(
            err.contains("recovery is pending") && err.contains("fence was refused"),
            "{err}"
        );
        assert_eq!(
            model.entries().len(),
            seen,
            "a batch ran while the recovery was pending"
        );
        assert_eq!(extra.calls(), (2, 0));
        let (streamed, done) = collect(&submit_faulty(&handle, &prompt, 8));
        assert_eq!(model.entries()[seen], (0, 0));
        let (want, eos) = chain(*prompt.last().unwrap(), 8);
        assert_eq!((streamed, done.tokens, done.eos), (want.clone(), want, eos));
        assert_eq!(extra.calls(), (3, 1));
        handle.shutdown();
        thread.join().unwrap();
    }

    /// A continuous scheduler of `rows` sequences over `model`, prefix reuse
    /// on: a request lands on the sequence that holds its longest prefix.
    fn start_faulty_continuous<E>(
        model: Arc<Faulty>,
        rows: usize,
        extra: E,
    ) -> (BatchHandle<Faulty, E>, std::thread::JoinHandle<()>)
    where
        E: MakeDriver<Faulty> + BatchRecovery + Send + 'static,
    {
        let cfg = BatchConfig {
            device: Device::Cpu,
            pool: SlotPool::new(rows, || Pos(0)),
            rings: RingPool::new(),
            opts: DecodeOptions {
                lookahead: 1,
                ..DecodeOptions::new(0)
            },
            max_rows: rows,
            reuse_prefix: true,
            share_prefix: false,
            prefill_batch: 1,
            prefill_decode: PrefillDecode::Ride,
            intake_per_tick: INTAKE_PER_TICK,
            mode: BatchMode::Continuous,
            extra,
            on_counters: None,
        };
        spawn(model, |m: &Faulty| m, cfg).unwrap()
    }

    /// Submit `prompts` together: held behind a closed job until all are
    /// queued, so one admission takes every one of them into the same steps.
    fn submit_together<E>(
        handle: &BatchHandle<Faulty, E>,
        prompts: &[Vec<u32>],
        max: usize,
    ) -> Vec<Receiver<BatchEvent>> {
        let gate = hold(handle);
        let rxs = prompts
            .iter()
            .map(|p| submit_faulty(handle, p, max))
            .collect();
        drop(gate);
        rxs
    }

    /// Round 1 of the continuous tests: `firsts` decoded together, each
    /// prompt extended by its reply and two more tokens, so the next round
    /// extends a cached prefix on every row.
    fn extend_after_one_round<E>(
        handle: &BatchHandle<Faulty, E>,
        firsts: &[Vec<u32>],
    ) -> Vec<Vec<u32>> {
        submit_together(handle, firsts, 4)
            .iter()
            .enumerate()
            .map(|(i, rx)| {
                let (_, done) = collect(rx);
                assert!(!done.eos, "the fixture wants a log to extend");
                let mut next = firsts[i].clone();
                next.extend(&done.tokens);
                next.extend([9, 2 + i as u32]);
                next
            })
            .collect()
    }

    /// A continuous step that moved every row's cache and then failed is
    /// recovered as a failed static batch is: the fence, the extra's abort,
    /// and every row handed back reset, so the next checkout of any of them
    /// is a `Fresh` prepare onto an empty cache (`batched-decode-design.md`
    /// §11.11). Both rows extended a cached prefix and the failure lands in
    /// the decode step after both prefilled: handed back as they were, the
    /// same prompts sent again would reuse logs whose caches hold one
    /// position more than they say. A round that succeeds runs no recovery.
    #[test]
    fn a_failed_continuous_step_hands_every_row_back_reset() {
        let model = Arc::new(Faulty::new(&Device::Cpu).unwrap());
        let extra = Recorder::default();
        let (handle, thread) = start_faulty_continuous(Arc::clone(&model), 2, extra.clone());
        let nexts = extend_after_one_round(&handle, &[vec![3u32, 1, 4], vec![5u32, 9]]);
        assert_eq!(
            extra.calls(),
            (0, 0),
            "a step that succeeded ran the recovery"
        );

        let seen = model.entries().len();
        model.arm_rows(Fault::RowsMutateThenFail, 0);
        for (i, rx) in submit_together(&handle, &nexts, 8).iter().enumerate() {
            let err = expect_error(rx);
            assert!(
                err.contains("moved every row's cache, then failed"),
                "row {i}: {err}"
            );
        }
        let failed = model.entries()[seen..].to_vec();
        assert!(
            failed.len() == 4 && failed.iter().all(|(pos, held)| *pos > 0 && pos == held),
            "two prefills extending a cached prefix, then one step over both rows: {failed:?}"
        );
        settle(&handle);
        assert_eq!(
            extra.calls(),
            (1, 1),
            "(fences, aborts) after one failed step"
        );

        let seen = model.entries().len();
        for (i, rx) in submit_together(&handle, &nexts, 8).iter().enumerate() {
            let (streamed, done) = collect(rx);
            assert_eq!(done.reuse, ReuseAction::Fresh, "row {i}: {done:?}");
            let (want, eos) = chain(*nexts[i].last().unwrap(), 8);
            assert_eq!(
                (streamed, done.tokens, done.eos),
                (want.clone(), want, eos),
                "row {i} after the failed step"
            );
        }
        assert_eq!(
            model.entries()[seen..seen + 2],
            [(0, 0), (0, 0)],
            "the retries' first forwards as (pos, held): a cache holding positions past pos \
             is the failed step's"
        );
        assert_eq!(
            extra.calls(),
            (1, 1),
            "the retries succeeded and recovered nothing"
        );
        handle.shutdown();
        thread.join().unwrap();
    }

    /// A refused fence after a failed continuous step resets nothing and
    /// keeps the step's sequences out of the pool; the requests that arrive
    /// meanwhile are refused with the fence's error, and no forward runs for
    /// them. Once the fence succeeds the sequence comes back reset and the
    /// request after decodes what a fresh sequence decodes.
    #[test]
    fn a_refused_fence_holds_a_failed_steps_rows_until_it_succeeds() {
        let model = Arc::new(Faulty::new(&Device::Cpu).unwrap());
        let extra = Recorder::default();
        extra.refusals.store(2, Ordering::Release);
        let (handle, thread) = start_faulty_continuous(Arc::clone(&model), 1, extra.clone());
        let prompt = vec![3u32, 1, 4];
        model.arm_rows(Fault::RowsMutateThenFail, 0);
        expect_error(&submit_faulty(&handle, &prompt, 8));
        settle(&handle);
        assert_eq!(extra.calls(), (1, 0), "a refused fence aborted the extra");
        let seen = model.entries().len();
        let err = expect_error(&submit_faulty(&handle, &prompt, 8));
        assert!(
            err.contains("recovery is pending") && err.contains("fence was refused"),
            "{err}"
        );
        assert_eq!(
            model.entries().len(),
            seen,
            "a request ran while the recovery was pending"
        );
        assert_eq!(extra.calls(), (2, 0));
        let (streamed, done) = collect(&submit_faulty(&handle, &prompt, 8));
        assert_eq!(model.entries()[seen], (0, 0));
        let (want, eos) = chain(*prompt.last().unwrap(), 8);
        assert_eq!((streamed, done.tokens, done.eos), (want.clone(), want, eos));
        assert_eq!(extra.calls(), (3, 1));
        handle.shutdown();
        thread.join().unwrap();
    }

    /// A continuous step refused before any row's state moved, reported as
    /// such on its error ([`lloom_tensor::StateTouched::No`], read by
    /// [`BatchRecovery::touched`]), runs no recovery: every row goes
    /// back to its own slot with its cached prefix. The refusal lands after
    /// each row decoded one token, so the drained rows' caches hold a
    /// position past the prompt that only the drain's settle logs. The same
    /// prompts sent again rewind both positions the log names past the
    /// prompt's last token and re-feed that token where it was: a log
    /// missing the decoded token would put it one position too far.
    #[test]
    fn an_untouched_continuous_failure_keeps_every_rows_prefix() {
        let model = Arc::new(Faulty::new(&Device::Cpu).unwrap());
        let extra = Recorder::default();
        let (handle, thread) = start_faulty_continuous(Arc::clone(&model), 2, extra.clone());
        let nexts = extend_after_one_round(&handle, &[vec![3u32, 1, 4], vec![5u32, 9]]);

        model.arm_rows(Fault::RowsRefuseUntouched, 1);
        let mut decoded = Vec::new();
        for (i, rx) in submit_together(&handle, &nexts, 8).iter().enumerate() {
            let mut streamed = Vec::new();
            let err = loop {
                match rx.recv_timeout(Duration::from_secs(30)).expect("an event") {
                    BatchEvent::Tokens(ids) => streamed.extend(ids),
                    BatchEvent::Done(d) => panic!("row {i} succeeded: {:?}", d.tokens),
                    BatchEvent::Error(e) => break e,
                }
            };
            assert!(
                err.contains("refused before any row moved"),
                "row {i}: {err}"
            );
            assert_eq!(
                streamed.len(),
                1,
                "row {i} decoded one token before the refusal"
            );
            decoded.push(streamed);
        }
        settle(&handle);
        assert_eq!(
            extra.calls(),
            (0, 0),
            "an untouched failure ran the recovery"
        );

        let seen = model.entries().len();
        for (i, rx) in submit_together(&handle, &nexts, 8).iter().enumerate() {
            let (streamed, done) = collect(rx);
            assert_eq!(
                (done.reuse, done.reused),
                (ReuseAction::Rewound(2), nexts[i].len() - 1),
                "row {i}: the prompt and its decoded token {:?} were kept",
                decoded[i]
            );
            let (want, eos) = chain(*nexts[i].last().unwrap(), 8);
            assert_eq!(
                (streamed, done.tokens, done.eos),
                (want.clone(), want, eos),
                "row {i} after the refused step"
            );
        }
        let last = |i: usize| nexts[i].len() - 1;
        assert_eq!(
            model.entries()[seen..seen + 2],
            [(last(0), last(0)), (last(1), last(1))],
            "the retries re-feed each prompt's last token where it was"
        );
        assert_eq!(extra.calls(), (0, 0));
        handle.shutdown();
        thread.join().unwrap();
    }
}
