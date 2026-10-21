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
    DecodeStats, PrefillDecode, ReuseAction, RowDriver, RowId, Session,
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
    // Admitted in order; a row the driver refuses is dropped by `admit`
    // (the driver's contract), the rows after it are handed back untouched.
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
            Err(e) => {
                let mut back = driver.drain_rows();
                rest.reverse();
                back.append(&mut rest);
                return (back, Err(e));
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
    decoder: Driver<'m, M>,
    inflight: HashMap<RowId, InFlight>,
    waiting: VecDeque<(BatchRequest, Sender<BatchEvent>)>,
    closed: VecDeque<ClosedJob<M, E>>,
    rx: Receiver<Job<M, E>>,
    counters: BatchCounters,
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
    steps: Vec<(u64, f64, f64, lloom::nn::StepReport)>,
}

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
    pub mode: BatchMode<M, E>,
    pub extra: E,
    /// Called after every step with the counters (a log line).
    pub on_counters: Option<CountersHook>,
}

impl<'m, M, E> BatchScheduler<'m, M, E>
where
    M: BatchedCausalLM + 'm,
    M::Cache: CacheReset + CacheRewind + CacheShare,
    E: MakeDriver<M>,
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
            decoder,
            inflight: HashMap::new(),
            waiting: VecDeque::new(),
            closed: VecDeque::new(),
            rx,
            counters: BatchCounters::default(),
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

    /// Take the queue: block when there is nothing to do, drain otherwise.
    /// `false` on shutdown.
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
        loop {
            match self.rx.try_recv() {
                Ok(job) => {
                    if !self.enqueue(job) {
                        return false;
                    }
                }
                Err(TryRecvError::Empty) => return true,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
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
    fn admit_waiting(&mut self) {
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
                Err(e) => {
                    let _ = tx.send(BatchEvent::Error(e.to_string()));
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
            lloom_tensor::kernels::metal::profile::set_step(self.counters.steps);
            lloom_tensor::kernels::metal::profile::now_s()
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
        if let (Some(w), Some(t0)) = (self.window.as_mut(), t0) {
            let t1 = lloom_tensor::kernels::metal::profile::now_s();
            w.steps.push((self.counters.steps, t0, t1, report));
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
                for (session, m) in sessions.into_iter().zip(meta) {
                    let (slot, _, tx, _, _) = m.expect("a kept row");
                    self.pool.release(slot, session.into_sequence());
                    let _ = tx.send(BatchEvent::Error(e.to_string()));
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
    /// (`<dump>.batch<n>.csv`) and the steps (`<dump>.batch<n>.steps.csv`).
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
            w.steps.iter().filter(|x| x.3.pieces > 0).count(),
            w.steps.iter().filter(|x| x.3.held > 0).count(),
            s.gpu_busy_union_s,
            wall,
            s.gpu_busy_union_s / wall.max(1e-9),
            s.timed_command_buffers,
            idle,
            starved,
        );
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
            "step,host_start_s,host_end_s,live,held,pieces,piece_tokens,retired,pending\n",
        );
        for (step, t0, t1, r) in &w.steps {
            st.push_str(&format!(
                "{step},{t0:.6},{t1:.6},{},{},{},{},{},{}\n",
                r.live, r.held, r.pieces, r.piece_tokens, r.retired, r.pending
            ));
        }
        for (p, body) in [
            (format!("{path}.batch{n}.csv"), cb),
            (format!("{path}.batch{n}.steps.csv"), st),
        ] {
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
                            // A failed step ends every row in it with the error;
                            // the decoder's state is not trusted past it, so
                            // every row is drained and its sequence returned to
                            // the pool reset.
                            let msg = e.to_string();
                            let slots: Vec<usize> =
                                self.inflight.values().map(|f| f.slot).collect();
                            for (_, f) in self.inflight.drain() {
                                let _ = f.tx.send(BatchEvent::Error(msg.clone()));
                            }
                            let sessions = self.decoder.drain_rows();
                            for (slot, mut s) in slots.into_iter().zip(sessions) {
                                s.reset();
                                self.pool.release(slot, s.into_sequence());
                            }
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
    E: MakeDriver<M> + Send + 'static,
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
    fn hold(handle: &BatchHandle<Chain, ()>) -> Sender<()> {
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
}
