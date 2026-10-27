//! Private optimized libtest experiment; not shipping-binary e2e acceptance.
//! Only an explicitly admitted exact ignored test may load a real checkpoint.
use super::*;
use crate::app::common::{MemGauge, MemReport};
use crate::app::{load_engine, resolve_options, Shape};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

#[derive(Clone, Deserialize)]
struct Settings {
    serve_argv: Vec<String>,
    release: String,
    hold_ms: u64,
    writer_timeout_ms: u64,
    cohorts: usize,
    warmup_requests: usize,
    trace_file: PathBuf,
    ready_file: PathBuf,
}

struct State {
    settings: Settings,
    release_at: Release,
    device: lloom_tensor::Device,
    trace: ParkingMutex<File>,
    error: ParkingMutex<Option<String>>,
    next: std::sync::atomic::AtomicUsize,
    returned: std::sync::atomic::AtomicUsize,
    writers: std::sync::atomic::AtomicUsize,
    peak_writers: std::sync::atomic::AtomicUsize,
    memory: ParkingMutex<Option<MemGauge>>,
    memory_started: AtomicBool,
    memory_closed: AtomicBool,
    memory_start_ns: std::sync::atomic::AtomicU64,
    samples: std::sync::atomic::AtomicU64,
}
impl State {
    fn fail(&self, message: impl Into<String>) {
        let mut error = self.error.lock();
        if error.is_none() {
            *error = Some(message.into());
        }
    }
    fn event(&self, value: Value) {
        let result = (|| -> Result<()> {
            let mut file = self.trace.lock();
            serde_json::to_writer(&mut *file, &value)?;
            file.write_all(b"\n")?;
            file.flush()?;
            Ok(())
        })();
        if let Err(error) = result {
            self.fail(format!("benchmark trace: {error:#}"));
        }
    }
    fn start_memory(&self) {
        let mut memory = self.memory.lock();
        if !self.memory_closed.load(Ordering::Acquire)
            && !self.memory_started.swap(true, Ordering::AcqRel)
        {
            self.memory_start_ns.store(uptime_ns(), Ordering::Release);
            *memory = Some(MemGauge::start(&self.device, true));
        }
    }
    fn finish_memory(&self, reason: &str) {
        self.finish_memory_with(reason, |gauge| gauge.finish(&self.device));
    }
    fn finish_memory_with(&self, reason: &str, finish: impl FnOnce(MemGauge) -> MemReport) {
        // Keep the same lock through join AND report publication. Taking the
        // gauge is not completion: the server-return caller can race a last
        // handler already joining it. No Turn is held by this post-return path.
        let mut memory = self.memory.lock();
        self.memory_closed.store(true, Ordering::Release);
        if let Some(gauge) = memory.take() {
            let report = finish(gauge);
            self.samples.store(report.samples, Ordering::Release);
            self.event(json!({"kind":"memory-finish", "at_ns":uptime_ns(), "reason":reason,
                "started_ns":self.memory_start_ns.load(Ordering::Acquire), "sample_period_ms":10,
                "report":report.json(), "scope":"sampled process/device, not exact retained payload bytes"}));
        }
    }
    fn handle(
        self: &Arc<Self>,
        shared: &Arc<Shared>,
        path: &str,
        request: Request,
        first: FirstToken,
    ) -> Result<()> {
        if path != CHAT {
            return handle_model_request(shared, path, request, first);
        }
        let ordinal = self.next.fetch_add(1, Ordering::AcqRel);
        let priced = ordinal >= self.settings.warmup_requests;
        let index = ordinal.saturating_sub(self.settings.warmup_requests);
        let is_a = priced && index.is_multiple_of(2);
        if priced {
            self.start_memory();
        }
        let start_ns = uptime_ns();
        let recv_ns = first.recv;
        let role = if !priced {
            "warm"
        } else if is_a {
            "A"
        } else {
            "B"
        };
        let mut publication = None;
        let (_observation, queued, timing) = shared.turns.observe_next_take_timed();
        let ticket = Arc::new(ParkingMutex::new(None));
        let (entered, entry) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let mut pace = is_a.then(|| {
            Pace::start(
                self.clone(),
                shared.clone(),
                ordinal,
                ticket.clone(),
                entry,
                release,
            )
        });
        let state = self.clone();
        let returned = handle_model_request_with_stream(
            shared,
            path,
            request,
            first,
            |request| {
                let q = queued
                    .try_recv()
                    .map_err(|e| anyhow::anyhow!("real ticket witness: {e}"))?;
                *ticket.lock() = Some(q);
                let writer = ChunkedWriter::start(request, "text/event-stream")?;
                let inner: Box<dyn Write + Send> =
                    Box::new(CountedWriter::new(writer.w, state.clone()));
                if is_a {
                    Ok(ChunkedWriter {
                        w: Box::new(RoleThenGate {
                            inner,
                            pending: Vec::new(),
                            attempted: None,
                            role_passed: false,
                            entered: Some(entered),
                            released,
                            expected_first_reasoning: None,
                            reject_after_gate: false,
                            max_pause: Duration::from_millis(state.settings.writer_timeout_ms),
                        }),
                    })
                } else {
                    Ok(ChunkedWriter { w: inner })
                }
            },
            |turn| {
                // Fixed-size fields only while either policy still owns Turn.
                publication = Some((uptime_ns(), shared.drains.active.load(Ordering::Acquire)));
                if priced && matches!(state.release_at, Release::Publication) {
                    drop(turn.take());
                }
            },
        );
        let full_return_ns = uptime_ns();
        let writers_at_return = self.writers.load(Ordering::Acquire);
        let contexts_at_return = shared.drains.active.load(Ordering::Acquire);
        let pace_receipt = if let Some(pace) = &mut pace {
            match pace.finish(returned.is_ok()) {
                Ok(receipt) => receipt,
                Err(error) => {
                    self.fail(format!("pace cleanup: {error:#}"));
                    None
                }
            }
        } else {
            None
        };
        // All serialization/file I/O is after handle_model_request_with_stream
        // returned and its Turn was dropped for BOTH policies.
        self.event(
            json!({"kind":"request-start", "ordinal":ordinal, "priced":priced,
            "role":role, "at_ns":start_ns, "recv_ns":recv_ns}),
        );
        if let Some((at_ns, contexts)) = publication {
            self.event(
                json!({"kind":"settlement-callback", "ordinal":ordinal, "at_ns":at_ns,
                "response_contexts":contexts,
                "scope":"publication clock only when healthy SSE and bound continuation independently validate success; callback also releases decode-error path"}),
            );
        }
        if let Ok(t) = timing.try_recv() {
            self.event(json!({"kind":"queue", "ordinal":ordinal, "ticket":t.ticket,
                "serving_at_assignment":t.serving, "assigned_ns":t.assigned_ns,
                "acquired_ns":t.acquired_ns, "queue_wait_ns":t.acquired_ns.saturating_sub(t.assigned_ns)}));
        } else {
            self.fail("missing real take timing (B=1 required)");
        }
        if let Some(pace) = pace_receipt {
            self.event(json!({"kind":"writer-release", "ordinal":ordinal,
                "entered_ns":pace.entered_ns, "target_ns":pace.target_ns,
                "send_begin_ns":pace.send_begin_ns,"send_end_ns":pace.send_end_ns,
                "release_sent":pace.release_sent,"canceled":pace.canceled,
                "serving":pace.serving, "A_ticket_held":pace.held}));
            if returned.is_ok()
                && (pace.canceled
                    || !pace.release_sent
                    || (matches!(self.release_at, Release::FullSettlement) && !pace.held))
            {
                self.fail("FullSettlement released A ticket before exogenous writer release");
            }
        } else if is_a && returned.is_ok() {
            self.fail("A had no visible writer gate");
        }
        self.event(
            json!({"kind":"full-return", "ordinal":ordinal, "at_ns":full_return_ns,
            "ok":returned.is_ok(), "error":returned.as_ref().err().map(|e|format!("{e:#}")),
            "open_writers":writers_at_return,
            "response_contexts":contexts_at_return}),
        );
        if returned.is_err() {
            self.fail(format!("request {ordinal} failed"));
        }
        if priced && self.returned.fetch_add(1, Ordering::AcqRel) + 1 == self.settings.cohorts * 2 {
            self.finish_memory("all priced handlers returned");
        }
        returned
    }
}

// Counts owned socket-writer objects, NOT complete reply storage or bytes.
struct CountedWriter {
    inner: Option<Box<dyn Write + Send>>,
    state: Arc<State>,
}
impl CountedWriter {
    fn new(inner: Box<dyn Write + Send>, state: Arc<State>) -> Self {
        let live = state.writers.fetch_add(1, Ordering::AcqRel) + 1;
        state.peak_writers.fetch_max(live, Ordering::AcqRel);
        Self {
            inner: Some(inner),
            state,
        }
    }
}
impl Write for CountedWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.inner.as_mut().unwrap().write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.as_mut().unwrap().flush()
    }
}
impl Drop for CountedWriter {
    fn drop(&mut self) {
        drop(self.inner.take());
        self.state.writers.fetch_sub(1, Ordering::AcqRel);
    }
}

// Time is intentional exogenous workload, never a non-progress safety oracle.
// Both policies use the same entry+D target, independent of B progress.
#[derive(Clone, Copy, Debug)]
struct PaceReceipt {
    entered_ns: u64,
    target_ns: u64,
    send_begin_ns: u64,
    send_end_ns: u64,
    release_sent: bool,
    serving: u64,
    held: bool,
    canceled: bool,
}
struct Pace {
    cancel: Arc<AtomicBool>,
    release: Option<mpsc::Sender<()>>,
    thread: Option<std::thread::JoinHandle<std::result::Result<Option<PaceReceipt>, String>>>,
}
impl Pace {
    fn start(
        state: Arc<State>,
        shared: Arc<Shared>,
        _ordinal: usize,
        ticket: Arc<ParkingMutex<Option<(u64, u64)>>>,
        entered: mpsc::Receiver<u64>,
        release: mpsc::Sender<()>,
    ) -> Self {
        let cancel = Arc::new(AtomicBool::new(false));
        let flag = cancel.clone();
        let writer = release.clone();
        let hold_ms = state.settings.hold_ms;
        let thread = std::thread::spawn(move || {
            let entered_ns = loop {
                if flag.load(Ordering::Acquire) {
                    return Ok(None);
                }
                match entered.recv_timeout(Duration::from_millis(10)) {
                    Ok(at) => break at,
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        return Err("A had no visible writer gate".into())
                    }
                }
            };
            let Some(target_ns) = hold_ms
                .checked_mul(1_000_000)
                .and_then(|d| entered_ns.checked_add(d))
            else {
                let _ = writer.send(());
                return Err("writer target overflow".into());
            };
            while !flag.load(Ordering::Acquire) {
                let remaining = target_ns.saturating_sub(uptime_ns());
                if remaining == 0 {
                    break;
                }
                std::thread::sleep(Duration::from_nanos(remaining.min(10_000_000)));
            }
            let serving = shared.turns.serving_for_test();
            let own = *ticket.lock();
            let held = own.is_some_and(|(t, _)| t == serving);
            let canceled = flag.load(Ordering::Acquire);
            let send_begin_ns = uptime_ns();
            let release_sent = writer.send(()).is_ok();
            let receipt = PaceReceipt {
                entered_ns,
                target_ns,
                send_begin_ns,
                send_end_ns: uptime_ns(),
                release_sent,
                serving,
                held,
                canceled,
            };
            // Bounded primitive receipt only; caller serializes AFTER full return.
            Ok(Some(receipt))
        });
        Self {
            cancel,
            release: Some(release),
            thread: Some(thread),
        }
    }
    fn abort(&mut self) {
        self.cancel.store(true, Ordering::Release);
        if let Some(release) = self.release.take() {
            let _ = release.send(());
        }
    }
    fn finish(&mut self, outcome_ok: bool) -> Result<Option<PaceReceipt>> {
        if !outcome_ok {
            self.abort();
        }
        let receipt = match self.thread.take() {
            Some(thread) => thread
                .join()
                .map_err(|_| anyhow::anyhow!("pacing thread panicked"))?
                .map_err(anyhow::Error::msg)?,
            None => None,
        };
        self.release.take();
        Ok(receipt)
    }
}
impl Drop for Pace {
    fn drop(&mut self) {
        self.abort();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

// Host-only channels/threads: no State, model, Device or native API constructed.
#[test]
fn failed_settlement_releases_and_cancels_pace_before_join() {
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let (release, released) = mpsc::channel();
    let (armed, ready) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        armed.send(()).unwrap();
        released
            .recv_timeout(WAIT)
            .expect("error cleanup must release before join");
        assert!(
            flag.load(Ordering::Acquire),
            "error cleanup cancels before waking waiter"
        );
        Ok(None)
    });
    let mut pace = Pace {
        cancel: cancel.clone(),
        release: Some(release),
        thread: Some(thread),
    };
    ready.recv_timeout(WAIT).unwrap();
    assert!(pace
        .finish(false)
        .expect("failed settlement owns bounded pace cleanup")
        .is_none());
    assert!(cancel.load(Ordering::Acquire));
    assert!(pace.thread.is_none());
    assert!(pace.release.is_none());
}
#[test]
fn successful_settlement_preserves_normal_pace_completion() {
    let cancel = Arc::new(AtomicBool::new(false));
    let (release, released) = mpsc::channel();
    let thread = std::thread::spawn(|| Ok(None));
    let mut pace = Pace {
        cancel: cancel.clone(),
        release: Some(release),
        thread: Some(thread),
    };
    assert!(pace.finish(true).unwrap().is_none());
    assert!(
        !cancel.load(Ordering::Acquire),
        "normal timed completion was not canceled"
    );
    assert!(
        matches!(released.try_recv(), Err(mpsc::TryRecvError::Disconnected)),
        "normal finish did not synthesize release"
    );
}
// Host-only fixture; real CPU gauge/report. No Metal model/device creation.
struct MemoryTraceFile(PathBuf);
impl Drop for MemoryTraceFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
struct ReleaseOnDrop(Option<mpsc::Sender<()>>);
impl Drop for ReleaseOnDrop {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}
fn cpu_memory_fixture() -> (MemoryTraceFile, Arc<State>) {
    let path = std::env::temp_dir().join(format!(
        "946-memory-finish-{}-{}.jsonl",
        std::process::id(),
        uptime_ns()
    ));
    let trace = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let trace_file = MemoryTraceFile(path.clone());
    let state = Arc::new(State {
        settings: Settings {
            serve_argv: Vec::new(),
            release: "FullSettlement".into(),
            hold_ms: 1,
            writer_timeout_ms: 10,
            cohorts: 1,
            warmup_requests: 1,
            trace_file: path.clone(),
            ready_file: path.with_extension("unused-ready"),
        },
        release_at: Release::FullSettlement,
        device: lloom_tensor::Device::Cpu,
        trace: ParkingMutex::new(trace),
        error: ParkingMutex::new(None),
        next: Default::default(),
        returned: Default::default(),
        writers: Default::default(),
        peak_writers: Default::default(),
        memory: ParkingMutex::new(None),
        memory_started: AtomicBool::new(false),
        memory_closed: AtomicBool::new(false),
        memory_start_ns: Default::default(),
        samples: Default::default(),
    });
    (trace_file, state)
}

// Actual shared finish helper; real CPU samples=0/sampled=false. This tests
// publication/lock custody, not a fabricated nonzero native memory report.
#[test]
fn memory_finish_publishes_before_a_contender_can_observe_terminal_state() {
    let (trace_file, state) = cpu_memory_fixture();
    let path = trace_file.0.clone();
    state.start_memory();
    std::thread::scope(|scope| {
        let (entered, entry) = mpsc::channel();
        let (release, released) = mpsc::channel();
        // On every panic path this release drops before scope joins workers.
        let release_on_drop = ReleaseOnDrop(Some(release));
        let worker_state = state.clone();
        let worker = scope.spawn(move || {
            worker_state.finish_memory_with("controlled finish", |gauge| {
                entered.send(()).unwrap();
                released
                    .recv_timeout(WAIT)
                    .expect("controlled finish released during cleanup");
                gauge.finish(&lloom_tensor::Device::Cpu)
            });
        });
        entry
            .recv_timeout(WAIT)
            .expect("finisher owns the actual gauge");
        // This is an actual contender on the SAME memory mutex, not a sleep
        // or a signal before an unobserved lock attempt.
        let contender = state.memory.try_lock();
        let exposed_terminal = contender.is_some();
        drop(contender);
        assert!(
            !exposed_terminal,
            "in-flight memory finish must not expose empty/zero terminal state"
        );
        drop(release_on_drop);
        worker.join().expect("memory finisher joined");
    });
    state.finish_memory_with("second finisher", |_| panic!("gauge finished twice"));
    assert!(state.memory.lock().is_none());
    assert!(state.memory_closed.load(Ordering::Acquire));
    assert_eq!(
        state.samples.load(Ordering::Acquire),
        0,
        "real CPU report, not a fabricated native sample"
    );
    let events: Vec<Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        events.len(),
        1,
        "exact one report published before terminal state"
    );
    assert_eq!(events[0]["kind"], "memory-finish");
    assert_eq!(events[0]["reason"], "controlled finish");
    assert_eq!(events[0]["report"]["samples"], 0);
    assert_eq!(events[0]["report"]["sampled"], false);
    drop(state);
    drop(trace_file);
}

// Last tail is paused before returned increments. A deadline is only a
// failure; success below requires actual sender disconnection after join.
#[test]
fn private_dispatch_tail_finishes_before_memory_finalization() {
    let (trace_file, state) = cpu_memory_fixture();
    state.start_memory();
    let (tail_sender, tails) = mpsc::channel::<std::convert::Infallible>();
    std::thread::scope(|scope| {
        let (entered, entry) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let release_on_drop = ReleaseOnDrop(Some(release));
        let released = Arc::new(ParkingMutex::new(released));
        let worker_state = state.clone();
        // Same cloned-dispatch captured-Sender lifetime as the native path.
        let dispatch = move || {
            let _keep_dispatch_alive = &tail_sender;
            entered.send(()).unwrap();
            released
                .lock()
                .recv_timeout(WAIT)
                .expect("paused private tail released during cleanup");
            worker_state.returned.fetch_add(1, Ordering::AcqRel);
        };
        let pending = dispatch.clone();
        drop(dispatch); // No root sender copy survives.
        let worker = scope.spawn(pending);
        entry
            .recv_timeout(WAIT)
            .expect("private tail paused before returned");
        let before = after_private_dispatch_tails(&tails, Duration::ZERO, || {
            assert_eq!(
                state.returned.load(Ordering::Acquire),
                1,
                "private tail finalizer ran before the last returned increment"
            );
            state.finish_memory("after private tails");
        });
        assert!(
            before.is_err(),
            "live dispatch is not completion at an expired deadline"
        );
        assert_eq!(state.returned.load(Ordering::Acquire), 0);
        assert!(!state.memory_closed.load(Ordering::Acquire));
        drop(release_on_drop);
        worker.join().expect("private tail joined");
    });
    after_private_dispatch_tails(&tails, WAIT, || {
        assert_eq!(state.returned.load(Ordering::Acquire), 1);
        state.finish_memory("after private tails");
    })
    .expect("only disconnected dispatch authority permits finalization");
    let events: Vec<Value> = std::fs::read_to_string(&trace_file.0)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0]["reason"], "after private tails");
    let error = owner_after_private_tails::<()>(
        Err(anyhow::anyhow!("original owner failure")),
        Err(anyhow::anyhow!("secondary private-tail deadline")),
    )
    .unwrap_err();
    assert_eq!(error.to_string(), "original owner failure");
    drop(state);
    drop(trace_file);
}

// No completion messages: each cloned private dispatch owns a Sender, even
// before entering its body. Disconnection alone proves all clones are gone.
fn after_private_dispatch_tails<T>(
    tails: &mpsc::Receiver<std::convert::Infallible>,
    timeout: Duration,
    finish: impl FnOnce() -> T,
) -> Result<T> {
    match tails.recv_timeout(timeout) {
        Err(mpsc::RecvTimeoutError::Disconnected) => Ok(finish()),
        Err(mpsc::RecvTimeoutError::Timeout) => {
            anyhow::bail!("private dispatch tails did not finish within {timeout:?}")
        }
        Ok(never) => match never {},
    }
}

fn owner_after_private_tails<T>(owner: Result<()>, tails: Result<T>) -> Result<T> {
    match owner {
        Err(primary) => {
            if let Err(secondary) = tails {
                eprintln!("[private dispatch-tail cleanup also failed: {secondary:#}]");
            }
            Err(primary)
        }
        Ok(()) => tails,
    }
}

struct FinishMemory(Arc<State>);
impl Drop for FinishMemory {
    fn drop(&mut self) {
        self.0.finish_memory("entrypoint cleanup/unwind");
    }
}

#[test]
#[ignore = "real checkpoint + Metal: separately admitted instrumented scheduling experiment only"]
fn real_model_http_overlap_private_policy() -> Result<()> {
    let config = PathBuf::from(
        std::env::var("LLOOM_OVERLAP_CONFIG").context("explicit private config required")?,
    );
    let bytes = std::fs::read(&config)?;
    let sha = format!("{:x}", Sha256::digest(&bytes));
    anyhow::ensure!(
        sha == std::env::var("LLOOM_OVERLAP_CONFIG_SHA256")?,
        "private config changed"
    );
    let settings: Settings = serde_json::from_slice(&bytes)?;
    anyhow::ensure!(
        settings.cohorts > 0 && settings.cohorts.checked_mul(2).is_some(),
        "finite nonzero cohorts"
    );
    anyhow::ensure!(settings.warmup_requests > 0, "explicit warmup required");
    let expected_priced = settings
        .cohorts
        .checked_mul(2)
        .context("cohort count overflow")?;
    let expected_total = settings
        .warmup_requests
        .checked_add(expected_priced)
        .context("warmup+cohort count overflow")?;
    anyhow::ensure!(
        settings.writer_timeout_ms > settings.hold_ms,
        "writer failure budget must exceed declared hold"
    );
    let release_at = match settings.release.as_str() {
        "Publication" => Release::Publication,
        "FullSettlement" => Release::FullSettlement,
        _ => anyhow::bail!("private release arm required"),
    };
    let args = ServeArgs::try_parse_from(&settings.serve_argv)?;
    let options = resolve_options(&args.common, &|o| args.front(o), Frontend::Serve)?;
    anyhow::ensure!(options.scheduler.max_sequences == 1, "B=1 experiment only");
    anyhow::ensure!(
        options.server.host == "127.0.0.1" && options.server.port == 0,
        "private benchmark requires an ephemeral loopback listener"
    );
    anyhow::ensure!(
        options
            .load
            .model
            .as_ref()
            .is_some_and(|p| p.is_absolute() && p.exists()),
        "explicit existing pinned model selector required"
    );
    anyhow::ensure!(
        !options.memory.force,
        "memory-admission bypass is not a benchmark arm"
    );
    let seed = seed_text(options.server.pipette.prompt_seed_file.as_deref())?;
    let shape = Shape::of(&options, Frontend::Serve, seed);
    let reuse = shape.reuse_prefix;
    let trace = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&settings.trace_file)?;
    let load_start = Instant::now();
    let (engine, path) = load_engine(&options.load, shape)?;
    if !(engine.device().is_metal() && engine.batch() == 1) {
        engine.shutdown();
        anyhow::bail!("actual Metal B=1 owner required");
    }
    let api = engine
        .device()
        .as_metal_device()?
        .commands()
        .actual_command_api();
    let state = Arc::new(State {
        settings: settings.clone(),
        release_at,
        device: engine.device().clone(),
        trace: ParkingMutex::new(trace),
        error: ParkingMutex::new(None),
        next: Default::default(),
        returned: Default::default(),
        writers: Default::default(),
        peak_writers: Default::default(),
        memory: ParkingMutex::new(None),
        memory_started: AtomicBool::new(false),
        memory_closed: AtomicBool::new(false),
        memory_start_ns: Default::default(),
        samples: Default::default(),
    });
    state.event(json!({"kind":"owner", "api":api,"contract":lloom_tensor::tuning::contract().name(),
        "config_sha256":sha,"release":settings.release,"tier":"optimized instrumented cfg(test), not shipping equivalence"}));
    let _finish = FinishMemory(state.clone());
    let dispatch = state.clone();
    let ready = state.clone();
    let (tail_sender, tails) = mpsc::channel::<std::convert::Infallible>();
    let result = run_owner_dispatched(
        options,
        engine,
        path,
        Some((load_start, reuse)),
        None,
        move |shared, path, request, first| {
            // Captured by D and EACH clone, including not-yet-entered workers.
            // No message/count protocol or surviving root sender copy.
            let _keep_dispatch_alive = &tail_sender;
            dispatch.handle(shared, path, request, first)
        },
        move |shared, configuration| {
            let tiny_http::ListenAddr::IP(address) = shared.server.server_addr() else {
                anyhow::bail!("IP listener required")
            };
            let body = json!({"kind":"ready","host":"127.0.0.1","port":address.port(),"configuration":configuration,
                "benchmark":{"private":true,"pid":std::process::id(),"release":ready.settings.release}});
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&ready.settings.ready_file)?;
            serde_json::to_writer(&mut file, &body)?;
            file.write_all(b"\n")?;
            file.flush()?;
            Ok(())
        },
    );
    let tail_result = after_private_dispatch_tails(&tails, WAIT, || {
        state.finish_memory("server returned after private tails");
    });
    state.event(json!({"kind":"dispatch-tails", "completed":tail_result.is_ok(),
        "authority":"private D clone Sender disconnection", "error":tail_result.as_ref().err().map(|e|format!("{e:#}"))}));
    state.event(json!({"kind":"summary","requests":state.next.load(Ordering::Acquire),"priced_returned":state.returned.load(Ordering::Acquire),
        "peak_owned_writers":state.peak_writers.load(Ordering::Acquire),"open_writers":state.writers.load(Ordering::Acquire),
        "exact_retained_payload_bytes":Value::Null,"scope":"not instrumented; do not infer from raw credit or process footprint"}));
    owner_after_private_tails(result, tail_result)?;
    anyhow::ensure!(
        state.next.load(Ordering::Acquire) == expected_total,
        "request matrix incomplete"
    );
    anyhow::ensure!(
        state.returned.load(Ordering::Acquire) == expected_priced,
        "priced requests not all fully returned"
    );
    anyhow::ensure!(
        state.writers.load(Ordering::Acquire) == 0,
        "writer ownership not released"
    );
    anyhow::ensure!(
        state.samples.load(Ordering::Acquire) > 0,
        "sampled memory was not observed"
    );
    if let Some(error) = state.error.lock().clone() {
        anyhow::bail!(error);
    }
    Ok(())
}
