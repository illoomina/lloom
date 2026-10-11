//! The LibAFL campaign around one harness: the observers, feedbacks, stages
//! and launcher every binary in `src/bin/` shares, plus the three ways to run
//! a harness without fuzzing (`--once`, `--replay`, `--instrumentation`).
//!
//! # The shape, and why it is one `Launcher` per binary
//!
//! AFL++'s `-M`/`-S` is one main instance doing the deterministic and cmplog
//! work and secondaries doing havoc, all syncing through the corpus. Here the
//! two roles are two BUILDS — `--features cmplog` is the main, the regular
//! build the secondary — so they cannot be clients of one `CentralizedLauncher`
//! (one binary). They are clients of one **LLMP broker** instead: the main
//! spawns it (`--broker`), the secondary joins the same port (`--no-broker`),
//! and every interesting input either finds is broadcast to the other. Each
//! build has its own `EventConfig` name, so an input received from the other
//! build is re-executed locally rather than trusted — the right behaviour when
//! the two binaries' edge ids are not comparable.
//!
//! Clients are separate processes (the launcher re-executes the binary with
//! `AFL_LAUNCHER_CLIENT` set), each with its own Metal device; nothing is
//! forked with a device open.
//!
//! # What is measured before anything is fuzzed
//!
//! The banner prints the guard count SanitizerCoverage reported at init
//! (`MAX_EDGES_FOUND`) against the map size; a binary with zero guards is
//! refused, because a campaign over an uninstrumented binary is #504's
//! `evicted = 0` with a fuzzer around it.
//!
//! # Coverage a harness reports itself
//!
//! Beside the edges, every campaign carries a second map, the shape map: a
//! harness marks points in it with [`shape_hit`] — the shape probes it
//! reached, the kernels the census saw (`fuzz_geometry`, §21.8). GPU code is
//! not compiled by rustc, so no edge ever sees which Metal kernel ran; this
//! map is how that becomes coverage. A harness that never marks it leaves it
//! zero, and a zero map is never interesting. Likewise a harness may register
//! a structured mutator ([`set_structured_mutator`]) that runs as one more
//! mutational stage; without one the stage skips.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Duration;

use anyhow::{anyhow, Context};
use clap::Parser;
use libafl::{
    corpus::{Corpus, InMemoryOnDiskCorpus, OnDiskCorpus},
    events::{launcher::Launcher, ClientDescription, EventConfig},
    executors::{inprocess::InProcessExecutor, ExitKind},
    feedback_or, feedback_or_fast,
    feedbacks::{CrashFeedback, MaxMapFeedback, TimeFeedback, TimeoutFeedback},
    fuzzer::{Fuzzer, StdFuzzer},
    generators::RandBytesGenerator,
    inputs::{BytesInput, HasTargetBytes},
    monitors::MultiMonitor,
    mutators::{havoc_mutations, scheduled::HavocScheduledMutator},
    observers::{CanTrack, HitcountsMapObserver, TimeObserver},
    schedulers::{IndexesLenTimeMinimizerScheduler, QueueScheduler},
    stages::{CalibrationStage, StdMutationalStage},
    state::{HasCorpus, HasMaxSize, StdState},
    Error,
};
use libafl_bolts::{
    core_affinity::Cores,
    current_nanos,
    rands::StdRand,
    shmem::{ShMemProvider, StdShMemProvider},
    tuples::tuple_list,
};
use libafl_targets::{std_edges_map_observer, EDGES_MAP_DEFAULT_SIZE, MAX_EDGES_FOUND};

use crate::record::{replay, Record};

/// Points in the shape map. Sized for the probes' `(site, condition,
/// outcome)` triples and the census labels with room to spare; a
/// collision merges two points, it loses none of the edges.
pub const SHAPE_MAP_SIZE: usize = 1 << 14;

static mut SHAPE_MAP: [u8; SHAPE_MAP_SIZE] = [0; SHAPE_MAP_SIZE];

/// Mark the point `key` in the shape map for the running execution.
pub fn shape_hit(key: &str) {
    // FNV-1a: stable across builds, so a corpus keeps its meaning.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in key.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x0100_0000_01b3);
    }
    let i = (h % SHAPE_MAP_SIZE as u64) as usize;
    // SAFETY: one client is one thread; the map is written here and read by
    // the observer between executions, on that same thread.
    unsafe {
        let m = &raw mut SHAPE_MAP;
        (*m)[i] = (*m)[i].saturating_add(1);
    }
}

/// A structured mutation: rewrite the input in place from a random seed;
/// `false` if it did not change.
pub type StructuredMutate = fn(&mut Vec<u8>, u64) -> bool;

static STRUCTURED: std::sync::OnceLock<StructuredMutate> = std::sync::OnceLock::new();

/// Register this binary's structured mutator (once, before [`run`]).
pub fn set_structured_mutator(f: StructuredMutate) {
    let _ = STRUCTURED.set(f);
}

/// The stage [`set_structured_mutator`] feeds.
struct Structured;

impl libafl_bolts::Named for Structured {
    fn name(&self) -> &std::borrow::Cow<'static, str> {
        static NAME: std::borrow::Cow<'static, str> = std::borrow::Cow::Borrowed("Structured");
        &NAME
    }
}

impl<S: libafl::state::HasRand> libafl::mutators::Mutator<BytesInput, S> for Structured {
    fn mutate(
        &mut self,
        state: &mut S,
        input: &mut BytesInput,
    ) -> Result<libafl::mutators::MutationResult, Error> {
        use libafl::inputs::{HasMutatorBytes, ResizableMutator};
        use libafl_bolts::rands::Rand;
        let Some(f) = STRUCTURED.get() else {
            return Ok(libafl::mutators::MutationResult::Skipped);
        };
        let mut bytes = input.mutator_bytes().to_vec();
        if !f(&mut bytes, state.rand_mut().next()) {
            return Ok(libafl::mutators::MutationResult::Skipped);
        }
        input.resize(bytes.len(), 0);
        input.mutator_bytes_mut().copy_from_slice(&bytes);
        Ok(libafl::mutators::MutationResult::Mutated)
    }

    fn post_exec(
        &mut self,
        _state: &mut S,
        _new_corpus_id: Option<libafl::corpus::CorpusId>,
    ) -> Result<(), Error> {
        Ok(())
    }
}

/// The command line every harness binary takes.
#[derive(Parser, Debug, Clone)]
#[command(about = "A LibAFL campaign over one of lloom's stateful fuzz models (lloom #706)")]
pub struct Cli {
    /// Corpus directory (shared between the main and the secondary).
    #[arg(long, default_value = "fuzz-out/corpus")]
    pub corpus: PathBuf,
    /// Where crashing / oracle-failing inputs go.
    #[arg(long, default_value = "fuzz-out/solutions")]
    pub solutions: PathBuf,
    /// Seed inputs to load before fuzzing (a directory of files). Empty:
    /// random seeds are generated.
    #[arg(long)]
    pub seeds: Option<PathBuf>,
    /// Cores to run clients on, LibAFL syntax ("0", "0-3", "all").
    #[arg(long, default_value = "0")]
    pub cores: String,
    /// The LLMP broker port. The main spawns the broker; a secondary joins it.
    #[arg(long, default_value_t = 1337)]
    pub port: u16,
    /// Spawn the broker (the main instance). A secondary passes
    /// `--no-broker` and joins the main's port.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    pub broker: bool,
    /// How many of the last executed inputs the persistent record keeps.
    #[arg(long, default_value_t = 64)]
    pub record: usize,
    /// Where records are written on a crash / oracle failure.
    #[arg(long, default_value = "fuzz-out/records")]
    pub record_dir: PathBuf,
    /// Per-execution timeout.
    #[arg(long, default_value_t = 5000)]
    pub timeout_ms: u64,
    /// The longest input the mutators produce.
    #[arg(long, default_value_t = 512)]
    pub max_len: usize,
    /// Run one input (a file) through the harness in this process and exit.
    #[arg(long)]
    pub once: Option<PathBuf>,
    /// Replay a record directory (or one input) in order in this process.
    #[arg(long)]
    pub replay: Option<PathBuf>,
    /// Print the instrumentation readout (guards, map size) and exit.
    #[arg(long)]
    pub instrumentation: bool,
    /// Do not refuse an uninstrumented binary (for the positive control).
    #[arg(long)]
    pub allow_uninstrumented: bool,
    /// One client, no broker, restarted by re-executing the binary instead
    /// of forking it (`--cores`, `--port`, `--broker` are ignored). For a
    /// harness that compiles Metal kernels at run time.
    #[arg(long)]
    pub single: bool,
    /// Redirect the clients' stdout and stderr to this file (the launcher
    /// silences them otherwise); the in-process crash report lands here.
    #[arg(long)]
    pub client_log: Option<PathBuf>,
}

/// The guards SanitizerCoverage reported at init, and the map they index.
pub fn instrumentation() -> (usize, usize) {
    // SAFETY: written only by `__sanitizer_cov_trace_pc_guard_init`, which the
    // module constructors run before `main`; read-only here.
    let guards = unsafe { std::ptr::read_volatile(&raw const MAX_EDGES_FOUND) };
    (guards, EDGES_MAP_DEFAULT_SIZE)
}

/// The build's arm, for the banner and the event configuration name.
pub fn arm() -> &'static str {
    if cfg!(feature = "cmplog") {
        "cmplog"
    } else {
        "pcguard"
    }
}

fn banner(name: &str) -> anyhow::Result<()> {
    let (guards, map) = instrumentation();
    println!(
        "lloom-fuzz {name} [{}] guards={guards} edges_map={map} record={}",
        arm(),
        Record.capacity()
    );
    Ok(())
}

/// Run `harness` under `cli`: once, as a replay, or as the campaign.
///
/// `harness` takes the raw input and returns the oracle's verdict; a panic
/// inside it is a crash like any other (the record's hook runs first).
pub fn run<H>(cli: &Cli, name: &str, harness: H) -> anyhow::Result<()>
where
    H: FnMut(&[u8]) -> Result<(), String> + 'static,
{
    if cli.max_len > crate::record::MAX_INPUT {
        return Err(anyhow!(
            "--max-len {} exceeds persistent record capacity {}",
            cli.max_len,
            crate::record::MAX_INPUT
        ));
    }
    let record = Record::install(&cli.record_dir, cli.record);
    banner(name)?;
    let (guards, map) = instrumentation();
    if cli.instrumentation {
        return Ok(());
    }
    if guards == 0 && !cli.allow_uninstrumented {
        return Err(anyhow!(
            "this binary reports 0 SanitizerCoverage guards: it was built without the \
             instrumentation flags in lloom-fuzz/.cargo/config.toml (or without --target). \
             A campaign over it would measure nothing. --allow-uninstrumented overrides."
        ));
    }
    if guards >= map {
        return Err(anyhow!(
            "{guards} guards do not fit the {map}-entry edges map: rebuild with \
             LIBAFL_EDGES_MAP_DEFAULT_SIZE raised (see README)"
        ));
    }

    let harness = Rc::new(RefCell::new(harness));

    if cli.once.is_some() || cli.replay.is_some() {
        Record::install_crash_dump()?;
    }
    if let Some(path) = &cli.once {
        let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
        record.push(&bytes)?;
        return match (harness.borrow_mut())(&bytes) {
            Ok(()) => {
                println!("ok: {} ({} bytes)", path.display(), bytes.len());
                Ok(())
            }
            Err(m) => {
                let dir = record.dump(&m);
                Err(anyhow!(
                    "oracle failure on {}: {m}\nrecord: {dir:?}",
                    path.display()
                ))
            }
        };
    }
    if let Some(path) = &cli.replay {
        let h = Rc::clone(&harness);
        return match replay(path, move |bytes| {
            record
                .push(bytes)
                .map_err(|e| format!("recorder failure: {e}"))?;
            (h.borrow_mut())(bytes)
        }) {
            Ok(n) => {
                println!("replayed {n} input(s) from {}: all clean", path.display());
                Ok(())
            }
            Err((i, m)) => {
                let dir = record.dump(&m);
                Err(anyhow!("input {i} failed: {m}\nrecord: {dir:?}"))
            }
        };
    }

    fuzz(cli, name, harness)
}

/// Every executor invocation changes the same persistent harness, including
/// comparison tracing. Record before entering it so replay keeps that order.
fn execute_recorded<H>(record: Record, harness: &RefCell<H>, name: &str, bytes: &[u8]) -> ExitKind
where
    H: FnMut(&[u8]) -> Result<(), String>,
{
    if let Err(error) = record.push(bytes) {
        eprintln!("[{name}] recorder failure (harness not executed): {error}");
        return ExitKind::Crash;
    }
    match (harness.borrow_mut())(bytes) {
        Ok(()) => ExitKind::Ok,
        Err(message) => {
            let directory = record.dump(&message);
            eprintln!("[{name}] oracle failure: {message}\n[{name}] record: {directory:?}");
            ExitKind::Crash
        }
    }
}

fn fuzz<H>(cli: &Cli, name: &str, harness: Rc<RefCell<H>>) -> anyhow::Result<()>
where
    H: FnMut(&[u8]) -> Result<(), String> + 'static,
{
    std::fs::create_dir_all(&cli.corpus)?;
    std::fs::create_dir_all(&cli.solutions)?;
    let timeout = Duration::from_millis(cli.timeout_ms);
    let config_name = format!("{name}-{}", arm());
    let record = Record;

    // One client's fuzzer, over whichever event manager runs it: the
    // launcher's LLMP clients, or `--single`'s restarting manager.
    macro_rules! client {
        ($state:expr, $mgr:expr) => {{
            let state = $state;
            let mut mgr = $mgr;
            let edges_observer =
                HitcountsMapObserver::new(unsafe { std_edges_map_observer("edges") })
                    .track_indices();
            let time_observer = TimeObserver::new("time");
            // SAFETY: the map is a static of `SHAPE_MAP_SIZE` bytes, written
            // only by `shape_hit` on this client's one thread.
            let shape_observer = unsafe {
                libafl::observers::StdMapObserver::from_mut_ptr(
                    "shape",
                    (&raw mut SHAPE_MAP).cast::<u8>(),
                    SHAPE_MAP_SIZE,
                )
            };
            let map_feedback = MaxMapFeedback::new(&edges_observer);
            let shape_feedback = MaxMapFeedback::new(&shape_observer);
            let calibration = CalibrationStage::new(&map_feedback);
            let mut feedback = feedback_or!(
                map_feedback,
                shape_feedback,
                TimeFeedback::new(&time_observer)
            );
            let mut objective = feedback_or_fast!(CrashFeedback::new(), TimeoutFeedback::new());

            let mut state = match state {
                Some(s) => s,
                None => StdState::new(
                    StdRand::with_seed(current_nanos()),
                    InMemoryOnDiskCorpus::new(&cli.corpus)?,
                    OnDiskCorpus::new(&cli.solutions)?,
                    &mut feedback,
                    &mut objective,
                )?,
            };
            state.set_max_size(cli.max_len);

            let scheduler =
                IndexesLenTimeMinimizerScheduler::new(&edges_observer, QueueScheduler::new());
            let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

            let h = Rc::clone(&harness);
            let mut exec_harness = |input: &BytesInput| -> ExitKind {
                let bytes = input.target_bytes();
                execute_recorded(record, &h, name, &bytes)
            };
            #[allow(deprecated)]
            let executor = InProcessExecutor::with_timeout(
                &mut exec_harness,
                tuple_list!(edges_observer, shape_observer, time_observer),
                &mut fuzzer,
                &mut state,
                &mut mgr,
                timeout,
            )?;
            // Comparison tracing shares the one in-process executor: separate
            // typed panic hooks would cast the same global executor pointer.
            #[cfg(feature = "cmplog")]
            let mut executor = libafl::executors::ShadowExecutor::new(
                executor,
                tuple_list!(libafl_targets::CmpLogObserver::new("cmplog", true)),
            );
            #[cfg(not(feature = "cmplog"))]
            let mut executor = executor;
            // The recorder wraps the final executor-installed panic/signal hooks
            // before the first target invocation, including corpus seeding.
            Record::install_crash_dump()?;

            if state.must_load_initial_inputs() {
                match &cli.seeds {
                    Some(dir) if dir.is_dir() => {
                        state.load_initial_inputs(
                            &mut fuzzer,
                            &mut executor,
                            &mut mgr,
                            &[dir.clone()],
                        )?;
                    }
                    _ => {
                        let mut generator = RandBytesGenerator::new(
                            std::num::NonZeroUsize::new(64).expect("64 is non-zero"),
                        );
                        state.generate_initial_inputs(
                            &mut fuzzer,
                            &mut executor,
                            &mut generator,
                            &mut mgr,
                            8,
                        )?;
                    }
                }
                println!("[{name}] corpus: {} input(s)", state.corpus().count());
            }

            let mutational = StdMutationalStage::new(HavocScheduledMutator::new(havoc_mutations()));
            let structured = StdMutationalStage::new(Structured);

            #[cfg(feature = "cmplog")]
            {
                use libafl::mutators::token_mutations::I2SRandReplace;
                use libafl::stages::ShadowTracingStage;

                let tracing = ShadowTracingStage::new();
                let i2s = StdMutationalStage::new(HavocScheduledMutator::new(tuple_list!(
                    I2SRandReplace::new()
                )));
                let mut stages = tuple_list!(calibration, tracing, i2s, structured, mutational);
                fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)?;
            }
            #[cfg(not(feature = "cmplog"))]
            {
                let mut stages = tuple_list!(calibration, structured, mutational);
                fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)?;
            }
            Ok(())
        }};
    }

    if cli.single {
        // One client, restarted by re-executing this binary rather than by
        // forking it: LibAFL's LLMP clients always fork their fuzzer child
        // (`fork(true)` in `setup_restarting_mgr_llmp`), and in such a child
        // a Metal library load has failed with MTLCompilerService
        // unreachable (`measurements/issue-761-shape-probe-spike.md` §4).
        remove_stale_shmem_server_socket();
        let monitor = MultiMonitor::new(|s| println!("{s}"));
        let mut shmem_provider = StdShMemProvider::new().map_err(|e| anyhow!("shmem: {e}"))?;
        let (state, mgr) = match libafl::events::SimpleRestartingEventManager::launch(
            monitor,
            &mut shmem_provider,
            false,
        ) {
            Ok(r) => r,
            Err(Error::ShuttingDown) => return Ok(()),
            Err(e) => return Err(anyhow!("restarting manager: {e}")),
        };
        let run = || -> Result<(), Error> { client!(state, mgr) };
        return match run() {
            Ok(()) | Err(Error::ShuttingDown) => Ok(()),
            Err(e) => Err(anyhow!("client: {e}")),
        };
    }

    let mut run_client = |state: Option<_>, mgr, _client: ClientDescription| -> Result<(), Error> {
        client!(state, mgr)
    };

    let cores = Cores::from_cmdline(&cli.cores).map_err(|e| anyhow!("--cores: {e}"))?;
    if cli.broker {
        remove_stale_shmem_server_socket();
    }
    let shmem_provider = StdShMemProvider::new().map_err(|e| anyhow!("shmem: {e}"))?;
    let monitor = MultiMonitor::new(|s| println!("{s}"));
    let client_log: Option<String> = cli.client_log.as_ref().map(|p| p.display().to_string());
    let result = Launcher::builder()
        .shmem_provider(shmem_provider)
        .configuration(EventConfig::from_name(&config_name))
        .monitor(monitor)
        .run_client(&mut run_client)
        .cores(&cores)
        .broker_port(cli.port)
        .spawn_broker(cli.broker)
        // Re-execute the binary for each client rather than fork it: a forked
        // child inherits the parent's Metal device and command queue, and
        // every execution then dies in the driver (measured: 2 252 of 2 252
        // executions were "objectives" on the first cb_liveness campaign).
        // Each client still forks its fuzzer child (LibAFL's LLMP restarting
        // manager always does), before any device is open: the harnesses
        // open theirs lazily, in the child. A harness that compiles Metal
        // kernels there has seen the compiler service unreachable; `--single`
        // re-executes instead.
        .fork(false)
        .stdout_file(client_log.as_deref())
        .stderr_file(client_log.as_deref())
        .build()
        .launch();
    match result {
        Ok(()) | Err(Error::ShuttingDown) => Ok(()),
        Err(e) => Err(anyhow!("launcher: {e}")),
    }
}

/// On macOS LibAFL's shared memory goes through a served provider whose unix
/// socket lives in the working directory (`./libafl_unix_shmem_server`). A
/// campaign killed by a signal leaves the file behind, and the next launch
/// then fails with `ConnectionRefused` instead of starting a server (measured
/// on the second campaign of this crate's life). Remove it iff nothing answers
/// on it — a live campaign in the same directory keeps its socket.
fn remove_stale_shmem_server_socket() {
    let path = std::path::Path::new("./libafl_unix_shmem_server");
    if !path.exists() {
        return;
    }
    if std::os::unix::net::UnixStream::connect(path).is_ok() {
        return;
    }
    if std::fs::remove_file(path).is_ok() {
        eprintln!(
            "removed a stale {} (nothing was listening on it)",
            path.display()
        );
    }
}
