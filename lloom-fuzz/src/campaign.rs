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

    if let Some(path) = &cli.once {
        let bytes = std::fs::read(path).with_context(|| path.display().to_string())?;
        record.push(&bytes);
        return match (harness.borrow_mut())(&bytes) {
            Ok(()) => {
                println!("ok: {} ({} bytes)", path.display(), bytes.len());
                Ok(())
            }
            Err(m) => {
                let dir = record.dump(&m);
                Err(anyhow!("oracle failure on {}: {m}\nrecord: {dir:?}", path.display()))
            }
        };
    }
    if let Some(path) = &cli.replay {
        let h = Rc::clone(&harness);
        return match replay(path, move |bytes| {
            record.push(bytes);
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

fn fuzz<H>(cli: &Cli, name: &str, harness: Rc<RefCell<H>>) -> anyhow::Result<()>
where
    H: FnMut(&[u8]) -> Result<(), String> + 'static,
{
    std::fs::create_dir_all(&cli.corpus)?;
    std::fs::create_dir_all(&cli.solutions)?;
    let timeout = Duration::from_millis(cli.timeout_ms);
    let config_name = format!("{name}-{}", arm());
    let record = Record;

    let mut run_client = |state: Option<_>, mut mgr, _client: ClientDescription| -> Result<(), Error> {
        let edges_observer =
            HitcountsMapObserver::new(unsafe { std_edges_map_observer("edges") }).track_indices();
        let time_observer = TimeObserver::new("time");
        let map_feedback = MaxMapFeedback::new(&edges_observer);
        let calibration = CalibrationStage::new(&map_feedback);
        let mut feedback = feedback_or!(map_feedback, TimeFeedback::new(&time_observer));
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

        let scheduler = IndexesLenTimeMinimizerScheduler::new(&edges_observer, QueueScheduler::new());
        let mut fuzzer = StdFuzzer::new(scheduler, feedback, objective);

        let h = Rc::clone(&harness);
        let mut exec_harness = |input: &BytesInput| -> ExitKind {
            let bytes = input.target_bytes();
            let bytes: &[u8] = &bytes;
            record.push(bytes);
            match (h.borrow_mut())(bytes) {
                Ok(()) => ExitKind::Ok,
                Err(m) => {
                    let dir = record.dump(&m);
                    eprintln!("[{name}] oracle failure: {m}\n[{name}] record: {dir:?}");
                    ExitKind::Crash
                }
            }
        };
        #[allow(deprecated)]
        let mut executor = InProcessExecutor::with_timeout(
            &mut exec_harness,
            tuple_list!(edges_observer, time_observer),
            &mut fuzzer,
            &mut state,
            &mut mgr,
            timeout,
        )?;

        if state.must_load_initial_inputs() {
            match &cli.seeds {
                Some(dir) if dir.is_dir() => {
                    state.load_initial_inputs(&mut fuzzer, &mut executor, &mut mgr, &[dir.clone()])?;
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

        #[cfg(feature = "cmplog")]
        {
            use libafl::mutators::token_mutations::I2SRandReplace;
            use libafl::stages::TracingStage;
            use libafl_targets::CmpLogObserver;

            let cmplog_observer = CmpLogObserver::new("cmplog", true);
            let h2 = Rc::clone(&harness);
            let mut trace_harness = |input: &BytesInput| -> ExitKind {
                let bytes = input.target_bytes();
                match (h2.borrow_mut())(&bytes) {
                    Ok(()) => ExitKind::Ok,
                    Err(_) => ExitKind::Crash,
                }
            };
            #[allow(deprecated)]
            let tracer = InProcessExecutor::with_timeout(
                &mut trace_harness,
                tuple_list!(cmplog_observer),
                &mut fuzzer,
                &mut state,
                &mut mgr,
                timeout,
            )?;
            let tracing = TracingStage::new(tracer);
            let i2s = StdMutationalStage::new(HavocScheduledMutator::new(tuple_list!(
                I2SRandReplace::new()
            )));
            let mut stages = tuple_list!(calibration, tracing, i2s, mutational);
            fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)?;
        }
        #[cfg(not(feature = "cmplog"))]
        {
            let mut stages = tuple_list!(calibration, mutational);
            fuzzer.fuzz_loop(&mut stages, &mut executor, &mut state, &mut mgr)?;
        }
        Ok(())
    };

    let cores = Cores::from_cmdline(&cli.cores).map_err(|e| anyhow!("--cores: {e}"))?;
    let shmem_provider = StdShMemProvider::new().map_err(|e| anyhow!("shmem: {e}"))?;
    let monitor = MultiMonitor::new(|s| println!("{s}"));
    let result = Launcher::builder()
        .shmem_provider(shmem_provider)
        .configuration(EventConfig::from_name(&config_name))
        .monitor(monitor)
        .run_client(&mut run_client)
        .cores(&cores)
        .broker_port(cli.port)
        .spawn_broker(cli.broker)
        .stdout_file(None)
        .build()
        .launch();
    match result {
        Ok(()) | Err(Error::ShuttingDown) => Ok(()),
        Err(e) => Err(anyhow!("launcher: {e}")),
    }
}
