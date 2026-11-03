//! A continuous step that fails part-way is recovered as a failed static
//! batch is — the fence, the extra's abort and the driver's (the drafter's
//! contexts, where the speculative loop holds them), then every row reset —
//! and the next requests, prefix reuse on and no cold flag, decode what fresh
//! sequences decode (`batched-decode-design.md` §11.11).
//!
//! The scheduler runs its continuous driver: the K=1 batched loop
//! (`BatchDecoder`), or the speculative one (`SpecBatchDecoder`, which the
//! scheduler's `extra` builds and hands the drafter to, as `--dspark
//! --batch N` on the pool arms does). The failure is injected after the
//! model's own work, so the state that work moved stays moved and the work
//! is still on the device when the error reaches the scheduler: a prompt's
//! prefill forward, a decode step's `forward_rows` over both rows (encoded,
//! not yet committed), or a speculative round's ragged verify (enqueued,
//! both rows' windows open). Every row of the failed step gets the error.
//!
//! Each test then sends both prompts extended by a token, so a sequence
//! returned as it was would be extended in place; then the same prompts with
//! the cold flag, which resets the sequence first, as the reference. A second
//! scheduler that never failed decodes the same extension warm; it is
//! printed beside each row to tell a stale state from a clean reuse. The
//! tokens are compared before the recovery's counts are, so a mutant that
//! drops the fence or an abort shows what it decodes before it fails.
//!
//! Runs on the CPU over own rewindable caches, and with the `metal` feature
//! natively: own paged caches (`--rows private`) and rows of one shared
//! `BatchPool` (`--rows pool`), where a reset releases the row's pages into
//! the quarantine while the failed step's work still writes them.

#[path = "support/batch_recovery.rs"]
mod batch_recovery;

use batch_recovery::{config, round, Faulty, Garbage, Site};
use lloom::models::lfm2::{Config, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::{DecodeOptions, Session, SpecBatchDecoder};
use lloom::profile::{Profile, RewindProof, RewindableExact};
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_engine::{
    spawn_batch, BatchConfig, BatchHandle, BatchMode, BatchRecovery, Driver, MakeDriver, RingPool,
    SlotPool,
};
use lloom_tensor::{DType, Device, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const ROWS: usize = 2;
const MAX: usize = 6;

/// The scheduler's continuous driver.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Loop {
    /// The K=1 batched loop the scheduler builds itself.
    K1,
    /// The speculative batched loop, built from the `extra`'s drafter.
    Speculative,
}

/// Where the rows' state lives: the scheduler's sequences, built over
/// `cfg`'s cache on a device ([`own_rows`], [`shared_rows`]).
type Rows<P> = fn(&Config, DType, &Device) -> SlotPool<State<P>>;

/// A cache of its own per sequence.
fn own_rows<P: Profile>(cfg: &Config, dtype: DType, device: &Device) -> SlotPool<State<P>> {
    SlotPool::new(ROWS, || State::<P>::new(true, dtype, cfg, device).unwrap())
}

/// Every sequence a slot of one shared pool, its batched arms on, as
/// `lloom-serve --batch N --rows pool` builds them.
#[cfg(feature = "metal")]
fn shared_rows<P: Profile<Rows = lloom::profile::Paged>>(
    cfg: &Config,
    dtype: DType,
    device: &Device,
) -> SlotPool<State<P>> {
    let pool = lloom::models::lfm2::BatchPool::new(cfg, device, ROWS, 128).unwrap();
    pool.set_batched_arms(true);
    let mut caches = (0..ROWS)
        .map(|_| State::<P>::for_batch(true, dtype, cfg, device, &pool).unwrap())
        .collect::<Vec<_>>()
        .into_iter();
    SlotPool::new(ROWS, || caches.next().expect("one cache per slot"))
}

/// The scheduler's `extra`: the drafter the speculative loop takes when the
/// scheduler starts, or nothing (the K=1 loop), and a count of the recovery
/// the scheduler runs. The fence is the default one.
struct Extra {
    spec: Option<Garbage>,
    fences: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
}

impl<P: Profile> MakeDriver<Faulty<P>> for Extra {
    fn make_driver<'m>(
        &mut self,
        _model: &'m Faulty<P>,
        opts: &DecodeOptions,
        device: &Device,
    ) -> Result<Option<Driver<'m, Faulty<P>>>> {
        let Some(spec) = self.spec.take() else {
            return Ok(None);
        };
        let rewind = RewindProof::<P>::ask()
            .ok_or_else(|| lloom_tensor::Error::msg("the speculative loop needs a Ring profile"))?;
        let dec = SpecBatchDecoder::<'m, Faulty<P>, Garbage, Session<'m, Faulty<P>>>::new(
            rewind,
            spec,
            opts.clone(),
            device.clone(),
        )?;
        Ok(Some(Box::new(dec)))
    }
}

impl BatchRecovery for Extra {
    fn fence(&mut self, device: &Device) -> Result<()> {
        self.fences.fetch_add(1, Ordering::AcqRel);
        lloom_engine::recovery::fence_failed_execution(device)
    }
    fn abort(&mut self) {
        self.aborts.fetch_add(1, Ordering::AcqRel);
    }
}

struct Served<P: Profile> {
    model: Arc<Faulty<P>>,
    handle: BatchHandle<Faulty<P>, Extra>,
    thread: std::thread::JoinHandle<()>,
    fences: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
    /// The drafter's own count, wherever it ended up (the speculative loop
    /// holds it): zero on the K=1 loop, which has none.
    drafter_aborts: Arc<AtomicUsize>,
}

impl<P: Profile> Served<P> {
    /// The continuous scheduler over two sequences of `cfg`'s cache on
    /// `device`, prefix reuse on.
    fn start(device: Device, cfg: Config, dtype: DType, driver: Loop, rows: Rows<P>) -> Self {
        let inner = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
            .with_dtype(dtype)
            .model::<P>(&device)
            .unwrap();
        let model = Arc::new(Faulty::around(inner, &device));
        let pool = rows(&cfg, dtype, &device);
        let spec = Garbage::new(ROWS);
        let drafter_aborts = Arc::clone(&spec.aborts);
        let fences = Arc::new(AtomicUsize::new(0));
        let aborts = Arc::new(AtomicUsize::new(0));
        let bcfg = BatchConfig {
            device: device.clone(),
            pool,
            rings: RingPool::new(),
            opts: DecodeOptions {
                lookahead: 1,
                ..DecodeOptions::new(0)
            },
            max_rows: ROWS,
            reuse_prefix: true,
            share_prefix: false,
            prefill_batch: 1,
            prefill_decode: lloom::nn::PrefillDecode::Ride,
            intake_per_tick: lloom_engine::options::DEFAULT_INTAKE_PER_TICK,
            mode: BatchMode::Continuous,
            extra: Extra {
                spec: (driver == Loop::Speculative).then_some(spec),
                fences: Arc::clone(&fences),
                aborts: Arc::clone(&aborts),
            },
            on_counters: None,
        };
        let (handle, thread) = spawn_batch(Arc::clone(&model), |m: &Faulty<P>| m, bcfg).unwrap();
        Self {
            model,
            handle,
            thread,
            fences,
            aborts,
            drafter_aborts,
        }
    }

    /// `(fences, extra aborts, drafter aborts)` the scheduler has run.
    fn recovery(&self) -> (usize, usize, usize) {
        (
            self.fences.load(Ordering::Acquire),
            self.aborts.load(Ordering::Acquire),
            self.drafter_aborts.load(Ordering::Acquire),
        )
    }

    /// `prompts` together through the continuous driver.
    fn round(
        &self,
        prompts: &[Vec<u32>],
        cold: bool,
    ) -> Vec<std::result::Result<Vec<u32>, String>> {
        round(&self.handle, prompts, cold, MAX)
    }

    fn stop(self) {
        self.handle.shutdown();
        self.thread.join().unwrap();
    }
}

/// Round 1 caches both rows' prompts and replies; round 2 extends them and
/// fails at `site`'s `n`-th call; round 3 extends round 2's prompts by one
/// token, reusing whatever each sequence came back with; round 4 sends
/// round 3's prompts cold, from reset sequences — the reference.
#[allow(clippy::too_many_arguments)]
fn a_failed_step_then_the_next<P: Profile>(
    arm: &str,
    device: Device,
    cfg: Config,
    dtype: DType,
    driver: Loop,
    rows: Rows<P>,
    site: Site,
    n: usize,
) {
    // The K=1 loop's batched step does not tap: its target loads untapped,
    // as `--batch N` without `--dspark` does.
    let mut cfg = cfg;
    if driver == Loop::K1 {
        cfg.tap_layers.clear();
    }
    let clean = Served::<P>::start(device.clone(), cfg.clone(), dtype, driver, rows);
    let served = Served::<P>::start(device.clone(), cfg, dtype, driver, rows);
    let first = [vec![1u32, 2, 3, 4], vec![5u32, 6, 7]];
    let replies = served.round(&first, false);
    let mut extended: Vec<Vec<u32>> = Vec::new();
    for (i, (p, r)) in first.iter().zip(&replies).enumerate() {
        let r = r.as_ref().expect("round 1 is served");
        let mut e = p.clone();
        e.extend(r);
        // Two tokens past the cached log, so the extension's prefill is a
        // multi-position forward on every driver ([`Site::Prefill`]).
        e.extend([9 + i as u32, 30 + i as u32]);
        extended.push(e);
    }
    assert_eq!(
        clean.round(&first, false),
        replies,
        "the clean run's round 1"
    );
    assert_eq!(
        served.recovery(),
        (0, 0, 0),
        "a round that succeeded ran the recovery"
    );

    served.model.arm(site, n);
    let failed = served.round(&extended, false);
    assert_eq!(served.model.failed(), 1, "the {site:?} injection fired");
    for (i, f) in failed.iter().enumerate() {
        let why = f
            .as_ref()
            .expect_err("every row of the failed step gets its error");
        assert!(why.contains("injected"), "row {i}: {why}");
    }
    let queued = served.model.queued();
    eprintln!("ARM {arm} SITE {site:?} queued at the failure: {queued:?}");
    if site == Site::Rows && device.is_metal() {
        assert!(
            queued[0].open_dispatches > 0,
            "the failed step's forward was committed before its error reached the scheduler: \
             nothing of it was left queued to reset beside ({queued:?})"
        );
    }

    let next: Vec<Vec<u32>> = extended
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let mut n = e.clone();
            n.push(20 + i as u32);
            n
        })
        .collect();
    let reused = served.round(&next, false);
    let reference = served.round(&next, true);
    let warm = clean.round(&next, false);
    clean.stop();
    let mut wrong = Vec::new();
    for (i, (got, want)) in reused.iter().zip(&reference).enumerate() {
        let want = want.as_ref().expect("the cold reference is served");
        let warm = warm[i].as_ref().expect("the clean extension is served");
        eprintln!(
            "ARM {arm} SITE {site:?} row {i}: after {got:?} cold {want:?} warm {warm:?} \
             (after==cold {}, after==warm {})",
            got.as_ref().is_ok_and(|g| g == want),
            got.as_ref().is_ok_and(|g| g == warm)
        );
        match got {
            Ok(got) if got == want => {}
            Ok(got) => wrong.push(format!(
                "row {i}: decoded {got:?}, a fresh sequence {want:?}"
            )),
            Err(e) => wrong.push(format!("row {i}: refused: {e}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "{arm}: the requests after the failed step: {}",
        wrong.join("; ")
    );
    let drafter = usize::from(driver == Loop::Speculative);
    assert_eq!(
        served.recovery(),
        (1, 1, drafter),
        "(fences, extra aborts, drafter aborts): only the failed step recovered, and the \
         drafter's contexts were dropped where the driver holds them"
    );
    served.stop();
}

/// The failure lands in row 0's prefill forward of round 2, after it
/// advanced the conv phase and wrote KV; row 1's prefill never ran.
#[test]
fn a_continuous_step_failing_in_a_prefill_hands_its_rows_back_reset() {
    a_failed_step_then_the_next::<RewindableExact>(
        "cpu-rewindable-k1",
        Device::Cpu,
        config(),
        DType::F32,
        Loop::K1,
        own_rows,
        Site::Prefill,
        1,
    );
}

/// The failure lands in round 2's first decode step, after `forward_rows`
/// advanced both rows' caches by a position the scheduler never recorded.
#[test]
fn a_continuous_decode_step_failing_after_both_rows_moved_hands_them_back_reset() {
    a_failed_step_then_the_next::<RewindableExact>(
        "cpu-rewindable-k1",
        Device::Cpu,
        config(),
        DType::F32,
        Loop::K1,
        own_rows,
        Site::Rows,
        1,
    );
}

/// The speculative loop: the failure lands in round 2's first round, after
/// the ragged verify over both rows was enqueued, both caches inside the
/// windows it opened.
#[test]
fn a_speculative_round_failing_with_its_verify_enqueued_hands_its_rows_back_reset() {
    a_failed_step_then_the_next::<RewindableExact>(
        "cpu-rewindable-spec",
        Device::Cpu,
        config(),
        DType::F32,
        Loop::Speculative,
        own_rows,
        Site::VerifyRowsEnqueue,
        1,
    );
}

#[cfg(feature = "metal")]
use batch_recovery::native_config;

#[cfg(feature = "metal")]
#[test]
fn natively_a_paged_decode_step_failing_after_both_rows_moved_hands_them_back_reset() {
    a_failed_step_then_the_next::<lloom::profile::PoolSpecExact>(
        "metal-private-paged-k1",
        Device::new_metal(0).unwrap(),
        native_config(true),
        DType::F16,
        Loop::K1,
        own_rows,
        Site::Rows,
        1,
    );
}

#[cfg(feature = "metal")]
#[test]
fn natively_a_pool_rows_decode_step_failing_after_both_rows_moved_hands_them_back_reset() {
    a_failed_step_then_the_next::<lloom::profile::PoolSpecExact>(
        "metal-pool-rows-k1",
        Device::new_metal(0).unwrap(),
        native_config(true),
        DType::F16,
        Loop::K1,
        shared_rows,
        Site::Rows,
        1,
    );
}

#[cfg(feature = "metal")]
#[test]
fn natively_a_pool_rows_prefill_failing_hands_its_rows_back_reset() {
    a_failed_step_then_the_next::<lloom::profile::PoolSpecExact>(
        "metal-pool-rows-k1",
        Device::new_metal(0).unwrap(),
        native_config(true),
        DType::F16,
        Loop::K1,
        shared_rows,
        Site::Prefill,
        1,
    );
}

#[cfg(feature = "metal")]
#[test]
fn natively_a_pool_rows_speculative_round_failing_with_its_verify_enqueued_hands_its_rows_back_reset(
) {
    a_failed_step_then_the_next::<lloom::profile::PoolSpecExact>(
        "metal-pool-rows-spec",
        Device::new_metal(0).unwrap(),
        native_config(true),
        DType::F16,
        Loop::Speculative,
        shared_rows,
        Site::VerifyRowsEnqueue,
        1,
    );
}

#[cfg(feature = "metal")]
#[test]
fn natively_a_rewindable_speculative_round_failing_with_its_verify_enqueued_hands_its_rows_back_reset(
) {
    a_failed_step_then_the_next::<RewindableExact>(
        "metal-rewindable-spec",
        Device::new_metal(0).unwrap(),
        native_config(false),
        DType::F16,
        Loop::Speculative,
        own_rows,
        Site::VerifyRowsEnqueue,
        1,
    );
}
