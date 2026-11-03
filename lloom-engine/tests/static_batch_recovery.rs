//! A static speculative batch that fails part-way hands every sequence back
//! reset, so the next request — prefix reuse on, no cold flag — decodes what
//! a fresh sequence decodes (`batched-decode-design.md` §11.10).
//!
//! The batch runs the pipelined speculative driver, as the
//! `--dspark --batch N --rows private` body does, over a tiny LFM2 on the
//! rewindable arms with an injected failure that lands after the model's own
//! work: a prefill forward that has advanced the conv phase and written KV,
//! or a verify completion that resolved one row's window while the other
//! row's verify is still enqueued, its cache inside an open speculative
//! window. Every row of the failed batch gets the error.
//!
//! Each test then sends both rows' prompts extended by a token, so a
//! sequence returned as it was would be extended in place; and the same
//! prompts again with the cold flag, which resets the sequence, as the
//! reference. The scheduler's `extra` counts the recovery it runs: one fence
//! and one abort for the failed batch, none for any batch that succeeded.
//!
//! Runs on the CPU, and with the `metal` feature natively too, on the
//! rewindable arms and on the private paged pool-spec cache
//! (`--rows private --cache pool-spec`), so the fence waits on real queued
//! verifies. All through the public route a server uses (`spawn_batch`,
//! `BatchHandle::submit`).

#[path = "support/batch_recovery.rs"]
mod batch_recovery;

use batch_recovery::{config, padded, round, round_reuse, small_config, Faulty, Garbage, Site};
use lloom::models::lfm2::{Config, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::{DecodeOptions, Session};
use lloom::profile::{Profile, RewindProof, RewindableExact};
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_engine::{
    spawn_batch, BatchConfig, BatchHandle, BatchMode, BatchRecovery, Driver, MakeDriver, RingPool,
    SlotPool, StaticBody,
};
use lloom_tensor::{DType, Device, Result};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const ROWS: usize = 2;
const MAX: usize = 6;

/// The scheduler's `extra` on the static arm: the batch thread's one
/// drafter, and a count of the recovery the scheduler runs.
struct Drafter {
    spec: Garbage,
    fences: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
}

impl<P: Profile> MakeDriver<Faulty<P>> for Drafter {
    fn make_driver<'m>(
        &mut self,
        _model: &'m Faulty<P>,
        _opts: &DecodeOptions,
        _device: &Device,
    ) -> Result<Option<Driver<'m, Faulty<P>>>> {
        Ok(None)
    }
}

impl BatchRecovery for Drafter {
    fn fence(&mut self, device: &Device) -> Result<()> {
        self.fences.fetch_add(1, Ordering::AcqRel);
        lloom_engine::recovery::fence_failed_execution(device)
    }
    fn abort(&mut self) {
        self.aborts.fetch_add(1, Ordering::AcqRel);
    }
}

/// The static body `--dspark --batch N --rows private` serves: every row
/// through the pipelined driver over the batch thread's one drafter.
fn pipelined<P: Profile>() -> StaticBody<Faulty<P>, Drafter> {
    Box::new(|idle, sessions, prompts, specs, on_tokens| {
        let caps: Vec<usize> = specs.iter().map(|s| s.max_tokens).collect();
        let rewind = RewindProof::<P>::ask().expect("a rewindable profile");
        Session::decode_speculative_pipelined_with_caps(
            sessions,
            rewind,
            &mut idle.extra.spec,
            prompts,
            idle.opts,
            &caps,
            &specs[0].eos_ids,
            idle.device,
            on_tokens,
        )
    })
}

struct Served<P: Profile> {
    model: Arc<Faulty<P>>,
    handle: BatchHandle<Faulty<P>, Drafter>,
    thread: std::thread::JoinHandle<()>,
    fences: Arc<AtomicUsize>,
    aborts: Arc<AtomicUsize>,
}

impl<P: Profile> Served<P> {
    /// The static scheduler over two private sequences of `cfg`'s cache on
    /// `device`, prefix reuse on, the pipelined body.
    fn start(device: Device, cfg: Config, dtype: DType) -> Self {
        let inner = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
            .with_dtype(dtype)
            .model::<P>(&device)
            .unwrap();
        let model =
            Arc::new(Faulty::around(inner, &device).with_rope_end(cfg.max_position_embeddings));
        let pool = SlotPool::new(ROWS, || {
            State::<P>::new(true, dtype, &cfg, &device).unwrap()
        });
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
            mode: BatchMode::Static(pipelined::<P>()),
            extra: Drafter {
                spec: Garbage::new(ROWS),
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
        }
    }

    fn recovery(&self) -> (usize, usize) {
        (
            self.fences.load(Ordering::Acquire),
            self.aborts.load(Ordering::Acquire),
        )
    }

    /// One static batch over `prompts`: each row's tokens, or its error.
    fn batch(
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

/// Batch 1 caches both rows' prompts and replies; batch 2 extends them and
/// fails at `site`'s `n`-th call; batch 3 extends batch 2's prompts by one
/// token, reusing whatever each sequence came back with; batch 4 sends
/// batch 3's prompts cold, from reset sequences — the reference. Every row
/// of batch 3 is compared before the recovery's counts are, so a sequence
/// handed back as it was fails on what it decodes.
///
/// A second scheduler runs batch 1 then batch 3 with no failure between:
/// what a clean sequence decodes when batch 3 EXTENDS it. A warm extension
/// is not byte-identical to a cold prefill on every arm (the zero-padded
/// conv context, `sequence-reuse-design.md` §2.5), so it is printed beside
/// each row to tell a stale state from a clean reuse; the assertion is the
/// cold reference, which is what a reset sequence decodes.
fn a_failed_batch_then_the_next<P: Profile>(
    arm: &str,
    device: Device,
    cfg: Config,
    dtype: DType,
    site: Site,
    n: usize,
) {
    let clean = Served::<P>::start(device.clone(), cfg.clone(), dtype);
    let served = Served::<P>::start(device, cfg, dtype);
    let first = [vec![1u32, 2, 3, 4], vec![5u32, 6, 7]];
    let replies = served.batch(&first, false);
    let mut extended: Vec<Vec<u32>> = Vec::new();
    for (i, (p, r)) in first.iter().zip(&replies).enumerate() {
        let r = r.as_ref().expect("batch 1 is served");
        let mut e = p.clone();
        e.extend(r);
        e.push(9 + i as u32);
        extended.push(e);
    }
    assert_eq!(
        served.recovery(),
        (0, 0),
        "a batch that succeeded ran the recovery"
    );
    assert_eq!(
        clean.batch(&first, false),
        replies,
        "the clean run's batch 1"
    );

    served.model.arm(site, n);
    let failed = served.batch(&extended, false);
    assert_eq!(served.model.failed(), 1, "the {site:?} injection fired");
    for (i, f) in failed.iter().enumerate() {
        let why = f
            .as_ref()
            .expect_err("every row of the failed batch gets its error");
        assert!(why.contains("injected"), "row {i}: {why}");
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
    let reused = served.batch(&next, false);
    let reference = served.batch(&next, true);
    let warm = clean.batch(&next, false);
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
        "{arm}: the requests after the failed batch: {}",
        wrong.join("; ")
    );
    assert_eq!(
        served.recovery(),
        (1, 1),
        "(fences, aborts): only the failed batch recovered"
    );
    served.stop();
}

/// The failure lands in row 0's first prefill forward of batch 2, after it
/// advanced the conv phase and wrote KV; row 1 never ran.
#[test]
fn a_static_batch_failing_in_a_prefill_hands_its_rows_back_reset() {
    a_failed_batch_then_the_next::<RewindableExact>(
        "cpu-rewindable",
        Device::Cpu,
        config(),
        DType::F32,
        Site::Prefill,
        1,
    );
}

/// The failure lands in row 0's first verify completion of batch 2, after it
/// resolved row 0's window; row 1's verify is still enqueued, its cache
/// inside the window the verify opened.
#[test]
fn a_static_batch_failing_with_a_sibling_verify_pending_hands_its_rows_back_reset() {
    a_failed_batch_then_the_next::<RewindableExact>(
        "cpu-rewindable",
        Device::Cpu,
        config(),
        DType::F32,
        Site::VerifyComplete,
        1,
    );
}

#[cfg(feature = "metal")]
use batch_recovery::native_config;

#[cfg(feature = "metal")]
#[test]
fn natively_a_paged_static_batch_failing_in_a_prefill_hands_its_rows_back_reset() {
    a_failed_batch_then_the_next::<lloom::profile::PoolSpecExact>(
        "metal-private-paged",
        Device::new_metal(0).unwrap(),
        native_config(true),
        DType::F16,
        Site::Prefill,
        1,
    );
}

#[cfg(feature = "metal")]
#[test]
fn natively_a_paged_static_batch_failing_with_a_sibling_verify_pending_hands_its_rows_back_reset() {
    a_failed_batch_then_the_next::<lloom::profile::PoolSpecExact>(
        "metal-private-paged",
        Device::new_metal(0).unwrap(),
        native_config(true),
        DType::F16,
        Site::VerifyComplete,
        1,
    );
}

#[cfg(feature = "metal")]
#[test]
fn natively_a_rewindable_static_batch_failing_with_a_sibling_verify_pending_hands_its_rows_back_reset(
) {
    a_failed_batch_then_the_next::<RewindableExact>(
        "metal-rewindable",
        Device::new_metal(0).unwrap(),
        native_config(false),
        DType::F16,
        Site::VerifyComplete,
        1,
    );
}

/// Batch 1 caches both rows' prompts and replies. Batch 2 extends them past
/// the caches' `max_position_embeddings`: row 0's prefill is refused at the
/// ceiling before any state moved, reported untouched, and row 1 never
/// runs. Batch 3 extends batch 1's logs by a token. With every prefix kept,
/// each row is prepared as a scheduler that never saw batch 2 prepares it,
/// and decodes what that one decodes; no recovery runs (decision 105,
/// `batched-decode-design.md` §11.10).
fn a_batch_refused_at_its_ceiling_keeps_its_prefixes<P: Profile>(
    arm: &str,
    device: Device,
    cfg: Config,
    dtype: DType,
) {
    let ceiling = cfg.max_position_embeddings;
    let clean = Served::<P>::start(device.clone(), cfg.clone(), dtype);
    let served = Served::<P>::start(device, cfg, dtype);
    let first = [vec![1u32, 2, 3, 4], vec![5u32, 6, 7]];
    let replies = served.batch(&first, false);
    assert_eq!(
        clean.batch(&first, false),
        replies,
        "the clean run's batch 1"
    );
    let cached: Vec<Vec<u32>> = first
        .iter()
        .zip(&replies)
        .map(|(p, r)| {
            let mut c = p.clone();
            c.extend(r.as_ref().expect("batch 1 is served"));
            c
        })
        .collect();
    let too_long: Vec<Vec<u32>> = cached.iter().map(|c| padded(c, ceiling + 1)).collect();
    for (i, f) in served.batch(&too_long, false).iter().enumerate() {
        let why = f
            .as_ref()
            .expect_err("a prompt past the ceiling is refused");
        eprintln!("ARM {arm} row {i} refused: {why}");
    }
    assert_eq!(served.model.failed(), 0, "nothing was injected");

    let next: Vec<Vec<u32>> = cached
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut n = c.clone();
            n.push(20 + i as u32);
            n
        })
        .collect();
    let reused = round_reuse(&served.handle, &next, false, MAX);
    let warm = round_reuse(&clean.handle, &next, false, MAX);
    clean.stop();
    for (i, (got, want)) in reused.iter().zip(&warm).enumerate() {
        eprintln!("ARM {arm} row {i}: after {got:?} never-refused {want:?}");
        let got = got.as_ref().expect("the row after the refusal is served");
        let want = want.as_ref().expect("the never-refused row is served");
        assert_eq!(
            got, want,
            "{arm} row {i}: (tokens, reuse, kept positions) after the refused batch"
        );
        assert!(
            got.2 >= first[i].len(),
            "{arm} row {i}: the cached prefix was not kept: {got:?}"
        );
    }
    assert_eq!(
        served.recovery(),
        (0, 0),
        "(fences, aborts): a refusal that touched nothing ran the recovery"
    );
    served.stop();
}

/// Batch 2 extends batch 1's logs to four positions short of the ceiling:
/// the prefill fits, and a verify window a few rounds in does not. The
/// model refuses that window's pass before the pass moves anything, but the
/// window is open by then, and every lane's position is past its log by
/// the tokens it accepted. The driver reports that failure as touched, so
/// the batch is fenced and reset, and batch 3 decodes what a fresh sequence
/// decodes. Reported as the model reports it, the rows would come back with
/// the window open (`batched-decode-design.md` §11.10).
fn a_batch_refused_inside_a_verify_window_is_reset<P: Profile>(
    arm: &str,
    device: Device,
    cfg: Config,
    dtype: DType,
) {
    let ceiling = cfg.max_position_embeddings;
    let served = Served::<P>::start(device, cfg, dtype);
    let first = [vec![1u32, 2, 3, 4], vec![5u32, 6, 7]];
    let replies = served.batch(&first, false);
    let cached: Vec<Vec<u32>> = first
        .iter()
        .zip(&replies)
        .map(|(p, r)| {
            let mut c = p.clone();
            c.extend(r.as_ref().expect("batch 1 is served"));
            c
        })
        .collect();
    let near: Vec<Vec<u32>> = cached.iter().map(|c| padded(c, ceiling - 4)).collect();
    for (i, f) in served.batch(&near, false).iter().enumerate() {
        let why = f.as_ref().expect_err("a reply past the ceiling is refused");
        eprintln!("ARM {arm} row {i} refused: {why}");
    }
    assert_eq!(served.model.failed(), 0, "nothing was injected");

    let next: Vec<Vec<u32>> = cached
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let mut n = c.clone();
            n.push(20 + i as u32);
            n
        })
        .collect();
    let reused = served.batch(&next, false);
    let reference = served.batch(&next, true);
    for (i, (got, want)) in reused.iter().zip(&reference).enumerate() {
        eprintln!("ARM {arm} row {i}: after {got:?} cold {want:?}");
        assert_eq!(
            got.as_ref().ok(),
            Some(want.as_ref().expect("the cold reference is served")),
            "{arm} row {i}: the request after the refused verify window"
        );
    }
    assert_eq!(
        served.recovery(),
        (1, 1),
        "(fences, aborts): the refused window's batch recovered once"
    );
    served.stop();
}

#[test]
fn a_static_batch_refused_at_its_ceiling_keeps_its_prefixes() {
    a_batch_refused_at_its_ceiling_keeps_its_prefixes::<RewindableExact>(
        "cpu-rewindable",
        Device::Cpu,
        small_config(),
        DType::F32,
    );
}

#[test]
fn a_static_batch_refused_inside_a_verify_window_is_reset() {
    a_batch_refused_inside_a_verify_window_is_reset::<RewindableExact>(
        "cpu-rewindable",
        Device::Cpu,
        small_config(),
        DType::F32,
    );
}

/// The failure is the real model's: row 0's first prefill of batch 2 runs
/// at the end of the RoPE table, its first (conv) layer writes its state
/// and the second fails, reported touched.
#[test]
fn a_static_batch_failing_past_its_first_layer_hands_its_rows_back_reset() {
    a_failed_batch_then_the_next::<RewindableExact>(
        "cpu-rewindable",
        Device::Cpu,
        config(),
        DType::F32,
        Site::PrefillPastRope,
        1,
    );
}
