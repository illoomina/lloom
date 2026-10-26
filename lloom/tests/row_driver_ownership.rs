//! Who owns what across a row driver's admission, retirement and drain
//! (`RowDriver`, `batched-decode-design.md` §12.4).
//!
//! The speculative driver gives every live row one of the drafter's
//! sequence slots. Each slot must go back to the free list exactly once
//! whatever order a caller retires, cancels, takes and drains rows in: a
//! slot returned twice is later handed to two live rows, whose drafter
//! contexts then alias. The schedules below run on the CPU over a tiny LFM2
//! and a drafter that records the slots it is asked to draft for, so the
//! aliasing is read where it happens (two rows drafting through one slot)
//! and the conservation as a count (how many rows the driver admits before
//! it refuses for want of a slot).
//!
//! The drafter proposes garbage: greedy verification accepts only the
//! target's own argmax, so the rows' streams do not depend on it and every
//! schedule here is about ownership, not output.

use lloom::generation::Sampling;
use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::speculator::{Draft, LayerTaps, SeqId, Speculator, TargetProbe};
use lloom::nn::{DecodeOptions, RowDriver, RowId, Session, SpecBatchDecoder};
use lloom::profile::RewindableExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{DType, Device, Result};
use parking_lot::Mutex;
use std::sync::Arc;

const K: usize = 2;
const EOS: u32 = 47;
const PROMPT: [u32; 4] = [1, 2, 3, 4];

/// Six layers with a five-layer tap and the rewindable arms: the shape the
/// speculative driver needs from its target (`Probes<LayerTaps<5>>`).
fn config() -> Config {
    Config {
        vocab_size: 48,
        hidden_size: 8,
        intermediate_size: 16,
        num_hidden_layers: 6,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
            LayerType::Conv,
            LayerType::FullAttention,
        ],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: Some(EOS),
        kv_append: KvAppend::InPlace,
        conv_state: ConvState::RotatingRing { k: K + 1 },
        tap_layers: vec![1, 2, 3, 4, 5],
        tap_dtype: DType::F32,
        ..Default::default()
    }
}

/// What the drafter was asked to do: the slots of every `draft` call, and
/// a switch that makes the next `draft` fail (a drafter error mid-step).
#[derive(Default)]
struct Record {
    drafts: Vec<Vec<usize>>,
    fail_next: bool,
}

/// `n` sequence slots; every call recorded; garbage drafts.
struct SlotDrafter {
    n: usize,
    record: Arc<Mutex<Record>>,
}

impl Speculator for SlotDrafter {
    type Probe = LayerTaps<5>;
    type Family = lloom::nn::speculator::DSpark;

    fn n_seq(&self) -> usize {
        self.n
    }

    fn max_draft(&self) -> usize {
        K
    }

    fn begin(&mut self, _seq: SeqId, _prompt: &[u32]) -> Result<()> {
        Ok(())
    }

    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        let mut r = self.record.lock();
        r.drafts.push(active.iter().map(|s| s.0).collect());
        if std::mem::take(&mut r.fail_next) {
            lloom_tensor::bail!("test: the drafter failed mid-step");
        }
        Ok(active.iter().map(|_| Draft::host(vec![0; K])).collect())
    }

    fn observe(
        &mut self,
        _seq: SeqId,
        _probe: &<Self::Probe as TargetProbe>::Output,
        _accepted: &[u32],
    ) -> Result<()> {
        Ok(())
    }
}

type Target = Model<RewindableExact>;
type Row<'m> = Session<'m, Target>;
type Driver<'m> = SpecBatchDecoder<'m, Target, SlotDrafter, Row<'m>>;

struct Fixture {
    cfg: Config,
    model: Target,
    dev: Device,
}

fn fixture() -> Result<Fixture> {
    let dev = Device::Cpu;
    let cfg = config();
    let model = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F32)
        .model::<RewindableExact>(&dev)?;
    Ok(Fixture { cfg, model, dev })
}

impl Fixture {
    fn session(&self) -> Result<Row<'_>> {
        Ok(Session::new(
            &self.model,
            State::<RewindableExact>::new(true, DType::F32, &self.cfg, &self.dev)?,
        ))
    }

    fn driver(&self, n_seq: usize) -> Result<(Driver<'_>, Arc<Mutex<Record>>)> {
        let record = Arc::new(Mutex::new(Record::default()));
        let spec = SlotDrafter {
            n: n_seq,
            record: Arc::clone(&record),
        };
        let driver = SpecBatchDecoder::new(
            lloom::profile::RewindProof::of(),
            spec,
            DecodeOptions::new(64),
            self.dev.clone(),
        )?;
        Ok((driver, record))
    }
}

/// Admit one greedy row of `PROMPT` capped at `cap`.
fn admit<'m>(driver: &mut Driver<'m>, session: Row<'m>, cap: usize) -> Result<RowId> {
    driver
        .admit_with_eos(session, PROMPT.to_vec(), Sampling::ArgMax, 0, &[EOS], cap)
        .map_err(|e| lloom_tensor::Error::msg(e.to_string()))
}

/// Every draft call named each slot at most once, and only slots the
/// drafter has.
fn assert_drafts_unique(record: &Mutex<Record>, n_seq: usize, schedule: &str) {
    for (i, call) in record.lock().drafts.iter().enumerate() {
        let mut seen = call.clone();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(
            seen.len(),
            call.len(),
            "{schedule}: draft call {i} named one drafter slot for two live rows: {call:?}"
        );
        assert!(
            call.iter().all(|&s| s < n_seq),
            "{schedule}: draft call {i} named a slot the drafter does not have: {call:?}"
        );
    }
}

/// How many rows the driver admits now before it refuses: with every row
/// out of it, its drafter's slot count. The rows are drained back after.
fn capacity<'m>(f: &'m Fixture, driver: &mut Driver<'m>, n_seq: usize) -> Result<usize> {
    let mut admitted = 0usize;
    // One past the slots: a free list holding a slot twice admits it.
    for _ in 0..=n_seq {
        let s = f.session()?;
        match driver.admit_with_eos(s, PROMPT.to_vec(), Sampling::ArgMax, 0, &[EOS], 4) {
            Ok(_) => admitted += 1,
            Err(e) => {
                let why = e.to_string();
                assert!(
                    why.contains("every drafter slot is taken"),
                    "refused for another reason: {why}"
                );
                break;
            }
        }
    }
    driver.drain_rows();
    Ok(admitted)
}

/// Step until every row is retired and taken, at most `max` steps.
fn run_out(driver: &mut Driver<'_>, max: usize) -> Result<usize> {
    let mut taken = 0;
    for _ in 0..max {
        if driver.is_idle() {
            break;
        }
        driver.step(&mut |_, _| Ok(()))?;
        taken += driver.take_finished().len();
    }
    Ok(taken)
}

/// A row cancelled and retired by a step, then drained before
/// `take_finished`: its drafter slot went back at the retirement, and the
/// drain must not hand it back a second time. Refilled to the drafter's
/// width, no two rows draft through one slot and one more row is refused.
#[test]
fn retire_then_drain_returns_the_drafter_slot_once() -> Result<()> {
    const N: usize = 2;
    let f = fixture()?;
    let (mut driver, record) = f.driver(N)?;
    let a = admit(&mut driver, f.session()?, 8)?;
    assert!(driver.cancel(a));
    let report = driver.step(&mut |_, _| Ok(()))?;
    assert_eq!(report.retired, 1, "the cancelled row retires at the step");
    let back = driver.drain_rows();
    assert_eq!(
        back.len(),
        1,
        "the drain hands the retired row's session back"
    );
    // Refill to the width and step: every live row drafts through its own slot.
    admit(&mut driver, f.session()?, 3)?;
    admit(&mut driver, f.session()?, 3)?;
    let third = driver.admit_with_eos(
        f.session()?,
        PROMPT.to_vec(),
        Sampling::ArgMax,
        0,
        &[EOS],
        3,
    );
    // Asserted BEFORE the rows are stepped, so this test stands on its own
    // absolute reference rather than on `release_drafter_slot`'s
    // `debug_assert` — which fires during the stepping, and which a release
    // build compiles out entirely (CONTRIBUTING §3.1b: make the coverage
    // intrinsic to the test that claims it).
    assert!(
        third.is_err(),
        "a third row was admitted to a two-slot drafter: the drain returned \
         the retired row's slot again"
    );
    run_out(&mut driver, 16)?;
    assert_drafts_unique(&record, N, "retire then drain");
    assert_eq!(capacity(&f, &mut driver, N)?, N, "slots conserved");
    Ok(())
}

/// The other orders a caller can end rows in, each then refilled: every
/// slot comes back exactly once and no two live rows share one.
///
/// - `take then drain`: the scheduler's success path, `take_finished`
///   before anything else.
/// - `cancel then drain`: a live row drained without a retiring step.
/// - `failure then drain`: a step that fails mid-round (the drafter
///   errors), its rows drained as `BatchScheduler::run` does.
#[test]
fn every_end_of_a_row_conserves_the_drafter_slots() -> Result<()> {
    const N: usize = 2;
    let f = fixture()?;
    for schedule in ["take then drain", "cancel then drain", "failure then drain"] {
        let (mut driver, record) = f.driver(N)?;
        let a = admit(&mut driver, f.session()?, 3)?;
        admit(&mut driver, f.session()?, 3)?;
        match schedule {
            "take then drain" => {
                let taken = run_out(&mut driver, 16)?;
                assert_eq!(taken, 2, "{schedule}: both rows finish");
                assert!(driver.drain_rows().is_empty());
            }
            "cancel then drain" => {
                driver.step(&mut |_, _| Ok(()))?;
                assert!(driver.cancel(a));
                assert_eq!(driver.drain_rows().len(), 2, "{schedule}");
            }
            "failure then drain" => {
                record.lock().fail_next = true;
                let failed = driver.step(&mut |_, _| Ok(()));
                assert!(
                    failed.is_err(),
                    "{schedule}: the injected failure fails the step"
                );
                assert_eq!(driver.drain_rows().len(), 2, "{schedule}");
            }
            _ => unreachable!(),
        }
        admit(&mut driver, f.session()?, 3)?;
        admit(&mut driver, f.session()?, 3)?;
        run_out(&mut driver, 16)?;
        assert_drafts_unique(&record, N, schedule);
        assert_eq!(
            capacity(&f, &mut driver, N)?,
            N,
            "{schedule}: slots conserved"
        );
    }
    Ok(())
}
