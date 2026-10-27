//! The batched loop's counter cut on a decode-only step
//! (`decode.batched.cut_hold`, [`BatchCutHold`]).
//!
//! Held, a decode-only step commits only the buffers it commits itself — the
//! step's readback (staged on the lookahead arm, waited on the blocking arm),
//! the forward's pacer marker and whatever the model reads back inside the
//! forward — however many dispatches the forward has, and the period it
//! found is back after it. `ahead` holds on the lookahead arm only. A step
//! that feeds prompt pieces keeps the period. Every arm decodes the same ids,
//! on LFM2 and on Qwen4-exp (resident, and streamed through a pool that
//! misses, whose host plan reads the routing back inside the forward), at
//! B = 2, 4 and 8.
//!
//! The device starts each run at a small period ([`PERIOD`]), so the tiny
//! forward cuts many times under it. The reference a held step is checked
//! against is absolute: two commits a step on LFM2 (no readback inside its
//! forward), and on Qwen4-exp the same run with the device's period at
//! `usize::MAX`, where nothing cuts anywhere.

use lloom::generation::Sampling;
use lloom::nn::model::BatchedCausalLM;
use lloom::nn::{BatchCutHold, BatchDecoder, DecodeOptions, Session};
use lloom_tensor::{Device, Result};

/// An id no argmax produces here: every run ends on its step count.
const NO_EOS: u32 = u32::MAX - 1;
/// The period each run starts the device at.
const PERIOD: usize = 4;
/// Tokens a row decodes.
const STEPS: usize = 8;

/// What one [`BatchDecoder::step`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Step {
    live: usize,
    pieces: usize,
    retired: usize,
    /// Command buffers committed during the call.
    commits: u64,
    /// The device's counter-cut period after it.
    period_after: usize,
}

impl Step {
    fn shape(&self) -> (usize, usize, usize) {
        (self.live, self.pieces, self.retired)
    }
}

#[derive(Debug)]
struct Run {
    ids: Vec<Vec<u32>>,
    steps: Vec<Step>,
}

impl Run {
    /// The decode-only steps whose window holds nothing but the step: every
    /// row live, no piece, nothing retired, and the step before the same (a
    /// join or a retirement leaves a gather in the open buffer that the next
    /// step's commits would carry).
    fn steady(&self, b: usize) -> Vec<usize> {
        let quiet = |s: &Step| s.live == b && s.pieces == 0 && s.retired == 0;
        (1..self.steps.len())
            .filter(|&i| quiet(&self.steps[i]) && quiet(&self.steps[i - 1]))
            .collect()
    }

    fn commits_at(&self, at: &[usize]) -> Vec<u64> {
        at.iter().map(|&i| self.steps[i].commits).collect()
    }
}

fn prompts(b: usize, vocab: usize, long: bool) -> Vec<Vec<u32>> {
    (0..b)
        .map(|r| {
            let len = if long { 3 + (r * 5) % 7 } else { 3 + r % 3 };
            (0..len)
                .map(|i| ((r * 7 + i * 3) % (vocab - 2) + 1) as u32)
                .collect()
        })
        .collect()
}

#[cfg(feature = "metal")]
fn epoch(dev: &Device) -> u64 {
    match dev {
        Device::Metal(m) => m.gpu_clock().current_epoch().0,
        _ => 0,
    }
}

#[cfg(not(feature = "metal"))]
fn epoch(_dev: &Device) -> u64 {
    0
}

#[cfg(feature = "metal")]
fn period_of(dev: &Device) -> usize {
    match dev {
        Device::Metal(m) => m.compute_per_buffer(),
        _ => 0,
    }
}

#[cfg(not(feature = "metal"))]
fn period_of(_dev: &Device) -> usize {
    0
}

/// One batch through `BatchDecoder` under `arm`, the device at `period`.
#[allow(clippy::too_many_arguments)]
fn run<M: BatchedCausalLM>(
    model: &M,
    make: &dyn Fn() -> Result<M::Cache>,
    dev: &Device,
    prompts: &[Vec<u32>],
    lookahead: usize,
    chunk: usize,
    arm: BatchCutHold,
    period: usize,
) -> Result<Run> {
    let _period = dev.cut_period_hold(period);
    let opts = DecodeOptions {
        lookahead,
        prefill_chunk: chunk,
        ..DecodeOptions::new(STEPS)
    };
    let mut dec: BatchDecoder<'_, M, Session<'_, M>> =
        BatchDecoder::new(opts, &[NO_EOS], dev.clone())?.with_cut_hold(arm);
    assert_eq!(dec.cut_hold(), arm);
    let mut rows = Vec::with_capacity(prompts.len());
    for p in prompts {
        let s = Session::new(model, make()?);
        rows.push(
            dec.admit(s, p.clone(), Sampling::ArgMax, 0)
                .map_err(|r| r.error)?,
        );
    }
    let mut ids = vec![Vec::new(); prompts.len()];
    let mut steps = Vec::new();
    while !dec.is_idle() {
        let e0 = epoch(dev);
        let rep = dec.step(&mut |row, t| {
            let k = rows
                .iter()
                .position(|r| *r == row)
                .expect("an admitted row");
            ids[k].extend_from_slice(t);
            Ok(())
        })?;
        steps.push(Step {
            live: rep.live,
            pieces: rep.pieces,
            retired: rep.retired,
            commits: epoch(dev) - e0,
            period_after: period_of(dev),
        });
        assert!(steps.len() < 200, "the batch never drained");
    }
    let finished = dec.take_finished();
    assert_eq!(finished.len(), prompts.len());
    for row in &ids {
        assert_eq!(row.len(), STEPS, "a row ended before its cap");
    }
    Ok(Run { ids, steps })
}

#[test]
fn the_arm_names_parse_and_the_default_is_off() -> Result<()> {
    for arm in [BatchCutHold::Off, BatchCutHold::Ahead, BatchCutHold::All] {
        assert_eq!(BatchCutHold::parse(arm.name()), Ok(arm));
        assert_eq!(arm.to_string(), arm.name());
    }
    assert!(BatchCutHold::parse("on").is_err());
    assert_eq!(BatchCutHold::default(), BatchCutHold::Off);
    assert_eq!(
        lloom::tuning::tuning().ring.batched_cut_hold,
        "off",
        "the arm's library default"
    );
    Ok(())
}

#[cfg(feature = "metal")]
mod metal {
    use super::*;
    use lloom::models::lfm2::{Config as LfmConfig, State};
    use lloom::models::lfm2_testing::Lfm2Tiny;
    use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
    use lloom::models::qwen4exp::{
        Config as QwenConfig, LoadOptions, SlotBudget, StreamLoader, StreamOptions,
    };
    use lloom::nn::model::CausalLM;
    use lloom::profile::CrownExact;
    use lloom::testing::fixture::{Backend, Fixture};
    use lloom::testing::tiny::TinyFamily;
    use lloom_tensor::DType;

    const ARMS: [BatchCutHold; 3] = [BatchCutHold::Off, BatchCutHold::Ahead, BatchCutHold::All];

    fn metal() -> Option<Device> {
        if Backend::current().at_scale() {
            println!("at scale: the tiny model's commit counts are the tiny model's");
            return None;
        }
        Device::new_metal(0).ok()
    }

    /// Every arm decodes the reference's ids, step for step in the same
    /// shape, and leaves the period where it found it after every step.
    fn same_run(tag: &str, reference: &Run, run: &Run) {
        assert_eq!(run.ids, reference.ids, "{tag}: the ids moved");
        let shape = |r: &Run| r.steps.iter().map(Step::shape).collect::<Vec<_>>();
        assert_eq!(shape(run), shape(reference), "{tag}: the steps moved");
        for (i, s) in run.steps.iter().enumerate() {
            assert_eq!(
                s.period_after, PERIOD,
                "{tag}: step {i} left the period at {}, not the {PERIOD} it found",
                s.period_after
            );
        }
    }

    /// LFM2, one sequence a row: held, a steady step commits the step's
    /// readback and the forward's marker and nothing else.
    #[test]
    fn a_held_lfm2_decode_step_commits_only_its_readback_and_its_forward() -> Result<()> {
        let Some(dev) = metal() else { return Ok(()) };
        let Some(fx) = Fixture::<Lfm2Tiny>::new(
            "batched_cut_hold_lfm2",
            <Lfm2Tiny as TinyFamily>::template(),
        )?
        else {
            return Ok(());
        };
        let cfg: LfmConfig = fx.config(|_| {});
        let model = fx.model_in::<CrownExact>(&cfg, Some(DType::F32), &dev)?;
        let make = || State::<CrownExact>::new(true, DType::F32, &cfg, &dev);
        for b in [2usize, 4, 8] {
            let ps = prompts(b, cfg.vocab_size, false);
            for lookahead in [1usize, 0] {
                let reference = run(
                    &model,
                    &make,
                    &dev,
                    &ps,
                    lookahead,
                    0,
                    BatchCutHold::Off,
                    PERIOD,
                )?;
                let steady = reference.steady(b);
                assert!(
                    steady.len() >= 3,
                    "b={b} la={lookahead}: {} steady steps, the fixture needs more",
                    steady.len()
                );
                let off = reference.commits_at(&steady);
                println!("CUT_HOLD lfm2 b={b} la={lookahead} off={off:?}");
                assert!(
                    off.iter().all(|&c| c > 2),
                    "b={b} la={lookahead}: with nothing held a forward of the tiny model cuts at \
                     period {PERIOD}, so a step commits more than its own two buffers: {off:?}"
                );
                for arm in [BatchCutHold::Ahead, BatchCutHold::All] {
                    let r = run(&model, &make, &dev, &ps, lookahead, 0, arm, PERIOD)?;
                    let tag = format!("lfm2 b={b} la={lookahead} {arm}");
                    same_run(&tag, &reference, &r);
                    let got = r.commits_at(&steady);
                    println!("CUT_HOLD {tag} commits={got:?}");
                    let holds = arm == BatchCutHold::All || lookahead >= 1;
                    if holds {
                        assert!(
                            got.iter().all(|&c| c == 2),
                            "{tag}: a held step commits its readback and its forward's marker, \
                             two buffers, whatever the forward's dispatch count: {got:?}"
                        );
                    } else {
                        assert_eq!(
                            got, off,
                            "{tag}: `ahead` holds nothing on the blocking arm, so its steps \
                             cut where the unheld ones do"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    /// A step that carries prompt pieces beside live rows keeps the period:
    /// under `all` its commits are the unheld step's.
    #[test]
    fn a_step_that_feeds_pieces_keeps_the_period() -> Result<()> {
        let Some(dev) = metal() else { return Ok(()) };
        let Some(fx) = Fixture::<Lfm2Tiny>::new(
            "batched_cut_hold_pieces",
            <Lfm2Tiny as TinyFamily>::template(),
        )?
        else {
            return Ok(());
        };
        let cfg: LfmConfig = fx.config(|_| {});
        let model = fx.model_in::<CrownExact>(&cfg, Some(DType::F32), &dev)?;
        let make = || State::<CrownExact>::new(true, DType::F32, &cfg, &dev);
        let b = 4;
        let ps = prompts(b, cfg.vocab_size, true);
        let off = run(&model, &make, &dev, &ps, 1, 2, BatchCutHold::Off, PERIOD)?;
        let all = run(&model, &make, &dev, &ps, 1, 2, BatchCutHold::All, PERIOD)?;
        same_run("lfm2 pieces all", &off, &all);
        let riding: Vec<usize> = (0..off.steps.len())
            .filter(|&i| off.steps[i].pieces > 0 && off.steps[i].live > 0)
            .collect();
        assert!(
            !riding.is_empty(),
            "the fixture must ride pieces beside live rows"
        );
        let (o, a) = (off.commits_at(&riding), all.commits_at(&riding));
        println!("CUT_HOLD lfm2 pieces riding_steps={riding:?} off={o:?} all={a:?}");
        assert_eq!(a, o, "a step that feeds pieces is cut as it was");
        assert!(
            o.iter().any(|&c| c > 2),
            "the riding steps must cut at period {PERIOD}, or the comparison sees nothing: {o:?}"
        );
        Ok(())
    }

    fn stream(slots: usize) -> StreamOptions {
        StreamOptions {
            slots: SlotBudget::Fixed(slots),
            loader: StreamLoader::Pread,
            prefetch: false,
            io_hints: false,
            drafter_bytes: 0,
        }
    }

    fn qwen_fixture(name: &str) -> Result<Option<(Fixture<Qwen4ExpTiny>, Device)>> {
        let Some(dev) = metal() else { return Ok(None) };
        Ok(Fixture::<Qwen4ExpTiny>::new(name, QwenConfig::tiny())?.map(|fx| (fx, dev)))
    }

    /// One Qwen4-exp model per run, built by `model` (a streamed tier's pool
    /// state is the run's own), every arm against the run where nothing cuts.
    fn qwen_arms<M: BatchedCausalLM>(
        tag: &str,
        model: &dyn Fn() -> Result<M>,
        cfg: &QwenConfig,
        dev: &Device,
    ) -> Result<()>
    where
        M: CausalLM<Config = QwenConfig>,
    {
        for b in [2usize, 4, 8] {
            let ps = prompts(b, cfg.vocab_size, false);
            for lookahead in [0usize, 1] {
                let one = |arm: BatchCutHold, period: usize| -> Result<Run> {
                    let m = model()?;
                    let make = || m.new_cache(cfg, dev);
                    run(&m, &make, dev, &ps, lookahead, 0, arm, period)
                };
                let uncut = one(BatchCutHold::Off, usize::MAX)?;
                let off = one(BatchCutHold::Off, PERIOD)?;
                assert_eq!(
                    off.ids, uncut.ids,
                    "{tag} b={b} la={lookahead}: the period moved the ids"
                );
                let steady = off.steady(b);
                assert!(
                    steady.len() >= 3,
                    "{tag} b={b} la={lookahead}: {} steady steps",
                    steady.len()
                );
                let (u, o) = (uncut.commits_at(&steady), off.commits_at(&steady));
                println!("CUT_HOLD {tag} b={b} la={lookahead} uncut={u:?} off={o:?}");
                assert!(
                    o.iter().zip(&u).all(|(o, u)| o > u),
                    "{tag} b={b} la={lookahead}: at period {PERIOD} the forward must cut, or a \
                     held step reads like an unheld one: off {o:?} against uncut {u:?}"
                );
                for arm in [BatchCutHold::Ahead, BatchCutHold::All] {
                    let r = one(arm, PERIOD)?;
                    let t = format!("{tag} b={b} la={lookahead} {arm}");
                    same_run(&t, &off, &r);
                    let got = r.commits_at(&steady);
                    println!("CUT_HOLD {t} commits={got:?}");
                    if arm == BatchCutHold::All || lookahead >= 1 {
                        assert_eq!(
                            got, u,
                            "{t}: a held step commits what the step itself commits — the run \
                             where nothing cuts"
                        );
                    } else {
                        assert_eq!(got, o, "{t}: `ahead` holds nothing on the blocking arm");
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn a_held_qwen4exp_decode_step_commits_only_what_the_step_commits() -> Result<()> {
        let Some((fx, dev)) = qwen_fixture("batched_cut_hold_qwen")? else {
            return Ok(());
        };
        let cfg = fx.config(|_| {});
        let opts = LoadOptions::default();
        qwen_arms(
            "qwen resident",
            &|| fx.resident::<CrownExact>(&cfg, &dev, &opts),
            &cfg,
            &dev,
        )
    }

    /// The streamed tier through a pool that misses: the host plan reads
    /// every layer's routing back inside the forward, and lands its misses
    /// with copy dispatches.
    #[test]
    fn a_held_streamed_qwen4exp_step_keeps_its_readbacks_and_its_ids() -> Result<()> {
        let Some((fx, dev)) = qwen_fixture("batched_cut_hold_qwen_streamed")? else {
            return Ok(());
        };
        let cfg = fx.config(|_| {});
        let opts = LoadOptions::default();
        qwen_arms(
            "qwen streamed",
            &|| fx.streamed::<CrownExact>(&cfg, &dev, &opts, &stream(6)),
            &cfg,
            &dev,
        )
    }
}
