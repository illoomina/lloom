//! `Prepared::path` (`sequence-reuse-design.md` §2.3f): the one outcome
//! `Session::prepare` reports of what it found and did, so a turn report
//! reads it instead of re-deriving `prepare`'s predicates. Each case drives
//! the tiny Qwen3.5 into one situation and pins the path against what the
//! case set up: the log's length, the overrun the run left, where the
//! request leaves the log, the checkpoint the run left. The engine's
//! `turn::prepare` classification of the same cases is `turn_reuse.rs`'s.

use lloom::generation::poll::PollMechanism;
use lloom::generation::{Blocking, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::nn::decoder::{DecodeRecovery, StateCapacity};
use lloom::nn::model::CausalLM;
use lloom::nn::{
    DecodeOptions, DivergenceBy, IdentityProposer, OverrunBy, PreparePath, ReuseAction, Session,
};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{Device, Result};

type Model = Qwen3_5<CrownExact>;

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

fn model(dev: &Device, standing: bool, slots: usize) -> Result<(Model, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<CrownExact>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        recovery: if standing {
            DecodeRecovery::Standing
        } else {
            DecodeRecovery::None
        },
        checkpoints: slots,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn ids(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| 2 + (i * 7 + salt) % 60).collect()
}

fn greedy(s: &mut Session<'_, Model>, prompt: &[u32], n: usize, dev: &Device) -> Result<Vec<u32>> {
    let out = s.decode(
        &mut IdentityProposer,
        &mut Blocking::new(&[u32::MAX], dev)?,
        &mut LogitsProcessor::from_sampling(0, Sampling::ArgMax),
        prompt,
        &DecodeOptions::new(n),
        &mut |_: &[u32]| Ok(()),
    )?;
    Ok(out.ids)
}

/// A run on the fixed-cadence ring stopping on the stream's first fresh id
/// from the sixth on: one forward past it (the overrun). Returns the ids.
fn overrun_run(
    m: &Model,
    cfg: &Config,
    s: &mut Session<'_, Model>,
    prompt: &[u32],
    dev: &Device,
) -> Result<Vec<u32>> {
    let stream = {
        let mut fresh = Session::new(m, m.new_cache(cfg, dev)?);
        greedy(&mut fresh, prompt, 24, dev)?
    };
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let eos = vec![stream[cut]];
    let mut rb = Ring::new(&eos, dev, 128, EosRead::Poll(PollMechanism::default()))?
        .with_poll_budget(20_000);
    let out = s.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut LogitsProcessor::from_sampling(0, Sampling::ArgMax),
        prompt,
        &DecodeOptions {
            lookahead: 1,
            ..DecodeOptions::new(24)
        },
        &mut |_: &[u32]| Ok(()),
    )?;
    assert!(
        out.eos && out.stats.overrun >= 1,
        "the ring overran its EOS"
    );
    Ok(out.ids)
}

fn path(
    boundary: usize,
    overrun: usize,
    overrun_by: OverrunBy,
    diverged_at: Option<usize>,
    divergence_by: DivergenceBy,
) -> PreparePath {
    PreparePath {
        boundary,
        overrun,
        overrun_by,
        diverged_at,
        divergence_by,
    }
}

/// The cases with no overrun: cold, extend, resume, the whole log again
/// (a refeed through the window), a divergence the window takes back, one
/// a checkpoint does, one nothing reaches, and no overlap.
#[test]
fn each_preparation_reports_the_path_it_took() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let (m, cfg) = model(&dev, true, 2)?;
    let p = ids(20, 1);

    let mut s = Session::new(&m, m.new_cache(&cfg, &dev)?);
    let got = s.prepare(&p)?;
    assert_eq!(got.action, ReuseAction::Fresh);
    assert_eq!(
        got.path,
        path(0, 0, OverrunBy::None, None, DivergenceBy::None),
        "cold"
    );
    let reply = greedy(&mut s, &got.suffix, 4, &dev)?;
    let log = [p.clone(), reply.clone()].concat();
    assert_eq!(s.tokens(), log.as_slice());

    // The whole log with its held logits: resume; without them (resume
    // off): the last id fed again, through the window.
    let got = s.prepare(&log)?;
    assert_eq!(got.action, ReuseAction::Resume);
    assert_eq!(
        got.path,
        path(log.len(), 0, OverrunBy::None, None, DivergenceBy::None),
        "resume"
    );
    let got = s.prepare_with(&log, false)?;
    assert_eq!(got.action, ReuseAction::Rewound(1));
    assert_eq!(
        got.path,
        path(log.len(), 0, OverrunBy::None, None, DivergenceBy::Window(1)),
        "refeed"
    );
    greedy(&mut s, &got.suffix, 4, &dev)?;
    let log = s.tokens().to_vec();

    // Extend.
    let full = [log.clone(), ids(5, 2)].concat();
    let got = s.prepare(&full)?;
    assert_eq!(got.action, ReuseAction::Extend);
    assert_eq!(
        got.path,
        path(log.len(), 0, OverrunBy::None, None, DivergenceBy::None),
        "extend"
    );
    greedy(&mut s, &got.suffix, 4, &dev)?;
    let log = s.tokens().to_vec();

    // The last id replaced: the window takes it back.
    let mut one = log.clone();
    *one.last_mut().unwrap() ^= 1;
    let got = s.prepare(&one)?;
    assert_eq!(got.action, ReuseAction::Rewound(1));
    assert_eq!(
        got.path,
        path(
            log.len(),
            0,
            OverrunBy::None,
            Some(log.len() - 1),
            DivergenceBy::Window(1)
        ),
        "a divergence inside the window"
    );
    greedy(&mut s, &got.suffix, 4, &dev)?;
    let log = s.tokens().to_vec();

    // The last reply re-rendered away: the turn checkpoint at the previous
    // prompt's end (the pass's own, the ring's default).
    let at = full.len();
    let rerender = [full.clone(), ids(3, 3)].concat();
    let got = s.prepare(&rerender)?;
    assert_eq!(got.action, ReuseAction::Rewound(log.len() - at));
    assert_eq!(
        got.path,
        path(
            log.len(),
            0,
            OverrunBy::None,
            Some(at),
            DivergenceBy::Checkpoint(at)
        ),
        "a divergence a checkpoint reaches"
    );

    // No overlap.
    let other: Vec<u32> = ids(12, 40).iter().map(|t| t + 1).collect();
    let got = s.prepare(&other)?;
    assert_eq!(got.action, ReuseAction::Fresh);
    assert_eq!(got.path.diverged_at, Some(0), "no overlap: {:?}", got.path);
    assert_eq!(got.path.divergence_by, DivergenceBy::Reset);

    // No checkpoint slots: a divergence past the window resets.
    let (m0, cfg0) = model(&dev, true, 0)?;
    let mut s = Session::new(&m0, m0.new_cache(&cfg0, &dev)?);
    greedy(&mut s, &p, 6, &dev)?;
    let log = s.tokens().to_vec();
    let got = s.prepare(&[p.clone(), ids(4, 9)].concat())?;
    assert_eq!(got.action, ReuseAction::Fresh);
    assert_eq!(
        got.path,
        path(
            log.len(),
            0,
            OverrunBy::None,
            Some(p.len()),
            DivergenceBy::Reset
        ),
        "past the window, no checkpoint"
    );
    Ok(())
}

/// An overrun on the fixed-cadence ring (one forward past the EOS): the
/// standing window undoes it; without the window a checkpoint restores; with
/// neither, a reset.
#[test]
fn an_overrun_reports_how_it_was_taken_back() -> Result<()> {
    pin();
    let dev = Device::Cpu;
    let p = ids(20, 5);
    for (standing, slots, want) in [
        (true, 2, OverrunBy::Window),
        (false, 2, OverrunBy::Checkpoint(20)),
        (false, 0, OverrunBy::Reset),
    ] {
        let (m, cfg) = model(&dev, standing, slots)?;
        let mut s = Session::new(&m, m.new_cache(&cfg, &dev)?);
        let reply = overrun_run(&m, &cfg, &mut s, &p, &dev)?;
        let log = s.tokens().len();
        let over = s.overrun();
        let full = [p.clone(), reply, ids(3, 6)].concat();
        let got = s.prepare(&full)?;
        eprintln!(
            "PATH standing {standing} slots {slots}: {:?} {:?}",
            got.action, got.path
        );
        assert_eq!(got.path.boundary, log);
        assert_eq!(got.path.overrun, over);
        assert_eq!(
            got.path.overrun_by, want,
            "standing {standing}, slots {slots}"
        );
        assert_eq!(
            got.path.diverged_at, None,
            "the request extends the log: the recovery's drop is not a divergence"
        );
        match want {
            OverrunBy::Window => assert_eq!(got.action, ReuseAction::Extend),
            OverrunBy::Checkpoint(at) => assert_eq!(got.action, ReuseAction::Rewound(log - at)),
            _ => assert_eq!(got.action, ReuseAction::Fresh),
        }
    }
    Ok(())
}

/// Under the EOS gate the forwards past the EOS wrote nothing the state
/// keeps: `prepare` takes the overrun back by its counters, and says so.
#[cfg(feature = "metal")]
#[test]
fn a_gated_overrun_reports_the_gate() -> Result<()> {
    pin();
    let dev = Device::new_metal(0)?;
    let (m, cfg) = model(&dev, true, 2)?;
    let p = ids(20, 5);
    let stream = {
        let mut fresh = Session::new(&m, m.new_cache(&cfg, &dev)?);
        greedy(&mut fresh, &p, 24, &dev)?
    };
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let eos = vec![stream[cut]];
    let mut s = Session::new(&m, m.new_cache(&cfg, &dev)?);
    let mut rb = Ring::new(
        &eos,
        &dev,
        lloom::generation::readback::RING_HISTORY_SLOTS,
        EosRead::Poll(PollMechanism::default()),
    )?;
    // No reader: `max_tokens` stops the run, 12 forwards past the EOS.
    let out = s.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut LogitsProcessor::from_sampling(0, Sampling::ArgMax),
        &p,
        &DecodeOptions {
            lookahead: 1,
            stream_via_reader: true,
            adaptive_commit: true,
            run_ahead: 8,
            eos_gate: true,
            ..DecodeOptions::new(cut + 13)
        },
        &mut |_: &[u32]| Ok(()),
    )?;
    assert!(
        out.stats.overrun > 1 && out.stats.overrun_unwritten,
        "{:?}",
        out.stats.overrun
    );
    let log = s.tokens().len();
    let full = [p.clone(), out.ids.clone(), ids(3, 6)].concat();
    let got = s.prepare(&full)?;
    eprintln!("PATH gated: {:?} {:?}", got.action, got.path);
    assert_eq!(got.action, ReuseAction::Extend);
    assert_eq!(
        got.path,
        path(
            log,
            out.stats.overrun,
            OverrunBy::Gate,
            None,
            DivergenceBy::None
        )
    );
    Ok(())
}
