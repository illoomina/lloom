//! The session reconciled to the accepted conversational frontier
//! (`Session::accept`, `sequence-reuse-design.md` §2.3f, decision 42(a)):
//! a front end accepts fewer ids than the run produced (a stop string, a
//! cancellation), and the session takes the rest back where it can (the
//! rewind horizon, the gate's counters) and keeps an explicit obligation
//! where it cannot (the next `prepare` restores a checkpoint and replays).
//!
//! The reference is the run that produced exactly the accepted ids (the
//! blocking arm at a `max_tokens` of the accepted count), extended by the
//! same next request: the exact case's continuation is its own bit for bit,
//! and the obligation's replays to it (the tiny model's greedy margins are
//! wide, `sequence-reuse-design.md` §2.3b's contract table).

use lloom::generation::poll::PollMechanism;
use lloom::generation::{Blocking, EosRead, LogitsProcessor, Ring, Sampling};
use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::nn::decoder::{DecodeRecovery, StateCapacity};
use lloom::nn::model::CausalLM;
use lloom::nn::{
    Acceptance, DecodeOptions, DivergenceBy, IdentityProposer, OverrunBy, ReuseAction, Session,
};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{Device, Result};

type Model = Qwen3_5<CrownExact>;

fn pin() {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
}

fn model(dev: &Device) -> Result<(Model, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<CrownExact>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        recovery: DecodeRecovery::Standing,
        checkpoints: 2,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn ids(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| 2 + (i * 7 + salt) % 60).collect()
}

/// A blocking greedy run of at most `max` ids whose consumer stops it after
/// `stop_after` ids (a host stop: the device never sees it).
fn run(
    s: &mut Session<'_, Model>,
    prompt: &[u32],
    max: usize,
    stop_after: Option<usize>,
    dev: &Device,
) -> Result<Vec<u32>> {
    let mut seen = 0usize;
    let out = s.decode_until(
        &mut IdentityProposer,
        &mut Blocking::new(&[u32::MAX], dev)?,
        &mut LogitsProcessor::from_sampling(0, Sampling::ArgMax),
        prompt,
        &DecodeOptions::new(max),
        &mut |batch: &[u32]| {
            seen += batch.len();
            Ok(stop_after.is_none_or(|k| seen < k))
        },
    )?;
    Ok(out.ids)
}

/// The run's ids past the boundary, `past` of them, then the next request
/// (the accepted history and a new turn) against the reference's.
fn case(
    dev: &Device,
    past: usize,
) -> Result<(Acceptance, ReuseAction, DivergenceBy, OverrunBy, bool)> {
    let (m, cfg) = model(dev)?;
    let p = ids(20, 1);
    let produced = 6;
    let accepted = produced - past;
    let new = ids(3, 9);

    let mut reference = Session::new(&m, m.new_cache(&cfg, dev)?);
    let acc = run(&mut reference, &p, accepted, None, dev)?;
    let full = [p.clone(), acc.clone(), new.clone()].concat();
    let r = reference.prepare(&full)?;
    assert_eq!(r.action, ReuseAction::Extend);
    let want = run(&mut reference, &r.suffix, 4, None, dev)?;

    let mut s = Session::new(&m, m.new_cache(&cfg, dev)?);
    let got_ids = run(&mut s, &p, 32, Some(produced), dev)?;
    assert_eq!(
        got_ids.len(),
        produced,
        "the host stop after {produced} ids"
    );
    assert_eq!(&got_ids[..accepted], acc.as_slice(), "the same stream");
    let a = s.accept(p.len() + accepted)?;
    assert_eq!(
        s.tokens().len(),
        p.len() + accepted,
        "the log ends at the boundary"
    );
    let got = s.prepare(&full)?;
    let cont = run(&mut s, &got.suffix, 4, None, dev)?;
    Ok((
        a,
        got.action,
        got.path.divergence_by,
        got.path.overrun_by,
        cont == want,
    ))
}

/// Past by one: the standing window takes it back, exactly, and the next
/// request extends. Past by three: an obligation, met by the next request's
/// restore of the prompt's checkpoint and its replay of the accepted ids.
/// The boundary at the log is nothing to do; past the log is refused.
fn accept_cases(dev: &Device) -> Result<()> {
    let (a, action, _, overrun_by, same) = case(dev, 1)?;
    eprintln!(
        "ACCEPT past 1: {a:?} then {action:?} {overrun_by:?}, continuation = reference {same}"
    );
    assert_eq!(a, Acceptance::Exact { dropped: 1 });
    assert_eq!(action, ReuseAction::Extend);
    assert_eq!(overrun_by, OverrunBy::None);
    assert!(same, "the exact case continues as the reference");

    let (a, action, _, overrun_by, same) = case(dev, 3)?;
    eprintln!(
        "ACCEPT past 3: {a:?} then {action:?} {overrun_by:?}, continuation = reference {same}"
    );
    assert_eq!(
        a,
        Acceptance::Obligation {
            ahead: 3,
            restore_point: Some(20)
        },
        "past the window: an obligation to restore the prompt's checkpoint"
    );
    assert_eq!(overrun_by, OverrunBy::Checkpoint(20), "met by the restore");
    assert_eq!(action, ReuseAction::Rewound(3), "the accepted ids replayed");
    assert!(same, "the replay continues as the reference");

    // An overrun first: the fixed-cadence ring stops on an EOS one forward
    // late, and the front end accepts two ids fewer than the run returned.
    // The overrun is taken back by the window, which the ids past the
    // boundary then cannot reach: an obligation for those two alone.
    let (m, cfg) = model(dev)?;
    let p = ids(20, 1);
    let stream = run(
        &mut Session::new(&m, m.new_cache(&cfg, dev)?),
        &p,
        24,
        None,
        dev,
    )?;
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let mut s = Session::new(&m, m.new_cache(&cfg, dev)?);
    let out = s.decode(
        &mut IdentityProposer,
        &mut Ring::new(
            &[stream[cut]],
            dev,
            128,
            EosRead::Poll(PollMechanism::default()),
        )?
        .with_poll_budget(20_000),
        &mut LogitsProcessor::from_sampling(0, Sampling::ArgMax),
        &p,
        &DecodeOptions {
            lookahead: 1,
            ..DecodeOptions::new(24)
        },
        &mut |_: &[u32]| Ok(()),
    )?;
    assert_eq!(
        (out.ids.len(), out.stats.overrun),
        (cut, 1),
        "the EOS one forward late"
    );
    let a = s.accept(p.len() + cut - 2)?;
    eprintln!("ACCEPT overrun 1, past 2: {a:?}");
    assert_eq!(
        a,
        Acceptance::Obligation {
            ahead: 2,
            restore_point: Some(20)
        },
        "the overrun reconciled by the window; the two ids past the boundary are the obligation"
    );
    let full = [p.clone(), stream[..cut - 2].to_vec(), ids(3, 9)].concat();
    let got = s.prepare(&full)?;
    assert_eq!(got.path.overrun_by, OverrunBy::Checkpoint(20));
    assert_eq!(got.action, ReuseAction::Rewound(cut - 2));

    let (m, cfg) = model(dev)?;
    let mut s = Session::new(&m, m.new_cache(&cfg, dev)?);
    run(&mut s, &ids(10, 3), 4, None, dev)?;
    let log = s.tokens().len();
    assert_eq!(s.accept(log)?, Acceptance::Whole);
    assert!(s.accept(log + 1).is_err(), "a boundary past the log");
    Ok(())
}

#[test]
fn the_session_is_reconciled_to_the_accepted_frontier_on_the_cpu() -> Result<()> {
    pin();
    accept_cases(&Device::Cpu)
}

#[cfg(feature = "metal")]
#[test]
fn the_session_is_reconciled_to_the_accepted_frontier_on_metal() -> Result<()> {
    pin();
    accept_cases(&Device::new_metal(0)?)
}
