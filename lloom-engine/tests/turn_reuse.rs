//! The turn report (`lloom_engine::turn`, `sequence-reuse-design.md`
//! §2.3f): every case `Session::prepare` can take, through
//! `DecodeSession::decode_until` on the tiny Qwen3.5 (a standing window of
//! one step, two turn checkpoints), each report checked against what the
//! session itself held — the log's length before the request, the
//! checkpoint the test hinted — never against the report's own fields.

use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::nn::decoder::{DecodeRecovery, StateCapacity};
use lloom::nn::model::CausalLM;
use lloom::nn::{ReuseAction, Sequence};
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_engine::turn::{Outcome, Reason, StateReady, TurnReuse};
use lloom_engine::{DecodeParams, DecodeSession, Decoded, ReadbackArm};
use lloom_tensor::{Device, Result};

type Model = Qwen3_5<CrownExact>;

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

fn detok(ids: &[u32]) -> Result<String> {
    Ok(ids.iter().map(|t| format!("<{t}>")).collect())
}

/// One request on `s`: greedy, the blocking arm, `max` ids, the turn
/// checkpoint `tail` ids before the prompt's end. Returns the run and the
/// log's length before it.
fn run(
    s: &mut DecodeSession<'_, Model>,
    full: &[u32],
    max: usize,
    tail: usize,
) -> Result<(Decoded, usize)> {
    let boundary = s.tokens().len();
    let params = DecodeParams {
        max_tokens: max,
        eos_ids: vec![u32::MAX],
        readback: ReadbackArm::Blocking,
        turn_tail: tail,
        ..DecodeParams::default()
    };
    let d = s.decode_until(full, detok, &params, |_| true)?;
    Ok((d, boundary))
}

const P: [u32; 20] = [
    1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39,
];

/// A session that has run `P` for six ids, its checkpoint hinted three ids
/// before the prompt's end.
fn after_turn_one<'m>(
    model: &'m Model,
    cfg: &Config,
    dev: &Device,
) -> Result<DecodeSession<'m, Model>> {
    let mut s = DecodeSession::from_sequence(
        model,
        Sequence::new(model.new_cache(cfg, dev)?),
        dev.clone(),
    );
    let (d, boundary) = run(&mut s, &P, 6, 3)?;
    assert_eq!(boundary, 0);
    assert_eq!(
        d.turn,
        TurnReuse {
            outcome: Outcome::Fresh,
            reason: Reason::Cold,
            prefilled: P.len(),
            ..TurnReuse::default()
        },
        "a cold session"
    );
    Ok(s)
}

/// A token the vocabulary has that `ids` does not use at `at`.
fn other(ids: &[u32], at: usize) -> u32 {
    if ids[at] == 2 {
        4
    } else {
        2
    }
}

#[test]
fn every_prepare_case_reports_what_the_session_held() -> Result<()> {
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::Cpu;
    let (model, cfg) = model(&dev)?;
    let at = P.len() - 3;

    // Extend: the whole log, then three new ids.
    let mut s = after_turn_one(&model, &cfg, &dev)?;
    let log = s.tokens().to_vec();
    let full = [log.clone(), vec![41, 43, 45]].concat();
    let (d, boundary) = run(&mut s, &full, 4, 0)?;
    assert_eq!(boundary, log.len());
    assert_eq!(d.reuse, ReuseAction::Extend);
    assert_eq!(
        d.turn,
        TurnReuse {
            boundary,
            outcome: Outcome::Extend,
            reused: boundary,
            prefilled: 3,
            ..TurnReuse::default()
        },
        "extend"
    );

    // The log whole: resumed from the held logits, or its last id fed again.
    let mut s = after_turn_one(&model, &cfg, &dev)?;
    let log = s.tokens().to_vec();
    let (d, boundary) = run(&mut s, &log, 2, 0)?;
    match d.reuse {
        ReuseAction::Resume => assert_eq!(
            d.turn,
            TurnReuse {
                boundary,
                outcome: Outcome::Resume,
                reused: boundary,
                ..TurnReuse::default()
            },
            "resume"
        ),
        ReuseAction::Rewound(1) => assert_eq!(
            d.turn,
            TurnReuse {
                boundary,
                outcome: Outcome::Rewind,
                reason: Reason::Refeed,
                reused: boundary - 1,
                prefilled: 1,
                ..TurnReuse::default()
            },
            "refeed"
        ),
        a => panic!("the log whole: {a:?}"),
    }

    // The last id replaced: one step back, inside the standing window.
    let mut s = after_turn_one(&model, &cfg, &dev)?;
    let log = s.tokens().to_vec();
    let n = log.len();
    let full = [log[..n - 1].to_vec(), vec![other(&log, n - 1)]].concat();
    let (d, boundary) = run(&mut s, &full, 2, 0)?;
    assert_eq!(d.reuse, ReuseAction::Rewound(1));
    assert_eq!(
        d.turn,
        TurnReuse {
            boundary,
            outcome: Outcome::Rewind,
            reason: Reason::Diverged,
            diverged_at: Some(n - 1),
            reused: n - 1,
            prefilled: 1,
            ..TurnReuse::default()
        },
        "rewind"
    );

    // Diverging two ids past the hinted checkpoint: restored there, and the
    // two ids between it and the divergence prefilled again.
    let mut s = after_turn_one(&model, &cfg, &dev)?;
    let log = s.tokens().to_vec();
    let p = at + 2;
    let full = [P[..p].to_vec(), vec![other(&P, p), 47]].concat();
    let (d, boundary) = run(&mut s, &full, 2, 0)?;
    assert_eq!(d.reuse, ReuseAction::Rewound(log.len() - at));
    assert_eq!(
        d.turn,
        TurnReuse {
            boundary,
            outcome: Outcome::Restore,
            reason: Reason::Diverged,
            diverged_at: Some(p),
            reused: at,
            replayed: 2,
            prefilled: 2,
            ..TurnReuse::default()
        },
        "restore"
    );

    // Diverging before every checkpoint: fresh, the shared ids replayed.
    let mut s = after_turn_one(&model, &cfg, &dev)?;
    let full = [P[..2].to_vec(), vec![other(&P, 2), 49, 51]].concat();
    let (d, boundary) = run(&mut s, &full, 2, 0)?;
    assert_eq!(d.reuse, ReuseAction::Fresh);
    assert_eq!(
        d.turn,
        TurnReuse {
            boundary,
            outcome: Outcome::Fresh,
            reason: Reason::Diverged,
            diverged_at: Some(2),
            replayed: 2,
            prefilled: 3,
            ..TurnReuse::default()
        },
        "past every checkpoint"
    );

    // Nothing in common.
    let mut s = after_turn_one(&model, &cfg, &dev)?;
    let full = vec![other(&P, 0), 53, 55];
    let (d, boundary) = run(&mut s, &full, 2, 0)?;
    assert_eq!(
        d.turn,
        TurnReuse {
            boundary,
            outcome: Outcome::Fresh,
            reason: Reason::NoOverlap,
            diverged_at: Some(0),
            prefilled: 3,
            ..TurnReuse::default()
        },
        "no overlap"
    );

    // An owner that reuses nothing reset the session first: the cold
    // report says so by policy.
    let cold = TurnReuse {
        outcome: Outcome::Fresh,
        reason: Reason::Cold,
        prefilled: 5,
        ..TurnReuse::default()
    };
    assert_eq!(cold.with_policy(Some(false)).reason, Reason::PolicyOff);
    assert_eq!(cold.with_policy(Some(true)).reason, Reason::Cold);
    assert_eq!(cold.with_policy(None).reason, Reason::Cold);
    Ok(())
}

/// The overrun cases, on the ring arms (Metal): the fixed cadence's one
/// forward past the EOS is taken back by the standing window and the next
/// request extends; the adaptive commit with a consumer slow at the EOS runs
/// past the window, and the next request restores the prompt pass's own
/// checkpoint (its end: no hint) and replays the reply.
#[cfg(feature = "metal")]
#[test]
fn an_overrun_is_reported_with_how_the_state_was_made_ready() -> Result<()> {
    use lloom_engine::{DecodeConfig, ReadbackConfig};
    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
    let dev = Device::new_metal(0)?;
    let (model, cfg) = model(&dev)?;
    let prompt: Vec<u32> = vec![1, 3, 5, 7, 9];
    let config = |spelling: &str| {
        let mut readback = ReadbackConfig::default();
        ReadbackConfig::apply_spelling(&mut readback, spelling).expect("a readback spelling");
        readback.eos_gate = false;
        DecodeConfig {
            readback,
            ..DecodeConfig::default()
        }
    };
    // The reply and the EOS to stop it on, from a blocking run.
    let stream = {
        let mut s = DecodeSession::from_sequence(
            &model,
            Sequence::new(model.new_cache(&cfg, &dev)?),
            dev.clone(),
        );
        run(&mut s, &prompt, 32, 0)?.0.tokens
    };
    let cut = (6..stream.len())
        .find(|&i| !stream[..i].contains(&stream[i]))
        .expect("a fresh id to stop on");
    let eos = stream[cut];
    for (spelling, slow) in [("ring-fixed", false), ("ring-nonblocking", true)] {
        let c = config(spelling);
        let mut s = DecodeSession::from_sequence(
            &model,
            Sequence::new(model.new_cache(&cfg, &dev)?),
            dev.clone(),
        )
        .with_config(c.clone());
        let params = DecodeParams {
            max_tokens: 3000,
            eos_ids: vec![eos],
            readback: c.readback.arm(),
            ..DecodeParams::default()
        };
        let mut delivered = 0usize;
        let first = s.decode_until(&prompt, detok, &params, |frag| {
            delivered += frag.matches('<').count();
            if slow && delivered >= cut {
                std::thread::sleep(std::time::Duration::from_millis(1000));
            }
            true
        })?;
        assert_eq!(first.tokens, stream[..cut], "{spelling}");
        let overrun = first.stats.overrun;
        let log = s.tokens().to_vec();
        assert_eq!(
            log,
            [prompt.clone(), first.tokens.clone()].concat(),
            "{spelling}"
        );
        let full = [log.clone(), vec![41, 43, 45]].concat();
        let (d, boundary) = run(&mut s, &full, 2, 0)?;
        assert_eq!(boundary, log.len());
        let want = if slow {
            assert!(
                overrun > 1,
                "{spelling}: the slow consumer let the host past the window"
            );
            TurnReuse {
                boundary,
                overrun,
                state: StateReady::Checkpoint(prompt.len()),
                outcome: Outcome::Restore,
                reason: Reason::Overrun,
                reused: prompt.len(),
                replayed: first.tokens.len(),
                prefilled: 3,
                ..TurnReuse::default()
            }
        } else {
            assert_eq!(overrun, 1, "{spelling}");
            TurnReuse {
                boundary,
                overrun,
                state: StateReady::Window,
                outcome: Outcome::Extend,
                reused: boundary,
                prefilled: 3,
                ..TurnReuse::default()
            }
        };
        assert_eq!(d.turn, want, "{spelling}");
    }
    Ok(())
}
