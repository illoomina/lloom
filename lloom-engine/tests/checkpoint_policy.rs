//! The turn checkpoint each request leaves (`sequence-reuse-design.md`
//! §2.3b): one live session on the tiny Qwen3.5 (a one-step standing
//! window, two checkpoint slots) driven through
//! `DecodeSession::decode_until`, alternating a stripped turn (the front
//! end's tail of 2) with a persisted one (tail 0), tiled and untiled
//! prefills, an extend and a restore. After every request the positions the
//! state's checkpoints stand at are read off the state and checked against
//! the positions the request's own prompt names; the text is never read.
//!
//! The regression case is the strip → persist transition without a reset: a
//! hinted request used to leave the ring hint-driven, and the zero-tail
//! request after it supplied no policy, so no pass left a checkpoint.

use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::nn::decoder::{DecodeRecovery, StateCapacity};
use lloom::nn::model::CausalLM;
use lloom::nn::Sequence;
use lloom::profile::CrownExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_engine::turn::Outcome;
use lloom_engine::{DecodeParams, DecodeSession, Decoded, ReadbackArm};
use lloom_tensor::{Device, Result};

type Model = Qwen3_5<CrownExact>;

fn model(dev: &Device, slots: usize) -> Result<(Model, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<CrownExact>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        recovery: DecodeRecovery::Standing,
        checkpoints: slots,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

fn detok(ids: &[u32]) -> Result<String> {
    Ok(ids.iter().map(|t| format!("<{t}>")).collect())
}

/// One request: greedy, the blocking arm, `max` ids, the front end's
/// checkpoint `tail`, the prefill in tiles of `tile` ids (0: one pass).
fn run(
    s: &mut DecodeSession<'_, Model>,
    full: &[u32],
    max: usize,
    tail: usize,
    tile: usize,
) -> Result<Decoded> {
    let params = DecodeParams {
        max_tokens: max,
        eos_ids: vec![u32::MAX],
        readback: ReadbackArm::Blocking,
        turn_tail: tail,
        prefill_tile: tile,
        ..DecodeParams::default()
    };
    s.decode_until(full, detok, &params, |_| true)
}

fn positions(s: &mut DecodeSession<'_, Model>) -> Vec<usize> {
    let mut p = s.cache_mut().checkpoint_positions();
    p.sort_unstable();
    p
}

/// `n` ids the tiny vocabulary has, distinct from the run's own.
fn new_ids(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32).map(|i| 2 + (i * 7 + salt) % 60).collect()
}

fn the_live_session_alternates_policies(dev: &Device) -> Result<()> {
    let (model, cfg) = model(dev, 2)?;
    let mut s = DecodeSession::from_sequence(
        &model,
        Sequence::new(model.new_cache(&cfg, dev)?),
        dev.clone(),
    );

    // 1. Stripped: the checkpoint two ids before the prompt's end.
    let p1: Vec<u32> = new_ids(20, 1);
    run(&mut s, &p1, 4, 2, 0)?;
    let e1 = p1.len() - 2;
    assert_eq!(positions(&mut s), vec![e1], "1: a hinted request");

    // 2. Persisted, no reset: tail 0 is the prompt's end, not the policy
    // the hinted request before it left.
    let full2 = [s.tokens().to_vec(), new_ids(5, 2)].concat();
    let d = run(&mut s, &full2, 4, 0, 0)?;
    assert_eq!(d.turn.outcome, Outcome::Extend, "2: extends");
    let e2 = full2.len();
    assert_eq!(
        positions(&mut s),
        vec![e1, e2],
        "2: strip -> persist leaves the zero-tail request its checkpoint at the prompt's end"
    );

    // 3. Persisted, tiled by 4: one checkpoint, at the prompt's end, not
    // one per tile (which would evict the earlier turns').
    let full3 = [s.tokens().to_vec(), new_ids(12, 3)].concat();
    let d = run(&mut s, &full3, 4, 0, 4)?;
    assert_eq!(d.turn.outcome, Outcome::Extend, "3: extends");
    assert!(d.turn.prefilled > 4, "3: a tiled prefill");
    let e3 = full3.len();
    assert_eq!(
        positions(&mut s),
        vec![e2, e3],
        "3: a tiled zero-tail request leaves one checkpoint, at its end"
    );

    // 4. Stripped, the last reply re-rendered away: the state is restored
    // to the checkpoint at the previous prompt's end, and this request's
    // own stands two ids before its end.
    let full4 = [full3.clone(), new_ids(6, 4)].concat();
    let d = run(&mut s, &full4, 4, 2, 0)?;
    assert_eq!(d.turn.outcome, Outcome::Restore, "4: restores");
    assert_eq!(d.turn.diverged_at, Some(e3), "4: diverges at the reply");
    let e4 = full4.len() - 2;
    assert_eq!(
        positions(&mut s),
        vec![e3, e4],
        "4: the restored-to checkpoint survives; the oldest goes"
    );

    // 5. Persisted, tiled, after the restore: the end again.
    let full5 = [s.tokens().to_vec(), new_ids(9, 5)].concat();
    let d = run(&mut s, &full5, 4, 0, 4)?;
    assert_eq!(d.turn.outcome, Outcome::Extend, "5: extends");
    let e5 = full5.len();
    assert_eq!(positions(&mut s), vec![e4, e5], "5: persist after strip");
    Ok(())
}

/// A fresh session's tiled zero-tail request: one checkpoint at its end.
fn a_tiled_first_request_leaves_one_checkpoint(dev: &Device) -> Result<()> {
    let (model, cfg) = model(dev, 2)?;
    let mut s = DecodeSession::from_sequence(
        &model,
        Sequence::new(model.new_cache(&cfg, dev)?),
        dev.clone(),
    );
    let p = new_ids(16, 7);
    run(&mut s, &p, 3, 0, 4)?;
    assert_eq!(
        positions(&mut s),
        vec![p.len()],
        "a tiled prefill's checkpoint is the prompt's end, not its last two tiles' ends"
    );
    Ok(())
}

/// No checkpoint slots: every policy is a no-op and every request runs.
fn no_checkpoint_capacity(dev: &Device) -> Result<()> {
    let (model, cfg) = model(dev, 0)?;
    let mut s = DecodeSession::from_sequence(
        &model,
        Sequence::new(model.new_cache(&cfg, dev)?),
        dev.clone(),
    );
    let p = new_ids(20, 9);
    run(&mut s, &p, 3, 2, 0)?;
    let full = [s.tokens().to_vec(), new_ids(8, 10)].concat();
    run(&mut s, &full, 3, 0, 4)?;
    assert!(positions(&mut s).is_empty(), "no slots, no checkpoints");
    Ok(())
}

macro_rules! on_cpu_and_metal {
    ($($name:ident => $case:ident),* $(,)?) => {
        mod cpu {
            $(
                #[test]
                fn $name() -> lloom_tensor::Result<()> {
                    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
                    super::$case(&lloom_tensor::Device::Cpu)
                }
            )*
        }
        #[cfg(feature = "metal")]
        mod metal {
            $(
                #[test]
                fn $name() -> lloom_tensor::Result<()> {
                    let _ = lloom_tensor::tuning::pin_regime(lloom_tensor::tuning::Contract::Deterministic);
                    super::$case(&lloom_tensor::Device::new_metal(0)?)
                }
            )*
        }
    };
}

on_cpu_and_metal! {
    every_request_sets_its_own_checkpoint_policy => the_live_session_alternates_policies,
    a_tiled_prefill_leaves_one_checkpoint_at_its_end => a_tiled_first_request_leaves_one_checkpoint,
    no_checkpoint_slots_keeps_none => no_checkpoint_capacity,
}
