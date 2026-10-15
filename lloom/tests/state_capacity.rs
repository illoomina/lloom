//! `StateCapacity`'s three quantities, each buying its own storage and work
//! and nothing of the others' (`profiles-and-memory-design.md` §4.4a):
//! - the verify window pools the recurrent state at a rewindable profile,
//!   `verify + 1` phases a channel, and keeps no decode step undoable;
//! - the decode recovery keeps the last step undoable (the standing
//!   window), and buys no verify storage at the shipped `No` profile;
//! - the checkpoints keep a ring of frames whatever the other two are.
//!
//! The library default buys none of them. Where a ring serves both a
//! verify window and the standing window it holds the longer of the two,
//! not their sum: a verify settles the standing window before it opens.
//!
//! The model is the tiny Qwen3.5 dense on the CPU at both profiles.

use lloom::models::qwen3_5::testing::Qwen3_5Tiny;
use lloom::models::qwen3_5::{Config, Qwen3_5};
use lloom::models::qwen4exp::{GdnState, KvCache};
use lloom::nn::decoder::{
    DecodeRecovery, DecoderState, HybridState, Layer, PreNormState, StateCapacity,
};
use lloom::nn::model::{CacheRewind, CausalLM};
use lloom::profile::{CrownExact, Profile, RewindableExact};
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::{Device, Result, Tensor};

type LayerState = PreNormState<HybridState<GdnState, KvCache>, ()>;

fn model<P: Profile>(
    dev: &Device,
    verify: usize,
    recovery: DecodeRecovery,
    checkpoints: usize,
) -> Result<(Qwen3_5<P>, Config)> {
    let ckpt = TinyCheckpoint::<Qwen3_5Tiny>::new(Config::tiny(), Weights::default());
    let mut model = ckpt.model::<P>(dev)?;
    let cap = model.capacity();
    model.set_capacity(StateCapacity {
        verify,
        recovery,
        checkpoints,
        ..cap
    });
    Ok((model, ckpt.cfg.clone()))
}

/// The phases of every recurrent layer's `S` channel: one value, or `None`
/// when the channels are unpooled.
fn phases<L: Layer<State = LayerState>>(st: &DecoderState<L>) -> Option<usize> {
    let mut seen = Vec::new();
    for l in &st.layers {
        if let HybridState::A(g) = &l.mixer {
            seen.push(g.s.pool_slot(g.layer).map(|(p, ..)| p.phases()));
        }
    }
    assert!(!seen.is_empty(), "the fixture has recurrent layers");
    assert!(
        seen.windows(2).all(|w| w[0] == w[1]),
        "every layer the same: {seen:?}"
    );
    seen[0]
}

/// A prefill of a few tokens, then one decode step: the state as a run
/// leaves it.
fn run<M, L>(model: &M, cfg: &M::Config, dev: &Device) -> Result<DecoderState<L>>
where
    M: CausalLM<Cache = DecoderState<L>>,
    L: Layer,
{
    let mut st = model.new_cache(cfg, dev)?;
    let prompt = Tensor::from_vec(vec![1u32, 3, 5, 7, 9], (1, 5), dev)?;
    model.forward(&prompt, 0, &mut st)?;
    let step = Tensor::from_vec(vec![11u32], (1, 1), dev)?;
    let pos = st.pos;
    model.forward(&step, pos, &mut st)?;
    Ok(st)
}

/// What one configuration bought: the phases, the decode step's horizon,
/// the widest window one pass holds, and the checkpoint slots.
fn bought<P: Profile>(
    verify: usize,
    recovery: DecodeRecovery,
    checkpoints: usize,
) -> Result<(Option<usize>, usize, usize, usize)> {
    let dev = Device::Cpu;
    let (model, cfg) = model::<P>(&dev, verify, recovery, checkpoints)?;
    let st = run(&model, &cfg, &dev)?;
    let widest = (1..=8)
        .take_while(|&t| st.ring_holds(t))
        .last()
        .unwrap_or(0);
    Ok((
        phases(&st),
        st.rewind_horizon(),
        widest,
        st.checkpoint_slots(),
    ))
}

#[test]
fn the_default_buys_no_window_no_recovery_and_no_checkpoints() -> Result<()> {
    let cap = StateCapacity::new(64);
    assert_eq!(
        (cap.verify, cap.recovery, cap.checkpoints),
        (0, DecodeRecovery::None, 0),
        "the library default"
    );
    assert_eq!(cap.recurrent_phases(true), None);
    assert_eq!(cap.recurrent_phases(false), None);
    for (name, got) in [
        ("No", bought::<CrownExact>(0, DecodeRecovery::None, 0)?),
        (
            "Ring",
            bought::<RewindableExact>(0, DecodeRecovery::None, 0)?,
        ),
    ] {
        eprintln!("CAPACITY default {name}: {got:?}");
        assert_eq!(got, (None, 0, 0, 0), "{name}: nothing bought");
    }
    Ok(())
}

#[test]
fn a_verify_window_pools_the_state_and_keeps_no_step_undoable() -> Result<()> {
    let ring = bought::<RewindableExact>(3, DecodeRecovery::None, 0)?;
    eprintln!("CAPACITY verify 3 Ring: {ring:?}");
    assert_eq!(ring.0, Some(4), "the window and its base");
    assert_eq!(ring.1, 0, "a verify window keeps no decode step undoable");
    assert_eq!(ring.2, 3, "one pass holds the window");
    let no = bought::<CrownExact>(3, DecodeRecovery::None, 0)?;
    eprintln!("CAPACITY verify 3 No: {no:?}");
    assert_eq!(no, (None, 0, 0, 0), "the `No` profile pools nothing");
    Ok(())
}

#[test]
fn the_standing_window_keeps_one_step_and_no_verify_storage() -> Result<()> {
    let no = bought::<CrownExact>(0, DecodeRecovery::Standing, 0)?;
    eprintln!("CAPACITY standing No: {no:?}");
    assert_eq!(no, (None, 1, 0, 0), "the shipped shape: one step, unpooled");
    let ring = bought::<RewindableExact>(0, DecodeRecovery::Standing, 0)?;
    eprintln!("CAPACITY standing Ring: {ring:?}");
    assert_eq!(
        ring,
        (Some(2), 1, 1, 0),
        "the base and the step's phase; a window of one"
    );
    Ok(())
}

#[test]
fn a_ring_serving_both_holds_the_longer_not_the_sum() -> Result<()> {
    for (verify, want) in [(1usize, 2usize), (3, 4)] {
        let got = bought::<RewindableExact>(verify, DecodeRecovery::Standing, 0)?;
        eprintln!("CAPACITY verify {verify} + standing Ring: {got:?}");
        assert_eq!(
            got.0,
            Some(want),
            "verify {verify}: max(verify, 1) + 1 phases"
        );
        assert_eq!(got.1, 1, "verify {verify}: the step stays undoable");
        assert_eq!(got.2, verify, "verify {verify}: one pass holds the window");
    }
    Ok(())
}

#[test]
fn the_checkpoints_are_their_own() -> Result<()> {
    for (verify, recovery) in [
        (0, DecodeRecovery::None),
        (0, DecodeRecovery::Standing),
        (3, DecodeRecovery::None),
        (3, DecodeRecovery::Standing),
    ] {
        let without = bought::<RewindableExact>(verify, recovery, 0)?;
        let with = bought::<RewindableExact>(verify, recovery, 2)?;
        eprintln!("CAPACITY checkpoints ({verify}, {recovery:?}): 0 {without:?} 2 {with:?}");
        assert_eq!((without.3, with.3), (0, 2), "the slots are the field's");
        assert_eq!(
            (with.0, with.1, with.2),
            (without.0, without.1, without.2),
            "({verify}, {recovery:?}): the checkpoints change nothing else"
        );
    }
    Ok(())
}
