//! Qwen4-exp under the pass-plan cache (`DESIGN.md` §9.2aa) after the load
//! warmup (`Session::warmup`): with the cache paused over the warmup, the
//! first prompt decodes the ids the pool path decodes, bit for bit, and the
//! warmup leaves nothing in the cache.
//!
//! The warmup calls `forward` with no driver bracket, so this model opens its
//! own passes. Those are keyed apart from a driver's (`PassOpener`, tested in
//! `qwen4exp_pass_plan_openers`), so a warmup plan could not be served to the
//! driver's decode; the pause is for the warmup's own contract, to leave no
//! plan behind.
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::{testing, Config, Qwen4Exp, Resident};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom::profile::CrownExact;
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice, PassKey, PassKind};
use lloom_tensor::{Device, Result};

/// The recorder's bind observer and the dispatch clock are process globals,
/// so two tests planning passes at once record each other's dispatches.
static SERIAL: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

const PROMPT: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
const STEPS: usize = 12;

fn tiny() -> Config {
    let mut cfg = Config::tiny();
    cfg.n_experts = 6;
    cfg.n_experts_per_tok = 2;
    cfg
}

fn metal(dev: &Device) -> &MetalDevice {
    match dev {
        Device::Metal(m) => m,
        _ => unreachable!("a Metal device"),
    }
}

/// `steps` greedy ids for `PROMPT` through the driver (`Session::decode`),
/// on a fresh session over `model`.
fn decode(model: &Qwen4Exp<CrownExact, Resident>, dev: &Device, steps: usize) -> Result<Vec<u32>> {
    let cfg = tiny();
    let mut session = Session::new(model, model.new_cache(&cfg, dev)?);
    decode_on(&mut session, dev, steps)
}

fn decode_on(
    session: &mut Session<'_, Qwen4Exp<CrownExact, Resident>>,
    dev: &Device,
    steps: usize,
) -> Result<Vec<u32>> {
    let mut spec = IdentityProposer;
    let mut rb = Blocking::new(&[u32::MAX], dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(steps);
    let mut sink = |_ids: &[u32]| -> Result<()> { Ok(()) };
    Ok(session
        .decode(&mut spec, &mut rb, &mut sampler, &PROMPT, &opts, &mut sink)?
        .ids)
}

/// The pool path's ids: a fresh device, no plans, no warmup.
fn pool_reference() -> Result<Vec<u32>> {
    let dev = Device::new_metal(0)?;
    let model = testing::resident_model::<CrownExact>(&tiny(), &dev);
    let want = decode(&model, &dev, STEPS)?;
    assert_eq!(want.len(), STEPS);
    assert!(
        want.iter().any(|&t| t != want[0]),
        "a degenerate stream cannot tell the arms apart: {want:?}"
    );
    Ok(want)
}

#[test]
fn a_warmup_under_the_paused_cache_records_nothing_and_decodes_the_pool_paths_ids() -> Result<()> {
    let _serial = SERIAL.lock();
    let want = pool_reference()?;

    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let cfg = tiny();
    let model = testing::resident_model::<CrownExact>(&cfg, &dev);
    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    {
        let _paused = m.pause_pass_plans()?;
        assert!(
            !m.pass_plans_enabled(),
            "the cache is set aside while paused"
        );
        session.warmup(&dev)?;
    }
    assert!(
        m.pass_plans_enabled(),
        "the cache is back when the guard drops"
    );
    let c = m.pass_plan_counters().expect("cache on");
    assert_eq!(
        (c.recorded, c.observed, c.served, c.plans),
        (0, 0, 0, 0),
        "the warmup's passes reached the cache: {c:?}"
    );
    let got = decode_on(&mut session, &dev, STEPS)?;
    eprintln!(
        "[paused warmup] ids={got:?} want={want:?} counters={:?}",
        m.pass_plan_counters()
    );
    assert_eq!(
        got, want,
        "a planned session moved its ids after the warmup"
    );
    // The run itself was planned: the decode key was served, from its own
    // sequence.
    let c = m.pass_plan_counters().expect("cache on");
    assert!(
        c.served > 0,
        "nothing was served, so nothing was tested: {c:?}"
    );
    assert_eq!(c.evicted, 0, "a key drifted under its own plan: {c:?}");
    Ok(())
}

#[test]
fn a_key_pending_at_the_pause_is_not_closed_by_the_paused_passes() -> Result<()> {
    let _serial = SERIAL.lock();
    let want = pool_reference()?;

    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let cfg = tiny();
    let model = testing::resident_model::<CrownExact>(&cfg, &dev);
    // A one-id run ends on a key's first sight: left pending, its recorder
    // installed.
    decode(&model, &dev, 1)?;
    assert!(
        m.arena_recording_excluded().is_some(),
        "the fixture must leave a key pending"
    );

    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    {
        let _paused = m.pause_pass_plans()?;
        assert!(
            m.arena_recording_excluded().is_none(),
            "the pending key's recorder still observes the paused passes"
        );
        session.warmup(&dev)?;
    }
    assert_eq!(decode_on(&mut session, &dev, STEPS)?, want);
    Ok(())
}

#[test]
fn a_pause_inside_an_open_pass_is_refused() -> Result<()> {
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    m.begin_pass(PassKey::new(PassKind::Decode { b: 1 }))?;
    let refused = m.pause_pass_plans().err().map(|e| e.to_string());
    m.finish_pass()?;
    assert!(
        refused
            .as_deref()
            .is_some_and(|e| e.contains("a pass is open")),
        "{refused:?}"
    );
    assert!(m.pass_plans_enabled());
    Ok(())
}
