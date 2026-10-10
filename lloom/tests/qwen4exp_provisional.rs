//! Provisional steps (`CausalLM::provisional`, `Session::decode` on the
//! blocking arm): a device-planned decode step returns its logits before its
//! settle, the loop settles it after its readback — the pass complete, so the
//! settle waits for nothing — and a step that computed on a missing expert is
//! redone there and its token drawn again, before it is handed out. The
//! stream is the resident model's, token for token, under argmax and under a
//! seeded temperature draw (the redraw puts the sampler back first: a draw it
//! did not rewind would advance the rng once more and the stream would
//! leave the resident one).
//!
//! Metal only (the deferral is the device queue's). The tiny model's pool of
//! 6 slots for 8 experts at top-2 arms the device plan and misses on it
//! (`qwen4exp_device_plan.rs`).
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::testing::Qwen4ExpTiny;
use lloom::models::qwen4exp::{
    Config, ExpertResidency, LoadOptions, Qwen4Exp, SlotBudget, StreamLoader, StreamOptions,
};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, DecodeOut, IdentityProposer, Session};
use lloom::profile::CrownExact;
use lloom::testing::fixture::Fixture;
use lloom_tensor::{Device, Result};

fn stream(slots: usize) -> StreamOptions {
    StreamOptions {
        slots: SlotBudget::Fixed(slots),
        loader: StreamLoader::Pread,
        prefetch: false,
        io_hints: false,
        drafter_bytes: 0,
    }
}

const PROMPT: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
const STEPS: usize = 24;
/// An id no draw produces: the runs end on their step count.
const NO_EOS: u32 = u32::MAX - 1;

fn fixture(name: &str) -> Result<Option<(Fixture<Qwen4ExpTiny>, Device)>> {
    let Ok(dev) = Device::new_metal(0) else {
        return Ok(None);
    };
    if lloom::testing::fixture::Backend::current().at_scale() {
        println!("at scale: the tiny model's provisional tests do not run the checkpoint");
        return Ok(None);
    }
    Ok(Fixture::<Qwen4ExpTiny>::new(name, Config::tiny())?.map(|fx| (fx, dev)))
}

/// One run through the session on the blocking arm; the session's cache's
/// provisional redos beside it.
fn run<R: ExpertResidency>(
    model: &Qwen4Exp<CrownExact, R>,
    cfg: &Config,
    dev: &Device,
    sampling: Sampling,
) -> Result<(DecodeOut, u64)> {
    let cache = model.new_cache(cfg, dev)?;
    let mut session = Session::new(model, cache);
    let mut rb = Blocking::new(&[NO_EOS], dev)?;
    let mut sampler = LogitsProcessor::from_sampling(7, sampling);
    let out = session.decode(
        &mut IdentityProposer,
        &mut rb,
        &mut sampler,
        &PROMPT,
        &DecodeOptions::new(STEPS),
        &mut |_| Ok(()),
    )?;
    Ok((out, session.cache_mut().provisional_redos))
}

fn check(name: &str, sampling: Sampling) -> Result<()> {
    let Some((fx, dev)) = fixture(name)? else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let opts = LoadOptions::default();
    let (want, _) = {
        let resident = fx.resident::<CrownExact>(&cfg, &dev, &opts)?;
        run(&resident, &cfg, &dev, sampling.clone())?
    };
    let streamed = fx.streamed::<CrownExact>(&cfg, &dev, &opts, &stream(6))?;
    let (out, redos) = run(&streamed, &cfg, &dev, sampling)?;
    let st = streamed.stream_stats();
    println!(
        "PROVISIONAL {name} provisional={} redraws={} provisional_redos={redos} device_plans={} settled_misses={}",
        out.stats.provisional, out.stats.redraws, st.device_plans, st.settled_misses
    );
    assert_eq!(out.ids.len(), STEPS);
    assert_eq!(
        out.ids, want.ids,
        "{name}: the provisional stream differs from the resident one"
    );
    assert!(
        out.stats.provisional,
        "{name}: the run's steps were not provisional"
    );
    assert!(
        out.stats.redraws > 0 && redos > 0,
        "{name}: no provisional step was redone — the identity says nothing about the redraw \
         (redraws {}, redos {redos})",
        out.stats.redraws
    );
    assert!(
        !streamed.step_settle_trailing(),
        "{name}: the run ended with a provisional step unsettled"
    );
    Ok(())
}

#[test]
fn a_provisional_stream_is_the_resident_stream_under_argmax_metal() -> Result<()> {
    check("qwen4exp_provisional_argmax_metal", Sampling::ArgMax)
}

#[test]
fn a_redraw_puts_the_sampler_back_first_metal() -> Result<()> {
    check(
        "qwen4exp_provisional_temperature_metal",
        Sampling::All { temperature: 1.0 },
    )
}

/// A resident model declines provisional steps: it has nothing to settle,
/// and a deferral would only cut a command buffer a step.
#[test]
fn a_resident_model_declines_provisional_steps_metal() -> Result<()> {
    let Some((fx, dev)) = fixture("qwen4exp_provisional_resident_metal")? else {
        return Ok(());
    };
    let cfg = fx.config(|_| {});
    let resident = fx.resident::<CrownExact>(&cfg, &dev, &LoadOptions::default())?;
    let (out, _) = run(&resident, &cfg, &dev, Sampling::ArgMax)?;
    assert!(
        !out.stats.provisional,
        "a resident model took provisional steps"
    );
    assert_eq!(out.stats.redraws, 0);
    Ok(())
}
