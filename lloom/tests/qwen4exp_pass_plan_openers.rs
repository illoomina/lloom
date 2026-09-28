//! One pass key names one allocation sequence (`DESIGN.md` §9.2aa): a pass
//! the model brackets itself and a driver's pass of the same shape are
//! different keys, so neither is served the other's plan.
//!
//! A driver opens its pass before `forward`, so the pass holds the model's
//! embedding; Qwen4-exp brackets a pass no driver holds after it — a direct
//! `forward` such as the `/v1` logprobs path runs. The two sequences are one
//! allocation and one dispatch apart, which moves an ordinal and its
//! position together, so the drift guard alone cannot tell them apart.
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::qwen4exp::{testing, Config, Qwen4Exp, Resident};
use lloom::nn::model::CausalLM;
use lloom::nn::{DecodeOptions, IdentityProposer, Session};
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice, PassKey, PassKind, PassOpener};
use lloom_tensor::{DType, Device, Result, Tensor};

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

/// `PROMPT`'s prefill, the key both openers take for it.
fn prefill_key() -> PassKey {
    PassKey::new(PassKind::Prefill {
        w: PROMPT.len() as u32,
        fresh: true,
    })
}

/// `STEPS` greedy ids for `PROMPT` through the driver (`Session::decode`).
fn decode(session: &mut Session<'_, Qwen4Exp<Resident>>, dev: &Device) -> Result<Vec<u32>> {
    session.reset();
    let mut spec = IdentityProposer;
    let mut rb = Blocking::new(&[u32::MAX], dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions::new(STEPS);
    let mut sink = |_ids: &[u32]| -> Result<()> { Ok(()) };
    Ok(session
        .decode(&mut spec, &mut rb, &mut sampler, &PROMPT, &opts, &mut sink)?
        .ids)
}

/// `PROMPT`'s last-position logits from a direct `forward`, as the logprobs
/// path takes them: no driver bracket, so the model opens its own pass.
fn direct(session: &mut Session<'_, Qwen4Exp<Resident>>, dev: &Device) -> Result<Vec<f32>> {
    session.reset();
    let input = Tensor::new(PROMPT.as_slice(), dev)?.unsqueeze(0)?;
    let logits = session.model().forward(&input, 0, session.cache_mut())?;
    session.reset();
    logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

/// The pool path's ids and direct logits: a fresh device, no plans.
fn pool_reference() -> Result<(Vec<u32>, Vec<f32>)> {
    let dev = Device::new_metal(0)?;
    let cfg = tiny();
    let model = testing::resident_model(&cfg, &dev);
    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    let ids = decode(&mut session, &dev)?;
    let logits = direct(&mut session, &dev)?;
    assert!(
        ids.iter().any(|&t| t != ids[0]),
        "a degenerate stream cannot tell the arms apart: {ids:?}"
    );
    Ok((ids, logits))
}

/// Both openers' plans for `PROMPT`'s prefill are cached, and no plan was
/// evicted: a plan served to the other opener's sequence drifts (or, where
/// the shift keeps size and position, aliases), and either shows here.
fn assert_two_plans_and_no_drift(m: &MetalDevice, what: &str) {
    let c = m.pass_plan_counters().expect("cache on");
    eprintln!(
        "[{what}] counters={c:?} keys={:?}",
        m.pass_plan_report().map(|(k, _)| k).unwrap_or_default()
    );
    assert!(
        m.pass_plan_cached(&prefill_key()),
        "{what}: the driver's prefill has no plan: {c:?}"
    );
    assert!(
        m.pass_plan_cached(&prefill_key().opened_by(PassOpener::Model)),
        "{what}: the model's own prefill has no plan: {c:?}"
    );
    assert_eq!(c.evicted, 0, "{what}: a plan drifted under a pass: {c:?}");
}

/// The mechanism, with no model: a driver's plan for a shape is cached, and
/// a pass the model opens for that shape is recorded, not served.
#[test]
fn a_model_opened_pass_is_not_served_a_drivers_plan() -> Result<()> {
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let x = Tensor::arange(0f32, 4096f32, &dev)?;
    let pass = |key: PassKey| -> Result<Vec<f32>> {
        m.begin_pass(key)?;
        let y = x.affine(2.0, 1.0)?.affine(1.0, 3.0)?;
        m.finish_pass()?;
        y.narrow(0, 0, 4)?.to_vec1::<f32>()
    };
    let driver = PassKey::new(PassKind::Decode { b: 1 });
    let own = driver.opened_by(PassOpener::Model);
    assert_ne!(driver, own, "the opener is part of the key");

    // Recorded, then closed by its second sight, then served.
    for _ in 0..3 {
        assert_eq!(pass(driver)?, [4.0, 6.0, 8.0, 10.0]);
    }
    let c = m.pass_plan_counters().expect("cache on");
    assert_eq!(
        (c.recorded, c.served, c.plans),
        (1, 1, 1),
        "the driver's key must be cached and served before the model opens: {c:?}"
    );
    assert!(m.pass_plan_cached(&driver) && !m.pass_plan_cached(&own));

    // The model's first pass of the same shape is a first sight.
    assert_eq!(pass(own)?, [4.0, 6.0, 8.0, 10.0]);
    let c = m.pass_plan_counters().expect("cache on");
    assert_eq!(
        (c.recorded, c.served),
        (2, 1),
        "the model's pass was served the driver's plan: {c:?}"
    );
    // And it plans on its own sightings.
    for _ in 0..2 {
        assert_eq!(pass(own)?, [4.0, 6.0, 8.0, 10.0]);
    }
    let c = m.pass_plan_counters().expect("cache on");
    assert_eq!((c.plans, c.served, c.evicted), (2, 2, 0), "{c:?}");
    assert!(m.pass_plan_cached(&own));
    Ok(())
}

/// The logprobs shape first: direct forwards plan the model's own prefill,
/// then the driver decodes the same prompt. Its prefill is a first sight,
/// not the direct forwards' plan, and every id is the pool path's.
#[test]
fn a_direct_forward_does_not_plan_the_drivers_prefill() -> Result<()> {
    let _serial = SERIAL.lock();
    let (want_ids, want_logits) = pool_reference()?;

    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let cfg = tiny();
    let model = testing::resident_model(&cfg, &dev);
    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    // Recorded, closed, served: the third sight runs on the model's plan.
    for round in 0..3 {
        assert_eq!(
            direct(&mut session, &dev)?,
            want_logits,
            "direct forward {round} moved its logits"
        );
    }
    let served = m.pass_plan_counters().expect("cache on").served;
    assert!(
        served > 0,
        "the direct forwards were never served: {served}"
    );
    for round in 0..2 {
        assert_eq!(
            decode(&mut session, &dev)?,
            want_ids,
            "the driver's decode {round} moved its ids"
        );
    }
    assert_two_plans_and_no_drift(m, "direct, then driver");
    Ok(())
}

/// The admission (`prepare_pass`) reads the plan of the pass that will run:
/// a pass a plan serves owes no transient, and the model's own plan says
/// nothing about a driver's pass of the same shape, nor the reverse. The
/// budget admits what is allocated plus a margin that covers the KV slab a
/// reset cache grows again (a few KB on this model) and not the prefill's
/// transient (`PREFILL_TRANSIENT_BYTES_PER_TOKEN` × 9 = 27 MB): the model's
/// served prefill passes, and the driver's unplanned one is refused.
#[test]
fn admission_reads_the_plan_of_the_pass_that_runs() -> Result<()> {
    use lloom::models::qwen4exp::{Cache, MemCheck};
    let _serial = SERIAL.lock();
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let cfg = tiny();
    let mut model = testing::resident_model(&cfg, &dev);
    let mut cache: Cache = model.new_cache(&cfg, &dev)?;
    let input = Tensor::new(PROMPT.as_slice(), &dev)?.unsqueeze(0)?;
    let forward = |model: &Qwen4Exp<Resident>, cache: &mut Cache| -> Result<()> {
        cache.reset();
        model.forward(&input, 0, cache)?;
        cache.reset();
        Ok(())
    };
    // Recorded, closed, served: the model's own prefill has a plan, the
    // driver's has none.
    for _ in 0..3 {
        forward(&model, &mut cache)?;
    }
    let own = prefill_key().opened_by(PassOpener::Model);
    assert!(m.pass_plan_cached(&own) && !m.pass_plan_cached(&prefill_key()));

    let margin = 8 << 20;
    assert!(
        margin < lloom::models::qwen4exp::PREFILL_TRANSIENT_BYTES_PER_TOKEN * PROMPT.len(),
        "the margin must not admit the transient"
    );
    let mtl = m.device();
    let budget = mtl.current_allocated_size() + margin;
    model.set_mem_check(MemCheck::Refuse);
    model.set_memory_fraction(budget as f64 / mtl.recommended_max_working_set_size() as f64);
    let served = forward(&model, &mut cache);
    assert!(
        served.is_ok(),
        "the model's served prefill was charged a transient: {served:?}"
    );
    cache.reset();
    let refused = model.prepare_pass(&mut cache, PROMPT.len());
    assert!(
        refused.is_err(),
        "the driver's unplanned prefill was admitted on the model's plan"
    );
    Ok(())
}

/// The other order: the driver plans its prefill first, then direct
/// forwards of the same prompt compute the pool path's logits bit for bit.
#[test]
fn a_drivers_prefill_plan_is_not_served_to_a_direct_forward() -> Result<()> {
    let _serial = SERIAL.lock();
    let (want_ids, want_logits) = pool_reference()?;

    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let cfg = tiny();
    let model = testing::resident_model(&cfg, &dev);
    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    for round in 0..3 {
        assert_eq!(
            decode(&mut session, &dev)?,
            want_ids,
            "the driver's decode {round} moved its ids"
        );
    }
    assert!(m.pass_plan_cached(&prefill_key()));
    for round in 0..3 {
        assert_eq!(
            direct(&mut session, &dev)?,
            want_logits,
            "direct forward {round} moved its logits"
        );
    }
    assert_two_plans_and_no_drift(m, "driver, then direct");
    Ok(())
}
