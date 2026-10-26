//! A decode or prefill pass whose body fails after its pass opened is
//! aborted, never finished and never left open (`decode.rs`'s `OpenPass`,
//! `MetalDevice::abort_pass`, `DESIGN.md` §9.2aa).
//!
//! `begin_decode_step` can succeed and the forward behind it fail — a
//! kernel refused, an allocation declined, a sticky queue error polled
//! mid-forward — or unwind. A loop that returns through `?` there skips the
//! pass's end: the pass stays current, the next request's `begin_pass` is
//! refused, and `drop_pass_plans` refuses to run. Closing it with the
//! success-path finish is wrong too: the pass may be the one observing a
//! pending key's liveness, and a finish builds that key's plan from the
//! failed body's partial observation.
//!
//! The failure injected here is in the model: the tiny LFM2 runs the decode
//! step's forward, then returns an error or panics, inside the step's pass.
//! The witnesses: the device has no pass open after the failed request, the
//! failed pass's observation built no plan and left no key pending (the key
//! it observed records again on its next sight), and the next request
//! decodes exactly what an uninjected one decodes, its plans still served.
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::model::{CausalLM, Model as ModelTrait, Provisional};
use lloom::nn::{DecodeOptions, IdentityProposer, Session, VarBuilder};
use lloom::profile::RewindableExact;
use lloom::testing::tiny::{TinyCheckpoint, Weights};
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice, PassPlanCounters};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicUsize, Ordering};

type Inner = Model<RewindableExact>;

const EOS: u32 = 47;
const SHORT: [u32; 4] = [1, 2, 3, 4];
const LONG: [u32; 5] = [5, 6, 7, 8, 9];

fn config() -> Config {
    Config {
        vocab_size: 48,
        hidden_size: 128,
        intermediate_size: 256,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_position_embeddings: 128,
        conv_l_cache: 3,
        conv_bias: false,
        layer_types: vec![LayerType::Conv, LayerType::FullAttention, LayerType::Conv],
        tie_embedding: true,
        bos_token_id: None,
        eos_token_id: Some(EOS),
        kv_append: KvAppend::InPlace,
        conv_state: ConvState::RotatingRing { k: 2 },
        conv_prefill_ctx: true,
        ..Default::default()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fail {
    Err,
    Unwind,
}

/// The tiny LFM2 with an injection in its forward: armed with `n`, the
/// `n`-th single-position forward from the arming runs to its end and then
/// fails — an error, or a panic — inside the decode step's pass.
struct FailingBody {
    inner: Inner,
    left: AtomicUsize,
    unwind: std::sync::atomic::AtomicBool,
    failed: AtomicUsize,
}

impl FailingBody {
    fn arm(&self, n: usize, how: Fail) {
        self.unwind.store(how == Fail::Unwind, Ordering::Relaxed);
        self.left.store(n, Ordering::Relaxed);
    }

    fn failed(&self) -> usize {
        self.failed.load(Ordering::Relaxed)
    }
}

impl ModelTrait for FailingBody {
    type Config = Config;
    fn from_weights(_vb: VarBuilder, _cfg: &Config) -> Result<Self> {
        lloom_tensor::bail!("built from the tiny checkpoint")
    }
}

impl CausalLM for FailingBody {
    type Cache = <Inner as CausalLM>::Cache;
    fn new_cache(&self, cfg: &Config, dev: &Device) -> Result<Self::Cache> {
        self.inner.new_cache(cfg, dev)
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        self.inner.embed(tokens)
    }
    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Self::Cache,
    ) -> Result<Tensor> {
        let out = self.inner.forward_embeds(embeds, pos, cache)?;
        if embeds.dim(1)? == 1 {
            let left = self.left.load(Ordering::Relaxed);
            if left > 0 {
                self.left.store(left - 1, Ordering::Relaxed);
                if left == 1 {
                    self.failed.fetch_add(1, Ordering::Relaxed);
                    if self.unwind.load(Ordering::Relaxed) {
                        panic!("injected: the step's forward unwound inside its pass");
                    }
                    lloom_tensor::bail!("injected: the step's forward failed inside its pass")
                }
            }
        }
        Ok(out)
    }
    fn prepare_pass(&self, cache: &mut Self::Cache, seq_len: usize) -> Result<()> {
        self.inner.prepare_pass(cache, seq_len)
    }
    fn note_input_ids(&self, cache: &mut Self::Cache, ids: &[u32]) -> Result<()> {
        self.inner.note_input_ids(cache, ids)
    }
    fn checkpoint_at(&self, cache: &mut Self::Cache, pos: usize) -> Result<()> {
        self.inner.checkpoint_at(cache, pos)
    }
    fn settle_pending(&self, cache: &mut Self::Cache) -> Result<()> {
        self.inner.settle_pending(cache)
    }
    fn provisional(&self, cache: &mut Self::Cache, mode: Provisional) -> Result<Provisional> {
        self.inner.provisional(cache, mode)
    }
    fn settle_step(&self, cache: &mut Self::Cache) -> Result<Option<Tensor>> {
        self.inner.settle_step(cache)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

fn model(dev: &Device) -> Result<(FailingBody, Config)> {
    let cfg = config();
    let inner = TinyCheckpoint::<Lfm2Tiny>::new(cfg.clone(), Weights::default())
        .with_dtype(DType::F16)
        .model::<RewindableExact>(dev)?;
    Ok((
        FailingBody {
            inner,
            left: AtomicUsize::new(0),
            unwind: std::sync::atomic::AtomicBool::new(false),
            failed: AtomicUsize::new(0),
        },
        cfg,
    ))
}

/// One greedy request over its own sequence, through the blocking readback.
fn request(
    model: &FailingBody,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    max: usize,
) -> Result<Vec<u32>> {
    let mut session = Session::new(
        model,
        State::<RewindableExact>::new(true, DType::F16, cfg, dev)?,
    );
    let mut readback = Blocking::new(&[EOS], dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let out = session.decode(
        &mut IdentityProposer,
        &mut readback,
        &mut sampler,
        prompt,
        &DecodeOptions::new(max),
        &mut |_| Ok(()),
    )?;
    Ok(out.ids)
}

/// The request with the injection armed, its failure returned whichever way
/// the body failed.
fn failing_request(
    model: &FailingBody,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    how: Fail,
) -> String {
    model.arm(1, how);
    let before = model.failed();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        request(model, cfg, dev, prompt, 6)
    }));
    model.arm(0, Fail::Err);
    assert_eq!(model.failed(), before + 1, "the injection fired once");
    match (how, outcome) {
        (Fail::Err, Ok(Err(e))) => e.to_string(),
        (Fail::Unwind, Err(panic)) => panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .unwrap_or_else(|| "a panic".into()),
        (how, outcome) => panic!(
            "the {how:?} injection did not fail the request that way: {:?}",
            outcome.map(|r| r.map(|_| ()))
        ),
    }
}

fn metal(dev: &Device) -> &MetalDevice {
    let Device::Metal(m) = dev else {
        unreachable!("a Metal fixture")
    };
    m
}

fn counters(m: &MetalDevice) -> PassPlanCounters {
    m.pass_plan_counters().expect("the plan cache is on")
}

/// A pass observing the liveness a new prefill key's plan is built from
/// fails: its observation builds no plan, the key is not left pending, no
/// pass is left open, and the next request — the same prompt — records the
/// key afresh, builds its plan from complete passes and decodes what an
/// uninjected request decodes, with the other keys' plans still served.
fn an_observing_pass_whose_body_fails_publishes_nothing(how: Fail) -> Result<()> {
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    let (model, cfg) = model(&dev)?;
    m.enable_pass_plans(ArenaLayout::Packed);

    // Warm: the decode key and the short prompt's prefill key are planned.
    let warm = request(&model, &cfg, &dev, &SHORT, 6)?;
    assert!(!warm.is_empty(), "the fixture decodes something");
    let warmed = counters(m);
    assert!(warmed.plans >= 2, "{warmed:?}");

    // The long prompt's prefill is a new key: its first sight records, and
    // the first decode step after it observes what escaped it. That step's
    // body fails.
    let why = failing_request(&model, &cfg, &dev, &LONG, how);
    assert!(why.contains("injected"), "{why}");
    let failed = counters(m);
    eprintln!("BODY_FAILED how={how:?} why={why} warmed={warmed:?} failed={failed:?}");
    assert!(!m.pass_open(), "the failed body left its pass open");
    assert_eq!(failed.aborted, warmed.aborted + 1, "{failed:?}");
    assert_eq!(
        failed.plans, warmed.plans,
        "the failed body's observation was published as a plan"
    );
    assert_eq!(
        failed.observed,
        warmed.observed + 1,
        "the failing pass observed"
    );
    assert_eq!(
        failed.recorded,
        warmed.recorded + 1,
        "the new key's first sight"
    );

    // The next request: the long prompt's key is not pending, so its prefill
    // records it again, the step after observes it, and its plan is built
    // from those two complete passes.
    let after = request(&model, &cfg, &dev, &LONG, 6)?;
    let next = counters(m);
    assert_eq!(next.recorded, failed.recorded + 1, "{next:?}");
    assert_eq!(next.observed, failed.observed + 1, "{next:?}");
    assert_eq!(next.plans, failed.plans + 1, "{next:?}");
    assert_eq!(next.aborted, failed.aborted, "{next:?}");
    assert!(!m.pass_open());
    let reference = request(&model, &cfg, &dev, &LONG, 6)?;
    let served = counters(m);
    assert!(
        served.served > next.served,
        "the plans are served: {served:?}"
    );
    assert_eq!(
        after, reference,
        "the request after a failed body decoded something else"
    );
    eprintln!("BODY_FAILED_NEXT how={how:?} next={next:?} served={served:?} ids={after:?}");
    Ok(())
}

#[test]
fn an_observing_pass_whose_body_errs_publishes_nothing() -> Result<()> {
    an_observing_pass_whose_body_fails_publishes_nothing(Fail::Err)
}

#[test]
fn an_observing_pass_whose_body_unwinds_publishes_nothing() -> Result<()> {
    an_observing_pass_whose_body_fails_publishes_nothing(Fail::Unwind)
}

/// A served pass whose body fails keeps its plan: the next request is
/// served from it, and decodes what an uninjected request decodes.
#[test]
fn a_served_pass_whose_body_fails_keeps_its_plan() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    let (model, cfg) = model(&dev)?;
    m.enable_pass_plans(ArenaLayout::Packed);
    let reference = request(&model, &cfg, &dev, &SHORT, 6)?;
    let warmed = counters(m);
    // The short prompt again: every pass is served. The second step fails.
    model.arm(2, Fail::Err);
    let failed = request(&model, &cfg, &dev, &SHORT, 6);
    model.arm(0, Fail::Err);
    assert!(failed.is_err());
    let after_fail = counters(m);
    assert!(!m.pass_open());
    assert_eq!(after_fail.aborted, warmed.aborted + 1, "{after_fail:?}");
    assert_eq!(after_fail.plans, warmed.plans, "{after_fail:?}");
    assert_eq!(after_fail.evicted, warmed.evicted, "{after_fail:?}");
    let after = request(&model, &cfg, &dev, &SHORT, 6)?;
    let next = counters(m);
    assert_eq!(
        next.recorded, after_fail.recorded,
        "a cached key recorded again"
    );
    assert!(next.served > after_fail.served);
    assert_eq!(after, reference);
    Ok(())
}

/// With the plan cache off, an armed arena recording over a step whose body
/// fails installs nothing from that step: it is armed again, and the next
/// request records over complete steps and decodes what an uninjected
/// request decodes.
#[test]
fn a_recorded_legacy_step_whose_body_fails_installs_no_arena() -> Result<()> {
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    let (model, cfg) = model(&dev)?;
    let reference = request(&model, &cfg, &dev, &SHORT, 6)?;
    // With the cache off every pass is a decode step to the arming, the
    // prefill's included: step 0 is the prefill, steps 1 and 2 the first two
    // decode steps, recorded. The second decode step — the last recorded —
    // fails.
    m.arm_arena_recording(ArenaLayout::Packed, 1, 2);
    model.arm(2, Fail::Err);
    let failed = request(&model, &cfg, &dev, &SHORT, 6);
    model.arm(0, Fail::Err);
    assert!(failed.is_err());
    assert!(
        m.arena_auto_report().is_none(),
        "an arena was installed from a step that failed part-way: {:?}",
        m.arena_auto_report()
    );
    assert!(m.arena().is_none());
    assert!(
        m.arena_recording_excluded().is_none(),
        "the failed step's recording was kept"
    );
    let after = request(&model, &cfg, &dev, &SHORT, 6)?;
    let report = m
        .arena_auto_report()
        .expect("the next request installs the arena");
    eprintln!("LEGACY_ARENA report={report:?}");
    assert_eq!(report.installed_at_step, 2, "{report:?}");
    assert_eq!(after, reference);
    m.clear_arena();
    Ok(())
}
