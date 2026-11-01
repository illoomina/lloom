//! A decode or prefill pass whose body fails after its pass opened is
//! aborted, never finished and never left open (`decode.rs`'s `OpenPass`,
//! `MetalDevice::abort_pass`, `docs/pass-plan-design.md` §24).
//!
//! `begin_decode_step` can succeed and the forward behind it fail — a
//! kernel refused, an allocation declined, a sticky queue error polled
//! mid-forward — or unwind. A loop that returns through `?` there skips the
//! pass's end: the pass stays current, the next request's `begin_pass` is
//! refused, and `drop_pass_plans` refuses to run. Closing it with the
//! success-path finish is wrong too: the pass may be the one observing a
//! pending key's liveness, and a finish builds that key's plan from the
//! failed body's observation.
//!
//! Every driver that brackets a pass does it through `OpenPass`: the single
//! sequence's decode, prefill and verify, the row driver's arrival prefill,
//! ragged step and batched step, the batched and serial speculative
//! verifies, a branch group, and a pass Qwen4-exp opens itself. Each has a
//! test here that fails its body and finds the pass aborted.
//!
//! The failure is injected in the model: [`FailingBody`] runs the armed
//! entry point (a [`Site`]), then returns an error or panics, inside the
//! pass its driver opened; Qwen4-exp's own pass fails in its observer.
//! The witnesses: the device has no pass open after the failed request, the
//! failed pass's observation built no plan and left no key pending (the key
//! it observed records again on its next sight), and the next request
//! decodes exactly what an uninjected one decodes, its plans still served.
//!
//! The LFM2 tests build through `lloom::testing::fixture`: `LLOOM_FIXTURE=real`
//! runs them on LFM2.5-2.6B. The Qwen3.5/3.6 test runs only on a real
//! checkpoint (`LLOOM_PASS_BODY_REAL_QWEN3_5`, a GGUF's first shard).
#![cfg(feature = "metal")]

use lloom::generation::{Blocking, LogitsProcessor, Sampling};
use lloom::models::lfm2::{Config, ConvState, KvAppend, LayerType, Model, State};
use lloom::models::lfm2_testing::Lfm2Tiny;
use lloom::nn::model::{
    BatchedCausalLM, CausalLM, Model as ModelTrait, Provisional, RaggedRow, Row,
};
use lloom::nn::speculator::{
    Anchor, DSpark, Draft, LayerTaps, Probes, SeqId, Speculator, TargetProbe, VerifyRowRef,
};
use lloom::nn::{CheckpointPolicy, DecodeOptions, IdentityProposer, Session, VarBuilder};
use lloom::profile::{CrownExact, PoolSpecExact, Profile, RewindableExact};
use lloom::testing::fixture::Fixture;
use lloom_tensor::metal_backend::{ArenaLayout, MetalDevice, PassPlanCounters};
use lloom_tensor::{DType, Device, Result, Tensor};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

const EOS: u32 = 47;
const SHORT: [u32; 4] = [1, 2, 3, 4];
const LONG: [u32; 5] = [5, 6, 7, 8, 9];

/// The tiny geometry, and the arms every LFM2 test here runs (at the real
/// geometry too).
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
        ..Default::default()
    }
    .with_arms()
}

trait Arms {
    fn with_arms(self) -> Self;
}

impl Arms for Config {
    fn with_arms(mut self) -> Self {
        self.kv_append = KvAppend::InPlace;
        self.conv_state = ConvState::RotatingRing { k: 2 };
        self.conv_prefill_ctx = true;
        self
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fail {
    Err,
    Unwind,
}

/// Which of the model's entry points an armed injection fails: each is the
/// body of one driver's pass.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Site {
    /// `forward` of one position: a decode step.
    Step,
    /// `forward` of more than one position: a prefill.
    Prefill,
    /// `BatchedCausalLM::forward_rows`: a batched decode step.
    Rows,
    /// `BatchedCausalLM::forward_ragged`: a ragged step or a branch group.
    Ragged,
    /// `Probes::verify_and_probe` (either form): a serial verify.
    Verify,
    /// `Probes::verify_rows_enqueue`: a batched verify.
    VerifyRows,
}

/// `inner` with an injection in its entry points: armed with `n` at a
/// [`Site`], the `n`-th call of that site from the arming runs to its end
/// and then fails — an error, or a panic — inside the pass its driver
/// opened. Every other method is the inner model's.
struct FailingBody<M> {
    inner: M,
    site: parking_lot::Mutex<Site>,
    left: AtomicUsize,
    unwind: AtomicBool,
    failed: AtomicUsize,
}

impl<M> FailingBody<M> {
    fn new(inner: M) -> Self {
        Self {
            inner,
            site: parking_lot::Mutex::new(Site::Step),
            left: AtomicUsize::new(0),
            unwind: AtomicBool::new(false),
            failed: AtomicUsize::new(0),
        }
    }

    fn arm(&self, n: usize, how: Fail) {
        self.arm_at(Site::Step, n, how);
    }

    fn arm_at(&self, site: Site, n: usize, how: Fail) {
        *self.site.lock() = site;
        self.unwind.store(how == Fail::Unwind, Ordering::Relaxed);
        self.left.store(n, Ordering::Relaxed);
    }

    fn failed(&self) -> usize {
        self.failed.load(Ordering::Relaxed)
    }

    /// After a forward of `positions`: fail here if this is the armed one.
    fn after_forward(&self, positions: usize) -> Result<()> {
        self.after(if positions == 1 {
            Site::Step
        } else {
            Site::Prefill
        })
    }

    /// After a call of `site`: fail here if this is the armed one.
    fn after(&self, site: Site) -> Result<()> {
        if *self.site.lock() != site {
            return Ok(());
        }
        let left = self.left.load(Ordering::Relaxed);
        if left == 0 {
            return Ok(());
        }
        self.left.store(left - 1, Ordering::Relaxed);
        if left == 1 {
            self.failed.fetch_add(1, Ordering::Relaxed);
            if self.unwind.load(Ordering::Relaxed) {
                panic!("injected: the {site:?} body unwound inside its pass");
            }
            lloom_tensor::bail!("injected: the {site:?} body failed inside its pass")
        }
        Ok(())
    }
}

impl<M: CausalLM> ModelTrait for FailingBody<M> {
    type Config = M::Config;
    fn from_weights(_vb: VarBuilder, _cfg: &M::Config) -> Result<Self> {
        lloom_tensor::bail!("built around a model")
    }
}

impl<M: CausalLM> CausalLM for FailingBody<M> {
    type Cache = M::Cache;
    fn new_cache(&self, cfg: &M::Config, dev: &Device) -> Result<Self::Cache> {
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
        self.after_forward(embeds.dim(1)?)?;
        Ok(out)
    }
    fn forward(&self, tokens: &Tensor, pos: usize, cache: &mut Self::Cache) -> Result<Tensor> {
        let out = self.inner.forward(tokens, pos, cache)?;
        self.after_forward(tokens.dim(1)?)?;
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
    fn eos_gate_cores(&self) -> bool {
        self.inner.eos_gate_cores()
    }
    fn open_checkpoint_request(
        &self,
        cache: &mut Self::Cache,
        policy: CheckpointPolicy,
        end: usize,
    ) -> Result<()> {
        self.inner.open_checkpoint_request(cache, policy, end)
    }
    fn close_checkpoint_request(&self, cache: &mut Self::Cache) -> Result<()> {
        self.inner.close_checkpoint_request(cache)
    }
    fn pin_checkpoint(&self, cache: &mut Self::Cache, pos: Option<usize>) -> Result<Option<usize>> {
        self.inner.pin_checkpoint(cache, pos)
    }
    fn pins_checkpoints(&self, cache: &Self::Cache) -> bool {
        self.inner.pins_checkpoints(cache)
    }
    fn pinned_checkpoint(&self, cache: &Self::Cache) -> Option<usize> {
        self.inner.pinned_checkpoint(cache)
    }
    fn host_feed(
        &self,
        cache: &mut Self::Cache,
    ) -> Result<Option<lloom::nn::ngram_memory::HostFeed>> {
        self.inner.host_feed(cache)
    }
    fn host_feed_end(&self, cache: &mut Self::Cache, published: &[u32]) -> Result<()> {
        self.inner.host_feed_end(cache, published)
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

impl<M: lloom::profile::Shaped> lloom::profile::Shaped for FailingBody<M> {
    type Profile = M::Profile;
}

impl<M: BatchedCausalLM> BatchedCausalLM for FailingBody<M> {
    fn forward_rows(&self, tokens: &Tensor, rows: &mut [Row<'_, Self::Cache>]) -> Result<Tensor> {
        let out = self.inner.forward_rows(tokens, rows)?;
        self.after(Site::Rows)?;
        Ok(out)
    }
    fn forward_ragged(
        &self,
        tokens: &Tensor,
        rows: &mut [RaggedRow<'_, Self::Cache>],
    ) -> Result<Tensor> {
        let out = self.inner.forward_ragged(tokens, rows)?;
        self.after(Site::Ragged)?;
        Ok(out)
    }
}

impl<P: TargetProbe, M: Probes<P>> Probes<P> for FailingBody<M> {
    fn verify_and_probe(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<Vec<u32>>, P::Output)> {
        let out = self
            .inner
            .verify_and_probe(drafts, anchors, cache, eos_ids)?;
        self.after(Site::Verify)?;
        Ok(out)
    }
    fn verify_and_probe_sampled(
        &self,
        drafts: &[Draft],
        anchors: &[Anchor],
        cache: &mut Self::Cache,
        eos_ids: &[u32],
        sampler: &mut LogitsProcessor,
    ) -> Result<(Vec<Vec<u32>>, P::Output)> {
        let out = self
            .inner
            .verify_and_probe_sampled(drafts, anchors, cache, eos_ids, sampler)?;
        self.after(Site::Verify)?;
        Ok(out)
    }
    fn prepare_verify(&self, cache: &mut Self::Cache, t: usize) -> Result<()> {
        self.inner.prepare_verify(cache, t)
    }
    type Pending = M::Pending;
    fn verify_enqueue(
        &self,
        draft: &Draft,
        anchor: &Anchor,
        cache: &mut Self::Cache,
    ) -> Result<Self::Pending> {
        self.inner.verify_enqueue(draft, anchor, cache)
    }
    fn verify_complete(
        &self,
        pending: Self::Pending,
        cache: &mut Self::Cache,
        eos_ids: &[u32],
    ) -> Result<(Vec<u32>, P::Output)> {
        self.inner.verify_complete(pending, cache, eos_ids)
    }
    type PendingRows = M::PendingRows;
    fn verify_rows_enqueue(
        &self,
        rows: &mut [VerifyRowRef<'_, Self::Cache>],
    ) -> Result<Self::PendingRows> {
        let pending = self.inner.verify_rows_enqueue(rows)?;
        self.after(Site::VerifyRows)?;
        Ok(pending)
    }
    fn verify_rows_complete(
        &self,
        pending: Self::PendingRows,
        caches: &mut [&mut Self::Cache],
        eos_sets: &[&[u32]],
    ) -> Result<Vec<(Vec<u32>, P::Output)>> {
        self.inner.verify_rows_complete(pending, caches, eos_sets)
    }
    fn take_probe(&self, cache: &mut Self::Cache) -> Result<P::Output> {
        self.inner.take_probe(cache)
    }
    fn probe_drops(&self, cache: &Self::Cache) -> u64 {
        self.inner.probe_drops(cache)
    }
    fn verifies_anchor_alone(&self) -> bool {
        self.inner.verifies_anchor_alone()
    }
}

type Lfm2Body = FailingBody<Model<RewindableExact>>;

/// What a test here runs on: the fixture, its device, the failing model in
/// profile `P`, and the configuration it was built from.
type Under<P> = (Fixture<Lfm2Tiny>, Device, FailingBody<Model<P>>, Config);

/// The LFM2 under test, at the fixture's geometry with this file's arms,
/// on Metal. `None` (with the fixture's line) when the switch asks for a
/// checkpoint this machine does not have.
fn lfm2(test: &str) -> Result<Option<Under<RewindableExact>>> {
    lfm2_as::<RewindableExact>(test, |_| {})
}

/// [`lfm2`] in profile `P`, with `arms` applied over this file's.
fn lfm2_as<P: Profile>(test: &str, arms: impl FnOnce(&mut Config)) -> Result<Option<Under<P>>> {
    let Some(fx) = Fixture::<Lfm2Tiny>::new(test, config())? else {
        return Ok(None);
    };
    let dev = Device::new_metal(0)?;
    let cfg = fx.config(|c| {
        *c = c.clone().with_arms();
        arms(c)
    });
    let inner = fx.model_in::<P>(&cfg, Some(DType::F16), &dev)?;
    Ok(Some((fx, dev, FailingBody::new(inner), cfg)))
}

/// One greedy request over `cache`, through the blocking readback.
fn decode_with<M: CausalLM>(
    model: &M,
    cache: M::Cache,
    dev: &Device,
    prompt: &[u32],
    max: usize,
    eos: u32,
) -> Result<Vec<u32>> {
    let mut session = Session::new(model, cache);
    let mut readback = Blocking::new(&[eos], dev)?;
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

/// One greedy LFM2 request over its own sequence.
fn request(
    model: &Lfm2Body,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    max: usize,
) -> Result<Vec<u32>> {
    let cache = State::<RewindableExact>::new(true, DType::F16, cfg, dev)?;
    decode_with(model, cache, dev, prompt, max, EOS)
}

/// Run `request` with the injection armed at the first single-position
/// forward, and return its failure whichever way the body failed.
fn failing<T>(
    model: &FailingBody<impl CausalLM>,
    how: Fail,
    request: impl FnOnce() -> Result<T>,
) -> String {
    model.arm(1, how);
    let before = model.failed();
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(request));
    model.arm(0, Fail::Err);
    assert_eq!(model.failed(), before + 1, "the injection fired once");
    match (how, outcome) {
        (Fail::Err, Ok(Err(e))) => e.to_string(),
        (Fail::Unwind, Err(panic)) => panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "a panic".into()),
        (how, outcome) => panic!(
            "the {how:?} injection did not fail the request that way: {:?}",
            outcome.map(|r| r.map(|_| ()))
        ),
    }
}

fn failing_request(
    model: &Lfm2Body,
    cfg: &Config,
    dev: &Device,
    prompt: &[u32],
    how: Fail,
) -> String {
    failing(model, how, || request(model, cfg, dev, prompt, 6))
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
fn an_observing_pass_whose_body_fails_publishes_nothing(test: &str, how: Fail) -> Result<()> {
    let Some((_fx, dev, model, cfg)) = lfm2(test)? else {
        return Ok(());
    };
    let m = metal(&dev);
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
    an_observing_pass_whose_body_fails_publishes_nothing("pass_body_observing_errs", Fail::Err)
}

#[test]
fn an_observing_pass_whose_body_unwinds_publishes_nothing() -> Result<()> {
    an_observing_pass_whose_body_fails_publishes_nothing(
        "pass_body_observing_unwinds",
        Fail::Unwind,
    )
}

/// A served pass whose body fails keeps its plan: the next request is
/// served from it, and decodes what an uninjected request decodes.
#[test]
fn a_served_pass_whose_body_fails_keeps_its_plan() -> Result<()> {
    let Some((_fx, dev, model, cfg)) = lfm2("pass_body_served")? else {
        return Ok(());
    };
    let m = metal(&dev);
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
    let Some((_fx, dev, model, cfg)) = lfm2("pass_body_legacy")? else {
        return Ok(());
    };
    let m = metal(&dev);
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

/// After a call whose armed body failed inside a driver's pass: the call
/// failed with the injection, no pass is left open, and exactly that pass
/// was aborted — not finished, so nothing it recorded or observed became a
/// plan. Returns the counters after it.
fn assert_aborted<T>(
    m: &MetalDevice,
    before: PassPlanCounters,
    out: Result<T>,
    what: &str,
) -> PassPlanCounters {
    let err = match out {
        Ok(_) => panic!("{what}: the armed body did not fail the call"),
        Err(e) => e.to_string(),
    };
    assert!(err.contains("injected"), "{what}: {err}");
    assert!(!m.pass_open(), "{what}: the failed body left its pass open");
    let after = counters(m);
    eprintln!("SITE_ABORTED {what} why={err} before={before:?} after={after:?}");
    assert_eq!(
        after.aborted,
        before.aborted + 1,
        "{what}: the failed pass was finished, not aborted: {after:?}"
    );
    after
}

/// A drafter that proposes a constant block for `n` sequences: the target's
/// correction advances every stream, so each round runs a verify pass.
struct Garbage {
    n: usize,
    k: usize,
}

impl Speculator for Garbage {
    type Probe = LayerTaps<5>;
    type Family = DSpark;
    fn n_seq(&self) -> usize {
        self.n
    }
    fn max_draft(&self) -> usize {
        self.k
    }
    fn begin(&mut self, _: SeqId, _: &[u32]) -> Result<()> {
        Ok(())
    }
    fn draft(&mut self, active: &[SeqId]) -> Result<Vec<Draft>> {
        Ok(active
            .iter()
            .map(|_| Draft::host(vec![0; self.k]))
            .collect())
    }
    fn observe(
        &mut self,
        _: SeqId,
        _: &<LayerTaps<5> as TargetProbe>::Output,
        _: &[u32],
    ) -> Result<()> {
        Ok(())
    }
}

/// The taps `LayerTaps<5>` reads, on layers the fixture has.
fn with_taps(c: &mut Config) {
    c.tap_layers = vec![0, 1, 2, 0, 1];
    c.tap_dtype = DType::F32;
}

/// A branch group's ragged pass (`branch::ragged_suffixes`) whose body fails
/// is aborted: it was the end skipped on an error, which left the pass open.
#[test]
fn a_branch_groups_pass_whose_body_fails_is_aborted() -> Result<()> {
    use lloom::nn::branch::{prefill_trunk, ragged_suffixes};
    // Branching adopts a trunk on the Cat arm alone, outside the rewindable arm.
    let Some((_fx, dev, model, cfg)) =
        lfm2_as::<CrownExact>("pass_body_branch", |c| c.kv_append = KvAppend::Cat)?
    else {
        return Ok(());
    };
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let cache = || State::<CrownExact>::new(true, DType::F16, &cfg, &dev);
    let run = || -> Result<Vec<Vec<f32>>> {
        let mut rb = Blocking::new(&[EOS], &dev)?;
        let opts = DecodeOptions::new(1);
        let trunk = prefill_trunk(&model, cache()?, &mut rb, &LONG, &opts)?;
        let suffixes: [&[u32]; 2] = [&[10, 11], &[12]];
        let logits = ragged_suffixes(&model, &trunk, cache, &mut rb, &suffixes, &opts)?;
        logits
            .iter()
            .map(|t| t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>())
            .collect()
    };
    let reference = run()?;
    let before = counters(m);
    model.arm_at(Site::Ragged, 1, Fail::Err);
    let failed = run();
    model.arm(0, Fail::Err);
    assert_aborted(m, before, failed, "branch group");
    assert_eq!(
        run()?,
        reference,
        "the branches after a failed group read other logits"
    );
    Ok(())
}

/// One batched request over three rows under the row driver, greedy:
/// every row's ids.
fn batched<P: Profile>(
    model: &FailingBody<Model<P>>,
    cfg: &Config,
    dev: &Device,
    chunk: usize,
) -> Result<Vec<Vec<u32>>> {
    let prompts: [&[u32]; 3] = [&LONG, &SHORT, &[20, 21, 22, 23, 24, 25, 26]];
    let mut sessions: Vec<Session<'_, FailingBody<Model<P>>>> = prompts
        .iter()
        .map(|_| {
            Ok(Session::new(
                model,
                State::<P>::new(true, DType::F16, cfg, dev)?,
            ))
        })
        .collect::<Result<_>>()?;
    let opts = DecodeOptions {
        lookahead: 1,
        prefill_chunk: chunk,
        ..DecodeOptions::new(5)
    };
    let outs = Session::decode_batched_admitted(
        &mut sessions,
        &mut LogitsProcessor::from_sampling(0, Sampling::ArgMax),
        &prompts,
        &[0, 1, 2],
        &opts,
        &[u32::MAX],
        dev,
        &mut |_, _| Ok(()),
    )?;
    Ok(outs.into_iter().map(|o| o.ids).collect())
}

/// The row driver's chunked step (`forward_ragged_live`, prompt pieces
/// beside the decode rows) whose body fails is aborted: it was the end
/// skipped on an error.
#[test]
fn a_ragged_row_step_whose_body_fails_is_aborted() -> Result<()> {
    let Some((_fx, dev, model, cfg)) = lfm2("pass_body_ragged_rows")? else {
        return Ok(());
    };
    let m = metal(&dev);
    // The pool path's ids are the reference; two clean runs under plans
    // (recorded, then served) must read them before any body fails.
    let reference = batched(&model, &cfg, &dev, 4)?;
    m.enable_pass_plans(ArenaLayout::Packed);
    for round in 0..2 {
        let clean = batched(&model, &cfg, &dev, 4)?;
        eprintln!(
            "CLEAN ragged row step round={round} same_as_pool={} counters={:?}",
            clean == reference,
            counters(m)
        );
        assert_eq!(
            clean, reference,
            "ragged row step: clean round {round} under plans"
        );
    }
    let before = counters(m);
    model.arm_at(Site::Ragged, 2, Fail::Err);
    let failed = batched(&model, &cfg, &dev, 4);
    model.arm(0, Fail::Err);
    assert_aborted(m, before, failed, "ragged row step");
    assert_eq!(batched(&model, &cfg, &dev, 4)?, reference);
    Ok(())
}

/// The row driver's batched decode step (`forward_live`, a step with no
/// prompt piece in it) whose body fails is aborted: it was the end skipped on
/// an error.
#[test]
fn a_batched_decode_step_whose_body_fails_is_aborted() -> Result<()> {
    let Some((_fx, dev, model, cfg)) = lfm2("pass_body_batched_decode")? else {
        return Ok(());
    };
    let m = metal(&dev);
    // The pool path's ids are the reference; two clean runs under plans
    // (recorded, then served) must read them before any body fails.
    let reference = batched(&model, &cfg, &dev, 4)?;
    m.enable_pass_plans(ArenaLayout::Packed);
    for round in 0..2 {
        let clean = batched(&model, &cfg, &dev, 4)?;
        eprintln!(
            "CLEAN batched decode step round={round} same_as_pool={} counters={:?}",
            clean == reference,
            counters(m)
        );
        assert_eq!(
            clean, reference,
            "batched decode step: clean round {round} under plans"
        );
    }
    let before = counters(m);
    model.arm_at(Site::Rows, 2, Fail::Err);
    let failed = batched(&model, &cfg, &dev, 4);
    model.arm(0, Fail::Err);
    assert_aborted(m, before, failed, "batched decode step");
    assert_eq!(batched(&model, &cfg, &dev, 4)?, reference);
    Ok(())
}

/// A row's whole prompt at its arrival (`admit_whole`) whose prefill body
/// fails is aborted: it was the success end, run on the failed prefill.
#[test]
fn a_rows_arrival_prefill_whose_body_fails_is_aborted() -> Result<()> {
    let Some((_fx, dev, model, cfg)) = lfm2("pass_body_rows_prefill")? else {
        return Ok(());
    };
    let m = metal(&dev);
    // The pool path's ids are the reference; two clean runs under plans
    // (recorded, then served) must read them before any body fails.
    let reference = batched(&model, &cfg, &dev, 0)?;
    m.enable_pass_plans(ArenaLayout::Packed);
    for round in 0..2 {
        let clean = batched(&model, &cfg, &dev, 0)?;
        eprintln!(
            "CLEAN arrival prefill round={round} same_as_pool={} counters={:?}",
            clean == reference,
            counters(m)
        );
        assert_eq!(
            clean, reference,
            "arrival prefill: clean round {round} under plans"
        );
    }
    let before = counters(m);
    model.arm_at(Site::Prefill, 2, Fail::Err);
    let failed = batched(&model, &cfg, &dev, 0);
    model.arm(0, Fail::Err);
    assert_aborted(m, before, failed, "arrival prefill");
    assert_eq!(batched(&model, &cfg, &dev, 0)?, reference);
    Ok(())
}

/// The serial speculative round's verify (`Session::speculative_round_keyed`)
/// whose body fails is aborted: it was the success end, run on the failed
/// verify.
#[test]
fn a_serial_verify_whose_body_fails_is_aborted() -> Result<()> {
    // A K = 1 window: the rotating ring of 2 rewinds it.
    let Some((_fx, dev, model, cfg)) = lfm2_as::<RewindableExact>("pass_body_verify", with_taps)?
    else {
        return Ok(());
    };
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let run = || -> Result<Vec<u32>> {
        let mut session = Session::new(
            &model,
            State::<RewindableExact>::new(true, DType::F16, &cfg, &dev)?,
        );
        let mut spec = Garbage { n: 1, k: 1 };
        let mut rb = Blocking::new(&[u32::MAX], &dev)?;
        let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
        let out = session.decode_speculative(
            lloom::profile::RewindProof::of(),
            &mut spec,
            &mut rb,
            &mut sampler,
            &LONG,
            &DecodeOptions::new(6),
            &[u32::MAX],
            &mut |_: &[u32]| Ok(()),
        )?;
        Ok(out.ids)
    };
    let reference = run()?;
    let before = counters(m);
    model.arm_at(Site::Verify, 2, Fail::Err);
    let failed = run();
    model.arm(0, Fail::Err);
    assert_aborted(m, before, failed, "serial verify");
    assert_eq!(run()?, reference);
    Ok(())
}

/// The batched speculative round's verify (`verify_rows_enqueue`) whose
/// body fails is aborted: it was the success end, run on the failed verify.
#[test]
fn a_batched_verify_whose_body_fails_is_aborted() -> Result<()> {
    use lloom::models::lfm2::{AttnImpl, BatchPool};
    // The batched verify's arms: the page pool, FlashDecoding over it.
    let Some((_fx, dev, model, cfg)) = lfm2_as::<PoolSpecExact>("pass_body_batched_verify", |c| {
        with_taps(c);
        c.kv_append = KvAppend::Paged;
        c.attn_impl = AttnImpl::FlashDecoding;
        c.flash_page_size = 32;
        c.flash_pages_per_chunk = 1;
    })?
    else {
        return Ok(());
    };
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let prompts: [&[u32]; 2] = [&LONG, &SHORT];
    let run = || -> Result<Vec<Vec<u32>>> {
        let pool = BatchPool::new(&cfg, &dev, prompts.len(), 128)?;
        let mut sessions: Vec<Session<'_, FailingBody<Model<PoolSpecExact>>>> = prompts
            .iter()
            .map(|_| {
                Ok(Session::new(
                    &model,
                    State::<PoolSpecExact>::for_batch(true, DType::F16, &cfg, &dev, &pool)?,
                ))
            })
            .collect::<Result<_>>()?;
        let mut spec = Garbage {
            n: prompts.len(),
            k: 1,
        };
        let outs = Session::decode_spec_batched(
            &mut sessions,
            lloom::profile::RewindProof::of(),
            &mut spec,
            &prompts,
            &DecodeOptions::new(6),
            &[u32::MAX],
            &dev,
            &mut |_, _| Ok(()),
        )?;
        Ok(outs.into_iter().map(|o| o.ids).collect())
    };
    let reference = run()?;
    let before = counters(m);
    model.arm_at(Site::VerifyRows, 2, Fail::Err);
    let failed = run();
    model.arm(0, Fail::Err);
    assert_aborted(m, before, failed, "batched verify");
    assert_eq!(run()?, reference);
    Ok(())
}

/// An observer that fails at one layer's carry: a body failure inside a
/// pass the model opened itself.
struct FailingObserver {
    at: usize,
    armed: bool,
}

impl lloom::nn::observer::ForwardObserver for FailingObserver {
    fn at<P: lloom::nn::observer::HookPoint>(&mut self, layer: usize, _: &Tensor) -> Result<()> {
        if self.armed && layer == self.at {
            lloom_tensor::bail!("injected: layer {layer}'s observer failed inside the model's pass")
        }
        Ok(())
    }
    fn wants<P: lloom::nn::observer::HookPoint>(&self, layer: usize) -> bool {
        self.armed && layer == self.at
    }
}

/// A pass Qwen4-exp opens itself (no driver holds one: a direct forward, as
/// the logprobs path takes), whose body fails at a layer, is aborted: it was
/// the end skipped on an error, which left the model's pass open. Both of
/// the model's pass shapes: a prefill chunk and a decode step.
#[test]
fn a_model_opened_pass_whose_body_fails_is_aborted() -> Result<()> {
    use lloom::models::qwen4exp::{testing, Config as QConfig};
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    m.enable_pass_plans(ArenaLayout::Packed);
    let mut qcfg = QConfig::tiny();
    qcfg.n_experts = 6;
    qcfg.n_experts_per_tok = 2;
    let model = testing::resident_model::<CrownExact>(&qcfg, &dev);
    let prompt: [u32; 9] = [3, 5, 2, 8, 8, 1, 9, 3, 5];
    let ids = |v: &[u32]| -> Result<Tensor> { Tensor::new(v, &dev)?.unsqueeze(0) };
    let logits = |t: &Tensor| -> Result<Vec<f32>> {
        t.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
    };
    let run = |obs: &mut FailingObserver| -> Result<(Vec<f32>, Vec<f32>)> {
        let mut cache = model.new_cache(&qcfg, &dev)?;
        let embeds = model.embed(&ids(&prompt)?)?;
        // Read each pass's logits before the next pass: a served plan may
        // hand the next pass the memory they live in.
        let pre = logits(&model.forward_embeds(&embeds, &mut cache, obs)?)?;
        let step = logits(&model.decode(&ids(&[4])?, &mut cache, obs)?)?;
        Ok((pre, step))
    };
    let reference = run(&mut FailingObserver {
        at: 1,
        armed: false,
    })?;
    for what in ["prefill chunk", "decode step"] {
        let before = counters(m);
        let mut cache = model.new_cache(&qcfg, &dev)?;
        let embeds = model.embed(&ids(&prompt)?)?;
        let mut obs = FailingObserver {
            at: 1,
            armed: what == "prefill chunk",
        };
        let failed = match model.forward_embeds(&embeds, &mut cache, &mut obs) {
            Err(e) => Err(e),
            Ok(_) => {
                obs.armed = true;
                model.decode(&ids(&[4])?, &mut cache, &mut obs)
            }
        };
        assert_aborted(m, before, failed, what);
        assert_eq!(
            run(&mut FailingObserver {
                at: 1,
                armed: false
            })?,
            reference,
            "{what}: the next request read other logits"
        );
    }
    Ok(())
}

/// `n` ids of the vocabulary's ordinary range, from `seed`.
fn real_ids(vocab: usize, n: usize, seed: u64) -> Vec<u32> {
    let (lo, span) = (100u64, (vocab as u64).saturating_sub(1100).max(1));
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (lo + (x >> 33) % span) as u32
        })
        .collect()
}

/// The failing body and a turn-checkpoint restore on a real Qwen3.5/3.6
/// checkpoint (`LLOOM_PASS_BODY_REAL_QWEN3_5`, a GGUF's first shard; skipped,
/// with a line, where unset), on the generic decoder whose state carries
/// GatedDeltaNet's recurrent S and conv window:
/// - a forced body error in the pass observing a new prefill key: no pass is
///   left open, one abort, no plan built from it, and the next request
///   decodes what an uninjected request decodes;
/// - a forced restore: a session that decoded prompt `a` serves `a + b` by
///   restoring the checkpoint at `a`'s end, and decodes what a fresh request
///   over `a + b` decodes through the same two prefill forwards.
#[test]
fn a_failing_body_and_a_restore_on_a_real_qwen3_5_checkpoint() -> Result<()> {
    use lloom::models::qwen3_5::{LoadOptions, Qwen3_5};
    use lloom::nn::ReuseAction;
    use lloom::profile::CrownExact;
    let Some(path) = std::env::var_os("LLOOM_PASS_BODY_REAL_QWEN3_5") else {
        eprintln!("PASS-BODY-REAL qwen3_5 SKIPPED: LLOOM_PASS_BODY_REAL_QWEN3_5 unset");
        return Ok(());
    };
    let dev = Device::new_metal(0)?;
    let m = metal(&dev);
    let shards = lloom::hub::gguf_shards(std::path::Path::new(&path));
    let opts = LoadOptions {
        kv_capacity: 2048,
        checkpoints: 2,
        purpose: lloom::nn::attention::route::ExecutionPurpose::Production,
        ..LoadOptions::default()
    };
    let t = std::time::Instant::now();
    let model = FailingBody::new(Qwen3_5::<CrownExact>::from_gguf_shards(
        &shards, &dev, &opts,
    )?);
    let cfg = model.inner.config().clone();
    eprintln!(
        "PASS-BODY-REAL qwen3_5 loaded {} in {:.1} s",
        std::path::Path::new(&path).display(),
        t.elapsed().as_secs_f64()
    );
    m.enable_pass_plans(ArenaLayout::Packed);
    let eos = (cfg.vocab_size - 1) as u32;
    let (a, b, c) = (
        real_ids(cfg.vocab_size, 48, 1),
        real_ids(cfg.vocab_size, 16, 2),
        real_ids(cfg.vocab_size, 40, 3),
    );
    let request = |prompt: &[u32]| -> Result<Vec<u32>> {
        decode_with(&model, model.new_cache(&cfg, &dev)?, &dev, prompt, 8, eos)
    };

    // The forced body error. `a` plans the decode key; `c`'s prefill is a new
    // key, and the first decode step after it observes it and fails.
    let warm = request(&a)?;
    let warmed = counters(m);
    let why = failing(&model, Fail::Err, || request(&c));
    let failed = counters(m);
    eprintln!("PASS-BODY-REAL qwen3_5 failed: why={why} warmed={warmed:?} failed={failed:?}");
    assert!(why.contains("injected"), "{why}");
    assert!(!m.pass_open(), "the failed body left its pass open");
    assert_eq!(failed.aborted, warmed.aborted + 1, "{failed:?}");
    assert_eq!(
        failed.plans, warmed.plans,
        "the failed observation became a plan"
    );
    let after = request(&c)?;
    let reference = request(&c)?;
    let next = counters(m);
    eprintln!("PASS-BODY-REAL qwen3_5 next: ids={after:?} counters={next:?}");
    assert_eq!(next.aborted, failed.aborted);
    assert!(
        next.plans > failed.plans,
        "the next request built the key's plan"
    );
    assert_eq!(
        after, reference,
        "the request after a failed body decoded something else"
    );

    // The forced restore.
    let mut session = Session::new(&model, model.new_cache(&cfg, &dev)?);
    session.cache_mut().checkpoint_at(a.len());
    let mut readback = Blocking::new(&[eos], &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let first = session.decode(
        &mut IdentityProposer,
        &mut readback,
        &mut sampler,
        &a,
        &DecodeOptions::new(8),
        &mut |_| Ok(()),
    )?;
    assert_eq!(first.ids, warm, "the session's first turn is the request's");
    let full: Vec<u32> = a.iter().chain(&b).copied().collect();
    let prepared = session.prepare(&full)?;
    eprintln!("PASS-BODY-REAL qwen3_5 restore: {prepared:?}");
    assert_eq!(prepared.reused, a.len(), "{prepared:?}");
    assert!(
        matches!(prepared.action, ReuseAction::Rewound(_)),
        "{prepared:?}"
    );
    let restored = session.decode(
        &mut IdentityProposer,
        &mut readback,
        &mut sampler,
        &prepared.suffix,
        &DecodeOptions::new(8),
        &mut |_| Ok(()),
    )?;
    let mut split = DecodeOptions::new(8);
    split.prefill_ubatch = a.len();
    let mut fresh = Session::new(&model, model.new_cache(&cfg, &dev)?);
    let mut readback = Blocking::new(&[eos], &dev)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let fresh = fresh.decode(
        &mut IdentityProposer,
        &mut readback,
        &mut sampler,
        &full,
        &split,
        &mut |_| Ok(()),
    )?;
    eprintln!(
        "PASS-BODY-REAL qwen3_5 restored={:?} fresh={:?} fresh_units={}",
        restored.ids, fresh.ids, fresh.stats.prefill_units
    );
    assert_eq!(
        fresh.stats.prefill_units, 2,
        "the fresh request's two forwards"
    );
    assert_eq!(
        restored.ids, fresh.ids,
        "the restored state decoded something the same forwards from scratch did not"
    );
    Ok(())
}

#[path = "support/pass_input_route.rs"]
mod pass_input_route;
