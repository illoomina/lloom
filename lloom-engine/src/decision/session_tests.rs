//! Public core Session seam, exercised through real paced prefill, no weights.
use lloom::generation::Blocking;
use lloom::nn::{CacheReset, CacheRewind, CausalLM, DecodeOptions, Model, Session, VarBuilder};
use lloom_tensor::{Device, Result, Tensor};
use std::cell::Cell;

#[derive(Default)]
struct Tokens(Vec<u32>);
impl CacheReset for Tokens {
    fn reset(&mut self) {
        self.0.clear();
    }
}
impl CacheRewind for Tokens {
    fn rewind_horizon(&self) -> usize {
        self.0.len()
    }
    fn rewind(&mut self, n: usize) -> Result<()> {
        if n > self.0.len() {
            lloom_tensor::bail!("past history");
        }
        self.0.truncate(self.0.len() - n);
        Ok(())
    }
}

#[derive(Default)]
struct Tiny {
    fail: Cell<bool>,
    calls: Cell<usize>,
}
impl Model for Tiny {
    type Config = ();
    fn from_weights(_: VarBuilder, _: &()) -> Result<Self> {
        Ok(Self::default())
    }
}
impl CausalLM for Tiny {
    type Cache = Tokens;
    fn new_cache(&self, _: &(), _: &Device) -> Result<Tokens> {
        Ok(Tokens::default())
    }
    fn vocab_size(&self) -> usize {
        3
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(tokens.clone())
    }
    fn forward_embeds(&self, tokens: &Tensor, pos: usize, cache: &mut Tokens) -> Result<Tensor> {
        assert_eq!(pos, cache.0.len());
        self.calls.set(self.calls.get() + 1);
        let rows = tokens.to_vec2::<u32>()?;
        cache.0.extend_from_slice(&rows[0]);
        // Fail after cache mutation, as a real forward may do. This test does
        // not pretend the entire prefill transaction was rolled back.
        if self.fail.get() {
            lloom_tensor::bail!("injected forward failure after cache write");
        }
        Tensor::new(
            &[[*cache.0.last().unwrap() as f32, cache.0.len() as f32, 0.0]],
            &Device::Cpu,
        )
    }
}

fn options() -> DecodeOptions {
    DecodeOptions {
        prefill_tile: 2,
        ..DecodeOptions::new(1)
    }
}

#[test]
fn held_logits_follow_last_position_and_reset_prepare_invalidation() -> Result<()> {
    let model = Tiny::default();
    let mut session = Session::new(&model, Tokens::default());
    let mut readback = Blocking::new(&[2], &Device::Cpu)?;
    assert!(session.last_logits().is_none());
    session.prefill(&mut readback, &[1, 2, 3], &options())?;
    assert_eq!(
        model.calls.get(),
        2,
        "existing tile loop, not a raw whole-prompt forward"
    );
    assert_eq!(
        session.last_logits().unwrap().to_vec1::<f32>()?,
        vec![3.0, 3.0, 0.0]
    );
    session.prefill(&mut readback, &[], &options())?;
    assert_eq!(model.calls.get(), 2);
    assert!(session.last_logits().is_some());
    assert!(session.prepare(&[1, 2, 3])?.suffix.is_empty());
    assert!(
        session.last_logits().is_some(),
        "unchanged complete prompt resumes held prediction"
    );
    assert_eq!(session.prepare(&[1, 2, 4])?.suffix, vec![4]);
    assert!(session.last_logits().is_none());
    session.prefill(&mut readback, &[4], &options())?;
    assert_eq!(
        session.last_logits().unwrap().to_vec1::<f32>()?,
        vec![4.0, 3.0, 0.0]
    );
    session.reset();
    assert!(session.last_logits().is_none());
    assert_eq!(session.pos(), 0);
    Ok(())
}

#[test]
fn failed_nonempty_prefill_does_not_expose_previous_prediction() -> Result<()> {
    let model = Tiny::default();
    let mut session = Session::new(&model, Tokens::default());
    let mut readback = Blocking::new(&[2], &Device::Cpu)?;
    session.prefill(&mut readback, &[1, 2], &options())?;
    assert!(session.last_logits().is_some());
    model.fail.set(true);
    assert!(session.prefill(&mut readback, &[3, 4], &options()).is_err());
    assert!(
        session.last_logits().is_none(),
        "stale prediction survived failed prefill"
    );
    assert_eq!(
        session.cache_mut().0.len(),
        4,
        "failure need not roll back cache"
    );
    session.reset();
    model.fail.set(false);
    session.prefill(&mut readback, &[9], &options())?;
    assert_eq!(
        session.last_logits().unwrap().to_vec1::<f32>()?,
        vec![9.0, 1.0, 0.0]
    );
    Ok(())
}

/// Runs the production baseline driver through this crate's normal dependency,
/// unlike lloom's self-dev graph (which deliberately enables fuzzing/profile).
#[test]
fn baseline_instrument_storage_matches_the_actual_shipping_or_diagnostic_graph() -> Result<()> {
    use lloom::generation::{LogitsProcessor, Sampling};
    use lloom::nn::IdentityProposer;
    lloom_tensor::tuning::install(lloom_tensor::tuning::Tuning::default()).unwrap();
    let expected = std::env::var("LLOOM_TEST_EXPECT_PROFILE_CAPABILITY").ok();
    if let Some(expected) = expected.as_deref() {
        assert_eq!(
            lloom_tensor::tuning::PROFILE_INSTRUMENTS,
            match expected {
                "absent" => false,
                "available" => true,
                other => panic!("unknown test capability expectation: {other}"),
            },
            "unexpected transitive diagnostic feature union"
        );
    }
    let model = Tiny::default();
    let mut session = Session::new(&model, Tokens::default());
    let mut readback = Blocking::new(&[2], &Device::Cpu)?;
    let mut sampler = LogitsProcessor::from_sampling(0, Sampling::ArgMax);
    let opts = DecodeOptions {
        prefill_tile: 2,
        ..DecodeOptions::new(3)
    };
    let out = session.decode(
        &mut IdentityProposer,
        &mut readback,
        &mut sampler,
        &[1, 1],
        &opts,
        &mut |_| Ok(()),
    )?;
    assert_eq!(out.ids, vec![1, 1, 1]);
    assert_eq!(out.stats.produced, 3);
    assert!(out.stats.prefill_census.is_none());
    assert!(out.stats.step0_census.is_none());
    match expected.as_deref() {
        Some("absent") => {
            assert!(out.stats.step_ns.is_empty());
            assert_eq!(
                out.stats.step_ns.capacity(),
                0,
                "shipping baseline reserved token-sized diagnostic storage"
            );
        }
        Some("available") => assert_eq!(out.stats.step_ns.len(), 3),
        _ => assert!(out.stats.step_ns.is_empty() || out.stats.step_ns.len() == 3),
    }
    println!(
        "actual_profile_capability={} step_entries={} step_capacity={} ids={:?}",
        lloom_tensor::tuning::PROFILE_INSTRUMENTS,
        out.stats.step_ns.len(),
        out.stats.step_ns.capacity(),
        out.ids
    );
    Ok(())
}
