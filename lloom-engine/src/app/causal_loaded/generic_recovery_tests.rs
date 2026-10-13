//! Actual CausalEngine error/retry paths, with a deliberately faulting inert CPU
//! model. V8, max3positions, at most4forwards/test; not Loaded/native admission.
use super::*;
use crate::decision::{CausalLabels, CausalTask, Question};
use lloom::nn::{Model, VarBuilder};
use lloom_tensor::Tensor;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

#[derive(Default)]
struct Counts {
    caches: AtomicUsize,
    forwards: AtomicUsize,
    resets: AtomicUsize,
    fences: AtomicUsize,
    successful_fences: AtomicUsize,
    fail_forward: AtomicBool,
    refuse_fence: AtomicBool,
    panic_fence: AtomicBool,
}
struct FaultModel(Arc<Counts>);
struct FaultCache {
    pos: usize,
    serial: usize,
    counts: Arc<Counts>,
}
impl CacheReset for FaultCache {
    fn reset(&mut self) {
        assert_eq!(self.serial, 0, "only the held cache is reset by retry");
        assert!(
            self.counts.successful_fences.load(Ordering::Relaxed)
                > self.counts.resets.load(Ordering::Relaxed),
            "reset preceded successful family recovery"
        );
        self.counts.resets.fetch_add(1, Ordering::Relaxed);
        self.pos = 0;
    }
}
impl CacheRewind for FaultCache {
    fn rewind_horizon(&self) -> usize {
        0
    }
    fn rewind(&mut self, n: usize) -> Result<()> {
        if n != 0 {
            lloom_tensor::bail!("fault fixture has no rewind window")
        }
        Ok(())
    }
}
impl RetainState for FaultCache {
    type Retained = usize;
    fn retain_state(&mut self) -> Result<usize> {
        Ok(self.pos)
    }
    fn restore_retained(&mut self, saved: &usize) -> Result<()> {
        self.pos = *saved;
        Ok(())
    }
}
impl Model for FaultModel {
    type Config = ();
    fn from_weights(_: VarBuilder, _: &()) -> Result<Self> {
        lloom_tensor::bail!("fault fixture is never loaded from weights")
    }
}
impl CausalLM for FaultModel {
    type Cache = FaultCache;
    fn new_cache(&self, _: &(), device: &Device) -> Result<FaultCache> {
        assert!(device.is_cpu());
        let serial = self.0.caches.fetch_add(1, Ordering::Relaxed);
        assert!(serial < 4, "fixture cache bound exceeded");
        Ok(FaultCache {
            pos: 0,
            serial,
            counts: Arc::clone(&self.0),
        })
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(tokens.clone())
    }
    fn forward_embeds(
        &self,
        tokens: &Tensor,
        pos: usize,
        cache: &mut FaultCache,
    ) -> Result<Tensor> {
        let call = self.0.forwards.fetch_add(1, Ordering::Relaxed);
        assert!(call < 4, "fixture forward bound exceeded");
        if pos != cache.pos {
            lloom_tensor::bail!("fault cache not reset before retry")
        }
        cache.pos += tokens.dim(1)?;
        assert!(cache.pos <= 3);
        if self.0.fail_forward.swap(false, Ordering::Relaxed) {
            lloom_tensor::bail!("intentional first-forward failure after cache write")
        }
        Tensor::from_vec(
            vec![1f32, 0., -1., -1., -1., -1., -1., -1.],
            (1, 8),
            &Device::Cpu,
        )
    }
    fn vocab_size(&self) -> usize {
        8
    }
}
struct FamilyRecovery(Arc<Counts>);
impl Recovery<FaultModel> for FamilyRecovery {
    fn before_reuse(&self, model: &FaultModel, device: &Device) -> Result<()> {
        assert!(
            Arc::ptr_eq(&self.0, &model.0),
            "recovery received another model"
        );
        assert!(device.is_cpu());
        let call = self.0.fences.fetch_add(1, Ordering::Relaxed);
        assert!(call < 3, "fixture recovery bound exceeded");
        if self.0.panic_fence.swap(false, Ordering::Relaxed) {
            panic!("intentional family recovery unwind");
        }
        if self.0.refuse_fence.swap(false, Ordering::Relaxed) {
            lloom_tensor::bail!("intentional family recovery refusal")
        }
        self.0.successful_fences.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}
fn engine(recovery: bool) -> Result<(CausalEngine<FaultModel>, Arc<Counts>)> {
    let counts = Arc::new(Counts::default());
    counts.fail_forward.store(true, Ordering::Relaxed);
    let model = FaultModel(Arc::clone(&counts));
    let cache = model.new_cache(&(), &Device::Cpu)?;
    // Move the existing bounded fixture's tokenizer; no tokenizer copy or model
    // loading. Other inert fixture fields are dropped without submitted work.
    let tokenizer = super::tests::engine()?.tokenizer;
    let engine = CausalEngine {
        model,
        config: (),
        device: Device::Cpu,
        capacity: 3,
        tokenizer,
        policy: None,
        decode: DecodeConfig::default(),
        recovery: recovery.then(|| {
            Box::new(FamilyRecovery(Arc::clone(&counts))) as Box<dyn Recovery<FaultModel>>
        }),
        vision: None,
        ragged: None,
        chat: Mutex::new(Some(Chat {
            sequence: Some(Sequence::new(cache)),
            rings: Some(RingPool::new()),
            extra: (),
            needs_recovery: false,
        })),
    };
    Ok((engine, counts))
}
fn params() -> DecodeParams {
    DecodeParams {
        // Zero means unbounded. Finite1 + greedy EOS0 on Blocking stops
        // before a sampled-token forward: only the successful prompt is held.
        max_tokens: 1,
        eos_ids: vec![0],
        readback: ReadbackArm::Blocking,
        ..DecodeParams::default()
    }
}
fn held(engine: &CausalEngine<FaultModel>, needs_recovery: bool, tokens: &[u32], cache_pos: usize) {
    let held = engine.chat.lock();
    let held = held.as_ref().expect("actual owner lost held custody");
    assert_eq!(held.needs_recovery, needs_recovery);
    let seq = held.sequence.as_ref().unwrap();
    assert_eq!(seq.tokens(), tokens);
    assert_eq!(
        seq.pos(),
        tokens.len(),
        "held Sequence position changed on error"
    );
    assert_eq!(seq.cache().pos, cache_pos);
    assert_eq!(seq.cache().serial, 0, "retry replaced the held cache");
    assert!(held.rings.is_some());
}
fn request() -> Result<CausalRequest> {
    let question = Question::Choice {
        instructions: OrderedJson::String("choose".into()),
        criteria: vec![("a".into(), None), ("b".into(), None)],
    };
    let labels = CausalLabels::systemone(&question, |code| match code.trim() {
        "a" => Ok(vec![0]),
        "b" => Ok(vec![1]),
        _ => lloom_tensor::bail!("unexpected fault fixture label"),
    })?;
    CausalRequest::new(
        vec![CausalTask::new(
            "q".into(),
            question,
            vec![2, 3],
            labels.into_readout(),
            1.0.try_into()?,
        )?],
        &[2],
        3,
    )
}

#[test]
fn actual_first_forward_error_keeps_custody_and_refusal_precedes_retry_reset() -> Result<()> {
    let (engine, counts) = engine(true)?;
    let error = engine.complete(&[2], &params(), &mut |_| true).unwrap_err();
    assert!(error
        .to_string()
        .contains("first-forward failure after cache write"));
    held(&engine, true, &[], 1); // Sequence pos/log did not advance with failed forward.
    counts.refuse_fence.store(true, Ordering::Relaxed);
    let error = engine.complete(&[3], &params(), &mut |_| true).unwrap_err();
    assert!(error.to_string().contains("family recovery refusal"));
    held(&engine, true, &[], 1);
    assert_eq!(counts.forwards.load(Ordering::Relaxed), 1);
    assert_eq!(counts.resets.load(Ordering::Relaxed), 0);
    let decoded = engine.complete(&[3], &params(), &mut |_| true)?;
    assert_eq!(decoded.stop, crate::Stop::Eos);
    assert!(
        decoded.tokens.is_empty(),
        "sampled EOS was returned as output"
    );
    held(&engine, false, &[3], 1);
    assert_eq!(counts.fences.load(Ordering::Relaxed), 2);
    assert_eq!(counts.resets.load(Ordering::Relaxed), 1);
    // Unmarked checkout takes no additional family fence.
    drop(engine.checkout()?);
    assert_eq!(counts.fences.load(Ordering::Relaxed), 2);
    Ok(())
}

#[test]
fn actual_decision_failure_and_success_preserve_marker_until_held_decode_reset() -> Result<()> {
    let (engine, counts) = engine(true)?;
    let request = request()?;
    let error = engine.decide(&request).unwrap_err();
    assert!(error
        .to_string()
        .contains("first-forward failure after cache write"));
    held(&engine, true, &[], 0); // Fresh decision cache never replaced held chat.
    assert_eq!(engine.decide(&request)?.answers().len(), 1);
    held(&engine, true, &[], 0); // Successful Held operation cannot clear prior marker.
    assert_eq!(counts.fences.load(Ordering::Relaxed), 1);
    assert_eq!(counts.resets.load(Ordering::Relaxed), 0);
    let decoded = engine.complete(&[3], &params(), &mut |_| true)?;
    assert_eq!(decoded.stop, crate::Stop::Eos);
    assert!(
        decoded.tokens.is_empty(),
        "sampled EOS was returned as output"
    );
    held(&engine, false, &[3], 1);
    assert_eq!(counts.fences.load(Ordering::Relaxed), 2);
    assert_eq!(counts.resets.load(Ordering::Relaxed), 1);
    assert_eq!(counts.caches.load(Ordering::Relaxed), 3);
    Ok(())
}

#[test]
fn actual_recovery_unwind_restores_marker_and_legacy_contract_stays_explicit() -> Result<()> {
    let (engine, counts) = engine(true)?;
    assert!(engine.complete(&[2], &params(), &mut |_| true).is_err());
    counts.panic_fence.store(true, Ordering::Relaxed);
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.complete(&[3], &params(), &mut |_| true)
    }));
    let payload = unwind.expect_err("family recovery did not unwind");
    assert_eq!(
        payload.downcast_ref::<&str>().copied(),
        Some("intentional family recovery unwind")
    );
    held(&engine, true, &[], 1);
    assert_eq!(counts.forwards.load(Ordering::Relaxed), 1);
    assert_eq!(counts.resets.load(Ordering::Relaxed), 0);
    let decoded = engine.complete(&[3], &params(), &mut |_| true)?;
    assert_eq!(decoded.stop, crate::Stop::Eos);
    assert!(
        decoded.tokens.is_empty(),
        "sampled EOS was returned as output"
    );
    held(&engine, false, &[3], 1);
    let (legacy, counts) = self::engine(false)?;
    assert!(legacy.complete(&[2], &params(), &mut |_| true).is_err());
    held(&legacy, false, &[], 1);
    let error = legacy.complete(&[3], &params(), &mut |_| true).unwrap_err();
    assert!(error
        .to_string()
        .contains("fault cache not reset before retry"));
    assert_eq!(counts.fences.load(Ordering::Relaxed), 0);
    assert_eq!(counts.resets.load(Ordering::Relaxed), 0);
    Ok(())
}
