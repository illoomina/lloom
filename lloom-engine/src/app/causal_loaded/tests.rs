use super::*;
use crate::decision::{CausalLabels, CausalTask, Question};
use lloom::nn::{Model, VarBuilder};
use lloom_tensor::Tensor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

// Ownership/callback controls only: V8, max3positions, no weights/checkpoint,
// native device or admission proof. Tiny's constant logits are not a model oracle.
pub(super) struct Tiny {
    caches: AtomicUsize,
}
pub(super) struct Cache {
    pub(super) pos: usize,
}
impl CacheReset for Cache {
    fn reset(&mut self) {
        self.pos = 0;
    }
}
impl CacheRewind for Cache {
    fn rewind_horizon(&self) -> usize {
        0
    }
    fn rewind(&mut self, n: usize) -> Result<()> {
        if n != 0 {
            lloom_tensor::bail!("tiny has no rewind window")
        }
        Ok(())
    }
}
impl RetainState for Cache {
    type Retained = usize;
    fn retain_state(&mut self) -> Result<usize> {
        Ok(self.pos)
    }
    fn restore_retained(&mut self, saved: &usize) -> Result<()> {
        self.pos = *saved;
        Ok(())
    }
}
impl Model for Tiny {
    type Config = ();
    fn from_weights(_: VarBuilder, _: &()) -> Result<Self> {
        Ok(Self {
            caches: AtomicUsize::new(0),
        })
    }
}
impl CausalLM for Tiny {
    type Cache = Cache;
    fn new_cache(&self, _: &(), _: &Device) -> Result<Cache> {
        self.caches.fetch_add(1, Ordering::Relaxed);
        Ok(Cache { pos: 0 })
    }
    fn embed(&self, tokens: &Tensor) -> Result<Tensor> {
        Ok(tokens.clone())
    }
    fn forward_embeds(&self, tokens: &Tensor, pos: usize, cache: &mut Cache) -> Result<Tensor> {
        assert_eq!(pos, cache.pos);
        cache.pos += tokens.dim(1)?;
        assert!(cache.pos <= 3);
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
pub(super) fn engine() -> Result<CausalEngine<Tiny>> {
    let tokenizer = Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [("a".into(), 0), ("b".into(), 1), ("[UNK]".into(), 7)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .map_err(Error::wrap)?,
    );
    let model = Tiny {
        caches: AtomicUsize::new(0),
    };
    let cache = model.new_cache(&(), &Device::Cpu)?;
    Ok(CausalEngine {
        model,
        config: (),
        device: Device::Cpu,
        capacity: 3,
        tokenizer,
        policy: None,
        decode: DecodeConfig::default(),
        recovery: None,
        chat: Mutex::new(Some(Chat {
            sequence: Some(Sequence::new(cache)),
            rings: Some(RingPool::new()),
            extra: (),
            needs_recovery: false,
        })),
    })
}
fn request() -> Result<CausalRequest> {
    let question = Question::Choice {
        instructions: OrderedJson::String("choose".into()),
        criteria: vec![("a".into(), None), ("b".into(), None)],
    };
    let labels = CausalLabels::systemone(&question, |text| match text.trim() {
        "a" => Ok(vec![0]),
        "b" => Ok(vec![1]),
        _ => lloom_tensor::bail!("unexpected tiny fixture label {text:?}"),
    })?;
    let task = CausalTask::new(
        "q".into(),
        question,
        vec![2, 3],
        labels.into_readout(),
        Temperature::try_from(1.)?,
    )?;
    CausalRequest::new(vec![task], &[2], 3)
}
fn params() -> DecodeParams {
    DecodeParams {
        max_tokens: 1,
        eos_ids: vec![7],
        readback: ReadbackArm::Blocking,
        ..DecodeParams::default()
    }
}

#[test]
fn callback_reentry_is_busy_then_decisions_and_chat_remain_available() -> Result<()> {
    let engine = engine()?;
    let request = request()?;
    let mut callbacks = 0;
    engine.complete(&[2, 3], &params(), &mut |_| {
        callbacks += 1;
        assert!(engine
            .decide(&request)
            .unwrap_err()
            .to_string()
            .contains("busy"));
        assert!(engine
            .complete(&[2], &params(), &mut |_| true)
            .unwrap_err()
            .to_string()
            .contains("busy"));
        true
    })?;
    assert!(callbacks > 0, "actual generation did not invoke callback");
    assert_eq!(
        engine.model.caches.load(Ordering::Relaxed),
        1,
        "Busy allocated a decision cache"
    );
    assert_eq!(engine.decide(&request)?.answers().len(), 1);
    assert_eq!(engine.model.caches.load(Ordering::Relaxed), 2);
    engine.complete(&[3], &params(), &mut |_| true)?;
    assert!(engine.chat.lock().is_some());
    Ok(())
}

#[test]
fn callback_unwind_restores_decode_session_and_clears_busy() -> Result<()> {
    let engine = engine()?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.complete(&[2, 3], &params(), &mut |_| {
            panic!("intentional bounded callback unwind")
        })
    }));
    assert!(
        result.is_err(),
        "actual generation did not reach unwind callback"
    );
    assert!(engine.chat.lock().is_some(), "RAII lost the chat sequence");
    assert_eq!(engine.decide(&request()?)?.answers().len(), 1);
    engine.complete(&[3], &params(), &mut |_| true)?;
    Ok(())
}

#[test]
fn callback_other_thread_gets_busy_without_waiting_for_original_request() -> Result<()> {
    let engine = Arc::new(engine()?);
    let request = Arc::new(request()?);
    let mut callbacks = 0;
    let mut child = None;
    let mut received = None;
    let completion = engine.complete(&[2, 3], &params(), &mut |_| {
        callbacks += 1;
        if callbacks > 1 {
            return true;
        }
        let (send, receive) = std::sync::mpsc::channel();
        let engine = engine.clone();
        let request = request.clone();
        child = Some(std::thread::spawn(move || {
            let result = engine
                .decide(&request)
                .map(|_| ())
                .map_err(|e| e.to_string());
            // The original request may have released its lease after timeout.
            // Retain the child until after that release; never detach on failure.
            let _ = send.send(result);
        }));
        received = Some(receive.recv_timeout(std::time::Duration::from_secs(2)));
        received.as_ref().is_some_and(|result| result.is_ok())
    });
    // `complete` has dropped its lease before joining, including its error path.
    if let Some(child) = child {
        child.join().expect("callback thread panicked");
    }
    completion?;
    assert!(callbacks > 0);
    let result = received
        .expect("callback did not spawn the bounded child")
        .expect("reentrant callback thread blocked on original request");
    assert!(result.unwrap_err().contains("busy"));
    assert!(engine.chat.lock().is_some());
    Ok(())
}
