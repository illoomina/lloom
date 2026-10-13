//! Actual ImageRows, bounded inert CPU forwards; no Loaded/native/weights.
use super::*;
use crate::decision::{CausalLabels, CausalRequest, CausalTask, OrderedJson, Question};
use lloom::nn::branch::RetainState;
use lloom::nn::{CacheReset, DecodeOptions};
use lloom_tensor::{Device, Result as TensorResult};
use std::cell::Cell;
use std::rc::Rc;

#[derive(Default)]
struct Counts {
    caches: Cell<usize>,
    calls: Cell<usize>,
    text: Cell<usize>,
    captures: Cell<usize>,
    restores: Cell<usize>,
}
#[derive(Default)]
struct Tiny {
    counts: Rc<Counts>,
    fault: Cell<u8>,
    rows: Mutex<Vec<(usize, Vec<f32>)>>,
}
struct Cache {
    pos: usize,
    counts: Rc<Counts>,
}
impl CacheReset for Cache {
    fn reset(&mut self) {
        self.pos = 0;
    }
}
impl RetainState for Cache {
    type Retained = usize;
    fn retain_state(&mut self) -> TensorResult<usize> {
        self.counts.captures.set(self.counts.captures.get() + 1);
        Ok(self.pos)
    }
    fn restore_retained(&mut self, saved: &usize) -> TensorResult<()> {
        self.counts.restores.set(self.counts.restores.get() + 1);
        self.pos = *saved;
        Ok(())
    }
}
impl Model for Tiny {
    type Config = ();
    fn from_weights(_: VarBuilder, _: &()) -> TensorResult<Self> {
        lloom_tensor::bail!("inert adapter control has no weights")
    }
}
impl CausalLM for Tiny {
    type Cache = Cache;
    fn new_cache(&self, _: &(), device: &Device) -> TensorResult<Cache> {
        assert!(device.is_cpu());
        self.counts.caches.set(self.counts.caches.get() + 1);
        assert!(self.counts.caches.get() <= 4);
        Ok(Cache {
            pos: 0,
            counts: Rc::clone(&self.counts),
        })
    }
    fn vocab_size(&self) -> usize {
        128
    }
    fn embed(&self, tokens: &Tensor) -> TensorResult<Tensor> {
        let ids = tokens.to_vec2::<u32>()?;
        assert_eq!(ids.len(), 1);
        assert!(ids[0].len() <= 5);
        let values = ids[0]
            .iter()
            .flat_map(|&id| [id as f32, -(id as f32)])
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (1, ids[0].len(), 2), &Device::Cpu)
    }
    fn forward(&self, tokens: &Tensor, pos: usize, cache: &mut Cache) -> TensorResult<Tensor> {
        self.counts.text.set(self.counts.text.get() + 1);
        self.forward_embeds(&self.embed(tokens)?, pos, cache)
    }
    fn forward_embeds(
        &self,
        embeds: &Tensor,
        pos: usize,
        cache: &mut Cache,
    ) -> TensorResult<Tensor> {
        self.counts.calls.set(self.counts.calls.get() + 1);
        assert!(self.counts.calls.get() <= 7);
        assert_eq!(pos, cache.pos);
        cache.pos += embeds.dim(1)?;
        assert!(cache.pos <= 5);
        self.rows
            .lock()
            .push((pos, embeds.flatten_all()?.to_vec1::<f32>()?));
        match self.fault.replace(0) {
            1 => lloom_tensor::bail!("intentional adapter model error"),
            2 => panic!("intentional adapter model unwind"),
            _ => (),
        }
        let mut logits = vec![0f32; 128];
        logits[1] = 1.;
        logits[2] = -1.;
        Tensor::from_vec(logits, (1, 128), &Device::Cpu)
    }
}
fn features() -> TensorResult<Tensor> {
    Tensor::new(&[[2f32, 3.], [5., 7.]], &Device::Cpu)
}
fn request(prompts: [Vec<u32>; 2]) -> TensorResult<CausalRequest> {
    let tasks = prompts
        .into_iter()
        .enumerate()
        .map(|(i, prompt)| {
            let question = Question::Choice {
                instructions: OrderedJson::String("pick".into()),
                criteria: vec![("A".into(), None), ("B".into(), None)],
            };
            let labels = CausalLabels::systemone(&question, |text| match text.trim() {
                "A" => Ok(vec![1]),
                "B" => Ok(vec![2]),
                _ => lloom_tensor::bail!("unexpected fixture label"),
            })?;
            CausalTask::new(
                format!("q{i}"),
                question,
                prompt,
                labels.into_readout(),
                1.0.try_into()?,
            )
        })
        .collect::<TensorResult<Vec<_>>>()?;
    CausalRequest::new(tasks, &[7, 99, 99], 5)
}
fn buffer<M: CausalLM>(adapter: &ImageRows<'_, M>) -> (usize, usize) {
    let slot = adapter.chunk.lock();
    let chunk = slot.as_ref().expect("chunk custody lost");
    assert!(chunk.is_empty(), "completed chunk not cleared");
    (chunk.as_ptr() as usize, chunk.capacity())
}
#[test]
fn actual_tree_splices_shared_rows_once_and_validates_before_cache() -> TensorResult<()> {
    let model = Tiny::default();
    let prepared = request([vec![7, 99, 99, 3, 4], vec![7, 99, 99, 5, 6]])?;
    prepared.validate_shared_token_positions(99, 2)?;
    prepared.validate_tokens(128)?;
    let adapter = ImageRows::try_new(&model, prepared.trunk_tokens().to_vec(), 99, features()?, 0)?;
    let before = buffer(&adapter);
    assert!(before.1 >= 3);
    let mut readback = lloom::generation::Blocking::new(&[0], &Device::Cpu)?;
    let options = DecodeOptions {
        prefill_tile: 8,
        ..DecodeOptions::new(1)
    };
    let response = prepared.execute(&adapter, &(), &mut readback, &options)?;
    assert_eq!(response.answers().len(), 2);
    assert_eq!(prepared.input_tokens(), 10);
    assert_eq!(adapter.spliced(), 2);
    assert_eq!(buffer(&adapter), before);
    assert_eq!(
        model.rows.lock().as_slice(),
        &[
            (0, vec![7., -7., 2., 3., 5., 7.]),
            (3, vec![3., -3., 4., -4.]),
            (3, vec![5., -5., 6., -6.]),
        ]
    );
    assert_eq!(model.counts.caches.get(), 1);
    assert_eq!(model.counts.calls.get(), 3);
    assert_eq!(model.counts.text.get(), 2);
    assert_eq!(model.counts.captures.get(), 1);
    assert_eq!(model.counts.restores.get(), 2);
    let malformed = request([vec![7, 99, 99, 128, 4], vec![7, 99, 99, 5, 6]])?;
    assert!(malformed
        .execute(&adapter, &(), &mut readback, &options)
        .unwrap_err()
        .to_string()
        .contains("outside loaded vocabulary"));
    assert_eq!(
        model.counts.caches.get(),
        1,
        "invalid token allocated a branch cache"
    );
    let suffix_image = request([vec![7, 99, 99, 99, 4], vec![7, 99, 99, 5, 6]])?;
    assert!(suffix_image
        .validate_shared_token_positions(99, 2)
        .unwrap_err()
        .to_string()
        .contains("shared trunk"));
    Ok(())
}
#[test]
fn actual_image_chunk_capacity_restores_on_error_and_exact_unwind() -> TensorResult<()> {
    let model = Tiny::default();
    let adapter = ImageRows::try_new(&model, vec![7, 99, 99], 99, features()?, 0)?;
    let before = buffer(&adapter);
    let ids = [7, 99, 99];
    let tokens = Tensor::new(&[ids], &Device::Cpu)?;
    for fault in [1, 2, 0] {
        let mut cache = adapter.new_cache(&(), &Device::Cpu)?;
        adapter.note_input_ids(&mut cache, &ids)?;
        model.fault.set(fault);
        if fault == 2 {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                adapter.forward(&tokens, 0, &mut cache)
            }));
            let payload = result.expect_err("adapter did not unwind");
            assert_eq!(
                payload.downcast_ref::<&str>().copied(),
                Some("intentional adapter model unwind")
            );
        } else if fault == 1 {
            assert!(adapter
                .forward(&tokens, 0, &mut cache)
                .unwrap_err()
                .to_string()
                .contains("intentional adapter model error"));
        } else {
            adapter.forward(&tokens, 0, &mut cache)?;
        }
        assert_eq!(
            buffer(&adapter),
            before,
            "Err/unwind discarded or replaced image chunk capacity"
        );
        assert!(!adapter.pending.load(Ordering::Relaxed));
        assert_eq!(cache.pos, 3);
    }
    assert_eq!(model.counts.calls.get(), 3);
    assert_eq!(
        adapter.spliced(),
        6,
        "attempted rows are counted before base forward success"
    );
    Ok(())
}
#[test]
fn actual_note_refusal_and_legacy_capacity_keep_text_forward_callable() -> TensorResult<()> {
    let model = Tiny::default();
    let adapter = ImageRows::new(&model, vec![7, 99, 99], 99, features()?, 0);
    let mut cache = adapter.new_cache(&(), &Device::Cpu)?;
    assert!(adapter
        .note_input_ids(&mut cache, &[99; 4])
        .unwrap_err()
        .to_string()
        .contains("prompt bound"));
    assert_eq!(buffer(&adapter).1, 0, "refused oversized note allocated");
    adapter.note_input_ids(&mut cache, &[7, 99, 99])?;
    let ids = adapter.chunk.lock().take().expect("note lost chunk");
    let held = ImageChunk {
        slot: &adapter.chunk,
        ids,
    };
    let before = (held.ids.as_ptr() as usize, held.ids.capacity());
    assert!(adapter
        .note_input_ids(&mut cache, &[99])
        .unwrap_err()
        .to_string()
        .contains("in flight"));
    assert!(adapter
        .forward(&Tensor::new(&[[99u32]], &Device::Cpu)?, 0, &mut cache)
        .unwrap_err()
        .to_string()
        .contains("already in flight"));
    drop(held);
    assert_eq!(buffer(&adapter), before);
    assert_eq!(model.counts.calls.get(), 0);
    adapter.note_input_ids(&mut cache, &[7, 99, 99])?;
    adapter.forward(
        &Tensor::new(&[[7u32, 99, 99]], &Device::Cpu)?,
        0,
        &mut cache,
    )?;
    assert_eq!(buffer(&adapter), before);
    adapter.note_input_ids(&mut cache, &[4])?;
    adapter.forward(&Tensor::new(&[[4u32]], &Device::Cpu)?, 3, &mut cache)?;
    assert_eq!(model.counts.text.get(), 1);
    assert_eq!(buffer(&adapter), before);
    assert_eq!(adapter.spliced(), 2);
    Ok(())
}
