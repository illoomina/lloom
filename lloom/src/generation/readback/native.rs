//! Explicit request-local native strategy over the existing pooled ring.
//! No context is stored in RingPool or handed to its independently Send reader.

use super::*;
use crate::generation::native::{PreparedNativeBudget, PreparedNativeSampler};
use crate::nn::CausalLM;
use lloom_tensor::metal_backend::sampling::SamplePublisher;
use lloom_tensor::{MetalStorage, Storage};

/// Construct, drive and drop on the decode thread for one request. The shared
/// error progress deliberately makes this adapter !Send/!Sync. Its publisher
/// retains exact-origin error storage after borrowed draws drop; GPU epoch
/// custody retains physical bytes on early host errors. No reset API exists.
///
/// ```compile_fail
/// fn require_send<T: Send>() {}
/// require_send::<lloom_tensor::metal_backend::sampling::SamplePublisher>();
/// ```
pub struct NativeRing<'a, M: CausalLM> {
    ring: &'a mut Ring,
    model: &'a M,
    native: PreparedNativeSampler,
    publisher: SamplePublisher,
    embeds: Tensor,
    forcing: Option<PreparedNativeBudget>,
}

fn storage(t: &Tensor) -> Result<MetalStorage> {
    let (storage, layout) = t.storage_and_layout();
    if !layout.is_contiguous() {
        lloom_tensor::bail!("native ring: contiguous publication storage required");
    }
    let Storage::Metal(storage) = &*storage else {
        lloom_tensor::bail!("native ring: Metal publication storage required");
    };
    Ok(storage.clone())
}

fn stamp(publisher: &SamplePublisher, ring: &TokenRing, g: u32, history: &Tensor) -> Result<()> {
    publisher.publish_marker(
        &storage(ring.step_counter())?,
        g,
        &storage(history)?,
        &storage(&ring.input_for(1)?)?,
        &storage(ring.eos_flag())?,
    )?;
    ring.signal_publication(g)
}

impl<'a, M: CausalLM> NativeRing<'a, M> {
    /// Existing admission owns all allocations/quotes. `embeds` is prepared
    /// `[1,1,hidden]` storage; unsupported model embedding routes fail cold.
    /// The prior request must have completed its final drain before borrowing
    /// its pooled ring. This constructor never clears or resets prior errors.
    pub fn new(
        ring: &'a mut Ring,
        model: &'a M,
        native: PreparedNativeSampler,
        embeds: Tensor,
        forcing: Option<PreparedNativeBudget>,
    ) -> Result<Self> {
        model.check_guarded_embedding(&embeds)?;
        if ring.history.ring.quantum != 1
            || ring.history.undrained() != 0
            || !ring.unstamped.is_empty()
        {
            lloom_tensor::bail!(
                "native ring: one-token geometry and prior request final drain required"
            );
        }
        native.check_output(&embeds, model.vocab_size())?;
        if let Some(budget) = forcing.as_ref() {
            budget.check_for(embeds.device(), model.vocab_size())?;
        }
        let publisher = native.publisher();
        Ok(Self {
            ring,
            model,
            native,
            publisher,
            embeds,
            forcing,
        })
    }

    fn check_error(&mut self) -> Result<()> {
        self.publisher.check_error().map(|_| ())
    }

    fn stamp_pending(&mut self) -> Result<()> {
        for &(seg, g) in &self.ring.unstamped {
            stamp(
                &self.publisher,
                &self.ring.ring,
                g,
                &self.ring.history.ring.segments[seg],
            )?;
        }
        self.ring.unstamped.clear();
        Ok(())
    }
}

impl<M: CausalLM> ReadbackStrategy for NativeRing<'_, M> {
    const PASS_INPUT: lloom_tensor::backend::PassInput =
        lloom_tensor::backend::PassInput::PreparedEmbeddings;

    fn device(&self) -> &Device {
        self.ring.device()
    }
    fn arms(&self) -> ReadbackArms {
        self.ring.arms()
    }
    fn admits_lookahead(&self) -> bool {
        self.ring.admits_lookahead()
    }
    fn loop_commits(&self) -> bool {
        self.ring.loop_commits()
    }
    fn check_sampler(&self, sampling: &Sampling) -> Result<()> {
        self.native.check_sampling_kind(sampling)
    }
    fn check_request(&self, sampler: &LogitsProcessor) -> Result<()> {
        crate::nn::decode::refuse_prepared_step(&crate::nn::decode::Asks::of(sampler, false))?;
        self.check_sampler(sampler.sampling())
    }
    fn begin_run(&mut self) {
        self.ring.begin_run();
    }
    fn published_count(&self) -> Option<usize> {
        self.ring.published_count()
    }
    #[cfg(all(feature = "metal", any(debug_assertions, feature = "profile")))]
    fn terminal_observers(&self) -> crate::nn::terminal_trace::Observers {
        self.ring.terminal_observers()
    }
    fn stage_prompt_chunk(&mut self, tokens: &[u32]) -> Result<Tensor> {
        self.ring.stage_prompt_chunk(tokens)
    }
    fn sample_and_feed(
        &mut self,
        sampler: &mut LogitsProcessor,
        logits: &Tensor,
    ) -> Result<Tensor> {
        self.check_error()?;
        let token = self.native.sample(sampler, logits, self.forcing.as_mut())?;
        self.ring.fused_step = false;
        self.ring.counters.ring_publishes += 1;
        let (seg, offset) = self.ring.history.begin_step()?;
        self.ring.generation =
            self.ring.generation.checked_add(1).ok_or_else(|| {
                lloom_tensor::Error::Msg("native ring: generation overflow".into())
            })?;
        self.ring.reserve_reader_window();
        let segment = &self.ring.history.ring.segments[seg];
        token.copy_into_tensor(segment, offset)?;
        let input = self.ring.ring.next_input_slot_view()?;
        token.copy_into_tensor(&input, 0)?;
        let eos = storage(self.ring.ring.eos_flag())?;
        token.eos_flag_into(&storage(self.ring.ring.eos_ids_tensor())?, &eos)?;
        // Only this boundary can index the model table. A failed request leaves
        // prepared embedding bytes unchanged; no stale raw token is indexed.
        self.model.embed_guarded_into(&token, &self.embeds)?;
        match self.ring.eos {
            EosRead::Poll(_) | EosRead::Blocking => token.publish_marker(
                &storage(self.ring.ring.step_counter())?,
                self.ring.generation,
                &storage(segment)?,
                &storage(&self.ring.ring.input_for(1)?)?,
                &eos,
            )?,
            EosRead::HostOnDrain | EosRead::GpuBatched | EosRead::GpuBatchedNonblocking => {
                match self.ring.unstamped.last_mut() {
                    Some((s, g)) if *s == seg => *g = self.ring.generation,
                    _ => self.ring.unstamped.push((seg, self.ring.generation)),
                }
            }
        }
        if matches!(self.ring.eos, EosRead::Poll(_) | EosRead::Blocking) {
            self.ring.ring.signal_publication(self.ring.generation)?;
        }
        self.ring.history.wrote_one();
        self.ring.ring.input_for(1)
    }
    fn forward_input<N: CausalLM>(
        &mut self,
        model: &N,
        _input: &Tensor,
        pos: usize,
        cache: &mut N::Cache,
    ) -> Result<Tensor> {
        if std::any::type_name::<M>() != std::any::type_name::<N>()
            || !std::ptr::eq(
                self.model as *const M as *const (),
                model as *const N as *const (),
            )
        {
            lloom_tensor::bail!("native ring: forward must use its prepared model instance");
        }
        self.check_error()?;
        model.forward_prepared_embeds(&self.embeds, pos, cache)
    }
    fn set_thinking_budget(&mut self, budget: Option<&ThinkingBudget>) -> Result<()> {
        self.forcing = budget
            .map(|b| PreparedNativeBudget::new(b.clone(), self.device(), self.model.vocab_size()))
            .transpose()?;
        Ok(())
    }
    fn publish_marker(&mut self, force_live: bool) -> Result<()> {
        self.check_error()?;
        let publisher = &self.publisher;
        self.ring
            .publish_marker_with(force_live, |ring, g, history| {
                stamp(publisher, ring, g, history)
            })
    }
    fn retracts_draw(&self) -> bool {
        matches!(self.ring.eos, EosRead::Poll(_) | EosRead::Blocking)
            && self.forcing.as_ref().is_none_or(|f| !f.live())
    }
    fn retract_draw(&mut self) -> Result<()> {
        self.check_error()?;
        if !self.retracts_draw() {
            lloom_tensor::bail!("native ring: this sampler/budget arm cannot retract");
        }
        self.ring.retract_last_draw()
    }
    fn eos_of_last(&mut self) -> Result<bool> {
        let publisher = &mut self.publisher;
        self.ring
            .eos_of_last_with(|| publisher.check_error().map(|_| ()))
    }
    fn take_ids(&mut self) -> Result<Vec<u32>> {
        self.check_error()?;
        let ids = self.ring.take_ids()?;
        self.check_error()?;
        Ok(ids)
    }
    fn take_ids_into(&mut self, out: &mut Vec<u32>) -> Result<()> {
        self.check_error()?;
        let old_len = out.len();
        self.ring.take_ids_into(out)?;
        if let Err(e) = self.check_error() {
            out.truncate(old_len);
            return Err(e);
        }
        Ok(())
    }
    fn drain_all(&mut self) -> Result<Vec<u32>> {
        self.check_error()?;
        self.stamp_pending()?;
        let ids = self.ring.drain_all()?;
        // EOS truncation/cancellation can leave no undrained ids while queued
        // lookahead still owns the error word. Complete that request's final
        // error observation even in the legacy empty-tail shortcut.
        if !self.publisher.check_error()? {
            self.device().synchronize()?;
            if !self.publisher.check_error()? {
                lloom_tensor::bail!("native ring: final error writer did not complete");
            }
        }
        Ok(ids)
    }
    fn eos_in_drained(&self, batch: &[u32]) -> Option<usize> {
        self.ring.eos_in_drained(batch)
    }
    fn note_eos_truncation(&mut self, dropped: usize) {
        self.ring.note_eos_truncation(dropped)
    }
    fn eos_read_arm(&self) -> &'static str {
        self.ring.eos_read_arm()
    }
    fn eos_ids_tensor(&self) -> Option<&Tensor> {
        self.ring.eos_ids_tensor()
    }
    fn reader(&self) -> Option<RingReader> {
        self.ring.reader()
    }
    fn stamps_each_token(&self, adaptive: bool) -> bool {
        self.ring.stamps_each_token(adaptive)
    }
    fn arm_publish_event(
        &mut self,
        for_reader: bool,
        eos_park: bool,
        adaptive: bool,
    ) -> Result<()> {
        self.ring.arm_publish_event(for_reader, eos_park, adaptive)
    }
    fn close_window(&mut self, ramp: bool) -> Result<()> {
        self.check_error()?;
        self.ring.close_window(ramp)
    }
    fn set_fused_publish(&mut self, _on: bool) { /* argmax fusion is not this sampler */
    }
    fn publishes_fused(&self) -> bool {
        false
    }
    fn counters(&self) -> ReadbackCounters {
        self.ring.counters()
    }
}

impl<M: CausalLM> Drop for NativeRing<'_, M> {
    fn drop(&mut self) {
        // Never publish failed/cancelled request tails through a later request's
        // legacy stamp. Queued error/consumer buffers retain epoch custody; the
        // next request gets a fresh sticky guard, never resets these bytes.
        self.ring.unstamped.clear();
        self.ring.counters.ring_discarded += self.ring.history.forget_undrained() as u64;
    }
}
